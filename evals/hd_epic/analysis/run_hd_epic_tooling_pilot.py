from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from answering import QwenVLMAnswerer, parse_choice_letter
from evals.hd_epic.loader import load_examples
from evals.hd_epic.runner import _build_frame_timestamp_labels, _build_timestamped_mcq_prompt, _sample_uniform_video_frames
from pipeline.experiments.hd_epic_mcq_shortlist_joint import (
    _build_answer_only_prompt,
    _build_mcq_style_frame_labels,
    _build_query_encoder_and_archive,
    _build_segment_index,
    _build_selection_only_prompt,
    _choice_spans_for_example,
    _parse_final_answer_letter,
    _parse_selection_candidate,
    _sample_candidate_frames,
)
from pipeline.config import PIPELINE_CONFIG
from pipeline.core.io import append_jsonl, log_line, write_json
from pipeline.core.metrics import summarize_layer2_hits
from pipeline.core.retrieve import extract_target_text
from evals.hd_epic.temporal import example_scope_for_video, gold_spans_for_video
from pipeline.experiments.hd_epic_mcq_shortlist_joint import (
    BudgetConfig,
    _candidate_pool_hierarchical,
    _prompt_choices,
    _write_status,
    default_answer_config,
    shortlist_candidates_cover_choices,
)
from pipeline.tools.compare_time import compare_time_tool
from pipeline.tools.parse_query import parse_query_tool
from pipeline.tools.verify_consistency import verify_consistency_tool


DEFAULT_SPLIT = REPO_ROOT / "results" / "pipeline" / "splits" / "hd_epic_tooling_pilot_v1.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "pipeline" / "analysis" / "hd_epic_tooling_pilot_v1"
TEMPORAL_TASKS = {
    "fine_grained_action_localization",
    "recipe_step_localization",
}
LENGTH_BUCKET_ORDER = ["short", "medium", "long"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the HD-EPIC tooling pilot.")
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--methods", nargs="+", default=["pure_vlm", "tooling"], choices=["pure_vlm", "tooling"])
    parser.add_argument("--max-examples-per-task", type=int, default=None)
    parser.add_argument("--tasks", nargs="+", default=None)
    return parser


def _load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_examples_by_id(task_names: list[str]) -> dict[str, Any]:
    examples = load_examples(task_names, REPO_ROOT)
    return {example.example_id: example for example in examples}


def _selected_entries(manifest: dict[str, Any], *, tasks: set[str] | None, max_examples_per_task: int | None) -> list[dict[str, Any]]:
    entries = [entry for entry in manifest["entries"] if tasks is None or entry["task_name"] in tasks]
    if max_examples_per_task is None:
        return entries
    counts: Counter[str] = Counter()
    trimmed: list[dict[str, Any]] = []
    for entry in entries:
        if counts[entry["task_name"]] >= max_examples_per_task:
            continue
        trimmed.append(entry)
        counts[entry["task_name"]] += 1
    return trimmed


def _example_groups(entries: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        grouped[entry["evaluation_video_id"]].append(entry)
    for group in grouped.values():
        group.sort(key=lambda item: (item["task_name"], item["example_id"]))
    return dict(sorted(grouped.items()))


def _tool_summary_for_temporal_choices(example: Any, shortlisted_candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    parse_payload = parse_query_tool({"question": example.question, "choices": example.choices})
    choice_refs = [item for item in parse_payload["choice_time_references"] if item is not None]
    if not choice_refs:
        return [], ""
    candidate_refs = [
        {
            "start_time_sec": float(candidate["start_time_sec"]),
            "end_time_sec": float(candidate["end_time_sec"]),
        }
        for candidate in shortlisted_candidates
    ]
    compared = compare_time_tool(
        {
            "candidates": candidate_refs,
            "references": choice_refs,
            "tolerance_sec": 10.0,
        }
    )
    lines = [
        "Temporal tool summary:",
        f"- query_family: {parse_payload['query_family']}",
        f"- primary_entity: {parse_payload['primary_entity']}",
    ]
    records: list[dict[str, Any]] = []
    for item in compared["pairwise"]:
        record = {
            "candidate_index": int(item["candidate_index"]) + 1,
            "nearest_reference_index": item["nearest_reference_index"],
            "nearest_gap_sec": item["nearest_gap_sec"],
            "nearest_overlap_sec": item["nearest_overlap_sec"],
            "within_tolerance": item["within_tolerance"],
        }
        records.append(record)
        option_label = None
        if item["nearest_reference_index"] is not None:
            option_label = chr(ord("A") + int(item["nearest_reference_index"]))
        lines.append(
            f"- Candidate {record['candidate_index']}: nearest option {option_label}, "
            f"gap={None if record['nearest_gap_sec'] is None else round(float(record['nearest_gap_sec']), 3)}s, "
            f"overlap={None if record['nearest_overlap_sec'] is None else round(float(record['nearest_overlap_sec']), 3)}s"
        )
    return records, "\n".join(lines)


def _selected_option_interval(example: Any, selected_letter: str | None) -> dict[str, Any] | None:
    if selected_letter is None:
        return None
    idx = ord(selected_letter) - ord("A")
    choice_spans = _choice_spans_for_example(example)
    if idx < 0 or idx >= len(choice_spans):
        return None
    return choice_spans[idx]


def _tolerance_hit(candidate: dict[str, Any], gold_spans: list[dict[str, Any]], tolerance_sec: float) -> float | None:
    if not gold_spans:
        return None
    start = float(candidate["start_time_sec"])
    end = float(candidate["end_time_sec"])
    for span in gold_spans:
        g_start = float(span["start_time_sec"])
        g_end = float(span["end_time_sec"])
        if max(start, g_start) <= min(end, g_end):
            return 1.0
        if end < g_start and g_start - end <= tolerance_sec:
            return 1.0
        if g_end < start and start - g_end <= tolerance_sec:
            return 1.0
    return 0.0


def _row_summary(rows: list[dict[str, Any]], *, method_name: str) -> dict[str, Any]:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    temporal_rows = [row for row in ok_rows if row["task_name"] in TEMPORAL_TASKS]
    payload = {
        "method_name": method_name,
        "example_count": len(ok_rows),
        "mcq_accuracy": round(sum(float(row["mcq_correct"]) for row in ok_rows) / len(ok_rows), 6) if ok_rows else None,
    }
    if temporal_rows:
        payload.update(
            {
                "candidate_pool_hit_any": round(
                    sum(float(row["candidate_pool_hit_any"]) for row in temporal_rows if row.get("candidate_pool_hit_any") is not None)
                    / len([row for row in temporal_rows if row.get("candidate_pool_hit_any") is not None]),
                    6,
                )
                if any(row.get("candidate_pool_hit_any") is not None for row in temporal_rows)
                else None,
                "selected_evidence_hit1": round(
                    sum(float(row["selected_evidence_hit1"]) for row in temporal_rows if row.get("selected_evidence_hit1") is not None)
                    / len([row for row in temporal_rows if row.get("selected_evidence_hit1") is not None]),
                    6,
                )
                if any(row.get("selected_evidence_hit1") is not None for row in temporal_rows)
                else None,
                "selected_evidence_hit1_tol_5s": round(
                    sum(float(row["selected_evidence_hit1_tol_5s"]) for row in temporal_rows if row.get("selected_evidence_hit1_tol_5s") is not None)
                    / len([row for row in temporal_rows if row.get("selected_evidence_hit1_tol_5s") is not None]),
                    6,
                )
                if any(row.get("selected_evidence_hit1_tol_5s") is not None for row in temporal_rows)
                else None,
                "selected_evidence_hit1_tol_10s": round(
                    sum(float(row["selected_evidence_hit1_tol_10s"]) for row in temporal_rows if row.get("selected_evidence_hit1_tol_10s") is not None)
                    / len([row for row in temporal_rows if row.get("selected_evidence_hit1_tol_10s") is not None]),
                    6,
                )
                if any(row.get("selected_evidence_hit1_tol_10s") is not None for row in temporal_rows)
                else None,
                "answer_accuracy_given_selected_hit": round(
                    sum(float(row["mcq_correct"]) for row in temporal_rows if row.get("selected_evidence_hit1") == 1.0)
                    / len([row for row in temporal_rows if row.get("selected_evidence_hit1") == 1.0]),
                    6,
                )
                if any(row.get("selected_evidence_hit1") == 1.0 for row in temporal_rows)
                else None,
                "answer_accuracy_given_selected_miss": round(
                    sum(float(row["mcq_correct"]) for row in temporal_rows if row.get("selected_evidence_hit1") == 0.0)
                    / len([row for row in temporal_rows if row.get("selected_evidence_hit1") == 0.0]),
                    6,
                )
                if any(row.get("selected_evidence_hit1") == 0.0 for row in temporal_rows)
                else None,
            }
        )
        tp = sum(1 for row in temporal_rows if row.get("selected_evidence_hit1") == 1.0 and row["mcq_correct"] == 1.0)
        fn = sum(1 for row in temporal_rows if row.get("selected_evidence_hit1") == 1.0 and row["mcq_correct"] == 0.0)
        fp = sum(1 for row in temporal_rows if row.get("selected_evidence_hit1") == 0.0 and row["mcq_correct"] == 1.0)
        tn = sum(1 for row in temporal_rows if row.get("selected_evidence_hit1") == 0.0 and row["mcq_correct"] == 0.0)
        payload.update({"tp_count": tp, "fn_count": fn, "fp_count": fp, "tn_count": tn})
    return payload


def _build_group_summaries(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    return {group_key: _row_summary(group_rows, method_name="group") for group_key, group_rows in sorted(grouped.items())}


def _paired_counts(rows_by_method: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    if "pure_vlm" not in rows_by_method or "tooling" not in rows_by_method:
        return {}
    base = {row["example_id"]: row for row in rows_by_method["pure_vlm"] if row.get("status") == "ok"}
    tool = {row["example_id"]: row for row in rows_by_method["tooling"] if row.get("status") == "ok"}
    common_ids = sorted(set(base) & set(tool))
    wins = losses = ties = 0
    for example_id in common_ids:
        b = float(base[example_id]["mcq_correct"])
        t = float(tool[example_id]["mcq_correct"])
        if t > b:
            wins += 1
        elif t < b:
            losses += 1
        else:
            ties += 1
    return {
        "paired_n": len(common_ids),
        "tooling_wins": wins,
        "tooling_losses": losses,
        "ties": ties,
    }


def _run_pure_example(*, example: Any, video_id: str, answerer: QwenVLMAnswerer, budget_config: BudgetConfig) -> dict[str, Any]:
    video_path = REPO_ROOT / "dataset" / "data" / "HD-EPIC" / "Videos" / video_id.split("-", 1)[0] / f"{video_id}.mp4"
    scope_start_sec, scope_end_sec = example_scope_for_video(example, video_id)
    if scope_start_sec is None:
        scope_start_sec = 0.0
    if scope_end_sec is None:
        capture = cv2.VideoCapture(str(video_path))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        capture.release()
        scope_end_sec = frames / fps if fps > 0.0 else scope_start_sec
    labeled_options = _prompt_choices(example)
    frames, _, timestamps_sec, _ = _sample_uniform_video_frames(
        video_path=video_path,
        frame_budget=budget_config.total_frames,
        start_time_sec=float(scope_start_sec),
        end_time_sec=float(scope_end_sec),
    )
    prompt = _build_timestamped_mcq_prompt(
        question=example.question,
        labeled_options=labeled_options,
        clip_start_sec=float(scope_start_sec),
        clip_end_sec=float(scope_end_sec),
    )
    generation = answerer.generate_text_from_frames(
        frames=frames,
        prompt=prompt,
        frame_texts=_build_frame_timestamp_labels(timestamps_sec),
    )
    predicted_letter = parse_choice_letter(generation.raw_text, options_count=len(example.choices))
    predicted_idx = (ord(predicted_letter) - ord("A")) if predicted_letter is not None else None
    return {
        "example_id": example.example_id,
        "task_name": example.task_name,
        "mcq_correct": 1.0 if predicted_idx == example.correct_idx else 0.0,
        "raw_answer": generation.raw_text,
        "total_frames_seen_per_question": float(len(frames)),
        "selected_evidence_hit1": None,
        "selected_evidence_hit1_tol_5s": None,
        "selected_evidence_hit1_tol_10s": None,
        "candidate_pool_hit_any": None,
        "shortlist_hit_any": None,
        "status": "ok",
    }


def _run_tooling_example(
    *,
    example: Any,
    video_id: str,
    answerer: QwenVLMAnswerer,
    budget_config: BudgetConfig,
    query_encoder: Any,
    archive: Any,
    index_bundle: dict[str, Any],
) -> dict[str, Any]:
    video_path = REPO_ROOT / "dataset" / "data" / "HD-EPIC" / "Videos" / video_id.split("-", 1)[0] / f"{video_id}.mp4"
    query_text = extract_target_text(example.question)
    query_embedding = query_encoder.encode_texts([query_text], batch_size=PIPELINE_CONFIG.retrieval.openclip_batch_size)[0]
    gold_spans = gold_spans_for_video(example, video_id)
    scope_start_sec, scope_end_sec = example_scope_for_video(example, video_id)
    layer3_hits, candidate_hits = _candidate_pool_hierarchical(
        query_embedding=query_embedding,
        video_id=video_id,
        scope_start_sec=scope_start_sec,
        scope_end_sec=scope_end_sec,
        index_bundle=index_bundle,
    )
    if not candidate_hits:
        return {
            "example_id": example.example_id,
            "task_name": example.task_name,
            "status": "error",
            "message": "No candidate hits produced.",
        }

    labeled_options = _prompt_choices(example)
    choice_spans = _choice_spans_for_example(example)
    if choice_spans:
        shortlist_ids, shortlist_scored_candidates = shortlist_candidates_cover_choices(
            candidates=candidate_hits,
            choice_spans=choice_spans,
            max_keep=budget_config.shortlist_k,
        )
        shortlisted_candidates = [candidate for candidate in candidate_hits if candidate["candidate_id"] in shortlist_ids]
    else:
        shortlist_scored_candidates = []
        shortlisted_candidates = candidate_hits[: budget_config.shortlist_k]

    selection_frames = []
    selection_frame_texts = []
    display_id_to_candidate: dict[int, dict[str, Any]] = {}
    per_candidate_frames: dict[int, list[Any]] = {}
    per_candidate_times: dict[int, list[float]] = {}
    for display_index, candidate in enumerate(shortlisted_candidates, start=1):
        display_id_to_candidate[display_index] = candidate
        frames, timestamps_sec = _sample_candidate_frames(
            sampler="l1_keyframes",
            video_path=video_path,
            candidate=candidate,
            frame_budget=budget_config.frames_per_candidate,
            archive=archive,
            query_embedding=query_embedding,
        )
        per_candidate_frames[display_index] = frames
        per_candidate_times[display_index] = [float(ts) for ts in timestamps_sec]
        selection_frames.extend(frames)
        selection_frame_texts.extend(
            [f"Candidate {display_index} | {label}" for label in _build_mcq_style_frame_labels(timestamps_sec)]
        )

    tool_records, tool_summary = _tool_summary_for_temporal_choices(example, shortlisted_candidates)
    selection_prompt = _build_selection_only_prompt(
        question=example.question,
        labeled_options=labeled_options,
        candidates=shortlisted_candidates,
    )
    if tool_summary:
        selection_prompt += "\n\n" + tool_summary
    selection_generation = answerer.generate_text_from_frames(
        frames=selection_frames,
        prompt=selection_prompt,
        frame_texts=selection_frame_texts,
    )
    selected_display_id = _parse_selection_candidate(
        selection_generation.raw_text,
        candidates=shortlisted_candidates,
        choice_spans=choice_spans,
        options_count=len(example.choices),
    )
    if selected_display_id is None:
        selected_display_id = 1
    selected_candidate = display_id_to_candidate[selected_display_id]
    answer_frames = per_candidate_frames[selected_display_id]
    answer_times = per_candidate_times[selected_display_id]
    answer_prompt = _build_answer_only_prompt(
        question=example.question,
        labeled_options=labeled_options,
        selected_candidate=selected_candidate,
    )
    if choice_spans:
        selected_interval = {
            "start_time_sec": float(selected_candidate["start_time_sec"]),
            "end_time_sec": float(selected_candidate["end_time_sec"]),
        }
        selected_compare = compare_time_tool(
            {
                "candidates": [selected_interval],
                "references": choice_spans,
                "tolerance_sec": 10.0,
            }
        )
        nearest = selected_compare["best_match"]
        option_label = chr(ord("A") + int(nearest["nearest_reference_index"])) if nearest["nearest_reference_index"] is not None else None
        answer_prompt += (
            "\n\nTemporal tool summary for the selected candidate:\n"
            f"- nearest option: {option_label}\n"
            f"- nearest gap: {nearest['nearest_gap_sec']}\n"
            f"- nearest overlap: {nearest['nearest_overlap_sec']}\n"
            "- Use this as a consistency hint, not as an automatic answer."
        )
    answer_generation = answerer.generate_text_from_frames(
        frames=answer_frames,
        prompt=answer_prompt,
        frame_texts=_build_mcq_style_frame_labels(answer_times),
    )
    selected_letter = _parse_final_answer_letter(answer_generation.raw_text, options_count=len(example.choices))
    selected_choice_idx = (ord(selected_letter) - ord("A")) if selected_letter is not None else None
    selected_interval = _selected_option_interval(example, selected_letter)
    consistency = None
    if selected_interval is not None:
        consistency = verify_consistency_tool(
            {
                "selected_interval": {
                    "start_time_sec": float(selected_candidate["start_time_sec"]),
                    "end_time_sec": float(selected_candidate["end_time_sec"]),
                },
                "answer_interval": selected_interval,
                "tolerance_sec": 10.0,
            }
        )

    candidate_pool_metrics = summarize_layer2_hits(
        layer2_hits=candidate_hits,
        gold_spans=gold_spans,
        top_k=len(candidate_hits),
    ) if gold_spans else {}
    shortlist_metrics = summarize_layer2_hits(
        layer2_hits=shortlisted_candidates,
        gold_spans=gold_spans,
        top_k=len(shortlisted_candidates),
    ) if gold_spans else {}
    selected_metrics = summarize_layer2_hits(
        layer2_hits=[selected_candidate],
        gold_spans=gold_spans,
        top_k=1,
    ) if gold_spans else {}

    return {
        "example_id": example.example_id,
        "task_name": example.task_name,
        "mcq_correct": 1.0 if selected_choice_idx == example.correct_idx else 0.0,
        "raw_answer": answer_generation.raw_text,
        "query_text": query_text,
        "candidate_pool_hit_any": float(candidate_pool_metrics[f"Layer2 Hit@{len(candidate_hits)}_gap0"]) if gold_spans else None,
        "shortlist_hit_any": float(shortlist_metrics[f"Layer2 Hit@{len(shortlisted_candidates)}_gap0"]) if gold_spans else None,
        "selected_evidence_hit1": float(selected_metrics["Layer2 Hit@1_gap0"]) if gold_spans else None,
        "selected_evidence_hit1_tol_5s": _tolerance_hit(selected_candidate, gold_spans, 5.0),
        "selected_evidence_hit1_tol_10s": _tolerance_hit(selected_candidate, gold_spans, 10.0),
        "selected_evidence": selected_candidate,
        "selection_raw_answer": selection_generation.raw_text,
        "tool_parse": parse_query_tool({"question": example.question, "choices": example.choices}),
        "tool_temporal_compare": tool_records,
        "tool_consistency": consistency,
        "shortlist_scored_candidates": shortlist_scored_candidates,
        "total_frames_seen_per_question": float(len(selection_frames) + len(answer_frames)),
        "status": "ok",
    }


def _run_method(
    *,
    method_name: str,
    entries: list[dict[str, Any]],
    examples_by_id: dict[str, Any],
    output_dir: Path,
    budget_config: BudgetConfig,
    answer_config: Any,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "rows.jsonl"
    log_path = output_dir / "progress.log"
    debug_dir = output_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    answerer = QwenVLMAnswerer(answer_config)
    grouped = _example_groups(entries)
    rows: list[dict[str, Any]] = []
    try:
        for video_id, group_entries in grouped.items():
            log_line(log_path, f"[video_start] method={method_name} video={video_id} count={len(group_entries)}")
            archive = query_encoder = index_bundle = None
            if method_name == "tooling":
                archive, query_encoder = _build_query_encoder_and_archive(
                    repo_root=REPO_ROOT,
                    video_id=video_id,
                    device=PIPELINE_CONFIG.retrieval.device,
                )
                index_bundle = _build_segment_index(archive=archive)
            for index, entry in enumerate(group_entries, start=1):
                example = examples_by_id[entry["example_id"]]
                if method_name == "pure_vlm":
                    row = _run_pure_example(
                        example=example,
                        video_id=video_id,
                        answerer=answerer,
                        budget_config=budget_config,
                    )
                else:
                    row = _run_tooling_example(
                        example=example,
                        video_id=video_id,
                        answerer=answerer,
                        budget_config=budget_config,
                        query_encoder=query_encoder,
                        archive=archive,
                        index_bundle=index_bundle,
                    )
                row.update(
                    {
                        "evaluation_video_id": video_id,
                        "query_family": entry["query_family"],
                        "length_bucket": entry["length_bucket"],
                        "method_name": method_name,
                    }
                )
                rows.append(row)
                append_jsonl(rows_path, row)
                (debug_dir / f"{len(rows):03d}_{entry['example_id']}.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
                log_line(
                    log_path,
                    f"[progress] method={method_name} video={video_id} row={index}/{len(group_entries)} "
                    f"example_id={entry['example_id']} mcq={row.get('mcq_correct')}",
                )
            if archive is not None:
                del archive
            if query_encoder is not None:
                del query_encoder
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        answerer.unload()
    return rows


def main() -> None:
    args = _build_parser().parse_args()
    manifest = _load_manifest(args.split)
    selected_tasks = set(args.tasks) if args.tasks else None
    entries = _selected_entries(manifest, tasks=selected_tasks, max_examples_per_task=args.max_examples_per_task)
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    examples_by_id = _load_examples_by_id(sorted({entry["task_name"] for entry in entries}))
    budget_config = BudgetConfig(total_frames=16)
    answer_config = default_answer_config()
    answer_config.max_new_tokens = 192
    started_at = time.perf_counter()
    rows_by_method: dict[str, list[dict[str, Any]]] = {}

    for method_name in args.methods:
        method_output = output_root / method_name
        rows_by_method[method_name] = _run_method(
            method_name=method_name,
            entries=entries,
            examples_by_id=examples_by_id,
            output_dir=method_output,
            budget_config=budget_config,
            answer_config=answer_config,
        )

    final_payload = {
        "split_name": manifest["split_name"],
        "elapsed_sec": round(time.perf_counter() - started_at, 3),
        "methods": {method_name: _row_summary(rows, method_name=method_name) for method_name, rows in rows_by_method.items()},
        "per_task": {
            method_name: _build_group_summaries(rows, "task_name")
            for method_name, rows in rows_by_method.items()
        },
        "per_query_family": {
            method_name: _build_group_summaries(rows, "query_family")
            for method_name, rows in rows_by_method.items()
        },
        "per_length_bucket": {
            method_name: _build_group_summaries(rows, "length_bucket")
            for method_name, rows in rows_by_method.items()
        },
        "paired_vs_baseline": _paired_counts(rows_by_method),
    }
    write_json(output_root / "final_summary.json", final_payload)


if __name__ == "__main__":
    main()
