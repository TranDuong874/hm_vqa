from __future__ import annotations

import json
import os
import sys
from pathlib import Path
import re

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from evals.hd_epic.loader import filter_examples_for_video, load_examples
from evals.hd_epic.temporal import gold_spans_for_video
from pipeline.experiments.hd_epic_mcq_shortlist_joint import (
    _build_query_encoder_and_archive,
    _sample_candidate_frames,
)


VIDEO_ID = os.getenv("VIDEO_ID", "P01-20240203-135502")
RUN_DIR = Path(
    os.getenv(
        "RUN_DIR",
        str(
            REPO_ROOT
            / "results"
            / "pipeline"
            / "analysis"
            / "mcq_split_final_stage_think4_l1"
            / VIDEO_ID
        ),
    )
)
OUTPUT_DIR = Path(
    os.getenv(
        "OUTPUT_DIR",
        str(RUN_DIR / "inspection"),
    )
)
TASKS = ["fine_grained_action_localization", "recipe_step_localization"]
FRAME_SAMPLER = os.getenv("FRAME_SAMPLER", "l1_keyframes").strip().lower()
ONLY_WRONG = os.getenv("ONLY_WRONG", "0").lower() in {"1", "true", "yes", "on"}
DEVICE = os.getenv("DEVICE", "cpu").strip().lower()
FILTER_MODE = os.getenv("FILTER_MODE", "all").strip().lower()


def _load_rows(rows_path: Path) -> list[dict]:
    rows = []
    if not rows_path.exists():
        raise FileNotFoundError(rows_path)
    with rows_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _sanitize_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)


def _extract_action_name(question: str, fallback: str) -> str:
    match = re.search(r"<([^>]+)>", question)
    if match:
        return match.group(1).strip()
    return fallback.strip()


def _format_range_name(start_time_sec: float, end_time_sec: float) -> str:
    return f"{start_time_sec:010.3f}s_to_{end_time_sec:010.3f}s"


def _read_debug_text(run_dir: Path, row_index: int, example_id: str) -> str:
    debug_path = run_dir / "debug" / f"{row_index:03d}_{example_id}.txt"
    if not debug_path.exists():
        return ""
    return debug_path.read_text(encoding="utf-8")


def _extract_debug_section(text: str, header: str, next_headers: list[str]) -> str:
    start = text.find(f"{header}\n")
    if start < 0:
        return ""
    start += len(header) + 1
    end = len(text)
    for next_header in next_headers:
        marker = f"\n{next_header}\n"
        pos = text.find(marker, start)
        if pos >= 0:
            end = min(end, pos)
    return text[start:end].strip()


def _extract_prompt_sections(debug_text: str) -> tuple[str, str]:
    visual_prompt = _extract_debug_section(
        debug_text,
        "visual_prompt:",
        ["visual_frame_texts:", "visual_raw_answer:", "baseline_prompt:"],
    )
    if visual_prompt:
        return visual_prompt, visual_prompt

    selection_prompt = _extract_debug_section(
        debug_text,
        "selection_prompt:",
        ["selection_frame_texts:", "selection_raw_answer:", "answer_prompt:", "answer_frame_texts:"],
    )
    answer_prompt = _extract_debug_section(
        debug_text,
        "answer_prompt:",
        ["answer_frame_texts:", "answer_raw_answer:", "baseline_prompt:"],
    )
    return selection_prompt, answer_prompt


def main() -> None:
    rows = _load_rows(RUN_DIR / "rows.jsonl")
    examples = {
        example.example_id: example
        for example in filter_examples_for_video(load_examples(TASKS, REPO_ROOT), VIDEO_ID)
    }
    video_path = REPO_ROOT / "dataset" / "hd_epic_structured" / VIDEO_ID / "video.mp4"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    archive = None
    query_encoder = None
    try:
        archive, query_encoder = _build_query_encoder_and_archive(
            repo_root=REPO_ROOT,
            video_id=VIDEO_ID,
            device=DEVICE,
        )
        exported = []
        for row_idx, row in enumerate(rows, start=1):
            if row.get("status") != "ok":
                continue
            if FILTER_MODE == "mcq_wrong":
                if float(row.get("mcq_correct", 0.0)) != 0.0:
                    continue
            elif FILTER_MODE == "shortlist_hit_selected_miss":
                if float(row.get("shortlist_hit_any", 0.0)) != 1.0:
                    continue
                if float(row.get("selected_evidence_hit1", 0.0)) != 0.0:
                    continue
            elif ONLY_WRONG and float(row.get("mcq_correct", 0.0)) != 0.0:
                continue
            example = examples.get(row["example_id"])
            if example is None:
                continue

            query_embedding = query_encoder.encode_texts([row["query_text"]], batch_size=1)[0]
            candidate_by_id = {int(candidate["candidate_id"]): candidate for candidate in row["candidate_pool"]}
            shortlist_ids = [int(value) for value in row["shortlist_ids"]]
            shortlisted_candidates = [candidate_by_id[candidate_id] for candidate_id in shortlist_ids if candidate_id in candidate_by_id]
            gold_spans = list(gold_spans_for_video(example, VIDEO_ID))

            action_name = _sanitize_filename(_extract_action_name(row["question"], row["query_text"]))
            selected = row.get("selected_evidence") or {}
            selected_start = float(selected.get("start_time_sec", 0.0))
            selected_end = float(selected.get("end_time_sec", 0.0))
            case_dir = OUTPUT_DIR / f"{row_idx:03d}_{action_name}_{_sanitize_filename(row['example_id'])}"
            case_dir.mkdir(parents=True, exist_ok=True)
            debug_text = _read_debug_text(RUN_DIR, row_idx, row["example_id"])
            selection_prompt, answer_prompt = _extract_prompt_sections(debug_text)

            manifest = {
                "example_id": row["example_id"],
                "question": row["question"],
                "query_text": row["query_text"],
                "correct_idx": row["correct_idx"],
                "mcq_correct": row["mcq_correct"],
                "selected_evidence_hit1": row["selected_evidence_hit1"],
                "shortlist_hit_any": row["shortlist_hit_any"],
                "candidate_pool_hit_any": row["candidate_pool_hit_any"],
                "shortlist_display_mapping": row.get("shortlist_display_mapping", {}),
                "shortlist_selected_display_id": row.get("shortlist_selected_display_id"),
                "shortlist_selected_choice_letter": row.get("shortlist_selected_choice_letter"),
                "selection_raw_answer": row.get("selection_raw_answer", ""),
                "answer_raw_answer": row.get("answer_raw_answer", ""),
                "selection_prompt": selection_prompt,
                "answer_prompt": answer_prompt,
                "gold_spans": gold_spans,
                "candidates": [],
                "gold_samples": [],
            }

            for display_index, candidate in enumerate(shortlisted_candidates, start=1):
                frames, timestamps_sec = _sample_candidate_frames(
                    sampler=FRAME_SAMPLER,
                    video_path=video_path,
                    candidate=candidate,
                    frame_budget=int(row.get("frames_per_clip", 4)),
                    archive=archive,
                    query_embedding=query_embedding,
                )
                candidate_dir = case_dir / (
                    f"c{display_index}_{_sanitize_filename(_format_range_name(float(candidate['start_time_sec']), float(candidate['end_time_sec'])))}"
                )
                candidate_dir.mkdir(parents=True, exist_ok=True)
                candidate_entry = {
                    "display_index": display_index,
                    "candidate": candidate,
                    "frames": [],
                }
                for frame_index, (frame, timestamp_sec) in enumerate(zip(frames, timestamps_sec), start=1):
                    filename = (
                        f"frame_{frame_index:02d}_"
                        f"{timestamp_sec:010.3f}s.jpg"
                    )
                    frame_path = candidate_dir / filename
                    frame.save(frame_path, format="JPEG", quality=95)
                    candidate_entry["frames"].append(
                        {
                            "frame_index": frame_index,
                            "timestamp_sec": float(timestamp_sec),
                            "path": str(frame_path),
                        }
                    )
                manifest["candidates"].append(candidate_entry)

            for gold_index, gold_span in enumerate(gold_spans, start=1):
                gold_candidate = {
                    "candidate_id": gold_index,
                    "start_time_sec": float(gold_span["start_time_sec"]),
                    "end_time_sec": float(gold_span["end_time_sec"]),
                    "video_id": gold_span["video_id"],
                }
                gold_frames, gold_timestamps_sec = _sample_candidate_frames(
                    sampler=FRAME_SAMPLER,
                    video_path=video_path,
                    candidate=gold_candidate,
                    frame_budget=int(row.get("frames_per_clip", 4)),
                    archive=archive,
                    query_embedding=query_embedding,
                )
                gold_dir = case_dir / (
                    f"gold_{action_name}_{_sanitize_filename(_format_range_name(gold_candidate['start_time_sec'], gold_candidate['end_time_sec']))}"
                )
                gold_dir.mkdir(parents=True, exist_ok=True)
                gold_entry = {
                    "gold_index": gold_index,
                    "gold_span": gold_span,
                    "frames": [],
                }
                for frame_index, (frame, timestamp_sec) in enumerate(zip(gold_frames, gold_timestamps_sec), start=1):
                    filename = (
                        f"frame_{frame_index:02d}_"
                        f"{timestamp_sec:010.3f}s.jpg"
                    )
                    frame_path = gold_dir / filename
                    frame.save(frame_path, format="JPEG", quality=95)
                    gold_entry["frames"].append(
                        {
                            "frame_index": frame_index,
                            "timestamp_sec": float(timestamp_sec),
                            "path": str(frame_path),
                        }
                    )
                manifest["gold_samples"].append(gold_entry)

            (case_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            prompt_text = (
                f"Question: {row['question']}\n"
                f"Correct idx: {row['correct_idx']}\n"
                f"Selection prompt:\n{selection_prompt}\n\n"
                f"Answer prompt:\n{answer_prompt}\n"
            )
            (case_dir / "prompt.txt").write_text(prompt_text, encoding="utf-8")
            (case_dir / "meta.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            exported.append(str(case_dir))

        index = {
            "video_id": VIDEO_ID,
            "run_dir": str(RUN_DIR),
            "output_dir": str(OUTPUT_DIR),
            "only_wrong": ONLY_WRONG,
            "filter_mode": FILTER_MODE,
            "frame_sampler": FRAME_SAMPLER,
            "exported_cases": exported,
        }
        (OUTPUT_DIR / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
        print(str(OUTPUT_DIR))
        print(f"exported_cases={len(exported)}")
    finally:
        if query_encoder is not None:
            del query_encoder


if __name__ == "__main__":
    main()
