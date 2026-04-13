from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from answering import QwenVLMAnswerer
from evals.hd_epic.loader import filter_examples_for_video, load_examples
from evals.hd_epic.runner import (
    _build_frame_timestamp_labels,
    _build_timestamped_mcq_prompt,
    _sanitize_prompt_text,
    _sample_uniform_video_frames,
)
from pipeline.experiments.hd_epic_mcq_shortlist_joint import (
    _build_answer_only_prompt,
    _build_mcq_style_frame_labels,
    _build_query_encoder_and_archive,
    _build_segment_index,
    _choice_spans_for_example,
    _parse_final_answer_letter,
    _sample_candidate_frames,
    MIN_KEYFRAME_GAP_SEC,
)
from pipeline.config import PIPELINE_CONFIG
from pipeline.core.io import append_jsonl, log_line
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


VIDEO_ID = os.getenv("VIDEO_ID", "P01-20240203-135502")
TASKS = [
    "fine_grained_action_localization",
    "recipe_step_localization",
]
LIMIT = int(os.getenv("LIMIT", "20"))
OUTPUT_TAG = os.getenv("OUTPUT_TAG", "agentic_selection")
FRAMES_PER_CANDIDATE = int(os.getenv("FRAMES_PER_CANDIDATE", "4"))
MAX_AGENT_LOOPS = int(os.getenv("MAX_AGENT_LOOPS", "4"))
THINKING_MAX_NEW_TOKENS = int(os.getenv("THINKING_MAX_NEW_TOKENS", "192"))


def _selection_prompt(
    *,
    question: str,
    labeled_options: list[str],
    candidates: list[dict[str, Any]],
    scratchpad: list[str],
    loop_index: int,
    tool_called: bool,
) -> str:
    candidate_lines = []
    for display_index, candidate in enumerate(candidates, start=1):
        candidate_lines.append(
            f"Candidate {display_index}: from "
            f"{candidate['start_time_sec']:.3f}s to {candidate['end_time_sec']:.3f}s"
        )
    tool_block = "\n".join(
        [
            "Available tools:",
            "- CALL PARSE_QUERY",
            "- CALL COMPARE_TIME",
            "- CALL VERIFY_CONSISTENCY CANDIDATE n OPTION X",
            "",
            "Rules:",
            "- Call at most one tool per turn.",
            "- On the first turn, you must call a tool.",
            "- Do not emit FINAL CANDIDATE before at least one tool call has been made.",
            "- After at least one tool call, finish as soon as you have enough evidence.",
            "- Finish with exactly one line: FINAL CANDIDATE: n",
            "- Do not answer the MCQ yet.",
        ]
    )
    state_line = f"Current turn: {loop_index}/{MAX_AGENT_LOOPS}. Tool already called: {'yes' if tool_called else 'no'}."
    history = "\n".join(scratchpad).strip()
    if history:
        history = f"\n\nCurrent reasoning state:\n{history}"
    return (
        "You are selecting the best candidate clip for a temporal video MCQ.\n"
        "Use the visual evidence, timestamps, and tools to decide which candidate most plausibly matches the option times.\n\n"
        f"Question: {_sanitize_prompt_text(question)}\n"
        "Options:\n"
        + "\n".join(labeled_options)
        + "\n\nCandidates:\n"
        + "\n".join(candidate_lines)
        + "\n\n"
        + state_line
        + "\n\n"
        + tool_block
        + history
    )


def _parse_agent_call(raw_text: str, *, max_candidate: int, options_count: int) -> tuple[str, dict[str, Any]] | None:
    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        verify_match = re.search(
            r"CALL\s+VERIFY_CONSISTENCY\s+CANDIDATE\s+([1-9][0-9]*)\s+OPTION\s+([A-Z])",
            line,
            flags=re.IGNORECASE,
        )
        if verify_match:
            candidate_id = int(verify_match.group(1))
            option_letter = verify_match.group(2).upper()
            if 1 <= candidate_id <= max_candidate and 0 <= (ord(option_letter) - ord("A")) < options_count:
                return ("VERIFY_CONSISTENCY", {"candidate_id": candidate_id, "option_letter": option_letter})
        if re.search(r"CALL\s+PARSE_QUERY", line, flags=re.IGNORECASE):
            return ("PARSE_QUERY", {})
        if re.search(r"CALL\s+COMPARE_TIME", line, flags=re.IGNORECASE):
            return ("COMPARE_TIME", {})
        final_match = re.search(r"FINAL\s+CANDIDATE\s*:\s*([1-9][0-9]*)", line, flags=re.IGNORECASE)
        if final_match:
            value = int(final_match.group(1))
            if 1 <= value <= max_candidate:
                return ("FINAL", {"candidate_id": value})
    return None


def _run_agentic_selection(
    *,
    answerer: QwenVLMAnswerer,
    frames: list[Any],
    frame_texts: list[str],
    question: str,
    labeled_options: list[str],
    shortlisted_candidates: list[dict[str, Any]],
    choice_spans: list[dict[str, Any]],
) -> tuple[int, list[dict[str, Any]], list[str]]:
    scratchpad: list[str] = []
    tool_trace: list[dict[str, Any]] = []
    tool_called = False

    for loop_index in range(1, MAX_AGENT_LOOPS + 1):
        prompt = _selection_prompt(
            question=question,
            labeled_options=labeled_options,
            candidates=shortlisted_candidates,
            scratchpad=scratchpad,
            loop_index=loop_index,
            tool_called=tool_called,
        )
        generation = answerer.generate_text_from_frames(
            frames=frames,
            prompt=prompt,
            frame_texts=frame_texts,
        )
        raw_text = generation.raw_text
        parsed = _parse_agent_call(
            raw_text,
            max_candidate=len(shortlisted_candidates),
            options_count=len(labeled_options),
        )
        tool_trace.append(
            {
                "loop": loop_index,
                "prompt": prompt,
                "raw_answer": raw_text,
            }
        )
        if parsed is None:
            scratchpad.append(f"Loop {loop_index}: unparsable response -> default to nearest candidate later.")
            continue
        action, payload = parsed
        if action == "FINAL":
            if not tool_called:
                scratchpad.append(
                    f"Loop {loop_index}: ignored FINAL because at least one tool call is required before finalizing."
                )
                continue
            return int(payload["candidate_id"]), tool_trace, scratchpad
        if action == "PARSE_QUERY":
            tool_output = parse_query_tool({"question": question, "choices": labeled_options})
            scratchpad.append(f"Loop {loop_index} tool PARSE_QUERY:\n{tool_output}")
            tool_trace[-1]["tool_name"] = "PARSE_QUERY"
            tool_trace[-1]["tool_output"] = tool_output
            tool_called = True
            continue
        if action == "COMPARE_TIME":
            tool_output = compare_time_tool(
                {
                    "candidates": [
                        {
                            "start_time_sec": float(candidate["start_time_sec"]),
                            "end_time_sec": float(candidate["end_time_sec"]),
                        }
                        for candidate in shortlisted_candidates
                    ],
                    "references": choice_spans,
                    "tolerance_sec": 10.0,
                }
            )
            scratchpad.append(f"Loop {loop_index} tool COMPARE_TIME:\n{tool_output}")
            tool_trace[-1]["tool_name"] = "COMPARE_TIME"
            tool_trace[-1]["tool_output"] = tool_output
            tool_called = True
            continue
        if action == "VERIFY_CONSISTENCY":
            candidate = shortlisted_candidates[int(payload["candidate_id"]) - 1]
            option_index = ord(payload["option_letter"]) - ord("A")
            choice_span = choice_spans[option_index]
            tool_output = verify_consistency_tool(
                {
                    "selected_interval": {
                        "start_time_sec": float(candidate["start_time_sec"]),
                        "end_time_sec": float(candidate["end_time_sec"]),
                    },
                    "answer_interval": choice_span,
                    "tolerance_sec": 10.0,
                }
            )
            scratchpad.append(
                f"Loop {loop_index} tool VERIFY_CONSISTENCY Candidate {payload['candidate_id']} "
                f"Option {payload['option_letter']}:\n{tool_output}"
            )
            tool_trace[-1]["tool_name"] = "VERIFY_CONSISTENCY"
            tool_trace[-1]["tool_output"] = tool_output
            tool_called = True
            continue

    fallback_compare = compare_time_tool(
        {
            "candidates": [
                {
                    "start_time_sec": float(candidate["start_time_sec"]),
                    "end_time_sec": float(candidate["end_time_sec"]),
                }
                for candidate in shortlisted_candidates
            ],
            "references": choice_spans,
            "tolerance_sec": 10.0,
        }
    )
    best_index = int(fallback_compare["best_match"]["candidate_index"]) + 1
    scratchpad.append(f"Fallback after {MAX_AGENT_LOOPS} loops -> Candidate {best_index}")
    return best_index, tool_trace, scratchpad


def main() -> None:
    output_dir = REPO_ROOT / "results" / "pipeline" / "analysis" / f"mcq_{OUTPUT_TAG}" / VIDEO_ID
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "rows.jsonl"
    log_path = output_dir / "progress.log"
    debug_dir = output_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.perf_counter()

    budget_config = BudgetConfig()
    budget_config.frames_per_candidate = FRAMES_PER_CANDIDATE
    answer_config = default_answer_config()
    answer_config.max_new_tokens = THINKING_MAX_NEW_TOKENS
    answerer = QwenVLMAnswerer(answer_config)
    rows: list[dict[str, Any]] = []
    video_path = REPO_ROOT / "dataset" / "data" / "HD-EPIC" / "Videos" / VIDEO_ID.split("-", 1)[0] / f"{VIDEO_ID}.mp4"
    config_payload = {"tasks": TASKS, "limit": LIMIT, "pipeline": PIPELINE_CONFIG.to_dict()}

    examples = filter_examples_for_video(load_examples(TASKS, REPO_ROOT), VIDEO_ID)
    examples = [example for example in examples if example.answer_type == "temporal_option" and example.gold_spans][:LIMIT]

    archive = None
    query_encoder = None
    try:
        log_line(log_path, f"[start] method=ours_agentic_selection video={VIDEO_ID} limit={LIMIT}")
        archive, query_encoder = _build_query_encoder_and_archive(
            repo_root=REPO_ROOT,
            video_id=VIDEO_ID,
            device=PIPELINE_CONFIG.retrieval.device,
        )
        index_bundle = _build_segment_index(archive=archive)

        _write_status(
            rows=rows,
            total_examples=len(examples),
            started_at=started_at,
            status="running",
            video_id=VIDEO_ID,
            method_name="ours_agentic_selection",
            output_dir=output_dir,
            budget_config=budget_config,
            answer_config=answer_config,
            config_payload=config_payload,
        )

        for index, example in enumerate(examples, start=1):
            query_text = extract_target_text(example.question)
            query_embedding = query_encoder.encode_texts(
                [query_text],
                batch_size=PIPELINE_CONFIG.retrieval.openclip_batch_size,
            )[0]
            gold_spans = gold_spans_for_video(example, VIDEO_ID)
            scope_start_sec, scope_end_sec = example_scope_for_video(example, VIDEO_ID)
            _layer3_hits, candidate_hits = _candidate_pool_hierarchical(
                query_embedding=query_embedding,
                video_id=VIDEO_ID,
                scope_start_sec=scope_start_sec,
                scope_end_sec=scope_end_sec,
                index_bundle=index_bundle,
            )
            if not candidate_hits:
                row = {
                    "example_id": example.example_id,
                    "task_name": example.task_name,
                    "question": example.question,
                    "query_text": query_text,
                    "status": "error",
                    "message": "No candidate hits produced.",
                }
                append_jsonl(rows_path, row)
                rows.append(row)
                continue

            labeled_options = _prompt_choices(example)
            choice_spans = _choice_spans_for_example(example)
            shortlist_ids, shortlist_scored_candidates = shortlist_candidates_cover_choices(
                candidates=candidate_hits,
                choice_spans=choice_spans,
                max_keep=budget_config.shortlist_k,
            )
            shortlisted_candidates = [candidate for candidate in candidate_hits if candidate["candidate_id"] in shortlist_ids]

            selection_frames = []
            selection_frame_texts = []
            per_candidate_frames: dict[int, list[Any]] = {}
            per_candidate_times: dict[int, list[float]] = {}
            for display_index, candidate in enumerate(shortlisted_candidates, start=1):
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

            selected_display_id, tool_trace, scratchpad = _run_agentic_selection(
                answerer=answerer,
                frames=selection_frames,
                frame_texts=selection_frame_texts,
                question=example.question,
                labeled_options=labeled_options,
                shortlisted_candidates=shortlisted_candidates,
                choice_spans=choice_spans,
            )
            selected_candidate = shortlisted_candidates[selected_display_id - 1]
            answer_frames = per_candidate_frames[selected_display_id]
            answer_times = per_candidate_times[selected_display_id]
            answer_prompt = _build_answer_only_prompt(
                question=example.question,
                labeled_options=labeled_options,
                selected_candidate=selected_candidate,
            )
            answer_generation = answerer.generate_text_from_frames(
                frames=answer_frames,
                prompt=answer_prompt,
                frame_texts=_build_mcq_style_frame_labels(answer_times),
            )
            selected_letter = _parse_final_answer_letter(answer_generation.raw_text, options_count=len(example.choices))
            selected_choice_idx = (ord(selected_letter) - ord("A")) if selected_letter is not None else None

            baseline_hit = candidate_hits[0]
            baseline_frames, _, baseline_timestamps, _ = _sample_uniform_video_frames(
                video_path=video_path,
                frame_budget=budget_config.baseline_answer_frame_budget,
                start_time_sec=float(baseline_hit["start_time_sec"]),
                end_time_sec=float(baseline_hit["end_time_sec"]),
            )
            baseline_prompt = _build_timestamped_mcq_prompt(
                question=example.question,
                labeled_options=labeled_options,
                clip_start_sec=float(baseline_hit["start_time_sec"]),
                clip_end_sec=float(baseline_hit["end_time_sec"]),
            )
            baseline_generation = answerer.generate_text_from_frames(
                frames=baseline_frames,
                prompt=baseline_prompt,
                frame_texts=_build_frame_timestamp_labels(baseline_timestamps),
            )
            baseline_letter = _parse_final_answer_letter(baseline_generation.raw_text, options_count=len(example.choices))
            baseline_choice_idx = (ord(baseline_letter) - ord("A")) if baseline_letter is not None else None

            baseline_metrics = summarize_layer2_hits(layer2_hits=[baseline_hit], gold_spans=gold_spans, top_k=1)
            selected_metrics = summarize_layer2_hits(layer2_hits=[selected_candidate], gold_spans=gold_spans, top_k=1)
            shortlist_metrics = summarize_layer2_hits(
                layer2_hits=shortlisted_candidates,
                gold_spans=gold_spans,
                top_k=len(shortlisted_candidates),
            )
            candidate_pool_metrics = summarize_layer2_hits(
                layer2_hits=candidate_hits,
                gold_spans=gold_spans,
                top_k=len(candidate_hits),
            )

            row = {
                "example_id": example.example_id,
                "task_name": example.task_name,
                "question": example.question,
                "query_text": query_text,
                "correct_idx": int(example.correct_idx),
                "candidate_pool_hit_any": float(candidate_pool_metrics[f"Layer2 Hit@{len(candidate_hits)}_gap0"]),
                "shortlist_hit_any": float(shortlist_metrics[f"Layer2 Hit@{len(shortlisted_candidates)}_gap0"]),
                "selected_evidence_hit1": float(selected_metrics["Layer2 Hit@1_gap0"]),
                "selected_evidence": selected_candidate,
                "mcq_correct": 1.0 if selected_choice_idx == example.correct_idx else 0.0,
                "baseline_top1_mcq_correct": 1.0 if baseline_choice_idx == example.correct_idx else 0.0,
                "baseline_top1_l2_hit1": float(baseline_metrics["Layer2 Hit@1_gap0"]),
                "shortlist_candidate_valid": 1.0,
                "shortlist_answer_valid": 1.0 if selected_letter is not None else 0.0,
                "shortlist_size": float(len(shortlisted_candidates)),
                "total_frames_seen_per_question": float(len(selection_frames) * min(MAX_AGENT_LOOPS, len(tool_trace)) + len(answer_frames)),
                "num_candidate_clips_seen": float(len(shortlisted_candidates)),
                "frames_per_clip": float(budget_config.frames_per_candidate),
                "selected_display_id": selected_display_id,
                "selection_tool_trace": tool_trace,
                "selection_scratchpad": scratchpad,
                "answer_raw_answer": answer_generation.raw_text,
                "status": "ok",
            }
            append_jsonl(rows_path, row)
            rows.append(row)

            debug_text = (
                f"example_id: {example.example_id}\n"
                f"question: {example.question}\n"
                f"query_text: {query_text}\n"
                f"correct_idx: {example.correct_idx}\n\n"
                f"choice_spans:\n" + "\n".join(str(span) for span in choice_spans) + "\n\n"
                f"candidate_pool:\n" + "\n".join(str(candidate) for candidate in candidate_hits) + "\n\n"
                f"shortlist_scored_candidates:\n" + "\n".join(str(item) for item in shortlist_scored_candidates) + "\n\n"
                f"shortlist_ids: {shortlist_ids}\n"
                f"selected_display_id: {selected_display_id}\n\n"
                f"selection_frame_texts:\n" + "\n".join(selection_frame_texts) + "\n\n"
                f"selection_tool_trace:\n" + "\n\n".join(str(item) for item in tool_trace) + "\n\n"
                f"selection_scratchpad:\n" + "\n".join(scratchpad) + "\n\n"
                f"answer_prompt:\n{answer_prompt}\n\n"
                f"answer_raw_answer:\n{answer_generation.raw_text}\n"
            )
            (debug_dir / f"{index:03d}_{example.example_id}.txt").write_text(debug_text, encoding="utf-8")
            log_line(
                log_path,
                f"[progress] row={index}/{len(examples)} example_id={example.example_id} "
                f"mcq={row['mcq_correct']:.0f} selected_hit={row['selected_evidence_hit1']:.0f}",
            )
            _write_status(
                rows=rows,
                total_examples=len(examples),
                started_at=started_at,
                status="running",
                video_id=VIDEO_ID,
                method_name="ours_agentic_selection",
                output_dir=output_dir,
                budget_config=budget_config,
                answer_config=answer_config,
                config_payload=config_payload,
            )

        _write_status(
            rows=rows,
            total_examples=len(examples),
            started_at=started_at,
            status="completed",
            video_id=VIDEO_ID,
            method_name="ours_agentic_selection",
            output_dir=output_dir,
            budget_config=budget_config,
            answer_config=answer_config,
            config_payload=config_payload,
        )
    finally:
        if query_encoder is not None:
            del query_encoder
        answerer.unload()


if __name__ == "__main__":
    main()
