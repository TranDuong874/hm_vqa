from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from answering import AnswerConfig, build_answerer, parse_choice_letter
from evals.hd_epic.loader import filter_examples_for_video, load_examples
from evals.hd_epic.runner import (
    _build_frame_timestamp_labels,
    _build_timestamped_mcq_prompt,
    _label_prompt_choices,
    _normalize_choice_for_prompt,
    _sanitize_prompt_text,
    _sample_uniform_video_frames,
)

from pipeline.config import PIPELINE_CONFIG
from pipeline.core.features import build_query_encoder, load_feature_archive
from pipeline.core.io import append_jsonl, log_line, write_json
from pipeline.core.metrics import summarize_layer2_hits, summarize_layer3_hits
from pipeline.core.retrieve import (
    extract_target_text,
    gather_segment_embeddings,
    rank_segments,
    restrict_segments_to_hits,
    select_ranked_hits,
)
from pipeline.core.segments import build_adaptive_layer3_segments, build_fixed_windows, mean_pool_segments
from evals.hd_epic.temporal import example_scope_for_video, gold_spans_for_video, parse_choice_spans


DEFAULT_VIDEO_IDS = [
    "P01-20240203-135502",
    "P01-20240204-142301",
    "P01-20240203-132119",
    "P01-20240204-121042",
    "P01-20240203-184214",
]

DEFAULT_TASKS = [
    "fine_grained_action_localization",
    "recipe_step_localization",
]


@dataclass(slots=True)
class BudgetConfig:
    total_frames: int = 20
    shortlist_k: int = 5
    frames_per_candidate: int = 4
    baseline_answer_frame_budget: int = 16
    frame_bundle_max_gap_sec: float = 5.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_answer_config() -> AnswerConfig:
    return AnswerConfig(
        model_id="Qwen/Qwen3-VL-2B-Instruct",
        max_new_tokens=192,
        load_in_4bit=True,
        load_in_8bit=False,
        image_max_size=224,
    )


def format_seconds_to_hhmmss(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    hours = total_ms // 3_600_000
    total_ms %= 3_600_000
    minutes = total_ms // 60_000
    total_ms %= 60_000
    secs = total_ms // 1000
    millis = total_ms % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def interval_distance_sec(
    start_a: float,
    end_a: float,
    start_b: float,
    end_b: float,
) -> float:
    if max(start_a, start_b) <= min(end_a, end_b):
        return 0.0
    if end_a < start_b:
        return start_b - end_a
    return start_a - end_b


def shortlist_candidates_cover_choices(
    *,
    candidates: list[dict[str, Any]],
    choice_spans: list[dict[str, float]],
    max_keep: int,
) -> tuple[list[int], list[dict[str, Any]]]:
    scored: list[dict[str, Any]] = []
    for candidate in candidates:
        distances: list[float] = []
        for span in choice_spans:
            distances.append(
                interval_distance_sec(
                    float(candidate["start_time_sec"]),
                    float(candidate["end_time_sec"]),
                    float(span["start_time_sec"]),
                    float(span["end_time_sec"]),
                )
            )
        best_distance = min(distances) if distances else float("inf")
        best_choice_index = distances.index(best_distance) if distances else -1
        scored.append(
            {
                "candidate_id": int(candidate["candidate_id"]),
                "best_choice_index": best_choice_index,
                "best_distance_sec": round(best_distance, 6),
                "choice_distances_sec": [round(distance, 6) for distance in distances],
                "retrieval_score": float(candidate["score"]),
            }
        )

    shortlist_ids: list[int] = []
    for choice_index in range(len(choice_spans)):
        matching = [item for item in scored if item["best_choice_index"] == choice_index]
        if not matching:
            continue
        matching.sort(
            key=lambda item: (
                item["choice_distances_sec"][choice_index],
                -item["retrieval_score"],
                item["candidate_id"],
            )
        )
        candidate_id = matching[0]["candidate_id"]
        if candidate_id not in shortlist_ids:
            shortlist_ids.append(candidate_id)
        if len(shortlist_ids) >= max_keep:
            break

    if len(shortlist_ids) < max_keep:
        scored.sort(key=lambda item: (item["best_distance_sec"], -item["retrieval_score"], item["candidate_id"]))
        for item in scored:
            if item["candidate_id"] not in shortlist_ids:
                shortlist_ids.append(item["candidate_id"])
            if len(shortlist_ids) >= max_keep:
                break

    return shortlist_ids, scored


def parse_joint_response(raw_text: str, *, max_candidate: int, options_count: int) -> tuple[int | None, str | None]:
    candidate = None
    candidate_match = re.search(
        r"SELECTED\s+CANDIDATE\s*:\s*(?:Candidate\s+)?([1-9][0-9]*)",
        raw_text,
        flags=re.IGNORECASE,
    )
    if candidate_match:
        value = int(candidate_match.group(1))
        if 1 <= value <= max_candidate:
            candidate = value
    letters = "".join(chr(ord("A") + i) for i in range(options_count))
    answer_match = re.search(
        rf"FINAL\s+ANSWER\s*:\s*(?:Option\s+)?([{letters}])\b",
        raw_text,
        flags=re.IGNORECASE,
    )
    final_letter = answer_match.group(1).upper() if answer_match else parse_choice_letter(raw_text, options_count=options_count)
    return candidate, final_letter


def build_visual_joint_prompt(*, question: str, labeled_options: list[str], candidates: list[dict[str, Any]]) -> str:
    candidate_lines = []
    for display_index, candidate in enumerate(candidates, start=1):
        candidate_lines.append(
            "Candidate "
            f"{display_index}: from {format_seconds_to_hhmmss(float(candidate['start_time_sec']))} "
            f"to {format_seconds_to_hhmmss(float(candidate['end_time_sec']))}"
        )
    return (
        "You are given timestamped evidence frames from shortlisted retrieved candidate clips.\n"
        "Each frame is preceded by its candidate id and timestamp.\n"
        "Choose the candidate clip that best matches the question and then answer the multiple-choice question.\n"
        "The final answer MUST be temporally consistent with the selected candidate clip.\n"
        "Candidate ids are the local shortlist ids shown below.\n"
        "Reply with exactly two lines and nothing else:\n"
        "SELECTED CANDIDATE: n\n"
        "FINAL ANSWER: X\n"
        "Example:\n"
        "SELECTED CANDIDATE: 2\n"
        "FINAL ANSWER: B\n\n"
        f"Question: {_sanitize_prompt_text(question)}\n"
        "Options:\n"
        + "\n".join(labeled_options)
        + "\n\nShortlisted candidates:\n"
        + "\n".join(candidate_lines)
    )


DEFAULT_MIN_KEYFRAME_GAP_SEC = 1.0
MIN_KEYFRAME_GAP_SEC = DEFAULT_MIN_KEYFRAME_GAP_SEC


def _build_selection_only_prompt(
    *,
    question: str,
    labeled_options: list[str],
    candidates: list[dict[str, Any]],
) -> str:
    candidate_lines = []
    for display_index, candidate in enumerate(candidates, start=1):
        candidate_lines.append(
            "Candidate "
            f"{display_index}: from {format_seconds_to_hhmmss(float(candidate['start_time_sec']))} "
            f"to {format_seconds_to_hhmmss(float(candidate['end_time_sec']))}"
        )
    return (
        "You are given timestamped evidence frames from shortlisted retrieved candidate clips.\n"
        "Each frame is preceded by its candidate id and timestamp.\n"
        "Choose which candidate clip is the best evidence for answering the question.\n"
        "Your choice must be temporally consistent with the option intervals and the shown frame timestamps.\n"
        "Do not select a candidate if its shown timestamps are clearly far from the plausible option times.\n"
        "Do not answer the multiple-choice question yet.\n"
        "Return only one line in this exact format and nothing else:\n"
        "SELECTED CANDIDATE: n\n\n"
        f"Question: {_sanitize_prompt_text(question)}\n"
        "Options:\n"
        + "\n".join(labeled_options)
        + "\n\nShortlisted candidates:\n"
        + "\n".join(candidate_lines)
    )


def _parse_hhmmss_to_sec(value: str) -> float | None:
    match = re.fullmatch(r"(\d{2}):(\d{2}):(\d{2})\.(\d{3})", value.strip())
    if not match:
        return None
    hours, minutes, seconds, millis = [int(group) for group in match.groups()]
    return float(hours * 3600 + minutes * 60 + seconds + millis / 1000.0)


def _map_interval_to_candidate(
    *,
    start_time_sec: float,
    end_time_sec: float,
    candidates: list[dict[str, Any]],
) -> int | None:
    if not candidates:
        return None
    best_index = min(
        range(len(candidates)),
        key=lambda idx: (
            interval_distance_sec(
                float(candidates[idx]["start_time_sec"]),
                float(candidates[idx]["end_time_sec"]),
                start_time_sec,
                end_time_sec,
            ),
            idx,
        ),
    )
    return best_index + 1


def _parse_selection_candidate(
    raw_text: str,
    *,
    candidates: list[dict[str, Any]],
    choice_spans: list[dict[str, Any]],
    options_count: int,
) -> int | None:
    max_candidate = len(candidates)
    for line in raw_text.splitlines():
        candidate_match = re.search(
            r"\bSELECTED\s+CANDIDATE\s*:\s*([1-9][0-9]*)\b",
            line,
            flags=re.IGNORECASE,
        )
        if candidate_match:
            value = int(candidate_match.group(1))
            if 1 <= value <= max_candidate:
                return value

    fallback_patterns = [
        r"\bSELECTED\s*:\s*([1-9][0-9]*)\b",
        r"\bCHOSEN\s*CANDIDATE\s*:\s*([1-9][0-9]*)\b",
        r"\bCANDIDATE\s*#?\s*([1-9][0-9]*)\b",
        r"\bANSWER\s*:\s*CANDIDATE\s*([1-9][0-9]*)\b",
    ]
    for pattern in fallback_patterns:
        matches = re.findall(pattern, raw_text, flags=re.IGNORECASE)
        if matches:
            value = int(matches[-1])
            if 1 <= value <= max_candidate:
                return value

    choice_letter = parse_choice_letter(raw_text, options_count=options_count)
    if choice_letter is not None:
        choice_index = ord(choice_letter) - ord("A")
        if 0 <= choice_index < len(choice_spans):
            span = choice_spans[choice_index]
            return _map_interval_to_candidate(
                start_time_sec=float(span["start_time_sec"]),
                end_time_sec=float(span["end_time_sec"]),
                candidates=candidates,
            )

    timestamp_matches = re.findall(r"\b\d{2}:\d{2}:\d{2}\.\d{3}\b", raw_text)
    if len(timestamp_matches) >= 2:
        start_time_sec = _parse_hhmmss_to_sec(timestamp_matches[0])
        end_time_sec = _parse_hhmmss_to_sec(timestamp_matches[1])
        if start_time_sec is not None and end_time_sec is not None:
            return _map_interval_to_candidate(
                start_time_sec=start_time_sec,
                end_time_sec=end_time_sec,
                candidates=candidates,
            )
    return None


def _fallback_display_id_from_distances(
    *,
    shortlist_scored_candidates: list[dict[str, Any]],
    candidate_id_to_display_id: dict[int, int],
) -> int | None:
    best_candidate_id = None
    best_distance = None
    for candidate in shortlist_scored_candidates:
        candidate_id = int(candidate.get("candidate_id", -1))
        distance = candidate.get("best_distance_sec")
        if distance is None:
            continue
        if best_distance is None or float(distance) < best_distance:
            best_distance = float(distance)
            best_candidate_id = candidate_id
    if best_candidate_id is None:
        return None
    return candidate_id_to_display_id.get(best_candidate_id)


def _sample_exact_video_frames(
    *,
    video_path: Path,
    target_times_sec: list[float],
) -> tuple[list[Any], list[float], list[int]]:
    import cv2
    import numpy as np
    from PIL import Image

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        capture.release()
        raise RuntimeError(f"Invalid FPS for {video_path}: {fps}")

    frames: list[Any] = []
    actual_times: list[float] = []
    frame_indices: list[int] = []
    try:
        for target_time_sec in target_times_sec:
            frame_index = max(0, int(round(float(target_time_sec) * fps)))
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            actual_index = int(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            actual_time_sec = actual_index / fps
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(np.asarray(frame_rgb)))
            actual_times.append(float(actual_time_sec))
            frame_indices.append(actual_index)
    finally:
        capture.release()
    return frames, actual_times, frame_indices


def _select_diverse_keyframe_times(
    *,
    query_embedding: torch.Tensor,
    archive: Any,
    candidate: dict[str, Any],
    frames_per_candidate: int,
    min_gap_sec: float,
) -> list[float]:
    start_time_sec = float(candidate["start_time_sec"])
    end_time_sec = float(candidate["end_time_sec"])
    if end_time_sec < start_time_sec:
        end_time_sec = start_time_sec

    scores = torch.matmul(archive.frame_embeddings, query_embedding).detach().cpu()
    candidate_frames: list[tuple[float, float]] = []
    for frame_index, timestamp in enumerate(archive.timestamps.tolist()):
        time_sec = float(timestamp)
        if time_sec < start_time_sec or time_sec > end_time_sec:
            continue
        candidate_frames.append((time_sec, float(scores[frame_index])))

    if not candidate_frames:
        return []

    ranked = sorted(candidate_frames, key=lambda item: item[1], reverse=True)
    selected_times: list[float] = []
    for time_sec, _score in ranked:
        if all(abs(time_sec - existing) >= min_gap_sec for existing in selected_times):
            selected_times.append(time_sec)
        if len(selected_times) >= frames_per_candidate:
            break

    if len(selected_times) < frames_per_candidate:
        selected_set = set(selected_times)
        for time_sec, _score in ranked:
            if time_sec not in selected_set:
                selected_times.append(time_sec)
                selected_set.add(time_sec)
            if len(selected_times) >= frames_per_candidate:
                break

    if len(selected_times) < frames_per_candidate:
        if end_time_sec <= start_time_sec:
            fill_times = [start_time_sec] * (frames_per_candidate - len(selected_times))
        else:
            step = (end_time_sec - start_time_sec) / max(frames_per_candidate - 1, 1)
            fill_times = [start_time_sec + step * index for index in range(frames_per_candidate)]
        for time_sec in fill_times:
            selected_times.append(float(time_sec))
            if len(selected_times) >= frames_per_candidate:
                break

    return sorted(selected_times[:frames_per_candidate])


def _sample_candidate_frames(
    *,
    sampler: str,
    video_path: Path,
    candidate: dict[str, Any],
    frame_budget: int,
    archive: Any,
    query_embedding: torch.Tensor,
) -> tuple[list[Any], list[float]]:
    if sampler == "l1_keyframes":
        keyframe_times = _select_diverse_keyframe_times(
            query_embedding=query_embedding,
            archive=archive,
            candidate=candidate,
            frames_per_candidate=frame_budget,
            min_gap_sec=DEFAULT_MIN_KEYFRAME_GAP_SEC,
        )
        frames, actual_times, _ = _sample_exact_video_frames(
            video_path=video_path,
            target_times_sec=keyframe_times,
        )
        return frames, actual_times

    frames, _, timestamps_sec, _ = _sample_uniform_video_frames(
        video_path=video_path,
        frame_budget=frame_budget,
        start_time_sec=float(candidate["start_time_sec"]),
        end_time_sec=float(candidate["end_time_sec"]),
    )
    return frames, [float(ts) for ts in timestamps_sec]


def _build_answer_only_prompt(
    *,
    question: str,
    labeled_options: list[str],
    selected_candidate: dict[str, Any],
) -> str:
    base = _build_timestamped_mcq_prompt(
        question=question,
        labeled_options=labeled_options,
        clip_start_sec=float(selected_candidate["start_time_sec"]),
        clip_end_sec=float(selected_candidate["end_time_sec"]),
    )
    return (
        base
        + "\nThe final answer MUST be temporally consistent with the shown frame timestamps.\n"
        + "Only choose an option whose time interval overlaps or is nearest to the shown timestamps.\n"
        + "If one option is far away in time from the shown timestamps, do not choose it.\n\n"
        + "Reason briefly about which option interval matches the shown frames and timestamps.\n"
        + "Finish with exactly this final line:\nFINAL ANSWER: X"
    )


def _parse_final_answer_letter(raw_text: str, *, options_count: int) -> str | None:
    letters = "".join(chr(ord("A") + i) for i in range(options_count))
    answer_match = re.search(
        rf"FINAL\s+ANSWER\s*:\s*([{letters}])\b",
        raw_text,
        flags=re.IGNORECASE,
    )
    if answer_match:
        return answer_match.group(1).upper()
    return parse_choice_letter(raw_text, options_count=options_count)


def _build_mcq_style_frame_labels(timestamps_sec: list[float]) -> list[str]:
    return [
        f"Frame {index + 1} timestamp: {format_seconds_to_hhmmss(float(time_sec))}"
        for index, time_sec in enumerate(timestamps_sec)
    ]


def _serialize_answer_config(config: AnswerConfig) -> dict[str, Any]:
    return asdict(config)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 6)


def _mean_optional(values: list[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    if not present:
        return None
    return round(sum(present) / len(present), 6)


def _answer_accuracy_given_hit(rows: list[dict[str, Any]], *, hit_key: str, answer_key: str) -> float | None:
    hit_rows: list[dict[str, Any]] = []
    for row in rows:
        if row.get("status") != "ok":
            continue
        value = row.get(hit_key)
        if value is None:
            continue
        if float(value) > 0.0:
            hit_rows.append(row)
    if not hit_rows:
        return None
    return round(sum(float(row[answer_key]) for row in hit_rows) / len(hit_rows), 6)


def _answer_accuracy_given_miss(rows: list[dict[str, Any]], *, hit_key: str, answer_key: str) -> float | None:
    miss_rows: list[dict[str, Any]] = []
    for row in rows:
        if row.get("status") != "ok":
            continue
        value = row.get(hit_key)
        if value is None:
            continue
        if float(value) <= 0.0:
            miss_rows.append(row)
    if not miss_rows:
        return None
    return round(sum(float(row[answer_key]) for row in miss_rows) / len(miss_rows), 6)


def _grounded_confusion_counts(
    rows: list[dict[str, Any]],
    *,
    hit_key: str,
    answer_key: str,
) -> dict[str, int] | None:
    support_rows = [row for row in rows if row.get("status") == "ok" and row.get(hit_key) is not None]
    if not support_rows:
        return None
    counts = {
        "support_count": len(support_rows),
        "tp_count": 0,
        "fn_count": 0,
        "fp_count": 0,
        "tn_count": 0,
        "selected_hit_count": 0,
        "selected_miss_count": 0,
    }
    for row in support_rows:
        hit = float(row[hit_key]) > 0.0
        correct = float(row[answer_key]) > 0.0
        if hit:
            counts["selected_hit_count"] += 1
            if correct:
                counts["tp_count"] += 1
            else:
                counts["fn_count"] += 1
        else:
            counts["selected_miss_count"] += 1
            if correct:
                counts["fp_count"] += 1
            else:
                counts["tn_count"] += 1
    return counts


def _write_status(
    *,
    rows: list[dict[str, Any]],
    total_examples: int,
    started_at: float,
    status: str,
    video_id: str,
    method_name: str,
    output_dir: Path,
    budget_config: BudgetConfig,
    answer_config: AnswerConfig,
    config_payload: dict[str, Any],
) -> dict[str, Any]:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    confusion = _grounded_confusion_counts(
        ok_rows,
        hit_key="selected_evidence_hit1",
        answer_key="mcq_correct",
    )
    summary = {
        "mcq_accuracy": _mean([float(row["mcq_correct"]) for row in ok_rows]),
        "candidate_pool_hit_any": _mean_optional([row.get("candidate_pool_hit_any") for row in ok_rows]),
        "selected_evidence_hit1": _mean_optional([row.get("selected_evidence_hit1") for row in ok_rows]),
        "answer_accuracy_given_selected_hit": _answer_accuracy_given_hit(
            ok_rows,
            hit_key="selected_evidence_hit1",
            answer_key="mcq_correct",
        ),
        "answer_accuracy_given_selected_miss": _answer_accuracy_given_miss(
            ok_rows,
            hit_key="selected_evidence_hit1",
            answer_key="mcq_correct",
        ),
        "total_frames_seen_per_question": _mean_optional([row.get("total_frames_seen_per_question") for row in ok_rows]),
        "num_candidate_clips_seen": _mean_optional([row.get("num_candidate_clips_seen") for row in ok_rows]),
        "frames_per_clip": _mean_optional([row.get("frames_per_clip") for row in ok_rows]),
        "baseline_top1_mcq_accuracy": _mean_optional([row.get("baseline_top1_mcq_correct") for row in ok_rows]),
        "baseline_top1_l2_hit1": _mean_optional([row.get("baseline_top1_l2_hit1") for row in ok_rows]),
        "shortlist_hit_any": _mean_optional([row.get("shortlist_hit_any") for row in ok_rows]),
        "shortlist_candidate_valid_rate": _mean_optional([row.get("shortlist_candidate_valid") for row in ok_rows]),
        "shortlist_answer_valid_rate": _mean_optional([row.get("shortlist_answer_valid") for row in ok_rows]),
        "shortlist_size": _mean_optional([row.get("shortlist_size") for row in ok_rows]),
    }
    if confusion is None:
        summary.update(
            {
                "selected_evidence_support_count": None,
                "selected_evidence_hit_count": None,
                "selected_evidence_miss_count": None,
                "tp_count": None,
                "fn_count": None,
                "fp_count": None,
                "tn_count": None,
                "tp_rate": None,
                "fn_rate": None,
                "fp_rate": None,
                "tn_rate": None,
                "answer_not_correct_when_selected_miss": None,
            }
        )
    else:
        support_count = confusion["support_count"]
        summary.update(
            {
                "selected_evidence_support_count": support_count,
                "selected_evidence_hit_count": confusion["selected_hit_count"],
                "selected_evidence_miss_count": confusion["selected_miss_count"],
                "tp_count": confusion["tp_count"],
                "fn_count": confusion["fn_count"],
                "fp_count": confusion["fp_count"],
                "tn_count": confusion["tn_count"],
                "tp_rate": round(confusion["tp_count"] / support_count, 6),
                "fn_rate": round(confusion["fn_count"] / support_count, 6),
                "fp_rate": round(confusion["fp_count"] / support_count, 6),
                "tn_rate": round(confusion["tn_count"] / support_count, 6),
                "answer_not_correct_when_selected_miss": round(
                    confusion["tn_count"] / confusion["selected_miss_count"],
                    6,
                )
                if confusion["selected_miss_count"] > 0
                else None,
            }
        )
    payload = {
        "status": status,
        "method_name": method_name,
        "video_id": video_id,
        "run_state": {
            "total_examples": total_examples,
            "completed_examples": len(rows),
            "elapsed_sec": round(time.perf_counter() - started_at, 3),
        },
        "config": {
            "tasks": config_payload["tasks"],
            "limit": config_payload["limit"],
            "pipeline": config_payload["pipeline"],
            "budget": budget_config.to_dict(),
            "answer": _serialize_answer_config(answer_config),
            "method": method_name,
        },
        "summary": summary,
    }
    write_json(output_dir / "rolling_summary.json", payload)
    if status in {"completed", "failed"}:
        write_json(output_dir / "final_summary.json", payload)
    return payload


def _build_query_encoder_and_archive(*, repo_root: Path, video_id: str, device: str) -> tuple[Any, Any]:
    archive = load_feature_archive(repo_root, video_id)
    query_encoder = build_query_encoder(
        repo_root=repo_root,
        model_name=archive.model_name,
        pretrained_name=archive.pretrained_name,
        device=device,
    )
    return archive, query_encoder


def _build_segment_index(*, archive: Any, prefix: str = "l3") -> dict[str, Any]:
    l2_segments = build_fixed_windows(
        timestamps=archive.timestamps,
        window_seconds=PIPELINE_CONFIG.segmentation.l2_window_seconds,
        stride_seconds=PIPELINE_CONFIG.segmentation.l2_window_stride_seconds,
        prefix="l2",
    )
    l2_embeddings = mean_pool_segments(archive.frame_embeddings, l2_segments)
    l3_segments, _ = build_adaptive_layer3_segments(
        l2_segments=l2_segments,
        l2_embeddings=l2_embeddings,
        config=PIPELINE_CONFIG.segmentation,
        prefix=prefix,
    )
    l3_embeddings = mean_pool_segments(archive.frame_embeddings, l3_segments)
    return {
        "l2_segments": l2_segments,
        "l2_embeddings": l2_embeddings,
        "l3_segments": l3_segments,
        "l3_embeddings": l3_embeddings,
    }


def _layer3_candidates_for_scope(*, l3_segments: list[Any], scope_start_sec: float | None, scope_end_sec: float | None) -> list[Any]:
    return [
        segment
        for segment in l3_segments
        if (
            scope_start_sec is None
            or max(float(segment.start_time_sec), float(scope_start_sec)) <= min(float(segment.end_time_sec), float(scope_end_sec))
        )
    ]


def _candidate_pool_hierarchical(
    *,
    query_embedding: torch.Tensor,
    video_id: str,
    scope_start_sec: float | None,
    scope_end_sec: float | None,
    index_bundle: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    l2_segments = index_bundle["l2_segments"]
    l2_embeddings = index_bundle["l2_embeddings"]
    l3_segments = index_bundle["l3_segments"]
    l3_embeddings = index_bundle["l3_embeddings"]

    layer3_candidates = _layer3_candidates_for_scope(
        l3_segments=l3_segments,
        scope_start_sec=scope_start_sec,
        scope_end_sec=scope_end_sec,
    )
    layer3_candidate_embeddings = gather_segment_embeddings(
        all_segments=l3_segments,
        all_embeddings=l3_embeddings,
        selected_segments=layer3_candidates,
    )
    all_layer3_hits = rank_segments(
        query_embedding=query_embedding,
        segment_embeddings=layer3_candidate_embeddings,
        segments=layer3_candidates,
        video_id=video_id,
        top_k=max(len(layer3_candidates), 1),
    )
    layer3_hits = select_ranked_hits(
        hits=all_layer3_hits,
        mode=PIPELINE_CONFIG.retrieval.selection_mode,
        top_k=PIPELINE_CONFIG.retrieval.layer3_top_k,
        relative_alpha=PIPELINE_CONFIG.retrieval.layer3_relative_alpha,
        max_keep=PIPELINE_CONFIG.retrieval.layer3_max_keep,
    )
    layer2_candidates = restrict_segments_to_hits(
        segments=l2_segments,
        parent_hits=layer3_hits,
        scope_start_sec=scope_start_sec,
        scope_end_sec=scope_end_sec,
    )
    layer2_candidate_embeddings = gather_segment_embeddings(
        all_segments=l2_segments,
        all_embeddings=l2_embeddings,
        selected_segments=layer2_candidates,
    )
    all_layer2_hits = rank_segments(
        query_embedding=query_embedding,
        segment_embeddings=layer2_candidate_embeddings,
        segments=layer2_candidates,
        video_id=video_id,
        top_k=max(len(layer2_candidates), 1),
    )
    layer2_hits = select_ranked_hits(
        hits=all_layer2_hits,
        mode=PIPELINE_CONFIG.retrieval.selection_mode,
        top_k=PIPELINE_CONFIG.retrieval.layer2_top_k,
        relative_alpha=PIPELINE_CONFIG.retrieval.layer2_relative_alpha,
        max_keep=PIPELINE_CONFIG.retrieval.layer2_max_keep,
    )
    candidate_hits = []
    for candidate_index, hit in enumerate(layer2_hits, start=1):
        payload = hit.to_dict()
        payload["candidate_id"] = candidate_index
        candidate_hits.append(payload)
    return [hit.to_dict() for hit in layer3_hits], candidate_hits


def _candidate_pool_flat_l2(
    *,
    query_embedding: torch.Tensor,
    video_id: str,
    scope_start_sec: float | None,
    scope_end_sec: float | None,
    index_bundle: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    l2_segments = index_bundle["l2_segments"]
    l2_embeddings = index_bundle["l2_embeddings"]

    layer2_candidates = restrict_segments_to_hits(
        segments=l2_segments,
        parent_hits=[],
        scope_start_sec=scope_start_sec,
        scope_end_sec=scope_end_sec,
    )
    layer2_candidate_embeddings = gather_segment_embeddings(
        all_segments=l2_segments,
        all_embeddings=l2_embeddings,
        selected_segments=layer2_candidates,
    )
    all_layer2_hits = rank_segments(
        query_embedding=query_embedding,
        segment_embeddings=layer2_candidate_embeddings,
        segments=layer2_candidates,
        video_id=video_id,
        top_k=max(len(layer2_candidates), 1),
    )
    layer2_hits = select_ranked_hits(
        hits=all_layer2_hits,
        mode=PIPELINE_CONFIG.retrieval.selection_mode,
        top_k=PIPELINE_CONFIG.retrieval.layer2_top_k,
        relative_alpha=PIPELINE_CONFIG.retrieval.layer2_relative_alpha,
        max_keep=PIPELINE_CONFIG.retrieval.layer2_max_keep,
    )
    candidate_hits = []
    for candidate_index, hit in enumerate(layer2_hits, start=1):
        payload = hit.to_dict()
        payload["candidate_id"] = candidate_index
        candidate_hits.append(payload)
    return [], candidate_hits


def _frame_hit_to_segment(
    *,
    frame_index: int,
    score: float,
    timestamps: Any,
    video_id: str,
    scope_start_sec: float | None,
    scope_end_sec: float | None,
) -> dict[str, Any]:
    center_time_sec = float(timestamps[frame_index])
    half_window = PIPELINE_CONFIG.segmentation.l2_window_seconds / 2.0
    start_time_sec = center_time_sec - half_window
    end_time_sec = center_time_sec + half_window
    if scope_start_sec is not None:
        start_time_sec = max(start_time_sec, float(scope_start_sec))
        end_time_sec = max(end_time_sec, start_time_sec)
    if scope_end_sec is not None:
        end_time_sec = min(end_time_sec, float(scope_end_sec))
        start_time_sec = min(start_time_sec, end_time_sec)
    return {
        "segment_id": f"frame_bundle_{frame_index}",
        "score": float(score),
        "start_index": int(frame_index),
        "end_index": int(frame_index),
        "start_time_sec": round(float(start_time_sec), 3),
        "end_time_sec": round(float(end_time_sec), 3),
        "video_id": video_id,
    }


def _merge_frame_bundles(
    *,
    ranked_frame_candidates: list[dict[str, Any]],
    max_keep: int,
    max_gap_sec: float,
    video_id: str,
    scope_start_sec: float | None = None,
    scope_end_sec: float | None = None,
) -> list[dict[str, Any]]:
    ordered = sorted(ranked_frame_candidates, key=lambda item: float(item["time_sec"]))
    bundles: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for frame in ordered:
        frame_time = float(frame["time_sec"])
        if current is None:
            current = {
                "start_time_sec": frame_time,
                "end_time_sec": frame_time,
                "score": float(frame["score"]),
                "video_id": video_id,
            }
            continue
        if frame_time - float(current["end_time_sec"]) <= max_gap_sec:
            current["end_time_sec"] = frame_time
            current["score"] = max(float(current["score"]), float(frame["score"]))
        else:
            bundles.append(current)
            current = {
                "start_time_sec": frame_time,
                "end_time_sec": frame_time,
                "score": float(frame["score"]),
                "video_id": video_id,
            }
    if current is not None:
        bundles.append(current)

    bundles.sort(key=lambda item: float(item["score"]), reverse=True)
    selected = bundles[: max_keep]
    normalized: list[dict[str, Any]] = []
    half_window = PIPELINE_CONFIG.segmentation.l2_window_seconds / 2.0
    for index, bundle in enumerate(selected, start=1):
        start_time_sec = max(0.0, float(bundle["start_time_sec"]) - half_window)
        end_time_sec = float(bundle["end_time_sec"]) + half_window
        if scope_start_sec is not None:
            start_time_sec = max(start_time_sec, float(scope_start_sec))
        if scope_end_sec is not None:
            end_time_sec = min(end_time_sec, float(scope_end_sec))
        end_time_sec = max(end_time_sec, start_time_sec)
        normalized.append(
            {
                "segment_id": f"openclip_bundle_{index}",
                "candidate_id": index,
                "score": float(bundle["score"]),
                "start_time_sec": round(start_time_sec, 3),
                "end_time_sec": round(end_time_sec, 3),
                "video_id": video_id,
            }
        )
    return normalized


def _candidate_pool_flat_openclip(
    *,
    query_embedding: torch.Tensor,
    archive: Any,
    video_id: str,
    scope_start_sec: float | None,
    scope_end_sec: float | None,
    budget_config: BudgetConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    frame_scores = torch.matmul(archive.frame_embeddings, query_embedding).cpu().numpy()
    ranked_frames: list[dict[str, Any]] = []
    for frame_index, score in enumerate(frame_scores.tolist()):
        time_sec = float(archive.timestamps[frame_index])
        if scope_start_sec is not None and time_sec < float(scope_start_sec):
            continue
        if scope_end_sec is not None and time_sec > float(scope_end_sec):
            continue
        ranked_frames.append(
            {
                "frame_index": frame_index,
                "time_sec": time_sec,
                "score": float(score),
                "video_id": video_id,
            }
        )
    ranked_frames.sort(key=lambda item: float(item["score"]), reverse=True)
    top_ranked = ranked_frames[: max(budget_config.shortlist_k * 10, 1)]
    candidate_hits = _merge_frame_bundles(
        ranked_frame_candidates=top_ranked,
        max_keep=budget_config.shortlist_k,
        max_gap_sec=budget_config.frame_bundle_max_gap_sec,
        video_id=video_id,
        scope_start_sec=scope_start_sec,
        scope_end_sec=scope_end_sec,
    )
    return [], candidate_hits


def _choice_spans_for_example(example: Any) -> list[dict[str, Any]]:
    parsed_choice_spans = [span for choice in example.choices for span in parse_choice_spans(choice)]
    matching_choice_spans = [span for span in parsed_choice_spans if span.video_alias == "video 1"]
    return [span.to_dict() for span in (matching_choice_spans or parsed_choice_spans)]


def _prompt_choices(example: Any) -> list[str]:
    prompt_options = [_normalize_choice_for_prompt(choice) for choice in example.choices]
    return _label_prompt_choices(prompt_options)


def _run_retrieval_method(
    *,
    method_name: str,
    video_id: str,
    tasks: list[str],
    limit: int,
    output_dir: Path,
    budget_config: BudgetConfig,
    answer_config: AnswerConfig,
    candidate_strategy: str,
    answerer: Any | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "rows.jsonl"
    log_path = output_dir / "progress.log"
    debug_dir = output_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.perf_counter()
    rows: list[dict[str, Any]] = []
    video_path = REPO_ROOT / "dataset" / "hd_epic_structured" / video_id / "video.mp4"
    config_payload = {
        "tasks": list(tasks),
        "limit": limit,
        "pipeline": PIPELINE_CONFIG.to_dict(),
    }
    active_answerer = answerer or build_answerer(answer_config)
    log_line(log_path, f"[start] method={method_name} video={video_id} limit={limit}")

    examples = filter_examples_for_video(load_examples(tasks, REPO_ROOT), video_id)
    examples = [example for example in examples if example.answer_type == "temporal_option" and example.gold_spans][:limit]

    archive = None
    query_encoder = None
    try:
        archive, query_encoder = _build_query_encoder_and_archive(
            repo_root=REPO_ROOT,
            video_id=video_id,
            device=PIPELINE_CONFIG.retrieval.device,
        )
        index_bundle = _build_segment_index(archive=archive)

        _write_status(
            rows=rows,
            total_examples=len(examples),
            started_at=started_at,
            status="running",
            video_id=video_id,
            method_name=method_name,
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
            gold_spans = gold_spans_for_video(example, video_id)
            scope_start_sec, scope_end_sec = example_scope_for_video(example, video_id)

            if candidate_strategy == "ours":
                layer3_hits, candidate_hits = _candidate_pool_hierarchical(
                    query_embedding=query_embedding,
                    video_id=video_id,
                    scope_start_sec=scope_start_sec,
                    scope_end_sec=scope_end_sec,
                    index_bundle=index_bundle,
                )
            elif candidate_strategy == "direct_layer2":
                layer3_hits, candidate_hits = _candidate_pool_flat_l2(
                    query_embedding=query_embedding,
                    video_id=video_id,
                    scope_start_sec=scope_start_sec,
                    scope_end_sec=scope_end_sec,
                    index_bundle=index_bundle,
                )
            elif candidate_strategy == "direct_openclip":
                layer3_hits, candidate_hits = _candidate_pool_flat_openclip(
                    query_embedding=query_embedding,
                    archive=archive,
                    video_id=video_id,
                    scope_start_sec=scope_start_sec,
                    scope_end_sec=scope_end_sec,
                    budget_config=budget_config,
                )
            else:
                raise ValueError(f"Unsupported candidate strategy: {candidate_strategy}")

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
                log_line(log_path, f"[error] row={index}/{len(examples)} example_id={example.example_id} no_candidates")
                _write_status(
                    rows=rows,
                    total_examples=len(examples),
                    started_at=started_at,
                    status="running",
                    video_id=video_id,
                    method_name=method_name,
                    output_dir=output_dir,
                    budget_config=budget_config,
                    answer_config=answer_config,
                    config_payload=config_payload,
                )
                continue

            labeled_options = _prompt_choices(example)
            choice_spans = _choice_spans_for_example(example)
            shortlist_ids, shortlist_scored_candidates = shortlist_candidates_cover_choices(
                candidates=candidate_hits,
                choice_spans=choice_spans,
                max_keep=budget_config.shortlist_k,
            )
            shortlisted_candidates = [candidate for candidate in candidate_hits if candidate["candidate_id"] in shortlist_ids]
            shortlist_candidate_valid = 1.0 if len(shortlisted_candidates) == min(budget_config.shortlist_k, len(candidate_hits)) else 0.0

            display_id_to_candidate: dict[int, dict[str, Any]] = {}
            selection_raw_answer: str | None = None
            answer_raw_answer: str | None = None
            visual_raw_answer: str | None = None

            if candidate_strategy == "ours":
                visual_frames: list[Any] = []
                visual_frame_texts: list[str] = []
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
                    visual_frames.extend(frames)
                    visual_frame_texts.extend(
                        [f"Candidate {display_index} | {label}" for label in _build_mcq_style_frame_labels(timestamps_sec)]
                    )

                visual_prompt = build_visual_joint_prompt(
                    question=example.question,
                    labeled_options=labeled_options,
                    candidates=shortlisted_candidates,
                )
                visual_generation = active_answerer.generate_text_from_frames(
                    frames=visual_frames,
                    prompt=visual_prompt,
                    frame_texts=visual_frame_texts,
                )
                visual_raw_answer = visual_generation.raw_text
                selected_display_id, selected_letter = parse_joint_response(
                    visual_generation.raw_text,
                    max_candidate=len(shortlisted_candidates),
                    options_count=len(example.choices),
                )
                shortlist_answer_valid = 1.0 if selected_letter is not None else 0.0
                if selected_display_id is None:
                    selected_display_id = 1
                    shortlist_candidate_valid = 0.0
                selected_candidate = display_id_to_candidate[selected_display_id]
                selected_choice_idx = (ord(selected_letter) - ord("A")) if selected_letter is not None else None
                total_frames_seen = float(len(visual_frames))
            else:
                visual_frames = []
                visual_frame_texts = []
                for display_index, candidate in enumerate(shortlisted_candidates, start=1):
                    display_id_to_candidate[display_index] = candidate
                    frames, _, timestamps_sec, _ = _sample_uniform_video_frames(
                        video_path=video_path,
                        frame_budget=budget_config.frames_per_candidate,
                        start_time_sec=float(candidate["start_time_sec"]),
                        end_time_sec=float(candidate["end_time_sec"]),
                    )
                    visual_frames.extend(frames)
                    visual_frame_texts.extend(
                        [f"Candidate {display_index} | {label}" for label in _build_frame_timestamp_labels(timestamps_sec)]
                    )

                visual_prompt = build_visual_joint_prompt(
                    question=example.question,
                    labeled_options=labeled_options,
                    candidates=shortlisted_candidates,
                )
                visual_generation = active_answerer.generate_text_from_frames(
                    frames=visual_frames,
                    prompt=visual_prompt,
                    frame_texts=visual_frame_texts,
                )
                visual_raw_answer = visual_generation.raw_text
                selected_display_id, selected_letter = parse_joint_response(
                    visual_generation.raw_text,
                    max_candidate=len(shortlisted_candidates),
                    options_count=len(example.choices),
                )
                shortlist_answer_valid = 1.0 if selected_letter is not None else 0.0
                if selected_display_id is None:
                    selected_display_id = 1
                    shortlist_candidate_valid = 0.0
                selected_candidate = display_id_to_candidate[selected_display_id]
                selected_choice_idx = (ord(selected_letter) - ord("A")) if selected_letter is not None else None
                total_frames_seen = float(len(visual_frames))

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
            baseline_generation = active_answerer.generate_text_from_frames(
                frames=baseline_frames,
                prompt=baseline_prompt,
                frame_texts=_build_frame_timestamp_labels(baseline_timestamps),
            )
            baseline_letter = parse_choice_letter(baseline_generation.raw_text, options_count=len(example.choices))
            baseline_choice_idx = (ord(baseline_letter) - ord("A")) if baseline_letter is not None else None

            baseline_l2_metrics = summarize_layer2_hits(layer2_hits=[baseline_hit], gold_spans=gold_spans, top_k=1)
            selected_l2_metrics = summarize_layer2_hits(layer2_hits=[selected_candidate], gold_spans=gold_spans, top_k=1)
            shortlist_any_metrics = summarize_layer2_hits(
                layer2_hits=shortlisted_candidates,
                gold_spans=gold_spans,
                top_k=len(shortlisted_candidates),
            )
            candidate_pool_metrics = summarize_layer2_hits(
                layer2_hits=candidate_hits,
                gold_spans=gold_spans,
                top_k=len(candidate_hits),
            )
            layer3_metrics = summarize_layer3_hits(
                layer3_hits=layer3_hits,
                gold_spans=gold_spans,
                coverage_threshold=PIPELINE_CONFIG.retrieval.layer3_coverage_threshold,
            ) if layer3_hits else {}

            row = {
                "example_id": example.example_id,
                "task_name": example.task_name,
                "question": example.question,
                "query_text": query_text,
                "correct_idx": int(example.correct_idx),
                "layer3_metrics": layer3_metrics,
                "candidate_pool": candidate_hits,
                "candidate_pool_hit_any": float(candidate_pool_metrics[f"Layer2 Hit@{len(candidate_hits)}_gap0"]),
                "baseline_top1_hit": baseline_hit,
                "baseline_top1_raw_answer": baseline_generation.raw_text,
                "baseline_top1_mcq_correct": 1.0 if baseline_choice_idx == example.correct_idx else 0.0,
                "baseline_top1_l2_hit1": float(baseline_l2_metrics["Layer2 Hit@1_gap0"]),
                "shortlist_ids": shortlist_ids,
                "shortlist_display_mapping": {
                    str(display_index): int(candidate["candidate_id"])
                    for display_index, candidate in display_id_to_candidate.items()
                },
                "shortlist_scored_candidates": shortlist_scored_candidates,
                "shortlist_selected_display_id": selected_display_id,
                "shortlist_selected_choice_letter": selected_letter,
                "selection_raw_answer": selection_raw_answer,
                "answer_raw_answer": answer_raw_answer,
                "visual_raw_answer": visual_raw_answer,
                "selected_evidence": selected_candidate,
                "mcq_correct": 1.0 if selected_choice_idx == example.correct_idx else 0.0,
                "selected_evidence_hit1": float(selected_l2_metrics["Layer2 Hit@1_gap0"]),
                "shortlist_hit_any": float(shortlist_any_metrics[f"Layer2 Hit@{len(shortlisted_candidates)}_gap0"]),
                "shortlist_candidate_valid": shortlist_candidate_valid,
                "shortlist_answer_valid": shortlist_answer_valid,
                "shortlist_size": float(len(shortlisted_candidates)),
                "total_frames_seen_per_question": total_frames_seen,
                "num_candidate_clips_seen": float(len(shortlisted_candidates)),
                "frames_per_clip": float(budget_config.frames_per_candidate),
                "status": "ok",
            }
            append_jsonl(rows_path, row)
            rows.append(row)

            debug_sections = [
                f"example_id: {example.example_id}",
                f"task_name: {example.task_name}",
                f"question: {example.question}",
                f"query_text: {query_text}",
                f"correct_idx: {example.correct_idx}",
                "",
                "choice_spans:",
                *[str(span) for span in choice_spans],
                "",
                "candidate_pool:",
                *[str(candidate) for candidate in candidate_hits],
                "",
                "shortlist_scored_candidates:",
                *[str(item) for item in shortlist_scored_candidates],
                "",
                f"shortlist_ids: {shortlist_ids}",
                "",
                f"shortlist_display_mapping: { {display_index: int(candidate['candidate_id']) for display_index, candidate in display_id_to_candidate.items()} }",
                "",
            ]
            if candidate_strategy == "ours":
                debug_sections.extend(
                    [
                        f"visual_prompt:\n{visual_prompt}",
                        "",
                        "visual_frame_texts:",
                        *visual_frame_texts,
                        "",
                        f"visual_raw_answer:\n{visual_raw_answer}",
                        "",
                    ]
                )
            else:
                debug_sections.extend(
                    [
                        f"visual_prompt:\n{visual_prompt}",
                        "",
                        "visual_frame_texts:",
                        *visual_frame_texts,
                        "",
                        f"visual_raw_answer:\n{visual_raw_answer}",
                        "",
                    ]
                )
            debug_sections.extend(
                [
                    f"baseline_prompt:\n{baseline_prompt}",
                    "",
                    f"baseline_raw_answer:\n{baseline_generation.raw_text}",
                ]
            )
            debug_text = "\n".join(debug_sections) + "\n"
            (debug_dir / f"{index:02d}_{example.example_id}.txt").write_text(debug_text, encoding="utf-8")

            log_line(
                log_path,
                f"[progress] row={index}/{len(examples)} example_id={example.example_id} "
                f"mcq={row['mcq_correct']:.0f} "
                f"selected_hit={row['selected_evidence_hit1']:.0f} shortlist_size={int(row['shortlist_size'])}",
            )
            _write_status(
                rows=rows,
                total_examples=len(examples),
                started_at=started_at,
                status="running",
                video_id=video_id,
                method_name=method_name,
                output_dir=output_dir,
                budget_config=budget_config,
                answer_config=answer_config,
                config_payload=config_payload,
            )

        final_payload = _write_status(
            rows=rows,
            total_examples=len(examples),
            started_at=started_at,
            status="completed",
            video_id=video_id,
            method_name=method_name,
            output_dir=output_dir,
            budget_config=budget_config,
            answer_config=answer_config,
            config_payload=config_payload,
        )
        log_line(log_path, "[done]")
        return final_payload
    except Exception as exc:
        log_line(log_path, f"[error] type={type(exc).__name__} error={exc}")
        final_payload = _write_status(
            rows=rows,
            total_examples=len(examples),
            started_at=started_at,
            status="failed",
            video_id=video_id,
            method_name=method_name,
            output_dir=output_dir,
            budget_config=budget_config,
            answer_config=answer_config,
            config_payload=config_payload,
        )
        raise
    finally:
        if query_encoder is not None:
            del query_encoder
        if answerer is None:
            active_answerer.unload()


def run_pure_vlm_method(
    *,
    video_id: str,
    tasks: list[str],
    limit: int,
    output_dir: Path,
    budget_config: BudgetConfig,
    answer_config: AnswerConfig,
    answerer: QwenVLMAnswerer | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "rows.jsonl"
    log_path = output_dir / "progress.log"
    debug_dir = output_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.perf_counter()
    rows: list[dict[str, Any]] = []
    video_path = REPO_ROOT / "dataset" / "hd_epic_structured" / video_id / "video.mp4"
    config_payload = {
        "tasks": list(tasks),
        "limit": limit,
        "pipeline": PIPELINE_CONFIG.to_dict(),
    }
    active_answerer = answerer or QwenVLMAnswerer(answer_config)
    log_line(log_path, f"[start] method=pure_vlm video={video_id} limit={limit}")

    examples = filter_examples_for_video(load_examples(tasks, REPO_ROOT), video_id)
    examples = [example for example in examples if example.answer_type == "temporal_option" and example.gold_spans][:limit]
    archive = load_feature_archive(REPO_ROOT, video_id)

    try:
        _write_status(
            rows=rows,
            total_examples=len(examples),
            started_at=started_at,
            status="running",
            video_id=video_id,
            method_name="pure_vlm",
            output_dir=output_dir,
            budget_config=budget_config,
            answer_config=answer_config,
            config_payload=config_payload,
        )

        for index, example in enumerate(examples, start=1):
            scope_start_sec, scope_end_sec = example_scope_for_video(example, video_id)
            if scope_start_sec is None:
                scope_start_sec = 0.0
            if scope_end_sec is None:
                scope_end_sec = float(archive.duration_sec)
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
            generation = active_answerer.generate_text_from_frames(
                frames=frames,
                prompt=prompt,
                frame_texts=_build_frame_timestamp_labels(timestamps_sec),
            )
            predicted_letter = parse_choice_letter(generation.raw_text, options_count=len(example.choices))
            predicted_idx = (ord(predicted_letter) - ord("A")) if predicted_letter is not None else None
            row = {
                "example_id": example.example_id,
                "task_name": example.task_name,
                "question": example.question,
                "correct_idx": int(example.correct_idx),
                "raw_answer": generation.raw_text,
                "mcq_correct": 1.0 if predicted_idx == example.correct_idx else 0.0,
                "candidate_pool_hit_any": None,
                "selected_evidence_hit1": None,
                "shortlist_hit_any": None,
                "shortlist_candidate_valid": None,
                "shortlist_answer_valid": None,
                "shortlist_size": None,
                "total_frames_seen_per_question": float(len(frames)),
                "num_candidate_clips_seen": 1.0,
                "frames_per_clip": float(len(frames)),
                "status": "ok",
            }
            append_jsonl(rows_path, row)
            rows.append(row)
            (debug_dir / f"{index:02d}_{example.example_id}.txt").write_text(
                f"prompt:\n{prompt}\n\nframe_texts:\n" + "\n".join(_build_frame_timestamp_labels(timestamps_sec)) + f"\n\nraw_answer:\n{generation.raw_text}\n",
                encoding="utf-8",
            )
            log_line(
                log_path,
                f"[progress] row={index}/{len(examples)} example_id={example.example_id} mcq={row['mcq_correct']:.0f}",
            )
            _write_status(
                rows=rows,
                total_examples=len(examples),
                started_at=started_at,
                status="running",
                video_id=video_id,
                method_name="pure_vlm",
                output_dir=output_dir,
                budget_config=budget_config,
                answer_config=answer_config,
                config_payload=config_payload,
            )

        payload = _write_status(
            rows=rows,
            total_examples=len(examples),
            started_at=started_at,
            status="completed",
            video_id=video_id,
            method_name="pure_vlm",
            output_dir=output_dir,
            budget_config=budget_config,
            answer_config=answer_config,
            config_payload=config_payload,
        )
        log_line(log_path, "[done]")
        return payload
    finally:
        if answerer is None:
            active_answerer.unload()


def run_ours_method(
    *,
    video_id: str,
    tasks: list[str],
    limit: int,
    output_dir: Path,
    budget_config: BudgetConfig,
    answer_config: AnswerConfig,
    answerer: Any | None = None,
) -> dict[str, Any]:
    return _run_retrieval_method(
        method_name="ours",
        video_id=video_id,
        tasks=tasks,
        limit=limit,
        output_dir=output_dir,
        budget_config=budget_config,
        answer_config=answer_config,
        candidate_strategy="ours",
        answerer=answerer,
    )


def run_direct_layer2_method(
    *,
    video_id: str,
    tasks: list[str],
    limit: int,
    output_dir: Path,
    budget_config: BudgetConfig,
    answer_config: AnswerConfig,
    answerer: Any | None = None,
) -> dict[str, Any]:
    return _run_retrieval_method(
        method_name="direct_layer2_retrieval",
        video_id=video_id,
        tasks=tasks,
        limit=limit,
        output_dir=output_dir,
        budget_config=budget_config,
        answer_config=answer_config,
        candidate_strategy="direct_layer2",
        answerer=answerer,
    )


def run_direct_openclip_method(
    *,
    video_id: str,
    tasks: list[str],
    limit: int,
    output_dir: Path,
    budget_config: BudgetConfig,
    answer_config: AnswerConfig,
    answerer: Any | None = None,
) -> dict[str, Any]:
    return _run_retrieval_method(
        method_name="direct_open_clip",
        video_id=video_id,
        tasks=tasks,
        limit=limit,
        output_dir=output_dir,
        budget_config=budget_config,
        answer_config=answer_config,
        candidate_strategy="direct_openclip",
        answerer=answerer,
    )
