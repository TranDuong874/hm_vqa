from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from hm_vqa.schema import RetrievalExample, TimeSpan
from evals.common.vlm_baseline_runner import BaselineExample
from evals.hd_epic.paths import (
    ANNOTATION_ROOT,
    FGAL24_MANIFEST,
    HD_EPIC_FGAL24_DERIVED_ROOT,
    HD_EPIC_FGAL24_OPENCLIP_ROOT,
    RAW_VIDEO_ROOT,
    RESULTS_ROOT,
    REPO_ROOT,
    STRUCTURED_VIDEO_ROOT,
)


DEFAULT_MANIFEST = FGAL24_MANIFEST
DEFAULT_VIDEO_ROOT = STRUCTURED_VIDEO_ROOT
DEFAULT_FEATURE_ROOT = HD_EPIC_FGAL24_OPENCLIP_ROOT
DEFAULT_DERIVED_CACHE_ROOT = HD_EPIC_FGAL24_DERIVED_ROOT
DEFAULT_OUTPUT_ROOT = RESULTS_ROOT / "ablations"
DEFAULT_RETRIEVAL_TASKS = ("fine_grained_action_localization",)
TIME_PATTERN = re.compile(r"<TIME\s+([^<>]+?)\s+video\s+(\d+)>")


@dataclass(slots=True)
class TemporalInputVideo:
    label: str
    video_id: str
    start_time_sec: float
    end_time_sec: float


@dataclass(slots=True)
class TemporalExample:
    example_id: str
    task_name: str
    question: str
    choices: list[str]
    correct_idx: int
    input_videos: list[TemporalInputVideo]


def parse_timecode(value: str) -> float:
    hours, minutes, seconds = value.strip().split(":")
    return int(hours) * 3600.0 + int(minutes) * 60.0 + float(seconds)


def _task_path(repo_root: Path, task_name: str) -> Path:
    candidates = [
        repo_root / "dataset/hd-epic-annotations/vqa-benchmark" / f"{task_name}.json",
        ANNOTATION_ROOT / f"{task_name}.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing HD-EPIC annotation file for task {task_name}: {candidates[0]}")


def load_task_examples(repo_root: Path, task_name: str) -> list[TemporalExample]:
    payload = json.loads(_task_path(repo_root, task_name).read_text(encoding="utf-8"))
    examples: list[TemporalExample] = []
    for example_id, row in payload.items():
        input_videos: list[TemporalInputVideo] = []
        for label, video in dict(row.get("inputs") or {}).items():
            input_videos.append(
                TemporalInputVideo(
                    label=str(label),
                    video_id=str(video["id"]),
                    start_time_sec=parse_timecode(str(video["start_time"])),
                    end_time_sec=parse_timecode(str(video["end_time"])),
                )
            )
        examples.append(
            TemporalExample(
                example_id=str(example_id),
                task_name=task_name,
                question=str(row.get("question", "")),
                choices=[str(choice) for choice in row.get("choices", [])],
                correct_idx=int(row.get("correct_idx", 0)),
                input_videos=input_videos,
            )
        )
    return examples


def load_temporal_examples_for_video(
    *,
    repo_root: Path,
    video_id: str,
    tasks: tuple[str, ...],
) -> list[TemporalExample]:
    rows: list[TemporalExample] = []
    for task_name in tasks:
        for example in load_task_examples(repo_root, task_name):
            if any(video.video_id == video_id for video in example.input_videos):
                rows.append(example)
    return rows


def example_scope_for_video(example: TemporalExample, video_id: str) -> tuple[float | None, float | None]:
    for video in example.input_videos:
        if video.video_id == video_id:
            return float(video.start_time_sec), float(video.end_time_sec)
    return None, None


def gold_spans_for_video(example: TemporalExample, video_id: str) -> list[tuple[float, float]]:
    if not example.choices or example.correct_idx < 0 or example.correct_idx >= len(example.choices):
        return []
    label_by_video_id = {video.video_id: video.label for video in example.input_videos}
    target_label = label_by_video_id.get(video_id)
    spans: list[tuple[float, float]] = []
    matches = TIME_PATTERN.findall(example.choices[example.correct_idx])
    for index in range(0, len(matches) - 1, 2):
        start_text, start_video_number = matches[index]
        end_text, end_video_number = matches[index + 1]
        if start_video_number != end_video_number:
            continue
        if target_label is not None and f"video {start_video_number}" != target_label:
            continue
        spans.append((parse_timecode(start_text), parse_timecode(end_text)))
    return spans


def _resolve_video_path(video_root: Path, video_id: str) -> Path:
    structured_path = video_root / video_id / "video.mp4"
    if structured_path.exists():
        return structured_path

    direct_path = video_root / f"{video_id}.mp4"
    if direct_path.exists():
        return direct_path

    participant = video_id.split("-", 1)[0]
    participant_dir = video_root / participant
    candidates = sorted(participant_dir.glob(f"{video_id}*.mp4")) if participant_dir.exists() else []
    if candidates:
        return candidates[0]

    raw_participant_dir = RAW_VIDEO_ROOT / participant
    raw_candidates = sorted(raw_participant_dir.glob(f"{video_id}*.mp4")) if raw_participant_dir.exists() else []
    if raw_candidates:
        return raw_candidates[0]

    return structured_path


def load_hd_epic_examples(
    manifest_path: Path = DEFAULT_MANIFEST,
    *,
    video_root: Path = DEFAULT_VIDEO_ROOT,
    limit: int | None = None,
) -> list[BaselineExample]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_rows = payload.get("mcq_rows") or payload.get("rows") or []
    if limit is not None:
        manifest_rows = manifest_rows[:limit]
    task_names = sorted({str(row["task_name"]) for row in manifest_rows})
    eval_examples = [example for task_name in task_names for example in load_task_examples(REPO_ROOT, task_name)]
    by_example_id = {example.example_id: example for example in eval_examples}
    temporal_examples = []
    for video_id in sorted({str(row["video_id"]) for row in manifest_rows}):
        temporal_examples.extend(
            load_temporal_examples_for_video(
                repo_root=REPO_ROOT,
                video_id=video_id,
                tasks=tuple(task_names),
            )
        )
    temporal_by_key = {
        (video.video_id, example.example_id): example
        for example in temporal_examples
        for video in example.input_videos
        if video.video_id
    }

    examples: list[BaselineExample] = []
    for row in manifest_rows:
        eval_example = by_example_id.get(str(row["example_id"]))
        if eval_example is None:
            continue
        video_id = str(row["video_id"])
        temporal_example = temporal_by_key.get((video_id, str(row["example_id"])))
        scope_start_sec = None
        scope_end_sec = None
        gold_spans = []
        if temporal_example is not None:
            scope_start_sec, scope_end_sec = example_scope_for_video(temporal_example, video_id)
            gold_spans = gold_spans_for_video(temporal_example, video_id)
        examples.append(
            BaselineExample(
                example_id=str(row["example_id"]),
                video_id=video_id,
                video_path=str(_resolve_video_path(video_root, video_id)),
                question=str(eval_example.question),
                options=[str(choice) for choice in eval_example.choices],
                correct_index=int(eval_example.correct_idx),
                metadata={
                    "task_name": str(row.get("task_name", "")),
                    "duration_bucket": row.get("duration_bucket"),
                    "duration": row.get("duration_sec"),
                    "question_rank_in_video_sample": row.get("question_rank_in_video_sample"),
                    "question_count_video": row.get("question_count_video"),
                    "mcq_eval": bool(row.get("mcq_eval", True)),
                    "scope_start_sec": scope_start_sec,
                    "scope_end_sec": scope_end_sec,
                    "gold_spans": gold_spans,
                },
            )
        )
    return examples


def load_retrieval_examples_for_video(
    video_id: str,
    *,
    video_root: Path = RAW_VIDEO_ROOT,
    tasks: tuple[str, ...] = DEFAULT_RETRIEVAL_TASKS,
) -> list[RetrievalExample]:
    examples = load_temporal_examples_for_video(repo_root=REPO_ROOT, video_id=video_id, tasks=tasks)
    video_path = _resolve_video_path(video_root, video_id)
    rows: list[RetrievalExample] = []
    for example in examples:
        scope_start, scope_end = example_scope_for_video(example, video_id)
        gold_spans = [TimeSpan(start_sec=start, end_sec=end) for start, end in gold_spans_for_video(example, video_id)]
        rows.append(
            RetrievalExample(
                example_id=f"{example.example_id}::{video_id}",
                dataset="hd_epic",
                split="fgal",
                video_id=video_id,
                video_path=video_path,
                query=example.question,
                time_scope=(
                    TimeSpan(start_sec=float(scope_start), end_sec=float(scope_end))
                    if scope_start is not None and scope_end is not None
                    else None
                ),
                gold_spans=gold_spans,
                metadata={
                    "task_name": example.task_name,
                    "source_example_id": example.example_id,
                    "choices": example.choices,
                    "correct_idx": example.correct_idx,
                },
            )
        )
    return rows
