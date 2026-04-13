from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .schema import EvalExample, InputVideoRef, TemporalSpan


TASK_FILES = {
    "fine_grained_action_localization": "dataset/hd-epic-annotations/vqa-benchmark/fine_grained_action_localization.json",
    "fine_grained_action_recognition": "dataset/hd-epic-annotations/vqa-benchmark/fine_grained_action_recognition.json",
    "fine_grained_how_recognition": "dataset/hd-epic-annotations/vqa-benchmark/fine_grained_how_recognition.json",
    "fine_grained_why_recognition": "dataset/hd-epic-annotations/vqa-benchmark/fine_grained_why_recognition.json",
    "recipe_step_localization": "dataset/hd-epic-annotations/vqa-benchmark/recipe_step_localization.json",
    "recipe_step_recognition": "dataset/hd-epic-annotations/vqa-benchmark/recipe_step_recognition.json",
    "ingredient_ingredients_order": "dataset/hd-epic-annotations/vqa-benchmark/ingredient_ingredients_order.json",
    "object_motion_object_movement_counting": "dataset/hd-epic-annotations/vqa-benchmark/object_motion_object_movement_counting.json",
}

TEMPORAL_TASKS = {
    "fine_grained_action_localization",
    "recipe_step_localization",
}

PILOT_TASKS = [
    "fine_grained_action_localization",
    "recipe_step_localization",
    "fine_grained_action_recognition",
    "recipe_step_recognition",
    "ingredient_ingredients_order",
    "object_motion_object_movement_counting",
    "fine_grained_how_recognition",
    "fine_grained_why_recognition",
]

TIME_PAIR_PATTERN = re.compile(
    r"<TIME\s+(?P<start>\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s+(?P<alias_start>video\s+\d+)>\s+to\s+<TIME\s+"
    r"(?P<end>\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s+(?P<alias_end>video\s+\d+)>",
    re.IGNORECASE,
)


def parse_timecode(value: str | None) -> float | None:
    if value is None:
        return None
    hours, minutes, seconds = value.split(":")
    return int(hours) * 3600.0 + int(minutes) * 60.0 + float(seconds)


def parse_choice_spans(choice: Any) -> list[TemporalSpan]:
    if not isinstance(choice, str):
        return []
    spans: list[TemporalSpan] = []
    for match in TIME_PAIR_PATTERN.finditer(choice):
        alias_start = match.group("alias_start").strip().lower()
        alias_end = match.group("alias_end").strip().lower()
        alias = alias_start if alias_start == alias_end else alias_start
        spans.append(
            TemporalSpan(
                video_alias=alias,
                start_time_sec=float(parse_timecode(match.group("start")) or 0.0),
                end_time_sec=float(parse_timecode(match.group("end")) or 0.0),
            )
        )
    return spans


def _normalize_input_videos(raw_inputs: dict[str, Any]) -> list[InputVideoRef]:
    normalized: list[InputVideoRef] = []
    for alias, raw in raw_inputs.items():
        alias_norm = alias.strip().lower()
        normalized.append(
            InputVideoRef(
                alias=alias_norm,
                video_id=str(raw["id"]),
                start_time_sec=parse_timecode(raw.get("start_time")),
                end_time_sec=parse_timecode(raw.get("end_time")),
            )
        )
    return normalized


def load_task_examples(task_name: str, repo_root: Path) -> list[EvalExample]:
    relative_path = TASK_FILES[task_name]
    data = json.loads((repo_root / relative_path).read_text())
    examples: list[EvalExample] = []
    for example_id in sorted(data.keys()):
        item = data[example_id]
        input_videos = _normalize_input_videos(item.get("inputs", {}))
        unique_video_ids = sorted({video.video_id for video in input_videos})
        primary_video_id = unique_video_ids[0] if len(unique_video_ids) == 1 else None
        answer_type = "temporal_option" if task_name in TEMPORAL_TASKS else "mcq_text"
        gold_spans = parse_choice_spans(item["choices"][item["correct_idx"]]) if answer_type == "temporal_option" else []
        examples.append(
            EvalExample(
                example_id=example_id,
                task_name=task_name,
                question=item["question"],
                choices=item["choices"],
                correct_idx=int(item["correct_idx"]),
                input_videos=input_videos,
                primary_video_id=primary_video_id,
                gold_spans=gold_spans,
                answer_type=answer_type,
            )
        )
    return examples


def load_examples(task_names: list[str], repo_root: Path) -> list[EvalExample]:
    examples: list[EvalExample] = []
    for task_name in task_names:
        examples.extend(load_task_examples(task_name, repo_root))
    return examples


def filter_examples_for_video(examples: list[EvalExample], video_id: str) -> list[EvalExample]:
    filtered: list[EvalExample] = []
    for example in examples:
        if any(video.video_id == video_id for video in example.input_videos):
            filtered.append(example)

    return filtered


def count_examples_by_task(examples: list[EvalExample]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for example in examples:
        counts[example.task_name] = counts.get(example.task_name, 0) + 1
    return counts
