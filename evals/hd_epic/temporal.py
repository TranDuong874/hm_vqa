from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


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

TIME_PAIR_PATTERN = re.compile(
    r"<TIME\s+(?P<start>\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s+(?P<alias_start>video\s+\d+)>\s+to\s+<TIME\s+"
    r"(?P<end>\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s+(?P<alias_end>video\s+\d+)>",
    re.IGNORECASE,
)


@dataclass(slots=True)
class InputVideoRef:
    alias: str
    video_id: str
    start_time_sec: float | None
    end_time_sec: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TemporalSpan:
    video_alias: str
    start_time_sec: float
    end_time_sec: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TemporalExample:
    example_id: str
    task_name: str
    question: str
    correct_idx: int
    input_videos: list[InputVideoRef]
    gold_spans: list[TemporalSpan]

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "task_name": self.task_name,
            "question": self.question,
            "correct_idx": self.correct_idx,
            "input_videos": [video.to_dict() for video in self.input_videos],
            "gold_spans": [span.to_dict() for span in self.gold_spans],
        }


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
        normalized.append(
            InputVideoRef(
                alias=alias.strip().lower(),
                video_id=str(raw["id"]),
                start_time_sec=parse_timecode(raw.get("start_time")),
                end_time_sec=parse_timecode(raw.get("end_time")),
            )
        )
    return normalized


def load_temporal_examples_for_video(
    *,
    repo_root: Path,
    video_id: str,
    tasks: tuple[str, ...],
) -> list[TemporalExample]:
    examples: list[TemporalExample] = []
    for task_name in tasks:
        data = json.loads((repo_root / TASK_FILES[task_name]).read_text())
        for example_id in sorted(data.keys()):
            item = data[example_id]
            input_videos = _normalize_input_videos(item.get("inputs", {}))
            matching = [video for video in input_videos if video.video_id == video_id]
            if not matching:
                continue
            gold_spans = parse_choice_spans(item["choices"][int(item["correct_idx"])])
            if not gold_spans:
                continue
            examples.append(
                TemporalExample(
                    example_id=example_id,
                    task_name=task_name,
                    question=item["question"],
                    correct_idx=int(item["correct_idx"]),
                    input_videos=input_videos,
                    gold_spans=gold_spans,
                )
            )
    return examples


def example_scope_for_video(example: TemporalExample, video_id: str) -> tuple[float | None, float | None]:
    matching = [video for video in example.input_videos if video.video_id == video_id]
    if len(matching) == 1 and matching[0].start_time_sec is not None and matching[0].end_time_sec is not None:
        return matching[0].start_time_sec, matching[0].end_time_sec
    return None, None


def gold_spans_for_video(example: TemporalExample, video_id: str) -> list[dict[str, Any]]:
    alias_to_video_id = {video.alias: video.video_id for video in example.input_videos}
    return [
        {
            "video_id": alias_to_video_id.get(span.video_alias, span.video_alias),
            "start_time_sec": float(span.start_time_sec),
            "end_time_sec": float(span.end_time_sec),
        }
        for span in example.gold_spans
        if alias_to_video_id.get(span.video_alias, span.video_alias) == video_id
    ]
