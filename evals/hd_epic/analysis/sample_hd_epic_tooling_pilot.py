from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = REPO_ROOT / "results" / "pipeline" / "splits" / "hd_epic_tooling_pilot_v1.json"
DEFAULT_SEED = 7
DEFAULT_PER_TASK = 30
PER_VIDEO_CAP = 4
TASKS = [
    "fine_grained_action_localization",
    "recipe_step_localization",
    "fine_grained_action_recognition",
    "recipe_step_recognition",
    "ingredient_ingredients_order",
    "object_motion_object_movement_counting",
    "fine_grained_how_recognition",
    "fine_grained_why_recognition",
]
TASK_FILES = {
    task_name: REPO_ROOT / "dataset" / "hd-epic-annotations" / "vqa-benchmark" / f"{task_name}.json"
    for task_name in TASKS
}
QUERY_FAMILY_BY_TASK = {
    "fine_grained_action_localization": "Entity->Time",
    "recipe_step_localization": "Entity->Time",
    "fine_grained_action_recognition": "Time->Entity",
    "recipe_step_recognition": "Time->Entity",
    "ingredient_ingredients_order": "Entity->Entity",
    "object_motion_object_movement_counting": "Time->Entity",
    "fine_grained_how_recognition": "Entity->Entity",
    "fine_grained_why_recognition": "Entity->Entity",
}
TEMPORAL_TASKS = {
    "fine_grained_action_localization",
    "recipe_step_localization",
}


@dataclass(slots=True)
class CandidateExample:
    task_name: str
    example_id: str
    question: str
    correct_idx: int
    evaluation_video_id: str
    input_video_ids: list[str]
    video_length_sec: float
    length_bucket: str
    hardness: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_name": self.task_name,
            "example_id": self.example_id,
            "question": self.question,
            "correct_idx": self.correct_idx,
            "evaluation_video_id": self.evaluation_video_id,
            "input_video_ids": self.input_video_ids,
            "video_length_sec": self.video_length_sec,
            "length_bucket": self.length_bucket,
            "hardness": self.hardness,
            "query_family": QUERY_FAMILY_BY_TASK[self.task_name],
        }


def _parse_timecode(value: str | None) -> float | None:
    if value is None:
        return None
    hours, minutes, seconds = value.split(":")
    return int(hours) * 3600.0 + int(minutes) * 60.0 + float(seconds)


def _duration_for_video(video_id: str, cache: dict[str, float]) -> float | None:
    if video_id in cache:
        return cache[video_id]
    person_id = video_id.split("-", 1)[0]
    video_path = REPO_ROOT / "dataset" / "data" / "HD-EPIC" / "Videos" / person_id / f"{video_id}.mp4"
    if not video_path.exists():
        cache[video_id] = None  # type: ignore[assignment]
        return None
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            cache[video_id] = None  # type: ignore[assignment]
            return None
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if fps <= 0.0 or frames <= 0:
            cache[video_id] = None  # type: ignore[assignment]
            return None
        duration = frames / fps
        cache[video_id] = float(duration)
        return float(duration)
    finally:
        capture.release()


def _length_bucket(duration_sec: float) -> str:
    if duration_sec < 300.0:
        return "short"
    if duration_sec < 900.0:
        return "medium"
    return "long"


def _gold_video_ids(task_name: str, item: dict[str, Any]) -> list[str]:
    if task_name not in TEMPORAL_TASKS:
        return []
    choice = item["choices"][int(item["correct_idx"])]
    if not isinstance(choice, str):
        return []
    ids: list[str] = []
    for alias, video in item.get("inputs", {}).items():
        if alias in choice:
            ids.append(str(video["id"]))
    return ids


def _evaluation_video_id(task_name: str, item: dict[str, Any]) -> str | None:
    gold_ids = _gold_video_ids(task_name, item)
    if gold_ids:
        return gold_ids[0]
    video_one = item.get("inputs", {}).get("video 1")
    if video_one is not None:
        return str(video_one["id"])
    for raw in item.get("inputs", {}).values():
        if "id" in raw:
            return str(raw["id"])
    return None


def _hardness_label(task_name: str, item: dict[str, Any]) -> str | None:
    if task_name not in TEMPORAL_TASKS:
        return None
    stat = item.get("stat")
    if not isinstance(stat, dict):
        return None
    scores = {name: int(value) for name, value in stat.items() if isinstance(value, int)}
    if not scores:
        return None
    return max(scores.items(), key=lambda pair: pair[1])[0]


def _load_task_candidates(task_name: str, duration_cache: dict[str, float]) -> list[CandidateExample]:
    payload = json.loads(TASK_FILES[task_name].read_text(encoding="utf-8"))
    candidates: list[CandidateExample] = []
    for example_id in sorted(payload.keys()):
        item = payload[example_id]
        evaluation_video_id = _evaluation_video_id(task_name, item)
        if evaluation_video_id is None:
            continue
        duration_sec = _duration_for_video(evaluation_video_id, duration_cache)
        if duration_sec is None:
            continue
        candidates.append(
            CandidateExample(
                task_name=task_name,
                example_id=example_id,
                question=item["question"],
                correct_idx=int(item["correct_idx"]),
                evaluation_video_id=evaluation_video_id,
                input_video_ids=sorted(
                    str(raw["id"])
                    for raw in item.get("inputs", {}).values()
                    if isinstance(raw, dict) and raw.get("id")
                ),
                video_length_sec=float(duration_sec),
                length_bucket=_length_bucket(float(duration_sec)),
                hardness=_hardness_label(task_name, item),
            )
        )
    return candidates


def _sample_task(candidates: list[CandidateExample], *, per_task: int, rng: random.Random) -> list[CandidateExample]:
    by_bucket: dict[str, list[CandidateExample]] = defaultdict(list)
    for candidate in candidates:
        by_bucket[candidate.length_bucket].append(candidate)
    for bucket_candidates in by_bucket.values():
        rng.shuffle(bucket_candidates)

    video_counts: Counter[str] = Counter()
    selected: list[CandidateExample] = []
    target_order = ["short", "medium", "long"]
    while len(selected) < min(per_task, len(candidates)):
        progress = False
        for bucket in target_order:
            pool = by_bucket[bucket]
            if not pool:
                continue
            picked_index = None
            for index, candidate in enumerate(pool):
                if video_counts[candidate.evaluation_video_id] < PER_VIDEO_CAP:
                    picked_index = index
                    break
            if picked_index is None:
                picked_index = 0
            candidate = pool.pop(picked_index)
            selected.append(candidate)
            video_counts[candidate.evaluation_video_id] += 1
            progress = True
            if len(selected) >= min(per_task, len(candidates)):
                break
        if not progress:
            break
    return selected


def build_pilot_manifest(*, per_task: int = DEFAULT_PER_TASK, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    rng = random.Random(seed)
    duration_cache: dict[str, float] = {}
    selected_entries: list[CandidateExample] = []
    task_summaries: list[dict[str, Any]] = []

    for task_name in TASKS:
        candidates = _load_task_candidates(task_name, duration_cache)
        sampled = _sample_task(candidates, per_task=per_task, rng=rng)
        selected_entries.extend(sampled)
        task_summaries.append(
            {
                "task_name": task_name,
                "query_family": QUERY_FAMILY_BY_TASK[task_name],
                "target_count": per_task,
                "available_count": len(candidates),
                "actual_count": len(sampled),
                "shortfall": max(per_task - len(sampled), 0),
                "video_counts": dict(sorted(Counter(item.evaluation_video_id for item in sampled).items())),
                "length_bucket_counts": dict(sorted(Counter(item.length_bucket for item in sampled).items())),
                "hardness_counts": dict(sorted(Counter(item.hardness for item in sampled if item.hardness).items())),
            }
        )

    selected_entries.sort(key=lambda item: (item.task_name, item.evaluation_video_id, item.example_id))
    payload = {
        "split_name": "hd_epic_tooling_pilot_v1",
        "seed": seed,
        "per_task_target": per_task,
        "per_video_cap": PER_VIDEO_CAP,
        "length_bucket_rule": {
            "short": "<300s",
            "medium": "300s-899.999s",
            "long": ">=900s",
        },
        "tasks": TASKS,
        "query_family_by_task": QUERY_FAMILY_BY_TASK,
        "task_summaries": task_summaries,
        "global_video_counts": dict(sorted(Counter(item.evaluation_video_id for item in selected_entries).items())),
        "global_length_bucket_counts": dict(sorted(Counter(item.length_bucket for item in selected_entries).items())),
        "entries": [item.to_dict() for item in selected_entries],
    }
    return payload


def main() -> None:
    payload = build_pilot_manifest()
    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(DEFAULT_OUTPUT)


if __name__ == "__main__":
    main()
