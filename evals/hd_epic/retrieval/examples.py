from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evals.common.vlm_baseline_runner import BaselineExample
from evals.hd_epic.dataset import TemporalExample, load_temporal_examples_for_video


REPO_ROOT = Path("/home/tranduong/dev/hm_vqa")
DEFAULT_TASKS = ("fine_grained_action_localization",)


def _participant_video_ids(feature_root: Path, participant: str) -> list[str]:
    video_ids = [path.name for path in sorted(feature_root.iterdir()) if path.is_dir() and path.name.startswith(f"{participant}-")]
    return video_ids


def _load_manifest_rows(manifest_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(manifest_path.read_text())
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"Manifest missing rows list: {manifest_path}")
    return [dict(row) for row in rows]


def _build_examples(*, video_root: Path, video_ids: list[str]) -> list[BaselineExample]:
    examples: list[BaselineExample] = []
    for video_id in video_ids:
        temporal_examples = load_temporal_examples_for_video(
            repo_root=REPO_ROOT,
            video_id=video_id,
            tasks=DEFAULT_TASKS,
        )
        for example in temporal_examples:
            examples.append(
                BaselineExample(
                    example_id=example.example_id,
                    video_id=video_id,
                    video_path=str(video_root / video_id.split("-")[0] / f"{video_id}.mp4"),
                    question=example.question,
                    options=[],
                    correct_index=None,
                    metadata={"task_name": example.task_name},
                )
            )
    return examples


def _build_examples_from_manifest(*, video_root: Path, manifest_rows: list[dict[str, Any]]) -> list[BaselineExample]:
    rows_by_video: dict[str, set[str]] = {}
    for row in manifest_rows:
        video_id = str(row["video_id"])
        example_id = str(row["example_id"])
        rows_by_video.setdefault(video_id, set()).add(example_id)

    examples: list[BaselineExample] = []
    for video_id in sorted(rows_by_video):
        temporal_examples = load_temporal_examples_for_video(
            repo_root=REPO_ROOT,
            video_id=video_id,
            tasks=DEFAULT_TASKS,
        )
        allowed_example_ids = rows_by_video[video_id]
        for example in temporal_examples:
            if str(example.example_id) not in allowed_example_ids:
                continue
            examples.append(
                BaselineExample(
                    example_id=example.example_id,
                    video_id=video_id,
                    video_path=str(video_root / video_id.split("-")[0] / f"{video_id}.mp4"),
                    question=example.question,
                    options=[],
                    correct_index=None,
                    metadata={"task_name": example.task_name},
                )
            )
    return examples


def _temporal_examples_by_video(video_ids: list[str]) -> dict[str, list[TemporalExample]]:
    return {
        video_id: load_temporal_examples_for_video(
            repo_root=REPO_ROOT,
            video_id=video_id,
            tasks=DEFAULT_TASKS,
        )
        for video_id in video_ids
    }


def _lookup_temporal_example(
    temporal_examples: dict[str, list[TemporalExample]],
    *,
    video_id: str,
    example_id: str,
) -> TemporalExample:
    for example in temporal_examples[video_id]:
        if example.example_id == example_id:
            return example
    raise KeyError(f"Missing temporal example {example_id} for {video_id}")

