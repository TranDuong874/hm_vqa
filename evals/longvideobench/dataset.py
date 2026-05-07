from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hm_vqa.schema import BenchmarkItem, QAExample, RetrievalExample
from evals.longvideobench.paths import LVB_FULL_MANIFEST, LVB_FULL_VIDEO_ROOT


def _rows_from_manifest(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows", payload)
    if not isinstance(rows, list):
        raise TypeError(f"Expected list rows in {path}")
    return payload if isinstance(payload, dict) else {}, rows


def load_benchmark_items(
    manifest_path: Path = LVB_FULL_MANIFEST,
    *,
    video_root: Path = LVB_FULL_VIDEO_ROOT,
    limit: int | None = None,
) -> list[BenchmarkItem]:
    payload, rows = _rows_from_manifest(manifest_path)
    if limit is not None:
        rows = rows[:limit]
    items: list[BenchmarkItem] = []
    for row in rows:
        example_id = str(row["id"])
        video_id = str(row["video_id"])
        question = str(row["question"])
        duration = row.get("duration")
        metadata = {
            "split": payload.get("source_split"),
            "question_category": row.get("question_category"),
            "level": row.get("level"),
            "duration_group": row.get("duration_group"),
            "duration": duration,
            "topic_category": row.get("topic_category"),
            "subtitle_path": row.get("subtitle_path"),
            "starting_timestamp_for_subtitles": row.get("starting_timestamp_for_subtitles"),
        }
        items.append(
            BenchmarkItem(
                retrieval=RetrievalExample(
                    example_id=example_id,
                    dataset="longvideobench",
                    split=str(payload.get("source_split") or "val"),
                    video_id=video_id,
                    video_path=video_root / str(row["video_path"]),
                    query=question,
                    duration_sec=float(duration) if duration not in (None, "") else None,
                    metadata=metadata,
                ),
                qa=QAExample(
                    example_id=example_id,
                    question=question,
                    answer_type="mcq",
                    choices=[str(option) for option in row["candidates"]],
                    answer_index=int(row["correct_choice"]) if "correct_choice" in row else None,
                    metadata=metadata,
                ),
            )
        )
    return items

