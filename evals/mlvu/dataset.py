from __future__ import annotations

import json
from pathlib import Path

from evals.common.vlm_baseline_runner import BaselineExample


DEFAULT_DATA_ROOT = Path("/home/tranduong/dev/dataset/mlvu_test")
DEFAULT_MANIFEST = DEFAULT_DATA_ROOT / "test-ground-truth/test_mcq_gt.json"
DEFAULT_VIDEO_ROOT = DEFAULT_DATA_ROOT / "extracted/video"
DEFAULT_FEATURE_ROOT = Path("/home/tranduong/dev/hm_vqa/local_storage/flat_files/mlvu/openclip_1fps_test_mcq_v1")
DEFAULT_DERIVED_CACHE_ROOT = Path("/home/tranduong/dev/hm_vqa/local_storage/flat_files/mlvu/derived_test_mcq_v1")
DEFAULT_OUTPUT_ROOT = Path("/home/tranduong/dev/hm_vqa/results/mlvu/ablations")


def _answer_index(answer: str, choices: list[str]) -> int | None:
    normalized = answer.strip().lower()
    for idx, choice in enumerate(choices):
        if normalized == choice.strip().lower():
            return idx
    if len(answer.strip()) == 1 and answer.strip().isalpha():
        idx = ord(answer.strip().upper()) - ord("A")
        if 0 <= idx < len(choices):
            return idx
    return None


def load_mlvu_examples(
    manifest_path: Path = DEFAULT_MANIFEST,
    *,
    video_root: Path = DEFAULT_VIDEO_ROOT,
    limit: int | None = None,
) -> list[BaselineExample]:
    rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise TypeError(f"Expected list rows in {manifest_path}")
    examples: list[BaselineExample] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        video_name = str(row["video"])
        choices = [str(candidate) for candidate in row.get("candidates", [])]
        answer = str(row.get("answer", ""))
        examples.append(
            BaselineExample(
                example_id=str(row.get("question_id") or f"{video_name}::{len(examples)}"),
                video_id=Path(video_name).stem,
                video_path=str(video_root / video_name),
                question=str(row.get("question", "")),
                options=choices,
                correct_index=_answer_index(answer, choices),
                metadata={
                    "video": video_name,
                    "duration": row.get("duration"),
                    "question_type": row.get("question_type"),
                    "question_id": row.get("question_id"),
                    "answer": answer,
                },
            )
        )
        if limit is not None and len(examples) >= limit:
            break
    return examples
