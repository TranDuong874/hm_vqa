from __future__ import annotations

import json
from pathlib import Path

from evals.common.vlm_baseline_runner import BaselineExample


DEFAULT_MANIFEST = Path(
    "/home/tranduong/dev/hm_vqa/local_storage/flat_files/manifests/video_mme/video_mme_stratified_50_50_50_no_subs.json"
)
DEFAULT_VIDEO_ROOT = Path("/home/tranduong/dev/dataset/Video-MME/videos_subset_50_50_50")
DEFAULT_FEATURE_ROOT = Path("/home/tranduong/dev/hm_vqa/local_storage/flat_files/video_mme_features")
DEFAULT_DERIVED_CACHE_ROOT = Path("/home/tranduong/dev/hm_vqa/local_storage/flat_files/video_mme_derived_subset_50_50_50")
DEFAULT_OUTPUT_ROOT = Path("/home/tranduong/dev/hm_vqa/results/video_mme/ablations")


def load_video_mme_examples(
    manifest_path: Path = DEFAULT_MANIFEST,
    *,
    video_root: Path = DEFAULT_VIDEO_ROOT,
    limit: int | None = None,
) -> list[BaselineExample]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        rows = payload.get("rows") or payload.get("questions") or payload.get("data") or []
    else:
        rows = payload
    examples: list[BaselineExample] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        answer = str(row.get("answer") or row.get("correct_answer") or row.get("label") or "A").strip()
        correct_index = ord(answer[0].upper()) - ord("A") if answer else 0
        url = str(row.get("url") or row.get("video_id") or row.get("video") or row.get("videoID"))
        question_id = str(row.get("question_id") or row.get("qid") or row.get("id") or len(examples))
        options = row.get("options") or row.get("choices") or row.get("candidates") or []
        examples.append(
            BaselineExample(
                example_id=f"{url}::{question_id}",
                video_id=url,
                video_path=str(video_root / f"{url}.mp4"),
                question=str(row.get("question", "")),
                options=[str(option) for option in options],
                correct_index=int(correct_index),
                metadata={
                    "question_id": question_id,
                    "duration": str(row.get("duration", "")),
                    "domain": str(row.get("domain", "")),
                    "sub_category": str(row.get("sub_category", "")),
                    "task_type": str(row.get("task_type", "")),
                    "url": url,
                },
            )
        )
        if limit is not None and len(examples) >= limit:
            break
    return examples
