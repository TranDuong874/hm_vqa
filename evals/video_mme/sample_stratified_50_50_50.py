from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from .dataloader import DEFAULT_PARQUET, VideoMMELoader


DEFAULT_OUTPUT = Path("evals/video_mme/manifests/video_mme_stratified_50_50_50_no_subs.json")
DEFAULT_SEED = 7
DEFAULT_COUNTS = {"short": 50, "medium": 50, "long": 50}


def _video_sort_key(item: dict[str, Any]) -> tuple[str, str]:
    return (str(item.get("video_id", "")), str(item.get("url", "")))


def _counter_sorted(counter: Counter[str]) -> dict[str, int]:
    return {key: int(counter[key]) for key in sorted(counter)}


def _pair_counter_sorted(counter: Counter[tuple[str, str]]) -> dict[str, int]:
    return {f"{key[0]} | {key[1]}": int(counter[key]) for key in sorted(counter)}


def _build_stats(videos: list[dict[str, Any]]) -> dict[str, Any]:
    video_duration = Counter(str(item["duration"]) for item in videos)
    video_domain = Counter(str(item["domain"]) for item in videos)
    video_subcat = Counter(str(item["sub_category"]) for item in videos)
    video_buckets = Counter((str(item["duration"]), str(item["domain"])) for item in videos)
    num_questions = sum(len(item.get("questions", [])) for item in videos)
    return {
        "num_videos": len(videos),
        "num_questions": num_questions,
        "video_duration": _counter_sorted(video_duration),
        "video_domain": _counter_sorted(video_domain),
        "video_sub_category": _counter_sorted(video_subcat),
        "video_duration_domain": _pair_counter_sorted(video_buckets),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a deterministic Video-MME 50/50/50 no-subtitle split.")
    parser.add_argument("--parquet-path", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    loader = VideoMMELoader(parquet_path=args.parquet_path)
    videos = [
        {
            "video_id": video.video_id,
            "duration": video.duration,
            "domain": video.domain,
            "sub_category": video.sub_category,
            "url": video.url,
            "questions": [
                {
                    "question_id": q.question_id,
                    "task_type": q.task_type,
                    "question": q.question,
                    "options": q.options,
                    "answer": q.answer,
                }
                for q in video.questions
            ],
        }
        for video in loader.load()
    ]

    rng = random.Random(args.seed)
    sampled: list[dict[str, Any]] = []
    for duration, target in DEFAULT_COUNTS.items():
        bucket = [item for item in videos if str(item["duration"]) == duration]
        if len(bucket) < target:
            raise ValueError(f"Need {target} videos for duration={duration}, found {len(bucket)}")
        rng.shuffle(bucket)
        sampled.extend(sorted(bucket[:target], key=_video_sort_key))

    sampled = sorted(sampled, key=_video_sort_key)
    payload = {
        "name": "video_mme_stratified_50_50_50_no_subs",
        "seed": args.seed,
        "uses_subtitles": False,
        "source": {
            "parquet_path": str(args.parquet_path),
        },
        "requested_video_counts": DEFAULT_COUNTS,
        "stats": _build_stats(sampled),
        "videos": sampled,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {args.output_json}")
    print(json.dumps(payload["stats"], indent=2))


if __name__ == "__main__":
    main()
