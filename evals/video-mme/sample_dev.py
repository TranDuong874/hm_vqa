from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("thirdparty/Video-RAG-master/evals/videomme_json_file.json")
DEFAULT_OUTPUT = Path("evals/video-mme/video_mme_dev50.json")


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected top-level list in {path}")
    return payload


def _bucket_key(item: dict[str, Any]) -> tuple[str, str]:
    return (str(item["duration"]), str(item["domain"]))


def _video_sort_key(item: dict[str, Any]) -> tuple[str, str]:
    return (str(item.get("video_id", "")), str(item.get("url", "")))


def _stratified_video_sample(
    videos: list[dict[str, Any]],
    *,
    sample_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    if sample_size <= 0:
        return []
    if sample_size >= len(videos):
        return list(sorted(videos, key=_video_sort_key))

    rng = random.Random(seed)
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in videos:
        buckets[_bucket_key(item)].append(item)
    for items in buckets.values():
        rng.shuffle(items)

    total = len(videos)
    target_counts: dict[tuple[str, str], int] = {}
    remainders: list[tuple[float, tuple[str, str]]] = []
    selected_total = 0

    for key, items in buckets.items():
        raw = sample_size * (len(items) / total)
        base = math.floor(raw)
        if base == 0 and len(items) > 0:
            base = 1
        base = min(base, len(items))
        target_counts[key] = base
        selected_total += base
        remainders.append((raw - math.floor(raw), key))

    if selected_total > sample_size:
        removable = sorted(
            (
                (target_counts[key] - 1, len(buckets[key]), key)
                for key in target_counts
                if target_counts[key] > 1
            ),
            key=lambda item: (item[0], item[1]),
            reverse=True,
        )
        idx = 0
        while selected_total > sample_size and idx < len(removable):
            _, _, key = removable[idx]
            if target_counts[key] > 1:
                target_counts[key] -= 1
                selected_total -= 1
            else:
                idx += 1

    if selected_total < sample_size:
        for _, key in sorted(remainders, key=lambda item: item[0], reverse=True):
            capacity = len(buckets[key]) - target_counts[key]
            while capacity > 0 and selected_total < sample_size:
                target_counts[key] += 1
                selected_total += 1
                capacity -= 1
            if selected_total >= sample_size:
                break

    if selected_total < sample_size:
        for key, items in sorted(buckets.items(), key=lambda item: len(item[1]), reverse=True):
            capacity = len(items) - target_counts[key]
            while capacity > 0 and selected_total < sample_size:
                target_counts[key] += 1
                selected_total += 1
                capacity -= 1
            if selected_total >= sample_size:
                break

    sampled: list[dict[str, Any]] = []
    for key, items in sorted(buckets.items(), key=lambda item: item[0]):
        sampled.extend(items[: target_counts[key]])

    sampled = sorted(sampled, key=_video_sort_key)
    if len(sampled) != sample_size:
        raise RuntimeError(f"Expected {sample_size} sampled videos, got {len(sampled)}")
    return sampled


def _question_rows(videos: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in videos:
        for question in item.get("questions", []):
            rows.append(
                {
                    "video_id": item["video_id"],
                    "duration": item["duration"],
                    "domain": item["domain"],
                    "sub_category": item["sub_category"],
                    "url": item["url"],
                    "question_id": question["question_id"],
                    "task_type": question["task_type"],
                    "question": question["question"],
                    "options": question["options"],
                    "answer": question["answer"],
                }
            )
    return rows


def _counter_sorted(counter: Counter[str]) -> dict[str, int]:
    return {key: int(counter[key]) for key in sorted(counter)}


def _pair_counter_sorted(counter: Counter[tuple[str, str]]) -> dict[str, int]:
    return {f"{key[0]} | {key[1]}": int(counter[key]) for key in sorted(counter)}


def _build_stats(videos: list[dict[str, Any]]) -> dict[str, Any]:
    video_duration = Counter(str(item["duration"]) for item in videos)
    video_domain = Counter(str(item["domain"]) for item in videos)
    video_subcat = Counter(str(item["sub_category"]) for item in videos)
    video_buckets = Counter(_bucket_key(item) for item in videos)
    rows = _question_rows(videos)
    task_type = Counter(str(row["task_type"]) for row in rows)
    return {
        "num_videos": len(videos),
        "num_questions": len(rows),
        "video_duration": _counter_sorted(video_duration),
        "video_domain": _counter_sorted(video_domain),
        "video_sub_category": _counter_sorted(video_subcat),
        "video_duration_domain": _pair_counter_sorted(video_buckets),
        "question_task_type": _counter_sorted(task_type),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample a Video-MME development split stratified by duration x domain.")
    parser.add_argument("--input-json", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    input_path = Path(args.input_json)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    videos = _load_manifest(input_path)
    sampled = _stratified_video_sample(videos, sample_size=args.sample_size, seed=args.seed)
    stats = _build_stats(sampled)

    payload = {
        "source_json": str(input_path),
        "sample_size": int(args.sample_size),
        "seed": int(args.seed),
        "stats": stats,
        "videos": sampled,
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(payload["stats"], indent=2, ensure_ascii=False))
    print(f"saved={output_path}")


if __name__ == "__main__":
    main()
