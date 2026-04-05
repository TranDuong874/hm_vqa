from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from common import ensure_local_video
from dataloader import DEFAULT_DATASET, DEFAULT_SPLIT, load_official_videomme_rows


DEFAULT_INPUT = None
DEFAULT_OUTPUT = Path("evals/video-mme/video_mme_dev50.json")


def _load_manifest(path: Path | None, hf_dataset: str, hf_split: str) -> list[dict[str, Any]]:
    if path is None:
        return load_official_videomme_rows(hf_dataset, hf_split)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "videos" in payload:
        return list(payload["videos"])
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unsupported manifest format in {path}")


def _bucket_key(item: dict[str, Any], bucket_fields: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(str(item[field]) for field in bucket_fields)


def _video_sort_key(item: dict[str, Any]) -> tuple[str, str]:
    return (str(item.get("video_id", "")), str(item.get("url", "")))


def _stratified_video_sample(
    videos: list[dict[str, Any]],
    *,
    sample_size: int,
    seed: int,
    bucket_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    if sample_size <= 0:
        return []
    if sample_size >= len(videos):
        return list(sorted(videos, key=_video_sort_key))

    rng = random.Random(seed)

    if not bucket_fields:
        shuffled = list(videos)
        rng.shuffle(shuffled)
        return sorted(shuffled[:sample_size], key=_video_sort_key)

    buckets: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for item in videos:
        buckets[_bucket_key(item, bucket_fields)].append(item)
    for items in buckets.values():
        rng.shuffle(items)

    total = len(videos)
    target_counts: dict[tuple[str, ...], int] = {}
    remainders: list[tuple[float, tuple[str, ...]]] = []
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


def _sample_with_duration_quotas(
    videos: list[dict[str, Any]],
    *,
    duration_quota: dict[str, int],
    seed: int,
    bucket_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    sampled: list[dict[str, Any]] = []
    for duration, quota in duration_quota.items():
        group = [item for item in videos if str(item["duration"]) == duration]
        if quota > len(group):
            raise ValueError(f"Requested {quota} videos for duration={duration}, but only {len(group)} available.")

        effective_bucket_fields = bucket_fields
        if "duration" in effective_bucket_fields:
            effective_bucket_fields = tuple(field for field in effective_bucket_fields if field != "duration")

        sampled.extend(
            _stratified_video_sample(
                group,
                sample_size=quota,
                seed=seed,
                bucket_fields=effective_bucket_fields,
            )
        )
    return sorted(sampled, key=_video_sort_key)


def _validate_local_videos(
    *,
    sampled: list[dict[str, Any]],
    video_root: Path,
) -> list[dict[str, Any]]:
    validated: list[dict[str, Any]] = []
    missing: list[str] = []
    for item in sampled:
        url_id = str(item["url"])
        try:
            ensure_local_video(video_root=video_root, url_id=url_id)
            validated.append(item)
        except FileNotFoundError:
            missing.append(url_id)
    if missing:
        preview = ", ".join(missing[:5])
        more = "" if len(missing) <= 5 else f" (+{len(missing) - 5} more)"
        raise FileNotFoundError(f"Missing local videos for sampled manifest: {preview}{more}")
    return validated


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
    video_buckets = Counter((str(item["duration"]), str(item["domain"])) for item in videos)
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
    parser = argparse.ArgumentParser(description="Sample a reusable Video-MME development split.")
    parser.add_argument("--input-json", default=None)
    parser.add_argument("--hf-dataset", default=DEFAULT_DATASET)
    parser.add_argument("--hf-split", default=DEFAULT_SPLIT)
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--bucket-fields",
        nargs="*",
        default=["duration", "domain"],
        help="Fields used for proportional stratification buckets.",
    )
    parser.add_argument(
        "--duration-quota",
        nargs="*",
        default=None,
        help="Optional per-duration quotas, e.g. short=25 medium=25",
    )
    parser.add_argument("--video-root", default="dataset/Video-MME")
    parser.add_argument("--validate-local", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input_json) if args.input_json else None
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    videos = _load_manifest(input_path, args.hf_dataset, args.hf_split)
    bucket_fields = tuple(args.bucket_fields)

    duration_quota: dict[str, int] | None = None
    if args.duration_quota:
        duration_quota = {}
        for item in args.duration_quota:
            if "=" not in item:
                raise ValueError(f"Invalid duration quota item: {item}")
            key, value = item.split("=", 1)
            duration_quota[key] = int(value)

    if duration_quota is not None:
        sampled = _sample_with_duration_quotas(
            videos,
            duration_quota=duration_quota,
            seed=args.seed,
            bucket_fields=bucket_fields,
        )
        sample_size = sum(duration_quota.values())
    else:
        sampled = _stratified_video_sample(
            videos,
            sample_size=args.sample_size,
            seed=args.seed,
            bucket_fields=bucket_fields,
        )
        sample_size = int(args.sample_size)

    if args.validate_local:
        sampled = _validate_local_videos(
            sampled=sampled,
            video_root=Path(args.video_root),
        )

    stats = _build_stats(sampled)

    payload = {
        "source_json": str(input_path) if input_path is not None else None,
        "source_dataset": args.hf_dataset,
        "source_split": args.hf_split,
        "sample_size": int(sample_size),
        "seed": int(args.seed),
        "bucket_fields": list(bucket_fields),
        "duration_quota": duration_quota,
        "video_root": str(args.video_root),
        "validate_local": bool(args.validate_local),
        "stats": stats,
        "videos": sampled,
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(payload["stats"], indent=2, ensure_ascii=False))
    print(f"saved={output_path}")


if __name__ == "__main__":
    main()
