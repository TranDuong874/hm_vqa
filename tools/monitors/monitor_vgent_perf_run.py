#!/usr/bin/env python3
"""Summarize cached-VGent performance shards."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from statistics import mean, median


STAMP_RE = re.compile(r"^\[(?P<ts>[^\]]+)\] (?P<msg>.*)$")
START_RE = re.compile(r"item_start .*example_id=(?P<id>\S+)")
DONE_RE = re.compile(r"item_done ")


def parse_ts(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[idx]


def read_rows(path: Path) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    total = 0
    correct = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            correct += int(bool(row.get("correct")))
    return total, correct


def read_log(path: Path) -> tuple[list[float], str]:
    if not path.exists():
        return [], "missing log"

    durations: list[float] = []
    current_start: datetime | None = None
    last_msg = ""

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = STAMP_RE.match(line.strip())
            if not match:
                continue
            ts = parse_ts(match.group("ts"))
            msg = match.group("msg")
            last_msg = msg

            if START_RE.search(msg):
                current_start = ts
            elif DONE_RE.search(msg) and current_start is not None:
                durations.append((ts - current_start).total_seconds())
                current_start = None

    return durations, last_msg


def fmt(value: float | None) -> str:
    return "--" if value is None else f"{value:.1f}s"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="results/vgent/rag_api_qwen2vl7b_cached_graph_v1",
        help="VGent result root.",
    )
    parser.add_argument(
        "--prefix",
        default="perf_cached_graph_100_v4_vast_shard_",
        help="Shard directory prefix.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    shards = sorted(p for p in root.glob(f"{args.prefix}*") if p.is_dir())
    if not shards:
        print(f"No shards found under {root} with prefix {args.prefix!r}")
        return

    all_durations: list[float] = []
    total_rows = 0
    total_correct = 0

    print("shard\trows\tacc\tmean\tmedian\tp90\tlast_log")
    for shard in shards:
        rows, correct = read_rows(shard / "rows.jsonl")
        durations, last_msg = read_log(shard / "run.log")
        total_rows += rows
        total_correct += correct
        all_durations.extend(durations)
        acc = correct / rows if rows else None
        print(
            f"{shard.name}\t{rows}\t"
            f"{'--' if acc is None else f'{acc:.4f}'}\t"
            f"{fmt(mean(durations) if durations else None)}\t"
            f"{fmt(median(durations) if durations else None)}\t"
            f"{fmt(percentile(durations, 0.9))}\t"
            f"{last_msg}"
        )

    acc = total_correct / total_rows if total_rows else None
    print()
    print(
        "TOTAL\t"
        f"rows={total_rows}\t"
        f"acc={'--' if acc is None else f'{acc:.4f}'}\t"
        f"mean={fmt(mean(all_durations) if all_durations else None)}\t"
        f"median={fmt(median(all_durations) if all_durations else None)}\t"
        f"p90={fmt(percentile(all_durations, 0.9))}"
    )


if __name__ == "__main__":
    main()
