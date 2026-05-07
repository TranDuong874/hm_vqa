from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path


def fmt_eta(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.0f} min"
    return f"{minutes / 60:.2f} h"


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset", default="lvb")
    parser.add_argument("--sample-fps", default="1")
    parser.add_argument("--chunk-size", default="64")
    parser.add_argument("--status-file", type=Path, required=True)
    args = parser.parse_args()

    dataset_dir = args.output_root / args.dataset
    graph_dir = args.output_root / "graphs" / f"{args.dataset}_{args.sample_fps}fps_{args.chunk_size}"
    log_path = args.output_root / "run_bge_graph_full.log"

    video_dirs = sorted(path for path in dataset_dir.iterdir() if path.is_dir()) if dataset_dir.exists() else []
    complete_inputs = []
    incomplete_inputs = []
    for video_dir in video_dirs:
        chunks_path = video_dir / "chunks.jsonl"
        meta_path = video_dir / "meta.json"
        if not chunks_path.exists() or not meta_path.exists():
            incomplete_inputs.append(video_dir)
            continue
        try:
            planned = int(json.loads(meta_path.read_text(encoding="utf-8")).get("chunks_planned") or 0)
        except Exception:
            incomplete_inputs.append(video_dir)
            continue
        rows = count_lines(chunks_path)
        if planned > 0 and rows >= planned:
            complete_inputs.append(video_dir)
        else:
            incomplete_inputs.append(video_dir)

    graph_files = sorted(graph_dir.glob("*.pkl")) if graph_dir.exists() else []
    embedding_meta = sorted(dataset_dir.glob("*/bge_embeddings_meta.json")) if dataset_dir.exists() else []
    done_names = {path.stem for path in graph_files}
    remaining = [path.name for path in complete_inputs if path.name not in done_names]

    now = dt.datetime.now()
    windows = []
    for minutes in (5, 10, 30, 60):
        cutoff = now.timestamp() - minutes * 60
        recent = [path for path in graph_files if path.stat().st_mtime >= cutoff]
        rate_per_min = len(recent) / minutes
        eta = len(remaining) / rate_per_min * 60 if rate_per_min > 0 else None
        windows.append((minutes, len(recent), rate_per_min, eta))

    newest_graph = max((path.stat().st_mtime for path in graph_files), default=None)
    newest_text = "n/a"
    seconds_since = "n/a"
    if newest_graph is not None:
        newest_dt = dt.datetime.fromtimestamp(newest_graph)
        newest_text = newest_dt.strftime("%Y-%m-%d %H:%M:%S")
        seconds_since = f"{(now - newest_dt).total_seconds():.1f}"

    active = False
    for pid in filter(str.isdigit, os.listdir("/proc")):
        try:
            cmdline = Path("/proc") / pid / "cmdline"
            text = cmdline.read_text(errors="ignore").replace("\x00", " ")
        except Exception:
            continue
        if "build_vgent_bge_from_chunks.py" in text:
            active = True
            break

    tail = []
    if log_path.exists():
        lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        tail = lines[-5:]

    args.status_file.parent.mkdir(parents=True, exist_ok=True)
    args.status_file.write_text(
        "\n".join(
            [
                "# Live Vgent Graph Build",
                "",
                f"Updated: `{now.strftime('%Y-%m-%d %H:%M:%S')}`",
                f"Output: `{args.output_root}`",
                "",
                "| Metric | Value |",
                "|---|---:|",
                f"| Process active | `{str(active).lower()}` |",
                f"| Complete chunk inputs | {len(complete_inputs)} |",
                f"| Graph files | {len(graph_files)}/{len(complete_inputs)} |",
                f"| BGE embedding caches | {len(embedding_meta)}/{len(complete_inputs)} |",
                f"| Remaining graphs | {len(remaining)} |",
                f"| Incomplete input videos | {len(incomplete_inputs)} |",
                f"| Newest graph update | {newest_text} |",
                f"| Seconds since newest graph | {seconds_since} |",
                "",
                "| Window | New graphs | Graphs/min | ETA from rate |",
                "|---|---:|---:|---:|",
                *[
                    f"| last {minutes}m | {count} | {rate:.2f} | {fmt_eta(eta)} |"
                    for minutes, count, rate, eta in windows
                ],
                "",
                "## Remaining Sample",
                "",
                "```text",
                *remaining[:20],
                "```",
                "",
                "## Log Tail",
                "",
                "```text",
                *tail,
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
