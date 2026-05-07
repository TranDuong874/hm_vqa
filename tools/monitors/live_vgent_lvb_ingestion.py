from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path


DEFAULT_MANIFEST = Path("local_storage/flat_files/manifests/longvideobench/lvb_val_full_1337_v1.json")
DEFAULT_OUTPUT = Path("local_storage/flat_files/vgent/offline_graph_cache_qwen2vl7b_api_lvb_fullval_1fps64_336_1024_w3")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path: Path) -> list[dict]:
    payload = read_json(path)
    return payload.get("rows", payload) if isinstance(payload, dict) else payload


def load_video_plan(manifest: Path, sample_fps: float, chunk_size: int) -> dict[str, dict]:
    videos: dict[str, dict] = {}
    for row in read_rows(manifest):
        video_id = str(row["video_id"])
        if video_id in videos:
            continue
        duration = float(row.get("duration") or 0.0)
        chunks = max(1, math.ceil(duration * sample_fps / chunk_size))
        videos[video_id] = {
            "duration": duration,
            "chunks": chunks,
            "duration_group": row.get("duration_group"),
        }
    return videos


def iter_chunk_rows(output_root: Path):
    for chunks_path in sorted((output_root / "lvb").glob("*/chunks.jsonl")):
        video_id = chunks_path.parent.name
        try:
            file_mtime = chunks_path.stat().st_mtime
        except FileNotFoundError:
            continue
        for line in chunks_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            yield video_id, file_mtime, json.loads(line)


def format_eta(hours: float | None) -> str:
    if hours is None:
        return "n/a"
    if hours < 1:
        return f"{hours * 60:.0f} min"
    return f"{hours:.2f} h"


def render_status(*, manifest: Path, output_root: Path, sample_fps: float, chunk_size: int, workers: int) -> None:
    plan = load_video_plan(manifest, sample_fps, chunk_size)
    total_chunks = sum(item["chunks"] for item in plan.values())
    total_videos = len(plan)

    seen_videos: set[str] = set()
    ok = 0
    err = 0
    partial = 0
    elapsed_sum = 0.0
    gen_sum = 0.0
    rows = 0
    by_group: dict[str, dict[str, int]] = {}
    rows_by_file_mtime: list[float] = []
    newest_mtime = 0.0
    for video_id, file_mtime, row in iter_chunk_rows(output_root):
        rows += 1
        rows_by_file_mtime.append(file_mtime)
        newest_mtime = max(newest_mtime, file_mtime)
        seen_videos.add(video_id)
        status = row.get("status")
        if status == "ok":
            ok += 1
        else:
            err += 1
        if row.get("parsed", {}).get("partial"):
            partial += 1
        elapsed_sum += float(row.get("elapsed_sec") or 0.0)
        gen_sum += float(row.get("generation_sec") or 0.0)
        group = str(plan.get(video_id, {}).get("duration_group", "unknown"))
        bucket = by_group.setdefault(group, {"rows": 0, "ok": 0, "err": 0})
        bucket["rows"] += 1
        bucket["ok"] += int(status == "ok")
        bucket["err"] += int(status != "ok")

    now_ts = time.time()
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    pct = rows / total_chunks * 100 if total_chunks else 0.0
    mean_elapsed = elapsed_sum / rows if rows else 0.0
    mean_gen = gen_sum / ok if ok else 0.0
    eta_chunks = max(total_chunks - rows, 0)
    ideal_eta_sec = eta_chunks * mean_elapsed / max(1, int(workers)) if rows else 0.0

    throughput_windows = [5, 10, 15, 30, 60]
    throughput_rows: list[tuple[int, int, float, float | None]] = []
    for minutes in throughput_windows:
        cutoff = now_ts - minutes * 60
        completed = sum(1 for mtime in rows_by_file_mtime if mtime >= cutoff)
        rate = completed / minutes if minutes > 0 else 0.0
        eta_hours = (eta_chunks / rate / 60) if rate > 0 else None
        throughput_rows.append((minutes, completed, rate, eta_hours))

    print("# Live Vgent LVB Ingestion")
    print()
    print(f"Updated: `{now}`")
    print(f"Output: `{output_root}`")
    print()
    print("| Metric | Value |")
    print("|---|---:|")
    print(f"| Videos touched | {len(seen_videos)}/{total_videos} |")
    print(f"| Chunks written | {rows}/{total_chunks} ({pct:.2f}%) |")
    print(f"| OK chunks | {ok} |")
    print(f"| Error chunks | {err} |")
    print(f"| Partial-parse chunks | {partial} |")
    print(f"| Mean generation sec / OK chunk | {mean_gen:.2f} |")
    print(f"| Mean elapsed sec / written chunk | {mean_elapsed:.2f} |")
    if newest_mtime > 0:
        print(f"| Newest chunk file update | {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(newest_mtime))} |")
        print(f"| Seconds since newest update | {now_ts - newest_mtime:.1f} |")
    print(f"| Idealized worker ETA, misleading | {ideal_eta_sec / 3600:.2f} h |")
    live_10m = next((eta for minutes, _, _, eta in throughput_rows if minutes == 10), None)
    live_30m = next((eta for minutes, _, _, eta in throughput_rows if minutes == 30), None)
    print(f"| Live ETA from 10m file writes | {format_eta(live_10m)} |")
    print(f"| Live ETA from 30m file writes | {format_eta(live_30m)} |")
    print()
    print("| Window | Rows in touched files | Approx chunks/min | ETA from rate |")
    print("|---|---:|---:|---:|")
    for minutes, completed, rate, eta_hours in throughput_rows:
        print(f"| last {minutes}m | {completed} | {rate:.2f} | {format_eta(eta_hours)} |")
    print()
    print("| Duration group | Rows | OK | Error |")
    print("|---|---:|---:|---:|")
    for group in sorted(by_group, key=lambda x: int(x) if x.isdigit() else 999999):
        item = by_group[group]
        print(f"| {group} | {item['rows']} | {item['ok']} | {item['err']} |")


def main_args(
    *,
    output_root: Path,
    workers: int,
    manifest: Path = DEFAULT_MANIFEST,
    sample_fps: float = 1.0,
    chunk_size: int = 64,
) -> None:
    render_status(
        manifest=manifest,
        output_root=output_root,
        sample_fps=sample_fps,
        chunk_size=chunk_size,
        workers=workers,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    main_args(
        manifest=args.manifest,
        output_root=args.output_root,
        sample_fps=args.sample_fps,
        chunk_size=args.chunk_size,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
