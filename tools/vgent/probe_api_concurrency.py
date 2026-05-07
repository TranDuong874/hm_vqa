from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

def _repo_root() -> Path:
    path = Path(__file__).resolve()
    for parent in path.parents:
        if (parent / "src").exists() and (parent / "evals").exists():
            return parent
    return path.parents[2]


REPO_ROOT = _repo_root()
VGENT_ROOT = REPO_ROOT / "thirdparty" / "Vgent"
if str(VGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(VGENT_ROOT))

from answering.factory import build_answerer
from answering.qwen_vl import AnswerConfig
from segmentation.video import sample_video_selected_indices
from utils.prompts import GRAPH_PROMPT


def load_manifest(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("rows", payload) if isinstance(payload, dict) else payload


def pick_video(manifest: Path, video_root: Path, min_chunks: int, chunk_size: int, sample_fps: float) -> tuple[str, Path]:
    for row in load_manifest(manifest):
        duration = float(row.get("duration") or 0.0)
        if duration * sample_fps >= min_chunks * chunk_size:
            video_path = video_root / str(row["video_path"])
            if video_path.exists():
                return str(row["video_id"]), video_path
    raise RuntimeError("No video with enough chunks found.")


def run_one(answerer, video_path: Path, chunk_id: int, chunk_size: int, sample_fps: float, image_max_size: int, max_new_tokens: int) -> dict:
    start = chunk_id * chunk_size
    end = start + chunk_size
    t0 = time.perf_counter()
    try:
        frames, _, _ = sample_video_selected_indices(
            video_path,
            sample_fps,
            target_indices=list(range(start, end)),
            image_max_size=image_max_size,
        )
        t1 = time.perf_counter()
        generation = answerer.generate_text_from_frames(
            frames=frames,
            prompt=GRAPH_PROMPT,
            max_new_tokens=max_new_tokens,
        )
        return {
            "chunk_id": chunk_id,
            "status": "ok",
            "decode_sec": round(t1 - t0, 3),
            "generation_sec": generation.generation_sec,
            "elapsed_sec": round(time.perf_counter() - t0, 3),
            "chars": len(generation.raw_text or ""),
        }
    except Exception as exc:
        return {
            "chunk_id": chunk_id,
            "status": "error",
            "elapsed_sec": round(time.perf_counter() - t0, 3),
            "error": f"{type(exc).__name__}: {exc}",
        }


def parse_metric(text: str, name: str) -> float | None:
    prefix = name + "{"
    for line in text.splitlines():
        if line.startswith(prefix) or line.startswith(name + " "):
            try:
                return float(line.rsplit(" ", 1)[-1])
            except ValueError:
                return None
    return None


def poll_metrics(metrics_url: str, api_key: str | None, stop: threading.Event, samples: list[dict]) -> None:
    while not stop.is_set():
        try:
            request = urllib.request.Request(metrics_url)
            if api_key:
                request.add_header("Authorization", f"Bearer {api_key}")
            with urllib.request.urlopen(request, timeout=2.0) as response:
                text = response.read().decode("utf-8", errors="replace")
            samples.append(
                {
                    "time": time.time(),
                    "running": parse_metric(text, "vllm:num_requests_running"),
                    "waiting": parse_metric(text, "vllm:num_requests_waiting"),
                    "kv_cache": parse_metric(text, "vllm:kv_cache_usage_perc"),
                }
            )
        except Exception as exc:
            samples.append({"time": time.time(), "error": f"{type(exc).__name__}: {exc}"})
        stop.wait(0.5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("local_storage/flat_files/manifests/longvideobench/lvb_val_full_1337_v1.json"))
    parser.add_argument("--video-root", type=Path, default=Path("local_storage/flat_files/longvideobench/videos_full_val_1337_v1"))
    parser.add_argument("--api-base-url", required=True)
    parser.add_argument("--api-key-env-var", required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--requests", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--image-max-size", type=int, default=336)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--timeout-sec", type=float, default=240.0)
    parser.add_argument("--metrics-url", default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if not os.getenv(args.api_key_env_var):
        raise RuntimeError(f"Missing {args.api_key_env_var}")
    total_requests = args.requests or args.concurrency
    video_id, video_path = pick_video(args.manifest, args.video_root, total_requests, args.chunk_size, args.sample_fps)
    answerer = build_answerer(
        AnswerConfig(
            model_id=args.model_id,
            backend="api",
            image_max_size=args.image_max_size,
            max_new_tokens=args.max_new_tokens,
            api_base_url=args.api_base_url,
            api_key_env_var=args.api_key_env_var,
            api_timeout_sec=args.timeout_sec,
            api_requests_per_minute=10000,
            api_tokens_per_minute=100000000,
        )
    )

    started = time.perf_counter()
    rows: list[dict] = []
    metric_samples: list[dict] = []
    stop_metrics = threading.Event()
    metric_thread = None
    if args.metrics_url:
        metric_thread = threading.Thread(
            target=poll_metrics,
            args=(args.metrics_url, os.getenv(args.api_key_env_var), stop_metrics, metric_samples),
            daemon=True,
        )
        metric_thread.start()
    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = [
                executor.submit(
                    run_one,
                    answerer,
                    video_path,
                    chunk_id,
                    args.chunk_size,
                    args.sample_fps,
                    args.image_max_size,
                    args.max_new_tokens,
                )
                for chunk_id in range(total_requests)
            ]
            for future in as_completed(futures):
                row = future.result()
                rows.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)
    finally:
        stop_metrics.set()
        if metric_thread is not None:
            metric_thread.join(timeout=3.0)
        answerer.unload()

    wall = round(time.perf_counter() - started, 3)
    ok = [row for row in rows if row["status"] == "ok"]
    running_values = [sample.get("running") for sample in metric_samples if isinstance(sample.get("running"), (int, float))]
    waiting_values = [sample.get("waiting") for sample in metric_samples if isinstance(sample.get("waiting"), (int, float))]
    kv_values = [sample.get("kv_cache") for sample in metric_samples if isinstance(sample.get("kv_cache"), (int, float))]
    summary = {
        "video_id": video_id,
        "video_path": str(video_path),
        "concurrency": args.concurrency,
        "requests": total_requests,
        "ok": len(ok),
        "errors": len(rows) - len(ok),
        "wall_sec": wall,
        "throughput_req_per_sec": round(len(rows) / wall, 4) if wall > 0 else 0,
        "mean_elapsed_sec": round(statistics.mean(row["elapsed_sec"] for row in rows), 3) if rows else None,
        "mean_generation_sec": round(statistics.mean(row["generation_sec"] for row in ok), 3) if ok else None,
        "metrics_samples": len(metric_samples),
        "max_vllm_requests_running": max(running_values) if running_values else None,
        "max_vllm_requests_waiting": max(waiting_values) if waiting_values else None,
        "max_kv_cache_usage": max(kv_values) if kv_values else None,
        "rows": sorted(rows, key=lambda row: row["chunk_id"]),
    }
    print(json.dumps({"summary": summary}, indent=2, ensure_ascii=False))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
