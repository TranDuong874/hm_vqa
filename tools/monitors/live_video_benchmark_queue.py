from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
QUEUE_ROOT = REPO_ROOT / "results" / "run_queue" / "videomme_mlvu_ours_v1"
STATUS_JSON = QUEUE_ROOT / "status.json"
STATUS_MD = QUEUE_ROOT / "live_status.md"
VIDEO_MME_HYBRID_ROWS = (
    REPO_ROOT
    / "results"
    / "video_mme"
    / "hybrid"
    / "Qwen3-VL-2B-Instruct_uniform8_hm8_16f_336"
    / "rows.jsonl"
)
MLVU_HYBRID_ROWS = (
    REPO_ROOT
    / "results"
    / "mlvu"
    / "hybrid"
    / "Qwen3-VL-2B-Instruct_uniform8_hm8_16f_336"
    / "rows.jsonl"
)
VIDEO_MME_L1_HM_ROWS = (
    REPO_ROOT
    / "results"
    / "video_mme"
    / "ablations"
    / "Qwen3-VL-2B-Instruct_l1_plus_l3_rerank_l2_l3fixed60s_s60s_l2w5_l2s5_l3k10_keep3_l2encviclip_16f_336"
    / "rows.jsonl"
)
MLVU_L1_HM_ROWS = (
    REPO_ROOT
    / "results"
    / "mlvu"
    / "ablations"
    / "Qwen3-VL-2B-Instruct_l1_plus_l3_rerank_l2_l3fixed60s_s60s_l2w5_l2s5_l3k10_keep3_l2encviclip_16f_336"
    / "rows.jsonl"
)

STAGES = [
    {
        "key": "video_mme_ingest",
        "label": "VideoMME 100h OpenCLIP ingestion",
        "total": 250,
        "progress_type": "meta_count",
        "progress_path": REPO_ROOT / "local_storage" / "flat_files" / "video_mme" / "openclip_1fps_100h_250_v1",
        "log_path": QUEUE_ROOT / "video_mme_ingest.log",
    },
    {
        "key": "mlvu_ingest",
        "label": "MLVU test OpenCLIP ingestion",
        "total": 349,
        "progress_type": "meta_count",
        "progress_path": REPO_ROOT / "local_storage" / "flat_files" / "mlvu" / "openclip_1fps_test_mcq_v1",
        "log_path": QUEUE_ROOT / "mlvu_ingest.log",
    },
    {
        "key": "video_mme_ours",
        "label": "VideoMME ours QA",
        "total": 750,
        "progress_type": "jsonl_rows",
        "progress_path": REPO_ROOT
        / "results"
        / "video_mme"
        / "ablations"
        / "Qwen3-VL-2B-Instruct_l3_rerank_l2_l3fixed60s_s60s_l2w5_l2s5_l3k10_keep3_l2encviclip_16f_336"
        / "rows.jsonl",
        "log_path": QUEUE_ROOT / "video_mme_ours.log",
    },
    {
        "key": "mlvu_ours",
        "label": "MLVU ours QA",
        "total": 502,
        "progress_type": "jsonl_rows",
        "progress_path": REPO_ROOT
        / "results"
        / "mlvu"
        / "ablations"
        / "Qwen3-VL-2B-Instruct_l3_rerank_l2_l3fixed60s_s60s_l2w5_l2s5_l3k10_keep3_l2encviclip_16f_336"
        / "rows.jsonl",
        "log_path": QUEUE_ROOT / "mlvu_ours.log",
    },
    {
        "key": "video_mme_pure_vlm",
        "label": "VideoMME pure uniform 16f baseline",
        "total": 750,
        "progress_type": "jsonl_rows",
        "progress_path": REPO_ROOT
        / "results"
        / "video_mme"
        / "pure_vlm"
        / "Qwen3-VL-2B-Instruct_frames_16f_336"
        / "rows.jsonl",
        "log_path": QUEUE_ROOT / "video_mme_pure_vlm.log",
    },
    {
        "key": "mlvu_pure_vlm",
        "label": "MLVU pure uniform 16f baseline",
        "total": 502,
        "progress_type": "jsonl_rows",
        "progress_path": REPO_ROOT
        / "results"
        / "mlvu"
        / "pure_vlm"
        / "Qwen3-VL-2B-Instruct_frames_16f_336"
        / "rows.jsonl",
        "log_path": QUEUE_ROOT / "mlvu_pure_vlm.log",
    },
    {
        "key": "video_mme_hybrid_uniform8_hm8",
        "label": "VideoMME hybrid uniform8 + HM8",
        "total": 750,
        "progress_type": "jsonl_rows",
        "progress_path": VIDEO_MME_HYBRID_ROWS,
        "log_path": QUEUE_ROOT / "video_mme_hybrid_uniform8_hm8.log",
    },
    {
        "key": "mlvu_hybrid_uniform8_hm8",
        "label": "MLVU hybrid uniform8 + HM8",
        "total": 502,
        "progress_type": "jsonl_rows",
        "progress_path": MLVU_HYBRID_ROWS,
        "log_path": QUEUE_ROOT / "mlvu_hybrid_uniform8_hm8.log",
    },
    {
        "key": "video_mme_l1_8_hm8",
        "label": "VideoMME hybrid L1 8 + HM8",
        "total": 750,
        "progress_type": "jsonl_rows",
        "progress_path": VIDEO_MME_L1_HM_ROWS,
        "log_path": QUEUE_ROOT / "video_mme_l1_8_hm8.log",
    },
    {
        "key": "mlvu_l1_8_hm8",
        "label": "MLVU hybrid L1 8 + HM8",
        "total": 502,
        "progress_type": "jsonl_rows",
        "progress_path": MLVU_L1_HM_ROWS,
        "log_path": QUEUE_ROOT / "mlvu_l1_8_hm8.log",
    },
]


def _load_status() -> dict:
    if STATUS_JSON.exists():
        return json.loads(STATUS_JSON.read_text(encoding="utf-8"))
    return {"status": "not_started", "stages": {}}


def _completed(stage: dict) -> int:
    path = Path(stage["progress_path"])
    if stage["progress_type"] == "meta_count":
        if not path.exists():
            return 0
        return sum(1 for _ in path.glob("*/meta.json"))
    if stage["progress_type"] == "jsonl_rows":
        if not path.exists():
            return 0
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    return 0


def _duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "unknown"
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, sec = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {sec}s"
    return f"{sec}s"


def _completed_elapsed_from_log(log_path: Path) -> float | None:
    """Return elapsed time for the first completed execution in an append log.

    The queue is often restarted with resume enabled. For ingestion stages this
    appends later all-skip runs to the same log, so status.json may report only
    the final skip pass. The first start-to-exit pair is the real initial stage
    execution that produced the cache.
    """
    if not log_path.exists():
        return None
    start_pattern = re.compile(r"^\[start\] (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
    exit_pattern = re.compile(r"^\[exit\] (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) code=0")
    current_start: float | None = None
    first_start: float | None = None
    last_success_exit: float | None = None
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if match := start_pattern.match(line):
            parsed = time.mktime(time.strptime(match.group(1), "%Y-%m-%d %H:%M:%S"))
            if first_start is None:
                first_start = parsed
            if current_start is None:
                current_start = parsed
            continue
        if match := exit_pattern.match(line):
            parsed = time.mktime(time.strptime(match.group(1), "%Y-%m-%d %H:%M:%S"))
            last_success_exit = parsed
            if current_start is not None:
                return max(parsed - current_start, 0.0)
    if first_start is not None and last_success_exit is not None:
        return max(last_success_exit - first_start, 0.0)
    return None


def _rate_eta(stage: dict, stage_status: dict, completed: int) -> tuple[str, str, str]:
    total = int(stage["total"])
    started_at = stage_status.get("started_at")
    finished_at = stage_status.get("finished_at")
    if stage_status.get("status") == "completed":
        elapsed = _completed_elapsed_from_log(Path(stage["log_path"]))
        if started_at and finished_at:
            status_elapsed = float(finished_at) - float(started_at)
            if elapsed is None or status_elapsed > elapsed:
                elapsed = status_elapsed
        return "done", "0s", _duration(elapsed)
    if not started_at or completed <= 0:
        return "unknown", "unknown", "unknown"
    elapsed = max(time.time() - float(started_at), 1.0)
    initial_completed = int(stage_status.get("initial_completed") or 0)
    delta_completed = max(completed - initial_completed, 0)
    if delta_completed <= 0:
        return "unknown", "unknown", _duration(elapsed)
    rate_per_min = delta_completed / elapsed * 60.0
    remaining = max(total - completed, 0)
    eta = remaining / (delta_completed / elapsed)
    return f"{rate_per_min:.2f}/min", _duration(eta), _duration(elapsed)


def write_status() -> None:
    status = _load_status()
    stage_statuses = status.get("stages", {})
    lines = [
        "# Video Benchmark Queue Status",
        "",
        f"Updated: `{time.strftime('%Y-%m-%d %H:%M:%S')}`",
        "",
        f"Queue status: `{status.get('status', 'unknown')}`",
        f"Current stage: `{status.get('current_stage') or 'none'}`",
        "",
        "| Stage | Status | Progress | Rate | ETA | Elapsed | Log |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for stage in STAGES:
        stage_status = stage_statuses.get(stage["key"], {})
        completed = _completed(stage)
        if stage_status.get("status") == "completed":
            completed = int(stage["total"])
        rate, eta, elapsed = _rate_eta(stage, stage_status, completed)
        log_rel = Path(stage["log_path"]).relative_to(REPO_ROOT)
        lines.append(
            f"| {stage['label']} | `{stage_status.get('status', 'pending')}` | "
            f"{completed}/{stage['total']} | {rate} | {eta} | {elapsed} | `{log_rel}` |"
        )
    if status.get("error"):
        lines.extend(["", "## Error", "", f"```text\n{status['error']}\n```"])
    QUEUE_ROOT.mkdir(parents=True, exist_ok=True)
    STATUS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Write one Markdown status file for the VideoMME/MLVU queue.")
    parser.add_argument("--watch", action="store_true", help="Refresh the Markdown status every interval seconds.")
    parser.add_argument("--interval", type=float, default=30.0)
    args = parser.parse_args()
    while True:
        write_status()
        if not args.watch:
            break
        time.sleep(max(float(args.interval), 1.0))


if __name__ == "__main__":
    main()
