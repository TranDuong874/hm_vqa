from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
QUEUE_ROOT = REPO_ROOT / "results" / "run_queue" / "videomme_mlvu_ours_v1"
STATUS_JSON = QUEUE_ROOT / "status.json"
STATUS_MD = QUEUE_ROOT / "live_status.md"
RETRIEVAL_CONFIG = REPO_ROOT / "configs" / "retrieval" / "fixed60_l2_viclip_keep3_16f.json"
INGEST_BATCH_SIZE = int(os.environ.get("HM_VQA_INGEST_BATCH_SIZE", "128"))

VIDEO_MME_FEATURE_ROOT = REPO_ROOT / "local_storage" / "flat_files" / "video_mme" / "openclip_1fps_100h_250_v1"
MLVU_FEATURE_ROOT = REPO_ROOT / "local_storage" / "flat_files" / "mlvu" / "openclip_1fps_test_mcq_v1"

VIDEO_MME_ROWS = (
    REPO_ROOT
    / "results"
    / "video_mme"
    / "ablations"
    / "Qwen3-VL-2B-Instruct_l3_rerank_l2_l3fixed60s_s60s_l2w5_l2s5_l3k10_keep3_l2encviclip_16f_336"
    / "rows.jsonl"
)
MLVU_ROWS = (
    REPO_ROOT
    / "results"
    / "mlvu"
    / "ablations"
    / "Qwen3-VL-2B-Instruct_l3_rerank_l2_l3fixed60s_s60s_l2w5_l2s5_l3k10_keep3_l2encviclip_16f_336"
    / "rows.jsonl"
)
VIDEO_MME_BASELINE_ROWS = (
    REPO_ROOT
    / "results"
    / "video_mme"
    / "pure_vlm"
    / "Qwen3-VL-2B-Instruct_frames_16f_336"
    / "rows.jsonl"
)
MLVU_BASELINE_ROWS = (
    REPO_ROOT
    / "results"
    / "mlvu"
    / "pure_vlm"
    / "Qwen3-VL-2B-Instruct_frames_16f_336"
    / "rows.jsonl"
)


@dataclass(slots=True)
class Stage:
    key: str
    label: str
    command: list[str]
    log_path: Path
    total: int
    progress_type: str
    progress_path: Path


def _python() -> str:
    return str(REPO_ROOT / ".venv" / "bin" / "python")


def _stages() -> list[Stage]:
    common_answer_args = [
        "--retrieval-config",
        str(RETRIEVAL_CONFIG),
        "--model-id",
        "Qwen/Qwen3-VL-2B-Instruct",
        "--backend",
        "local",
        "--load-in-4bit",
        "--max-new-tokens",
        "32",
        "--image-max-size",
        "336",
    ]
    baseline_answer_args = [
        "--model-id",
        "Qwen/Qwen3-VL-2B-Instruct",
        "--backend",
        "local",
        "--load-in-4bit",
        "--max-new-tokens",
        "32",
        "--max-frames",
        "16",
        "--image-max-size",
        "336",
    ]
    return [
        Stage(
            key="video_mme_ingest",
            label="VideoMME 100h OpenCLIP ingestion",
            command=[
                _python(),
                "evals/video_mme/ingestion/run_ingest.py",
                "--batch-size",
                str(INGEST_BATCH_SIZE),
                "--device",
                "cuda",
            ],
            log_path=QUEUE_ROOT / "video_mme_ingest.log",
            total=250,
            progress_type="meta_count",
            progress_path=VIDEO_MME_FEATURE_ROOT,
        ),
        Stage(
            key="mlvu_ingest",
            label="MLVU test OpenCLIP ingestion",
            command=[
                _python(),
                "evals/mlvu/ingestion/run_ingest.py",
                "--batch-size",
                str(INGEST_BATCH_SIZE),
                "--device",
                "cuda",
            ],
            log_path=QUEUE_ROOT / "mlvu_ingest.log",
            total=349,
            progress_type="meta_count",
            progress_path=MLVU_FEATURE_ROOT,
        ),
        Stage(
            key="video_mme_ours",
            label="VideoMME ours QA",
            command=[_python(), "evals/video_mme/inference/run_retrieval_qa.py", *common_answer_args],
            log_path=QUEUE_ROOT / "video_mme_ours.log",
            total=750,
            progress_type="jsonl_rows",
            progress_path=VIDEO_MME_ROWS,
        ),
        Stage(
            key="mlvu_ours",
            label="MLVU ours QA",
            command=[_python(), "evals/mlvu/inference/run_retrieval_qa.py", *common_answer_args],
            log_path=QUEUE_ROOT / "mlvu_ours.log",
            total=502,
            progress_type="jsonl_rows",
            progress_path=MLVU_ROWS,
        ),
        Stage(
            key="video_mme_pure_vlm",
            label="VideoMME pure uniform 16f baseline",
            command=[_python(), "evals/video_mme/inference/run_pure_vlm.py", *baseline_answer_args],
            log_path=QUEUE_ROOT / "video_mme_pure_vlm.log",
            total=750,
            progress_type="jsonl_rows",
            progress_path=VIDEO_MME_BASELINE_ROWS,
        ),
        Stage(
            key="mlvu_pure_vlm",
            label="MLVU pure uniform 16f baseline",
            command=[_python(), "evals/mlvu/inference/run_pure_vlm.py", *baseline_answer_args],
            log_path=QUEUE_ROOT / "mlvu_pure_vlm.log",
            total=502,
            progress_type="jsonl_rows",
            progress_path=MLVU_BASELINE_ROWS,
        ),
    ]


def _now() -> float:
    return time.time()


def _read_status() -> dict:
    if STATUS_JSON.exists():
        return json.loads(STATUS_JSON.read_text(encoding="utf-8"))
    return {}


def _write_status(payload: dict) -> None:
    QUEUE_ROOT.mkdir(parents=True, exist_ok=True)
    STATUS_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_status_md(payload)


def _format_duration(seconds: float | None) -> str:
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


def _completed_for_stage(stage: Stage) -> int:
    if stage.progress_type == "meta_count":
        if not stage.progress_path.exists():
            return 0
        return sum(1 for _ in stage.progress_path.glob("*/meta.json"))
    if stage.progress_type == "jsonl_rows":
        if not stage.progress_path.exists():
            return 0
        with stage.progress_path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    return 0


def _stage_eta(stage: Stage, stage_payload: dict) -> tuple[int, str, str]:
    completed = _completed_for_stage(stage)
    total = max(int(stage.total), 1)
    started_at = stage_payload.get("started_at")
    initial_completed = int(stage_payload.get("initial_completed") or 0)
    delta_completed = max(completed - initial_completed, 0)
    if not started_at or delta_completed <= 0:
        return completed, "unknown", "unknown"
    elapsed = max(_now() - float(started_at), 1.0)
    rate = delta_completed / elapsed
    remaining = max(total - completed, 0)
    eta = remaining / rate if rate > 0 else None
    return completed, f"{rate * 60:.2f}/min", _format_duration(eta)


def _write_status_md(payload: dict) -> None:
    stages = _stages()
    current_key = payload.get("current_stage")
    lines = [
        "# Video Benchmark Queue Status",
        "",
        f"Updated: `{time.strftime('%Y-%m-%d %H:%M:%S')}`",
        "",
        f"Queue status: `{payload.get('status', 'unknown')}`",
        f"Current stage: `{current_key or 'none'}`",
        "",
        "| Stage | Status | Progress | Rate | ETA | Log |",
        "|---|---:|---:|---:|---:|---|",
    ]
    stage_payloads = payload.get("stages", {})
    for stage in stages:
        stage_payload = stage_payloads.get(stage.key, {})
        completed, rate, eta = _stage_eta(stage, stage_payload)
        status = stage_payload.get("status", "pending")
        if status == "completed":
            completed = stage.total
            eta = "0s"
        log_rel = stage.log_path.relative_to(REPO_ROOT)
        lines.append(
            f"| {stage.label} | `{status}` | {completed}/{stage.total} | {rate} | {eta} | `{log_rel}` |"
        )
    if payload.get("error"):
        lines.extend(["", "## Error", "", f"```text\n{payload['error']}\n```"])
    STATUS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_stage(stage: Stage, payload: dict) -> None:
    stage_payloads = payload.setdefault("stages", {})
    stage_payload = stage_payloads.setdefault(stage.key, {})
    if stage_payload.get("status") == "completed":
        return

    stage.log_path.parent.mkdir(parents=True, exist_ok=True)
    stage_payload.update(
        {
            "status": "running",
            "started_at": _now(),
            "initial_completed": _completed_for_stage(stage),
            "command": stage.command,
        }
    )
    payload.update({"status": "running", "current_stage": stage.key, "error": None})
    _write_status(payload)

    env = os.environ.copy()
    env["PYTHONPATH"] = ".:src"
    with stage.log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[start] {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(stage.command)}\n")
        log.flush()
        process = subprocess.Popen(
            stage.command,
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            sys.stdout.write(line)
            sys.stdout.flush()
        return_code = process.wait()
        log.write(f"[exit] {time.strftime('%Y-%m-%d %H:%M:%S')} code={return_code}\n")
        log.flush()

    if return_code != 0:
        stage_payload.update({"status": "failed", "finished_at": _now(), "return_code": return_code})
        payload.update({"status": "failed", "current_stage": stage.key, "error": f"{stage.label} failed with exit code {return_code}"})
        _write_status(payload)
        raise SystemExit(return_code)

    stage_payload.update({"status": "completed", "finished_at": _now(), "return_code": return_code})
    _write_status(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VideoMME and MLVU ingestion plus HM-VQA QA runs in order.")
    parser.add_argument("--reset-status", action="store_true", help="Forget previous queue status, but keep resumable cache/result files.")
    args = parser.parse_args()

    QUEUE_ROOT.mkdir(parents=True, exist_ok=True)
    payload = {} if args.reset_status else _read_status()
    payload.setdefault("created_at", _now())
    payload.setdefault("stages", {})
    payload["status"] = "running"
    _write_status(payload)

    for stage in _stages():
        _run_stage(stage, payload)

    payload.update({"status": "completed", "current_stage": None, "finished_at": _now(), "error": None})
    _write_status(payload)


if __name__ == "__main__":
    main()
