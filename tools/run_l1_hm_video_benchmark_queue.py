from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
QUEUE_ROOT = REPO_ROOT / "results" / "run_queue" / "videomme_mlvu_ours_v1"
STATUS_JSON = QUEUE_ROOT / "status.json"

MLVU_ROWS = (
    REPO_ROOT
    / "results/mlvu/ablations"
    / "Qwen3-VL-2B-Instruct_l1_plus_l3_rerank_l2_l3fixed60s_s60s_l2w5_l2s5_l3k10_keep3_l2encviclip_16f_336"
    / "rows.jsonl"
)
VIDEO_MME_ROWS = (
    REPO_ROOT
    / "results/video_mme/ablations"
    / "Qwen3-VL-2B-Instruct_l1_plus_l3_rerank_l2_l3fixed60s_s60s_l2w5_l2s5_l3k10_keep3_l2encviclip_16f_336"
    / "rows.jsonl"
)


def _read_status() -> dict:
    if STATUS_JSON.exists():
        return json.loads(STATUS_JSON.read_text(encoding="utf-8"))
    return {"created_at": time.time(), "status": "running", "current_stage": None, "stages": {}, "error": None}


def _write_status(status: dict) -> None:
    QUEUE_ROOT.mkdir(parents=True, exist_ok=True)
    STATUS_JSON.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _row_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _set_stage(state: dict, key: str, **updates: object) -> None:
    stages = state.setdefault("stages", {})
    stage = stages.setdefault(key, {})
    stage.update(updates)
    _write_status(state)


def _mlvu_process_running() -> bool:
    proc = subprocess.run(
        ["pgrep", "-f", "evals/mlvu/inference/run_retrieval_qa.py.*l1_plus_l3_rerank_l2"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return proc.returncode == 0


def _monitor_existing_mlvu(status: dict) -> None:
    key = "mlvu_l1_8_hm8"
    started = time.time()
    initial_completed = _row_count(MLVU_ROWS)
    status["status"] = "running"
    status["current_stage"] = key
    _set_stage(
        status,
        key,
        status="running",
        started_at=started,
        initial_completed=initial_completed,
        command=["external tmux session", "mlvu_l1_8_hm_8"],
    )
    while True:
        completed = _row_count(MLVU_ROWS)
        if completed >= 502:
            _set_stage(status, key, status="completed", finished_at=time.time(), return_code=0)
            return
        if not _mlvu_process_running():
            _set_stage(status, key, status="failed", finished_at=time.time(), return_code=1)
            raise RuntimeError(f"MLVU L1+HM stopped early at {completed}/502")
        time.sleep(10)


def _run_video_mme(status: dict) -> None:
    key = "video_mme_l1_8_hm8"
    if _row_count(VIDEO_MME_ROWS) >= 750:
        _set_stage(status, key, status="completed", started_at=time.time(), finished_at=time.time(), return_code=0)
        return

    command = [
        str(REPO_ROOT / ".venv/bin/python"),
        "evals/video_mme/inference/run_retrieval_qa.py",
        "--manifest",
        str(REPO_ROOT / "local_storage/flat_files/manifests/video_mme/video_mme_100h_250videos_no_subs_v1.json"),
        "--video-root",
        "/home/tranduong/dev/dataset/Video-MME/videos_subset_100h_250_v1",
        "--feature-root",
        str(REPO_ROOT / "local_storage/flat_files/video_mme/openclip_1fps_100h_250_v1"),
        "--derived-cache-root",
        str(REPO_ROOT / "local_storage/flat_files/video_mme/derived_100h_250_v1"),
        "--output-root",
        str(REPO_ROOT / "results/video_mme/ablations"),
        "--method",
        "l1_plus_l3_rerank_l2",
        "--l3-segmentation",
        "fixed",
        "--l3-window-seconds",
        "60",
        "--l3-stride-seconds",
        "60",
        "--top-l3-segments",
        "10",
        "--l3-rerank-keep",
        "3",
        "--l2-segmentation",
        "fixed",
        "--l2-window-seconds",
        "5",
        "--l2-stride-seconds",
        "5",
        "--l2-rerank-encoder",
        "viclip",
        "--l3-rerank-evidence-source",
        "reranked_l3",
        "--max-frames",
        "16",
        "--image-max-size",
        "336",
        "--evidence-text-mode",
        "frames",
        "--model-id",
        "Qwen/Qwen3-VL-2B-Instruct",
        "--backend",
        "local",
        "--load-in-4bit",
        "--max-new-tokens",
        "32",
    ]
    log_path = QUEUE_ROOT / "video_mme_l1_8_hm8.log"
    status["status"] = "running"
    status["current_stage"] = key
    _set_stage(
        status,
        key,
        status="running",
        started_at=time.time(),
        initial_completed=_row_count(VIDEO_MME_ROWS),
        command=command,
    )
    env = {"PYTHONPATH": "src:.", **dict(__import__("os").environ)}
    with log_path.open("ab") as log:
        log.write(f"[start] {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(command)}\n".encode())
        proc = subprocess.Popen(command, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        return_code = proc.wait()
        log.write(f"[exit] {time.strftime('%Y-%m-%d %H:%M:%S')} code={return_code}\n".encode())
    _set_stage(
        status,
        key,
        status="completed" if return_code == 0 else "failed",
        finished_at=time.time(),
        return_code=return_code,
    )
    if return_code != 0:
        raise RuntimeError(f"VideoMME L1+HM exited with code {return_code}")


def main() -> None:
    status = _read_status()
    status["error"] = None
    status["status"] = "running"
    try:
        _monitor_existing_mlvu(status)
        _run_video_mme(status)
        status["status"] = "completed"
        status["current_stage"] = None
        status["error"] = None
        status["finished_at"] = time.time()
        _write_status(status)
    except Exception as exc:
        status["status"] = "failed"
        status["error"] = f"{type(exc).__name__}: {exc}"
        status["finished_at"] = time.time()
        _write_status(status)
        raise


if __name__ == "__main__":
    main()
