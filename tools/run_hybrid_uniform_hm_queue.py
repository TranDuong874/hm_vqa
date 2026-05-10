from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
QUEUE_ROOT = REPO_ROOT / "results" / "run_queue" / "videomme_mlvu_ours_v1"
STATUS_JSON = QUEUE_ROOT / "status.json"

VIDEO_MME_HM_ROWS = (
    REPO_ROOT
    / "results"
    / "video_mme"
    / "ablations"
    / "Qwen3-VL-2B-Instruct_l3_rerank_l2_l3fixed60s_s60s_l2w5_l2s5_l3k10_keep3_l2encviclip_16f_336"
    / "rows.jsonl"
)
VIDEO_MME_PURE_ROWS = (
    REPO_ROOT / "results" / "video_mme" / "pure_vlm" / "Qwen3-VL-2B-Instruct_frames_16f_336" / "rows.jsonl"
)
VIDEO_MME_HYBRID_ROOT = (
    REPO_ROOT / "results" / "video_mme" / "hybrid" / "Qwen3-VL-2B-Instruct_uniform8_hm8_16f_336"
)

MLVU_HM_ROWS = (
    REPO_ROOT
    / "results"
    / "mlvu"
    / "ablations"
    / "Qwen3-VL-2B-Instruct_l3_rerank_l2_l3fixed60s_s60s_l2w5_l2s5_l3k10_keep3_l2encviclip_16f_336"
    / "rows.jsonl"
)
MLVU_PURE_ROWS = (
    REPO_ROOT / "results" / "mlvu" / "pure_vlm" / "Qwen3-VL-2B-Instruct_frames_16f_336" / "rows.jsonl"
)
MLVU_HYBRID_ROOT = REPO_ROOT / "results" / "mlvu" / "hybrid" / "Qwen3-VL-2B-Instruct_uniform8_hm8_16f_336"


@dataclass(slots=True)
class Stage:
    key: str
    label: str
    command: list[str]
    log_path: Path
    progress_path: Path


def _python() -> str:
    return str(REPO_ROOT / ".venv" / "bin" / "python")


def _stages() -> list[Stage]:
    common_args = [
        "--model-id",
        "Qwen/Qwen3-VL-2B-Instruct",
        "--backend",
        "local",
        "--load-in-4bit",
        "--max-new-tokens",
        "32",
        "--image-max-size",
        "336",
        "--uniform-count",
        "8",
        "--hm-count",
        "8",
        "--hm-select",
        "top_score",
    ]
    return [
        Stage(
            key="video_mme_hybrid_uniform8_hm8",
            label="VideoMME hybrid uniform8 + HM8",
            command=[
                _python(),
                "evals/common/run_hybrid_from_rows.py",
                "--hm-rows",
                str(VIDEO_MME_HM_ROWS),
                "--pure-rows",
                str(VIDEO_MME_PURE_ROWS),
                "--output-dir",
                str(VIDEO_MME_HYBRID_ROOT),
                *common_args,
            ],
            log_path=QUEUE_ROOT / "video_mme_hybrid_uniform8_hm8.log",
            progress_path=VIDEO_MME_HYBRID_ROOT / "rows.jsonl",
        ),
        Stage(
            key="mlvu_hybrid_uniform8_hm8",
            label="MLVU hybrid uniform8 + HM8",
            command=[
                _python(),
                "evals/common/run_hybrid_from_rows.py",
                "--hm-rows",
                str(MLVU_HM_ROWS),
                "--pure-rows",
                str(MLVU_PURE_ROWS),
                "--output-dir",
                str(MLVU_HYBRID_ROOT),
                *common_args,
            ],
            log_path=QUEUE_ROOT / "mlvu_hybrid_uniform8_hm8.log",
            progress_path=MLVU_HYBRID_ROOT / "rows.jsonl",
        ),
    ]


def _read_status() -> dict:
    if STATUS_JSON.exists():
        return json.loads(STATUS_JSON.read_text(encoding="utf-8"))
    return {"created_at": time.time(), "stages": {}}


def _write_status(status: dict) -> None:
    QUEUE_ROOT.mkdir(parents=True, exist_ok=True)
    STATUS_JSON.write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")


def _row_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _run_stage(stage: Stage) -> None:
    status = _read_status()
    stage_payload = status.setdefault("stages", {}).setdefault(stage.key, {})
    if stage_payload.get("status") == "completed":
        return

    started_at = time.time()
    stage_payload.update(
        {
            "status": "running",
            "started_at": started_at,
            "finished_at": None,
            "initial_completed": _row_count(stage.progress_path),
            "command": stage.command,
        }
    )
    status.update({"status": "running", "current_stage": stage.key, "error": None})
    _write_status(status)

    stage.log_path.parent.mkdir(parents=True, exist_ok=True)
    with stage.log_path.open("a", encoding="utf-8") as log:
        log.write(f"[start] {time.strftime('%Y-%m-%d %H:%M:%S')} stage={stage.key}\n")
        log.flush()
        result = subprocess.run(stage.command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
        log.write(f"[exit] {time.strftime('%Y-%m-%d %H:%M:%S')} code={result.returncode} stage={stage.key}\n")
        log.flush()

    status = _read_status()
    stage_payload = status.setdefault("stages", {}).setdefault(stage.key, {})
    stage_payload.update(
        {
            "status": "completed" if result.returncode == 0 else "failed",
            "finished_at": time.time(),
            "return_code": result.returncode,
        }
    )
    if result.returncode != 0:
        status.update({"status": "failed", "current_stage": stage.key, "error": f"{stage.key} failed"})
        _write_status(status)
        raise SystemExit(result.returncode)
    status.update({"status": "running", "current_stage": None})
    _write_status(status)


def main() -> None:
    for stage in _stages():
        _run_stage(stage)
    status = _read_status()
    status.update({"status": "completed", "current_stage": None, "finished_at": time.time(), "error": None})
    _write_status(status)


if __name__ == "__main__":
    main()
