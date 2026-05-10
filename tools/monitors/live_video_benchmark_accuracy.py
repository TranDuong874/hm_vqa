from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT = REPO_ROOT / "results" / "run_queue" / "videomme_mlvu_ours_v1" / "live_accuracy.md"

RUNS = [
    {
        "dataset": "VideoMME 100h",
        "method": "HM-VQA fixed60 + L2 ViCLIP",
        "root": REPO_ROOT
        / "results"
        / "video_mme"
        / "ablations"
        / "Qwen3-VL-2B-Instruct_l3_rerank_l2_l3fixed60s_s60s_l2w5_l2s5_l3k10_keep3_l2encviclip_16f_336",
        "total": 750,
    },
    {
        "dataset": "VideoMME 100h",
        "method": "Pure VLM uniform 16f",
        "root": REPO_ROOT
        / "results"
        / "video_mme"
        / "pure_vlm"
        / "Qwen3-VL-2B-Instruct_frames_16f_336",
        "total": 750,
    },
    {
        "dataset": "MLVU test",
        "method": "HM-VQA fixed60 + L2 ViCLIP",
        "root": REPO_ROOT
        / "results"
        / "mlvu"
        / "ablations"
        / "Qwen3-VL-2B-Instruct_l3_rerank_l2_l3fixed60s_s60s_l2w5_l2s5_l3k10_keep3_l2encviclip_16f_336",
        "total": 502,
    },
    {
        "dataset": "MLVU test",
        "method": "Pure VLM uniform 16f",
        "root": REPO_ROOT
        / "results"
        / "mlvu"
        / "pure_vlm"
        / "Qwen3-VL-2B-Instruct_frames_16f_336",
        "total": 502,
    },
    {
        "dataset": "VideoMME 100h",
        "method": "Hybrid uniform8 + HM8",
        "root": REPO_ROOT
        / "results"
        / "video_mme"
        / "hybrid"
        / "Qwen3-VL-2B-Instruct_uniform8_hm8_16f_336",
        "total": 750,
    },
    {
        "dataset": "MLVU test",
        "method": "Hybrid uniform8 + HM8",
        "root": REPO_ROOT
        / "results"
        / "mlvu"
        / "hybrid"
        / "Qwen3-VL-2B-Instruct_uniform8_hm8_16f_336",
        "total": 502,
    },
    {
        "dataset": "VideoMME 100h",
        "method": "Hybrid L1 8 + HM8",
        "root": REPO_ROOT
        / "results"
        / "video_mme"
        / "ablations"
        / "Qwen3-VL-2B-Instruct_l1_plus_l3_rerank_l2_l3fixed60s_s60s_l2w5_l2s5_l3k10_keep3_l2encviclip_16f_336",
        "total": 750,
    },
    {
        "dataset": "MLVU test",
        "method": "Hybrid L1 8 + HM8",
        "root": REPO_ROOT
        / "results"
        / "mlvu"
        / "ablations"
        / "Qwen3-VL-2B-Instruct_l1_plus_l3_rerank_l2_l3fixed60s_s60s_l2w5_l2s5_l3k10_keep3_l2encviclip_16f_336",
        "total": 502,
    },
]


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _row_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _fmt_float(value: object, digits: int = 4) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "-"


def write_accuracy() -> None:
    lines = [
        "# Video Benchmark Accuracy",
        "",
        f"Updated: `{time.strftime('%Y-%m-%d %H:%M:%S')}`",
        "",
        "| Dataset | Method | Completed | Answered | Scored | Accuracy | Avg Gen Sec |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for run in RUNS:
        root = Path(run["root"])
        summary = _read_json(root / "rolling_summary.json") or _read_json(root / "final_summary.json")
        completed = int(summary.get("completed") or _row_count(root / "rows.jsonl"))
        total = int(summary.get("total") or run["total"])
        lines.append(
            "| {dataset} | {method} | {completed}/{total} | {answered} | {scored} | {acc} | {gen} |".format(
                dataset=run["dataset"],
                method=run["method"],
                completed=completed,
                total=total,
                answered=summary.get("answered", "-"),
                scored=summary.get("scored", "-"),
                acc=_fmt_float(summary.get("choice_accuracy")),
                gen=_fmt_float(summary.get("avg_generation_sec"), digits=3),
            )
        )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Write live VideoMME/MLVU accuracy table.")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=30.0)
    args = parser.parse_args()
    while True:
        write_accuracy()
        if not args.watch:
            break
        time.sleep(max(float(args.interval), 1.0))


if __name__ == "__main__":
    main()
