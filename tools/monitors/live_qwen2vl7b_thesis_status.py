from __future__ import annotations

import json
import subprocess
from pathlib import Path


RUNS = [
    (
        "LongVideoBench",
        "VLM",
        1337,
        Path("results/longvideobench/api_qwen2vl7b_full1337_v1/pure_vlm/Qwen2-VL-7B-Instruct_frames_16f_336/rows.jsonl"),
    ),
    (
        "LongVideoBench",
        "HM-VQA",
        1337,
        Path("results/longvideobench/api_qwen2vl7b_full1337_v1/ablations/Qwen2-VL-7B-Instruct_l3_rerank_l2_l3fixed60s_s60s_l2w5_l2s5_l3k10_keep3_l2encviclip_16f_336/rows.jsonl"),
    ),
    (
        "VideoMME",
        "VLM",
        315,
        Path("results/video_mme/near100h_balanced315_qwen2vl7b_api_v1/pure_vlm/Qwen2-VL-7B-Instruct_frames_16f_336/rows.jsonl"),
    ),
    (
        "VideoMME",
        "HM-VQA",
        315,
        Path("results/video_mme/near100h_balanced315_qwen2vl7b_api_v1/ablations/Qwen2-VL-7B-Instruct_l3_rerank_l2_l3fixed60s_s60s_l2w5_l2s5_l3k10_keep3_l2encviclip_16f_336/rows.jsonl"),
    ),
    (
        "MLVU",
        "VLM",
        502,
        Path("results/mlvu/full_test_mcq_qwen2vl7b_api_v1/pure_vlm/Qwen2-VL-7B-Instruct_frames_16f_336/rows.jsonl"),
    ),
    (
        "MLVU",
        "HM-VQA",
        502,
        Path("results/mlvu/full_test_mcq_qwen2vl7b_api_v1/fixed_retrieval_diagnostics/Qwen2-VL-7B-Instruct_l3_rerank_l2_l3fixed60s_s60s_l2w5_l2s5_l3k10_keep3_l2encviclip_l2qfull_evitop_l2_evitextchunks_16f_336/rows.jsonl"),
    ),
]


def summarize(path: Path) -> tuple[int, int, int, float | None]:
    if not path.exists():
        return 0, 0, 0, None
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    scored = [row for row in rows if row.get("choice_correct") is not None]
    correct = sum(1 for row in scored if row.get("choice_correct") is True)
    acc = correct / len(scored) if scored else None
    return len(rows), len(scored), correct, acc


def running_processes() -> str:
    result = subprocess.run(
        ["pgrep", "-af", "reanswer_saved_evidence_rows.py|evals/.*/inference/run_.*\\.py"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.stdout.strip()


def main() -> None:
    print("| Dataset | Method | Rows | Scored | Correct | Accuracy |")
    print("|---|---:|---:|---:|---:|---:|")
    by_dataset: dict[str, dict[str, float | None]] = {}
    for dataset, method, total, path in RUNS:
        rows, scored, correct, acc = summarize(path)
        by_dataset.setdefault(dataset, {})[method] = acc
        acc_text = "--" if acc is None else f"{acc:.4f}"
        print(f"| {dataset} | {method} | {rows}/{total} | {scored} | {correct} | {acc_text} |")

    print("\nLaTeX-ready Qwen2-VL-7B rows:")
    for dataset in ["LongVideoBench", "VideoMME", "MLVU"]:
        vlm = by_dataset.get(dataset, {}).get("VLM")
        hm = by_dataset.get(dataset, {}).get("HM-VQA")
        if vlm is None or hm is None:
            print(f"{dataset} & Qwen2-VL-7B & -- & -- & -- \\\\")
            continue
        print(f"{dataset} & Qwen2-VL-7B & {vlm:.4f} & \\textbf{{{hm:.4f}}} & {hm - vlm:+.4f} \\\\")

    procs = running_processes()
    print("\nRunning eval processes:")
    print(procs if procs else "None")


if __name__ == "__main__":
    main()
