from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from statistics import mean


ROOT = Path("results/longvideobench/api_qwen3vl8b_full1337_v1")
RUNS = [
    (
        "VLM",
        ROOT / "pure_vlm" / "Qwen3-VL-8B-Instruct_frames_16f_336",
    ),
    (
        "HM-VQA",
        ROOT
        / "ablations"
        / "Qwen3-VL-8B-Instruct_l3_rerank_l2_l3fixed60s_s60s_l2w5_l2s5_l3k10_keep3_l2encviclip_16f_336",
    ),
]
TOTAL = 1337


def _load_rows(path: Path) -> list[dict]:
    rows_path = path / "rows.jsonl"
    if not rows_path.exists():
        return []
    rows: list[dict] = []
    with rows_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def _correct(row: dict) -> bool:
    if "choice_correct" in row:
        return bool(row["choice_correct"])
    if "correct" in row:
        return bool(row["correct"])
    return False


def _last_log(path: Path) -> str:
    log_path = path / "progress.log"
    if not log_path.exists():
        return "-"
    last = "-"
    with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.strip():
                last = line.strip()
    return last


def _eta(path: Path, completed: int) -> str:
    if completed <= 0:
        return "unknown"
    log_path = path / "progress.log"
    if not log_path.exists():
        return "unknown"
    start_time: datetime | None = None
    last_time: datetime | None = None
    pattern = re.compile(r"^\[(.*?)\]")
    with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            match = pattern.match(line)
            if not match:
                continue
            try:
                current = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue
            if start_time is None:
                start_time = current
            last_time = current
    if start_time is None or last_time is None or last_time <= start_time:
        return "unknown"
    elapsed = (last_time - start_time).total_seconds()
    rate = completed / elapsed
    if rate <= 0:
        return "unknown"
    remaining = int((TOTAL - completed) / rate)
    hours, rem = divmod(max(remaining, 0), 3600)
    minutes, seconds = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    return f"{minutes}m{seconds:02d}s"


def main() -> None:
    print("run\trows\tacc\tavg_gen_sec\teta\tlast_log")
    for label, path in RUNS:
        rows = _load_rows(path)
        completed = len(rows)
        acc = sum(1 for row in rows if _correct(row)) / completed if completed else 0.0
        gen_secs = [float(row["generation_sec"]) for row in rows if isinstance(row.get("generation_sec"), (int, float))]
        avg_gen = mean(gen_secs) if gen_secs else 0.0
        print(
            f"{label}\t{completed}/{TOTAL}\t{acc:.4f}\t{avg_gen:.2f}s\t{_eta(path, completed)}\t{_last_log(path)}"
        )


if __name__ == "__main__":
    main()
