from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path


ROOT = Path("results/longvideobench/api_qwen2vl7b_full1337_v1")
HYBRID_ROOT = Path("results/longvideobench/api_qwen2vl7b_hybrid_l1_hm_v1")
PURE_DIR = ROOT / "pure_vlm/Qwen2-VL-7B-Instruct_frames_16f_336"
L1_DIR = ROOT / "ablations/Qwen2-VL-7B-Instruct_l1_16f_336"
L3_TOP3_DIR = ROOT / "fixed_l3_top3/Qwen2-VL-7B-Instruct_l3_l3fixed60s_s60s_16f_336"
HM_DIR = (
    ROOT
    / "ablations/Qwen2-VL-7B-Instruct_l3_rerank_l2_l3fixed60s_s60s_l2w5_l2s5_l3k10_keep3_l2encviclip_16f_336"
)
OUT = ROOT / "live_matrix.md"


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_rows(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    rows_path = path / "rows.jsonl"
    if not rows_path.exists():
        return rows
    for line in rows_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[str(row["example_id"])] = row
    return rows


def parse_done_times(path: Path) -> list[datetime]:
    if not path.exists():
        return []
    times: list[datetime] = []
    pattern = re.compile(r"^\[(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] \[item_(?:done|blocked)\]")
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.match(line)
        if match:
            times.append(datetime.strptime(match.group("ts"), "%Y-%m-%d %H:%M:%S"))
    return times


def fmt_eta(seconds: float | None) -> str:
    if seconds is None:
        return "--"
    seconds = max(float(seconds), 0.0)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def rate_and_eta(run_dir: Path, completed: int, total: int) -> tuple[str, str]:
    times = parse_done_times(run_dir / "progress.log")
    if len(times) < 2:
        return "--", "--"
    recent = times[-50:]
    if len(recent) < 2:
        return "--", "--"
    elapsed = (recent[-1] - recent[0]).total_seconds()
    if elapsed <= 0:
        return "--", "--"
    rate = (len(recent) - 1) / elapsed
    remaining = max(total - completed, 0)
    return f"{rate:.3f} q/s", fmt_eta(remaining / rate if rate > 0 else None)


def acc(rows: list[dict]) -> float:
    scored = [row for row in rows if row.get("choice_correct") is not None]
    if not scored:
        return 0.0
    return sum(1 for row in scored if row.get("choice_correct") is True) / len(scored)


def wlt(candidate: dict[str, dict], baseline: dict[str, dict], ids: list[str]) -> tuple[int, int, int]:
    wins = losses = ties = 0
    for example_id in ids:
        cand = candidate[example_id].get("choice_correct")
        base = baseline[example_id].get("choice_correct")
        if cand == base:
            ties += 1
        elif cand is True and base is not True:
            wins += 1
        else:
            losses += 1
    return wins, losses, ties


def duration_bucket(row: dict) -> str:
    group = row.get("duration_group")
    try:
        group_int = int(group)
    except (TypeError, ValueError):
        return "unknown"
    if group_int <= 15:
        return "15"
    if group_int <= 60:
        return "60"
    if group_int <= 600:
        return "600"
    return "3600"


def main() -> None:
    runs = [
        ("Pure VLM uniform 16f", PURE_DIR),
        ("L1 direct 16f", L1_DIR),
        ("Fixed L3 60s top3 16f", L3_TOP3_DIR),
        ("HM fixed60 + fixed L2 5s ViCLIP keep3 16f", HM_DIR),
        ("Hybrid L1 8 + HM 8", HYBRID_ROOT / "l1_8_hm_8"),
        ("Hybrid RRF union top16", HYBRID_ROOT / "rrf_union_top16"),
        ("Hybrid L1-projected L3 + L2 rerank", HYBRID_ROOT / "l1_l3_union_l2rerank_l3top3"),
        ("Hybrid L1/HM L2 union + ViCLIP", HYBRID_ROOT / "l1_hm_l2_union_viclip_top_l2"),
    ]
    summaries = {label: load_json(path / "rolling_summary.json") for label, path in runs}
    rows_by_label = {label: load_rows(path) for label, path in runs}
    pure_summary = summaries["Pure VLM uniform 16f"]
    pure_rows = rows_by_label["Pure VLM uniform 16f"]
    total = int(pure_summary.get("total") or 1337)

    lines = [
        "# Live LVB Qwen2-VL-7B API Matrix",
        "",
        f"Updated: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`",
        "",
        f"Pure run: `{PURE_DIR}`",
        f"L1 run: `{L1_DIR}`",
        f"L3 top3 run: `{L3_TOP3_DIR}`",
        f"HM run: `{HM_DIR}`",
        f"Hybrid root: `{HYBRID_ROOT}`",
        "",
        "| Method | Completed | Accuracy | Matched vs Pure | Delta vs Pure | W/L/T vs Pure | Recent throughput | ETA |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for label, run_dir in runs:
        summary = summaries[label]
        run_rows = rows_by_label[label]
        completed = int(summary.get("completed") or len(run_rows))
        matched_ids = sorted(set(pure_rows) & set(run_rows))
        if label == "Pure VLM uniform 16f":
            matched = len(pure_rows)
            delta = 0.0
            wins, losses, ties = 0, 0, len(pure_rows)
        else:
            matched = len(matched_ids)
            run_acc = acc([run_rows[i] for i in matched_ids]) if matched_ids else 0.0
            pure_acc = acc([pure_rows[i] for i in matched_ids]) if matched_ids else 0.0
            delta = run_acc - pure_acc
            wins, losses, ties = wlt(run_rows, pure_rows, matched_ids) if matched_ids else (0, 0, 0)
        rate, eta = rate_and_eta(run_dir, completed, total)
        lines.append(
            f"| {label} | {completed}/{total} | {float(summary.get('choice_accuracy') or 0.0):.4f} | "
            f"{matched} | {delta:+.4f} | {wins}/{losses}/{ties} | {rate} | {eta} |"
        )

    hm_rows = rows_by_label["HM fixed60 + fixed L2 5s ViCLIP keep3 16f"]
    ids = sorted(set(pure_rows) & set(hm_rows))
    lines.extend(
        [
            "",
            "## HM Matched Accuracy By Duration Group",
            "",
            "| Group | Matched | Pure | HM | Delta | W/L/T |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )

    for group in ("15", "60", "600", "3600", "unknown"):
        group_ids = [example_id for example_id in ids if duration_bucket(pure_rows[example_id]) == group]
        if not group_ids:
            continue
        group_pure = acc([pure_rows[i] for i in group_ids])
        group_hm = acc([hm_rows[i] for i in group_ids])
        group_wlt = wlt(hm_rows, pure_rows, group_ids)
        lines.append(
            f"| {group} | {len(group_ids)} | {group_pure:.4f} | {group_hm:.4f} | {group_hm - group_pure:+.4f} | "
            f"{group_wlt[0]}/{group_wlt[1]}/{group_wlt[2]} |"
        )

    lines.extend(
        [
            "",
            "## Per-Method Matched Accuracy By Duration Group",
            "",
            "| Method | 15 | 60 | 600 | 3600 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for label, _ in runs:
        run_rows = rows_by_label[label]
        cells: list[str] = []
        for group in ("15", "60", "600", "3600"):
            group_ids = [
                example_id
                for example_id in sorted(set(pure_rows) & set(run_rows))
                if duration_bucket(pure_rows[example_id]) == group
            ]
            if not group_ids:
                cells.append("--")
            else:
                cells.append(f"{acc([run_rows[i] for i in group_ids]):.4f} ({len(group_ids)})")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    lines.extend(
        [
        "",
            "Monitor command:",
            "",
            "```bash",
            "while true; do",
            "  PYTHONPATH=.:src .venv/bin/python tools/monitors/live_lvb_qwen2_api_matrix.py",
            "  sleep 2",
            "done",
            "```",
        ]
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
