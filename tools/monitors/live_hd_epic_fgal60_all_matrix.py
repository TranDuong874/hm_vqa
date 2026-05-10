from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


OUT = Path("results/experimentations_plan/live_hd_epic_fgal60_all_matrix.md")
MANIFEST = Path("results/hd_epic/manifests/fgal_60videos_allq_v1.json")
ROOT = Path("results/hd_epic/ablations_fgal60_allq")

RUN_REGISTRY = [
    {
        "key": "hm_viclip",
        "label": "Fixed60 + fixed L2 5s + ViCLIP",
        "path": ROOT / "l3_rerank_l2_l3fixed60s_s60s_l3k10_l2w5_l2s5_l2encviclip_cov0p6",
    },
    {
        "key": "l1_project_l3",
        "label": "L1 frame-score projected to fixed L3",
        "path": ROOT / "l1_project_l3_l3fixed60s_s60s_l3k10_cov0p6",
    },
    {
        "key": "l2_openclip",
        "label": "Fixed60 + fixed L2 5s + OpenCLIP rerank",
        "path": ROOT / "l3_rerank_l2_l3fixed60s_s60s_l3k10_l2w5_l2s5_cov0p6",
    },
    {
        "key": "fixed_l3",
        "label": "Fixed60 L3 only",
        "path": ROOT / "l3_l3fixed60s_s60s_l3k10_cov0p6",
    },
]


def load_manifest_total() -> int:
    if not MANIFEST.exists():
        return 0
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rows = payload.get("rows", payload if isinstance(payload, list) else [])
    return len(rows)


def load_rows(path: Path) -> dict[str, dict]:
    rows_path = path / "rows.jsonl"
    if not rows_path.exists():
        return {}
    rows: dict[str, dict] = {}
    for line in rows_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[str(row["example_id"])] = row
    return rows


def load_summary(path: Path) -> dict:
    for name in ("rolling_summary.json", "final_summary.json"):
        summary_path = path / name
        if summary_path.exists():
            return json.loads(summary_path.read_text(encoding="utf-8"))
    return {}


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    total = load_manifest_total()
    rows_by_key = {run["key"]: load_rows(Path(run["path"])) for run in RUN_REGISTRY}
    summaries = {run["key"]: load_summary(Path(run["path"])) for run in RUN_REGISTRY}
    existing = [run for run in RUN_REGISTRY if rows_by_key[run["key"]] or summaries[run["key"]]]

    lines = [
        "# Live HD-EPIC FGAL60-All Retrieval Matrix",
        "",
        f"Updated: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`",
        "",
        f"Manifest: `{MANIFEST}`",
        f"Total examples: `{total}`",
        "",
        "| Method | Completed | @1 MeanCov | @1 Hit | @3 MeanCov | @3 Hit | @5 MeanCov | @5 Hit | AvgDur@3(s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for run in RUN_REGISTRY:
        key = run["key"]
        rows = rows_by_key[key]
        summary = summaries[key]
        completed = int(summary.get("completed", len(rows))) if summary else len(rows)
        lines.append(
            f"| {run['label']} | {completed}/{total or summary.get('total', 0)} | "
            f"{float(summary.get('mean_best_coverage_at_1', 0.0)):.4f} | "
            f"{float(summary.get('coverage_recall_at_1', 0.0)):.4f} | "
            f"{float(summary.get('mean_best_coverage_at_3', 0.0)):.4f} | "
            f"{float(summary.get('coverage_recall_at_3', 0.0)):.4f} | "
            f"{float(summary.get('mean_best_coverage_at_5', 0.0)):.4f} | "
            f"{float(summary.get('coverage_recall_at_5', 0.0)):.4f} | "
            f"{float(summary.get('avg_total_duration_at_3', 0.0)):.1f} |"
        )
    lines.append("")

    if len(existing) >= 2:
        base = existing[0]
        base_key = base["key"]
        ids = sorted(
            set(rows_by_key[base_key]).intersection(
                *(set(rows_by_key[run["key"]]) for run in existing[1:])
            )
        )
        lines += [
            f"## Matched Comparison vs `{base['label']}`",
            "",
            f"Matched examples: `{len(ids)}`",
            "",
            "| Method | Delta @1 MeanCov | Delta @3 MeanCov | Delta @5 MeanCov | @3 W/L/T vs HM |",
            "|---|---:|---:|---:|---:|",
        ]
        for run in existing[1:]:
            key = run["key"]
            if not ids:
                lines.append(f"| {run['label']} | +0.0000 | +0.0000 | +0.0000 | 0/0/0 |")
                continue
            deltas = []
            for k in (1, 3, 5):
                metric = f"best_coverage_at_{k}"
                base_cov = mean([rows_by_key[base_key][example_id]["metrics"][metric] for example_id in ids])
                cov = mean([rows_by_key[key][example_id]["metrics"][metric] for example_id in ids])
                deltas.append(cov - base_cov)
            metric = "best_coverage_at_3"
            wins = sum(1 for example_id in ids if rows_by_key[base_key][example_id]["metrics"][metric] > rows_by_key[key][example_id]["metrics"][metric])
            losses = sum(1 for example_id in ids if rows_by_key[base_key][example_id]["metrics"][metric] < rows_by_key[key][example_id]["metrics"][metric])
            ties = len(ids) - wins - losses
            lines.append(f"| {run['label']} | {deltas[0]:+.4f} | {deltas[1]:+.4f} | {deltas[2]:+.4f} | {wins}/{losses}/{ties} |")
        lines.append("")

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
