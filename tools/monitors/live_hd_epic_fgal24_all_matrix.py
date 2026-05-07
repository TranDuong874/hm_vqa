from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path


OUT = Path("results/experimentations_plan/live_hd_epic_fgal24_all_matrix.md")
MANIFEST = Path("results/hd_epic/manifests/fgal_24videos_allq_v1.json")

MANIFEST_BUCKET_LABELS = {
    "short": "short <5m",
    "medium": "medium 5-15m",
    "long": "long >15m",
}

RUN_REGISTRY = [
    {
        "key": "fixed60_fixed15_l2_viclip",
        "label": "Fixed60 + fixed L2 15s + ViCLIP",
        "path": Path(
            "results/hd_epic/localization_retrieval_fgal24_all_v1/"
            "l3_rerank_l2_l3fixed60s_s60s_l3k10_l2w15_l2s15_l2encviclip_cov0p6"
        ),
        "priority": 10,
    },
    {
        "key": "fixed60_fixed5_l2_viclip",
        "label": "Fixed60 + fixed L2 5s + ViCLIP",
        "path": Path(
            "results/hd_epic/localization_retrieval_fgal24_all_v1/"
            "l3_rerank_l2_l3fixed60s_s60s_l3k10_l2w5_l2s5_l2encviclip_cov0p6"
        ),
        "priority": 20,
    },
    {
        "key": "fixed60_adaptive_l2_viclip",
        "label": "Fixed60 + adaptive L2 + ViCLIP",
        "path": Path(
            "results/hd_epic/localization_retrieval_fgal24_all_v1/"
            "l3_rerank_l2_l3fixed60s_s60s_l3k10_l2l3_local_contrast_min3_max12_p65_l2encviclip_cov0p6"
        ),
        "priority": 30,
    },
    {
        "key": "adaptive_l3_adaptive_l2_viclip",
        "label": "Adaptive L3 + adaptive L2 + ViCLIP",
        "path": Path(
            "results/hd_epic/localization_retrieval_fgal24_all_v1/"
            "l3_rerank_l2_l3k10_l2l3_local_contrast_min3_max12_p65_l2encviclip_cov0p6"
        ),
        "priority": 40,
    },
    {
        "key": "adaptive_l3_fixed5_l2_viclip",
        "label": "Adaptive L3 + fixed L2 5s + ViCLIP",
        "path": Path(
            "results/hd_epic/localization_retrieval_fgal24_all_v1/"
            "l3_rerank_l2_l3k10_l2w5_l2s5_l2encviclip_cov0p6"
        ),
        "priority": 50,
    },
    {
        "key": "fixed60_l3_only",
        "label": "Fixed60 L3 only",
        "path": Path(
            "results/hd_epic/localization_retrieval_fgal24_all_v1/"
            "l3_l3fixed60s_s60s_l3k10_cov0p6"
        ),
        "priority": 60,
    },
    {
        "key": "fixed60_fixed10_l2_viclip",
        "label": "Fixed60 + fixed L2 10s + ViCLIP",
        "path": Path(
            "results/hd_epic/localization_retrieval_fgal24_all_v1/"
            "l3_rerank_l2_l3fixed60s_s60s_l3k10_l2w10_l2s10_l2encviclip_cov0p6"
        ),
        "priority": 70,
    },
    {
        "key": "fixed60_adaptive_l2_openclip",
        "label": "Fixed60 + adaptive L2 + OpenCLIP",
        "path": Path(
            "results/hd_epic/localization_retrieval_fgal24_all_v1/"
            "l3_rerank_l2_l3fixed60s_s60s_l3k10_l2l3_local_contrast_min3_max12_p65_cov0p6"
        ),
        "priority": 80,
    },
]


def load_rows(path: Path) -> dict[str, dict]:
    rows_path = path / "rows.jsonl"
    rows: dict[str, dict] = {}
    if not rows_path.exists():
        return rows
    for line in rows_path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            rows[str(row["example_id"])] = row
    return rows


def load_summary(path: Path) -> dict:
    summary_path = path / "rolling_summary.json"
    return json.loads(summary_path.read_text()) if summary_path.exists() else {}


def load_manifest_buckets(path: Path = MANIFEST) -> dict[str, str]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    manifest_rows = data.get("rows", data if isinstance(data, list) else [])
    buckets: dict[str, str] = {}
    for row in manifest_rows:
        example_id = row.get("example_id")
        raw_bucket = row.get("duration_bucket")
        if example_id is None or raw_bucket is None:
            continue
        buckets[str(example_id)] = MANIFEST_BUCKET_LABELS.get(str(raw_bucket), str(raw_bucket))
    return buckets


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def scope_duration(row: dict) -> float:
    return float(row.get("scope_end_sec") or 0.0) - float(row.get("scope_start_sec") or 0.0)


def duration_bucket(seconds: float) -> str:
    if seconds < 300:
        return "short <5m"
    if seconds < 900:
        return "medium 5-15m"
    return "long >15m"


def metric_table(
    *,
    rows: dict[str, dict[str, dict]],
    methods: list[dict],
    current_key: str,
    ids: list[str],
    title: str,
) -> list[str]:
    lines: list[str] = [f"## {title}", "", f"Matched examples: `{len(ids)}`", ""]
    if not ids:
        lines += ["No matched rows yet.", ""]
        return lines

    for k in (1, 3, 5):
        coverage_key = f"best_coverage_at_{k}"
        hit_key = f"coverage_recall_at_{k}"
        duration_key = f"total_duration_at_{k}"
        base_cov = mean([rows[current_key][example_id]["metrics"][coverage_key] for example_id in ids])

        lines += [
            f"### @{k}",
            "",
            "| Method | MeanCov | Hit | AvgTotalDur(s) | DeltaCov vs current |",
            "|---|---:|---:|---:|---:|",
        ]
        for method in methods:
            key = str(method["key"])
            label = str(method["label"])
            cov = mean([rows[key][example_id]["metrics"][coverage_key] for example_id in ids])
            hit = mean([rows[key][example_id]["metrics"][hit_key] for example_id in ids])
            dur = mean([rows[key][example_id]["metrics"][duration_key] for example_id in ids])
            lines.append(f"| {label} | {cov:.4f} | {hit:.4f} | {dur:.1f} | {cov - base_cov:+.4f} |")
        lines.append("")

    lines += [
        "### Pairwise W/L/T Against Current",
        "",
        "| Baseline | @1 W/L/T | @3 W/L/T | @5 W/L/T |",
        "|---|---:|---:|---:|",
    ]
    for method in methods:
        key = str(method["key"])
        label = str(method["label"])
        if key == current_key:
            continue
        cells: list[str] = []
        for k in (1, 3, 5):
            coverage_key = f"best_coverage_at_{k}"
            wins = sum(
                1
                for example_id in ids
                if rows[current_key][example_id]["metrics"][coverage_key]
                > rows[key][example_id]["metrics"][coverage_key]
            )
            losses = sum(
                1
                for example_id in ids
                if rows[current_key][example_id]["metrics"][coverage_key]
                < rows[key][example_id]["metrics"][coverage_key]
            )
            ties = len(ids) - wins - losses
            cells.append(f"{wins}/{losses}/{ties}")
        lines.append(f"| {label} | {cells[0]} | {cells[1]} | {cells[2]} |")
    lines.append("")
    return lines


def main() -> None:
    methods = [
        method
        for method in sorted(RUN_REGISTRY, key=lambda item: int(item["priority"]))
        if (Path(method["path"]) / "rows.jsonl").exists()
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not methods:
        OUT.write_text("# Live HD-EPIC FGAL24-All Retrieval Matrix\n\nNo FGAL24-all result rows found yet.\n")
        return

    current_key = str(methods[0]["key"])
    rows = {str(method["key"]): load_rows(Path(method["path"])) for method in methods}
    summaries = {str(method["key"]): load_summary(Path(method["path"])) for method in methods}
    matched_ids = sorted(set.intersection(*(set(method_rows) for method_rows in rows.values())))
    manifest_buckets = load_manifest_buckets()

    bucketed_ids: dict[str, list[str]] = defaultdict(list)
    for example_id in matched_ids:
        bucket = manifest_buckets.get(example_id)
        if bucket is None:
            bucket = duration_bucket(scope_duration(rows[current_key][example_id]))
        bucketed_ids[bucket].append(example_id)

    lines: list[str] = [
        "# Live HD-EPIC FGAL24-All Retrieval Matrix",
        "",
        f"Updated: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`",
        "",
        "Manifest: `results/hd_epic/manifests/fgal_24videos_allq_v1.json`",
        "",
        "This run uses all available FGAL questions from the same 24-video diverse set.",
        "",
        "Duration buckets use the FGAL24 manifest source-video duration, not the per-question scope length.",
        "",
        f"Current baseline for deltas: `{methods[0]['label']}`",
        "",
    ]
    current_summary = summaries[current_key]
    lines.append(
        "Current progress: `{}/{}`".format(
            current_summary.get("completed", len(rows[current_key])),
            current_summary.get("total", "?"),
        )
    )
    lines += ["", "Compared methods:", ""]
    for method in methods:
        key = str(method["key"])
        label = str(method["label"])
        summary = summaries[key]
        lines.append(f"- `{label}`: `{summary.get('completed', len(rows[key]))}/{summary.get('total', '?')}`")
    lines.append("")

    missing = [
        method
        for method in sorted(RUN_REGISTRY, key=lambda item: int(item["priority"]))
        if not (Path(method["path"]) / "rows.jsonl").exists()
    ]
    if missing:
        lines += ["Not included yet because `rows.jsonl` does not exist:", ""]
        for method in missing:
            lines.append(f"- `{method['label']}`")
        lines.append("")

    lines.extend(
        metric_table(
            rows=rows,
            methods=methods,
            current_key=current_key,
            ids=matched_ids,
            title="Overall FGAL24-All Matched Comparison",
        )
    )
    lines += ["## Duration Buckets", "", "| Bucket | Matched n |", "|---|---:|"]
    for bucket in ("short <5m", "medium 5-15m", "long >15m"):
        lines.append(f"| {bucket} | {len(bucketed_ids[bucket])} |")
    lines.append("")
    for bucket in ("short <5m", "medium 5-15m", "long >15m"):
        lines.extend(
            metric_table(
                rows=rows,
                methods=methods,
                current_key=current_key,
                ids=bucketed_ids[bucket],
                title=f"By Duration: {bucket}",
            )
        )

    OUT.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
