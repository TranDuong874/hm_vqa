from __future__ import annotations

from typing import Any


DEFAULT_RECALL_K = (1, 3, 5)


def _interval_iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    if inter <= 0.0:
        return 0.0
    union = max(a_end, b_end) - min(a_start, b_start)
    return inter / union if union > 0.0 else 0.0


def _coverage_ratio(hit: dict[str, float], gold_span: dict[str, Any]) -> float:
    inter = max(
        0.0,
        min(float(hit["end_time_sec"]), float(gold_span["end_time_sec"]))
        - max(float(hit["start_time_sec"]), float(gold_span["start_time_sec"])),
    )
    gold_duration = max(1e-6, float(gold_span["end_time_sec"]) - float(gold_span["start_time_sec"]))
    return inter / gold_duration


def _best_coverage(hit: dict[str, float], gold_spans: list[dict[str, Any]]) -> float:
    return max((_coverage_ratio(hit, span) for span in gold_spans), default=0.0)


def _gap0_overlap(hit: dict[str, float], gold_spans: list[dict[str, Any]]) -> float:
    for span in gold_spans:
        if float(hit["end_time_sec"]) >= float(span["start_time_sec"]) and float(hit["start_time_sec"]) <= float(span["end_time_sec"]):
            return 1.0
    return 0.0


def _duration(hit: dict[str, Any]) -> float:
    return max(0.0, float(hit["end_time_sec"]) - float(hit["start_time_sec"]))


def _metrics_for_hits(
    *,
    hits: list[dict[str, float]],
    gold_spans: list[dict[str, Any]],
    coverage_threshold: float,
    recall_ks: tuple[int, ...] = DEFAULT_RECALL_K,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for k in recall_ks:
        metrics[f"best_coverage_at_{k}"] = 0.0
        metrics[f"coverage_recall_at_{k}"] = 0.0
        metrics[f"gap_recall_at_{k}"] = 0.0
        metrics[f"avg_window_duration_at_{k}"] = 0.0
        metrics[f"total_duration_at_{k}"] = 0.0
    if not hits or not gold_spans:
        return metrics
    best_coverages = [_best_coverage(hit, gold_spans) for hit in hits]
    gap_hits = [_gap0_overlap(hit, gold_spans) for hit in hits]
    durations = [_duration(hit) for hit in hits]
    for k in recall_ks:
        scoped_coverages = best_coverages[:k]
        scoped_gaps = gap_hits[:k]
        scoped_durations = durations[:k]
        best_coverage = max(scoped_coverages, default=0.0)
        metrics[f"best_coverage_at_{k}"] = float(best_coverage)
        metrics[f"coverage_recall_at_{k}"] = 1.0 if best_coverage >= coverage_threshold else 0.0
        metrics[f"gap_recall_at_{k}"] = 1.0 if any(scoped_gaps) else 0.0
        metrics[f"avg_window_duration_at_{k}"] = (
            sum(scoped_durations) / len(scoped_durations) if scoped_durations else 0.0
        )
        metrics[f"total_duration_at_{k}"] = sum(scoped_durations)
    return metrics


def _summarize(rows: list[dict[str, Any]], total: int) -> dict[str, Any]:
    scored = len(rows)
    summary: dict[str, Any] = {
        "completed": scored,
        "total": total,
        "scored": scored,
    }
    for k in DEFAULT_RECALL_K:
        if not rows:
            summary[f"mean_best_coverage_at_{k}"] = 0.0
            summary[f"coverage_recall_at_{k}"] = 0.0
            summary[f"gap_recall_at_{k}"] = 0.0
            summary[f"avg_window_duration_at_{k}"] = 0.0
            summary[f"avg_total_duration_at_{k}"] = 0.0
            continue
        summary[f"mean_best_coverage_at_{k}"] = (
            sum(float(row["metrics"][f"best_coverage_at_{k}"]) for row in rows) / scored
        )
        summary[f"coverage_recall_at_{k}"] = (
            sum(float(row["metrics"][f"coverage_recall_at_{k}"]) for row in rows) / scored
        )
        summary[f"gap_recall_at_{k}"] = (
            sum(float(row["metrics"][f"gap_recall_at_{k}"]) for row in rows) / scored
        )
        summary[f"avg_window_duration_at_{k}"] = (
            sum(float(row["metrics"][f"avg_window_duration_at_{k}"]) for row in rows) / scored
        )
        summary[f"avg_total_duration_at_{k}"] = (
            sum(float(row["metrics"][f"total_duration_at_{k}"]) for row in rows) / scored
        )
    return summary

