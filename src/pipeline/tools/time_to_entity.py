from __future__ import annotations

from typing import Any

from .time_utils import format_seconds, interval_gap, interval_overlap, parse_interval


def time_to_entity_tool(arguments: dict[str, Any]) -> dict[str, Any]:
    interval = parse_interval(arguments.get("interval"))
    candidates = list(arguments.get("candidates") or [])
    top_k = max(int(arguments.get("top_k", 5)), 1)

    if interval is None:
        raise ValueError("`interval` is required")
    if not candidates:
        raise ValueError("`candidates` is required")

    ranked: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_interval = parse_interval(candidate.get("interval")) or parse_interval(candidate)
        if candidate_interval is None:
            continue
        overlap = interval_overlap(interval, candidate_interval)
        gap = interval_gap(interval, candidate_interval)
        ranked.append(
            {
                "entity": candidate.get("entity") or candidate.get("label") or candidate.get("text"),
                "interval": candidate_interval.to_dict(),
                "display": f"{format_seconds(candidate_interval.start_time_sec)} to {format_seconds(candidate_interval.end_time_sec)}",
                "overlap_sec": overlap,
                "gap_sec": gap,
                "score": float(candidate.get("score", 0.0)),
                "metadata": candidate,
            }
        )

    ranked.sort(key=lambda item: (item["gap_sec"], -item["overlap_sec"], -item["score"]))
    return {
        "query_interval": interval.to_dict(),
        "top_k": top_k,
        "results": ranked[:top_k],
    }
