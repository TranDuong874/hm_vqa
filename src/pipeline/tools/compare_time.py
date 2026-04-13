from __future__ import annotations

from typing import Any

from .time_utils import (
    format_seconds,
    interval_overlap,
    nearest_reference,
    normalize_intervals,
)


def compare_time_tool(arguments: dict[str, Any]) -> dict[str, Any]:
    candidates = normalize_intervals(arguments.get("candidates"))
    references = normalize_intervals(arguments.get("references"))
    tolerance_sec = float(arguments.get("tolerance_sec", 0.0))

    if not candidates:
        raise ValueError("`candidates` must contain at least one interval")
    if not references:
        raise ValueError("`references` must contain at least one interval")

    pairwise: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(candidates):
        nearest_index, nearest_gap = nearest_reference(candidate, references)
        nearest_overlap = None
        if nearest_index is not None:
            nearest_overlap = interval_overlap(candidate, references[nearest_index])
        pairwise.append(
            {
                "candidate_index": candidate_index,
                "candidate": candidate.to_dict(),
                "nearest_reference_index": nearest_index,
                "nearest_gap_sec": nearest_gap,
                "nearest_overlap_sec": nearest_overlap,
                "within_tolerance": bool(nearest_gap is not None and nearest_gap <= tolerance_sec),
                "display": (
                    f"{format_seconds(candidate.start_time_sec)} to {format_seconds(candidate.end_time_sec)}"
                ),
            }
        )

    best = min(pairwise, key=lambda item: (float(item["nearest_gap_sec"] or 0.0), -float(item["nearest_overlap_sec"] or 0.0)))
    return {
        "tolerance_sec": tolerance_sec,
        "best_match": best,
        "pairwise": pairwise,
        "all_within_tolerance": [item for item in pairwise if item["within_tolerance"]],
    }
