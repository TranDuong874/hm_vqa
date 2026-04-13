from __future__ import annotations

from typing import Any

from .time_utils import interval_gap, interval_overlap, parse_interval


def verify_consistency_tool(arguments: dict[str, Any]) -> dict[str, Any]:
    selected_interval = parse_interval(arguments.get("selected_interval"))
    answer_interval = parse_interval(arguments.get("answer_interval"))
    tolerance_sec = float(arguments.get("tolerance_sec", 0.0))

    if selected_interval is None:
        raise ValueError("`selected_interval` is required")
    if answer_interval is None:
        raise ValueError("`answer_interval` is required")

    overlap = interval_overlap(selected_interval, answer_interval)
    gap = interval_gap(selected_interval, answer_interval)
    consistent = overlap > 0.0 or gap <= tolerance_sec

    return {
        "consistent": consistent,
        "overlap_sec": overlap,
        "gap_sec": gap,
        "tolerance_sec": tolerance_sec,
        "reason": (
            "intervals overlap"
            if overlap > 0.0
            else f"gap {gap:.3f}s {'within' if consistent else 'exceeds'} tolerance"
        ),
    }
