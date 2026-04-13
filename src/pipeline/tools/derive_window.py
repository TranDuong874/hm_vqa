from __future__ import annotations

from typing import Any

from .time_utils import TimeInterval, format_seconds, parse_interval


def _window_for_before(anchor: TimeInterval, window_sec: float) -> TimeInterval:
    end_time = max(anchor.start_time_sec, 0.0)
    start_time = max(end_time - window_sec, 0.0)
    return TimeInterval(start_time, end_time)


def _window_for_after(anchor: TimeInterval, window_sec: float) -> TimeInterval:
    start_time = max(anchor.end_time_sec, 0.0)
    end_time = start_time + window_sec
    return TimeInterval(start_time, end_time)


def _window_for_between(left: TimeInterval, right: TimeInterval, pad_sec: float) -> TimeInterval:
    start_time = max(left.end_time_sec - pad_sec, 0.0)
    end_time = max(right.start_time_sec + pad_sec, start_time)
    return TimeInterval(start_time, end_time)


def derive_window_tool(arguments: dict[str, Any]) -> dict[str, Any]:
    mode = str(arguments.get("mode", "")).strip().lower()
    window_sec = float(arguments.get("window_sec", 30.0))
    pad_sec = float(arguments.get("pad_sec", 0.0))

    if mode not in {"before", "after", "between"}:
        raise ValueError("`mode` must be one of: before, after, between")

    if mode in {"before", "after"}:
        anchor = parse_interval(arguments.get("anchor_interval"))
        if anchor is None:
            raise ValueError("`anchor_interval` is required for before/after")
        window = _window_for_before(anchor, window_sec) if mode == "before" else _window_for_after(anchor, window_sec)
    else:
        left = parse_interval(arguments.get("left_interval"))
        right = parse_interval(arguments.get("right_interval"))
        if left is None or right is None:
            raise ValueError("`left_interval` and `right_interval` are required for between")
        window = _window_for_between(left, right, pad_sec)

    return {
        "mode": mode,
        "window_sec": window_sec,
        "pad_sec": pad_sec,
        "interval": window.to_dict(),
        "display": f"{format_seconds(window.start_time_sec)} to {format_seconds(window.end_time_sec)}",
    }
