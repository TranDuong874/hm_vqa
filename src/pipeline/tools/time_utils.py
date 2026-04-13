from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any


TIME_PATTERN = re.compile(
    r"(?P<hours>\d{1,2}):(?P<minutes>\d{2}):(?P<seconds>\d{2}(?:\.\d{1,3})?)"
)


@dataclass(slots=True)
class TimeInterval:
    start_time_sec: float
    end_time_sec: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def format_seconds(seconds: float) -> str:
    total = max(float(seconds), 0.0)
    hours = int(total // 3600)
    minutes = int((total % 3600) // 60)
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def parse_timecode(value: str) -> float | None:
    match = TIME_PATTERN.search(str(value))
    if not match:
        return None
    hours = int(match.group("hours"))
    minutes = int(match.group("minutes"))
    seconds = float(match.group("seconds"))
    return hours * 3600.0 + minutes * 60.0 + seconds


def parse_interval(value: Any) -> TimeInterval | None:
    if value is None:
        return None
    if isinstance(value, TimeInterval):
        return value
    if isinstance(value, dict):
        start = value.get("start_time_sec")
        end = value.get("end_time_sec")
        if start is None or end is None:
            return None
        return TimeInterval(float(start), float(end))
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return TimeInterval(float(value[0]), float(value[1]))

    text = str(value)
    if "to" in text:
        left, right = text.split("to", 1)
        start = parse_timecode(left)
        end = parse_timecode(right)
        if start is not None and end is not None:
            return TimeInterval(start, end)
    times = [parse_timecode(match.group(0)) for match in TIME_PATTERN.finditer(text)]
    if len(times) >= 2 and times[0] is not None and times[1] is not None:
        return TimeInterval(float(times[0]), float(times[1]))
    return None


def interval_overlap(a: TimeInterval, b: TimeInterval) -> float:
    return max(0.0, min(a.end_time_sec, b.end_time_sec) - max(a.start_time_sec, b.start_time_sec))


def interval_gap(a: TimeInterval, b: TimeInterval) -> float:
    if interval_overlap(a, b) > 0.0:
        return 0.0
    if a.end_time_sec < b.start_time_sec:
        return float(b.start_time_sec - a.end_time_sec)
    return float(a.start_time_sec - b.end_time_sec)


def interval_midpoint(a: TimeInterval) -> float:
    return (float(a.start_time_sec) + float(a.end_time_sec)) / 2.0


def nearest_reference(
    candidate: TimeInterval,
    references: list[TimeInterval],
) -> tuple[int | None, float | None]:
    if not references:
        return None, None
    best_index: int | None = None
    best_gap: float | None = None
    for index, reference in enumerate(references):
        gap = interval_gap(candidate, reference)
        if best_gap is None or gap < best_gap:
            best_gap = gap
            best_index = index
    return best_index, best_gap


def normalize_intervals(values: list[Any] | None) -> list[TimeInterval]:
    intervals: list[TimeInterval] = []
    for value in values or []:
        interval = parse_interval(value)
        if interval is not None:
            intervals.append(interval)
    return intervals
