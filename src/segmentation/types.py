from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Segment:
    segment_id: str
    start_index: int
    end_index: int
    start_time_sec: float
    end_time_sec: float
    duration_sec: float
