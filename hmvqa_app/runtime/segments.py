from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class Segment:
    segment_id: str
    start_index: int
    end_index: int
    start_time_sec: float
    end_time_sec: float
    duration_sec: float


def segment_fixed_windows(
    *,
    timestamps: np.ndarray,
    window_seconds: float,
    stride_seconds: float | None,
    prefix: str = "window",
) -> list[Segment]:
    if len(timestamps) == 0:
        return []
    if window_seconds <= 0.0:
        raise ValueError("window_seconds must be positive")

    stride = float(window_seconds if stride_seconds is None else stride_seconds)
    if stride <= 0.0:
        raise ValueError("stride_seconds must be positive")

    segments: list[Segment] = []
    video_start = float(timestamps[0])
    video_end = float(timestamps[-1])
    start_time = video_start
    segment_index = 0

    while start_time < video_end:
        end_time = min(start_time + float(window_seconds), video_end)
        start_index = int(np.searchsorted(timestamps, start_time, side="left"))
        end_index = int(np.searchsorted(timestamps, end_time, side="right") - 1)
        if end_index < start_index:
            start_time += stride
            continue
        actual_start = float(timestamps[start_index])
        actual_end = float(timestamps[end_index])
        segments.append(
            Segment(
                segment_id=f"{prefix}_{segment_index:04d}",
                start_index=start_index,
                end_index=end_index,
                start_time_sec=actual_start,
                end_time_sec=actual_end,
                duration_sec=max(0.0, actual_end - actual_start),
            )
        )
        segment_index += 1
        start_time += stride
        if end_time >= video_end:
            break

    return segments
