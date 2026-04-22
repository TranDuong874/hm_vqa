from __future__ import annotations

import numpy as np
import torch

from .types import Segment


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


def _moving_average_1d(values: np.ndarray, kernel_size: int) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float32)
    size = min(max(int(kernel_size), 1), int(values.size))
    if size == 1:
        return values.astype(np.float32)
    kernel = np.ones((size,), dtype=np.float32) / float(size)
    return np.convolve(values.astype(np.float32), kernel, mode="same").astype(np.float32)


def _segment_from_indices(
    *,
    timestamps: np.ndarray,
    start_index: int,
    end_index: int,
    segment_id: str,
) -> Segment:
    start = int(start_index)
    end = max(start, int(end_index))
    start_time = float(timestamps[start])
    end_time = float(timestamps[end])
    return Segment(
        segment_id=segment_id,
        start_index=start,
        end_index=end,
        start_time_sec=start_time,
        end_time_sec=end_time,
        duration_sec=max(0.0, end_time - start_time),
    )


def _best_boundary_in_range(
    *,
    timestamps: np.ndarray,
    contrast: np.ndarray,
    parent_start: int,
    start_index: int,
    low_time: float,
    high_time: float,
) -> int | None:
    candidates: list[tuple[float, int]] = []
    for frame_index in range(start_index + 1, len(timestamps)):
        time_sec = float(timestamps[frame_index])
        if time_sec < low_time:
            continue
        if time_sec > high_time:
            break
        contrast_index = frame_index - parent_start
        if 0 <= contrast_index < len(contrast):
            candidates.append((float(contrast[contrast_index]), frame_index - 1))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return int(candidates[0][1])


def segment_l3_local_contrast_windows(
    *,
    timestamps: np.ndarray,
    frame_embeddings: torch.Tensor,
    l3_segments: list[Segment],
    min_duration_sec: float = 3.0,
    max_duration_sec: float = 12.0,
    fast_kernel_size: int = 1,
    slow_kernel_size: int = 9,
    peak_percentile: float = 75.0,
    min_peak_gap_sec: float = 2.0,
    prefix: str = "l2_l3local_contrast",
) -> list[Segment]:
    """Build local L2 windows inside L3 segments using fast-minus-slow drift peaks."""
    if len(timestamps) == 0 or not l3_segments:
        return []
    if frame_embeddings.ndim != 2:
        raise ValueError("frame_embeddings must be 2D")
    if frame_embeddings.shape[0] != len(timestamps):
        raise ValueError("frame_embeddings and timestamps must have matching length")
    if min_duration_sec <= 0.0:
        raise ValueError("min_duration_sec must be positive")
    if max_duration_sec < min_duration_sec:
        raise ValueError("max_duration_sec must be >= min_duration_sec")
    if not 0.0 <= peak_percentile <= 100.0:
        raise ValueError("peak_percentile must be in [0, 100]")

    normalized = torch.nn.functional.normalize(frame_embeddings, dim=-1)
    drift = np.zeros((len(timestamps),), dtype=np.float32)
    if normalized.shape[0] > 1:
        similarities = (normalized[:-1] * normalized[1:]).sum(dim=-1)
        drift[1:] = (1.0 - similarities).cpu().numpy().astype(np.float32)

    output: list[Segment] = []
    segment_index = 0
    for parent_index, parent in enumerate(l3_segments):
        parent_start = max(0, int(parent.start_index))
        parent_end = min(len(timestamps) - 1, int(parent.end_index))
        if parent_end < parent_start:
            continue
        if float(timestamps[parent_end]) - float(timestamps[parent_start]) <= max_duration_sec:
            output.append(
                _segment_from_indices(
                    timestamps=timestamps,
                    start_index=parent_start,
                    end_index=parent_end,
                    segment_id=f"{prefix}_{segment_index:04d}",
                )
            )
            segment_index += 1
            continue

        local_drift = drift[parent_start : parent_end + 1]
        fast = _moving_average_1d(local_drift, fast_kernel_size)
        slow = _moving_average_1d(local_drift, slow_kernel_size)
        contrast = np.maximum(fast - slow, 0.0).astype(np.float32)
        if contrast.size:
            contrast[0] = 0.0

        positive = contrast[contrast > 0.0]
        threshold = float(np.percentile(positive, peak_percentile)) if positive.size else float("inf")
        peaks: list[int] = []
        for local_index in range(1, max(len(contrast) - 1, 1)):
            left = contrast[local_index - 1] if local_index - 1 >= 0 else -np.inf
            right = contrast[local_index + 1] if local_index + 1 < len(contrast) else -np.inf
            if float(contrast[local_index]) < threshold:
                continue
            if float(contrast[local_index]) < float(left) or float(contrast[local_index]) < float(right):
                continue
            frame_index = parent_start + local_index
            if frame_index <= parent_start or frame_index > parent_end:
                continue
            peaks.append(frame_index - 1)

        boundaries: list[int] = []
        current_start = parent_start
        last_boundary_time: float | None = None
        for peak_boundary in sorted(peaks):
            boundary_time = float(timestamps[peak_boundary])
            current_duration = boundary_time - float(timestamps[current_start])
            tail_duration = float(timestamps[parent_end]) - float(timestamps[min(peak_boundary + 1, parent_end)])
            if current_duration < min_duration_sec or tail_duration < min_duration_sec:
                continue
            if last_boundary_time is not None and boundary_time - last_boundary_time < min_peak_gap_sec:
                continue
            while current_duration > max_duration_sec:
                forced = _best_boundary_in_range(
                    timestamps=timestamps,
                    contrast=contrast,
                    parent_start=parent_start,
                    start_index=current_start,
                    low_time=float(timestamps[current_start]) + min_duration_sec,
                    high_time=float(timestamps[current_start]) + max_duration_sec,
                )
                if forced is None or forced < current_start:
                    forced = int(np.searchsorted(timestamps, float(timestamps[current_start]) + max_duration_sec, side="right") - 1)
                    forced = max(current_start, min(forced, parent_end - 1))
                boundaries.append(forced)
                last_boundary_time = float(timestamps[forced])
                current_start = min(forced + 1, parent_end)
                current_duration = boundary_time - float(timestamps[current_start])
            boundaries.append(peak_boundary)
            last_boundary_time = boundary_time
            current_start = min(peak_boundary + 1, parent_end)

        while float(timestamps[parent_end]) - float(timestamps[current_start]) > max_duration_sec:
            forced = _best_boundary_in_range(
                timestamps=timestamps,
                contrast=contrast,
                parent_start=parent_start,
                start_index=current_start,
                low_time=float(timestamps[current_start]) + min_duration_sec,
                high_time=float(timestamps[current_start]) + max_duration_sec,
            )
            if forced is None or forced < current_start:
                forced = int(np.searchsorted(timestamps, float(timestamps[current_start]) + max_duration_sec, side="right") - 1)
                forced = max(current_start, min(forced, parent_end - 1))
            boundaries.append(forced)
            current_start = min(forced + 1, parent_end)

        split_start = parent_start
        unique_boundaries = sorted(set(boundary for boundary in boundaries if parent_start <= boundary < parent_end))
        for boundary in unique_boundaries:
            if float(timestamps[boundary]) - float(timestamps[split_start]) < min_duration_sec:
                continue
            output.append(
                _segment_from_indices(
                    timestamps=timestamps,
                    start_index=split_start,
                    end_index=boundary,
                    segment_id=f"{prefix}_{segment_index:04d}",
                )
            )
            segment_index += 1
            split_start = boundary + 1
        if split_start <= parent_end:
            if output and float(timestamps[parent_end]) - float(timestamps[split_start]) < min_duration_sec:
                previous = output.pop()
                output.append(
                    _segment_from_indices(
                        timestamps=timestamps,
                        start_index=previous.start_index,
                        end_index=parent_end,
                        segment_id=previous.segment_id,
                    )
                )
            else:
                output.append(
                    _segment_from_indices(
                        timestamps=timestamps,
                        start_index=split_start,
                        end_index=parent_end,
                        segment_id=f"{prefix}_{segment_index:04d}",
                    )
                )
                segment_index += 1

    return output
