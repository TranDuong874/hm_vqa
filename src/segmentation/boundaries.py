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


def segment_change_threshold(
    *,
    timestamps: np.ndarray,
    frame_embeddings: torch.Tensor,
    change_threshold: float,
    prefix: str = "semantic",
) -> list[Segment]:
    if len(timestamps) == 0:
        return []
    if frame_embeddings.ndim != 2:
        raise ValueError("frame_embeddings must be 2D")
    if frame_embeddings.shape[0] != len(timestamps):
        raise ValueError("frame_embeddings and timestamps must have matching length")
    if change_threshold < 0.0:
        raise ValueError("change_threshold must be non-negative")

    normalized = torch.nn.functional.normalize(frame_embeddings, dim=-1)
    if normalized.shape[0] <= 1:
        return [
            Segment(
                segment_id=f"{prefix}_0000",
                start_index=0,
                end_index=0,
                start_time_sec=float(timestamps[0]),
                end_time_sec=float(timestamps[0]),
                duration_sec=0.0,
            )
        ]

    similarities = (normalized[:-1] * normalized[1:]).sum(dim=-1)
    changes = 1.0 - similarities
    boundary_indices = [
        int(index)
        for index, value in enumerate(changes.tolist())
        if float(value) >= float(change_threshold)
    ]

    segments: list[Segment] = []
    start_index = 0
    segment_index = 0
    for boundary_index in boundary_indices:
        end_index = boundary_index
        if end_index < start_index:
            continue
        start_time = float(timestamps[start_index])
        end_time = float(timestamps[end_index])
        segments.append(
            Segment(
                segment_id=f"{prefix}_{segment_index:04d}",
                start_index=start_index,
                end_index=end_index,
                start_time_sec=start_time,
                end_time_sec=end_time,
                duration_sec=max(0.0, end_time - start_time),
            )
        )
        segment_index += 1
        start_index = boundary_index + 1

    if start_index <= len(timestamps) - 1:
        start_time = float(timestamps[start_index])
        end_time = float(timestamps[-1])
        segments.append(
            Segment(
                segment_id=f"{prefix}_{segment_index:04d}",
                start_index=start_index,
                end_index=len(timestamps) - 1,
                start_time_sec=start_time,
                end_time_sec=end_time,
                duration_sec=max(0.0, end_time - start_time),
            )
        )

    return segments


def _normalize_scores(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float32)
    minimum = float(values.min())
    maximum = float(values.max())
    if maximum - minimum <= 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    normalized = (values - minimum) / (maximum - minimum)
    return normalized.astype(np.float32)


def segment_fused_change_threshold(
    *,
    timestamps: np.ndarray,
    frame_embeddings: torch.Tensor,
    motion_energy: np.ndarray,
    change_threshold: float,
    motion_weight: float = 0.5,
    prefix: str = "semantic",
) -> list[Segment]:
    if len(timestamps) == 0:
        return []
    if frame_embeddings.ndim != 2:
        raise ValueError("frame_embeddings must be 2D")
    if frame_embeddings.shape[0] != len(timestamps):
        raise ValueError("frame_embeddings and timestamps must have matching length")
    if len(motion_energy) != len(timestamps):
        raise ValueError("motion_energy and timestamps must have matching length")
    if change_threshold < 0.0 or change_threshold > 1.0:
        raise ValueError("change_threshold must be in [0, 1]")
    if motion_weight < 0.0 or motion_weight > 1.0:
        raise ValueError("motion_weight must be in [0, 1]")

    normalized = torch.nn.functional.normalize(frame_embeddings, dim=-1)
    semantic_change = np.zeros((len(timestamps),), dtype=np.float32)
    if normalized.shape[0] > 1:
        similarities = (normalized[:-1] * normalized[1:]).sum(dim=-1)
        semantic_change[1:] = (1.0 - similarities).cpu().numpy().astype(np.float32)
    semantic_change = _normalize_scores(semantic_change)
    motion_change = _normalize_scores(np.asarray(motion_energy, dtype=np.float32))
    fused_change = ((1.0 - motion_weight) * semantic_change) + (motion_weight * motion_change)

    boundary_indices = [
        index - 1
        for index, value in enumerate(fused_change.tolist())
        if index > 0 and float(value) >= float(change_threshold)
    ]

    segments: list[Segment] = []
    start_index = 0
    segment_index = 0
    for boundary_index in boundary_indices:
        if boundary_index < start_index:
            continue
        start_time = float(timestamps[start_index])
        end_time = float(timestamps[boundary_index])
        segments.append(
            Segment(
                segment_id=f"{prefix}_{segment_index:04d}",
                start_index=start_index,
                end_index=boundary_index,
                start_time_sec=start_time,
                end_time_sec=end_time,
                duration_sec=max(0.0, end_time - start_time),
            )
        )
        segment_index += 1
        start_index = boundary_index + 1

    if start_index <= len(timestamps) - 1:
        start_time = float(timestamps[start_index])
        end_time = float(timestamps[-1])
        segments.append(
            Segment(
                segment_id=f"{prefix}_{segment_index:04d}",
                start_index=start_index,
                end_index=len(timestamps) - 1,
                start_time_sec=start_time,
                end_time_sec=end_time,
                duration_sec=max(0.0, end_time - start_time),
            )
        )

    return segments


def constrain_segments_by_duration_with_overlap(
    *,
    timestamps: np.ndarray,
    segments: list[Segment],
    min_duration_sec: float = 15.0,
    max_duration_sec: float = 60.0,
    overlap_seconds: float = 5.0,
    prefix: str = "bounded",
) -> list[Segment]:
    if not segments:
        return []
    if max_duration_sec <= 0.0:
        raise ValueError("max_duration_sec must be positive")
    if min_duration_sec < 0.0:
        raise ValueError("min_duration_sec must be non-negative")
    if min_duration_sec > max_duration_sec:
        raise ValueError("min_duration_sec must not exceed max_duration_sec")
    if overlap_seconds < 0.0:
        raise ValueError("overlap_seconds must be non-negative")
    if overlap_seconds >= max_duration_sec:
        raise ValueError("overlap_seconds must be smaller than max_duration_sec")

    merged: list[Segment] = []
    merged_index = 0
    index = 0
    while index < len(segments):
        start_segment = segments[index]
        end_segment = start_segment
        while (
            index + 1 < len(segments)
            and (end_segment.end_time_sec - start_segment.start_time_sec) < min_duration_sec
        ):
            index += 1
            end_segment = segments[index]
        merged.append(
            Segment(
                segment_id=f"{prefix}_merged_{merged_index:04d}",
                start_index=start_segment.start_index,
                end_index=end_segment.end_index,
                start_time_sec=start_segment.start_time_sec,
                end_time_sec=end_segment.end_time_sec,
                duration_sec=max(0.0, end_segment.end_time_sec - start_segment.start_time_sec),
            )
        )
        merged_index += 1
        index += 1

    if len(merged) >= 2 and merged[-1].duration_sec < min_duration_sec:
        previous = merged[-2]
        tail = merged[-1]
        merged[-2] = Segment(
            segment_id=previous.segment_id,
            start_index=previous.start_index,
            end_index=tail.end_index,
            start_time_sec=previous.start_time_sec,
            end_time_sec=tail.end_time_sec,
            duration_sec=max(0.0, tail.end_time_sec - previous.start_time_sec),
        )
        merged.pop()

    constrained: list[Segment] = []
    constrained_index = 0
    stride_seconds = max_duration_sec - overlap_seconds
    video_end_time = float(timestamps[-1])
    for segment in merged:
        if segment.duration_sec <= max_duration_sec:
            constrained.append(
                Segment(
                    segment_id=f"{prefix}_{constrained_index:04d}",
                    start_index=segment.start_index,
                    end_index=segment.end_index,
                    start_time_sec=segment.start_time_sec,
                    end_time_sec=segment.end_time_sec,
                    duration_sec=max(0.0, segment.end_time_sec - segment.start_time_sec),
                )
            )
            constrained_index += 1
            continue

        chunk_start_time = segment.start_time_sec
        while chunk_start_time < segment.end_time_sec:
            chunk_end_time = min(chunk_start_time + max_duration_sec, segment.end_time_sec, video_end_time)
            chunk_start_index = int(
                np.searchsorted(timestamps, chunk_start_time, side="left").clip(segment.start_index, segment.end_index)
            )
            chunk_end_index = int(
                (np.searchsorted(timestamps, chunk_end_time, side="right") - 1).clip(
                    segment.start_index, segment.end_index
                )
            )
            if chunk_end_index < chunk_start_index:
                break
            actual_start = float(timestamps[chunk_start_index])
            actual_end = float(timestamps[chunk_end_index])
            constrained.append(
                Segment(
                    segment_id=f"{prefix}_{constrained_index:04d}",
                    start_index=chunk_start_index,
                    end_index=chunk_end_index,
                    start_time_sec=actual_start,
                    end_time_sec=actual_end,
                    duration_sec=max(0.0, actual_end - actual_start),
                )
            )
            constrained_index += 1
            if chunk_end_time >= segment.end_time_sec:
                break
            next_start_time = chunk_start_time + stride_seconds
            if next_start_time <= chunk_start_time:
                break
            chunk_start_time = next_start_time

    return constrained


def group_segments_by_drift(
    *,
    base_segments: list[Segment],
    segment_embeddings: torch.Tensor,
    drift_threshold: float,
    smoothing_kernel_size: int = 3,
    min_duration_sec: float = 15.0,
    max_duration_sec: float = 60.0,
    adaptive_percentile: float | None = None,
    adaptive_floor: float | None = None,
    adaptive_window_size: int | None = None,
    prefix: str = "grouped",
) -> list[Segment]:
    if not base_segments:
        return []
    if segment_embeddings.ndim != 2:
        raise ValueError("segment_embeddings must be 2D")
    if segment_embeddings.shape[0] != len(base_segments):
        raise ValueError("segment_embeddings and base_segments must have matching length")
    if smoothing_kernel_size <= 0:
        raise ValueError("smoothing_kernel_size must be positive")

    if len(base_segments) == 1:
        segment = base_segments[0]
        return [
            Segment(
                segment_id=f"{prefix}_0000",
                start_index=segment.start_index,
                end_index=segment.end_index,
                start_time_sec=segment.start_time_sec,
                end_time_sec=segment.end_time_sec,
                duration_sec=max(0.0, segment.end_time_sec - segment.start_time_sec),
            )
        ]

    normalized = torch.nn.functional.normalize(segment_embeddings, dim=-1)
    similarities = (normalized[:-1] * normalized[1:]).sum(dim=-1).cpu().numpy().astype(np.float32)
    drift = 1.0 - similarities
    if smoothing_kernel_size > 1:
        kernel = np.ones((smoothing_kernel_size,), dtype=np.float32) / float(smoothing_kernel_size)
        smoothed = np.convolve(drift, kernel, mode="same")
    else:
        smoothed = drift

    effective_thresholds = np.full_like(smoothed, float(drift_threshold), dtype=np.float32)
    if adaptive_percentile is not None:
        percentile = float(adaptive_percentile)
        if percentile < 0.0 or percentile > 100.0:
            raise ValueError("adaptive_percentile must be in [0, 100]")
        if adaptive_window_size is None:
            local_threshold = float(np.percentile(smoothed, percentile))
            effective_thresholds.fill(local_threshold)
        else:
            window_size = int(adaptive_window_size)
            if window_size <= 0:
                raise ValueError("adaptive_window_size must be positive")
            radius = window_size // 2
            local_thresholds = np.zeros_like(smoothed, dtype=np.float32)
            for index in range(len(smoothed)):
                start = max(0, index - radius)
                end = min(len(smoothed), index + radius + 1)
                local_thresholds[index] = float(np.percentile(smoothed[start:end], percentile))
            effective_thresholds = local_thresholds
        if adaptive_floor is not None:
            effective_thresholds = np.maximum(effective_thresholds, float(adaptive_floor))

    segments: list[Segment] = []
    group_start = 0
    segment_index = 0
    for boundary_index in range(len(base_segments) - 1):
        current_duration = base_segments[boundary_index].end_time_sec - base_segments[group_start].start_time_sec
        force_split = current_duration >= max_duration_sec
        boundary_hit = smoothed[boundary_index] >= effective_thresholds[boundary_index] and current_duration >= min_duration_sec
        if not force_split and not boundary_hit:
            continue
        start_segment = base_segments[group_start]
        end_segment = base_segments[boundary_index]
        segments.append(
            Segment(
                segment_id=f"{prefix}_{segment_index:04d}",
                start_index=start_segment.start_index,
                end_index=end_segment.end_index,
                start_time_sec=start_segment.start_time_sec,
                end_time_sec=end_segment.end_time_sec,
                duration_sec=max(0.0, end_segment.end_time_sec - start_segment.start_time_sec),
            )
        )
        segment_index += 1
        group_start = boundary_index + 1

    if group_start < len(base_segments):
        start_segment = base_segments[group_start]
        end_segment = base_segments[-1]
        if segments and (end_segment.end_time_sec - start_segment.start_time_sec) < min_duration_sec:
            previous = segments.pop()
            start_segment = Segment(
                segment_id=start_segment.segment_id,
                start_index=previous.start_index,
                end_index=start_segment.end_index,
                start_time_sec=previous.start_time_sec,
                end_time_sec=start_segment.end_time_sec,
                duration_sec=max(0.0, start_segment.end_time_sec - previous.start_time_sec),
            )
            segment_index -= 1
        segments.append(
            Segment(
                segment_id=f"{prefix}_{segment_index:04d}",
                start_index=start_segment.start_index,
                end_index=end_segment.end_index,
                start_time_sec=start_segment.start_time_sec,
                end_time_sec=end_segment.end_time_sec,
                duration_sec=max(0.0, end_segment.end_time_sec - start_segment.start_time_sec),
            )
        )

    return segments
