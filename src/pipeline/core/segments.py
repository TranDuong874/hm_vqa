from __future__ import annotations

from typing import Iterable

import numpy as np
import torch

from .schema import SegmentationConfig
from .types import Segment


def build_fixed_windows(
    *,
    timestamps: np.ndarray,
    window_seconds: float,
    stride_seconds: float,
    prefix: str,
) -> list[Segment]:
    if len(timestamps) == 0:
        return []
    segments: list[Segment] = []
    video_start = float(timestamps[0])
    video_end = float(timestamps[-1])
    segment_index = 0
    start_time = video_start
    while start_time < video_end:
        end_time = min(start_time + float(window_seconds), video_end)
        start_index = int(np.searchsorted(timestamps, start_time, side="left"))
        end_index = int(np.searchsorted(timestamps, end_time, side="right") - 1)
        if end_index < start_index:
            start_time += float(stride_seconds)
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
        if end_time >= video_end:
            break
        start_time += float(stride_seconds)
    return segments


def mean_pool_segments(frame_embeddings: torch.Tensor, segments: Iterable[Segment]) -> torch.Tensor:
    pooled: list[torch.Tensor] = []
    for segment in segments:
        pooled.append(frame_embeddings[segment.start_index : segment.end_index + 1].mean(dim=0))
    if not pooled:
        width = frame_embeddings.shape[-1] if frame_embeddings.ndim == 2 else 0
        return torch.empty((0, width), dtype=torch.float32)
    return torch.nn.functional.normalize(torch.stack(pooled, dim=0), dim=-1)


def compute_adjacent_drift(segment_embeddings: torch.Tensor) -> np.ndarray:
    if segment_embeddings.ndim != 2 or segment_embeddings.shape[0] <= 1:
        return np.zeros((0,), dtype=np.float32)
    normalized = torch.nn.functional.normalize(segment_embeddings, dim=-1)
    similarities = (normalized[:-1] * normalized[1:]).sum(dim=-1).cpu().numpy().astype(np.float32)
    return 1.0 - similarities


def moving_average(values: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size <= 1 or values.size == 0:
        return values.astype(np.float32, copy=True)
    kernel = np.ones((kernel_size,), dtype=np.float32) / float(kernel_size)
    return np.convolve(values, kernel, mode="same").astype(np.float32)


def robust_local_zscores(
    values: np.ndarray,
    *,
    window_size: int,
    eps: float,
) -> np.ndarray:
    if values.size == 0:
        return np.zeros((0,), dtype=np.float32)
    radius = max(int(window_size) // 2, 0)
    scores = np.zeros_like(values, dtype=np.float32)
    for index in range(len(values)):
        start = max(0, index - radius)
        end = min(len(values), index + radius + 1)
        local = values[start:end]
        median = float(np.median(local))
        mad = float(np.median(np.abs(local - median)))
        scores[index] = float((values[index] - median) / max(mad, eps))
    return scores


def local_maxima(values: np.ndarray) -> list[int]:
    peaks: list[int] = []
    for index, value in enumerate(values):
        left = values[index - 1] if index > 0 else float("-inf")
        right = values[index + 1] if index + 1 < len(values) else float("-inf")
        if float(value) > float(left) and float(value) >= float(right):
            peaks.append(index)
    return peaks


def select_peaks_by_nms(
    *,
    peak_indices: list[int],
    peak_scores: np.ndarray,
    l2_segments: list[Segment],
    min_peak_distance_sec: float,
) -> list[int]:
    if not peak_indices:
        return []
    ordered = sorted(peak_indices, key=lambda index: float(peak_scores[index]), reverse=True)
    kept: list[int] = []
    for index in ordered:
        boundary_time = float(l2_segments[index].end_time_sec)
        if any(abs(boundary_time - float(l2_segments[kept_index].end_time_sec)) < float(min_peak_distance_sec) for kept_index in kept):
            continue
        kept.append(index)
    kept.sort()
    return kept


def _segments_from_cut_indices(l2_segments: list[Segment], cut_indices: list[int], prefix: str) -> list[Segment]:
    if not l2_segments:
        return []
    cuts = sorted({int(index) for index in cut_indices if 0 <= int(index) < len(l2_segments) - 1})
    start_window = 0
    segments: list[Segment] = []
    segment_index = 0
    for cut_index in cuts:
        first = l2_segments[start_window]
        last = l2_segments[cut_index]
        segments.append(
            Segment(
                segment_id=f"{prefix}_{segment_index:04d}",
                start_index=first.start_index,
                end_index=last.end_index,
                start_time_sec=float(first.start_time_sec),
                end_time_sec=float(last.end_time_sec),
                duration_sec=max(0.0, float(last.end_time_sec) - float(first.start_time_sec)),
            )
        )
        segment_index += 1
        start_window = cut_index + 1
    first = l2_segments[start_window]
    last = l2_segments[-1]
    segments.append(
        Segment(
            segment_id=f"{prefix}_{segment_index:04d}",
            start_index=first.start_index,
            end_index=last.end_index,
            start_time_sec=float(first.start_time_sec),
            end_time_sec=float(last.end_time_sec),
            duration_sec=max(0.0, float(last.end_time_sec) - float(first.start_time_sec)),
        )
    )
    return segments


def _merge_short_segments(
    *,
    segments: list[Segment],
    min_duration_sec: float,
    prefix: str,
) -> list[Segment]:
    if not segments:
        return []
    merged: list[Segment] = []
    pending = segments[0]
    for current in segments[1:]:
        if pending.duration_sec < float(min_duration_sec):
            pending = Segment(
                segment_id=pending.segment_id,
                start_index=pending.start_index,
                end_index=current.end_index,
                start_time_sec=pending.start_time_sec,
                end_time_sec=current.end_time_sec,
                duration_sec=max(0.0, float(current.end_time_sec) - float(pending.start_time_sec)),
            )
            continue
        merged.append(pending)
        pending = current
    if merged and pending.duration_sec < float(min_duration_sec):
        previous = merged.pop()
        pending = Segment(
            segment_id=previous.segment_id,
            start_index=previous.start_index,
            end_index=pending.end_index,
            start_time_sec=previous.start_time_sec,
            end_time_sec=pending.end_time_sec,
            duration_sec=max(0.0, float(pending.end_time_sec) - float(previous.start_time_sec)),
        )
    merged.append(pending)
    output: list[Segment] = []
    for index, segment in enumerate(merged):
        output.append(
            Segment(
                segment_id=f"{prefix}_{index:04d}",
                start_index=segment.start_index,
                end_index=segment.end_index,
                start_time_sec=segment.start_time_sec,
                end_time_sec=segment.end_time_sec,
                duration_sec=segment.duration_sec,
            )
        )
    return output


def _split_long_segments(
    *,
    segments: list[Segment],
    l2_segments: list[Segment],
    max_duration_sec: float,
    prefix: str,
) -> list[Segment]:
    if not segments:
        return []
    split_segments: list[Segment] = []
    segment_index = 0
    for segment in segments:
        if segment.duration_sec <= float(max_duration_sec):
            split_segments.append(
                Segment(
                    segment_id=f"{prefix}_{segment_index:04d}",
                    start_index=segment.start_index,
                    end_index=segment.end_index,
                    start_time_sec=segment.start_time_sec,
                    end_time_sec=segment.end_time_sec,
                    duration_sec=segment.duration_sec,
                )
            )
            segment_index += 1
            continue

        l2_start = next(index for index, item in enumerate(l2_segments) if item.start_index == segment.start_index)
        l2_end = next(index for index, item in enumerate(l2_segments) if item.end_index == segment.end_index)
        current_start = l2_start
        while current_start <= l2_end:
            current_end = current_start
            while current_end < l2_end:
                first = l2_segments[current_start]
                last = l2_segments[current_end + 1]
                duration = float(last.end_time_sec) - float(first.start_time_sec)
                if duration > float(max_duration_sec):
                    break
                current_end += 1
            first = l2_segments[current_start]
            last = l2_segments[current_end]
            split_segments.append(
                Segment(
                    segment_id=f"{prefix}_{segment_index:04d}",
                    start_index=int(first.start_index),
                    end_index=int(last.end_index),
                    start_time_sec=float(first.start_time_sec),
                    end_time_sec=float(last.end_time_sec),
                    duration_sec=max(0.0, float(last.end_time_sec) - float(first.start_time_sec)),
                )
            )
            segment_index += 1
            current_start = current_end + 1
    return split_segments


def build_adaptive_layer3_segments(
    *,
    l2_segments: list[Segment],
    l2_embeddings: torch.Tensor,
    config: SegmentationConfig,
    prefix: str = "l3",
) -> tuple[list[Segment], dict[str, list[float] | list[int]]]:
    if not l2_segments:
        return [], {"drift": [], "smoothed_drift": [], "drift_z": [], "kept_peak_indices": []}
    if len(l2_segments) == 1:
        only = l2_segments[0]
        segment = Segment(
            segment_id=f"{prefix}_0000",
            start_index=only.start_index,
            end_index=only.end_index,
            start_time_sec=only.start_time_sec,
            end_time_sec=only.end_time_sec,
            duration_sec=only.duration_sec,
        )
        return [segment], {"drift": [], "smoothed_drift": [], "drift_z": [], "kept_peak_indices": []}

    drift = compute_adjacent_drift(l2_embeddings)
    smoothed = moving_average(drift, int(config.drift_smoothing_kernel))
    zscores = robust_local_zscores(
        smoothed,
        window_size=int(config.adaptive_local_window),
        eps=float(config.mad_epsilon),
    )
    peak_candidates = [
        index
        for index in local_maxima(smoothed)
        if float(zscores[index]) >= float(config.min_peak_z)
    ]
    kept_peaks = select_peaks_by_nms(
        peak_indices=peak_candidates,
        peak_scores=zscores,
        l2_segments=l2_segments,
        min_peak_distance_sec=float(config.min_peak_distance_sec),
    )
    coarse_segments = _segments_from_cut_indices(l2_segments, kept_peaks, prefix=prefix)
    coarse_segments = _merge_short_segments(
        segments=coarse_segments,
        min_duration_sec=float(config.min_segment_duration_sec),
        prefix=prefix,
    )
    coarse_segments = _split_long_segments(
        segments=coarse_segments,
        l2_segments=l2_segments,
        max_duration_sec=float(config.max_segment_duration_sec),
        prefix=prefix,
    )
    diagnostics = {
        "drift": drift.astype(float).tolist(),
        "smoothed_drift": smoothed.astype(float).tolist(),
        "drift_z": zscores.astype(float).tolist(),
        "kept_peak_indices": [int(index) for index in kept_peaks],
    }
    return coarse_segments, diagnostics
