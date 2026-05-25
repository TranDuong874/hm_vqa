from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .types import Segment


def robust_stats(
    values: np.ndarray,
    *,
    eps: float = 1e-6,
    min_scale: float = 0.0,
) -> tuple[float, float]:
    if values.size == 0:
        return 0.0, float(max(eps, min_scale))
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    scale = float(max((1.4826 * mad) + eps, min_scale))
    return med, scale


def moving_average(values: np.ndarray, *, kernel_size: int = 3) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0 or kernel_size <= 1:
        return values.copy()
    half = int(kernel_size) // 2
    smoothed = np.zeros_like(values, dtype=np.float32)
    for index in range(len(values)):
        start = max(0, index - half)
        end = min(len(values), index + half + 1)
        smoothed[index] = float(np.mean(values[start:end]))
    return smoothed


def compute_semantic_drift_ema(
    frame_embeddings: torch.Tensor,
    *,
    rho: float = 0.05,
) -> np.ndarray:
    if frame_embeddings.ndim != 2:
        raise ValueError("frame_embeddings must be 2D")
    if frame_embeddings.shape[0] == 0:
        return np.empty((0,), dtype=np.float32)
    if rho <= 0.0 or rho > 1.0:
        raise ValueError("rho must be in (0, 1]")

    normalized = torch.nn.functional.normalize(frame_embeddings.float(), dim=-1).cpu().numpy().astype(np.float32)
    drift = np.zeros((normalized.shape[0],), dtype=np.float32)
    context = normalized[0].copy()
    context /= max(float(np.linalg.norm(context)), 1e-8)

    for index in range(1, normalized.shape[0]):
        current = normalized[index]
        current /= max(float(np.linalg.norm(current)), 1e-8)
        drift[index] = float(
            1.0 - (float(np.dot(current, context)) / (max(float(np.linalg.norm(current) * np.linalg.norm(context)), 1e-8)))
        )
        context = ((1.0 - float(rho)) * context) + (float(rho) * current)
        context /= max(float(np.linalg.norm(context)), 1e-8)

    return drift


def detect_duration_constrained_fused_peaks(
    *,
    semantic_drift: np.ndarray,
    motion_energy: np.ndarray,
    trailing_window: int = 30,
    peak_neighborhood: int = 2,
    smooth_kernel_size: int = 3,
    min_len: int = 15,
    max_len: int = 60,
    w_sem: float = 0.7,
    w_mot: float = 0.3,
    k_strong: float = 2.5,
    k_weak: float = 1.0,
    robust_min_scale: float = 0.01,
    zscore_clip: float = 6.0,
) -> dict[str, Any]:
    sem = np.asarray(semantic_drift, dtype=np.float32)
    mot = np.asarray(motion_energy, dtype=np.float32)
    if sem.ndim != 1 or mot.ndim != 1:
        raise ValueError("semantic_drift and motion_energy must be 1D")
    if len(sem) != len(mot):
        raise ValueError("semantic_drift and motion_energy must have matching length")
    if trailing_window <= 0:
        raise ValueError("trailing_window must be positive")
    if peak_neighborhood < 0:
        raise ValueError("peak_neighborhood must be non-negative")
    if min_len <= 0 or max_len <= 0 or min_len > max_len:
        raise ValueError("min_len/max_len must be positive and min_len <= max_len")
    if robust_min_scale < 0.0:
        raise ValueError("robust_min_scale must be non-negative")
    if zscore_clip <= 0.0:
        raise ValueError("zscore_clip must be positive")

    length = len(sem)
    z_sem = np.zeros((length,), dtype=np.float32)
    z_mot = np.zeros((length,), dtype=np.float32)

    for index in range(trailing_window, length):
        sem_med, sem_scale = robust_stats(
            sem[index - trailing_window : index],
            min_scale=robust_min_scale,
        )
        mot_med, mot_scale = robust_stats(
            mot[index - trailing_window : index],
            min_scale=robust_min_scale,
        )
        z_sem[index] = float(np.clip((sem[index] - sem_med) / sem_scale, -zscore_clip, zscore_clip))
        z_mot[index] = float(np.clip((mot[index] - mot_med) / mot_scale, -zscore_clip, zscore_clip))

    fused = (float(w_sem) * z_sem) + (float(w_mot) * z_mot)
    fused_smooth = moving_average(fused, kernel_size=smooth_kernel_size)

    fused_z = np.zeros((length,), dtype=np.float32)
    for index in range(trailing_window, length):
        fused_med, fused_scale = robust_stats(
            fused_smooth[index - trailing_window : index],
            min_scale=robust_min_scale,
        )
        fused_z[index] = float(np.clip((fused_smooth[index] - fused_med) / fused_scale, -zscore_clip, zscore_clip))

    boundaries: list[dict[str, Any]] = []
    segment_start = 0

    while segment_start + min_len < length:
        remaining = length - segment_start
        if remaining <= max_len:
            break
        search_start = max(segment_start + min_len, trailing_window, peak_neighborhood)
        search_end = min(segment_start + max_len, length - peak_neighborhood - 1)
        if search_start > search_end:
            break

        best_peak_index: int | None = None
        best_peak_score = float("-inf")
        best_strong_peak_index: int | None = None
        best_strong_peak_score = float("-inf")

        for index in range(search_start, search_end + 1):
            local_window = fused_smooth[index - peak_neighborhood : index + peak_neighborhood + 1]
            is_peak = bool(fused_smooth[index] >= float(np.max(local_window)))
            if not is_peak:
                continue

            score = float(fused_z[index])
            if score >= float(k_weak) and score > best_peak_score:
                best_peak_index = index
                best_peak_score = score
            if score >= float(k_strong) and score > best_strong_peak_score:
                best_strong_peak_index = index
                best_strong_peak_score = score

        cut_index = (
            best_strong_peak_index
            if best_strong_peak_index is not None
            else best_peak_index
            if best_peak_index is not None
            else search_end
        )
        boundaries.append(
            {
                "frame_idx": int(cut_index),
                "score": float(fused_smooth[cut_index]),
                "zscore": float(fused_z[cut_index]),
                "segment_start": int(segment_start),
                "segment_end": int(cut_index),
                "segment_len": int(cut_index - segment_start + 1),
                "forced_cut": bool(best_peak_index is None),
            }
        )
        segment_start = cut_index + 1

    return {
        "semantic_drift": sem,
        "motion_score": mot,
        "z_sem": z_sem,
        "z_mot": z_mot,
        "fused_score": fused,
        "fused_score_smooth": fused_smooth,
        "fused_z": fused_z,
        "boundaries": boundaries,
    }


def segment_fused_adaptive_peaks(
    *,
    timestamps: np.ndarray,
    frame_embeddings: torch.Tensor,
    motion_energy: np.ndarray,
    rho: float = 0.05,
    trailing_window: int = 30,
    peak_neighborhood: int = 2,
    smooth_kernel_size: int = 3,
    min_duration_sec: float = 15.0,
    max_duration_sec: float = 60.0,
    w_sem: float = 0.7,
    w_mot: float = 0.3,
    k_strong: float = 2.5,
    k_weak: float = 1.0,
    robust_min_scale: float = 0.01,
    zscore_clip: float = 6.0,
    prefix: str = "fused_peak",
) -> dict[str, Any]:
    timestamps = np.asarray(timestamps, dtype=np.float32)
    if timestamps.ndim != 1:
        raise ValueError("timestamps must be 1D")
    if len(timestamps) == 0:
        return {
            "segments": [],
            "semantic_drift": np.empty((0,), dtype=np.float32),
            "motion_score": np.empty((0,), dtype=np.float32),
            "z_sem": np.empty((0,), dtype=np.float32),
            "z_mot": np.empty((0,), dtype=np.float32),
            "fused_score": np.empty((0,), dtype=np.float32),
            "fused_score_smooth": np.empty((0,), dtype=np.float32),
            "fused_z": np.empty((0,), dtype=np.float32),
            "boundaries": [],
        }
    if frame_embeddings.ndim != 2:
        raise ValueError("frame_embeddings must be 2D")
    if frame_embeddings.shape[0] != len(timestamps):
        raise ValueError("frame_embeddings and timestamps must have matching length")
    if len(motion_energy) != len(timestamps):
        raise ValueError("motion_energy and timestamps must have matching length")

    semantic_drift = compute_semantic_drift_ema(frame_embeddings, rho=rho)
    detection = detect_duration_constrained_fused_peaks(
        semantic_drift=semantic_drift,
        motion_energy=np.asarray(motion_energy, dtype=np.float32),
        trailing_window=trailing_window,
        peak_neighborhood=peak_neighborhood,
        smooth_kernel_size=smooth_kernel_size,
        min_len=max(int(round(min_duration_sec)), 1),
        max_len=max(int(round(max_duration_sec)), 1),
        w_sem=w_sem,
        w_mot=w_mot,
        k_strong=k_strong,
        k_weak=k_weak,
        robust_min_scale=robust_min_scale,
        zscore_clip=zscore_clip,
    )

    segments: list[Segment] = []
    segment_start = 0
    segment_index = 0
    for boundary in detection["boundaries"]:
        end_index = int(boundary["frame_idx"])
        if end_index < segment_start:
            continue
        segments.append(
            Segment(
                segment_id=f"{prefix}_{segment_index:04d}",
                start_index=int(segment_start),
                end_index=end_index,
                start_time_sec=float(timestamps[segment_start]),
                end_time_sec=float(timestamps[end_index]),
                duration_sec=max(0.0, float(timestamps[end_index] - timestamps[segment_start])),
            )
        )
        segment_index += 1
        segment_start = end_index + 1

    if segment_start <= len(timestamps) - 1:
        segments.append(
            Segment(
                segment_id=f"{prefix}_{segment_index:04d}",
                start_index=int(segment_start),
                end_index=len(timestamps) - 1,
                start_time_sec=float(timestamps[segment_start]),
                end_time_sec=float(timestamps[-1]),
                duration_sec=max(0.0, float(timestamps[-1] - timestamps[segment_start])),
            )
        )

    return {
        **detection,
        "segments": segments,
    }
