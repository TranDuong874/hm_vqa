from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


@dataclass(slots=True)
class Segment:
    segment_id: str
    level: str
    start_index: int
    end_index: int
    start_time_sec: float
    end_time_sec: float
    duration_sec: float
    peak_score: float
    mean_score: float


def sample_video(video_path: Path, fps: float) -> tuple[list[Image.Image], list[np.ndarray], np.ndarray, float]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    native_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if native_fps <= 0.0 or total_frames <= 0:
        capture.release()
        raise RuntimeError(f"Invalid fps/frame count for video: {video_path}")

    step = max(int(round(native_fps / fps)), 1)
    pil_frames: list[Image.Image] = []
    bgr_frames: list[np.ndarray] = []
    timestamps: list[float] = []

    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % step == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frames.append(Image.fromarray(frame_rgb))
            bgr_frames.append(frame.copy())
            timestamps.append(frame_index / native_fps)
        frame_index += 1

    capture.release()
    if not pil_frames:
        raise RuntimeError(f"No frames sampled from video: {video_path}")
    return pil_frames, bgr_frames, np.asarray(timestamps, dtype=np.float32), native_fps


def minmax_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    low = float(values.min())
    high = float(values.max())
    if math.isclose(low, high):
        return np.zeros_like(values, dtype=np.float32)
    return (values - low) / (high - low)


def cosine_drift(embeddings: torch.Tensor) -> np.ndarray:
    if embeddings.shape[0] <= 1:
        return np.zeros((embeddings.shape[0],), dtype=np.float32)
    prev = embeddings[:-1]
    curr = embeddings[1:]
    similarity = torch.sum(prev * curr, dim=-1).clamp(-1.0, 1.0)
    drift = 1.0 - similarity
    drift = torch.cat([torch.zeros(1, dtype=drift.dtype), drift], dim=0)
    return drift.cpu().numpy().astype(np.float32)


def frame_energy_diff(frames_bgr: list[np.ndarray]) -> np.ndarray:
    if len(frames_bgr) <= 1:
        return np.zeros((len(frames_bgr),), dtype=np.float32)
    energies = [0.0]
    prev_gray = cv2.cvtColor(frames_bgr[0], cv2.COLOR_BGR2GRAY).astype(np.float32)
    for frame in frames_bgr[1:]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        diff = np.mean(np.abs(gray - prev_gray)) / 255.0
        energies.append(float(diff))
        prev_gray = gray
    return np.asarray(energies, dtype=np.float32)


def segment_by_threshold(
    *,
    timestamps: np.ndarray,
    score: np.ndarray,
    threshold: float,
    min_seconds: float,
    max_seconds: float | None,
    level: str,
) -> list[Segment]:
    if len(timestamps) == 0:
        return []

    boundaries = [0]
    last_boundary = 0
    for index in range(1, len(timestamps)):
        duration = float(timestamps[index] - timestamps[last_boundary])
        trigger = float(score[index]) >= float(threshold) and duration >= float(min_seconds)
        too_long = max_seconds is not None and duration >= float(max_seconds)
        if trigger or too_long:
            boundaries.append(index)
            last_boundary = index
    if boundaries[-1] != len(timestamps) - 1:
        boundaries.append(len(timestamps) - 1)

    segments: list[Segment] = []
    for segment_index in range(len(boundaries) - 1):
        start_idx = int(boundaries[segment_index])
        end_idx = int(boundaries[segment_index + 1])
        if end_idx <= start_idx:
            continue
        start_time = float(timestamps[start_idx])
        end_time = float(timestamps[end_idx])
        local_score = score[start_idx : end_idx + 1]
        segments.append(
            Segment(
                segment_id=f"{level}_{segment_index:04d}",
                level=level,
                start_index=start_idx,
                end_index=end_idx,
                start_time_sec=start_time,
                end_time_sec=end_time,
                duration_sec=max(0.0, end_time - start_time),
                peak_score=float(np.max(local_score)),
                mean_score=float(np.mean(local_score)),
            )
        )
    return segments


def export_segment_clips(
    *,
    segments: list[Segment],
    frames_bgr: list[np.ndarray],
    output_dir: Path,
    export_fps: float,
    max_segments: int | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    export_list = segments if max_segments is None else segments[:max_segments]
    for segment in export_list:
        clip_path = output_dir / f"{segment.segment_id}.mp4"
        clip_frames = frames_bgr[segment.start_index : segment.end_index + 1]
        if not clip_frames:
            continue
        height, width = clip_frames[0].shape[:2]
        writer = cv2.VideoWriter(
            str(clip_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(export_fps),
            (width, height),
        )
        for frame in clip_frames:
            writer.write(frame)
        writer.release()


def plot_signals(
    *,
    timestamps: np.ndarray,
    energy_signal: np.ndarray | None,
    openclip_signal: np.ndarray | None,
    dino_signal: np.ndarray | None,
    combined_score: np.ndarray,
    low_segments: list[Segment],
    high_segments: list[Segment],
    low_threshold: float,
    high_threshold: float,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(18, 12), sharex=True)

    if energy_signal is not None:
        axes[0].plot(timestamps, energy_signal, label="energy_diff", color="tab:orange")
        axes[0].legend(loc="upper right")
    if openclip_signal is not None:
        axes[1].plot(timestamps, openclip_signal, label="openclip_drift", color="tab:blue")
        axes[1].legend(loc="upper right")
    if dino_signal is not None:
        axes[2].plot(timestamps, dino_signal, label="dino_drift", color="tab:green")
        axes[2].legend(loc="upper right")

    axes[3].plot(timestamps, combined_score, label="combined_score", color="tab:red")
    axes[3].axhline(low_threshold, color="tab:purple", linestyle="--", alpha=0.7, label="low_threshold")
    axes[3].axhline(high_threshold, color="black", linestyle="--", alpha=0.7, label="high_threshold")
    for segment in low_segments:
        axes[3].axvspan(segment.start_time_sec, segment.end_time_sec, color="tab:blue", alpha=0.08)
    for segment in high_segments:
        axes[3].axvspan(segment.start_time_sec, segment.end_time_sec, color="tab:red", alpha=0.10)
    axes[3].legend(loc="upper right")

    axes[0].set_title("Energy difference")
    axes[1].set_title("OpenCLIP drift")
    axes[2].set_title("DINO drift")
    axes[3].set_title("Combined score + segments")
    axes[3].set_xlabel("Time (sec)")
    for axis in axes:
        axis.grid(True, alpha=0.25)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
