from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from segmentation import probe_video_sampling, sample_video, sample_video_selected_indices

from .types import FrameHit, SampledVideo


def load_video_frames(
    video_path: str | Path,
    sample_fps: float,
    *,
    image_max_size: int | None = None,
) -> SampledVideo:
    path = Path(video_path)
    frames, timestamps, native_fps = sample_video(path, sample_fps, image_max_size=image_max_size)
    return SampledVideo(
        video_path=path,
        frames=frames,
        timestamps=timestamps,
        native_fps=native_fps,
    )


def load_selected_video_frames(
    video_path: str | Path,
    *,
    sample_fps: float,
    target_indices: list[int],
    image_max_size: int | None = None,
) -> tuple[list[Image.Image], list[FrameHit], float]:
    path = Path(video_path)
    frames, timestamps, native_fps = sample_video_selected_indices(
        path,
        sample_fps,
        target_indices=target_indices,
        image_max_size=image_max_size,
    )
    hits = [
        FrameHit(frame_index=int(index), time_sec=float(timestamp), score=0.0)
        for index, timestamp in zip(target_indices, timestamps)
    ]
    return frames, hits, native_fps


def select_uniform_frames(
    *,
    frames: list[Image.Image],
    timestamps: np.ndarray,
    max_frames: int,
) -> tuple[list[Image.Image], list[FrameHit]]:
    if len(frames) <= max_frames:
        indices = list(range(len(frames)))
    else:
        indices = torch.linspace(0, len(frames) - 1, max_frames).round().long().tolist()
    selected_frames = [frames[index] for index in indices]
    hits = [FrameHit(frame_index=int(index), time_sec=float(timestamps[index]), score=0.0) for index in indices]
    return selected_frames, hits


def select_uniform_video_frames(
    *,
    video_path: str | Path,
    sample_fps: float,
    max_frames: int,
    image_max_size: int | None = None,
) -> tuple[list[Image.Image], list[FrameHit], dict[str, float | int]]:
    sampling = probe_video_sampling(Path(video_path), sample_fps)
    if sampling.sampled_count <= max_frames:
        indices = list(range(sampling.sampled_count))
    else:
        indices = torch.linspace(0, sampling.sampled_count - 1, max_frames).round().long().tolist()
    frames, hits, _ = load_selected_video_frames(
        video_path,
        sample_fps=sample_fps,
        target_indices=indices,
        image_max_size=image_max_size,
    )
    return frames, hits, {
        "native_fps": float(sampling.native_fps),
        "duration_sec": float(sampling.duration_sec),
        "sampled_count": int(sampling.sampled_count),
    }


def export_frames(
    *,
    frames: list[Image.Image],
    hits: list[FrameHit],
    output_dir: str | Path,
) -> None:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    for index, (frame, frame_hit) in enumerate(zip(frames, hits)):
        time_tag = f"{frame_hit.time_sec:.2f}s".replace(".", "_")
        frame.save(directory / f"{index:02d}_{time_tag}.png")
