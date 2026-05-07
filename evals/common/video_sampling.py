from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def _resize_frame_if_needed(frame: Image.Image, max_size: int | None) -> Image.Image:
    if max_size is None or max_size <= 0:
        return frame
    resized = frame.copy()
    resized.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return resized


def sample_uniform_video_frames(
    *,
    video_path: str | Path,
    frame_budget: int,
    start_time_sec: float = 0.0,
    end_time_sec: float | None = None,
    image_max_size: int | None = None,
) -> tuple[list[Image.Image], list[int], list[float], float]:
    """Sample a fixed number of original-video frames uniformly over a time range.

    The return shape is shared by existing eval runners: frames, native frame
    indices, timestamps in seconds, and native FPS.
    """
    if frame_budget <= 0:
        return [], [], [], 0.0

    path = Path(video_path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")

    native_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if native_fps <= 0.0 or total_frames <= 0:
        capture.release()
        raise RuntimeError(f"Invalid fps/frame count for video: {path}")

    duration_sec = total_frames / native_fps
    start = max(float(start_time_sec), 0.0)
    end = float(end_time_sec) if end_time_sec is not None else duration_sec
    end = min(max(end, start), duration_sec)

    if frame_budget == 1 or end <= start:
        timestamps = [min(max((start + end) / 2.0, 0.0), duration_sec)]
    else:
        timestamps = np.linspace(start, end, num=frame_budget, endpoint=True).astype(float).tolist()

    frames: list[Image.Image] = []
    frame_indices: list[int] = []
    actual_times: list[float] = []
    for timestamp in timestamps:
        native_index = int(round(float(timestamp) * native_fps))
        native_index = max(0, min(native_index, total_frames - 1))
        capture.set(cv2.CAP_PROP_POS_FRAMES, native_index)
        ok, frame = capture.read()
        if not ok:
            capture.set(cv2.CAP_PROP_POS_MSEC, float(timestamp) * 1000.0)
            ok, frame = capture.read()
        if not ok:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(_resize_frame_if_needed(Image.fromarray(frame_rgb), image_max_size))
        frame_indices.append(native_index)
        actual_times.append(native_index / native_fps)

    capture.release()
    if not frames:
        raise RuntimeError(f"No frames sampled from video: {path}")
    return frames, frame_indices, actual_times, native_fps


def build_frame_timestamp_labels(timestamps_sec: list[float]) -> list[str]:
    return [f"Frame {index + 1} at {float(timestamp):.2f}s" for index, timestamp in enumerate(timestamps_sec)]
