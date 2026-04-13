from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


@dataclass(slots=True)
class VideoSamplingInfo:
    native_fps: float
    total_frames: int
    sampled_step: int
    sampled_count: int
    duration_sec: float


def _resize_frame_if_needed(frame: Image.Image, max_size: int | None) -> Image.Image:
    if max_size is None or max_size <= 0:
        return frame
    resized = frame.copy()
    resized.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return resized

def _open_video(video_path: Path) -> tuple[cv2.VideoCapture, float, int]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    native_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if native_fps <= 0.0 or total_frames <= 0:
        capture.release()
        raise RuntimeError(f"Invalid fps/frame count for video: {video_path}")
    return capture, native_fps, total_frames


def _frame_step(native_fps: float, sample_fps: float) -> int:
    if sample_fps <= 0.0:
        raise ValueError("sample_fps must be positive")
    return max(int(round(native_fps / sample_fps)), 1)


def probe_video_sampling(video_path: Path, sample_fps: float) -> VideoSamplingInfo:
    capture, native_fps, total_frames = _open_video(video_path)
    capture.release()
    step = _frame_step(native_fps, sample_fps)
    sampled_count = int(math.ceil(total_frames / step))
    duration_sec = float(total_frames / native_fps)
    return VideoSamplingInfo(
        native_fps=native_fps,
        total_frames=total_frames,
        sampled_step=step,
        sampled_count=sampled_count,
        duration_sec=duration_sec,
    )


def sample_video(
    video_path: Path,
    sample_fps: float,
    *,
    image_max_size: int | None = None,
) -> tuple[list[Image.Image], np.ndarray, float]:
    capture, native_fps, _ = _open_video(video_path)
    step = _frame_step(native_fps, sample_fps)
    frames: list[Image.Image] = []
    timestamps: list[float] = []

    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % step == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(_resize_frame_if_needed(Image.fromarray(frame_rgb), image_max_size))
            timestamps.append(frame_index / native_fps)
        frame_index += 1

    capture.release()
    if not frames:
        raise RuntimeError(f"No frames sampled from video: {video_path}")
    return frames, np.asarray(timestamps, dtype=np.float32), native_fps


def parse_timecode(value: str) -> float:
    hours, minutes, seconds = value.split(":")
    return int(hours) * 3600.0 + int(minutes) * 60.0 + float(seconds)


def sample_video_segment(
    video_path: Path,
    fps: float,
    *,
    start_time_sec: float,
    end_time_sec: float,
    image_max_size: int | None = None,
) -> tuple[list[Image.Image], np.ndarray, float]:
    capture, native_fps, total_frames = _open_video(video_path)

    start_frame = max(int(math.floor(start_time_sec * native_fps)), 0)
    end_frame = min(int(math.ceil(end_time_sec * native_fps)), total_frames - 1)
    step = _frame_step(native_fps, fps)

    frames: list[Image.Image] = []
    timestamps: list[float] = []

    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_index = start_frame
    while frame_index <= end_frame:
        ok, frame = capture.read()
        if not ok:
            break
        relative_index = frame_index - start_frame
        if relative_index % step == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(_resize_frame_if_needed(Image.fromarray(frame_rgb), image_max_size))
            timestamps.append(frame_index / native_fps)
        frame_index += 1

    capture.release()
    if not frames:
        raise RuntimeError(f"No frames sampled from segment: {video_path} {start_time_sec:.3f}-{end_time_sec:.3f}")
    return frames, np.asarray(timestamps, dtype=np.float32), native_fps


def sample_video_selected_indices(
    video_path: Path,
    fps: float,
    *,
    target_indices: list[int],
    image_max_size: int | None = None,
) -> tuple[list[Image.Image], list[float], float]:
    if not target_indices:
        return [], [], 0.0

    wanted = sorted(set(int(index) for index in target_indices))
    wanted_set = set(wanted)
    captured: dict[int, Image.Image] = {}
    times: dict[int, float] = {}

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    native_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if native_fps <= 0.0 or total_frames <= 0:
        capture.release()
        raise RuntimeError(f"Invalid fps/frame count for video: {video_path}")

    step = _frame_step(native_fps, fps)
    frame_index = 0
    sampled_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % step == 0:
            if sampled_index in wanted_set:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                captured[sampled_index] = _resize_frame_if_needed(Image.fromarray(frame_rgb), image_max_size)
                times[sampled_index] = frame_index / native_fps
                if len(captured) == len(wanted):
                    break
            sampled_index += 1
        frame_index += 1

    capture.release()
    missing = [index for index in wanted if index not in captured]
    if missing:
        raise RuntimeError(f"Failed to sample requested frame indices from {video_path}: {missing[:8]}")

    ordered_frames = [captured[index] for index in target_indices]
    ordered_times = [float(times[index]) for index in target_indices]
    return ordered_frames, ordered_times, native_fps


def compute_motion_energy_from_images(
    frames: list[Image.Image],
    *,
    resize_width: int = 64,
    resize_height: int = 64,
) -> np.ndarray:
    if not frames:
        return np.empty((0,), dtype=np.float32)

    energies = np.zeros((len(frames),), dtype=np.float32)
    previous_gray: np.ndarray | None = None
    for index, frame in enumerate(frames):
        gray = cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (resize_width, resize_height), interpolation=cv2.INTER_AREA)
        if previous_gray is not None:
            diff = np.abs(gray.astype(np.float32) - previous_gray.astype(np.float32))
            energies[index] = float(diff.mean() / 255.0)
        previous_gray = gray
    return energies


def compute_motion_energy_for_frame_indices(
    video_path: Path,
    *,
    target_frame_indices: list[int],
    resize_width: int = 64,
    resize_height: int = 64,
) -> np.ndarray:
    if not target_frame_indices:
        return np.empty((0,), dtype=np.float32)

    ordered = [int(index) for index in target_frame_indices]
    unique_sorted = sorted(set(ordered))
    capture, _, total_frames = _open_video(video_path)
    energies_by_frame: dict[int, float] = {}
    previous_gray: np.ndarray | None = None

    target_pointer = 0
    next_target = unique_sorted[target_pointer]
    frame_index = 0
    while frame_index < total_frames and target_pointer < len(unique_sorted):
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index == next_target:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (resize_width, resize_height), interpolation=cv2.INTER_AREA)
            if previous_gray is None:
                energies_by_frame[next_target] = 0.0
            else:
                diff = np.abs(gray.astype(np.float32) - previous_gray.astype(np.float32))
                energies_by_frame[next_target] = float(diff.mean() / 255.0)
            previous_gray = gray
            target_pointer += 1
            if target_pointer < len(unique_sorted):
                next_target = unique_sorted[target_pointer]
        frame_index += 1

    capture.release()
    missing = [index for index in unique_sorted if index not in energies_by_frame]
    if missing:
        raise RuntimeError(f"Failed to compute motion energy for indices from {video_path}: {missing[:8]}")

    return np.asarray([float(energies_by_frame[index]) for index in ordered], dtype=np.float32)
