from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any


def _sanitize_prompt_text(text: str) -> str:
    text = re.sub(r"<TIME\s+([^>]+?)\s+video\s+\d+>", r"\1", text, flags=re.IGNORECASE)
    text = re.sub(r"<TIME\s+([^>]+?)>", r"\1", text, flags=re.IGNORECASE)
    text = re.sub(r"\bvideo\s+\d+\b", "video", text, flags=re.IGNORECASE)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def _normalize_choice_for_prompt(choice: Any) -> str:
    if isinstance(choice, str):
        return _sanitize_prompt_text(choice)
    if isinstance(choice, list):
        return _sanitize_prompt_text(" -> ".join(str(item) for item in choice))
    return _sanitize_prompt_text(str(choice))


def _label_prompt_choices(choices: list[str]) -> list[str]:
    return [f"{chr(ord('A') + index)}. {choice}" for index, choice in enumerate(choices)]


def _format_seconds_for_prompt(value: float) -> str:
    hours = int(value // 3600)
    minutes = int((value % 3600) // 60)
    seconds = value - (hours * 3600) - (minutes * 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"


def _build_frame_timestamp_labels(timestamps_sec: list[float]) -> list[str]:
    return [f"Frame {index + 1} timestamp: {_format_seconds_for_prompt(float(time_sec))}" for index, time_sec in enumerate(timestamps_sec)]


def _build_timestamped_mcq_prompt(
    *,
    question: str,
    labeled_options: list[str],
    clip_start_sec: float,
    clip_end_sec: float,
) -> str:
    return (
        "You are given frames sampled from one candidate clip of a long video.\n"
        f"The shown clip spans {_format_seconds_for_prompt(clip_start_sec)} to {_format_seconds_for_prompt(clip_end_sec)}.\n"
        "Each frame is preceded by its timestamp, and the frames are shown in chronological order.\n"
        "Use the shown frame timestamps to align the visual evidence with the answer intervals.\n"
        "Think briefly about which option interval overlaps the shown frames, then reply with only one letter.\n\n"
        f"Question: {_sanitize_prompt_text(question)}\n"
        "Options:\n"
        + "\n".join(labeled_options)
    )


def _sample_uniform_video_frames(
    *,
    video_path: str | Path,
    frame_budget: int,
    start_time_sec: float | None,
    end_time_sec: float | None,
) -> tuple[list[Any], list[int], list[float], float]:
    import cv2
    import numpy as np
    from PIL import Image

    started = time.perf_counter()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    native_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if native_fps <= 0.0 or total_frames <= 0:
        capture.release()
        raise RuntimeError(f"Invalid fps/frame count for video: {video_path}")

    start_frame = max(0, int(round((start_time_sec or 0.0) * native_fps)))
    end_frame = total_frames - 1 if end_time_sec is None else min(total_frames - 1, int(round(end_time_sec * native_fps)))
    if end_frame < start_frame:
        capture.release()
        raise RuntimeError(f"Invalid scoped range for video: {video_path} ({start_time_sec}, {end_time_sec})")

    target_indices = np.linspace(start_frame, end_frame, num=max(frame_budget, 1))
    raw_indices = [int(index) for index in np.round(target_indices).tolist()]
    frame_indices = sorted(dict.fromkeys(raw_indices))

    frames: list[Image.Image] = []
    timestamps_sec: list[float] = []
    for frame_index in frame_indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))
        timestamps_sec.append(frame_index / native_fps)

    capture.release()
    if not frames:
        raise RuntimeError(f"No frames sampled from video scope: {video_path}")
    return frames, frame_indices, timestamps_sec, round(time.perf_counter() - started, 3)


__all__ = [
    "_build_frame_timestamp_labels",
    "_build_timestamped_mcq_prompt",
    "_label_prompt_choices",
    "_normalize_choice_for_prompt",
    "_sample_uniform_video_frames",
    "_sanitize_prompt_text",
]
