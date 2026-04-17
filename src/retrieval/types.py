from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from segmentation import Segment


@dataclass(slots=True)
class PipelineConfig:
    sample_fps: float = 2.0
    window_seconds: float = 5.0
    window_stride_seconds: float = 2.5
    layer2_pooling: str = "mean"
    top_windows: int = 5
    max_evidence_frames: int = 8
    openclip_batch_size: int = 16
    image_max_size: int | None = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(slots=True)
class SampledVideo:
    video_path: Path
    frames: list[Image.Image]
    timestamps: np.ndarray
    native_fps: float


@dataclass(slots=True)
class SegmentHit:
    segment_id: str
    score: float
    start_index: int
    end_index: int
    start_time_sec: float
    end_time_sec: float


@dataclass(slots=True)
class FrameHit:
    frame_index: int
    time_sec: float
    score: float


@dataclass(slots=True)
class VideoIndex:
    sampled_video: SampledVideo
    frame_embeddings: torch.Tensor
    window_segments: list[Segment]
    window_embeddings: torch.Tensor


@dataclass(slots=True)
class EvidencePackage:
    question: str
    options: list[str]
    window_hits: list[SegmentHit]
    frame_hits: list[FrameHit]
    evidence_frames: list[Image.Image]
