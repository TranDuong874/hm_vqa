from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class SegmentationConfig:
    l2_window_seconds: float = 5.0
    l2_window_stride_seconds: float = 5.0
    drift_smoothing_kernel: int = 3
    adaptive_local_window: int = 11
    min_peak_z: float = 1.5
    min_peak_distance_sec: float = 10.0
    min_segment_duration_sec: float = 15.0
    max_segment_duration_sec: float = 60.0
    mad_epsilon: float = 1e-6


@dataclass(slots=True)
class RetrievalConfig:
    selection_mode: str = "relative_threshold"
    layer3_top_k: int = 3
    layer2_top_k: int = 3
    layer3_relative_alpha: float = 0.8
    layer3_max_keep: int = 10
    layer2_relative_alpha: float = 0.7
    layer2_max_keep: int = 20
    layer3_coverage_threshold: float = 0.6
    layer1_top_k: int = 1
    openclip_batch_size: int = 32
    device: str = "cuda"
    query_mode: str = "target_only"


@dataclass(slots=True)
class PipelineConfig:
    repo_root: Path
    tasks: tuple[str, ...]
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    output_root: Path = Path("results/pipeline")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["repo_root"] = str(self.repo_root)
        payload["output_root"] = str(self.output_root)
        return payload
