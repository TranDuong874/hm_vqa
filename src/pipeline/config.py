from __future__ import annotations

from pathlib import Path

from .core.schema import PipelineConfig, RetrievalConfig, SegmentationConfig


PIPELINE_CONFIG = PipelineConfig(
    repo_root=Path(__file__).resolve().parents[2],
    tasks=(
        "fine_grained_action_localization",
        "recipe_step_localization",
    ),
    segmentation=SegmentationConfig(
        l2_window_seconds=5.0,
        l2_window_stride_seconds=5.0,
        drift_smoothing_kernel=3,
        adaptive_local_window=11,
        min_peak_z=1.5,
        min_peak_distance_sec=10.0,
        min_segment_duration_sec=15.0,
        max_segment_duration_sec=60.0,
    ),
    retrieval=RetrievalConfig(
        selection_mode="relative_threshold",
        layer3_top_k=3,
        layer2_top_k=3,
        layer3_relative_alpha=0.8,
        layer3_max_keep=10,
        layer2_relative_alpha=0.7,
        layer2_max_keep=20,
        layer3_coverage_threshold=0.6,
        layer1_top_k=1,
        openclip_batch_size=32,
        device="cuda",
        query_mode="target_only",
    ),
    output_root=Path("results/pipeline"),
)
