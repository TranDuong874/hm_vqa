from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class MemoryPolicy:
    name: str
    sample_fps: float = 1.0
    image_max_size: int | None = 336
    frame_encoder: str = "openclip"
    frame_encoder_model: str = "ViT-L-14"
    frame_encoder_pretrained: str = "laion2b_s32b_b82k"
    normalize_embeddings: bool = True
    l3_segmentation: str = "fixed"
    l3_window_seconds: float = 60.0
    l3_stride_seconds: float = 60.0
    l3_pooling: str = "mean"
    l2_segmentation: str = "fixed"
    l2_window_seconds: float = 5.0
    l2_stride_seconds: float = 5.0
    l2_local_min_duration_sec: float = 3.0
    l2_local_max_duration_sec: float = 12.0
    l2_local_fast_kernel_size: int = 1
    l2_local_slow_kernel_size: int = 9
    l2_local_peak_percentile: float = 65.0
    l2_pooling: str = "mean"
    l2_video_encoder: str | None = None
    l2_video_encoder_model: str | None = None
    l2_video_encoder_max_frames: int = 16


@dataclass(slots=True)
class RetrievalPolicy:
    method: str
    name: str = "custom"
    requires_memory: str | None = None
    vector_backend: str = "torch"
    sample_fps: float = 1.0
    feature_eval_fps: float | None = None
    max_frames: int = 16
    image_max_size: int | None = 336
    include_subtitles: bool = False
    top_l3_segments: int = 10
    l3_rerank_keep: int = 5
    l3_rerank_evidence_source: str = "reranked_l3"
    l3_segmentation: str = "fixed"
    l3_window_seconds: float = 60.0
    l3_stride_seconds: float = 60.0
    top_l2_segments: int = 10
    l2_window_seconds: float = 5.0
    l2_stride_seconds: float = 5.0
    l2_segmentation: str = "fixed"
    l2_local_min_duration_sec: float = 3.0
    l2_local_max_duration_sec: float = 12.0
    l2_local_fast_kernel_size: int = 1
    l2_local_slow_kernel_size: int = 9
    l2_local_peak_percentile: float = 75.0
    l2_scoring: str = "embedding"
    l2_frame_score_top_m: int = 4
    l2_frame_score_temperature: float = 0.07
    l2_rerank_encoder: str = "openclip"
    l2_rerank_query_mode: str = "target"
    l2_evidence_per_l3: int = 2
    l1_evidence_per_l2: int = 4
    evidence_text_mode: str = "frames"
    prompt_prefix: str = (
        "You are given retrieved evidence frames from a video. "
        "Use only the visible evidence and any provided subtitles."
    )


@dataclass(slots=True)
class AnswerPolicy:
    name: str
    model_id: str
    backend: str = "local"
    max_new_tokens: int = 32
    image_max_size: int | None = 336
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    enable_thinking: bool = False
    api_key_env_var: str = "ALIBABA_CLOUD_API"
    api_requests_per_minute: int = 60
    api_tokens_per_minute: int = 100000
