from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class AppConfig:
    root: Path = Path(__file__).resolve().parent
    cache_root: Path = Path(os.getenv("HMVQA_APP_CACHE_ROOT", Path(__file__).resolve().parent / ".cache" / "sessions"))
    static_root: Path = Path(__file__).resolve().parent / "static"
    schema_version: str = "hmvqa-app-v1"

    sample_fps: float = float(os.getenv("HMVQA_APP_SAMPLE_FPS", "1.0"))
    min_sample_fps: float = 0.25
    max_sample_fps: float = 4.0
    display_frame_size: int = int(os.getenv("HMVQA_APP_DISPLAY_FRAME_SIZE", "720"))
    image_max_size: int = int(os.getenv("HMVQA_APP_IMAGE_MAX_SIZE", "448"))
    openclip_device: str | None = os.getenv("HMVQA_APP_OPENCLIP_DEVICE", "cuda") or None
    viclip_device: str | None = os.getenv("HMVQA_APP_VICLIP_DEVICE", "cuda") or None
    unload_encoders_after_request: bool = os.getenv("HMVQA_APP_UNLOAD_ENCODERS", "1").lower() not in {"0", "false", "no"}
    openclip_batch_size: int = int(os.getenv("HMVQA_APP_OPENCLIP_BATCH_SIZE", "8"))
    viclip_batch_size: int = int(os.getenv("HMVQA_APP_VICLIP_BATCH_SIZE", "1"))
    use_viclip_l2: bool = os.getenv("HMVQA_APP_USE_VICLIP_L2", "1").lower() not in {"0", "false", "no"}
    l2_seconds: float = float(os.getenv("HMVQA_APP_L2_SECONDS", "5.0"))
    l3_seconds: float = float(os.getenv("HMVQA_APP_L3_SECONDS", "60.0"))
    default_evidence_frames: int = int(os.getenv("HMVQA_APP_EVIDENCE_FRAMES", "16"))

    default_backend: str = os.getenv("HMVQA_APP_BACKEND", "api")
    default_model_id: str = os.getenv("HMVQA_DEMO_MODEL_ID", os.getenv("HMVQA_APP_MODEL_ID", "qwen-vl-max-latest"))
    default_api_base_url: str = os.getenv(
        "HMVQA_DEMO_API_BASE_URL",
        os.getenv("HMVQA_APP_API_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
    )
    default_api_key_env_var: str = os.getenv("HMVQA_APP_API_KEY_ENV", "HMVQA_DEMO_API_KEY")
    allowed_model_ids: tuple[str, ...] = tuple(
        item.strip()
        for item in os.getenv(
            "HMVQA_APP_ALLOWED_MODELS",
            "qwen-vl-max-latest,qwen-vl-plus-latest,qwen2.5-vl-72b-instruct",
        ).split(",")
        if item.strip()
    )

    def clamp_sample_fps(self, value: float) -> float:
        return max(self.min_sample_fps, min(self.max_sample_fps, float(value)))


CONFIG = AppConfig()
