from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image, ImageOps
from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen3_5ForConditionalGeneration,
)


@dataclass(slots=True)
class AnswerConfig:
    model_id: str = "Qwen/Qwen3-VL-2B-Instruct"
    backend: str = "local"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_new_tokens: int = 384
    image_max_size: int | None = None
    enable_thinking: bool = False
    api_base_url: str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    api_key_env_var: str = "HMVQA_DEMO_API_KEY"
    api_timeout_sec: float = 180.0
    api_requests_per_minute: int = 60
    api_tokens_per_minute: int = 100000
    api_retry_attempts: int = 5
    api_retry_base_delay_sec: float = 2.0


@dataclass(slots=True)
class GenerationResult:
    raw_text: str
    generation_sec: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class QwenVLMAnswerer:
    def __init__(self, config: AnswerConfig | None = None) -> None:
        self.config = config or AnswerConfig()
        self.model: Any | None = None
        self.processor: AutoProcessor | None = None

    def load(self) -> None:
        if self.model is not None and self.processor is not None:
            return
        model_cls = _resolve_model_class(self.config.model_id)
        model_kwargs: dict[str, object] = {
            "attn_implementation": "sdpa",
            "torch_dtype": torch.bfloat16 if self.config.device == "cuda" else torch.float32,
        }
        self.model = model_cls.from_pretrained(self.config.model_id, **model_kwargs).to(self.config.device)
        self.processor = AutoProcessor.from_pretrained(self.config.model_id)

    def generate_text_from_frames(
        self,
        *,
        frames: list[Image.Image],
        prompt: str,
        frame_texts: list[str] | None = None,
        max_new_tokens: int | None = None,
    ) -> GenerationResult:
        self.load()
        assert self.model is not None
        assert self.processor is not None

        prepared_frames = [_prepare_frame(frame, image_max_size=self.config.image_max_size) for frame in frames]
        if frame_texts is not None and len(frame_texts) != len(prepared_frames):
            raise ValueError("frame_texts length must match frames length")

        content: list[dict[str, str]] = []
        for idx in range(len(prepared_frames)):
            if frame_texts is not None:
                content.append({"type": "text", "text": frame_texts[idx]})
            content.append({"type": "image", "image": f"frame_{idx}.png"})
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=prepared_frames, padding=True, return_tensors="pt")
        model_device = next(self.model.parameters()).device
        inputs = {key: value.to(model_device) for key, value in inputs.items()}

        started = time.perf_counter()
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=(max_new_tokens or self.config.max_new_tokens),
                do_sample=False,
            )
        elapsed = time.perf_counter() - started
        generated = outputs[0, inputs["input_ids"].shape[1] :]
        raw_text = self.processor.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
        del outputs, generated, inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return GenerationResult(raw_text=raw_text, generation_sec=round(elapsed, 3))


def _prepare_frame(frame: Image.Image, *, image_max_size: int | None) -> Image.Image:
    image = frame.copy().convert("RGB")
    if image_max_size is not None and image_max_size > 0:
        image = ImageOps.contain(image, (image_max_size, image_max_size))
    return image


def _resolve_model_class(model_id: str):
    normalized = (model_id or "").lower()
    if normalized.startswith("qwen/qwen2.5-"):
        return Qwen2_5_VLForConditionalGeneration
    if normalized.startswith("qwen/qwen2-"):
        return Qwen2VLForConditionalGeneration
    if normalized.startswith("qwen/qwen3.5-"):
        return Qwen3_5ForConditionalGeneration
    return Qwen3VLForConditionalGeneration
