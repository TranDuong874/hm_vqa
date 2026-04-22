from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
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
    max_new_tokens: int = 32
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    image_max_size: int | None = None
    enable_thinking: bool = False
    video_total_pixels: int = 20480 * 32 * 32
    video_min_pixels: int = 64 * 32 * 32
    api_base_url: str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    api_key_env_var: str = "ALIBABA_CLOUD_API"
    api_timeout_sec: float = 120.0
    api_requests_per_minute: int = 60
    api_tokens_per_minute: int = 100000
    api_retry_attempts: int = 5
    api_retry_base_delay_sec: float = 2.0


@dataclass(slots=True)
class PredictionResult:
    raw_text: str
    predicted_letter: str | None
    generation_sec: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(slots=True)
class GenerationResult:
    raw_text: str
    generation_sec: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


def _choice_letters(num_options: int) -> list[str]:
    if not (1 <= num_options <= 26):
        raise ValueError(f"Unsupported number of options: {num_options}")
    return [chr(ord("A") + index) for index in range(num_options)]


def build_mcq_letter_prompt(question: str, options: list[str], prefix: str) -> str:
    letters = _choice_letters(len(options))
    last_letters = ", ".join(letters[:-1]) + f", or {letters[-1]}" if len(letters) > 1 else letters[0]
    labeled_options = [f"{letter}. {option}" for letter, option in zip(letters, options)]
    return (
        f"{prefix}\n"
        "Answer the multiple-choice question using only the evidence shown.\n"
        f"Reply with only one letter: {last_letters}.\n\n"
        f"Question: {question}\n"
        f"Options:\n" + "\n".join(labeled_options)
    )


def _normalize_answer_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^A-Z0-9]+", " ", text.upper())).strip()


def parse_choice_letter(text: str, *, options_count: int = 4, options: list[str] | None = None) -> str | None:
    letters = "".join(_choice_letters(options_count))
    match = re.search(rf"\b([{letters}])\b", text.upper())
    if match:
        return match.group(1)
    if options:
        normalized_text = _normalize_answer_text(text)
        if normalized_text:
            for index, option in enumerate(options[:options_count]):
                normalized_option = _normalize_answer_text(option)
                if not normalized_option:
                    continue
                if normalized_text == normalized_option:
                    return chr(ord("A") + index)
                if len(normalized_option) >= 4 and normalized_option in normalized_text:
                    return chr(ord("A") + index)
    return None


class QwenVLMAnswerer:
    def __init__(self, config: AnswerConfig | None = None) -> None:
        self.config = config or AnswerConfig()
        self.model: Any | None = None
        self.processor: AutoProcessor | None = None

    def load(self) -> None:
        if self.model is not None and self.processor is not None:
            return
        model_kwargs: dict[str, object] = {
            "attn_implementation": "sdpa",
        }
        if self.config.load_in_4bit or self.config.load_in_8bit:
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as exc:
                raise RuntimeError(
                    "Quantized loading requires bitsandbytes support in transformers and the "
                    "'bitsandbytes' package to be installed."
                ) from exc

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=self.config.load_in_4bit,
                load_in_8bit=self.config.load_in_8bit,
                bnb_4bit_compute_dtype=torch.bfloat16 if self.config.device == "cuda" else torch.float32,
            )
            if self.config.device == "cuda":
                model_kwargs["device_map"] = {"": torch.cuda.current_device()}
            else:
                model_kwargs["device_map"] = {"": self.config.device}
            model_kwargs["low_cpu_mem_usage"] = True
        else:
            model_kwargs["torch_dtype"] = torch.bfloat16 if self.config.device == "cuda" else torch.float32

        model_cls = _resolve_model_class(self.config.model_id)
        self.model = model_cls.from_pretrained(
            self.config.model_id,
            **model_kwargs,
        )
        if not (self.config.load_in_4bit or self.config.load_in_8bit):
            self.model = self.model.to(self.config.device)
        self.processor = AutoProcessor.from_pretrained(self.config.model_id)

    def unload(self) -> None:
        if self.model is not None:
            del self.model
        if self.processor is not None:
            del self.processor
        self.model = None
        self.processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def answer_frames(
        self,
        *,
        frames: list[Image.Image],
        question: str,
        options: list[str],
        prompt_prefix: str,
        frame_texts: list[str] | None = None,
    ) -> PredictionResult:
        prompt = build_mcq_letter_prompt(question, options, prefix=prompt_prefix)
        generation = self.generate_text_from_frames(frames=frames, prompt=prompt, frame_texts=frame_texts)
        return PredictionResult(
            raw_text=generation.raw_text,
            predicted_letter=parse_choice_letter(generation.raw_text, options_count=len(options), options=options),
            generation_sec=generation.generation_sec,
            prompt_tokens=generation.prompt_tokens,
            completion_tokens=generation.completion_tokens,
            total_tokens=generation.total_tokens,
        )

    def answer_video(
        self,
        *,
        video_path: str | Path,
        question: str,
        options: list[str],
        prompt_prefix: str,
        sample_fps: float = 2.0,
        max_frames: int = 128,
        extra_text: str | None = None,
    ) -> PredictionResult:
        prompt = build_mcq_letter_prompt(question, options, prefix=prompt_prefix)
        if extra_text:
            prompt = f"{extra_text.strip()}\n\n{prompt}"
        generation = self.generate_text_from_video(
            video_path=video_path,
            prompt=prompt,
            sample_fps=sample_fps,
            max_frames=max_frames,
        )
        return PredictionResult(
            raw_text=generation.raw_text,
            predicted_letter=parse_choice_letter(generation.raw_text, options_count=len(options), options=options),
            generation_sec=generation.generation_sec,
            prompt_tokens=generation.prompt_tokens,
            completion_tokens=generation.completion_tokens,
            total_tokens=generation.total_tokens,
        )

    def generate_text(
        self,
        *,
        prompt: str,
        max_new_tokens: int | None = None,
    ) -> GenerationResult:
        self.load()
        assert self.model is not None
        assert self.processor is not None

        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text],
            padding=True,
            return_tensors="pt",
        )
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
        del outputs
        del generated
        del inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return GenerationResult(raw_text=raw_text, generation_sec=round(elapsed, 3))

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
        inputs = self.processor(
            text=[text],
            images=prepared_frames,
            padding=True,
            return_tensors="pt",
        )
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
        del outputs
        del generated
        del inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return GenerationResult(raw_text=raw_text, generation_sec=round(elapsed, 3))

    def generate_text_from_video(
        self,
        *,
        video_path: str | Path,
        prompt: str,
        sample_fps: float = 2.0,
        max_frames: int = 128,
    ) -> GenerationResult:
        self.load()
        assert self.model is not None
        assert self.processor is not None

        try:
            from qwen_vl_utils import process_vision_info
        except ImportError as exc:
            raise RuntimeError(
                "Native Qwen video input requires qwen-vl-utils in the active environment."
            ) from exc

        content: list[dict[str, object]] = [
            {
                "type": "video",
                "video": str(Path(video_path)),
                "total_pixels": int(self.config.video_total_pixels),
                "min_pixels": int(self.config.video_min_pixels),
                "max_frames": int(max_frames),
                "sample_fps": float(sample_fps),
            },
            {"type": "text", "text": prompt},
        ]
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            [messages],
            return_video_kwargs=True,
            image_patch_size=16,
            return_video_metadata=True,
        )
        if video_inputs is not None:
            video_inputs, video_metadatas = zip(*video_inputs)
            video_inputs = list(video_inputs)
            video_metadatas = list(video_metadatas)
        else:
            video_metadatas = None

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            video_metadata=video_metadatas,
            **video_kwargs,
            do_resize=False,
            padding=True,
            return_tensors="pt",
        )
        model_device = next(self.model.parameters()).device
        inputs = {
            key: (value.to(model_device) if hasattr(value, "to") else value)
            for key, value in inputs.items()
        }

        started = time.perf_counter()
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
            )
        elapsed = time.perf_counter() - started
        generated = outputs[0, inputs["input_ids"].shape[1] :]
        raw_text = self.processor.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
        del outputs
        del generated
        del inputs
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
