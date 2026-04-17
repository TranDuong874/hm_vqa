from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image, ImageOps
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, Qwen3_5ForConditionalGeneration


@dataclass(slots=True)
class AnswerConfig:
    model_id: str = "Qwen/Qwen3-VL-2B-Instruct"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_new_tokens: int = 32
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    image_max_size: int | None = None
    enable_thinking: bool = False


@dataclass(slots=True)
class PredictionResult:
    raw_text: str
    predicted_letter: str | None
    generation_sec: float


@dataclass(slots=True)
class GenerationResult:
    raw_text: str
    generation_sec: float


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
    ) -> PredictionResult:
        prompt = build_mcq_letter_prompt(question, options, prefix=prompt_prefix)
        generation = self.generate_text_from_frames(frames=frames, prompt=prompt)
        return PredictionResult(
            raw_text=generation.raw_text,
            predicted_letter=parse_choice_letter(generation.raw_text, options_count=len(options), options=options),
            generation_sec=generation.generation_sec,
        )

    def generate_text(
        self,
        *,
        prompt: str,
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

    def generate_text_from_frames(
        self,
        *,
        frames: list[Image.Image],
        prompt: str,
        frame_texts: list[str] | None = None,
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
    if normalized.startswith("qwen/qwen3.5-"):
        return Qwen3_5ForConditionalGeneration
    return Qwen3VLForConditionalGeneration
