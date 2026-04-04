from __future__ import annotations

import re
import time
from dataclasses import dataclass

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


@dataclass(slots=True)
class AnswerConfig:
    model_id: str = "Qwen/Qwen3-VL-2B-Instruct"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_new_tokens: int = 32


@dataclass(slots=True)
class PredictionResult:
    raw_text: str
    predicted_letter: str | None
    generation_sec: float


def build_mcq_letter_prompt(question: str, options: list[str], prefix: str) -> str:
    return (
        f"{prefix}\n"
        "Answer the multiple-choice question using only the evidence shown.\n"
        "Reply with only one letter: A, B, C, or D.\n\n"
        f"Question: {question}\n"
        f"Options:\n" + "\n".join(options)
    )


def parse_choice_letter(text: str) -> str | None:
    match = re.search(r"\b([A-D])\b", text.upper())
    return match.group(1) if match else None


class QwenVLMAnswerer:
    def __init__(self, config: AnswerConfig | None = None) -> None:
        self.config = config or AnswerConfig()
        self.model: Qwen3VLForConditionalGeneration | None = None
        self.processor: AutoProcessor | None = None

    def load(self) -> None:
        if self.model is not None and self.processor is not None:
            return
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.config.model_id,
            torch_dtype=torch.bfloat16 if self.config.device == "cuda" else torch.float32,
            attn_implementation="sdpa",
        ).to(self.config.device)
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
        self.load()
        assert self.model is not None
        assert self.processor is not None

        content = [{"type": "image", "image": f"frame_{idx}.png"} for idx in range(len(frames))]
        content.append({"type": "text", "text": build_mcq_letter_prompt(question, options, prefix=prompt_prefix)})
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text],
            images=frames,
            padding=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.config.device) for key, value in inputs.items()}

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
        return PredictionResult(
            raw_text=raw_text,
            predicted_letter=parse_choice_letter(raw_text),
            generation_sec=round(elapsed, 3),
        )
