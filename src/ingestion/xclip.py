from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModel, AutoProcessor

from .base import batched, l2_normalize, resolve_device


class XCLIPEncoder:
    def __init__(
        self,
        model_id: str = "microsoft/xclip-base-patch32",
        device: str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.model_id = model_id
        self.device = resolve_device(device)
        self.dtype = dtype or (torch.float16 if self.device.startswith("cuda") else torch.float32)
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(self.model_id).to(self.device)
        if self.device.startswith("cuda"):
            self.model = self.model.to(self.dtype)
        self.model.eval()
        self.num_frames = int(getattr(self.model.config, "num_frames", 8))

    def _resample_video(self, frames: list[Any]) -> list[Any]:
        if not frames:
            raise ValueError("Video frame list must not be empty")
        if len(frames) == self.num_frames:
            return list(frames)
        if len(frames) == 1:
            return [frames[0] for _ in range(self.num_frames)]
        indices = torch.linspace(0, len(frames) - 1, steps=self.num_frames).round().long().tolist()
        return [frames[int(index)] for index in indices]

    def encode_videos(
        self,
        videos: list[list[Any]],
        *,
        batch_size: int = 2,
        normalize: bool = True,
    ) -> torch.Tensor:
        outputs: list[torch.Tensor] = []
        with torch.inference_mode():
            for video_batch in batched(videos, batch_size):
                pixel_value_batches = []
                for frames in video_batch:
                    inputs = self.processor(images=self._resample_video(frames), return_tensors="pt")
                    pixel_value_batches.append(inputs["pixel_values"])
                pixel_values = torch.cat(pixel_value_batches, dim=0).to(self.device)
                if self.device.startswith("cuda"):
                    pixel_values = pixel_values.to(self.dtype)
                embeddings = self.model.get_video_features(pixel_values=pixel_values)
                if hasattr(embeddings, "pooler_output"):
                    embeddings = embeddings.pooler_output
                outputs.append(embeddings.float().cpu())
        result = torch.cat(outputs, dim=0) if outputs else torch.empty((0, 0), dtype=torch.float32)
        return l2_normalize(result) if normalize and result.numel() else result

    def encode_texts(
        self,
        texts: list[str],
        *,
        batch_size: int = 16,
        normalize: bool = True,
    ) -> torch.Tensor:
        outputs: list[torch.Tensor] = []
        with torch.inference_mode():
            for text_batch in batched(texts, batch_size):
                inputs = self.processor(text=text_batch, return_tensors="pt", padding=True, truncation=True)
                input_ids = inputs["input_ids"].to(self.device)
                attention_mask = inputs["attention_mask"].to(self.device)
                embeddings = self.model.get_text_features(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                if hasattr(embeddings, "pooler_output"):
                    embeddings = embeddings.pooler_output
                outputs.append(embeddings.float().cpu())
        result = torch.cat(outputs, dim=0) if outputs else torch.empty((0, 0), dtype=torch.float32)
        return l2_normalize(result) if normalize and result.numel() else result
