from __future__ import annotations

from collections.abc import Iterable
from typing import TypeVar

import torch
from PIL import Image
from transformers import XCLIPModel, XCLIPProcessor

from .open_clip import l2_normalize, resolve_device


T = TypeVar("T")


def batched(items: list[T], batch_size: int) -> Iterable[list[T]]:
    size = 1 if batch_size <= 0 else int(batch_size)
    for start in range(0, len(items), size):
        yield items[start : start + size]


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
        self.model = XCLIPModel.from_pretrained(self.model_id, torch_dtype=self.dtype).to(self.device)
        self.processor = XCLIPProcessor.from_pretrained(self.model_id)
        self.model.eval()
        self.num_frames = int(self.model.config.vision_config.num_frames)
        self.image_size = int(self.model.config.vision_config.image_size)

    @staticmethod
    def _feature_tensor(output: torch.Tensor | object) -> torch.Tensor:
        if isinstance(output, torch.Tensor):
            return output
        pooler_output = getattr(output, "pooler_output", None)
        if isinstance(pooler_output, torch.Tensor):
            return pooler_output
        last_hidden_state = getattr(output, "last_hidden_state", None)
        if isinstance(last_hidden_state, torch.Tensor):
            return last_hidden_state[:, 0]
        raise TypeError(f"Unsupported X-CLIP output type: {type(output)!r}")

    def encode_video_clips(
        self,
        clips: list[list[Image.Image]],
        *,
        batch_size: int = 4,
        normalize: bool = True,
    ) -> torch.Tensor:
        outputs: list[torch.Tensor] = []
        with torch.inference_mode():
            for clip_batch in batched(clips, batch_size):
                pixel_values = self.processor.image_processor(images=clip_batch, return_tensors="pt")["pixel_values"]
                pixel_values = pixel_values.to(self.device, dtype=self.dtype)
                embeddings = self._feature_tensor(self.model.get_video_features(pixel_values=pixel_values))
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
                encoded = self.processor.tokenizer(text_batch, padding=True, truncation=True, return_tensors="pt")
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                embeddings = self._feature_tensor(self.model.get_text_features(**encoded))
                outputs.append(embeddings.float().cpu())
        result = torch.cat(outputs, dim=0) if outputs else torch.empty((0, 0), dtype=torch.float32)
        return l2_normalize(result) if normalize and result.numel() else result
