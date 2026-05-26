from __future__ import annotations

from collections.abc import Iterable
from typing import Any, TypeVar

import open_clip
import torch


T = TypeVar("T")


def resolve_device(device: str | None = None) -> str:
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def batched(items: list[T], batch_size: int) -> Iterable[list[T]]:
    size = 1 if batch_size <= 0 else int(batch_size)
    for start in range(0, len(items), size):
        yield items[start : start + size]


def l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(x, dim=-1)


class OpenCLIPEncoder:
    def __init__(
        self,
        model_name: str = "ViT-L-14",
        pretrained: str = "laion2b_s32b_b82k",
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = resolve_device(device)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_name,
            pretrained=self.pretrained,
            device=self.device,
        )
        self.tokenizer = open_clip.get_tokenizer(self.model_name)
        self.model_dtype = next(self.model.parameters()).dtype
        self.model.eval()

    def encode_images(
        self,
        images: list[Any],
        *,
        batch_size: int = 16,
        normalize: bool = True,
    ) -> torch.Tensor:
        outputs: list[torch.Tensor] = []
        with torch.inference_mode():
            for image_batch in batched(images, batch_size):
                image_tensor = torch.stack([self.preprocess(image) for image in image_batch], dim=0).to(self.device)
                image_tensor = image_tensor.to(self.model_dtype)
                embeddings = self.model.encode_image(image_tensor)
                outputs.append(embeddings.float().cpu())
        result = torch.cat(outputs, dim=0) if outputs else torch.empty((0, 0), dtype=torch.float32)
        return l2_normalize(result) if normalize and result.numel() else result

    def encode_texts(
        self,
        texts: list[str],
        *,
        batch_size: int = 32,
        normalize: bool = True,
    ) -> torch.Tensor:
        outputs: list[torch.Tensor] = []
        with torch.inference_mode():
            for text_batch in batched(texts, batch_size):
                tokenized = self.tokenizer(text_batch).to(self.device)
                embeddings = self.model.encode_text(tokenized)
                outputs.append(embeddings.float().cpu())
        result = torch.cat(outputs, dim=0) if outputs else torch.empty((0, 0), dtype=torch.float32)
        return l2_normalize(result) if normalize and result.numel() else result
