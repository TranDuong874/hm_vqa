from __future__ import annotations

from typing import Any

import open_clip
import torch

from .base import batched, l2_normalize, resolve_device


class OpenCLIPEncoder:
    def __init__(
        self,
        model_name: str = "ViT-L-14",
        pretrained: str = "laion2b_s32b_b82k",
        device: str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = resolve_device(device)
        self.dtype = dtype or (torch.float16 if self.device.startswith("cuda") else torch.float32)
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
