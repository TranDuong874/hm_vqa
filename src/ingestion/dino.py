from __future__ import annotations

from typing import Any

import torch
from transformers import AutoImageProcessor, AutoModel

from .base import batched, l2_normalize, resolve_device


class DINOEncoder:
    def __init__(
        self,
        model_id: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        device: str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.model_id = model_id
        self.device = resolve_device(device)
        self.dtype = dtype or (torch.float16 if self.device.startswith("cuda") else torch.float32)
        self.processor = AutoImageProcessor.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(self.model_id).to(self.device)
        if self.device.startswith("cuda"):
            self.model = self.model.to(self.dtype)
        self.model.eval()

    def encode_images(
        self,
        images: list[Any],
        *,
        batch_size: int = 8,
        normalize: bool = True,
    ) -> torch.Tensor:
        outputs: list[torch.Tensor] = []
        with torch.inference_mode():
            for image_batch in batched(images, batch_size):
                inputs = self.processor(images=image_batch, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(self.device)
                if self.device.startswith("cuda"):
                    pixel_values = pixel_values.to(self.dtype)
                model_outputs = self.model(pixel_values=pixel_values)
                embeddings = model_outputs.last_hidden_state[:, 0]
                outputs.append(embeddings.float().cpu())
        result = torch.cat(outputs, dim=0) if outputs else torch.empty((0, 0), dtype=torch.float32)
        return l2_normalize(result) if normalize and result.numel() else result

