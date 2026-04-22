from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any
from typing import TypeVar

import numpy as np
import torch
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import AutoConfig, AutoModel
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from .open_clip import l2_normalize, resolve_device


T = TypeVar("T")


def batched(items: list[T], batch_size: int) -> Iterable[list[T]]:
    size = 1 if batch_size <= 0 else int(batch_size)
    for start in range(0, len(items), size):
        yield items[start : start + size]


class ViCLIPEncoder:
    def __init__(
        self,
        model_id: str = "OpenGVLab/ViCLIP-L-14-hf",
        device: str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.model_id = model_id
        self.device = resolve_device(device)
        self.dtype = dtype or (torch.float16 if self.device.startswith("cuda") else torch.float32)
        self.local_dir = Path(
            snapshot_download(
                repo_id=self.model_id,
                allow_patterns=[
                    "config.json",
                    "configuration_viclip.py",
                    "viclip.py",
                    "viclip_text.py",
                    "viclip_vision.py",
                    "simple_tokenizer.py",
                    "bpe_simple_vocab_16e6.txt.gz",
                    "model.safetensors",
                ],
            )
        )
        config = AutoConfig.from_pretrained(self.local_dir, trust_remote_code=True)
        config.tokenizer_path = str(self.local_dir / "bpe_simple_vocab_16e6.txt.gz")
        model_class = get_class_from_dynamic_module("viclip.ViCLIP", str(self.local_dir))
        if not hasattr(model_class, "all_tied_weights_keys"):
            model_class.all_tied_weights_keys = {}
        self.model = AutoModel.from_pretrained(
            self.local_dir,
            trust_remote_code=True,
            config=config,
            low_cpu_mem_usage=False,
        ).to(self.device)
        text_encoder = getattr(self.model, "text_encoder", None)
        if text_encoder is not None and hasattr(text_encoder, "context_length"):
            attn_mask = self._materialize_attention_mask(text_encoder)
            for block in getattr(getattr(text_encoder, "transformer", None), "resblocks", []):
                attn_mask = getattr(block, "attn_mask", None)
                if attn_mask is not None and getattr(attn_mask, "is_meta", False):
                    block.attn_mask = self._materialize_attention_mask(text_encoder)
        if self.device.startswith("cuda"):
            self.model = self.model.to(self.dtype)
        text_encoder = getattr(self.model, "text_encoder", None)
        if text_encoder is not None:
            text_encoder.float()
        self.model.eval()
        self.num_frames = int(getattr(self.model, "video_input_num_frames", 8))
        self.image_size = int(getattr(self.model, "inputs_image_res", 224))
        self._mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self._std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    def _materialize_attention_mask(self, text_encoder: Any) -> torch.Tensor:
        context_length = int(getattr(text_encoder, "context_length", 77))
        mask = torch.empty(context_length, context_length, device="cpu")
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        resized = image.convert("RGB").resize((self.image_size, self.image_size), Image.BICUBIC)
        array = np.asarray(resized, dtype=np.float32)
        array = (array / 255.0 - self._mean) / self._std
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        return tensor

    def encode_video_clips(
        self,
        clips: list[list[Image.Image]],
        *,
        batch_size: int = 2,
        normalize: bool = True,
    ) -> torch.Tensor:
        outputs: list[torch.Tensor] = []
        model_dtype = next(self.model.parameters()).dtype
        with torch.inference_mode():
            for clip_batch in batched(clips, batch_size):
                video_tensor = torch.stack(
                    [
                        torch.stack([self._preprocess_image(frame) for frame in clip], dim=0)
                        for clip in clip_batch
                    ],
                    dim=0,
                ).to(self.device, dtype=model_dtype)
                embeddings = self.model.get_vid_features(video_tensor)
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
                embeddings = [self.model.get_text_features(text, None, {}) for text in text_batch]
                outputs.append(torch.cat([embedding.float().cpu() for embedding in embeddings], dim=0))
        result = torch.cat(outputs, dim=0) if outputs else torch.empty((0, 0), dtype=torch.float32)
        return l2_normalize(result) if normalize and result.numel() else result
