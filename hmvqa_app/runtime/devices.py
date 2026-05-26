from __future__ import annotations

import gc
from typing import Any

import torch


def release_model(owner: Any, attribute: str) -> None:
    model = getattr(owner, attribute, None)
    if model is None:
        return
    setattr(owner, attribute, None)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
