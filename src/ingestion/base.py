from __future__ import annotations

from collections.abc import Iterable
from typing import TypeVar

import torch


T = TypeVar("T")


def resolve_device(device: str | None = None) -> str:
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def normalize_batch_size(batch_size: int | None) -> int:
    if batch_size is None or batch_size <= 0:
        return 1
    return int(batch_size)


def batched(items: list[T], batch_size: int) -> Iterable[list[T]]:
    size = normalize_batch_size(batch_size)
    for start in range(0, len(items), size):
        yield items[start : start + size]


def l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(x, dim=-1)

