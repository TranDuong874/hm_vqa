from __future__ import annotations

import torch

from hmvqa_app.runtime.segments import Segment


def pool_segments(
    frame_embeddings: torch.Tensor,
    segments: list[Segment],
    *,
    pooling: str = "mean",
) -> torch.Tensor:
    pooled: list[torch.Tensor] = []
    for segment in segments:
        segment_embeddings = frame_embeddings[segment.start_index : segment.end_index + 1]
        if segment_embeddings.numel() == 0:
            continue
        if pooling == "mean":
            pooled_embedding = segment_embeddings.mean(dim=0)
        else:
            raise ValueError(f"Unsupported runtime segment pooling: {pooling}")
        pooled.append(pooled_embedding)
    if not pooled:
        width = frame_embeddings.shape[-1] if frame_embeddings.ndim == 2 else 0
        return torch.empty((0, width), dtype=torch.float32)
    return torch.nn.functional.normalize(torch.stack(pooled, dim=0), dim=-1)
