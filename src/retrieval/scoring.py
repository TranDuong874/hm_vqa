from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from segmentation import Segment, segment_fixed_windows

from .types import FrameHit, PipelineConfig, SampledVideo, SegmentHit


def build_window_segments(sampled_video: SampledVideo, config: PipelineConfig) -> list[Segment]:
    return segment_fixed_windows(
        timestamps=sampled_video.timestamps,
        window_seconds=config.window_seconds,
        stride_seconds=config.window_stride_seconds,
    )


def pool_segments(
    frame_embeddings: torch.Tensor,
    segments: list[Segment],
    *,
    pooling: str = "mean",
) -> torch.Tensor:
    pooled: list[torch.Tensor] = []
    for segment in segments:
        segment_embeddings = frame_embeddings[segment.start_index : segment.end_index + 1]
        if pooling == "mean":
            pooled_embedding = segment_embeddings.mean(dim=0)
        elif pooling == "weighted_mean":
            mean_embedding = torch.nn.functional.normalize(segment_embeddings.mean(dim=0), dim=0)
            weights = torch.softmax(torch.matmul(segment_embeddings, mean_embedding) / 0.1, dim=0)
            pooled_embedding = (segment_embeddings * weights.unsqueeze(1)).sum(dim=0)
        elif pooling == "mean_max_concat":
            pooled_embedding = torch.cat(
                [
                    segment_embeddings.mean(dim=0),
                    segment_embeddings.max(dim=0).values,
                ],
                dim=0,
            )
        else:
            raise ValueError(f"Unsupported segment pooling: {pooling}")
        pooled.append(pooled_embedding)
    if not pooled:
        width = frame_embeddings.shape[-1] if frame_embeddings.ndim == 2 else 0
        if pooling == "mean_max_concat":
            width *= 2
        return torch.empty((0, width), dtype=torch.float32)
    result = torch.stack(pooled, dim=0)
    return torch.nn.functional.normalize(result, dim=-1)


def mean_pool_segments(frame_embeddings: torch.Tensor, segments: list[Segment]) -> torch.Tensor:
    return pool_segments(frame_embeddings, segments, pooling="mean")


def adapt_query_embedding_for_segment_pooling(query_embedding: torch.Tensor, *, pooling: str = "mean") -> torch.Tensor:
    if pooling in {"mean", "weighted_mean"}:
        return query_embedding
    if pooling == "mean_max_concat":
        duplicated = torch.cat([query_embedding, query_embedding], dim=0)
        return torch.nn.functional.normalize(duplicated, dim=0)
    raise ValueError(f"Unsupported segment pooling: {pooling}")


def retrieve_top_segments_from_frame_scores(
    query_embedding: torch.Tensor,
    frame_embeddings: torch.Tensor,
    segments: list[Segment],
    *,
    top_k: int,
    top_m: int = 8,
    aggregation: str = "topm_mean",
    temperature: float = 0.07,
) -> list[SegmentHit]:
    if len(segments) == 0 or frame_embeddings.numel() == 0:
        return []
    frame_scores = torch.matmul(frame_embeddings, query_embedding).cpu()
    scored: list[SegmentHit] = []
    for segment in segments:
        segment_scores = frame_scores[segment.start_index : segment.end_index + 1]
        if segment_scores.numel() == 0:
            continue
        if aggregation == "topm_mean":
            k = min(max(top_m, 1), int(segment_scores.numel()))
            top_scores = torch.topk(segment_scores, k=k).values
            score = float(top_scores.mean().item())
        elif aggregation == "max":
            score = float(segment_scores.max().item())
        elif aggregation == "logsumexp_mean":
            tau = max(float(temperature), 1e-6)
            score = float(
                (
                    (tau * torch.logsumexp(segment_scores / tau, dim=0))
                    - (tau * torch.log(torch.tensor(float(segment_scores.numel()))))
                ).item()
            )
        else:
            raise ValueError(f"Unsupported frame-score aggregation: {aggregation}")
        scored.append(
            SegmentHit(
                segment_id=segment.segment_id,
                score=score,
                start_index=segment.start_index,
                end_index=segment.end_index,
                start_time_sec=segment.start_time_sec,
                end_time_sec=segment.end_time_sec,
            )
        )
    scored.sort(key=lambda hit: hit.score, reverse=True)
    return scored[:top_k]


def build_query_text(question: str, options: list[str] | None = None) -> str:
    if not options:
        return question
    lines = [question, "", "Options:"]
    for index, option in enumerate(options):
        letter = chr(ord("A") + index)
        lines.append(f"{letter}. {option}")
    return "\n".join(lines)


def retrieve_top_segments(
    query_embedding: torch.Tensor,
    segment_embeddings: torch.Tensor,
    segments: list[Segment],
    top_k: int,
) -> list[SegmentHit]:
    if len(segments) == 0 or segment_embeddings.numel() == 0:
        return []
    scores = torch.matmul(segment_embeddings, query_embedding).cpu().numpy()
    order = np.argsort(-scores)
    results: list[SegmentHit] = []
    for idx in order[:top_k]:
        segment = segments[int(idx)]
        results.append(
            SegmentHit(
                segment_id=segment.segment_id,
                score=float(scores[int(idx)]),
                start_index=segment.start_index,
                end_index=segment.end_index,
                start_time_sec=segment.start_time_sec,
                end_time_sec=segment.end_time_sec,
            )
        )
    return results


def collect_segment_frame_indices(segment_hits: list[SegmentHit]) -> list[int]:
    candidate_indices: set[int] = set()
    for hit in segment_hits:
        candidate_indices.update(range(hit.start_index, hit.end_index + 1))
    return sorted(candidate_indices)


def retrieve_top_frames(
    query_embedding: torch.Tensor,
    frame_embeddings: torch.Tensor,
    timestamps: np.ndarray,
    *,
    top_k: int,
    allowed_indices: list[int] | None = None,
) -> list[FrameHit]:
    if frame_embeddings.numel() == 0:
        return []

    if allowed_indices is None:
        candidate_indices = list(range(frame_embeddings.shape[0]))
    else:
        candidate_indices = sorted(set(int(index) for index in allowed_indices if 0 <= int(index) < frame_embeddings.shape[0]))
    if not candidate_indices:
        return []

    candidate_tensor = frame_embeddings[candidate_indices]
    scores = torch.matmul(candidate_tensor, query_embedding).cpu().numpy()
    order = np.argsort(-scores)[:top_k]

    hits: list[FrameHit] = []
    for rank_index in order:
        frame_index = candidate_indices[int(rank_index)]
        hits.append(
            FrameHit(
                frame_index=frame_index,
                time_sec=float(timestamps[frame_index]),
                score=float(scores[int(rank_index)]),
            )
        )
    hits.sort(key=lambda hit: hit.frame_index)
    return hits


def select_evidence_frames(
    *,
    frames: list[Image.Image],
    frame_hits: list[FrameHit],
) -> list[Image.Image]:
    return [frames[hit.frame_index] for hit in frame_hits]
