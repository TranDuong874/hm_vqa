from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch

from .schema import RetrievalConfig
from .features import FeatureArchive, build_query_encoder
from .types import FrameHit, Segment, SegmentHit


def extract_target_text(question: str) -> str:
    match = re.search(r"<([^>]+)>", question)
    if match:
        text = match.group(1).strip()
        if text:
            return text
    return question.strip()


def overlaps_scope(
    start_time_sec: float,
    end_time_sec: float,
    scope_start_sec: float | None,
    scope_end_sec: float | None,
) -> bool:
    if scope_start_sec is None or scope_end_sec is None:
        return True
    return max(float(start_time_sec), float(scope_start_sec)) <= min(float(end_time_sec), float(scope_end_sec))


def encode_query_text(
    *,
    repo_root: Path,
    archive: FeatureArchive,
    query_text: str,
    config: RetrievalConfig,
) -> torch.Tensor:
    encoder = build_query_encoder(
        repo_root=repo_root,
        model_name=archive.model_name,
        pretrained_name=archive.pretrained_name,
        device=config.device,
    )
    try:
        return encoder.encode_texts([query_text], batch_size=config.openclip_batch_size)[0]
    finally:
        del encoder


def rank_segments(
    *,
    query_embedding: torch.Tensor,
    segment_embeddings: torch.Tensor,
    segments: list[Segment],
    video_id: str,
    top_k: int,
) -> list[SegmentHit]:
    if not segments or segment_embeddings.numel() == 0:
        return []
    scores = torch.matmul(segment_embeddings, query_embedding).cpu().numpy()
    order = scores.argsort()[::-1][: max(int(top_k), 1)]
    hits: list[SegmentHit] = []
    for index in order.tolist():
        segment = segments[int(index)]
        hits.append(
            SegmentHit(
                segment_id=segment.segment_id,
                score=float(scores[int(index)]),
                start_index=segment.start_index,
                end_index=segment.end_index,
                start_time_sec=segment.start_time_sec,
                end_time_sec=segment.end_time_sec,
                video_id=video_id,
            )
        )
    return hits


def select_ranked_hits(
    *,
    hits: list[SegmentHit],
    mode: str,
    top_k: int,
    relative_alpha: float,
    max_keep: int,
) -> list[SegmentHit]:
    if not hits:
        return []
    if str(mode) == "topk":
        return hits[: max(int(top_k), 1)]

    if str(mode) != "relative_threshold":
        raise ValueError(f"Unsupported selection mode: {mode}")

    top_score = float(hits[0].score)
    floor = float(relative_alpha) * top_score
    selected = [hit for hit in hits if float(hit.score) >= floor]
    if not selected:
        selected = [hits[0]]
    return selected[: max(int(max_keep), 1)]


def restrict_segments_to_hits(
    *,
    segments: list[Segment],
    parent_hits: list[SegmentHit],
    scope_start_sec: float | None,
    scope_end_sec: float | None,
) -> list[Segment]:
    if not segments:
        return []
    restricted: list[Segment] = []
    for segment in segments:
        if not overlaps_scope(segment.start_time_sec, segment.end_time_sec, scope_start_sec, scope_end_sec):
            continue
        if parent_hits and not any(
            overlaps_scope(
                segment.start_time_sec,
                segment.end_time_sec,
                float(parent.start_time_sec),
                float(parent.end_time_sec),
            )
            for parent in parent_hits
        ):
            continue
        restricted.append(segment)
    return restricted


def gather_segment_embeddings(
    *,
    all_segments: list[Segment],
    all_embeddings: torch.Tensor,
    selected_segments: list[Segment],
) -> torch.Tensor:
    if not selected_segments:
        width = int(all_embeddings.shape[-1]) if all_embeddings.ndim == 2 else 0
        return torch.empty((0, width), dtype=torch.float32)
    index_by_id = {segment.segment_id: index for index, segment in enumerate(all_segments)}
    indices = [index_by_id[segment.segment_id] for segment in selected_segments]
    return all_embeddings[torch.tensor(indices, dtype=torch.long)]


def rank_frames_in_segments(
    *,
    query_embedding: torch.Tensor,
    frame_embeddings: torch.Tensor,
    candidate_segments: list[SegmentHit],
    frame_timestamps: Any,
    video_id: str,
    top_k: int,
) -> list[FrameHit]:
    hits: list[FrameHit] = []
    if not candidate_segments:
        return hits
    frame_scores = torch.matmul(frame_embeddings, query_embedding).cpu()
    for segment_hit in candidate_segments:
        start_index = int(segment_hit.start_index)
        end_index = int(segment_hit.end_index)
        if end_index < start_index:
            continue
        local_scores = frame_scores[start_index : end_index + 1]
        if local_scores.numel() == 0:
            continue
        local_top_k = min(max(int(top_k), 1), int(local_scores.numel()))
        values, indices = torch.topk(local_scores, k=local_top_k)
        for value, local_index in zip(values.tolist(), indices.tolist(), strict=True):
            frame_index = start_index + int(local_index)
            hits.append(
                FrameHit(
                    frame_index=frame_index,
                    time_sec=float(frame_timestamps[frame_index]),
                    score=float(value),
                    video_id=video_id,
                )
            )
    hits.sort(key=lambda hit: hit.score, reverse=True)
    return hits[: max(int(top_k), 1)]
