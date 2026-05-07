from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import torch

from evals.common.retrieval_ablation_runner import AblationRetriever
from evals.common.video_sampling import sample_uniform_video_frames as _sample_uniform_video_frames
from ingestion.viclip import ViCLIPEncoder

from .intervals import (
    _hit_dict,
    _rank_score_by_id,
    _segment_in_scope,
    _segment_time_overlap,
    _segment_topm_score,
    _temporal_nms_frame_hits,
)
from .metrics import _best_coverage
from .queries import _decompose_target_text


DEFAULT_RECALL_K = (1, 3, 5)
L1_ANCHOR_K = 5
L1_NMS_SEC = 2.0
L2_NEIGHBOR_RADIUS = 1
L2_SCORE_TOP_M = 4
VICLIP_L2_MAX_FRAMES = 16
_VICLIP_ENCODER: ViCLIPEncoder | None = None


def _get_viclip_encoder() -> ViCLIPEncoder:
    global _VICLIP_ENCODER
    if _VICLIP_ENCODER is None:
        _VICLIP_ENCODER = ViCLIPEncoder()
    return _VICLIP_ENCODER


def _viclip_l2_cache_dir(retriever: AblationRetriever, video_id: str) -> Path:
    key = retriever._stable_hash(
        {
            "version": 1,
            "encoder": "viclip",
            "model_id": "OpenGVLab/ViCLIP-L-14-hf",
            "segmentation": retriever.config.l2_segmentation,
            "window_seconds": retriever.config.l2_window_seconds,
            "stride_seconds": retriever.config.l2_stride_seconds,
            "local_min_duration_sec": retriever.config.l2_local_min_duration_sec,
            "local_max_duration_sec": retriever.config.l2_local_max_duration_sec,
            "local_fast_kernel_size": retriever.config.l2_local_fast_kernel_size,
            "local_slow_kernel_size": retriever.config.l2_local_slow_kernel_size,
            "local_peak_percentile": retriever.config.l2_local_peak_percentile,
            "sample_fps": retriever.config.sample_fps,
            "feature_eval_fps": retriever.config.feature_eval_fps,
        }
    )
    return retriever.derived_cache_root / video_id / f"l2_viclip_{key}"


def _ensure_viclip_l2_embeddings(
    *,
    retriever: AblationRetriever,
    artifacts: Any,
) -> torch.Tensor:
    retriever._ensure_l2(artifacts)
    assert artifacts.l2_segments is not None
    cache_dir = _viclip_l2_cache_dir(retriever, artifacts.video_id)
    segments_path = cache_dir / "segments.json"
    embeddings_path = cache_dir / "embeddings.pt"
    if segments_path.exists() and embeddings_path.exists():
        return torch.load(embeddings_path, map_location="cpu").float()

    encoder = _get_viclip_encoder()
    embedding_rows: list[torch.Tensor] = []
    for index, segment in enumerate(artifacts.l2_segments):
        candidate_budget = min(
            VICLIP_L2_MAX_FRAMES,
            max(1, int(segment.end_index) - int(segment.start_index) + 1),
        )
        frames, _, _, _ = _sample_uniform_video_frames(
            video_path=artifacts.video_path,
            frame_budget=candidate_budget,
            start_time_sec=float(segment.start_time_sec),
            end_time_sec=float(segment.end_time_sec),
        )
        if len(frames) >= encoder.num_frames:
            step_indices = torch.linspace(0, len(frames) - 1, steps=encoder.num_frames)
            selected = [frames[int(round(float(index)))] for index in step_indices.tolist()]
        else:
            selected = list(frames)
            while len(selected) < encoder.num_frames:
                selected.append(selected[-1])
        clip_embedding = encoder.encode_video_clips([selected], batch_size=1).float().cpu()
        embedding_rows.append(clip_embedding[0])
        del frames
        del selected
        if (index + 1) % 16 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    embeddings = torch.stack(embedding_rows, dim=0) if embedding_rows else torch.empty((0, 0), dtype=torch.float32)
    cache_dir.mkdir(parents=True, exist_ok=True)
    segments_path.write_text(json.dumps(retriever._serialize_segments(artifacts.l2_segments), indent=2), encoding="utf-8")
    torch.save(embeddings, embeddings_path)
    return embeddings


def _segment_id_for_frame_index(l3_hits: list[Any], frame_index: int) -> str | None:
    for hit in l3_hits:
        if int(hit.start_index) <= int(frame_index) <= int(hit.end_index):
            return str(hit.segment_id)
    return None


def _rerank_l3_hits_with_decomposed_l1(
    *,
    retriever: AblationRetriever,
    artifacts: Any,
    target_text: str,
    l3_hits: list[Any],
    allowed_indices: list[int],
    scope_start_sec: float | None,
    scope_end_sec: float | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not l3_hits or not allowed_indices:
        return [], {"queries": []}

    query_specs = _decompose_target_text(target_text)
    if not query_specs:
        return [], {"queries": []}

    l3_prior_rank = _rank_score_by_id(
        [(str(hit.segment_id), float(hit.score)) for hit in l3_hits],
        key_fn=lambda item: float(item[1]),
    )
    l3_stats: dict[str, dict[str, float]] = {
        str(hit.segment_id): {
            "full_votes": 0.0,
            "action_votes": 0.0,
            "object_votes": 0.0,
            "full_rank_sum": 0.0,
            "action_rank_sum": 0.0,
            "object_rank_sum": 0.0,
            "full_best_rank": 0.0,
            "action_best_rank": 0.0,
            "object_best_rank": 0.0,
        }
        for hit in l3_hits
    }

    query_debug: list[dict[str, Any]] = []
    for query_kind, query_text in query_specs:
        query_embedding = retriever._encoder.encode_texts([query_text]).cpu()[0]
        query_embedding = torch.nn.functional.normalize(query_embedding, dim=0)
        frame_hits = retriever._frame_hits_from_indices(
            artifacts=artifacts,
            query_embedding=query_embedding,
            allowed_indices=allowed_indices,
        )
        anchor_hits = _temporal_nms_frame_hits(
            frame_hits,
            max_hits=L1_ANCHOR_K,
            min_gap_sec=L1_NMS_SEC,
        )
        rank_scores = _rank_score_by_id(
            [(int(hit.frame_index), float(hit.score)) for hit in anchor_hits],
            key_fn=lambda item: float(item[1]),
        )
        debug_hits: list[dict[str, Any]] = []
        for hit in anchor_hits:
            segment_id = _segment_id_for_frame_index(l3_hits, int(hit.frame_index))
            debug_hits.append(
                {
                    "frame_index": int(hit.frame_index),
                    "time_sec": float(hit.time_sec),
                    "score": float(hit.score),
                    "segment_id": segment_id,
                }
            )
            if segment_id is None:
                continue
            stats = l3_stats[segment_id]
            vote_key = f"{query_kind}_votes"
            rank_key = f"{query_kind}_rank_sum"
            best_rank_key = f"{query_kind}_best_rank"
            stats[vote_key] += 1.0
            stats[rank_key] += float(rank_scores.get(int(hit.frame_index), 0.0))
            stats[best_rank_key] = max(
                float(stats[best_rank_key]),
                float(rank_scores.get(int(hit.frame_index), 0.0)),
            )
        query_debug.append({"kind": query_kind, "text": query_text, "anchor_hits": debug_hits})

    query_weights = {"full": 1.0, "action": 0.75, "object": 0.75}
    reranked_hits: list[dict[str, Any]] = []
    for hit in l3_hits:
        segment_id = str(hit.segment_id)
        stats = l3_stats[segment_id]
        weighted_votes = sum(
            query_weights[kind] * float(stats[f"{kind}_votes"])
            for kind in ("full", "action", "object")
        )
        weighted_rank_sum = sum(
            query_weights[kind] * float(stats[f"{kind}_rank_sum"])
            for kind in ("full", "action", "object")
        )
        weighted_best_rank = sum(
            query_weights[kind] * float(stats[f"{kind}_best_rank"])
            for kind in ("full", "action", "object")
        )
        prior_rank = float(l3_prior_rank.get(segment_id, 0.0))
        rerank_score = weighted_votes + (0.35 * weighted_rank_sum) + (0.15 * weighted_best_rank) + (0.05 * prior_rank)
        reranked_hits.append(
            _hit_dict(
                start_time_sec=float(hit.start_time_sec),
                end_time_sec=float(hit.end_time_sec),
                score=rerank_score,
                scope_start_sec=scope_start_sec,
                scope_end_sec=scope_end_sec,
                source="l3_rerank_l1_decomp",
                segment_id=segment_id,
                components={
                    "weighted_votes": weighted_votes,
                    "weighted_rank_sum": weighted_rank_sum,
                    "weighted_best_rank": weighted_best_rank,
                    "l3_prior_rank": prior_rank,
                    "full_votes": stats["full_votes"],
                    "action_votes": stats["action_votes"],
                    "object_votes": stats["object_votes"],
                },
            )
        )

    reranked_hits.sort(
        key=lambda item: (
            float(item["score"]),
            float(item.get("components", {}).get("weighted_votes", 0.0)),
            float(item.get("components", {}).get("l3_prior_rank", 0.0)),
        ),
        reverse=True,
    )
    return reranked_hits[: max(DEFAULT_RECALL_K)], {"queries": query_debug}


def _rerank_l3_hits_with_decomposed_l1_l2(
    *,
    retriever: AblationRetriever,
    artifacts: Any,
    target_text: str,
    query_embedding: torch.Tensor,
    l3_hits: list[Any],
    allowed_indices: list[int],
    scope_start_sec: float | None,
    scope_end_sec: float | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not l3_hits or not allowed_indices:
        return [], {"queries": [], "l2_candidates": []}

    retriever._ensure_l2(artifacts)
    assert artifacts.l2_segments is not None
    candidate_l2_indices = _candidate_l2_indices_from_l3_hits(
        artifacts,
        l3_hits,
        scope_start_sec=scope_start_sec,
        scope_end_sec=scope_end_sec,
    )
    if not candidate_l2_indices:
        return _rerank_l3_hits_with_decomposed_l1(
            retriever=retriever,
            artifacts=artifacts,
            target_text=target_text,
            l3_hits=l3_hits,
            allowed_indices=allowed_indices,
            scope_start_sec=scope_start_sec,
            scope_end_sec=scope_end_sec,
        )

    query_specs = _decompose_target_text(target_text)
    l3_prior_rank = _rank_score_by_id(
        [(str(hit.segment_id), float(hit.score)) for hit in l3_hits],
        key_fn=lambda item: float(item[1]),
    )
    frame_scores_full = torch.matmul(artifacts.frame_embeddings, query_embedding).cpu()
    l2_query_rank = _rank_score_by_id(
        [
            (
                segment_index,
                _segment_topm_score(
                    frame_scores=frame_scores_full,
                    start_index=int(artifacts.l2_segments[segment_index].start_index),
                    end_index=int(artifacts.l2_segments[segment_index].end_index),
                    top_m=L2_SCORE_TOP_M,
                ),
            )
            for segment_index in sorted(candidate_l2_indices)
        ],
        key_fn=lambda item: float(item[1]),
    )

    l2_votes: dict[int, dict[str, float]] = {
        int(segment_index): {
            "full_votes": 0.0,
            "action_votes": 0.0,
            "object_votes": 0.0,
            "full_rank_sum": 0.0,
            "action_rank_sum": 0.0,
            "object_rank_sum": 0.0,
            "full_best_rank": 0.0,
            "action_best_rank": 0.0,
            "object_best_rank": 0.0,
        }
        for segment_index in candidate_l2_indices
    }
    query_debug: list[dict[str, Any]] = []
    for query_kind, query_text in query_specs:
        query_vec = retriever._encoder.encode_texts([query_text]).cpu()[0]
        query_vec = torch.nn.functional.normalize(query_vec, dim=0)
        frame_hits = retriever._frame_hits_from_indices(
            artifacts=artifacts,
            query_embedding=query_vec,
            allowed_indices=allowed_indices,
        )
        anchor_hits = _temporal_nms_frame_hits(
            frame_hits,
            max_hits=L1_ANCHOR_K,
            min_gap_sec=L1_NMS_SEC,
        )
        rank_scores = _rank_score_by_id(
            [(int(hit.frame_index), float(hit.score)) for hit in anchor_hits],
            key_fn=lambda item: float(item[1]),
        )
        debug_hits: list[dict[str, Any]] = []
        for hit in anchor_hits:
            matched_l2_index = None
            for segment_index in sorted(candidate_l2_indices):
                segment = artifacts.l2_segments[segment_index]
                if int(segment.start_index) <= int(hit.frame_index) <= int(segment.end_index):
                    matched_l2_index = int(segment_index)
                    break
            segment_id = _segment_id_for_frame_index(l3_hits, int(hit.frame_index))
            debug_hits.append(
                {
                    "frame_index": int(hit.frame_index),
                    "time_sec": float(hit.time_sec),
                    "score": float(hit.score),
                    "l2_index": matched_l2_index,
                    "segment_id": segment_id,
                }
            )
            if matched_l2_index is None:
                continue
            stats = l2_votes[matched_l2_index]
            vote_key = f"{query_kind}_votes"
            rank_key = f"{query_kind}_rank_sum"
            best_rank_key = f"{query_kind}_best_rank"
            stats[vote_key] += 1.0
            stats[rank_key] += float(rank_scores.get(int(hit.frame_index), 0.0))
            stats[best_rank_key] = max(
                float(stats[best_rank_key]),
                float(rank_scores.get(int(hit.frame_index), 0.0)),
            )
        query_debug.append({"kind": query_kind, "text": query_text, "anchor_hits": debug_hits})

    query_weights = {"full": 1.0, "action": 0.75, "object": 0.75}
    l2_to_parent: dict[int, str] = {}
    l2_scored: list[dict[str, Any]] = []
    for segment_index in sorted(candidate_l2_indices):
        segment = artifacts.l2_segments[segment_index]
        parent_segment_id = None
        for l3_hit in l3_hits:
            if _segment_time_overlap(
                float(segment.start_time_sec),
                float(segment.end_time_sec),
                float(l3_hit.start_time_sec),
                float(l3_hit.end_time_sec),
            ):
                parent_segment_id = str(l3_hit.segment_id)
                break
        if parent_segment_id is None:
            continue
        l2_to_parent[segment_index] = parent_segment_id
        stats = l2_votes[segment_index]
        weighted_votes = sum(query_weights[kind] * float(stats[f"{kind}_votes"]) for kind in ("full", "action", "object"))
        weighted_rank_sum = sum(query_weights[kind] * float(stats[f"{kind}_rank_sum"]) for kind in ("full", "action", "object"))
        weighted_best_rank = sum(query_weights[kind] * float(stats[f"{kind}_best_rank"]) for kind in ("full", "action", "object"))
        query_rank = float(l2_query_rank.get(segment_index, 0.0))
        parent_prior = float(l3_prior_rank.get(parent_segment_id, 0.0))
        score = weighted_votes + (0.35 * weighted_rank_sum) + (0.15 * weighted_best_rank) + (0.35 * query_rank) + (0.05 * parent_prior)
        l2_scored.append(
            {
                "segment_index": segment_index,
                "segment_id": str(segment.segment_id),
                "parent_segment_id": parent_segment_id,
                "start_time_sec": float(segment.start_time_sec),
                "end_time_sec": float(segment.end_time_sec),
                "score": score,
                "components": {
                    "weighted_votes": weighted_votes,
                    "weighted_rank_sum": weighted_rank_sum,
                    "weighted_best_rank": weighted_best_rank,
                    "l2_query_rank": query_rank,
                    "l3_prior_rank": parent_prior,
                },
            }
        )

    parent_scores: dict[str, dict[str, float]] = {}
    for item in l2_scored:
        parent_id = str(item["parent_segment_id"])
        existing = parent_scores.get(parent_id)
        if existing is None:
            parent_scores[parent_id] = {
                "best_l2_score": float(item["score"]),
                "sum_top2_l2_score": float(item["score"]),
                "l2_count": 1.0,
            }
        else:
            best = max(float(existing["best_l2_score"]), float(item["score"]))
            sum_top2 = float(existing["sum_top2_l2_score"])
            if float(item["score"]) > float(existing["best_l2_score"]):
                sum_top2 = float(existing["best_l2_score"]) + float(item["score"])
            else:
                sum_top2 = max(sum_top2, float(existing["best_l2_score"]) + float(item["score"]))
            existing["best_l2_score"] = best
            existing["sum_top2_l2_score"] = sum_top2
            existing["l2_count"] = float(existing["l2_count"]) + 1.0

    reranked_hits: list[dict[str, Any]] = []
    for hit in l3_hits:
        segment_id = str(hit.segment_id)
        parent = parent_scores.get(segment_id, {"best_l2_score": 0.0, "sum_top2_l2_score": 0.0, "l2_count": 0.0})
        prior_rank = float(l3_prior_rank.get(segment_id, 0.0))
        rerank_score = float(parent["best_l2_score"]) + (0.25 * float(parent["sum_top2_l2_score"])) + (0.05 * prior_rank)
        reranked_hits.append(
            _hit_dict(
                start_time_sec=float(hit.start_time_sec),
                end_time_sec=float(hit.end_time_sec),
                score=rerank_score,
                scope_start_sec=scope_start_sec,
                scope_end_sec=scope_end_sec,
                source="l3_rerank_l1_decomp_l2",
                segment_id=segment_id,
                components={
                    "best_l2_score": float(parent["best_l2_score"]),
                    "sum_top2_l2_score": float(parent["sum_top2_l2_score"]),
                    "l2_count": float(parent["l2_count"]),
                    "l3_prior_rank": prior_rank,
                },
            )
        )
    reranked_hits.sort(
        key=lambda item: (
            float(item["score"]),
            float(item.get("components", {}).get("best_l2_score", 0.0)),
            float(item.get("components", {}).get("l3_prior_rank", 0.0)),
        ),
        reverse=True,
    )
    return reranked_hits[: max(DEFAULT_RECALL_K)], {"queries": query_debug, "l2_candidates": l2_scored}


def _rerank_l3_hits_with_l2(
    *,
    retriever: AblationRetriever,
    artifacts: Any,
    query_embedding: torch.Tensor,
    target_text: str,
    l3_hits: list[Any],
    scope_start_sec: float | None,
    scope_end_sec: float | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not l3_hits:
        return [], {"l2_candidates": []}

    assert artifacts.l2_segments is not None
    candidate_indices = _candidate_l2_indices_from_l3_hits(
        artifacts,
        l3_hits,
        scope_start_sec=scope_start_sec,
        scope_end_sec=scope_end_sec,
    )
    if not candidate_indices:
        return [], {"l2_candidates": []}

    frame_scores = torch.matmul(artifacts.frame_embeddings, query_embedding).cpu()
    l3_prior_rank = _rank_score_by_id(
        [(str(hit.segment_id), float(hit.score)) for hit in l3_hits],
        key_fn=lambda item: float(item[1]),
    )

    viclip_query_embedding: torch.Tensor | None = None
    viclip_l2_embeddings: torch.Tensor | None = None
    if retriever.config.l2_rerank_encoder == "viclip":
        viclip_query_embedding = _get_viclip_encoder().encode_texts([target_text])[0].float().cpu()
        viclip_l2_embeddings = _ensure_viclip_l2_embeddings(retriever=retriever, artifacts=artifacts)

    l2_items: list[dict[str, Any]] = []
    for segment_index in sorted(candidate_indices):
        segment = artifacts.l2_segments[segment_index]
        parent_segment_id = None
        for l3_hit in l3_hits:
            if _segment_time_overlap(
                float(segment.start_time_sec),
                float(segment.end_time_sec),
                float(l3_hit.start_time_sec),
                float(l3_hit.end_time_sec),
            ):
                parent_segment_id = str(l3_hit.segment_id)
                break
        if parent_segment_id is None:
            continue
        if retriever.config.l2_rerank_encoder == "viclip":
            assert viclip_query_embedding is not None
            assert viclip_l2_embeddings is not None
            l2_score = float(torch.dot(viclip_l2_embeddings[int(segment_index)], viclip_query_embedding).item())
        else:
            l2_score = _segment_topm_score(
                frame_scores=frame_scores,
                start_index=int(segment.start_index),
                end_index=int(segment.end_index),
                top_m=L2_SCORE_TOP_M,
            )
        l2_items.append(
            {
                "segment_index": int(segment_index),
                "segment_id": str(segment.segment_id),
                "parent_segment_id": parent_segment_id,
                "start_time_sec": float(segment.start_time_sec),
                "end_time_sec": float(segment.end_time_sec),
                "raw_score": float(l2_score),
            }
        )

    l2_rank = _rank_score_by_id(
        [(int(item["segment_index"]), float(item["raw_score"])) for item in l2_items],
        key_fn=lambda item: float(item[1]),
    )
    for item in l2_items:
        item["rank_score"] = float(l2_rank.get(int(item["segment_index"]), 0.0))

    parent_scores: dict[str, list[float]] = {}
    for item in l2_items:
        parent_scores.setdefault(str(item["parent_segment_id"]), []).append(float(item["rank_score"]))

    reranked_hits: list[dict[str, Any]] = []
    for hit in l3_hits:
        segment_id = str(hit.segment_id)
        child_scores = sorted(parent_scores.get(segment_id, []), reverse=True)
        best_child = child_scores[0] if child_scores else 0.0
        top2_sum = sum(child_scores[:2]) if child_scores else 0.0
        prior_rank = float(l3_prior_rank.get(segment_id, 0.0))
        score = best_child + (0.35 * top2_sum) + (0.05 * prior_rank)
        reranked_hits.append(
            _hit_dict(
                start_time_sec=float(hit.start_time_sec),
                end_time_sec=float(hit.end_time_sec),
                score=score,
                scope_start_sec=scope_start_sec,
                scope_end_sec=scope_end_sec,
                source="l3_rerank_l2",
                segment_id=segment_id,
                components={
                    "best_l2_rank": best_child,
                    "top2_l2_rank_sum": top2_sum,
                    "l3_prior_rank": prior_rank,
                    "l2_count": float(len(child_scores)),
                    "l2_rerank_encoder": 1.0 if retriever.config.l2_rerank_encoder == "viclip" else 0.0,
                },
            )
        )
    reranked_hits.sort(
        key=lambda item: (
            float(item["score"]),
            float(item.get("components", {}).get("best_l2_rank", 0.0)),
            float(item.get("components", {}).get("l3_prior_rank", 0.0)),
        ),
        reverse=True,
    )
    l2_items.sort(key=lambda item: float(item["rank_score"]), reverse=True)
    return reranked_hits[: max(DEFAULT_RECALL_K)], {"l2_candidates": l2_items, "l2_rerank_encoder": retriever.config.l2_rerank_encoder}


def _candidate_l2_indices_from_frame_hits(
    artifacts: Any,
    frame_hits: list[Any],
    *,
    neighbor_radius: int,
    scope_start_sec: float | None,
    scope_end_sec: float | None,
) -> set[int]:
    assert artifacts.l2_segments is not None
    selected: set[int] = set()
    for frame_hit in frame_hits:
        frame_index = int(frame_hit.frame_index)
        for segment_index, segment in enumerate(artifacts.l2_segments):
            if int(segment.start_index) <= frame_index <= int(segment.end_index):
                low = max(0, segment_index - int(neighbor_radius))
                high = min(len(artifacts.l2_segments) - 1, segment_index + int(neighbor_radius))
                for neighbor_index in range(low, high + 1):
                    neighbor = artifacts.l2_segments[neighbor_index]
                    if _segment_in_scope(
                        neighbor.start_time_sec,
                        neighbor.end_time_sec,
                        scope_start_sec,
                        scope_end_sec,
                    ):
                        selected.add(neighbor_index)
                break
    return selected


def _candidate_l2_indices_from_l3_hits(
    artifacts: Any,
    l3_hits: list[Any],
    *,
    scope_start_sec: float | None,
    scope_end_sec: float | None,
) -> set[int]:
    assert artifacts.l2_segments is not None
    selected: set[int] = set()
    for segment_index, l2_segment in enumerate(artifacts.l2_segments):
        if not _segment_in_scope(l2_segment.start_time_sec, l2_segment.end_time_sec, scope_start_sec, scope_end_sec):
            continue
        if any(
            _segment_time_overlap(
                l2_segment.start_time_sec,
                l2_segment.end_time_sec,
                l3_hit.start_time_sec,
                l3_hit.end_time_sec,
            )
            for l3_hit in l3_hits
        ):
            selected.add(segment_index)
    return selected


def _score_l2_candidates(
    *,
    artifacts: Any,
    candidate_indices: set[int],
    query_embedding: torch.Tensor,
    frame_hits: list[Any],
    l3_hits: list[Any],
    scope_start_sec: float | None,
    scope_end_sec: float | None,
    source: str,
    gold_spans: list[dict[str, Any]] | None = None,
    oracle: bool = False,
    max_keep: int = max(DEFAULT_RECALL_K),
) -> list[dict[str, Any]]:
    assert artifacts.l2_segments is not None
    if not candidate_indices:
        return []

    frame_scores = torch.matmul(artifacts.frame_embeddings, query_embedding).cpu()
    l2_raw_items: list[tuple[int, float]] = []
    for segment_index in sorted(candidate_indices):
        segment = artifacts.l2_segments[segment_index]
        l2_raw_items.append(
            (
                segment_index,
                _segment_topm_score(
                    frame_scores=frame_scores,
                    start_index=int(segment.start_index),
                    end_index=int(segment.end_index),
                    top_m=L2_SCORE_TOP_M,
                ),
            )
        )
    l2_rank = _rank_score_by_id(l2_raw_items, key_fn=lambda item: float(item[1]))

    frame_rank_items = [
        (int(hit.frame_index), float(hit.score))
        for hit in sorted(frame_hits, key=lambda item: float(item.score), reverse=True)
    ]
    frame_rank = _rank_score_by_id(frame_rank_items, key_fn=lambda item: float(item[1]))

    l3_rank_items = [
        (str(hit.segment_id), float(hit.score))
        for hit in sorted(l3_hits, key=lambda item: float(item.score), reverse=True)
    ]
    l3_rank = _rank_score_by_id(l3_rank_items, key_fn=lambda item: float(item[1]))

    hits: list[dict[str, Any]] = []
    for segment_index in sorted(candidate_indices):
        segment = artifacts.l2_segments[segment_index]
        contained_frame_scores = [
            frame_rank.get(int(frame_hit.frame_index), 0.0)
            for frame_hit in frame_hits
            if int(segment.start_index) <= int(frame_hit.frame_index) <= int(segment.end_index)
        ]
        l1_component = max(contained_frame_scores, default=0.0)
        overlapping_l3_scores = [
            l3_rank.get(str(l3_hit.segment_id), 0.0)
            for l3_hit in l3_hits
            if _segment_time_overlap(
                segment.start_time_sec,
                segment.end_time_sec,
                l3_hit.start_time_sec,
                l3_hit.end_time_sec,
            )
        ]
        l3_component = max(overlapping_l3_scores, default=0.0)
        l2_component = float(l2_rank.get(segment_index, 0.0))
        score = (1.0 * l1_component) + (0.25 * l2_component) + (0.25 * l3_component)
        hit = _hit_dict(
            start_time_sec=float(segment.start_time_sec),
            end_time_sec=float(segment.end_time_sec),
            score=score,
            scope_start_sec=scope_start_sec,
            scope_end_sec=scope_end_sec,
            source=source,
            segment_id=str(segment.segment_id),
            components={
                "l1_rank": l1_component,
                "l2_query_rank": l2_component,
                "l3_overlap_rank": l3_component,
            },
        )
        if oracle and gold_spans is not None:
            hit["oracle_coverage"] = _best_coverage(hit, gold_spans)
        hits.append(hit)

    if oracle:
        hits.sort(key=lambda item: (float(item.get("oracle_coverage", 0.0)), float(item["score"])), reverse=True)
    else:
        hits.sort(key=lambda item: float(item["score"]), reverse=True)
    return hits[: max_keep]


def _dedupe_l2_supplements(
    l2_hits: list[dict[str, Any]],
    *,
    kept_hits: list[dict[str, Any]],
    max_keep: int,
) -> list[dict[str, Any]]:
    supplements: list[dict[str, Any]] = []
    kept_centers = [
        (float(hit["start_time_sec"]) + float(hit["end_time_sec"])) / 2.0
        for hit in kept_hits
    ]
    for hit in l2_hits:
        center = (float(hit["start_time_sec"]) + float(hit["end_time_sec"])) / 2.0
        if any(abs(center - kept_center) < L1_NMS_SEC for kept_center in kept_centers):
            continue
        supplements.append(hit)
        kept_centers.append(center)
        if len(supplements) >= max_keep:
            break
    return supplements

