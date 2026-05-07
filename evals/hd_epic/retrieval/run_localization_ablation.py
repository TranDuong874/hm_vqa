from __future__ import annotations

import argparse
import gc
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from evals.hd_epic.paths import HD_EPIC_P01_DERIVED_ROOT, HD_EPIC_P01_OPENCLIP_ROOT, RAW_VIDEO_ROOT, RESULTS_ROOT, REPO_ROOT
from evals.common.retrieval_ablation_runner import (
    AblationRetriever,
    AblationRunConfig,
)
from evals.common.vlm_baseline_runner import BaselineExample, _append_jsonl, _log_line, _write_json
from ingestion.viclip import ViCLIPEncoder
from evals.hd_epic.dataset import (
    TemporalExample,
    example_scope_for_video,
    gold_spans_for_video,
    load_temporal_examples_for_video,
)
from evals.common.video_sampling import sample_uniform_video_frames as _sample_uniform_video_frames
from retrieval import (
    collect_segment_frame_indices,
    adapt_query_embedding_for_segment_pooling,
    retrieve_top_segments,
    retrieve_top_segments_from_frame_scores,
)
from .examples import (
    _build_examples,
    _build_examples_from_manifest,
    _load_manifest_rows,
    _lookup_temporal_example,
    _participant_video_ids,
    _temporal_examples_by_video,
)
from .intervals import (
    _fixed_windows_from_frame_hits,
    _frame_interval,
    _legacy_l1_bundles,
    _segment_in_scope,
    _temporal_nms_frame_hits,
)
from .metrics import _best_coverage, _metrics_for_hits, _summarize
from .queries import _extract_target_text, _query_texts_for_method
from .rerank import (
    _candidate_l2_indices_from_frame_hits,
    _candidate_l2_indices_from_l3_hits,
    _dedupe_l2_supplements,
    _rerank_l3_hits_with_decomposed_l1,
    _rerank_l3_hits_with_decomposed_l1_l2,
    _rerank_l3_hits_with_l2,
    _score_l2_candidates,
)


DEFAULT_VIDEO_ROOT = RAW_VIDEO_ROOT
DEFAULT_FEATURE_ROOT = HD_EPIC_P01_OPENCLIP_ROOT
DEFAULT_DERIVED_CACHE_ROOT = HD_EPIC_P01_DERIVED_ROOT
DEFAULT_OUTPUT_ROOT = RESULTS_ROOT / "localization_retrieval_p01"
DEFAULT_TASKS = ("fine_grained_action_localization",)
DEFAULT_COVERAGE_THRESHOLD = 0.6
DEFAULT_RECALL_K = (1, 3, 5)
L1_RAW_K = 50
L1_ANCHOR_K = 5
L1_NMS_SEC = 2.0
L2_NEIGHBOR_RADIUS = 1
L2_SCORE_TOP_M = 4
FIXED_CONTEXT_SECONDS = 5.0
SUPPLEMENT_METHODS = {
    "l1_to_l2",
    "l3_to_l2",
    "l3_rerank_l2",
    "l3_l1_to_l2",
    "l3_l1_to_l2_oracle",
    "l3_to_l1_plus_l2",
    "l3_to_l1_plus_l2_oracle",
    "l3_l1_to_l2_prompt",
    "l3_rerank_l1_decomp_l2",
}
L1_PROPOSAL_METHODS = {
    "l1",
    "l3_to_l1",
    "l3_to_l1_prompt",
    "l3_rerank_l1_decomp",
    "l3_rerank_l1_decomp_l2",
    "l1_to_l2",
    "l3_l1_to_l2",
    "l3_l1_to_l2_oracle",
    "l3_to_l1_plus_l2",
    "l3_to_l1_plus_l2_oracle",
    "l3_l1_to_l2_prompt",
}
PROMPT_QUERY_METHODS = {
    "l3_to_l1_prompt",
    "l3_l1_to_l2_prompt",
}
PLUS_L2_BASELINE_SLOTS = 3
PLUS_L2_SUPPLEMENT_SLOTS = 2
TARGET_PATTERN = re.compile(r"<([^>]+)>")
_VICLIP_ENCODER: ViCLIPEncoder | None = None
VICLIP_L2_MAX_FRAMES = 16
PHRASAL_VERB_PARTICLES = {
    "up",
    "down",
    "on",
    "off",
    "out",
    "in",
    "into",
    "onto",
    "over",
    "under",
    "back",
    "away",
    "through",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run retrieval-only HD-EPIC localization ablations.")
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--derived-cache-root", type=Path, default=DEFAULT_DERIVED_CACHE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--participant", default="P01")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional HD-EPIC manifest JSON with rows[{example_id, video_id}] to restrict evaluation.",
    )
    parser.add_argument(
        "--method",
        choices=[
            "l1",
            "l2",
            "l3",
            "l3_rerank_l2",
            "l3_rerank_l1_decomp",
            "l3_rerank_l1_decomp_l2",
            "l3_to_l1",
            "l3_to_l1_prompt",
            "l1_to_l2",
            "l3_to_l2",
            "l3_l1_to_l2",
            "l3_l1_to_l2_oracle",
            "l3_to_l1_plus_l2",
            "l3_to_l1_plus_l2_oracle",
            "l3_l1_to_l2_prompt",
        ],
        required=True,
    )
    parser.add_argument("--l2-window-seconds", type=float, default=5.0)
    parser.add_argument("--l2-stride-seconds", type=float, default=5.0)
    parser.add_argument("--l2-segmentation", choices=["fixed", "l3_local_contrast"], default="fixed")
    parser.add_argument("--l2-local-min-duration-sec", type=float, default=3.0)
    parser.add_argument("--l2-local-max-duration-sec", type=float, default=12.0)
    parser.add_argument("--l2-local-fast-kernel-size", type=int, default=1)
    parser.add_argument("--l2-local-slow-kernel-size", type=int, default=9)
    parser.add_argument("--l2-local-peak-percentile", type=float, default=75.0)
    parser.add_argument(
        "--feature-eval-fps",
        type=float,
        default=1.0,
        help="Downsample cached frame embeddings for retrieval evaluation. Use 1.0 to match prior HD-EPIC retrieval runs.",
    )
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--top-l2-segments", type=int, default=max(DEFAULT_RECALL_K))
    parser.add_argument("--top-l3-segments", type=int, default=max(DEFAULT_RECALL_K))
    parser.add_argument("--l2-rerank-encoder", choices=["openclip", "viclip"], default="openclip")
    parser.add_argument("--l3-segmentation", choices=["fused_adaptive", "fixed"], default="fused_adaptive")
    parser.add_argument("--l3-window-seconds", type=float, default=60.0)
    parser.add_argument("--l3-stride-seconds", type=float, default=60.0)
    parser.add_argument("--coverage-threshold", type=float, default=DEFAULT_COVERAGE_THRESHOLD)
    parser.add_argument("--limit-videos", type=int, default=None)
    parser.add_argument("--limit-examples", type=int, default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()






























































def _segment_hits_for_method(
    method: str,
    retrieval_info: dict[str, Any],
    sample_fps: float,
    *,
    scope_start_sec: float | None,
    scope_end_sec: float | None,
    l2_window_seconds: float,
) -> list[dict[str, float]]:
    if "hits" in retrieval_info:
        return retrieval_info["hits"]
    if method == "l1":
        return _legacy_l1_bundles(
            retrieval_info["frame_hits"],
            max_keep=max(DEFAULT_RECALL_K),
            max_gap_sec=5.0,
            half_window_sec=float(l2_window_seconds) / 2.0,
            scope_start_sec=scope_start_sec,
            scope_end_sec=scope_end_sec,
        )
    if method == "l2":
        key = "l2_hits"
    else:
        key = "l3_hits"
    return [
        {
            "start_time_sec": float(hit.start_time_sec),
            "end_time_sec": float(hit.end_time_sec),
            "score": float(hit.score),
        }
        for hit in retrieval_info[key]
    ]




def _scoped_retrieve(
    retriever: AblationRetriever,
    *,
    example: BaselineExample,
    scope_start_sec: float | None,
    scope_end_sec: float | None,
    gold_spans: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    artifacts = retriever._load_video(example)
    method = retriever.config.method
    routing_method = method.removesuffix("_prompt")
    query_text = _extract_target_text(example.question)
    query_embeddings = retriever._encoder.encode_texts(_query_texts_for_method(query_text, method)).cpu()
    query_embedding = torch.nn.functional.normalize(query_embeddings.mean(dim=0), dim=0)
    pooled_query_embedding = adapt_query_embedding_for_segment_pooling(query_embedding, pooling="mean")

    scoped_frame_indices = [
        idx
        for idx, time_sec in enumerate(artifacts.timestamps.tolist())
        if _segment_in_scope(float(time_sec), float(time_sec), scope_start_sec, scope_end_sec)
    ]

    def _retrieve_l1(allowed_indices: list[int] | None = None) -> list[Any]:
        frame_hits = retriever._frame_hits_from_indices(
            artifacts=artifacts,
            query_embedding=query_embedding,
            allowed_indices=allowed_indices if allowed_indices is not None else scoped_frame_indices,
        )
        return frame_hits

    def _retrieve_l2_global() -> list[Any]:
        retriever._ensure_l2(artifacts)
        assert artifacts.l2_segments is not None
        assert artifacts.l2_embeddings is not None
        candidate_indices = [
            idx
            for idx, segment in enumerate(artifacts.l2_segments)
            if _segment_in_scope(segment.start_time_sec, segment.end_time_sec, scope_start_sec, scope_end_sec)
        ]
        segments = [artifacts.l2_segments[idx] for idx in candidate_indices]
        embeddings = artifacts.l2_embeddings[torch.tensor(candidate_indices, dtype=torch.long)] if candidate_indices else artifacts.l2_embeddings[:0]
        if retriever.config.l2_scoring == "embedding":
            l2_hits = retrieve_top_segments(
                query_embedding=pooled_query_embedding,
                segment_embeddings=embeddings,
                segments=segments,
                top_k=retriever.config.top_l2_segments,
            )
        else:
            l2_hits = retrieve_top_segments_from_frame_scores(
                query_embedding=query_embedding,
                frame_embeddings=artifacts.frame_embeddings,
                segments=segments,
                top_k=retriever.config.top_l2_segments,
                top_m=retriever.config.l2_frame_score_top_m,
                aggregation=retriever.config.l2_scoring,
                temperature=retriever.config.l2_frame_score_temperature,
            )
        return {"frame_hits": [], "l2_hits": l2_hits, "l3_hits": []}

    def _retrieve_l3() -> list[Any]:
        retriever._ensure_l3(artifacts)
        assert artifacts.l3_segments is not None
        assert artifacts.l3_embeddings is not None
        candidate_indices = [
            idx
            for idx, segment in enumerate(artifacts.l3_segments)
            if _segment_in_scope(segment.start_time_sec, segment.end_time_sec, scope_start_sec, scope_end_sec)
        ]
        segments = [artifacts.l3_segments[idx] for idx in candidate_indices]
        embeddings = (
            artifacts.l3_embeddings[torch.tensor(candidate_indices, dtype=torch.long)]
            if candidate_indices
            else artifacts.l3_embeddings[:0]
        )
        return retrieve_top_segments(
            query_embedding=pooled_query_embedding,
            segment_embeddings=embeddings,
            segments=segments,
            top_k=retriever.config.top_l3_segments,
        )

    if routing_method == "l1":
        frame_hits = _retrieve_l1()
        return {"frame_hits": frame_hits, "l2_hits": [], "l3_hits": []}

    if routing_method == "l2":
        return _retrieve_l2_global()

    if routing_method == "l3":
        l3_hits = _retrieve_l3()
        return {"frame_hits": [], "l2_hits": [], "l3_hits": l3_hits}

    if routing_method == "l3_rerank_l2":
        l3_hits = _retrieve_l3()
        retriever._ensure_l2(artifacts)
        hits, debug = _rerank_l3_hits_with_l2(
            retriever=retriever,
            artifacts=artifacts,
            query_embedding=query_embedding,
            target_text=query_text,
            l3_hits=l3_hits,
            scope_start_sec=scope_start_sec,
            scope_end_sec=scope_end_sec,
        )
        return {"hits": hits, "frame_hits": [], "l2_hits": [], "l3_hits": l3_hits, "debug": debug}

    if routing_method == "l3_rerank_l1_decomp":
        l3_hits = _retrieve_l3()
        allowed_indices = collect_segment_frame_indices(l3_hits)
        hits, debug = _rerank_l3_hits_with_decomposed_l1(
            retriever=retriever,
            artifacts=artifacts,
            target_text=query_text,
            l3_hits=l3_hits,
            allowed_indices=allowed_indices,
            scope_start_sec=scope_start_sec,
            scope_end_sec=scope_end_sec,
        )
        frame_hits = _retrieve_l1(allowed_indices=allowed_indices) if allowed_indices else []
        return {
            "hits": hits,
            "frame_hits": frame_hits,
            "l2_hits": [],
            "l3_hits": l3_hits,
            "debug": debug,
        }

    if routing_method == "l3_rerank_l1_decomp_l2":
        l3_hits = _retrieve_l3()
        allowed_indices = collect_segment_frame_indices(l3_hits)
        hits, debug = _rerank_l3_hits_with_decomposed_l1_l2(
            retriever=retriever,
            artifacts=artifacts,
            target_text=query_text,
            query_embedding=query_embedding,
            l3_hits=l3_hits,
            allowed_indices=allowed_indices,
            scope_start_sec=scope_start_sec,
            scope_end_sec=scope_end_sec,
        )
        frame_hits = _retrieve_l1(allowed_indices=allowed_indices) if allowed_indices else []
        return {
            "hits": hits,
            "frame_hits": frame_hits,
            "l2_hits": [],
            "l3_hits": l3_hits,
            "debug": debug,
        }

    if routing_method == "l3_to_l1":
        l3_hits = _retrieve_l3()
        allowed_indices = collect_segment_frame_indices(l3_hits)
        frame_hits = _retrieve_l1(allowed_indices=allowed_indices)
        anchor_hits = _temporal_nms_frame_hits(
            frame_hits,
            max_hits=L1_ANCHOR_K,
            min_gap_sec=L1_NMS_SEC,
        )
        hits = _fixed_windows_from_frame_hits(
            anchor_hits,
            max_keep=max(DEFAULT_RECALL_K),
            window_seconds=FIXED_CONTEXT_SECONDS,
            scope_start_sec=scope_start_sec,
            scope_end_sec=scope_end_sec,
            source="l3_to_l1_fixed_context",
        )
        return {"hits": hits, "frame_hits": frame_hits, "l2_hits": [], "l3_hits": l3_hits}

    if routing_method == "l1_to_l2":
        frame_hits = _retrieve_l1()
        retriever._ensure_l2(artifacts)
        candidate_indices = _candidate_l2_indices_from_frame_hits(
            artifacts,
            frame_hits,
            neighbor_radius=L2_NEIGHBOR_RADIUS,
            scope_start_sec=scope_start_sec,
            scope_end_sec=scope_end_sec,
        )
        hits = _score_l2_candidates(
            artifacts=artifacts,
            candidate_indices=candidate_indices,
            query_embedding=query_embedding,
            frame_hits=frame_hits,
            l3_hits=[],
            scope_start_sec=scope_start_sec,
            scope_end_sec=scope_end_sec,
            source="l1_to_l2_fixed",
        )
        return {"hits": hits, "frame_hits": frame_hits, "l2_hits": [], "l3_hits": []}

    if routing_method == "l3_to_l2":
        l3_hits = _retrieve_l3()
        retriever._ensure_l2(artifacts)
        candidate_indices = _candidate_l2_indices_from_l3_hits(
            artifacts,
            l3_hits,
            scope_start_sec=scope_start_sec,
            scope_end_sec=scope_end_sec,
        )
        hits = _score_l2_candidates(
            artifacts=artifacts,
            candidate_indices=candidate_indices,
            query_embedding=query_embedding,
            frame_hits=[],
            l3_hits=l3_hits,
            scope_start_sec=scope_start_sec,
            scope_end_sec=scope_end_sec,
            source="l3_to_l2_fixed",
        )
        return {"hits": hits, "frame_hits": [], "l2_hits": [], "l3_hits": l3_hits}

    if routing_method in {"l3_l1_to_l2", "l3_l1_to_l2_oracle"}:
        l3_hits = _retrieve_l3()
        allowed_indices = collect_segment_frame_indices(l3_hits)
        frame_hits = _retrieve_l1(allowed_indices=allowed_indices)
        retriever._ensure_l2(artifacts)
        candidate_indices = _candidate_l2_indices_from_frame_hits(
            artifacts,
            frame_hits,
            neighbor_radius=L2_NEIGHBOR_RADIUS,
            scope_start_sec=scope_start_sec,
            scope_end_sec=scope_end_sec,
        )
        candidate_indices.update(
            _candidate_l2_indices_from_l3_hits(
                artifacts,
                l3_hits,
                scope_start_sec=scope_start_sec,
                scope_end_sec=scope_end_sec,
            )
        )
        hits = _score_l2_candidates(
            artifacts=artifacts,
            candidate_indices=candidate_indices,
            query_embedding=query_embedding,
            frame_hits=frame_hits,
            l3_hits=l3_hits,
            scope_start_sec=scope_start_sec,
            scope_end_sec=scope_end_sec,
            source=("l3_l1_to_l2_oracle" if routing_method == "l3_l1_to_l2_oracle" else "l3_l1_to_l2_fixed"),
            gold_spans=gold_spans,
            oracle=(routing_method == "l3_l1_to_l2_oracle"),
        )
        return {"hits": hits, "frame_hits": frame_hits, "l2_hits": [], "l3_hits": l3_hits}

    if routing_method in {"l3_to_l1_plus_l2", "l3_to_l1_plus_l2_oracle"}:
        l3_hits = _retrieve_l3()
        allowed_indices = collect_segment_frame_indices(l3_hits)
        frame_hits = _retrieve_l1(allowed_indices=allowed_indices)
        anchor_hits = _temporal_nms_frame_hits(
            frame_hits,
            max_hits=L1_ANCHOR_K,
            min_gap_sec=L1_NMS_SEC,
        )
        baseline_hits = _fixed_windows_from_frame_hits(
            anchor_hits,
            max_keep=max(DEFAULT_RECALL_K),
            window_seconds=FIXED_CONTEXT_SECONDS,
            scope_start_sec=scope_start_sec,
            scope_end_sec=scope_end_sec,
            source="l3_to_l1_fixed_context",
        )

        retriever._ensure_l2(artifacts)
        candidate_indices = _candidate_l2_indices_from_frame_hits(
            artifacts,
            frame_hits,
            neighbor_radius=L2_NEIGHBOR_RADIUS,
            scope_start_sec=scope_start_sec,
            scope_end_sec=scope_end_sec,
        )
        candidate_indices.update(
            _candidate_l2_indices_from_l3_hits(
                artifacts,
                l3_hits,
                scope_start_sec=scope_start_sec,
                scope_end_sec=scope_end_sec,
            )
        )
        l2_hits = _score_l2_candidates(
            artifacts=artifacts,
            candidate_indices=candidate_indices,
            query_embedding=query_embedding,
            frame_hits=frame_hits,
            l3_hits=l3_hits,
            scope_start_sec=scope_start_sec,
            scope_end_sec=scope_end_sec,
            source=(
                "l3_to_l1_plus_l2_oracle"
                if routing_method == "l3_to_l1_plus_l2_oracle"
                else "l3_to_l1_plus_l2_fixed"
            ),
            gold_spans=gold_spans,
            oracle=(routing_method == "l3_to_l1_plus_l2_oracle"),
            max_keep=20,
        )
        kept_baseline = baseline_hits[:PLUS_L2_BASELINE_SLOTS]
        l2_supplements = _dedupe_l2_supplements(
            l2_hits,
            kept_hits=kept_baseline,
            max_keep=PLUS_L2_SUPPLEMENT_SLOTS,
        )
        hits = (kept_baseline + l2_supplements)[: max(DEFAULT_RECALL_K)]
        return {"hits": hits, "frame_hits": frame_hits, "l2_hits": [], "l3_hits": l3_hits}

    raise ValueError(f"Unsupported method: {method}")














def _output_name(config: AblationRunConfig, *, coverage_threshold: float) -> str:
    name = config.method
    if config.method in {"l3", "l3_rerank_l2"} and config.l3_segmentation == "fixed":
        name += f"_l3fixed{config.l3_window_seconds:g}s_s{config.l3_stride_seconds:g}s"
    if config.top_l3_segments != max(DEFAULT_RECALL_K):
        name += f"_l3k{config.top_l3_segments:g}"
    if config.top_l2_segments != max(DEFAULT_RECALL_K):
        name += f"_l2k{config.top_l2_segments:g}"
    if config.method in {"l2", *SUPPLEMENT_METHODS}:
        if config.l2_segmentation == "fixed":
            name += f"_l2w{config.l2_window_seconds:g}_l2s{config.l2_stride_seconds:g}"
        else:
            name += (
                f"_l2{config.l2_segmentation}"
                f"_min{config.l2_local_min_duration_sec:g}"
                f"_max{config.l2_local_max_duration_sec:g}"
                f"_p{config.l2_local_peak_percentile:g}"
            )
        if config.l2_scoring != "embedding":
            name += f"_score{config.l2_scoring}"
    if config.method == "l3_rerank_l2" and config.l2_rerank_encoder != "openclip":
        name += f"_l2enc{config.l2_rerank_encoder}"
    return f"{name}_cov{str(coverage_threshold).replace('.', 'p')}"


def main() -> None:
    args = _parse_args()

    if args.method in L1_PROPOSAL_METHODS:
        max_frames = L1_RAW_K
        l2_window_seconds = 5.0
        l2_stride_seconds = 5.0
        l2_segmentation = "fixed"
    elif args.method == "l2":
        max_frames = max(DEFAULT_RECALL_K)
        l2_window_seconds = args.l2_window_seconds
        l2_stride_seconds = args.l2_stride_seconds
        l2_segmentation = args.l2_segmentation
    elif args.method == "l3_rerank_l2":
        max_frames = max(DEFAULT_RECALL_K)
        l2_window_seconds = args.l2_window_seconds
        l2_stride_seconds = args.l2_stride_seconds
        l2_segmentation = args.l2_segmentation
    else:
        max_frames = max(DEFAULT_RECALL_K)
        l2_window_seconds = 5.0
        l2_stride_seconds = 5.0
        l2_segmentation = "fixed"

    run_config = AblationRunConfig(
        method=args.method,
        sample_fps=args.sample_fps,
        feature_eval_fps=args.feature_eval_fps,
        max_frames=max_frames,
        image_max_size=336,
        include_subtitles=False,
        l2_window_seconds=l2_window_seconds,
        l2_stride_seconds=l2_stride_seconds,
        l2_segmentation=l2_segmentation,
        l2_local_min_duration_sec=args.l2_local_min_duration_sec,
        l2_local_max_duration_sec=args.l2_local_max_duration_sec,
        l2_local_fast_kernel_size=args.l2_local_fast_kernel_size,
        l2_local_slow_kernel_size=args.l2_local_slow_kernel_size,
        l2_local_peak_percentile=args.l2_local_peak_percentile,
        top_l2_segments=args.top_l2_segments,
        top_l3_segments=args.top_l3_segments,
        l2_scoring=args.l2_scoring if hasattr(args, "l2_scoring") else "embedding",
        l2_frame_score_top_m=args.l2_frame_score_top_m if hasattr(args, "l2_frame_score_top_m") else 4,
        l2_frame_score_temperature=args.l2_frame_score_temperature if hasattr(args, "l2_frame_score_temperature") else 0.07,
        l2_rerank_encoder=args.l2_rerank_encoder,
        l3_segmentation=args.l3_segmentation,
        l3_window_seconds=args.l3_window_seconds,
        l3_stride_seconds=args.l3_stride_seconds,
    )

    output_dir = args.output_root / _output_name(run_config, coverage_threshold=args.coverage_threshold)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "rows.jsonl"
    progress_path = output_dir / "progress.log"
    error_path = output_dir / "error.log"
    rolling_summary_path = output_dir / "rolling_summary.json"
    if not args.resume:
        for path in (rows_path, progress_path, error_path, rolling_summary_path, output_dir / "final_summary.json"):
            if path.exists():
                path.unlink()

    manifest_rows: list[dict[str, Any]] | None = None
    if args.manifest is not None:
        manifest_rows = _load_manifest_rows(args.manifest)
        if args.limit_examples is not None:
            manifest_rows = manifest_rows[: max(int(args.limit_examples), 0)]
        if args.limit_videos is not None:
            allowed_videos = sorted({str(row["video_id"]) for row in manifest_rows})[: max(int(args.limit_videos), 0)]
            allowed_video_set = set(allowed_videos)
            manifest_rows = [row for row in manifest_rows if str(row["video_id"]) in allowed_video_set]
        video_ids = sorted({str(row["video_id"]) for row in manifest_rows})
        examples = _build_examples_from_manifest(video_root=args.video_root, manifest_rows=manifest_rows)
    else:
        video_ids = _participant_video_ids(args.feature_root, args.participant)
        if args.limit_videos is not None:
            video_ids = video_ids[: max(int(args.limit_videos), 0)]
        examples = _build_examples(video_root=args.video_root, video_ids=video_ids)
    if args.limit_examples is not None:
        examples = examples[: max(int(args.limit_examples), 0)]
    temporal_examples = _temporal_examples_by_video(video_ids)

    existing_rows: list[dict[str, Any]] = []
    if args.resume and rows_path.exists():
        with rows_path.open("r", encoding="utf-8") as handle:
            existing_rows = [json.loads(line) for line in handle if line.strip()]
    completed_ids = {str(row["example_id"]) for row in existing_rows}
    pending_examples = [example for example in examples if example.example_id not in completed_ids]

    _log_line(
        progress_path,
        f"[start] total={len(examples)} pending={len(pending_examples)} method={run_config.method} cov={args.coverage_threshold}",
    )
    _write_json(
        output_dir / "config.json",
        {
            "participant": args.participant,
            "manifest": str(args.manifest) if args.manifest is not None else None,
            "feature_root": str(args.feature_root),
            "video_root": str(args.video_root),
            "derived_cache_root": str(args.derived_cache_root),
            "coverage_threshold": args.coverage_threshold,
            "feature_eval_fps": args.feature_eval_fps,
            "recall_k": list(DEFAULT_RECALL_K),
            "run_config": asdict(run_config),
        },
    )
    _write_json(rolling_summary_path, _summarize(existing_rows, len(examples)))

    retriever = AblationRetriever(
        feature_root=args.feature_root,
        derived_cache_root=args.derived_cache_root,
        config=run_config,
        encoder_device="cuda",
    )
    rows = list(existing_rows)
    try:
        for index, example in enumerate(pending_examples, start=len(existing_rows) + 1):
            _log_line(progress_path, f"[item_start] index={index}/{len(examples)} example_id={example.example_id} video={example.video_id}")
            try:
                temporal_example = _lookup_temporal_example(
                    temporal_examples,
                    video_id=example.video_id,
                    example_id=example.example_id,
                )
                scope_start_sec, scope_end_sec = example_scope_for_video(temporal_example, example.video_id)
                gold_spans = gold_spans_for_video(temporal_example, example.video_id)
                retrieval_info = _scoped_retrieve(
                    retriever,
                    example=example,
                    scope_start_sec=scope_start_sec,
                    scope_end_sec=scope_end_sec,
                    gold_spans=gold_spans,
                )
                hits = _segment_hits_for_method(
                    run_config.method,
                    retrieval_info,
                    run_config.sample_fps,
                    scope_start_sec=scope_start_sec,
                    scope_end_sec=scope_end_sec,
                    l2_window_seconds=run_config.l2_window_seconds,
                )
                metrics = _metrics_for_hits(
                    hits=hits,
                    gold_spans=gold_spans,
                    coverage_threshold=args.coverage_threshold,
                )
                row = {
                    "example_id": example.example_id,
                    "video_id": example.video_id,
                    "task_name": temporal_example.task_name,
                    "question": example.question,
                    "scope_start_sec": scope_start_sec,
                    "scope_end_sec": scope_end_sec,
                    "gold_spans": gold_spans,
                    "retrieved_hits": hits,
                    "metrics": metrics,
                }
                if "debug" in retrieval_info:
                    row["retrieval_debug"] = retrieval_info["debug"]
                rows.append(row)
                _append_jsonl(rows_path, row)
                _write_json(rolling_summary_path, _summarize(rows, len(examples)))
                _log_line(
                    progress_path,
                    f"[item_done] index={index}/{len(examples)} example_id={example.example_id} cov5={metrics['best_coverage_at_5']:.3f} cov_r5={metrics['coverage_recall_at_5']:.0f}",
                )
            except Exception as exc:
                if isinstance(exc, RuntimeError) and "Failed to open video:" in str(exc):
                    _log_line(
                        progress_path,
                        f"[item_skip] index={index}/{len(examples)} example_id={example.example_id} reason=missing_video",
                    )
                    with error_path.open("a", encoding="utf-8") as handle:
                        handle.write(f"{example.example_id}: SKIPPED missing_video: {exc}\n")
                    continue
                _log_line(progress_path, f"[item_error] index={index}/{len(examples)} example_id={example.example_id} error={type(exc).__name__}: {exc}")
                with error_path.open("a", encoding="utf-8") as handle:
                    handle.write(f"{example.example_id}: {type(exc).__name__}: {exc}\n")
                raise
    finally:
        retriever.close()

    final_summary = {
        "participant": args.participant,
        "method": run_config.method,
        "window_seconds": run_config.l2_window_seconds,
        "stride_seconds": run_config.l2_stride_seconds,
        "coverage_threshold": args.coverage_threshold,
        **_summarize(rows, len(examples)),
    }
    _write_json(output_dir / "final_summary.json", final_summary)
    _log_line(
        progress_path,
        f"[done] scored={final_summary['scored']} mean_cov5={final_summary['mean_best_coverage_at_5']:.4f} cov_r5={final_summary['coverage_recall_at_5']:.4f}",
    )


if __name__ == "__main__":
    main()
