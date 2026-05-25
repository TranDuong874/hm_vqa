from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from answering.factory import build_answerer
from answering.qwen_vl import AnswerConfig
from hm_vqa.config import load_retrieval_policy
from hm_vqa.policies import RetrievalPolicy
from evals.common.vlm_baseline_runner import (
    BaselineExample,
    _append_jsonl,
    _is_api_content_filter_error,
    _load_resume_rows,
    _load_subtitles,
    _log_line,
    _merge_frame_texts,
    _rewrite_jsonl,
    _subtitle_texts_for_frames,
    _summarize_rows,
    _write_json,
)
from ingestion import OpenCLIPEncoder
from ingestion.viclip import ViCLIPEncoder
from evals.common.fused_adaptive import segment_fused_adaptive_peaks
from evals.common.video_sampling import sample_uniform_video_frames as _sample_uniform_video_frames
from retrieval.faiss_index import load_or_build_ip_index, search_ip_index, write_ip_index
from retrieval import (
    adapt_query_embedding_for_segment_pooling,
    build_query_text,
    build_window_segments,
    collect_segment_frame_indices,
    load_selected_video_frames,
    pool_segments,
    retrieve_top_frames,
    retrieve_top_segments,
    retrieve_top_segments_from_frame_scores,
)
from retrieval.types import FrameHit, PipelineConfig, SegmentHit
from segmentation import Segment, segment_fixed_windows, segment_l3_local_contrast_windows
from segmentation.video import compute_motion_energy_for_frame_indices


DEFAULT_L3_RERANK_KEEP = 5
L2_SCORE_TOP_M = 4
VICLIP_L2_MAX_FRAMES = 16
VICLIP_BATCH_SIZE = int(os.environ.get("HM_VQA_VICLIP_BATCH_SIZE", "8"))
DEFAULT_L2_EVIDENCE_PER_L3 = 2
DEFAULT_L1_EVIDENCE_PER_L2 = 4
TARGET_PATTERN = re.compile(r"<([^<>]+)>")
_VICLIP_ENCODER: ViCLIPEncoder | None = None


@dataclass(slots=True)
class AblationRunConfig(RetrievalPolicy):
    """Compatibility name for historical runners.

    The canonical policy type is `hm_vqa.policies.RetrievalPolicy`; old output
    schemas and CLI code still refer to `AblationRunConfig`, so keep the alias
    while moving policy semantics into shared source code.
    """

    top_l2_segments: int = 3
    top_l3_segments: int = 2
    l3_segmentation: str = "fused_adaptive"
    l2_evidence_per_l3: int = DEFAULT_L2_EVIDENCE_PER_L3
    l1_evidence_per_l2: int = DEFAULT_L1_EVIDENCE_PER_L2


@dataclass(slots=True)
class VideoArtifacts:
    video_id: str
    video_path: Path
    timestamps: np.ndarray
    frame_embeddings: torch.Tensor
    native_fps: float
    l2_segments: list[Segment] | None = None
    l2_embeddings: torch.Tensor | None = None
    l3_segments: list[Segment] | None = None
    l3_embeddings: torch.Tensor | None = None
    faiss_indices: dict[str, Any] = field(default_factory=dict)


def add_retrieval_ablation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--retrieval-config", type=Path, default=None)
    parser.add_argument("--method", choices=["l1", "l2", "l3", "l3_rerank_l2", "l1_plus_l3_rerank_l2"], default=None)
    parser.add_argument(
        "--vector-backend",
        choices=["torch", "faiss"],
        default="torch",
        help="Backend for dense inner-product top-k search. Torch is the historical default; FAISS is optional.",
    )
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--feature-eval-fps", type=float, default=None)
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--image-max-size", type=int, default=336)
    parser.add_argument("--include-subtitles", action="store_true")
    parser.add_argument("--l2-window-seconds", type=float, default=5.0)
    parser.add_argument("--l2-stride-seconds", type=float, default=5.0)
    parser.add_argument("--l2-segmentation", choices=["fixed", "l3_local_contrast"], default="fixed")
    parser.add_argument("--l2-local-min-duration-sec", type=float, default=3.0)
    parser.add_argument("--l2-local-max-duration-sec", type=float, default=12.0)
    parser.add_argument("--l2-local-fast-kernel-size", type=int, default=1)
    parser.add_argument("--l2-local-slow-kernel-size", type=int, default=9)
    parser.add_argument("--l2-local-peak-percentile", type=float, default=75.0)
    parser.add_argument(
        "--l2-scoring",
        choices=["embedding", "topm_mean", "max", "logsumexp_mean", "softmax_mean"],
        default="embedding",
    )
    parser.add_argument("--l2-frame-score-top-m", type=int, default=4)
    parser.add_argument("--l2-frame-score-temperature", type=float, default=0.07)
    parser.add_argument("--top-l2-segments", type=int, default=3)
    parser.add_argument("--top-l3-segments", type=int, default=2)
    parser.add_argument("--l2-rerank-encoder", choices=["openclip", "viclip"], default="openclip")
    parser.add_argument(
        "--l2-rerank-query-mode",
        choices=["target", "full"],
        default="target",
        help="Text used for L2 reranking. 'target' preserves existing behavior; 'full' uses question plus options.",
    )
    parser.add_argument("--l3-rerank-keep", type=int, default=DEFAULT_L3_RERANK_KEEP)
    parser.add_argument(
        "--l3-rerank-evidence-source",
        choices=["reranked_l3", "top_l2", "top_l2_per_l3"],
        default="reranked_l3",
        help="For l3_rerank_l2, choose final evidence from reranked parent L3 ranges or top L2 windows inside them.",
    )
    parser.add_argument("--l2-evidence-per-l3", type=int, default=DEFAULT_L2_EVIDENCE_PER_L3)
    parser.add_argument("--l1-evidence-per-l2", type=int, default=DEFAULT_L1_EVIDENCE_PER_L2)
    parser.add_argument("--l3-segmentation", choices=["fused_adaptive", "fixed"], default="fused_adaptive")
    parser.add_argument("--l3-window-seconds", type=float, default=60.0)
    parser.add_argument("--l3-stride-seconds", type=float, default=60.0)
    parser.add_argument(
        "--evidence-text-mode",
        choices=["frames", "chunks"],
        default="frames",
        help="Format per-image text as flat frame labels or as grouped temporal evidence chunks.",
    )
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--backend", choices=["local", "api"], default="local")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--api-base-url", default=None)
    parser.add_argument("--api-key-env-var", default="ALIBABA_CLOUD_API")
    parser.add_argument("--api-requests-per-minute", type=int, default=60)
    parser.add_argument("--api-tokens-per-minute", type=int, default=100000)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")


def build_retrieval_run_config(args: argparse.Namespace) -> AblationRunConfig:
    if args.retrieval_config is not None:
        policy = load_retrieval_policy(args.retrieval_config)
        return AblationRunConfig(**asdict(policy))
    if args.method is None:
        raise ValueError("Either --method or --retrieval-config is required.")
    return AblationRunConfig(
        method=args.method,
        vector_backend=args.vector_backend,
        sample_fps=args.sample_fps,
        feature_eval_fps=args.feature_eval_fps,
        max_frames=args.max_frames,
        image_max_size=args.image_max_size,
        include_subtitles=args.include_subtitles,
        l2_window_seconds=args.l2_window_seconds,
        l2_stride_seconds=args.l2_stride_seconds,
        l2_segmentation=args.l2_segmentation,
        l2_local_min_duration_sec=args.l2_local_min_duration_sec,
        l2_local_max_duration_sec=args.l2_local_max_duration_sec,
        l2_local_fast_kernel_size=args.l2_local_fast_kernel_size,
        l2_local_slow_kernel_size=args.l2_local_slow_kernel_size,
        l2_local_peak_percentile=args.l2_local_peak_percentile,
        l2_scoring=args.l2_scoring,
        l2_frame_score_top_m=args.l2_frame_score_top_m,
        l2_frame_score_temperature=args.l2_frame_score_temperature,
        l2_rerank_encoder=args.l2_rerank_encoder,
        l2_rerank_query_mode=args.l2_rerank_query_mode,
        top_l2_segments=args.top_l2_segments,
        top_l3_segments=args.top_l3_segments,
        l3_rerank_keep=args.l3_rerank_keep,
        l3_rerank_evidence_source=args.l3_rerank_evidence_source,
        l2_evidence_per_l3=args.l2_evidence_per_l3,
        l1_evidence_per_l2=args.l1_evidence_per_l2,
        l3_segmentation=args.l3_segmentation,
        l3_window_seconds=args.l3_window_seconds,
        l3_stride_seconds=args.l3_stride_seconds,
        evidence_text_mode=args.evidence_text_mode,
    )


def build_retrieval_output_name(*, model_id: str, run_config: AblationRunConfig) -> str:
    output_name = f"{model_id.split('/')[-1]}_{run_config.method}"
    if run_config.method in {"l3", "l3_rerank_l2", "l1_plus_l3_rerank_l2"} and run_config.l3_segmentation == "fixed":
        output_name += f"_l3fixed{run_config.l3_window_seconds:g}s_s{run_config.l3_stride_seconds:g}s"
    if run_config.method in {"l2", "l3_rerank_l2", "l1_plus_l3_rerank_l2"}:
        if run_config.l2_segmentation == "fixed":
            output_name += f"_l2w{run_config.l2_window_seconds:g}_l2s{run_config.l2_stride_seconds:g}"
        else:
            output_name += (
                f"_l2{run_config.l2_segmentation}"
                f"_min{run_config.l2_local_min_duration_sec:g}"
                f"_max{run_config.l2_local_max_duration_sec:g}"
                f"_p{run_config.l2_local_peak_percentile:g}"
            )
        if run_config.l2_scoring != "embedding":
            output_name += f"_score{run_config.l2_scoring}"
    if run_config.method in {"l3_rerank_l2", "l1_plus_l3_rerank_l2"}:
        output_name += f"_l3k{run_config.top_l3_segments:g}"
        if run_config.l3_rerank_keep != DEFAULT_L3_RERANK_KEEP:
            output_name += f"_keep{run_config.l3_rerank_keep:g}"
        if run_config.l2_rerank_encoder != "openclip":
            output_name += f"_l2enc{run_config.l2_rerank_encoder}"
        if run_config.l2_rerank_query_mode != "target":
            output_name += f"_l2q{run_config.l2_rerank_query_mode}"
        if run_config.l3_rerank_evidence_source != "reranked_l3":
            output_name += f"_evi{run_config.l3_rerank_evidence_source}"
            if run_config.l3_rerank_evidence_source == "top_l2_per_l3":
                output_name += f"_l2p{run_config.l2_evidence_per_l3:g}_l1p{run_config.l1_evidence_per_l2:g}"
    if run_config.evidence_text_mode != "frames":
        output_name += f"_evitext{run_config.evidence_text_mode}"
    if run_config.vector_backend != "torch":
        output_name += f"_vec{run_config.vector_backend}"
    output_name += f"_{run_config.max_frames}f_{run_config.image_max_size}"
    return output_name


def _format_time_range(start_time_sec: float, end_time_sec: float) -> str:
    return f"{float(start_time_sec):.1f}s-{float(end_time_sec):.1f}s"


def _find_covering_hit(time_sec: float, hits: list[SegmentHit]) -> SegmentHit | None:
    for hit in hits:
        if float(hit.start_time_sec) <= float(time_sec) <= float(hit.end_time_sec):
            return hit
    return None


def _build_chunked_frame_texts(
    *,
    frame_hits: list[FrameHit],
    retrieval_info: dict[str, Any],
    subtitle_texts: list[str] | None,
) -> list[str]:
    l3_hits = list(retrieval_info.get("l3_hits") or [])
    l2_hits = list(retrieval_info.get("selected_l2_hits") or retrieval_info.get("l2_hits") or [])
    chunk_id_by_key: dict[str, int] = {}
    frame_texts: list[str] = []

    for frame_pos, frame_hit in enumerate(frame_hits):
        time_sec = float(frame_hit.time_sec)
        l2_hit = _find_covering_hit(time_sec, l2_hits)
        l3_hit = _find_covering_hit(time_sec, l3_hits)
        if l2_hit is not None:
            key = f"L2:{l2_hit.segment_id}"
        elif l3_hit is not None:
            key = f"L3:{l3_hit.segment_id}"
        else:
            key = "ungrouped"

        if key not in chunk_id_by_key:
            chunk_id_by_key[key] = len(chunk_id_by_key) + 1
            parts = [f"Evidence chunk {chunk_id_by_key[key]} begins."]
            if l3_hit is not None:
                parts.append(
                    f"L3 segment: {_format_time_range(l3_hit.start_time_sec, l3_hit.end_time_sec)}."
                )
            if l2_hit is not None:
                parts.append(
                    f"L2 window inside that segment: {_format_time_range(l2_hit.start_time_sec, l2_hit.end_time_sec)}."
                )
            parts.append("Frames in this chunk are temporally ordered.")
        else:
            parts = [f"Evidence chunk {chunk_id_by_key[key]} continues."]

        parts.append(f"Frame {frame_pos + 1} at {time_sec:.1f}s.")
        if subtitle_texts and frame_pos < len(subtitle_texts) and subtitle_texts[frame_pos]:
            parts.append(f"Subtitle near this frame: {subtitle_texts[frame_pos]}")
        frame_texts.append(" ".join(parts))
    return frame_texts


def _chunk_prompt_prefix(base_prefix: str) -> str:
    return (
        f"{base_prefix}\n"
        "The visual evidence is presented as temporal chunks. "
        "Each chunk belongs to a retrieved L3 segment and, when available, an L2 sub-window. "
        "Use the chunk labels and timestamps to compare the answer options. "
        "Do not assume unshown time ranges are visible."
    )


def _extract_target_text(question: str) -> str:
    match = TARGET_PATTERN.search(question)
    if match:
        return match.group(1).strip()
    return question


def _segment_time_overlap(start_a: float, end_a: float, start_b: float, end_b: float) -> bool:
    return max(float(start_a), float(start_b)) <= min(float(end_a), float(end_b))


def _segment_topm_score(
    *,
    frame_scores: torch.Tensor,
    start_index: int,
    end_index: int,
    top_m: int = L2_SCORE_TOP_M,
) -> float:
    segment_scores = frame_scores[int(start_index) : int(end_index) + 1]
    if segment_scores.numel() == 0:
        return 0.0
    k = min(max(int(top_m), 1), int(segment_scores.numel()))
    return float(torch.topk(segment_scores, k=k).values.mean().item())


def _rank_score_by_id(items: list[tuple[Any, float]]) -> dict[Any, float]:
    if not items:
        return {}
    ordered = sorted(items, key=lambda item: float(item[1]), reverse=True)
    denom = max(len(ordered) - 1, 1)
    return {item_id: 1.0 - (rank / denom) for rank, (item_id, _) in enumerate(ordered)}


def _get_viclip_encoder() -> ViCLIPEncoder:
    global _VICLIP_ENCODER
    if _VICLIP_ENCODER is None:
        _VICLIP_ENCODER = ViCLIPEncoder()
    return _VICLIP_ENCODER


def _temporal_nms_frame_hits(
    frame_hits: list[FrameHit],
    *,
    max_hits: int,
    min_gap_sec: float,
) -> list[FrameHit]:
    selected: list[FrameHit] = []
    for hit in sorted(frame_hits, key=lambda item: float(item.score), reverse=True):
        if len(selected) >= max_hits:
            break
        if all(abs(float(hit.time_sec) - float(kept.time_sec)) >= min_gap_sec for kept in selected):
            selected.append(hit)
    return sorted(selected, key=lambda item: int(item.frame_index))


def _expanded_segment_hits_from_frame_hits(
    *,
    timestamps: np.ndarray,
    frame_hits: list[FrameHit],
    expansion_seconds: float,
    allowed_indices: list[int],
) -> list[SegmentHit]:
    allowed = set(int(index) for index in allowed_indices)
    segment_hits: list[SegmentHit] = []
    for idx, hit in enumerate(frame_hits):
        center_time = float(hit.time_sec)
        start_time = center_time - float(expansion_seconds)
        end_time = center_time + float(expansion_seconds)
        indices = [
            int(frame_index)
            for frame_index, time_sec in enumerate(timestamps.tolist())
            if frame_index in allowed and start_time <= float(time_sec) <= end_time
        ]
        if not indices:
            continue
        start_index = min(indices)
        end_index = max(indices)
        segment_hits.append(
            SegmentHit(
                segment_id=f"expand_{idx:04d}",
                score=float(hit.score),
                start_index=start_index,
                end_index=end_index,
                start_time_sec=float(timestamps[start_index]),
                end_time_sec=float(timestamps[end_index]),
            )
        )
    return segment_hits


def _segment_sample_indices(segment_hit: SegmentHit, count: int) -> list[int]:
    indices = np.arange(int(segment_hit.start_index), int(segment_hit.end_index) + 1)
    if len(indices) <= count:
        return [int(index) for index in indices.tolist()]
    positions = np.linspace(0, len(indices) - 1, num=count)
    return [int(indices[int(round(position))]) for position in positions]


def _sample_viclip_clip_from_open_capture(
    *,
    capture: cv2.VideoCapture,
    native_fps: float,
    total_frames: int,
    start_time_sec: float,
    end_time_sec: float,
    frame_budget: int,
    required_frames: int,
) -> list[Image.Image]:
    start = max(float(start_time_sec), 0.0)
    end = max(float(end_time_sec), start)
    if frame_budget <= 1 or end <= start:
        timestamps = [(start + end) / 2.0]
    else:
        timestamps = np.linspace(start, end, num=int(frame_budget), endpoint=True).astype(float).tolist()

    frames: list[Image.Image] = []
    for timestamp in timestamps:
        native_index = int(round(float(timestamp) * float(native_fps)))
        native_index = max(0, min(native_index, int(total_frames) - 1))
        capture.set(cv2.CAP_PROP_POS_FRAMES, native_index)
        ok, frame = capture.read()
        if not ok:
            capture.set(cv2.CAP_PROP_POS_MSEC, float(timestamp) * 1000.0)
            ok, frame = capture.read()
        if not ok:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))

    if not frames:
        raise RuntimeError(f"No frames sampled for ViCLIP clip at {start:.3f}-{end:.3f}s")
    if len(frames) >= required_frames:
        step_indices = torch.linspace(0, len(frames) - 1, steps=int(required_frames))
        return [frames[int(round(float(step)))] for step in step_indices.tolist()]
    selected = list(frames)
    while len(selected) < required_frames:
        selected.append(selected[-1])
    return selected


class AblationRetriever:
    def __init__(
        self,
        *,
        feature_root: Path,
        derived_cache_root: Path,
        config: AblationRunConfig,
        encoder_device: str | None = None,
    ) -> None:
        self.feature_root = feature_root
        self.derived_cache_root = derived_cache_root
        self.config = config
        self.encoder_device = encoder_device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline_config = PipelineConfig(
            sample_fps=config.sample_fps,
            window_seconds=config.l2_window_seconds,
            window_stride_seconds=config.l2_stride_seconds,
            layer2_pooling="mean",
            top_windows=config.top_l2_segments,
            max_evidence_frames=config.max_frames,
            image_max_size=config.image_max_size,
            device=self.encoder_device,
        )
        self._encoder = OpenCLIPEncoder(device=self.encoder_device)
        self._video_cache: dict[str, VideoArtifacts] = {}
        self._query_cache: dict[str, torch.Tensor] = {}

    def close(self) -> None:
        del self._encoder
        self._video_cache.clear()
        self._query_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _stable_hash(self, payload: dict[str, Any]) -> str:
        raw = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha1(raw).hexdigest()[:12]

    def _video_derived_dir(self, video_id: str) -> Path:
        return self.derived_cache_root / video_id

    def _l2_cache_dir(self, video_id: str) -> Path:
        key = self._stable_hash(
            {
                "version": 4,
                "sample_fps": self.config.sample_fps,
                "feature_eval_fps": self.config.feature_eval_fps,
                "segmentation": self.config.l2_segmentation,
                "window_seconds": self.config.l2_window_seconds,
                "stride_seconds": self.config.l2_stride_seconds,
                "local_min_duration_sec": self.config.l2_local_min_duration_sec,
                "local_max_duration_sec": self.config.l2_local_max_duration_sec,
                "local_fast_kernel_size": self.config.l2_local_fast_kernel_size,
                "local_slow_kernel_size": self.config.l2_local_slow_kernel_size,
                "local_peak_percentile": self.config.l2_local_peak_percentile,
                "scoring": self.config.l2_scoring,
                "frame_score_top_m": self.config.l2_frame_score_top_m,
                "frame_score_temperature": self.config.l2_frame_score_temperature,
                "pooling": "mean",
            }
        )
        return self._video_derived_dir(video_id) / f"l2_{key}"

    def _l3_cache_dir(self, video_id: str) -> Path:
        key = self._stable_hash(
            {
                "version": 2,
                "sample_fps": self.config.sample_fps,
                "feature_eval_fps": self.config.feature_eval_fps,
                "segmentation": self.config.l3_segmentation,
                "window_seconds": self.config.l3_window_seconds,
                "stride_seconds": self.config.l3_stride_seconds,
                "pooling": "mean",
            }
        )
        return self._video_derived_dir(video_id) / f"l3_{key}"

    def _motion_cache_path(self, video_id: str) -> Path:
        key = self._stable_hash(
            {
                "version": 1,
                "sample_fps": self.config.sample_fps,
                "feature_eval_fps": self.config.feature_eval_fps,
            }
        )
        return self._video_derived_dir(video_id) / f"motion_{key}.npy"

    def _viclip_l2_cache_dir(self, video_id: str) -> Path:
        key = self._stable_hash(
            {
                "version": 1,
                "encoder": "viclip",
                "model_id": "OpenGVLab/ViCLIP-L-14-hf",
                "sample_fps": self.config.sample_fps,
                "feature_eval_fps": self.config.feature_eval_fps,
                "segmentation": self.config.l2_segmentation,
                "window_seconds": self.config.l2_window_seconds,
                "stride_seconds": self.config.l2_stride_seconds,
                "local_min_duration_sec": self.config.l2_local_min_duration_sec,
                "local_max_duration_sec": self.config.l2_local_max_duration_sec,
                "local_fast_kernel_size": self.config.l2_local_fast_kernel_size,
                "local_slow_kernel_size": self.config.l2_local_slow_kernel_size,
                "local_peak_percentile": self.config.l2_local_peak_percentile,
            }
        )
        return self._video_derived_dir(video_id) / f"l2_viclip_{key}"

    def _frame_faiss_path(self, artifacts: VideoArtifacts) -> Path:
        key = self._stable_hash(
            {
                "version": 1,
                "kind": "frame_openclip_ip",
                "sample_fps": self.config.sample_fps,
                "feature_eval_fps": self.config.feature_eval_fps,
                "count": int(artifacts.frame_embeddings.shape[0]),
                "dim": int(artifacts.frame_embeddings.shape[1]) if artifacts.frame_embeddings.ndim == 2 else 0,
            }
        )
        return self._video_derived_dir(artifacts.video_id) / f"faiss_frame_{key}.index"

    def _cached_faiss_index(self, *, artifacts: VideoArtifacts, cache_key: str, path: Path, embeddings: torch.Tensor) -> Any:
        cached = artifacts.faiss_indices.get(cache_key)
        if cached is not None:
            return cached
        index = load_or_build_ip_index(path, embeddings)
        artifacts.faiss_indices[cache_key] = index
        return index

    def _segment_hits_from_scores(
        self,
        *,
        segments: list[Segment],
        scores: np.ndarray,
        order: np.ndarray,
    ) -> list[SegmentHit]:
        hits: list[SegmentHit] = []
        for score, idx in zip(scores, order, strict=False):
            segment = segments[int(idx)]
            hits.append(
                SegmentHit(
                    segment_id=segment.segment_id,
                    score=float(score),
                    start_index=int(segment.start_index),
                    end_index=int(segment.end_index),
                    start_time_sec=float(segment.start_time_sec),
                    end_time_sec=float(segment.end_time_sec),
                )
            )
        return hits

    def _retrieve_top_segments(
        self,
        *,
        artifacts: VideoArtifacts,
        layer: str,
        query_embedding: torch.Tensor,
        segment_embeddings: torch.Tensor,
        segments: list[Segment],
        top_k: int,
    ) -> list[SegmentHit]:
        if self.config.vector_backend == "faiss" and segment_embeddings.numel():
            index_path: Path | None = None
            if layer == "l2" and segments is artifacts.l2_segments:
                index_path = self._l2_cache_dir(artifacts.video_id) / "embeddings.faiss"
            elif layer == "l3" and segments is artifacts.l3_segments:
                index_path = self._l3_cache_dir(artifacts.video_id) / "embeddings.faiss"
            if index_path is not None:
                index = self._cached_faiss_index(
                    artifacts=artifacts,
                    cache_key=f"{layer}:{index_path}",
                    path=index_path,
                    embeddings=segment_embeddings,
                )
                scores, order = search_ip_index(index, query_embedding, top_k)
                return self._segment_hits_from_scores(segments=segments, scores=scores, order=order)
        return retrieve_top_segments(
            query_embedding=query_embedding,
            segment_embeddings=segment_embeddings,
            segments=segments,
            top_k=top_k,
            backend=self.config.vector_backend,
        )

    def _serialize_segments(self, segments: list[Segment]) -> list[dict[str, Any]]:
        return [
            {
                "segment_id": segment.segment_id,
                "start_index": int(segment.start_index),
                "end_index": int(segment.end_index),
                "start_time_sec": float(segment.start_time_sec),
                "end_time_sec": float(segment.end_time_sec),
                "duration_sec": float(segment.duration_sec),
            }
            for segment in segments
        ]

    def _deserialize_segments(self, payload: list[dict[str, Any]]) -> list[Segment]:
        return [Segment(**item) for item in payload]

    def _scope_bounds(self, example: BaselineExample) -> tuple[float | None, float | None]:
        start = example.metadata.get("scope_start_sec")
        end = example.metadata.get("scope_end_sec")
        if start is None or end is None:
            return None, None
        return float(start), float(end)

    def _frame_indices_in_scope(self, artifacts: VideoArtifacts, start: float | None, end: float | None) -> list[int] | None:
        if start is None or end is None:
            return None
        return [
            int(index)
            for index, time_sec in enumerate(artifacts.timestamps.tolist())
            if float(start) <= float(time_sec) <= float(end)
        ]

    def _filter_indices_to_scope(
        self,
        indices: list[int],
        artifacts: VideoArtifacts,
        start: float | None,
        end: float | None,
    ) -> list[int]:
        if start is None or end is None:
            return indices
        return [
            int(index)
            for index in indices
            if 0 <= int(index) < len(artifacts.timestamps)
            and float(start) <= float(artifacts.timestamps[int(index)]) <= float(end)
        ]

    def _segments_and_embeddings_in_scope(
        self,
        *,
        segments: list[Segment],
        embeddings: torch.Tensor,
        start: float | None,
        end: float | None,
    ) -> tuple[list[Segment], torch.Tensor]:
        if start is None or end is None:
            return segments, embeddings
        scoped: list[tuple[int, Segment]] = [
            (index, segment)
            for index, segment in enumerate(segments)
            if _segment_time_overlap(segment.start_time_sec, segment.end_time_sec, start, end)
        ]
        if not scoped:
            return [], embeddings[:0]
        indices = torch.tensor([index for index, _ in scoped], dtype=torch.long)
        return [segment for _, segment in scoped], embeddings.index_select(0, indices)

    def _query_embedding(self, question: str, options: list[str]) -> torch.Tensor:
        target_text = _extract_target_text(question)
        query_text = target_text if target_text != question else build_query_text(question, options)
        cache_key = f"openclip::{query_text}"
        cached = self._query_cache.get(cache_key)
        if cached is not None:
            return cached
        embedding = self._encoder.encode_texts([query_text])[0].cpu()
        self._query_cache[cache_key] = embedding
        return embedding

    def _load_dense_feature_cache(self, cache_dir: Path) -> tuple[np.ndarray, torch.Tensor, float]:
        timestamps = np.load(cache_dir / "timestamps.npy").astype(np.float32)
        frame_embeddings = torch.load(cache_dir / "frame_embeddings.pt", map_location="cpu").float()
        metadata = json.loads((cache_dir / "metadata.json").read_text(encoding="utf-8"))
        return timestamps, frame_embeddings, float(metadata["native_fps"])

    def _load_sharded_feature_cache(self, cache_dir: Path) -> tuple[np.ndarray, torch.Tensor, float]:
        metadata = json.loads((cache_dir / "meta.json").read_text(encoding="utf-8"))
        shards = sorted(cache_dir.glob("shard_*.pt"))
        if not shards:
            raise FileNotFoundError(f"No shard_*.pt files found under {cache_dir}")
        frame_indices_list: list[torch.Tensor] = []
        timestamps_list: list[torch.Tensor] = []
        embeddings_list: list[torch.Tensor] = []
        for shard_path in shards:
            shard = torch.load(shard_path, map_location="cpu")
            frame_indices_list.append(shard["frame_idx"].to(torch.int64).cpu())
            timestamps_list.append(shard["timestamp_sec"].to(torch.float32).cpu())
            embeddings_list.append(shard["openclip"].to(torch.float32).cpu())
        frame_indices = torch.cat(frame_indices_list, dim=0)
        timestamps = torch.cat(timestamps_list, dim=0)
        embeddings = torch.cat(embeddings_list, dim=0)
        order = frame_indices.argsort()
        timestamps = timestamps[order]
        embeddings = embeddings[order]
        native_fps = float(metadata.get("native_fps") or metadata.get("fps") or 0.0)
        return timestamps.numpy().astype(np.float32), embeddings.float(), native_fps

    def _downsample_features_for_eval(
        self,
        *,
        timestamps: np.ndarray,
        frame_embeddings: torch.Tensor,
        target_fps: float | None,
    ) -> tuple[np.ndarray, torch.Tensor]:
        if target_fps is None or target_fps <= 0 or len(timestamps) <= 1:
            return timestamps, frame_embeddings
        duration = max(float(timestamps[-1]) - float(timestamps[0]), 1e-6)
        current_fps = (len(timestamps) - 1) / duration
        if current_fps <= target_fps * 1.25:
            return timestamps, frame_embeddings
        step_sec = 1.0 / float(target_fps)
        selected: list[int] = []
        next_time = float(timestamps[0])
        for index, time_sec in enumerate(timestamps.tolist()):
            if float(time_sec) + 1e-6 >= next_time:
                selected.append(index)
                next_time = float(time_sec) + step_sec
        if not selected:
            selected = [0]
        indices = torch.tensor(selected, dtype=torch.long)
        return timestamps[selected].astype(np.float32), frame_embeddings[indices].contiguous()

    def _load_video(self, example: BaselineExample) -> VideoArtifacts:
        cached = self._video_cache.get(example.video_id)
        if cached is not None:
            return cached
        cache_dir = self.feature_root / example.video_id
        if (cache_dir / "timestamps.npy").exists() and (cache_dir / "frame_embeddings.pt").exists():
            timestamps, frame_embeddings, native_fps = self._load_dense_feature_cache(cache_dir)
        elif (cache_dir / "meta.json").exists() and any(cache_dir.glob("shard_*.pt")):
            timestamps, frame_embeddings, native_fps = self._load_sharded_feature_cache(cache_dir)
        else:
            raise FileNotFoundError(f"Unsupported or missing feature cache for {example.video_id} under {cache_dir}")
        timestamps, frame_embeddings = self._downsample_features_for_eval(
            timestamps=timestamps,
            frame_embeddings=frame_embeddings,
            target_fps=self.config.feature_eval_fps,
        )
        artifacts = VideoArtifacts(
            video_id=example.video_id,
            video_path=Path(example.video_path),
            timestamps=timestamps,
            frame_embeddings=frame_embeddings,
            native_fps=native_fps,
        )
        self._video_cache[example.video_id] = artifacts
        return artifacts

    def _ensure_l2(self, artifacts: VideoArtifacts) -> None:
        if artifacts.l2_segments is not None and artifacts.l2_embeddings is not None:
            return
        cache_dir = self._l2_cache_dir(artifacts.video_id)
        segments_path = cache_dir / "segments.json"
        embeddings_path = cache_dir / "embeddings.pt"
        if segments_path.exists() and embeddings_path.exists():
            artifacts.l2_segments = self._deserialize_segments(json.loads(segments_path.read_text(encoding="utf-8")))
            artifacts.l2_embeddings = torch.load(embeddings_path, map_location="cpu").float()
            return
        if self.config.l2_segmentation == "fixed":
            sampled = type("Sampled", (), {"timestamps": artifacts.timestamps})()
            segments = build_window_segments(sampled, self.pipeline_config)
        elif self.config.l2_segmentation == "l3_local_contrast":
            self._ensure_l3(artifacts)
            assert artifacts.l3_segments is not None
            segments = segment_l3_local_contrast_windows(
                timestamps=artifacts.timestamps,
                frame_embeddings=artifacts.frame_embeddings,
                l3_segments=artifacts.l3_segments,
                min_duration_sec=self.config.l2_local_min_duration_sec,
                max_duration_sec=self.config.l2_local_max_duration_sec,
                fast_kernel_size=self.config.l2_local_fast_kernel_size,
                slow_kernel_size=self.config.l2_local_slow_kernel_size,
                peak_percentile=self.config.l2_local_peak_percentile,
                prefix="l2_l3local_contrast",
            )
        else:
            raise ValueError(f"Unsupported l2_segmentation: {self.config.l2_segmentation}")
        embeddings = pool_segments(artifacts.frame_embeddings, segments, pooling="mean")
        artifacts.l2_segments = segments
        artifacts.l2_embeddings = embeddings
        cache_dir.mkdir(parents=True, exist_ok=True)
        segments_path.write_text(json.dumps(self._serialize_segments(segments), indent=2), encoding="utf-8")
        torch.save(embeddings, embeddings_path)
        if self.config.vector_backend == "faiss" and embeddings.numel():
            index = write_ip_index(cache_dir / "embeddings.faiss", embeddings)
            artifacts.faiss_indices[f"l2:{cache_dir / 'embeddings.faiss'}"] = index

    def _ensure_viclip_l2_embeddings(self, artifacts: VideoArtifacts) -> torch.Tensor:
        self._ensure_l2(artifacts)
        assert artifacts.l2_segments is not None
        cache_dir = self._viclip_l2_cache_dir(artifacts.video_id)
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
            if not frames:
                continue
            if len(frames) >= encoder.num_frames:
                step_indices = torch.linspace(0, len(frames) - 1, steps=encoder.num_frames)
                selected = [frames[int(round(float(step)))] for step in step_indices.tolist()]
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
        segments_path.write_text(json.dumps(self._serialize_segments(artifacts.l2_segments), indent=2), encoding="utf-8")
        torch.save(embeddings, embeddings_path)
        return embeddings

    def _encode_viclip_l2_segment(self, *, artifacts: VideoArtifacts, segment: Segment) -> torch.Tensor:
        encoder = _get_viclip_encoder()
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
        if not frames:
            raise RuntimeError(f"No frames sampled for segment {segment.segment_id} in {artifacts.video_id}")
        if len(frames) >= encoder.num_frames:
            step_indices = torch.linspace(0, len(frames) - 1, steps=encoder.num_frames)
            selected = [frames[int(round(float(step)))] for step in step_indices.tolist()]
        else:
            selected = list(frames)
            while len(selected) < encoder.num_frames:
                selected.append(selected[-1])
        embedding = encoder.encode_video_clips([selected], batch_size=1).float().cpu()[0]
        del frames
        del selected
        return embedding

    def _ensure_viclip_l2_embeddings_for_indices(
        self,
        *,
        artifacts: VideoArtifacts,
        segment_indices: set[int],
    ) -> dict[int, torch.Tensor]:
        self._ensure_l2(artifacts)
        assert artifacts.l2_segments is not None
        cache_dir = self._viclip_l2_cache_dir(artifacts.video_id)
        full_embeddings_path = cache_dir / "embeddings.pt"
        if full_embeddings_path.exists():
            full_embeddings = torch.load(full_embeddings_path, map_location="cpu").float()
            return {int(index): full_embeddings[int(index)] for index in segment_indices}

        segment_cache_dir = cache_dir / "segment_embeddings"
        segment_cache_dir.mkdir(parents=True, exist_ok=True)
        embeddings: dict[int, torch.Tensor] = {}
        missing_segment_indices: list[int] = []
        for segment_index in sorted(segment_indices):
            segment_path = segment_cache_dir / f"{int(segment_index):06d}.pt"
            if segment_path.exists():
                embeddings[int(segment_index)] = torch.load(segment_path, map_location="cpu").float()
                continue
            missing_segment_indices.append(int(segment_index))

        if missing_segment_indices:
            encoder = _get_viclip_encoder()
            capture = cv2.VideoCapture(str(artifacts.video_path))
            if not capture.isOpened():
                raise RuntimeError(f"Failed to open video for ViCLIP L2 sampling: {artifacts.video_path}")
            native_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if native_fps <= 0.0 or total_frames <= 0:
                capture.release()
                raise RuntimeError(f"Invalid video metadata for ViCLIP L2 sampling: {artifacts.video_path}")
            try:
                for batch_start in range(0, len(missing_segment_indices), max(VICLIP_BATCH_SIZE, 1)):
                    batch_indices = missing_segment_indices[batch_start : batch_start + max(VICLIP_BATCH_SIZE, 1)]
                    clips: list[list[Image.Image]] = []
                    valid_indices: list[int] = []
                    for segment_index in batch_indices:
                        segment = artifacts.l2_segments[int(segment_index)]
                        candidate_budget = min(
                            VICLIP_L2_MAX_FRAMES,
                            max(1, int(segment.end_index) - int(segment.start_index) + 1),
                        )
                        try:
                            clip = _sample_viclip_clip_from_open_capture(
                                capture=capture,
                                native_fps=native_fps,
                                total_frames=total_frames,
                                start_time_sec=float(segment.start_time_sec),
                                end_time_sec=float(segment.end_time_sec),
                                frame_budget=candidate_budget,
                                required_frames=encoder.num_frames,
                            )
                        except RuntimeError:
                            embedding = self._encode_viclip_l2_segment(artifacts=artifacts, segment=segment)
                            segment_path = segment_cache_dir / f"{int(segment_index):06d}.pt"
                            torch.save(embedding, segment_path)
                            embeddings[int(segment_index)] = embedding
                            continue
                        clips.append(clip)
                        valid_indices.append(int(segment_index))
                    if clips:
                        clip_embeddings = encoder.encode_video_clips(
                            clips,
                            batch_size=max(VICLIP_BATCH_SIZE, 1),
                        ).float().cpu()
                        for offset, segment_index in enumerate(valid_indices):
                            embedding = clip_embeddings[offset]
                            segment_path = segment_cache_dir / f"{int(segment_index):06d}.pt"
                            torch.save(embedding, segment_path)
                            embeddings[int(segment_index)] = embedding
                    del clips
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            finally:
                capture.release()
        (cache_dir / "segments.json").write_text(json.dumps(self._serialize_segments(artifacts.l2_segments), indent=2), encoding="utf-8")
        return embeddings

    def _ensure_l3(self, artifacts: VideoArtifacts) -> None:
        if artifacts.l3_segments is not None and artifacts.l3_embeddings is not None:
            return
        cache_dir = self._l3_cache_dir(artifacts.video_id)
        segments_path = cache_dir / "segments.json"
        embeddings_path = cache_dir / "embeddings.pt"
        if segments_path.exists() and embeddings_path.exists():
            artifacts.l3_segments = self._deserialize_segments(json.loads(segments_path.read_text(encoding="utf-8")))
            artifacts.l3_embeddings = torch.load(embeddings_path, map_location="cpu").float()
            return
        if self.config.l3_segmentation == "fixed":
            segments = segment_fixed_windows(
                timestamps=artifacts.timestamps,
                window_seconds=self.config.l3_window_seconds,
                stride_seconds=self.config.l3_stride_seconds,
                prefix="l3_fixed",
            )
        else:
            if self.config.l3_segmentation != "fused_adaptive":
                raise ValueError(f"Unsupported l3_segmentation: {self.config.l3_segmentation}")
            motion_cache_path = self._motion_cache_path(artifacts.video_id)
            if motion_cache_path.exists():
                motion_energy = np.load(motion_cache_path).astype(np.float32)
            else:
                target_frame_indices = [int(round(float(ts) * float(artifacts.native_fps))) for ts in artifacts.timestamps.tolist()]
                motion_energy = compute_motion_energy_for_frame_indices(
                    artifacts.video_path,
                    target_frame_indices=target_frame_indices,
                ).astype(np.float32)
                motion_cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(motion_cache_path, motion_energy)
            result = segment_fused_adaptive_peaks(
                timestamps=artifacts.timestamps,
                frame_embeddings=artifacts.frame_embeddings,
                motion_energy=motion_energy,
                prefix="l3_fused_adaptive",
            )
            segments = result["segments"]
        embeddings = pool_segments(artifacts.frame_embeddings, segments, pooling="mean")
        artifacts.l3_segments = segments
        artifacts.l3_embeddings = embeddings
        cache_dir.mkdir(parents=True, exist_ok=True)
        segments_path.write_text(json.dumps(self._serialize_segments(segments), indent=2), encoding="utf-8")
        torch.save(embeddings, embeddings_path)
        if self.config.vector_backend == "faiss" and embeddings.numel():
            index = write_ip_index(cache_dir / "embeddings.faiss", embeddings)
            artifacts.faiss_indices[f"l3:{cache_dir / 'embeddings.faiss'}"] = index

    def _frame_hits_from_indices(
        self,
        *,
        artifacts: VideoArtifacts,
        query_embedding: torch.Tensor,
        allowed_indices: list[int] | None,
    ) -> list[FrameHit]:
        if self.config.vector_backend == "faiss" and allowed_indices is None and artifacts.frame_embeddings.numel():
            index = self._cached_faiss_index(
                artifacts=artifacts,
                cache_key=f"frame:{self._frame_faiss_path(artifacts)}",
                path=self._frame_faiss_path(artifacts),
                embeddings=artifacts.frame_embeddings,
            )
            scores, order = search_ip_index(index, query_embedding, self.config.max_frames)
            hits = [
                FrameHit(
                    frame_index=int(frame_index),
                    time_sec=float(artifacts.timestamps[int(frame_index)]),
                    score=float(score),
                )
                for score, frame_index in zip(scores, order, strict=False)
            ]
            hits.sort(key=lambda hit: hit.frame_index)
            return hits
        return retrieve_top_frames(
            query_embedding=query_embedding,
            frame_embeddings=artifacts.frame_embeddings,
            timestamps=artifacts.timestamps,
            top_k=self.config.max_frames,
            allowed_indices=allowed_indices,
            backend=self.config.vector_backend,
        )

    def _frame_hits_for_indices(
        self,
        *,
        artifacts: VideoArtifacts,
        query_embedding: torch.Tensor,
        indices: list[int],
        required_indices: list[int] | None = None,
    ) -> list[FrameHit]:
        unique_indices = sorted(set(int(index) for index in indices if 0 <= int(index) < artifacts.frame_embeddings.shape[0]))
        if not unique_indices:
            return []
        scores = torch.matmul(artifacts.frame_embeddings[unique_indices], query_embedding).cpu().numpy()
        score_by_index = {index: float(scores[offset]) for offset, index in enumerate(unique_indices)}
        selected = {
            int(index)
            for index in (required_indices or [])
            if int(index) in score_by_index
        }
        remaining = sorted(
            (index for index in unique_indices if index not in selected),
            key=lambda index: score_by_index[index],
            reverse=True,
        )
        for index in remaining:
            if len(selected) >= self.config.max_frames:
                break
            selected.add(index)
        if len(selected) > self.config.max_frames:
            selected = set(sorted(selected)[: self.config.max_frames])
        return [
            FrameHit(frame_index=index, time_sec=float(artifacts.timestamps[index]), score=float(score_by_index[index]))
            for index in sorted(selected)
        ]

    def _candidate_l2_indices_from_l3_hits(
        self,
        *,
        artifacts: VideoArtifacts,
        l3_hits: list[SegmentHit],
    ) -> set[int]:
        assert artifacts.l2_segments is not None
        selected: set[int] = set()
        for segment_index, l2_segment in enumerate(artifacts.l2_segments):
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

    def _rerank_l3_hits_with_l2(
        self,
        *,
        artifacts: VideoArtifacts,
        query_embedding: torch.Tensor,
        target_text: str,
        l3_hits: list[SegmentHit],
    ) -> tuple[list[SegmentHit], dict[str, Any]]:
        if not l3_hits:
            return [], {"l2_candidates": []}

        self._ensure_l2(artifacts)
        assert artifacts.l2_segments is not None
        candidate_indices = self._candidate_l2_indices_from_l3_hits(artifacts=artifacts, l3_hits=l3_hits)
        if not candidate_indices:
            return l3_hits[: self.config.l3_rerank_keep], {"l2_candidates": []}

        frame_scores = torch.matmul(artifacts.frame_embeddings, query_embedding).cpu()
        l3_prior_rank = _rank_score_by_id(
            [(str(hit.segment_id), float(hit.score)) for hit in l3_hits],
        )

        viclip_query_embedding: torch.Tensor | None = None
        viclip_l2_embeddings: dict[int, torch.Tensor] | None = None
        if self.config.l2_rerank_encoder == "viclip":
            viclip_query_embedding = _get_viclip_encoder().encode_texts([target_text])[0].float().cpu()
            viclip_l2_embeddings = self._ensure_viclip_l2_embeddings_for_indices(
                artifacts=artifacts,
                segment_indices=candidate_indices,
            )

        l2_items: list[dict[str, Any]] = []
        for segment_index in sorted(candidate_indices):
            segment = artifacts.l2_segments[segment_index]
            parent_segment_id = None
            for l3_hit in l3_hits:
                if _segment_time_overlap(
                    segment.start_time_sec,
                    segment.end_time_sec,
                    l3_hit.start_time_sec,
                    l3_hit.end_time_sec,
                ):
                    parent_segment_id = str(l3_hit.segment_id)
                    break
            if parent_segment_id is None:
                continue
            if self.config.l2_rerank_encoder == "viclip":
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
        )
        for item in l2_items:
            item["rank_score"] = float(l2_rank.get(int(item["segment_index"]), 0.0))

        parent_scores: dict[str, list[float]] = {}
        for item in l2_items:
            parent_scores.setdefault(str(item["parent_segment_id"]), []).append(float(item["rank_score"]))

        reranked_hits: list[SegmentHit] = []
        for hit in l3_hits:
            segment_id = str(hit.segment_id)
            child_scores = sorted(parent_scores.get(segment_id, []), reverse=True)
            best_child = child_scores[0] if child_scores else 0.0
            top2_sum = sum(child_scores[:2]) if child_scores else 0.0
            prior_rank = float(l3_prior_rank.get(segment_id, 0.0))
            score = best_child + (0.35 * top2_sum) + (0.05 * prior_rank)
            reranked_hits.append(
                SegmentHit(
                    segment_id=segment_id,
                    score=float(score),
                    start_index=int(hit.start_index),
                    end_index=int(hit.end_index),
                    start_time_sec=float(hit.start_time_sec),
                    end_time_sec=float(hit.end_time_sec),
                )
            )
        reranked_hits.sort(key=lambda item: float(item.score), reverse=True)
        l2_items.sort(key=lambda item: float(item["rank_score"]), reverse=True)
        l2_hits = [
            SegmentHit(
                segment_id=str(item["segment_id"]),
                score=float(item["rank_score"]),
                start_index=int(artifacts.l2_segments[int(item["segment_index"])].start_index),
                end_index=int(artifacts.l2_segments[int(item["segment_index"])].end_index),
                start_time_sec=float(item["start_time_sec"]),
                end_time_sec=float(item["end_time_sec"]),
            )
            for item in l2_items
        ]
        return reranked_hits[: self.config.l3_rerank_keep], {
            "l2_candidates": l2_hits,
            "l2_rerank_encoder": self.config.l2_rerank_encoder,
        }

    def retrieve(self, *, example: BaselineExample) -> tuple[list[int], dict[str, Any]]:
        artifacts = self._load_video(example)
        query_embedding = self._query_embedding(example.question, example.options)
        pooled_query_embedding = adapt_query_embedding_for_segment_pooling(query_embedding, pooling="mean")
        scope_start_sec, scope_end_sec = self._scope_bounds(example)
        scoped_frame_indices = self._frame_indices_in_scope(artifacts, scope_start_sec, scope_end_sec)

        if self.config.method == "l1":
            frame_hits = self._frame_hits_from_indices(
                artifacts=artifacts,
                query_embedding=query_embedding,
                allowed_indices=scoped_frame_indices,
            )
            return [int(hit.frame_index) for hit in frame_hits], {"l2_hits": [], "l3_hits": [], "frame_hits": frame_hits}

        if self.config.method == "l2":
            self._ensure_l2(artifacts)
            assert artifacts.l2_segments is not None
            assert artifacts.l2_embeddings is not None
            scoped_l2_segments, scoped_l2_embeddings = self._segments_and_embeddings_in_scope(
                segments=artifacts.l2_segments,
                embeddings=artifacts.l2_embeddings,
                start=scope_start_sec,
                end=scope_end_sec,
            )
            if self.config.l2_scoring == "embedding":
                l2_hits = self._retrieve_top_segments(
                    artifacts=artifacts,
                    layer="l2",
                    query_embedding=pooled_query_embedding,
                    segment_embeddings=scoped_l2_embeddings,
                    segments=scoped_l2_segments,
                    top_k=self.config.top_l2_segments,
                )
            else:
                l2_hits = retrieve_top_segments_from_frame_scores(
                    query_embedding=query_embedding,
                    frame_embeddings=artifacts.frame_embeddings,
                    segments=scoped_l2_segments,
                    top_k=self.config.top_l2_segments,
                    top_m=self.config.l2_frame_score_top_m,
                    aggregation=self.config.l2_scoring,
                    temperature=self.config.l2_frame_score_temperature,
                )
            allowed_l2_indices = self._filter_indices_to_scope(
                collect_segment_frame_indices(l2_hits),
                artifacts,
                scope_start_sec,
                scope_end_sec,
            )
            frame_hits = self._frame_hits_from_indices(
                artifacts=artifacts,
                query_embedding=query_embedding,
                allowed_indices=allowed_l2_indices,
            )
            return [int(hit.frame_index) for hit in frame_hits], {"l2_hits": l2_hits, "l3_hits": [], "frame_hits": frame_hits}

        if self.config.method in {"l3_rerank_l2", "l1_plus_l3_rerank_l2"}:
            self._ensure_l3(artifacts)
            self._ensure_l2(artifacts)
            assert artifacts.l3_segments is not None
            assert artifacts.l3_embeddings is not None
            l1_frame_hits: list[FrameHit] = []
            l1_budget = 0
            hm_budget = self.config.max_frames
            if self.config.method == "l1_plus_l3_rerank_l2":
                l1_budget = max(0, self.config.max_frames // 2)
                hm_budget = max(0, self.config.max_frames - l1_budget)
                if l1_budget > 0:
                    all_l1_hits = self._frame_hits_from_indices(
                        artifacts=artifacts,
                        query_embedding=query_embedding,
                        allowed_indices=scoped_frame_indices,
                    )
                    l1_frame_hits = all_l1_hits[:l1_budget]
            scoped_l3_segments, scoped_l3_embeddings = self._segments_and_embeddings_in_scope(
                segments=artifacts.l3_segments,
                embeddings=artifacts.l3_embeddings,
                start=scope_start_sec,
                end=scope_end_sec,
            )
            l3_hits = self._retrieve_top_segments(
                artifacts=artifacts,
                layer="l3",
                query_embedding=pooled_query_embedding,
                segment_embeddings=scoped_l3_embeddings,
                segments=scoped_l3_segments,
                top_k=self.config.top_l3_segments,
            )
            target_text = (
                build_query_text(example.question, example.options)
                if self.config.l2_rerank_query_mode == "full"
                else _extract_target_text(example.question)
            )
            reranked_l3_hits, rerank_debug = self._rerank_l3_hits_with_l2(
                artifacts=artifacts,
                query_embedding=query_embedding,
                target_text=target_text,
                l3_hits=l3_hits,
            )
            l2_candidates = list(rerank_debug.get("l2_candidates", []))
            selected_l2_hits: list[SegmentHit] = []
            if self.config.l3_rerank_evidence_source == "top_l2_per_l3":
                for l3_hit in reranked_l3_hits:
                    parent_l2_hits = [
                        l2_hit
                        for l2_hit in l2_candidates
                        if _segment_time_overlap(
                            l2_hit.start_time_sec,
                            l2_hit.end_time_sec,
                            l3_hit.start_time_sec,
                            l3_hit.end_time_sec,
                        )
                    ]
                    selected_l2_hits.extend(parent_l2_hits[: self.config.l2_evidence_per_l3])
            elif self.config.l3_rerank_evidence_source == "top_l2":
                selected_l2_hits = [
                    l2_hit
                    for l2_hit in l2_candidates
                    if any(
                        _segment_time_overlap(
                            l2_hit.start_time_sec,
                            l2_hit.end_time_sec,
                            l3_hit.start_time_sec,
                            l3_hit.end_time_sec,
                        )
                        for l3_hit in reranked_l3_hits
                    )
                ][: self.config.top_l2_segments]
            if selected_l2_hits:
                required_indices: list[int] = []
                for hit in selected_l2_hits:
                    required_indices.extend(_segment_sample_indices(hit, count=self.config.l1_evidence_per_l2))
                evidence_indices = self._filter_indices_to_scope(
                    collect_segment_frame_indices(selected_l2_hits),
                    artifacts,
                    scope_start_sec,
                    scope_end_sec,
                )
                required_indices = self._filter_indices_to_scope(
                    required_indices,
                    artifacts,
                    scope_start_sec,
                    scope_end_sec,
                )
                frame_hits = self._frame_hits_for_indices(
                    artifacts=artifacts,
                    query_embedding=query_embedding,
                    indices=evidence_indices,
                    required_indices=required_indices,
                )
            else:
                fallback_indices = self._filter_indices_to_scope(
                    collect_segment_frame_indices(reranked_l3_hits),
                    artifacts,
                    scope_start_sec,
                    scope_end_sec,
                )
                frame_hits = self._frame_hits_from_indices(
                    artifacts=artifacts,
                    query_embedding=query_embedding,
                    allowed_indices=fallback_indices,
                )
            if self.config.method == "l1_plus_l3_rerank_l2":
                l1_indices = [int(hit.frame_index) for hit in l1_frame_hits]
                hm_indices = [int(hit.frame_index) for hit in frame_hits]
                combined_indices: list[int] = []
                seen_indices: set[int] = set()
                for index in l1_indices:
                    if index not in seen_indices:
                        combined_indices.append(index)
                        seen_indices.add(index)
                    if len(combined_indices) >= l1_budget:
                        break
                for index in hm_indices:
                    if index not in seen_indices:
                        combined_indices.append(index)
                        seen_indices.add(index)
                    if len(combined_indices) >= self.config.max_frames:
                        break
                if len(combined_indices) < self.config.max_frames:
                    fill_indices = self._filter_indices_to_scope(
                        collect_segment_frame_indices(selected_l2_hits or reranked_l3_hits),
                        artifacts,
                        scope_start_sec,
                        scope_end_sec,
                    )
                    fill_hits = self._frame_hits_for_indices(
                        artifacts=artifacts,
                        query_embedding=query_embedding,
                        indices=fill_indices,
                        required_indices=[],
                    )
                    for hit in fill_hits:
                        index = int(hit.frame_index)
                        if index not in seen_indices:
                            combined_indices.append(index)
                            seen_indices.add(index)
                        if len(combined_indices) >= self.config.max_frames:
                            break
                score_by_index = {
                    int(hit.frame_index): float(hit.score)
                    for hit in [*l1_frame_hits, *frame_hits]
                }
                frame_hits = [
                    FrameHit(
                        frame_index=index,
                        time_sec=float(artifacts.timestamps[index]),
                        score=float(score_by_index.get(index, 0.0)),
                    )
                    for index in sorted(combined_indices[: self.config.max_frames])
                ]
            return [int(hit.frame_index) for hit in frame_hits], {
                "l2_hits": selected_l2_hits or l2_candidates,
                "l3_hits": reranked_l3_hits,
                "frame_hits": frame_hits,
                "l2_rerank_encoder": rerank_debug.get("l2_rerank_encoder"),
                "l3_rerank_evidence_source": self.config.l3_rerank_evidence_source,
                "selected_l2_hits": selected_l2_hits,
                "l1_plus_l3_rerank_l2_l1_budget": l1_budget,
                "l1_plus_l3_rerank_l2_hm_budget": hm_budget,
            }

        if self.config.method != "l3":
            raise ValueError(f"Unsupported ablation method: {self.config.method}")
        self._ensure_l3(artifacts)
        assert artifacts.l3_segments is not None
        assert artifacts.l3_embeddings is not None
        scoped_l3_segments, scoped_l3_embeddings = self._segments_and_embeddings_in_scope(
            segments=artifacts.l3_segments,
            embeddings=artifacts.l3_embeddings,
            start=scope_start_sec,
            end=scope_end_sec,
        )
        l3_hits = self._retrieve_top_segments(
            artifacts=artifacts,
            layer="l3",
            query_embedding=pooled_query_embedding,
            segment_embeddings=scoped_l3_embeddings,
            segments=scoped_l3_segments,
            top_k=self.config.top_l3_segments,
        )
        allowed_l3_indices = self._filter_indices_to_scope(
            collect_segment_frame_indices(l3_hits),
            artifacts,
            scope_start_sec,
            scope_end_sec,
        )
        frame_hits = self._frame_hits_from_indices(
            artifacts=artifacts,
            query_embedding=query_embedding,
            allowed_indices=allowed_l3_indices,
        )
        return [int(hit.frame_index) for hit in frame_hits], {"l2_hits": [], "l3_hits": l3_hits, "frame_hits": frame_hits}


def run_retrieval_ablation(
    *,
    examples: list[BaselineExample],
    feature_root: Path,
    derived_cache_root: Path,
    output_root: Path,
    run_config: AblationRunConfig,
    answer_config: AnswerConfig,
    subtitle_root: str | Path | None = None,
    subtitle_tar: str | Path | None = None,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    rows_path = output_root / "rows.jsonl"
    progress_path = output_root / "progress.log"
    error_path = output_root / "error.log"
    rolling_summary_path = output_root / "rolling_summary.json"

    rows, _ = _load_resume_rows(rows_path)
    completed_example_ids = {str(row["example_id"]) for row in rows}
    pending_examples = [example for example in examples if example.example_id not in completed_example_ids]
    if rows_path.exists():
        _rewrite_jsonl(rows_path, rows)
    _write_json(rolling_summary_path, {"completed": len(rows), "total": len(examples), **_summarize_rows(rows)})

    answerer = build_answerer(answer_config)
    retriever = AblationRetriever(
        feature_root=feature_root,
        derived_cache_root=derived_cache_root,
        config=run_config,
        encoder_device="cuda" if torch.cuda.is_available() else "cpu",
    )
    try:
        if pending_examples:
            _log_line(
                progress_path,
                f"[start] total={len(examples)} method={run_config.method} sample_fps={run_config.sample_fps} max_frames={run_config.max_frames}",
            )
        for index, example in enumerate(pending_examples, start=len(rows) + 1):
            subtitle_context = None
            try:
                _log_line(progress_path, f"[item_start] index={index}/{len(examples)} example_id={example.example_id} video={example.video_id}")
                item_start_time = time.perf_counter()
                retrieve_start_time = time.perf_counter()
                target_indices, retrieval_info = retriever.retrieve(example=example)
                retrieve_sec = time.perf_counter() - retrieve_start_time
                frame_load_start_time = time.perf_counter()
                frames, frame_hits, _ = load_selected_video_frames(
                    Path(example.video_path),
                    sample_fps=run_config.sample_fps,
                    target_indices=target_indices,
                    image_max_size=run_config.image_max_size,
                )
                frame_load_sec = time.perf_counter() - frame_load_start_time
                frame_times = [float(hit.time_sec) for hit in frame_hits]
                subtitle_texts: list[str] | None = None
                if run_config.include_subtitles:
                    subtitle_path = example.metadata.get("subtitle_path")
                    if subtitle_path:
                        subtitles = _load_subtitles(
                            subtitle_path=str(subtitle_path),
                            subtitle_root=subtitle_root,
                            subtitle_tar=subtitle_tar,
                        )
                        subtitle_texts, subtitle_context = _subtitle_texts_for_frames(
                            frame_times=frame_times,
                            subtitles=subtitles,
                            starting_timestamp_for_subtitles=float(example.metadata.get("starting_timestamp_for_subtitles", 0.0)),
                            duration=(float(example.metadata["duration"]) if example.metadata.get("duration") is not None else None),
                        )
                if run_config.evidence_text_mode == "chunks":
                    frame_texts = _build_chunked_frame_texts(
                        frame_hits=frame_hits,
                        retrieval_info=retrieval_info,
                        subtitle_texts=subtitle_texts,
                    )
                    prompt_prefix = _chunk_prompt_prefix(run_config.prompt_prefix)
                else:
                    frame_texts = _merge_frame_texts(frame_times=frame_times, subtitle_texts=subtitle_texts)
                    prompt_prefix = run_config.prompt_prefix
                answer_start_time = time.perf_counter()
                prediction = answerer.answer_frames(
                    frames=frames,
                    question=example.question,
                    options=example.options,
                    prompt_prefix=prompt_prefix,
                    frame_texts=frame_texts,
                )
                answer_wall_sec = time.perf_counter() - answer_start_time
                item_wall_sec = time.perf_counter() - item_start_time
            except Exception as exc:
                if _is_api_content_filter_error(exc):
                    row = {
                        "example_id": example.example_id,
                        "video_id": example.video_id,
                        "video_path": example.video_path,
                        "question": example.question,
                        "options": example.options,
                        "correct_index": example.correct_index,
                        "gold_letter": (chr(ord("A") + int(example.correct_index)) if example.correct_index is not None and int(example.correct_index) >= 0 else None),
                        "predicted_letter": None,
                        "choice_correct": None,
                        "raw_answer": f"API_BLOCKED: {type(exc).__name__}: {exc}",
                        "generation_sec": None,
                        "prompt_tokens": None,
                        "completion_tokens": None,
                        "total_tokens": None,
                        "subtitle_context": subtitle_context,
                        "method": run_config.method,
                        "status": "api_blocked",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        **example.metadata,
                    }
                    rows.append(row)
                    _append_jsonl(rows_path, row)
                    _write_json(rolling_summary_path, {"completed": len(rows), "total": len(examples), **_summarize_rows(rows)})
                    _log_line(progress_path, f"[item_blocked] index={index}/{len(examples)} example_id={example.example_id} error={type(exc).__name__}: {exc}")
                    continue
                _log_line(progress_path, f"[item_error] index={index}/{len(examples)} example_id={example.example_id} error={type(exc).__name__}: {exc}")
                with error_path.open("a", encoding="utf-8") as handle:
                    handle.write(f"{example.example_id}: {type(exc).__name__}: {exc}\n")
                raise

            gold_letter = chr(ord("A") + int(example.correct_index)) if example.correct_index is not None and int(example.correct_index) >= 0 else None
            row = {
                "example_id": example.example_id,
                "video_id": example.video_id,
                "video_path": example.video_path,
                "question": example.question,
                "options": example.options,
                "correct_index": example.correct_index,
                "gold_letter": gold_letter,
                "predicted_letter": prediction.predicted_letter,
                "choice_correct": (prediction.predicted_letter == gold_letter) if gold_letter is not None else None,
                "raw_answer": prediction.raw_text,
                "generation_sec": prediction.generation_sec,
                "answer_wall_sec": answer_wall_sec,
                "retrieve_sec": retrieve_sec,
                "frame_load_sec": frame_load_sec,
                "item_wall_sec": item_wall_sec,
                "prompt_tokens": prediction.prompt_tokens,
                "completion_tokens": prediction.completion_tokens,
                "total_tokens": prediction.total_tokens,
                "frame_times": frame_times,
                "frame_texts": frame_texts,
                "subtitle_context": subtitle_context,
                "method": run_config.method,
                "l3_rerank_evidence_source": retrieval_info.get("l3_rerank_evidence_source"),
                "frames": [
                    {"frame_index": int(hit.frame_index), "time_sec": float(hit.time_sec), "score": float(hit.score)}
                    for hit in retrieval_info["frame_hits"]
                ],
                "l2_hits": [
                    {"segment_id": hit.segment_id, "score": float(hit.score), "start_time_sec": float(hit.start_time_sec), "end_time_sec": float(hit.end_time_sec)}
                    for hit in retrieval_info["l2_hits"]
                ],
                "l3_hits": [
                    {"segment_id": hit.segment_id, "score": float(hit.score), "start_time_sec": float(hit.start_time_sec), "end_time_sec": float(hit.end_time_sec)}
                    for hit in retrieval_info["l3_hits"]
                ],
                **example.metadata,
            }
            rows.append(row)
            _append_jsonl(rows_path, row)
            _write_json(rolling_summary_path, {"completed": len(rows), "total": len(examples), **_summarize_rows(rows)})
            _log_line(
                progress_path,
                (
                    f"[item_done] index={index}/{len(examples)} example_id={example.example_id} "
                    f"predicted={prediction.predicted_letter} correct={row['choice_correct']} "
                    f"retrieve_sec={retrieve_sec:.3f} frame_load_sec={frame_load_sec:.3f} "
                    f"answer_wall_sec={answer_wall_sec:.3f} gen_sec={prediction.generation_sec} "
                    f"item_wall_sec={item_wall_sec:.3f}"
                ),
            )
    finally:
        answerer.unload()
        retriever.close()

    summary = {
        "run_config": {
            "method": run_config.method,
            "vector_backend": run_config.vector_backend,
            "sample_fps": run_config.sample_fps,
            "feature_eval_fps": run_config.feature_eval_fps,
            "max_frames": run_config.max_frames,
            "image_max_size": run_config.image_max_size,
                "include_subtitles": run_config.include_subtitles,
                "evidence_text_mode": run_config.evidence_text_mode,
                "l2_window_seconds": run_config.l2_window_seconds,
            "l2_stride_seconds": run_config.l2_stride_seconds,
            "l2_segmentation": run_config.l2_segmentation,
            "l2_local_min_duration_sec": run_config.l2_local_min_duration_sec,
            "l2_local_max_duration_sec": run_config.l2_local_max_duration_sec,
            "l2_local_fast_kernel_size": run_config.l2_local_fast_kernel_size,
            "l2_local_slow_kernel_size": run_config.l2_local_slow_kernel_size,
            "l2_local_peak_percentile": run_config.l2_local_peak_percentile,
            "l2_scoring": run_config.l2_scoring,
            "l2_frame_score_top_m": run_config.l2_frame_score_top_m,
            "l2_frame_score_temperature": run_config.l2_frame_score_temperature,
            "l2_rerank_encoder": run_config.l2_rerank_encoder,
            "l2_rerank_query_mode": run_config.l2_rerank_query_mode,
            "top_l2_segments": run_config.top_l2_segments,
            "top_l3_segments": run_config.top_l3_segments,
            "l3_rerank_keep": run_config.l3_rerank_keep,
            "l3_rerank_evidence_source": run_config.l3_rerank_evidence_source,
            "l2_evidence_per_l3": run_config.l2_evidence_per_l3,
            "l1_evidence_per_l2": run_config.l1_evidence_per_l2,
        },
        "answer_config": {
            "model_id": answer_config.model_id,
            "backend": answer_config.backend,
            "load_in_4bit": answer_config.load_in_4bit,
            "load_in_8bit": answer_config.load_in_8bit,
        },
        **_summarize_rows(rows),
    }
    _write_json(output_root / "final_summary.json", summary)
    _log_line(progress_path, f"[done] scored={summary['scored']} accuracy={summary['choice_accuracy']}")
    return summary
