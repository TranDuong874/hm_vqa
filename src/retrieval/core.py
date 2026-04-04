from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from analysis import Segment, minmax_normalize, sample_video


@dataclass(slots=True)
class PipelineConfig:
    sample_fps: float = 4.0
    low_threshold: float = 0.32
    high_threshold: float = 0.58
    low_min_seconds: float = 2.0
    high_min_seconds: float = 12.0
    openclip_weight: float = 0.45
    dino_weight: float = 0.35
    energy_weight: float = 0.20
    top_high: int = 2
    top_low: int = 3
    max_evidence_frames: int = 8
    openclip_batch_size: int = 16
    dino_batch_size: int = 16
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(slots=True)
class SampledVideo:
    video_path: Path
    pil_frames: list[Image.Image]
    timestamps: np.ndarray
    native_fps: float


@dataclass(slots=True)
class VideoIndex:
    sampled_video: SampledVideo
    low_segments: list[Segment]
    high_segments: list[Segment]
    low_embeddings: torch.Tensor
    high_embeddings: torch.Tensor
    energy_signal: np.ndarray
    openclip_signal: np.ndarray
    dino_signal: np.ndarray
    combined_score: np.ndarray


@dataclass(slots=True)
class EvidencePackage:
    question: str
    options: list[str]
    high_hits: list[dict[str, Any]]
    low_hits: list[dict[str, Any]]
    evidence_frames: list[Image.Image]
    evidence_meta: list[dict[str, Any]]


def load_video_frames(video_path: str | Path, sample_fps: float) -> SampledVideo:
    path = Path(video_path)
    pil_frames, _, timestamps, native_fps = sample_video(path, sample_fps)
    return SampledVideo(
        video_path=path,
        pil_frames=pil_frames,
        timestamps=timestamps,
        native_fps=native_fps,
    )


def build_score(
    *,
    energy_signal: np.ndarray,
    openclip_signal: np.ndarray,
    dino_signal: np.ndarray,
    config: PipelineConfig,
) -> np.ndarray:
    weighted_parts = [
        minmax_normalize(openclip_signal) * config.openclip_weight,
        minmax_normalize(dino_signal) * config.dino_weight,
        minmax_normalize(energy_signal) * config.energy_weight,
    ]
    total = config.openclip_weight + config.dino_weight + config.energy_weight
    return np.sum(np.stack(weighted_parts, axis=0), axis=0) / total


def mean_pool_segments(frame_embeddings: torch.Tensor, segments: list[Segment]) -> torch.Tensor:
    pooled: list[torch.Tensor] = []
    for segment in segments:
        segment_embeddings = frame_embeddings[segment.start_index : segment.end_index + 1]
        pooled.append(segment_embeddings.mean(dim=0))
    if not pooled:
        width = frame_embeddings.shape[-1] if frame_embeddings.ndim == 2 else 0
        return torch.empty((0, width), dtype=torch.float32)
    result = torch.stack(pooled, dim=0)
    return torch.nn.functional.normalize(result, dim=-1)


def retrieve_top_segments(
    query_embedding: torch.Tensor,
    segment_embeddings: torch.Tensor,
    segments: list[Segment],
    top_k: int,
) -> list[dict[str, Any]]:
    if len(segments) == 0 or segment_embeddings.numel() == 0:
        return []
    scores = torch.matmul(segment_embeddings, query_embedding).cpu().numpy()
    order = np.argsort(-scores)
    results: list[dict[str, Any]] = []
    for idx in order[:top_k]:
        segment = segments[int(idx)]
        results.append(
            {
                "segment_id": segment.segment_id,
                "level": segment.level,
                "score": float(scores[int(idx)]),
                "start_index": segment.start_index,
                "end_index": segment.end_index,
                "start_time_sec": segment.start_time_sec,
                "end_time_sec": segment.end_time_sec,
                "duration_sec": segment.duration_sec,
            }
        )
    return results


def collect_candidate_low_segments(
    *,
    high_hits: list[dict[str, Any]],
    low_segments: list[Segment],
    low_embeddings: torch.Tensor,
    query_embedding: torch.Tensor,
    top_k: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for low_index, low_segment in enumerate(low_segments):
        for high_hit in high_hits:
            if low_segment.start_index >= high_hit["start_index"] and low_segment.end_index <= high_hit["end_index"]:
                score = float(torch.dot(low_embeddings[low_index], query_embedding).item())
                candidates.append(
                    {
                        "segment_id": low_segment.segment_id,
                        "level": low_segment.level,
                        "score": score,
                        "start_index": low_segment.start_index,
                        "end_index": low_segment.end_index,
                        "start_time_sec": low_segment.start_time_sec,
                        "end_time_sec": low_segment.end_time_sec,
                        "duration_sec": low_segment.duration_sec,
                    }
                )
                break
    candidates.sort(key=lambda row: row["score"], reverse=True)
    return candidates[:top_k]


def select_evidence_frames(
    *,
    pil_frames: list[Image.Image],
    timestamps: np.ndarray,
    low_hits: list[dict[str, Any]],
    max_frames: int,
) -> tuple[list[Image.Image], list[dict[str, Any]]]:
    selected_indices: list[int] = []
    per_segment = max(1, max_frames // max(len(low_hits), 1))

    for hit in low_hits:
        start = int(hit["start_index"])
        end = int(hit["end_index"])
        segment_indices = list(range(start, end + 1))
        if not segment_indices:
            continue
        if len(segment_indices) <= per_segment:
            picked = segment_indices
        else:
            chosen = torch.linspace(0, len(segment_indices) - 1, per_segment).round().long().tolist()
            picked = [segment_indices[index] for index in chosen]
        selected_indices.extend(picked)

    selected_indices = sorted(set(selected_indices))
    if len(selected_indices) > max_frames:
        chosen = torch.linspace(0, len(selected_indices) - 1, max_frames).round().long().tolist()
        selected_indices = [selected_indices[index] for index in chosen]

    frames = [pil_frames[index] for index in selected_indices]
    meta = [{"frame_index": int(index), "time_sec": float(timestamps[index])} for index in selected_indices]
    return frames, meta


def select_uniform_frames(
    *,
    pil_frames: list[Image.Image],
    timestamps: np.ndarray,
    max_frames: int,
) -> tuple[list[Image.Image], list[dict[str, Any]]]:
    if len(pil_frames) <= max_frames:
        indices = list(range(len(pil_frames)))
    else:
        indices = torch.linspace(0, len(pil_frames) - 1, max_frames).round().long().tolist()
    frames = [pil_frames[index] for index in indices]
    meta = [{"frame_index": int(index), "time_sec": float(timestamps[index])} for index in indices]
    return frames, meta


def export_frames(
    *,
    frames: list[Image.Image],
    meta: list[dict[str, Any]],
    output_dir: str | Path,
) -> None:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    for index, (frame, frame_meta) in enumerate(zip(frames, meta)):
        time_tag = f"{frame_meta['time_sec']:.2f}s".replace(".", "_")
        frame.save(directory / f"{index:02d}_{time_tag}.png")
