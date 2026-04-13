from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass(slots=True)
class FeatureArchive:
    video_id: str
    timestamps: np.ndarray
    frame_indices: torch.Tensor
    frame_embeddings: torch.Tensor
    fps: float
    total_frames: int
    model_name: str
    pretrained_name: str
    source_dir: Path

    @property
    def duration_sec(self) -> float:
        if self.fps > 0.0 and self.total_frames > 0:
            return float(self.total_frames / self.fps)
        if len(self.timestamps):
            return float(self.timestamps[-1])
        return 0.0


def _parse_openclip_model_string(value: str) -> tuple[str, str]:
    if " / " in value:
        model_name, pretrained = value.split(" / ", 1)
        return model_name.strip(), pretrained.strip()
    return "ViT-L-14", "datacomp_xl_s13b_b90k"


def _feature_dir(repo_root: Path, video_id: str) -> Path:
    candidates = sorted((repo_root / "dataset").glob(f"hd_epic_features_*/{video_id}"))
    if not candidates:
        raise FileNotFoundError(f"No precomputed HD-EPIC feature archive found for {video_id}")
    return candidates[0]


def load_feature_archive(repo_root: Path, video_id: str) -> FeatureArchive:
    feature_dir = _feature_dir(repo_root, video_id)
    meta_path = feature_dir / "meta.json"
    shard_paths = sorted(feature_dir.glob("shard_*.pt"))
    if not meta_path.exists() or not shard_paths:
        raise FileNotFoundError(f"Incomplete feature archive for {video_id}: {feature_dir}")

    metadata = json.loads(meta_path.read_text())
    frame_indices: list[torch.Tensor] = []
    timestamps: list[torch.Tensor] = []
    embeddings: list[torch.Tensor] = []
    for shard_path in shard_paths:
        payload = torch.load(shard_path, map_location="cpu", weights_only=False)
        frame_indices.append(payload["frame_idx"].to(torch.int64))
        timestamps.append(payload["timestamp_sec"].to(torch.float32))
        embeddings.append(payload["openclip"].to(torch.float32))

    all_frame_indices = torch.cat(frame_indices, dim=0)
    all_timestamps = torch.cat(timestamps, dim=0).numpy()
    all_embeddings = torch.nn.functional.normalize(torch.cat(embeddings, dim=0), dim=-1)
    model_name, pretrained_name = _parse_openclip_model_string(str(metadata.get("openclip_model", "")))
    return FeatureArchive(
        video_id=video_id,
        timestamps=all_timestamps,
        frame_indices=all_frame_indices,
        frame_embeddings=all_embeddings,
        fps=float(metadata.get("fps") or 0.0),
        total_frames=int(metadata.get("total_frames") or 0),
        model_name=model_name,
        pretrained_name=pretrained_name,
        source_dir=feature_dir,
    )


def _ensure_src_on_path(repo_root: Path) -> None:
    src_path = str(repo_root / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


def build_query_encoder(
    *,
    repo_root: Path,
    model_name: str,
    pretrained_name: str,
    device: str,
) -> Any:
    _ensure_src_on_path(repo_root)
    from ingestion import OpenCLIPEncoder

    return OpenCLIPEncoder(
        model_name=model_name,
        pretrained=pretrained_name,
        device=device,
    )
