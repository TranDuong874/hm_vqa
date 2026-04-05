from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

os.environ["HF_HUB_DISABLE_XET"] = "1"

import torch
from dotenv import load_dotenv

from segmentation import cosine_drift, frame_energy_diff, sample_video, segment_by_threshold
from retrieval import (
    EvidencePackage,
    PipelineConfig,
    VideoIndex,
    build_score,
    collect_candidate_low_segments,
    mean_pool_segments,
    retrieve_top_segments,
    select_evidence_frames,
)
from ingestion import DINOEncoder, OpenCLIPEncoder
from retrieval import SampledVideo


def configure_hf_env(env_path: str | Path | None = None) -> None:
    if env_path is None:
        load_dotenv()
    else:
        load_dotenv(Path(env_path))
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", hf_token)


class HMVQAPipeline:
    CACHE_VERSION = 1
    OPENCLIP_MODEL_NAME = "ViT-L-14"
    OPENCLIP_PRETRAINED = "laion2b_s32b_b82k"
    DINO_MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"

    def __init__(
        self,
        config: PipelineConfig | None = None,
        *,
        feature_cache_dir: str | Path | None = None,
        refresh_feature_cache: bool = False,
    ) -> None:
        self.config = config or PipelineConfig()
        self._openclip: OpenCLIPEncoder | None = None
        self._dino: DINOEncoder | None = None
        self.feature_cache_dir = Path(feature_cache_dir) if feature_cache_dir is not None else None
        self.refresh_feature_cache = bool(refresh_feature_cache)
        if self.feature_cache_dir is not None:
            self.feature_cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_openclip(self) -> OpenCLIPEncoder:
        if self._openclip is None:
            self._openclip = OpenCLIPEncoder(device=self.config.device)
        return self._openclip

    def _get_dino(self) -> DINOEncoder:
        if self._dino is None:
            self._dino = DINOEncoder(device=self.config.device)
        return self._dino

    def release_encoders(self) -> None:
        if self._openclip is not None:
            del self._openclip
        if self._dino is not None:
            del self._dino
        self._openclip = None
        self._dino = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _feature_cache_path(self, video_path: Path) -> Path | None:
        if self.feature_cache_dir is None:
            return None
        resolved = str(video_path.resolve())
        key = hashlib.sha256(
            f"{resolved}|fps={self.config.sample_fps:.6f}|openclip={self.OPENCLIP_MODEL_NAME}|"
            f"{self.OPENCLIP_PRETRAINED}|dino={self.DINO_MODEL_ID}|v={self.CACHE_VERSION}".encode("utf-8")
        ).hexdigest()[:16]
        return self.feature_cache_dir / f"{video_path.stem}_{key}.pt"

    def _load_cached_features(self, video_path: Path) -> dict[str, Any] | None:
        cache_path = self._feature_cache_path(video_path)
        if cache_path is None or self.refresh_feature_cache or not cache_path.exists():
            return None
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        expected = {
            "cache_version": self.CACHE_VERSION,
            "sample_fps": float(self.config.sample_fps),
            "openclip_model_name": self.OPENCLIP_MODEL_NAME,
            "openclip_pretrained": self.OPENCLIP_PRETRAINED,
            "dino_model_id": self.DINO_MODEL_ID,
        }
        for key, value in expected.items():
            if payload.get(key) != value:
                return None
        return payload

    def _save_cached_features(
        self,
        video_path: Path,
        *,
        timestamps,
        native_fps: float,
        openclip_embeddings: torch.Tensor,
        dino_embeddings: torch.Tensor,
        energy_signal,
    ) -> None:
        cache_path = self._feature_cache_path(video_path)
        if cache_path is None:
            return
        payload = {
            "cache_version": self.CACHE_VERSION,
            "video_path": str(video_path.resolve()),
            "sample_fps": float(self.config.sample_fps),
            "openclip_model_name": self.OPENCLIP_MODEL_NAME,
            "openclip_pretrained": self.OPENCLIP_PRETRAINED,
            "dino_model_id": self.DINO_MODEL_ID,
            "timestamps": timestamps,
            "native_fps": float(native_fps),
            "openclip_embeddings": openclip_embeddings.cpu(),
            "dino_embeddings": dino_embeddings.cpu(),
            "energy_signal": energy_signal,
        }
        torch.save(payload, cache_path)

    def build_index(self, video_path: str | Path) -> VideoIndex:
        path = Path(video_path)
        cached = self._load_cached_features(path)
        if cached is not None:
            pil_frames, _, sampled_timestamps, native_fps = sample_video(path, self.config.sample_fps, include_bgr=False)
            timestamps = cached["timestamps"]
            if len(pil_frames) != len(timestamps) or len(sampled_timestamps) != len(timestamps):
                cached = None
            else:
                openclip_embeddings = cached["openclip_embeddings"].float().cpu()
                dino_embeddings = cached["dino_embeddings"].float().cpu()
                energy_signal = cached["energy_signal"]
        if cached is None:
            pil_frames, bgr_frames, timestamps, native_fps = sample_video(path, self.config.sample_fps)
            openclip = self._get_openclip()
            dino = self._get_dino()
            openclip_embeddings = openclip.encode_images(pil_frames, batch_size=self.config.openclip_batch_size)
            dino_embeddings = dino.encode_images(pil_frames, batch_size=self.config.dino_batch_size)
            energy_signal = frame_energy_diff(bgr_frames)
            self._save_cached_features(
                path,
                timestamps=timestamps,
                native_fps=native_fps,
                openclip_embeddings=openclip_embeddings,
                dino_embeddings=dino_embeddings,
                energy_signal=energy_signal,
            )
        sampled = SampledVideo(
            video_path=path,
            pil_frames=pil_frames,
            timestamps=timestamps,
            native_fps=native_fps,
        )
        openclip_signal = cosine_drift(openclip_embeddings)
        dino_signal = cosine_drift(dino_embeddings)
        combined_score = build_score(
            energy_signal=energy_signal,
            openclip_signal=openclip_signal,
            dino_signal=dino_signal,
            config=self.config,
        )

        low_segments = segment_by_threshold(
            timestamps=sampled.timestamps,
            score=combined_score,
            threshold=self.config.low_threshold,
            min_seconds=self.config.low_min_seconds,
            max_seconds=None,
            level="low",
        )
        high_segments = segment_by_threshold(
            timestamps=sampled.timestamps,
            score=combined_score,
            threshold=self.config.high_threshold,
            min_seconds=self.config.high_min_seconds,
            max_seconds=None,
            level="high",
        )

        low_embeddings = mean_pool_segments(openclip_embeddings, low_segments)
        high_embeddings = mean_pool_segments(openclip_embeddings, high_segments)

        return VideoIndex(
            sampled_video=sampled,
            low_segments=low_segments,
            high_segments=high_segments,
            low_embeddings=low_embeddings,
            high_embeddings=high_embeddings,
            energy_signal=energy_signal,
            openclip_signal=openclip_signal,
            dino_signal=dino_signal,
            combined_score=combined_score,
        )

    def retrieve(
        self,
        *,
        index: VideoIndex,
        question: str,
        options: list[str],
    ) -> EvidencePackage:
        query_text = question + "\n" + "\n".join(options)
        query_embedding = self._get_openclip().encode_texts([query_text])[0]
        high_hits = retrieve_top_segments(
            query_embedding=query_embedding,
            segment_embeddings=index.high_embeddings,
            segments=index.high_segments,
            top_k=self.config.top_high,
        )
        low_hits = collect_candidate_low_segments(
            high_hits=high_hits,
            low_segments=index.low_segments,
            low_embeddings=index.low_embeddings,
            query_embedding=query_embedding,
            top_k=self.config.top_low,
        )
        evidence_frames, evidence_meta = select_evidence_frames(
            pil_frames=index.sampled_video.pil_frames,
            timestamps=index.sampled_video.timestamps,
            low_hits=low_hits,
            max_frames=self.config.max_evidence_frames,
        )
        return EvidencePackage(
            question=question,
            options=options,
            high_hits=high_hits,
            low_hits=low_hits,
            evidence_frames=evidence_frames,
            evidence_meta=evidence_meta,
        )


if __name__ == "__main__":
    configure_hf_env(Path(__file__).resolve().parents[1] / ".env")
    print("hm_vqa_pipeline is now a reusable module.")
    print("Import HMVQAPipeline from this file, retrieval helpers from src/retrieval, and answerers from src/answering.")
