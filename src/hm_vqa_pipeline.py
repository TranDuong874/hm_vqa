from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ["HF_HUB_DISABLE_XET"] = "1"

import cv2
import numpy as np
import torch
from dotenv import load_dotenv
from PIL import Image

from segmentation import sample_video_selected_indices
from retrieval import (
    adapt_query_embedding_for_segment_pooling,
    EvidencePackage,
    PipelineConfig,
    VideoIndex,
    build_query_text,
    build_window_segments,
    collect_segment_frame_indices,
    pool_segments,
    retrieve_top_frames,
    retrieve_top_segments,
    select_evidence_frames,
)
from ingestion import OpenCLIPEncoder
from retrieval import SampledVideo
from segmentation import Segment


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
    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self._openclip: OpenCLIPEncoder | None = None

    def _get_openclip(self) -> OpenCLIPEncoder:
        if self._openclip is None:
            self._openclip = OpenCLIPEncoder(device=self.config.device)
        return self._openclip

    def release_encoder(self) -> None:
        if self._openclip is not None:
            del self._openclip
        self._openclip = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _cache_root(self) -> Path:
        return Path(".cache") / "hm_vqa" / "video_index"

    def _cache_key(self, video_path: Path) -> str:
        payload = {
            "video_path": str(video_path.resolve()),
            "sample_fps": float(self.config.sample_fps),
            "image_max_size": int(self.config.image_max_size or 0),
            "window_seconds": float(self.config.window_seconds),
            "window_stride_seconds": float(self.config.window_stride_seconds),
            "layer2_pooling": self.config.layer2_pooling,
            "model_name": "ViT-L-14",
            "pretrained": "laion2b_s32b_b82k",
        }
        raw = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha1(raw).hexdigest()[:16]

    def _cache_dir(self, video_path: Path) -> Path:
        return self._cache_root() / self._cache_key(video_path)

    def _stream_sampled_frame_embeddings(
        self,
        video_path: Path,
    ) -> tuple[torch.Tensor, np.ndarray, float]:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        native_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if native_fps <= 0.0 or total_frames <= 0:
            capture.release()
            raise RuntimeError(f"Invalid fps/frame count for video: {video_path}")

        step = max(int(round(native_fps / self.config.sample_fps)), 1)
        batch_images: list[Image.Image] = []
        timestamps: list[float] = []
        embedding_batches: list[torch.Tensor] = []

        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % step == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                if self.config.image_max_size is not None and self.config.image_max_size > 0:
                    image.thumbnail((self.config.image_max_size, self.config.image_max_size), Image.Resampling.LANCZOS)
                batch_images.append(image)
                timestamps.append(frame_index / native_fps)
                if len(batch_images) >= self.config.openclip_batch_size:
                    embedding_batches.append(
                        self._get_openclip().encode_images(batch_images, batch_size=self.config.openclip_batch_size)
                    )
                    batch_images.clear()
            frame_index += 1

        capture.release()
        if batch_images:
            embedding_batches.append(
                self._get_openclip().encode_images(batch_images, batch_size=self.config.openclip_batch_size)
            )
        if not embedding_batches:
            raise RuntimeError(f"No frames sampled from video: {video_path}")

        embeddings = torch.cat(embedding_batches, dim=0)
        return embeddings, np.asarray(timestamps, dtype=np.float32), native_fps

    def _save_cached_index(
        self,
        *,
        cache_dir: Path,
        video_path: Path,
        timestamps: np.ndarray,
        native_fps: float,
        frame_embeddings: torch.Tensor,
        window_segments: list[Segment],
        window_embeddings: torch.Tensor,
    ) -> None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "video_path": str(video_path.resolve()),
            "sample_fps": float(self.config.sample_fps),
            "image_max_size": int(self.config.image_max_size or 0),
            "native_fps": float(native_fps),
        }
        (cache_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        np.save(cache_dir / "timestamps.npy", timestamps)
        torch.save(frame_embeddings, cache_dir / "frame_embeddings.pt")
        torch.save(window_embeddings, cache_dir / "window_embeddings.pt")
        (cache_dir / "window_segments.json").write_text(
            json.dumps(
                [
                    {
                        "segment_id": segment.segment_id,
                        "start_index": int(segment.start_index),
                        "end_index": int(segment.end_index),
                        "start_time_sec": float(segment.start_time_sec),
                        "end_time_sec": float(segment.end_time_sec),
                        "duration_sec": float(segment.duration_sec),
                    }
                    for segment in window_segments
                ],
                indent=2,
            ),
            encoding="utf-8",
        )

    def _load_cached_index(self, cache_dir: Path) -> VideoIndex | None:
        metadata_path = cache_dir / "metadata.json"
        timestamps_path = cache_dir / "timestamps.npy"
        frame_embeddings_path = cache_dir / "frame_embeddings.pt"
        window_embeddings_path = cache_dir / "window_embeddings.pt"
        segments_path = cache_dir / "window_segments.json"
        required = [metadata_path, timestamps_path, frame_embeddings_path, window_embeddings_path, segments_path]
        if not all(path.exists() for path in required):
            return None

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        timestamps = np.load(timestamps_path)
        frame_embeddings = torch.load(frame_embeddings_path, map_location="cpu")
        window_embeddings = torch.load(window_embeddings_path, map_location="cpu")
        window_segments = [Segment(**item) for item in json.loads(segments_path.read_text(encoding="utf-8"))]
        sampled = SampledVideo(
            video_path=Path(metadata["video_path"]),
            frames=[],
            timestamps=timestamps,
            native_fps=float(metadata["native_fps"]),
        )
        return VideoIndex(
            sampled_video=sampled,
            frame_embeddings=frame_embeddings,
            window_segments=window_segments,
            window_embeddings=window_embeddings,
        )

    def build_index(self, video_path: str | Path) -> VideoIndex:
        path = Path(video_path)
        cache_dir = self._cache_dir(path)
        cached = self._load_cached_index(cache_dir)
        if cached is not None:
            return cached

        openclip_embeddings, timestamps, native_fps = self._stream_sampled_frame_embeddings(path)
        sampled = SampledVideo(video_path=path, frames=[], timestamps=timestamps, native_fps=native_fps)
        window_segments = build_window_segments(sampled, self.config)
        window_embeddings = pool_segments(
            openclip_embeddings,
            window_segments,
            pooling=self.config.layer2_pooling,
        )
        index = VideoIndex(
            sampled_video=sampled,
            frame_embeddings=openclip_embeddings,
            window_segments=window_segments,
            window_embeddings=window_embeddings,
        )
        self._save_cached_index(
            cache_dir=cache_dir,
            video_path=path,
            timestamps=timestamps,
            native_fps=native_fps,
            frame_embeddings=openclip_embeddings,
            window_segments=window_segments,
            window_embeddings=window_embeddings,
        )
        return index

    def retrieve(
        self,
        *,
        index: VideoIndex,
        question: str,
        options: list[str],
    ) -> EvidencePackage:
        query_text = build_query_text(question, options)
        query_embedding = self._get_openclip().encode_texts([query_text])[0]
        window_query_embedding = adapt_query_embedding_for_segment_pooling(
            query_embedding,
            pooling=self.config.layer2_pooling,
        )
        window_hits = retrieve_top_segments(
            query_embedding=window_query_embedding,
            segment_embeddings=index.window_embeddings,
            segments=index.window_segments,
            top_k=self.config.top_windows,
        )
        candidate_indices = collect_segment_frame_indices(window_hits)
        frame_hits = retrieve_top_frames(
            query_embedding=query_embedding,
            frame_embeddings=index.frame_embeddings,
            timestamps=index.sampled_video.timestamps,
            allowed_indices=candidate_indices,
            top_k=self.config.max_evidence_frames,
        )
        if not frame_hits:
            frame_hits = retrieve_top_frames(
                query_embedding=query_embedding,
                frame_embeddings=index.frame_embeddings,
                timestamps=index.sampled_video.timestamps,
                top_k=self.config.max_evidence_frames,
            )
        evidence_frames, _, _ = sample_video_selected_indices(
            index.sampled_video.video_path,
            self.config.sample_fps,
            target_indices=[int(hit.frame_index) for hit in frame_hits],
            image_max_size=self.config.image_max_size,
        )
        return EvidencePackage(
            question=question,
            options=options,
            window_hits=window_hits,
            frame_hits=frame_hits,
            evidence_frames=evidence_frames,
        )


if __name__ == "__main__":
    configure_hf_env(Path(__file__).resolve().parents[1] / ".env")
    print("hm_vqa_pipeline is now a reusable module.")
    print("This minimal baseline keeps only OpenCLIP frames, fixed windows, and simple retrieval.")
