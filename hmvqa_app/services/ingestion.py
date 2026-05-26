from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from hmvqa_app.config import AppConfig
from hmvqa_app.runtime import (
    OpenCLIPEncoder,
    Segment,
    pool_segments,
    segment_fixed_windows,
    write_ip_index,
)
from hmvqa_app.runtime.devices import release_model
from hmvqa_app.services.session import SessionService
from hmvqa_app.services.storage import StorageService


@dataclass(slots=True)
class SampledFrames:
    frames: list[Image.Image]
    timestamps: np.ndarray
    native_fps: float


@dataclass(slots=True)
class MemoryArtifacts:
    frame_embeddings: torch.Tensor
    l2_embeddings: torch.Tensor
    l3_embeddings: torch.Tensor
    l2_segments: list[Segment]
    l3_segments: list[Segment]


class IngestionService:
    def __init__(self, config: AppConfig, storage: StorageService, sessions: SessionService) -> None:
        self.config = config
        self.storage = storage
        self.sessions = sessions
        self._encoder: OpenCLIPEncoder | None = None
        self._viclip_encoder: Any | None = None

    def _get_encoder(self) -> OpenCLIPEncoder:
        if self._encoder is None:
            self._encoder = OpenCLIPEncoder(device=self.config.openclip_device)
        return self._encoder

    def _get_viclip_encoder(self) -> Any:
        if self._viclip_encoder is None:
            from hmvqa_app.runtime.viclip import ViCLIPEncoder

            self._viclip_encoder = ViCLIPEncoder(device=self.config.viclip_device)
        return self._viclip_encoder

    def ingest(self, *, session_id: str, video_path: Path, original_name: str, sample_fps: float) -> None:
        started = time.perf_counter()
        sample_fps = self.config.clamp_sample_fps(sample_fps)
        try:
            if self._load_cached_if_ready(session_id):
                return

            self.sessions.patch(session_id, status="processing", progress=4, message="Opening video")
            sampled = self._sample_frames(session_id, video_path, sample_fps)
            artifacts = self._build_memory(session_id, sampled)
            self._write_artifacts(
                session_id=session_id,
                video_path=video_path,
                original_name=original_name,
                sample_fps=sample_fps,
                sampled=sampled,
                artifacts=artifacts,
                elapsed_sec=time.perf_counter() - started,
            )
            self._mark_ready(session_id, sampled)
        except Exception as exc:
            self.sessions.patch(session_id, status="error", progress=100, message="Ingestion failed.", error=str(exc))
        finally:
            if self.config.unload_encoders_after_request:
                self._release_openclip()
                self._release_viclip()

    def _load_cached_if_ready(self, session_id: str) -> bool:
        if not self.storage.is_ready(session_id):
            return False
        metadata = self.storage.read_json(self.storage.metadata_path(session_id))
        self.sessions.patch(
            session_id,
            status="ready",
            progress=100,
            message="Loaded cached memory. Ask a question about the video.",
            duration_sec=metadata.get("duration_sec"),
            sampled_frames=metadata.get("sampled_frames"),
            cache_hit=True,
        )
        return True

    def _build_memory(self, session_id: str, sampled: SampledFrames) -> MemoryArtifacts:
        frame_embeddings = self._encode_frame_embeddings(session_id, sampled.frames)

        self.sessions.patch(session_id, progress=72, message="Building L2/L3 memory")

        l2_segments = self._segments(sampled.timestamps, self.config.l2_seconds, "l2")
        l3_segments = self._segments(sampled.timestamps, self.config.l3_seconds, "l3")

        l2_embeddings = self._encode_l2_embeddings(sampled.frames, frame_embeddings, l2_segments)
        l3_embeddings = pool_segments(frame_embeddings, l3_segments, pooling="mean")

        return MemoryArtifacts(
            frame_embeddings=frame_embeddings,
            l2_embeddings=l2_embeddings,
            l3_embeddings=l3_embeddings,
            l2_segments=l2_segments,
            l3_segments=l3_segments,
        )

    def _encode_frame_embeddings(self, session_id: str, frames: list[Image.Image]) -> torch.Tensor:
        self._release_viclip()

        self.sessions.patch(
            session_id,
            progress=42,
            message=f"Encoding {len(frames)} sampled frames with OpenCLIP",
        )

        embeddings: list[torch.Tensor] = []
        encoder = self._get_encoder()

        for start in range(0, len(frames), self.config.openclip_batch_size):
            batch = frames[start : start + self.config.openclip_batch_size]
            embeddings.append(encoder.encode_images(batch, batch_size=self.config.openclip_batch_size))

            progress = 42 + int(((start + len(batch)) / max(len(frames), 1)) * 28)
            self.sessions.patch(
                session_id,
                progress=min(progress, 70),
                message=f"Encoding frames: {start + len(batch)}/{len(frames)}",
            )

        frame_embeddings = torch.cat(embeddings, dim=0)

        if self.config.use_viclip_l2:
            self._release_openclip()

        return frame_embeddings

    def _write_artifacts(
        self,
        *,
        session_id: str,
        video_path: Path,
        original_name: str,
        sample_fps: float,
        sampled: SampledFrames,
        artifacts: MemoryArtifacts,
        elapsed_sec: float,
    ) -> None:
        self.sessions.patch(session_id, progress=84, message="Writing FAISS indexes")
        session_dir = self.storage.session_dir(session_id)
        torch.save(artifacts.frame_embeddings, session_dir / "frame_embeddings.pt")
        torch.save(artifacts.l2_embeddings, session_dir / "l2_embeddings.pt")
        torch.save(artifacts.l3_embeddings, session_dir / "l3_embeddings.pt")
        np.save(session_dir / "timestamps.npy", sampled.timestamps)

        write_ip_index(session_dir / "frame.index", artifacts.frame_embeddings)
        write_ip_index(session_dir / "l2.index", artifacts.l2_embeddings)
        write_ip_index(session_dir / "l3.index", artifacts.l3_embeddings)

        self.storage.write_json(
            session_dir / "l2_segments.json",
            self.storage.segments_to_json(artifacts.l2_segments),
        )
        self.storage.write_json(
            session_dir / "l3_segments.json",
            self.storage.segments_to_json(artifacts.l3_segments),
        )
        self.storage.write_json(
            self.storage.metadata_path(session_id),
            {
                "schema_version": self.config.schema_version,
                "session_id": session_id,
                "video_name": original_name,
                "video_path": str(video_path),
                "native_fps": float(sampled.native_fps),
                "sample_fps": float(sample_fps),
                "duration_sec": float(sampled.timestamps[-1]) if len(sampled.timestamps) else 0.0,
                "sampled_frames": len(sampled.frames),
                "ingest_sec": round(elapsed_sec, 3),
                "openclip_device": self.config.openclip_device,
                "viclip_device": self.config.viclip_device,
                "l2_encoder": "viclip" if self.config.use_viclip_l2 else "openclip_mean",
                "l2_seconds": self.config.l2_seconds,
                "l3_seconds": self.config.l3_seconds,
            },
        )

    def _mark_ready(self, session_id: str, sampled: SampledFrames) -> None:
        self.sessions.patch(
            session_id,
            status="ready",
            progress=100,
            message="Ingestion complete. Ask a question about the video.",
            duration_sec=float(sampled.timestamps[-1]) if len(sampled.timestamps) else 0.0,
            sampled_frames=len(sampled.frames),
        )

    def _sample_frames(self, session_id: str, video_path: Path, sample_fps: float) -> SampledFrames:
        capture = cv2.VideoCapture(str(video_path))

        if not capture.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        native_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        if native_fps <= 0.0 or total_frames <= 0:
            capture.release()
            raise RuntimeError("Invalid video FPS or frame count.")

        step = max(int(round(native_fps / sample_fps)), 1)
        frame_dir = self.storage.frame_dir(session_id)
        frame_dir.mkdir(parents=True, exist_ok=True)
        frames: list[Image.Image] = []
        timestamps: list[float] = []
        sampled_index = 0
        native_index = 0
        last_progress = 5

        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                if native_index % step == 0:
                    image = self._resize_frame(frame)
                    frame_id = f"frame_{sampled_index:06d}.jpg"
                    image.save(self.storage.frame_path(session_id, frame_id), quality=90)

                    frames.append(image)
                    timestamps.append(native_index / native_fps)
                    sampled_index += 1

                if native_index % max(step * 8, 1) == 0:
                    progress = 5 + int(min(native_index / max(total_frames, 1), 1.0) * 35)
                    if progress > last_progress:
                        last_progress = progress
                        self.sessions.patch(
                            session_id,
                            progress=progress,
                            message=f"Sampling frames: {sampled_index} captured",
                        )

                native_index += 1
        finally:
            capture.release()

        if not frames:
            raise RuntimeError("No frames sampled from uploaded video.")

        return SampledFrames(
            frames=frames,
            timestamps=np.asarray(timestamps, dtype=np.float32),
            native_fps=native_fps,
        )

    def _encode_l2_embeddings(
        self,
        frames: list[Image.Image],
        frame_embeddings: torch.Tensor,
        segments: list[Segment],
    ) -> torch.Tensor:
        if not self.config.use_viclip_l2:
            return pool_segments(frame_embeddings, segments, pooling="mean")

        self._release_openclip()
        encoder = self._get_viclip_encoder()
        clips = [self._clip_from_segment(frames, segment, encoder.num_frames) for segment in segments]
        embeddings: list[torch.Tensor] = []
        total = len(clips)

        try:
            for start in range(0, total, self.config.viclip_batch_size):
                batch = clips[start : start + self.config.viclip_batch_size]
                batch_embeddings = encoder.encode_video_clips(batch, batch_size=self.config.viclip_batch_size)
                embeddings.append(batch_embeddings.float().cpu())

            if embeddings:
                return torch.cat(embeddings, dim=0)

            return torch.empty((0, frame_embeddings.shape[-1]), dtype=torch.float32)
        finally:
            self._release_viclip()

    def _resize_frame(self, frame: np.ndarray) -> Image.Image:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb).convert("RGB")
        image.thumbnail(
            (self.config.display_frame_size, self.config.display_frame_size),
            Image.Resampling.LANCZOS,
        )
        return image

    def _release_openclip(self) -> None:
        release_model(self, "_encoder")

    def _release_viclip(self) -> None:
        release_model(self, "_viclip_encoder")

    @staticmethod
    def _clip_from_segment(frames: list[Image.Image], segment: Segment, num_frames: int) -> list[Image.Image]:
        start = max(0, int(segment.start_index))
        end = min(len(frames) - 1, int(segment.end_index))
        if end < start:
            return [frames[0]] * max(1, int(num_frames))
        count = max(1, int(num_frames))
        local_indices = np.linspace(start, end, num=count, dtype=np.int64)
        return [frames[int(index)].copy() for index in local_indices]

    @staticmethod
    def _segments(timestamps: np.ndarray, seconds: float, prefix: str) -> list[Segment]:
        segments = segment_fixed_windows(
            timestamps=timestamps,
            window_seconds=seconds,
            stride_seconds=seconds,
            prefix=prefix,
        )
        if segments:
            return segments
        return [
            Segment(
                segment_id=f"{prefix}_0000",
                start_index=0,
                end_index=max(0, len(timestamps) - 1),
                start_time_sec=float(timestamps[0]) if len(timestamps) else 0.0,
                end_time_sec=float(timestamps[-1]) if len(timestamps) else 0.0,
                duration_sec=max(float(timestamps[-1] - timestamps[0]), 0.0) if len(timestamps) else 0.0,
            )
        ]
