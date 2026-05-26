from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hmvqa_app.config import AppConfig
from hmvqa_app.runtime import Segment


class StorageService:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.config.cache_root.mkdir(parents=True, exist_ok=True)

    def cache_key_for_upload(self, temp_video_path: Path, *, original_name: str, sample_fps: float) -> str:
        digest = hashlib.sha1()
        with temp_video_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        payload = {
            "schema": self.config.schema_version,
            "sha1": digest.hexdigest(),
            "name": Path(original_name).name,
            "sample_fps": float(sample_fps),
            "image_max_size": int(self.config.image_max_size),
            "l2": float(self.config.l2_seconds),
            "l3": float(self.config.l3_seconds),
            "l2_encoder": "viclip" if self.config.use_viclip_l2 else "openclip_mean",
        }
        return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]

    def session_dir(self, session_id: str) -> Path:
        return self.config.cache_root / session_id

    def frame_dir(self, session_id: str) -> Path:
        return self.session_dir(session_id) / "frames"

    def frame_path(self, session_id: str, frame_id: str) -> Path:
        return self.frame_dir(session_id) / frame_id

    def source_path(self, session_id: str, suffix: str) -> Path:
        return self.session_dir(session_id) / f"source{suffix}"

    def metadata_path(self, session_id: str) -> Path:
        return self.session_dir(session_id) / "metadata.json"

    def chat_path(self, session_id: str) -> Path:
        return self.session_dir(session_id) / "chat_history.json"

    def is_ready(self, session_id: str) -> bool:
        metadata_path = self.metadata_path(session_id)
        if not metadata_path.exists():
            return False
        try:
            metadata = self.read_json(metadata_path)
        except Exception:
            return False
        return metadata.get("schema_version") == self.config.schema_version and (self.session_dir(session_id) / "frame.index").exists()

    def prepare_video(self, temp_video_path: Path, *, original_name: str, sample_fps: float) -> tuple[str, Path, bool]:
        session_id = self.cache_key_for_upload(temp_video_path, original_name=original_name, sample_fps=sample_fps)
        session_dir = self.session_dir(session_id)
        session_dir.mkdir(parents=True, exist_ok=True)
        suffix = self.safe_suffix(original_name)
        video_path = self.source_path(session_id, suffix)
        if not video_path.exists():
            shutil.copy2(temp_video_path, video_path)
        return session_id, video_path, self.is_ready(session_id)

    @staticmethod
    def safe_suffix(filename: str) -> str:
        suffix = Path(filename).suffix.lower()
        return suffix if suffix in {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"} else ".mp4"

    @staticmethod
    def write_json(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @staticmethod
    def read_json(path: Path) -> Any:
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def segments_to_json(segments: list[Segment]) -> list[dict[str, Any]]:
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

    @staticmethod
    def segments_from_json(rows: list[dict[str, Any]]) -> list[Segment]:
        return [Segment(**row) for row in rows]

    def load_artifacts(self, session_id: str) -> dict[str, Any]:
        session_dir = self.session_dir(session_id)
        if not self.metadata_path(session_id).exists():
            raise FileNotFoundError("Session artifacts are not ready.")
        return {
            "session_dir": session_dir,
            "metadata": self.read_json(self.metadata_path(session_id)),
            "timestamps": np.load(session_dir / "timestamps.npy"),
            "frame_embeddings": torch.load(session_dir / "frame_embeddings.pt", map_location="cpu"),
            "l2_segments": self.segments_from_json(self.read_json(session_dir / "l2_segments.json")),
            "l3_segments": self.segments_from_json(self.read_json(session_dir / "l3_segments.json")),
        }

    def read_chat_history(self, session_id: str) -> list[dict[str, Any]]:
        path = self.chat_path(session_id)
        if not path.exists():
            return []
        payload = self.read_json(path)
        return payload if isinstance(payload, list) else []

    def append_chat_messages(self, session_id: str, messages: list[dict[str, Any]]) -> None:
        history = self.read_chat_history(session_id)
        now = time.time()
        for message in messages:
            row = dict(message)
            row.setdefault("created_at", now)
            history.append(row)
        self.write_json(self.chat_path(session_id), history)

    def list_sessions(self) -> list[dict[str, Any]]:
        sessions: list[dict[str, Any]] = []
        if not self.config.cache_root.exists():
            return sessions
        for session_dir in self.config.cache_root.iterdir():
            if not session_dir.is_dir():
                continue
            session_id = session_dir.name
            metadata: dict[str, Any] = {}
            if self.metadata_path(session_id).exists():
                try:
                    metadata = self.read_json(self.metadata_path(session_id))
                except Exception:
                    metadata = {}
            source_candidates = sorted(session_dir.glob("source.*"))
            chat_history = self.read_chat_history(session_id)
            updated_at = max(
                [path.stat().st_mtime for path in [self.metadata_path(session_id), self.chat_path(session_id), *source_candidates] if path.exists()],
                default=session_dir.stat().st_mtime,
            )
            sessions.append(
                {
                    "session_id": session_id,
                    "video_name": metadata.get("video_name") or (source_candidates[0].name if source_candidates else session_id),
                    "status": "ready" if self.is_ready(session_id) else "processing",
                    "duration_sec": metadata.get("duration_sec"),
                    "sampled_frames": metadata.get("sampled_frames"),
                    "chat_count": len(chat_history),
                    "updated_at": updated_at,
                }
            )
        sessions.sort(key=lambda item: float(item.get("updated_at") or 0.0), reverse=True)
        return sessions
