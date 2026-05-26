from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image

from hmvqa_app.config import AppConfig
from hmvqa_app.runtime import OpenCLIPEncoder, Segment, read_ip_index, search_ip_index
from hmvqa_app.runtime.devices import release_model
from hmvqa_app.schemas import EvidenceItem, Mode
from hmvqa_app.services.storage import StorageService


@dataclass(slots=True)
class RetrievalResult:
    frames: list[Image.Image]
    evidence: list[EvidenceItem]
    timing: dict[str, float]
    debug: dict[str, Any]


class RetrievalService:
    def __init__(self, config: AppConfig, storage: StorageService) -> None:
        self.config = config
        self.storage = storage
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

    def retrieve(self, *, session_id: str, question: str, mode: Mode, evidence_frames: int) -> RetrievalResult:
        started = time.perf_counter()
        try:
            if mode == "pure_vlm":
                frames, evidence, debug = self._retrieve_uniform(session_id, evidence_frames)
            elif mode == "hmvqa":
                frames, evidence, debug = self._retrieve_hmvqa(session_id, question, evidence_frames)
            else:
                raise ValueError("mode must be 'hmvqa' or 'pure_vlm'")
            return RetrievalResult(
                frames=frames,
                evidence=evidence,
                timing={"retrieval_sec": round(time.perf_counter() - started, 3)},
                debug=debug,
            )
        finally:
            if self.config.unload_encoders_after_request:
                self._release_openclip()
                self._release_viclip()

    def _retrieve_uniform(self, session_id: str, max_frames: int) -> tuple[list[Image.Image], list[EvidenceItem], dict[str, Any]]:
        data = self.storage.load_artifacts(session_id)
        timestamps: np.ndarray = data["timestamps"]
        total = len(timestamps)
        count = min(max(int(max_frames), 1), total)

        if count <= 0:
            raise RuntimeError("No sampled frames available for pure VLM mode.")

        indices = list(dict.fromkeys(np.linspace(0, total - 1, num=count, dtype=np.int64).tolist()))
        selected = [(int(index), None) for index in indices]
        frames, evidence = self._materialize(session_id, timestamps, selected, source="pure_vlm")

        return frames, evidence, {"mode": "pure_vlm", "requested_frames": int(max_frames)}

    def _retrieve_hmvqa(self, session_id: str, question: str, max_frames: int) -> tuple[list[Image.Image], list[EvidenceItem], dict[str, Any]]:
        data = self.storage.load_artifacts(session_id)
        session_dir = self.storage.session_dir(session_id)
        timestamps: np.ndarray = data["timestamps"]
        metadata: dict[str, Any] = data["metadata"]
        frame_embeddings: torch.Tensor = data["frame_embeddings"]
        l2_segments: list[Segment] = data["l2_segments"]
        l3_segments: list[Segment] = data["l3_segments"]

        l2_encoder = metadata.get("l2_encoder") or "openclip_mean"
        openclip_query, l2_query = self._encode_queries(question, l2_encoder)

        l3_scores, l3_indices = search_ip_index(
            read_ip_index(session_dir / "l3.index"),
            openclip_query,
            top_k=min(5, len(l3_segments)),
        )
        l2_scores, l2_indices = search_ip_index(
            read_ip_index(session_dir / "l2.index"),
            l2_query,
            top_k=min(24, len(l2_segments)),
        )

        l2_by_parent: dict[int, float] = {int(idx): -1.0 for idx in l3_indices}
        for score, l2_idx in zip(l2_scores, l2_indices, strict=False):
            l2_segment = l2_segments[int(l2_idx)]
            for l3_idx in l3_indices:
                l3_segment = l3_segments[int(l3_idx)]
                if self._overlaps(l2_segment, l3_segment):
                    l2_by_parent[int(l3_idx)] = max(l2_by_parent[int(l3_idx)], float(score))

        reranked: list[tuple[float, int, float, float]] = []
        for l3_score, l3_idx in zip(l3_scores, l3_indices, strict=False):
            l2_bonus = max(l2_by_parent.get(int(l3_idx), -1.0), 0.0)
            reranked.append((
                float(l3_score) + 0.35 * l2_bonus,
                int(l3_idx),
                float(l3_score),
                float(l2_bonus),
            ))
        reranked.sort(reverse=True)
        selected_l3 = reranked[:3]

        allowed: set[int] = set()
        selected_l3_indices = {idx for _, idx, _, _ in selected_l3}
        for idx in selected_l3_indices:
            segment = l3_segments[idx]
            allowed.update(range(segment.start_index, segment.end_index + 1))

        for _, l2_idx in zip(l2_scores[:8], l2_indices[:8], strict=False):
            segment = l2_segments[int(l2_idx)]
            if any(self._overlaps(segment, l3_segments[idx]) for idx in selected_l3_indices):
                allowed.update(range(segment.start_index, segment.end_index + 1))

        if not allowed:
            _, global_indices = search_ip_index(
                read_ip_index(session_dir / "frame.index"),
                openclip_query,
                top_k=max_frames,
            )
            allowed = {int(index) for index in global_indices}

        allowed_indices = sorted(index for index in allowed if 0 <= index < frame_embeddings.shape[0])
        if not allowed_indices:
            raise RuntimeError("No evidence candidates retrieved.")

        candidate = frame_embeddings[allowed_indices]
        frame_scores = torch.matmul(candidate, openclip_query).detach().float().cpu().numpy()
        order = np.argsort(-frame_scores)[: min(max_frames, len(allowed_indices))]
        selected = sorted(
            [(allowed_indices[int(local_idx)], float(frame_scores[int(local_idx)])) for local_idx in order],
            key=lambda item: item[0],
        )
        frames, evidence = self._materialize(session_id, timestamps, selected, source="hmvqa")

        return frames, evidence, {
            "mode": "hmvqa",
            "l2_encoder": l2_encoder,
            "selected_l3": [
                {
                    "segment_id": l3_segments[idx].segment_id,
                    "start": l3_segments[idx].start_time_sec,
                    "end": l3_segments[idx].end_time_sec,
                    "score": round(score, 4),
                    "l3_score": round(l3_score, 4),
                    "l2_bonus": round(l2_bonus, 4),
                }
                for score, idx, l3_score, l2_bonus in selected_l3
            ],
        }

    def _encode_queries(self, question: str, l2_encoder: str) -> tuple[torch.Tensor, torch.Tensor]:
        openclip_query = self._get_encoder().encode_texts([question], batch_size=1)[0]

        if l2_encoder != "viclip":
            self._release_openclip()
            return openclip_query, openclip_query

        self._release_openclip()

        try:
            l2_query = self._get_viclip_encoder().encode_texts([question], batch_size=1)[0]
            return openclip_query, l2_query.float().cpu()
        finally:
            self._release_viclip()

    def _materialize(
        self,
        session_id: str,
        timestamps: np.ndarray,
        selected: list[tuple[int, float | None]],
        *,
        source: str,
    ) -> tuple[list[Image.Image], list[EvidenceItem]]:
        frames: list[Image.Image] = []
        evidence: list[EvidenceItem] = []
        for rank, (frame_index, score) in enumerate(selected, start=1):
            frame_id = f"frame_{int(frame_index):06d}.jpg"
            path = self.storage.frame_path(session_id, frame_id)
            frames.append(Image.open(path).convert("RGB"))
            evidence.append(
                EvidenceItem(
                    frame_id=frame_id,
                    url=f"/api/sessions/{session_id}/frames/{frame_id}",
                    timestamp=float(timestamps[int(frame_index)]),
                    rank=rank,
                    score=None if score is None else round(float(score), 4),
                    source=source,
                )
            )
        return frames, evidence

    @staticmethod
    def _overlaps(child: Segment, parent: Segment) -> bool:
        return child.start_index <= parent.end_index and child.end_index >= parent.start_index

    def _release_openclip(self) -> None:
        release_model(self, "_encoder")

    def _release_viclip(self) -> None:
        release_model(self, "_viclip_encoder")
