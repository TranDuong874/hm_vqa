from __future__ import annotations

import threading

from hmvqa_app.schemas import ProgressState


class SessionService:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._states: dict[str, ProgressState] = {}

    def create(self, session_id: str, video_name: str, *, cache_hit: bool = False) -> None:
        self.set(
            session_id,
            ProgressState(
                status="ready" if cache_hit else "uploaded",
                progress=100 if cache_hit else 1,
                message="Cache ready. Ask a question about the video." if cache_hit else "Video uploaded.",
                video_name=video_name,
                cache_hit=cache_hit,
            ),
        )

    def set(self, session_id: str, state: ProgressState) -> None:
        with self._lock:
            self._states[session_id] = state

    def patch(self, session_id: str, **updates: object) -> None:
        with self._lock:
            current = self._states.get(session_id)
            if current is None:
                current = ProgressState(status="unknown", progress=0, message="")
            data = current.to_dict()
            data.update(updates)
            self._states[session_id] = ProgressState(**data)

    def get(self, session_id: str) -> ProgressState | None:
        with self._lock:
            return self._states.get(session_id)

    def all(self) -> dict[str, ProgressState]:
        with self._lock:
            return dict(self._states)

    def remove(self, session_id: str) -> None:
        with self._lock:
            self._states.pop(session_id, None)
