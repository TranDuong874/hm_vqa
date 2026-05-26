from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field


Mode = Literal["hmvqa", "pure_vlm"]


@dataclass(slots=True)
class ProgressState:
    status: str
    progress: int
    message: str
    video_name: str | None = None
    duration_sec: float | None = None
    sampled_frames: int | None = None
    cache_hit: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RetrieveRequest(BaseModel):
    question: str = Field(min_length=1)
    mode: Mode = "hmvqa"
    evidence_frames: int = Field(default=16, ge=1, le=64)


class AnswerRequest(RetrieveRequest):
    model_id: str = "qwen-vl-max-latest"
    max_new_tokens: int = Field(default=384, ge=1, le=4096)
    enable_thinking: bool = False


class EvidenceItem(BaseModel):
    frame_id: str
    url: str
    timestamp: float
    rank: int
    score: float | None = None
    source: str


class RetrieveResponse(BaseModel):
    mode: Mode
    evidence: list[EvidenceItem]
    timing: dict[str, float]
    debug: dict[str, Any] = Field(default_factory=dict)


class AnswerResponse(BaseModel):
    answer_text: str
    predicted_letter: str | None = None
    mode: Mode
    evidence: list[EvidenceItem]
    timing: dict[str, float]
    debug: dict[str, Any] = Field(default_factory=dict)
    usage: dict[str, int | None] = Field(default_factory=dict)
