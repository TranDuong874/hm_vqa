from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class Segment:
    segment_id: str
    start_index: int
    end_index: int
    start_time_sec: float
    end_time_sec: float
    duration_sec: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SegmentHit:
    segment_id: str
    score: float
    start_index: int
    end_index: int
    start_time_sec: float
    end_time_sec: float
    video_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class FrameHit:
    frame_index: int
    time_sec: float
    score: float
    video_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EvalRow:
    example_id: str
    task_name: str
    video_id: str
    question: str
    query_text: str
    gold_spans: list[dict[str, Any]]
    layer3_hits: list[dict[str, Any]]
    layer2_hits: list[dict[str, Any]]
    layer1_hits: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    status: str = "ok"
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
