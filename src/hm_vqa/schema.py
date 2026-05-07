from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class TimeSpan:
    start_sec: float
    end_sec: float

    @property
    def duration_sec(self) -> float:
        return max(0.0, float(self.end_sec) - float(self.start_sec))


@dataclass(frozen=True, slots=True)
class SubtitleSegment:
    start_sec: float
    end_sec: float
    text: str


@dataclass(slots=True)
class RetrievalExample:
    """Dataset-independent input for evidence retrieval."""

    example_id: str
    dataset: str
    split: str
    video_id: str
    video_path: Path
    query: str
    duration_sec: float | None = None
    time_scope: TimeSpan | None = None
    gold_spans: list[TimeSpan] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class QAExample:
    """Dataset-independent input for answer generation/evaluation."""

    example_id: str
    question: str
    answer_type: str = "mcq"
    choices: list[str] | None = None
    answer_index: int | None = None
    reference_answers: list[str] | None = None
    subtitles: list[SubtitleSegment] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BenchmarkItem:
    retrieval: RetrievalExample
    qa: QAExample | None = None


@dataclass(slots=True)
class EvidenceFrame:
    frame_index: int
    time_sec: float
    score: float = 0.0
    image_path: Path | None = None
    text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievalOutput:
    example_id: str
    evidence_frames: list[EvidenceFrame]
    retrieval_info: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AnswerOutput:
    example_id: str
    raw_answer: str
    predicted_index: int | None = None
    predicted_text: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

