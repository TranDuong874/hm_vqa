from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


AnswerType = Literal["temporal_option", "mcq_text"]
EvalLevel = Literal["retrieval_only", "oracle_evidence_answering", "end_to_end"]
FeatureSource = Literal["precomputed_hd_epic", "live_compute"]
StatusKind = Literal["ok", "skipped", "error"]


@dataclass(slots=True)
class InputVideoRef:
    alias: str
    video_id: str
    start_time_sec: float | None
    end_time_sec: float | None


@dataclass(slots=True)
class TemporalSpan:
    video_alias: str
    start_time_sec: float
    end_time_sec: float


@dataclass(slots=True)
class EvalExample:
    example_id: str
    task_name: str
    question: str
    choices: list[Any]
    correct_idx: int
    input_videos: list[InputVideoRef]
    primary_video_id: str | None
    gold_spans: list[TemporalSpan]
    answer_type: AnswerType

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SourceResolution:
    feature_source: FeatureSource
    video_id: str
    source_path: str
    available: bool
    source_metadata: dict[str, Any] = field(default_factory=dict)
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PredictionPayload:
    predicted_choice_idx: int | None
    predicted_spans: list[dict[str, Any]]
    is_correct_choice: bool | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TracePayload:
    window_hits: list[dict[str, Any]]
    frame_hits: list[dict[str, Any]]
    evidence_frame_indices: list[int]
    evidence_timestamps_sec: list[float]
    notes: list[str]
    layer3_window_hits: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TimingPayload:
    load_sec: float
    index_sec: float
    retrieve_sec: float
    answer_sec: float
    total_sec: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class StatusPayload:
    kind: StatusKind
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ResultRow:
    run_id: str
    eval_level: EvalLevel
    method_name: str
    task_name: str
    example_id: str
    question: str
    choices: list[Any]
    correct_idx: int
    input_videos: list[dict[str, Any]]
    feature_source: FeatureSource
    video_context: dict[str, Any]
    prediction: dict[str, Any]
    trace: dict[str, Any]
    timing: dict[str, Any]
    status: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
