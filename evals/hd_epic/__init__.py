"""Minimal HD-EPIC dataset adapters and prompt/frame helpers."""

from .temporal import (
    example_scope_for_video,
    gold_spans_for_video,
    load_temporal_examples_for_video,
    parse_choice_spans,
)

__all__ = [
    "example_scope_for_video",
    "gold_spans_for_video",
    "load_temporal_examples_for_video",
    "parse_choice_spans",
]
