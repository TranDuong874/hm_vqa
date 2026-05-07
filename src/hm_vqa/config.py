from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import TypeVar

from .policies import AnswerPolicy, MemoryPolicy, RetrievalPolicy

T = TypeVar("T")


def _filter_kwargs(cls: type[T], payload: dict) -> dict:
    valid = {field.name for field in fields(cls)}
    return {key: value for key, value in payload.items() if key in valid}


def load_json_config(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_memory_policy(path: str | Path) -> MemoryPolicy:
    payload = load_json_config(path)
    return MemoryPolicy(**_filter_kwargs(MemoryPolicy, payload))


def load_retrieval_policy(path: str | Path) -> RetrievalPolicy:
    payload = load_json_config(path)
    return RetrievalPolicy(**_filter_kwargs(RetrievalPolicy, payload))


def load_answer_policy(path: str | Path) -> AnswerPolicy:
    payload = load_json_config(path)
    return AnswerPolicy(**_filter_kwargs(AnswerPolicy, payload))

