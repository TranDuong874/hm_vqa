from __future__ import annotations

from .qwen_vl import AnswerConfig, QwenVLMAnswerer


def build_answerer(config: AnswerConfig):
    return QwenVLMAnswerer(config)
