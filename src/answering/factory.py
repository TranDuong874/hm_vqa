from __future__ import annotations

from .qwen_vl import AnswerConfig, QwenVLMAnswerer
from .qwen_api import QwenAPIAnswerer


def build_answerer(config: AnswerConfig):
    if config.backend == "api":
        return QwenAPIAnswerer(config)
    return QwenVLMAnswerer(config)
