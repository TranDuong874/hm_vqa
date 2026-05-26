from __future__ import annotations

from hmvqa_app.runtime.qwen_api import QwenAPIAnswerer
from hmvqa_app.runtime.qwen_vl import AnswerConfig, QwenVLMAnswerer


def build_answerer(config: AnswerConfig):
    if config.backend == "api":
        return QwenAPIAnswerer(config)
    return QwenVLMAnswerer(config)
