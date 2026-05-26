from __future__ import annotations

import json
import threading
import time
from textwrap import dedent
from typing import Any

from hmvqa_app.config import AppConfig
from hmvqa_app.runtime.answering import build_answerer
from hmvqa_app.runtime.qwen_vl import AnswerConfig
from hmvqa_app.schemas import AnswerRequest, AnswerResponse
from hmvqa_app.services.retrieval import RetrievalResult


class AnswerService:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._lock = threading.Lock()
        self._answerers: dict[str, Any] = {}

    def answer(self, *, request: AnswerRequest, retrieval: RetrievalResult) -> AnswerResponse:
        started = time.perf_counter()
        prompt = self._build_prompt(request.question, request.mode, retrieval)
        frame_texts = [
            self._frame_context(item.rank, item.timestamp, item.score)
            for item in retrieval.evidence
        ]
        answerer = self._get_answerer(request)
        generation = answerer.generate_text_from_frames(
            frames=retrieval.frames,
            prompt=prompt,
            frame_texts=frame_texts,
            max_new_tokens=request.max_new_tokens,
        )
        timing = dict(retrieval.timing)
        timing["answer_sec"] = round(time.perf_counter() - started, 3)
        if generation.generation_sec is not None:
            timing["generation_sec"] = float(generation.generation_sec)
        return AnswerResponse(
            answer_text=generation.raw_text,
            predicted_letter=None,
            mode=request.mode,
            evidence=retrieval.evidence,
            timing=timing,
            debug=retrieval.debug,
            usage={
                "prompt_tokens": generation.prompt_tokens,
                "completion_tokens": generation.completion_tokens,
                "total_tokens": generation.total_tokens,
            },
        )

    def _get_answerer(self, request: AnswerRequest) -> Any:
        model_id = request.model_id.strip() or self.config.default_model_id
        if model_id not in self.config.allowed_model_ids:
            raise ValueError(f"Unsupported model_id: {model_id}")
        config = AnswerConfig(
            backend="api",
            model_id=model_id,
            max_new_tokens=int(request.max_new_tokens),
            image_max_size=self.config.image_max_size,
            enable_thinking=bool(request.enable_thinking),
            api_base_url=self.config.default_api_base_url,
            api_key_env_var=self.config.default_api_key_env_var,
            api_timeout_sec=180.0,
        )
        key = json.dumps(
            {
                "backend": config.backend,
                "model": config.model_id,
                "base": config.api_base_url,
                "env": config.api_key_env_var,
                "tokens": config.max_new_tokens,
                "thinking": config.enable_thinking,
            },
            sort_keys=True,
        )
        with self._lock:
            if key not in self._answerers:
                self._answerers[key] = build_answerer(config)
            return self._answerers[key]

    @staticmethod
    def _build_prompt(question: str, mode: str, retrieval: RetrievalResult) -> str:
        evidence_name = "uniformly sampled frames" if mode == "pure_vlm" else "retrieved visual evidence frames"
        evidence_lines = [
            AnswerService._evidence_line(item.rank, item.timestamp, item.score)
            for item in retrieval.evidence
        ]
        l3_lines = [
            f"- {item['segment_id']}: {item['start']:.2f}s-{item['end']:.2f}s, score {item['score']:.4f}"
            for item in retrieval.debug.get("selected_l3", [])
        ]
        coarse_regions = "\n".join(l3_lines) if l3_lines else "- none"
        coarse_section = "" if mode == "pure_vlm" else f"Retrieved coarse video regions:\n{coarse_regions}\n\n"
        evidence_section = "\n".join(evidence_lines) if evidence_lines else "- none"
        return dedent(
            f"""\
            You are HM-VQA answering a question about one uploaded video.
            Use only the {evidence_name}. If the evidence is insufficient, say that clearly.
            Do not invent details that are not visible in the evidence.
            Format your response in Markdown with exactly these sections:
            ### Answer
            One concise answer.
            ### Visual evidence
            - Cite the relevant frame timestamps and visible details.
            ### Why this evidence was chosen
            - Briefly connect the frames to the question.

            User question:
            {question}

            {coarse_section}Evidence frames:
            {evidence_section}
            """
        ).strip()

    @staticmethod
    def _frame_context(rank: int, timestamp: float, score: float | None) -> str:
        if score is None:
            return f"Evidence frame {rank} at {timestamp:.2f} seconds. Uniformly sampled frame."
        return f"Evidence frame {rank} at {timestamp:.2f} seconds. Retrieval score {score:.4f}."

    @staticmethod
    def _evidence_line(rank: int, timestamp: float, score: float | None) -> str:
        if score is None:
            return f"- Frame {rank}: timestamp {timestamp:.2f}s"
        return f"- Frame {rank}: timestamp {timestamp:.2f}s, retrieval score {score:.4f}"
