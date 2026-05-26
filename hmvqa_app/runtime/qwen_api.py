from __future__ import annotations

import base64
import io
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from openai import APIConnectionError, APIStatusError, APITimeoutError, InternalServerError, RateLimitError
from PIL import Image

from hmvqa_app.runtime.qwen_vl import AnswerConfig, GenerationResult, _prepare_frame


def _image_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


@dataclass(slots=True)
class _UsageSnapshot:
    timestamp: float
    total_tokens: int


class _APIRateLimiter:
    def __init__(self, *, rpm: int, tpm: int) -> None:
        self.rpm = max(int(rpm), 1)
        self.tpm = max(int(tpm), 1)
        self.request_times: deque[float] = deque()
        self.token_times: deque[_UsageSnapshot] = deque()
        self._lock = threading.Lock()

    def _prune(self, now: float) -> None:
        cutoff = now - 60.0
        while self.request_times and self.request_times[0] <= cutoff:
            self.request_times.popleft()
        while self.token_times and self.token_times[0].timestamp <= cutoff:
            self.token_times.popleft()

    def before_request(self, estimated_tokens: int) -> None:
        estimated_tokens = max(int(estimated_tokens), 1)
        while True:
            with self._lock:
                now = time.monotonic()
                self._prune(now)
                waits: list[float] = []
                if len(self.request_times) >= self.rpm:
                    waits.append(60.0 - (now - self.request_times[0]) + 0.01)
                token_total = sum(item.total_tokens for item in self.token_times)
                if token_total + estimated_tokens > self.tpm and self.token_times:
                    waits.append(60.0 - (now - self.token_times[0].timestamp) + 0.01)
                if not waits:
                    return
            time.sleep(max(waits))

    def after_request(self, total_tokens: int) -> None:
        with self._lock:
            now = time.monotonic()
            self._prune(now)
            self.request_times.append(now)
            self.token_times.append(_UsageSnapshot(timestamp=now, total_tokens=max(int(total_tokens), 1)))


class QwenAPIAnswerer:
    def __init__(self, config: AnswerConfig | None = None) -> None:
        self.config = config or AnswerConfig()
        self.client: OpenAI | None = None
        self._limiter = _APIRateLimiter(
            rpm=self.config.api_requests_per_minute,
            tpm=self.config.api_tokens_per_minute,
        )

    def load(self) -> None:
        if self.client is not None:
            return
        load_dotenv()
        api_key = os.getenv(self.config.api_key_env_var)
        if not api_key:
            raise RuntimeError(f"Missing API key in environment variable: {self.config.api_key_env_var}")
        self.client = OpenAI(
            api_key=api_key,
            base_url=self.config.api_base_url,
            timeout=self.config.api_timeout_sec,
        )

    def generate_text_from_frames(
        self,
        *,
        frames: list[Image.Image],
        prompt: str,
        frame_texts: list[str] | None = None,
        max_new_tokens: int | None = None,
    ) -> GenerationResult:
        self.load()
        assert self.client is not None

        prepared_frames = [_prepare_frame(frame, image_max_size=self.config.image_max_size) for frame in frames]
        if frame_texts is not None and len(frame_texts) != len(prepared_frames):
            raise ValueError("frame_texts length must match frames length")

        content: list[dict[str, Any]] = []
        for idx, frame in enumerate(prepared_frames):
            if frame_texts is not None:
                content.append({"type": "text", "text": frame_texts[idx]})
            content.append({"type": "image_url", "image_url": {"url": _image_to_data_url(frame)}})
        content.append({"type": "text", "text": prompt})

        estimated_tokens = max(
            len(prompt) // 3 + sum(len(text or "") // 3 for text in (frame_texts or [])) + len(prepared_frames) * 1200,
            1,
        )
        self._limiter.before_request(estimated_tokens)
        attempts = max(int(self.config.api_retry_attempts), 1)
        started = time.perf_counter()
        completion = None
        for attempt in range(attempts):
            try:
                completion = self.client.chat.completions.create(
                    model=self.config.model_id,
                    messages=[{"role": "user", "content": content}],
                    max_tokens=(max_new_tokens or self.config.max_new_tokens),
                    temperature=0,
                    extra_body={"enable_thinking": bool(self.config.enable_thinking)},
                )
                break
            except (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError) as exc:
                if attempt + 1 >= attempts:
                    raise
                delay = self.config.api_retry_base_delay_sec * (2 ** attempt)
                if isinstance(exc, RateLimitError):
                    delay = max(delay, 5.0)
                time.sleep(delay)
            except APIStatusError as exc:
                status = int(getattr(exc, "status_code", 0) or 0)
                retryable = status in {408, 409, 429, 500, 502, 503, 504}
                if (not retryable) or attempt + 1 >= attempts:
                    raise
                time.sleep(self.config.api_retry_base_delay_sec * (2 ** attempt))
        assert completion is not None
        elapsed = time.perf_counter() - started
        usage = getattr(completion, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
        self._limiter.after_request(total_tokens if total_tokens > 0 else estimated_tokens)
        raw_text = (completion.choices[0].message.content or "").strip()
        return GenerationResult(
            raw_text=raw_text,
            generation_sec=round(elapsed, 3),
            prompt_tokens=prompt_tokens or None,
            completion_tokens=completion_tokens or None,
            total_tokens=total_tokens or None,
        )

    def answer_video(self, *, video_path: str | Path, **_: Any) -> GenerationResult:
        raise NotImplementedError("API backend supports retrieved frame-list input only.")
