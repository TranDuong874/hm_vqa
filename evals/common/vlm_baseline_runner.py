from __future__ import annotations

import json
import tarfile
import threading
import traceback
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from answering.factory import build_answerer
from answering.qwen_vl import AnswerConfig
from hm_vqa.schema import BenchmarkItem, QAExample, RetrievalExample, SubtitleSegment, TimeSpan
from retrieval import select_uniform_video_frames


@dataclass(slots=True)
class BaselineExample:
    example_id: str
    video_id: str
    video_path: str
    question: str
    options: list[str]
    correct_index: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_benchmark_item(
        self,
        *,
        dataset: str,
        split: str = "",
        video_root: str | Path | None = None,
        answer_type: str = "mcq",
    ) -> BenchmarkItem:
        duration = self.metadata.get("duration")
        duration_sec = float(duration) if duration not in (None, "") else None
        scope = None
        if self.metadata.get("scope_start_sec") is not None and self.metadata.get("scope_end_sec") is not None:
            scope = TimeSpan(
                start_sec=float(self.metadata["scope_start_sec"]),
                end_sec=float(self.metadata["scope_end_sec"]),
            )
        subtitles = self.metadata.get("subtitles")
        subtitle_segments = None
        if isinstance(subtitles, list):
            subtitle_segments = [
                SubtitleSegment(
                    start_sec=float(item.get("start", 0.0)),
                    end_sec=float(item.get("end", 0.0)),
                    text=str(item.get("text", "")),
                )
                for item in subtitles
                if isinstance(item, dict)
            ]
        video_path = Path(self.video_path)
        if video_root is not None and not video_path.is_absolute():
            video_path = Path(video_root) / video_path
        return BenchmarkItem(
            retrieval=RetrievalExample(
                example_id=self.example_id,
                dataset=dataset,
                split=split,
                video_id=self.video_id,
                video_path=video_path,
                query=self.question,
                duration_sec=duration_sec,
                time_scope=scope,
                metadata=dict(self.metadata),
            ),
            qa=QAExample(
                example_id=self.example_id,
                question=self.question,
                answer_type=answer_type,
                choices=list(self.options),
                answer_index=self.correct_index,
                subtitles=subtitle_segments,
                metadata=dict(self.metadata),
            ),
        )


@dataclass(slots=True)
class BaselineRunConfig:
    input_mode: str = "frames"
    sample_fps: float = 1.0
    max_frames: int = 8
    image_max_size: int | None = 336
    prompt_prefix: str = "You are given frames sampled uniformly from a video."
    output_root: str = "results/baselines"
    include_subtitles: bool = False
    workers: int = 1


def _parse_timecode(value: str) -> float:
    hours, minutes, seconds = value.split(":")
    return int(hours) * 3600.0 + int(minutes) * 60.0 + float(seconds)


def _normalize_subtitles(
    subtitles: list[dict[str, Any]],
    *,
    starting_timestamp_for_subtitles: float,
    duration: float | None,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for subtitle in subtitles:
        if "timestamp" in subtitle:
            start, end = subtitle["timestamp"]
            start = float(start) - float(starting_timestamp_for_subtitles)
            if isinstance(end, (int, float)):
                end = float(end) - float(starting_timestamp_for_subtitles)
            else:
                end = float(duration if duration is not None else start)
            text = str(subtitle.get("text", "")).strip()
        else:
            start = _parse_timecode(str(subtitle["start"])) - float(starting_timestamp_for_subtitles)
            end = _parse_timecode(str(subtitle["end"])) - float(starting_timestamp_for_subtitles)
            text = str(subtitle.get("line", "")).strip()
        if end - start < 1.0:
            midpoint = (start + end) / 2.0
            start = midpoint - 0.5
            end = midpoint + 0.5
        normalized.append({"start": start, "end": end, "text": text})
    return normalized


def _subtitle_texts_for_frames(
    *,
    frame_times: list[float],
    subtitles: list[dict[str, Any]],
    starting_timestamp_for_subtitles: float,
    duration: float | None,
) -> tuple[list[str], str]:
    normalized = _normalize_subtitles(
        subtitles,
        starting_timestamp_for_subtitles=starting_timestamp_for_subtitles,
        duration=duration,
    )
    per_frame: list[str] = []
    used_texts: list[str] = []
    for frame_time in frame_times:
        texts = [
            subtitle["text"]
            for subtitle in normalized
            if subtitle["text"] and float(subtitle["start"]) < frame_time < float(subtitle["end"])
        ]
        unique_texts: list[str] = []
        for text in texts:
            if text not in unique_texts:
                unique_texts.append(text)
                used_texts.append(text)
        per_frame.append("\n".join(unique_texts))
    subtitle_context = "\n".join(dict.fromkeys(text for text in used_texts if text))
    return per_frame, subtitle_context


def _load_subtitles(
    *,
    subtitle_path: str,
    subtitle_root: str | Path | None = None,
    subtitle_tar: str | Path | None = None,
) -> list[dict[str, Any]]:
    if subtitle_root is not None:
        candidate = Path(subtitle_root) / subtitle_path
        if candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8"))
    if subtitle_tar is not None:
        with tarfile.open(subtitle_tar) as archive:
            member_name = f"subtitles/{subtitle_path}"
            try:
                extracted = archive.extractfile(member_name)
            except KeyError:
                extracted = None
            if extracted is not None:
                return json.load(extracted)
    raise FileNotFoundError(f"Subtitle file not found: {subtitle_path}")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        return vars(value)
    return str(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=_json_default) + "\n")


def _rewrite_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")


def _log_line(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp}] {message}\n")


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    answered = [row for row in rows if row.get("predicted_letter") is not None]
    scored = [row for row in rows if row.get("choice_correct") is not None]
    correct = sum(1 for row in scored if row.get("choice_correct"))
    prompt_tokens = sum(int(row.get("prompt_tokens") or 0) for row in rows)
    completion_tokens = sum(int(row.get("completion_tokens") or 0) for row in rows)
    total_tokens = sum(int(row.get("total_tokens") or 0) for row in rows)
    generation_values = [float(value) for value in (row.get("generation_sec") for row in rows) if value is not None]
    return {
        "questions": total,
        "answered": len(answered),
        "scored": len(scored),
        "choice_accuracy": (correct / len(scored)) if scored else None,
        "avg_generation_sec": (
            round(sum(generation_values) / len(generation_values), 3) if generation_values else None
        ),
        "prompt_tokens": prompt_tokens or None,
        "completion_tokens": completion_tokens or None,
        "total_tokens": total_tokens or None,
    }


def _timestamp_label(time_sec: float) -> str:
    return f"Frame at {time_sec:.1f}s"


def _merge_frame_texts(*, frame_times: list[float], subtitle_texts: list[str] | None) -> list[str]:
    labels: list[str] = []
    for index, time_sec in enumerate(frame_times):
        parts = [_timestamp_label(time_sec)]
        if subtitle_texts is not None:
            subtitle_text = subtitle_texts[index].strip()
            if subtitle_text:
                parts.append(f"Subtitle near this frame: {subtitle_text}")
        labels.append("\n".join(parts))
    return labels


def _row_is_complete(row: dict[str, Any]) -> bool:
    return bool(row.get("example_id")) and row.get("raw_answer") not in (None, "")


def _load_resume_rows(path: Path) -> tuple[list[dict[str, Any]], int]:
    if not path.exists():
        return [], 0
    completed_rows: list[dict[str, Any]] = []
    dropped_rows = 0
    seen_example_ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                dropped_rows += 1
                continue
            example_id = str(row.get("example_id") or "")
            if not _row_is_complete(row):
                dropped_rows += 1
                continue
            if example_id in seen_example_ids:
                dropped_rows += 1
                continue
            completed_rows.append(row)
            seen_example_ids.add(example_id)
    return completed_rows, dropped_rows


def _is_api_content_filter_error(exc: Exception) -> bool:
    text = str(exc)
    return type(exc).__name__ == "BadRequestError" and "data_inspection_failed" in text


def run_pure_vlm_baseline(
    *,
    examples: Iterable[BaselineExample],
    video_root: str | Path,
    output_root: str | Path,
    run_config: BaselineRunConfig,
    answer_config: AnswerConfig,
    subtitle_root: str | Path | None = None,
    subtitle_tar: str | Path | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "rows.jsonl"
    progress_path = output_dir / "progress.log"
    error_path = output_dir / "error.log"
    rolling_summary_path = output_dir / "rolling_summary.json"

    example_list = list(examples)
    total_examples = len(example_list)
    rows, dropped_rows = _load_resume_rows(rows_path)
    completed_example_ids = {str(row["example_id"]) for row in rows}
    pending_examples = [example for example in example_list if example.example_id not in completed_example_ids]
    if rows_path.exists():
        _rewrite_jsonl(rows_path, rows)
    initial_summary = {
        "completed": len(rows),
        "total": total_examples,
        **_summarize_rows(rows),
    }
    _write_json(rolling_summary_path, initial_summary)
    answerer = build_answerer(answer_config)
    if int(run_config.workers) > 1 and answer_config.backend == "api":
        answerer.load()
    frame_sample_cache: OrderedDict[
        tuple[str, float, int, int | None],
        tuple[list[Any], list[Any], dict[str, float | int]],
    ] = OrderedDict()
    frame_sample_cache_size = 16
    subtitle_cache: dict[str, list[dict[str, Any]]] = {}
    cache_lock = threading.Lock()
    subtitle_lock = threading.Lock()
    write_lock = threading.Lock()

    def _get_uniform_frames(video_path: Path) -> tuple[list[Any], list[Any], dict[str, float | int]]:
        cache_key = (
            str(video_path),
            float(run_config.sample_fps),
            int(run_config.max_frames),
            run_config.image_max_size,
        )
        with cache_lock:
            cached = frame_sample_cache.get(cache_key)
            if cached is not None:
                frame_sample_cache.move_to_end(cache_key)
                frames, frame_hits, sampling = cached
                return list(frames), list(frame_hits), dict(sampling)

            frames, frame_hits, sampling = select_uniform_video_frames(
                video_path=video_path,
                sample_fps=run_config.sample_fps,
                max_frames=run_config.max_frames,
                image_max_size=run_config.image_max_size,
            )
            frame_sample_cache[cache_key] = (frames, frame_hits, sampling)
            while len(frame_sample_cache) > frame_sample_cache_size:
                frame_sample_cache.popitem(last=False)
            return list(frames), list(frame_hits), dict(sampling)

    def _get_subtitles(subtitle_path: str) -> list[dict[str, Any]]:
        with subtitle_lock:
            cached = subtitle_cache.get(subtitle_path)
            if cached is not None:
                return cached
            loaded = _load_subtitles(
                subtitle_path=subtitle_path,
                subtitle_root=subtitle_root,
                subtitle_tar=subtitle_tar,
            )
            subtitle_cache[subtitle_path] = loaded
            return loaded

    def _blocked_row(
        *,
        example: BaselineExample,
        exc: Exception,
        frame_hits: list[Any],
        sampling: dict[str, float | int] | None,
        subtitle_context: str,
    ) -> dict[str, Any]:
        return {
            "example_id": example.example_id,
            "video_id": example.video_id,
            "video_path": example.video_path,
            "question": example.question,
            "options": example.options,
            "correct_index": example.correct_index,
            "gold_letter": (
                chr(ord("A") + int(example.correct_index))
                if example.correct_index is not None and int(example.correct_index) >= 0
                else None
            ),
            "predicted_letter": None,
            "choice_correct": None,
            "raw_answer": f"API_BLOCKED: {type(exc).__name__}: {exc}",
            "generation_sec": None,
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "input_mode": run_config.input_mode,
            "effective_sample_fps": float(run_config.sample_fps),
            "sampled_frame_count": int(sampling["sampled_count"]) if sampling is not None else None,
            "subtitle_context": subtitle_context,
            "frames": [
                {"frame_index": int(hit.frame_index), "time_sec": float(hit.time_sec), "score": float(hit.score)}
                for hit in frame_hits
            ],
            "status": "api_blocked",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            **example.metadata,
        }

    def _run_example(example: BaselineExample, *, index: int) -> tuple[dict[str, Any], str]:
        video_path = Path(video_root) / example.video_path
        subtitle_texts: list[str] | None = None
        subtitle_context = ""
        frame_hits = []
        sampling = None
        frame_texts: list[str] | None = None
        try:
            if run_config.input_mode == "frames":
                frames, frame_hits, sampling = _get_uniform_frames(video_path)
                frame_times = [float(hit.time_sec) for hit in frame_hits]
            else:
                frames = []
                frame_times = []
            if run_config.include_subtitles:
                subtitle_path = example.metadata.get("subtitle_path")
                if subtitle_path:
                    subtitles = _get_subtitles(str(subtitle_path))
                    subtitle_texts, subtitle_context = _subtitle_texts_for_frames(
                        frame_times=frame_times,
                        subtitles=subtitles,
                        starting_timestamp_for_subtitles=float(
                            example.metadata.get("starting_timestamp_for_subtitles", 0.0)
                        ),
                        duration=(
                            float(example.metadata["duration"])
                            if example.metadata.get("duration") is not None
                            else None
                        ),
                    )
            if run_config.input_mode == "frames":
                frame_texts = _merge_frame_texts(frame_times=frame_times, subtitle_texts=subtitle_texts)
                prediction = answerer.answer_frames(
                    frames=frames,
                    question=example.question,
                    options=example.options,
                    prompt_prefix=run_config.prompt_prefix,
                    frame_texts=frame_texts,
                )
            else:
                extra_text = None
                if subtitle_context:
                    extra_text = f"Supplementary subtitles from the video:\n{subtitle_context}"
                prediction = answerer.answer_video(
                    video_path=video_path,
                    question=example.question,
                    options=example.options,
                    prompt_prefix=run_config.prompt_prefix,
                    sample_fps=run_config.sample_fps,
                    max_frames=run_config.max_frames,
                    extra_text=extra_text,
                )
        except Exception as exc:
            if _is_api_content_filter_error(exc):
                row = _blocked_row(
                    example=example,
                    exc=exc,
                    frame_hits=frame_hits,
                    sampling=sampling,
                    subtitle_context=subtitle_context,
                )
                return row, (
                    f"[item_blocked] index={index}/{total_examples} "
                    f"example_id={example.example_id} error={type(exc).__name__}: {exc}"
                )
            raise

        predicted_letter = prediction.predicted_letter
        gold_letter = (
            chr(ord("A") + int(example.correct_index))
            if example.correct_index is not None and int(example.correct_index) >= 0
            else None
        )
        row = {
            "example_id": example.example_id,
            "video_id": example.video_id,
            "video_path": example.video_path,
            "question": example.question,
            "options": example.options,
            "correct_index": example.correct_index,
            "gold_letter": gold_letter,
            "predicted_letter": predicted_letter,
            "choice_correct": (predicted_letter == gold_letter) if gold_letter is not None else None,
            "raw_answer": prediction.raw_text,
            "generation_sec": prediction.generation_sec,
            "prompt_tokens": prediction.prompt_tokens,
            "completion_tokens": prediction.completion_tokens,
            "total_tokens": prediction.total_tokens,
            "input_mode": run_config.input_mode,
            "effective_sample_fps": float(run_config.sample_fps),
            "sampled_frame_count": int(sampling["sampled_count"]) if sampling is not None else None,
            "subtitle_context": subtitle_context,
            "frames": [
                {"frame_index": int(hit.frame_index), "time_sec": float(hit.time_sec), "score": float(hit.score)}
                for hit in frame_hits
            ],
            **example.metadata,
        }
        return row, (
            f"[item_done] index={index}/{total_examples} example_id={example.example_id} "
            f"predicted={predicted_letter} correct={row['choice_correct']} "
            f"gen_sec={prediction.generation_sec} prompt_tokens={prediction.prompt_tokens} "
            f"completion_tokens={prediction.completion_tokens} total_tokens={prediction.total_tokens}"
        )

    def _record_row(row: dict[str, Any], message: str) -> None:
        with write_lock:
            rows.append(row)
            _append_jsonl(rows_path, row)
            rolling_summary = {
                "completed": len(rows),
                "total": total_examples,
                **_summarize_rows(rows),
            }
            _write_json(rolling_summary_path, rolling_summary)
            _log_line(progress_path, message)

    try:
        if rows:
            _log_line(
                progress_path,
                f"[resume] kept_completed={len(rows)} rerun_incomplete={dropped_rows} remaining={len(pending_examples)} total={total_examples}",
            )
        else:
            _log_line(
                progress_path,
                f"[start] total={total_examples} input_mode={run_config.input_mode} sample_fps={run_config.sample_fps} max_frames={run_config.max_frames} include_subtitles={run_config.include_subtitles} workers={run_config.workers}",
            )
        workers = max(int(run_config.workers), 1)
        use_parallel = workers > 1 and answer_config.backend == "api"
        if workers > 1 and not use_parallel:
            _log_line(progress_path, f"[workers_ignored] backend={answer_config.backend} workers={workers}")

        if use_parallel:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_item = {}
                for pending_index, example in enumerate(pending_examples, start=1):
                    index = len(rows) + pending_index
                    _log_line(
                        progress_path,
                        f"[item_start] index={index}/{total_examples} example_id={example.example_id} video={example.video_id}",
                    )
                    future = executor.submit(_run_example, example, index=index)
                    future_to_item[future] = (index, example)
                for future in as_completed(future_to_item):
                    index, example = future_to_item[future]
                    try:
                        row, message = future.result()
                    except Exception:
                        _log_line(
                            progress_path,
                            f"[item_error] index={index}/{total_examples} example_id={example.example_id}",
                        )
                        with error_path.open("a", encoding="utf-8") as handle:
                            handle.write(
                                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] index={index}/{total_examples} "
                                f"example_id={example.example_id}\n{traceback.format_exc()}\n"
                            )
                        raise
                    _record_row(row, message)
        else:
            for pending_index, example in enumerate(pending_examples, start=1):
                index = len(rows) + 1
                _log_line(
                    progress_path,
                    f"[item_start] index={index}/{total_examples} example_id={example.example_id} video={example.video_id}",
                )
                try:
                    row, message = _run_example(example, index=index)
                except Exception:
                    _log_line(
                        progress_path,
                        f"[item_error] index={index}/{total_examples} example_id={example.example_id}",
                    )
                    with error_path.open("a", encoding="utf-8") as handle:
                        handle.write(
                            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] index={index}/{total_examples} "
                            f"example_id={example.example_id}\n{traceback.format_exc()}\n"
                        )
                    raise
                _record_row(row, message)
    finally:
        answerer.unload()

    summary = {
        "run_config": asdict(run_config),
        "answer_config": asdict(answer_config),
        **_summarize_rows(rows),
    }
    _write_json(output_dir / "final_summary.json", summary)
    _log_line(progress_path, f"[done] scored={summary['scored']} accuracy={summary['choice_accuracy']}")
    return summary
