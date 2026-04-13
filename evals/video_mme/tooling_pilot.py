from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import torch

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
for path in (ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from answering import AnswerConfig, QwenVLMAnswerer, build_mcq_letter_prompt, parse_choice_letter
from evals.video_mme.common import ROOT, ensure_local_video
from evals.video_mme.dataloader import VideoMMELoader, VideoMMEQuestion, VideoMMEVideo
from hm_vqa_pipeline import HMVQAPipeline, configure_hf_env
from pipeline.tools import (
    format_seconds,
    parse_interval,
    route_query_tool,
    select_evidence_tool,
    verify_consistency_tool,
)
from retrieval import (
    FrameHit,
    PipelineConfig,
    export_frames,
    load_selected_video_frames,
    select_uniform_video_frames,
)


DEFAULT_MANIFEST = ROOT / "evals" / "video_mme" / "manifests" / "video_mme_tooling_pilot_3videos_no_subs.json"
DEFAULT_VIDEO_ROOT = Path("/home/tranduong/dev/dataset/Video-MME/videos_subset_50_50_50")
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "video_mme" / "tooling_pilot_3videos_4fps_no_subs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Video-MME 3-video tooling pilot.")
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--methods", nargs="+", default=["pure_vlm", "tooling"], choices=["pure_vlm", "tooling"])
    parser.add_argument("--sample-fps", type=float, default=4.0)
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--image-max-size", type=int, default=224)
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--answer-device", default="cuda")
    parser.add_argument("--retrieval-device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--top-windows", type=int, default=5)
    parser.add_argument("--max-sampled-frames", type=int, default=0)
    parser.add_argument("--save-frames", action="store_true")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    return parser.parse_args()


def _iter_rows(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _row_key(url: str, question_id: str) -> str:
    return f"{url}::{question_id}"


def _load_completed_question_ids(rows_path: Path) -> set[str]:
    completed: set[str] = set()
    for row in _iter_rows(rows_path):
        url = row.get("url")
        question_id = row.get("question_id")
        if url and question_id:
            completed.add(_row_key(str(url), str(question_id)))
    return completed


def _serialize_frame_hits(hits: list[FrameHit]) -> list[dict[str, Any]]:
    return [
        {
            "frame_index": int(hit.frame_index),
            "time_sec": float(hit.time_sec),
            "score": float(hit.score),
        }
        for hit in hits
    ]


def _serialize_window_hits(hits: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "segment_id": str(hit.segment_id),
            "score": float(hit.score),
            "start_index": int(hit.start_index),
            "end_index": int(hit.end_index),
            "start_time_sec": float(hit.start_time_sec),
            "end_time_sec": float(hit.end_time_sec),
        }
        for hit in hits
    ]


def _select_videos(loader: VideoMMELoader) -> list[VideoMMEVideo]:
    return loader.load()


def _build_group_summary(rows: list[dict[str, Any]], field: str) -> dict[str, dict[str, float | int]]:
    total: Counter[str] = Counter()
    correct: Counter[str] = Counter()
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = str(row[field])
        total[key] += 1
        if bool(row.get("choice_correct")):
            correct[key] += 1
    summary: dict[str, dict[str, float | int]] = {}
    for key in sorted(total):
        denom = total[key]
        summary[key] = {
            "questions": int(denom),
            "correct": int(correct[key]),
            "accuracy": float(correct[key] / denom) if denom else 0.0,
        }
    return summary


def _build_method_summary(rows: list[dict[str, Any]], *, method_name: str, elapsed_sec: float) -> dict[str, Any]:
    valid_rows = [row for row in rows if row.get("status") == "ok"]
    correct = sum(1 for row in valid_rows if bool(row.get("choice_correct")))
    return {
        "method": method_name,
        "questions": len(valid_rows),
        "choice_accuracy": float(correct / len(valid_rows)) if valid_rows else 0.0,
        "elapsed_sec": round(elapsed_sec, 3),
        "by_duration": _build_group_summary(valid_rows, "duration"),
        "by_task_type": _build_group_summary(valid_rows, "task_type"),
        "by_video": _build_group_summary(valid_rows, "url"),
    }


def _paired_vs_baseline(rows_by_method: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    baseline_rows = {
        _row_key(str(row["url"]), str(row["question_id"])): row
        for row in rows_by_method.get("pure_vlm", [])
        if row.get("status") == "ok"
    }
    tooling_rows = {
        _row_key(str(row["url"]), str(row["question_id"])): row
        for row in rows_by_method.get("tooling", [])
        if row.get("status") == "ok"
    }
    wins = losses = ties = 0
    for key in sorted(set(baseline_rows) & set(tooling_rows)):
        base_ok = bool(baseline_rows[key].get("choice_correct"))
        tool_ok = bool(tooling_rows[key].get("choice_correct"))
        if tool_ok and not base_ok:
            wins += 1
        elif base_ok and not tool_ok:
            losses += 1
        else:
            ties += 1
    return {
        "tooling_wins": wins,
        "tooling_losses": losses,
        "ties": ties,
        "paired_questions": wins + losses + ties,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_method_rolling_summary(
    *,
    rows_path: Path,
    output_dir: Path,
    method_name: str,
    started_at: float,
) -> None:
    rows = list(_iter_rows(rows_path))
    payload = _build_method_summary(rows, method_name=method_name, elapsed_sec=time.perf_counter() - started_at)
    _write_json(output_dir / "rolling_summary.json", payload)


def _frame_labels_from_hits(hits: list[dict[str, Any]]) -> list[str]:
    return [f"Frame {index + 1} timestamp: {format_seconds(float(hit['time_sec']))}" for index, hit in enumerate(hits)]


def _question_prefix() -> str:
    return "These are uniformly sampled frames from the input video. Do not use subtitles."


def _tooling_prefix(query_family: str) -> str:
    return (
        "These are retrieval-selected evidence frames from the input video. "
        "They were chosen by a scripted tooling pipeline. "
        f"Routed query family: {query_family}. "
        "Use only the visible evidence. Do not use subtitles."
    )


def _compute_effective_sample_fps(
    *,
    requested_fps: float,
    duration_sec: float,
    max_sampled_frames: int,
    min_frames: int,
) -> float:
    if duration_sec <= 0.0:
        return requested_fps
    capped_fps = requested_fps
    if max_sampled_frames > 0:
        capped_fps = min(capped_fps, max_sampled_frames / duration_sec)
    min_required_fps = max(float(min_frames) / duration_sec, 1e-3)
    return max(min(capped_fps, requested_fps), min_required_fps)


def _run_pure_question(
    *,
    question: VideoMMEQuestion,
    frames: list[Any],
    frame_hits: list[FrameHit],
    answerer: QwenVLMAnswerer,
) -> dict[str, Any]:
    prediction = answerer.answer_frames(
        frames=frames,
        question=question.question,
        options=question.options,
        prompt_prefix=_question_prefix(),
    )
    return {
        "video_id": question.video_id,
        "url": question.url,
        "duration": question.duration,
        "domain": question.domain,
        "sub_category": question.sub_category,
        "question_id": question.question_id,
        "task_type": question.task_type,
        "question": question.question,
        "options": question.options,
        "gold_letter": question.answer,
        "predicted_letter": prediction.predicted_letter,
        "choice_correct": prediction.predicted_letter == question.answer,
        "raw_answer": prediction.raw_text,
        "generation_sec": prediction.generation_sec,
        "frame_budget": len(frames),
        "total_frames_seen_per_question": float(len(frames)),
        "frames": _serialize_frame_hits(frame_hits),
        "status": "ok",
    }


def _run_tooling_question(
    *,
    question: VideoMMEQuestion,
    index: Any,
    pipeline: HMVQAPipeline,
    answerer: QwenVLMAnswerer,
    max_frames: int,
) -> dict[str, Any]:
    route = route_query_tool({"question": question.question, "choices": question.options})
    evidence = pipeline.retrieve(index=index, question=question.question, options=question.options)
    pipeline.release_encoder()

    selected_payload = select_evidence_tool(
        {
            "frames": _serialize_frame_hits(evidence.frame_hits),
            "limit": max_frames,
            "min_gap_sec": 1.0,
        }
    )
    selected_hits = list(selected_payload["selected_frames"])
    selected_frames, _, _ = load_selected_video_frames(
        index.sampled_video.video_path,
        sample_fps=pipeline.config.sample_fps,
        target_indices=[int(hit["frame_index"]) for hit in selected_hits],
        image_max_size=pipeline.config.image_max_size,
    )

    prompt = build_mcq_letter_prompt(
        question.question,
        question.options,
        prefix=_tooling_prefix(str(route["query_family"])),
    )
    generation = answerer.generate_text_from_frames(
        frames=selected_frames,
        prompt=prompt,
        frame_texts=_frame_labels_from_hits(selected_hits),
    )
    predicted_letter = parse_choice_letter(generation.raw_text, options_count=len(question.options))

    consistency = None
    choice_intervals = [value for value in route.get("choice_time_references", []) if value is not None]
    if choice_intervals and predicted_letter is not None:
        predicted_index = ord(predicted_letter) - ord("A")
        if 0 <= predicted_index < len(choice_intervals):
            answer_interval = parse_interval(choice_intervals[predicted_index])
            if answer_interval is not None and selected_hits:
                selected_interval = {
                    "start_time_sec": float(selected_hits[0]["time_sec"]),
                    "end_time_sec": float(selected_hits[-1]["time_sec"]),
                }
                consistency = verify_consistency_tool(
                    {
                        "selected_interval": selected_interval,
                        "answer_interval": answer_interval.to_dict(),
                        "tolerance_sec": 10.0,
                    }
                )

    return {
        "video_id": question.video_id,
        "url": question.url,
        "duration": question.duration,
        "domain": question.domain,
        "sub_category": question.sub_category,
        "question_id": question.question_id,
        "task_type": question.task_type,
        "question": question.question,
        "options": question.options,
        "gold_letter": question.answer,
        "predicted_letter": predicted_letter,
        "choice_correct": predicted_letter == question.answer,
        "raw_answer": generation.raw_text,
        "generation_sec": generation.generation_sec,
        "query_family": route.get("query_family"),
        "tool_route": route,
        "window_hits": _serialize_window_hits(evidence.window_hits),
        "frame_hits": _serialize_frame_hits(evidence.frame_hits),
        "selected_frames": selected_hits,
        "tool_consistency": consistency,
        "total_frames_seen_per_question": float(len(selected_hits)),
        "status": "ok",
    }


def _run_method(
    *,
    method_name: str,
    videos: list[VideoMMEVideo],
    args: argparse.Namespace,
    answerer: QwenVLMAnswerer,
    output_dir: Path,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "rows.jsonl"
    debug_root = output_dir / "debug_frames"
    started_at = time.perf_counter()
    mode = "a" if args.resume and rows_path.exists() and not args.force else "w"
    if args.force and rows_path.exists():
        rows_path.unlink()
    completed = _load_completed_question_ids(rows_path) if args.resume and not args.force else set()

    rows: list[dict[str, Any]] = list(_iter_rows(rows_path)) if mode == "a" else []

    for video_index, video in enumerate(videos, start=1):
        video_path = ensure_local_video(video_root=args.video_root, url_id=video.url)
        pending_questions = [
            question
            for question in video.questions
            if _row_key(video.url, question.question_id) not in completed
        ]
        if not pending_questions:
            continue

        duration_sec = None
        effective_sample_fps = float(args.sample_fps)

        if method_name == "pure_vlm":
            uniform_frames, uniform_frame_hits, sampling_meta = select_uniform_video_frames(
                video_path=video_path,
                sample_fps=effective_sample_fps,
                max_frames=args.max_frames,
                image_max_size=args.image_max_size,
            )
            duration_sec = float(sampling_meta["duration_sec"])
            effective_sample_fps = _compute_effective_sample_fps(
                requested_fps=args.sample_fps,
                duration_sec=duration_sec,
                max_sampled_frames=args.max_sampled_frames,
                min_frames=max(args.max_frames, args.top_windows),
            )
            if effective_sample_fps != float(args.sample_fps):
                uniform_frames, uniform_frame_hits, sampling_meta = select_uniform_video_frames(
                    video_path=video_path,
                    sample_fps=effective_sample_fps,
                    max_frames=args.max_frames,
                    image_max_size=args.image_max_size,
                )
                duration_sec = float(sampling_meta["duration_sec"])
            index = None
            pipeline = None
        else:
            sampled = None
            sampling_meta = None
            if args.max_sampled_frames > 0:
                _, _, sampling_meta = select_uniform_video_frames(
                    video_path=video_path,
                    sample_fps=args.sample_fps,
                    max_frames=max(args.max_frames, args.top_windows),
                    image_max_size=args.image_max_size,
                )
                duration_sec = float(sampling_meta["duration_sec"])
                effective_sample_fps = _compute_effective_sample_fps(
                    requested_fps=args.sample_fps,
                    duration_sec=duration_sec,
                    max_sampled_frames=args.max_sampled_frames,
                    min_frames=max(args.max_frames, args.top_windows),
                )
            pipeline = HMVQAPipeline(
                PipelineConfig(
                    sample_fps=effective_sample_fps,
                    top_windows=args.top_windows,
                    max_evidence_frames=args.max_frames,
                    image_max_size=args.image_max_size,
                    device=args.retrieval_device,
                )
            )
            index = pipeline.build_index(video_path)
            pipeline.release_encoder()
            if duration_sec is None:
                duration_sec = float(index.sampled_video.timestamps[-1]) if len(index.sampled_video.timestamps) else 0.0

        for question in pending_questions:
            if method_name == "pure_vlm":
                row = _run_pure_question(
                    question=question,
                    frames=uniform_frames,
                    frame_hits=uniform_frame_hits,
                    answerer=answerer,
                )
                row["effective_sample_fps"] = float(effective_sample_fps)
                row["video_duration_sec"] = float(duration_sec)
                row["sampled_frame_count"] = int(sampling_meta["sampled_count"])
                if args.save_frames:
                    export_frames(
                        frames=uniform_frames,
                        hits=uniform_frame_hits,
                        output_dir=debug_root / question.question_id,
                    )
            else:
                assert pipeline is not None
                assert index is not None
                row = _run_tooling_question(
                    question=question,
                    index=index,
                    pipeline=pipeline,
                    answerer=answerer,
                    max_frames=args.max_frames,
                )
                row["effective_sample_fps"] = float(effective_sample_fps)
                row["video_duration_sec"] = float(duration_sec)
                row["sampled_frame_count"] = int(len(index.sampled_video.timestamps))
                if args.save_frames and row["selected_frames"]:
                    selected_hits = [
                        FrameHit(
                            frame_index=int(hit["frame_index"]),
                            time_sec=float(hit["time_sec"]),
                            score=float(hit["score"]),
                        )
                        for hit in row["selected_frames"]
                    ]
                    selected_frames, _, _ = load_selected_video_frames(
                        video_path,
                        sample_fps=effective_sample_fps,
                        target_indices=[int(hit.frame_index) for hit in selected_hits],
                        image_max_size=args.image_max_size,
                    )
                    export_frames(
                        frames=selected_frames,
                        hits=selected_hits,
                        output_dir=debug_root / question.question_id,
                    )

            with rows_path.open("a", encoding="utf-8") as sink:
                sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.append(row)
            completed.add(_row_key(video.url, question.question_id))
            _write_method_rolling_summary(
                rows_path=rows_path,
                output_dir=output_dir,
                method_name=method_name,
                started_at=started_at,
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if method_name != "pure_vlm":
            del index
            del pipeline
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _write_json(
        output_dir / "final_summary.json",
        _build_method_summary(list(_iter_rows(rows_path)), method_name=method_name, elapsed_sec=time.perf_counter() - started_at),
    )
    return list(_iter_rows(rows_path))


def main() -> None:
    args = parse_args()
    if args.load_in_4bit and args.load_in_8bit:
        raise ValueError("Choose only one of --load-in-4bit or --load-in-8bit.")
    if not args.load_in_4bit and not args.load_in_8bit:
        args.load_in_4bit = True

    configure_hf_env(ROOT / ".env")
    args.output_root.mkdir(parents=True, exist_ok=True)

    loader = VideoMMELoader(args.manifest_path)
    videos = _select_videos(loader)
    answerer = QwenVLMAnswerer(
        AnswerConfig(
            model_id=args.model_id,
            device=args.answer_device,
            max_new_tokens=args.max_new_tokens,
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
            image_max_size=args.image_max_size,
        )
    )

    rows_by_method: dict[str, list[dict[str, Any]]] = {}
    started_at = time.perf_counter()
    try:
        for method_name in args.methods:
            rows_by_method[method_name] = _run_method(
                method_name=method_name,
                videos=videos,
                args=args,
                answerer=answerer,
                output_dir=args.output_root / method_name,
            )
    finally:
        answerer.unload()

    final_payload = {
        "manifest_path": str(args.manifest_path),
        "video_root": str(args.video_root),
        "output_root": str(args.output_root),
        "sample_fps": float(args.sample_fps),
        "max_frames": int(args.max_frames),
        "max_sampled_frames": int(args.max_sampled_frames),
        "image_max_size": int(args.image_max_size),
        "model_id": args.model_id,
        "elapsed_sec": round(time.perf_counter() - started_at, 3),
        "methods": {
            method_name: _build_method_summary(
                rows,
                method_name=method_name,
                elapsed_sec=time.perf_counter() - started_at,
            )
            for method_name, rows in rows_by_method.items()
        },
        "paired_vs_baseline": _paired_vs_baseline(rows_by_method),
        "per_duration": {
            method_name: _build_group_summary([row for row in rows if row.get("status") == "ok"], "duration")
            for method_name, rows in rows_by_method.items()
        },
        "per_task_type": {
            method_name: _build_group_summary([row for row in rows if row.get("status") == "ok"], "task_type")
            for method_name, rows in rows_by_method.items()
        },
        "per_video": {
            method_name: _build_group_summary([row for row in rows if row.get("status") == "ok"], "url")
            for method_name, rows in rows_by_method.items()
        },
    }
    _write_json(args.output_root / "final_summary.json", final_payload)


if __name__ == "__main__":
    main()
