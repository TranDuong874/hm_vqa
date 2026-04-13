from __future__ import annotations

import argparse
import gc
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from evals.video_mme.common import ROOT, ensure_local_video
from evals.video_mme.dataloader import VideoMMELoader
from hm_vqa_pipeline import configure_hf_env
from answering import AnswerConfig, QwenVLMAnswerer
from retrieval import export_frames, load_video_frames, select_uniform_frames
import torch


DEFAULT_MANIFEST = ROOT / "evals" / "video_mme" / "manifests" / "video_mme_stratified_50_50_50_no_subs.json"
DEFAULT_VIDEO_ROOT = Path("/home/tranduong/dev/dataset/Video-MME/videos_subset_50_50_50")
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "video_mme" / "pure_vlm_16f_224_no_subs_50_50_50"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the pure-VLM baseline on a stratified Video-MME subset.")
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--image-max-size", type=int, default=224)
    parser.add_argument("--limit-videos", type=int, default=None)
    parser.add_argument("--target-urls", nargs="*", default=None)
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--answer-device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--save-frames", action="store_true")
    parser.add_argument("--resume", action="store_true", default=True)
    return parser.parse_args()


def _select_videos(loader: VideoMMELoader, *, target_urls: list[str] | None, limit_videos: int | None):
    videos = loader.load()
    if target_urls:
        order = {url: idx for idx, url in enumerate(target_urls)}
        videos = [video for video in videos if video.url in order]
        videos.sort(key=lambda video: order[video.url])
    if limit_videos is not None:
        videos = videos[:limit_videos]
    return videos


def _iter_rows(path: Path) -> Iterable[dict[str, object]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _build_summary(rows_path: Path, *, args: argparse.Namespace, elapsed_sec: float) -> dict[str, object]:
    total_questions = 0
    total_correct = 0
    by_duration_total: Counter[str] = Counter()
    by_duration_correct: Counter[str] = Counter()
    by_domain_total: Counter[str] = Counter()
    by_domain_correct: Counter[str] = Counter()
    by_task_total: Counter[str] = Counter()
    by_task_correct: Counter[str] = Counter()
    by_video_total: Counter[str] = Counter()
    by_video_correct: Counter[str] = Counter()

    for row in _iter_rows(rows_path):
        total_questions += 1
        duration = str(row["duration"])
        domain = str(row["domain"])
        task_type = str(row["task_type"])
        url = str(row["url"])
        correct = bool(row["choice_correct"])
        if correct:
            total_correct += 1
        by_duration_total[duration] += 1
        by_domain_total[domain] += 1
        by_task_total[task_type] += 1
        by_video_total[url] += 1
        if correct:
            by_duration_correct[duration] += 1
            by_domain_correct[domain] += 1
            by_task_correct[task_type] += 1
            by_video_correct[url] += 1

    def build_acc_map(total: Counter[str], correct: Counter[str]) -> dict[str, dict[str, float | int]]:
        items: dict[str, dict[str, float | int]] = {}
        for key in sorted(total):
            denom = total[key]
            items[key] = {
                "questions": int(denom),
                "correct": int(correct[key]),
                "accuracy": float(correct[key] / denom) if denom else 0.0,
            }
        return items

    return {
        "manifest_path": str(args.manifest_path),
        "video_root": str(args.video_root),
        "output_root": str(args.output_root),
        "method": "pure_vlm",
        "uses_subtitles": False,
        "sample_fps": float(args.sample_fps),
        "max_frames": int(args.max_frames),
        "image_max_size": int(args.image_max_size),
        "model_id": args.model_id,
        "questions": total_questions,
        "choice_accuracy": float(total_correct / total_questions) if total_questions else 0.0,
        "elapsed_sec": round(elapsed_sec, 3),
        "by_duration": build_acc_map(by_duration_total, by_duration_correct),
        "by_domain": build_acc_map(by_domain_total, by_domain_correct),
        "by_task_type": build_acc_map(by_task_total, by_task_correct),
        "by_video": build_acc_map(by_video_total, by_video_correct),
    }


def _load_completed_question_ids(rows_path: Path) -> set[str]:
    completed: set[str] = set()
    for row in _iter_rows(rows_path):
        question_id = row.get("question_id")
        url = row.get("url")
        if question_id and url:
            completed.add(f"{url}::{question_id}")
    return completed


def _write_rolling_summary(rows_path: Path, output_root: Path, *, args: argparse.Namespace, started: float) -> None:
    summary = _build_summary(rows_path, args=args, elapsed_sec=time.perf_counter() - started)
    (output_root / "rolling_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.load_in_4bit and args.load_in_8bit:
        raise ValueError("Choose only one of --load-in-4bit or --load-in-8bit.")

    configure_hf_env(ROOT / ".env")
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_root / "rows.jsonl"
    final_summary_path = args.output_root / "final_summary.json"

    loader = VideoMMELoader(args.manifest_path)
    videos = _select_videos(loader, target_urls=args.target_urls, limit_videos=args.limit_videos)

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

    completed_question_ids = _load_completed_question_ids(rows_path) if args.resume else set()
    started = time.perf_counter()

    try:
        mode = "a" if args.resume and rows_path.exists() else "w"
        with rows_path.open(mode, encoding="utf-8") as sink:
            for video_idx, video in enumerate(videos, start=1):
                pending_questions = [
                    question
                    for question in video.questions
                    if f"{video.url}::{question.question_id}" not in completed_question_ids
                ]
                if not pending_questions:
                    print(f"[video {video_idx}/{len(videos)}] {video.url} questions=0 (already completed)")
                    continue
                video_path = ensure_local_video(video_root=args.video_root, url_id=video.url)
                sampled = load_video_frames(video_path, args.sample_fps)
                print(f"[video {video_idx}/{len(videos)}] {video.url} questions={len(pending_questions)}")

                for question in pending_questions:
                    frames, frame_hits = select_uniform_frames(
                        frames=sampled.frames,
                        timestamps=sampled.timestamps,
                        max_frames=args.max_frames,
                    )
                    prediction = answerer.answer_frames(
                        frames=frames,
                        question=question.question,
                        options=question.options,
                        prompt_prefix="These are uniformly sampled frames from the input video. Do not use subtitles.",
                    )

                    row = {
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
                        "raw_text": prediction.raw_text,
                        "generation_sec": prediction.generation_sec,
                        "frame_budget": args.max_frames,
                        "frames": [
                            {
                                "frame_index": int(hit.frame_index),
                                "time_sec": float(hit.time_sec),
                            }
                            for hit in frame_hits
                        ],
                    }
                    sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                    sink.flush()
                    completed_question_ids.add(f"{video.url}::{question.question_id}")

                    if args.save_frames:
                        question_dir = args.output_root / "debug_frames" / question.question_id
                        export_frames(frames=frames, hits=frame_hits, output_dir=question_dir)

                    print(
                        f"  question_id={question.question_id} pred={row['predicted_letter']} "
                        f"gold={row['gold_letter']} ok={int(bool(row['choice_correct']))}"
                    )
                    del frames
                    del frame_hits
                    del prediction
                    del row
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                del sampled
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                _write_rolling_summary(rows_path, args.output_root, args=args, started=started)
    finally:
        answerer.unload()

    summary = _build_summary(rows_path, args=args, elapsed_sec=time.perf_counter() - started)
    final_summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[done] accuracy={summary['choice_accuracy']:.4f} questions={summary['questions']}")
    print(f"[saved] {final_summary_path}")


if __name__ == "__main__":
    main()
