from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from common import ROOT, ensure_local_video
from dataloader import VideoMMELoader
from hm_vqa_pipeline import configure_hf_env
from answering import AnswerConfig, QwenVLMAnswerer
from retrieval import export_frames, load_video_frames, select_uniform_frames


VIDEO_ROOT = ROOT / "dataset" / "Video-MME"
OUTPUT_ROOT = ROOT / "results" / "video_mme" / "baseline_qwen3vl"
SAMPLE_FPS = 4.0
MAX_FRAMES = 16

ANSWER_CONFIG = AnswerConfig()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fixed-frame Qwen3-VL baseline on Video-MME.")
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--hf-dataset", default="lmms-lab/Video-MME")
    parser.add_argument("--hf-split", default="test")
    parser.add_argument("--video-root", type=Path, default=VIDEO_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--target-urls", nargs="*", default=None)
    parser.add_argument("--sample-fps", type=float, default=SAMPLE_FPS)
    parser.add_argument("--max-frames", type=int, default=MAX_FRAMES)
    parser.add_argument("--model-id", default=ANSWER_CONFIG.model_id)
    parser.add_argument("--answer-device", default=ANSWER_CONFIG.device)
    parser.add_argument("--max-new-tokens", type=int, default=ANSWER_CONFIG.max_new_tokens)
    parser.add_argument("--load-in-4bit", action="store_true", default=ANSWER_CONFIG.load_in_4bit)
    parser.add_argument("--load-in-8bit", action="store_true", default=ANSWER_CONFIG.load_in_8bit)
    return parser.parse_args()


def select_videos(loader: VideoMMELoader, target_urls: list[str] | None) -> list:
    videos = loader.load()
    if not target_urls:
        return videos
    picked = [video for video in videos if video.url in target_urls]
    picked.sort(key=lambda video: target_urls.index(video.url))
    return picked


if __name__ == "__main__":
    args = parse_args()
    if args.load_in_4bit and args.load_in_8bit:
        raise ValueError("Choose only one of --load-in-4bit or --load-in-8bit.")
    configure_hf_env(ROOT / ".env")
    args.output_root.mkdir(parents=True, exist_ok=True)

    loader = VideoMMELoader(
        args.manifest_path,
        hf_dataset=args.hf_dataset,
        hf_split=args.hf_split,
    )
    videos = select_videos(loader, args.target_urls)
    answer_config = AnswerConfig(
        model_id=args.model_id,
        device=args.answer_device,
        max_new_tokens=args.max_new_tokens,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
    )
    answerer = QwenVLMAnswerer(answer_config)

    all_results: list[dict[str, object]] = []
    started = time.perf_counter()

    try:
        for video in videos:
            video_path = ensure_local_video(video_root=args.video_root, url_id=video.url)
            sampled = load_video_frames(video_path, args.sample_fps)
            video_root = args.output_root / video.url
            video_root.mkdir(parents=True, exist_ok=True)

            for question in video.questions:
                frames, meta = select_uniform_frames(
                    pil_frames=sampled.pil_frames,
                    timestamps=sampled.timestamps,
                    max_frames=args.max_frames,
                )

                question_root = video_root / question.question_id
                question_root.mkdir(parents=True, exist_ok=True)
                export_frames(frames=frames, meta=meta, output_dir=question_root / "frames")

                prediction = answerer.answer_frames(
                    frames=frames,
                    question=question.question,
                    options=question.options,
                    prompt_prefix="These are uniformly sampled video frames from a longer video.",
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
                    "frames": meta,
                }
                (question_root / "result.json").write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")
                all_results.append(row)
                print(
                    f"question_id={question.question_id} frames={args.max_frames} "
                    f"pred={row['predicted_letter']} gold={row['gold_letter']} ok={row['choice_correct']}"
                )
    finally:
        answerer.unload()

    summary = {
        "manifest_path": str(args.manifest_path) if args.manifest_path is not None else None,
        "hf_dataset": args.hf_dataset,
        "hf_split": args.hf_split,
        "video_root": str(args.video_root),
        "videos": len(videos),
        "questions": len(all_results),
        "choice_accuracy": sum(1 for row in all_results if row["choice_correct"]) / max(len(all_results), 1),
        "elapsed_sec": round(time.perf_counter() - started, 3),
        "sample_fps": args.sample_fps,
        "max_frames": args.max_frames,
        "answer_config": {
            "model_id": answer_config.model_id,
            "device": answer_config.device,
            "max_new_tokens": answer_config.max_new_tokens,
            "load_in_4bit": answer_config.load_in_4bit,
            "load_in_8bit": answer_config.load_in_8bit,
        },
        "results": all_results,
    }
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"summary_accuracy: {summary['choice_accuracy']:.3f}")
    print(f"saved: {args.output_root / 'summary.json'}")
