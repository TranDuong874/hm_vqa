from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from common import ROOT, ensure_local_video
from dataloader import VideoMMELoader
from hm_vqa_pipeline import HMVQAPipeline, configure_hf_env
from answering import AnswerConfig, QwenVLMAnswerer
from retrieval import PipelineConfig, export_frames


MANIFEST_PATH = ROOT / "evals" / "video-mme" / "video_mme_dev50.json"
VIDEO_ROOT = ROOT / "dataset" / "Video-MME"
OUTPUT_ROOT = ROOT / "results" / "video_mme" / "ours_dev50"
ALLOW_DOWNLOAD = False
TARGET_URLS: list[str] | None = [
    "fFjv93ACGo8",
    "N1cdUjctpG8",
    "dphq5X-rMew",
]

PIPELINE_CONFIG = PipelineConfig(max_evidence_frames=64)
ANSWER_CONFIG = AnswerConfig()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HM-VQA on Video-MME.")
    parser.add_argument("--manifest-path", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--video-root", type=Path, default=VIDEO_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--allow-download", action="store_true", default=ALLOW_DOWNLOAD)
    parser.add_argument("--target-urls", nargs="*", default=TARGET_URLS)
    parser.add_argument("--sample-fps", type=float, default=PIPELINE_CONFIG.sample_fps)
    parser.add_argument("--low-threshold", type=float, default=PIPELINE_CONFIG.low_threshold)
    parser.add_argument("--high-threshold", type=float, default=PIPELINE_CONFIG.high_threshold)
    parser.add_argument("--low-min-seconds", type=float, default=PIPELINE_CONFIG.low_min_seconds)
    parser.add_argument("--high-min-seconds", type=float, default=PIPELINE_CONFIG.high_min_seconds)
    parser.add_argument("--openclip-weight", type=float, default=PIPELINE_CONFIG.openclip_weight)
    parser.add_argument("--dino-weight", type=float, default=PIPELINE_CONFIG.dino_weight)
    parser.add_argument("--energy-weight", type=float, default=PIPELINE_CONFIG.energy_weight)
    parser.add_argument("--top-high", type=int, default=PIPELINE_CONFIG.top_high)
    parser.add_argument("--top-low", type=int, default=PIPELINE_CONFIG.top_low)
    parser.add_argument("--max-evidence-frames", type=int, default=PIPELINE_CONFIG.max_evidence_frames)
    parser.add_argument("--openclip-batch-size", type=int, default=PIPELINE_CONFIG.openclip_batch_size)
    parser.add_argument("--dino-batch-size", type=int, default=PIPELINE_CONFIG.dino_batch_size)
    parser.add_argument("--pipeline-device", default=PIPELINE_CONFIG.device)
    parser.add_argument("--model-id", default=ANSWER_CONFIG.model_id)
    parser.add_argument("--answer-device", default=ANSWER_CONFIG.device)
    parser.add_argument("--max-new-tokens", type=int, default=ANSWER_CONFIG.max_new_tokens)
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
    configure_hf_env(ROOT / ".env")
    args.output_root.mkdir(parents=True, exist_ok=True)

    loader = VideoMMELoader(args.manifest_path)
    videos = select_videos(loader, args.target_urls)
    pipeline_config = PipelineConfig(
        sample_fps=args.sample_fps,
        low_threshold=args.low_threshold,
        high_threshold=args.high_threshold,
        low_min_seconds=args.low_min_seconds,
        high_min_seconds=args.high_min_seconds,
        openclip_weight=args.openclip_weight,
        dino_weight=args.dino_weight,
        energy_weight=args.energy_weight,
        top_high=args.top_high,
        top_low=args.top_low,
        max_evidence_frames=args.max_evidence_frames,
        openclip_batch_size=args.openclip_batch_size,
        dino_batch_size=args.dino_batch_size,
        device=args.pipeline_device,
    )
    answer_config = AnswerConfig(
        model_id=args.model_id,
        device=args.answer_device,
        max_new_tokens=args.max_new_tokens,
    )
    pipeline = HMVQAPipeline(pipeline_config)
    answerer = QwenVLMAnswerer(answer_config)

    all_results: list[dict[str, object]] = []
    started = time.perf_counter()

    for video in videos:
        video_path = ensure_local_video(video_root=args.video_root, url_id=video.url, allow_download=args.allow_download)
        video_root = args.output_root / video.url
        video_root.mkdir(parents=True, exist_ok=True)

        index = pipeline.build_index(video_path)
        retrieval_packages = []
        for question in video.questions:
            package = pipeline.retrieve(
                index=index,
                question=question.question,
                options=question.options,
            )
            question_root = video_root / question.question_id
            question_root.mkdir(parents=True, exist_ok=True)
            export_frames(
                frames=package.evidence_frames,
                meta=package.evidence_meta,
                output_dir=question_root / "evidence_frames",
            )
            retrieval_packages.append((question, package, question_root))

        pipeline.release_encoders()

        for question, package, question_root in retrieval_packages:
            prediction = answerer.answer_frames(
                frames=package.evidence_frames,
                question=question.question,
                options=question.options,
                prompt_prefix="These are retrieved evidence frames from a longer video.",
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
                "high_hits": package.high_hits,
                "low_hits": package.low_hits,
                "evidence_frames": package.evidence_meta,
            }
            (question_root / "result.json").write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")
            all_results.append(row)
            print(
                f"question_id={question.question_id} frames={len(package.evidence_frames)} "
                f"pred={row['predicted_letter']} gold={row['gold_letter']} ok={row['choice_correct']}"
            )

        answerer.unload()

    summary = {
        "manifest_path": str(args.manifest_path),
        "video_root": str(args.video_root),
        "videos": len(videos),
        "questions": len(all_results),
        "choice_accuracy": sum(1 for row in all_results if row["choice_correct"]) / max(len(all_results), 1),
        "elapsed_sec": round(time.perf_counter() - started, 3),
        "pipeline_config": {
            "sample_fps": pipeline_config.sample_fps,
            "low_threshold": pipeline_config.low_threshold,
            "high_threshold": pipeline_config.high_threshold,
            "low_min_seconds": pipeline_config.low_min_seconds,
            "high_min_seconds": pipeline_config.high_min_seconds,
            "openclip_weight": pipeline_config.openclip_weight,
            "dino_weight": pipeline_config.dino_weight,
            "energy_weight": pipeline_config.energy_weight,
            "top_high": pipeline_config.top_high,
            "top_low": pipeline_config.top_low,
            "max_evidence_frames": pipeline_config.max_evidence_frames,
        },
        "answer_config": {
            "model_id": answer_config.model_id,
            "device": answer_config.device,
            "max_new_tokens": answer_config.max_new_tokens,
        },
        "results": all_results,
    }
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"summary_accuracy: {summary['choice_accuracy']:.3f}")
    print(f"saved: {args.output_root / 'summary.json'}")
