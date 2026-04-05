from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch

from common import ROOT, ensure_local_video
from dataloader import VideoMMELoader, VideoMMEQuestion, VideoMMEVideo
from segmentation import (
    cosine_drift,
    sample_video,
    sample_video_selected_indices,
    sample_video_with_energy,
    segment_by_threshold,
)
from answering import AnswerConfig, QwenVLMAnswerer
from hm_vqa_pipeline import HMVQAPipeline, configure_hf_env
from ingestion import DINOEncoder, OpenCLIPEncoder
from retrieval import (
    PipelineConfig,
    SampledVideo,
    VideoIndex,
    build_score,
    collect_candidate_low_segments,
    export_frames,
    mean_pool_segments,
    retrieve_top_segments,
    select_evidence_indices,
    select_evidence_frames,
)


VIDEO_ROOT = ROOT / "dataset" / "Video-MME"
OUTPUT_ROOT = ROOT / "results" / "video_mme" / "ours"
FEATURE_CACHE_ROOT = ROOT / "cache" / "video_mme_features"

PIPELINE_CONFIG = PipelineConfig(max_evidence_frames=16)
ANSWER_CONFIG = AnswerConfig()


@dataclass(slots=True)
class BatchVideoState:
    video: VideoMMEVideo
    video_path: Path
    sample_fps: float
    timestamps: object | None = None
    native_fps: float | None = None
    energy_signal: object | None = None
    openclip_embeddings: torch.Tensor | None = None
    dino_embeddings: torch.Tensor | None = None
    index: VideoIndex | None = None


@dataclass(slots=True)
class QuestionPackage:
    question: VideoMMEQuestion
    question_root: Path
    high_hits: list[dict[str, object]]
    low_hits: list[dict[str, object]]
    evidence_meta: list[dict[str, object]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HM-VQA retrieval pipeline on Video-MME in staged batches.")
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--hf-dataset", default="lmms-lab/Video-MME")
    parser.add_argument("--hf-split", default="test")
    parser.add_argument("--video-root", type=Path, default=VIDEO_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--target-urls", nargs="*", default=None)
    parser.add_argument("--video-batch-size", type=int, default=10)
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
    parser.add_argument("--feature-cache-dir", type=Path, default=FEATURE_CACHE_ROOT)
    parser.add_argument("--refresh-feature-cache", action="store_true", default=False)
    parser.add_argument("--model-id", default=ANSWER_CONFIG.model_id)
    parser.add_argument("--answer-device", default=ANSWER_CONFIG.device)
    parser.add_argument("--max-new-tokens", type=int, default=ANSWER_CONFIG.max_new_tokens)
    parser.add_argument("--load-in-4bit", action="store_true", default=ANSWER_CONFIG.load_in_4bit)
    parser.add_argument("--load-in-8bit", action="store_true", default=ANSWER_CONFIG.load_in_8bit)
    return parser.parse_args()


def select_videos(loader: VideoMMELoader, target_urls: list[str] | None) -> list[VideoMMEVideo]:
    videos = loader.load()
    if not target_urls:
        return videos
    picked = [video for video in videos if video.url in target_urls]
    picked.sort(key=lambda video: target_urls.index(video.url))
    return picked


def chunked(items: list[VideoMMEVideo], chunk_size: int) -> Iterable[list[VideoMMEVideo]]:
    for index in range(0, len(items), chunk_size):
        yield items[index : index + chunk_size]


def release_gpu_object(obj: object | None) -> None:
    if obj is not None:
        del obj
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_query_text(question: str, options: list[str]) -> str:
    return question + "\n" + "\n".join(options)


def video_feature_dir(root: Path, video_url: str) -> Path:
    return root / video_url


def openclip_cache_path(root: Path, video_url: str) -> Path:
    return video_feature_dir(root, video_url) / "openclip_features.pt"


def dino_cache_path(root: Path, video_url: str) -> Path:
    return video_feature_dir(root, video_url) / "dino_features.pt"


def feature_signature(*, sample_fps: float, encoder_name: str, encoder_id: str) -> str:
    payload = f"v=1|fps={sample_fps:.6f}|encoder={encoder_name}|id={encoder_id}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def load_openclip_cache(path: Path, *, signature: str) -> dict[str, object] | None:
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("signature") != signature:
        return None
    return payload


def load_dino_cache(path: Path, *, signature: str) -> dict[str, object] | None:
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("signature") != signature:
        return None
    return payload


def prepare_batch(
    *,
    videos: list[VideoMMEVideo],
    video_root: Path,
    sample_fps: float,
) -> list[BatchVideoState]:
    states: list[BatchVideoState] = []
    for video in videos:
        video_path = ensure_local_video(video_root=video_root, url_id=video.url)
        states.append(BatchVideoState(video=video, video_path=video_path, sample_fps=sample_fps))
    return states


def run_openclip_pass(
    states: list[BatchVideoState],
    config: PipelineConfig,
    *,
    feature_cache_dir: Path,
    refresh_feature_cache: bool,
) -> None:
    signature = feature_signature(
        sample_fps=config.sample_fps,
        encoder_name="openclip",
        encoder_id=f"{HMVQAPipeline.OPENCLIP_MODEL_NAME}|{HMVQAPipeline.OPENCLIP_PRETRAINED}",
    )
    missing_states: list[BatchVideoState] = []
    for state in states:
        cache_path = openclip_cache_path(feature_cache_dir, state.video.url)
        if not refresh_feature_cache:
            payload = load_openclip_cache(cache_path, signature=signature)
            if payload is not None:
                state.timestamps = payload["timestamps"]
                state.native_fps = float(payload["native_fps"])
                state.energy_signal = payload["energy_signal"]
                state.openclip_embeddings = payload["openclip_embeddings"].float().cpu()
                print(f"openclip_cached url={state.video.url} frames={len(state.timestamps)}")
                continue
        missing_states.append(state)

    if not missing_states:
        return

    encoder = OpenCLIPEncoder(device=config.device)
    try:
        for state in missing_states:
            pil_frames, timestamps, native_fps, energy_signal = sample_video_with_energy(
                state.video_path,
                config.sample_fps,
            )
            state.openclip_embeddings = encoder.encode_images(
                pil_frames,
                batch_size=config.openclip_batch_size,
            )
            state.timestamps = timestamps
            state.native_fps = float(native_fps)
            state.energy_signal = energy_signal
            cache_path = openclip_cache_path(feature_cache_dir, state.video.url)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "signature": signature,
                    "video_url": state.video.url,
                    "sample_fps": float(config.sample_fps),
                    "native_fps": float(native_fps),
                    "timestamps": timestamps,
                    "energy_signal": state.energy_signal,
                    "openclip_embeddings": state.openclip_embeddings.cpu(),
                },
                cache_path,
            )
            print(f"openclip_ingested url={state.video.url} frames={len(pil_frames)}")
    finally:
        release_gpu_object(encoder)


def run_dino_pass(
    states: list[BatchVideoState],
    config: PipelineConfig,
    *,
    feature_cache_dir: Path,
    refresh_feature_cache: bool,
) -> None:
    signature = feature_signature(
        sample_fps=config.sample_fps,
        encoder_name="dino",
        encoder_id=HMVQAPipeline.DINO_MODEL_ID,
    )
    missing_states: list[BatchVideoState] = []
    for state in states:
        cache_path = dino_cache_path(feature_cache_dir, state.video.url)
        if not refresh_feature_cache:
            payload = load_dino_cache(cache_path, signature=signature)
            if payload is not None:
                state.dino_embeddings = payload["dino_embeddings"].float().cpu()
                if state.timestamps is None:
                    state.timestamps = payload["timestamps"]
                if state.native_fps is None:
                    state.native_fps = float(payload["native_fps"])
                print(f"dino_cached url={state.video.url} frames={len(state.timestamps)}")
                continue
        missing_states.append(state)

    if not missing_states:
        return

    encoder = DINOEncoder(device=config.device)
    try:
        for state in missing_states:
            pil_frames, _, timestamps, native_fps = sample_video(
                state.video_path,
                config.sample_fps,
                include_bgr=False,
            )
            if state.timestamps is None:
                state.timestamps = timestamps
            if state.native_fps is None:
                state.native_fps = float(native_fps)
            if len(timestamps) != len(state.timestamps):
                raise RuntimeError(f"Timestamp length mismatch for {state.video.url}")
            state.dino_embeddings = encoder.encode_images(
                pil_frames,
                batch_size=config.dino_batch_size,
            )
            cache_path = dino_cache_path(feature_cache_dir, state.video.url)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "signature": signature,
                    "video_url": state.video.url,
                    "sample_fps": float(config.sample_fps),
                    "native_fps": float(native_fps),
                    "timestamps": timestamps,
                    "dino_embeddings": state.dino_embeddings.cpu(),
                },
                cache_path,
            )
            print(f"dino_ingested url={state.video.url} frames={len(pil_frames)}")
    finally:
        release_gpu_object(encoder)


def run_segmentation_pass(states: list[BatchVideoState], config: PipelineConfig) -> None:
    for state in states:
        if (
            state.openclip_embeddings is None
            or state.dino_embeddings is None
            or state.timestamps is None
            or state.native_fps is None
            or state.energy_signal is None
        ):
            raise RuntimeError(f"Missing embeddings for {state.video.url}")

        openclip_signal = cosine_drift(state.openclip_embeddings)
        dino_signal = cosine_drift(state.dino_embeddings)
        combined_score = build_score(
            energy_signal=state.energy_signal,
            openclip_signal=openclip_signal,
            dino_signal=dino_signal,
            config=config,
        )

        low_segments = segment_by_threshold(
            timestamps=state.timestamps,
            score=combined_score,
            threshold=config.low_threshold,
            min_seconds=config.low_min_seconds,
            max_seconds=None,
            level="low",
        )
        high_segments = segment_by_threshold(
            timestamps=state.timestamps,
            score=combined_score,
            threshold=config.high_threshold,
            min_seconds=config.high_min_seconds,
            max_seconds=None,
            level="high",
        )

        state.index = VideoIndex(
            sampled_video=SampledVideo(
                video_path=state.video_path,
                pil_frames=[],
                timestamps=state.timestamps,
                native_fps=state.native_fps,
            ),
            low_segments=low_segments,
            high_segments=high_segments,
            low_embeddings=mean_pool_segments(state.openclip_embeddings, low_segments),
            high_embeddings=mean_pool_segments(state.openclip_embeddings, high_segments),
            energy_signal=state.energy_signal,
            openclip_signal=openclip_signal,
            dino_signal=dino_signal,
            combined_score=combined_score,
        )
        print(
            f"segmented url={state.video.url} high={len(high_segments)} low={len(low_segments)} "
            f"frames={len(state.timestamps)}"
        )


def run_retrieval_pass(
    *,
    states: list[BatchVideoState],
    config: PipelineConfig,
    output_root: Path,
) -> list[QuestionPackage]:
    encoder = OpenCLIPEncoder(device=config.device)
    packages: list[QuestionPackage] = []
    try:
        for state in states:
            if state.index is None:
                raise RuntimeError(f"Missing segmented index for {state.video.url}")
            video_root = output_root / state.video.url
            video_root.mkdir(parents=True, exist_ok=True)

            for question in state.video.questions:
                query_text = build_query_text(question.question, question.options)
                query_embedding = encoder.encode_texts([query_text])[0]
                high_hits = retrieve_top_segments(
                    query_embedding=query_embedding,
                    segment_embeddings=state.index.high_embeddings,
                    segments=state.index.high_segments,
                    top_k=config.top_high,
                )
                low_hits = collect_candidate_low_segments(
                    high_hits=high_hits,
                    low_segments=state.index.low_segments,
                    low_embeddings=state.index.low_embeddings,
                    query_embedding=query_embedding,
                    top_k=config.top_low,
                )
                _, evidence_meta = select_evidence_indices(
                    timestamps=state.index.sampled_video.timestamps,
                    low_hits=low_hits,
                    max_frames=config.max_evidence_frames,
                )
                question_root = video_root / question.question_id
                question_root.mkdir(parents=True, exist_ok=True)
                packages.append(
                    QuestionPackage(
                        question=question,
                        question_root=question_root,
                        high_hits=high_hits,
                        low_hits=low_hits,
                        evidence_meta=evidence_meta,
                    )
                )
                print(
                    f"retrieved question_id={question.question_id} high={len(high_hits)} "
                    f"low={len(low_hits)} frames={len(evidence_meta)}"
                )
    finally:
        release_gpu_object(encoder)
    return packages


def run_answer_pass(
    *,
    states: list[BatchVideoState],
    question_packages: list[QuestionPackage],
    answer_config: AnswerConfig,
) -> list[dict[str, object]]:
    answerer = QwenVLMAnswerer(answer_config)
    results: list[dict[str, object]] = []
    try:
        packages_by_url: dict[str, list[QuestionPackage]] = {}
        for package in question_packages:
            packages_by_url.setdefault(package.question.url, []).append(package)

        for state in states:
            packages = packages_by_url.get(state.video.url, [])
            if not packages:
                continue
            needed_indices = sorted(
                {
                    int(item["frame_index"])
                    for package in packages
                    for item in package.evidence_meta
                }
            )
            selected_frames, selected_times, _ = sample_video_selected_indices(
                state.video_path,
                state.sample_fps,
                target_indices=needed_indices,
            )
            frame_lookup = {index: frame for index, frame in zip(needed_indices, selected_frames)}
            time_lookup = {index: time_sec for index, time_sec in zip(needed_indices, selected_times)}
            for package in packages:
                question = package.question
                frames = [frame_lookup[int(item["frame_index"])] for item in package.evidence_meta]
                for item in package.evidence_meta:
                    frame_index = int(item["frame_index"])
                    item["time_sec"] = float(time_lookup[frame_index])
                export_frames(
                    frames=frames,
                    meta=package.evidence_meta,
                    output_dir=package.question_root / "evidence_frames",
                )
                prediction = answerer.answer_frames(
                    frames=frames,
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
                (package.question_root / "result.json").write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")
                results.append(row)
                print(
                    f"question_id={question.question_id} frames={len(frames)} "
                    f"pred={row['predicted_letter']} gold={row['gold_letter']} ok={row['choice_correct']}"
                )
    finally:
        answerer.unload()
    return results


if __name__ == "__main__":
    args = parse_args()
    if args.load_in_4bit and args.load_in_8bit:
        raise ValueError("Choose only one of --load-in-4bit or --load-in-8bit.")
    configure_hf_env(ROOT / ".env")
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.feature_cache_dir.mkdir(parents=True, exist_ok=True)

    loader = VideoMMELoader(
        args.manifest_path,
        hf_dataset=args.hf_dataset,
        hf_split=args.hf_split,
    )
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
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
    )

    all_results: list[dict[str, object]] = []
    started = time.perf_counter()

    for batch_index, video_batch in enumerate(chunked(videos, args.video_batch_size), start=1):
        print(f"batch_start index={batch_index} videos={len(video_batch)}")
        batch_states = prepare_batch(
            videos=video_batch,
            video_root=args.video_root,
            sample_fps=args.sample_fps,
        )
        run_openclip_pass(
            batch_states,
            pipeline_config,
            feature_cache_dir=args.feature_cache_dir,
            refresh_feature_cache=args.refresh_feature_cache,
        )
        run_dino_pass(
            batch_states,
            pipeline_config,
            feature_cache_dir=args.feature_cache_dir,
            refresh_feature_cache=args.refresh_feature_cache,
        )
        run_segmentation_pass(batch_states, pipeline_config)
        question_packages = run_retrieval_pass(
            states=batch_states,
            config=pipeline_config,
            output_root=args.output_root,
        )
        batch_results = run_answer_pass(
            states=batch_states,
            question_packages=question_packages,
            answer_config=answer_config,
        )
        all_results.extend(batch_results)
        print(f"batch_done index={batch_index} questions={len(batch_results)}")

    summary = {
        "manifest_path": str(args.manifest_path) if args.manifest_path is not None else None,
        "hf_dataset": args.hf_dataset,
        "hf_split": args.hf_split,
        "video_root": str(args.video_root),
        "videos": len(videos),
        "questions": len(all_results),
        "choice_accuracy": sum(1 for row in all_results if row["choice_correct"]) / max(len(all_results), 1),
        "elapsed_sec": round(time.perf_counter() - started, 3),
        "video_batch_size": args.video_batch_size,
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
            "openclip_batch_size": pipeline_config.openclip_batch_size,
            "dino_batch_size": pipeline_config.dino_batch_size,
            "device": pipeline_config.device,
            "feature_cache_dir": str(args.feature_cache_dir),
            "refresh_feature_cache": bool(args.refresh_feature_cache),
        },
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
