from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import yt_dlp
from dotenv import load_dotenv

from analysis import (
    DINO_BATCH_SIZE,
    DINO_WEIGHT,
    ENERGY_WEIGHT,
    EXPORT_HIGH_SEGMENTS,
    EXPORT_LOW_SEGMENTS,
    HIGH_MAX_SECONDS,
    HIGH_MIN_SECONDS,
    HIGH_THRESHOLD,
    LOW_MAX_SECONDS,
    LOW_MIN_SECONDS,
    LOW_THRESHOLD,
    MAX_EXPORTED_SEGMENTS_PER_LEVEL,
    OPENCLIP_BATCH_SIZE,
    OPENCLIP_WEIGHT,
    SAMPLE_FPS,
    USE_DINO,
    USE_ENERGY,
    USE_OPENCLIP,
    export_segment_clips,
    frame_energy_diff,
    minmax_normalize,
    plot_signals,
    sample_video,
    segment_by_threshold,
    cosine_drift,
)
from ingestion import DINOEncoder, OpenCLIPEncoder


load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
    os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", hf_token)


MANIFEST_PATH = Path("thirdparty/Video-RAG-master/evals/videomme_json_file.json")
VIDEO_ROOT = Path("dataset/Video-MME")
ANALYSIS_ROOT = Path("results/analysis")

# Hardcode a small mixed set for inspection. Tweak this list directly.
TARGET_URLS = [
    "fFjv93ACGo8",  # short
    "N1cdUjctpG8",  # short
    "dphq5X-rMew",  # medium
]

DEVICE = "cuda"


def ensure_local_video(url_id: str) -> Path:
    video_path = VIDEO_ROOT / f"{url_id}.mp4"
    if video_path.exists():
        return video_path
    VIDEO_ROOT.mkdir(parents=True, exist_ok=True)
    youtube_url = f"https://www.youtube.com/watch?v={url_id}"
    download_opts = {
        "outtmpl": str(video_path),
        "format": "mp4/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best",
        "merge_output_format": "mp4",
        "quiet": False,
        "noplaylist": True,
    }
    with yt_dlp.YoutubeDL(download_opts) as ydl:
        ydl.download([youtube_url])
    return video_path


def build_score(
    *,
    energy_signal: np.ndarray | None,
    openclip_signal: np.ndarray | None,
    dino_signal: np.ndarray | None,
) -> np.ndarray:
    weighted_parts: list[np.ndarray] = []
    weight_total = 0.0
    if energy_signal is not None:
        weighted_parts.append(minmax_normalize(energy_signal) * ENERGY_WEIGHT)
        weight_total += ENERGY_WEIGHT
    if openclip_signal is not None:
        weighted_parts.append(minmax_normalize(openclip_signal) * OPENCLIP_WEIGHT)
        weight_total += OPENCLIP_WEIGHT
    if dino_signal is not None:
        weighted_parts.append(minmax_normalize(dino_signal) * DINO_WEIGHT)
        weight_total += DINO_WEIGHT
    if not weighted_parts or weight_total <= 0.0:
        raise RuntimeError("No enabled signals were computed.")
    return np.sum(np.stack(weighted_parts, axis=0), axis=0) / weight_total


if __name__ == "__main__":
    manifest = json.loads(MANIFEST_PATH.read_text())
    by_url = {row["url"]: row for row in manifest}

    ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)

    openclip = OpenCLIPEncoder(device=DEVICE) if USE_OPENCLIP else None
    dino = DINOEncoder(device=DEVICE) if USE_DINO else None

    batch_summary = []

    for url_id in TARGET_URLS:
        item = by_url[url_id]
        output_dir = ANALYSIS_ROOT / url_id
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        video_path = ensure_local_video(url_id)

        print(f"video_id: {item['video_id']}")
        print(f"url: {url_id}")
        print(f"duration: {item['duration']}")
        print(f"domain: {item['domain']}")
        print(f"sub_category: {item['sub_category']}")
        print(f"video_path: {video_path}")
        print(f"output_dir: {output_dir}")

        total_started = time.perf_counter()

        decode_started = time.perf_counter()
        pil_frames, bgr_frames, timestamps, native_fps = sample_video(video_path, SAMPLE_FPS)
        decode_elapsed = time.perf_counter() - decode_started
        print(f"sampled_frames: {len(pil_frames)}")
        print(f"decode_elapsed_sec: {decode_elapsed:.3f}")

        energy_signal = frame_energy_diff(bgr_frames) if USE_ENERGY else None

        openclip_signal = None
        openclip_elapsed = None
        if openclip is not None:
            started = time.perf_counter()
            openclip_embeddings = openclip.encode_images(pil_frames, batch_size=OPENCLIP_BATCH_SIZE)
            openclip_signal = cosine_drift(openclip_embeddings)
            openclip_elapsed = time.perf_counter() - started
            print(f"openclip_embeddings: {tuple(openclip_embeddings.shape)}")
            print(f"openclip_elapsed_sec: {openclip_elapsed:.3f}")

        dino_signal = None
        dino_elapsed = None
        if dino is not None:
            started = time.perf_counter()
            dino_embeddings = dino.encode_images(pil_frames, batch_size=DINO_BATCH_SIZE)
            dino_signal = cosine_drift(dino_embeddings)
            dino_elapsed = time.perf_counter() - started
            print(f"dino_embeddings: {tuple(dino_embeddings.shape)}")
            print(f"dino_elapsed_sec: {dino_elapsed:.3f}")

        combined_score = build_score(
            energy_signal=energy_signal,
            openclip_signal=openclip_signal,
            dino_signal=dino_signal,
        )

        low_segments = segment_by_threshold(
            timestamps=timestamps,
            score=combined_score,
            threshold=LOW_THRESHOLD,
            min_seconds=LOW_MIN_SECONDS,
            max_seconds=LOW_MAX_SECONDS,
            level="low",
        )
        high_segments = segment_by_threshold(
            timestamps=timestamps,
            score=combined_score,
            threshold=HIGH_THRESHOLD,
            min_seconds=HIGH_MIN_SECONDS,
            max_seconds=HIGH_MAX_SECONDS,
            level="high",
        )

        plot_signals(
            timestamps=timestamps,
            energy_signal=minmax_normalize(energy_signal) if energy_signal is not None else None,
            openclip_signal=minmax_normalize(openclip_signal) if openclip_signal is not None else None,
            dino_signal=minmax_normalize(dino_signal) if dino_signal is not None else None,
            combined_score=combined_score,
            low_segments=low_segments,
            high_segments=high_segments,
            output_path=output_dir / "signals.png",
        )

        if EXPORT_LOW_SEGMENTS:
            export_segment_clips(
                segments=low_segments,
                frames_bgr=bgr_frames,
                timestamps=timestamps,
                output_dir=output_dir / "clips_low",
                native_fps=native_fps,
                max_segments=MAX_EXPORTED_SEGMENTS_PER_LEVEL,
            )
        if EXPORT_HIGH_SEGMENTS:
            export_segment_clips(
                segments=high_segments,
                frames_bgr=bgr_frames,
                timestamps=timestamps,
                output_dir=output_dir / "clips_high",
                native_fps=native_fps,
                max_segments=MAX_EXPORTED_SEGMENTS_PER_LEVEL,
            )

        manifest_data = {
            "video_id": item["video_id"],
            "url": url_id,
            "duration": item["duration"],
            "domain": item["domain"],
            "sub_category": item["sub_category"],
            "video_path": str(video_path),
            "sample_fps": SAMPLE_FPS,
            "sampled_frames": len(pil_frames),
            "weights": {
                "openclip": OPENCLIP_WEIGHT if USE_OPENCLIP else 0.0,
                "dino": DINO_WEIGHT if USE_DINO else 0.0,
                "energy": ENERGY_WEIGHT if USE_ENERGY else 0.0,
            },
            "thresholds": {
                "low": LOW_THRESHOLD,
                "high": HIGH_THRESHOLD,
            },
            "timings_sec": {
                "decode": decode_elapsed,
                "openclip": openclip_elapsed,
                "dino": dino_elapsed,
                "total": time.perf_counter() - total_started,
            },
            "score_stats": {
                "mean": float(np.mean(combined_score)),
                "std": float(np.std(combined_score)),
                "max": float(np.max(combined_score)),
            },
            "segments": {
                "low": [asdict(segment) for segment in low_segments],
                "high": [asdict(segment) for segment in high_segments],
            },
        }
        (output_dir / "segments.json").write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")

        batch_summary.append(
            {
                "video_id": item["video_id"],
                "url": url_id,
                "duration": item["duration"],
                "sampled_frames": len(pil_frames),
                "low_segments": len(low_segments),
                "high_segments": len(high_segments),
                "output_dir": str(output_dir),
            }
        )

        print(f"low_segments: {len(low_segments)}")
        print(f"high_segments: {len(high_segments)}")
        print(f"saved: {output_dir}")
        print("")

    (ANALYSIS_ROOT / "batch_summary.json").write_text(json.dumps(batch_summary, indent=2), encoding="utf-8")
    print(f"batch_summary: {ANALYSIS_ROOT / 'batch_summary.json'}")
