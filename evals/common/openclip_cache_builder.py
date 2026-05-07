from __future__ import annotations

import json
from pathlib import Path

import cv2
import torch
from PIL import Image

from ingestion import OpenCLIPEncoder


def _flush_shard(
    *,
    output_dir: Path,
    shard_index: int,
    frame_indices: list[int],
    timestamps: list[float],
    embeddings: list[torch.Tensor],
) -> int:
    if not frame_indices:
        return shard_index
    payload = {
        "frame_idx": torch.tensor(frame_indices, dtype=torch.int64),
        "timestamp_sec": torch.tensor(timestamps, dtype=torch.float32),
        "openclip": torch.cat(embeddings, dim=0).to(torch.float32),
    }
    torch.save(payload, output_dir / f"shard_{shard_index:05d}.pt")
    return shard_index + 1


def _encode_video(
    *,
    video_path: Path,
    output_dir: Path,
    sample_fps: float,
    batch_size: int,
    shard_size: int,
    encoder: OpenCLIPEncoder,
) -> dict[str, object]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    native_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if native_fps <= 0.0 or total_frames <= 0:
        capture.release()
        raise RuntimeError(f"Invalid fps/frame count for video: {video_path}")

    step = max(int(round(native_fps / sample_fps)), 1)
    batch_images: list[Image.Image] = []
    batch_frame_indices: list[int] = []
    batch_timestamps: list[float] = []
    shard_frame_indices: list[int] = []
    shard_timestamps: list[float] = []
    shard_embeddings: list[torch.Tensor] = []
    shard_index = 0
    sampled_count = 0

    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % step == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            batch_images.append(Image.fromarray(frame_rgb))
            batch_frame_indices.append(frame_index)
            batch_timestamps.append(frame_index / native_fps)
            sampled_count += 1

            if len(batch_images) >= batch_size:
                embeddings = encoder.encode_images(batch_images, batch_size=batch_size)
                shard_frame_indices.extend(batch_frame_indices)
                shard_timestamps.extend(batch_timestamps)
                shard_embeddings.append(embeddings)
                batch_images = []
                batch_frame_indices = []
                batch_timestamps = []
                current = sum(t.shape[0] for t in shard_embeddings)
                if current >= shard_size:
                    shard_index = _flush_shard(
                        output_dir=output_dir,
                        shard_index=shard_index,
                        frame_indices=shard_frame_indices,
                        timestamps=shard_timestamps,
                        embeddings=shard_embeddings,
                    )
                    shard_frame_indices = []
                    shard_timestamps = []
                    shard_embeddings = []
        frame_index += 1
    capture.release()

    if batch_images:
        embeddings = encoder.encode_images(batch_images, batch_size=batch_size)
        shard_frame_indices.extend(batch_frame_indices)
        shard_timestamps.extend(batch_timestamps)
        shard_embeddings.append(embeddings)
    shard_index = _flush_shard(
        output_dir=output_dir,
        shard_index=shard_index,
        frame_indices=shard_frame_indices,
        timestamps=shard_timestamps,
        embeddings=shard_embeddings,
    )
    return {
        "native_fps": native_fps,
        "native_total_frames": total_frames,
        "sample_every_n_frames": step,
        "sampled_frames": sampled_count,
        "duration_sec": float(total_frames / native_fps),
        "num_shards": shard_index,
    }


def build_openclip_feature_cache(
    *,
    videos: list[tuple[str, Path]],
    output_root: Path,
    sample_fps: float = 1.0,
    batch_size: int = 64,
    shard_size: int = 5000,
    device: str = "cuda",
    model_name: str = "ViT-L-14",
    pretrained: str = "datacomp_xl_s13b_b90k",
    force: bool = False,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    encoder = OpenCLIPEncoder(model_name=model_name, pretrained=pretrained, device=device)
    try:
        for video_id, video_path in videos:
            output_dir = output_root / video_id
            meta_path = output_dir / "meta.json"
            if meta_path.exists() and sorted(output_dir.glob("shard_*.pt")) and not force:
                print(f"[skip] {video_id}")
                continue
            output_dir.mkdir(parents=True, exist_ok=True)
            for shard in output_dir.glob("shard_*.pt"):
                shard.unlink()
            stats = _encode_video(
                video_path=video_path,
                output_dir=output_dir,
                sample_fps=float(sample_fps),
                batch_size=int(batch_size),
                shard_size=int(shard_size),
                encoder=encoder,
            )
            meta = {
                "source_video": str(video_path.resolve()),
                "fps": float(sample_fps),
                "total_frames": int(stats["sampled_frames"]),
                "sample_every_n_frames": int(stats["sample_every_n_frames"]),
                "native_fps": float(stats["native_fps"]),
                "native_total_frames": int(stats["native_total_frames"]),
                "openclip_model": f"{model_name} / {pretrained}",
                "batch_size": int(batch_size),
                "shard_size": int(shard_size),
                "duration_sec": round(float(stats["duration_sec"]), 3),
            }
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            print(f"[done] {video_id} sampled={stats['sampled_frames']} duration={stats['duration_sec']:.1f}s")
    finally:
        del encoder
