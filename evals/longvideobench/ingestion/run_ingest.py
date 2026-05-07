from __future__ import annotations

import argparse
from pathlib import Path

from evals.common.openclip_cache_builder import build_openclip_feature_cache
from evals.longvideobench.dataset import load_benchmark_items
from evals.longvideobench.paths import LVB_FULL_MANIFEST, LVB_FULL_OPENCLIP_ROOT, LVB_FULL_VIDEO_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description="Build OpenCLIP frame stores for LongVideoBench videos.")
    parser.add_argument("--manifest", type=Path, default=LVB_FULL_MANIFEST)
    parser.add_argument("--video-root", type=Path, default=LVB_FULL_VIDEO_ROOT)
    parser.add_argument("--output-root", type=Path, default=LVB_FULL_OPENCLIP_ROOT)
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--shard-size", type=int, default=5000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-name", default="ViT-L-14")
    parser.add_argument("--pretrained", default="datacomp_xl_s13b_b90k")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    items = load_benchmark_items(args.manifest, video_root=args.video_root, limit=args.limit)
    video_map = {
        item.retrieval.video_id: item.retrieval.video_path
        for item in items
    }
    build_openclip_feature_cache(
        videos=sorted(video_map.items()),
        output_root=args.output_root,
        sample_fps=args.sample_fps,
        batch_size=args.batch_size,
        shard_size=args.shard_size,
        device=args.device,
        model_name=args.model_name,
        pretrained=args.pretrained,
        force=args.force,
    )


if __name__ == "__main__":
    main()

