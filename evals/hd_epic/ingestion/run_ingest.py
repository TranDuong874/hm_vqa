from __future__ import annotations

import argparse
from pathlib import Path

from evals.common.openclip_cache_builder import build_openclip_feature_cache
from evals.hd_epic.dataset import DEFAULT_FEATURE_ROOT, DEFAULT_MANIFEST, DEFAULT_VIDEO_ROOT, load_hd_epic_examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Build 1 FPS OpenCLIP caches for the HD-EPIC ablation manifest.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--shard-size", type=int, default=5000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-name", default="ViT-L-14")
    parser.add_argument("--pretrained", default="datacomp_xl_s13b_b90k")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    examples = load_hd_epic_examples(args.manifest, video_root=args.video_root)
    video_map = {example.video_id: Path(example.video_path) for example in examples}
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
