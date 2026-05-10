from __future__ import annotations

import argparse
import json
from pathlib import Path

from answering.qwen_vl import AnswerConfig
from evals.common.vlm_baseline_runner import BaselineRunConfig, run_pure_vlm_baseline
from evals.mlvu.dataset import DEFAULT_MANIFEST, DEFAULT_OUTPUT_ROOT, DEFAULT_VIDEO_ROOT, load_mlvu_examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pure uniform-frame VLM baseline on MLVU.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT.parent / "pure_vlm")
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--image-max-size", type=int, default=336)
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--backend", choices=["local", "api"], default="local")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--api-base-url", default=None)
    parser.add_argument("--api-key-env-var", default="ALIBABA_CLOUD_API")
    parser.add_argument("--api-requests-per-minute", type=int, default=60)
    parser.add_argument("--api-tokens-per-minute", type=int, default=100000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    args = parser.parse_args()

    examples = load_mlvu_examples(args.manifest, video_root=args.video_root, limit=args.limit)
    output_dir = args.output_root / f"{args.model_id.split('/')[-1]}_frames_{args.max_frames}f_{args.image_max_size}"
    summary = run_pure_vlm_baseline(
        examples=examples,
        video_root=args.video_root,
        output_root=output_dir,
        run_config=BaselineRunConfig(
            input_mode="frames",
            sample_fps=args.sample_fps,
            max_frames=args.max_frames,
            image_max_size=args.image_max_size,
            prompt_prefix="You are given frames sampled uniformly from a video. Use only the visible evidence.",
            include_subtitles=False,
            workers=args.workers,
        ),
        answer_config=AnswerConfig(
            model_id=args.model_id,
            backend=args.backend,
            max_new_tokens=args.max_new_tokens,
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
            image_max_size=args.image_max_size,
            enable_thinking=args.enable_thinking,
            api_base_url=args.api_base_url or AnswerConfig.api_base_url,
            api_key_env_var=args.api_key_env_var,
            api_requests_per_minute=args.api_requests_per_minute,
            api_tokens_per_minute=args.api_tokens_per_minute,
        ),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
