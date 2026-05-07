from __future__ import annotations

import argparse
from pathlib import Path

from answering.qwen_vl import AnswerConfig
from evals.common.retrieval_ablation_runner import (
    add_retrieval_ablation_args,
    build_retrieval_output_name,
    build_retrieval_run_config,
    run_retrieval_ablation,
)
from evals.video_mme.dataset import (
    DEFAULT_DERIVED_CACHE_ROOT,
    DEFAULT_FEATURE_ROOT,
    DEFAULT_MANIFEST,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_VIDEO_ROOT,
    load_video_mme_examples,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run retrieval ablations on a Video-MME manifest.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--derived-cache-root", type=Path, default=DEFAULT_DERIVED_CACHE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    add_retrieval_ablation_args(parser)
    args = parser.parse_args()

    examples = load_video_mme_examples(args.manifest, video_root=args.video_root, limit=args.limit)
    run_config = build_retrieval_run_config(args)
    method_output_root = args.output_root / build_retrieval_output_name(model_id=args.model_id, run_config=run_config)
    run_retrieval_ablation(
        examples=examples,
        feature_root=args.feature_root,
        derived_cache_root=args.derived_cache_root,
        output_root=method_output_root,
        run_config=run_config,
        answer_config=AnswerConfig(
            model_id=args.model_id,
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
            image_max_size=args.image_max_size,
        ),
    )


if __name__ == "__main__":
    main()
