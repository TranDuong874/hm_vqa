from __future__ import annotations

import argparse
import re
from dataclasses import replace
from pathlib import Path

from answering.qwen_vl import AnswerConfig
from evals.common.retrieval_ablation_runner import (
    add_retrieval_ablation_args,
    build_retrieval_output_name,
    build_retrieval_run_config,
    run_retrieval_ablation,
)
from evals.hd_epic.dataset import (
    DEFAULT_DERIVED_CACHE_ROOT,
    DEFAULT_FEATURE_ROOT,
    DEFAULT_MANIFEST,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_VIDEO_ROOT,
    load_hd_epic_examples,
)

TIME_TAG_PATTERN = re.compile(r"<TIME\s+([^>]+?)\s+video\s+\d+>", flags=re.IGNORECASE)
BARE_TIME_TAG_PATTERN = re.compile(r"<TIME\s+([^>]+?)>", flags=re.IGNORECASE)
VIDEO_ALIAS_PATTERN = re.compile(r"\bvideo\s+\d+\b", flags=re.IGNORECASE)

HD_EPIC_TIMESTAMP_PROMPT_PREFIX = (
    "You are given retrieved evidence frames from a long video. "
    "Each frame label shows its timestamp, and the frames are in chronological order. "
    "Use the shown timestamps to align the visual evidence with the answer intervals. "
    "Reply with only one letter."
)


def _sanitize_prompt_text(text: str) -> str:
    text = TIME_TAG_PATTERN.sub(r"\1", text)
    text = BARE_TIME_TAG_PATTERN.sub(r"\1", text)
    text = VIDEO_ALIAS_PATTERN.sub("video", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def _prepare_mcq_examples(examples):
    prepared = []
    for example in examples:
        prepared.append(
            replace(
                example,
                question=_sanitize_prompt_text(example.question),
                options=[_sanitize_prompt_text(option) for option in example.options],
            )
        )
    return prepared


def main() -> None:
    parser = argparse.ArgumentParser(description="Run retrieval ablations on the HD-EPIC FGAL benchmark manifest.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--derived-cache-root", type=Path, default=DEFAULT_DERIVED_CACHE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    add_retrieval_ablation_args(parser)
    args = parser.parse_args()

    examples = _prepare_mcq_examples(
        load_hd_epic_examples(args.manifest, video_root=args.video_root, limit=args.limit)
    )
    run_config = build_retrieval_run_config(args)
    run_config.prompt_prefix = HD_EPIC_TIMESTAMP_PROMPT_PREFIX
    method_output_root = args.output_root / build_retrieval_output_name(model_id=args.model_id, run_config=run_config)
    run_retrieval_ablation(
        examples=examples,
        feature_root=args.feature_root,
        derived_cache_root=args.derived_cache_root,
        output_root=method_output_root,
        run_config=run_config,
        answer_config=AnswerConfig(
            model_id=args.model_id,
            backend=args.backend,
            max_new_tokens=args.max_new_tokens,
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
            image_max_size=args.image_max_size,
            enable_thinking=args.enable_thinking,
            api_key_env_var=args.api_key_env_var,
            api_requests_per_minute=args.api_requests_per_minute,
            api_tokens_per_minute=args.api_tokens_per_minute,
        ),
    )


if __name__ == "__main__":
    main()
