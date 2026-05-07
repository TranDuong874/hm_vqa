from __future__ import annotations

import argparse
import re
from pathlib import Path

from answering.qwen_vl import AnswerConfig
from evals.common.vlm_baseline_runner import BaselineRunConfig, run_pure_vlm_baseline
from evals.hd_epic.dataset import DEFAULT_MANIFEST, DEFAULT_OUTPUT_ROOT, DEFAULT_VIDEO_ROOT, load_hd_epic_examples


TIME_TAG_PATTERN = re.compile(r"<TIME\s+([^>]+?)\s+video\s+\d+>", flags=re.IGNORECASE)
BARE_TIME_TAG_PATTERN = re.compile(r"<TIME\s+([^>]+?)>", flags=re.IGNORECASE)
VIDEO_ALIAS_PATTERN = re.compile(r"\bvideo\s+\d+\b", flags=re.IGNORECASE)

HD_EPIC_TIMESTAMP_PROMPT_PREFIX = (
    "You are given frames sampled uniformly from a long video. "
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
        example.question = _sanitize_prompt_text(example.question)
        example.options = [_sanitize_prompt_text(option) for option in example.options]
        prepared.append(example)
    return prepared


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pure VLM baseline on the HD-EPIC FGAL benchmark manifest.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT / "pure_vlm")
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=24)
    parser.add_argument("--image-max-size", type=int, default=224)
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--backend", choices=["local", "api"], default="local")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--api-key-env-var", default="ALIBABA_CLOUD_API")
    parser.add_argument("--api-requests-per-minute", type=int, default=60)
    parser.add_argument("--api-tokens-per-minute", type=int, default=100000)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    args = parser.parse_args()

    examples = _prepare_mcq_examples(
        load_hd_epic_examples(args.manifest, video_root=args.video_root, limit=args.limit)
    )
    output_dir = args.output_root / f"{args.model_id.split('/')[-1]}_frames_{args.max_frames}f_{args.image_max_size}"
    run_pure_vlm_baseline(
        examples=examples,
        video_root=args.video_root,
        output_root=output_dir,
        run_config=BaselineRunConfig(
            input_mode="frames",
            sample_fps=args.sample_fps,
            max_frames=args.max_frames,
            image_max_size=args.image_max_size,
            prompt_prefix=HD_EPIC_TIMESTAMP_PROMPT_PREFIX,
            include_subtitles=False,
        ),
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
