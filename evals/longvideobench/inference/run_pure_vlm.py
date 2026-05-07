from __future__ import annotations

import argparse
import json
from pathlib import Path

from answering.qwen_vl import AnswerConfig
from evals.longvideobench.paths import LVB_FULL_MANIFEST, LVB_FULL_VIDEO_ROOT, SUBTITLE_ROOT, SUBTITLE_TAR

from evals.common.vlm_baseline_runner import BaselineExample, BaselineRunConfig, run_pure_vlm_baseline


DEFAULT_MANIFEST = LVB_FULL_MANIFEST
DEFAULT_VIDEO_ROOT = LVB_FULL_VIDEO_ROOT
DEFAULT_SUBTITLE_ROOT = SUBTITLE_ROOT
DEFAULT_SUBTITLE_TAR = SUBTITLE_TAR
DEFAULT_OUTPUT_ROOT = Path("/home/tranduong/dev/hm_vqa/results/longvideobench/pure_vlm")


def _load_examples(manifest_path: Path, *, limit: int | None = None) -> list[BaselineExample]:
    payload = json.loads(manifest_path.read_text())
    rows = payload["rows"]
    if limit is not None:
        rows = rows[:limit]
    examples: list[BaselineExample] = []
    for row in rows:
        examples.append(
            BaselineExample(
                example_id=str(row["id"]),
                video_id=str(row["video_id"]),
                video_path=str(row["video_path"]),
                question=str(row["question"]),
                options=[str(option) for option in row["candidates"]],
                correct_index=int(row["correct_choice"]) if "correct_choice" in row else None,
                metadata={
                    "split": payload.get("source_split"),
                    "question_category": row.get("question_category"),
                    "level": row.get("level"),
                    "duration_group": row.get("duration_group"),
                    "duration": row.get("duration"),
                    "topic_category": row.get("topic_category"),
                    "subtitle_path": row.get("subtitle_path"),
                    "starting_timestamp_for_subtitles": row.get("starting_timestamp_for_subtitles"),
                },
            )
        )
    return examples


def _default_video_root(manifest_path: Path) -> Path:
    return DEFAULT_VIDEO_ROOT


def _validate_video_root(video_root: Path, examples: list[BaselineExample]) -> None:
    if not video_root.exists():
        raise RuntimeError(
            f"Video root does not exist: {video_root}\n"
            "This runner expects a read-only source video directory that already contains the referenced .mp4 files."
        )
    missing = [example.video_path for example in examples if not (video_root / example.video_path).exists()]
    if missing:
        sample = ", ".join(sorted(set(missing))[:8])
        raise RuntimeError(
            f"{len(set(missing))} referenced videos are missing under {video_root}.\n"
            f"Sample missing files: {sample}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--video-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--input-mode", choices=["frames", "video"], default="video")
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, required=True)
    parser.add_argument("--image-max-size", type=int, default=336)
    parser.add_argument("--backend", choices=["local", "api"], default="local")
    parser.add_argument("--include-subtitles", action="store_true")
    parser.add_argument("--subtitle-root", type=Path, default=DEFAULT_SUBTITLE_ROOT)
    parser.add_argument("--subtitle-tar", type=Path, default=DEFAULT_SUBTITLE_TAR)
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--api-base-url", default=None)
    parser.add_argument("--api-key-env-var", default="ALIBABA_CLOUD_API")
    parser.add_argument("--api-requests-per-minute", type=int, default=60)
    parser.add_argument("--api-tokens-per-minute", type=int, default=100000)
    parser.add_argument("--api-timeout-sec", type=float, default=120.0)
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers for API backend. Local backend stays sequential.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    args = parser.parse_args()

    video_root = args.video_root or _default_video_root(args.manifest)
    examples = _load_examples(args.manifest, limit=args.limit)
    _validate_video_root(video_root, examples)
    summary = run_pure_vlm_baseline(
        examples=examples,
        video_root=video_root,
        output_root=args.output_root / f"{args.model_id.split('/')[-1]}_{args.input_mode}_{args.max_frames}f_{args.image_max_size}",
        run_config=BaselineRunConfig(
            input_mode=args.input_mode,
            sample_fps=args.sample_fps,
            max_frames=args.max_frames,
            image_max_size=args.image_max_size,
            prompt_prefix=(
                "You are given a video and must answer using only the visual evidence and any provided subtitles."
                if args.input_mode == "video"
                else "You are given frames sampled uniformly from a video. Use only the visible evidence."
            ),
            output_root=str(args.output_root),
            include_subtitles=args.include_subtitles,
            workers=args.workers,
        ),
        answer_config=AnswerConfig(
            model_id=args.model_id,
            backend=args.backend,
            image_max_size=args.image_max_size,
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
            api_base_url=args.api_base_url or AnswerConfig.api_base_url,
            api_key_env_var=args.api_key_env_var,
            api_requests_per_minute=args.api_requests_per_minute,
            api_tokens_per_minute=args.api_tokens_per_minute,
            api_timeout_sec=args.api_timeout_sec,
        ),
        subtitle_root=args.subtitle_root,
        subtitle_tar=args.subtitle_tar,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
