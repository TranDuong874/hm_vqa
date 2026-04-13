from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from answering import build_answerer

from pipeline.experiments.hd_epic_mcq_shortlist_joint import (
    BudgetConfig,
    DEFAULT_TASKS,
    default_answer_config,
    run_ours_method,
)


def _default_output_dir(video_id: str) -> Path:
    return Path("results/pipeline/hd_epic_mcq_shortlist_joint") / video_id.lower()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the canonical shortlist-joint HD-EPIC MCQ pipeline.")
    parser.add_argument("video_id", help="HD-EPIC video id to evaluate.")
    parser.add_argument("--limit", type=int, default=10_000, help="Maximum number of examples to evaluate.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional output directory override.")
    parser.add_argument("--model-id", type=str, default=None, help="Override the answer model id.")
    parser.add_argument("--enable-thinking", action="store_true", help="Enable model thinking mode when supported.")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    answer_config = default_answer_config()
    if args.model_id:
        answer_config.model_id = str(args.model_id)
    answer_config.enable_thinking = bool(args.enable_thinking)
    answerer = build_answerer(answer_config)
    try:
        run_ours_method(
            video_id=str(args.video_id),
            tasks=list(DEFAULT_TASKS),
            limit=int(args.limit),
            output_dir=args.output_dir or _default_output_dir(str(args.video_id)),
            budget_config=BudgetConfig(),
            answer_config=answer_config,
            answerer=answerer,
        )
    finally:
        answerer.unload()


if __name__ == "__main__":
    main()
