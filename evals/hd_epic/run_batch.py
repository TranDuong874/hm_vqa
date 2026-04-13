from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from answering import QwenVLMAnswerer

from pipeline.experiments.hd_epic_mcq_shortlist_joint import (
    DEFAULT_TASKS,
    DEFAULT_VIDEO_IDS,
    BudgetConfig,
    default_answer_config,
)
from pipeline.core.io import log_line, write_json
from evals.ablations import direct_layer2_retrieval, direct_open_clip
from evals.methods import ours, pure_vlm


METHODS = {
    "ours": ours.run_method,
    "pure_vlm": pure_vlm.run_method,
    "direct_layer2_retrieval": direct_layer2_retrieval.run_method,
    "direct_open_clip": direct_open_clip.run_method,
}


def _default_output_root() -> Path:
    return Path("results/pipeline/comparisons/batch_default")


def _mean_optional(values: list[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    if not present:
        return None
    return round(sum(present) / len(present), 6)


def _sum_optional(values: list[int | float | None]) -> int | None:
    present = [int(value) for value in values if value is not None]
    if not present:
        return None
    return sum(present)


def _aggregate_method_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    summaries = [run["summary"] for run in runs]
    selected_support_count = _sum_optional([summary.get("selected_evidence_support_count") for summary in summaries])
    selected_hit_count = _sum_optional([summary.get("selected_evidence_hit_count") for summary in summaries])
    selected_miss_count = _sum_optional([summary.get("selected_evidence_miss_count") for summary in summaries])
    tp_count = _sum_optional([summary.get("tp_count") for summary in summaries])
    fn_count = _sum_optional([summary.get("fn_count") for summary in summaries])
    fp_count = _sum_optional([summary.get("fp_count") for summary in summaries])
    tn_count = _sum_optional([summary.get("tn_count") for summary in summaries])
    return {
        "retrieval_quality": {
            "candidate_pool_hit_any": _mean_optional([summary.get("candidate_pool_hit_any") for summary in summaries]),
            "selected_evidence_hit1": _mean_optional([summary.get("selected_evidence_hit1") for summary in summaries]),
        },
        "answer_quality": {
            "mcq_accuracy": _mean_optional([summary.get("mcq_accuracy") for summary in summaries]),
        },
        "conversion_quality": {
            "answer_accuracy_given_selected_hit": _mean_optional(
                [summary.get("answer_accuracy_given_selected_hit") for summary in summaries]
            ),
            "answer_accuracy_given_selected_miss": _mean_optional(
                [summary.get("answer_accuracy_given_selected_miss") for summary in summaries]
            ),
            "answer_not_correct_when_selected_miss": _mean_optional(
                [summary.get("answer_not_correct_when_selected_miss") for summary in summaries]
            ),
            "selected_evidence_support_count": selected_support_count,
            "selected_evidence_hit_count": selected_hit_count,
            "selected_evidence_miss_count": selected_miss_count,
            "tp_count": tp_count,
            "fn_count": fn_count,
            "fp_count": fp_count,
            "tn_count": tn_count,
            "tp_rate": round(tp_count / selected_support_count, 6)
            if tp_count is not None and selected_support_count
            else None,
            "fn_rate": round(fn_count / selected_support_count, 6)
            if fn_count is not None and selected_support_count
            else None,
            "fp_rate": round(fp_count / selected_support_count, 6)
            if fp_count is not None and selected_support_count
            else None,
            "tn_rate": round(tn_count / selected_support_count, 6)
            if tn_count is not None and selected_support_count
            else None,
        },
        "budget": {
            "total_frames_seen_per_question": _mean_optional(
                [summary.get("total_frames_seen_per_question") for summary in summaries]
            ),
            "num_candidate_clips_seen": _mean_optional(
                [summary.get("num_candidate_clips_seen") for summary in summaries]
            ),
            "frames_per_clip": _mean_optional([summary.get("frames_per_clip") for summary in summaries]),
        },
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the matched HD-EPIC comparison batch.")
    parser.add_argument("--limit", type=int, default=10_000, help="Maximum number of examples per video.")
    parser.add_argument("--output-root", type=Path, default=None, help="Optional output root override.")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(METHODS.keys()),
        choices=list(METHODS.keys()),
        help="Methods to run.",
    )
    parser.add_argument(
        "--videos",
        nargs="+",
        default=DEFAULT_VIDEO_IDS,
        help="Video ids to evaluate.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    output_root = args.output_root or _default_output_root()
    output_root.mkdir(parents=True, exist_ok=True)
    log_path = output_root / "progress.log"
    budget_config = BudgetConfig()
    answer_config = default_answer_config()
    overall_started = time.perf_counter()
    completed: dict[str, list[dict[str, Any]]] = {method_name: [] for method_name in args.methods}

    answerer = QwenVLMAnswerer(answer_config)
    try:
        for method_name in args.methods:
            runner = METHODS[method_name]
            for video_id in args.videos:
                log_line(log_path, f"[start] method={method_name} video={video_id} limit={args.limit}")
                output_dir = output_root / method_name / video_id.lower()
                result = runner(
                    video_id=video_id,
                    tasks=list(DEFAULT_TASKS),
                    limit=int(args.limit),
                    output_dir=output_dir,
                    budget_config=budget_config,
                    answer_config=answer_config,
                    answerer=answerer,
                )
                completed[method_name].append(
                    {
                        "video_id": video_id,
                        "summary": result["summary"],
                        "elapsed_sec": result["run_state"]["elapsed_sec"],
                        "completed_examples": result["run_state"]["completed_examples"],
                    }
                )
                aggregate = {
                    "status": "running",
                    "elapsed_sec": round(time.perf_counter() - overall_started, 3),
                    "methods": {
                        name: {
                            "per_video": runs,
                            "aggregate": _aggregate_method_runs(runs),
                        }
                        for name, runs in completed.items()
                        if runs
                    },
                }
                write_json(output_root / "rolling_summary.json", aggregate)
                log_line(log_path, f"[done] method={method_name} video={video_id}")

        final_payload = {
            "status": "completed",
            "elapsed_sec": round(time.perf_counter() - overall_started, 3),
            "methods": {
                name: {
                    "per_video": runs,
                    "aggregate": _aggregate_method_runs(runs),
                }
                for name, runs in completed.items()
            },
        }
        write_json(output_root / "final_summary.json", final_payload)
        log_line(log_path, "[complete]")
    finally:
        answerer.unload()


if __name__ == "__main__":
    main()
