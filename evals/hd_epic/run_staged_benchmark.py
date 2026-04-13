from __future__ import annotations

import argparse
import json
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

from answering import build_answerer

from pipeline.core.io import log_line, write_json
from pipeline.experiments.hd_epic_mcq_shortlist_joint import (
    DEFAULT_TASKS,
    BudgetConfig,
    default_answer_config,
)
from evals.ablations import direct_layer2_retrieval, direct_open_clip
from evals.methods import ours, pure_vlm


DEFAULT_VIDEOS = [
    "P01-20240203-184045",
    "P01-20240203-152323",
    "P01-20240203-152956",
    "P01-20240203-161757",
    "P01-20240204-152537",
    "P01-20240202-171220",
    "P01-20240202-161948",
    "P01-20240203-132119",
    "P01-20240203-123350",
    "P01-20240203-135502",
]

METHOD_ORDER = [
    ("pure_vlm", pure_vlm.run_method),
    ("ours", ours.run_method),
    ("direct_open_clip", direct_open_clip.run_method),
    ("direct_layer2_retrieval", direct_layer2_retrieval.run_method),
]


def _default_output_root() -> Path:
    return REPO_ROOT / "results" / "pipeline" / "comparisons" / "benchmark_10videos_staged_v1"


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


def _write_batch_state(
    *,
    output_root: Path,
    completed: dict[str, list[dict[str, Any]]],
    started_at: float,
    status: str,
) -> None:
    payload = {
        "status": status,
        "elapsed_sec": round(time.perf_counter() - started_at, 3),
        "methods": {
            method_name: {
                "per_video": runs,
                "aggregate": _aggregate_method_runs(runs),
            }
            for method_name, runs in completed.items()
            if runs
        },
    }
    write_json(output_root / "rolling_summary.json", payload)
    if status in {"completed", "failed"}:
        write_json(output_root / "final_summary.json", payload)


def _load_existing_summary(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    if payload.get("status") != "completed":
        return None
    return payload


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run staged HD-EPIC benchmark one method at a time.")
    parser.add_argument("--limit", type=int, default=10_000, help="Maximum number of examples per video.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Output directory. Defaults to results/pipeline/comparisons/benchmark_10videos_staged_v1",
    )
    parser.add_argument(
        "--videos",
        nargs="+",
        default=DEFAULT_VIDEOS,
        help="Video ids to evaluate.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[name for name, _ in METHOD_ORDER],
        choices=[name for name, _ in METHOD_ORDER],
        help="Subset of methods to run, still executed in the given order.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(DEFAULT_TASKS),
        help="Task names to evaluate. Defaults to the canonical HD-EPIC task set.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Override the answer model id.",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable model thinking mode when supported.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun method/video pairs even if final_summary.json already exists.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    output_root = args.output_root or _default_output_root()
    output_root.mkdir(parents=True, exist_ok=True)
    log_path = output_root / "progress.log"
    started_at = time.perf_counter()
    budget_config = BudgetConfig()
    answer_config = default_answer_config()
    if args.model_id:
        answer_config.model_id = str(args.model_id)
    answer_config.enable_thinking = bool(args.enable_thinking)
    selected_methods = {name for name in args.methods}
    tasks = list(args.tasks)
    completed: dict[str, list[dict[str, Any]]] = {name: [] for name, _ in METHOD_ORDER if name in selected_methods}

    answerer = build_answerer(answer_config)
    try:
        for method_name, runner in METHOD_ORDER:
            if method_name not in selected_methods:
                continue
            log_line(log_path, f"[method_start] method={method_name}")
            for video_id in args.videos:
                output_dir = output_root / method_name / video_id.lower()
                final_path = output_dir / "final_summary.json"
                existing = None if args.force else _load_existing_summary(final_path)
                if existing is not None:
                    log_line(log_path, f"[skip] method={method_name} video={video_id} reason=completed_exists")
                    completed[method_name].append(
                        {
                            "video_id": video_id,
                            "summary": existing["summary"],
                            "elapsed_sec": existing["run_state"]["elapsed_sec"],
                            "completed_examples": existing["run_state"]["completed_examples"],
                        }
                    )
                    _write_batch_state(
                        output_root=output_root,
                        completed=completed,
                        started_at=started_at,
                        status="running",
                    )
                    continue

                log_line(log_path, f"[video_start] method={method_name} video={video_id} limit={args.limit}")
                result = runner(
                    video_id=video_id,
                    tasks=tasks,
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
                _write_batch_state(
                    output_root=output_root,
                    completed=completed,
                    started_at=started_at,
                    status="running",
                )
                log_line(log_path, f"[video_done] method={method_name} video={video_id}")
            log_line(log_path, f"[method_done] method={method_name}")

        _write_batch_state(
            output_root=output_root,
            completed=completed,
            started_at=started_at,
            status="completed",
        )
        log_line(log_path, "[complete]")
    except Exception as exc:
        log_line(log_path, f"[error] type={type(exc).__name__} error={exc}")
        _write_batch_state(
            output_root=output_root,
            completed=completed,
            started_at=started_at,
            status="failed",
        )
        raise
    finally:
        answerer.unload()


if __name__ == "__main__":
    main()
