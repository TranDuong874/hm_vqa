from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from evals.hd_epic.loader import filter_examples_for_video, load_examples
from src.pipeline.config import PIPELINE_CONFIG
from src.pipeline.core.features import build_query_encoder, load_feature_archive
from src.pipeline.core.retrieve import extract_target_text
from evals.hd_epic.temporal import example_scope_for_video, gold_spans_for_video


DEFAULT_TASKS = [
    "fine_grained_action_localization",
    "recipe_step_localization",
]
DEFAULT_METHODS = ["pure_vlm", "ours"]
DEFAULT_TOLERANCES = [1.0, 2.5, 5.0]


@dataclass(slots=True)
class ExampleAnalysis:
    method: str
    video_id: str
    example_id: str
    task_name: str
    query_text: str
    mcq_correct: float
    sampled_timestamps_sec: list[float]
    sampled_similarities: list[float]
    min_frame_distance_sec: float
    mean_frame_distance_sec: float
    frame_precision_by_tol: dict[str, float]
    frame_hit_by_tol: dict[str, float]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze grounding quality of saved benchmark runs.")
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=Path("results/pipeline/comparisons/benchmark_10videos_stratified_v1"),
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=DEFAULT_METHODS,
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=DEFAULT_TASKS,
    )
    parser.add_argument(
        "--tolerances",
        nargs="+",
        type=float,
        default=DEFAULT_TOLERANCES,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/pipeline/analysis/grounding_quality"),
    )
    return parser.parse_args()


def _load_example_index(tasks: list[str]) -> dict[tuple[str, str], Any]:
    examples = load_examples(tasks, REPO_ROOT)
    index: dict[tuple[str, str], Any] = {}
    video_to_examples: defaultdict[str, list[Any]] = defaultdict(list)
    for example in examples:
        for video in example.input_videos:
            video_to_examples[video.video_id].append(example)
    for video_id, items in video_to_examples.items():
        filtered = filter_examples_for_video(items, video_id)
        for example in filtered:
            index[(video_id, example.example_id)] = example
    return index


def _sample_uniform_timestamps(
    *,
    frame_budget: int,
    start_time_sec: float,
    end_time_sec: float,
    fps: float,
    total_frames: int,
) -> list[float]:
    start_frame = max(0, int(round(start_time_sec * fps)))
    end_frame = min(total_frames - 1, int(round(end_time_sec * fps)))
    if end_frame < start_frame:
        end_frame = start_frame
    target_indices = np.linspace(start_frame, end_frame, num=max(frame_budget, 1))
    raw_indices = [int(index) for index in np.round(target_indices).tolist()]
    frame_indices = sorted(dict.fromkeys(raw_indices))
    return [frame_index / fps for frame_index in frame_indices]


def _distance_to_gold(time_sec: float, gold_spans: list[dict[str, Any]]) -> float:
    if not gold_spans:
        return float("inf")
    best = float("inf")
    for span in gold_spans:
        start_time_sec = float(span["start_time_sec"])
        end_time_sec = float(span["end_time_sec"])
        if start_time_sec <= time_sec <= end_time_sec:
            return 0.0
        if time_sec < start_time_sec:
            best = min(best, start_time_sec - time_sec)
        else:
            best = min(best, time_sec - end_time_sec)
    return best


def _nearest_frame_indices(timestamps: np.ndarray, sampled_timestamps_sec: list[float]) -> list[int]:
    if not sampled_timestamps_sec:
        return []
    indices: list[int] = []
    for time_sec in sampled_timestamps_sec:
        idx = int(np.abs(timestamps - float(time_sec)).argmin())
        indices.append(idx)
    return indices


def _ensure_query_encoder(
    *,
    encoders: dict[tuple[str, str], Any],
    model_name: str,
    pretrained_name: str,
    device: str,
) -> Any:
    key = (model_name, pretrained_name)
    if key not in encoders:
        encoders[key] = build_query_encoder(
            repo_root=REPO_ROOT,
            model_name=model_name,
            pretrained_name=pretrained_name,
            device=device,
        )
    return encoders[key]


def _sampled_timestamps_for_row(
    *,
    method: str,
    row: dict[str, Any],
    example: Any,
    archive: Any,
) -> list[float]:
    if method == "pure_vlm":
        scope_start_sec, scope_end_sec = example_scope_for_video(example, archive.video_id)
        start_time_sec = float(scope_start_sec or 0.0)
        end_time_sec = float(scope_end_sec or archive.duration_sec)
        frame_budget = int(round(float(row.get("frames_per_clip") or row.get("total_frames_seen_per_question") or 20.0)))
        return _sample_uniform_timestamps(
            frame_budget=frame_budget,
            start_time_sec=start_time_sec,
            end_time_sec=end_time_sec,
            fps=float(archive.fps),
            total_frames=int(archive.total_frames),
        )

    selected = row.get("selected_evidence")
    if not selected:
        return []
    start_time_sec = float(selected["start_time_sec"])
    end_time_sec = float(selected["end_time_sec"])
    frame_budget = int(round(float(row.get("frames_per_clip") or 4.0)))
    return _sample_uniform_timestamps(
        frame_budget=frame_budget,
        start_time_sec=start_time_sec,
        end_time_sec=end_time_sec,
        fps=float(archive.fps),
        total_frames=int(archive.total_frames),
    )


def _analyze_method(
    *,
    benchmark_root: Path,
    method: str,
    tolerances: list[float],
    example_index: dict[tuple[str, str], Any],
) -> tuple[dict[str, Any], list[ExampleAnalysis]]:
    method_root = benchmark_root / method
    if not method_root.exists():
        raise FileNotFoundError(f"Missing method directory: {method_root}")

    analyses: list[ExampleAnalysis] = []
    archives: dict[str, Any] = {}
    encoders: dict[tuple[str, str], Any] = {}

    for rows_path in sorted(method_root.glob("*/rows.jsonl")):
        video_id = rows_path.parent.name.upper()
        archive = archives.get(video_id)
        if archive is None:
            archive = load_feature_archive(REPO_ROOT, video_id)
            archives[video_id] = archive
        encoder = _ensure_query_encoder(
            encoders=encoders,
            model_name=archive.model_name,
            pretrained_name=archive.pretrained_name,
            device="cuda",
        )

        rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
        query_texts = []
        valid_rows: list[dict[str, Any]] = []
        examples: list[Any] = []
        for row in rows:
            if row.get("status") != "ok":
                continue
            example = example_index.get((video_id, row["example_id"]))
            if example is None:
                continue
            query_texts.append(extract_target_text(example.question))
            valid_rows.append(row)
            examples.append(example)

        if not valid_rows:
            continue
        query_embeddings = encoder.encode_texts(query_texts, batch_size=PIPELINE_CONFIG.retrieval.openclip_batch_size)

        for row, example, query_embedding, query_text in zip(valid_rows, examples, query_embeddings, query_texts, strict=True):
            gold_spans = gold_spans_for_video(example, video_id)
            sampled_timestamps_sec = _sampled_timestamps_for_row(
                method=method,
                row=row,
                example=example,
                archive=archive,
            )
            frame_indices = _nearest_frame_indices(archive.timestamps, sampled_timestamps_sec)
            if frame_indices:
                sampled_embeddings = archive.frame_embeddings[frame_indices]
                similarities = torch.matmul(sampled_embeddings, query_embedding).detach().cpu().tolist()
            else:
                similarities = []

            distances = [_distance_to_gold(time_sec, gold_spans) for time_sec in sampled_timestamps_sec]
            min_distance = min(distances) if distances else float("inf")
            mean_distance = float(sum(distances) / len(distances)) if distances else float("inf")
            frame_precision_by_tol: dict[str, float] = {}
            frame_hit_by_tol: dict[str, float] = {}
            for tolerance in tolerances:
                key = f"{tolerance:g}s"
                hits = [1.0 if distance <= tolerance else 0.0 for distance in distances]
                frame_precision_by_tol[key] = float(sum(hits) / len(hits)) if hits else 0.0
                frame_hit_by_tol[key] = 1.0 if any(hits) else 0.0

            analyses.append(
                ExampleAnalysis(
                    method=method,
                    video_id=video_id,
                    example_id=row["example_id"],
                    task_name=example.task_name,
                    query_text=query_text,
                    mcq_correct=float(row.get("mcq_correct") or 0.0),
                    sampled_timestamps_sec=sampled_timestamps_sec,
                    sampled_similarities=[float(value) for value in similarities],
                    min_frame_distance_sec=float(min_distance),
                    mean_frame_distance_sec=float(mean_distance),
                    frame_precision_by_tol=frame_precision_by_tol,
                    frame_hit_by_tol=frame_hit_by_tol,
                )
            )

    for encoder in encoders.values():
        del encoder

    total = len(analyses)
    if total == 0:
        raise RuntimeError(f"No rows analyzed for method {method}")

    aggregate: dict[str, Any] = {
        "examples": total,
        "mcq_accuracy": round(sum(item.mcq_correct for item in analyses) / total, 6),
        "examples_with_gold_on_video": sum(
            1 for item in analyses if np.isfinite(item.min_frame_distance_sec) and np.isfinite(item.mean_frame_distance_sec)
        ),
        "examples_without_gold_on_video": sum(
            1 for item in analyses if not np.isfinite(item.min_frame_distance_sec) or not np.isfinite(item.mean_frame_distance_sec)
        ),
        "min_frame_distance_sec_mean": None,
        "mean_frame_distance_sec_mean": None,
        "sampled_frame_similarity_mean": round(
            sum((sum(item.sampled_similarities) / len(item.sampled_similarities)) if item.sampled_similarities else 0.0 for item in analyses) / total,
            6,
        ),
        "sampled_frame_similarity_max_mean": round(
            sum((max(item.sampled_similarities) if item.sampled_similarities else 0.0) for item in analyses) / total,
            6,
        ),
    }
    finite_min = [item.min_frame_distance_sec for item in analyses if np.isfinite(item.min_frame_distance_sec)]
    finite_mean = [item.mean_frame_distance_sec for item in analyses if np.isfinite(item.mean_frame_distance_sec)]
    if finite_min:
        aggregate["min_frame_distance_sec_mean"] = round(sum(finite_min) / len(finite_min), 6)
    if finite_mean:
        aggregate["mean_frame_distance_sec_mean"] = round(sum(finite_mean) / len(finite_mean), 6)

    for tolerance in tolerances:
        key = f"{tolerance:g}s"
        hit_examples = [item for item in analyses if item.frame_hit_by_tol[key] > 0.0]
        miss_examples = [item for item in analyses if item.frame_hit_by_tol[key] <= 0.0]
        correct_examples = [item for item in analyses if item.mcq_correct > 0.0]
        correct_with_hit = [item for item in correct_examples if item.frame_hit_by_tol[key] > 0.0]
        correct_with_miss = [item for item in correct_examples if item.frame_hit_by_tol[key] <= 0.0]
        aggregate[f"frame_recall@{key}"] = round(sum(item.frame_hit_by_tol[key] for item in analyses) / total, 6)
        aggregate[f"frame_precision@{key}"] = round(sum(item.frame_precision_by_tol[key] for item in analyses) / total, 6)
        aggregate[f"mcq_accuracy_given_frame_hit@{key}"] = round(sum(item.mcq_correct for item in hit_examples) / len(hit_examples), 6) if hit_examples else None
        aggregate[f"mcq_accuracy_given_frame_miss@{key}"] = round(sum(item.mcq_correct for item in miss_examples) / len(miss_examples), 6) if miss_examples else None
        aggregate[f"correct_answers_with_frame_hit@{key}"] = len(correct_with_hit)
        aggregate[f"correct_answers_with_frame_miss@{key}"] = len(correct_with_miss)
        aggregate[f"correct_answer_grounded_rate@{key}"] = round(len(correct_with_hit) / len(correct_examples), 6) if correct_examples else None

    return aggregate, analyses


def _row_to_dict(item: ExampleAnalysis) -> dict[str, Any]:
    return {
        "method": item.method,
        "video_id": item.video_id,
        "example_id": item.example_id,
        "task_name": item.task_name,
        "query_text": item.query_text,
        "mcq_correct": item.mcq_correct,
        "sampled_timestamps_sec": item.sampled_timestamps_sec,
        "sampled_similarities": item.sampled_similarities,
        "min_frame_distance_sec": item.min_frame_distance_sec,
        "mean_frame_distance_sec": item.mean_frame_distance_sec,
        "frame_precision_by_tol": item.frame_precision_by_tol,
        "frame_hit_by_tol": item.frame_hit_by_tol,
    }


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir / args.benchmark_root.name
    output_dir.mkdir(parents=True, exist_ok=True)

    example_index = _load_example_index(list(args.tasks))
    final_payload: dict[str, Any] = {
        "benchmark_root": str(args.benchmark_root),
        "methods": {},
        "tolerances_sec": list(args.tolerances),
    }

    for method in args.methods:
        aggregate, analyses = _analyze_method(
            benchmark_root=args.benchmark_root,
            method=method,
            tolerances=list(args.tolerances),
            example_index=example_index,
        )
        final_payload["methods"][method] = {"aggregate": aggregate}
        method_dir = output_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)
        (method_dir / "rows.jsonl").write_text(
            "\n".join(json.dumps(_row_to_dict(item), ensure_ascii=True) for item in analyses) + "\n",
            encoding="utf-8",
        )
        (method_dir / "final_summary.json").write_text(
            json.dumps({"aggregate": aggregate}, indent=2) + "\n",
            encoding="utf-8",
        )

    report_lines = ["# Grounding Quality Analysis", ""]
    for method in args.methods:
        aggregate = final_payload["methods"][method]["aggregate"]
        report_lines.append(f"## {method}")
        report_lines.append(f"- examples: {aggregate['examples']}")
        report_lines.append(f"- mcq_accuracy: {aggregate['mcq_accuracy']}")
        report_lines.append(f"- min_frame_distance_sec_mean: {aggregate['min_frame_distance_sec_mean']}")
        report_lines.append(f"- mean_frame_distance_sec_mean: {aggregate['mean_frame_distance_sec_mean']}")
        report_lines.append(f"- sampled_frame_similarity_mean: {aggregate['sampled_frame_similarity_mean']}")
        report_lines.append(f"- sampled_frame_similarity_max_mean: {aggregate['sampled_frame_similarity_max_mean']}")
        for tolerance in args.tolerances:
            key = f"{tolerance:g}s"
            report_lines.append(f"- frame_recall@{key}: {aggregate[f'frame_recall@{key}']}")
            report_lines.append(f"- frame_precision@{key}: {aggregate[f'frame_precision@{key}']}")
            report_lines.append(f"- mcq_accuracy_given_frame_hit@{key}: {aggregate[f'mcq_accuracy_given_frame_hit@{key}']}")
            report_lines.append(f"- mcq_accuracy_given_frame_miss@{key}: {aggregate[f'mcq_accuracy_given_frame_miss@{key}']}")
            report_lines.append(f"- correct_answers_with_frame_hit@{key}: {aggregate[f'correct_answers_with_frame_hit@{key}']}")
            report_lines.append(f"- correct_answers_with_frame_miss@{key}: {aggregate[f'correct_answers_with_frame_miss@{key}']}")
            report_lines.append(f"- correct_answer_grounded_rate@{key}: {aggregate[f'correct_answer_grounded_rate@{key}']}")
        report_lines.append("")

    (output_dir / "final_summary.json").write_text(json.dumps(final_payload, indent=2) + "\n", encoding="utf-8")
    (output_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
