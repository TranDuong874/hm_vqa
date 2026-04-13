from __future__ import annotations

import argparse
import sys
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pipeline.config import PIPELINE_CONFIG
from pipeline.core.features import build_query_encoder, load_feature_archive
from pipeline.core.io import (
    append_jsonl,
    build_run_state,
    load_jsonl,
    log_line,
    stable_run_id,
    write_json,
)
from pipeline.core.metrics import summarize_layer2_hits, summarize_layer3_hits, summarize_rows, summarize_selected_segment_hits
from pipeline.core.retrieve import (
    extract_target_text,
    gather_segment_embeddings,
    rank_frames_in_segments,
    rank_segments,
    select_ranked_hits,
    restrict_segments_to_hits,
)
from pipeline.core.schema import PipelineConfig
from pipeline.core.segments import build_adaptive_layer3_segments, build_fixed_windows, mean_pool_segments
from evals.hd_epic.temporal import example_scope_for_video, gold_spans_for_video, load_temporal_examples_for_video


def _default_output_dir(video_id: str, *, limit: int | None) -> Path:
    base = PIPELINE_CONFIG.output_root / "hd_epic" / video_id.lower()
    if limit is not None:
        return base.parent / f"{base.name}_limit{int(limit)}"
    return base


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean retrieval-only HD-EPIC baseline.")
    parser.add_argument("video_id", help="HD-EPIC video id to evaluate.")
    parser.add_argument("--limit", type=int, default=None, help="Optional example limit for smoke tests.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional output directory override.")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True, help="Resume from rows.jsonl if present.")
    return parser


def _row_identity(example: Any) -> str:
    if hasattr(example, "example_id"):
        return str(example.example_id)
    return str(example["example_id"])


def _write_status_files(
    *,
    output_dir: Path,
    config_payload: dict[str, Any],
    video_id: str,
    rows: list[dict[str, Any]],
    total_examples: int,
    started_at: float,
    status: str,
    final_message: str | None = None,
) -> None:
    run_state = build_run_state(
        video_id=video_id,
        current_video_id=video_id,
        total_examples=total_examples,
        rows=rows,
        started_at=started_at,
        status=status,
    )
    summary = summarize_rows(rows=rows, total_examples=total_examples)
    payload = {
        "config": config_payload,
        "run_state": run_state,
        "summary": summary,
    }
    if final_message is not None:
        payload["message"] = final_message
    write_json(output_dir / "rolling_summary.json", payload)
    if status in {"completed", "failed"}:
        write_json(output_dir / "final_summary.json", payload)


def run_with_config(
    *,
    config: PipelineConfig,
    video_id: str,
    output_dir: Path | None = None,
    limit: int | None = None,
    resume: bool = True,
) -> None:
    repo_root = config.repo_root
    video_id = str(video_id)
    output_dir = output_dir or _default_output_dir(video_id, limit=limit)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "rows.jsonl"
    log_path = output_dir / "progress.log"

    config_payload = {
        "video_id": video_id,
        "limit": limit,
        "pipeline": config.to_dict(),
    }
    write_json(output_dir / "config.json", config_payload)

    started_at = __import__("time").perf_counter()
    existing_rows = load_jsonl(rows_path) if resume else []
    completed_ids = {_row_identity(row) for row in existing_rows if row.get("status") == "ok"}
    rows = list(existing_rows)
    examples: list[Any] = []
    run_id = "pipeline_pending"
    query_encoder = None

    log_line(log_path, f"[start] video={video_id} resume={resume} existing_rows={len(existing_rows)}")

    try:
        archive = load_feature_archive(repo_root, video_id)
        query_encoder = build_query_encoder(
            repo_root=repo_root,
            model_name=archive.model_name,
            pretrained_name=archive.pretrained_name,
            device=config.retrieval.device,
        )
        l2_segments = build_fixed_windows(
            timestamps=archive.timestamps,
            window_seconds=config.segmentation.l2_window_seconds,
            stride_seconds=config.segmentation.l2_window_stride_seconds,
            prefix="l2",
        )
        l2_embeddings = mean_pool_segments(archive.frame_embeddings, l2_segments)
        l3_segments, l3_diagnostics = build_adaptive_layer3_segments(
            l2_segments=l2_segments,
            l2_embeddings=l2_embeddings,
            config=config.segmentation,
            prefix="l3",
        )
        l3_embeddings = mean_pool_segments(archive.frame_embeddings, l3_segments)

        examples = load_temporal_examples_for_video(
            repo_root=repo_root,
            video_id=video_id,
            tasks=config.tasks,
        )
        if limit is not None:
            examples = examples[: max(int(limit), 0)]

        run_id = stable_run_id(
            {
                "video_id": video_id,
                "tasks": list(config.tasks),
                "segmentation": asdict(config.segmentation),
                "retrieval": asdict(config.retrieval),
                "limit": limit,
            }
        )
        log_line(
            log_path,
            f"[index_ready] video={video_id} examples={len(examples)} l2_segments={len(l2_segments)} l3_segments={len(l3_segments)} run_id={run_id}",
        )

        _write_status_files(
            output_dir=output_dir,
            config_payload=config_payload,
            video_id=video_id,
            rows=rows,
            total_examples=len(examples),
            started_at=started_at,
            status="running",
        )

        for index, example in enumerate(examples, start=1):
            example_id = _row_identity(example)
            if example_id in completed_ids:
                log_line(log_path, f"[resume_skip] row={index}/{len(examples)} example_id={example_id}")
                continue

            query_text = extract_target_text(example.question)
            query_embedding = query_encoder.encode_texts(
                [query_text],
                batch_size=config.retrieval.openclip_batch_size,
            )[0]

            gold_spans = gold_spans_for_video(example, video_id)
            scope_start_sec, scope_end_sec = example_scope_for_video(example, video_id)

            layer3_candidates = [
                segment
                for segment in l3_segments
                if (
                    scope_start_sec is None
                    or max(float(segment.start_time_sec), float(scope_start_sec)) <= min(float(segment.end_time_sec), float(scope_end_sec))
                )
            ]
            layer3_candidate_embeddings = gather_segment_embeddings(
                all_segments=l3_segments,
                all_embeddings=l3_embeddings,
                selected_segments=layer3_candidates,
            )
            all_layer3_hits = rank_segments(
                query_embedding=query_embedding,
                segment_embeddings=layer3_candidate_embeddings,
                segments=layer3_candidates,
                video_id=video_id,
                top_k=max(len(layer3_candidates), 1),
            )
            layer3_hits = select_ranked_hits(
                hits=all_layer3_hits,
                mode=config.retrieval.selection_mode,
                top_k=config.retrieval.layer3_top_k,
                relative_alpha=config.retrieval.layer3_relative_alpha,
                max_keep=config.retrieval.layer3_max_keep,
            )

            layer2_candidates = restrict_segments_to_hits(
                segments=l2_segments,
                parent_hits=layer3_hits,
                scope_start_sec=scope_start_sec,
                scope_end_sec=scope_end_sec,
            )
            layer2_candidate_embeddings = gather_segment_embeddings(
                all_segments=l2_segments,
                all_embeddings=l2_embeddings,
                selected_segments=layer2_candidates,
            )
            all_layer2_hits = rank_segments(
                query_embedding=query_embedding,
                segment_embeddings=layer2_candidate_embeddings,
                segments=layer2_candidates,
                video_id=video_id,
                top_k=max(len(layer2_candidates), 1),
            )
            layer2_hits = select_ranked_hits(
                hits=all_layer2_hits,
                mode=config.retrieval.selection_mode,
                top_k=config.retrieval.layer2_top_k,
                relative_alpha=config.retrieval.layer2_relative_alpha,
                max_keep=config.retrieval.layer2_max_keep,
            )
            layer1_hits = rank_frames_in_segments(
                query_embedding=query_embedding,
                frame_embeddings=archive.frame_embeddings,
                candidate_segments=layer2_hits,
                frame_timestamps=archive.timestamps,
                video_id=video_id,
                top_k=config.retrieval.layer1_top_k,
            )

            layer3_metrics = summarize_layer3_hits(
                layer3_hits=[hit.to_dict() for hit in layer3_hits],
                gold_spans=gold_spans,
                coverage_threshold=config.retrieval.layer3_coverage_threshold,
            )
            layer3_selected_metrics = summarize_selected_segment_hits(
                hits=[hit.to_dict() for hit in layer3_hits],
                gold_spans=gold_spans,
                prefix="Layer3",
            )
            layer2_metrics = summarize_layer2_hits(
                layer2_hits=[hit.to_dict() for hit in layer2_hits],
                gold_spans=gold_spans,
                top_k=config.retrieval.layer2_top_k,
            )
            layer2_selected_metrics = summarize_selected_segment_hits(
                hits=[hit.to_dict() for hit in layer2_hits],
                gold_spans=gold_spans,
                prefix="Layer2",
            )
            row = {
                "run_id": run_id,
                "example_id": example_id,
                "task_name": example.task_name,
                "video_id": video_id,
                "question": example.question,
                "query_text": query_text,
                "gold_spans": gold_spans,
                "layer3_hits": [hit.to_dict() for hit in layer3_hits],
                "layer2_hits": [hit.to_dict() for hit in layer2_hits],
                "layer1_hits": [hit.to_dict() for hit in layer1_hits],
                "metrics": {
                    **{key: round(float(value), 6) for key, value in layer3_metrics.items()},
                    **{key: round(float(value), 6) for key, value in layer3_selected_metrics.items()},
                    **{key: round(float(value), 6) for key, value in layer2_metrics.items()},
                    **{key: round(float(value), 6) for key, value in layer2_selected_metrics.items()},
                },
                "diagnostics": {
                    "scope_start_sec": scope_start_sec,
                    "scope_end_sec": scope_end_sec,
                    "l3_peak_indices": l3_diagnostics["kept_peak_indices"],
                },
                "status": "ok",
            }
            append_jsonl(rows_path, row)
            rows.append(row)
            completed_ids.add(example_id)
            log_line(
                log_path,
                f"[progress] row={index}/{len(examples)} example_id={example_id} layer3_hit3={row['metrics']['Layer3CoverageHit@3']:.0f} layer2_hit1={row['metrics']['Layer2 Hit@1_gap0']:.0f} layer2_d1={row['metrics']['Layer2 mean_top1_distance_sec']:.3f}",
            )
            _write_status_files(
                output_dir=output_dir,
                config_payload=config_payload,
                video_id=video_id,
                rows=rows,
                total_examples=len(examples),
                started_at=started_at,
                status="running",
            )

        _write_status_files(
            output_dir=output_dir,
            config_payload=config_payload,
            video_id=video_id,
            rows=rows,
            total_examples=len(examples),
            started_at=started_at,
            status="completed",
            final_message="run completed",
        )
        log_line(log_path, f"[done] video={video_id} rows={len(rows)}")
    except Exception as exc:
        error_row = {
            "run_id": run_id,
            "example_id": "__fatal__",
            "task_name": "fatal",
            "video_id": video_id,
            "question": "",
            "query_text": "",
            "gold_spans": [],
            "layer3_hits": [],
            "layer2_hits": [],
            "layer1_hits": [],
            "metrics": {},
            "status": "error",
            "message": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        append_jsonl(rows_path, error_row)
        rows.append(error_row)
        log_line(log_path, f"[fatal] video={video_id} error={type(exc).__name__}: {exc}")
        _write_status_files(
            output_dir=output_dir,
            config_payload=config_payload,
            video_id=video_id,
            rows=rows,
            total_examples=len(examples),
            started_at=started_at,
            status="failed",
            final_message=f"{type(exc).__name__}: {exc}",
        )
        raise
    finally:
        if query_encoder is not None:
            del query_encoder


def main() -> None:
    args = _build_arg_parser().parse_args()
    run_with_config(
        config=PIPELINE_CONFIG,
        video_id=str(args.video_id),
        output_dir=args.output_dir,
        limit=args.limit,
        resume=bool(args.resume),
    )


__all__ = ["main", "run_with_config"]


if __name__ == "__main__":
    main()
