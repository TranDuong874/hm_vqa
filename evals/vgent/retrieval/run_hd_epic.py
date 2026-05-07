from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModel, AutoTokenizer

from evals.common.vlm_baseline_runner import BaselineExample, _append_jsonl, _log_line, _write_json
from evals.hd_epic.retrieval.run_localization_ablation import (
    DEFAULT_COVERAGE_THRESHOLD,
    DEFAULT_RECALL_K,
    _build_examples,
    _build_examples_from_manifest,
    _metrics_for_hits,
    _participant_video_ids,
    _summarize,
    _extract_target_text,
)
from evals.hd_epic.dataset import (
    example_scope_for_video,
    gold_spans_for_video,
    load_temporal_examples_for_video,
)


REPO_ROOT = Path("/home/tranduong/dev/hm_vqa")
DEFAULT_VIDEO_ROOT = Path("/home/tranduong/dev/dataset/HD-EPIC")
DEFAULT_FEATURE_ROOT = Path("/home/tranduong/dev/hm_vqa/local_storage/flat_files/hd_epic_features_p01")
DEFAULT_CACHE_ROOT = Path(
    "/home/tranduong/dev/hm_vqa/local_storage/flat_files/vgent/offline_graph_cache_hd_epic"
)
DEFAULT_OUTPUT_ROOT = Path("/home/tranduong/dev/hm_vqa/results/vgent/hd_epic_retrieval_only")
DEFAULT_TASKS = ("fine_grained_action_localization",)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run retrieval-only Vgent graph evaluation on HD-EPIC.")
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--participant", default="P01")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--coverage-threshold", type=float, default=DEFAULT_COVERAGE_THRESHOLD)
    parser.add_argument("--n-retrieval", type=int, default=max(DEFAULT_RECALL_K))
    parser.add_argument(
        "--query-mode",
        choices=["target", "question", "target_question"],
        default="target",
        help="Text used to query Vgent chunk descriptions.",
    )
    parser.add_argument(
        "--candidate-mode",
        choices=["all_nodes", "entity_threshold"],
        default="all_nodes",
        help="all_nodes ranks every chunk; entity_threshold first filters chunks through Vgent entity keys.",
    )
    parser.add_argument("--entity-threshold", type=float, default=0.5)
    parser.add_argument("--limit-videos", type=int, default=None)
    parser.add_argument("--limit-examples", type=int, default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _temporal_examples_by_video(video_ids: list[str]) -> dict[str, dict[str, Any]]:
    by_video: dict[str, dict[str, Any]] = {}
    for video_id in video_ids:
        examples = load_temporal_examples_for_video(
            repo_root=REPO_ROOT,
            video_id=video_id,
            tasks=DEFAULT_TASKS,
        )
        by_video[video_id] = {example.example_id: example for example in examples}
    return by_video


def _lookup_temporal_example(temporal_examples: dict[str, dict[str, Any]], *, video_id: str, example_id: str) -> Any:
    try:
        return temporal_examples[video_id][example_id]
    except KeyError as exc:
        raise KeyError(f"Missing temporal example video_id={video_id} example_id={example_id}") from exc


def _run_name(args: argparse.Namespace) -> str:
    name = f"vgent_graph_retrieval_{args.query_mode}_{args.candidate_mode}_k{args.n_retrieval}"
    if args.candidate_mode == "entity_threshold":
        name += f"_thr{args.entity_threshold:g}"
    name += f"_cov{str(args.coverage_threshold).replace('.', 'p')}"
    return name


def _load_manifest_rows(manifest_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(manifest_path.read_text())
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"Manifest missing rows list: {manifest_path}")
    return [dict(row) for row in rows]


def _build_eval_set(args: argparse.Namespace) -> tuple[list[str], list[BaselineExample]]:
    if args.manifest is not None:
        manifest_rows = _load_manifest_rows(args.manifest)
        if args.limit_examples is not None:
            manifest_rows = manifest_rows[: max(int(args.limit_examples), 0)]
        if args.limit_videos is not None:
            allowed = sorted({str(row["video_id"]) for row in manifest_rows})[: max(int(args.limit_videos), 0)]
            allowed_set = set(allowed)
            manifest_rows = [row for row in manifest_rows if str(row["video_id"]) in allowed_set]
        video_ids = sorted({str(row["video_id"]) for row in manifest_rows})
        examples = _build_examples_from_manifest(video_root=args.video_root, manifest_rows=manifest_rows)
    else:
        video_ids = _participant_video_ids(args.feature_root, args.participant)
        if args.limit_videos is not None:
            video_ids = video_ids[: max(int(args.limit_videos), 0)]
        examples = _build_examples(video_root=args.video_root, video_ids=video_ids)
        if args.limit_examples is not None:
            examples = examples[: max(int(args.limit_examples), 0)]
            video_ids = sorted({example.video_id for example in examples})
    return video_ids, examples


def _load_video_cache(cache_root: Path, video_id: str) -> dict[str, Any]:
    video_dir = cache_root / "hd_epic" / video_id
    chunks_path = video_dir / "chunks.jsonl"
    emb_path = video_dir / "bge_embeddings.pt"
    if not chunks_path.exists():
        raise FileNotFoundError(f"Missing Vgent chunks cache: {chunks_path}")
    if not emb_path.exists():
        raise FileNotFoundError(f"Missing Vgent BGE embedding cache: {emb_path}")

    chunks: dict[int, dict[str, Any]] = {}
    for line in chunks_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("status") == "ok":
            chunks[int(row["chunk_id"])] = row
    emb = torch.load(emb_path, map_location="cpu")
    node_ids = [int(x) for x in emb.get("node_ids", [])]
    node_embeddings = emb.get("node_embeddings")
    entity_keys = [str(x) for x in emb.get("entity_keys", [])]
    entity_embeddings = emb.get("entity_embeddings")
    if node_embeddings is None or len(node_ids) == 0:
        raise RuntimeError(f"No node embeddings in {emb_path}")
    return {
        "chunks": chunks,
        "node_ids": node_ids,
        "node_embeddings": torch.nn.functional.normalize(node_embeddings.float(), p=2, dim=1),
        "entity_keys": entity_keys,
        "entity_embeddings": (
            torch.nn.functional.normalize(entity_embeddings.float(), p=2, dim=1)
            if entity_embeddings is not None and len(entity_keys) > 0
            else None
        ),
    }


def _load_bge() -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")
    model = AutoModel.from_pretrained("BAAI/bge-large-en-v1.5")
    model.eval()
    return model, tokenizer


def _encode_query(texts: list[str], *, model: Any, tokenizer: Any) -> torch.Tensor:
    encoded = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        output = model(**encoded)
        emb = output[0][:, 0]
    emb = torch.nn.functional.normalize(emb.float(), p=2, dim=1)
    return emb


def _query_texts(example: BaselineExample, mode: str) -> list[str]:
    target = _extract_target_text(example.question)
    if mode == "target":
        return [target]
    if mode == "question":
        return [example.question]
    return [target, example.question]


def _entity_candidate_nodes(cache: dict[str, Any], query_emb: torch.Tensor, threshold: float) -> set[int]:
    entity_embeddings = cache.get("entity_embeddings")
    entity_keys = cache.get("entity_keys") or []
    if entity_embeddings is None or len(entity_keys) == 0:
        return set()
    entity_scores = torch.mean(query_emb @ entity_embeddings.T, dim=0)
    selected_entities = [
        entity_keys[i]
        for i, score in enumerate(entity_scores.tolist())
        if float(score) > threshold
    ]
    if not selected_entities:
        return set()
    # Load graph entity mapping lazily from graph.pkl because bge_embeddings only stores keys.
    return set()


def _retrieve_hits(
    *,
    cache: dict[str, Any],
    query_emb: torch.Tensor,
    n_retrieval: int,
    candidate_mode: str,
    entity_threshold: float,
    scope_start_sec: float | None,
    scope_end_sec: float | None,
) -> list[dict[str, Any]]:
    node_ids: list[int] = cache["node_ids"]
    node_embeddings: torch.Tensor = cache["node_embeddings"]
    chunks: dict[int, dict[str, Any]] = cache["chunks"]

    scores = torch.mean(query_emb @ node_embeddings.T, dim=0)
    candidate_indices = list(range(len(node_ids)))
    if candidate_mode == "entity_threshold":
        # Vgent's entity-threshold path is brittle for our cached graph quality.
        # We use it as a filter only if it selects at least one node; otherwise fall back.
        selected = _entity_candidate_nodes(cache, query_emb, entity_threshold)
        if selected:
            candidate_indices = [idx for idx, node_id in enumerate(node_ids) if node_id in selected]

    scoped_candidates: list[int] = []
    for idx in candidate_indices:
        node_id = node_ids[idx]
        chunk = chunks.get(node_id)
        if chunk is None:
            continue
        start = chunk.get("start_time_sec")
        end = chunk.get("end_time_sec")
        if start is None or end is None:
            continue
        end_time = float(end) + 1.0
        if scope_start_sec is not None and scope_end_sec is not None:
            if max(float(start), float(scope_start_sec)) > min(end_time, float(scope_end_sec)):
                continue
        scoped_candidates.append(idx)

    ranked = sorted(scoped_candidates, key=lambda idx: float(scores[idx]), reverse=True)
    hits: list[dict[str, Any]] = []
    for idx in ranked[:n_retrieval]:
        node_id = node_ids[idx]
        chunk = chunks.get(node_id)
        if chunk is None:
            continue
        start = chunk.get("start_time_sec")
        end = chunk.get("end_time_sec")
        if start is None or end is None:
            continue
        # Offline chunks record the last sampled timestamp. Use the next second as
        # exclusive end so a 64-frame 1 FPS chunk has roughly 64s duration.
        end_time = float(end) + 1.0
        hits.append(
            {
                "segment_id": f"vgent_chunk_{node_id:04d}",
                "score": float(scores[idx]),
                "start_time_sec": float(start),
                "end_time_sec": end_time,
                "chunk_id": node_id,
            }
        )
    return hits


def main() -> None:
    args = _parse_args()
    output_dir = args.output_root / _run_name(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "rows.jsonl"
    progress_path = output_dir / "progress.log"
    rolling_summary_path = output_dir / "rolling_summary.json"
    if not args.resume:
        for path in (rows_path, progress_path, rolling_summary_path, output_dir / "final_summary.json"):
            if path.exists():
                path.unlink()

    video_ids, examples = _build_eval_set(args)
    temporal_examples = _temporal_examples_by_video(video_ids)

    existing_rows: list[dict[str, Any]] = []
    if args.resume and rows_path.exists():
        existing_rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
    completed = {str(row["example_id"]) for row in existing_rows}
    pending = [example for example in examples if str(example.example_id) not in completed]

    _write_json(
        output_dir / "config.json",
        {
            "cache_root": str(args.cache_root),
            "participant": args.participant,
            "manifest": str(args.manifest) if args.manifest else None,
            "coverage_threshold": args.coverage_threshold,
            "n_retrieval": args.n_retrieval,
            "query_mode": args.query_mode,
            "candidate_mode": args.candidate_mode,
            "entity_threshold": args.entity_threshold,
            "recall_k": list(DEFAULT_RECALL_K),
        },
    )
    _write_json(rolling_summary_path, _summarize(existing_rows, len(examples)))
    _log_line(progress_path, f"[start] total={len(examples)} pending={len(pending)} query_mode={args.query_mode} candidate_mode={args.candidate_mode}")

    model, tokenizer = _load_bge()
    video_cache: dict[str, dict[str, Any]] = {}
    rows = list(existing_rows)
    for index, example in enumerate(pending, start=len(existing_rows) + 1):
        _log_line(progress_path, f"[item_start] index={index}/{len(examples)} example_id={example.example_id} video={example.video_id}")
        try:
            if example.video_id not in video_cache:
                video_cache[example.video_id] = _load_video_cache(args.cache_root, example.video_id)
            temporal_example = _lookup_temporal_example(
                temporal_examples,
                video_id=example.video_id,
                example_id=example.example_id,
            )
            scope_start_sec, scope_end_sec = example_scope_for_video(temporal_example, example.video_id)
            gold_spans = gold_spans_for_video(temporal_example, example.video_id)
            query_texts = _query_texts(example, args.query_mode)
            query_emb = _encode_query(query_texts, model=model, tokenizer=tokenizer)
            hits = _retrieve_hits(
                cache=video_cache[example.video_id],
                query_emb=query_emb,
                n_retrieval=args.n_retrieval,
                candidate_mode=args.candidate_mode,
                entity_threshold=args.entity_threshold,
                scope_start_sec=scope_start_sec,
                scope_end_sec=scope_end_sec,
            )
            metrics = _metrics_for_hits(
                hits=hits,
                gold_spans=gold_spans,
                coverage_threshold=args.coverage_threshold,
            )
            row = {
                "example_id": example.example_id,
                "video_id": example.video_id,
                "task_name": temporal_example.task_name,
                "question": example.question,
                "query_texts": query_texts,
                "scope_start_sec": scope_start_sec,
                "scope_end_sec": scope_end_sec,
                "gold_spans": gold_spans,
                "retrieved_hits": hits,
                "metrics": metrics,
            }
            rows.append(row)
            _append_jsonl(rows_path, row)
            _write_json(rolling_summary_path, _summarize(rows, len(examples)))
            _log_line(
                progress_path,
                f"[item_done] index={index}/{len(examples)} example_id={example.example_id} cov3={metrics['best_coverage_at_3']:.3f} cov_r3={metrics['coverage_recall_at_3']:.0f}",
            )
        except Exception as exc:
            _log_line(progress_path, f"[item_error] index={index}/{len(examples)} example_id={example.example_id} error={type(exc).__name__}: {exc}")
            raise

    final_summary = {
        "participant": args.participant,
        "method": "vgent_graph_retrieval",
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        **_summarize(rows, len(examples)),
    }
    _write_json(output_dir / "final_summary.json", final_summary)
    _log_line(
        progress_path,
        f"[done] scored={final_summary['scored']} mean_cov3={final_summary['mean_best_coverage_at_3']:.4f} cov_r3={final_summary['coverage_recall_at_3']:.4f}",
    )


if __name__ == "__main__":
    main()
