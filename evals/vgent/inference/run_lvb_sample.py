from __future__ import annotations

import argparse
import json
import pickle
import os
import sys
import tarfile
import time
import hashlib
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from evals.longvideobench.paths import LVB_FULL_MANIFEST, LVB_FULL_VIDEO_ROOT, SUBTITLE_TAR

REPO_ROOT = Path("/home/tranduong/dev/hm_vqa")
VGENT_ROOT = REPO_ROOT / "thirdparty" / "Vgent"
if str(VGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(VGENT_ROOT))

from utils.vgent import Vgent  # noqa: E402


def _load_subtitles(
    *,
    subtitle_path: str,
    subtitle_tar: Path | None = None,
) -> list[dict]:
    if subtitle_tar is not None:
        with tarfile.open(subtitle_tar) as archive:
            member_name = f"subtitles/{subtitle_path}"
            try:
                extracted = archive.extractfile(member_name)
            except KeyError:
                extracted = None
            if extracted is not None:
                return json.load(extracted)
    raise FileNotFoundError(f"Subtitle file not found: {subtitle_path}")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


DEFAULT_MANIFEST = LVB_FULL_MANIFEST
DEFAULT_VIDEO_ROOT = LVB_FULL_VIDEO_ROOT
DEFAULT_SUBTITLE_TAR = SUBTITLE_TAR
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "longvideobench" / "baselines"


@dataclass(slots=True)
class SampleRow:
    example_id: str
    video_id: str
    video_path: Path
    question: str
    prompt: str
    candidates: list[str]
    answer: str
    letters: list[str]
    duration: float | None
    duration_group: int | None
    question_category: str | None
    subtitle_path: str | None
    starting_timestamp_for_subtitles: float


def _normalize_subtitles_for_vgent(
    *,
    subtitle_path: str | None,
    subtitle_tar: Path | None,
    starting_timestamp_for_subtitles: float,
    duration: float | None,
) -> list[tuple[int, str]] | None:
    if not subtitle_path:
        return None
    subtitles = _load_subtitles(subtitle_path=subtitle_path, subtitle_tar=subtitle_tar)
    normalized: list[tuple[int, str]] = []
    for subtitle in subtitles:
        if "timestamp" in subtitle:
            start, end = subtitle["timestamp"]
            start = float(start) - float(starting_timestamp_for_subtitles)
            if not isinstance(end, (int, float)):
                end = float(duration if duration is not None else start)
            else:
                end = float(end) - float(starting_timestamp_for_subtitles)
            text = str(subtitle.get("text", "")).strip()
        else:
            continue
        if text:
            normalized.append((int(max(start, 0.0)), text))
    return normalized or None


def _build_prompt(question: str, candidates: list[str]) -> str:
    lines = [f"Question: {question}", "Options:"]
    for idx, candidate in enumerate(candidates):
        lines.append(f"({chr(ord('A') + idx)}) {candidate}")
    return "\n".join(lines)


def _log(log_path: Path, message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}\n"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line)
    print(line, end="", flush=True)


def _load_rows(manifest_path: Path, *, video_root: Path, limit: int | None, duration_groups: set[int] | None) -> list[SampleRow]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    out: list[SampleRow] = []
    for item in payload["rows"]:
        duration_group = item.get("duration_group")
        if duration_groups is not None and duration_group not in duration_groups:
            continue
        candidates = [str(option) for option in item["candidates"]]
        correct_index = int(item["correct_choice"])
        out.append(
            SampleRow(
                example_id=str(item["id"]),
                video_id=str(item["video_id"]),
                video_path=video_root / str(item["video_path"]),
                question=str(item["question"]),
                prompt=_build_prompt(str(item["question"]), candidates),
                candidates=candidates,
                answer=chr(ord("A") + correct_index),
                letters=[chr(ord("A") + idx) for idx in range(len(candidates))],
                duration=float(item["duration"]) if item.get("duration") is not None else None,
                duration_group=duration_group,
                question_category=item.get("question_category"),
                subtitle_path=item.get("subtitle_path"),
                starting_timestamp_for_subtitles=float(item.get("starting_timestamp_for_subtitles", 0.0)),
            )
        )
        if limit is not None and len(out) >= limit:
            break
    return out


def _stable_shard_id(value: str, num_shards: int) -> int:
    digest = hashlib.md5(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % num_shards


def _load_completed_ids(rows_path: Path) -> set[str]:
    completed: set[str] = set()
    if not rows_path.exists():
        return completed
    with rows_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                completed.add(str(json.loads(line)["example_id"]))
            except Exception:
                continue
    return completed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--subtitle-tar", type=Path, default=DEFAULT_SUBTITLE_TAR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model-name", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--duration-groups", default="600,3600")
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--n-retrieval", type=int, default=4)
    parser.add_argument("--n-refine", type=int, default=2)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--uniform-frame", type=int, default=128)
    parser.add_argument("--total-pixels", type=int, default=4096)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--skip-graph", action="store_true")
    parser.add_argument(
        "--graph-cache-root",
        type=Path,
        default=None,
        help="Optional Vgent-compatible graph cache root. Expects graphs under <root>/lvb_<fps:g>fps_<chunk_size>/",
    )
    parser.add_argument("--output-name", default="vgent_sample")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--keep-raw-video", action="store_true")
    parser.add_argument("--max-chunk-scale", type=int, default=8)
    parser.add_argument("--decode-size", type=int, default=336)
    parser.add_argument("--decode-batch-size", type=int, default=128)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--shard-by-video",
        action="store_true",
        help="Assign all questions from the same video to the same shard.",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", str(args.cpu_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.cpu_threads))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        import torch

        torch.set_num_threads(args.cpu_threads)
        torch.set_num_interop_threads(max(1, min(args.cpu_threads, 2)))
    except Exception:
        pass

    duration_groups = {int(part) for part in args.duration_groups.split(",") if part.strip()}
    rows = _load_rows(
        args.manifest,
        video_root=args.video_root,
        limit=args.limit,
        duration_groups=duration_groups,
    )
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("--shard-index must be in [0, num_shards)")
    if args.num_shards > 1:
        if args.shard_by_video:
            rows = [
                row
                for row in rows
                if _stable_shard_id(row.video_id, args.num_shards) == args.shard_index
            ]
        else:
            rows = [
                row
                for idx, row in enumerate(rows)
                if idx % args.num_shards == args.shard_index
            ]
    output_dir = args.output_root / args.output_name
    rows_path = output_dir / "rows.jsonl"
    log_path = output_dir / "run.log"
    graph_base = args.graph_cache_root if args.graph_cache_root is not None else output_dir / "graphs"
    graph_dir = graph_base / f"lvb_{args.fps:g}fps_{args.chunk_size}"
    output_dir.mkdir(parents=True, exist_ok=True)
    graph_dir.mkdir(parents=True, exist_ok=True)
    completed_ids = _load_completed_ids(rows_path) if args.resume else set()
    if rows_path.exists() and not args.resume:
        rows_path.unlink()
    if log_path.exists() and not args.resume:
        log_path.unlink()

    vgent_args = SimpleNamespace(
        model_name=args.model_name,
        output_path=str(output_dir),
        chunk_size=args.chunk_size,
        task="lvb",
        data_path="",
        uniform_frame=args.uniform_frame,
        n_retrieval=args.n_retrieval,
        n_refine=args.n_refine,
        fps=args.fps,
        graph_path=str(graph_base),
        total_pixels=args.total_pixels,
        duration=["long", "medium", "short"],
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        keep_raw_video=args.keep_raw_video,
        max_chunk_scale=args.max_chunk_scale,
        cpu_threads=args.cpu_threads,
        decode_size=args.decode_size,
        decode_batch_size=args.decode_batch_size,
    )

    if completed_ids:
        rows = [row for row in rows if row.example_id not in completed_ids]
    _log(
        log_path,
        "start total={} model={} chunk_size={} n_retrieval={} n_refine={} skip_graph={} shard={}/{} shard_by_video={} resume_completed={}".format(
            len(rows),
            args.model_name,
            args.chunk_size,
            args.n_retrieval,
            args.n_refine,
            args.skip_graph,
            args.shard_index,
            args.num_shards,
            args.shard_by_video,
            len(completed_ids),
        ),
    )
    vgent = Vgent(vgent_args)
    results: list[dict] = []
    last_video_path: Path | None = None
    last_loaded_video = None
    try:
        for index, row in enumerate(rows, start=1):
            _log(log_path, f"item_start index={index}/{len(rows)} example_id={row.example_id} video={row.video_id}")
            if last_video_path == row.video_path and last_loaded_video is not None:
                raw_video, fps, video_inputs, size_list = last_loaded_video
                _log(log_path, f"video_reused frames={len(video_inputs[0])} fps={fps}")
            else:
                raw_video, _, _, _, fps, video_inputs, size_list = vgent.load_video(str(row.video_path), vgent_args)
                if type(video_inputs) is not list:
                    video_inputs = [video_inputs]
                last_video_path = row.video_path
                last_loaded_video = (raw_video, fps, video_inputs, size_list)
                _log(log_path, f"video_loaded frames={len(video_inputs[0])} fps={fps}")
            if type(video_inputs) is not list:
                video_inputs = [video_inputs]
            subtitles = _normalize_subtitles_for_vgent(
                subtitle_path=row.subtitle_path,
                subtitle_tar=args.subtitle_tar,
                starting_timestamp_for_subtitles=row.starting_timestamp_for_subtitles,
                duration=row.duration,
            )
            threshold = vgent_args.chunk_size * vgent_args.n_retrieval
            if args.skip_graph or len(video_inputs[0]) < threshold:
                video_graph, entity_graph = (None, None)
                _log(log_path, f"graph_skipped threshold={threshold} reason={'flag' if args.skip_graph else 'short_video'}")
            else:
                graph_key = row.video_path.stem.split(".")[0]
                graph_cache_path = graph_dir / f"{graph_key}.pkl"
                if graph_cache_path.exists():
                    _log(log_path, f"graph_cache_hit path={graph_cache_path.name}")
                    saved_graph = pickle.loads(graph_cache_path.read_bytes())
                    video_graph, entity_graph = saved_graph["video_graph"], saved_graph["entity_graph"]
                else:
                    _log(log_path, "graph_start")
                    video_graph, entity_graph = vgent.construct_graph(video_inputs, subtitles)
                    graph_cache_path.write_bytes(pickle.dumps({"video_graph": video_graph, "entity_graph": entity_graph}))
                    _log(log_path, f"graph_cached path={graph_cache_path.name}")
                _log(log_path, f"graph_done nodes={len(video_graph.nodes())}")
            _log(log_path, "extract_keywords_start")
            query_list, llm_info = vgent.extract_keywords(row.question, row.candidates, video_inputs)
            _log(log_path, f"extract_keywords_done keywords={len(query_list)}")
            _log(log_path, "retrieve_start")
            retrieved_node_list = vgent.retrieve_nodes(row.question, query_list, video_inputs, row.candidates, video_graph, entity_graph, subtitles, llm_info)
            _log(log_path, f"retrieve_done nodes={len(retrieved_node_list.get('nodes', []))}")
            _log(log_path, "refine_start")
            refined_node_list, sql_check, check_result = vgent.refine_nodes(retrieved_node_list, row.question, llm_info, row.candidates, video_inputs, subtitles, size_list)
            _log(log_path, f"refine_done nodes={len(refined_node_list.get('nodes', []))}")
            _log(log_path, "aggregate_start")
            pred = vgent.aggregate_nodes(
                refined_node_list,
                llm_info,
                video_inputs,
                raw_video,
                size_list,
                subtitles,
                row.prompt,
                {
                    "question": row.question,
                    "candidates": row.candidates,
                    "letters": row.letters,
                },
                video_graph,
                sql_check,
                check_result,
                fps,
            )
            _log(log_path, f"aggregate_done pred={pred} gold={row.answer}")
            result = {
                "example_id": row.example_id,
                "video_id": row.video_id,
                "question": row.question,
                "candidates": row.candidates,
                "pred": pred,
                "answer": row.answer,
                "correct": pred == row.answer,
                "duration_group": row.duration_group,
                "question_category": row.question_category,
                "retrieved_nodes": refined_node_list.get("nodes", []),
                "llm_info": llm_info,
            }
            results.append(result)
            _append_jsonl(rows_path, result)
            _log(log_path, f"item_done index={index}/{len(rows)} correct={result['correct']}")
    finally:
        del vgent

    summary = {
        "model_name": args.model_name,
        "questions": len(results),
        "correct": sum(1 for row in results if row["correct"]),
        "accuracy": (sum(1 for row in results if row["correct"]) / len(results)) if results else None,
        "chunk_size": args.chunk_size,
        "n_retrieval": args.n_retrieval,
        "n_refine": args.n_refine,
        "fps": args.fps,
    }
    _write_json(output_dir / "final_summary.json", summary)
    _log(log_path, f"finished questions={summary['questions']} accuracy={summary['accuracy']}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
