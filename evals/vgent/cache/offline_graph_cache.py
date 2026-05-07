from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
import json
import pickle
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import torch
from tqdm import tqdm

from hm_vqa.storage import FLAT_STORAGE_ROOT, LVB_FULL_MANIFEST, LVB_FULL_VIDEO_ROOT

def _repo_root() -> Path:
    path = Path(__file__).resolve()
    for parent in path.parents:
        if (parent / "src").exists() and (parent / "evals").exists():
            return parent
    return path.parents[3]


REPO_ROOT = _repo_root()
VGENT_ROOT = REPO_ROOT / "thirdparty" / "Vgent"
if str(VGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(VGENT_ROOT))

from models.utils import resize_video  # noqa: E402
from answering.factory import build_answerer  # noqa: E402
from answering.qwen_vl import AnswerConfig  # noqa: E402
from segmentation.video import probe_video_sampling, sample_video_selected_indices  # noqa: E402
from utils.prompts import GRAPH_PROMPT  # noqa: E402
from utils.retrieval import compute_text_similarity  # noqa: E402


DEFAULT_LVB_MANIFEST = LVB_FULL_MANIFEST
DEFAULT_LVB_VIDEO_ROOT = LVB_FULL_VIDEO_ROOT
DEFAULT_HD_VIDEO_ROOT = Path("/home/tranduong/dev/dataset/HD-EPIC")
DEFAULT_OUTPUT_ROOT = FLAT_STORAGE_ROOT / "vgent" / "offline_graph_cache"


@dataclass(slots=True)
class VideoItem:
    dataset: str
    video_id: str
    video_path: Path
    duration_sec: float | None = None


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.@-]+", "_", value).strip("_") or "video"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in vars(args).items():
        out[key] = str(value) if isinstance(value, Path) else value
    return out


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _load_lvb_videos(manifest: Path, video_root: Path) -> list[VideoItem]:
    payload = _read_json(manifest)
    rows = payload.get("rows", payload) if isinstance(payload, dict) else payload
    videos: dict[str, VideoItem] = {}
    for row in rows:
        video_id = str(row.get("video_id") or Path(str(row["video_path"])).stem)
        if video_id in videos:
            continue
        videos[video_id] = VideoItem(
            dataset="lvb",
            video_id=video_id,
            video_path=video_root / str(row["video_path"]),
            duration_sec=float(row["duration"]) if row.get("duration") is not None else None,
        )
    return list(videos.values())


def _load_hd_epic_videos(video_root: Path, participant: str, manifest: Path | None) -> list[VideoItem]:
    if manifest is not None:
        payload = _read_json(manifest)
        rows = payload.get("videos", payload.get("rows", payload)) if isinstance(payload, dict) else payload
        video_ids = []
        durations: dict[str, float] = {}
        for row in rows:
            video_id = str(row["video_id"])
            if video_id not in video_ids:
                video_ids.append(video_id)
            if row.get("duration_sec") is not None:
                durations[video_id] = float(row["duration_sec"])
    else:
        video_ids = [path.stem for path in sorted((video_root / participant).glob("*.mp4"))]
        durations = {}

    items: list[VideoItem] = []
    for video_id in video_ids:
        part = video_id.split("-")[0]
        participant_filter = participant.strip().upper() if participant else ""
        if participant_filter not in {"", "ALL"} and part != participant:
            continue
        items.append(
            VideoItem(
                dataset="hd_epic",
                video_id=video_id,
                video_path=video_root / part / f"{video_id}.mp4",
                duration_sec=durations.get(video_id),
            )
        )
    return items


def _parse_graph_json(raw_text: str) -> tuple[list[str], list[str], list[str], dict[str, Any] | None]:
    cleaned = raw_text.replace("```json", "").replace("```", "").strip()
    try:
        info = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            return _parse_partial_graph_json(cleaned)
        try:
            info = json.loads(match.group(0))
        except json.JSONDecodeError:
            return _parse_partial_graph_json(cleaned)

    entities: list[str] = []
    for entity in info.get("entities", []):
        if isinstance(entity, dict) and "entity name" in entity and "description" in entity:
            entities.append(f"{entity['entity name']}, {entity['description']}")
        elif isinstance(entity, str) and entity.strip():
            entities.append(entity.strip())
    actions: list[str] = []
    for action in info.get("actions", []):
        if isinstance(action, str) and action.strip():
            actions.append(action.strip())
            continue
        if not isinstance(action, dict) or "entity name" not in action:
            continue
        if "action description" in action:
            actions.append(f"{action['entity name']}, {action['action description']}")
        else:
            other_values = [str(value) for key, value in action.items() if key != "entity name"]
            if other_values:
                actions.append(f"{action['entity name']}, {'; '.join(other_values)}")
    scenes: list[str] = []
    for scene in info.get("scenes", []):
        if isinstance(scene, dict) and "location" in scene:
            scenes.append(str(scene["location"]))
        elif isinstance(scene, str) and scene.strip():
            scenes.append(scene.strip())
    return entities, actions, scenes, info


def _parse_partial_graph_json(raw_text: str) -> tuple[list[str], list[str], list[str], dict[str, Any] | None]:
    def extract_list(key: str) -> list[str]:
        match = re.search(rf'"{re.escape(key)}"\s*:\s*\[(.*?)(?=,\s*"[^"]+"\s*:|\}}\s*$|$)', raw_text, flags=re.DOTALL)
        if not match:
            return []
        values = re.findall(r'"([^"\\]*(?:\\.[^"\\]*)*)"', match.group(1))
        return [value.encode("utf-8").decode("unicode_escape").strip() for value in values if value.strip()]

    entities = extract_list("entities")
    actions = extract_list("actions")
    scenes = extract_list("scenes")
    parsed = {"entities": entities, "actions": actions, "scenes": scenes, "partial": True}
    return entities, actions, scenes, parsed


def _load_existing_chunks(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    chunks: dict[int, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("status") == "ok" and row.get("raw_text") and not (row.get("entities") or row.get("actions") or row.get("scenes")):
            entities, actions, scenes, parsed = _parse_graph_json(str(row.get("raw_text", "")))
            row["entities"] = entities
            row["actions"] = actions
            row["scenes"] = scenes
            row["parsed"] = parsed
        chunks[int(row["chunk_id"])] = row
    return chunks


def _build_graph_from_chunks(chunks: list[dict[str, Any]], *, embedding_model: Any, embedding_tokenizer: Any) -> tuple[nx.DiGraph, dict[str, set[int]]]:
    video_graph = nx.DiGraph()
    entity_graph: dict[str, set[int]] = {}
    for chunk in chunks:
        idx = int(chunk["chunk_id"])
        entities = list(chunk.get("entities", []))
        actions = list(chunk.get("actions", []))
        scenes = list(chunk.get("scenes", []))
        subtitles = chunk.get("subtitles")
        video_graph.add_node(idx, actions=actions, scenes=scenes, entities=entities, subtitles=subtitles)
        for entity in entities + actions + scenes:
            entity_name = str(entity).split(",")[0].lower().strip()
            if not entity_name:
                continue
            if len(entity_graph) == 0:
                entity_graph.setdefault(entity_name, set()).add(idx)
                continue
            keys = list(entity_graph.keys())
            entity_sim = compute_text_similarity([str(entity)], keys, embedding_model, embedding_tokenizer, return_all=True)
            max_sim_idx = max(range(len(entity_sim[0])), key=lambda i: float(entity_sim[0][i]))
            max_sim = float(entity_sim[0][max_sim_idx])
            if max_sim > 0.7:
                most_similar_entity = keys[max_sim_idx]
                entity_graph[most_similar_entity].add(idx)
                video_graph.add_edges_from((idx, i, {"label": most_similar_entity}) for i in entity_graph[most_similar_entity])
            else:
                entity_graph.setdefault(entity_name, set()).add(idx)
    return video_graph, entity_graph


def _node_text_from_chunk(chunk: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("entities", "actions", "scenes", "subtitles"):
        value = chunk.get(key)
        if isinstance(value, list):
            parts.extend(str(item) for item in value if str(item).strip())
        elif value:
            parts.append(str(value))
    return "; ".join(parts)


def _encode_texts(
    texts: list[str],
    *,
    embedding_model: Any,
    embedding_tokenizer: Any,
    batch_size: int = 64,
) -> torch.Tensor:
    if not texts:
        return torch.empty((0, 0), dtype=torch.float16)
    embeddings: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = embedding_tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            output = embedding_model(**encoded)
            batch_embeddings = torch.nn.functional.normalize(output[0][:, 0], p=2, dim=1)
            embeddings.append(batch_embeddings.cpu())
    return torch.cat(embeddings, dim=0).to(torch.float16)


def _write_embedding_cache(
    video_dir: Path,
    chunks: list[dict[str, Any]],
    entity_graph: dict[str, set[int]],
    *,
    embedding_model: Any,
    embedding_tokenizer: Any,
) -> None:
    node_ids: list[int] = []
    node_texts: list[str] = []
    for chunk in chunks:
        text = _node_text_from_chunk(chunk)
        if not text:
            continue
        node_ids.append(int(chunk["chunk_id"]))
        node_texts.append(text)

    entity_keys = sorted(str(key) for key in entity_graph.keys())
    node_embeddings = _encode_texts(
        node_texts,
        embedding_model=embedding_model,
        embedding_tokenizer=embedding_tokenizer,
    )
    entity_embeddings = _encode_texts(
        entity_keys,
        embedding_model=embedding_model,
        embedding_tokenizer=embedding_tokenizer,
    )
    embedding_path = video_dir / "bge_embeddings.pt"
    torch.save(
        {
            "embedding_model": "BAAI/bge-large-en-v1.5",
            "node_ids": node_ids,
            "node_texts": node_texts,
            "node_embeddings": node_embeddings,
            "entity_keys": entity_keys,
            "entity_embeddings": entity_embeddings,
        },
        embedding_path,
    )
    _write_json(
        video_dir / "bge_embeddings_meta.json",
        {
            "embedding_model": "BAAI/bge-large-en-v1.5",
            "node_count": len(node_ids),
            "entity_count": len(entity_keys),
            "embedding_path": str(embedding_path),
        },
    )


def _load_embedding_model() -> tuple[Any, Any]:
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")
    model = AutoModel.from_pretrained("BAAI/bge-large-en-v1.5")
    model.eval()
    return model, tokenizer


def _chunk_ranges(sampled_count: int, chunk_size: int) -> list[tuple[int, int]]:
    return [(start, min(start + chunk_size, sampled_count)) for start in range(0, sampled_count, chunk_size)]


def build_video_cache(
    item: VideoItem,
    *,
    output_root: Path,
    answerer: Any,
    sample_fps: float,
    chunk_size: int,
    image_max_size: int,
    max_new_tokens: int,
    max_chunks_per_video: int | None,
    chunk_workers: int,
    decode_workers: int,
    prefetch_chunks: int,
    build_graph: bool,
    rebuild: bool,
    embedding_model: Any | None = None,
    embedding_tokenizer: Any | None = None,
) -> Path:
    video_dir = output_root / item.dataset / _safe_name(item.video_id)
    chunks_path = video_dir / "chunks.jsonl"
    meta_path = video_dir / "meta.json"
    graph_path = video_dir / "graph.pkl"
    vgent_graph_path = output_root / "graphs" / f"{item.dataset}_{sample_fps:g}fps_{chunk_size}" / f"{_safe_name(item.video_id)}.pkl"
    if rebuild:
        for path in (chunks_path, graph_path, vgent_graph_path):
            if path.exists():
                path.unlink()

    sampling = probe_video_sampling(item.video_path, sample_fps)
    ranges = _chunk_ranges(sampling.sampled_count, chunk_size)
    if max_chunks_per_video is not None:
        ranges = ranges[: max(0, int(max_chunks_per_video))]

    existing = _load_existing_chunks(chunks_path)
    _write_json(
        meta_path,
        {
            "dataset": item.dataset,
            "video_id": item.video_id,
            "video_path": str(item.video_path),
            "duration_sec": sampling.duration_sec,
            "native_fps": sampling.native_fps,
            "sample_fps": sample_fps,
            "sampled_count": sampling.sampled_count,
            "chunk_size": chunk_size,
            "chunks_planned": len(ranges),
            "image_max_size": image_max_size,
            "model_id": answerer.config.model_id,
        },
    )

    def decode_chunk(chunk_id: int, start: int, end: int) -> dict[str, Any]:
        target_indices = list(range(start, end))
        started = time.perf_counter()
        try:
            frames, times, _ = sample_video_selected_indices(
                item.video_path,
                sample_fps,
                target_indices=target_indices,
                image_max_size=image_max_size,
            )
            return {
                "chunk_id": chunk_id,
                "start_sample_index": start,
                "end_sample_index": end,
                "started": started,
                "frames": frames,
                "times": times,
                "decode_sec": round(time.perf_counter() - started, 3),
            }
        except Exception as exc:
            return {
                "chunk_id": chunk_id,
                "start_sample_index": start,
                "end_sample_index": end,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_sec": round(time.perf_counter() - started, 3),
            }

    def generate_chunk(decoded: dict[str, Any]) -> dict[str, Any]:
        if decoded.get("status") == "error":
            return decoded
        started = float(decoded["started"])
        frames = decoded["frames"]
        times = decoded["times"]
        try:
            generation = answerer.generate_text_from_frames(
                frames=frames,
                prompt=GRAPH_PROMPT,
                max_new_tokens=max_new_tokens,
            )
            entities, actions, scenes, parsed = _parse_graph_json(generation.raw_text)
            row = {
                "chunk_id": decoded["chunk_id"],
                "start_sample_index": decoded["start_sample_index"],
                "end_sample_index": decoded["end_sample_index"],
                "start_time_sec": float(times[0]) if times else None,
                "end_time_sec": float(times[-1]) if times else None,
                "status": "ok",
                "entities": entities,
                "actions": actions,
                "scenes": scenes,
                "parsed": parsed,
                "raw_text": generation.raw_text,
                "generation_sec": generation.generation_sec,
                "decode_sec": decoded.get("decode_sec"),
                "elapsed_sec": round(time.perf_counter() - started, 3),
            }
        except Exception as exc:
            row = {
                "chunk_id": decoded["chunk_id"],
                "start_sample_index": decoded["start_sample_index"],
                "end_sample_index": decoded["end_sample_index"],
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "decode_sec": decoded.get("decode_sec"),
                "elapsed_sec": round(time.perf_counter() - started, 3),
            }
        return row

    def process_chunk(chunk_id: int, start: int, end: int) -> dict[str, Any]:
        return generate_chunk(decode_chunk(chunk_id, start, end))

    pending = [
        (chunk_id, start, end)
        for chunk_id, (start, end) in enumerate(ranges)
        if not (chunk_id in existing and existing[chunk_id].get("status") == "ok")
    ]
    if chunk_workers <= 1 or len(pending) <= 1:
        for chunk_id, start, end in tqdm(pending, desc=f"{item.video_id} chunks", leave=False):
            row = process_chunk(chunk_id, start, end)
            _append_jsonl(chunks_path, row)
            existing[chunk_id] = row
    elif decode_workers > 0:
        pending_iter = iter(pending)
        max_prefetch = max(int(prefetch_chunks), int(decode_workers), int(chunk_workers))
        decode_futures: dict[Any, tuple[int, int, int]] = {}
        api_futures: dict[Any, int] = {}

        def submit_decode(executor: ThreadPoolExecutor) -> bool:
            try:
                chunk_id, start, end = next(pending_iter)
            except StopIteration:
                return False
            decode_futures[executor.submit(decode_chunk, chunk_id, start, end)] = (chunk_id, start, end)
            return True

        with ThreadPoolExecutor(max_workers=decode_workers) as decode_executor, ThreadPoolExecutor(max_workers=chunk_workers) as api_executor:
            for _ in range(min(max_prefetch, len(pending))):
                submit_decode(decode_executor)
            with tqdm(total=len(pending), desc=f"{item.video_id} chunks", leave=False) as progress:
                while decode_futures or api_futures:
                    completed, _ = wait(set(decode_futures) | set(api_futures), return_when=FIRST_COMPLETED)
                    for future in completed:
                        if future in decode_futures:
                            decode_futures.pop(future)
                            decoded = future.result()
                            api_futures[api_executor.submit(generate_chunk, decoded)] = int(decoded["chunk_id"])
                            while len(decode_futures) + len(api_futures) < max_prefetch:
                                if not submit_decode(decode_executor):
                                    break
                        else:
                            api_futures.pop(future)
                            row = future.result()
                            _append_jsonl(chunks_path, row)
                            existing[int(row["chunk_id"])] = row
                            progress.update(1)
    else:
        with ThreadPoolExecutor(max_workers=chunk_workers) as executor:
            futures = {executor.submit(process_chunk, chunk_id, start, end): chunk_id for chunk_id, start, end in pending}
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"{item.video_id} chunks", leave=False):
                row = future.result()
                _append_jsonl(chunks_path, row)
                existing[int(row["chunk_id"])] = row

    ok_chunks = [existing[idx] for idx, _ in enumerate(ranges) if idx in existing and existing[idx].get("status") == "ok"]
    if not build_graph:
        _write_json(
            video_dir / "summary.json",
            {
                "video_id": item.video_id,
                "chunks_ok": len(ok_chunks),
                "chunks_total": len(ranges),
                "graph_nodes": None,
                "graph_edges": None,
                "entity_keys": None,
                "graph_path": str(graph_path),
                "vgent_compatible_graph_path": str(vgent_graph_path),
                "embedding_cache_path": str(video_dir / "bge_embeddings.pt"),
                "embedding_cache_meta_path": str(video_dir / "bge_embeddings_meta.json"),
                "graph_build": "skipped",
            },
        )
        return video_dir / "summary.json"

    if embedding_model is None or embedding_tokenizer is None:
        embedding_model, embedding_tokenizer = _load_embedding_model()
    video_graph, entity_graph = _build_graph_from_chunks(
        ok_chunks,
        embedding_model=embedding_model,
        embedding_tokenizer=embedding_tokenizer,
    )
    _write_embedding_cache(
        video_dir,
        ok_chunks,
        entity_graph,
        embedding_model=embedding_model,
        embedding_tokenizer=embedding_tokenizer,
    )
    payload = {
        "video_graph": video_graph,
        "entity_graph": entity_graph,
        "chunk_cache": str(chunks_path),
        "meta": json.loads(meta_path.read_text(encoding="utf-8")),
    }
    graph_path.write_bytes(pickle.dumps(payload))
    vgent_graph_path.parent.mkdir(parents=True, exist_ok=True)
    vgent_graph_path.write_bytes(pickle.dumps({"video_graph": video_graph, "entity_graph": entity_graph}))
    _write_json(
        video_dir / "summary.json",
        {
            "video_id": item.video_id,
            "chunks_ok": len(ok_chunks),
            "chunks_total": len(ranges),
            "graph_nodes": len(video_graph.nodes),
            "graph_edges": len(video_graph.edges),
            "entity_keys": len(entity_graph),
            "graph_path": str(graph_path),
            "vgent_compatible_graph_path": str(vgent_graph_path),
            "embedding_cache_path": str(video_dir / "bge_embeddings.pt"),
            "embedding_cache_meta_path": str(video_dir / "bge_embeddings_meta.json"),
        },
    )
    return video_dir / "summary.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["lvb", "hd_epic"], required=True)
    parser.add_argument("--lvb-manifest", type=Path, default=DEFAULT_LVB_MANIFEST)
    parser.add_argument("--lvb-video-root", type=Path, default=DEFAULT_LVB_VIDEO_ROOT)
    parser.add_argument("--hd-video-root", type=Path, default=DEFAULT_HD_VIDEO_ROOT)
    parser.add_argument("--hd-manifest", type=Path, default=None)
    parser.add_argument("--hd-participant", default="P01")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--backend", choices=["local", "api"], default="local")
    parser.add_argument("--api-base-url", default=AnswerConfig.api_base_url)
    parser.add_argument("--api-key-env-var", default=AnswerConfig.api_key_env_var)
    parser.add_argument("--api-timeout-sec", type=float, default=180.0)
    parser.add_argument("--api-requests-per-minute", type=int, default=2000)
    parser.add_argument("--api-tokens-per-minute", type=int, default=20000000)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--image-max-size", type=int, default=336)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--chunk-workers", type=int, default=1)
    parser.add_argument("--decode-workers", type=int, default=0)
    parser.add_argument("--prefetch-chunks", type=int, default=32)
    parser.add_argument("--limit-videos", type=int, default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-chunks-per-video", type=int, default=None)
    parser.add_argument("--skip-graph-build", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    if args.dataset == "lvb":
        videos = _load_lvb_videos(args.lvb_manifest, args.lvb_video_root)
    else:
        videos = _load_hd_epic_videos(args.hd_video_root, args.hd_participant, args.hd_manifest)
    videos = [video for video in videos if video.video_path.exists()]
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")
    if args.num_shards > 1:
        videos = [video for index, video in enumerate(videos) if index % args.num_shards == args.shard_index]
    if args.limit_videos is not None:
        videos = videos[: max(0, int(args.limit_videos))]
    if not videos:
        raise RuntimeError("No videos found for offline Vgent graph cache build.")

    answerer = build_answerer(
        AnswerConfig(
            model_id=args.model_id,
            backend=args.backend,
            load_in_4bit=args.load_in_4bit,
            image_max_size=args.image_max_size,
            max_new_tokens=args.max_new_tokens,
            api_base_url=args.api_base_url,
            api_key_env_var=args.api_key_env_var,
            api_timeout_sec=args.api_timeout_sec,
            api_requests_per_minute=args.api_requests_per_minute,
            api_tokens_per_minute=args.api_tokens_per_minute,
        )
    )
    run_summary = {
        "dataset": args.dataset,
        "videos": len(videos),
        "sample_fps": args.sample_fps,
        "chunk_size": args.chunk_size,
        "image_max_size": args.image_max_size,
        "model_id": args.model_id,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "items": [],
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_root / f"{args.dataset}_latest_config.json", _jsonable_args(args))
    try:
        if args.skip_graph_build:
            embedding_model, embedding_tokenizer = None, None
        else:
            embedding_model, embedding_tokenizer = _load_embedding_model()
        for video in tqdm(videos, desc=f"{args.dataset} videos"):
            summary_path = build_video_cache(
                video,
                output_root=args.output_root,
                answerer=answerer,
                sample_fps=args.sample_fps,
                chunk_size=args.chunk_size,
                image_max_size=args.image_max_size,
                max_new_tokens=args.max_new_tokens,
                max_chunks_per_video=args.max_chunks_per_video,
                chunk_workers=max(1, int(args.chunk_workers)),
                decode_workers=max(0, int(args.decode_workers)),
                prefetch_chunks=max(1, int(args.prefetch_chunks)),
                build_graph=not args.skip_graph_build,
                rebuild=args.rebuild,
                embedding_model=embedding_model,
                embedding_tokenizer=embedding_tokenizer,
            )
            run_summary["items"].append(json.loads(summary_path.read_text(encoding="utf-8")))
            _write_json(args.output_root / f"{args.dataset}_run_summary.json", run_summary)
    finally:
        answerer.unload()
    run_summary["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _write_json(args.output_root / f"{args.dataset}_run_summary.json", run_summary)
    print(json.dumps(run_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
