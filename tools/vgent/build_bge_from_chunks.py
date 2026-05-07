from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import networkx as nx
import torch

from evals.vgent.cache.offline_graph_cache import (
    _encode_texts,
    _load_embedding_model,
    _load_existing_chunks,
    _write_embedding_cache,
    _write_json,
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _entity_name(value: object) -> str:
    return str(value).split(",")[0].lower().strip()


def _batched_greedy_graph_from_chunks(chunks: list[dict], *, embedding_model, embedding_tokenizer) -> tuple[nx.DiGraph, dict[str, set[int]]]:
    """Same greedy entity grouping as Vgent-style code, but batch BGE encoding.

    The original loop calls BGE once for each candidate entity against all prior
    keys. This keeps the same ordering and threshold but precomputes vectors.
    """
    video_graph = nx.DiGraph()
    raw_terms: list[str] = []
    key_terms: list[str] = []

    for chunk in chunks:
        terms = list(chunk.get("entities", [])) + list(chunk.get("actions", [])) + list(chunk.get("scenes", []))
        raw_terms.extend(str(term) for term in terms if str(term).strip())
        key_terms.extend(_entity_name(term) for term in terms if _entity_name(term))

    vocab = sorted(set(raw_terms) | set(key_terms))
    embeddings = _encode_texts(
        vocab,
        embedding_model=embedding_model,
        embedding_tokenizer=embedding_tokenizer,
        batch_size=128,
    ).float()
    vector_by_text = {text: embeddings[idx] for idx, text in enumerate(vocab)}

    entity_graph: dict[str, set[int]] = {}
    key_vectors: list[torch.Tensor] = []
    key_names: list[str] = []

    for chunk in chunks:
        idx = int(chunk["chunk_id"])
        entities = list(chunk.get("entities", []))
        actions = list(chunk.get("actions", []))
        scenes = list(chunk.get("scenes", []))
        subtitles = chunk.get("subtitles")
        video_graph.add_node(idx, actions=actions, scenes=scenes, entities=entities, subtitles=subtitles)

        for term in entities + actions + scenes:
            raw_text = str(term)
            entity_name = _entity_name(raw_text)
            if not entity_name:
                continue
            if not entity_graph:
                entity_graph.setdefault(entity_name, set()).add(idx)
                key_names.append(entity_name)
                key_vectors.append(vector_by_text[entity_name])
                continue

            query_vector = vector_by_text.get(raw_text)
            if query_vector is None:
                continue
            similarities = torch.mv(torch.stack(key_vectors), query_vector)
            max_sim, max_idx = torch.max(similarities, dim=0)
            if float(max_sim) > 0.7:
                most_similar_entity = key_names[int(max_idx)]
                entity_graph[most_similar_entity].add(idx)
                video_graph.add_edges_from((idx, node_id, {"label": most_similar_entity}) for node_id in entity_graph[most_similar_entity])
            else:
                entity_graph.setdefault(entity_name, set()).add(idx)
                key_names.append(entity_name)
                key_vectors.append(vector_by_text[entity_name])

    return video_graph, entity_graph


def build_one(video_dir: Path, *, output_root: Path, dataset: str, sample_fps: str, chunk_size: str, rebuild: bool, embedding_model, embedding_tokenizer) -> dict | None:
    chunks_path = video_dir / "chunks.jsonl"
    meta_path = video_dir / "meta.json"
    graph_path = video_dir / "graph.pkl"
    embedding_meta_path = video_dir / "bge_embeddings_meta.json"
    if not chunks_path.exists() or not meta_path.exists():
        return None
    if not rebuild and graph_path.exists() and embedding_meta_path.exists():
        return None

    meta = load_json(meta_path)
    planned = int(meta.get("chunks_planned") or 0)
    chunks_by_id = _load_existing_chunks(chunks_path)
    ok_chunks = [
        chunks_by_id[idx]
        for idx in range(planned)
        if idx in chunks_by_id and chunks_by_id[idx].get("status") == "ok"
    ]
    if planned <= 0 or len(ok_chunks) < planned:
        return None

    video_graph, entity_graph = _batched_greedy_graph_from_chunks(
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

    vgent_graph_path = output_root / "graphs" / f"{dataset}_{sample_fps}fps_{chunk_size}" / f"{video_dir.name}.pkl"
    payload = {
        "video_graph": video_graph,
        "entity_graph": entity_graph,
        "chunk_cache": str(chunks_path),
        "meta": meta,
    }
    graph_path.write_bytes(pickle.dumps(payload))
    vgent_graph_path.parent.mkdir(parents=True, exist_ok=True)
    vgent_graph_path.write_bytes(pickle.dumps({"video_graph": video_graph, "entity_graph": entity_graph}))

    summary = {
        "video_id": video_dir.name,
        "chunks_ok": len(ok_chunks),
        "chunks_total": planned,
        "graph_nodes": len(video_graph.nodes),
        "graph_edges": len(video_graph.edges),
        "entity_keys": len(entity_graph),
        "graph_path": str(graph_path),
        "vgent_compatible_graph_path": str(vgent_graph_path),
        "embedding_cache_path": str(video_dir / "bge_embeddings.pt"),
        "embedding_cache_meta_path": str(embedding_meta_path),
        "graph_build": "complete",
    }
    _write_json(video_dir / "summary.json", summary)
    return summary


def run_once(args, embedding_model, embedding_tokenizer) -> list[dict]:
    dataset_dir = args.output_root / args.dataset
    built: list[dict] = []
    if not dataset_dir.exists():
        return built
    video_dirs = sorted(path for path in dataset_dir.iterdir() if path.is_dir())
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")
    if args.num_shards > 1:
        video_dirs = [path for index, path in enumerate(video_dirs) if index % args.num_shards == args.shard_index]
    for video_dir in video_dirs:
        summary = build_one(
            video_dir,
            output_root=args.output_root,
            dataset=args.dataset,
            sample_fps=args.sample_fps,
            chunk_size=args.chunk_size,
            rebuild=args.rebuild,
            embedding_model=embedding_model,
            embedding_tokenizer=embedding_tokenizer,
        )
        if summary is not None:
            built.append(summary)
            print(json.dumps(summary, ensure_ascii=False), flush=True)
    return built


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset", default="lvb")
    parser.add_argument("--sample-fps", default="1")
    parser.add_argument("--chunk-size", default="64")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval-sec", type=float, default=30.0)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()

    embedding_model, embedding_tokenizer = _load_embedding_model()
    while True:
        built = run_once(args, embedding_model, embedding_tokenizer)
        if not args.watch:
            print(f"built={len(built)}")
            return
        if not built:
            print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} no completed videos waiting", flush=True)
        time.sleep(args.interval_sec)


if __name__ == "__main__":
    main()
