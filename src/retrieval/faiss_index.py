from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch


def require_faiss() -> Any:
    try:
        import faiss  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("FAISS backend requested but faiss is not installed. Install faiss-cpu or faiss-gpu.") from exc
    return faiss


def tensor_to_float32_numpy(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().float().cpu().contiguous().numpy()
    return array.astype(np.float32, copy=False)


def build_ip_index(embeddings: torch.Tensor) -> Any:
    faiss = require_faiss()
    xb = tensor_to_float32_numpy(embeddings)
    if xb.ndim != 2:
        raise ValueError(f"Expected 2D embedding matrix, got shape {xb.shape}")
    index = faiss.IndexFlatIP(int(xb.shape[1]))
    index.add(xb)
    return index


def write_ip_index(path: Path, embeddings: torch.Tensor) -> Any:
    faiss = require_faiss()
    index = build_ip_index(embeddings)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    faiss.write_index(index, str(tmp_path))
    tmp_path.replace(path)
    return index


def read_ip_index(path: Path) -> Any:
    faiss = require_faiss()
    return faiss.read_index(str(path))


def load_or_build_ip_index(path: Path, embeddings: torch.Tensor) -> Any:
    if path.exists():
        return read_ip_index(path)
    return write_ip_index(path, embeddings)


def search_ip_index(index: Any, query_embedding: torch.Tensor, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    if top_k <= 0:
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int64)
    xq = tensor_to_float32_numpy(query_embedding).reshape(1, -1)
    if xq.shape[1] != index.d:
        raise ValueError(f"Query dimension {xq.shape[1]} does not match FAISS index dimension {index.d}")
    k = min(int(top_k), int(index.ntotal))
    scores, indices = index.search(xq, k)
    valid = indices[0] >= 0
    return scores[0][valid].astype(np.float32, copy=False), indices[0][valid].astype(np.int64, copy=False)
