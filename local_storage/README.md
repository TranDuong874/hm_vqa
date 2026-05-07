# Local Storage

This directory is the single local source of truth for reusable, non-result
artifacts produced by ingestion and indexing jobs.

`local_storage/flat_files/` contains file-backed storage only: benchmark
manifests, copied video subsets, OpenCLIP frame features, derived L2/L3 segment
caches, and Vgent description/graph caches.

It is intentionally **not** a vector database. Future FAISS or other vector DB
indexes should live under a separate top-level directory such as
`local_storage/vector_indexes/` so flat-file memory and vector-search indexes do
not get mixed.

Use `results/` only for experiment outputs: predictions, metrics, live matrices,
inspection exports, and reports.

New datasets should use the staged layout exposed by `hm_vqa.storage`:

```text
local_storage/flat_files/<dataset>/<split>/videos/
local_storage/flat_files/<dataset>/<split>/memory/<memory_policy>/
```

The existing LVB and HD-EPIC roots are kept in their historical names for
compatibility with already generated runs.
