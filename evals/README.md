# Evaluation Runners

`evals/` contains dataset entrypoints. Keep benchmark loaders and metrics here;
shared pipeline contracts, storage paths, and policy definitions live under
`src/hm_vqa/`.

## Layout

- `evals/common/`
  - Shared OpenCLIP ingestion, retrieval-ablation, and VLM baseline runner code.
Each dataset should stay small and follow this shape when the category exists:

- `dataset.py`
  - Dataset loader that returns canonical examples/items.
- `paths.py`
  - Dataset-local default paths.
- `ingestion/`
  - Offline local-storage builders.
- `inference/`
  - Runners that produce final VLM answers.
- `retrieval/`
  - Retrieval-only benchmark runners, only when the dataset has retrieval-only
    evaluation such as HD-EPIC localization.

Current dataset packages:

- `evals/hd_epic/`
  - `dataset.py`, `paths.py`, `ingestion`, `inference`, `retrieval`.
- `evals/longvideobench/`
  - `dataset.py`, `paths.py`, `ingestion`, `inference`.
- `evals/video_mme/`
  - `dataset.py`, `ingestion`, `inference`.
- `evals/vgent/`
  - `cache`, `inference`, `retrieval`. The `cache` package name follows
    Vgent's terminology; generated artifacts should still be written under
    `local_storage/flat_files/`.

## Ablation Policy

Prefer config files for new runs:

- `configs/memory/`: offline memory construction.
- `configs/retrieval/`: evidence retrieval and frame selection.
- `configs/answer/`: VLM backend settings.
- `configs/datasets/`: dataset roots and output roots.

Existing runners keep CLI flags for compatibility:

- retrieval level: `--method l1`, `--method l3`, `--method l3_rerank_l2`
- segmentation: `--l3-segmentation`, `--l2-segmentation`
- window policy: `--l3-window-seconds`, `--l2-window-seconds`, stride flags
- reranker: `--l2-rerank-encoder openclip|viclip`
- evidence policy: `--l3-rerank-evidence-source`, `--max-frames`
- answer backend: `--backend local|api`

Do not add new queue `.sh` files. Put repeatable commands in `RUNBOOK.md` and
run them directly or through `tmux`.
