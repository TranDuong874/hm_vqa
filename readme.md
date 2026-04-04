# HM-VQA

Evaluation-oriented long-video QA pipeline with:
- dense video indexing
- hierarchical retrieval
- frame-based final answering

## Structure

### `src/`
- `ingestion/`
  - model wrappers for feature extraction
  - current encoders:
    - OpenCLIP
    - DINOv3
    - X-CLIP
- `analysis/`
  - video sampling
  - signal computation
  - segmentation
  - plots and clip export helpers
- `retrieval/`
  - index dataclasses
  - score fusion
  - segment retrieval
  - evidence frame selection
- `answering/`
  - final answer-time model wrappers
  - current answerer:
    - Qwen3-VL
- `hm_vqa_pipeline.py`
  - dataset-agnostic orchestration for:
    - index building
    - retrieval
    - evidence packaging

### `evals/video-mme/`
- `dataloader.py`
  - Video-MME manifest loader
- `evaluate.py`
  - runs the HM-VQA method on Video-MME
- `baseline.py`
  - runs the direct Qwen3-VL baseline on Video-MME
- `sample_dev.py`
  - creates a local dev subset manifest

## Current benchmark setup

The current local benchmark uses:
- manifest: `evals/video-mme/video_mme_dev50.json`
- videos: `dataset/Video-MME/*.mp4`

The Video-MME runners can optionally download missing YouTube videos only if you enable that in the script. By default:
- `ALLOW_DOWNLOAD = False`

## Environment

This repo currently expects:
- a Python environment at `.venv`
- `PYTHONPATH=src`
- optional Hugging Face token in `.env`

Example `.env`:

```bash
HF_TOKEN=...
```

## Run

### HM-VQA on Video-MME

```bash
PYTHONPATH=src .venv/bin/python evals/video-mme/evaluate.py
```

### Direct Qwen3-VL baseline on Video-MME

```bash
PYTHONPATH=src .venv/bin/python evals/video-mme/baseline.py
```

## Notes

- `hm_vqa_pipeline.py` should stay dataset-agnostic.
- Dataset-specific logic belongs in `evals/<benchmark>/`.
- Future final-answer models should go in `src/answering/`.
- Future retrieval variants should go in `src/retrieval/`.
