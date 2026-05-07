# HM-VQA: Hierarchical Memory for Video Question Answering

HM-VQA is a hierarchical retrieval system designed for long-video understanding. It uses multi-granular video memory to bridge raw video frames and Vision-Language Model (VLM) reasoning:

- **L3 (Coarse):** Fixed or adaptive coarse temporal segments for broad retrieval.
- **L2 (Refinement):** Local temporal windows, optionally reranked with ViCLIP inside the retrieved L3 context.
- **L1 (Fine-grained):** Frame-level evidence selection for final visual answering.

## Core Thesis Story

The strongest evidence for the hierarchical memory approach currently lies in **temporal grounding and retrieval**, especially on HD-EPIC. End-to-end MCQ accuracy on LongVideoBench is more mixed: direct L1 retrieval is a strong baseline, so LVB should be reported as retrieval-augmented QA rather than as a clean win for every hierarchy variant.

### Key Findings

1. **HD-EPIC localization:** L3 plus L2/ViCLIP retrieval gives much stronger temporal evidence coverage than L3-only retrieval.
2. **LongVideoBench QA:** Retrieval helps over pure uniform-frame VLM baselines, but direct L1 retrieval remains highly competitive.
3. **Resource profile:** HM-style embedding retrieval is lightweight compared with Vgent-style graph construction, which is dominated by VLM description generation and graph/cache construction.
4. **Evaluation direction:** MCQ accuracy alone hides whether a method found correct evidence or the VLM guessed correctly from weak evidence.

## Repository Layout

### `src/`
- [`src/answering`](/home/tranduong/dev/hm_vqa/src/answering): VLM answerers (primarily Qwen-VL based).
- [`src/ingestion`](/home/tranduong/dev/hm_vqa/src/ingestion): Frame/video ingestion and OpenCLIP/ViCLIP feature caching.
- [`src/retrieval`](/home/tranduong/dev/hm_vqa/src/retrieval): Hierarchical index construction and retrieval logic.
- [`src/segmentation`](/home/tranduong/dev/hm_vqa/src/segmentation): Adaptive boundary detection and multi-scale windowing.

### `evals/`
- [`evals/hd_epic`](/home/tranduong/dev/hm_vqa/evals/hd_epic): HD-EPIC runners (Strongest results for localization).
- [`evals/longvideobench`](/home/tranduong/dev/hm_vqa/evals/longvideobench): LongVideoBench ingestion and inference runners.
- [`evals/video_mme`](/home/tranduong/dev/hm_vqa/evals/video_mme): VideoMME ingestion and inference runners.
- [`tools`](/home/tranduong/dev/hm_vqa/tools): Live monitors and Vgent utilities.

For reproducible headless commands, see [`RUNBOOK.md`](/home/tranduong/dev/hm_vqa/RUNBOOK.md).

## Latest Results (Summary)

### HD-EPIC Localization (FGAL24)
| Method | HitAny @5 |
|---|---:|
| L1 direct | `0.4217` |
| L3 adaptive | `0.7333` |
| L3 + L2 adaptive ViCLIP | **`0.8850`** |

### LongVideoBench (Full 1337)
| Method | Acc |
|---|---:|
| Pure VLM 16f | `0.4637` |
| L1 direct 16f | `0.5228` |
| HM L3 Adaptive 16f | `0.5198` |

## Environment

```bash
# Example: Run retrieval ablation on LongVideoBench
PYTHONPATH=.:src .venv/bin/python -m evals.longvideobench.inference.run_retrieval_qa --help
```
