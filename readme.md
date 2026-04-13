# HM-VQA

Hierarchical-memory video QA repo with:
- reusable pipeline support in [`src/pipeline`](/home/tranduong/dev/hm_vqa/src/pipeline)
- dataset-specific evaluation code in [`evals/`](/home/tranduong/dev/hm_vqa/evals)
- an MCP tool server in [`mcp/hm_vqa_server`](/home/tranduong/dev/hm_vqa/mcp/hm_vqa_server)

## Current Canonical Pipeline

The main non-agentic pipeline is the HD-EPIC shortlist pipeline in:
- [`hd_epic_mcq_shortlist_joint.py`](/home/tranduong/dev/hm_vqa/src/pipeline/experiments/hd_epic_mcq_shortlist_joint.py)

Current canonical behavior for `ours`:
- `L3 -> L2` candidate harvest
- deterministic `k=5` shortlist
- split final stage:
  - candidate selection
  - answer from selected candidate only
- `L1` keyframe sampling inside shortlisted `L2` clips

Main entrypoint:
- [`run_mcq_shortlist_joint.py`](/home/tranduong/dev/hm_vqa/evals/hd_epic/run_mcq_shortlist_joint.py)

Comparison runners:
- [`run_batch.py`](/home/tranduong/dev/hm_vqa/evals/hd_epic/run_batch.py)
- [`run_staged_benchmark.py`](/home/tranduong/dev/hm_vqa/evals/hd_epic/run_staged_benchmark.py)

## Repo Layout

### `src/`
- model wrappers and reusable pipeline support
- `src/pipeline/`
  - config
  - retrieval/segmentation helpers
  - metrics/io/types
  - tools
  - experiments

### `evals/`
- dataset-specific loaders, runners, and analysis
- `evals/hd_epic/`
  - HD-EPIC loaders, temporal helpers, and runners
- `evals/hd_epic/analysis/`
  - HD-EPIC-specific analyses and pilot tooling utilities
- `evals/methods/`
  - benchmark-facing method wrappers
- `evals/ablations/`
  - benchmark-facing ablation wrappers

### `mcp/`
- MCP server and tool adapters for HM-VQA tool use

### `docs/`
- [`roadmap.md`](/home/tranduong/dev/hm_vqa/docs/roadmap.md)
- [`taxonomy.md`](/home/tranduong/dev/hm_vqa/docs/taxonomy.md)

## Important Results Kept

- pure VLM baseline:
  - [`results/hd_epic_uniform_qwen_longvideo`](/home/tranduong/dev/hm_vqa/results/hd_epic_uniform_qwen_longvideo)
- Stage 7 baseline:
  - [`results/hd_epic_stage7_visual_verify_w5_s5_batch`](/home/tranduong/dev/hm_vqa/results/hd_epic_stage7_visual_verify_w5_s5_batch)
- current comparison benchmark:
  - [`results/pipeline/comparisons`](/home/tranduong/dev/hm_vqa/results/pipeline/comparisons)
- current promoted pipeline analysis:
  - [`results/pipeline/analysis/mcq_split_final_stage_think4_l1`](/home/tranduong/dev/hm_vqa/results/pipeline/analysis/mcq_split_final_stage_think4_l1)
- grounding analyses:
  - [`results/pipeline/analysis/grounding_quality`](/home/tranduong/dev/hm_vqa/results/pipeline/analysis/grounding_quality)
  - [`results/pipeline/analysis/grounding_quality_extreme`](/home/tranduong/dev/hm_vqa/results/pipeline/analysis/grounding_quality_extreme)

## Environment

Expected local setup:
- Python env in `.venv`
- repo root on `PYTHONPATH`
- optional `.env` with tokens

Example:

```bash
PYTHONPATH=src .venv/bin/python evals/hd_epic/run_mcq_shortlist_joint.py P01-20240203-135502 --limit 50
```
