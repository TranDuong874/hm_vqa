# HM-VQA Runtime App

This is the clean application/runtime path for HM-VQA. Its runtime code is self-contained under `hmvqa_app/` and does not import from `src/`. It is intentionally separate from:

- `demo/`: older prototype UI.
- `evals/`: experiments, ablations, and benchmark runners.
- `tools/`: one-off operational scripts.

The runtime exposes two evidence modes:

- `hmvqa`: query-focused HM-VQA evidence retrieval.
- `pure_vlm`: uniform frame sampling baseline.

By default, the runtime uses the full retrieval path with CUDA OpenCLIP and CUDA ViCLIP, but it loads them sequentially. During ingestion the app unloads ViCLIP before OpenCLIP frame encoding, then unloads OpenCLIP before ViCLIP L2 encoding. This keeps the final demo path close to the thesis method while avoiding both encoders being resident on a 6GB GPU at the same time.

## Setup

From the repository root:

```bash
pip install -r requirements.txt -r hmvqa_app/requirements.txt
```

The answer step is API-only for the demo and uses Alibaba Cloud / DashScope OpenAI-compatible mode:

```bash
export HMVQA_DEMO_API_KEY="your_api_key"
export HMVQA_APP_API_BASE_URL="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
export HMVQA_APP_MODEL_ID="qwen-vl-max-latest"
export HMVQA_APP_ALLOWED_MODELS="qwen-vl-max-latest,qwen-vl-plus-latest,qwen2.5-vl-72b-instruct"

# Defaults for the final retrieval path:
export HMVQA_APP_OPENCLIP_DEVICE=cuda
export HMVQA_APP_VICLIP_DEVICE=cuda
export HMVQA_APP_USE_VICLIP_L2=1
export HMVQA_APP_UNLOAD_ENCODERS=1

# Lower-VRAM fallback if ViCLIP still does not fit:
# export HMVQA_APP_USE_VICLIP_L2=0
# export HMVQA_APP_OPENCLIP_DEVICE=cpu
# export HMVQA_APP_VICLIP_DEVICE=cpu
```

Run the app:

```bash
uvicorn hmvqa_app.main:app --host 127.0.0.1 --port 7860
```

Open `http://127.0.0.1:7860`.

## API

Upload a video:

```bash
curl -F "video=@/path/to/video.mp4" -F "sample_fps=1" http://127.0.0.1:7860/api/videos
```

Check ingestion progress:

```bash
curl http://127.0.0.1:7860/api/sessions/<session_id>/progress
```

Retrieve evidence only:

```bash
curl -X POST http://127.0.0.1:7860/api/sessions/<session_id>/retrieve \
  -H "Content-Type: application/json" \
  -d '{"question":"What is happening?", "mode":"hmvqa", "evidence_frames":16}'
```

Answer with retrieved evidence:

```bash
curl -X POST http://127.0.0.1:7860/api/sessions/<session_id>/answer \
  -H "Content-Type: application/json" \
  -d '{"question":"What is happening?", "mode":"hmvqa", "evidence_frames":16, "model_id":"qwen-vl-max-latest"}'
```

Clear one session or all app cache:

```bash
curl -X DELETE http://127.0.0.1:7860/api/sessions/<session_id>
curl -X DELETE http://127.0.0.1:7860/api/cache
```

## Manual Module Checks

Ingest one video:

```bash
python -m hmvqa_app.cli ingest /path/to/video.mp4 --sample-fps 1
```

Retrieve without answering:

```bash
python -m hmvqa_app.cli retrieve <session_id> "What is happening?" --mode hmvqa --frames 16
```

Run the pure VLM evidence baseline:

```bash
python -m hmvqa_app.cli retrieve <session_id> "What is happening?" --mode pure_vlm --frames 16
```

Answer from cached evidence:

```bash
python -m hmvqa_app.cli answer <session_id> "What is happening?" --mode hmvqa --frames 16
```

## Cache Layout

Artifacts are stored under:

```text
hmvqa_app/.cache/sessions/<session_id>/
  source.<ext>
  metadata.json
  timestamps.npy
  frame_embeddings.pt
  l2_embeddings.pt
  l3_embeddings.pt
  frame.index
  l2.index
  l3.index
  l2_segments.json
  l3_segments.json
  frames/
```

The `session_id` is deterministic for the uploaded video and final runtime config. Re-uploading the same video with the same config reuses cached artifacts.

## Runtime Code Boundary

The app-local selected implementation lives under `hmvqa_app/runtime/`:

- OpenCLIP and ViCLIP encoders.
- Fixed-window segmentation and mean pooling.
- FAISS index helpers.
- Qwen API/local frame-list answerers. The public demo UI only exposes the Alibaba API path.

Keep broad ablations, adaptive segmentation, old experiment configs, and benchmark runners outside this app path.
