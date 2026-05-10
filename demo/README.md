# HM-VQA Demo

This demo runs a local browser UI with a FastAPI backend.

It supports:

- video upload by drag/drop or file picker
- per-video sessions
- ingestion progress
- adjustable demo-only sampling FPS at upload time
- OpenCLIP frame memory with L2 and L3 windows
- FAISS indexes for frame, L2, and L3 retrieval
- HM-VQA retrieval or uniform frame sampling for side-by-side comparison
- adjustable evidence frame count per question
- evidence frames shown beside the chat answer
- API or local Qwen-VL answer backend

## Setup

From the repository root:

```bash
pip install -r requirements.txt -r demo/requirements.txt
```

For an OpenAI-compatible API backend, set an API key environment variable:

```bash
export HMVQA_DEMO_API_KEY="your_api_key"
```

Then run:

```bash
uvicorn demo.app:app --host 127.0.0.1 --port 7860
```

Open `http://127.0.0.1:7860`.

The checked-in UI defaults currently point at:

- model: `Qwen/Qwen3-VL-8B-Instruct`
- API base URL: `http://108.255.76.60:53861/v1`
- API key env var: `HMVQA_DEMO_API_KEY`

## Model Settings

In the sidebar:

- `Backend`: choose `API` or `Local`.
- `Model ID`: for example `Qwen/Qwen3-VL-8B-Instruct`.
- `API Base URL`: use your OpenAI-compatible `/v1` endpoint.
- `API Key Env`: defaults to `HMVQA_DEMO_API_KEY`.
- `API Key`: optional local-only convenience field. If filled, the backend sets it for the current process.
- `Evidence mode`: use `HM-VQA retrieval` for query-focused evidence or `Uniform sampling` for an evenly spaced baseline.
- `Frames`: number of evidence frames sent to the answer model.

Before uploading a video, set `Sampling FPS` in the upload panel. Higher FPS gives denser memory and more precise retrieval, but ingestion takes longer and uses more storage. This is demo-only and does not alter benchmark artifacts.

Local backend loads the model with Transformers in the current Python environment and needs enough GPU memory.

## Cache

Uploaded videos and indexes are stored under:

```text
demo/.cache/sessions/
```

Delete that directory to clear demo sessions.
