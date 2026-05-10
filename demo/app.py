from __future__ import annotations

import json
import os
import shutil
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from answering.factory import build_answerer
from answering.qwen_vl import AnswerConfig
from ingestion import OpenCLIPEncoder
from retrieval.faiss_index import read_ip_index, search_ip_index, write_ip_index
from retrieval.scoring import pool_segments
from segmentation import Segment, segment_fixed_windows


DEMO_ROOT = Path(__file__).resolve().parent
CACHE_ROOT = DEMO_ROOT / ".cache" / "sessions"
STATIC_ROOT = DEMO_ROOT / "static"
OVERWRITE_ON_UPLOAD = {"mlvu_test_ego_8_8m_6of7.mp4"}

SAMPLE_FPS = float(os.getenv("HMVQA_DEMO_SAMPLE_FPS", "1.0"))
MIN_SAMPLE_FPS = 0.25
MAX_SAMPLE_FPS = 4.0
DISPLAY_FRAME_SIZE = int(os.getenv("HMVQA_DEMO_DISPLAY_FRAME_SIZE", "720"))
OPENCLIP_BATCH_SIZE = int(os.getenv("HMVQA_DEMO_OPENCLIP_BATCH_SIZE", "16"))
L3_SECONDS = float(os.getenv("HMVQA_DEMO_L3_SECONDS", "60"))
L2_SECONDS = float(os.getenv("HMVQA_DEMO_L2_SECONDS", "5"))
MAX_EVIDENCE_FRAMES = int(os.getenv("HMVQA_DEMO_MAX_EVIDENCE_FRAMES", "16"))


@dataclass(slots=True)
class ProgressState:
    status: str
    progress: int
    message: str
    video_name: str | None = None
    duration_sec: float | None = None
    sampled_frames: int | None = None
    error: str | None = None


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    backend: str = "api"
    model_id: str = "Qwen/Qwen2-VL-7B-Instruct"
    api_base_url: str = "http://localhost:8000/v1"
    api_key_env_var: str = "HMVQA_DEMO_API_KEY"
    api_key: str | None = None
    max_new_tokens: int = 384
    enable_thinking: bool = False
    retrieval_mode: str = "hm"
    evidence_frames: int = Field(default=MAX_EVIDENCE_FRAMES, ge=1, le=64)


class SessionStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._states: dict[str, ProgressState] = {}

    def create(self, session_id: str, video_name: str) -> None:
        self.set(
            session_id,
            ProgressState(
                status="uploaded",
                progress=1,
                message="Video uploaded. Waiting for ingestion.",
                video_name=video_name,
            ),
        )

    def set(self, session_id: str, state: ProgressState) -> None:
        with self._lock:
            self._states[session_id] = state

    def patch(self, session_id: str, **updates: Any) -> None:
        with self._lock:
            current = self._states.get(session_id)
            if current is None:
                current = ProgressState(status="unknown", progress=0, message="")
            data = asdict(current)
            data.update(updates)
            self._states[session_id] = ProgressState(**data)

    def get(self, session_id: str) -> ProgressState | None:
        with self._lock:
            return self._states.get(session_id)

    def remove(self, session_id: str) -> None:
        with self._lock:
            self._states.pop(session_id, None)


store = SessionStore()
encoder_lock = threading.Lock()
encoder: OpenCLIPEncoder | None = None
answerer_lock = threading.Lock()
answerers: dict[str, Any] = {}

app = FastAPI(title="HM-VQA Demo", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=STATIC_ROOT), name="static")


def _get_encoder() -> OpenCLIPEncoder:
    global encoder
    with encoder_lock:
        if encoder is None:
            encoder = OpenCLIPEncoder()
        return encoder


def _get_answerer(request: ChatRequest) -> Any:
    backend = request.backend.strip().lower()
    if backend not in {"api", "local"}:
        raise HTTPException(status_code=400, detail="backend must be 'api' or 'local'")
    api_key_env = request.api_key_env_var.strip() or "HMVQA_DEMO_API_KEY"
    if request.api_key:
        os.environ[api_key_env] = request.api_key

    config = AnswerConfig(
        backend=backend,
        model_id=request.model_id.strip(),
        max_new_tokens=int(request.max_new_tokens),
        image_max_size=448,
        enable_thinking=bool(request.enable_thinking),
        api_base_url=request.api_base_url.strip(),
        api_key_env_var=api_key_env,
        api_timeout_sec=180.0,
    )
    key = json.dumps(
        {
            "backend": config.backend,
            "model": config.model_id,
            "base": config.api_base_url,
            "env": config.api_key_env_var,
            "tokens": config.max_new_tokens,
            "thinking": config.enable_thinking,
        },
        sort_keys=True,
    )
    with answerer_lock:
        if key not in answerers:
            answerers[key] = build_answerer(config)
        return answerers[key]


def _safe_suffix(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    return suffix if suffix in {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"} else ".mp4"


def _session_dir(session_id: str) -> Path:
    return CACHE_ROOT / session_id


def _frame_path(session_dir: Path, frame_index: int) -> Path:
    return session_dir / "frames" / f"frame_{frame_index:06d}.jpg"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _purge_demo_sessions_for_video(video_name: str) -> int:
    """Remove old demo-session cache for files we intentionally re-demo."""
    if Path(video_name).name not in OVERWRITE_ON_UPLOAD or not CACHE_ROOT.exists():
        return 0

    removed = 0
    for session_dir in CACHE_ROOT.iterdir():
        if not session_dir.is_dir():
            continue
        metadata_path = session_dir / "metadata.json"
        if not metadata_path.exists():
            continue
        try:
            metadata = _read_json(metadata_path)
        except Exception:
            continue
        if metadata.get("video_name") == video_name:
            shutil.rmtree(session_dir, ignore_errors=True)
            store.remove(session_dir.name)
            removed += 1
    return removed


def _resize_for_display(image: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    image.thumbnail((DISPLAY_FRAME_SIZE, DISPLAY_FRAME_SIZE), Image.Resampling.LANCZOS)
    return image


def _clamp_sample_fps(value: float) -> float:
    return max(MIN_SAMPLE_FPS, min(MAX_SAMPLE_FPS, float(value)))


def _sample_video_frames(
    video_path: Path,
    session_dir: Path,
    session_id: str,
    sample_fps: float,
) -> tuple[list[Image.Image], np.ndarray, float]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    native_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if native_fps <= 0.0 or total_frames <= 0:
        capture.release()
        raise RuntimeError("Invalid video FPS or frame count.")

    sample_fps = _clamp_sample_fps(sample_fps)
    step = max(int(round(native_fps / sample_fps)), 1)
    frame_dir = session_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    frames: list[Image.Image] = []
    timestamps: list[float] = []
    sampled_index = 0
    native_index = 0
    last_progress = 5
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if native_index % step == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = _resize_for_display(Image.fromarray(rgb))
            image.save(_frame_path(session_dir, sampled_index), quality=90)
            frames.append(image)
            timestamps.append(native_index / native_fps)
            sampled_index += 1
        if native_index % max(step * 8, 1) == 0:
            progress = 5 + int(min(native_index / max(total_frames, 1), 1.0) * 35)
            if progress > last_progress:
                last_progress = progress
                store.patch(session_id, progress=progress, message=f"Sampling frames: {sampled_index} captured")
        native_index += 1

    capture.release()
    if not frames:
        raise RuntimeError("No frames sampled from uploaded video.")
    return frames, np.asarray(timestamps, dtype=np.float32), native_fps


def _ensure_segments(timestamps: np.ndarray, *, seconds: float, prefix: str) -> list[Segment]:
    segments = segment_fixed_windows(
        timestamps=timestamps,
        window_seconds=seconds,
        stride_seconds=seconds,
        prefix=prefix,
    )
    if segments:
        return segments
    return [
        Segment(
            segment_id=f"{prefix}_0000",
            start_index=0,
            end_index=max(0, len(timestamps) - 1),
            start_time_sec=float(timestamps[0]) if len(timestamps) else 0.0,
            end_time_sec=float(timestamps[-1]) if len(timestamps) else 0.0,
            duration_sec=max(float(timestamps[-1] - timestamps[0]), 0.0) if len(timestamps) else 0.0,
        )
    ]


def _segments_to_json(segments: list[Segment]) -> list[dict[str, Any]]:
    return [
        {
            "segment_id": segment.segment_id,
            "start_index": int(segment.start_index),
            "end_index": int(segment.end_index),
            "start_time_sec": float(segment.start_time_sec),
            "end_time_sec": float(segment.end_time_sec),
            "duration_sec": float(segment.duration_sec),
        }
        for segment in segments
    ]


def _segments_from_json(rows: list[dict[str, Any]]) -> list[Segment]:
    return [Segment(**row) for row in rows]


def ingest_video(session_id: str, video_path: Path, original_name: str, sample_fps: float) -> None:
    session_dir = _session_dir(session_id)
    started = time.perf_counter()
    sample_fps = _clamp_sample_fps(sample_fps)
    try:
        store.patch(session_id, status="processing", progress=4, message="Opening video")
        frames, timestamps, native_fps = _sample_video_frames(video_path, session_dir, session_id, sample_fps)

        store.patch(session_id, progress=42, message=f"Encoding {len(frames)} sampled frames with OpenCLIP")
        embeddings: list[torch.Tensor] = []
        clip = _get_encoder()
        for start in range(0, len(frames), OPENCLIP_BATCH_SIZE):
            batch = frames[start : start + OPENCLIP_BATCH_SIZE]
            embeddings.append(clip.encode_images(batch, batch_size=OPENCLIP_BATCH_SIZE))
            progress = 42 + int(((start + len(batch)) / max(len(frames), 1)) * 28)
            store.patch(session_id, progress=min(progress, 70), message=f"Encoding frames: {start + len(batch)}/{len(frames)}")
        frame_embeddings = torch.cat(embeddings, dim=0)

        store.patch(session_id, progress=72, message="Building L2 and L3 memory")
        l2_segments = _ensure_segments(timestamps, seconds=L2_SECONDS, prefix="l2")
        l3_segments = _ensure_segments(timestamps, seconds=L3_SECONDS, prefix="l3")
        l2_embeddings = pool_segments(frame_embeddings, l2_segments, pooling="mean")
        l3_embeddings = pool_segments(frame_embeddings, l3_segments, pooling="mean")

        store.patch(session_id, progress=84, message="Writing FAISS indexes")
        torch.save(frame_embeddings, session_dir / "frame_embeddings.pt")
        torch.save(l2_embeddings, session_dir / "l2_embeddings.pt")
        torch.save(l3_embeddings, session_dir / "l3_embeddings.pt")
        np.save(session_dir / "timestamps.npy", timestamps)
        write_ip_index(session_dir / "frame.index", frame_embeddings)
        write_ip_index(session_dir / "l2.index", l2_embeddings)
        write_ip_index(session_dir / "l3.index", l3_embeddings)
        _write_json(session_dir / "l2_segments.json", _segments_to_json(l2_segments))
        _write_json(session_dir / "l3_segments.json", _segments_to_json(l3_segments))
        _write_json(
            session_dir / "metadata.json",
            {
                "session_id": session_id,
                "video_name": original_name,
                "video_path": str(video_path),
                "native_fps": native_fps,
                "sample_fps": sample_fps,
                "duration_sec": float(timestamps[-1]) if len(timestamps) else 0.0,
                "sampled_frames": len(frames),
                "ingest_sec": round(time.perf_counter() - started, 3),
            },
        )
        store.patch(
            session_id,
            status="ready",
            progress=100,
            message="Ingestion complete. Ask a question about the video.",
            duration_sec=float(timestamps[-1]) if len(timestamps) else 0.0,
            sampled_frames=len(frames),
        )
    except Exception as exc:
        store.patch(
            session_id,
            status="error",
            progress=100,
            message="Ingestion failed.",
            error=str(exc),
        )


def _load_index(session_id: str) -> dict[str, Any]:
    session_dir = _session_dir(session_id)
    metadata_path = session_dir / "metadata.json"
    if not metadata_path.exists():
        raise HTTPException(status_code=409, detail="Session is not ready yet.")
    return {
        "session_dir": session_dir,
        "metadata": _read_json(metadata_path),
        "timestamps": np.load(session_dir / "timestamps.npy"),
        "frame_embeddings": torch.load(session_dir / "frame_embeddings.pt", map_location="cpu"),
        "l2_segments": _segments_from_json(_read_json(session_dir / "l2_segments.json")),
        "l3_segments": _segments_from_json(_read_json(session_dir / "l3_segments.json")),
        "frame_index": read_ip_index(session_dir / "frame.index"),
        "l2_index": read_ip_index(session_dir / "l2.index"),
        "l3_index": read_ip_index(session_dir / "l3.index"),
    }


def _overlaps(child: Segment, parent: Segment) -> bool:
    return child.start_index <= parent.end_index and child.end_index >= parent.start_index


def _materialize_evidence(
    session_dir: Path,
    timestamps: np.ndarray,
    selected: list[tuple[int, float | None]],
    *,
    source: str,
) -> tuple[list[Image.Image], list[dict[str, Any]]]:
    frames: list[Image.Image] = []
    evidence: list[dict[str, Any]] = []
    for rank, (frame_index, score) in enumerate(selected, start=1):
        path = _frame_path(session_dir, frame_index)
        frames.append(Image.open(path).convert("RGB"))
        evidence.append(
            {
                "rank": rank,
                "frame_index": int(frame_index),
                "timestamp": float(timestamps[frame_index]),
                "score": None if score is None else round(float(score), 4),
                "source": source,
                "url": f"/api/sessions/{session_dir.name}/frames/{path.name}",
            }
        )
    return frames, evidence


def _retrieve_uniform(session_id: str, max_frames: int) -> tuple[list[Image.Image], list[dict[str, Any]], dict[str, Any]]:
    data = _load_index(session_id)
    session_dir: Path = data["session_dir"]
    timestamps: np.ndarray = data["timestamps"]
    total = len(timestamps)
    count = min(max(int(max_frames), 1), total)
    indices = np.linspace(0, total - 1, num=count, dtype=np.int64)
    selected = [(int(index), None) for index in dict.fromkeys(indices.tolist())]
    frames, evidence = _materialize_evidence(session_dir, timestamps, selected, source="uniform")
    return frames, evidence, {"mode": "uniform", "requested_frames": int(max_frames)}


def _retrieve_evidence(session_id: str, question: str, max_frames: int) -> tuple[list[Image.Image], list[dict[str, Any]], dict[str, Any]]:
    data = _load_index(session_id)
    session_dir: Path = data["session_dir"]
    timestamps: np.ndarray = data["timestamps"]
    frame_embeddings: torch.Tensor = data["frame_embeddings"]
    l2_segments: list[Segment] = data["l2_segments"]
    l3_segments: list[Segment] = data["l3_segments"]

    query = _get_encoder().encode_texts([question], batch_size=1)[0]
    l3_scores, l3_indices = search_ip_index(data["l3_index"], query, top_k=min(5, len(l3_segments)))
    l2_scores, l2_indices = search_ip_index(data["l2_index"], query, top_k=min(24, len(l2_segments)))

    l2_by_parent: dict[int, float] = {idx: -1.0 for idx in range(len(l3_segments))}
    for score, l2_idx in zip(l2_scores, l2_indices, strict=False):
        l2_segment = l2_segments[int(l2_idx)]
        for l3_idx in l3_indices:
            l3_segment = l3_segments[int(l3_idx)]
            if _overlaps(l2_segment, l3_segment):
                l2_by_parent[int(l3_idx)] = max(l2_by_parent[int(l3_idx)], float(score))

    reranked: list[tuple[float, int, float, float]] = []
    for l3_score, l3_idx in zip(l3_scores, l3_indices, strict=False):
        l2_bonus = max(l2_by_parent.get(int(l3_idx), -1.0), 0.0)
        reranked.append((float(l3_score) + 0.35 * l2_bonus, int(l3_idx), float(l3_score), float(l2_bonus)))
    reranked.sort(reverse=True)
    selected_l3 = reranked[:3] if reranked else []

    allowed: set[int] = set()
    selected_l3_indices = {idx for _, idx, _, _ in selected_l3}
    for idx in selected_l3_indices:
        segment = l3_segments[idx]
        allowed.update(range(segment.start_index, segment.end_index + 1))
    for score, l2_idx in zip(l2_scores[:8], l2_indices[:8], strict=False):
        segment = l2_segments[int(l2_idx)]
        if any(_overlaps(segment, l3_segments[idx]) for idx in selected_l3_indices):
            allowed.update(range(segment.start_index, segment.end_index + 1))

    if not allowed:
        _, global_indices = search_ip_index(data["frame_index"], query, top_k=max_frames)
        allowed = {int(index) for index in global_indices}

    allowed_indices = sorted(index for index in allowed if 0 <= index < frame_embeddings.shape[0])
    candidate = frame_embeddings[allowed_indices]
    frame_scores = torch.matmul(candidate, query).detach().float().cpu().numpy()
    order = np.argsort(-frame_scores)[: min(max_frames, len(allowed_indices))]
    selected = sorted(
        [
            (allowed_indices[int(local_idx)], float(frame_scores[int(local_idx)]))
            for local_idx in order
        ],
        key=lambda item: item[0],
    )

    frames, evidence = _materialize_evidence(session_dir, timestamps, selected, source="hm")

    retrieval_debug = {
        "mode": "hm",
        "requested_frames": int(max_frames),
        "selected_l3": [
            {
                "segment_id": l3_segments[idx].segment_id,
                "start": l3_segments[idx].start_time_sec,
                "end": l3_segments[idx].end_time_sec,
                "score": round(score, 4),
                "l3_score": round(l3_score, 4),
                "l2_bonus": round(l2_bonus, 4),
            }
            for score, idx, l3_score, l2_bonus in selected_l3
        ],
        "l2_hits": [
            {
                "segment_id": l2_segments[int(idx)].segment_id,
                "start": l2_segments[int(idx)].start_time_sec,
                "end": l2_segments[int(idx)].end_time_sec,
                "score": round(float(score), 4),
            }
            for score, idx in zip(l2_scores[:8], l2_indices[:8], strict=False)
        ],
    }
    return frames, evidence, retrieval_debug


def _build_prompt(question: str, evidence: list[dict[str, Any]], retrieval_debug: dict[str, Any]) -> str:
    mode = retrieval_debug.get("mode", "hm")
    evidence_name = "uniformly sampled frames" if mode == "uniform" else "retrieved visual evidence frames"
    evidence_lines = [
        (
            f"- Frame {item['rank']}: timestamp {item['timestamp']:.2f}s"
            + (f", retrieval score {item['score']:.4f}" if item.get("score") is not None else "")
        )
        for item in evidence
    ]
    l3_lines = [
        f"- {item['segment_id']}: {item['start']:.2f}s-{item['end']:.2f}s, score {item['score']:.4f}"
        for item in retrieval_debug.get("selected_l3", [])
    ]
    return (
        "You are HM-VQA answering a question about one uploaded video.\n"
        f"Use only the {evidence_name}. If the evidence is insufficient, say that clearly.\n"
        "Do not invent details that are not visible in the evidence.\n"
        "Format your response in Markdown with exactly these sections:\n"
        "### Answer\n"
        "One concise answer.\n"
        "### Visual evidence\n"
        "- Cite the relevant frame timestamps and visible details.\n"
        "### Why this evidence was chosen\n"
        "- Briefly connect the retrieved frames to the question.\n\n"
        f"User question:\n{question}\n\n"
        + ("Retrieved coarse video regions:\n" + ("\n".join(l3_lines) if l3_lines else "- none") + "\n\n" if mode != "uniform" else "")
        + "Evidence frames:\n"
        + ("\n".join(evidence_lines) if evidence_lines else "- none")
    )


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_ROOT / "index.html")


@app.post("/api/sessions")
async def create_session(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    sample_fps: float = Form(default=SAMPLE_FPS),
) -> dict[str, Any]:
    original_name = video.filename or "uploaded_video.mp4"
    sample_fps = _clamp_sample_fps(sample_fps)
    _purge_demo_sessions_for_video(original_name)
    session_id = uuid.uuid4().hex[:12]
    session_dir = _session_dir(session_id)
    session_dir.mkdir(parents=True, exist_ok=True)
    suffix = _safe_suffix(original_name)
    video_path = session_dir / f"source{suffix}"
    with video_path.open("wb") as handle:
        shutil.copyfileobj(video.file, handle)
    store.create(session_id, original_name)
    background_tasks.add_task(ingest_video, session_id, video_path, original_name, sample_fps)
    return {"session_id": session_id, "sample_fps": sample_fps}


@app.get("/api/sessions/{session_id}/progress")
def progress(session_id: str) -> dict[str, Any]:
    state = store.get(session_id)
    if state is None:
        metadata = _session_dir(session_id) / "metadata.json"
        if metadata.exists():
            meta = _read_json(metadata)
            return {
                "status": "ready",
                "progress": 100,
                "message": "Ingestion complete. Ask a question about the video.",
                "video_name": meta.get("video_name"),
                "duration_sec": meta.get("duration_sec"),
                "sampled_frames": meta.get("sampled_frames"),
            }
        raise HTTPException(status_code=404, detail="Unknown session.")
    return asdict(state)


@app.post("/api/sessions/{session_id}/chat")
def chat(session_id: str, request: ChatRequest) -> dict[str, Any]:
    state = store.get(session_id)
    if state is not None and state.status != "ready":
        raise HTTPException(status_code=409, detail=f"Session is {state.status}: {state.message}")

    retrieval_mode = request.retrieval_mode.strip().lower()
    max_frames = int(request.evidence_frames)
    if retrieval_mode == "uniform":
        frames, evidence, retrieval_debug = _retrieve_uniform(session_id, max_frames)
    elif retrieval_mode in {"hm", "hm-vqa", "retrieval"}:
        frames, evidence, retrieval_debug = _retrieve_evidence(session_id, request.question, max_frames)
    else:
        raise HTTPException(status_code=400, detail="retrieval_mode must be 'hm' or 'uniform'")
    prompt = _build_prompt(request.question, evidence, retrieval_debug)
    frame_texts = [
        (
            f"Evidence frame {item['rank']} at {item['timestamp']:.2f} seconds."
            + (f" Retrieval score {item['score']:.4f}." if item.get("score") is not None else " Uniformly sampled frame.")
        )
        for item in evidence
    ]
    try:
        answerer = _get_answerer(request)
        generation = answerer.generate_text_from_frames(
            frames=frames,
            prompt=prompt,
            frame_texts=frame_texts,
            max_new_tokens=request.max_new_tokens,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "answer": generation.raw_text,
        "generation_sec": generation.generation_sec,
        "evidence": evidence,
        "retrieval": retrieval_debug,
        "usage": {
            "prompt_tokens": generation.prompt_tokens,
            "completion_tokens": generation.completion_tokens,
            "total_tokens": generation.total_tokens,
        },
    }


@app.get("/api/sessions/{session_id}/frames/{filename}")
def frame(session_id: str, filename: str) -> FileResponse:
    if "/" in filename or "\\" in filename or not filename.startswith("frame_"):
        raise HTTPException(status_code=400, detail="Invalid frame filename.")
    path = _session_dir(session_id) / "frames" / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Frame not found.")
    return FileResponse(path)
