from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from hmvqa_app.config import CONFIG
from hmvqa_app.schemas import AnswerRequest, RetrieveRequest
from hmvqa_app.services.answering import AnswerService
from hmvqa_app.services.ingestion import IngestionService
from hmvqa_app.services.retrieval import RetrievalService
from hmvqa_app.services.session import SessionService
from hmvqa_app.services.storage import StorageService


storage = StorageService(CONFIG)
sessions = SessionService()
ingestion = IngestionService(CONFIG, storage, sessions)
retrieval = RetrievalService(CONFIG, storage)
answering = AnswerService(CONFIG)

app = FastAPI(title="HM-VQA Runtime", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=CONFIG.static_root), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(CONFIG.static_root / "index.html")


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "schema_version": CONFIG.schema_version,
        "cache_root": str(CONFIG.cache_root),
        "defaults": {
            "sample_fps": CONFIG.sample_fps,
            "use_viclip_l2": CONFIG.use_viclip_l2,
            "openclip_device": CONFIG.openclip_device,
            "viclip_device": CONFIG.viclip_device,
            "unload_encoders": CONFIG.unload_encoders_after_request,
            "l2_seconds": CONFIG.l2_seconds,
            "l3_seconds": CONFIG.l3_seconds,
            "model_id": CONFIG.default_model_id,
            "allowed_model_ids": list(CONFIG.allowed_model_ids),
        },
    }


@app.post("/api/videos")
async def upload_video(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    sample_fps: float = Form(default=CONFIG.sample_fps),
) -> dict[str, Any]:
    original_name = video.filename or "uploaded_video.mp4"
    sample_fps = CONFIG.clamp_sample_fps(sample_fps)
    suffix = storage.safe_suffix(original_name)

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        temp_path = Path(handle.name)
        shutil.copyfileobj(video.file, handle)

    try:
        session_id, video_path, cache_hit = storage.prepare_video(
            temp_path,
            original_name=original_name,
            sample_fps=sample_fps,
        )
    finally:
        temp_path.unlink(missing_ok=True)

    sessions.create(session_id, original_name, cache_hit=cache_hit)

    if not cache_hit:
        background_tasks.add_task(
            ingestion.ingest,
            session_id=session_id,
            video_path=video_path,
            original_name=original_name,
            sample_fps=sample_fps,
        )

    return {"session_id": session_id, "sample_fps": sample_fps, "cache_hit": cache_hit}


@app.post("/api/sessions/{session_id}/ingest")
def ingest_session(session_id: str, background_tasks: BackgroundTasks) -> dict[str, Any]:
    metadata_path = storage.metadata_path(session_id)
    session_dir = storage.session_dir(session_id)
    source_candidates = sorted(session_dir.glob("source.*"))

    if not source_candidates:
        raise HTTPException(status_code=404, detail="Session source video not found.")

    original_name = session_id
    sample_fps = CONFIG.sample_fps

    if metadata_path.exists():
        metadata = storage.read_json(metadata_path)
        original_name = metadata.get("video_name") or original_name
        sample_fps = float(metadata.get("sample_fps") or sample_fps)

    sessions.create(session_id, original_name, cache_hit=False)
    background_tasks.add_task(
        ingestion.ingest,
        session_id=session_id,
        video_path=source_candidates[0],
        original_name=original_name,
        sample_fps=sample_fps,
    )

    return {"session_id": session_id, "status": "queued"}


@app.get("/api/sessions")
def list_sessions() -> dict[str, Any]:
    rows = storage.list_sessions()
    active = {row["session_id"]: row for row in rows}

    for session_id, state in sessions.all().items():
        row = active.setdefault(
            session_id,
            {
                "session_id": session_id,
                "video_name": state.video_name or session_id,
                "chat_count": len(storage.read_chat_history(session_id)),
            },
        )
        row.update(
            {
                "status": state.status,
                "progress": state.progress,
                "message": state.message,
                "duration_sec": state.duration_sec,
                "sampled_frames": state.sampled_frames,
                "cache_hit": state.cache_hit,
                "error": state.error,
            }
        )

    sorted_sessions = sorted(
        active.values(),
        key=lambda item: float(item.get("updated_at") or 0.0),
        reverse=True,
    )
    return {"sessions": sorted_sessions}


@app.get("/api/sessions/{session_id}/progress")
def progress(session_id: str) -> dict[str, Any]:
    state = sessions.get(session_id)
    if state is not None:
        return state.to_dict()

    if storage.is_ready(session_id):
        return _cached_ready_state(session_id)

    raise HTTPException(status_code=404, detail="Unknown session.")


@app.get("/api/sessions/{session_id}/history")
def history(session_id: str) -> dict[str, Any]:
    if not storage.session_dir(session_id).exists():
        raise HTTPException(status_code=404, detail="Unknown session.")
    return {"session_id": session_id, "messages": storage.read_chat_history(session_id)}


@app.post("/api/sessions/{session_id}/retrieve")
def retrieve(session_id: str, request: RetrieveRequest) -> dict[str, Any]:
    _ensure_ready(session_id)

    try:
        result = retrieval.retrieve(
            session_id=session_id,
            question=request.question,
            mode=request.mode,
            evidence_frames=request.evidence_frames,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "mode": request.mode,
        "evidence": _dump_evidence(result.evidence),
        "timing": result.timing,
        "debug": result.debug,
    }


@app.post("/api/sessions/{session_id}/answer")
def answer(session_id: str, request: AnswerRequest) -> dict[str, Any]:
    _ensure_ready(session_id)

    try:
        retrieved = retrieval.retrieve(
            session_id=session_id,
            question=request.question,
            mode=request.mode,
            evidence_frames=request.evidence_frames,
        )
        response = answering.answer(request=request, retrieval=retrieved)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    payload = response.model_dump()
    _append_answer_history(session_id, request.question, payload)

    return payload


@app.get("/api/sessions/{session_id}/frames/{frame_id}")
def frame(session_id: str, frame_id: str) -> FileResponse:
    if "/" in frame_id or "\\" in frame_id or not frame_id.startswith("frame_"):
        raise HTTPException(status_code=400, detail="Invalid frame id.")
    path = storage.frame_path(session_id, frame_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Frame not found.")
    return FileResponse(path)


@app.delete("/api/sessions/{session_id}")
def clear_session(session_id: str) -> dict[str, Any]:
    shutil.rmtree(storage.session_dir(session_id), ignore_errors=True)
    sessions.remove(session_id)
    return {"session_id": session_id, "status": "cleared"}


@app.delete("/api/cache")
def clear_cache() -> dict[str, Any]:
    shutil.rmtree(CONFIG.cache_root, ignore_errors=True)
    CONFIG.cache_root.mkdir(parents=True, exist_ok=True)
    return {"status": "cleared", "cache_root": str(CONFIG.cache_root)}


def _ensure_ready(session_id: str) -> None:
    state = sessions.get(session_id)
    if state is not None and state.status != "ready":
        raise HTTPException(status_code=409, detail=f"Session is {state.status}: {state.message}")

    if not storage.is_ready(session_id):
        raise HTTPException(status_code=409, detail="Session artifacts are not ready.")


def _cached_ready_state(session_id: str) -> dict[str, Any]:
    metadata = storage.read_json(storage.metadata_path(session_id))
    return {
        "status": "ready",
        "progress": 100,
        "message": "Loaded cached memory. Ask a question about the video.",
        "video_name": metadata.get("video_name"),
        "duration_sec": metadata.get("duration_sec"),
        "sampled_frames": metadata.get("sampled_frames"),
        "cache_hit": True,
        "error": None,
    }


def _dump_evidence(evidence: list[Any]) -> list[dict[str, Any]]:
    return [item.model_dump() for item in evidence]


def _append_answer_history(session_id: str, question: str, payload: dict[str, Any]) -> None:
    storage.append_chat_messages(
        session_id,
        [
            {"role": "user", "text": question},
            {
                "role": "assistant",
                "text": payload["answer_text"],
                "mode": payload["mode"],
                "evidence": payload["evidence"],
                "timing": payload["timing"],
                "usage": payload["usage"],
            },
        ],
    )
