from __future__ import annotations

import argparse
from pathlib import Path

from hmvqa_app.config import CONFIG
from hmvqa_app.schemas import AnswerRequest
from hmvqa_app.services.answering import AnswerService
from hmvqa_app.services.ingestion import IngestionService
from hmvqa_app.services.retrieval import RetrievalService
from hmvqa_app.services.session import SessionService
from hmvqa_app.services.storage import StorageService


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual HM-VQA runtime checks")
    sub = parser.add_subparsers(dest="cmd", required=True)
    ingest_cmd = sub.add_parser("ingest")
    ingest_cmd.add_argument("video", type=Path)
    ingest_cmd.add_argument("--sample-fps", type=float, default=CONFIG.sample_fps)
    retrieve_cmd = sub.add_parser("retrieve")
    retrieve_cmd.add_argument("session_id")
    retrieve_cmd.add_argument("question")
    retrieve_cmd.add_argument("--mode", choices=["hmvqa", "pure_vlm"], default="hmvqa")
    retrieve_cmd.add_argument("--frames", type=int, default=CONFIG.default_evidence_frames)
    answer_cmd = sub.add_parser("answer")
    answer_cmd.add_argument("session_id")
    answer_cmd.add_argument("question")
    answer_cmd.add_argument("--mode", choices=["hmvqa", "pure_vlm"], default="hmvqa")
    answer_cmd.add_argument("--frames", type=int, default=CONFIG.default_evidence_frames)

    args = parser.parse_args()
    storage = StorageService(CONFIG)
    sessions = SessionService()

    if args.cmd == "ingest":
        session_id, video_path, cache_hit = storage.prepare_video(
            args.video,
            original_name=args.video.name,
            sample_fps=CONFIG.clamp_sample_fps(args.sample_fps),
        )
        sessions.create(session_id, args.video.name, cache_hit=cache_hit)
        IngestionService(CONFIG, storage, sessions).ingest(
            session_id=session_id,
            video_path=video_path,
            original_name=args.video.name,
            sample_fps=args.sample_fps,
        )

        state = sessions.get(session_id)
        print({"session_id": session_id, "cache_hit": cache_hit, "state": state.to_dict() if state else None})
        return

    retrieval = RetrievalService(CONFIG, storage)
    result = retrieval.retrieve(
        session_id=args.session_id,
        question=args.question,
        mode=args.mode,
        evidence_frames=args.frames,
    )

    if args.cmd == "retrieve":
        print({
            "evidence": [item.model_dump() for item in result.evidence],
            "timing": result.timing,
            "debug": result.debug,
        })
        return

    request = AnswerRequest(question=args.question, mode=args.mode, evidence_frames=args.frames)
    response = AnswerService(CONFIG).answer(request=request, retrieval=result)
    print(response.model_dump())


if __name__ == "__main__":
    main()
