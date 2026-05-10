from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from answering.qwen_api import QwenAPIAnswerer
from answering.qwen_vl import AnswerConfig
from evals.common.retrieval_ablation_runner import _chunk_prompt_prefix
from retrieval.frames import load_selected_video_frames


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [row for row in rows if row.get("choice_correct") is not None]
    correct = sum(1 for row in scored if row.get("choice_correct") is True)
    return {
        "answered": len(rows),
        "scored": len(scored),
        "correct": correct,
        "choice_accuracy": (correct / len(scored)) if scored else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-answer saved evidence rows with an API VLM.")
    parser.add_argument("--source-rows", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--api-base-url", required=True)
    parser.add_argument("--api-key-env-var", default="ALIBABA_CLOUD_API")
    parser.add_argument("--api-requests-per-minute", type=int, default=720)
    parser.add_argument("--api-tokens-per-minute", type=int, default=1000000)
    parser.add_argument("--image-max-size", type=int, default=336)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_dir / "rows.jsonl"
    summary_path = args.output_dir / "final_summary.json"
    progress_path = args.output_dir / "progress.log"

    source_rows = _load_jsonl(args.source_rows)
    output_rows = _load_jsonl(rows_path)
    completed = {str(row["example_id"]) for row in output_rows}
    pending = [row for row in source_rows if str(row.get("example_id")) not in completed]

    answerer = QwenAPIAnswerer(
        AnswerConfig(
            model_id=args.model_id,
            backend="api",
            image_max_size=args.image_max_size,
            max_new_tokens=args.max_new_tokens,
            enable_thinking=False,
            api_base_url=args.api_base_url,
            api_key_env_var=args.api_key_env_var,
            api_requests_per_minute=args.api_requests_per_minute,
            api_tokens_per_minute=args.api_tokens_per_minute,
        )
    )
    answerer.load()
    write_lock = threading.Lock()
    prompt_prefix = _chunk_prompt_prefix(
        "You are given retrieved evidence frames from a video. Use only the visible evidence and any provided subtitles."
    )

    def run_one(row: dict[str, Any]) -> dict[str, Any]:
        selected_indices = [int(frame["frame_index"]) for frame in row.get("frames", [])]
        frames, _, _ = load_selected_video_frames(
            video_path=Path(str(row["video_path"])),
            sample_fps=1.0,
            target_indices=selected_indices,
            image_max_size=args.image_max_size,
        )
        started = time.perf_counter()
        prediction = answerer.answer_frames(
            frames=frames,
            question=str(row["question"]),
            options=[str(option) for option in row["options"]],
            prompt_prefix=prompt_prefix,
            frame_texts=[str(text) for text in row.get("frame_texts", [])],
        )
        gold_letter = row.get("gold_letter")
        out = dict(row)
        out.update(
            {
                "predicted_letter": prediction.predicted_letter,
                "choice_correct": (
                    prediction.predicted_letter == gold_letter if gold_letter is not None else None
                ),
                "raw_answer": prediction.raw_text,
                "generation_sec": prediction.generation_sec,
                "answer_wall_sec": round(time.perf_counter() - started, 3),
                "prompt_tokens": prediction.prompt_tokens,
                "completion_tokens": prediction.completion_tokens,
                "total_tokens": prediction.total_tokens,
                "reanswered_from": str(args.source_rows),
                "model_id": args.model_id,
            }
        )
        return out

    try:
        with ThreadPoolExecutor(max_workers=max(int(args.workers), 1)) as executor:
            futures = {executor.submit(run_one, row): row for row in pending}
            for future in as_completed(futures):
                row = future.result()
                with write_lock:
                    output_rows.append(row)
                    _append_jsonl(rows_path, row)
                    progress_path.write_text(
                        f"completed={len(output_rows)}/{len(source_rows)} "
                        f"accuracy={_summarize(output_rows)['choice_accuracy']}\n",
                        encoding="utf-8",
                    )
    finally:
        answerer.unload()

    summary = {"total": len(source_rows), **_summarize(output_rows)}
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
