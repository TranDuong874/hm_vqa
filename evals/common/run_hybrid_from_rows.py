from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from answering.factory import build_answerer
from answering.qwen_vl import AnswerConfig
from evals.common.vlm_baseline_runner import (
    _append_jsonl,
    _load_resume_rows,
    _log_line,
    _merge_frame_texts,
    _rewrite_jsonl,
    _summarize_rows,
    _write_json,
)
from retrieval import load_selected_video_frames


def _load_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[str(row["example_id"])] = row
    return rows


def _pick_uniform_frames(frames: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if len(frames) <= count:
        return list(frames)
    indices = np.linspace(0, len(frames) - 1, count).round().astype(int).tolist()
    return [frames[index] for index in indices]


def _pick_hm_frames(frames: list[dict[str, Any]], count: int, mode: str) -> list[dict[str, Any]]:
    if len(frames) <= count:
        return list(frames)
    if mode == "top_score":
        selected = sorted(frames, key=lambda item: float(item.get("score") or 0.0), reverse=True)[:count]
        return sorted(selected, key=lambda item: float(item["time_sec"]))
    if mode == "chronological":
        indices = np.linspace(0, len(frames) - 1, count).round().astype(int).tolist()
        return [frames[index] for index in indices]
    raise ValueError(f"Unsupported HM selection mode: {mode}")


def _combine_frames(
    *,
    pure_frames: list[dict[str, Any]],
    hm_frames: list[dict[str, Any]],
    uniform_count: int,
    hm_count: int,
    hm_select: str,
    max_frames: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[int] = set()

    def add(items: list[dict[str, Any]], source: str) -> None:
        for item in items:
            index = int(item["frame_index"])
            if index in seen:
                continue
            copied = dict(item)
            copied["source"] = source
            selected.append(copied)
            seen.add(index)

    add(_pick_uniform_frames(pure_frames, uniform_count), "uniform")
    add(_pick_hm_frames(hm_frames, hm_count, hm_select), "hm")
    if len(selected) < max_frames:
        add(_pick_hm_frames(hm_frames, len(hm_frames), hm_select), "hm_fill")
    if len(selected) < max_frames:
        add(_pick_uniform_frames(pure_frames, len(pure_frames)), "uniform_fill")
    return sorted(selected[:max_frames], key=lambda item: float(item["time_sec"]))


def _hybrid_frame_texts(selected_frames: list[dict[str, Any]], frame_times: list[float]) -> list[str]:
    labels: list[str] = []
    for item, time_sec in zip(selected_frames, frame_times):
        source = str(item.get("source") or "")
        if source.startswith("uniform"):
            role = "global context frame"
        elif source.startswith("hm"):
            role = "retrieved evidence frame"
        else:
            role = "evidence frame"
        labels.append(f"Frame at {time_sec:.1f}s ({role})")
    return labels


def _summary_payload(rows: list[dict[str, Any]], total: int) -> dict[str, Any]:
    return {
        "completed": len(rows),
        "total": total,
        **_summarize_rows(rows),
    }


def run_hybrid_from_rows(
    *,
    hm_rows_path: Path,
    pure_rows_path: Path,
    output_dir: Path,
    answer_config: AnswerConfig,
    uniform_count: int,
    hm_count: int,
    hm_select: str,
    max_frames: int,
    sample_fps: float,
    image_max_size: int,
    prompt_prefix: str,
) -> dict[str, Any]:
    hm_rows = _load_rows(hm_rows_path)
    pure_rows = _load_rows(pure_rows_path)
    example_ids = sorted(set(hm_rows) & set(pure_rows))

    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "rows.jsonl"
    progress_path = output_dir / "progress.log"
    rolling_summary_path = output_dir / "rolling_summary.json"
    final_summary_path = output_dir / "final_summary.json"

    rows, dropped_rows = _load_resume_rows(rows_path)
    completed_ids = {str(row["example_id"]) for row in rows}
    pending_ids = [example_id for example_id in example_ids if example_id not in completed_ids]
    if rows_path.exists():
        _rewrite_jsonl(rows_path, rows)

    _write_json(rolling_summary_path, _summary_payload(rows, len(example_ids)))
    _log_line(
        progress_path,
        (
            f"[start] total={len(example_ids)} completed={len(rows)} pending={len(pending_ids)} "
            f"dropped_resume_rows={dropped_rows} uniform_count={uniform_count} hm_count={hm_count}"
        ),
    )

    answerer = build_answerer(answer_config)
    answerer.load()
    try:
        for index, example_id in enumerate(pending_ids, start=len(rows) + 1):
            hm_row = hm_rows[example_id]
            pure_row = pure_rows[example_id]
            selected_frames = _combine_frames(
                pure_frames=list(pure_row.get("frames") or []),
                hm_frames=list(hm_row.get("frames") or []),
                uniform_count=uniform_count,
                hm_count=hm_count,
                hm_select=hm_select,
                max_frames=max_frames,
            )
            target_indices = [int(item["frame_index"]) for item in selected_frames]
            frames, frame_hits, _ = load_selected_video_frames(
                hm_row["video_path"],
                sample_fps=sample_fps,
                target_indices=target_indices,
                image_max_size=image_max_size,
            )
            frame_times = [float(hit.time_sec) for hit in frame_hits]
            frame_texts = _hybrid_frame_texts(selected_frames, frame_times)
            started = time.monotonic()
            prediction = answerer.answer_frames(
                frames=frames,
                question=hm_row["question"],
                options=[str(option) for option in hm_row.get("options", [])],
                prompt_prefix=prompt_prefix,
                frame_texts=frame_texts,
            )
            item_wall_sec = time.monotonic() - started
            gold_letter = hm_row.get("gold_letter")
            row = {
                "example_id": example_id,
                "video_id": hm_row.get("video_id"),
                "video_path": hm_row.get("video_path"),
                "question": hm_row.get("question"),
                "options": hm_row.get("options"),
                "correct_index": hm_row.get("correct_index"),
                "gold_letter": gold_letter,
                "predicted_letter": prediction.predicted_letter,
                "choice_correct": (
                    prediction.predicted_letter == gold_letter if gold_letter is not None else None
                ),
                "raw_answer": prediction.raw_text,
                "generation_sec": prediction.generation_sec,
                "item_wall_sec": item_wall_sec,
                "prompt_tokens": prediction.prompt_tokens,
                "completion_tokens": prediction.completion_tokens,
                "total_tokens": prediction.total_tokens,
                "method": "hybrid_uniform_hm",
                "uniform_count": uniform_count,
                "hm_count": hm_count,
                "hm_select": hm_select,
                "frame_texts": frame_texts,
                "frames": [
                    {
                        "frame_index": int(hit.frame_index),
                        "time_sec": float(hit.time_sec),
                        "score": float(source.get("score") or 0.0),
                        "source": source.get("source"),
                    }
                    for hit, source in zip(frame_hits, selected_frames)
                ],
                "hm_predicted_letter": hm_row.get("predicted_letter"),
                "pure_predicted_letter": pure_row.get("predicted_letter"),
                "hm_correct": hm_row.get("choice_correct"),
                "pure_correct": pure_row.get("choice_correct"),
                **{
                    key: value
                    for key, value in hm_row.items()
                    if key
                    not in {
                        "example_id",
                        "video_id",
                        "video_path",
                        "question",
                        "options",
                        "correct_index",
                        "gold_letter",
                        "predicted_letter",
                        "choice_correct",
                        "raw_answer",
                        "generation_sec",
                        "prompt_tokens",
                        "completion_tokens",
                        "total_tokens",
                        "method",
                        "frames",
                        "frame_texts",
                    }
                },
            }
            rows.append(row)
            _append_jsonl(rows_path, row)
            if index == len(example_ids) or index % 10 == 0:
                _write_json(rolling_summary_path, _summary_payload(rows, len(example_ids)))
            _log_line(
                progress_path,
                (
                    f"[item_done] index={index}/{len(example_ids)} example_id={example_id} "
                    f"predicted={prediction.predicted_letter} correct={row['choice_correct']} "
                    f"gen_sec={prediction.generation_sec:.3f} item_wall_sec={item_wall_sec:.3f}"
                ),
            )
    finally:
        unload = getattr(answerer, "unload", None)
        if callable(unload):
            unload()

    summary = _summary_payload(rows, len(example_ids))
    _write_json(rolling_summary_path, summary)
    _write_json(final_summary_path, summary)
    _log_line(progress_path, f"[done] completed={len(rows)}/{len(example_ids)}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run hybrid uniform + HM evidence QA from saved rows.")
    parser.add_argument("--hm-rows", type=Path, required=True)
    parser.add_argument("--pure-rows", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--uniform-count", type=int, default=8)
    parser.add_argument("--hm-count", type=int, default=8)
    parser.add_argument("--hm-select", choices=["top_score", "chronological"], default="top_score")
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--image-max-size", type=int, default=336)
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--backend", choices=["local", "api"], default="local")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--api-key-env-var", default="ALIBABA_CLOUD_API")
    parser.add_argument("--api-requests-per-minute", type=int, default=60)
    parser.add_argument("--api-tokens-per-minute", type=int, default=100000)
    parser.add_argument(
        "--prompt-prefix",
        default=(
            "You are given a mix of uniformly sampled global context frames and "
            "retrieved evidence frames from a video. Use only the visible evidence."
        ),
    )
    args = parser.parse_args()

    summary = run_hybrid_from_rows(
        hm_rows_path=args.hm_rows,
        pure_rows_path=args.pure_rows,
        output_dir=args.output_dir,
        answer_config=AnswerConfig(
            model_id=args.model_id,
            backend=args.backend,
            max_new_tokens=args.max_new_tokens,
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
            image_max_size=args.image_max_size,
            enable_thinking=args.enable_thinking,
            api_key_env_var=args.api_key_env_var,
            api_requests_per_minute=args.api_requests_per_minute,
            api_tokens_per_minute=args.api_tokens_per_minute,
        ),
        uniform_count=args.uniform_count,
        hm_count=args.hm_count,
        hm_select=args.hm_select,
        max_frames=args.max_frames,
        sample_fps=args.sample_fps,
        image_max_size=args.image_max_size,
        prompt_prefix=args.prompt_prefix,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
