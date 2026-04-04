from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

os.environ["HF_HUB_DISABLE_XET"] = "1"

import cv2
import torch
import yt_dlp
from dotenv import load_dotenv
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


ROOT = Path(__file__).resolve().parents[2]

load_dotenv(ROOT / ".env")
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
    os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", hf_token)


MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
DEVICE = "cuda"
FPS = 4.0
FRAME_BUDGET = 64
MAX_NEW_TOKENS = 16

MANIFEST_PATH = ROOT / "thirdparty" / "Video-RAG-master" / "evals" / "videomme_json_file.json"
VIDEO_ROOT = ROOT / "dataset" / "Video-MME"
OUTPUT_PATH = ROOT / "results" / "baselines" / "qwen3_vl_4fps_three_videos.json"
PROGRESS_PATH = ROOT / "results" / "baselines" / "qwen3_vl_4fps_three_videos.progress.jsonl"

TARGET_URLS = [
    "fFjv93ACGo8",
    "N1cdUjctpG8",
    "dphq5X-rMew",
]


def ensure_local_video(url_id: str) -> Path:
    video_path = VIDEO_ROOT / f"{url_id}.mp4"
    if video_path.exists():
        return video_path
    VIDEO_ROOT.mkdir(parents=True, exist_ok=True)
    youtube_url = f"https://www.youtube.com/watch?v={url_id}"
    download_opts = {
        "outtmpl": str(video_path),
        "format": "mp4/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best",
        "merge_output_format": "mp4",
        "quiet": False,
        "noplaylist": True,
    }
    with yt_dlp.YoutubeDL(download_opts) as ydl:
        ydl.download([youtube_url])
    return video_path


def build_prompt(question: str, options: list[str]) -> str:
    options_text = "\n".join(options)
    return (
        "Answer the multiple-choice question about the video.\n"
        "Choose exactly one option.\n"
        "Reply with only the option letter: A, B, C, or D.\n\n"
        f"Question: {question}\n"
        f"Options:\n{options_text}"
    )


def parse_choice_letter(text: str) -> str | None:
    match = re.search(r"\b([A-D])\b", text.upper())
    return match.group(1) if match else None


def sample_video_frames(
    video_path: Path,
    fps: float,
    max_frames: int | None = None,
) -> tuple[list[Image.Image], int, float, dict[str, object]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    native_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if native_fps <= 0.0 or total_frames <= 0:
        capture.release()
        raise RuntimeError(f"Invalid fps/frame count for video: {video_path}")

    step = max(int(round(native_fps / fps)), 1)
    pil_frames: list[Image.Image] = []
    frame_indices: list[int] = []
    frame_index = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % step == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frames.append(Image.fromarray(frame_rgb))
            frame_indices.append(frame_index)
        frame_index += 1

    capture.release()
    if not pil_frames:
        raise RuntimeError(f"No sampled frames from video: {video_path}")

    if max_frames is not None and len(pil_frames) > max_frames:
        sampled_positions = torch.linspace(0, len(pil_frames) - 1, max_frames).round().long().tolist()
        pil_frames = [pil_frames[index] for index in sampled_positions]
        frame_indices = [frame_indices[index] for index in sampled_positions]

    sampled_frames = len(pil_frames)
    effective_fps = sampled_frames / max(total_frames / native_fps, 1e-6)
    metadata = {
        "total_num_frames": total_frames,
        "fps": float(native_fps),
        "frames_indices": frame_indices,
        "duration": float(total_frames / native_fps),
        "height": height,
        "width": width,
    }
    return pil_frames, sampled_frames, float(effective_fps), metadata


def run_one_question(
    *,
    model: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    video_frames: list[Image.Image],
    video_metadata: dict[str, object],
    question: str,
    options: list[str],
) -> dict[str, object]:
    prompt = build_prompt(question, options)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": "local.mp4",
                    "max_pixels": 360 * 420,
                    "fps": FPS,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        videos=[video_frames],
        video_metadata=[video_metadata],
        do_sample_frames=False,
        padding=True,
        return_tensors="pt",
    )
    inputs = {key: value.to(DEVICE) for key, value in inputs.items()}

    started = time.perf_counter()
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )
    elapsed = time.perf_counter() - started
    generated = outputs[0, inputs["input_ids"].shape[1] :]
    raw_text = processor.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
    return {
        "raw_text": raw_text,
        "predicted_letter": parse_choice_letter(raw_text),
        "generation_sec": round(elapsed, 3),
    }


if __name__ == "__main__":
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    selected = [item for item in payload if item["url"] in TARGET_URLS]
    selected.sort(key=lambda item: TARGET_URLS.index(item["url"]))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()
    if PROGRESS_PATH.exists():
        PROGRESS_PATH.unlink()

    print(f"model_id: {MODEL_ID}")
    print(f"fps: {FPS}")
    print(f"frame_budget: {FRAME_BUDGET}")
    print(f"videos: {len(selected)}")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
        attn_implementation="sdpa",
    ).to(DEVICE)
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    all_results: list[dict[str, object]] = []
    total_started = time.perf_counter()

    for item in selected:
        video_path = ensure_local_video(item["url"])
        print(f"video_id: {item['video_id']} url: {item['url']} path: {video_path}")

        load_started = time.perf_counter()
        video_frames, sampled_frames, effective_fps, video_metadata = sample_video_frames(
            video_path,
            FPS,
            max_frames=FRAME_BUDGET,
        )
        load_elapsed = time.perf_counter() - load_started
        print(
            f"video_loaded frames={sampled_frames} effective_fps={effective_fps:.3f} "
            f"load_sec={load_elapsed:.3f}"
        )

        video_rows = []
        for question_row in item["questions"]:
            print(f"question_id: {question_row['question_id']}")
            result = run_one_question(
                model=model,
                processor=processor,
                video_frames=video_frames,
                video_metadata=video_metadata,
                question=question_row["question"],
                options=question_row["options"],
            )
            choice_correct = result["predicted_letter"] == question_row["answer"]
            row = {
                "video_id": item["video_id"],
                "url": item["url"],
                "duration": item["duration"],
                "domain": item["domain"],
                "sub_category": item["sub_category"],
                "question_id": question_row["question_id"],
                "task_type": question_row["task_type"],
                "question": question_row["question"],
                "options": question_row["options"],
                "gold_letter": question_row["answer"],
                "predicted_letter": result["predicted_letter"],
                "choice_correct": choice_correct,
                "raw_text": result["raw_text"],
                "generation_sec": result["generation_sec"],
                "video_load_sec": round(load_elapsed, 3),
                "sampled_frames": sampled_frames,
                "effective_fps": round(effective_fps, 3),
            }
            all_results.append(row)
            video_rows.append(row)
            with PROGRESS_PATH.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(
                f"pred={row['predicted_letter']} gold={row['gold_letter']} "
                f"ok={row['choice_correct']} gen_sec={row['generation_sec']}"
            )

        video_acc = sum(1 for row in video_rows if row["choice_correct"]) / max(len(video_rows), 1)
        print(f"video_accuracy: {video_acc:.3f}")

    summary = {
        "model_id": MODEL_ID,
        "fps": FPS,
        "frame_budget": FRAME_BUDGET,
        "target_urls": TARGET_URLS,
        "videos": len(selected),
        "questions": len(all_results),
        "choice_accuracy": (
            sum(1 for row in all_results if row["choice_correct"]) / max(len(all_results), 1)
        ),
        "elapsed_sec": round(time.perf_counter() - total_started, 3),
        "results": all_results,
    }
    OUTPUT_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved={OUTPUT_PATH}")
