from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_DATASET = "lmms-lab/Video-MME"
DEFAULT_SPLIT = "test"


@dataclass(slots=True)
class VideoMMEQuestion:
    video_id: str
    duration: str
    domain: str
    sub_category: str
    url: str
    question_id: str
    task_type: str
    question: str
    options: list[str]
    answer: str


@dataclass(slots=True)
class VideoMMEVideo:
    video_id: str
    duration: str
    domain: str
    sub_category: str
    url: str
    questions: list[VideoMMEQuestion]

    @property
    def video_filename(self) -> str:
        return f"{self.url}.mp4"

    def resolve_video_path(self, video_root: str | Path) -> Path:
        return Path(video_root) / self.video_filename


def _normalize_url_id(row: dict[str, Any]) -> str:
    if row.get("videoID"):
        return str(row["videoID"])
    raw_url = str(row.get("url", ""))
    if "watch?v=" in raw_url:
        return raw_url.split("watch?v=", 1)[1]
    return raw_url


def _group_flat_rows(flat_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in flat_rows:
        url_id = _normalize_url_id(row)
        key = (str(row["video_id"]), url_id)
        video = grouped.get(key)
        if video is None:
            video = {
                "video_id": str(row["video_id"]),
                "duration": str(row["duration"]),
                "domain": str(row["domain"]),
                "sub_category": str(row["sub_category"]),
                "url": url_id,
                "questions": [],
            }
            grouped[key] = video
        video["questions"].append(
            {
                "question_id": str(row["question_id"]),
                "task_type": str(row["task_type"]),
                "question": str(row["question"]),
                "options": list(row["options"]),
                "answer": str(row["answer"]),
            }
        )
    videos = list(grouped.values())
    videos.sort(key=lambda item: (str(item["video_id"]), str(item["url"])))
    for item in videos:
        item["questions"].sort(key=lambda q: str(q["question_id"]))
    return videos


def load_official_videomme_rows(
    dataset_name: str = DEFAULT_DATASET,
    split: str = DEFAULT_SPLIT,
) -> list[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, split=split)
    flat_rows = [dict(row) for row in dataset]
    return _group_flat_rows(flat_rows)


class VideoMMELoader:
    def __init__(
        self,
        manifest_path: str | Path | None = None,
        *,
        hf_dataset: str = DEFAULT_DATASET,
        hf_split: str = DEFAULT_SPLIT,
    ) -> None:
        self.manifest_path = Path(manifest_path) if manifest_path is not None else None
        self.hf_dataset = hf_dataset
        self.hf_split = hf_split
        self._videos: list[VideoMMEVideo] | None = None

    def _load_raw_videos(self) -> list[dict[str, Any]]:
        if self.manifest_path is None:
            return load_official_videomme_rows(self.hf_dataset, self.hf_split)

        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "videos" in payload:
            return payload["videos"]
        if isinstance(payload, list):
            if payload and "questions" not in payload[0] and "question_id" in payload[0]:
                return _group_flat_rows(payload)
            return payload
        raise ValueError(f"Unsupported Video-MME manifest format: {self.manifest_path}")

    def load(self) -> list[VideoMMEVideo]:
        if self._videos is not None:
            return self._videos

        raw_videos = self._load_raw_videos()
        videos: list[VideoMMEVideo] = []
        for item in raw_videos:
            questions = [
                VideoMMEQuestion(
                    video_id=str(item["video_id"]),
                    duration=str(item["duration"]),
                    domain=str(item["domain"]),
                    sub_category=str(item["sub_category"]),
                    url=str(item["url"]),
                    question_id=str(question["question_id"]),
                    task_type=str(question["task_type"]),
                    question=str(question["question"]),
                    options=list(question["options"]),
                    answer=str(question["answer"]),
                )
                for question in item.get("questions", [])
            ]
            videos.append(
                VideoMMEVideo(
                    video_id=str(item["video_id"]),
                    duration=str(item["duration"]),
                    domain=str(item["domain"]),
                    sub_category=str(item["sub_category"]),
                    url=str(item["url"]),
                    questions=questions,
                )
            )
        self._videos = videos
        return videos

    def iter_videos(self) -> Iterable[VideoMMEVideo]:
        return iter(self.load())

    def iter_questions(self) -> Iterable[VideoMMEQuestion]:
        for video in self.load():
            for question in video.questions:
                yield question

    def video_count(self) -> int:
        return len(self.load())

    def question_count(self) -> int:
        return sum(len(video.questions) for video in self.load())
