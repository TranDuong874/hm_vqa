from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_MANIFEST = Path("thirdparty/Video-RAG-master/evals/videomme_json_file.json")


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


class VideoMMELoader:
    def __init__(self, manifest_path: str | Path = DEFAULT_MANIFEST) -> None:
        self.manifest_path = Path(manifest_path)
        self._videos: list[VideoMMEVideo] | None = None

    def load(self) -> list[VideoMMEVideo]:
        if self._videos is not None:
            return self._videos

        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "videos" in payload:
            raw_videos = payload["videos"]
        elif isinstance(payload, list):
            raw_videos = payload
        else:
            raise ValueError(f"Unsupported Video-MME manifest format: {self.manifest_path}")

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

    def stats(self) -> dict[str, Any]:
        videos = self.load()
        question_rows = list(self.iter_questions())

        def count_by(items: list[Any], key: str) -> dict[str, int]:
            counts: dict[str, int] = {}
            for item in items:
                value = str(getattr(item, key))
                counts[value] = counts.get(value, 0) + 1
            return dict(sorted(counts.items()))

        duration_domain: dict[str, int] = {}
        for video in videos:
            key = f"{video.duration} | {video.domain}"
            duration_domain[key] = duration_domain.get(key, 0) + 1

        return {
            "videos": len(videos),
            "questions": len(question_rows),
            "video_duration": count_by(videos, "duration"),
            "video_domain": count_by(videos, "domain"),
            "question_task_type": count_by(question_rows, "task_type"),
            "video_duration_domain": dict(sorted(duration_domain.items())),
        }


if __name__ == "__main__":
    loader = VideoMMELoader()
    videos = loader.load()
    print(f"manifest: {loader.manifest_path}")
    print(f"videos: {loader.video_count()}")
    print(f"questions: {loader.question_count()}")
    if videos:
        first = videos[0]
        print(f"first_video_id: {first.video_id}")
        print(f"first_video_file: {first.video_filename}")
        if first.questions:
            print(f"first_question_id: {first.questions[0].question_id}")
            print(f"first_question: {first.questions[0].question}")
    print(json.dumps(loader.stats(), indent=2, ensure_ascii=False))
