from __future__ import annotations

from pathlib import Path
import sys

import yt_dlp


ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
CURRENT_DIR = Path(__file__).resolve().parent

for path in (CURRENT_DIR, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def ensure_local_video(
    *,
    video_root: str | Path,
    url_id: str,
    allow_download: bool = False,
) -> Path:
    root = Path(video_root)
    video_path = root / f"{url_id}.mp4"
    if video_path.exists():
        return video_path
    if not allow_download:
        raise FileNotFoundError(f"Missing video: {video_path}")

    root.mkdir(parents=True, exist_ok=True)
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
