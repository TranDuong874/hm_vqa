from __future__ import annotations

from pathlib import Path
import sys


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
) -> Path:
    root = Path(video_root)
    video_path = root / f"{url_id}.mp4"
    if video_path.exists():
        return video_path
    raise FileNotFoundError(f"Missing local video: {video_path}")
