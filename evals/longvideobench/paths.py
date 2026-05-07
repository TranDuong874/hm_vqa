from __future__ import annotations

from pathlib import Path

from hm_vqa.storage import LVB_FULL_DERIVED_ROOT, LVB_FULL_MANIFEST, LVB_FULL_OPENCLIP_ROOT, LVB_FULL_VIDEO_ROOT

DATASET_ROOT = Path("/home/tranduong/dev/dataset/LongVideoBench")
SUBTITLE_ROOT = DATASET_ROOT / "subtitles"
SUBTITLE_TAR = DATASET_ROOT / "subtitles.tar"
RESULTS_ROOT = Path("results/longvideobench")

__all__ = [
    "DATASET_ROOT",
    "LVB_FULL_DERIVED_ROOT",
    "LVB_FULL_MANIFEST",
    "LVB_FULL_OPENCLIP_ROOT",
    "LVB_FULL_VIDEO_ROOT",
    "RESULTS_ROOT",
    "SUBTITLE_ROOT",
    "SUBTITLE_TAR",
]
