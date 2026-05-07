from __future__ import annotations

from pathlib import Path

from hm_vqa.storage import (
    HD_EPIC_FGAL24_DERIVED_ROOT,
    HD_EPIC_FGAL24_OPENCLIP_ROOT,
    HD_EPIC_P01_DERIVED_ROOT,
    HD_EPIC_P01_OPENCLIP_ROOT,
)

REPO_ROOT = Path("/home/tranduong/dev/hm_vqa")
RAW_VIDEO_ROOT = Path("/home/tranduong/dev/dataset/HD-EPIC")
STRUCTURED_VIDEO_ROOT = REPO_ROOT / "dataset/hd_epic_structured"
ANNOTATION_ROOT = REPO_ROOT / "dataset/hd-epic-annotations/vqa-benchmark"
FGAL24_MANIFEST = REPO_ROOT / "results/hd_epic/manifests/fgal_24videos_600q_v1.json"
RESULTS_ROOT = REPO_ROOT / "results/hd_epic"

__all__ = [
    "ANNOTATION_ROOT",
    "FGAL24_MANIFEST",
    "HD_EPIC_FGAL24_DERIVED_ROOT",
    "HD_EPIC_FGAL24_OPENCLIP_ROOT",
    "HD_EPIC_P01_DERIVED_ROOT",
    "HD_EPIC_P01_OPENCLIP_ROOT",
    "RAW_VIDEO_ROOT",
    "RESULTS_ROOT",
    "REPO_ROOT",
    "STRUCTURED_VIDEO_ROOT",
]
