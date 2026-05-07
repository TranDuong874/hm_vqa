from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class LocalStorageLayout:
    root: Path = Path("local_storage/flat_files")

    def dataset(self, dataset: str, split: str) -> "DatasetStorageLayout":
        return DatasetStorageLayout(root=self.root, dataset=dataset, split=split)


@dataclass(frozen=True, slots=True)
class DatasetStorageLayout:
    root: Path
    dataset: str
    split: str

    @property
    def dataset_root(self) -> Path:
        return self.root / self.dataset / self.split

    @property
    def video_root(self) -> Path:
        return self.dataset_root / "videos"

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifests" / self.dataset / f"{self.split}.json"

    def memory_root(self, policy_name: str) -> Path:
        return self.dataset_root / "memory" / policy_name

    def openclip_root(self, policy_name: str = "openclip_1fps") -> Path:
        return self.memory_root(policy_name) / "l1_openclip"

    def derived_root(self, policy_name: str) -> Path:
        return self.memory_root(policy_name) / "derived"


# Compatibility paths for the current artifact layout. New datasets should use
# DatasetStorageLayout above; these constants keep existing result regeneration
# commands stable while the migration is incremental.
FLAT_STORAGE_ROOT = Path("local_storage/flat_files")
LVB_FULL_MANIFEST = FLAT_STORAGE_ROOT / "manifests" / "longvideobench" / "lvb_val_full_1337_v1.json"
LVB_FULL_VIDEO_ROOT = FLAT_STORAGE_ROOT / "longvideobench" / "videos_full_val_1337_v1"
LVB_FULL_OPENCLIP_ROOT = FLAT_STORAGE_ROOT / "longvideobench" / "openclip_1fps_lvb_val_full_1337_v1"
LVB_FULL_DERIVED_ROOT = FLAT_STORAGE_ROOT / "longvideobench" / "derived_full1337_benchmark_v1"
HD_EPIC_P01_OPENCLIP_ROOT = FLAT_STORAGE_ROOT / "hd_epic_features_p01"
HD_EPIC_P01_DERIVED_ROOT = FLAT_STORAGE_ROOT / "hd_epic_derived_localization_p01"
HD_EPIC_FGAL24_OPENCLIP_ROOT = FLAT_STORAGE_ROOT / "hd_epic_features_1fps_fgal24_v1"
HD_EPIC_FGAL24_DERIVED_ROOT = FLAT_STORAGE_ROOT / "hd_epic_derived_ablation_1fps_fgal24_v1"
VGENT_LVB_QWEN2_GRAPH_ROOT = (
    FLAT_STORAGE_ROOT / "vgent" / "offline_graph_cache_qwen2vl7b_api_lvb_fullval_1fps64_336_1024_w3"
)

