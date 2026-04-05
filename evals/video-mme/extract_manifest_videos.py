from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Iterable
from zipfile import ZipFile


def iter_manifest_urls(manifest_path: Path) -> list[str]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "videos" not in payload:
        raise ValueError(f"Unsupported manifest format: {manifest_path}")
    urls: list[str] = []
    for item in payload["videos"]:
        url = str(item["url"])
        if url not in urls:
            urls.append(url)
    return urls


def list_chunk_zips(dataset_root: Path) -> list[Path]:
    zips = sorted(dataset_root.glob("videos_chunked_*.zip"))
    if not zips:
        raise FileNotFoundError(f"No chunked zip files found under {dataset_root}")
    return zips


def build_zip_index(zip_paths: Iterable[Path], target_filenames: set[str]) -> dict[str, tuple[Path, str]]:
    index: dict[str, tuple[Path, str]] = {}
    for zip_path in zip_paths:
        with ZipFile(zip_path) as archive:
            for member in archive.namelist():
                filename = Path(member).name
                if filename in target_filenames and filename not in index:
                    index[filename] = (zip_path, member)
    return index


def extract_member(zip_path: Path, member: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["unzip", "-j", "-n", str(zip_path), member, "-d", str(output_dir)],
        check=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract only the videos referenced by a Video-MME manifest.")
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    urls = iter_manifest_urls(args.manifest_path)
    target_filenames = {f"{url}.mp4" for url in urls}
    chunk_zips = list_chunk_zips(args.dataset_root)
    zip_index = build_zip_index(chunk_zips, target_filenames)

    missing = sorted(target_filenames - set(zip_index))
    if missing:
        preview = ", ".join(missing[:5])
        more = "" if len(missing) <= 5 else f" (+{len(missing) - 5} more)"
        raise FileNotFoundError(f"Could not locate these videos in chunk zips: {preview}{more}")

    extracted = 0
    skipped = 0
    for url in urls:
        filename = f"{url}.mp4"
        destination = args.output_dir / filename
        if destination.exists():
            skipped += 1
            continue
        zip_path, member = zip_index[filename]
        extract_member(zip_path, member, args.output_dir)
        extracted += 1
        print(f"extracted url={url} zip={zip_path.name}")

    print(
        json.dumps(
            {
                "manifest_path": str(args.manifest_path),
                "dataset_root": str(args.dataset_root),
                "output_dir": str(args.output_dir),
                "requested_videos": len(urls),
                "extracted": extracted,
                "skipped_existing": skipped,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
