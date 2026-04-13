from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path


DEFAULT_MANIFEST = Path("evals/video_mme/manifests/video_mme_stratified_50_50_50_no_subs.json")
DEFAULT_ZIP_ROOT = Path("/home/tranduong/dev/dataset/Video-MME")
DEFAULT_OUTPUT_ROOT = Path("/home/tranduong/dev/dataset/Video-MME/videos_subset_50_50_50")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract only the sampled Video-MME videos from the chunked zip files.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--zip-root", type=Path, default=DEFAULT_ZIP_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    wanted = {f"{item['url']}.mp4" for item in payload["videos"]}
    remaining = set(wanted)
    args.output_root.mkdir(parents=True, exist_ok=True)

    for zip_path in sorted(args.zip_root.glob("videos_chunked_*.zip")):
        if not remaining:
            break
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = set(zf.namelist())
            matched = sorted(name for name in names if Path(name).name in remaining)
            for member in matched:
                target = args.output_root / Path(member).name
                if target.exists():
                    remaining.discard(target.name)
                    continue
                with zf.open(member) as src, target.open("wb") as dst:
                    dst.write(src.read())
                remaining.discard(target.name)
                print(f"extracted {target.name} from {zip_path.name}")

    if remaining:
        missing = sorted(remaining)
        preview = ", ".join(missing[:10])
        raise RuntimeError(f"Failed to extract {len(remaining)} videos: {preview}")

    print(f"done: extracted {len(wanted)} videos to {args.output_root}")


if __name__ == "__main__":
    main()
