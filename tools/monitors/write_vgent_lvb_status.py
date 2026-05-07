from __future__ import annotations

import argparse
import io
from contextlib import redirect_stdout
from pathlib import Path

import live_vgent_lvb_ingestion


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("local_storage/flat_files/vgent/offline_graph_cache_qwen2vl7b_api_lvb_fullval_1fps64_336_1024_w3"),
    )
    parser.add_argument(
        "--status-file",
        type=Path,
        default=Path("local_storage/flat_files/vgent/offline_graph_cache_qwen2vl7b_api_lvb_fullval_1fps64_336_1024_w3/live_status.md"),
    )
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        live_vgent_lvb_ingestion.main_args(
            output_root=args.output_root,
            workers=args.workers,
        )
    args.status_file.parent.mkdir(parents=True, exist_ok=True)
    args.status_file.write_text(buffer.getvalue(), encoding="utf-8")


if __name__ == "__main__":
    main()
