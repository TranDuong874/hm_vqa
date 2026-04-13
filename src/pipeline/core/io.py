from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any


def stable_run_id(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha1(serialized.encode("utf-8")).hexdigest()[:12]
    return f"pipeline_{digest}"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_line(log_path: Path, message: str) -> None:
    ensure_dir(log_path.parent)
    line = f"[{timestamp()}] {message}"
    print(line, flush=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def build_run_state(
    *,
    video_id: str,
    current_video_id: str,
    total_examples: int,
    rows: list[dict[str, Any]],
    started_at: float,
    status: str,
) -> dict[str, Any]:
    ok_examples = sum(1 for row in rows if row.get("status") == "ok")
    skipped_examples = sum(1 for row in rows if row.get("status") == "skipped")
    error_examples = sum(1 for row in rows if row.get("status") == "error")
    return {
        "video_id": video_id,
        "current_video_id": current_video_id,
        "total_examples": int(total_examples),
        "seen_examples": len(rows),
        "scored_examples": ok_examples,
        "skipped_examples": skipped_examples,
        "error_examples": error_examples,
        "elapsed_sec": round(time.perf_counter() - started_at, 3),
        "status": status,
    }
