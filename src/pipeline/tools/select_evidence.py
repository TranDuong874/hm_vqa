from __future__ import annotations

from typing import Any


def select_evidence_tool(arguments: dict[str, Any]) -> dict[str, Any]:
    frames = list(arguments.get("frames") or [])
    limit = max(int(arguments.get("limit", 4)), 1)
    min_gap_sec = float(arguments.get("min_gap_sec", 1.0))

    if not frames:
        raise ValueError("`frames` is required")

    sorted_frames = sorted(
        (
            {
                "frame_index": frame.get("frame_index"),
                "time_sec": float(frame["time_sec"]),
                "score": float(frame.get("score", 0.0)),
                "metadata": frame,
            }
            for frame in frames
            if "time_sec" in frame
        ),
        key=lambda item: item["score"],
        reverse=True,
    )

    selected: list[dict[str, Any]] = []
    for frame in sorted_frames:
        if any(abs(frame["time_sec"] - chosen["time_sec"]) < min_gap_sec for chosen in selected):
            continue
        selected.append(frame)
        if len(selected) >= limit:
            break

    selected.sort(key=lambda item: item["time_sec"])
    return {
        "limit": limit,
        "min_gap_sec": min_gap_sec,
        "selected_frames": selected,
        "selected_count": len(selected),
    }
