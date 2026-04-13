from __future__ import annotations

from typing import Any


def segment_to_gold_distance(segment: dict[str, Any], gold_spans: list[dict[str, Any]]) -> float:
    segment_start = float(segment["start_time_sec"])
    segment_end = float(segment["end_time_sec"])
    best = float("inf")
    for gold_span in gold_spans:
        gold_start = float(gold_span["start_time_sec"])
        gold_end = float(gold_span["end_time_sec"])
        if segment_end >= gold_start and segment_start <= gold_end:
            return 0.0
        if segment_end < gold_start:
            gap = gold_start - segment_end
        else:
            gap = segment_start - gold_end
        best = min(best, float(gap))
    return 0.0 if best == float("inf") else float(best)


def frame_hits_gold(frame_hit: dict[str, Any], gold_spans: list[dict[str, Any]]) -> bool:
    time_sec = float(frame_hit["time_sec"])
    return any(float(span["start_time_sec"]) <= time_sec <= float(span["end_time_sec"]) for span in gold_spans)


def target_coverage(segment: dict[str, Any], gold: dict[str, Any]) -> float:
    if str(segment["video_id"]) != str(gold["video_id"]):
        return 0.0
    start = max(float(segment["start_time_sec"]), float(gold["start_time_sec"]))
    end = min(float(segment["end_time_sec"]), float(gold["end_time_sec"]))
    overlap = max(0.0, end - start)
    gold_duration = float(gold["end_time_sec"]) - float(gold["start_time_sec"])
    if gold_duration <= 0.0:
        return 0.0
    return overlap / gold_duration


def summarize_layer2_hits(
    *,
    layer2_hits: list[dict[str, Any]],
    gold_spans: list[dict[str, Any]],
    top_k: int,
) -> dict[str, float]:
    selected = layer2_hits[: max(int(top_k), 1)]
    if not selected:
        return {
            "Layer2 Hit@1_gap0": 0.0,
            f"Layer2 Hit@{top_k}_gap0": 0.0,
            "Layer2 mean_top1_distance_sec": float("inf"),
            f"Layer2 mean_top{top_k}_distance_sec": float("inf"),
            f"Layer2 mean_top{top_k}_min_distance_sec": float("inf"),
        }
    distances = [segment_to_gold_distance(hit, gold_spans) for hit in selected]
    return {
        "Layer2 Hit@1_gap0": 1.0 if distances[0] <= 0.0 else 0.0,
        f"Layer2 Hit@{top_k}_gap0": 1.0 if any(distance <= 0.0 for distance in distances) else 0.0,
        "Layer2 mean_top1_distance_sec": float(distances[0]),
        f"Layer2 mean_top{top_k}_distance_sec": float(sum(distances) / len(distances)),
        f"Layer2 mean_top{top_k}_min_distance_sec": float(min(distances)),
    }


def summarize_layer3_hits(
    *,
    layer3_hits: list[dict[str, Any]],
    gold_spans: list[dict[str, Any]],
    coverage_threshold: float,
) -> dict[str, float]:
    def _hit_at(top_k: int) -> float:
        coverage = max(
            (
                target_coverage(hit, gold)
                for hit in layer3_hits[:top_k]
                for gold in gold_spans
            ),
            default=0.0,
        )
        return 1.0 if coverage >= float(coverage_threshold) else 0.0

    return {
        "Layer3CoverageHit@1": _hit_at(1),
        "Layer3CoverageHit@3": _hit_at(3),
        "Layer3CoverageHit@5": _hit_at(5),
    }


def summarize_selected_segment_hits(
    *,
    hits: list[dict[str, Any]],
    gold_spans: list[dict[str, Any]],
    prefix: str,
) -> dict[str, float]:
    if not hits:
        return {
            f"{prefix} SelectedCount": 0.0,
            f"{prefix} Hit@Any_gap0": 0.0,
            f"{prefix} mean_selected_distance_sec": float("inf"),
            f"{prefix} min_selected_distance_sec": float("inf"),
        }
    distances = [segment_to_gold_distance(hit, gold_spans) for hit in hits]
    return {
        f"{prefix} SelectedCount": float(len(hits)),
        f"{prefix} Hit@Any_gap0": 1.0 if any(distance <= 0.0 for distance in distances) else 0.0,
        f"{prefix} mean_selected_distance_sec": float(sum(distances) / len(distances)),
        f"{prefix} min_selected_distance_sec": float(min(distances)),
    }


def summarize_rows(
    *,
    rows: list[dict[str, Any]],
    total_examples: int,
) -> dict[str, Any]:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    error_rows = [row for row in rows if row.get("status") == "error"]
    skipped_rows = [row for row in rows if row.get("status") == "skipped"]

    def _mean(key: str) -> float:
        if not ok_rows:
            return 0.0
        return round(sum(float(row["metrics"][key]) for row in ok_rows) / len(ok_rows), 6)

    frame_hits = [1.0 if row.get("layer1_hits") and frame_hits_gold(row["layer1_hits"][0], row["gold_spans"]) else 0.0 for row in ok_rows]
    summary = {
        "counts": {
            "total_examples": int(total_examples),
            "seen_examples": len(rows),
            "ok_examples": len(ok_rows),
            "skipped_examples": len(skipped_rows),
            "error_examples": len(error_rows),
        },
        "metrics": {
            "Layer3CoverageHit@1": _mean("Layer3CoverageHit@1"),
            "Layer3CoverageHit@3": _mean("Layer3CoverageHit@3"),
            "Layer3CoverageHit@5": _mean("Layer3CoverageHit@5"),
            "Layer2 Hit@1_gap0": _mean("Layer2 Hit@1_gap0"),
            "Layer2 Hit@3_gap0": _mean("Layer2 Hit@3_gap0"),
            "Layer2 mean_top1_distance_sec": _mean("Layer2 mean_top1_distance_sec"),
            "Layer2 mean_top3_distance_sec": _mean("Layer2 mean_top3_distance_sec"),
            "Layer2 mean_top3_min_distance_sec": _mean("Layer2 mean_top3_min_distance_sec"),
            "Layer3 SelectedCount": _mean("Layer3 SelectedCount"),
            "Layer3 Hit@Any_gap0": _mean("Layer3 Hit@Any_gap0"),
            "Layer3 mean_selected_distance_sec": _mean("Layer3 mean_selected_distance_sec"),
            "Layer3 min_selected_distance_sec": _mean("Layer3 min_selected_distance_sec"),
            "Layer2 SelectedCount": _mean("Layer2 SelectedCount"),
            "Layer2 Hit@Any_gap0": _mean("Layer2 Hit@Any_gap0"),
            "Layer2 mean_selected_distance_sec": _mean("Layer2 mean_selected_distance_sec"),
            "Layer2 min_selected_distance_sec": _mean("Layer2 min_selected_distance_sec"),
            "Layer1 FrameHit@1_gap0": round(sum(frame_hits) / len(frame_hits), 6) if frame_hits else 0.0,
        },
    }
    return summary
