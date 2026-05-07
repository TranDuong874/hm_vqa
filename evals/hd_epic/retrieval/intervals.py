from __future__ import annotations

from typing import Any

import torch


L2_SCORE_TOP_M = 4


def _frame_interval(time_sec: float, sample_fps: float) -> tuple[float, float]:
    half_width = 0.5 / max(sample_fps, 1e-6)
    return float(time_sec - half_width), float(time_sec + half_width)


def _legacy_l1_bundles(
    frame_hits: list[Any],
    *,
    max_keep: int,
    max_gap_sec: float,
    half_window_sec: float,
    scope_start_sec: float | None,
    scope_end_sec: float | None,
) -> list[dict[str, float]]:
    """Match the prior HD-EPIC direct L1 protocol: top frames -> merged evidence intervals."""
    ordered = sorted(frame_hits, key=lambda hit: float(hit.time_sec))
    bundles: list[dict[str, float]] = []
    current: dict[str, float] | None = None
    for hit in ordered:
        time_sec = float(hit.time_sec)
        score = float(hit.score)
        if current is None:
            current = {"start_time_sec": time_sec, "end_time_sec": time_sec, "score": score}
            continue
        if time_sec - float(current["end_time_sec"]) <= max_gap_sec:
            current["end_time_sec"] = time_sec
            current["score"] = max(float(current["score"]), score)
        else:
            bundles.append(current)
            current = {"start_time_sec": time_sec, "end_time_sec": time_sec, "score": score}
    if current is not None:
        bundles.append(current)

    bundles.sort(key=lambda hit: float(hit["score"]), reverse=True)
    normalized: list[dict[str, float]] = []
    for bundle in bundles[:max_keep]:
        start_time_sec = max(0.0, float(bundle["start_time_sec"]) - half_window_sec)
        end_time_sec = float(bundle["end_time_sec"]) + half_window_sec
        if scope_start_sec is not None:
            start_time_sec = max(start_time_sec, float(scope_start_sec))
        if scope_end_sec is not None:
            end_time_sec = min(end_time_sec, float(scope_end_sec))
        normalized.append(
            {
                "start_time_sec": float(start_time_sec),
                "end_time_sec": float(max(end_time_sec, start_time_sec)),
                "score": float(bundle["score"]),
            }
        )
    return normalized


def _clip_interval(
    start_time_sec: float,
    end_time_sec: float,
    *,
    scope_start_sec: float | None,
    scope_end_sec: float | None,
) -> tuple[float, float]:
    start = float(start_time_sec)
    end = float(end_time_sec)
    if scope_start_sec is not None:
        start = max(start, float(scope_start_sec))
    if scope_end_sec is not None:
        end = min(end, float(scope_end_sec))
    return start, max(start, end)


def _hit_dict(
    *,
    start_time_sec: float,
    end_time_sec: float,
    score: float,
    scope_start_sec: float | None,
    scope_end_sec: float | None,
    source: str,
    segment_id: str | None = None,
    components: dict[str, float] | None = None,
) -> dict[str, Any]:
    start, end = _clip_interval(
        start_time_sec,
        end_time_sec,
        scope_start_sec=scope_start_sec,
        scope_end_sec=scope_end_sec,
    )
    hit: dict[str, Any] = {
        "start_time_sec": float(start),
        "end_time_sec": float(end),
        "score": float(score),
        "source": source,
    }
    if segment_id is not None:
        hit["segment_id"] = str(segment_id)
    if components:
        hit["components"] = {key: float(value) for key, value in components.items()}
    return hit


def _fixed_windows_from_frame_hits(
    frame_hits: list[Any],
    *,
    max_keep: int,
    window_seconds: float,
    scope_start_sec: float | None,
    scope_end_sec: float | None,
    source: str,
) -> list[dict[str, Any]]:
    half_window = float(window_seconds) / 2.0
    hits: list[dict[str, Any]] = []
    for hit in frame_hits[:max_keep]:
        time_sec = float(hit.time_sec)
        hits.append(
            _hit_dict(
                start_time_sec=time_sec - half_window,
                end_time_sec=time_sec + half_window,
                score=float(hit.score),
                scope_start_sec=scope_start_sec,
                scope_end_sec=scope_end_sec,
                source=source,
                segment_id=f"frame_{int(hit.frame_index):06d}",
            )
        )
    return hits


def _temporal_nms_frame_hits(
    frame_hits: list[Any],
    *,
    max_hits: int,
    min_gap_sec: float,
) -> list[Any]:
    selected: list[Any] = []
    for hit in sorted(frame_hits, key=lambda item: float(item.score), reverse=True):
        if len(selected) >= max_hits:
            break
        if all(abs(float(hit.time_sec) - float(kept.time_sec)) >= min_gap_sec for kept in selected):
            selected.append(hit)
    return selected


def _rank_score_by_id(items: list[Any], *, key_fn: Any) -> dict[Any, float]:
    ordered = sorted(items, key=key_fn, reverse=True)
    if not ordered:
        return {}
    if len(ordered) == 1:
        return {ordered[0][0]: 1.0}
    return {item[0]: 1.0 - (rank / float(len(ordered) - 1)) for rank, item in enumerate(ordered)}


def _segment_time_overlap(
    start_a: float,
    end_a: float,
    start_b: float,
    end_b: float,
) -> bool:
    return max(float(start_a), float(start_b)) <= min(float(end_a), float(end_b))


def _segment_topm_score(
    *,
    frame_scores: torch.Tensor,
    start_index: int,
    end_index: int,
    top_m: int = L2_SCORE_TOP_M,
) -> float:
    segment_scores = frame_scores[int(start_index) : int(end_index) + 1]
    if segment_scores.numel() == 0:
        return 0.0
    k = min(max(int(top_m), 1), int(segment_scores.numel()))
    return float(torch.topk(segment_scores, k=k).values.mean().item())


def _segment_in_scope(
    start_time_sec: float,
    end_time_sec: float,
    scope_start_sec: float | None,
    scope_end_sec: float | None,
) -> bool:
    if scope_start_sec is None or scope_end_sec is None:
        return True
    return max(float(start_time_sec), float(scope_start_sec)) <= min(float(end_time_sec), float(scope_end_sec))

