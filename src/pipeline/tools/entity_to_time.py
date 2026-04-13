from __future__ import annotations

import re
from collections import Counter
from typing import Any

from .time_utils import format_seconds, parse_interval


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def _score_candidate(entity: str, candidate: dict[str, Any]) -> float:
    query_terms = Counter(_tokenize(entity))
    candidate_text = " ".join(
        str(candidate.get(key, ""))
        for key in ("label", "text", "entity", "description", "query_text")
    )
    candidate_terms = Counter(_tokenize(candidate_text))
    lexical = sum(min(query_terms[token], candidate_terms[token]) for token in query_terms)
    prior = float(candidate.get("score", 0.0))
    return float(lexical) + prior


def entity_to_time_tool(arguments: dict[str, Any]) -> dict[str, Any]:
    entity = str(arguments.get("entity", "")).strip()
    candidates = list(arguments.get("candidates") or [])
    top_k = max(int(arguments.get("top_k", 5)), 1)

    if not entity:
        raise ValueError("`entity` is required")
    if not candidates:
        raise ValueError("`candidates` is required")

    ranked: list[dict[str, Any]] = []
    for candidate in candidates:
        interval = parse_interval(candidate.get("interval")) or parse_interval(candidate)
        if interval is None:
            continue
        score = _score_candidate(entity, candidate)
        ranked.append(
            {
                "interval": interval.to_dict(),
                "display": f"{format_seconds(interval.start_time_sec)} to {format_seconds(interval.end_time_sec)}",
                "entity": candidate.get("entity") or candidate.get("label") or candidate.get("text"),
                "score": score,
                "source": candidate.get("source", "candidate_list"),
                "metadata": candidate,
            }
        )

    ranked.sort(key=lambda item: item["score"], reverse=True)
    return {
        "entity": entity,
        "top_k": top_k,
        "candidates_considered": len(candidates),
        "results": ranked[:top_k],
    }
