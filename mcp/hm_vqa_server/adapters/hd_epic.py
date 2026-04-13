from __future__ import annotations

from typing import Any

from src.pipeline.core.retrieve import extract_target_text


def parse_hd_epic_query(question: str) -> dict[str, Any]:
    query_text = extract_target_text(question)
    return {
        "question": question,
        "query_text": query_text,
    }
