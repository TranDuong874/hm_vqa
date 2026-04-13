from __future__ import annotations

import re
from typing import Any

from .time_utils import normalize_intervals, parse_interval


ANGLE_BRACKET_PATTERN = re.compile(r"<([^>]+)>")
BETWEEN_PATTERN = re.compile(r"between\s+(.+?)\s+and\s+(.+?)(?:[?.]|$)")
DID_BEFORE_AFTER_PATTERN = re.compile(r"did\s+(.+?)\s+happen\s+(before|after)\s+(.+?)(?:[?.]|$)")
ORDER_PATTERN = re.compile(r"in which order|order of events|order did", re.IGNORECASE)
BEFORE_AFTER_PATTERN = re.compile(r"\b(before|after)\b", re.IGNORECASE)
WHEN_PATTERN = re.compile(r"^(when did|when was|when were|what time)", re.IGNORECASE)


def _extract_entities(question: str) -> list[str]:
    entities = [match.group(1).strip() for match in ANGLE_BRACKET_PATTERN.finditer(question)]
    entities = [entity for entity in entities if entity]
    if entities:
        return entities

    lowered = question.lower()
    between = BETWEEN_PATTERN.search(lowered)
    if between:
        return [between.group(1).strip(), between.group(2).strip()]

    relation = DID_BEFORE_AFTER_PATTERN.search(lowered)
    if relation:
        return [relation.group(1).strip(), relation.group(3).strip()]

    return [question.strip()]


def _classify_query_family(question: str, has_choices: bool, has_time_refs: bool) -> str:
    text = question.lower().strip()
    if BETWEEN_PATTERN.search(text):
        return "between_events"
    if ORDER_PATTERN.search(text):
        return "order_of_events"
    if DID_BEFORE_AFTER_PATTERN.search(text):
        return "entity_temporal_relation"
    if BEFORE_AFTER_PATTERN.search(text) and any(token in text for token in ("what happened", "what was")):
        return "before_after"
    if WHEN_PATTERN.search(text):
        return "when_did"
    if has_time_refs and any(token in text for token in ("what happened", "what was", "who", "where")):
        return "context_at_time"
    if has_choices:
        return "when_did"
    return "entity_entity_relation"


def _entity_subtypes(question: str) -> list[str]:
    text = question.lower()
    subtypes: list[str] = []
    if any(token in text for token in ("who", "person", "people", "hand")):
        subtypes.append("person")
    if any(token in text for token in ("what object", "object", "bowl", "lid", "spoon", "blender")):
        subtypes.append("object")
    if any(token in text for token in ("what happened", "doing", "action", "turn", "pick", "put", "knead", "roll")):
        subtypes.append("action")
    if any(token in text for token in ("say", "speech", "sound", "audio")):
        subtypes.append("speech")
    if not subtypes:
        subtypes.append("entity")
    return sorted(set(subtypes))


def _procedure_for_family(query_family: str) -> list[dict[str, Any]]:
    if query_family == "when_did":
        return [
            {"stage": "retrieve_candidates", "notes": "Use adapters to retrieve time candidates for the entity."},
            {"stage": "rank", "tool": "entity_to_time"},
            {"stage": "align_options", "tool": "compare_time"},
            {"stage": "select_evidence", "tool": "select_evidence"},
            {"stage": "verify", "tool": "verify_consistency"},
        ]
    if query_family == "entity_temporal_relation":
        return [
            {"stage": "retrieve_candidates", "notes": "Retrieve time spans for each entity."},
            {"stage": "rank", "tool": "entity_to_time"},
            {"stage": "compare", "tool": "compare_time"},
        ]
    if query_family == "order_of_events":
        return [
            {"stage": "retrieve_candidates", "notes": "Retrieve time spans for each entity."},
            {"stage": "rank", "tool": "entity_to_time"},
            {"stage": "sort", "notes": "Order spans by time."},
        ]
    if query_family == "before_after":
        return [
            {"stage": "retrieve_anchor", "tool": "entity_to_time"},
            {"stage": "derive_window", "tool": "derive_window"},
            {"stage": "describe", "tool": "time_to_entity"},
        ]
    if query_family == "context_at_time":
        return [{"stage": "describe", "tool": "time_to_entity"}]
    if query_family == "between_events":
        return [
            {"stage": "retrieve_bounds", "tool": "entity_to_time"},
            {"stage": "derive_window", "tool": "derive_window"},
            {"stage": "describe", "tool": "time_to_entity"},
        ]
    return [
        {"stage": "locate", "tool": "entity_to_time"},
        {"stage": "describe", "tool": "time_to_entity"},
    ]


def route_query_tool(arguments: dict[str, Any]) -> dict[str, Any]:
    question = str(arguments.get("question", "")).strip()
    if not question:
        raise ValueError("`question` is required")

    raw_choices = arguments.get("choices") or []
    parsed_choice_intervals = []
    for choice in raw_choices:
        interval = None
        if isinstance(choice, dict):
            interval = parse_interval(choice.get("interval")) or parse_interval(choice)
        else:
            interval = parse_interval(choice)
        parsed_choice_intervals.append(interval.to_dict() if interval is not None else None)

    question_intervals = normalize_intervals([question])
    entities = _extract_entities(question)
    query_family = _classify_query_family(question, bool(raw_choices), bool(question_intervals))

    return {
        "question": question,
        "query_family": query_family,
        "primary_entity": entities[0] if entities else "",
        "secondary_entities": entities[1:],
        "entity_subtypes": _entity_subtypes(question),
        "question_time_references": [interval.to_dict() for interval in question_intervals],
        "choice_time_references": parsed_choice_intervals,
        "has_multiple_choice": bool(raw_choices),
        "procedure": _procedure_for_family(query_family),
    }


def parse_query_tool(arguments: dict[str, Any]) -> dict[str, Any]:
    return route_query_tool(arguments)
