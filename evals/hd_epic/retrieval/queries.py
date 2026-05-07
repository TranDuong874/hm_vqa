from __future__ import annotations

import re


PROMPT_QUERY_METHODS = {
    "l3_to_l1_prompt",
    "l3_l1_to_l2_prompt",
}
TARGET_PATTERN = re.compile(r"<([^>]+)>")
PHRASAL_VERB_PARTICLES = {
    "up", "down", "on", "off", "out", "in", "into", "onto", "over", "under", "back", "away", "through",
}


def _extract_target_text(question: str) -> str:
    match = TARGET_PATTERN.search(question)
    if match:
        return match.group(1).strip()
    lowered = question.strip()
    prefixes = (
        "when did the action ",
        "when did ",
        "where did ",
        "which action is ",
    )
    for prefix in prefixes:
        if lowered.lower().startswith(prefix):
            return lowered[len(prefix) :].strip(" ?.")
    return lowered


def _query_texts_for_method(target_text: str, method: str) -> list[str]:
    if method not in PROMPT_QUERY_METHODS:
        return [target_text]
    return [
        target_text,
        f"a video of the action {target_text}",
        f"an egocentric video of someone performing {target_text}",
        f"hands performing {target_text}",
        f"the moment when someone {target_text}",
    ]


def _decompose_target_text(target_text: str) -> list[tuple[str, str]]:
    tokens = [token for token in target_text.strip().split() if token]
    if not tokens:
        return []
    full_text = " ".join(tokens)
    queries: list[tuple[str, str]] = [("full", full_text)]
    if len(tokens) == 1:
        return queries

    if len(tokens) >= 3 and tokens[1].lower() in PHRASAL_VERB_PARTICLES:
        action_text = " ".join(tokens[:2])
        object_text = " ".join(tokens[2:])
    else:
        action_text = tokens[0]
        object_text = " ".join(tokens[1:])

    if action_text and action_text != full_text:
        queries.append(("action", action_text))
    if object_text and object_text != full_text and object_text != action_text:
        queries.append(("object", object_text))
    return queries

