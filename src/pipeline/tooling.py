from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .policies import get_policy
from .tools import route_query_tool


@dataclass(slots=True)
class ToolingPlan:
    question: str
    choices: list[str] = field(default_factory=list)
    query_family: str = ""
    procedure: list[dict[str, Any]] = field(default_factory=list)
    retrieval_policy: str | None = None
    policy: dict[str, Any] | None = None
    dataset: str | None = None
    category: str | None = None
    intent: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "question": self.question,
            "choices": list(self.choices),
            "query_family": self.query_family,
            "procedure": [dict(step) for step in self.procedure],
            "retrieval_policy": self.retrieval_policy,
            "policy": dict(self.policy) if self.policy is not None else None,
            "dataset": self.dataset,
            "category": self.category,
            "intent": self.intent,
        }
        payload.update(self.metadata)
        return payload


def build_tooling_plan(
    question: str,
    choices: list[str] | None = None,
    *,
    dataset: str | None = None,
    category: str | None = None,
    query_family: str | None = None,
    procedure: list[dict[str, Any]] | None = None,
    retrieval_policy: str | None = None,
    intent: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> ToolingPlan:
    normalized_question = str(question).strip()
    normalized_choices = [str(choice) for choice in (choices or [])]
    routed = route_query_tool({"question": normalized_question, "choices": normalized_choices})
    normalized_policy = str(retrieval_policy).strip() if retrieval_policy else None
    return ToolingPlan(
        question=normalized_question,
        choices=normalized_choices,
        query_family=str(query_family or routed["query_family"]),
        procedure=[dict(step) for step in (procedure or routed["procedure"])],
        retrieval_policy=normalized_policy,
        policy=get_policy(normalized_policy) if normalized_policy else None,
        dataset=dataset,
        category=category,
        intent=intent,
        metadata={
            "primary_entity": routed.get("primary_entity"),
            "secondary_entities": list(routed.get("secondary_entities") or []),
            "entity_subtypes": list(routed.get("entity_subtypes") or []),
            "question_time_references": list(routed.get("question_time_references") or []),
            "choice_time_references": list(routed.get("choice_time_references") or []),
            "has_multiple_choice": bool(routed.get("has_multiple_choice")),
            **(metadata or {}),
        },
    )
