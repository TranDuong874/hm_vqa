from __future__ import annotations

from typing import Any


_POLICIES: dict[str, dict[str, Any]] = {
    "direct_l2_l1_perception": {
        "name": "direct_l2_l1_perception",
        "intent": "Use for local perception questions where the answer can be read from one or a few nearby scenes.",
        "stages": [
            {
                "stage": "retrieve_l2",
                "layer": "L2",
                "action": "Retrieve top L2 windows directly from the question or anchored scene/object description.",
            },
            {
                "stage": "sample_l1_from_l2",
                "layer": "L1",
                "action": "Sample a few direct L1 frames from each kept L2 window for visual inspection.",
            },
            {
                "stage": "select_evidence",
                "layer": "L1",
                "action": "Keep the most informative evidence frames and answer from them.",
            },
        ],
    },
    "local_event_hierarchy": {
        "name": "local_event_hierarchy",
        "intent": "Use when an anchor event/object should be localized first, then refined to child windows and frames.",
        "stages": [
            {
                "stage": "retrieve_l3",
                "layer": "L3",
                "action": "Retrieve coarse event-level L3 segments for the anchor description.",
            },
            {
                "stage": "rerank_l3_with_l1",
                "layer": "L1",
                "action": "Sample a small number of L1 frames per L3 segment to remove obviously wrong segments.",
            },
            {
                "stage": "expand_to_l2",
                "layer": "L2",
                "action": "Open child L2 windows inside the kept L3 segments.",
            },
            {
                "stage": "sample_l1_from_l2",
                "layer": "L1",
                "action": "Sample direct L1 frames inside the retained L2 windows.",
            },
            {
                "stage": "select_evidence",
                "layer": "L1",
                "action": "Choose the final evidence bundle and answer.",
            },
        ],
    },
    "relative_before_after": {
        "name": "relative_before_after",
        "intent": "Use for before/after and short-range temporal relation questions.",
        "stages": [
            {
                "stage": "localize_anchor",
                "layer": "L3",
                "action": "Retrieve L3 segments for the anchor event or scene.",
            },
            {
                "stage": "inspect_neighbor_l2",
                "layer": "L2",
                "action": "Query nearby L2 windows around the anchor and inspect 2-3 local neighbors.",
            },
            {
                "stage": "inspect_neighbor_l3",
                "layer": "L3",
                "action": "Also inspect 2-3 preceding/following L3 segments for longer-range relation evidence.",
            },
            {
                "stage": "sample_l1",
                "layer": "L1",
                "action": "Sample L1 frames only from the kept nearby L2 and neighboring L3 regions.",
            },
            {
                "stage": "select_evidence",
                "layer": "L1",
                "action": "Choose the evidence set that best supports the relative relation answer.",
            },
        ],
    },
    "sequence_relation": {
        "name": "sequence_relation",
        "intent": "Use for multi-event or scene ordering questions.",
        "stages": [
            {
                "stage": "retrieve_l3_multi",
                "layer": "L3",
                "action": "Retrieve several candidate L3 segments for each event or scene description.",
            },
            {
                "stage": "compare_l3_order",
                "layer": "L3",
                "action": "Compare segment-level timing first before descending to finer layers.",
            },
            {
                "stage": "expand_finalists_to_l2",
                "layer": "L2",
                "action": "Open only the finalist L3 segments into L2 windows.",
            },
            {
                "stage": "sample_l1_finalists",
                "layer": "L1",
                "action": "Sample L1 frames from finalist windows to confirm the ordering decision.",
            },
            {
                "stage": "select_evidence",
                "layer": "L1",
                "action": "Keep the final evidence bundle and answer.",
            },
        ],
    },
    "tracking_relation": {
        "name": "tracking_relation",
        "intent": "Use for cross-scene reappearance and tracking-style questions.",
        "stages": [
            {
                "stage": "retrieve_entity_occurrences",
                "layer": "L3",
                "action": "Retrieve multiple L3 segments where the tracked entity may appear.",
            },
            {
                "stage": "compare_occurrences",
                "layer": "L2",
                "action": "Inspect L2 windows across candidate occurrences to find alternate appearances.",
            },
            {
                "stage": "sample_l1_occurrences",
                "layer": "L1",
                "action": "Sample L1 frames from the best occurrence candidates.",
            },
            {
                "stage": "select_evidence",
                "layer": "L1",
                "action": "Keep the evidence set that best identifies the alternate scene or location.",
            },
        ],
    },
}


def get_policy(name: str) -> dict[str, Any]:
    key = str(name).strip()
    if key not in _POLICIES:
        raise KeyError(f"Unknown hierarchical policy: {name}")
    return dict(_POLICIES[key])


def list_policies() -> list[str]:
    return sorted(_POLICIES)
