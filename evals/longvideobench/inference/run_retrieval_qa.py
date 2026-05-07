from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from answering.factory import build_answerer
from answering.qwen_vl import AnswerConfig
from evals.longvideobench.paths import (
    LVB_FULL_DERIVED_ROOT,
    LVB_FULL_MANIFEST,
    LVB_FULL_OPENCLIP_ROOT,
    LVB_FULL_VIDEO_ROOT,
    SUBTITLE_ROOT,
    SUBTITLE_TAR,
)
from evals.common.vlm_baseline_runner import (
    BaselineExample,
    _append_jsonl,
    _is_api_content_filter_error,
    _load_resume_rows,
    _normalize_subtitles,
    _load_subtitles,
    _log_line,
    _merge_frame_texts,
    _rewrite_jsonl,
    _subtitle_texts_for_frames,
    _summarize_rows,
    _write_json,
)
from ingestion import OpenCLIPEncoder
from ingestion.viclip import ViCLIPEncoder
from evals.common.video_sampling import sample_uniform_video_frames as _sample_uniform_video_frames
from retrieval import (
    adapt_query_embedding_for_segment_pooling,
    build_query_text,
    build_window_segments,
    collect_segment_frame_indices,
    load_selected_video_frames,
    pool_segments,
    retrieve_top_frames,
    retrieve_top_segments,
    retrieve_top_segments_from_frame_scores,
)
from retrieval.types import FrameHit, PipelineConfig, SegmentHit
from segmentation import Segment, segment_fused_adaptive_peaks, segment_l3_local_contrast_windows
from segmentation.video import compute_motion_energy_for_frame_indices


DEFAULT_MANIFEST = LVB_FULL_MANIFEST
DEFAULT_VIDEO_ROOT = LVB_FULL_VIDEO_ROOT
DEFAULT_FEATURE_ROOT = LVB_FULL_OPENCLIP_ROOT
DEFAULT_DERIVED_CACHE_ROOT = LVB_FULL_DERIVED_ROOT
DEFAULT_SUBTITLE_ROOT = SUBTITLE_ROOT
DEFAULT_SUBTITLE_TAR = SUBTITLE_TAR
DEFAULT_OUTPUT_ROOT = Path("/home/tranduong/dev/hm_vqa/results/longvideobench/ablations")
DEFAULT_L3_RERANK_K = 5
L2_SCORE_TOP_M = 4
VICLIP_L2_MAX_FRAMES = 16
_VICLIP_ENCODER: ViCLIPEncoder | None = None
GRAPH_PROMPT = """You are given representative frames from one short video segment.
Extract compact visual grounding as JSON only.

Return this schema exactly:
{
  "entities": ["..."],
  "actions": ["..."],
  "scenes": ["..."]
}

Rules:
- use short phrases only
- max 6 entities, 6 actions, 3 scenes
- only include visible evidence
- no markdown
"""


@dataclass(slots=True)
class AblationRunConfig:
    method: str
    sample_fps: float = 1.0
    max_frames: int = 16
    image_max_size: int | None = 336
    include_subtitles: bool = True
    l2_window_seconds: float = 5.0
    l2_stride_seconds: float = 5.0
    l2_segmentation: str = "fixed"
    l2_local_min_duration_sec: float = 3.0
    l2_local_max_duration_sec: float = 12.0
    l2_local_fast_kernel_size: int = 1
    l2_local_slow_kernel_size: int = 9
    l2_local_peak_percentile: float = 75.0
    l2_scoring: str = "embedding"
    l2_frame_score_top_m: int = 4
    l2_frame_score_temperature: float = 0.07
    top_l2_segments: int = 10
    top_l3_segments: int = 10
    l3_segmentation: str = "fused_adaptive"
    l3_window_seconds: float = 60.0
    l3_stride_seconds: float = 60.0
    l1_expansion_peaks: int = 6
    l1_expansion_candidates: int = 32
    l1_temporal_nms_sec: float = 4.0
    et_l2_min_video_sec: float = 0.0
    graph_frames_per_segment: int = 3
    prompt_prefix: str = "You are given retrieved evidence frames from a video. Use only the visible evidence and any provided subtitles."
    l2_rerank_encoder: str = "openclip"
    l3_rerank_keep: int = DEFAULT_L3_RERANK_K
    l3_rerank_evidence_source: str = "reranked_l3"
    l2_evidence_per_l3: int = 1
    l1_evidence_per_l2: int = 3


@dataclass(slots=True)
class VideoArtifacts:
    video_id: str
    video_path: Path
    timestamps: np.ndarray
    frame_embeddings: torch.Tensor
    native_fps: float
    l2_segments: list[Segment] | None = None
    l2_embeddings: torch.Tensor | None = None
    l3_segments: list[Segment] | None = None
    l3_embeddings: torch.Tensor | None = None
    l2_graph_nodes: list[dict[str, Any]] | None = None
    l2_graph_embeddings: torch.Tensor | None = None


@dataclass(slots=True)
class ETRetrievalPlan:
    plan_type: str
    l3_queries: list[str]
    l2_queries: list[str]
    l1_queries: list[str]
    relation: str = "none"
    anchor_query: str | None = None


@dataclass(slots=True)
class VgentQueryPlan:
    primary_query: str
    retrieval_queries: list[str]
    subqueries: list[str]
    keywords: list[str]
    multiple: bool
    time_focus: str
    tool: str
    global_reasoning: bool
    candidates_necessary: bool
    raw_controller: str | None = None
    raw_subqueries: str | None = None


@dataclass(slots=True)
class HMRouteDecision:
    route: str
    reasons: list[str]
    temporal_local: bool
    quoted_anchor: bool
    fine_motion: bool
    ambiguous_l3: bool
    l3_top1_score: float | None = None
    l3_top2_score: float | None = None
    l3_margin: float | None = None


def _strip_question_boilerplate(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"^(question\s*:\s*)", "", cleaned, flags=re.IGNORECASE)


def _normalize_graph_payload(raw_text: str) -> dict[str, list[str]]:
    cleaned = raw_text.replace("```json", "").replace("```", "").strip()
    payload: dict[str, Any] | None = None
    try:
        parsed = json.loads(cleaned)
        payload = parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                payload = parsed if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                payload = None
    payload = payload or {}

    def _as_list(value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        out: list[str] = []
        seen: set[str] = set()
        for item in value:
            text = re.sub(r"\s+", " ", str(item)).strip()
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(text[:120])
        return out

    return {
        "entities": _as_list(payload.get("entities")),
        "actions": _as_list(payload.get("actions")),
        "scenes": _as_list(payload.get("scenes")),
    }


def _segment_sample_indices(segment: Segment, count: int) -> list[int]:
    indices = np.arange(int(segment.start_index), int(segment.end_index) + 1)
    if len(indices) <= count:
        return [int(index) for index in indices.tolist()]
    positions = np.linspace(0, len(indices) - 1, num=count)
    return [int(indices[int(round(position))]) for position in positions]


def _simple_lemma(token: str) -> str:
    value = token.lower()
    if len(value) > 5 and value.endswith("ies"):
        return value[:-3] + "y"
    if len(value) > 4 and value.endswith("ing"):
        stem = value[:-3]
        if len(stem) >= 2 and stem[-1] == stem[-2]:
            stem = stem[:-1]
        return stem
    if len(value) > 3 and value.endswith("ed"):
        stem = value[:-2]
        if len(stem) >= 2 and stem[-1] == stem[-2]:
            stem = stem[:-1]
        return stem
    if len(value) > 3 and value.endswith("es"):
        return value[:-2]
    if len(value) > 3 and value.endswith("s"):
        return value[:-1]
    return value


def _tokenize_for_graph(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _normalized_graph_terms(texts: list[str]) -> list[str]:
    tokens: list[str] = []
    for text in texts:
        for token in _tokenize_for_graph(text):
            lemma = _simple_lemma(token)
            if lemma:
                tokens.append(lemma)
    deduped: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        deduped.append(token)
    return deduped


def _build_et_plan(question: str, options: list[str]) -> ETRetrievalPlan:
    q = _strip_question_boilerplate(question)
    lower = q.lower()
    option_text = " ".join(options)
    before_patterns = [" before ", " prior to ", " earlier "]
    after_patterns = [" after ", " following ", " once ", " later "]
    sequence_patterns = ["chronological order", "sequence", "which order", "first", "then", "finally"]
    compare_patterns = ["who ", "which ", "what object", "what objects", "what item", "what food", "what clothes"]

    relation = "none"
    if any(pattern in f" {lower} " for pattern in before_patterns):
        relation = "before"
    elif any(pattern in f" {lower} " for pattern in after_patterns):
        relation = "after"
    elif " while " in lower or " during " in lower or " at this moment" in lower:
        relation = "during"

    if any(pattern in lower for pattern in sequence_patterns) and ("order" in lower or "sequence" in lower):
        plan_type = "sequence_ordering"
    elif relation in {"before", "after"} or "change" in lower or "changed" in lower:
        plan_type = "before_after_reasoning"
    elif any(pattern in lower for pattern in compare_patterns):
        plan_type = "same_moment_comparison"
    else:
        plan_type = "direct_lookup"

    anchor_query = q
    if relation == "before":
        parts = re.split(r"\bbefore\b|\bprior to\b|\bearlier\b", q, maxsplit=1, flags=re.IGNORECASE)
        if len(parts) > 1 and parts[1].strip():
            anchor_query = parts[1].strip(" ?.")
    elif relation == "after":
        parts = re.split(r"\bafter\b|\bfollowing\b|\bonce\b", q, maxsplit=1, flags=re.IGNORECASE)
        if len(parts) > 1 and parts[1].strip():
            anchor_query = parts[0].strip(" ?.") + " " + parts[1].strip(" ?.")

    l3_queries = [q]
    l2_queries = [anchor_query, q] if anchor_query != q else [q]
    l1_queries = [q]
    if plan_type == "same_moment_comparison":
        l1_queries.extend(options)
        l2_queries.append(option_text)
    elif plan_type == "before_after_reasoning":
        l3_queries = [anchor_query, q]
        l2_queries = [anchor_query, q]
        l1_queries = [anchor_query, q]
    elif plan_type == "sequence_ordering":
        l3_queries.extend(options)
        l2_queries.extend(options)
        l1_queries.extend(options)

    def dedupe(items: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for item in items:
            normalized = re.sub(r"\s+", " ", item).strip()
            if not normalized:
                continue
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            result.append(normalized)
        return result

    return ETRetrievalPlan(
        plan_type=plan_type,
        l3_queries=dedupe(l3_queries),
        l2_queries=dedupe(l2_queries),
        l1_queries=dedupe(l1_queries),
        relation=relation,
        anchor_query=anchor_query,
    )


def _extract_json_object(raw_text: str) -> dict[str, Any] | None:
    cleaned = raw_text.replace("```json", "").replace("```", "").strip()
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None


def _dedupe_text_items(items: list[str], *, max_items: int | None = None) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for item in items:
        normalized = re.sub(r"\s+", " ", str(item)).strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
        if max_items is not None and len(deduped) >= max_items:
            break
    return deduped


def _is_local_temporal_question(question: str) -> tuple[bool, bool, bool]:
    lowered = f" {_strip_question_boilerplate(question).lower()} "
    quoted_anchor = bool(re.search(r"['\"“”].+?['\"“”]", question))
    fine_motion_terms = (
        " movement ",
        " hand ",
        " hands ",
        " gesture ",
        " pose ",
        " position ",
        " first character ",
        " appear on screen ",
        " appears on screen ",
        " happens first ",
    )
    temporal_terms = (
        " before ",
        " after ",
        " first ",
        " next ",
        " then ",
        " while ",
        " when ",
        " once ",
        " following ",
    )
    fine_motion = any(term in lowered for term in fine_motion_terms)
    temporal_local = quoted_anchor or fine_motion or any(term in lowered for term in temporal_terms)
    return temporal_local, quoted_anchor, fine_motion


def _decide_hm_route(
    *,
    question: str,
    l3_hits: list[SegmentHit],
) -> HMRouteDecision:
    temporal_local, quoted_anchor, fine_motion = _is_local_temporal_question(question)
    top1_score = float(l3_hits[0].score) if l3_hits else None
    top2_score = float(l3_hits[1].score) if len(l3_hits) > 1 else None
    l3_margin = (top1_score - top2_score) if top1_score is not None and top2_score is not None else None
    ambiguous_l3 = bool(l3_margin is not None and l3_margin < 0.03)

    reasons: list[str] = []
    if quoted_anchor:
        reasons.append("quoted_anchor")
    if fine_motion:
        reasons.append("fine_motion")
    if ambiguous_l3:
        reasons.append("ambiguous_l3")

    route = "l3_only"
    if quoted_anchor or (temporal_local and ambiguous_l3):
        route = "l3_to_l2_local"
    elif fine_motion and len(l3_hits) > 1:
        route = "l3_to_l2_local"

    return HMRouteDecision(
        route=route,
        reasons=reasons,
        temporal_local=temporal_local,
        quoted_anchor=quoted_anchor,
        fine_motion=fine_motion,
        ambiguous_l3=ambiguous_l3,
        l3_top1_score=top1_score,
        l3_top2_score=top2_score,
        l3_margin=l3_margin,
    )


def _decide_hm_route_v2(
    *,
    question: str,
    l3_hits: list[SegmentHit],
) -> HMRouteDecision:
    temporal_local, quoted_anchor, fine_motion = _is_local_temporal_question(question)
    top1_score = float(l3_hits[0].score) if l3_hits else None
    top2_score = float(l3_hits[1].score) if len(l3_hits) > 1 else None
    l3_margin = (top1_score - top2_score) if top1_score is not None and top2_score is not None else None
    ambiguous_l3 = bool(l3_margin is not None and l3_margin < 0.02)

    lowered = f" {_strip_question_boilerplate(question).lower()} "
    explicit_local_detail = any(
        term in lowered
        for term in (
            " hand movement ",
            " movement occurs ",
            " gesture ",
            " position ",
            " pose ",
            " first time she appeared ",
            " first time he appeared ",
            " when he first appears ",
            " when she first appears ",
        )
    )

    reasons: list[str] = []
    route = "l3_only"
    if quoted_anchor and explicit_local_detail and ambiguous_l3:
        route = "l3_to_l2_local"
        reasons.extend(["quoted_anchor", "explicit_local_detail", "ambiguous_l3"])

    return HMRouteDecision(
        route=route,
        reasons=reasons,
        temporal_local=temporal_local,
        quoted_anchor=quoted_anchor,
        fine_motion=fine_motion,
        ambiguous_l3=ambiguous_l3,
        l3_top1_score=top1_score,
        l3_top2_score=top2_score,
        l3_margin=l3_margin,
    )


def _decide_hm_route_v3(
    *,
    question: str,
    metadata: dict[str, Any],
    l3_hits: list[SegmentHit],
) -> HMRouteDecision:
    temporal_local, quoted_anchor, fine_motion = _is_local_temporal_question(question)
    top1_score = float(l3_hits[0].score) if l3_hits else None
    top2_score = float(l3_hits[1].score) if len(l3_hits) > 1 else None
    l3_margin = (top1_score - top2_score) if top1_score is not None and top2_score is not None else None
    ambiguous_l3 = bool(l3_margin is not None and l3_margin < 0.02)

    level = str(metadata.get("level") or "")
    try:
        duration_group = int(metadata.get("duration_group") or 0)
    except (TypeError, ValueError):
        duration_group = 0

    is_l1 = level == "L1-Perception"
    is_very_long = duration_group >= 3600
    route = "l3_to_l2_local" if (is_l1 and is_very_long and quoted_anchor) else "l3_only"

    reasons: list[str] = []
    if route == "l3_to_l2_local":
        reasons = ["lvl_L1", "dur_3600", "quoted_anchor"]

    return HMRouteDecision(
        route=route,
        reasons=reasons,
        temporal_local=temporal_local,
        quoted_anchor=quoted_anchor,
        fine_motion=fine_motion,
        ambiguous_l3=ambiguous_l3,
        l3_top1_score=top1_score,
        l3_top2_score=top2_score,
        l3_margin=l3_margin,
    )


def _decide_hm_runtime_router(
    *,
    question: str,
    artifacts: VideoArtifacts,
    l3_hits: list[SegmentHit],
) -> HMRouteDecision:
    temporal_local, quoted_anchor, fine_motion = _is_local_temporal_question(question)
    top1_score = float(l3_hits[0].score) if l3_hits else None
    top2_score = float(l3_hits[1].score) if len(l3_hits) > 1 else None
    l3_margin = (top1_score - top2_score) if top1_score is not None and top2_score is not None else None
    ambiguous_l3 = bool(l3_margin is not None and l3_margin < 0.02)

    video_duration_sec = float(artifacts.timestamps[-1]) if len(artifacts.timestamps) > 0 else 0.0
    is_long_video = video_duration_sec >= 600.0
    route = "l3_to_l2_local" if (is_long_video and quoted_anchor) else "l3_only"
    reasons: list[str] = []
    if route == "l3_to_l2_local":
        reasons = ["runtime_dur_ge_600", "quoted_anchor"]

    return HMRouteDecision(
        route=route,
        reasons=reasons,
        temporal_local=temporal_local,
        quoted_anchor=quoted_anchor,
        fine_motion=fine_motion,
        ambiguous_l3=ambiguous_l3,
        l3_top1_score=top1_score,
        l3_top2_score=top2_score,
        l3_margin=l3_margin,
    )


def _heuristic_vgent_plan(question: str, options: list[str]) -> VgentQueryPlan:
    stripped_question = _strip_question_boilerplate(question)
    et_plan = _build_et_plan(question, options)
    keywords = _dedupe_text_items([stripped_question, *et_plan.l2_queries, *options], max_items=6)
    tool = "none"
    lower = stripped_question.lower()
    if "how many" in lower or "count" in lower or "number of" in lower:
        tool = "count"
    elif "order" in lower or "sequence" in lower or "first" in lower or "then" in lower or "finally" in lower:
        tool = "order"
    subqueries: list[str] = []
    if et_plan.anchor_query and et_plan.anchor_query != stripped_question:
        subqueries.append(f"Does the video show {et_plan.anchor_query}?")
    for keyword in keywords[:3]:
        if keyword.lower() == stripped_question.lower():
            continue
        subqueries.append(f"Does the video show {keyword}?")
    if tool == "order":
        subqueries.extend([f"Does the video show {option}?" for option in options[:3]])
    return VgentQueryPlan(
        primary_query=(et_plan.anchor_query or stripped_question),
        retrieval_queries=_dedupe_text_items([stripped_question, *(et_plan.l3_queries or []), *keywords], max_items=8),
        subqueries=_dedupe_text_items(subqueries, max_items=4),
        keywords=_dedupe_text_items(keywords, max_items=6),
        multiple=(et_plan.plan_type in {"before_after_reasoning", "sequence_ordering"}),
        time_focus=("begin" if "beginning" in lower or "start" in lower else "end" if "end" in lower else "none"),
        tool=tool,
        global_reasoning=bool("main" in lower or "overall" in lower or "whole video" in lower),
        candidates_necessary=bool(tool in {"count", "order"}),
    )


VGENT_CONTROLLER_PROMPT = """You are planning retrieval for hierarchical video question answering.
Return JSON only.

Question: {question}
Candidates: {candidates}

Output schema:
{{
  "keywords": ["..."],
  "multiple": "yes" | "no",
  "time": "begin" | "end" | "none",
  "tool": "count" | "order" | "none",
  "candidates_necessary": "yes" | "no",
  "global": "yes" | "no",
  "primary_query": "...",
  "retrieval_queries": ["..."]
}}

Rules:
- Keep keywords short.
- Do not repeat equivalent phrases.
- retrieval_queries should be concrete text queries useful for retrieval.
- Use candidates only if they are necessary for retrieval.
"""


VGENT_SUBQUERY_PROMPT = """You are refining retrieved evidence for video question answering.
Return JSON only.

Question: {question}
Candidates: {candidates}
Primary query: {primary_query}
Keywords: {keywords}
Tool: {tool}
Multiple segments: {multiple}

Output schema:
{{
  "subqueries": ["..."]
}}

Rules:
- Write short verification subqueries.
- Prefer yes/no checks.
- For order questions, ask about the presence of key events/items.
- Do not mention timestamps.
- Produce at most 4 subqueries.
"""


def _build_vgent_plan(
    *,
    question: str,
    options: list[str],
    text_answerer: Any | None,
) -> VgentQueryPlan:
    fallback = _heuristic_vgent_plan(question, options)
    if text_answerer is None:
        return fallback

    candidates = "; ".join(options)
    try:
        controller_raw = text_answerer.generate_text(
            prompt=VGENT_CONTROLLER_PROMPT.format(question=question.strip(), candidates=candidates),
            max_new_tokens=192,
        ).raw_text
        controller_payload = _extract_json_object(controller_raw) or {}
    except Exception:
        controller_raw = None
        controller_payload = {}

    keywords = _dedupe_text_items(
        [*(controller_payload.get("keywords") or []), *fallback.keywords],
        max_items=6,
    )
    retrieval_queries = _dedupe_text_items(
        [str(controller_payload.get("primary_query") or ""), *(controller_payload.get("retrieval_queries") or []), *fallback.retrieval_queries],
        max_items=8,
    )
    primary_query = str(controller_payload.get("primary_query") or fallback.primary_query).strip() or fallback.primary_query
    multiple = str(controller_payload.get("multiple", "yes" if fallback.multiple else "no")).strip().lower() == "yes"
    time_focus = str(controller_payload.get("time", fallback.time_focus)).strip().lower()
    if time_focus not in {"begin", "end", "none"}:
        time_focus = "none"
    tool = str(controller_payload.get("tool", fallback.tool)).strip().lower()
    if tool not in {"count", "order", "none"}:
        tool = fallback.tool
    global_reasoning = str(controller_payload.get("global", "yes" if fallback.global_reasoning else "no")).strip().lower() == "yes"
    candidates_necessary = (
        str(controller_payload.get("candidates_necessary", "yes" if fallback.candidates_necessary else "no")).strip().lower() == "yes"
    )

    try:
        subquery_raw = text_answerer.generate_text(
            prompt=VGENT_SUBQUERY_PROMPT.format(
                question=question.strip(),
                candidates=candidates,
                primary_query=primary_query,
                keywords=", ".join(keywords),
                tool=tool,
                multiple=("yes" if multiple else "no"),
            ),
            max_new_tokens=192,
        ).raw_text
        subquery_payload = _extract_json_object(subquery_raw) or {}
    except Exception:
        subquery_raw = None
        subquery_payload = {}
    subqueries = _dedupe_text_items(
        [*(subquery_payload.get("subqueries") or []), *fallback.subqueries],
        max_items=4,
    )

    return VgentQueryPlan(
        primary_query=primary_query,
        retrieval_queries=(retrieval_queries or fallback.retrieval_queries),
        subqueries=subqueries,
        keywords=keywords,
        multiple=multiple,
        time_focus=time_focus,
        tool=tool,
        global_reasoning=global_reasoning,
        candidates_necessary=candidates_necessary,
        raw_controller=controller_raw,
        raw_subqueries=subquery_raw,
    )


def _load_examples(manifest_path: Path, *, limit: int | None = None) -> list[BaselineExample]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = payload["rows"]
    if limit is not None:
        rows = rows[:limit]
    examples: list[BaselineExample] = []
    for row in rows:
        examples.append(
            BaselineExample(
                example_id=str(row["id"]),
                video_id=str(row["video_id"]),
                video_path=str(row["video_path"]),
                question=str(row["question"]),
                options=[str(option) for option in row["candidates"]],
                correct_index=int(row["correct_choice"]) if "correct_choice" in row else None,
                metadata={
                    "split": payload.get("source_split"),
                    "question_category": row.get("question_category"),
                    "level": row.get("level"),
                    "duration_group": row.get("duration_group"),
                    "duration": row.get("duration"),
                    "topic_category": row.get("topic_category"),
                    "subtitle_path": row.get("subtitle_path"),
                    "starting_timestamp_for_subtitles": row.get("starting_timestamp_for_subtitles"),
                },
            )
        )
    return examples


def _validate_video_root(video_root: Path, examples: list[BaselineExample]) -> None:
    if not video_root.exists():
        raise RuntimeError(f"Video root does not exist: {video_root}")
    missing = [example.video_path for example in examples if not (video_root / example.video_path).exists()]
    if missing:
        sample = ", ".join(sorted(set(missing))[:8])
        raise RuntimeError(f"{len(set(missing))} referenced videos are missing under {video_root}. Sample: {sample}")


def _segment_overlaps(hit_start: int, hit_end: int, segment: Segment) -> bool:
    return not (segment.end_index < hit_start or segment.start_index > hit_end)


def _temporal_nms_frame_hits(
    frame_hits: list[FrameHit],
    *,
    max_hits: int,
    min_gap_sec: float,
) -> list[FrameHit]:
    selected: list[FrameHit] = []
    for hit in sorted(frame_hits, key=lambda item: float(item.score), reverse=True):
        if len(selected) >= max_hits:
            break
        if all(abs(float(hit.time_sec) - float(kept.time_sec)) >= min_gap_sec for kept in selected):
            selected.append(hit)
    return sorted(selected, key=lambda item: int(item.frame_index))


def _frame_hits_for_indices(
    *,
    query_embedding: torch.Tensor,
    frame_embeddings: torch.Tensor,
    timestamps: np.ndarray,
    indices: list[int],
    max_frames: int,
    required_indices: list[int] | None = None,
) -> list[FrameHit]:
    unique_indices = sorted(set(int(index) for index in indices if 0 <= int(index) < frame_embeddings.shape[0]))
    if not unique_indices:
        return []
    scores = torch.matmul(frame_embeddings[unique_indices], query_embedding).cpu().numpy()
    score_by_index = {index: float(scores[offset]) for offset, index in enumerate(unique_indices)}
    required = [int(index) for index in (required_indices or []) if int(index) in score_by_index]
    selected = set(required)
    remaining = sorted(
        (index for index in unique_indices if index not in selected),
        key=lambda index: score_by_index[index],
        reverse=True,
    )
    for index in remaining:
        if len(selected) >= max_frames:
            break
        selected.add(index)
    return [
        FrameHit(frame_index=index, time_sec=float(timestamps[index]), score=float(score_by_index[index]))
        for index in sorted(selected)
    ]


def _rank_score_by_id(items: list[tuple[Any, float]], *, key_fn: Any) -> dict[Any, float]:
    ordered = sorted(items, key=key_fn, reverse=True)
    if not ordered:
        return {}
    if len(ordered) == 1:
        return {ordered[0][0]: 1.0}
    return {item[0]: 1.0 - (rank / float(len(ordered) - 1)) for rank, item in enumerate(ordered)}


def _segment_time_overlap(start_a: float, end_a: float, start_b: float, end_b: float) -> bool:
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


def _get_viclip_encoder() -> ViCLIPEncoder:
    global _VICLIP_ENCODER
    if _VICLIP_ENCODER is None:
        _VICLIP_ENCODER = ViCLIPEncoder()
    return _VICLIP_ENCODER


class AblationRetriever:
    def __init__(
        self,
        *,
        feature_root: Path,
        derived_cache_root: Path,
        video_root: Path,
        config: AblationRunConfig,
        encoder_device: str,
    ) -> None:
        self.feature_root = feature_root
        self.derived_cache_root = derived_cache_root
        self.video_root = video_root
        self.config = config
        self.pipeline_config = PipelineConfig(
            sample_fps=config.sample_fps,
            window_seconds=config.l2_window_seconds,
            window_stride_seconds=config.l2_stride_seconds,
            layer2_pooling="mean",
            top_windows=config.top_l2_segments,
            max_evidence_frames=config.max_frames,
            image_max_size=config.image_max_size,
            device=encoder_device,
        )
        self._encoder = OpenCLIPEncoder(device=encoder_device)
        self._video_cache: dict[str, VideoArtifacts] = {}
        self._query_cache: dict[str, torch.Tensor] = {}

    def unload(self) -> None:
        del self._encoder
        self._video_cache.clear()
        self._query_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _query_embedding(self, question: str, options: list[str]) -> torch.Tensor:
        query_text = build_query_text(question, options)
        cache_key = f"openclip::{query_text}"
        cached = self._query_cache.get(cache_key)
        if cached is not None:
            return cached
        embedding = self._encoder.encode_texts([query_text])[0].cpu()
        self._query_cache[cache_key] = embedding
        return embedding

    def _text_embedding(self, text: str) -> torch.Tensor:
        query_text = _strip_question_boilerplate(text)
        cache_key = f"openclip_text::{query_text}"
        cached = self._query_cache.get(cache_key)
        if cached is not None:
            return cached
        embedding = self._encoder.encode_texts([query_text])[0].cpu()
        self._query_cache[cache_key] = embedding
        return embedding

    def _load_video(self, example: BaselineExample) -> VideoArtifacts:
        cached = self._video_cache.get(example.video_id)
        if cached is not None:
            return cached
        cache_dir = self.feature_root / example.video_id
        timestamps = np.load(cache_dir / "timestamps.npy").astype(np.float32)
        frame_embeddings = torch.load(cache_dir / "frame_embeddings.pt", map_location="cpu").float()
        metadata = json.loads((cache_dir / "metadata.json").read_text(encoding="utf-8"))
        artifacts = VideoArtifacts(
            video_id=example.video_id,
            video_path=self.video_root / example.video_path,
            timestamps=timestamps,
            frame_embeddings=frame_embeddings,
            native_fps=float(metadata["native_fps"]),
        )
        self._video_cache[example.video_id] = artifacts
        return artifacts

    def _video_derived_dir(self, video_id: str) -> Path:
        return self.derived_cache_root / video_id

    def _stable_hash(self, payload: dict[str, Any]) -> str:
        raw = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha1(raw).hexdigest()[:12]

    def _serialize_segments(self, segments: list[Segment]) -> list[dict[str, Any]]:
        return [
            {
                "segment_id": segment.segment_id,
                "start_index": int(segment.start_index),
                "end_index": int(segment.end_index),
                "start_time_sec": float(segment.start_time_sec),
                "end_time_sec": float(segment.end_time_sec),
                "duration_sec": float(segment.duration_sec),
            }
            for segment in segments
        ]

    def _deserialize_segments(self, payload: list[dict[str, Any]]) -> list[Segment]:
        return [Segment(**item) for item in payload]

    def _l2_cache_dir(self, video_id: str) -> Path:
        key = self._stable_hash(
            {
                "version": 3,
                "sample_fps": self.config.sample_fps,
                "segmentation": self.config.l2_segmentation,
                "window_seconds": self.config.l2_window_seconds,
                "stride_seconds": self.config.l2_stride_seconds,
                "local_min_duration_sec": self.config.l2_local_min_duration_sec,
                "local_max_duration_sec": self.config.l2_local_max_duration_sec,
                "local_fast_kernel_size": self.config.l2_local_fast_kernel_size,
                "local_slow_kernel_size": self.config.l2_local_slow_kernel_size,
                "local_peak_percentile": self.config.l2_local_peak_percentile,
                "scoring": self.config.l2_scoring,
                "frame_score_top_m": self.config.l2_frame_score_top_m,
                "frame_score_temperature": self.config.l2_frame_score_temperature,
                "pooling": "mean",
            }
        )
        return self._video_derived_dir(video_id) / f"l2_{key}"

    def _l3_cache_dir(self, video_id: str) -> Path:
        key = self._stable_hash(
            {
                "version": 3,
                "sample_fps": self.config.sample_fps,
                "method": self.config.l3_segmentation,
                "window_seconds": self.config.l3_window_seconds,
                "stride_seconds": self.config.l3_stride_seconds,
                "defaults": {
                    "rho": 0.05,
                    "trailing_window": 30,
                    "peak_neighborhood": 2,
                    "smooth_kernel_size": 3,
                    "min_duration_sec": 15.0,
                    "max_duration_sec": 60.0,
                    "w_sem": 0.7,
                    "w_mot": 0.3,
                    "k_strong": 2.5,
                    "k_weak": 1.0,
                    "robust_min_scale": 0.01,
                    "zscore_clip": 6.0,
                },
                "pooling": "mean",
            }
        )
        return self._video_derived_dir(video_id) / f"l3_{key}"

    def _motion_cache_path(self, video_id: str) -> Path:
        key = self._stable_hash({"version": 1, "sample_fps": self.config.sample_fps})
        return self._video_derived_dir(video_id) / f"motion_{key}.npy"

    def _graph_cache_dir(self, video_id: str) -> Path:
        key = self._stable_hash(
            {
                "version": 1,
                "sample_fps": self.config.sample_fps,
                "l2_segmentation": self.config.l2_segmentation,
                "l2_window_seconds": self.config.l2_window_seconds,
                "l2_stride_seconds": self.config.l2_stride_seconds,
                "l2_local_min_duration_sec": self.config.l2_local_min_duration_sec,
                "l2_local_max_duration_sec": self.config.l2_local_max_duration_sec,
                "l2_local_fast_kernel_size": self.config.l2_local_fast_kernel_size,
                "l2_local_slow_kernel_size": self.config.l2_local_slow_kernel_size,
                "l2_local_peak_percentile": self.config.l2_local_peak_percentile,
                "graph_frames_per_segment": self.config.graph_frames_per_segment,
                "include_subtitles": self.config.include_subtitles,
            }
        )
        return self._video_derived_dir(video_id) / f"grapha_{key}"

    def _viclip_l2_cache_dir(self, video_id: str) -> Path:
        key = self._stable_hash(
            {
                "version": 1,
                "encoder": "viclip",
                "model_id": "OpenGVLab/ViCLIP-L-14-hf",
                "sample_fps": self.config.sample_fps,
                "segmentation": self.config.l2_segmentation,
                "window_seconds": self.config.l2_window_seconds,
                "stride_seconds": self.config.l2_stride_seconds,
                "local_min_duration_sec": self.config.l2_local_min_duration_sec,
                "local_max_duration_sec": self.config.l2_local_max_duration_sec,
                "local_fast_kernel_size": self.config.l2_local_fast_kernel_size,
                "local_slow_kernel_size": self.config.l2_local_slow_kernel_size,
                "local_peak_percentile": self.config.l2_local_peak_percentile,
            }
        )
        return self._video_derived_dir(video_id) / f"l2_viclip_{key}"

    def _ensure_l2(self, artifacts: VideoArtifacts) -> None:
        if artifacts.l2_segments is not None and artifacts.l2_embeddings is not None:
            return
        cache_dir = self._l2_cache_dir(artifacts.video_id)
        segments_path = cache_dir / "segments.json"
        embeddings_path = cache_dir / "embeddings.pt"
        if segments_path.exists() and embeddings_path.exists():
            artifacts.l2_segments = self._deserialize_segments(json.loads(segments_path.read_text(encoding="utf-8")))
            artifacts.l2_embeddings = torch.load(embeddings_path, map_location="cpu").float()
            return

        if self.config.l2_segmentation == "fixed":
            sampled = type("Sampled", (), {"timestamps": artifacts.timestamps})()
            segments = build_window_segments(sampled, self.pipeline_config)
        elif self.config.l2_segmentation == "l3_local_contrast":
            self._ensure_l3(artifacts)
            assert artifacts.l3_segments is not None
            segments = segment_l3_local_contrast_windows(
                timestamps=artifacts.timestamps,
                frame_embeddings=artifacts.frame_embeddings,
                l3_segments=artifacts.l3_segments,
                min_duration_sec=self.config.l2_local_min_duration_sec,
                max_duration_sec=self.config.l2_local_max_duration_sec,
                fast_kernel_size=self.config.l2_local_fast_kernel_size,
                slow_kernel_size=self.config.l2_local_slow_kernel_size,
                peak_percentile=self.config.l2_local_peak_percentile,
                prefix="l2_l3local_contrast",
            )
        else:
            raise ValueError(f"Unsupported l2_segmentation: {self.config.l2_segmentation}")

        embeddings = pool_segments(artifacts.frame_embeddings, segments, pooling="mean")
        artifacts.l2_segments = segments
        artifacts.l2_embeddings = embeddings
        cache_dir.mkdir(parents=True, exist_ok=True)
        segments_path.write_text(json.dumps(self._serialize_segments(segments), indent=2), encoding="utf-8")
        torch.save(embeddings, embeddings_path)

    def _ensure_viclip_l2_embeddings(self, artifacts: VideoArtifacts) -> torch.Tensor:
        self._ensure_l2(artifacts)
        assert artifacts.l2_segments is not None
        cache_dir = self._viclip_l2_cache_dir(artifacts.video_id)
        segments_path = cache_dir / "segments.json"
        embeddings_path = cache_dir / "embeddings.pt"
        if segments_path.exists() and embeddings_path.exists():
            return torch.load(embeddings_path, map_location="cpu").float()

        encoder = _get_viclip_encoder()
        embedding_rows: list[torch.Tensor] = []
        for index, segment in enumerate(artifacts.l2_segments):
            candidate_budget = min(
                VICLIP_L2_MAX_FRAMES,
                max(1, int(segment.end_index) - int(segment.start_index) + 1),
            )
            frames, _, _, _ = _sample_uniform_video_frames(
                video_path=artifacts.video_path,
                frame_budget=candidate_budget,
                start_time_sec=float(segment.start_time_sec),
                end_time_sec=float(segment.end_time_sec),
            )
            if len(frames) >= encoder.num_frames:
                step_indices = torch.linspace(0, len(frames) - 1, steps=encoder.num_frames)
                selected = [frames[int(round(float(step)))] for step in step_indices.tolist()]
            else:
                selected = list(frames)
                while len(selected) < encoder.num_frames:
                    selected.append(selected[-1])
            clip_embedding = encoder.encode_video_clips([selected], batch_size=1).float().cpu()
            embedding_rows.append(clip_embedding[0])
            del frames
            del selected
            if (index + 1) % 16 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        embeddings = torch.stack(embedding_rows, dim=0) if embedding_rows else torch.empty((0, 0), dtype=torch.float32)
        cache_dir.mkdir(parents=True, exist_ok=True)
        segments_path.write_text(json.dumps(self._serialize_segments(artifacts.l2_segments), indent=2), encoding="utf-8")
        torch.save(embeddings, embeddings_path)
        return embeddings

    def _ensure_l3(self, artifacts: VideoArtifacts) -> None:
        if artifacts.l3_segments is not None and artifacts.l3_embeddings is not None:
            return
        cache_dir = self._l3_cache_dir(artifacts.video_id)
        segments_path = cache_dir / "segments.json"
        embeddings_path = cache_dir / "embeddings.pt"
        if segments_path.exists() and embeddings_path.exists():
            artifacts.l3_segments = self._deserialize_segments(json.loads(segments_path.read_text(encoding="utf-8")))
            artifacts.l3_embeddings = torch.load(embeddings_path, map_location="cpu").float()
            return

        if self.config.l3_segmentation == "fixed":
            sampled = type("Sampled", (), {"timestamps": artifacts.timestamps})()
            l3_pipeline_config = PipelineConfig(
                sample_fps=self.config.sample_fps,
                window_seconds=self.config.l3_window_seconds,
                window_stride_seconds=self.config.l3_stride_seconds,
                layer2_pooling="mean",
                top_windows=self.config.top_l3_segments,
                max_evidence_frames=self.config.max_frames,
                image_max_size=self.config.image_max_size,
                device=self.pipeline_config.device,
            )
            segments = build_window_segments(sampled, l3_pipeline_config)
        elif self.config.l3_segmentation == "fused_adaptive":
            motion_cache_path = self._motion_cache_path(artifacts.video_id)
            if motion_cache_path.exists():
                motion_energy = np.load(motion_cache_path).astype(np.float32)
            else:
                target_frame_indices = [int(round(float(ts) * float(artifacts.native_fps))) for ts in artifacts.timestamps.tolist()]
                motion_energy = compute_motion_energy_for_frame_indices(
                    artifacts.video_path,
                    target_frame_indices=target_frame_indices,
                ).astype(np.float32)
                motion_cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(motion_cache_path, motion_energy)

            result = segment_fused_adaptive_peaks(
                timestamps=artifacts.timestamps,
                frame_embeddings=artifacts.frame_embeddings,
                motion_energy=motion_energy,
                prefix="l3_fused_adaptive",
            )
            segments = result["segments"]
        else:
            raise ValueError(f"Unsupported l3_segmentation: {self.config.l3_segmentation}")
        embeddings = pool_segments(artifacts.frame_embeddings, segments, pooling="mean")
        artifacts.l3_segments = segments
        artifacts.l3_embeddings = embeddings
        cache_dir.mkdir(parents=True, exist_ok=True)
        segments_path.write_text(json.dumps(self._serialize_segments(segments), indent=2), encoding="utf-8")
        torch.save(embeddings, embeddings_path)

    def _ensure_l2_graph(
        self,
        *,
        example: BaselineExample,
        artifacts: VideoArtifacts,
        graph_answerer: Any,
        subtitle_root: str | Path | None,
        subtitle_tar: str | Path | None,
    ) -> None:
        if artifacts.l2_graph_nodes is not None and artifacts.l2_graph_embeddings is not None:
            return
        self._ensure_l2(artifacts)
        assert artifacts.l2_segments is not None

        cache_dir = self._graph_cache_dir(artifacts.video_id)
        nodes_path = cache_dir / "nodes.json"
        embeddings_path = cache_dir / "embeddings.pt"
        if nodes_path.exists() and embeddings_path.exists():
            artifacts.l2_graph_nodes = json.loads(nodes_path.read_text(encoding="utf-8"))
            artifacts.l2_graph_embeddings = torch.load(embeddings_path, map_location="cpu").float()
            return

        normalized_subtitles: list[dict[str, Any]] = []
        if self.config.include_subtitles:
            subtitle_path = example.metadata.get("subtitle_path")
            if subtitle_path:
                subtitles = _load_subtitles(
                    subtitle_path=str(subtitle_path),
                    subtitle_root=subtitle_root,
                    subtitle_tar=subtitle_tar,
                )
                normalized_subtitles = _normalize_subtitles(
                    subtitles,
                    starting_timestamp_for_subtitles=float(example.metadata.get("starting_timestamp_for_subtitles", 0.0)),
                    duration=(float(example.metadata["duration"]) if example.metadata.get("duration") is not None else None),
                )

        nodes: list[dict[str, Any]] = []
        graph_texts: list[str] = []
        for segment in artifacts.l2_segments:
            sample_indices = _segment_sample_indices(segment, self.config.graph_frames_per_segment)
            frames, _, _ = load_selected_video_frames(
                artifacts.video_path,
                sample_fps=self.config.sample_fps,
                target_indices=sample_indices,
                image_max_size=self.config.image_max_size,
            )
            generation = graph_answerer.generate_text_from_frames(
                frames=frames,
                prompt=GRAPH_PROMPT,
                max_new_tokens=96,
            )
            payload = _normalize_graph_payload(generation.raw_text)
            overlapping_subtitles = [
                subtitle["text"]
                for subtitle in normalized_subtitles
                if subtitle["text"]
                and not (
                    float(subtitle["end"]) < float(segment.start_time_sec)
                    or float(subtitle["start"]) > float(segment.end_time_sec)
                )
            ]
            subtitle_lines = list(dict.fromkeys(text for text in overlapping_subtitles if text))
            subtitle_text = "\n".join(subtitle_lines)
            subtitle_terms = _normalized_graph_terms(subtitle_lines)
            graph_parts = [
                *payload["entities"],
                *payload["actions"],
                *payload["scenes"],
            ]
            if subtitle_text:
                graph_parts.append(subtitle_text)
            graph_text = "; ".join(part for part in graph_parts if part)
            node = {
                "segment_id": segment.segment_id,
                "start_index": int(segment.start_index),
                "end_index": int(segment.end_index),
                "start_time_sec": float(segment.start_time_sec),
                "end_time_sec": float(segment.end_time_sec),
                "duration_sec": float(segment.duration_sec),
                "entities": payload["entities"],
                "actions": payload["actions"],
                "scenes": payload["scenes"],
                "subtitle_text": subtitle_text,
                "subtitle_terms": subtitle_terms,
                "exact_terms": _tokenize_for_graph(" ".join([*payload["entities"], *payload["actions"], *payload["scenes"], subtitle_text])),
                "normalized_terms": _normalized_graph_terms([*payload["entities"], *payload["actions"], *payload["scenes"], subtitle_text]),
                "graph_text": graph_text,
                "raw_text": generation.raw_text,
            }
            nodes.append(node)
            graph_texts.append(graph_text or "empty segment")

        graph_embeddings = self._encoder.encode_texts(graph_texts).cpu()
        cache_dir.mkdir(parents=True, exist_ok=True)
        nodes_path.write_text(json.dumps(nodes, indent=2, ensure_ascii=False), encoding="utf-8")
        torch.save(graph_embeddings, embeddings_path)
        artifacts.l2_graph_nodes = nodes
        artifacts.l2_graph_embeddings = graph_embeddings

    def _graph_node_hits(
        self,
        *,
        artifacts: VideoArtifacts,
        query_text: str,
        l3_hits: list[SegmentHit],
    ) -> list[SegmentHit]:
        assert artifacts.l2_graph_nodes is not None
        assert artifacts.l2_graph_embeddings is not None
        query_embedding = self._text_embedding(query_text)
        query_tokens = _tokenize_for_graph(query_text)
        query_norm_terms = _normalized_graph_terms([query_text])
        query_token_set = set(query_tokens)
        query_norm_set = set(query_norm_terms)

        allowed_nodes: list[tuple[int, dict[str, Any]]] = []
        for idx, node in enumerate(artifacts.l2_graph_nodes):
            if any(
                not (
                    float(node["end_time_sec"]) < float(hit.start_time_sec)
                    or float(node["start_time_sec"]) > float(hit.end_time_sec)
                )
                for hit in l3_hits
            ):
                allowed_nodes.append((idx, node))
        if not allowed_nodes:
            return []

        node_indices = [idx for idx, _ in allowed_nodes]
        node_embeddings = artifacts.l2_graph_embeddings[node_indices]
        embedding_scores = torch.matmul(node_embeddings, query_embedding).cpu().numpy()
        hits: list[SegmentHit] = []
        for offset, (node_idx, node) in enumerate(allowed_nodes):
            exact_terms = set(str(term) for term in node.get("exact_terms", []))
            normalized_terms = set(str(term) for term in node.get("normalized_terms", []))
            subtitle_terms = set(str(term) for term in node.get("subtitle_terms", []))
            exact_score = (len(query_token_set & exact_terms) / max(len(query_token_set), 1)) if query_token_set else 0.0
            lemma_score = (len(query_norm_set & normalized_terms) / max(len(query_norm_set), 1)) if query_norm_set else 0.0
            subtitle_score = (len(query_norm_set & subtitle_terms) / max(len(query_norm_set), 1)) if query_norm_set else 0.0
            score = (
                1.0 * exact_score
                + 0.7 * lemma_score
                + 0.8 * subtitle_score
                + 0.6 * float(embedding_scores[offset])
            )
            hits.append(
                SegmentHit(
                    segment_id=str(node["segment_id"]),
                    score=float(score),
                    start_index=int(node["start_index"]),
                    end_index=int(node["end_index"]),
                    start_time_sec=float(node["start_time_sec"]),
                    end_time_sec=float(node["end_time_sec"]),
                )
            )
        return sorted(hits, key=lambda item: float(item.score), reverse=True)[: self.config.top_l2_segments]

    def _frame_hits_from_indices(
        self,
        *,
        artifacts: VideoArtifacts,
        query_embedding: torch.Tensor,
        allowed_indices: list[int] | None,
        required_indices: list[int] | None = None,
    ) -> list[FrameHit]:
        if required_indices:
            return _frame_hits_for_indices(
                query_embedding=query_embedding,
                frame_embeddings=artifacts.frame_embeddings,
                timestamps=artifacts.timestamps,
                indices=allowed_indices or required_indices,
                max_frames=self.config.max_frames,
                required_indices=required_indices,
            )
        return retrieve_top_frames(
            query_embedding=query_embedding,
            frame_embeddings=artifacts.frame_embeddings,
            timestamps=artifacts.timestamps,
            top_k=self.config.max_frames,
            allowed_indices=allowed_indices,
        )

    def _candidate_l2_indices_from_l3_hits(
        self,
        *,
        artifacts: VideoArtifacts,
        l3_hits: list[SegmentHit],
    ) -> set[int]:
        assert artifacts.l2_segments is not None
        selected: set[int] = set()
        for segment_index, l2_segment in enumerate(artifacts.l2_segments):
            if any(
                _segment_time_overlap(
                    l2_segment.start_time_sec,
                    l2_segment.end_time_sec,
                    l3_hit.start_time_sec,
                    l3_hit.end_time_sec,
                )
                for l3_hit in l3_hits
            ):
                selected.add(segment_index)
        return selected

    def _rerank_l3_hits_with_l2(
        self,
        *,
        artifacts: VideoArtifacts,
        query_embedding: torch.Tensor,
        target_text: str,
        l3_hits: list[SegmentHit],
    ) -> tuple[list[SegmentHit], dict[str, Any]]:
        if not l3_hits:
            return [], {"l2_candidates": []}

        self._ensure_l2(artifacts)
        assert artifacts.l2_segments is not None
        candidate_indices = self._candidate_l2_indices_from_l3_hits(artifacts=artifacts, l3_hits=l3_hits)
        if not candidate_indices:
            return l3_hits[: self.config.l3_rerank_keep], {"l2_candidates": []}

        frame_scores = torch.matmul(artifacts.frame_embeddings, query_embedding).cpu()
        l3_prior_rank = _rank_score_by_id(
            [(str(hit.segment_id), float(hit.score)) for hit in l3_hits],
            key_fn=lambda item: float(item[1]),
        )

        viclip_query_embedding: torch.Tensor | None = None
        viclip_l2_embeddings: torch.Tensor | None = None
        if self.config.l2_rerank_encoder == "viclip":
            viclip_query_embedding = _get_viclip_encoder().encode_texts([target_text])[0].float().cpu()
            viclip_l2_embeddings = self._ensure_viclip_l2_embeddings(artifacts)

        l2_items: list[dict[str, Any]] = []
        for segment_index in sorted(candidate_indices):
            segment = artifacts.l2_segments[segment_index]
            parent_segment_id = None
            for l3_hit in l3_hits:
                if _segment_time_overlap(
                    segment.start_time_sec,
                    segment.end_time_sec,
                    l3_hit.start_time_sec,
                    l3_hit.end_time_sec,
                ):
                    parent_segment_id = str(l3_hit.segment_id)
                    break
            if parent_segment_id is None:
                continue
            if self.config.l2_rerank_encoder == "viclip":
                assert viclip_query_embedding is not None
                assert viclip_l2_embeddings is not None
                l2_score = float(torch.dot(viclip_l2_embeddings[int(segment_index)], viclip_query_embedding).item())
            else:
                l2_score = _segment_topm_score(
                    frame_scores=frame_scores,
                    start_index=int(segment.start_index),
                    end_index=int(segment.end_index),
                    top_m=L2_SCORE_TOP_M,
                )
            l2_items.append(
                {
                    "segment_index": int(segment_index),
                    "segment_id": str(segment.segment_id),
                    "parent_segment_id": parent_segment_id,
                    "start_time_sec": float(segment.start_time_sec),
                    "end_time_sec": float(segment.end_time_sec),
                    "raw_score": float(l2_score),
                }
            )

        l2_rank = _rank_score_by_id(
            [(int(item["segment_index"]), float(item["raw_score"])) for item in l2_items],
            key_fn=lambda item: float(item[1]),
        )
        for item in l2_items:
            item["rank_score"] = float(l2_rank.get(int(item["segment_index"]), 0.0))

        parent_scores: dict[str, list[float]] = {}
        for item in l2_items:
            parent_scores.setdefault(str(item["parent_segment_id"]), []).append(float(item["rank_score"]))

        reranked_hits: list[SegmentHit] = []
        for hit in l3_hits:
            segment_id = str(hit.segment_id)
            child_scores = sorted(parent_scores.get(segment_id, []), reverse=True)
            best_child = child_scores[0] if child_scores else 0.0
            top2_sum = sum(child_scores[:2]) if child_scores else 0.0
            prior_rank = float(l3_prior_rank.get(segment_id, 0.0))
            score = best_child + (0.35 * top2_sum) + (0.05 * prior_rank)
            reranked_hits.append(
                SegmentHit(
                    segment_id=segment_id,
                    score=float(score),
                    start_index=int(hit.start_index),
                    end_index=int(hit.end_index),
                    start_time_sec=float(hit.start_time_sec),
                    end_time_sec=float(hit.end_time_sec),
                )
            )
        reranked_hits.sort(key=lambda item: float(item.score), reverse=True)
        l2_items.sort(key=lambda item: float(item["rank_score"]), reverse=True)
        l2_hits = [
            SegmentHit(
                segment_id=str(item["segment_id"]),
                score=float(item["rank_score"]),
                start_index=int(artifacts.l2_segments[int(item["segment_index"])].start_index),
                end_index=int(artifacts.l2_segments[int(item["segment_index"])].end_index),
                start_time_sec=float(item["start_time_sec"]),
                end_time_sec=float(item["end_time_sec"]),
            )
            for item in l2_items
        ]
        return reranked_hits[: self.config.l3_rerank_keep], {
            "l2_candidates": l2_hits,
            "l2_rerank_encoder": self.config.l2_rerank_encoder,
        }

    def _merge_segment_hits(self, hits: list[SegmentHit], *, top_k: int) -> list[SegmentHit]:
        merged: dict[tuple[int, int, str], SegmentHit] = {}
        for hit in hits:
            key = (int(hit.start_index), int(hit.end_index), str(hit.segment_id))
            previous = merged.get(key)
            if previous is None or float(hit.score) > float(previous.score):
                merged[key] = hit
        return sorted(merged.values(), key=lambda item: float(item.score), reverse=True)[:top_k]

    def _merge_frame_hits(self, hits: list[FrameHit], *, top_k: int) -> list[FrameHit]:
        merged: dict[int, FrameHit] = {}
        for hit in hits:
            previous = merged.get(int(hit.frame_index))
            if previous is None or float(hit.score) > float(previous.score):
                merged[int(hit.frame_index)] = hit
        return sorted(merged.values(), key=lambda item: float(item.score), reverse=True)[:top_k]

    def _retrieve_l3_for_queries(
        self,
        *,
        artifacts: VideoArtifacts,
        queries: list[str],
        top_k_per_query: int | None = None,
    ) -> list[SegmentHit]:
        self._ensure_l3(artifacts)
        assert artifacts.l3_segments is not None
        assert artifacts.l3_embeddings is not None
        hits: list[SegmentHit] = []
        for query in queries:
            query_embedding = adapt_query_embedding_for_segment_pooling(self._text_embedding(query), pooling="mean")
            hits.extend(
                retrieve_top_segments(
                    query_embedding=query_embedding,
                    segment_embeddings=artifacts.l3_embeddings,
                    segments=artifacts.l3_segments,
                    top_k=top_k_per_query or self.config.top_l3_segments,
                )
            )
        return self._merge_segment_hits(
            hits,
            top_k=max(self.config.top_l3_segments, len(queries) * (top_k_per_query or self.config.top_l3_segments)),
        )

    def _retrieve_l2_for_queries(
        self,
        *,
        artifacts: VideoArtifacts,
        queries: list[str],
        top_k_per_query: int | None = None,
    ) -> list[SegmentHit]:
        self._ensure_l2(artifacts)
        assert artifacts.l2_segments is not None
        hits: list[SegmentHit] = []
        for query in queries:
            query_embedding = self._text_embedding(query)
            pooled = adapt_query_embedding_for_segment_pooling(query_embedding, pooling="mean")
            if self.config.l2_scoring == "embedding":
                assert artifacts.l2_embeddings is not None
                hits.extend(
                    retrieve_top_segments(
                        query_embedding=pooled,
                        segment_embeddings=artifacts.l2_embeddings,
                        segments=artifacts.l2_segments,
                        top_k=top_k_per_query or self.config.top_l2_segments,
                    )
                )
            else:
                hits.extend(
                    retrieve_top_segments_from_frame_scores(
                        query_embedding=query_embedding,
                        frame_embeddings=artifacts.frame_embeddings,
                        segments=artifacts.l2_segments,
                        top_k=top_k_per_query or self.config.top_l2_segments,
                        top_m=self.config.l2_frame_score_top_m,
                        aggregation=self.config.l2_scoring,
                        temperature=self.config.l2_frame_score_temperature,
                    )
                )
        return self._merge_segment_hits(
            hits,
            top_k=max(self.config.top_l2_segments, len(queries) * (top_k_per_query or self.config.top_l2_segments)),
        )

    def _retrieve_l1_for_queries(
        self,
        *,
        artifacts: VideoArtifacts,
        queries: list[str],
        allowed_indices: list[int] | None,
        top_k_per_query: int,
    ) -> list[FrameHit]:
        hits: list[FrameHit] = []
        for query in queries:
            hits.extend(
                retrieve_top_frames(
                    query_embedding=self._text_embedding(query),
                    frame_embeddings=artifacts.frame_embeddings,
                    timestamps=artifacts.timestamps,
                    top_k=top_k_per_query,
                    allowed_indices=allowed_indices,
                )
            )
        return self._merge_frame_hits(hits, top_k=max(top_k_per_query, len(queries) * top_k_per_query))

    def _l2_hits_from_frame_hits(
        self,
        *,
        artifacts: VideoArtifacts,
        frame_hits: list[FrameHit],
    ) -> list[SegmentHit]:
        self._ensure_l2(artifacts)
        assert artifacts.l2_segments is not None
        by_segment: dict[str, SegmentHit] = {}
        for frame_hit in sorted(frame_hits, key=lambda item: float(item.score), reverse=True):
            frame_index = int(frame_hit.frame_index)
            for segment in artifacts.l2_segments:
                if int(segment.start_index) <= frame_index <= int(segment.end_index):
                    existing = by_segment.get(segment.segment_id)
                    if existing is None or float(frame_hit.score) > float(existing.score):
                        by_segment[segment.segment_id] = SegmentHit(
                            segment_id=segment.segment_id,
                            score=float(frame_hit.score),
                            start_index=int(segment.start_index),
                            end_index=int(segment.end_index),
                            start_time_sec=float(segment.start_time_sec),
                            end_time_sec=float(segment.end_time_sec),
                        )
                    break
        return sorted(by_segment.values(), key=lambda item: float(item.score), reverse=True)

    def _temporal_neighbors(
        self,
        *,
        artifacts: VideoArtifacts,
        anchor: SegmentHit,
        relation: str,
        max_neighbors: int = 2,
    ) -> list[SegmentHit]:
        self._ensure_l2(artifacts)
        assert artifacts.l2_segments is not None
        if relation == "before":
            candidates = [segment for segment in artifacts.l2_segments if float(segment.end_time_sec) <= float(anchor.start_time_sec)]
            candidates = sorted(candidates, key=lambda segment: float(segment.end_time_sec), reverse=True)[:max_neighbors]
        elif relation == "after":
            candidates = [segment for segment in artifacts.l2_segments if float(segment.start_time_sec) >= float(anchor.end_time_sec)]
            candidates = sorted(candidates, key=lambda segment: float(segment.start_time_sec))[:max_neighbors]
        elif relation == "during":
            candidates = [
                segment
                for segment in artifacts.l2_segments
                if not (float(segment.end_time_sec) < float(anchor.start_time_sec) or float(segment.start_time_sec) > float(anchor.end_time_sec))
            ][:max_neighbors]
        else:
            candidates = []
        hits: list[SegmentHit] = []
        for offset, segment in enumerate(candidates):
            hits.append(
                SegmentHit(
                    segment_id=f"{relation}_{segment.segment_id}",
                    score=float(anchor.score) - 0.001 * offset,
                    start_index=int(segment.start_index),
                    end_index=int(segment.end_index),
                    start_time_sec=float(segment.start_time_sec),
                    end_time_sec=float(segment.end_time_sec),
                )
            )
        return hits

    def _retrieve_et_lite(self, *, example: BaselineExample, artifacts: VideoArtifacts) -> tuple[list[int], dict[str, Any]]:
        plan = _build_et_plan(example.question, example.options)
        query_embedding = self._query_embedding(example.question, example.options)
        pooled_query_embedding = adapt_query_embedding_for_segment_pooling(query_embedding, pooling="mean")

        self._ensure_l3(artifacts)
        assert artifacts.l3_segments is not None
        assert artifacts.l3_embeddings is not None
        l3_hits = retrieve_top_segments(
            query_embedding=pooled_query_embedding,
            segment_embeddings=artifacts.l3_embeddings,
            segments=artifacts.l3_segments,
            top_k=self.config.top_l3_segments,
        )
        l3_allowed_indices = collect_segment_frame_indices(l3_hits)
        l3_frame_hits = self._frame_hits_from_indices(
            artifacts=artifacts,
            query_embedding=query_embedding,
            allowed_indices=l3_allowed_indices,
        )

        metadata_duration = example.metadata.get("duration")
        try:
            video_duration_sec = float(metadata_duration) if metadata_duration is not None else None
        except (TypeError, ValueError):
            video_duration_sec = None
        if video_duration_sec is None and len(artifacts.timestamps) > 0:
            video_duration_sec = float(artifacts.timestamps[-1] - artifacts.timestamps[0])
        metadata_duration_group = example.metadata.get("duration_group")
        try:
            duration_group_sec = float(metadata_duration_group) if metadata_duration_group is not None else None
        except (TypeError, ValueError):
            duration_group_sec = None
        if duration_group_sec is not None:
            video_duration_sec = max(video_duration_sec or 0.0, duration_group_sec)

        use_same_moment_l2 = (
            plan.plan_type == "same_moment_comparison"
            and video_duration_sec is not None
            and video_duration_sec >= self.config.et_l2_min_video_sec
        )
        if use_same_moment_l2:
            self._ensure_l2(artifacts)
            assert artifacts.l2_segments is not None
            l2_hits = retrieve_top_segments_from_frame_scores(
                query_embedding=query_embedding,
                frame_embeddings=artifacts.frame_embeddings,
                segments=artifacts.l2_segments,
                top_k=self.config.top_l2_segments,
                top_m=self.config.l2_frame_score_top_m,
                aggregation=self.config.l2_scoring if self.config.l2_scoring != "embedding" else "topm_mean",
                temperature=self.config.l2_frame_score_temperature,
            )
            if l2_hits:
                frame_hits = self._frame_hits_from_indices(
                    artifacts=artifacts,
                    query_embedding=query_embedding,
                    allowed_indices=collect_segment_frame_indices(l2_hits),
                )
                return [int(hit.frame_index) for hit in frame_hits], {
                    "l2_hits": l2_hits,
                    "l3_hits": l3_hits,
                    "frame_hits": frame_hits,
                    "et_plan": {
                        "plan_type": plan.plan_type,
                        "relation": plan.relation,
                        "anchor_query": plan.anchor_query,
                        "mode": "E_then_local_event",
                        "video_duration_sec": video_duration_sec,
                        "duration_group_sec": duration_group_sec,
                        "et_l2_min_video_sec": self.config.et_l2_min_video_sec,
                        "l3_queries": plan.l3_queries,
                        "l2_queries": plan.l2_queries,
                        "l1_queries": plan.l1_queries,
                        "temporal_hits": [],
                    },
                }

        temporal_hits: list[SegmentHit] = []
        selected_l2_hits: list[SegmentHit] = []
        if plan.relation in {"before", "after"}:
            candidate_hits = retrieve_top_frames(
                query_embedding=query_embedding,
                frame_embeddings=artifacts.frame_embeddings,
                timestamps=artifacts.timestamps,
                top_k=max(self.config.l1_expansion_candidates, self.config.l1_expansion_peaks),
                allowed_indices=l3_allowed_indices,
            )
            peak_hits = _temporal_nms_frame_hits(
                candidate_hits,
                max_hits=self.config.l1_expansion_peaks,
                min_gap_sec=self.config.l1_temporal_nms_sec,
            )
            l1_projected_l2_hits = self._l2_hits_from_frame_hits(artifacts=artifacts, frame_hits=peak_hits)
            selected_l2_hits = [
                hit
                for hit in l1_projected_l2_hits
                if any(hit.end_index >= parent.start_index and hit.start_index <= parent.end_index for parent in l3_hits)
            ]
            temporal_hits = self._temporal_neighbors(
                artifacts=artifacts,
                anchor=(selected_l2_hits[0] if selected_l2_hits else l3_hits[0]),
                relation=plan.relation,
                max_neighbors=2,
            )
            selected_l2_hits = self._merge_segment_hits(
                [*selected_l2_hits[:1], *temporal_hits],
                top_k=max(self.config.top_l2_segments, 3),
            )

        required_indices = [int(hit.frame_index) for hit in l3_frame_hits]
        evidence_indices = [*required_indices, *collect_segment_frame_indices(selected_l2_hits)]
        if not evidence_indices:
            evidence_indices = l3_allowed_indices
        frame_hits = _frame_hits_for_indices(
            query_embedding=query_embedding,
            frame_embeddings=artifacts.frame_embeddings,
            timestamps=artifacts.timestamps,
            indices=evidence_indices,
            max_frames=self.config.max_frames,
            required_indices=required_indices,
        )
        return [int(hit.frame_index) for hit in frame_hits], {
            "l2_hits": selected_l2_hits,
            "l3_hits": l3_hits,
            "frame_hits": frame_hits,
            "et_plan": {
                "plan_type": plan.plan_type,
                "relation": plan.relation,
                "anchor_query": plan.anchor_query,
                "mode": "E_then_T_then_Evidence",
                "video_duration_sec": video_duration_sec,
                "duration_group_sec": duration_group_sec,
                "et_l2_min_video_sec": self.config.et_l2_min_video_sec,
                "l3_queries": plan.l3_queries,
                "l2_queries": plan.l2_queries,
                "l1_queries": plan.l1_queries,
                "temporal_hits": [
                    {
                        "segment_id": hit.segment_id,
                        "start_time_sec": float(hit.start_time_sec),
                        "end_time_sec": float(hit.end_time_sec),
                        "score": float(hit.score),
                    }
                    for hit in temporal_hits
                ],
            },
        }

    def _score_l2_segment_for_query(
        self,
        *,
        artifacts: VideoArtifacts,
        segment_index: int,
        query_text: str,
    ) -> float:
        if self.config.l2_rerank_encoder == "viclip":
            viclip_query_embedding = _get_viclip_encoder().encode_texts([query_text])[0].float().cpu()
            viclip_l2_embeddings = self._ensure_viclip_l2_embeddings(artifacts)
            return float(torch.dot(viclip_l2_embeddings[int(segment_index)], viclip_query_embedding).item())
        query_embedding = self._text_embedding(query_text)
        frame_scores = torch.matmul(artifacts.frame_embeddings, query_embedding).cpu()
        segment = artifacts.l2_segments[int(segment_index)]
        return _segment_topm_score(
            frame_scores=frame_scores,
            start_index=int(segment.start_index),
            end_index=int(segment.end_index),
            top_m=L2_SCORE_TOP_M,
        )

    def _rerank_l3_hits_with_subqueries(
        self,
        *,
        artifacts: VideoArtifacts,
        l3_hits: list[SegmentHit],
        subqueries: list[str],
    ) -> tuple[list[SegmentHit], list[SegmentHit], list[dict[str, Any]]]:
        if not l3_hits or not subqueries:
            return l3_hits, [], []

        self._ensure_l2(artifacts)
        assert artifacts.l2_segments is not None
        candidate_indices = self._candidate_l2_indices_from_l3_hits(artifacts=artifacts, l3_hits=l3_hits)
        if not candidate_indices:
            return l3_hits, [], []

        support_items: list[dict[str, Any]] = []
        for subquery in subqueries:
            segment_scores: list[tuple[int, float]] = []
            for segment_index in sorted(candidate_indices):
                score = self._score_l2_segment_for_query(
                    artifacts=artifacts,
                    segment_index=int(segment_index),
                    query_text=subquery,
                )
                segment_scores.append((int(segment_index), float(score)))
            rank_scores = _rank_score_by_id(segment_scores, key_fn=lambda item: float(item[1]))
            top_ranked = sorted(segment_scores, key=lambda item: float(item[1]), reverse=True)[: min(3, len(segment_scores))]
            for segment_index, raw_score in top_ranked:
                segment = artifacts.l2_segments[int(segment_index)]
                parent_segment_id = None
                for l3_hit in l3_hits:
                    if _segment_time_overlap(
                        segment.start_time_sec,
                        segment.end_time_sec,
                        l3_hit.start_time_sec,
                        l3_hit.end_time_sec,
                    ):
                        parent_segment_id = str(l3_hit.segment_id)
                        break
                if parent_segment_id is None:
                    continue
                support_items.append(
                    {
                        "query": subquery,
                        "segment_index": int(segment_index),
                        "segment_id": str(segment.segment_id),
                        "parent_segment_id": parent_segment_id,
                        "raw_score": float(raw_score),
                        "rank_score": float(rank_scores.get(int(segment_index), 0.0)),
                        "start_time_sec": float(segment.start_time_sec),
                        "end_time_sec": float(segment.end_time_sec),
                    }
                )

        parent_support: dict[str, list[float]] = {}
        for item in support_items:
            parent_support.setdefault(str(item["parent_segment_id"]), []).append(float(item["rank_score"]))

        reranked_hits: list[SegmentHit] = []
        for offset, hit in enumerate(l3_hits):
            support_scores = sorted(parent_support.get(str(hit.segment_id), []), reverse=True)
            support_gain = sum(support_scores[:3]) if support_scores else 0.0
            reranked_hits.append(
                SegmentHit(
                    segment_id=str(hit.segment_id),
                    score=float(hit.score) + 0.4 * float(support_gain) - 0.001 * offset,
                    start_index=int(hit.start_index),
                    end_index=int(hit.end_index),
                    start_time_sec=float(hit.start_time_sec),
                    end_time_sec=float(hit.end_time_sec),
                )
            )
        reranked_hits.sort(key=lambda item: float(item.score), reverse=True)

        support_hits = self._merge_segment_hits(
            [
                SegmentHit(
                    segment_id=str(item["segment_id"]),
                    score=float(item["rank_score"]),
                    start_index=int(artifacts.l2_segments[int(item["segment_index"])].start_index),
                    end_index=int(artifacts.l2_segments[int(item["segment_index"])].end_index),
                    start_time_sec=float(item["start_time_sec"]),
                    end_time_sec=float(item["end_time_sec"]),
                )
                for item in support_items
            ],
            top_k=max(self.config.top_l2_segments, len(subqueries)),
        )
        return reranked_hits[: self.config.l3_rerank_keep], support_hits, support_items

    def _retrieve_l3_l2_vgent(
        self,
        *,
        example: BaselineExample,
        artifacts: VideoArtifacts,
        text_answerer: Any | None,
    ) -> tuple[list[int], dict[str, Any]]:
        plan = _build_vgent_plan(
            question=example.question,
            options=example.options,
            text_answerer=text_answerer,
        )
        query_embedding = self._query_embedding(example.question, example.options)
        l3_hits = self._retrieve_l3_for_queries(
            artifacts=artifacts,
            queries=plan.retrieval_queries,
            top_k_per_query=max(2, min(4, self.config.top_l3_segments)),
        )
        if not l3_hits:
            return [], {
                "l2_hits": [],
                "l3_hits": [],
                "frame_hits": [],
                "vgent_plan": asdict(plan),
            }

        reranked_l3_hits, rerank_debug = self._rerank_l3_hits_with_l2(
            artifacts=artifacts,
            query_embedding=query_embedding,
            target_text=plan.primary_query,
            l3_hits=l3_hits,
        )
        final_l3_hits, support_l2_hits, support_items = self._rerank_l3_hits_with_subqueries(
            artifacts=artifacts,
            l3_hits=reranked_l3_hits,
            subqueries=plan.subqueries,
        )
        if not support_l2_hits:
            support_l2_hits = list(rerank_debug.get("l2_candidates", []))[: self.config.top_l2_segments]

        evidence_indices = collect_segment_frame_indices(support_l2_hits)
        if not evidence_indices:
            evidence_indices = collect_segment_frame_indices(final_l3_hits)
        frame_hits = self._frame_hits_from_indices(
            artifacts=artifacts,
            query_embedding=query_embedding,
            allowed_indices=evidence_indices,
        )
        return [int(hit.frame_index) for hit in frame_hits], {
            "l2_hits": support_l2_hits,
            "l3_hits": final_l3_hits,
            "frame_hits": frame_hits,
            "l2_rerank_encoder": self.config.l2_rerank_encoder,
            "vgent_plan": {
                **asdict(plan),
                "support_items": support_items,
            },
        }

    def _select_best_l2_per_parent(
        self,
        *,
        l3_hits: list[SegmentHit],
        l2_candidates: list[SegmentHit],
    ) -> list[SegmentHit]:
        selected: list[SegmentHit] = []
        for parent in l3_hits:
            best_child = next(
                (
                    hit
                    for hit in l2_candidates
                    if _segment_time_overlap(
                        hit.start_time_sec,
                        hit.end_time_sec,
                        parent.start_time_sec,
                        parent.end_time_sec,
                    )
                ),
                None,
            )
            if best_child is not None:
                selected.append(best_child)
        return self._merge_segment_hits(selected, top_k=len(l3_hits))

    def _select_l2_per_parent(
        self,
        *,
        l3_hits: list[SegmentHit],
        l2_candidates: list[SegmentHit],
        per_parent: int,
    ) -> list[SegmentHit]:
        selected: list[SegmentHit] = []
        per_parent = max(1, int(per_parent))
        for parent in l3_hits:
            children = [
                hit
                for hit in l2_candidates
                if _segment_time_overlap(
                    hit.start_time_sec,
                    hit.end_time_sec,
                    parent.start_time_sec,
                    parent.end_time_sec,
                )
            ]
            selected.extend(children[:per_parent])
        return self._merge_segment_hits(selected, top_k=max(len(l3_hits) * per_parent, 1))

    def _retrieve_hm_router_v1(
        self,
        *,
        example: BaselineExample,
        artifacts: VideoArtifacts,
        strict: bool = False,
        v3: bool = False,
        runtime_only: bool = False,
    ) -> tuple[list[int], dict[str, Any]]:
        query_embedding = self._query_embedding(example.question, example.options)
        pooled_query_embedding = adapt_query_embedding_for_segment_pooling(query_embedding, pooling="mean")
        self._ensure_l3(artifacts)
        assert artifacts.l3_segments is not None
        assert artifacts.l3_embeddings is not None
        l3_hits = retrieve_top_segments(
            query_embedding=pooled_query_embedding,
            segment_embeddings=artifacts.l3_embeddings,
            segments=artifacts.l3_segments,
            top_k=self.config.top_l3_segments,
        )
        if runtime_only:
            decision = _decide_hm_runtime_router(
                question=example.question,
                artifacts=artifacts,
                l3_hits=l3_hits,
            )
        elif v3:
            decision = _decide_hm_route_v3(
                question=example.question,
                metadata=example.metadata,
                l3_hits=l3_hits,
            )
        elif strict:
            decision = _decide_hm_route_v2(question=example.question, l3_hits=l3_hits)
        else:
            decision = _decide_hm_route(question=example.question, l3_hits=l3_hits)
        kept_l3_hits = l3_hits[: self.config.l3_rerank_keep]

        if decision.route == "l3_to_l2_local":
            reranked_l3_hits, rerank_debug = self._rerank_l3_hits_with_l2(
                artifacts=artifacts,
                query_embedding=query_embedding,
                target_text=example.question,
                l3_hits=l3_hits,
            )
            kept_l3_hits = reranked_l3_hits
            selected_l2_hits = self._select_best_l2_per_parent(
                l3_hits=kept_l3_hits,
                l2_candidates=list(rerank_debug.get("l2_candidates", [])),
            )
            required_indices: list[int] = []
            allowed_indices = collect_segment_frame_indices(selected_l2_hits)
            if selected_l2_hits:
                self._ensure_l2(artifacts)
                assert artifacts.l2_segments is not None
                segment_by_id = {segment.segment_id: segment for segment in artifacts.l2_segments}
                for hit in selected_l2_hits:
                    segment = segment_by_id.get(str(hit.segment_id))
                    if segment is None:
                        continue
                    required_indices.extend(_segment_sample_indices(segment, count=3))
            frame_hits = _frame_hits_for_indices(
                query_embedding=query_embedding,
                frame_embeddings=artifacts.frame_embeddings,
                timestamps=artifacts.timestamps,
                indices=allowed_indices or collect_segment_frame_indices(kept_l3_hits),
                max_frames=self.config.max_frames,
                required_indices=required_indices,
            )
            return [int(hit.frame_index) for hit in frame_hits], {
                "l2_hits": selected_l2_hits,
                "l3_hits": kept_l3_hits,
                "frame_hits": frame_hits,
                "l2_rerank_encoder": self.config.l2_rerank_encoder,
                "hm_route": asdict(decision),
            }

        frame_hits = self._frame_hits_from_indices(
            artifacts=artifacts,
            query_embedding=query_embedding,
            allowed_indices=collect_segment_frame_indices(kept_l3_hits),
        )
        return [int(hit.frame_index) for hit in frame_hits], {
            "l2_hits": [],
            "l3_hits": kept_l3_hits,
            "frame_hits": frame_hits,
            "hm_route": asdict(decision),
        }

    def _retrieve_graph_a(
        self,
        *,
        example: BaselineExample,
        artifacts: VideoArtifacts,
        graph_answerer: Any,
        subtitle_root: str | Path | None,
        subtitle_tar: str | Path | None,
    ) -> tuple[list[int], dict[str, Any]]:
        query_embedding = self._query_embedding(example.question, example.options)
        pooled_query_embedding = adapt_query_embedding_for_segment_pooling(query_embedding, pooling="mean")
        self._ensure_l3(artifacts)
        assert artifacts.l3_segments is not None
        assert artifacts.l3_embeddings is not None
        l3_hits = retrieve_top_segments(
            query_embedding=pooled_query_embedding,
            segment_embeddings=artifacts.l3_embeddings,
            segments=artifacts.l3_segments,
            top_k=self.config.top_l3_segments,
        )
        self._ensure_l2_graph(
            example=example,
            artifacts=artifacts,
            graph_answerer=graph_answerer,
            subtitle_root=subtitle_root,
            subtitle_tar=subtitle_tar,
        )
        graph_hits = self._graph_node_hits(
            artifacts=artifacts,
            query_text=example.question,
            l3_hits=l3_hits,
        )
        allowed_indices = collect_segment_frame_indices(graph_hits)
        if not allowed_indices:
            allowed_indices = collect_segment_frame_indices(l3_hits)
        frame_hits = self._frame_hits_from_indices(
            artifacts=artifacts,
            query_embedding=query_embedding,
            allowed_indices=allowed_indices,
        )
        return [int(hit.frame_index) for hit in frame_hits], {
            "l2_hits": graph_hits,
            "l3_hits": l3_hits,
            "frame_hits": frame_hits,
            "graph_mode": "phase1_branch_a_raw_query",
        }

    def retrieve(
        self,
        *,
        example: BaselineExample,
        graph_answerer: Any | None = None,
        subtitle_root: str | Path | None = None,
        subtitle_tar: str | Path | None = None,
    ) -> tuple[list[int], dict[str, Any]]:
        artifacts = self._load_video(example)
        query_embedding = self._query_embedding(example.question, example.options)
        pooled_query_embedding = adapt_query_embedding_for_segment_pooling(query_embedding, pooling="mean")

        if self.config.method == "et_lite":
            return self._retrieve_et_lite(example=example, artifacts=artifacts)

        if self.config.method == "graph_a":
            if graph_answerer is None:
                raise RuntimeError("graph_a retrieval requires a loaded answerer for offline graph extraction")
            return self._retrieve_graph_a(
                example=example,
                artifacts=artifacts,
                graph_answerer=graph_answerer,
                subtitle_root=subtitle_root,
                subtitle_tar=subtitle_tar,
            )

        if self.config.method == "l3_l2_vgent":
            return self._retrieve_l3_l2_vgent(
                example=example,
                artifacts=artifacts,
                text_answerer=graph_answerer,
            )

        if self.config.method == "hm_router_v1":
            return self._retrieve_hm_router_v1(
                example=example,
                artifacts=artifacts,
                strict=False,
            )

        if self.config.method == "hm_router_v2":
            return self._retrieve_hm_router_v1(
                example=example,
                artifacts=artifacts,
                strict=True,
            )

        if self.config.method == "hm_router_v3":
            return self._retrieve_hm_router_v1(
                example=example,
                artifacts=artifacts,
                v3=True,
            )

        if self.config.method == "hm_router_runtime":
            return self._retrieve_hm_router_v1(
                example=example,
                artifacts=artifacts,
                runtime_only=True,
            )

        if self.config.method == "l1":
            frame_hits = self._frame_hits_from_indices(artifacts=artifacts, query_embedding=query_embedding, allowed_indices=None)
            return [int(hit.frame_index) for hit in frame_hits], {"l2_hits": [], "l3_hits": [], "frame_hits": frame_hits}

        if self.config.method == "l2":
            self._ensure_l2(artifacts)
            assert artifacts.l2_segments is not None
            assert artifacts.l2_embeddings is not None
            if self.config.l2_scoring == "embedding":
                l2_hits = retrieve_top_segments(
                    query_embedding=pooled_query_embedding,
                    segment_embeddings=artifacts.l2_embeddings,
                    segments=artifacts.l2_segments,
                    top_k=self.config.top_l2_segments,
                )
            else:
                l2_hits = retrieve_top_segments_from_frame_scores(
                    query_embedding=query_embedding,
                    frame_embeddings=artifacts.frame_embeddings,
                    segments=artifacts.l2_segments,
                    top_k=self.config.top_l2_segments,
                    top_m=self.config.l2_frame_score_top_m,
                    aggregation=self.config.l2_scoring,
                    temperature=self.config.l2_frame_score_temperature,
                )
            frame_hits = self._frame_hits_from_indices(
                artifacts=artifacts,
                query_embedding=query_embedding,
                allowed_indices=collect_segment_frame_indices(l2_hits),
            )
            return [int(hit.frame_index) for hit in frame_hits], {"l2_hits": l2_hits, "l3_hits": [], "frame_hits": frame_hits}

        if self.config.method == "l3_rerank_l2":
            self._ensure_l3(artifacts)
            self._ensure_l2(artifacts)
            assert artifacts.l3_segments is not None
            assert artifacts.l3_embeddings is not None
            l3_hits = retrieve_top_segments(
                query_embedding=pooled_query_embedding,
                segment_embeddings=artifacts.l3_embeddings,
                segments=artifacts.l3_segments,
                top_k=self.config.top_l3_segments,
            )
            reranked_l3_hits, rerank_debug = self._rerank_l3_hits_with_l2(
                artifacts=artifacts,
                query_embedding=query_embedding,
                target_text=example.question,
                l3_hits=l3_hits,
            )
            selected_l2_hits: list[SegmentHit] = []
            required_indices: list[int] = []
            if self.config.l3_rerank_evidence_source == "reranked_l3":
                allowed_indices = collect_segment_frame_indices(reranked_l3_hits)
            else:
                l2_candidates = list(rerank_debug.get("l2_candidates", []))
                if self.config.l3_rerank_evidence_source == "top_l2":
                    selected_l2_hits = self._merge_segment_hits(
                        l2_candidates,
                        top_k=max(self.config.top_l2_segments, 1),
                    )
                elif self.config.l3_rerank_evidence_source == "top_l2_per_l3":
                    selected_l2_hits = self._select_l2_per_parent(
                        l3_hits=reranked_l3_hits,
                        l2_candidates=l2_candidates,
                        per_parent=self.config.l2_evidence_per_l3,
                    )
                else:
                    raise ValueError(f"Unsupported l3_rerank_evidence_source: {self.config.l3_rerank_evidence_source}")
                allowed_indices = collect_segment_frame_indices(selected_l2_hits)
                if selected_l2_hits and self.config.l1_evidence_per_l2 > 0:
                    assert artifacts.l2_segments is not None
                    segment_by_id = {str(segment.segment_id): segment for segment in artifacts.l2_segments}
                    for hit in selected_l2_hits:
                        segment = segment_by_id.get(str(hit.segment_id))
                        if segment is not None:
                            required_indices.extend(
                                _segment_sample_indices(segment, count=self.config.l1_evidence_per_l2)
                            )
                if not allowed_indices:
                    allowed_indices = collect_segment_frame_indices(reranked_l3_hits)
            frame_hits = self._frame_hits_from_indices(
                artifacts=artifacts,
                query_embedding=query_embedding,
                allowed_indices=allowed_indices,
                required_indices=required_indices,
            )
            return [int(hit.frame_index) for hit in frame_hits], {
                "l2_hits": selected_l2_hits or rerank_debug.get("l2_candidates", []),
                "l3_hits": reranked_l3_hits,
                "frame_hits": frame_hits,
                "l2_rerank_encoder": rerank_debug.get("l2_rerank_encoder"),
                "l3_rerank_evidence_source": self.config.l3_rerank_evidence_source,
            }

        if self.config.method != "l3":
            raise ValueError(f"Unsupported ablation method: {self.config.method}")

        self._ensure_l3(artifacts)
        assert artifacts.l3_segments is not None
        assert artifacts.l3_embeddings is not None
        l3_hits = retrieve_top_segments(
            query_embedding=pooled_query_embedding,
            segment_embeddings=artifacts.l3_embeddings,
            segments=artifacts.l3_segments,
            top_k=self.config.top_l3_segments,
        )
        frame_hits = self._frame_hits_from_indices(
            artifacts=artifacts,
            query_embedding=query_embedding,
            allowed_indices=collect_segment_frame_indices(l3_hits),
        )
        return [int(hit.frame_index) for hit in frame_hits], {"l2_hits": [], "l3_hits": l3_hits, "frame_hits": frame_hits}


def _build_output_name(*, model_id: str, run_config: AblationRunConfig) -> str:
    output_name = f"{model_id.split('/')[-1]}_{run_config.method}"
    if run_config.method in {"l3", "l3_rerank_l2"} and run_config.l3_segmentation == "fixed":
        output_name += f"_l3fixed{run_config.l3_window_seconds:g}s_s{run_config.l3_stride_seconds:g}s"
    if run_config.method in {"l2", "l3_rerank_l2", "et_lite", "graph_a", "l3_l2_vgent", "hm_router_v1", "hm_router_v2", "hm_router_v3", "hm_router_runtime"}:
        if run_config.l2_segmentation == "fixed":
            output_name += f"_l2w{run_config.l2_window_seconds:g}_l2s{run_config.l2_stride_seconds:g}"
        else:
            output_name += (
                f"_l2{run_config.l2_segmentation}"
                f"_min{run_config.l2_local_min_duration_sec:g}"
                f"_max{run_config.l2_local_max_duration_sec:g}"
                f"_p{run_config.l2_local_peak_percentile:g}"
            )
        if run_config.l2_scoring != "embedding":
            output_name += f"_score{run_config.l2_scoring}"
    if run_config.method == "l3_rerank_l2":
        output_name += f"_l3k{run_config.top_l3_segments:g}"
        if run_config.l3_rerank_keep != DEFAULT_L3_RERANK_K:
            output_name += f"_keep{run_config.l3_rerank_keep:g}"
        if run_config.l3_rerank_evidence_source != "reranked_l3":
            output_name += f"_evi{run_config.l3_rerank_evidence_source}"
            if run_config.l3_rerank_evidence_source == "top_l2_per_l3":
                output_name += f"_l2p{run_config.l2_evidence_per_l3:g}_l1p{run_config.l1_evidence_per_l2:g}"
        if run_config.l2_rerank_encoder != "openclip":
            output_name += f"_l2enc{run_config.l2_rerank_encoder}"
    if run_config.method == "l3_l2_vgent":
        output_name += f"_l3k{run_config.top_l3_segments:g}"
        if run_config.l3_rerank_keep != DEFAULT_L3_RERANK_K:
            output_name += f"_keep{run_config.l3_rerank_keep:g}"
        if run_config.l2_rerank_encoder != "openclip":
            output_name += f"_l2enc{run_config.l2_rerank_encoder}"
    if run_config.method == "hm_router_v1":
        output_name += f"_l3k{run_config.top_l3_segments:g}"
        if run_config.l3_rerank_keep != DEFAULT_L3_RERANK_K:
            output_name += f"_keep{run_config.l3_rerank_keep:g}"
        if run_config.l2_rerank_encoder != "openclip":
            output_name += f"_l2enc{run_config.l2_rerank_encoder}"
    if run_config.method == "hm_router_v2":
        output_name += f"_l3k{run_config.top_l3_segments:g}"
        if run_config.l3_rerank_keep != DEFAULT_L3_RERANK_K:
            output_name += f"_keep{run_config.l3_rerank_keep:g}"
        if run_config.l2_rerank_encoder != "openclip":
            output_name += f"_l2enc{run_config.l2_rerank_encoder}"
    if run_config.method == "hm_router_v3":
        output_name += f"_l3k{run_config.top_l3_segments:g}"
        if run_config.l3_rerank_keep != DEFAULT_L3_RERANK_K:
            output_name += f"_keep{run_config.l3_rerank_keep:g}"
        if run_config.l2_rerank_encoder != "openclip":
            output_name += f"_l2enc{run_config.l2_rerank_encoder}"
    if run_config.method == "hm_router_runtime":
        output_name += f"_l3k{run_config.top_l3_segments:g}"
        if run_config.l3_rerank_keep != DEFAULT_L3_RERANK_K:
            output_name += f"_keep{run_config.l3_rerank_keep:g}"
        if run_config.l2_rerank_encoder != "openclip":
            output_name += f"_l2enc{run_config.l2_rerank_encoder}"
    if run_config.method == "et_lite":
        output_name += (
            f"_peaks{run_config.l1_expansion_peaks:g}"
            f"_nms{run_config.l1_temporal_nms_sec:g}"
        )
        if run_config.et_l2_min_video_sec > 0:
            output_name += f"_etmin{run_config.et_l2_min_video_sec:g}s"
    if run_config.method == "graph_a":
        output_name += f"_graph{run_config.graph_frames_per_segment}f"
    output_name += f"_{run_config.max_frames}f_{run_config.image_max_size}"
    return output_name


def run_ablation(
    *,
    examples: list[BaselineExample],
    video_root: Path,
    feature_root: Path,
    derived_cache_root: Path,
    output_root: Path,
    run_config: AblationRunConfig,
    answer_config: AnswerConfig,
    subtitle_root: str | Path | None,
    subtitle_tar: str | Path | None,
    api_answer_workers: int = 1,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    rows_path = output_root / "rows.jsonl"
    progress_path = output_root / "progress.log"
    error_path = output_root / "error.log"
    rolling_summary_path = output_root / "rolling_summary.json"

    rows, _ = _load_resume_rows(rows_path)
    completed_example_ids = {str(row["example_id"]) for row in rows}
    pending_examples = [example for example in examples if example.example_id not in completed_example_ids]
    if rows_path.exists():
        _rewrite_jsonl(rows_path, rows)
    _write_json(rolling_summary_path, {"completed": len(rows), "total": len(examples), **_summarize_rows(rows)})

    answerer = build_answerer(answer_config)
    use_parallel_answer = answer_config.backend == "api" and int(api_answer_workers) > 1
    if use_parallel_answer:
        answerer.load()
    answer_executor: ThreadPoolExecutor | None = (
        ThreadPoolExecutor(max_workers=max(int(api_answer_workers), 1)) if use_parallel_answer else None
    )
    answer_futures: set[Future[tuple[dict[str, Any], str]]] = set()
    max_in_flight = max(int(api_answer_workers) * 2, 1)
    retriever = AblationRetriever(
        feature_root=feature_root,
        derived_cache_root=derived_cache_root,
        video_root=video_root,
        config=run_config,
        encoder_device="cuda" if torch.cuda.is_available() else "cpu",
    )

    def _blocked_row(
        *,
        example: BaselineExample,
        exc: Exception,
        subtitle_context: str | None,
    ) -> dict[str, Any]:
        return {
            "example_id": example.example_id,
            "video_id": example.video_id,
            "video_path": example.video_path,
            "question": example.question,
            "options": example.options,
            "correct_index": example.correct_index,
            "gold_letter": (chr(ord("A") + int(example.correct_index)) if example.correct_index is not None and int(example.correct_index) >= 0 else None),
            "predicted_letter": None,
            "choice_correct": None,
            "raw_answer": f"API_BLOCKED: {type(exc).__name__}: {exc}",
            "generation_sec": None,
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "subtitle_context": subtitle_context,
            "method": run_config.method,
            "status": "api_blocked",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            **example.metadata,
        }

    def _answer_row(
        *,
        index: int,
        example: BaselineExample,
        frames: list[Any],
        frame_texts: list[str],
        subtitle_context: str | None,
        retrieval_info: dict[str, Any],
    ) -> tuple[dict[str, Any], str]:
        try:
            prediction = answerer.answer_frames(
                frames=frames,
                question=example.question,
                options=example.options,
                prompt_prefix=run_config.prompt_prefix,
                frame_texts=frame_texts,
            )
        except Exception as exc:
            if _is_api_content_filter_error(exc):
                row = _blocked_row(example=example, exc=exc, subtitle_context=subtitle_context)
                return row, f"[item_blocked] index={index}/{len(examples)} example_id={example.example_id} error={type(exc).__name__}: {exc}"
            raise

        gold_letter = chr(ord("A") + int(example.correct_index)) if example.correct_index is not None and int(example.correct_index) >= 0 else None
        row = {
            "example_id": example.example_id,
            "video_id": example.video_id,
            "video_path": example.video_path,
            "question": example.question,
            "options": example.options,
            "correct_index": example.correct_index,
            "gold_letter": gold_letter,
            "predicted_letter": prediction.predicted_letter,
            "choice_correct": (prediction.predicted_letter == gold_letter) if gold_letter is not None else None,
            "raw_answer": prediction.raw_text,
            "generation_sec": prediction.generation_sec,
            "prompt_tokens": prediction.prompt_tokens,
            "completion_tokens": prediction.completion_tokens,
            "total_tokens": prediction.total_tokens,
            "subtitle_context": subtitle_context,
            "method": run_config.method,
            "frames": [
                {"frame_index": int(hit.frame_index), "time_sec": float(hit.time_sec), "score": float(hit.score)}
                for hit in retrieval_info["frame_hits"]
            ],
            "l2_hits": [
                {
                    "segment_id": hit.segment_id,
                    "score": float(hit.score),
                    "start_time_sec": float(hit.start_time_sec),
                    "end_time_sec": float(hit.end_time_sec),
                }
                for hit in retrieval_info["l2_hits"]
            ],
            "l3_hits": [
                {
                    "segment_id": hit.segment_id,
                    "score": float(hit.score),
                    "start_time_sec": float(hit.start_time_sec),
                    "end_time_sec": float(hit.end_time_sec),
                }
                for hit in retrieval_info["l3_hits"]
            ],
            "et_plan": retrieval_info.get("et_plan"),
            "graph_mode": retrieval_info.get("graph_mode"),
            "vgent_plan": retrieval_info.get("vgent_plan"),
            "hm_route": retrieval_info.get("hm_route"),
            **example.metadata,
        }
        message = (
            f"[item_done] index={index}/{len(examples)} example_id={example.example_id} "
            f"predicted={prediction.predicted_letter} correct={row['choice_correct']} gen_sec={prediction.generation_sec}"
        )
        return row, message

    def _record_row(row: dict[str, Any], message: str) -> None:
        rows.append(row)
        _append_jsonl(rows_path, row)
        _write_json(rolling_summary_path, {"completed": len(rows), "total": len(examples), **_summarize_rows(rows)})
        _log_line(progress_path, message)

    def _drain_finished(*, block: bool) -> None:
        if not answer_futures:
            return
        if block:
            done, _ = wait(answer_futures, return_when=FIRST_COMPLETED)
        else:
            done, _ = wait(answer_futures, timeout=0, return_when=FIRST_COMPLETED)
        for future in done:
            answer_futures.remove(future)
            row, message = future.result()
            _record_row(row, message)

    try:
        if pending_examples:
            _log_line(
                progress_path,
                f"[start] total={len(examples)} method={run_config.method} sample_fps={run_config.sample_fps} max_frames={run_config.max_frames} api_answer_workers={api_answer_workers}",
            )
        for index, example in enumerate(pending_examples, start=len(rows) + 1):
            subtitle_context = None
            try:
                _log_line(progress_path, f"[item_start] index={index}/{len(examples)} example_id={example.example_id} video={example.video_id}")
                target_indices, retrieval_info = retriever.retrieve(
                    example=example,
                    graph_answerer=answerer,
                    subtitle_root=subtitle_root,
                    subtitle_tar=subtitle_tar,
                )
                frames, frame_hits, _ = load_selected_video_frames(
                    video_root / example.video_path,
                    sample_fps=run_config.sample_fps,
                    target_indices=target_indices,
                    image_max_size=run_config.image_max_size,
                )
                frame_times = [float(hit.time_sec) for hit in frame_hits]
                subtitle_texts: list[str] | None = None
                if run_config.include_subtitles:
                    subtitle_path = example.metadata.get("subtitle_path")
                    if subtitle_path:
                        subtitles = _load_subtitles(
                            subtitle_path=str(subtitle_path),
                            subtitle_root=subtitle_root,
                            subtitle_tar=subtitle_tar,
                        )
                        subtitle_texts, subtitle_context = _subtitle_texts_for_frames(
                            frame_times=frame_times,
                            subtitles=subtitles,
                            starting_timestamp_for_subtitles=float(example.metadata.get("starting_timestamp_for_subtitles", 0.0)),
                            duration=(float(example.metadata["duration"]) if example.metadata.get("duration") is not None else None),
                        )
                frame_texts = _merge_frame_texts(frame_times=frame_times, subtitle_texts=subtitle_texts)
                if use_parallel_answer:
                    assert answer_executor is not None
                    answer_futures.add(
                        answer_executor.submit(
                            _answer_row,
                            index=index,
                            example=example,
                            frames=frames,
                            frame_texts=frame_texts,
                            subtitle_context=subtitle_context,
                            retrieval_info=retrieval_info,
                        )
                    )
                    while len(answer_futures) >= max_in_flight:
                        _drain_finished(block=True)
                    continue
                row, message = _answer_row(
                    index=index,
                    example=example,
                    frames=frames,
                    frame_texts=frame_texts,
                    subtitle_context=subtitle_context,
                    retrieval_info=retrieval_info,
                )
            except Exception as exc:
                _log_line(progress_path, f"[item_error] index={index}/{len(examples)} example_id={example.example_id} error={type(exc).__name__}: {exc}")
                with error_path.open("a", encoding="utf-8") as handle:
                    handle.write(f"{example.example_id}: {type(exc).__name__}: {exc}\n")
                raise
            _record_row(row, message)
        while answer_futures:
            _drain_finished(block=True)
    finally:
        if answer_executor is not None:
            answer_executor.shutdown(wait=True, cancel_futures=False)
        answerer.unload()
        retriever.unload()

    summary = {
        "run_config": {
            "method": run_config.method,
            "sample_fps": run_config.sample_fps,
            "max_frames": run_config.max_frames,
            "image_max_size": run_config.image_max_size,
            "include_subtitles": run_config.include_subtitles,
            "l2_window_seconds": run_config.l2_window_seconds,
            "l2_stride_seconds": run_config.l2_stride_seconds,
            "l2_segmentation": run_config.l2_segmentation,
            "l2_local_min_duration_sec": run_config.l2_local_min_duration_sec,
            "l2_local_max_duration_sec": run_config.l2_local_max_duration_sec,
            "l2_local_fast_kernel_size": run_config.l2_local_fast_kernel_size,
            "l2_local_slow_kernel_size": run_config.l2_local_slow_kernel_size,
            "l2_local_peak_percentile": run_config.l2_local_peak_percentile,
            "l2_scoring": run_config.l2_scoring,
            "l2_frame_score_top_m": run_config.l2_frame_score_top_m,
            "l2_frame_score_temperature": run_config.l2_frame_score_temperature,
            "top_l2_segments": run_config.top_l2_segments,
            "top_l3_segments": run_config.top_l3_segments,
            "l3_segmentation": run_config.l3_segmentation,
            "l3_window_seconds": run_config.l3_window_seconds,
            "l3_stride_seconds": run_config.l3_stride_seconds,
            "l1_expansion_peaks": run_config.l1_expansion_peaks,
            "l1_expansion_candidates": run_config.l1_expansion_candidates,
            "l1_temporal_nms_sec": run_config.l1_temporal_nms_sec,
            "et_l2_min_video_sec": run_config.et_l2_min_video_sec,
            "graph_frames_per_segment": run_config.graph_frames_per_segment,
            "l2_rerank_encoder": run_config.l2_rerank_encoder,
            "l3_rerank_keep": run_config.l3_rerank_keep,
        },
        "answer_config": {
            "model_id": answer_config.model_id,
            "backend": answer_config.backend,
            "load_in_4bit": answer_config.load_in_4bit,
            "load_in_8bit": answer_config.load_in_8bit,
            "api_base_url": answer_config.api_base_url,
            "api_key_env_var": answer_config.api_key_env_var,
            "api_requests_per_minute": answer_config.api_requests_per_minute,
            "api_tokens_per_minute": answer_config.api_tokens_per_minute,
            "api_timeout_sec": answer_config.api_timeout_sec,
            "api_answer_workers": api_answer_workers,
        },
        **_summarize_rows(rows),
    }
    _write_json(output_root / "final_summary.json", summary)
    _log_line(progress_path, f"[done] scored={summary['scored']} accuracy={summary['choice_accuracy']}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--derived-cache-root", type=Path, default=DEFAULT_DERIVED_CACHE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--subtitle-root", type=Path, default=DEFAULT_SUBTITLE_ROOT)
    parser.add_argument("--subtitle-tar", type=Path, default=DEFAULT_SUBTITLE_TAR)
    parser.add_argument("--method", choices=["l1", "l2", "l3", "l3_rerank_l2", "l3_l2_vgent", "hm_router_v1", "hm_router_v2", "hm_router_v3", "hm_router_runtime", "et_lite", "graph_a"], required=True)
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--image-max-size", type=int, default=336)
    parser.add_argument("--include-subtitles", action="store_true")
    parser.add_argument("--l2-window-seconds", type=float, default=5.0)
    parser.add_argument("--l2-stride-seconds", type=float, default=5.0)
    parser.add_argument("--l2-segmentation", choices=["fixed", "l3_local_contrast"], default="fixed")
    parser.add_argument("--l2-local-min-duration-sec", type=float, default=3.0)
    parser.add_argument("--l2-local-max-duration-sec", type=float, default=12.0)
    parser.add_argument("--l2-local-fast-kernel-size", type=int, default=1)
    parser.add_argument("--l2-local-slow-kernel-size", type=int, default=9)
    parser.add_argument("--l2-local-peak-percentile", type=float, default=75.0)
    parser.add_argument(
        "--l2-scoring",
        choices=["embedding", "topm_mean", "max", "logsumexp_mean", "softmax_mean"],
        default="embedding",
    )
    parser.add_argument("--l2-frame-score-top-m", type=int, default=4)
    parser.add_argument("--l2-frame-score-temperature", type=float, default=0.07)
    parser.add_argument("--top-l2-segments", type=int, default=10)
    parser.add_argument("--top-l3-segments", type=int, default=10)
    parser.add_argument("--l3-segmentation", choices=["fused_adaptive", "fixed"], default="fused_adaptive")
    parser.add_argument("--l3-window-seconds", type=float, default=60.0)
    parser.add_argument("--l3-stride-seconds", type=float, default=60.0)
    parser.add_argument("--l2-rerank-encoder", choices=["openclip", "viclip"], default="openclip")
    parser.add_argument("--l3-rerank-keep", type=int, default=DEFAULT_L3_RERANK_K)
    parser.add_argument(
        "--l3-rerank-evidence-source",
        choices=["reranked_l3", "top_l2", "top_l2_per_l3"],
        default="reranked_l3",
        help="For l3_rerank_l2, choose final frame evidence from reranked parent L3 ranges or L2 windows.",
    )
    parser.add_argument("--l2-evidence-per-l3", type=int, default=1)
    parser.add_argument("--l1-evidence-per-l2", type=int, default=3)
    parser.add_argument("--graph-frames-per-segment", type=int, default=3)
    parser.add_argument("--l1-expansion-peaks", type=int, default=6)
    parser.add_argument("--l1-expansion-candidates", type=int, default=32)
    parser.add_argument("--l1-temporal-nms-sec", type=float, default=4.0)
    parser.add_argument("--et-l2-min-video-sec", type=float, default=0.0)
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--backend", choices=["local", "api"], default="local")
    parser.add_argument("--api-base-url", default=None)
    parser.add_argument("--api-key-env-var", default="ALIBABA_CLOUD_API")
    parser.add_argument("--api-requests-per-minute", type=int, default=60)
    parser.add_argument("--api-tokens-per-minute", type=int, default=100000)
    parser.add_argument("--api-timeout-sec", type=float, default=120.0)
    parser.add_argument("--api-answer-workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    args = parser.parse_args()

    examples = _load_examples(args.manifest, limit=args.limit)
    _validate_video_root(args.video_root, examples)
    run_config = AblationRunConfig(
        method=args.method,
        sample_fps=args.sample_fps,
        max_frames=args.max_frames,
        image_max_size=args.image_max_size,
        include_subtitles=args.include_subtitles,
        l2_window_seconds=args.l2_window_seconds,
        l2_stride_seconds=args.l2_stride_seconds,
        l2_segmentation=args.l2_segmentation,
        l2_local_min_duration_sec=args.l2_local_min_duration_sec,
        l2_local_max_duration_sec=args.l2_local_max_duration_sec,
        l2_local_fast_kernel_size=args.l2_local_fast_kernel_size,
        l2_local_slow_kernel_size=args.l2_local_slow_kernel_size,
        l2_local_peak_percentile=args.l2_local_peak_percentile,
        l2_scoring=args.l2_scoring,
        l2_frame_score_top_m=args.l2_frame_score_top_m,
        l2_frame_score_temperature=args.l2_frame_score_temperature,
        top_l2_segments=args.top_l2_segments,
        top_l3_segments=args.top_l3_segments,
        l3_segmentation=args.l3_segmentation,
        l3_window_seconds=args.l3_window_seconds,
        l3_stride_seconds=args.l3_stride_seconds,
        l1_expansion_peaks=args.l1_expansion_peaks,
        l1_expansion_candidates=args.l1_expansion_candidates,
        l1_temporal_nms_sec=args.l1_temporal_nms_sec,
        et_l2_min_video_sec=args.et_l2_min_video_sec,
        graph_frames_per_segment=args.graph_frames_per_segment,
        l2_rerank_encoder=args.l2_rerank_encoder,
        l3_rerank_keep=args.l3_rerank_keep,
        l3_rerank_evidence_source=args.l3_rerank_evidence_source,
        l2_evidence_per_l3=args.l2_evidence_per_l3,
        l1_evidence_per_l2=args.l1_evidence_per_l2,
    )
    output_name = _build_output_name(model_id=args.model_id, run_config=run_config)
    summary = run_ablation(
        examples=examples,
        video_root=args.video_root,
        feature_root=args.feature_root,
        derived_cache_root=args.derived_cache_root,
        output_root=args.output_root / output_name,
        run_config=run_config,
        answer_config=AnswerConfig(
            model_id=args.model_id,
            backend=args.backend,
            image_max_size=args.image_max_size,
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
            api_base_url=args.api_base_url or AnswerConfig.api_base_url,
            api_key_env_var=args.api_key_env_var,
            api_requests_per_minute=args.api_requests_per_minute,
            api_tokens_per_minute=args.api_tokens_per_minute,
            api_timeout_sec=args.api_timeout_sec,
        ),
        subtitle_root=args.subtitle_root,
        subtitle_tar=args.subtitle_tar,
        api_answer_workers=args.api_answer_workers,
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
