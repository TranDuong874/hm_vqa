from __future__ import annotations

import json
import sys
from typing import Any

from .schemas import ToolSpec
from .tools import (
    compare_time_tool,
    derive_window_tool,
    entity_to_time_tool,
    parse_query_tool,
    route_query_tool,
    select_evidence_tool,
    time_to_entity_tool,
    verify_consistency_tool,
)


PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {"name": "hm-vqa-mcp", "version": "0.1.0"}


def _tool_specs() -> dict[str, ToolSpec]:
    return {
        "parse_query": ToolSpec(
            name="parse_query",
            description="Parse a video QA query into taxonomy, entities, and recommended procedure.",
            input_schema={
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "choices": {"type": "array", "items": {}},
                },
                "required": ["question"],
            },
            handler=parse_query_tool,
        ),
        "route_query": ToolSpec(
            name="route_query",
            description="Route a query to the taxonomy family and emit a tool procedure plan.",
            input_schema={
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "choices": {"type": "array", "items": {}},
                },
                "required": ["question"],
            },
            handler=route_query_tool,
        ),
        "compare_time": ToolSpec(
            name="compare_time",
            description="Compare candidate intervals against reference intervals with overlap and gap metrics.",
            input_schema={
                "type": "object",
                "properties": {
                    "candidates": {"type": "array", "items": {}},
                    "references": {"type": "array", "items": {}},
                    "tolerance_sec": {"type": "number"},
                },
                "required": ["candidates", "references"],
            },
            handler=compare_time_tool,
        ),
        "derive_window": ToolSpec(
            name="derive_window",
            description="Derive a time window before/after/between anchor intervals.",
            input_schema={
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "enum": ["before", "after", "between"]},
                    "anchor_interval": {},
                    "left_interval": {},
                    "right_interval": {},
                    "window_sec": {"type": "number"},
                    "pad_sec": {"type": "number"},
                },
                "required": ["mode"],
            },
            handler=derive_window_tool,
        ),
        "entity_to_time": ToolSpec(
            name="entity_to_time",
            description="Rerank candidate intervals for a target entity using lightweight lexical matching and prior scores.",
            input_schema={
                "type": "object",
                "properties": {
                    "entity": {"type": "string"},
                    "candidates": {"type": "array", "items": {}},
                    "top_k": {"type": "integer"},
                },
                "required": ["entity", "candidates"],
            },
            handler=entity_to_time_tool,
        ),
        "time_to_entity": ToolSpec(
            name="time_to_entity",
            description="Find entities closest to a requested time interval from a candidate set.",
            input_schema={
                "type": "object",
                "properties": {
                    "interval": {},
                    "candidates": {"type": "array", "items": {}},
                    "top_k": {"type": "integer"},
                },
                "required": ["interval", "candidates"],
            },
            handler=time_to_entity_tool,
        ),
        "select_evidence": ToolSpec(
            name="select_evidence",
            description="Select temporally diverse evidence frames from scored frame candidates.",
            input_schema={
                "type": "object",
                "properties": {
                    "frames": {"type": "array", "items": {}},
                    "limit": {"type": "integer"},
                    "min_gap_sec": {"type": "number"},
                },
                "required": ["frames"],
            },
            handler=select_evidence_tool,
        ),
        "verify_consistency": ToolSpec(
            name="verify_consistency",
            description="Check whether a selected interval is temporally consistent with an answer interval.",
            input_schema={
                "type": "object",
                "properties": {
                    "selected_interval": {},
                    "answer_interval": {},
                    "tolerance_sec": {"type": "number"},
                },
                "required": ["selected_interval", "answer_interval"],
            },
            handler=verify_consistency_tool,
        ),
    }


TOOLS = _tool_specs()


def _read_message() -> dict[str, Any] | None:
    headers: dict[str, str] = {}
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        if line in (b"\r\n", b"\n"):
            break
        decoded = line.decode("utf-8").strip()
        if not decoded:
            break
        name, _, value = decoded.partition(":")
        headers[name.lower()] = value.strip()

    content_length = int(headers.get("content-length", "0"))
    if content_length <= 0:
        return None
    body = sys.stdin.buffer.read(content_length)
    if not body:
        return None
    return json.loads(body.decode("utf-8"))


def _write_message(payload: dict[str, Any]) -> None:
    body = json.dumps(payload).encode("utf-8")
    sys.stdout.buffer.write(f"Content-Length: {len(body)}\r\n\r\n".encode("utf-8"))
    sys.stdout.buffer.write(body)
    sys.stdout.buffer.flush()


def _result(message_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": message_id, "result": result}


def _error(message_id: Any, code: int, message: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": message_id,
        "error": {"code": code, "message": message},
    }


def _handle_initialize(message_id: Any) -> dict[str, Any]:
    return _result(
        message_id,
        {
            "protocolVersion": PROTOCOL_VERSION,
            "serverInfo": SERVER_INFO,
            "capabilities": {
                "tools": {"listChanged": False},
            },
        },
    )


def _handle_tools_list(message_id: Any) -> dict[str, Any]:
    tools = [
        {
            "name": spec.name,
            "description": spec.description,
            "inputSchema": spec.input_schema,
        }
        for spec in TOOLS.values()
    ]
    return _result(message_id, {"tools": tools})


def _handle_tools_call(message_id: Any, params: dict[str, Any]) -> dict[str, Any]:
    name = str(params.get("name", "")).strip()
    arguments = params.get("arguments") or {}
    if name not in TOOLS:
        return _error(message_id, -32601, f"Unknown tool: {name}")
    try:
        output = TOOLS[name].handler(arguments)
    except Exception as exc:  # noqa: BLE001
        return _error(message_id, -32000, f"{type(exc).__name__}: {exc}")

    return _result(
        message_id,
        {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(output, indent=2, sort_keys=True),
                }
            ],
            "structuredContent": output,
            "isError": False,
        },
    )


def _dispatch(request: dict[str, Any]) -> dict[str, Any] | None:
    method = request.get("method")
    message_id = request.get("id")
    params = request.get("params") or {}

    if method == "initialize":
        return _handle_initialize(message_id)
    if method == "notifications/initialized":
        return None
    if method == "ping":
        return _result(message_id, {})
    if method == "tools/list":
        return _handle_tools_list(message_id)
    if method == "tools/call":
        return _handle_tools_call(message_id, params)
    if message_id is None:
        return None
    return _error(message_id, -32601, f"Unsupported method: {method}")


def main() -> int:
    while True:
        request = _read_message()
        if request is None:
            return 0
        response = _dispatch(request)
        if response is not None:
            _write_message(response)


if __name__ == "__main__":
    raise SystemExit(main())
