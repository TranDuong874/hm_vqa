# HM-VQA MCP Server

This package keeps MCP-facing tooling separate from the experiment runners in `src/pipeline/experiments` and dataset runners in `evals/`.

Current tool surface:

- `parse_query` (legacy alias for `route_query`)
- `route_query`
- `compare_time`
- `derive_window`
- `entity_to_time`
- `time_to_entity`
- `select_evidence`
- `verify_consistency`

Run locally over stdio:

```bash
.venv/bin/python -m mcp.hm_vqa_server.server
```

Current implementation notes:

- This is a small standalone stdio MCP server.
- The tools are deliberately lightweight and reusable.
- `route_query` emits a taxonomy label plus a suggested tool procedure for the query type.
- Retrieval-heavy dataset bindings should live in `adapters/` and call into pipeline/core code, not the other way around.
