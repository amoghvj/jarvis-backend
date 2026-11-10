"""Thin adapter exposing the canonical J.A.R.V.I.S. ToolRegistry and executor to LangChain / Groq.

This module never re-implements tool handlers or introduces a second registry.
All tool metadata comes from `app.tools.definitions.get_default_registry()` and
all execution goes through `app.tools.executor.execute_tool` with a server-constructed
`ToolContext(user_id=..., project_id=..., conn=...)`.
"""

from dataclasses import dataclass
import json
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field, create_model

from app.tools.definitions import get_default_registry
from app.tools.errors import redact
from app.tools.executor import execute_tool
from app.tools.registry import ToolRegistry
from app.tools.schemas import ToolContext, ToolDefinition, ToolResult


# Concise user-visible step labels per tool for `agent_run_steps`.
TOOL_ACTIVITY_LABELS: Dict[str, str] = {
    "get_incident": "Investigating incident details and linked repositories",
    "get_related_events": "Retrieving related telemetry events for the incident",
    "get_incident_timeline": "Reconstructing incident chronological timeline",
    "search_events": "Searching project telemetry events",
    "get_event_details": "Inspecting telemetry event stack trace and payload",
    "get_trace": "Inspecting distributed trace events and spans",
    "get_service_context": "Retrieving service telemetry health and runtime context",
    "list_installations": "Listing linked GitHub App installations",
    "list_repositories": "Listing accessible GitHub repositories",
    "get_repository_metadata": "Inspecting GitHub repository metadata",
    "get_repository_tree": "Browsing repository directory tree",
    "get_file_contents": "Reading source file from repository",
    "search_repository": "Searching repository source files",
    "get_commit": "Inspecting commit metadata",
    "get_commit_diff": "Inspecting commit diff and code changes",
    "get_file_history": "Retrieving commit history for file",
    "find_symbol": "Locating symbol definition in source code",
    "get_function_context": "Extracting function source code and context",
    "find_references": "Finding symbol references across repository",
    "find_callers": "Finding upstream callers of function or method",
    "analyze_dependencies": "Analyzing static module import dependencies",
    "find_related_files": "Ranking related repository files",
    "inspect_configuration": "Inspecting allowlisted configuration and manifest files",
    "inspect_tests": "Inspecting static test coverage for file or symbol",
}


def activity_label_for_tool(tool_name: Optional[str]) -> str:
    if not tool_name:
        return "Executing investigation tool"
    return TOOL_ACTIVITY_LABELS.get(tool_name, f"Running tool {tool_name}")


# Standard parameter names whose semantics are self-explanatory and documented in the system prompt.
# Omitting repeated boilerplate descriptions on these 17x-repeated parameters keeps all 24 tool
# schemas well within Groq's 7,000-token per-minute/request budget on `qwen/qwen3.8-27b`.
_REPEATED_PARAM_NAMES = {
    "installation_id",
    "owner",
    "repo",
    "ref",
    "path_prefix",
    "max_files",
    "max_results",
    "page",
    "per_page",
    "limit",
    "offset",
    "since",
    "until",
    "case_sensitive",
    "recursive",
    "window_hours",
    "max_source_bytes",
    "include_textual",
    "include_unresolved",
    "include_dependents",
    "include_source_env_refs",
    "include_events",
    "event_limit",
}


# Concise, high-signal tool descriptions for the OpenAI/Groq wire payload (preserving purpose,
# when to use, when not to use, and scope while fitting all 24 tools into ~3,300 prompt tokens).
LLM_WIRE_DESCRIPTIONS: Dict[str, str] = {
    "get_incident": (
        "Fetch incident metadata, status, severity, fingerprint, linked GitHub repositories "
        "(installation_id, owner, repo), and recent events. Primary first tool when incident_id is known; "
        "do not use for broad event search (use search_events). Scoped to active project."
    ),
    "get_related_events": (
        "List telemetry events for an incident (scope='incident') or all incidents sharing its fingerprint "
        "(scope='fingerprint'). Use after get_incident for occurrence history; for one event's full stack trace "
        "use get_event_details. Scoped to active project."
    ),
    "get_incident_timeline": (
        "Chronological incident timeline interleaving telemetry events (event_time) and lifecycle transitions "
        "(recorded_time) with derived duration/counts. Use to analyze incident progression/escalation; "
        "use get_incident first for metadata/repos. Scoped to active project."
    ),
    "search_events": (
        "Filter project telemetry events by service_id, environment, severity, event_type, classification, "
        "message_contains, error_type, source_file/function, trace_id, request_id, or time range. Use for "
        "broad telemetry exploration; if incident_id is known start with get_incident. Scoped to active project."
    ),
    "get_event_details": (
        "Fetch one complete telemetry event by event_uuid or event_id including full redacted stack_trace, "
        "source file/function/line, runtime, attributes, and payload. Use when an event summary needs deep "
        "inspection. Scoped to active project."
    ),
    "get_trace": (
        "Fetch all telemetry events sharing a trace_id in chronological order with span_id and service "
        "groupings. Use when trace_id is known; parent-child span hierarchy is not recorded. Scoped to active project."
    ),
    "get_service_context": (
        "Summarize telemetry for one service_id: environments, instances, runtimes, last startup event, "
        "recent error rates, recent incidents, and verified GitHub repositories. Use for service-level "
        "health/repo discovery. Scoped to active project."
    ),
    "list_installations": (
        "List GitHub App installations linked to the current user. Use only when installation_id is not "
        "already provided by get_incident or get_service_context."
    ),
    "list_repositories": (
        "List GitHub repositories accessible under an installation_id. Use when owner/repo are not already "
        "known from incident or service repository_context."
    ),
    "get_repository_metadata": (
        "Get default_branch, language, visibility, and URLs for owner/repo under installation_id. Do not "
        "use to list files (use get_repository_tree) or read files (use get_file_contents)."
    ),
    "get_repository_tree": (
        "List files and directories in owner/repo at ref. Use to browse directory structure; use "
        "get_file_contents for known file paths or search_repository for text search."
    ),
    "get_file_contents": (
        "Read full UTF-8 text of a known file path in owner/repo at ref (refuses >512KB/binary). Use when "
        "path is known from a stack trace or search; for a single function prefer get_function_context."
    ),
    "search_repository": (
        "Bounded literal substring search across files in owner/repo at ref. Use for error strings or "
        "identifiers in unknown files; for AST definitions/callers prefer find_symbol or find_callers."
    ),
    "get_commit": (
        "Get author, timestamp, message, parents, and stats for commit sha in owner/repo. To see line-by-line "
        "code patches use get_commit_diff."
    ),
    "get_commit_diff": (
        "Get changed files and unified diff patches for commit sha in owner/repo (optionally filtered to path). "
        "Use after get_file_history to inspect suspect commits."
    ),
    "get_file_history": (
        "List commits that modified path in owner/repo walking back from ref. Use to check recent changes "
        "to a file implicated in an incident."
    ),
    "find_symbol": (
        "Locate AST definitions of functions, classes, methods, constants, or TS types by symbol_name in "
        "Python/JS/TS at ref. For full function body and imports use get_function_context."
    ),
    "get_function_context": (
        "Extract complete untruncated source of one function/method (symbol_name, optional path or Class.method) "
        "with signature, decorators, class, imports, and related same-file symbols."
    ),
    "find_references": (
        "Find AST references (imports, calls, attributes, identifiers) to symbol_name across owner/repo at ref "
        "with import-binding resolution confidence. For direct call sites only prefer find_callers."
    ),
    "find_callers": (
        "Find direct statically visible call sites of symbol_name and their enclosing caller functions in "
        "owner/repo at ref. Does not trace dynamic dispatch."
    ),
    "analyze_dependencies": (
        "Build static import dependency graph for path or path_prefix in owner/repo at ref (internal, "
        "external, stdlib, unresolved). Never executes code."
    ),
    "find_related_files": (
        "Rank repository files related to path (and optional symbol_name) by imports, symbol references, "
        "and test/directory heuristics."
    ),
    "inspect_configuration": (
        "Inspect allowlisted config/manifest files (.env.example, pyproject.toml, requirements.txt, package.json, "
        "tsconfig, vercel.json) and env var references at ref. Never reads .env or secrets."
    ),
    "inspect_tests": (
        "Statically locate test files and test cases relevant to path and optional symbol_name at ref, "
        "reporting direct calls and assertions without executing tests."
    ),
}


def _compact_property_schema(prop: Any, prop_name: str = "") -> Dict[str, Any]:
    """Compact a single property's JSON schema for LLM function-calling while preserving types, enums, and descriptions."""
    if not isinstance(prop, dict):
        return {"type": "string"}

    out: Dict[str, Any] = {}
    if "anyOf" in prop and isinstance(prop["anyOf"], list):
        non_null = [
            branch
            for branch in prop["anyOf"]
            if isinstance(branch, dict) and branch.get("type") != "null"
        ]
        if non_null:
            merged = dict(non_null[0])
            if "description" in prop and "description" not in merged:
                merged["description"] = prop["description"]
            return _compact_property_schema(merged, prop_name=prop_name)

    if "type" in prop:
        out["type"] = prop["type"]
    if "enum" in prop:
        out["enum"] = prop["enum"]
    if "items" in prop and isinstance(prop["items"], dict):
        out["items"] = _compact_property_schema(prop["items"])
    if (
        prop_name not in ("installation_id", "owner", "repo", "page", "per_page", "limit", "offset")
        and "description" in prop
        and isinstance(prop["description"], str)
    ):
        desc = prop["description"].strip().split(". ")[0]
        if desc:
            out["description"] = desc[:80]
    if "format" in prop and prop["format"] in ("uuid", "date-time"):
        out["format"] = prop["format"]
    if "type" not in out and "enum" not in out:
        out["type"] = "string"
    return out


def compact_input_schema(raw_schema: Dict[str, Any]) -> Dict[str, Any]:
    """Produce a clean, token-efficient OpenAI/Groq JSON Schema from a tool's LLM-facing input_schema.

    Note: The canonical `execute_tool` still validates against the full Pydantic `input_model`
    (with all regex patterns, bounds, and `extra='forbid'`). Compacting the LLM-facing schema
    keeps all 24 tools well within Groq's per-request token budget.
    """
    props = raw_schema.get("properties") or {}
    compacted_props = {
        name: _compact_property_schema(spec, prop_name=name)
        for name, spec in props.items()
    }
    result: Dict[str, Any] = {
        "type": "object",
        "properties": compacted_props,
    }
    required = raw_schema.get("required")
    if isinstance(required, list) and required:
        result["required"] = [r for r in required if r in compacted_props]
    return result


def build_llm_args_model(tool: ToolDefinition) -> type[BaseModel]:
    """Build a Pydantic model representing the LLM-facing arguments (excluding server-injected context_inputs)."""
    if not tool.context_inputs:
        return tool.input_model

    hidden = set(tool.context_inputs)
    field_defs: Dict[str, Any] = {}
    for fname, f_info in tool.input_model.model_fields.items():
        if fname in hidden:
            continue
        annotation = f_info.annotation or Any
        field_defs[fname] = (annotation, f_info)

    return create_model(
        f"{tool.input_model.__name__}LLM",
        __config__=ConfigDict(extra="forbid"),
        **field_defs,
    )


def build_openai_tool_schemas(registry: Optional[ToolRegistry] = None) -> List[Dict[str, Any]]:
    """Convert all registered J.A.R.V.I.S. tools into OpenAI/Groq function-calling tool definitions."""
    reg = registry or get_default_registry()
    tools_out: List[Dict[str, Any]] = []
    for item in reg.describe(include_internal=False):
        name = item["name"]
        wire_desc = LLM_WIRE_DESCRIPTIONS.get(name, item["description"])
        tools_out.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": wire_desc,
                    "parameters": compact_input_schema(item["input_schema"]),
                },
            }
        )
    return tools_out


def build_langchain_tools(
    context: ToolContext,
    registry: Optional[ToolRegistry] = None,
    max_result_chars: int = 16000,
) -> List[StructuredTool]:
    """Adapt all tools in `ToolRegistry` into `langchain_core.tools.StructuredTool` instances.

    Each `StructuredTool` delegates execution directly to `app.tools.executor.execute_tool`
    with the server-injected `ToolContext`.
    """
    reg = registry or get_default_registry()
    lc_tools: List[StructuredTool] = []

    for name in reg.names():
        tool_def = reg.get(name)
        args_model = build_llm_args_model(tool_def)

        def _make_runner(t_name: str):
            def _run(**kwargs: Any) -> str:
                outcome = dispatch_tool_call(
                    tool_name=t_name,
                    raw_args=kwargs,
                    tool_call_id="direct",
                    context=context,
                    registry=reg,
                    max_result_chars=max_result_chars,
                )
                return outcome.serialized_envelope

            return _run

        lc_tools.append(
            StructuredTool(
                name=tool_def.name,
                description=tool_def.description,
                args_schema=args_model,
                func=_make_runner(tool_def.name),
            )
        )
    return lc_tools


def _summarize_tool_result(tool_name: str, result: ToolResult) -> str:
    """Create a concise human-readable summary of a tool execution."""
    if result.status != "success" or result.error is not None:
        code = result.error.code if result.error else "error"
        msg = result.error.message if result.error else "Tool execution failed"
        return redact(f"{tool_name} failed ({code}): {msg}")[:300]

    data = result.data or {}
    if tool_name == "get_incident":
        inc = data.get("incident") or {}
        repos = (data.get("repository_context") or {}).get("repositories") or []
        return f"Retrieved incident '{inc.get('title', '')}' ({inc.get('severity', '')}, {inc.get('status', '')}) with {len(repos)} linked repository(ies)."
    if tool_name == "get_related_events":
        events = data.get("events") or []
        return f"Retrieved {len(events)} related telemetry event(s) (scope={data.get('scope', 'incident')})."
    if tool_name == "get_incident_timeline":
        timeline = data.get("timeline") or []
        return f"Retrieved incident timeline with {len(timeline)} entry(ies)."
    if tool_name == "search_events":
        events = data.get("events") or []
        return f"Found {len(events)} matching telemetry event(s)."
    if tool_name == "get_event_details":
        ev = data.get("event") or {}
        return f"Retrieved full details for event {ev.get('event_id') or ev.get('id') or ''}."
    if tool_name == "get_trace":
        return f"Retrieved trace '{data.get('trace_id', '')}' with {data.get('total_events', 0)} event(s) across {len(data.get('spans') or [])} span(s)."
    if tool_name == "get_service_context":
        return f"Retrieved service context for '{data.get('service_id', '')}' ({len(data.get('recent_incidents') or [])} recent incident(s))."
    if tool_name == "get_file_contents":
        return f"Read file '{data.get('path', '')}' ({data.get('line_count', 0)} lines)."
    if tool_name == "search_repository":
        matches = data.get("matches") or []
        return f"Searched repository for '{data.get('query', '')}': {len(matches)} match(es) across {data.get('files_scanned', 0)} scanned file(s)."
    if tool_name == "find_symbol":
        defs = data.get("definitions") or []
        return f"Located {len(defs)} definition(s) for symbol '{data.get('symbol_name', '')}'."
    if tool_name == "get_function_context":
        return f"Extracted function context for '{data.get('symbol_name', '')}' (resolved={data.get('resolved', False)})."
    if tool_name == "find_references":
        return f"Found {data.get('reference_count', 0)} reference(s) for '{data.get('symbol_name', '')}'."
    if tool_name == "find_callers":
        return f"Found {data.get('caller_count', 0)} caller(s) for '{data.get('symbol_name', '')}'."
    return f"Completed {tool_name} successfully."


def _bound_envelope_json(envelope: Dict[str, Any], max_chars: int) -> str:
    """Serialize the tool envelope to JSON, bounding large data payloads while keeping valid JSON."""
    raw = redact(json.dumps(envelope, default=str, ensure_ascii=False))
    if len(raw) <= max_chars:
        return raw

    # Shrink large fields inside `data` while preserving valid JSON structure.
    compact_env = dict(envelope)
    data = compact_env.get("data")
    if isinstance(data, dict):
        trimmed_data: Dict[str, Any] = {}
        budget_per_field = max(500, (max_chars - 600) // max(1, len(data)))
        for k, v in data.items():
            serialized_v = json.dumps(v, default=str, ensure_ascii=False)
            if len(serialized_v) <= budget_per_field:
                trimmed_data[k] = v
            elif isinstance(v, str):
                trimmed_data[k] = v[: max(200, budget_per_field - 60)] + "\n...[truncated]..."
            elif isinstance(v, list):
                kept: List[Any] = []
                used = 2
                for item in v:
                    item_s = json.dumps(item, default=str, ensure_ascii=False)
                    if used + len(item_s) + 2 > budget_per_field:
                        break
                    kept.append(item)
                    used += len(item_s) + 2
                trimmed_data[k] = kept
                trimmed_data[f"{k}_truncated_count"] = max(0, len(v) - len(kept))
            else:
                trimmed_data[k] = redact(serialized_v)[: max(200, budget_per_field - 40)] + "...[truncated]"
        trimmed_data["_envelope_truncated"] = True
        compact_env["data"] = trimmed_data

    raw_compact = redact(json.dumps(compact_env, default=str, ensure_ascii=False))
    if len(raw_compact) <= max_chars:
        return raw_compact

    # Final fallback guarantee: valid JSON within max_chars
    preview_len = max(200, max_chars - 350)
    fallback = {
        "ok": bool(envelope.get("ok")),
        "operation": envelope.get("operation"),
        "summary": envelope.get("summary"),
        "truncated": True,
        "data_preview": raw[:preview_len] + "...[truncated]",
        "metadata": envelope.get("metadata"),
        "error": envelope.get("error"),
    }
    return json.dumps(fallback, default=str, ensure_ascii=False)


@dataclass(frozen=True)
class ToolDispatchOutcome:
    tool_name: str
    tool_call_id: str
    ok: bool
    summary: str
    envelope: Dict[str, Any]
    serialized_envelope: str
    tool_message: ToolMessage
    step_input_summary: Dict[str, Any]
    step_output_summary: Dict[str, Any]


def dispatch_tool_call(
    *,
    tool_name: Any,
    raw_args: Any,
    tool_call_id: str,
    context: ToolContext,
    registry: Optional[ToolRegistry] = None,
    max_result_chars: int = 16000,
) -> ToolDispatchOutcome:
    """Validate and execute one model tool call through `execute_tool`, returning a standardized envelope."""
    reg = registry or get_default_registry()
    safe_name = str(tool_name or "").strip()[:64] or "unknown_tool"
    safe_call_id = str(tool_call_id or "call_0")

    parsed_args: Any = raw_args
    parse_error: Optional[str] = None
    if isinstance(raw_args, str):
        raw_str = raw_args.strip()
        if not raw_str:
            parsed_args = {}
        else:
            try:
                parsed_args = json.loads(raw_str)
            except Exception:
                parse_error = "Tool arguments were not valid JSON"
                parsed_args = None
    elif raw_args is None:
        parsed_args = {}

    if isinstance(parsed_args, dict) and "_raw_args" in parsed_args and len(parsed_args) == 1:
        parse_error = "Tool arguments were not a valid JSON object"

    if parse_error is not None or not isinstance(parsed_args, dict):
        summary = f"{safe_name} failed (invalid_input): {parse_error or 'Tool arguments must be a JSON object'}"
        envelope: Dict[str, Any] = {
            "ok": False,
            "operation": safe_name,
            "summary": summary,
            "error": {
                "code": "invalid_input",
                "message": parse_error or "Tool arguments must be a JSON object",
                "retryable": False,
            },
            "metadata": {"duration_ms": 0, "source": "dispatcher", "github_requests": 0},
        }
        serialized = _bound_envelope_json(envelope, max_result_chars)
        return ToolDispatchOutcome(
            tool_name=safe_name,
            tool_call_id=safe_call_id,
            ok=False,
            summary=summary,
            envelope=envelope,
            serialized_envelope=serialized,
            tool_message=ToolMessage(
                content=serialized,
                tool_call_id=safe_call_id,
                name=safe_name,
            ),
            step_input_summary={"activity": activity_label_for_tool(safe_name), "args": {}},
            step_output_summary={"ok": False, "summary": summary, "error_code": "invalid_input"},
        )

    result: ToolResult = execute_tool(safe_name, parsed_args, context, registry=reg)
    ok = result.status == "success" and result.error is None
    summary = _summarize_tool_result(safe_name, result)
    meta_dict = result.metadata.model_dump(mode="json")

    if ok:
        envelope = {
            "ok": True,
            "operation": result.tool_name,
            "summary": summary,
            "data": result.data or {},
            "metadata": meta_dict,
        }
        step_out = {
            "ok": True,
            "activity": activity_label_for_tool(safe_name),
            "summary": summary,
            "duration_ms": result.metadata.duration_ms,
            "source": result.metadata.source,
        }
    else:
        err_info = result.error.model_dump(mode="json") if result.error else {
            "code": "internal_error",
            "message": "Tool execution failed",
            "retryable": False,
        }
        envelope = {
            "ok": False,
            "operation": result.tool_name,
            "summary": summary,
            "error": err_info,
            "metadata": meta_dict,
        }
        step_out = {
            "ok": False,
            "activity": activity_label_for_tool(safe_name),
            "summary": summary,
            "error_code": err_info.get("code"),
            "error_message": err_info.get("message"),
            "retryable": err_info.get("retryable", False),
            "duration_ms": result.metadata.duration_ms,
        }

    serialized = _bound_envelope_json(envelope, max_result_chars)
    return ToolDispatchOutcome(
        tool_name=result.tool_name,
        tool_call_id=safe_call_id,
        ok=ok,
        summary=summary,
        envelope=envelope,
        serialized_envelope=serialized,
        tool_message=ToolMessage(
            content=serialized,
            tool_call_id=safe_call_id,
            name=result.tool_name,
        ),
        step_input_summary={
            "activity": activity_label_for_tool(safe_name),
            "args": parsed_args,
        },
        step_output_summary=step_out,
    )
