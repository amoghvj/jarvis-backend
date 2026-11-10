"""Shared, allowlisted tool executor.

Resolve -> validate input -> verify authorization -> run handler under a timeout ->
validate/limit output -> normalize result. Raw exceptions are logged (redacted) and
never returned.
"""

import dataclasses
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Any, Optional
from uuid import UUID

from pydantic import ValidationError

from app import repositories as db_repo
from app.tools.errors import ToolError, redact
from app.tools.registry import ToolRegistry
from app.tools.schemas import (
    ToolContext,
    ToolDefinition,
    ToolErrorInfo,
    ToolMetadata,
    ToolResult,
)

logger = logging.getLogger("jarvis.tools")

# Handlers are blocking (HTTP + DB), so they run in worker threads that the
# executor can stop waiting for. Per-request HTTP timeouts and the context
# deadline make a timed-out handler stop promptly rather than run on.
_POOL = ThreadPoolExecutor(max_workers=8, thread_name_prefix="jarvis-tool")


def _summarize_validation(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors()[:10]:
        loc = ".".join(str(p) for p in err.get("loc", ())) or "input"
        parts.append(f"{loc}: {err.get('msg', 'invalid')}")  # never echoes the input value
    return redact("; ".join(parts))


def _check_authorization(tool: ToolDefinition, context: Any) -> None:
    if not tool.requires_user:
        return
    if not isinstance(context, ToolContext) or not isinstance(context.user_id, UUID) or context.conn is None:
        raise ToolError("unauthorized", "A verified user context is required")
    try:
        user = db_repo.get_user_by_id(context.conn, context.user_id)
    except Exception as exc:  # DB trouble must not look like an authorization decision
        logger.error("tool authorization lookup failed: %s", redact(f"{type(exc).__name__}: {exc}"))
        raise ToolError("internal_error", "Tool execution failed", retryable=True)
    if not user:
        raise ToolError("unauthorized", "A verified user context is required")


def _resolve_context_inputs(tool: ToolDefinition, context: Any) -> dict:
    """Resolve each injected input from trusted context and verify the caller may use it."""
    if not isinstance(context, ToolContext) or context.conn is None:
        raise ToolError("unauthorized", "A verified user context is required")
    resolved: dict = {}
    for name in tool.context_inputs:
        if name == "project_id":
            project_id = context.project_id
            if not isinstance(project_id, UUID):
                raise ToolError("project_context_required", "This tool requires a selected project")
            try:
                project = db_repo.get_project_for_user(context.conn, context.user_id, project_id)
            except Exception as exc:  # DB trouble must not look like an authorization decision
                logger.error("tool project lookup failed: %s", redact(f"{type(exc).__name__}: {exc}"))
                raise ToolError("internal_error", "Tool execution failed", retryable=True)
            if project is None:  # not found and not-yours are deliberately indistinguishable
                raise ToolError("project_not_found", "Project not found")
            resolved["project_id"] = project_id
        else:  # unreachable: the registry only admits names listed in CONTEXT_INPUTS
            raise ToolError("internal_error", "Tool execution failed")
    return resolved


def execute_tool(
    tool_name: str,
    raw_input: Optional[dict],
    context: Any,
    registry: Optional[ToolRegistry] = None,
) -> ToolResult:
    if registry is None:
        from app.tools.definitions import get_default_registry

        registry = get_default_registry()

    started = time.perf_counter()
    source = "unknown"
    run_context = context if isinstance(context, ToolContext) else None

    def finish(data=None, error: Optional[ToolError] = None) -> ToolResult:
        requests = run_context.stats.get("github_requests", 0) if run_context else 0
        metadata = ToolMetadata(
            duration_ms=int((time.perf_counter() - started) * 1000),
            source=source,
            github_requests=requests,
        )
        if error is not None:
            return ToolResult(
                tool_name=str(tool_name)[:64],
                status="error",
                metadata=metadata,
                error=ToolErrorInfo(
                    code=error.code, message=redact(error.message), retryable=error.retryable
                ),
            )
        return ToolResult(tool_name=tool_name, status="success", data=data, metadata=metadata)

    try:
        tool = registry.get(tool_name)
        source = tool.source

        if raw_input is None:
            raw_input = {}
        if not isinstance(raw_input, dict):
            raise ToolError("invalid_input", "Tool input must be a JSON object")

        # Fields the server injects (e.g. the selected project) are never accepted from the
        # caller: the model/LLM cannot choose, override or even name them. They are resolved from
        # trusted context and authorized BEFORE validation, then merged into the internal input.
        injected: dict = {}
        if tool.context_inputs:
            supplied = [name for name in tool.context_inputs if name in raw_input]
            if supplied:
                raise ToolError("invalid_input", f"{', '.join(supplied)}: supplied by the server and cannot be set")
            _check_authorization(tool, context)
            injected = _resolve_context_inputs(tool, context)
        try:
            parsed = tool.input_model.model_validate({**raw_input, **injected})
        except ValidationError as exc:
            raise ToolError("invalid_input", _summarize_validation(exc))

        if not tool.context_inputs:
            _check_authorization(tool, context)

        run_context = dataclasses.replace(
            context,
            deadline=time.monotonic() + tool.constraints.timeout_seconds,
            stats={},
        ) if isinstance(context, ToolContext) else ToolContext(
            user_id=UUID(int=0), conn=None,
            deadline=time.monotonic() + tool.constraints.timeout_seconds,
        )

        future = _POOL.submit(tool.handler, run_context, parsed)
        try:
            output = future.result(timeout=tool.constraints.timeout_seconds)
        except FutureTimeout:
            raise ToolError("timeout", "The tool exceeded its time limit", retryable=True)

        if not isinstance(output, tool.output_model):
            logger.error("tool %s returned an unexpected output type", tool.name)
            raise ToolError("internal_error", "Tool execution failed")
        data = output.model_dump(mode="json")
        if len(json.dumps(data, default=str)) > tool.constraints.max_output_bytes:
            raise ToolError(
                "result_too_large",
                "The tool result exceeded the output size limit; narrow the request",
            )
        return finish(data=data)

    except ToolError as exc:
        return finish(error=exc)
    except Exception as exc:
        # Raw details stay server-side and are redacted; callers get a generic error.
        logger.error(
            "tool %s failed: %s", str(tool_name)[:64], redact(f"{type(exc).__name__}: {exc}")
        )
        return finish(error=ToolError("internal_error", "Tool execution failed"))

