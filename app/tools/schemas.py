"""Core tool types: execution context, definitions and normalized results."""

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple, Type
from uuid import UUID

from pydantic import BaseModel


# Context-injectable input names. The executor owns how each one is resolved and authorized.
CONTEXT_INPUTS = ("project_id",)


@dataclass
class ToolContext:
    """Authorization and runtime context for one tool execution.

    `user_id` must come from a verified J.A.R.V.I.S. session (or other trusted
    server-side source), never from tool input. Tool inputs reject extra fields,
    so a caller cannot smuggle an identity in through them.
    """

    user_id: UUID
    conn: Any
    # Internal UUID of the project the caller selected. It must come from trusted backend
    # context (never from tool input or the model) and is ownership-checked by the executor.
    project_id: Optional[UUID] = None
    deadline: Optional[float] = None  # time.monotonic() value, set by the executor
    stats: Dict[str, int] = field(default_factory=dict)

    def time_remaining(self) -> float:
        if self.deadline is None:
            return float("inf")
        return self.deadline - time.monotonic()

    def count_request(self) -> None:
        self.stats["github_requests"] = self.stats.get("github_requests", 0) + 1


@dataclass(frozen=True)
class ToolConstraints:
    timeout_seconds: float = 20.0
    max_output_bytes: int = 1_000_000


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    input_model: Type[BaseModel]
    output_model: Type[BaseModel]
    handler: Callable[[ToolContext, Any], BaseModel]
    requires_user: bool = True
    source: str = "github"
    constraints: ToolConstraints = ToolConstraints()
    # Input fields the executor fills in from trusted context instead of accepting them from the
    # caller. They exist in input_model (the internal schema) but are removed from the schema
    # shown to an LLM. Each name must be a key of CONTEXT_INPUTS.
    context_inputs: Tuple[str, ...] = ()


class ToolMetadata(BaseModel):
    duration_ms: int
    source: str
    github_requests: int = 0


class ToolErrorInfo(BaseModel):
    code: str
    message: str
    retryable: bool = False


class ToolResult(BaseModel):
    tool_name: str
    status: str  # "success" | "error"
    data: Optional[Dict[str, Any]] = None
    metadata: ToolMetadata
    error: Optional[ToolErrorInfo] = None

