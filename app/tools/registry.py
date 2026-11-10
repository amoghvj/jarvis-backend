"""Explicit, allowlisted tool registry. It stores definitions and never runs them."""

import re
from typing import Any, Dict, List

from pydantic import BaseModel

from app.tools.errors import ToolError
from app.tools.schemas import CONTEXT_INPUTS, ToolDefinition

_NAME_RE = re.compile(r"[a-z][a-z0-9_]{2,63}")


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        if not _NAME_RE.fullmatch(tool.name):
            raise ValueError(f"Invalid tool name '{tool.name}'")
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        if not (
            isinstance(tool.input_model, type)
            and issubclass(tool.input_model, BaseModel)
            and isinstance(tool.output_model, type)
            and issubclass(tool.output_model, BaseModel)
        ):
            raise ValueError(f"Tool '{tool.name}' must declare Pydantic input and output models")
        if not callable(tool.handler):
            raise ValueError(f"Tool '{tool.name}' has no callable handler")
        for name in tool.context_inputs:
            if name not in CONTEXT_INPUTS or name not in tool.input_model.model_fields:
                raise ValueError(f"Tool '{tool.name}' declares unsupported context input '{name}'")
        self._tools[tool.name] = tool

    def get(self, name: Any) -> ToolDefinition:
        tool = self._tools.get(name) if isinstance(name, str) else None
        if tool is None:
            shown = name[:64] if isinstance(name, str) else "?"
            raise ToolError("unknown_tool", f"Unknown tool '{shown}'")
        return tool

    def names(self) -> List[str]:
        return sorted(self._tools)

    def describe(self, include_internal: bool = False) -> List[Dict[str, Any]]:
        """Machine-readable definitions for future agent orchestration code.

        By default this is the LLM-facing view: fields the executor injects from trusted context
        (for example `project_id`) are removed from the input schema. `include_internal=True`
        returns the full internal schema, which is what the executor validates against.
        """
        return [
            {
                "name": t.name,
                "description": t.description,
                "source": t.source,
                "requires_user": t.requires_user,
                "input_schema": self.input_schema(t, include_internal),
                "output_schema": t.output_model.model_json_schema(),
                "constraints": {
                    "timeout_seconds": t.constraints.timeout_seconds,
                    "max_output_bytes": t.constraints.max_output_bytes,
                },
            }
            for _, t in sorted(self._tools.items())
        ]

    @staticmethod
    def input_schema(tool: ToolDefinition, include_internal: bool = False) -> Dict[str, Any]:
        schema = tool.input_model.model_json_schema()
        if include_internal or not tool.context_inputs:
            return schema
        hidden = set(tool.context_inputs)
        schema["properties"] = {k: v for k, v in schema.get("properties", {}).items() if k not in hidden}
        if "required" in schema:
            schema["required"] = [r for r in schema["required"] if r not in hidden]
            if not schema["required"]:
                del schema["required"]
        return schema


