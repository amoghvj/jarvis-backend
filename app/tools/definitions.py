"""Explicit registration of all tools. Nothing is discovered or imported dynamically."""

from typing import Optional

from app.tools.code import configuration, dependencies, references, symbols, tests_analysis
from app.tools.github import commits, files, repositories
from app.tools.telemetry import events as telemetry_events
from app.tools.telemetry import incidents as telemetry_incidents
from app.tools.telemetry import service as telemetry_service
from app.tools.registry import ToolRegistry

_default_registry: Optional[ToolRegistry] = None


def build_default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    for tool in (
        *repositories.TOOLS,
        *files.TOOLS,
        *commits.TOOLS,
        *symbols.TOOLS,
        *references.TOOLS,
        *dependencies.TOOLS,
        *configuration.TOOLS,
        *tests_analysis.TOOLS,
        *telemetry_incidents.TOOLS,
        *telemetry_events.TOOLS,
        *telemetry_service.TOOLS,
    ):
        registry.register(tool)
    return registry


def get_default_registry() -> ToolRegistry:
    global _default_registry
    if _default_registry is None:
        _default_registry = build_default_registry()
    return _default_registry

