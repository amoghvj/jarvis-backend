"""`get_service_context`."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.tools.errors import ToolError
from app.tools.schemas import ToolContext, ToolDefinition
from app.tools.telemetry.common import (
    EVENT_COLUMNS, TELEMETRY_CONSTRAINTS, EventSummary, FilterText, IncidentSummary, RepositoryContext,
    TelemetryInput, TelemetryResult, event_summary, incident_summary, repository_context, result_fields,
)


class ServiceContextInput(TelemetryInput):
    service_id: FilterText
    environment: Optional[FilterText] = Field(default=None, description="Restrict to one environment")
    window_hours: int = Field(default=24, ge=1, le=168, description="Window for recent event/error frequency")


class EnvironmentSeen(BaseModel):
    environment: Optional[str] = None
    event_count: int
    last_event_at: datetime


class InstanceSeen(BaseModel):
    instance_id: Optional[str] = None
    event_count: int
    last_event_at: datetime


class RuntimeSeen(BaseModel):
    language: Optional[str] = None
    version: Optional[str] = None
    event_count: int
    last_event_at: datetime


class RecentActivity(BaseModel):
    window_hours: int
    total_events: int
    events_by_severity: Dict[str, int]
    error_events: int
    errors_per_hour: float


class ServiceContextOutput(TelemetryResult):
    service_id: str
    environment_filter: Optional[str] = None
    first_event_at: Optional[datetime] = None
    last_event_at: Optional[datetime] = None
    environments: List[EnvironmentSeen]
    instances: List[InstanceSeen]
    runtimes: List[RuntimeSeen]
    last_startup_event: Optional[EventSummary] = None
    recent_activity: RecentActivity
    recent_incidents: List[IncidentSummary]
    repository_context: RepositoryContext


def get_service_context(ctx: ToolContext, inp: ServiceContextInput) -> ServiceContextOutput:
    scope = "project_id = %s and service_id = %s"
    params: List[Any] = [inp.project_id, inp.service_id]
    if inp.environment is not None:
        scope += " and environment = %s"
        params.append(inp.environment)
    conn = ctx.conn

    overall = conn.execute(
        f"select count(*) as n, min(event_timestamp) as first_at, max(event_timestamp) as last_at "
        f"from public.telemetry_events where {scope}", params,
    ).fetchone()
    repos = repository_context(conn, inp.project_id, inp.service_id)
    if not overall["n"] and not any(r.matches_service for r in repos.repositories):
        raise ToolError("service_not_found", "No telemetry has been recorded for that service in this project")

    environments = conn.execute(
        f"select environment, count(*) as n, max(event_timestamp) as last_at from public.telemetry_events "
        f"where {scope} group by environment order by max(event_timestamp) desc, environment nulls last limit 20", params,
    ).fetchall()
    instances = conn.execute(
        f"select instance_id, count(*) as n, max(event_timestamp) as last_at from public.telemetry_events "
        f"where {scope} group by instance_id order by max(event_timestamp) desc, instance_id nulls last limit 20", params,
    ).fetchall()
    runtimes = conn.execute(
        f"""select runtime->>'language' as language, runtime->>'version' as version, count(*) as n,
                   max(event_timestamp) as last_at
            from public.telemetry_events where {scope}
            group by runtime->>'language', runtime->>'version'
            order by max(event_timestamp) desc, 1 nulls last, 2 nulls last limit 10""", params,
    ).fetchall()
    startup = conn.execute(
        f"select {EVENT_COLUMNS} from public.telemetry_events e where e.{scope.replace(' and ', ' and e.')} "
        "and e.event_type = 'startup' order by e.event_timestamp desc, e.id desc limit 1", params,
    ).fetchone()
    by_severity = conn.execute(
        f"select coalesce(severity, 'UNSPECIFIED') as severity, count(*) as n from public.telemetry_events "
        f"where {scope} and event_timestamp >= now() - make_interval(hours => %s) group by 1 order by 1",
        (*params, inp.window_hours),
    ).fetchall()
    severity_counts = {r["severity"]: int(r["n"]) for r in by_severity}
    errors = severity_counts.get("ERROR", 0) + severity_counts.get("CRITICAL", 0)

    incident_scope = "project_id = %s and service_id = %s" + (" and environment = %s" if inp.environment is not None else "")
    incidents = conn.execute(
        f"select id, title, status, severity, service_id, environment, event_count, first_seen_at, last_seen_at "
        f"from public.incidents where {incident_scope} order by last_seen_at desc nulls last, id limit 10", params,
    ).fetchall()

    notes = [
        "Derived only from received telemetry. No deployment, release or infrastructure information is "
        "available, and none is inferred; last_startup_event is the last SDK start observed, not a deployment.",
        "Instances, environments and runtimes lists are limited to the 20/20/10 most recently active.",
    ]
    return ServiceContextOutput(
        service_id=inp.service_id, environment_filter=inp.environment,
        first_event_at=overall["first_at"], last_event_at=overall["last_at"],
        environments=[EnvironmentSeen(environment=r["environment"], event_count=r["n"], last_event_at=r["last_at"]) for r in environments],
        instances=[InstanceSeen(instance_id=r["instance_id"], event_count=r["n"], last_event_at=r["last_at"]) for r in instances],
        runtimes=[RuntimeSeen(language=r["language"], version=r["version"], event_count=r["n"], last_event_at=r["last_at"]) for r in runtimes],
        last_startup_event=event_summary(startup) if startup else None,
        recent_activity=RecentActivity(
            window_hours=inp.window_hours, total_events=sum(severity_counts.values()),
            events_by_severity=severity_counts, error_events=errors,
            errors_per_hour=round(errors / inp.window_hours, 3),
        ),
        recent_incidents=[incident_summary(r) for r in incidents], repository_context=repos,
        **result_fields(None, notes),
    )


TOOLS = [
    ToolDefinition(
        name="get_service_context",
        description=(
            "Get what telemetry shows about one service: environments, instances and runtimes seen, last "
            "startup event, recent event and error frequency, recent incidents and its verified repositories. "
            "No deployment information is available or inferred."
        ),
        input_model=ServiceContextInput, output_model=ServiceContextOutput, handler=get_service_context,
        source="database", constraints=TELEMETRY_CONSTRAINTS, context_inputs=("project_id",),
    ),
]

