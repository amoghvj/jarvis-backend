"""`search_events`, `get_event_details` and `get_trace`."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

from app.tools.errors import ToolError
from app.tools.schemas import ToolContext, ToolDefinition
from app.tools.telemetry.common import (
    EVENT_COLUMNS, SEVERITIES, TELEMETRY_CONSTRAINTS, EventSummary, FilterText, Identifier, IncidentSummary,
    Pagination, Severity, SourceLocation, TelemetryInput, TelemetryResult, TimeRangeInput, event_summary,
    incident_summary, like_pattern, paginate, redact_json, redact_text, result_fields,
)

# --- search_events -----------------------------------------------------------


class SearchEventsInput(TimeRangeInput):
    service_id: Optional[FilterText] = None
    environment: Optional[FilterText] = None
    severity: Optional[Severity] = Field(default=None, description="Exact severity")
    min_severity: Optional[Severity] = Field(default=None, description="This severity or higher")
    event_type: Optional[FilterText] = Field(default=None, description="For example error, log or startup")
    message_contains: Optional[FilterText] = Field(default=None, description="Literal, case-insensitive text")
    error_type: Optional[FilterText] = None
    source_file: Optional[FilterText] = Field(default=None, description="Exact file path as reported by the SDK")
    source_function: Optional[FilterText] = None
    trace_id: Optional[Identifier] = None
    request_id: Optional[FilterText] = None

    @model_validator(mode="after")
    def _severity_choice(self) -> "SearchEventsInput":
        if self.severity and self.min_severity:
            raise ValueError("use either severity or min_severity, not both")
        return self


class SearchEventsOutput(TelemetryResult):
    events: List[EventSummary]
    pagination: Pagination
    applied_filters: Dict[str, Any]


def search_events(ctx: ToolContext, inp: SearchEventsInput) -> SearchEventsOutput:
    where, params = ["e.project_id = %s"], [inp.project_id]
    applied: Dict[str, Any] = {}

    def exact(column: str, name: str) -> None:
        value = getattr(inp, name)
        if value is not None:
            where.append(f"e.{column} = %s")
            params.append(value)
            applied[name] = value

    for column, name in (
        ("service_id", "service_id"), ("environment", "environment"), ("severity", "severity"),
        ("event_type", "event_type"), ("error_type", "error_type"), ("source_file", "source_file"),
        ("source_function", "source_function"), ("trace_id", "trace_id"), ("request_id", "request_id"),
    ):
        exact(column, name)
    if inp.min_severity:
        where.append("e.severity = any(%s)")
        params.append(list(SEVERITIES[SEVERITIES.index(inp.min_severity):]))
        applied["min_severity"] = inp.min_severity
    if inp.message_contains:
        where.append("e.message ilike %s escape '\\'")
        params.append(like_pattern(inp.message_contains))
        applied["message_contains"] = inp.message_contains
    if inp.since:
        where.append("e.event_timestamp >= %s")
        params.append(inp.since)
        applied["since"] = inp.since.isoformat()
    if inp.until:
        where.append("e.event_timestamp <= %s")
        params.append(inp.until)
        applied["until"] = inp.until.isoformat()
    rows = ctx.conn.execute(
        f"select {EVENT_COLUMNS} from public.telemetry_events e where {' and '.join(where)} "
        "order by e.event_timestamp desc, e.id desc limit %s offset %s",
        (*params, inp.limit + 1, inp.offset),
    ).fetchall()
    page, pagination = paginate(rows, inp.limit, inp.offset)
    return SearchEventsOutput(
        events=[event_summary(r) for r in page], pagination=pagination, applied_filters=applied,
        **result_fields(pagination, ["Ordered newest first; filters are combined with AND."]),
    )


# --- get_event_details -------------------------------------------------------

MAX_STACK_CHARS = 50_000


class EventDetailsInput(TelemetryInput):
    event_uuid: Optional[UUID] = Field(default=None, description="Internal event UUID from other tool results")
    event_id: Optional[FilterText] = Field(default=None, description="The event_id the SDK generated")

    @model_validator(mode="after")
    def _exactly_one(self) -> "EventDetailsInput":
        if (self.event_uuid is None) == (self.event_id is None):
            raise ValueError("provide exactly one of event_uuid or event_id")
        return self


class EventDetails(BaseModel):
    id: UUID
    event_id: str
    schema_version: Optional[str] = None
    event_type: str
    classification: str
    timestamp: datetime
    received_at: datetime
    application: Dict[str, Any]
    service_id: Optional[str] = None
    environment: Optional[str] = None
    instance_id: Optional[str] = None
    runtime: Dict[str, Any]
    severity: Optional[str] = None
    message: Optional[str] = None
    logger_name: Optional[str] = None
    error_type: Optional[str] = None
    stack_trace: Optional[str] = None
    stack_trace_truncated: bool = False
    source: SourceLocation
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    request_id: Optional[str] = None
    thread_name: Optional[str] = None
    attributes: Dict[str, Any]
    payload: Dict[str, Any]
    incident: Optional[IncidentSummary] = None


class EventDetailsOutput(TelemetryResult):
    event: EventDetails


def get_event_details(ctx: ToolContext, inp: EventDetailsInput) -> EventDetailsOutput:
    column, value = ("id", inp.event_uuid) if inp.event_uuid else ("event_id", inp.event_id)
    row = ctx.conn.execute(
        f"select * from public.telemetry_events e where e.project_id = %s and e.{column} = %s",
        (inp.project_id, value),
    ).fetchone()
    if row is None:
        raise ToolError("event_not_found", "Event not found in this project")
    incident = None
    if row["incident_id"]:
        inc = ctx.conn.execute(
            "select id, title, status, severity, service_id, environment, event_count, first_seen_at, last_seen_at "
            "from public.incidents where id = %s and project_id = %s",
            (row["incident_id"], inp.project_id),
        ).fetchone()
        incident = incident_summary(inc) if inc else None
    stack = redact_text(row["stack_trace"])
    stack_truncated = bool(stack and len(stack) > MAX_STACK_CHARS)
    event = EventDetails(
        id=row["id"], event_id=row["event_id"], schema_version=row["schema_version"], event_type=row["event_type"],
        classification=row["classification"], timestamp=row["event_timestamp"], received_at=row["received_at"],
        application=redact_json(row["application"]), service_id=row["service_id"], environment=row["environment"],
        instance_id=row["instance_id"], runtime=redact_json(row["runtime"]), severity=row["severity"],
        message=redact_text(row["message"]), logger_name=row["logger_name"], error_type=row["error_type"],
        stack_trace=stack[:MAX_STACK_CHARS] if stack else stack, stack_trace_truncated=stack_truncated,
        source=SourceLocation(file=row["source_file"], function=row["source_function"], line=row["source_line"]),
        trace_id=row["trace_id"], span_id=row["span_id"], request_id=row["request_id"],
        thread_name=row["thread_name"], attributes=redact_json(row["attributes"]),
        payload=redact_json(row["payload"]), incident=incident,
    )
    notes = ["payload is the original event as received, with secret-looking keys and values redacted."]
    fields = result_fields(None, notes)
    if stack_truncated:
        fields.update(complete=False, limits_hit=["stack_trace_truncated"])
        fields["notes"].append(f"The stack trace exceeded {MAX_STACK_CHARS} characters and was truncated.")
    return EventDetailsOutput(event=event, **fields)


# --- get_trace ---------------------------------------------------------------


class GetTraceInput(TelemetryInput):
    trace_id: Identifier
    limit: int = Field(default=100, ge=1, le=200)
    offset: int = Field(default=0, ge=0, le=10_000)


class SpanGroup(BaseModel):
    span_id: Optional[str] = None
    event_count: int
    first_event_at: datetime
    last_event_at: datetime


class ServiceGroup(BaseModel):
    service_id: Optional[str] = None
    event_count: int


class GetTraceOutput(TelemetryResult):
    trace_id: str
    events: List[EventSummary]
    spans: List[SpanGroup]
    services: List[ServiceGroup]
    first_event_at: Optional[datetime] = None
    last_event_at: Optional[datetime] = None
    total_events: int
    pagination: Pagination


def get_trace(ctx: ToolContext, inp: GetTraceInput) -> GetTraceOutput:
    rows = ctx.conn.execute(
        f"select {EVENT_COLUMNS} from public.telemetry_events e where e.project_id = %s and e.trace_id = %s "
        "order by e.event_timestamp, e.received_at, e.id limit %s offset %s",
        (inp.project_id, inp.trace_id, inp.limit + 1, inp.offset),
    ).fetchall()
    page, pagination = paginate(rows, inp.limit, inp.offset)
    spans = ctx.conn.execute(
        """select span_id, count(*) as n, min(event_timestamp) as first_at, max(event_timestamp) as last_at
           from public.telemetry_events where project_id = %s and trace_id = %s
           group by span_id order by min(event_timestamp), span_id nulls first limit 200""",
        (inp.project_id, inp.trace_id),
    ).fetchall()
    services = ctx.conn.execute(
        """select service_id, count(*) as n from public.telemetry_events
           where project_id = %s and trace_id = %s group by service_id order by count(*) desc, service_id limit 50""",
        (inp.project_id, inp.trace_id),
    ).fetchall()
    total = sum(r["n"] for r in services)
    if not total:
        raise ToolError("trace_not_found", "No events with that trace_id exist in this project")
    notes = [
        "Events are grouped by trace_id and span_id only. Parent/child span relationships are not recorded "
        "in the telemetry schema, so none are inferred and this is not a full distributed-trace reconstruction.",
    ]
    return GetTraceOutput(
        trace_id=inp.trace_id, events=[event_summary(r) for r in page],
        spans=[SpanGroup(span_id=s["span_id"], event_count=s["n"], first_event_at=s["first_at"], last_event_at=s["last_at"])
               for s in spans],
        services=[ServiceGroup(service_id=s["service_id"], event_count=s["n"]) for s in services],
        first_event_at=min((s["first_at"] for s in spans), default=None),
        last_event_at=max((s["last_at"] for s in spans), default=None),
        total_events=int(total), pagination=pagination, **result_fields(pagination, notes),
    )


TOOLS = [
    ToolDefinition(
        name="search_events",
        description=(
            "Purpose: Search and filter persisted project telemetry events by service_id, environment, severity, "
            "event_type, classification, message substring, error_type, source_file, source_function, trace_id, "
            "request_id, incident_id, and time range (since/until). "
            "When to use: When exploring project telemetry broadly, finding recent errors/logs, or when no incident_id is given. "
            "When not to use: If you already have an incident_id, start with get_incident; if you have a trace_id, prefer get_trace. "
            "Scope & limitations: Returns newest-first event summaries (paginated); automatically scoped to the active project."
        ),
        input_model=SearchEventsInput, output_model=SearchEventsOutput, handler=search_events,
        source="database", constraints=TELEMETRY_CONSTRAINTS, context_inputs=("project_id",),
    ),
    ToolDefinition(
        name="get_event_details",
        description=(
            "Purpose: Retrieve one complete persisted telemetry event by event_uuid (internal UUID) or event_id "
            "(SDK event ID), including full redacted stack_trace, source file/function/line, runtime, attributes, "
            "redacted raw payload, and linked incident summary. "
            "When to use: When an event summary from get_incident, get_related_events, or search_events needs stack trace or attribute inspection. "
            "When not to use: Do not call until you have a specific event_uuid or event_id. "
            "Scope & limitations: Automatically scoped to the active project; stack traces over 16 KB are marked truncated."
        ),
        input_model=EventDetailsInput, output_model=EventDetailsOutput, handler=get_event_details,
        source="database", constraints=TELEMETRY_CONSTRAINTS, context_inputs=("project_id",),
    ),
    ToolDefinition(
        name="get_trace",
        description=(
            "Purpose: Retrieve all persisted telemetry events sharing a trace_id in chronological order, along "
            "with span_id groupings and affected services. "
            "When to use: When an event or incident includes a trace_id and you need to inspect cross-step or cross-service execution flow. "
            "When not to use: Do not use without a known trace_id (use search_events or get_related_events instead). "
            "Scope & limitations: Groups by trace_id and span_id only; parent-child span hierarchy is not recorded or inferred; automatically scoped to the active project."
        ),
        input_model=GetTraceInput, output_model=GetTraceOutput, handler=get_trace,
        source="database", constraints=TELEMETRY_CONSTRAINTS, context_inputs=("project_id",),
    ),
]
