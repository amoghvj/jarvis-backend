"""`get_incident`, `get_related_events` and `get_incident_timeline`."""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from app.tools.errors import ToolError
from app.tools.schemas import ToolContext, ToolDefinition
from app.tools.telemetry.common import (
    EVENT_COLUMNS, INCIDENT_SUMMARY_COLUMNS, TELEMETRY_CONSTRAINTS, EventSummary, IncidentSummary,
    PageInput, Pagination, ProjectContext, RepositoryContext, Severity, TelemetryInput, TelemetryResult,
    TimeRangeInput, event_summary, incident_summary, paginate, redact_json, redact_text, repository_context,
    result_fields, SEVERITIES,
)

_FULL_INCIDENT_COLUMNS = (
    INCIDENT_SUMMARY_COLUMNS + ", fingerprint, acknowledged_at, resolved_at, incident_data, created_at, updated_at"
)


def _load_incident(conn: Any, project_id: UUID, incident_id: UUID, columns: str = _FULL_INCIDENT_COLUMNS) -> Dict[str, Any]:
    row = conn.execute(
        f"select {columns} from public.incidents where id = %s and project_id = %s", (incident_id, project_id)
    ).fetchone()
    if row is None:  # other projects' incidents are indistinguishable from nonexistent ones
        raise ToolError("incident_not_found", "Incident not found in this project")
    return row


def _project_context(conn: Any, project_id: UUID) -> ProjectContext:
    row = conn.execute("select id, project_id, name from public.projects where id = %s", (project_id,)).fetchone()
    if row is None:
        raise ToolError("project_not_found", "Project not found")
    return ProjectContext(project_id=row["project_id"], project_uuid=row["id"], project_name=redact_text(row["name"]) or "")


# --- get_incident ------------------------------------------------------------


class GetIncidentInput(PageInput):
    incident_id: UUID
    include_events: bool = Field(default=False, description="Also return the incident's most recent events")
    limit: int = Field(default=10, ge=1, le=50, description="Events per page when include_events is true")


class IncidentDetails(IncidentSummary):
    fingerprint: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime
    summary: Dict[str, Any] = {}


class GetIncidentOutput(TelemetryResult):
    incident: IncidentDetails
    project_context: ProjectContext
    repository_context: RepositoryContext
    events: Optional[List[EventSummary]] = None
    events_pagination: Optional[Pagination] = None


def get_incident(ctx: ToolContext, inp: GetIncidentInput) -> GetIncidentOutput:
    row = _load_incident(ctx.conn, inp.project_id, inp.incident_id)
    incident = IncidentDetails(
        **incident_summary(row).model_dump(), fingerprint=row["fingerprint"],
        acknowledged_at=row["acknowledged_at"], resolved_at=row["resolved_at"],
        created_at=row["created_at"], updated_at=row["updated_at"], summary=redact_json(row["incident_data"] or {}),
    )
    events = pagination = None
    if inp.include_events:
        rows = ctx.conn.execute(
            f"""select {EVENT_COLUMNS} from public.telemetry_events e
                where e.project_id = %s and e.incident_id = %s
                order by e.event_timestamp desc, e.id desc limit %s offset %s""",
            (inp.project_id, inp.incident_id, inp.limit + 1, inp.offset),
        ).fetchall()
        page, pagination = paginate(rows, inp.limit, inp.offset)
        events = [event_summary(r) for r in page]
    return GetIncidentOutput(
        incident=incident, project_context=_project_context(ctx.conn, inp.project_id),
        repository_context=repository_context(ctx.conn, inp.project_id, row["service_id"]),
        events=events, events_pagination=pagination,
        **result_fields(pagination, ["event_count is the number of events recorded against this incident."]),
    )


# --- get_related_events ------------------------------------------------------


class RelatedEventsInput(TimeRangeInput):
    incident_id: UUID
    scope: Literal["incident", "fingerprint"] = Field(
        default="incident",
        description="'incident': events grouped into this incident. 'fingerprint': events of every incident "
        "in this project that shares its fingerprint (recurrences).",
    )
    min_severity: Optional[Severity] = None


class RelatedEventsOutput(TelemetryResult):
    incident: IncidentSummary
    scope: str
    related_incident_ids: List[UUID]
    events: List[EventSummary]
    pagination: Pagination


def _time_and_severity_filters(inp: Any, where: List[str], params: List[Any]) -> None:
    if inp.since:
        where.append("e.event_timestamp >= %s")
        params.append(inp.since)
    if inp.until:
        where.append("e.event_timestamp <= %s")
        params.append(inp.until)
    min_severity = getattr(inp, "min_severity", None)
    if min_severity:
        where.append("e.severity = any(%s)")
        params.append(list(SEVERITIES[SEVERITIES.index(min_severity):]))


def get_related_events(ctx: ToolContext, inp: RelatedEventsInput) -> RelatedEventsOutput:
    row = _load_incident(ctx.conn, inp.project_id, inp.incident_id, _FULL_INCIDENT_COLUMNS)
    notes: List[str] = []
    if inp.scope == "fingerprint" and row["fingerprint"]:
        ids = [r["id"] for r in ctx.conn.execute(
            "select id from public.incidents where project_id = %s and fingerprint = %s order by created_at, id limit 200",
            (inp.project_id, row["fingerprint"]),
        ).fetchall()]
        notes.append("Fingerprint scope covers incidents sharing the same project, service, environment, "
                     "error type, message, source file and function.")
    else:
        ids = [inp.incident_id]
        if inp.scope == "fingerprint":
            notes.append("This incident has no fingerprint; only its own events are returned.")
    where = ["e.project_id = %s", "e.incident_id = any(%s)"]
    params: List[Any] = [inp.project_id, ids]
    _time_and_severity_filters(inp, where, params)
    rows = ctx.conn.execute(
        f"select {EVENT_COLUMNS} from public.telemetry_events e where {' and '.join(where)} "
        "order by e.event_timestamp desc, e.id desc limit %s offset %s",
        (*params, inp.limit + 1, inp.offset),
    ).fetchall()
    page, pagination = paginate(rows, inp.limit, inp.offset)
    return RelatedEventsOutput(
        incident=incident_summary(row), scope=inp.scope, related_incident_ids=ids,
        events=[event_summary(r) for r in page], pagination=pagination, **result_fields(pagination, notes),
    )


# --- get_incident_timeline ---------------------------------------------------


class TimelineInput(PageInput):
    incident_id: UUID
    limit: int = Field(default=100, ge=1, le=200, description="Recorded events per page")


class TimelineEntry(BaseModel):
    timestamp: datetime
    # event_time: when the SDK says the event happened. recorded_time: when J.A.R.V.I.S. stored the
    # lifecycle transition. The two clocks can differ (SDKs buffer and retry).
    timestamp_basis: Literal["event_time", "recorded_time"]
    entry_type: Literal["event", "lifecycle"]
    source: Literal["recorded_event", "recorded_lifecycle_transition"]
    summary: str
    event: Optional[EventSummary] = None
    change_type: Optional[str] = None
    from_status: Optional[str] = None
    to_status: Optional[str] = None
    from_severity: Optional[str] = None
    to_severity: Optional[str] = None
    note: Optional[str] = None


class DerivedSummary(BaseModel):
    """Computed from the recorded rows; NOT itself a recorded event."""

    derived: bool = True
    first_event_at: Optional[datetime] = None
    last_event_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    recorded_event_count: int
    peak_severity: Optional[str] = None
    current_status: str


class TimelineOutput(TelemetryResult):
    incident: IncidentSummary
    timeline: List[TimelineEntry]
    derived_summary: DerivedSummary
    pagination: Pagination


_MAX_HISTORY_ROWS = 500


def get_incident_timeline(ctx: ToolContext, inp: TimelineInput) -> TimelineOutput:
    row = _load_incident(ctx.conn, inp.project_id, inp.incident_id, INCIDENT_SUMMARY_COLUMNS)
    rows = ctx.conn.execute(
        f"""select {EVENT_COLUMNS} from public.telemetry_events e
            where e.project_id = %s and e.incident_id = %s
            order by e.event_timestamp, e.id limit %s offset %s""",
        (inp.project_id, inp.incident_id, inp.limit + 1, inp.offset),
    ).fetchall()
    page, pagination = paginate(rows, inp.limit, inp.offset)
    history = ctx.conn.execute(
        """select seq, change_type, from_status, to_status, from_severity, to_severity, note, created_at
           from public.incident_history where incident_id = %s and project_id = %s
           order by created_at, seq limit %s""",
        (inp.incident_id, inp.project_id, _MAX_HISTORY_ROWS),
    ).fetchall()

    # Lifecycle rows are placed against the page of events being shown: everything before the
    # first event on the first page, everything after the last event on the last page.
    first_ts = page[0]["event_timestamp"] if page else None
    last_ts = page[-1]["event_timestamp"] if page else None
    entries: List[tuple] = []
    for r in page:
        ev = event_summary(r)
        label = f"{ev.severity or ev.event_type}: {ev.message or ev.error_type or ''}".strip()[:200]
        entries.append((r["event_timestamp"], 1, str(r["id"]), TimelineEntry(
            timestamp=r["event_timestamp"], timestamp_basis="event_time", entry_type="event",
            source="recorded_event", summary=label, event=ev)))
    for h in history:
        ts = h["created_at"]
        if page and ((inp.offset > 0 and ts < first_ts) or (pagination.has_more and ts > last_ts)):
            continue
        if h["change_type"] == "created":
            text = "Incident created"
        elif h["change_type"] == "severity_escalated":
            text = f"Severity escalated {h['from_severity']} -> {h['to_severity']}"
        else:
            text = f"Status changed {h['from_status']} -> {h['to_status']}"
        entries.append((ts, 0, h["seq"], TimelineEntry(
            timestamp=ts, timestamp_basis="recorded_time", entry_type="lifecycle",
            source="recorded_lifecycle_transition", summary=text,
            change_type=h["change_type"], from_status=h["from_status"], to_status=h["to_status"],
            from_severity=h["from_severity"], to_severity=h["to_severity"], note=redact_text(h["note"]))))
    entries.sort(key=lambda t: (t[0], t[1], t[2]))

    agg = ctx.conn.execute(
        "select min(event_timestamp) as first_at, max(event_timestamp) as last_at, count(*) as n "
        "from public.telemetry_events where project_id = %s and incident_id = %s",
        (inp.project_id, inp.incident_id),
    ).fetchone()
    duration = (agg["last_at"] - agg["first_at"]).total_seconds() if agg["n"] else None
    summary = DerivedSummary(
        first_event_at=agg["first_at"], last_event_at=agg["last_at"], duration_seconds=duration,
        recorded_event_count=int(agg["n"]), peak_severity=row["severity"], current_status=row["status"],
    )
    notes = [
        "Entries marked recorded_* are stored rows. derived_summary is computed from them and is not a recorded event.",
        "Event entries use event_time; lifecycle entries use recorded_time (when the transition was stored).",
    ]
    if len(history) >= _MAX_HISTORY_ROWS:
        notes.append(f"Only the first {_MAX_HISTORY_ROWS} lifecycle records are shown.")
    return TimelineOutput(
        incident=incident_summary(row), timeline=[e[3] for e in entries], derived_summary=summary,
        pagination=pagination, **result_fields(pagination, notes),
    )


TOOLS = [
    ToolDefinition(
        name="get_incident",
        description=(
            "Purpose: Fetch full details for one incident (title, status, severity, fingerprint, service_id, "
            "environment, timestamps, event_count), its verified project/service GitHub repositories "
            "(with installation_id, owner, repo), and optionally its most recent telemetry events. "
            "When to use: Primary starting point whenever an incident_id (UUID) is known. "
            "When not to use: Do not use for broad event filtering without an incident_id (use search_events). "
            "Scope & limitations: Automatically scoped to the active project context; read-only."
        ),
        input_model=GetIncidentInput, output_model=GetIncidentOutput, handler=get_incident,
        source="database", constraints=TELEMETRY_CONSTRAINTS, context_inputs=("project_id",),
    ),
    ToolDefinition(
        name="get_related_events",
        description=(
            "Purpose: List persisted telemetry events belonging to an incident (scope='incident') or all "
            "incidents sharing its fingerprint (scope='fingerprint'), filtered by time range and min_severity. "
            "When to use: After get_incident when you need more event occurrences, stack locations, or cross-incident pattern history. "
            "When not to use: Do not use for full stack trace/payload of one event (use get_event_details) or arbitrary service searches (use search_events). "
            "Scope & limitations: Newest first, paginated; automatically scoped to the active project."
        ),
        input_model=RelatedEventsInput, output_model=RelatedEventsOutput, handler=get_related_events,
        source="database", constraints=TELEMETRY_CONSTRAINTS, context_inputs=("project_id",),
    ),
    ToolDefinition(
        name="get_incident_timeline",
        description=(
            "Purpose: Build a chronological timeline of an incident interleaving recorded telemetry events "
            "(event_time) with recorded lifecycle transitions (created, severity_escalated, status changes at recorded_time) and a derived summary. "
            "When to use: When analyzing how an incident evolved, escalated, or recurred over time. "
            "When not to use: Do not use just to fetch incident metadata or repository links (use get_incident). "
            "Scope & limitations: Automatically scoped to the active project; derived_summary is computed, not a raw event."
        ),
        input_model=TimelineInput, output_model=TimelineOutput, handler=get_incident_timeline,
        source="database", constraints=TELEMETRY_CONSTRAINTS, context_inputs=("project_id",),
    ),
]
_ = TelemetryInput
