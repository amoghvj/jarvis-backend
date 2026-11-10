"""Shared inputs, outputs, redaction and repository context for the telemetry/incident tools.

Every query in this package is scoped by `project_id`, the internal project UUID that the shared
executor injects from trusted context after verifying ownership. `project_id` is part of each
tool's INTERNAL input schema only; it is stripped from the LLM-facing schema.
"""

import re
from datetime import datetime
from typing import Annotated, Any, Dict, List, Literal, Optional, Tuple
from uuid import UUID

from pydantic import AfterValidator, AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from app import repositories_telemetry as repo
from app.tools.code.redaction import REDACTED, is_sensitive_key, scrub_text
from app.tools.schemas import ToolConstraints

TELEMETRY_CONSTRAINTS = ToolConstraints(timeout_seconds=20.0)
MESSAGE_PREVIEW_CHARS = 500
SEVERITIES = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
Severity = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
_TRACE_RE = re.compile(r"[A-Za-z0-9._:\-]{1,128}")
_MAX_JSON_DEPTH = 8


def _printable(value: str) -> str:
    if any(ord(c) < 32 or ord(c) == 127 for c in value):
        raise ValueError("must not contain control characters")
    return value


def _identifier(value: str) -> str:
    if not _TRACE_RE.fullmatch(value):
        raise ValueError("must be 1-128 characters: letters, digits and . _ : -")
    return value


FilterText = Annotated[str, Field(min_length=1, max_length=200), AfterValidator(_printable)]
Identifier = Annotated[str, AfterValidator(_identifier)]


class TelemetryInput(BaseModel):
    """Base for every telemetry tool input."""

    # Unknown fields are rejected, so identity or tenant fields can never be smuggled in.
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    project_id: UUID = Field(description="Internal project UUID; injected by the executor, never supplied")


class PageInput(TelemetryInput):
    limit: int = Field(default=25, ge=1, le=100)
    offset: int = Field(default=0, ge=0, le=10_000)


class TimeRangeInput(PageInput):
    since: Optional[AwareDatetime] = Field(default=None, description="Only events at or after this time")
    until: Optional[AwareDatetime] = Field(default=None, description="Only events at or before this time")

    @model_validator(mode="after")
    def _ordered(self) -> "TimeRangeInput":
        if self.since and self.until and self.since > self.until:
            raise ValueError("since must not be after until")
        return self


class Pagination(BaseModel):
    limit: int
    offset: int
    returned: int
    has_more: bool
    next_offset: Optional[int] = None


def paginate(rows: List[Any], limit: int, offset: int) -> Tuple[List[Any], Pagination]:
    """`rows` was fetched with limit + 1; the extra row only signals that more exist."""
    has_more = len(rows) > limit
    page = rows[:limit]
    return page, Pagination(
        limit=limit, offset=offset, returned=len(page), has_more=has_more,
        next_offset=offset + limit if has_more else None,
    )


class TelemetryResult(BaseModel):
    """Every telemetry result says whether it is complete."""

    complete: bool
    limits_hit: List[str]
    notes: List[str]


def result_fields(pagination: Optional[Pagination], notes: Optional[List[str]] = None) -> Dict[str, Any]:
    truncated = bool(pagination and pagination.has_more)
    return {
        "complete": not truncated,
        "limits_hit": ["page_limit"] if truncated else [],
        "notes": list(notes or []) + (
            ["More results exist; request the next page using next_offset."] if truncated else []
        ),
    }


# --- redaction ---------------------------------------------------------------


def redact_text(text: Optional[str]) -> Optional[str]:
    return scrub_text(text)[0] if text else text


def redact_json(value: Any, depth: int = 0) -> Any:
    """Redact secrets in arbitrary JSON: sensitive-looking keys lose their value, strings are scrubbed."""
    if depth > _MAX_JSON_DEPTH:
        return "[TRUNCATED]"
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for key, item in value.items():
            if is_sensitive_key(str(key)) and item not in (None, "") and not isinstance(item, (dict, list)):
                out[str(key)] = REDACTED
            else:
                out[str(key)] = redact_json(item, depth + 1)
        return out
    if isinstance(value, list):
        return [redact_json(v, depth + 1) for v in value]
    if isinstance(value, str):
        return scrub_text(value)[0]
    return value


def like_pattern(text: str) -> str:
    """Escape LIKE wildcards so a user's text is always matched literally."""
    return "%" + text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"


# --- event summaries ---------------------------------------------------------

EVENT_COLUMNS = (
    "e.id, e.event_id, e.event_type, e.classification, e.event_timestamp, e.received_at, e.service_id, "
    "e.environment, e.instance_id, e.severity, e.message, e.logger_name, e.error_type, e.source_file, "
    "e.source_function, e.source_line, e.trace_id, e.span_id, e.request_id, e.incident_id"
)


class SourceLocation(BaseModel):
    file: Optional[str] = None
    function: Optional[str] = None
    line: Optional[int] = None


class EventSummary(BaseModel):
    id: UUID
    event_id: str
    event_type: str
    classification: str
    timestamp: datetime
    received_at: datetime
    service_id: Optional[str] = None
    environment: Optional[str] = None
    instance_id: Optional[str] = None
    severity: Optional[str] = None
    message: Optional[str] = None
    message_truncated: bool = False
    logger_name: Optional[str] = None
    error_type: Optional[str] = None
    source: SourceLocation
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    request_id: Optional[str] = None
    incident_id: Optional[UUID] = None


def event_summary(row: Dict[str, Any]) -> EventSummary:
    message = redact_text(row["message"])
    truncated = bool(message and len(message) > MESSAGE_PREVIEW_CHARS)
    return EventSummary(
        id=row["id"], event_id=row["event_id"], event_type=row["event_type"],
        classification=row["classification"], timestamp=row["event_timestamp"], received_at=row["received_at"],
        service_id=row["service_id"], environment=row["environment"], instance_id=row["instance_id"],
        severity=row["severity"], message=message[:MESSAGE_PREVIEW_CHARS] if message else message,
        message_truncated=truncated, logger_name=row["logger_name"], error_type=row["error_type"],
        source=SourceLocation(file=row["source_file"], function=row["source_function"], line=row["source_line"]),
        trace_id=row["trace_id"], span_id=row["span_id"], request_id=row["request_id"],
        incident_id=row["incident_id"],
    )


# --- project, incident and repository context --------------------------------


class ProjectContext(BaseModel):
    project_id: str  # the external ID the SDK uses
    project_uuid: UUID
    project_name: str


class IncidentSummary(BaseModel):
    id: UUID
    title: str
    status: str
    severity: Optional[str] = None
    service_id: Optional[str] = None
    environment: Optional[str] = None
    event_count: int
    first_seen_at: Optional[datetime] = None
    last_seen_at: Optional[datetime] = None


INCIDENT_SUMMARY_COLUMNS = (
    "id, title, status, severity, service_id, environment, event_count, first_seen_at, last_seen_at"
)


def incident_summary(row: Dict[str, Any]) -> IncidentSummary:
    return IncidentSummary(
        id=row["id"], title=redact_text(row["title"]) or "", status=row["status"], severity=row["severity"],
        service_id=row["service_id"], environment=row["environment"], event_count=row["event_count"],
        first_seen_at=row["first_seen_at"], last_seen_at=row["last_seen_at"],
    )


class RepositoryEntry(BaseModel):
    installation_id: int
    owner: str
    repo: str
    repository_url: str
    default_branch: Optional[str] = None
    service_id: Optional[str] = None
    verification_status: str
    matches_service: Optional[bool] = None  # None when no service was in question
    last_seen_at: Optional[datetime] = None


class RepositoryContext(BaseModel):
    """Verified repositories only. Unverified mappings are never listed as tool targets."""

    repositories: List[RepositoryEntry]
    unverified_repository_count: int
    note: str


def repository_context(conn: Any, project_uuid: UUID, service_id: Optional[str]) -> RepositoryContext:
    rows = repo.verified_repositories(conn, project_uuid)
    entries = [
        RepositoryEntry(
            installation_id=r["installation_id"], owner=r["owner"], repo=r["repo"],
            repository_url=r["repository_url"], default_branch=r["default_branch"],
            service_id=r["service_id"] or None, verification_status=r["verification_status"],
            matches_service=(None if service_id is None else (r["service_id"] == service_id)),
            last_seen_at=r["last_seen_at"],
        )
        for r in rows
    ]
    entries.sort(key=lambda e: (e.matches_service is not True, e.service_id or "", e.owner, e.repo))
    unverified = repo.count_unverified_repositories(conn, project_uuid)
    note = (
        "Only repositories verified against your GitHub App installations are listed, and they may be "
        "used as targets for the repository tools. Repositories are never inferred from error messages "
        "or stack traces."
    )
    if unverified:
        note += f" {unverified} discovered repository mapping(s) are unverified and are not usable."
    return RepositoryContext(repositories=entries, unverified_repository_count=unverified, note=note)

