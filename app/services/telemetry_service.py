"""Telemetry ingestion: validation, normalization and idempotent, transactional persistence."""

import json
import logging
from dataclasses import dataclass, field
from datetime import timezone
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

import psycopg
from pydantic import ValidationError

from app import repositories_telemetry as repo
from app.models import Event
from app.services import incident_service
from app.tools.errors import redact

logger = logging.getLogger("jarvis.ingest")

MAX_BATCH_EVENTS = 100
MAX_PAYLOAD_BYTES = 1_000_000
MAX_EVENT_BYTES = 100_000
MAX_EVENT_ID_LENGTH = 200
MAX_PROJECT_ID_LENGTH = 128


@dataclass
class EventResult:
    index: int
    event_id: Optional[str]
    status: str  # accepted | duplicate | rejected
    error: Optional[str] = None
    incident_id: Optional[UUID] = None


@dataclass
class Ingested:
    results: List[EventResult] = field(default_factory=list)
    new_startup_events: List[Dict[str, Any]] = field(default_factory=list)

    def count(self, status: str) -> int:
        return sum(1 for r in self.results if r.status == status)


class InvalidClassification(ValueError):
    """The event's `classification` is present but not a supported value. Message is safe to return."""


def resolve_classification(payload: dict, event_type: str) -> str:
    """The canonical classification (`normal_behaviour` / `abnormal_behaviour`) for an event.

    * Field absent (SDK 0.1.0 and earlier): `startup` events are `normal_behaviour` (they are lifecycle
      events by definition); every other event falls back to `abnormal_behaviour`.
    * Field present: it must be a string equal to one of the two values after trimming whitespace and
      lower-casing, which is the only normalization applied. Anything else (unknown string, empty
      string, null, number, boolean, list, object) is rejected; nothing is truncated or replaced.
    * A valid but contradictory combination is stored exactly as sent and is never reinterpreted
      (for example `startup` + `abnormal_behaviour`, or `ERROR` + `normal_behaviour`). Incident
      eligibility is decided separately by `incident_service.is_incident_eligible`.
    """
    if "classification" not in payload:
        return incident_service.NORMAL_BEHAVIOUR if event_type == "startup" else incident_service.ABNORMAL_BEHAVIOUR
    raw = payload["classification"]
    value = raw.strip().lower() if isinstance(raw, str) else None
    if value not in incident_service.CLASSIFICATIONS:
        raise InvalidClassification(
            "classification must be 'normal_behaviour' or 'abnormal_behaviour' when provided"
        )
    return value


def _summarize(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors()[:3]:
        loc = ".".join(str(p) for p in err.get("loc", ())) or "event"
        parts.append(f"{loc}: {err.get('msg', 'invalid')}")  # never echoes the offending value
    return redact("; ".join(parts))


def parse_events(raw_events: List[Any]) -> Tuple[List[Tuple[int, Event, dict]], List[EventResult]]:
    """Validate each event independently so one bad event never discards the rest."""
    valid: List[Tuple[int, Event, dict]] = []
    rejected: List[EventResult] = []
    for index, raw in enumerate(raw_events):
        raw_id = raw.get("event_id") if isinstance(raw, dict) and isinstance(raw.get("event_id"), str) else None
        if not isinstance(raw, dict):
            rejected.append(EventResult(index, None, "rejected", "event must be a JSON object"))
            continue
        try:
            event = Event.model_validate(raw)
        except ValidationError as exc:
            rejected.append(EventResult(index, raw_id, "rejected", _summarize(exc)))
            continue
        problem = None
        if not event.event_id.strip() or len(event.event_id) > MAX_EVENT_ID_LENGTH:
            problem = f"event_id must be 1-{MAX_EVENT_ID_LENGTH} characters"
        elif not event.application.project_id.strip() or len(event.application.project_id) > MAX_PROJECT_ID_LENGTH:
            problem = f"application.project_id must be 1-{MAX_PROJECT_ID_LENGTH} characters"
        elif not event.type.strip() or len(event.type) > 64:
            problem = "type must be 1-64 characters"
        elif len(json.dumps(raw, default=str)) > MAX_EVENT_BYTES:
            problem = f"event exceeds the {MAX_EVENT_BYTES} byte limit"
        if problem is None:
            try:
                resolve_classification(raw, event.type.strip().lower())
            except InvalidClassification as exc:
                problem = str(exc)
        if problem:
            rejected.append(EventResult(index, raw_id, "rejected", problem))
            continue
        valid.append((index, event, raw))
    return valid, rejected


def _clean(value: Any) -> Any:
    """PostgreSQL text/jsonb cannot store NUL characters; replace them rather than fail the batch."""
    if isinstance(value, str):
        return value.replace("\x00", "\ufffd")
    if isinstance(value, dict):
        return {_clean(k): _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    return value


def _text(value: Any, limit: Optional[int] = None) -> Optional[str]:
    if value is None:
        return None
    text = str(value)
    return text[:limit] if limit else text


def normalize_event(event: Event, payload: dict, project_uuid: UUID) -> Dict[str, Any]:
    app = event.application
    details = event.details
    source = details.source
    execution = event.execution
    ts = event.timestamp if event.timestamp.tzinfo else event.timestamp.replace(tzinfo=timezone.utc)
    ts = ts.astimezone(timezone.utc)
    severity = incident_service.normalize_severity(details.severity)
    event_type = event.type.strip().lower()
    classification = resolve_classification(payload, event_type)  # raises InvalidClassification
    runtime = app.runtime.model_dump(mode="json") if app.runtime else {}
    normalized = {
        "event_id": event.event_id,
        "schema_version": _text(event.schema_version, 32),
        "event_type": event_type,
        "classification": classification,
        "event_timestamp": ts,
        "application": app.model_dump(mode="json"),
        "service_id": _text(app.service_id, 200),
        "environment": _text(app.environment, 100),
        "instance_id": _text(app.instance_id, 200),
        "runtime": runtime,
        "severity": severity,
        "message": _text(details.message),
        "logger_name": _text(details.logger_name, 300),
        "error_type": _text(details.error_type, 300),
        "stack_trace": _text(details.stack_trace),
        "source_file": _text(source.file_path if source else None, 500),
        "source_function": _text(source.function_name if source else None, 300),
        "source_line": source.line_number if source and isinstance(source.line_number, int) else None,
        "trace_id": _text(execution.trace_id if execution else None, 200),
        "span_id": _text(execution.span_id if execution else None, 200),
        "request_id": _text(execution.request_id if execution else None, 200),
        "thread_name": _text(execution.thread_name if execution else None, 200),
        "attributes": event.attributes or {},
        "payload": json.loads(json.dumps(payload, default=str)),
        "repository": _text(app.repository, 1000),
    }
    normalized = _clean(normalized)
    normalized["eligible"] = incident_service.is_incident_eligible(event_type, classification, severity)
    if normalized["eligible"]:
        normalized["fingerprint"] = incident_service.compute_fingerprint(
            project_uuid, normalized["service_id"], normalized["environment"], normalized["error_type"],
            normalized["message"], normalized["source_file"], normalized["source_function"],
        )
    return normalized


def ingest_events(
    conn: psycopg.Connection,
    valid: List[Tuple[int, Event, dict]],
    project_map: Dict[str, UUID],
    window_seconds: int = incident_service.DEFAULT_CORRELATION_WINDOW_SECONDS,
) -> Ingested:
    """Persist events and group them into incidents in ONE transaction.

    Duplicates (same project + event_id) are detected by the unique constraint and are skipped
    entirely: no second row, no incident count change, no repeated side effects. A failure
    anywhere rolls the whole batch back so the SDK can safely retry it.
    """
    out = Ingested()
    inserted: List[Tuple[UUID, UUID, Dict[str, Any], EventResult]] = []
    with conn.transaction():
        for index, event, payload in valid:
            project_uuid = project_map[event.application.project_id]
            normalized = normalize_event(event, payload, project_uuid)
            event_uuid = repo.insert_event(conn, project_uuid, normalized)
            result = EventResult(index, event.event_id, "accepted" if event_uuid else "duplicate")
            out.results.append(result)
            if event_uuid is None:
                continue
            inserted.append((project_uuid, event_uuid, normalized, result))
            if normalized["event_type"] == "startup":
                out.new_startup_events.append({**normalized, "project_uuid": project_uuid})
        # Lock/grouping order is deterministic (by project, fingerprint) so concurrent batches
        # that share fingerprints can never deadlock.
        eligible = sorted(
            (item for item in inserted if item[2]["eligible"]), key=lambda item: (str(item[0]), item[2]["fingerprint"])
        )
        for project_uuid, event_uuid, normalized, result in eligible:
            result.incident_id = incident_service.attach_event(conn, project_uuid, normalized, event_uuid, window_seconds)
    out.results.sort(key=lambda r: r.index)
    return out
