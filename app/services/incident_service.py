"""Deterministic incident grouping, severity escalation and lifecycle.

Grouping key (the fingerprint) is built from stable attributes only: project, service,
environment, error type, lightly normalized message, source file and source function. It never
includes timestamps, request/trace/span IDs, instance IDs or line numbers.

Concurrency: every grouping decision runs inside the caller's transaction after taking a
transaction-scoped advisory lock keyed by (project, fingerprint). Two concurrent batches carrying
the same error therefore serialize and end up in ONE incident, with no lost counter updates.
"""

import hashlib
import re
from typing import Any, Dict, Optional
from uuid import UUID

import psycopg

from app import repositories_telemetry as repo

DEFAULT_CORRELATION_WINDOW_SECONDS = 15 * 60
# The canonical event classifications (the SDK contract since jarvis-observe 0.1.1).
NORMAL_BEHAVIOUR = "normal_behaviour"
ABNORMAL_BEHAVIOUR = "abnormal_behaviour"
CLASSIFICATIONS = (NORMAL_BEHAVIOUR, ABNORMAL_BEHAVIOUR)
SEVERITY_ORDER = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3, "CRITICAL": 4}
INCIDENT_SEVERITIES = ("ERROR", "CRITICAL")
STATUSES = ("open", "acknowledged", "resolved")
# from-status -> statuses it may move to. `resolved -> open` is an explicit reopen by a user.
TRANSITIONS = {
    "open": ("acknowledged", "resolved"),
    "acknowledged": ("resolved", "open"),
    "resolved": ("open",),
}
_ADDRESS = re.compile(r"0x[0-9a-fA-F]{6,}")
_SPACE = re.compile(r"\s+")
MAX_NOTE_LENGTH = 500


class IncidentNotFound(Exception):
    pass


class InvalidTransition(Exception):
    """Message is safe to return to the caller."""


def normalize_severity(value: Optional[str]) -> Optional[str]:
    if not isinstance(value, str) or not value.strip():
        return None
    upper = value.strip().upper()
    return {"WARN": "WARNING", "FATAL": "CRITICAL", "SEVERE": "CRITICAL"}.get(upper, upper)[:32]


def severity_rank(value: Optional[str]) -> int:
    return SEVERITY_ORDER.get(value or "", -1)


def normalize_message(message: Optional[str]) -> str:
    """Light normalization only: memory addresses and whitespace. Numbers/IDs are kept."""
    text = _ADDRESS.sub("0x?", message or "")
    return _SPACE.sub(" ", text).strip()[:500]


def compute_fingerprint(
    project_uuid: UUID, service_id: Optional[str], environment: Optional[str],
    error_type: Optional[str], message: Optional[str], source_file: Optional[str],
    source_function: Optional[str],
) -> str:
    parts = [
        "v1", str(project_uuid), service_id or "", environment or "", error_type or "",
        normalize_message(message), source_file or "", source_function or "",
    ]
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def is_incident_eligible(event_type: str, classification: str, severity: Optional[str]) -> bool:
    """Only ERROR/CRITICAL events classified `abnormal_behaviour` can create or join an incident.

    * `startup` events never do, whatever their severity or classification.
    * Anything not `abnormal_behaviour` (that is, `normal_behaviour`) never does.
    * INFO, DEBUG and WARNING never do, even when classified `abnormal_behaviour`.
    * An `error`-type event that carries no severity at all is treated as ERROR (unchanged behaviour).
    """
    if event_type == "startup" or classification != ABNORMAL_BEHAVIOUR:
        return False
    return severity in INCIDENT_SEVERITIES or (severity is None and event_type == "error")


def _title(error_type: Optional[str], message: Optional[str]) -> str:
    first_line = (message or "").strip().splitlines()[0] if (message or "").strip() else ""
    title = f"{error_type}: {first_line}" if error_type and first_line else (error_type or first_line)
    return (title or "Unclassified error")[:200]


def attach_event(
    conn: psycopg.Connection, project_uuid: UUID, event: Dict[str, Any], event_uuid: UUID,
    window_seconds: int,
) -> UUID:
    """Associate an already-inserted eligible event with an incident (existing or new).

    Must run inside a transaction. Returns the incident ID.
    """
    fingerprint = event["fingerprint"]
    ts = event["event_timestamp"]
    severity = event["severity"] or "ERROR"
    repo.lock_incident_group(conn, f"{project_uuid}:{fingerprint}")
    incident = repo.find_open_incident(conn, project_uuid, fingerprint, ts, window_seconds)
    if incident is None:
        incident_id = repo.insert_incident(
            conn, project_uuid, title=_title(event["error_type"], event["message"]), severity=severity,
            fingerprint=fingerprint, service_id=event["service_id"], environment=event["environment"], ts=ts,
            data={
                "error_type": event["error_type"],
                "message": (event["message"] or "")[:500],
                "source_file": event["source_file"],
                "source_function": event["source_function"],
                "first_event_id": event["event_id"],
            },
        )
        repo.insert_history(conn, project_uuid, incident_id, "created", to_status="open", to_severity=severity)
    else:
        incident_id = incident["id"]
        escalated = severity_rank(severity) > severity_rank(incident["severity"])
        repo.bump_incident(conn, incident_id, ts, severity if escalated else None)
        if escalated:
            repo.insert_history(
                conn, project_uuid, incident_id, "severity_escalated",
                from_severity=incident["severity"], to_severity=severity,
            )
    repo.set_event_incident(conn, event_uuid, incident_id)
    return incident_id


def change_status(
    conn: psycopg.Connection, project_uuid: UUID, incident_id: UUID, new_status: str,
    actor_user_id: UUID, note: Optional[str] = None,
) -> dict:
    """Move an incident through its lifecycle and record the transition. Never automatic."""
    if new_status not in STATUSES:
        raise InvalidTransition(f"status must be one of: {', '.join(STATUSES)}")
    note = (note or "").strip()[:MAX_NOTE_LENGTH] or None
    with conn.transaction():
        current = repo.get_incident_for_update(conn, project_uuid, incident_id)
        if current is None:
            raise IncidentNotFound()
        if new_status not in TRANSITIONS.get(current["status"], ()):
            raise InvalidTransition(f"cannot move an incident from '{current['status']}' to '{new_status}'")
        updated = repo.set_incident_status(conn, incident_id, new_status)
        repo.insert_history(
            conn, project_uuid, incident_id, "status_changed", from_status=current["status"],
            to_status=new_status, actor_user_id=actor_user_id, note=note,
        )
    return updated

