"""Helpers shared by the project / telemetry / incident tests."""

import itertools
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from app import repositories as db_repo
from app.services import project_service, telemetry_service
from app.tools.schemas import ToolContext
from tests.conftest import bearer, make_key, make_user

BASE_TIME = datetime(2026, 11, 9, 8, 30, 0, tzinfo=timezone.utc)
_counter = itertools.count(1)


def iso(offset_seconds: float = 0) -> str:
    return (BASE_TIME + timedelta(seconds=offset_seconds)).isoformat().replace("+00:00", "Z")


def make_event(project_id: str, **kw) -> Dict[str, Any]:
    """An SDK-shaped event. Override any field with keyword arguments."""
    event: Dict[str, Any] = {
        "schema_version": "1.0",
        "event_id": kw.get("event_id") or f"evt-{next(_counter)}-{uuid.uuid4().hex[:6]}",
        "timestamp": kw.get("ts", iso()),
        "type": kw.get("type", "error"),
        "application": {
            "project_id": project_id,
            "service_id": kw.get("service", "backend"),
            "repository": kw.get("repository"),
            "environment": kw.get("env", "production"),
            "instance_id": kw.get("instance", "host-1"),
            "runtime": {"language": "python", "version": "3.12.2"},
        },
        "details": {
            "severity": kw.get("severity", "ERROR"),
            "message": kw.get("message", "connection refused"),
            "logger_name": kw.get("logger", "app.db"),
            "error_type": kw.get("error_type", "ConnectionError"),
            "stack_trace": kw.get("stack", "Traceback (most recent call last):\n  File \"app/db.py\", line 10\nConnectionError"),
            "source": {
                "file_path": kw.get("file", "app/db.py"),
                "function_name": kw.get("func", "connect"),
                "line_number": kw.get("line", 10),
            },
        },
        "execution": {
            "trace_id": kw.get("trace"), "span_id": kw.get("span"),
            "request_id": kw.get("request"), "thread_name": kw.get("thread", "MainThread"),
        },
        "attributes": kw.get("attributes", {"mechanism": "logging"}),
    }
    if "classification" in kw:
        event["classification"] = kw["classification"]
    return event


def startup_event(project_id: str, **kw) -> Dict[str, Any]:
    """The startup event the SDKs will emit once per init()."""
    kw.setdefault("severity", "INFO")
    kw.setdefault("message", "J.A.R.V.I.S. observability initialised")
    kw.setdefault("error_type", None)
    kw.setdefault("stack", None)
    kw.setdefault("classification", "normal_behaviour")
    return make_event(project_id, type="startup", **kw)


class Account:
    """A user with an API key, one project, and helpers to ingest events directly."""

    def __init__(self, db, project_external: Optional[str] = None, username: str = "someone"):
        self.db = db
        self.user = make_user(db, username=username)
        self.key = make_key(db, self.user["id"])
        self.external = project_external or f"proj-{uuid.uuid4().hex[:8]}"
        row = project_service.create_project(db, self.user["id"], self.external, "Project " + self.external)
        self.project_uuid = row["id"]

    def event(self, **kw) -> Dict[str, Any]:
        return make_event(self.external, **kw)

    def ingest(self, events: List[Dict[str, Any]], window: int = 900):
        valid, rejected = telemetry_service.parse_events(events)
        assert not rejected, rejected
        return telemetry_service.ingest_events(self.db, valid, {self.external: self.project_uuid}, window)

    def post(self, client, events: List[Dict[str, Any]]):
        return client.post("/v1/events", json={"events": events}, headers=bearer(self.key))

    def ctx(self, project_uuid=None) -> ToolContext:
        return ToolContext(user_id=self.user["id"], conn=self.db, project_id=project_uuid or self.project_uuid)

    def incidents(self) -> List[dict]:
        return self.db.execute(
            "select * from public.incidents where project_id = %s order by first_seen_at, id", (self.project_uuid,)
        ).fetchall()

    def events(self) -> List[dict]:
        return self.db.execute(
            "select * from public.telemetry_events where project_id = %s order by event_timestamp, id",
            (self.project_uuid,),
        ).fetchall()

    def history(self, incident_id) -> List[dict]:
        return self.db.execute(
            "select * from public.incident_history where incident_id = %s order by created_at, seq", (incident_id,)
        ).fetchall()


def link_installation(db, user_id, installation_id: Optional[int] = None, suspended: bool = False) -> int:
    installation_id = installation_id or secrets.randbelow(10**9) + 6_000_000_000
    db_repo.upsert_github_app_installation(
        db, user_id=user_id, installation_id=installation_id, github_account_id=4242,
        github_account_login="acme", github_account_type="Organization", repository_selection="selected",
        permissions={"contents": "read", "metadata": "read"},
        suspended_at=datetime.now(timezone.utc) if suspended else None,
    )
    return installation_id


def add_verified_repository(db, account: Account, installation_id: int, owner: str, repo: str,
                            service_id: str = "backend", branch: str = "main") -> None:
    db.execute(
        """insert into public.project_repositories
           (project_id, service_id, repository_url, owner, repo, installation_id, default_branch,
            verification_status, verified_at)
           values (%s, %s, %s, %s, %s, %s, %s, 'verified', now())""",
        (account.project_uuid, service_id, f"https://github.com/{owner}/{repo}", owner, repo, installation_id, branch),
    )
