"""Repository discovery from telemetry, and verification against the user's GitHub App installations.

A repository URL arriving in a startup event is UNTRUSTED metadata. It is stored as an
`unverified` mapping and is never used to mint tokens or to grant access. It becomes `verified`
only when the existing installation-authorization path (`access.open_installation` followed by
`access.find_repository`) finds that exact repository inside an installation that belongs to the
project's owner. `installation_id` is only ever set from that check.
"""

import logging
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit
from uuid import UUID

import psycopg

from app import repositories as db_repo
from app import repositories_telemetry as repo
from app.tools.code.redaction import scrub_text
from app.tools.errors import ToolError, redact
from app.tools.github import access
from app.tools.github.common import _owner as _validate_owner
from app.tools.github.common import _repo as _validate_repo
from app.tools.schemas import ToolContext

logger = logging.getLogger("jarvis.repositories")

MAX_URL_LENGTH = 300
MAX_INSTALLATIONS_CHECKED = 10
VERIFY_RETRY_AFTER = timedelta(minutes=5)
VERIFY_DEADLINE_SECONDS = 20.0
_GITHUB_HOSTS = ("github.com", "www.github.com")
_SCP = re.compile(r"git@github\.com:([^/\s]+)/([^/\s]+?)(?:\.git)?/?")
_USERINFO = re.compile(r"(?<=//)[^/@\s]*@")


def parse_repository_url(raw: Any) -> Optional[Tuple[str, str, str]]:
    """(owner, repo, canonical_url) for a github.com repository URL, else None."""
    if not isinstance(raw, str):
        return None
    url = raw.strip()
    if not url or len(url) > MAX_URL_LENGTH or any(ord(c) < 32 or ord(c) == 127 for c in url):
        return None
    owner = repo_name = None
    scp = _SCP.fullmatch(url)
    if scp:
        owner, repo_name = scp.group(1), scp.group(2)
    else:
        parts = urlsplit(url)
        if parts.scheme not in ("https", "ssh") or (parts.hostname or "").lower() not in _GITHUB_HOSTS:
            return None
        segments = [s for s in parts.path.split("/") if s]
        if len(segments) < 2:
            return None
        owner, repo_name = segments[0], segments[1]
        if repo_name.endswith(".git"):
            repo_name = repo_name[:-4]
    try:
        owner, repo_name = _validate_owner(owner), _validate_repo(repo_name)
    except ValueError:
        return None
    return owner, repo_name, f"https://github.com/{owner}/{repo_name}"


def _safe_display(raw: Any) -> str:
    """A bounded, credential-free rendering of an unusable URL, for display only."""
    text = _USERINFO.sub("", str(raw))[:MAX_URL_LENGTH]
    return scrub_text(text)[0]


def register_discovered_repository(
    conn: psycopg.Connection, project_uuid: UUID, service_id: Optional[str], raw_url: Any,
    source: str = "startup_event",
) -> Optional[dict]:
    """Record (or refresh) a repository seen in telemetry. Never verifies, never grants access."""
    if not isinstance(raw_url, str) or not raw_url.strip():
        return None
    parsed = parse_repository_url(raw_url)
    with conn.transaction():
        if parsed is None:
            return repo.upsert_project_repository(
                conn, project_uuid, (service_id or "")[:200], _safe_display(raw_url), None, None,
                "unsupported_url", "Not a recognised github.com repository URL", source,
            )
        owner, name, canonical = parsed
        return repo.upsert_project_repository(
            conn, project_uuid, (service_id or "")[:200], canonical, owner, name, "unverified",
            "Not yet matched to a GitHub App installation", source,
        )


def verify_repository(conn: psycopg.Connection, user_id: UUID, row: Dict[str, Any]) -> Dict[str, Any]:
    """Try to match one mapping to one of `user_id`'s installations. Returns the updated row."""
    if row["verification_status"] == "unsupported_url" or not row["owner"] or not row["repo"]:
        return row
    ctx = ToolContext(user_id=user_id, conn=conn, deadline=time.monotonic() + VERIFY_DEADLINE_SECONDS)
    installations = [
        i for i in db_repo.list_github_app_installations_for_user(conn, user_id) if i["suspended_at"] is None
    ][:MAX_INSTALLATIONS_CHECKED]
    problem: Optional[str] = None
    for installation in installations:
        try:
            client = access.open_installation(ctx, installation["installation_id"])
            entry = access.find_repository(client, row["owner"], row["repo"])
        except ToolError as exc:
            if exc.code != "repository_not_accessible":
                problem = exc.code
            continue
        except Exception as exc:  # never let verification break ingestion
            logger.error("repository verification failed: %s", redact(f"{type(exc).__name__}: {exc}"))
            problem = "internal_error"
            continue
        owner_info = entry.get("owner") if isinstance(entry.get("owner"), dict) else {}
        try:
            owner = _validate_owner(str(owner_info.get("login") or row["owner"]))
            name = _validate_repo(str(entry.get("name") or row["repo"]))
        except ValueError:
            problem = "github_bad_response"
            continue
        branch = entry.get("default_branch") if isinstance(entry.get("default_branch"), str) else None
        with conn.transaction():
            return repo.mark_repository_verified(
                conn, row["id"], int(installation["installation_id"]), owner, name, branch
            )
    if not installations:
        note = "No active GitHub App installation is linked to your account"
    elif problem:
        note = f"Verification could not be completed ({problem}); it will be retried"
    else:
        note = "Repository was not found in any of your GitHub App installations"
    with conn.transaction():
        updated = repo.mark_repository_unverified(conn, row["id"], note)
    return updated or row


def verify_if_due(conn: psycopg.Connection, user_id: UUID, row: Dict[str, Any]) -> Dict[str, Any]:
    """Verify an unverified mapping unless it was checked very recently."""
    if row["verification_status"] != "unverified":
        return row
    last = row.get("last_verification_at")
    if isinstance(last, datetime) and datetime.now(timezone.utc) - last < VERIFY_RETRY_AFTER:
        return row
    return verify_repository(conn, user_id, row)


def verify_pending(conn: psycopg.Connection, user_id: UUID, project_uuid: UUID, limit: int = 20) -> List[dict]:
    rows = [r for r in repo.list_project_repositories(conn, project_uuid, "unverified")][:limit]
    return [verify_repository(conn, user_id, r) for r in rows]


def process_startup_events(
    conn: psycopg.Connection, user_id: UUID, startup_events: List[Dict[str, Any]]
) -> None:
    """Post-commit hook: register/refresh repositories named by NEW startup events.

    Failures are logged (redacted) and swallowed: the events are already persisted, and
    `POST /projects/{id}/repositories/verify` can retry verification at any time.
    """
    seen = set()
    for event in startup_events:
        key = (event["project_uuid"], event["service_id"] or "", event.get("repository"))
        if not event.get("repository") or key in seen:
            continue
        seen.add(key)
        try:
            row = register_discovered_repository(conn, event["project_uuid"], event["service_id"], event["repository"])
            if row is not None:
                verify_if_due(conn, user_id, row)
        except Exception as exc:
            logger.error("repository discovery failed: %s", redact(f"{type(exc).__name__}: {exc}"))

