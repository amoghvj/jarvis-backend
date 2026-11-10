"""SQL for telemetry events, incidents, incident history and project repositories.

Every statement is scoped by the internal project UUID. Callers decide transaction boundaries.
"""

from typing import Any, Dict, List, Optional
from uuid import UUID

import psycopg
from psycopg.rows import Row
from psycopg.types.json import Jsonb

# --- events ------------------------------------------------------------------


def insert_event(conn: psycopg.Connection, project_uuid: UUID, e: Dict[str, Any]) -> Optional[UUID]:
    """Insert one event; returns its UUID, or None if (project_id, event_id) already exists."""
    row = conn.execute(
        """
        insert into public.telemetry_events (
            project_id, event_id, schema_version, event_type, classification, event_timestamp,
            application, service_id, environment, instance_id, runtime, severity, message,
            logger_name, error_type, stack_trace, source_file, source_function, source_line,
            trace_id, span_id, request_id, thread_name, attributes, payload
        ) values (
            %(project_id)s, %(event_id)s, %(schema_version)s, %(event_type)s, %(classification)s,
            %(event_timestamp)s, %(application)s, %(service_id)s, %(environment)s, %(instance_id)s,
            %(runtime)s, %(severity)s, %(message)s, %(logger_name)s, %(error_type)s, %(stack_trace)s,
            %(source_file)s, %(source_function)s, %(source_line)s, %(trace_id)s, %(span_id)s,
            %(request_id)s, %(thread_name)s, %(attributes)s, %(payload)s
        )
        on conflict (project_id, event_id) do nothing
        returning id
        """,
        {
            **e,
            "project_id": project_uuid,
            "application": Jsonb(e["application"]),
            "runtime": Jsonb(e["runtime"]),
            "attributes": Jsonb(e["attributes"]),
            "payload": Jsonb(e["payload"]),
        },
    ).fetchone()
    return row["id"] if row else None


def set_event_incident(conn: psycopg.Connection, event_uuid: UUID, incident_id: UUID) -> None:
    conn.execute("update public.telemetry_events set incident_id = %s where id = %s", (incident_id, event_uuid))


# --- incidents ---------------------------------------------------------------


def lock_incident_group(conn: psycopg.Connection, key: str) -> None:
    """Serialize grouping for one (project, fingerprint) until the transaction ends."""
    conn.execute("select pg_advisory_xact_lock(hashtextextended(%s, 0))", (key,))


def find_open_incident(
    conn: psycopg.Connection, project_uuid: UUID, fingerprint: str, ts: Any, window_seconds: int
) -> Optional[Row]:
    return conn.execute(
        """
        select id, severity, status, first_seen_at, last_seen_at
        from public.incidents
        where project_id = %(p)s and fingerprint = %(f)s
          and status in ('open', 'acknowledged')
          and last_seen_at  >= %(ts)s - make_interval(secs => %(w)s)
          and first_seen_at <= %(ts)s + make_interval(secs => %(w)s)
        order by last_seen_at desc, id
        limit 1
        for update
        """,
        {"p": project_uuid, "f": fingerprint, "ts": ts, "w": window_seconds},
    ).fetchone()


def insert_incident(
    conn: psycopg.Connection, project_uuid: UUID, *, title: str, severity: str, fingerprint: str,
    service_id: Optional[str], environment: Optional[str], ts: Any, data: Dict[str, Any],
) -> UUID:
    row = conn.execute(
        """
        insert into public.incidents (
            project_id, title, status, severity, incident_data, fingerprint, service_id,
            environment, first_seen_at, last_seen_at, event_count
        ) values (%s, %s, 'open', %s, %s, %s, %s, %s, %s, %s, 1)
        returning id
        """,
        (project_uuid, title, severity, Jsonb(data), fingerprint, service_id, environment, ts, ts),
    ).fetchone()
    return row["id"]


def bump_incident(
    conn: psycopg.Connection, incident_id: UUID, ts: Any, severity: Optional[str]
) -> None:
    conn.execute(
        """
        update public.incidents set
            event_count   = event_count + 1,
            first_seen_at = least(first_seen_at, %s),
            last_seen_at  = greatest(last_seen_at, %s),
            severity      = coalesce(%s, severity),
            updated_at    = now()
        where id = %s
        """,
        (ts, ts, severity, incident_id),
    )


def insert_history(
    conn: psycopg.Connection, project_uuid: UUID, incident_id: UUID, change_type: str, *,
    from_status: Optional[str] = None, to_status: Optional[str] = None,
    from_severity: Optional[str] = None, to_severity: Optional[str] = None,
    actor_user_id: Optional[UUID] = None, note: Optional[str] = None,
) -> None:
    conn.execute(
        """
        insert into public.incident_history (
            incident_id, project_id, change_type, from_status, to_status,
            from_severity, to_severity, actor_user_id, note
        ) values (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (incident_id, project_uuid, change_type, from_status, to_status, from_severity, to_severity,
         actor_user_id, note),
    )


def get_incident_for_update(conn: psycopg.Connection, project_uuid: UUID, incident_id: UUID) -> Optional[Row]:
    return conn.execute(
        "select id, status, severity from public.incidents where id = %s and project_id = %s for update",
        (incident_id, project_uuid),
    ).fetchone()


def set_incident_status(conn: psycopg.Connection, incident_id: UUID, status: str) -> Row:
    return conn.execute(
        """
        update public.incidents set
            status = %(s)s,
            acknowledged_at = case when %(s)s = 'acknowledged' then now() else acknowledged_at end,
            resolved_at = case when %(s)s = 'resolved' then now()
                              when %(s)s = 'open' then null else resolved_at end,
            updated_at = now()
        where id = %(id)s
        returning id, project_id, title, status, severity, service_id, environment, event_count,
                  first_seen_at, last_seen_at, acknowledged_at, resolved_at, created_at, updated_at
        """,
        {"s": status, "id": incident_id},
    ).fetchone()


INCIDENT_LIST_COLUMNS = (
    "id, project_id, title, status, severity, service_id, environment, event_count, first_seen_at, "
    "last_seen_at, acknowledged_at, resolved_at, created_at, updated_at"
)


def list_incidents(
    conn: psycopg.Connection, project_uuid: UUID, status: Optional[str], limit: int, offset: int
) -> List[Row]:
    return conn.execute(
        f"""
        select {INCIDENT_LIST_COLUMNS} from public.incidents
        where project_id = %s and (%s::text is null or status = %s)
        order by last_seen_at desc nulls last, id
        limit %s offset %s
        """,
        (project_uuid, status, status, limit, offset),
    ).fetchall()


# --- project repositories ----------------------------------------------------

_REPO_COLUMNS = (
    "id, project_id, service_id, repository_url, owner, repo, installation_id, default_branch, "
    "verification_status, verification_note, discovery_source, first_seen_at, last_seen_at, "
    "last_verification_at, verified_at"
)


def upsert_project_repository(
    conn: psycopg.Connection, project_uuid: UUID, service_id: str, repository_url: str,
    owner: Optional[str], repo: Optional[str], status: str, note: Optional[str], source: str,
) -> Row:
    """Register or refresh a discovered repository. Never downgrades or changes verification."""
    return conn.execute(
        f"""
        insert into public.project_repositories (
            project_id, service_id, repository_url, owner, repo, verification_status,
            verification_note, discovery_source
        ) values (%s, %s, %s, %s, %s, %s, %s, %s)
        on conflict (project_id, service_id, repository_url) do update set
            last_seen_at = clock_timestamp(), updated_at = clock_timestamp(),
            -- a verified row whose installation was deleted (installation_id nulled) is no longer verified
            verification_status = case
                when public.project_repositories.verification_status = 'verified'
                     and public.project_repositories.installation_id is null then 'unverified'
                else public.project_repositories.verification_status end
        returning {_REPO_COLUMNS}
        """,
        (project_uuid, service_id, repository_url, owner, repo, status, note, source),
    ).fetchone()


def get_project_repository(conn: psycopg.Connection, project_uuid: UUID, repository_id: UUID) -> Optional[Row]:
    return conn.execute(
        f"select {_REPO_COLUMNS} from public.project_repositories where id = %s and project_id = %s",
        (repository_id, project_uuid),
    ).fetchone()


def _effective(row: Row) -> Row:
    """A stored 'verified' only counts while its installation is still the owner's and active."""
    row = dict(row)
    still_valid = row.pop("installation_valid")
    if row["verification_status"] == "verified" and not still_valid:
        row.update(
            verification_status="unverified", installation_id=None, verified_at=None,
            verification_note="The GitHub App installation is no longer available; verify again",
        )
    return row


def list_project_repositories(
    conn: psycopg.Connection, project_uuid: UUID, status: Optional[str] = None
) -> List[Row]:
    """Mappings with their EFFECTIVE verification status (re-checked against the installation)."""
    rows = conn.execute(
        """
        select r.id, r.project_id, r.service_id, r.repository_url, r.owner, r.repo, r.installation_id,
               r.default_branch, r.verification_status, r.verification_note, r.discovery_source,
               r.first_seen_at, r.last_seen_at, r.last_verification_at, r.verified_at,
               (i.installation_id is not null) as installation_valid
        from public.project_repositories r
        join public.projects p on p.id = r.project_id
        left join public.github_app_installations i
          on i.installation_id = r.installation_id and i.user_id = p.user_id and i.suspended_at is null
        where r.project_id = %s
        order by r.service_id, r.repository_url
        """,
        (project_uuid,),
    ).fetchall()
    mapped = [_effective(r) for r in rows]
    return [r for r in mapped if status is None or r["verification_status"] == status]


def mark_repository_verified(
    conn: psycopg.Connection, repository_id: UUID, installation_id: int, owner: str, repo: str,
    default_branch: Optional[str],
) -> Row:
    return conn.execute(
        f"""
        update public.project_repositories set
            verification_status = 'verified', installation_id = %s, owner = %s, repo = %s,
            default_branch = %s, verification_note = null, verified_at = now(),
            last_verification_at = now(), updated_at = now()
        where id = %s
        returning {_REPO_COLUMNS}
        """,
        (installation_id, owner, repo, default_branch, repository_id),
    ).fetchone()


def mark_repository_unverified(conn: psycopg.Connection, repository_id: UUID, note: str) -> Row:
    return conn.execute(
        f"""
        update public.project_repositories set
            last_verification_at = now(), verification_note = %s, updated_at = now()
        where id = %s and verification_status <> 'verified'
        returning {_REPO_COLUMNS}
        """,
        (note, repository_id),
    ).fetchone()


def verified_repositories(conn: psycopg.Connection, project_uuid: UUID) -> List[Row]:
    """Repositories that are verified AND whose installation still belongs to the project owner
    and is not suspended. This is the only query that may feed repository tool targets."""
    return conn.execute(
        """
        select r.service_id, r.owner, r.repo, r.repository_url, r.installation_id, r.default_branch,
               r.verification_status, r.first_seen_at, r.last_seen_at, r.verified_at
        from public.project_repositories r
        join public.projects p on p.id = r.project_id
        join public.github_app_installations i
          on i.installation_id = r.installation_id and i.user_id = p.user_id
        where r.project_id = %s and r.verification_status = 'verified' and i.suspended_at is null
        order by r.service_id, r.owner, r.repo
        """,
        (project_uuid,),
    ).fetchall()


def count_unverified_repositories(conn: psycopg.Connection, project_uuid: UUID) -> int:
    row = conn.execute(
        """
        select count(*) as n from public.project_repositories r
        where r.project_id = %s and not exists (
            select 1 from public.projects p
            join public.github_app_installations i on i.user_id = p.user_id
            where p.id = r.project_id and i.installation_id = r.installation_id
              and i.suspended_at is null and r.verification_status = 'verified')
        """,
        (project_uuid,),
    ).fetchone()
    return int(row["n"])
