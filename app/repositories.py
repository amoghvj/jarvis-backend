"""All SQL lives here. Every statement is parameterised."""

from typing import Any, Dict, Iterable, List, Optional
from uuid import UUID

import psycopg
from psycopg.types.json import Jsonb

Row = Dict[str, Any]


# --- users -----------------------------------------------------------------

def upsert_user(
    conn: psycopg.Connection,
    *,
    github_id: int,
    username: str,
    email: Optional[str],
    avatar_url: Optional[str],
) -> Row:
    """Create the user on first login, otherwise refresh profile data.

    Atomic on the unique `github_id`, so concurrent first logins cannot create
    duplicates. An existing email is kept if GitHub supplied no verified one.
    """
    return conn.execute(
        """
        insert into public.users (github_id, github_username, email, avatar_url)
        values (%s, %s, %s, %s)
        on conflict (github_id) do update set
            github_username = excluded.github_username,
            email           = coalesce(excluded.email, public.users.email),
            avatar_url      = excluded.avatar_url,
            updated_at      = now()
        returning id, github_id, github_username, email, avatar_url, created_at,
                  (xmax = 0) as created
        """,
        (github_id, username, email, avatar_url),
    ).fetchone()


# --- sessions --------------------------------------------------------------

def create_session(conn: psycopg.Connection, user_id: UUID, token_hash: str, ttl_seconds: int) -> Row:
    return conn.execute(
        """
        insert into public.sessions (user_id, token_hash, expires_at)
        values (%s, %s, now() + make_interval(secs => %s))
        returning id, expires_at
        """,
        (user_id, token_hash, ttl_seconds),
    ).fetchone()


def delete_expired_sessions(conn: psycopg.Connection, user_id: UUID) -> None:
    conn.execute(
        "delete from public.sessions where user_id = %s and expires_at <= now()",
        (user_id,),
    )


def get_user_by_session_hash(conn: psycopg.Connection, token_hash: str) -> Optional[Row]:
    return conn.execute(
        """
        select u.id, u.github_id, u.github_username, u.email, u.avatar_url, u.created_at
        from public.sessions s
        join public.users u on u.id = s.user_id
        where s.token_hash = %s and s.expires_at > now()
        """,
        (token_hash,),
    ).fetchone()


def delete_session(conn: psycopg.Connection, token_hash: str) -> None:
    conn.execute("delete from public.sessions where token_hash = %s", (token_hash,))


# --- api keys --------------------------------------------------------------

def upsert_api_key(conn: psycopg.Connection, user_id: UUID, key_hash: str, key_prefix: str) -> Row:
    """Insert or atomically replace the user's single key (unique on user_id)."""
    return conn.execute(
        """
        insert into public.api_keys (user_id, key_hash, key_prefix)
        values (%s, %s, %s)
        on conflict (user_id) do update set
            key_hash     = excluded.key_hash,
            key_prefix   = excluded.key_prefix,
            created_at   = now(),
            last_used_at = null
        returning key_prefix, created_at
        """,
        (user_id, key_hash, key_prefix),
    ).fetchone()


def get_api_key_for_user(conn: psycopg.Connection, user_id: UUID) -> Optional[Row]:
    return conn.execute(
        "select key_prefix, created_at, last_used_at from public.api_keys where user_id = %s",
        (user_id,),
    ).fetchone()


def delete_api_key(conn: psycopg.Connection, user_id: UUID) -> bool:
    cur = conn.execute("delete from public.api_keys where user_id = %s", (user_id,))
    return cur.rowcount > 0


def find_api_key_account(conn: psycopg.Connection, key_hash: str) -> Optional[Row]:
    return conn.execute(
        """
        select k.id as key_id, u.id as user_id, u.github_username
        from public.api_keys k
        join public.users u on u.id = k.user_id
        where k.key_hash = %s
        """,
        (key_hash,),
    ).fetchone()


def touch_api_key(conn: psycopg.Connection, key_id: UUID) -> None:
    """Record usage, at most once a minute, to avoid a write on every request."""
    conn.execute(
        """
        update public.api_keys set last_used_at = now()
        where id = %s and (last_used_at is null or last_used_at < now() - interval '1 minute')
        """,
        (key_id,),
    )


# --- projects --------------------------------------------------------------
# `projects.project_id` is the external ID the SDK sends and is unique PER USER
# (UNIQUE (user_id, project_id)); `projects.id` is the internal UUID used by every relationship
# and every authorization decision.

_PROJECT_COLUMNS = "id, user_id, project_id, name, created_at, updated_at"


def get_user_projects_by_external_ids(
    conn: psycopg.Connection, user_id: UUID, project_ids: Iterable[str]
) -> Dict[str, UUID]:
    """External project_id -> internal UUID, restricted to projects owned by `user_id`."""
    rows = conn.execute(
        "select project_id, id from public.projects where user_id = %s and project_id = any(%s)",
        (user_id, list(project_ids)),
    ).fetchall()
    return {r["project_id"]: r["id"] for r in rows}


def get_external_ids_owned_by_others(
    conn: psycopg.Connection, user_id: UUID, project_ids: Iterable[str]
) -> set:
    rows = conn.execute(
        "select distinct project_id from public.projects where user_id <> %s and project_id = any(%s)",
        (user_id, list(project_ids)),
    ).fetchall()
    return {r["project_id"] for r in rows}


def insert_project(conn: psycopg.Connection, user_id: UUID, project_id: str, name: str) -> Optional[Row]:
    """Insert a project; None if this user already has that project_id (race-safe)."""
    return conn.execute(
        f"""
        insert into public.projects (user_id, project_id, name)
        values (%s, %s, %s)
        on conflict (user_id, project_id) do nothing
        returning {_PROJECT_COLUMNS}
        """,
        (user_id, project_id, name),
    ).fetchone()


def list_projects_for_user(conn: psycopg.Connection, user_id: UUID) -> List[Row]:
    return conn.execute(
        f"select {_PROJECT_COLUMNS} from public.projects where user_id = %s order by created_at, id",
        (user_id,),
    ).fetchall()


def get_project_for_user(conn: psycopg.Connection, user_id: UUID, project_uuid: UUID) -> Optional[Row]:
    """The project only if it belongs to `user_id` (the one authorization check for projects)."""
    return conn.execute(
        f"select {_PROJECT_COLUMNS} from public.projects where id = %s and user_id = %s",
        (project_uuid, user_id),
    ).fetchone()


def update_project_name(
    conn: psycopg.Connection, user_id: UUID, project_uuid: UUID, name: str
) -> Optional[Row]:
    return conn.execute(
        f"""
        update public.projects set name = %s, updated_at = now()
        where id = %s and user_id = %s
        returning {_PROJECT_COLUMNS}
        """,
        (name, project_uuid, user_id),
    ).fetchone()


# --- user lookup -----------------------------------------------------------

def get_user_by_id(conn: psycopg.Connection, user_id: UUID) -> Optional[Row]:
    return conn.execute(
        """
        select id, github_id, github_username, email, avatar_url, created_at
        from public.users
        where id = %s
        """,
        (user_id,),
    ).fetchone()


# --- github app installations & pending flows ------------------------------

def delete_expired_pending_installations(conn: psycopg.Connection, user_id: UUID) -> None:
    conn.execute(
        """
        delete from public.github_app_pending_installations
        where user_id = %s and (expires_at <= now() or consumed_at is not null)
        """,
        (user_id,),
    )


def create_pending_installation(
    conn: psycopg.Connection,
    *,
    user_id: UUID,
    state_hash: str,
    ttl_seconds: int,
    stage: str,
    installation_id: Optional[int] = None,
    setup_action: Optional[str] = None,
) -> Row:
    return conn.execute(
        """
        insert into public.github_app_pending_installations (
            user_id, installation_id, setup_action, stage, state_hash, expires_at
        )
        values (%s, %s, %s, %s, %s, now() + make_interval(secs => %s))
        returning id, user_id, installation_id, setup_action, stage, expires_at, created_at
        """,
        (user_id, installation_id, setup_action, stage, state_hash, ttl_seconds),
    ).fetchone()


def consume_install_flow_by_hash(
    conn: psycopg.Connection, state_hash: str, user_id: UUID
) -> Optional[Row]:
    return conn.execute(
        """
        update public.github_app_pending_installations
        set consumed_at = now()
        where state_hash = %s
          and user_id = %s
          and stage = 'install'
          and consumed_at is null
          and expires_at > now()
        returning id, user_id
        """,
        (state_hash, user_id),
    ).fetchone()


def get_pending_installation_by_hash(
    conn: psycopg.Connection, state_hash: str
) -> Optional[Row]:
    return conn.execute(
        """
        select id, user_id, installation_id, setup_action, stage, consumed_at,
               (expires_at <= now()) as is_expired
        from public.github_app_pending_installations
        where state_hash = %s
        """,
        (state_hash,),
    ).fetchone()


def consume_pending_installation(
    conn: psycopg.Connection, pending_id: UUID
) -> Optional[Row]:
    return conn.execute(
        """
        update public.github_app_pending_installations
        set consumed_at = now()
        where id = %s
          and stage = 'oauth'
          and installation_id is not null
          and consumed_at is null
          and expires_at > now()
        returning id, user_id, installation_id, setup_action
        """,
        (pending_id,),
    ).fetchone()


def upsert_github_app_installation(
    conn: psycopg.Connection,
    *,
    user_id: UUID,
    installation_id: int,
    github_account_id: int,
    github_account_login: str,
    github_account_type: str,
    repository_selection: Optional[str],
    permissions: Dict[str, Any],
    suspended_at: Optional[Any] = None,
) -> Optional[Row]:
    """Insert or update an installation owned by `user_id`.

    Returns None if `installation_id` is already linked to a different J.A.R.V.I.S. user.
    """
    return conn.execute(
        """
        insert into public.github_app_installations (
            user_id, installation_id, github_account_id, github_account_login,
            github_account_type, repository_selection, permissions, suspended_at
        )
        values (%s, %s, %s, %s, %s, %s, %s, %s)
        on conflict (installation_id) do update set
            github_account_id    = excluded.github_account_id,
            github_account_login = excluded.github_account_login,
            github_account_type  = excluded.github_account_type,
            repository_selection = excluded.repository_selection,
            permissions          = excluded.permissions,
            suspended_at         = excluded.suspended_at,
            updated_at           = now()
        where public.github_app_installations.user_id = excluded.user_id
        returning id, user_id, installation_id, github_account_id, github_account_login,
                  github_account_type, repository_selection, permissions, suspended_at,
                  created_at, updated_at
        """,
        (
            user_id,
            installation_id,
            github_account_id,
            github_account_login,
            github_account_type,
            repository_selection,
            Jsonb(permissions or {}),
            suspended_at,
        ),
    ).fetchone()


def list_github_app_installations_for_user(
    conn: psycopg.Connection, user_id: UUID
) -> List[Row]:
    return conn.execute(
        """
        select id, user_id, installation_id, github_account_id, github_account_login,
               github_account_type, repository_selection, permissions, suspended_at,
               created_at, updated_at
        from public.github_app_installations
        where user_id = %s
        order by created_at asc
        """,
        (user_id,),
    ).fetchall()


def get_github_app_installation_for_user(
    conn: psycopg.Connection, user_id: UUID, installation_id: int
) -> Optional[Row]:
    return conn.execute(
        """
        select id, user_id, installation_id, github_account_id, github_account_login,
               github_account_type, repository_selection, permissions, suspended_at,
               created_at, updated_at
        from public.github_app_installations
        where user_id = %s and installation_id = %s
        """,
        (user_id, installation_id),
    ).fetchone()


def delete_github_app_installation(
    conn: psycopg.Connection, user_id: UUID, installation_id: int
) -> bool:
    cur = conn.execute(
        """
        delete from public.github_app_installations
        where user_id = %s and installation_id = %s
        """,
        (user_id, installation_id),
    )
    return cur.rowcount > 0


