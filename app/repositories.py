"""All SQL lives here. Every statement is parameterised."""

from typing import Any, Dict, Iterable, Optional
from uuid import UUID

import psycopg

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

def get_project_owners(conn: psycopg.Connection, project_ids: Iterable[str]) -> Dict[str, UUID]:
    rows = conn.execute(
        "select project_id, user_id from public.projects where project_id = any(%s)",
        (list(project_ids),),
    ).fetchall()
    return {r["project_id"]: r["user_id"] for r in rows}

