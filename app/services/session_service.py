"""Persistent (database-backed) sessions, cookies and OAuth state handling."""

import hashlib
import hmac
import secrets
import time
from typing import Optional

import psycopg
from fastapi import Response

from app import repositories
from app.config import OAUTH_STATE_TTL_SECONDS, SESSION_TTL_SECONDS, Settings

SESSION_COOKIE = "jarvis_session"
STATE_COOKIE = "jarvis_oauth_state"
STATE_COOKIE_PATH = "/auth"


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


# --- sessions --------------------------------------------------------------

def create_session(conn: psycopg.Connection, user_id) -> str:
    """Persist a new session and return the raw token (only ever sent as a cookie)."""
    token = secrets.token_urlsafe(32)
    with conn.transaction():
        repositories.delete_expired_sessions(conn, user_id)
        repositories.create_session(conn, user_id, _hash_token(token), SESSION_TTL_SECONDS)
    return token


def get_user_for_token(conn: psycopg.Connection, token: Optional[str]) -> Optional[dict]:
    if not token:
        return None
    return repositories.get_user_by_session_hash(conn, _hash_token(token))


def revoke_session(conn: psycopg.Connection, token: Optional[str]) -> None:
    if token:
        repositories.delete_session(conn, _hash_token(token))


def set_session_cookie(response: Response, settings: Settings, token: str) -> None:
    response.set_cookie(
        SESSION_COOKIE,
        token,
        max_age=SESSION_TTL_SECONDS,
        httponly=True,
        secure=settings.cookie_secure,
        samesite=settings.cookie_samesite,
        path="/",
    )


def clear_session_cookie(response: Response, settings: Settings) -> None:
    # Attributes must match those used when the cookie was set.
    response.delete_cookie(
        SESSION_COOKIE,
        httponly=True,
        secure=settings.cookie_secure,
        samesite=settings.cookie_samesite,
        path="/",
    )


# --- OAuth state -----------------------------------------------------------

def _sign(settings: Settings, payload: str) -> str:
    return hmac.new(settings.session_secret.encode(), payload.encode(), hashlib.sha256).hexdigest()


def new_state() -> str:
    return secrets.token_urlsafe(32)


def set_state_cookie(response: Response, settings: Settings, state: str) -> None:
    """Bind the state to this browser with a short-lived signed cookie (no server storage needed)."""
    payload = f"{state}.{int(time.time()) + OAUTH_STATE_TTL_SECONDS}"
    response.set_cookie(
        STATE_COOKIE,
        f"{payload}.{_sign(settings, 'oauth-state:' + payload)}",
        max_age=OAUTH_STATE_TTL_SECONDS,
        httponly=True,
        secure=settings.cookie_secure,
        samesite="lax",  # sent on GitHub's top-level redirect back to the callback
        path=STATE_COOKIE_PATH,
    )


def clear_state_cookie(response: Response, settings: Settings) -> None:
    response.delete_cookie(
        STATE_COOKIE,
        httponly=True,
        secure=settings.cookie_secure,
        samesite="lax",
        path=STATE_COOKIE_PATH,
    )


def verify_state(settings: Settings, state_param: Optional[str], cookie_value: Optional[str]) -> bool:
    if not state_param or not cookie_value:
        return False
    parts = cookie_value.split(".")
    if len(parts) != 3:
        return False
    state, expires, signature = parts
    expected = _sign(settings, f"oauth-state:{state}.{expires}")
    if not hmac.compare_digest(signature, expected):
        return False
    if not expires.isdigit() or int(expires) < time.time():
        return False
    return hmac.compare_digest(state, state_param)

