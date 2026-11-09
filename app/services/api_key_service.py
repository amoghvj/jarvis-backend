"""API key generation, rotation, revocation and verification."""

import hashlib
import secrets
from dataclasses import dataclass
from typing import Optional
from uuid import UUID

import psycopg

from app import repositories

KEY_PREFIX = "jarvis_live_"
SECRET_BYTES = 32
DISPLAY_PREFIX_LENGTH = len(KEY_PREFIX) + 4  # safe-to-show leading characters


class InvalidApiKey(Exception):
    """Missing, malformed, revoked or rotated API key."""


@dataclass(frozen=True)
class ApiKeyAccount:
    key_id: UUID
    user_id: UUID
    github_username: str


def hash_api_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def generate_api_key() -> str:
    return KEY_PREFIX + secrets.token_urlsafe(SECRET_BYTES)


def rotate_api_key(conn: psycopg.Connection, user_id: UUID) -> dict:
    """Create a new key, atomically replacing any existing one.

    The old hash is overwritten by the same statement that stores the new one,
    inside a transaction, so there is never a moment with no valid key and the
    old key only dies if the new one was stored. The raw key is returned to the
    caller only; it is never stored or logged.
    """
    raw_key = generate_api_key()
    with conn.transaction():
        row = repositories.upsert_api_key(
            conn, user_id, hash_api_key(raw_key), raw_key[:DISPLAY_PREFIX_LENGTH]
        )
    return {"api_key": raw_key, "key_prefix": row["key_prefix"], "created_at": row["created_at"]}


def get_status(conn: psycopg.Connection, user_id: UUID) -> dict:
    row = repositories.get_api_key_for_user(conn, user_id)
    if not row:
        return {"active": False, "key_prefix": None, "created_at": None, "last_used_at": None}
    return {"active": True, **row}


def revoke_api_key(conn: psycopg.Connection, user_id: UUID) -> bool:
    with conn.transaction():
        return repositories.delete_api_key(conn, user_id)


def authenticate(conn: psycopg.Connection, authorization: Optional[str]) -> ApiKeyAccount:
    """Resolve `Authorization: Bearer <key>` to an account or raise InvalidApiKey."""
    if not authorization:
        raise InvalidApiKey("Missing credentials")
    scheme, _, token = authorization.partition(" ")
    token = token.strip()
    if scheme.lower() != "bearer" or not token.startswith(KEY_PREFIX):
        raise InvalidApiKey("Invalid credentials")

    row = repositories.find_api_key_account(conn, hash_api_key(token))
    if not row:
        raise InvalidApiKey("Invalid credentials")
    repositories.touch_api_key(conn, row["key_id"])
    return ApiKeyAccount(
        key_id=row["key_id"], user_id=row["user_id"], github_username=row["github_username"]
    )

