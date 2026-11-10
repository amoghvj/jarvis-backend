"""Serverless-safe PostgreSQL access (Supabase Transaction Pooler).

A short-lived connection is opened per request (no process-level pool, which
does not suit serverless). Connections use autocommit, so every write is wrapped
explicitly in `with conn.transaction():` and is committed *before* the HTTP
response is produced. `prepare_threshold=None` disables server-side prepared
statements, which the transaction pooler does not support.
"""

from typing import Iterator

import psycopg
from psycopg.rows import dict_row

from app.config import get_settings


def open_connection() -> psycopg.Connection:
    """Open a short-lived PostgreSQL connection compatible with the Supabase Transaction Pooler."""
    return psycopg.connect(
        get_settings().database_url,
        row_factory=dict_row,
        autocommit=True,
        prepare_threshold=None,
        connect_timeout=10,
    )


def get_db() -> Iterator[psycopg.Connection]:
    """FastAPI dependency yielding a request-scoped connection."""
    conn = open_connection()
    try:
        yield conn
    finally:
        conn.close()

