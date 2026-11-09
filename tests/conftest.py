"""Test fixtures.

The tests run the real SQL against the configured DATABASE_URL, but inside ONE
transaction that is always rolled back (the app's connection dependency is
replaced and nothing is ever committed). The additive `sessions` migration is
applied inside that same transaction, so the database is left unchanged.
GitHub's HTTP API is mocked; nothing contacts GitHub.
"""

import os
import re
import secrets
import uuid
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")  # supplies DATABASE_URL locally; real env wins

# Deterministic fake OAuth/app config (DATABASE_URL is deliberately untouched).
os.environ.update(
    {
        "GITHUB_CLIENT_ID": "test-client-id",
        "GITHUB_CLIENT_SECRET": "test-client-secret",
        "GITHUB_CALLBACK_URL": "https://jarvis-test.example/auth/github/callback",
        "FRONTEND_URL": "https://frontend-test.example",
        "SESSION_SECRET": "x" * 48,
        "POST_LOGIN_REDIRECT": "false",
    }
)
os.environ.pop("COOKIE_SAMESITE", None)
os.environ.setdefault("DATABASE_URL", "")

import psycopg  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from psycopg.rows import dict_row  # noqa: E402

from app import repositories  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.database import get_db  # noqa: E402
from app.main import app  # noqa: E402
from app.services import api_key_service, github_oauth  # noqa: E402

FRONTEND = "https://frontend-test.example"
BASE = "https://jarvis-test.example"
MIGRATION = (ROOT / "migrations" / "001_create_sessions.sql").read_text()


@pytest.fixture(autouse=True)
def _settings():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def db():
    url = os.environ.get("DATABASE_URL")
    if not url:
        pytest.skip("DATABASE_URL not configured")
    conn = psycopg.connect(url, row_factory=dict_row, prepare_threshold=None, autocommit=False)
    conn.execute(MIGRATION)
    try:
        yield conn
    finally:
        conn.rollback()  # never commit
        conn.close()


@pytest.fixture
def client(db):
    app.dependency_overrides[get_db] = lambda: db
    with TestClient(app, base_url=BASE, raise_server_exceptions=False) as c:
        yield c
    app.dependency_overrides.clear()


class FakeGitHub:
    """Controls what the mocked GitHub API returns."""

    def __init__(self):
        self.profile = github_oauth.GitHubProfile(
            github_id=secrets.randbelow(10**9) + 9_000_000_000,
            username="octo-test",
            avatar_url="https://avatars.example/a.png",
            email="octo@example.com",
        )

    def set(self, **kw):
        self.profile = github_oauth.GitHubProfile(**{**self.profile.__dict__, **kw})


@pytest.fixture
def github(monkeypatch):
    fake = FakeGitHub()
    monkeypatch.setattr(github_oauth, "exchange_code_for_token", lambda settings, code: "gho_fake")
    monkeypatch.setattr(github_oauth, "fetch_profile", lambda token: fake.profile)
    return fake


def login(client, github=None, **profile):
    """Run the full login flow; returns the callback response."""
    if profile and github:
        github.set(**profile)
    start = client.get("/auth/github", follow_redirects=False)
    state = parse_qs(urlsplit(start.headers["location"]).query)["state"][0]
    return client.get(
        "/auth/github/callback", params={"code": "abc", "state": state}, follow_redirects=False
    )


def csrf():
    return {"Origin": FRONTEND}


def make_user(db, **kw):
    return repositories.upsert_user(
        db,
        github_id=kw.get("github_id", secrets.randbelow(10**9) + 8_000_000_000),
        username=kw.get("username", "someone"),
        email=None,
        avatar_url=None,
    )


def make_project(db, user_id, project_id=None):
    project_id = project_id or f"proj-{uuid.uuid4().hex[:8]}"
    db.execute(
        "insert into public.projects (user_id, project_id, name) values (%s, %s, %s)",
        (user_id, project_id, project_id),
    )
    return project_id


def make_key(db, user_id):
    return api_key_service.rotate_api_key(db, user_id)["api_key"]


def sample_event(project_id):
    return {
        "events": [
            {
                "schema_version": "1.0",
                "event_id": "evt-1",
                "timestamp": "2026-11-09T08:30:00Z",
                "type": "error",
                "application": {
                    "project_id": project_id,
                    "service_id": "backend",
                    "runtime": {"language": "python", "version": "3.12.2"},
                },
                "details": {"severity": "ERROR", "message": "boom"},
                "execution": {"trace_id": None, "span_id": None},
                "attributes": {"mechanism": "logging"},
            }
        ]
    }


def bearer(key):
    return {"Authorization": f"Bearer {key}"}

