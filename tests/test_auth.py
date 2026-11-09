from urllib.parse import parse_qs, urlsplit

from app.services import github_oauth, session_service
from tests.conftest import FRONTEND, csrf, login


def _count(db, table="users"):
    return db.execute(f"select count(*) as n from public.{table}").fetchone()["n"]


# --- OAuth start / state ---------------------------------------------------

def test_authorize_redirect_uses_minimal_scopes_and_state(client):
    r = client.get("/auth/github", follow_redirects=False)
    assert r.status_code == 302
    url = urlsplit(r.headers["location"])
    q = parse_qs(url.query)
    assert url.netloc == "github.com"
    assert q["scope"] == ["read:user user:email"]  # no repo scope
    assert q["redirect_uri"] == ["https://jarvis-test.example/auth/github/callback"]
    assert len(q["state"][0]) >= 40
    cookie = r.headers["set-cookie"].lower()
    assert "httponly" in cookie and "secure" in cookie


def test_state_is_random_per_request(client):
    a = parse_qs(urlsplit(client.get("/auth/github", follow_redirects=False).headers["location"]).query)
    b = parse_qs(urlsplit(client.get("/auth/github", follow_redirects=False).headers["location"]).query)
    assert a["state"] != b["state"]


def test_callback_rejects_wrong_state(client, github):
    client.get("/auth/github", follow_redirects=False)
    r = client.get("/auth/github/callback", params={"code": "c", "state": "forged"})
    assert r.status_code == 400


def test_callback_rejects_missing_state_cookie(client, github):
    r = client.get("/auth/github/callback", params={"code": "c", "state": "whatever"})
    assert r.status_code == 400


def test_callback_rejects_tampered_state_cookie(client, github):
    start = client.get("/auth/github", follow_redirects=False)
    state = parse_qs(urlsplit(start.headers["location"]).query)["state"][0]
    client.cookies.clear()  # drop the genuine cookie so only the forged one is sent
    client.cookies.set(session_service.STATE_COOKIE, f"{state}.9999999999.deadbeef", path="/auth")
    r = client.get("/auth/github/callback", params={"code": "c", "state": state})
    assert r.status_code == 400


def test_callback_rejects_expired_state(client, github, monkeypatch):
    start = client.get("/auth/github", follow_redirects=False)
    state = parse_qs(urlsplit(start.headers["location"]).query)["state"][0]
    real = session_service.time.time
    monkeypatch.setattr(session_service.time, "time", lambda: real() + 3600)
    r = client.get("/auth/github/callback", params={"code": "c", "state": state})
    assert r.status_code == 400


def test_callback_handles_github_error_param(client, github):
    start = client.get("/auth/github", follow_redirects=False)
    state = parse_qs(urlsplit(start.headers["location"]).query)["state"][0]
    r = client.get("/auth/github/callback", params={"error": "access_denied", "state": state})
    assert r.status_code == 400 and "access_denied" in r.json()["detail"]
    assert "jarvis_session" not in client.cookies


def test_callback_maps_github_failure_to_502(client, monkeypatch):
    def boom(settings, code):
        raise github_oauth.GitHubOAuthError("GitHub rejected the authorization code")

    monkeypatch.setattr(github_oauth, "exchange_code_for_token", boom)
    r = login(client)
    assert r.status_code == 502


def test_state_cannot_be_replayed(client, github):
    start = client.get("/auth/github", follow_redirects=False)
    state = parse_qs(urlsplit(start.headers["location"]).query)["state"][0]
    first = client.get("/auth/github/callback", params={"code": "c", "state": state})
    assert first.status_code == 200
    # The state cookie was cleared, so a replay has nothing to match against.
    again = client.get("/auth/github/callback", params={"code": "c", "state": state})
    assert again.status_code == 400


# --- account creation / login ----------------------------------------------

def test_first_login_creates_user_and_session(client, github, db):
    before = _count(db)
    r = login(client, github)
    assert r.status_code == 200
    body = r.json()
    assert body["new_account"] is True
    assert body["user"]["github_username"] == "octo-test"
    assert body["user"]["email"] == "octo@example.com"
    assert _count(db) == before + 1

    cookie = r.headers["set-cookie"].lower()
    assert "jarvis_session=" in cookie and "httponly" in cookie and "secure" in cookie
    assert "samesite=none" in cookie  # cross-origin frontend in production
    assert "max-age=" in cookie


def test_session_stored_hashed_not_raw(client, github, db):
    login(client, github)
    raw = client.cookies.get("jarvis_session")
    rows = db.execute("select token_hash from public.sessions").fetchall()
    assert raw and all(raw not in r["token_hash"] and len(r["token_hash"]) == 64 for r in rows)


def test_existing_user_login_updates_profile_without_duplicate(client, github, db):
    login(client, github)
    n = _count(db)
    r = login(client, github, username="renamed", email="new@example.com")
    assert r.json()["new_account"] is False
    assert r.json()["user"]["github_username"] == "renamed"
    assert _count(db) == n  # duplicate GitHub ID prevented


def test_login_keeps_email_when_github_returns_none(client, github):
    login(client, github)
    r = login(client, github, email=None)
    assert r.json()["user"]["email"] == "octo@example.com"


def test_different_github_ids_are_different_accounts(client, github, db):
    login(client, github)
    n = _count(db)
    login(client, github, github_id=github.profile.github_id + 1, username="other")
    assert _count(db) == n + 1


def test_redirect_mode_redirects_to_frontend(client, github, monkeypatch):
    monkeypatch.setenv("POST_LOGIN_REDIRECT", "true")
    get_settings = __import__("app.config", fromlist=["get_settings"]).get_settings
    get_settings.cache_clear()
    r = login(client, github)
    assert r.status_code == 302 and r.headers["location"] == FRONTEND


# --- session validation / logout -------------------------------------------

def test_me_requires_authentication(client):
    assert client.get("/auth/me").status_code == 401


def test_me_rejects_garbage_cookie(client):
    client.cookies.set("jarvis_session", "not-a-real-token")
    assert client.get("/auth/me").status_code == 401


def test_me_returns_user_after_login(client, github):
    login(client, github)
    r = client.get("/auth/me")
    assert r.status_code == 200 and r.json()["github_username"] == "octo-test"
    assert "token" not in r.text.lower()


def test_expired_session_rejected(client, github, db):
    login(client, github)
    db.execute("update public.sessions set expires_at = now() - interval '1 second'")
    assert client.get("/auth/me").status_code == 401


def test_logout_invalidates_session_server_side(client, github):
    login(client, github)
    token = client.cookies.get("jarvis_session")
    r = client.post("/auth/logout", headers=csrf())
    assert r.status_code == 200
    assert "jarvis_session=" in r.headers["set-cookie"] and "max-age=0" in r.headers["set-cookie"].lower()
    # Replaying the stolen/old cookie must fail: the server-side session is gone.
    client.cookies.clear()
    client.cookies.set("jarvis_session", token)
    assert client.get("/auth/me").status_code == 401


def test_logout_requires_trusted_origin(client, github):
    login(client, github)
    assert client.post("/auth/logout").status_code == 403
    assert client.post("/auth/logout", headers={"Origin": "https://evil.example"}).status_code == 403
    assert client.get("/auth/me").status_code == 200  # still logged in


def test_logout_without_session_is_idempotent(client):
    assert client.post("/auth/logout", headers=csrf()).status_code == 200


def test_cors_allows_only_frontend_origin(client):
    ok = client.options(
        "/auth/me", headers={"Origin": FRONTEND, "Access-Control-Request-Method": "GET"}
    )
    assert ok.headers.get("access-control-allow-origin") == FRONTEND
    assert ok.headers.get("access-control-allow-credentials") == "true"
    bad = client.options(
        "/auth/me", headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "GET"}
    )
    assert "access-control-allow-origin" not in bad.headers

