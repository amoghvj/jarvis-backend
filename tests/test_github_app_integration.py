"""Unit and database tests for the GitHub App integration (Phases 1-3)."""

from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qs, urlsplit

import httpx
import jwt
import pytest

from app import repositories
from app.integrations.github import (
    client as gh_client,
    installation_service,
    jwt_service,
    oauth_service,
)
from app.integrations.github.config import (
    GitHubAppSettings,
    get_github_app_settings,
    normalize_private_key,
)
from tests.conftest import FRONTEND, TEST_PRIVATE_KEY_PEM, TEST_PUBLIC_KEY_PEM, csrf, login


# --- 1. Configuration & JWT generation -------------------------------------

def test_normalize_private_key_supports_multiline_and_escaped_newlines():
    escaped = '"' + TEST_PRIVATE_KEY_PEM.replace("\n", "\\n") + '"'
    normalized = normalize_private_key(escaped)
    assert normalized == TEST_PRIVATE_KEY_PEM
    assert normalized.startswith("-----BEGIN PRIVATE KEY-----")


def test_config_rejects_app_id_confused_with_client_id(monkeypatch):
    monkeypatch.setenv("GITHUB_APP_ID", "Iv1.testappclientid")
    with pytest.raises(RuntimeError, match="GITHUB_APP_ID"):
        get_github_app_settings()


def test_generate_app_jwt_claims_and_signature():
    settings = get_github_app_settings()
    fixed_now = 1_700_000_000
    token = jwt_service.generate_app_jwt(settings, now=fixed_now)
    decoded = jwt.decode(
        token,
        TEST_PUBLIC_KEY_PEM,
        algorithms=["RS256"],
        options={"verify_exp": False},
    )
    assert decoded["iss"] == str(settings.app_id)
    assert decoded["iat"] == fixed_now - 60
    assert decoded["exp"] == fixed_now + 540
    assert decoded["exp"] - fixed_now <= 600


def test_generate_app_jwt_does_not_leak_key_on_error():
    bad_settings = GitHubAppSettings(
        app_id=123,
        client_id="Iv1.abc",
        client_secret="secret",
        private_key="-----BEGIN PRIVATE KEY-----\nnot-valid-base64\n-----END PRIVATE KEY-----\n",
        app_slug="slug",
        callback_url="https://example.com/cb",
        base_settings=get_github_app_settings().base_settings,
    )
    with pytest.raises(jwt_service.GitHubAppJwtError) as exc_info:
        jwt_service.generate_app_jwt(bad_settings)
    assert "not-valid-base64" not in str(exc_info.value)


# --- 2. Installation access-token generation & caching ---------------------

def test_installation_access_token_generation_and_caching(monkeypatch):
    gh_client.clear_token_cache()
    settings = get_github_app_settings()
    calls = []
    future_exp = (datetime.now(timezone.utc) + timedelta(hours=1)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

    def fake_post(url, headers=None, timeout=None):
        calls.append((url, headers))
        return httpx.Response(
            201,
            json={
                "token": "ghs_dynamic_installation_token_value",
                "expires_at": future_exp,
                "permissions": {"contents": "read", "metadata": "read"},
                "repository_selection": "selected",
            },
        )

    monkeypatch.setattr(httpx, "post", fake_post)

    t1 = gh_client.get_installation_access_token(settings, 4242)
    t2 = gh_client.get_installation_access_token(settings, 4242)
    assert t1.token == "ghs_dynamic_installation_token_value"
    assert t2.token == t1.token
    assert len(calls) == 1  # second call served from cache
    assert calls[0][0].endswith("/app/installations/4242/access_tokens")
    assert calls[0][1]["Authorization"].startswith("Bearer ")


def test_installation_token_refreshes_when_near_expiry(monkeypatch):
    gh_client.clear_token_cache()
    settings = get_github_app_settings()
    counter = {"n": 0}

    def fake_post(url, headers=None, timeout=None):
        counter["n"] += 1
        # First token is within the 60s expiry buffer; second is fresh for 1h.
        delta = timedelta(seconds=30) if counter["n"] == 1 else timedelta(hours=1)
        exp = (datetime.now(timezone.utc) + delta).strftime("%Y-%m-%dT%H:%M:%SZ")
        return httpx.Response(
            201,
            json={"token": f"ghs_tok_{counter['n']}", "expires_at": exp},
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    first = gh_client.get_installation_access_token(settings, 777)
    second = gh_client.get_installation_access_token(settings, 777)
    assert first.token == "ghs_tok_1"
    assert second.token == "ghs_tok_2"
    assert counter["n"] == 2


def test_installation_token_handles_revoked_and_rate_limit(monkeypatch):
    gh_client.clear_token_cache()
    settings = get_github_app_settings()

    monkeypatch.setattr(
        httpx, "post", lambda *a, **k: httpx.Response(404, json={"message": "Not Found"})
    )
    with pytest.raises(gh_client.GitHubInstallationRevokedError):
        gh_client.get_installation_access_token(settings, 999)

    monkeypatch.setattr(
        httpx,
        "post",
        lambda *a, **k: httpx.Response(
            403, headers={"x-ratelimit-remaining": "0"}, json={"message": "rate limit"}
        ),
    )
    with pytest.raises(gh_client.GitHubRateLimitError):
        gh_client.get_installation_access_token(settings, 999)


def test_installation_request_retries_once_on_401(monkeypatch):
    gh_client.clear_token_cache()
    settings = get_github_app_settings()
    exp = (datetime.now(timezone.utc) + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    tokens = iter(["ghs_stale", "ghs_fresh"])

    monkeypatch.setattr(
        httpx,
        "post",
        lambda *a, **k: httpx.Response(
            201, json={"token": next(tokens), "expires_at": exp}
        ),
    )
    seen_auth = []

    def fake_request(method, url, headers=None, timeout=None, **kw):
        seen_auth.append(headers["Authorization"])
        if len(seen_auth) == 1:
            return httpx.Response(401, json={"message": "Bad credentials"})
        return httpx.Response(200, json={"ok": True})

    monkeypatch.setattr(httpx, "request", fake_request)
    resp = gh_client.installation_request(settings, 555, "GET", "/installation/repositories")
    assert resp.status_code == 200
    assert seen_auth == ["Bearer ghs_stale", "Bearer ghs_fresh"]


# --- 3. End-to-end installation & OAuth flow -------------------------------

class FakeGitHubAppOAuth:
    def __init__(self, github_id: int, username: str = "octo-test", installation_id: int = 55501):
        self.user = oauth_service.GitHubAppUser(github_id=github_id, login=username)
        self.installations = [
            {
                "id": installation_id,
                "app_id": 987654,
                "account": {
                    "id": github_id,
                    "login": username,
                    "type": "User",
                },
                "repository_selection": "selected",
                "permissions": {"contents": "read", "metadata": "read"},
                "suspended_at": None,
            }
        ]


@pytest.fixture
def mock_gh_app(monkeypatch, github):
    fake = FakeGitHubAppOAuth(github_id=github.profile.github_id)
    monkeypatch.setattr(
        oauth_service,
        "exchange_code_for_user_token",
        lambda settings, code: "ghu_mock_user_token",
    )
    monkeypatch.setattr(
        oauth_service,
        "fetch_authenticated_user",
        lambda token: fake.user,
    )
    monkeypatch.setattr(
        oauth_service,
        "fetch_user_installations",
        lambda token: fake.installations,
    )
    return fake


def test_install_requires_authenticated_session(client):
    r = client.get("/integrations/github/install", follow_redirects=False)
    assert r.status_code == 401


def test_install_initiates_flow_and_stores_only_hashed_token(client, github, db):
    login(client, github)
    r = client.get("/integrations/github/install", follow_redirects=False)
    assert r.status_code == 302
    assert r.headers["location"] == "https://github.com/apps/jarvis-incident-app/installations/new"

    cookie_val = client.cookies.get(oauth_service.INSTALL_FLOW_COOKIE)
    assert cookie_val is not None
    raw_token = cookie_val.split(".")[0]

    rows = db.execute(
        "select state_hash, stage, installation_id from public.github_app_pending_installations"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["stage"] == "install"
    assert rows[0]["installation_id"] is None
    assert rows[0]["state_hash"] == oauth_service.hash_state_token(raw_token)
    assert raw_token not in rows[0]["state_hash"]


def test_setup_requires_auth_and_validates_installation_id(client, github):
    assert client.get("/integrations/github/setup?installation_id=123").status_code == 401
    login(client, github)
    assert client.get("/integrations/github/setup").status_code == 400
    assert client.get("/integrations/github/setup?installation_id=abc").status_code == 400
    assert client.get("/integrations/github/setup?installation_id=0").status_code == 400
    assert (
        client.get(
            "/integrations/github/setup?installation_id=123&setup_action=invalid"
        ).status_code
        == 400
    )


def test_full_install_setup_callback_links_installation(client, github, mock_gh_app, db):
    login(client, github)
    client.get("/integrations/github/install", follow_redirects=False)

    setup = client.get(
        "/integrations/github/setup",
        params={"installation_id": "55501", "setup_action": "install"},
        follow_redirects=False,
    )
    assert setup.status_code == 302
    loc = urlsplit(setup.headers["location"])
    assert loc.netloc == "github.com" and loc.path == "/login/oauth/authorize"
    q = parse_qs(loc.query)
    assert q["client_id"] == ["Iv1.testappclientid"]
    state = q["state"][0]

    cb = client.get(
        "/integrations/github/callback",
        params={"code": "valid-code", "state": state},
        follow_redirects=False,
    )
    assert cb.status_code == 302
    cb_loc = urlsplit(cb.headers["location"])
    cb_q = parse_qs(cb_loc.query)
    assert f"{cb_loc.scheme}://{cb_loc.netloc}" == FRONTEND
    assert cb_q == {"github_app": ["connected"], "installation_id": ["55501"]}

    status = client.get("/integrations/github/status")
    assert status.status_code == 200
    body = status.json()
    assert body["connected"] is True
    assert len(body["installations"]) == 1
    assert body["installations"][0]["installation_id"] == 55501
    assert body["installations"][0]["account_login"] == "octo-test"
    assert body["installations"][0]["permissions"] == {"contents": "read", "metadata": "read"}


def test_replayed_state_is_rejected(client, github, mock_gh_app):
    login(client, github)
    setup = client.get(
        "/integrations/github/setup",
        params={"installation_id": "55501"},
        follow_redirects=False,
    )
    state = parse_qs(urlsplit(setup.headers["location"]).query)["state"][0]
    saved_cookie = client.cookies.get(oauth_service.OAUTH_STATE_COOKIE)

    first = client.get(
        "/integrations/github/callback",
        params={"code": "c", "state": state},
        follow_redirects=False,
    )
    assert parse_qs(urlsplit(first.headers["location"]).query)["github_app"] == ["connected"]

    # Attempt to replay the same state with the saved cookie.
    client.cookies.set(
        oauth_service.OAUTH_STATE_COOKIE, saved_cookie, path=oauth_service.COOKIE_PATH
    )
    second = client.get(
        "/integrations/github/callback",
        params={"code": "c", "state": state},
        follow_redirects=False,
    )
    q = parse_qs(urlsplit(second.headers["location"]).query)
    assert q["github_app"] == ["error"]
    assert q["reason"] == ["replayed_state"]


def test_expired_state_is_rejected(client, github, mock_gh_app, db):
    login(client, github)
    setup = client.get(
        "/integrations/github/setup",
        params={"installation_id": "55501"},
        follow_redirects=False,
    )
    state = parse_qs(urlsplit(setup.headers["location"]).query)["state"][0]
    db.execute(
        "update public.github_app_pending_installations set expires_at = now() - interval '1 minute'"
    )

    cb = client.get(
        "/integrations/github/callback",
        params={"code": "c", "state": state},
        follow_redirects=False,
    )
    q = parse_qs(urlsplit(cb.headers["location"]).query)
    assert q == {"github_app": ["error"], "reason": ["expired_state"]}
    assert client.get("/integrations/github/status").json()["connected"] is False


def test_identity_mismatch_is_rejected(client, github, mock_gh_app):
    login(client, github)
    setup = client.get(
        "/integrations/github/setup",
        params={"installation_id": "55501"},
        follow_redirects=False,
    )
    state = parse_qs(urlsplit(setup.headers["location"]).query)["state"][0]

    # GitHub OAuth returns a different GitHub account than the logged-in J.A.R.V.I.S. user.
    mock_gh_app.user = oauth_service.GitHubAppUser(
        github_id=github.profile.github_id + 999, login="impostor"
    )
    cb = client.get(
        "/integrations/github/callback",
        params={"code": "c", "state": state},
        follow_redirects=False,
    )
    q = parse_qs(urlsplit(cb.headers["location"]).query)
    assert q == {"github_app": ["error"], "reason": ["identity_mismatch"]}
    assert client.get("/integrations/github/status").json()["connected"] is False


def test_unowned_or_inaccessible_installation_is_rejected(client, github, mock_gh_app):
    login(client, github)
    # User passes an installation_id in the URL that is not in their /user/installations.
    setup = client.get(
        "/integrations/github/setup",
        params={"installation_id": "999999"},
        follow_redirects=False,
    )
    state = parse_qs(urlsplit(setup.headers["location"]).query)["state"][0]

    cb = client.get(
        "/integrations/github/callback",
        params={"code": "c", "state": state},
        follow_redirects=False,
    )
    q = parse_qs(urlsplit(cb.headers["location"]).query)
    assert q == {"github_app": ["error"], "reason": ["installation_not_accessible"]}
    assert client.get("/integrations/github/status").json()["connected"] is False


def test_suspended_installation_is_rejected(client, github, mock_gh_app):
    login(client, github)
    mock_gh_app.installations[0]["suspended_at"] = "2026-11-09T10:00:00Z"
    setup = client.get(
        "/integrations/github/setup",
        params={"installation_id": "55501"},
        follow_redirects=False,
    )
    state = parse_qs(urlsplit(setup.headers["location"]).query)["state"][0]
    cb = client.get(
        "/integrations/github/callback",
        params={"code": "c", "state": state},
        follow_redirects=False,
    )
    q = parse_qs(urlsplit(cb.headers["location"]).query)
    assert q == {"github_app": ["error"], "reason": ["installation_suspended"]}


def test_duplicate_installation_for_same_user_updates_idempotently(
    client, github, mock_gh_app, db
):
    login(client, github)
    for sel in ("selected", "all"):
        mock_gh_app.installations[0]["repository_selection"] = sel
        setup = client.get(
            "/integrations/github/setup",
            params={"installation_id": "55501", "setup_action": "update"},
            follow_redirects=False,
        )
        state = parse_qs(urlsplit(setup.headers["location"]).query)["state"][0]
        cb = client.get(
            "/integrations/github/callback",
            params={"code": "c", "state": state},
            follow_redirects=False,
        )
        assert parse_qs(urlsplit(cb.headers["location"]).query)["github_app"] == ["connected"]

    status = client.get("/integrations/github/status").json()
    assert len(status["installations"]) == 1
    assert status["installations"][0]["repository_selection"] == "all"


def test_cross_user_installation_conflict_is_rejected(client, github, mock_gh_app, db):
    # Another J.A.R.V.I.S. user already owns installation 55501.
    other_user = repositories.upsert_user(
        db, github_id=777111222, username="other-owner", email=None, avatar_url=None
    )
    repositories.upsert_github_app_installation(
        db,
        user_id=other_user["id"],
        installation_id=55501,
        github_account_id=888,
        github_account_login="shared-org",
        github_account_type="Organization",
        repository_selection="all",
        permissions={"contents": "read"},
    )

    login(client, github)
    mock_gh_app.installations[0]["account"] = {
        "id": 888,
        "login": "shared-org",
        "type": "Organization",
    }
    setup = client.get(
        "/integrations/github/setup",
        params={"installation_id": "55501"},
        follow_redirects=False,
    )
    state = parse_qs(urlsplit(setup.headers["location"]).query)["state"][0]
    cb = client.get(
        "/integrations/github/callback",
        params={"code": "c", "state": state},
        follow_redirects=False,
    )
    q = parse_qs(urlsplit(cb.headers["location"]).query)
    assert q == {"github_app": ["error"], "reason": ["installation_already_linked"]}

    # Ownership must remain with other_user.
    owner = db.execute(
        "select user_id from public.github_app_installations where installation_id = 55501"
    ).fetchone()["user_id"]
    assert owner == other_user["id"]


def test_disconnect_enforces_csrf_and_ownership(client, github, mock_gh_app, db):
    other_user = repositories.upsert_user(
        db, github_id=333444555, username="other-user", email=None, avatar_url=None
    )
    repositories.upsert_github_app_installation(
        db,
        user_id=other_user["id"],
        installation_id=88801,
        github_account_id=333444555,
        github_account_login="other-user",
        github_account_type="User",
        repository_selection="all",
        permissions={},
    )

    login(client, github)
    setup = client.get(
        "/integrations/github/setup",
        params={"installation_id": "55501"},
        follow_redirects=False,
    )
    state = parse_qs(urlsplit(setup.headers["location"]).query)["state"][0]
    client.get(
        "/integrations/github/callback",
        params={"code": "c", "state": state},
        follow_redirects=False,
    )

    # Missing Origin header -> 403 CSRF rejection.
    assert client.delete("/integrations/github/installations/55501").status_code == 403

    # Cannot disconnect another user's installation -> 404, and their row stays intact.
    assert (
        client.delete("/integrations/github/installations/88801", headers=csrf()).status_code
        == 404
    )
    assert (
        repositories.get_github_app_installation_for_user(db, other_user["id"], 88801)
        is not None
    )

    # Disconnecting own installation succeeds.
    r = client.delete("/integrations/github/installations/55501", headers=csrf())
    assert r.status_code == 200
    assert r.json()["installation_id"] == 55501
    assert "GitHub" in r.json()["message"]
    assert client.get("/integrations/github/status").json()["connected"] is False


def test_transaction_rollback_on_db_error_keeps_state_consumed(
    client, github, mock_gh_app, db, monkeypatch
):
    login(client, github)
    setup = client.get(
        "/integrations/github/setup",
        params={"installation_id": "55501"},
        follow_redirects=False,
    )
    state = parse_qs(urlsplit(setup.headers["location"]).query)["state"][0]
    saved_cookie = client.cookies.get(oauth_service.OAUTH_STATE_COOKIE)

    def broken_upsert(*a, **k):
        raise RuntimeError("simulated DB failure")

    monkeypatch.setattr(repositories, "upsert_github_app_installation", broken_upsert)
    r = client.get(
        "/integrations/github/callback",
        params={"code": "c", "state": state},
        follow_redirects=False,
    )
    assert r.status_code == 500
    monkeypatch.undo()

    # The state was already consumed before the failure, so replay is still rejected.
    client.cookies.set(
        oauth_service.OAUTH_STATE_COOKIE, saved_cookie, path=oauth_service.COOKIE_PATH
    )
    replay = client.get(
        "/integrations/github/callback",
        params={"code": "c", "state": state},
        follow_redirects=False,
    )
    q = parse_qs(urlsplit(replay.headers["location"]).query)
    assert q == {"github_app": ["error"], "reason": ["replayed_state"]}

