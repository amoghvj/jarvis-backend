"""Startup events, repository discovery and repository verification."""

import json
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from app import repositories_telemetry as repo
from app.integrations.github import client as app_tokens
from app.services import repository_discovery as rd
from app.tools.github import api
from tests.conftest import csrf, login
from tests.fake_github_api import FakeGitHubAPI, TokenSource
from tests.telemetry_helpers import Account, add_verified_repository, link_installation, startup_event

TOKEN = "ghs_installation_token_AAAA1111BBBB2222"


@pytest.fixture
def gh(monkeypatch):
    """Mocked GitHub: acme/webapp and acme/lib are installation repos; nothing else is."""
    fake = FakeGitHubAPI()
    monkeypatch.setattr(api, "_TRANSPORT", fake.transport)
    tokens = TokenSource([TOKEN])
    monkeypatch.setattr(app_tokens, "get_installation_access_token", tokens)
    fake.tokens = tokens
    return fake


def _startup(a: Account, **kw):
    """Ingest a startup event the way the route does (persist, then post-commit discovery)."""
    ingested = a.ingest([startup_event(a.external, **kw)])
    rd.process_startup_events(a.db, a.user["id"], ingested.new_startup_events)
    return ingested


def _mappings(db, a):
    return db.execute(
        "select * from public.project_repositories where project_id = %s order by service_id, repository_url",
        (a.project_uuid,),
    ).fetchall()


# --- URL parsing (untrusted input) ---------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("https://github.com/acme/webapp", ("acme", "webapp")),
        ("https://github.com/acme/webapp.git", ("acme", "webapp")),
        ("https://github.com/acme/webapp/", ("acme", "webapp")),
        ("https://www.github.com/acme/web.app-x_1", ("acme", "web.app-x_1")),
        ("https://github.com/acme/webapp/tree/main/src", ("acme", "webapp")),
        ("git@github.com:acme/webapp.git", ("acme", "webapp")),
        ("ssh://git@github.com/acme/webapp.git", ("acme", "webapp")),
        ("  https://github.com/acme/webapp  ", ("acme", "webapp")),
        ("https://ghp_abcdefghijklmnop123456@github.com/acme/webapp", ("acme", "webapp")),  # userinfo dropped
    ],
)
def test_parse_accepts_github_repository_urls(raw, expected):
    owner, repo_name, canonical = rd.parse_repository_url(raw)
    assert (owner, repo_name) == expected and canonical == f"https://github.com/{owner}/{repo_name}"
    assert "ghp_" not in canonical and "@" not in canonical


@pytest.mark.parametrize(
    "raw",
    [
        None, "", "   ", 42, "http://github.com/acme/webapp", "https://gitlab.com/acme/webapp",
        "https://github.com.evil.example/acme/webapp", "https://evil.example/github.com/acme/webapp",
        "https://github.com/acme", "https://github.com/", "https://github.com/ac me/webapp",
        "https://github.com/acme/..", "https://github.com/-bad-/webapp", "https://github.com/acme/web\napp",
        "file:///etc/passwd", "javascript:alert(1)", "https://github.com/" + "a" * 400 + "/x",
        "git@gitlab.com:acme/webapp.git",
    ],
)
def test_parse_rejects_everything_else(raw):
    assert rd.parse_repository_url(raw) is None


# --- startup events ---------------------------------------------------------------------------------


def test_startup_event_is_persisted_and_registers_an_unverified_repository(db, gh):
    a = Account(db)
    _startup(a, repository="https://github.com/acme/webapp", service="backend")
    (event,) = a.events()
    assert event["event_type"] == "startup" and event["classification"] == "normal_behaviour"
    assert event["runtime"]["language"] == "python" and event["instance_id"] == "host-1"
    assert event["application"]["repository"] == "https://github.com/acme/webapp"
    assert a.incidents() == []
    (m,) = _mappings(db, a)
    assert (m["service_id"], m["owner"], m["repo"]) == ("backend", "acme", "webapp")
    assert m["repository_url"] == "https://github.com/acme/webapp" and m["discovery_source"] == "startup_event"
    assert m["verification_status"] == "unverified" and m["installation_id"] is None  # nothing invented
    assert "No active GitHub App installation" in m["verification_note"]
    assert gh.requests == [] and gh.tokens.calls == []  # no installation, so no GitHub call or token


def test_each_startup_refreshes_the_mapping_without_duplicating_it(db, gh):
    a = Account(db)
    _startup(a, repository="https://github.com/acme/webapp")
    first = _mappings(db, a)[0]
    _startup(a, repository="https://github.com/acme/webapp.git")  # a new init(): same repo, other spelling
    rows = _mappings(db, a)
    assert len(rows) == 1 and rows[0]["last_seen_at"] > first["last_seen_at"]
    assert rows[0]["first_seen_at"] == first["first_seen_at"]


def test_a_project_can_have_several_repositories_and_services(db, gh):
    a = Account(db)
    _startup(a, repository="https://github.com/acme/webapp", service="backend")
    _startup(a, repository="https://github.com/acme/lib", service="worker")
    _startup(a, repository="https://github.com/acme/lib", service="backend")
    rows = {(m["service_id"], m["repo"]) for m in _mappings(db, a)}
    assert rows == {("backend", "webapp"), ("worker", "lib"), ("backend", "lib")}


def test_startup_without_repository_or_with_unusable_url(db, gh):
    a = Account(db)
    _startup(a)  # no repository configured
    assert _mappings(db, a) == []
    _startup(a, repository="https://user:hunter2@gitlab.example/team/proj.git", service="other")
    (m,) = _mappings(db, a)
    assert m["verification_status"] == "unsupported_url" and m["owner"] is None and m["installation_id"] is None
    assert "hunter2" not in json.dumps(dict(m), default=str) and "@" not in m["repository_url"]
    assert gh.requests == []


def test_startup_events_never_create_incidents_or_error_counts(db, gh):
    a = Account(db)
    _startup(a, repository="https://github.com/acme/webapp", severity="ERROR")
    assert a.incidents() == []


# --- verification ------------------------------------------------------------------------------------


def test_repository_in_an_owned_installation_is_verified(db, gh):
    a = Account(db)
    iid = link_installation(db, a.user["id"])
    _startup(a, repository="https://github.com/acme/webapp.git")
    (m,) = _mappings(db, a)
    assert m["verification_status"] == "verified" and m["installation_id"] == iid
    assert (m["owner"], m["repo"], m["default_branch"]) == ("acme", "webapp", "main")
    assert m["verified_at"] is not None and m["verification_note"] is None
    assert {r["method"] for r in gh.requests} == {"GET"} and "/installation/repositories" in gh.paths()


def test_ssh_style_url_is_verified_too(db, gh):
    a = Account(db)
    link_installation(db, a.user["id"])
    _startup(a, repository="git@github.com:acme/lib.git")
    assert _mappings(db, a)[0]["verification_status"] == "verified"


def test_repository_outside_every_installation_stays_unverified(db, gh):
    a = Account(db)
    link_installation(db, a.user["id"])
    _startup(a, repository="https://github.com/octocat/public-repo")  # public, but not in the installation
    (m,) = _mappings(db, a)
    assert m["verification_status"] == "unverified" and m["installation_id"] is None
    assert "not found in any of your GitHub App installations" in m["verification_note"]


def test_another_users_installation_is_never_used(db, gh):
    a, b = Account(db), Account(db)
    link_installation(db, b.user["id"])  # B has an installation containing acme/webapp; A has none
    _startup(a, repository="https://github.com/acme/webapp")
    (m,) = _mappings(db, a)
    assert m["verification_status"] == "unverified" and m["installation_id"] is None
    assert gh.tokens.calls == [] and gh.requests == []


def test_suspended_installation_does_not_verify(db, gh):
    a = Account(db)
    link_installation(db, a.user["id"], suspended=True)
    _startup(a, repository="https://github.com/acme/webapp")
    assert _mappings(db, a)[0]["verification_status"] == "unverified" and gh.requests == []


def test_installation_id_comes_from_the_database_not_from_telemetry(db, gh):
    a = Account(db)
    iid = link_installation(db, a.user["id"])
    ev = startup_event(a.external, repository="https://github.com/acme/webapp")
    ev["application"]["installation_id"] = 999999
    ev["installation_id"] = 888888
    ingested = a.ingest([ev])
    rd.process_startup_events(db, a.user["id"], ingested.new_startup_events)
    assert _mappings(db, a)[0]["installation_id"] == iid


def test_github_failure_leaves_the_mapping_unverified_and_ingestion_intact(db, gh):
    a = Account(db)
    link_installation(db, a.user["id"])
    gh.overrides["/installation/repositories"] = (500, {}, {"message": "boom"})
    ingested = _startup(a, repository="https://github.com/acme/webapp")
    assert ingested.count("accepted") == 1 and len(a.events()) == 1
    (m,) = _mappings(db, a)
    assert m["verification_status"] == "unverified" and "could not be completed" in m["verification_note"]


def test_recently_checked_unverified_mapping_is_not_rechecked_on_every_startup(db, gh):
    a = Account(db)
    _startup(a, repository="https://github.com/acme/webapp")  # no installation yet
    link_installation(db, a.user["id"])
    _startup(a, repository="https://github.com/acme/webapp")  # checked < 5 minutes ago: skipped
    assert _mappings(db, a)[0]["verification_status"] == "unverified"
    db.execute("update public.project_repositories set last_verification_at = %s where project_id = %s",
               (datetime.now(timezone.utc) - timedelta(minutes=10), a.project_uuid))
    _startup(a, repository="https://github.com/acme/webapp")  # stale: retried and now verified
    assert _mappings(db, a)[0]["verification_status"] == "verified"


def test_verified_repositories_query_only_returns_still_valid_verified_rows(db, gh):
    a = Account(db)
    iid = link_installation(db, a.user["id"])
    add_verified_repository(db, a, iid, "acme", "webapp", "backend")
    db.execute(
        "insert into public.project_repositories (project_id, service_id, repository_url, owner, repo) "
        "values (%s, 'x', 'https://github.com/acme/lib', 'acme', 'lib')", (a.project_uuid,))
    assert [r["repo"] for r in repo.verified_repositories(db, a.project_uuid)] == ["webapp"]
    assert repo.count_unverified_repositories(db, a.project_uuid) == 1
    db.execute("update public.github_app_installations set suspended_at = now() where installation_id = %s", (iid,))
    assert repo.verified_repositories(db, a.project_uuid) == []  # suspension revokes it immediately
    db.execute("update public.github_app_installations set suspended_at = null where installation_id = %s", (iid,))
    other = Account(db)
    db.execute("update public.github_app_installations set user_id = %s where installation_id = %s", (other.user["id"], iid))
    assert repo.verified_repositories(db, a.project_uuid) == []  # installation no longer the owner's
    db.execute("delete from public.github_app_installations where installation_id = %s", (iid,))
    assert repo.verified_repositories(db, a.project_uuid) == []  # uninstalled


def test_deleting_an_installation_still_works_and_demotes_its_verified_mappings(db, gh):
    a = Account(db)
    iid = link_installation(db, a.user["id"])
    add_verified_repository(db, a, iid, "acme", "webapp")
    assert repo.list_project_repositories(db, a.project_uuid)[0]["verification_status"] == "verified"
    db.execute("delete from public.github_app_installations where installation_id = %s", (iid,))  # the disconnect flow
    (listed,) = repo.list_project_repositories(db, a.project_uuid)
    assert listed["verification_status"] == "unverified" and listed["installation_id"] is None
    assert "no longer available" in listed["verification_note"]
    assert repo.verified_repositories(db, a.project_uuid) == []
    # A later startup event refreshes the stored status too, and re-verification can succeed again.
    new_iid = link_installation(db, a.user["id"])
    _startup(a, repository="https://github.com/acme/webapp")
    (row,) = _mappings(db, a)
    assert row["verification_status"] == "verified" and row["installation_id"] == new_iid


def test_suspended_installation_is_reported_unverified_in_listings(db, gh):
    a = Account(db)
    iid = link_installation(db, a.user["id"])
    add_verified_repository(db, a, iid, "acme", "webapp")
    db.execute("update public.github_app_installations set suspended_at = now() where installation_id = %s", (iid,))
    assert repo.list_project_repositories(db, a.project_uuid)[0]["verification_status"] == "unverified"

# --- HTTP endpoints ------------------------------------------------------------------------------------


def _owner_session(client, github, db):
    github.set(github_id=uuid.uuid4().int % 10**9 + 7_000_000_000, username="owner")
    client.cookies.clear()
    login(client, github)
    me = client.get("/auth/me").json()
    project = client.post("/projects", json={"project_id": "repo-proj"}, headers=csrf()).json()
    return uuid.UUID(me["id"]), project


def test_startup_over_http_then_verify_endpoint_after_installing_the_app(client, github, db, gh):
    user_id, project = _owner_session(client, github, db)
    key = __import__("tests.conftest", fromlist=["make_key"]).make_key(db, user_id)
    event = startup_event("repo-proj", repository="https://github.com/acme/webapp")
    r = client.post("/v1/events", json={"events": [event]}, headers={"Authorization": f"Bearer {key}"})
    assert r.status_code == 200
    listed = client.get(f"/projects/{project['id']}/repositories").json()["repositories"]
    assert len(listed) == 1 and listed[0]["verification_status"] == "unverified" and listed[0]["installation_id"] is None
    assert listed[0]["service_id"] == "backend" and listed[0]["repository_url"] == "https://github.com/acme/webapp"

    iid = link_installation(db, user_id)
    url = f"/projects/{project['id']}/repositories/verify"
    assert client.post(url).status_code == 403  # trusted origin required
    verified = client.post(url, headers=csrf()).json()["repositories"]
    assert verified[0]["verification_status"] == "verified" and verified[0]["installation_id"] == iid


def test_repository_endpoints_are_tenant_isolated(client, github, db, gh):
    user_id, project = _owner_session(client, github, db)
    github.set(github_id=uuid.uuid4().int % 10**9 + 7_000_000_000, username="intruder")
    client.cookies.clear()
    login(client, github)
    assert client.get(f"/projects/{project['id']}/repositories").status_code == 404
    assert client.post(f"/projects/{project['id']}/repositories/verify", headers=csrf()).status_code == 404
    assert user_id
