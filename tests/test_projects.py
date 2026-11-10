"""Project creation/management APIs, per-user project IDs and the migration itself."""

import uuid

import pytest

from tests.conftest import MIGRATIONS, csrf, login, make_user
from tests.telemetry_helpers import Account


_GITHUB_IDS = {}


def _login_as(client, github, username):
    # A username keeps the same GitHub identity across logins within one test run.
    github_id = _GITHUB_IDS.setdefault(username, uuid.uuid4().int % 10**9 + 7_000_000_000)
    github.set(github_id=github_id, username=username)
    client.cookies.clear()
    assert login(client, github).status_code in (200, 302)
    return client.get("/auth/me").json()


def _create(client, project_id, name=None, **extra):
    body = {"project_id": project_id, **({"name": name} if name is not None else {}), **extra}
    return client.post("/projects", json=body, headers=csrf())


# --- creation ----------------------------------------------------------------


def test_create_project_returns_project_owned_by_session_user(client, github, db):
    me = _login_as(client, github, "alice")
    r = _create(client, "my-platform", "My Platform")
    assert r.status_code == 201
    body = r.json()
    assert body["project_id"] == "my-platform" and body["name"] == "My Platform"
    uuid.UUID(body["id"])  # internal UUID is returned and distinct from the external ID
    row = db.execute("select user_id from public.projects where id = %s", (body["id"],)).fetchone()
    assert str(row["user_id"]) == me["id"]


def test_name_defaults_to_project_id_and_no_repository_mapping_is_created(client, github, db):
    _login_as(client, github, "alice")
    body = _create(client, "api-service").json()
    assert body["name"] == "api-service"
    assert db.execute("select count(*) as n from public.project_repositories where project_id = %s", (body["id"],)).fetchone()["n"] == 0


def test_client_cannot_supply_user_or_internal_ids(client, github, db):
    _login_as(client, github, "alice")
    other = make_user(db)
    for extra in ({"user_id": str(other["id"])}, {"id": str(uuid.uuid4())}, {"repository": "https://github.com/a/b"}):
        assert _create(client, "proj-extra", **extra).status_code == 422
    assert client.get("/projects").json()["projects"] == []


def test_create_requires_authentication_and_trusted_origin(client, github):
    assert client.post("/projects", json={"project_id": "abc-123"}, headers=csrf()).status_code == 401
    _login_as(client, github, "alice")
    assert client.post("/projects", json={"project_id": "abc-123"}).status_code == 403
    assert client.post("/projects", json={"project_id": "abc-123"}, headers={"Origin": "https://evil.example"}).status_code == 403


@pytest.mark.parametrize("bad", ["ab", "a" * 65, "-leading", "has space", "semi;colon", "slash/x", "ünï", ""])
def test_invalid_project_ids_rejected(client, github, bad):
    _login_as(client, github, "alice")
    r = _create(client, bad)
    assert r.status_code == 422, bad


@pytest.mark.parametrize("name", ["", "   ", "x" * 101, "tab\there"])
def test_invalid_names_rejected(client, github, name):
    _login_as(client, github, "alice")
    assert _create(client, "valid-id", name).status_code == 422


def test_boundary_project_ids_accepted(client, github):
    _login_as(client, github, "alice")
    assert _create(client, "abc").status_code == 201
    assert _create(client, "a" * 64).status_code == 201
    assert _create(client, "My.Proj_1-x").status_code == 201


# --- per-user uniqueness -----------------------------------------------------


def test_duplicate_project_id_rejected_for_same_user(client, github):
    _login_as(client, github, "alice")
    assert _create(client, "dup-project").status_code == 201
    r = _create(client, "dup-project")
    assert r.status_code == 409 and "already" in r.json()["detail"]


def test_same_project_id_allowed_for_different_users(client, github, db):
    _login_as(client, github, "alice")
    assert _create(client, "shared-name").status_code == 201
    _login_as(client, github, "bob")
    assert _create(client, "shared-name").status_code == 201
    assert db.execute("select count(*) as n from public.projects where project_id = 'shared-name'").fetchone()["n"] == 2


def test_failed_create_is_rolled_back_cleanly(client, github, db, monkeypatch):
    _login_as(client, github, "alice")
    from app import repositories

    def boom(*a, **k):
        raise RuntimeError("db down")

    monkeypatch.setattr(repositories, "insert_project", boom)
    assert _create(client, "will-fail").status_code == 500
    monkeypatch.undo()
    assert db.execute("select count(*) as n from public.projects where project_id = 'will-fail'").fetchone()["n"] == 0


# --- retrieval, update and isolation ------------------------------------------


def test_list_and_get_only_own_projects(client, github):
    _login_as(client, github, "alice")
    mine = _create(client, "alice-proj").json()
    _login_as(client, github, "bob")
    theirs = _create(client, "bob-proj").json()
    listed = client.get("/projects").json()["projects"]
    assert [p["project_id"] for p in listed] == ["bob-proj"]
    assert client.get(f"/projects/{theirs['id']}").status_code == 200
    assert client.get(f"/projects/{mine['id']}").status_code == 404  # exists, but not bob's
    assert client.get(f"/projects/{uuid.uuid4()}").status_code == 404
    assert client.get("/projects/not-a-uuid").status_code == 422


def test_rename_project_and_cross_user_denial(client, github):
    _login_as(client, github, "alice")
    mine = _create(client, "rename-me", "Old").json()
    r = client.patch(f"/projects/{mine['id']}", json={"name": "New Name"}, headers=csrf())
    assert r.status_code == 200 and r.json()["name"] == "New Name" and r.json()["project_id"] == "rename-me"
    assert client.patch(f"/projects/{mine['id']}", json={"name": "x"}).status_code == 403  # origin required
    assert client.patch(f"/projects/{mine['id']}", json={"name": ""}, headers=csrf()).status_code == 422
    assert client.patch(f"/projects/{mine['id']}", json={"project_id": "hijack", "name": "x"}, headers=csrf()).status_code == 422
    _login_as(client, github, "bob")
    assert client.patch(f"/projects/{mine['id']}", json={"name": "pwned"}, headers=csrf()).status_code == 404
    _login_as(client, github, "alice")
    assert client.get(f"/projects/{mine['id']}").json()["name"] == "New Name"


def test_project_endpoints_require_authentication(client):
    assert client.get("/projects").status_code == 401
    assert client.get(f"/projects/{uuid.uuid4()}").status_code == 401
    assert client.get(f"/projects/{uuid.uuid4()}/incidents").status_code == 401
    assert client.get(f"/projects/{uuid.uuid4()}/repositories").status_code == 401


# --- the migration -----------------------------------------------------------


def _constraints(db, table):
    rows = db.execute(
        "select conname, pg_get_constraintdef(oid) as definition from pg_constraint "
        "where conrelid = %s::regclass", (f"public.{table}",)).fetchall()
    return {r["conname"]: r["definition"] for r in rows}


def test_migration_replaces_global_uniqueness_with_per_user(db):
    constraints = _constraints(db, "projects")
    assert "projects_project_id_key" not in constraints
    assert constraints["projects_user_id_project_id_key"] == "UNIQUE (user_id, project_id)"
    assert "projects_pkey" in constraints  # the internal UUID primary key is preserved


def test_migration_is_idempotent_and_non_destructive(db):
    a = Account(db)
    a.ingest([a.event()])
    before = db.execute("select count(*) as n from public.telemetry_events").fetchone()["n"]
    for _ in range(2):
        for sql in MIGRATIONS:
            db.execute(sql)
    assert db.execute("select count(*) as n from public.telemetry_events").fetchone()["n"] == before
    assert db.execute("select count(*) as n from public.projects where id = %s", (a.project_uuid,)).fetchone()["n"] == 1


def test_new_tables_have_rls_enabled_and_expected_constraints(db):
    for table in ("telemetry_events", "project_repositories", "incident_history"):
        row = db.execute("select relrowsecurity from pg_class where oid = %s::regclass", (f"public.{table}",)).fetchone()
        assert row["relrowsecurity"] is True, table
    assert "UNIQUE (project_id, event_id)" in _constraints(db, "telemetry_events").values()
    assert "UNIQUE (project_id, service_id, repository_url)" in _constraints(db, "project_repositories").values()
    assert "incidents_status_check" in _constraints(db, "incidents")
