"""Migration 003 against a production-shaped database: upgrade with existing data, idempotency, atomicity.

Each test builds its own database on the throwaway local PostgreSQL server (never Supabase), so it can commit.
"""

import uuid
from contextlib import contextmanager

import psycopg
import pytest
from psycopg.rows import dict_row

from tests.conftest import BASE_SCHEMA, MIGRATIONS

M001, M002, M003 = MIGRATIONS
NEW_TABLES = ("telemetry_events", "project_repositories", "incident_history")
NEW_INCIDENT_COLUMNS = {"fingerprint", "service_id", "environment", "first_seen_at", "last_seen_at",
                        "event_count", "acknowledged_at", "resolved_at"}


@pytest.fixture
def make_db(test_database):
    """Create isolated databases on the throwaway server; they are dropped afterwards."""
    server_url, throwaway = test_database
    if not throwaway:
        pytest.skip("migration tests commit data and only run on an isolated test database")
    created = []

    @contextmanager
    def build(*scripts, seed=None):
        name = f"mig_{uuid.uuid4().hex[:10]}"
        with psycopg.connect(server_url, autocommit=True) as admin:
            admin.execute(f'create database "{name}"')
        created.append(name)
        url = server_url.rsplit("/", 1)[0] + "/" + name
        with psycopg.connect(url, autocommit=True, row_factory=dict_row) as conn:
            for script in scripts:
                conn.execute(script)
            if seed:
                conn.execute(seed)
            yield conn

    yield build
    with psycopg.connect(server_url, autocommit=True) as admin:
        for name in created:
            admin.execute(f'drop database if exists "{name}" with (force)')


PRE = (BASE_SCHEMA, M001, M002)  # the production schema before migration 003

SEED = """
insert into public.users (id, github_id, github_username) values
  ('00000000-0000-0000-0000-0000000000a1', 101, 'alice'), ('00000000-0000-0000-0000-0000000000b2', 202, 'bob');
insert into public.api_keys (user_id, key_hash, key_prefix) values
  ('00000000-0000-0000-0000-0000000000a1', repeat('a', 64), 'jarvis_live_abcd');
insert into public.projects (id, user_id, project_id, name, repository) values
  ('00000000-0000-0000-0000-00000000aa01', '00000000-0000-0000-0000-0000000000a1', 'alice-app', 'Alice App', 'https://github.com/a/a'),
  ('00000000-0000-0000-0000-00000000bb02', '00000000-0000-0000-0000-0000000000b2', 'bob-app', 'Bob App', null);
insert into public.incidents (id, project_id, title, status, severity, incident_data) values
  ('00000000-0000-0000-0000-0000000001a1', '00000000-0000-0000-0000-00000000aa01', 'Legacy open', 'open', 'ERROR', '{"k": 1}'),
  ('00000000-0000-0000-0000-0000000001a2', '00000000-0000-0000-0000-00000000aa01', 'Legacy resolved', 'resolved', null, '{}'),
  ('00000000-0000-0000-0000-0000000001a3', '00000000-0000-0000-0000-00000000bb02', 'Legacy odd status', 'closed', 'WARNING', '{}');
insert into public.github_app_installations (user_id, installation_id, github_account_id, github_account_login, github_account_type)
  values ('00000000-0000-0000-0000-0000000000a1', 4242, 1, 'alice', 'User');
insert into public.sessions (user_id, token_hash, expires_at)
  values ('00000000-0000-0000-0000-0000000000a1', repeat('b', 64), now() + interval '1 day');
"""

ORIGINAL_COLUMNS = {
    "users": "id, github_id, github_username, email, avatar_url, created_at, updated_at",
    "api_keys": "id, user_id, key_hash, key_prefix, created_at, last_used_at",
    "sessions": "id, user_id, token_hash, created_at, expires_at",
    "projects": "id, user_id, project_id, name, repository, created_at, updated_at",
    "incidents": "id, project_id, title, status, severity, incident_data, created_at, updated_at",
    "github_app_installations": "id, user_id, installation_id, github_account_id, github_account_login, "
                                "github_account_type, repository_selection, permissions, suspended_at, created_at, updated_at",
}


def _rows(conn):
    return {t: conn.execute(f"select {cols} from public.{t} order by 1").fetchall() for t, cols in ORIGINAL_COLUMNS.items()}


def _structure(conn):
    """Columns, constraints, indexes and RLS of every public table, for before/after comparison."""
    tables = [r["relname"] for r in conn.execute(
        "select relname from pg_class where relnamespace = 'public'::regnamespace and relkind = 'r' order by 1")]
    out = {}
    for t in tables:
        out[t] = {
            "columns": conn.execute(
                "select column_name, data_type, is_nullable, column_default from information_schema.columns "
                "where table_schema='public' and table_name=%s order by column_name", (t,)).fetchall(),
            "constraints": conn.execute(
                "select conname, pg_get_constraintdef(oid) as d from pg_constraint where conrelid = %s::regclass order by 1",
                (f"public.{t}",)).fetchall(),
            "indexes": conn.execute(
                "select indexname, indexdef from pg_indexes where schemaname='public' and tablename=%s order by 1", (t,)).fetchall(),
            "rls": conn.execute("select relrowsecurity from pg_class where oid = %s::regclass", (f"public.{t}",)).fetchone(),
        }
    return out


def _constraint_defs(conn, table):
    return {r["conname"]: r["d"] for r in conn.execute(
        "select conname, pg_get_constraintdef(oid) as d from pg_constraint where conrelid = %s::regclass",
        (f"public.{table}",))}


# --- upgrading a database that already has data --------------------------------------------------------------------


def test_upgrade_keeps_every_existing_row_and_column_value(make_db):
    with make_db(*PRE, seed=SEED) as conn:
        before = _rows(conn)
        assert [len(v) for v in before.values()] == [2, 1, 1, 2, 3, 1]
        conn.execute(M003)
        assert _rows(conn) == before  # nothing deleted, rewritten or invalidated
        legacy = conn.execute("select * from public.incidents order by title").fetchall()
        for row in legacy:  # the new columns are inert for old rows
            assert row["fingerprint"] is None and row["last_seen_at"] is None and row["event_count"] == 0
            assert row["acknowledged_at"] is None and row["resolved_at"] is None


def test_upgrade_adds_exactly_the_expected_objects_and_removes_only_the_old_unique(make_db):
    with make_db(*PRE, seed=SEED) as conn:
        before = _structure(conn)
        conn.execute(M003)
        after = _structure(conn)
        assert set(after) - set(before) == set(NEW_TABLES)
        for table, was in before.items():  # existing tables only gain things, apart from the one swapped constraint
            names_before = {c["conname"] for c in was["constraints"]}
            names_after = {c["conname"] for c in after[table]["constraints"]}
            removed = names_before - names_after
            assert removed == ({"projects_project_id_key"} if table == "projects" else set()), table
            assert {c["column_name"] for c in was["columns"]} <= {c["column_name"] for c in after[table]["columns"]}
            assert {i["indexname"] for i in was["indexes"]} - {"projects_project_id_key"} <= {i["indexname"] for i in after[table]["indexes"]}
            assert after[table]["rls"] == was["rls"]
        added = {c["column_name"] for c in after["incidents"]["columns"]} - {c["column_name"] for c in before["incidents"]["columns"]}
        assert added == NEW_INCIDENT_COLUMNS
        assert _constraint_defs(conn, "projects")["projects_user_id_project_id_key"] == "UNIQUE (user_id, project_id)"
        assert "projects_pkey" in _constraint_defs(conn, "projects")
        for table in NEW_TABLES:
            assert after[table]["rls"]["relrowsecurity"] is True, table


def test_rerunning_changes_nothing_and_is_repeatable(make_db):
    with make_db(*PRE, seed=SEED) as conn:
        conn.execute(M003)
        structure, rows = _structure(conn), _rows(conn)
        for _ in range(3):
            conn.execute(M003)
        assert _structure(conn) == structure and _rows(conn) == rows


def test_rerun_takes_no_exclusive_lock_on_existing_tables(make_db):
    """A repeat run must not queue behind (or block) readers: it takes no ACCESS EXCLUSIVE lock."""
    with make_db(*PRE, seed=SEED) as conn:
        conn.execute(M003)
        with psycopg.connect(conn.info.dsn.replace("password=", "password="), autocommit=False) as reader:
            reader.execute("lock table public.projects, public.incidents in access share mode")
            conn.execute("set lock_timeout = '1500ms'")
            conn.execute(M003)  # would raise LockNotAvailable if any ALTER TABLE were attempted
            reader.rollback()


def test_unmigrated_to_migrated_matches_a_fresh_build(make_db):
    """Upgrading in place yields the same structure as a database built straight to 003."""
    with make_db(*PRE) as upgraded, make_db(*PRE, M003) as fresh:
        upgraded.execute(M003)
        assert _structure(upgraded) == _structure(fresh)


# --- all-or-nothing and prerequisites ------------------------------------------------------------------------------


def test_failure_part_way_applies_nothing(make_db):
    """Run as one request, a failure late in the script rolls back the earlier steps too."""
    broken = "create table public.telemetry_events (id uuid primary key);"  # incompatible: no project_id column etc.
    with make_db(*PRE, broken) as conn:
        before = _structure(conn)
        with pytest.raises(psycopg.errors.Error):
            conn.execute(M003)
        after = _structure(conn)
        assert after == before  # the unique swap and the incident columns were rolled back
        assert "projects_project_id_key" in _constraint_defs(conn, "projects")
        assert "projects_user_id_project_id_key" not in _constraint_defs(conn, "projects")


def test_missing_prerequisites_fail_clearly_before_changing_anything(make_db):
    with make_db(BASE_SCHEMA, M001) as conn:  # migration 002 (github_app_installations) not applied
        before = _structure(conn)
        with pytest.raises(psycopg.errors.RaiseException, match="prerequisites are missing: public.github_app_installations"):
            conn.execute(M003)
        assert _structure(conn) == before


def test_installation_id_must_be_unique_prerequisite(make_db):
    non_unique = (BASE_SCHEMA, M001, "create table public.github_app_installations (installation_id bigint not null);")
    with make_db(*non_unique) as conn:
        with pytest.raises(psycopg.errors.RaiseException, match="installation_id must be UNIQUE"):
            conn.execute(M003)


# --- the unique-constraint swap ------------------------------------------------------------------------------------


def test_project_ids_become_unique_per_user_only(make_db):
    with make_db(*PRE, seed=SEED) as conn:
        conn.execute(M003)
        conn.execute("insert into public.projects (user_id, project_id, name) values "
                     "('00000000-0000-0000-0000-0000000000b2', 'alice-app', 'Bob reuses the id')")  # another user: allowed
        with pytest.raises(psycopg.errors.UniqueViolation):
            conn.execute("insert into public.projects (user_id, project_id, name) values "
                         "('00000000-0000-0000-0000-0000000000a1', 'alice-app', 'duplicate')")


def test_old_unique_constraint_is_found_by_definition_even_if_renamed(make_db):
    renamed = "alter table public.projects rename constraint projects_project_id_key to some_other_name;"
    with make_db(*PRE, renamed) as conn:
        conn.execute(M003)
        defs = _constraint_defs(conn, "projects")
        assert "some_other_name" not in defs and "UNIQUE (user_id, project_id)" in defs.values()


# --- existing incidents: compatible with the new code ---------------------------------------------------------------


def test_status_check_accepts_only_lifecycle_values_but_tolerates_legacy_rows(make_db):
    with make_db(*PRE, seed=SEED) as conn:
        conn.execute(M003)  # a legacy row with status 'closed' did not make the migration fail
        assert conn.execute("select status from public.incidents where title = 'Legacy odd status'").fetchone()["status"] == "closed"
        for status in ("open", "acknowledged", "resolved"):
            conn.execute("insert into public.incidents (project_id, title, status) values "
                         "('00000000-0000-0000-0000-00000000aa01', 'ok', %s)", (status,))
        for bad in ("closed", "OPEN", "", "done"):
            with pytest.raises(psycopg.errors.CheckViolation):
                conn.execute("insert into public.incidents (project_id, title, status) values "
                             "('00000000-0000-0000-0000-00000000aa01', 'bad', %s)", (bad,))
        # NOT VALID means the constraint is enforced for new/updated rows but legacy rows were never scanned.
        valid = conn.execute("select convalidated from pg_constraint where conname = 'incidents_status_check'").fetchone()
        assert valid["convalidated"] is False


def test_backend_works_on_a_migrated_database_with_legacy_incidents(make_db, monkeypatch):
    from app.services import incident_service, telemetry_service
    from tests.telemetry_helpers import make_event

    with make_db(*PRE, seed=SEED) as conn:
        conn.execute(M003)
        alice, project = "00000000-0000-0000-0000-0000000000a1", uuid.UUID("00000000-0000-0000-0000-00000000aa01")
        valid, rejected = telemetry_service.parse_events([make_event("alice-app", message="new failure")])
        assert not rejected
        with conn.transaction():
            telemetry_service.ingest_events(conn, valid, {"alice-app": project})
        incidents = conn.execute("select title, status, event_count from public.incidents where project_id = %s order by title",
                                 (project,)).fetchall()
        assert {i["title"] for i in incidents} == {"Legacy open", "Legacy resolved", "ConnectionError: new failure"}
        assert next(i for i in incidents if i["title"] == "Legacy open")["event_count"] == 0  # legacy rows are never absorbed
        legacy = uuid.UUID("00000000-0000-0000-0000-0000000001a1")
        updated = incident_service.change_status(conn, project, legacy, "acknowledged", uuid.UUID(alice))
        assert updated["status"] == "acknowledged" and updated["first_seen_at"] is None
        odd = uuid.UUID("00000000-0000-0000-0000-0000000001a3")
        bob_project = uuid.UUID("00000000-0000-0000-0000-00000000bb02")
        with pytest.raises(incident_service.InvalidTransition):
            incident_service.change_status(conn, bob_project, odd, "resolved", uuid.UUID(alice))  # no crash, clear refusal
        listed = conn.execute("select id from public.incidents where project_id = %s order by last_seen_at desc nulls last, id",
                              (project,)).fetchall()
        assert len(listed) == 3


# --- telemetry_events.classification CHECK -------------------------------------------------------------------------


EVENT_INSERT = """insert into public.telemetry_events (project_id, event_id, event_type, classification, event_timestamp, payload)
                  values (%s, %s, 'log', %s, now(), '{}'::jsonb)"""


@pytest.mark.parametrize("value", ["normal_behaviour", "abnormal_behaviour"])
def test_classification_check_accepts_the_two_canonical_values(db, value):
    from tests.telemetry_helpers import Account

    a = Account(db)
    db.execute(EVENT_INSERT, (a.project_uuid, "ok-" + value, value))


@pytest.mark.parametrize("value", ["NORMAL_BEHAVIOUR", "Abnormal_Behaviour", "error", "informational", "", " normal_behaviour",
                                   "normal_behaviour ", "normal", "abnormal", "unknown"])
def test_classification_check_rejects_everything_else(db, value):
    from tests.telemetry_helpers import Account

    a = Account(db)
    with pytest.raises(psycopg.errors.CheckViolation, match="telemetry_events_classification_check"):
        db.execute(EVENT_INSERT, (a.project_uuid, "bad", value))
    db.rollback()


def test_classification_is_still_not_null(db):
    from tests.telemetry_helpers import Account

    a = Account(db)
    with pytest.raises(psycopg.errors.NotNullViolation):
        db.execute(EVENT_INSERT, (a.project_uuid, "null-cls", None))
    db.rollback()


def test_check_is_added_to_a_table_created_by_an_earlier_draft(make_db):
    """A telemetry_events table without the check (CREATE TABLE IF NOT EXISTS skips it) gets it on rerun."""
    with make_db(*PRE, seed=SEED) as conn:
        conn.execute(M003)
        conn.execute("alter table public.telemetry_events drop constraint telemetry_events_classification_check")
        conn.execute(EVENT_INSERT, ("00000000-0000-0000-0000-00000000aa01", "legacy-1", "error"))  # old vocabulary
        conn.execute(M003)  # does not fail on the legacy row
        constraint = _constraint_defs(conn, "telemetry_events")["telemetry_events_classification_check"]
        assert "normal_behaviour" in constraint and "abnormal_behaviour" in constraint
        with pytest.raises(psycopg.errors.CheckViolation):
            conn.execute(EVENT_INSERT, ("00000000-0000-0000-0000-00000000aa01", "new-bad", "error"))
        conn.execute(EVENT_INSERT, ("00000000-0000-0000-0000-00000000aa01", "new-good", "abnormal_behaviour"))
        with pytest.raises(psycopg.errors.CheckViolation):  # validation reports the legacy row until it is fixed
            conn.execute("alter table public.telemetry_events validate constraint telemetry_events_classification_check")
        conn.execute("update public.telemetry_events set classification = 'abnormal_behaviour' where event_id = 'legacy-1'")
        conn.execute("alter table public.telemetry_events validate constraint telemetry_events_classification_check")


# --- foreign keys, cascades and behaviour of the existing flows -----------------------------------------------------


def test_foreign_keys_and_cascades(db):
    from tests.telemetry_helpers import Account

    a = Account(db)
    a.ingest([a.event()])
    assert a.events() and a.incidents()
    fks = {(r["conrelid"], r["confrelid"], r["confdeltype"]) for r in db.execute(
        "select conrelid::regclass::text as conrelid, confrelid::regclass::text as confrelid, confdeltype from pg_constraint "
        "where contype = 'f' and conrelid in ('public.telemetry_events'::regclass, 'public.project_repositories'::regclass, "
        "'public.incident_history'::regclass)")}
    assert fks == {
        ("telemetry_events", "projects", "c"), ("telemetry_events", "incidents", "n"),
        ("project_repositories", "projects", "c"), ("project_repositories", "github_app_installations", "n"),
        ("incident_history", "incidents", "c"), ("incident_history", "projects", "c"), ("incident_history", "users", "n"),
    }
    db.execute("delete from public.projects where id = %s", (a.project_uuid,))  # cascades through everything
    for table in NEW_TABLES:
        assert db.execute(f"select count(*) as n from public.{table} where project_id = %s", (a.project_uuid,)).fetchone()["n"] == 0
    assert db.execute("select count(*) as n from public.incidents where project_id = %s", (a.project_uuid,)).fetchone()["n"] == 0


def test_deleting_a_user_removes_their_telemetry_and_history_without_errors(db):
    from tests.telemetry_helpers import Account

    a = Account(db)
    a.ingest([a.event()])
    db.execute("delete from public.users where id = %s", (a.user["id"],))
    assert db.execute("select count(*) as n from public.telemetry_events where project_id = %s", (a.project_uuid,)).fetchone()["n"] == 0
    assert db.execute("select count(*) as n from public.incident_history where project_id = %s", (a.project_uuid,)).fetchone()["n"] == 0

