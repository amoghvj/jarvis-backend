"""Incident grouping, severity escalation, lifecycle and concurrency."""

import threading
import uuid
from datetime import timedelta

import psycopg
import pytest
from psycopg.rows import dict_row

from app.services import incident_service, project_service, telemetry_service
from tests.conftest import csrf, login
from tests.telemetry_helpers import BASE_TIME, Account, iso, make_event, startup_event


# --- fingerprinting ---------------------------------------------------------------


def test_fingerprint_ignores_unstable_fields_but_not_stable_ones():
    p = uuid.uuid4()
    base = incident_service.compute_fingerprint(p, "svc", "prod", "KeyError", "boom", "a.py", "f")
    assert base == incident_service.compute_fingerprint(p, "svc", "prod", "KeyError", "boom", "a.py", "f")
    for variant in (
        (uuid.uuid4(), "svc", "prod", "KeyError", "boom", "a.py", "f"),
        (p, "other", "prod", "KeyError", "boom", "a.py", "f"),
        (p, "svc", "dev", "KeyError", "boom", "a.py", "f"),
        (p, "svc", "prod", "ValueError", "boom", "a.py", "f"),
        (p, "svc", "prod", "KeyError", "bang", "a.py", "f"),
        (p, "svc", "prod", "KeyError", "boom", "b.py", "f"),
        (p, "svc", "prod", "KeyError", "boom", "a.py", "g"),
    ):
        assert incident_service.compute_fingerprint(*variant) != base


def test_message_normalization_is_light():
    n = incident_service.normalize_message
    assert n("  a   b\n c ") == "a b c"
    assert n("object at 0x7f12abc4d0 failed") == n("object at 0x7f99dd1100 failed")
    assert n("User 1 not found") != n("User 2 not found")  # numbers/IDs are NOT collapsed
    assert n("Timeout after 30s") != n("Timeout after 45s")


# --- grouping -----------------------------------------------------------------------


def test_error_creates_an_incident(db):
    a = Account(db)
    res = a.ingest([a.event(message="db down", error_type="ConnectionError")])
    (inc,) = a.incidents()
    assert res.results[0].incident_id == inc["id"]
    assert inc["title"] == "ConnectionError: db down" and inc["status"] == "open" and inc["severity"] == "ERROR"
    assert inc["event_count"] == 1 and len(inc["fingerprint"]) == 64
    assert (inc["service_id"], inc["environment"]) == ("backend", "production")
    assert inc["first_seen_at"] == inc["last_seen_at"]
    assert inc["incident_data"]["error_type"] == "ConnectionError"
    assert [h["change_type"] for h in a.history(inc["id"])] == ["created"]
    assert a.events()[0]["incident_id"] == inc["id"]


def test_same_fingerprint_within_window_groups_into_one_incident(db):
    a = Account(db)
    a.ingest([a.event(ts=iso(0), request="r1", trace="t1", instance="i1", line=10)])
    a.ingest([a.event(ts=iso(300), request="r2", trace="t2", instance="i2", line=99)])  # unstable fields differ
    (inc,) = a.incidents()
    assert inc["event_count"] == 2
    assert inc["last_seen_at"] - inc["first_seen_at"] == timedelta(seconds=300)
    assert {e["incident_id"] for e in a.events()} == {inc["id"]}


def test_stable_attribute_differences_make_separate_incidents(db):
    a = Account(db)
    a.ingest([
        a.event(message="m"), a.event(message="m", service="worker"), a.event(message="m", env="staging"),
        a.event(message="m", error_type="TimeoutError"), a.event(message="other"),
        a.event(message="m", file="app/other.py"), a.event(message="m", func="reconnect"),
    ])
    assert len(a.incidents()) == 7


def test_window_boundaries_and_configurable_window(db):
    a = Account(db)
    a.ingest([a.event(ts=iso(0))])
    a.ingest([a.event(ts=iso(14 * 60))])       # inside 15 minutes of the last event: same incident
    assert len(a.incidents()) == 1
    a.ingest([a.event(ts=iso(14 * 60 + 16 * 60))])  # 16 minutes after the last event: new incident
    assert len(a.incidents()) == 2
    b = Account(db)
    b.ingest([b.event(ts=iso(0))], window=60)
    b.ingest([b.event(ts=iso(90))], window=60)   # a 60s window separates events 90s apart
    assert len(b.incidents()) == 2
    c = Account(db)
    c.ingest([c.event(ts=iso(0))], window=3600)
    c.ingest([c.event(ts=iso(1800))], window=3600)
    assert len(c.incidents()) == 1


def test_out_of_order_timestamps_extend_the_incident_both_ways(db):
    a = Account(db)
    a.ingest([a.event(ts=iso(600))])
    a.ingest([a.event(ts=iso(0))])       # an older, buffered event arrives later
    (inc,) = a.incidents()
    assert inc["event_count"] == 2
    assert inc["first_seen_at"] == BASE_TIME and inc["last_seen_at"] == BASE_TIME + timedelta(seconds=600)


def test_batch_with_repeated_errors_counts_each_event_once(db):
    a = Account(db)
    a.ingest([a.event(ts=iso(i)) for i in range(5)])
    (inc,) = a.incidents()
    assert inc["event_count"] == 5 and len(a.events()) == 5


def test_incidents_are_per_project(db):
    a, b = Account(db), Account(db)
    a.ingest([a.event()])
    b.ingest([b.event()])
    assert a.incidents()[0]["id"] != b.incidents()[0]["id"]
    assert a.incidents()[0]["fingerprint"] != b.incidents()[0]["fingerprint"]


# --- eligibility and startup events ----------------------------------------------------


@pytest.mark.parametrize("kw", [
    {"severity": "INFO"}, {"severity": "WARNING"}, {"severity": "DEBUG"}, {"type": "log", "severity": "INFO"},
    {"severity": "ERROR", "classification": "normal_behaviour"},
])
def test_non_actionable_events_are_persisted_without_incidents(db, kw):
    a = Account(db)
    a.ingest([a.event(**kw)])
    assert len(a.events()) == 1 and a.incidents() == [] and a.events()[0]["incident_id"] is None


def test_startup_events_are_normal_telemetry_not_incidents(db):
    a = Account(db)
    a.ingest([a.event(message="real failure")])
    (inc,) = a.incidents()
    for i in range(3):
        a.ingest([startup_event(a.external, ts=iso(i * 10))])
    (after,) = a.incidents()
    assert after["event_count"] == 1 and after["severity"] == "ERROR" and after["id"] == inc["id"]
    assert len(a.history(inc["id"])) == 1  # no escalation, no timeline noise
    startups = [e for e in a.events() if e["event_type"] == "startup"]
    assert len(startups) == 3 and all(e["incident_id"] is None and e["classification"] == "normal_behaviour" for e in startups)


def test_startup_event_with_error_severity_still_never_creates_an_incident(db):
    a = Account(db)
    a.ingest([startup_event(a.external, severity="ERROR")])
    assert a.incidents() == []


def test_error_type_event_with_missing_severity_is_treated_as_error(db):
    a = Account(db)
    ev = a.event()
    ev["details"]["severity"] = None
    a.ingest([ev])
    (inc,) = a.incidents()
    assert inc["severity"] == "ERROR"


# --- severity escalation ----------------------------------------------------------------


def test_severity_only_escalates(db):
    a = Account(db)
    a.ingest([a.event(ts=iso(0), severity="ERROR")])
    a.ingest([a.event(ts=iso(10), severity="CRITICAL")])
    (inc,) = a.incidents()
    assert inc["severity"] == "CRITICAL"
    a.ingest([a.event(ts=iso(20), severity="ERROR")])
    (inc,) = a.incidents()
    assert inc["severity"] == "CRITICAL" and inc["event_count"] == 3  # never downgraded
    kinds = [(h["change_type"], h["from_severity"], h["to_severity"]) for h in a.history(inc["id"])]
    assert kinds == [("created", None, "ERROR"), ("severity_escalated", "ERROR", "CRITICAL")]


def test_severity_names_are_normalized(db):
    a = Account(db)
    a.ingest([a.event(severity="error"), a.event(severity="fatal", message="x")])
    sev = {i["title"]: i["severity"] for i in a.incidents()}
    assert sev["ConnectionError: connection refused"] == "ERROR" and sev["ConnectionError: x"] == "CRITICAL"


def test_escalation_inside_a_single_batch(db):
    a = Account(db)
    a.ingest([a.event(ts=iso(0), severity="ERROR"), a.event(ts=iso(1), severity="CRITICAL")])
    (inc,) = a.incidents()
    assert inc["severity"] == "CRITICAL" and inc["event_count"] == 2


# --- lifecycle ---------------------------------------------------------------------------


def _open_incident(db):
    a = Account(db)
    a.ingest([a.event()])
    return a, a.incidents()[0]


def test_lifecycle_transitions_are_recorded_with_actor_and_note(db):
    a, inc = _open_incident(db)
    acked = incident_service.change_status(db, a.project_uuid, inc["id"], "acknowledged", a.user["id"], "looking")
    assert acked["status"] == "acknowledged" and acked["acknowledged_at"] is not None and acked["resolved_at"] is None
    done = incident_service.change_status(db, a.project_uuid, inc["id"], "resolved", a.user["id"])
    assert done["status"] == "resolved" and done["resolved_at"] is not None
    reopened = incident_service.change_status(db, a.project_uuid, inc["id"], "open", a.user["id"], "regressed")
    assert reopened["status"] == "open" and reopened["resolved_at"] is None
    history = [(h["change_type"], h["from_status"], h["to_status"], h["actor_user_id"], h["note"]) for h in a.history(inc["id"])]
    assert history == [
        ("created", None, "open", None, None),
        ("status_changed", "open", "acknowledged", a.user["id"], "looking"),
        ("status_changed", "acknowledged", "resolved", a.user["id"], None),
        ("status_changed", "resolved", "open", a.user["id"], "regressed"),
    ]


@pytest.mark.parametrize(
    "setup,bad",
    [(["resolved"], "acknowledged"), (["acknowledged"], "acknowledged"), ([], "open"), ([], "bogus")],
)
def test_invalid_transitions_are_rejected_and_leave_no_trace(db, setup, bad):
    a, inc = _open_incident(db)
    for status in setup:
        incident_service.change_status(db, a.project_uuid, inc["id"], status, a.user["id"])
    expected = setup[-1] if setup else "open"
    with pytest.raises(incident_service.InvalidTransition):
        incident_service.change_status(db, a.project_uuid, inc["id"], bad, a.user["id"])
    assert db.execute("select status from public.incidents where id = %s", (inc["id"],)).fetchone()["status"] == expected
    assert len([h for h in a.history(inc["id"]) if h["change_type"] == "status_changed"]) == len(setup)


def test_transition_on_another_projects_incident_is_not_found(db):
    a, inc = _open_incident(db)
    b = Account(db)
    with pytest.raises(incident_service.IncidentNotFound):
        incident_service.change_status(db, b.project_uuid, inc["id"], "acknowledged", b.user["id"])
    assert db.execute("select status from public.incidents where id = %s", (inc["id"],)).fetchone()["status"] == "open"


def test_acknowledged_incident_keeps_grouping_but_resolved_one_does_not(db):
    a, inc = _open_incident(db)
    incident_service.change_status(db, a.project_uuid, inc["id"], "acknowledged", a.user["id"])
    a.ingest([a.event(ts=iso(60))])
    assert len(a.incidents()) == 1 and a.incidents()[0]["event_count"] == 2
    incident_service.change_status(db, a.project_uuid, inc["id"], "resolved", a.user["id"])
    a.ingest([a.event(ts=iso(120))])
    incidents = a.incidents()
    assert len(incidents) == 2 and incidents[0]["status"] == "resolved" and incidents[1]["status"] == "open"
    assert incidents[0]["event_count"] == 2  # the resolved incident did not absorb the new event


def test_incidents_are_never_resolved_automatically(db):
    a, inc = _open_incident(db)
    a.ingest([a.event(ts=iso(86400 * 30), message="a month later")])  # a month of silence, then more errors
    statuses = {i["id"]: i["status"] for i in a.incidents()}
    assert statuses[inc["id"]] == "open"  # inactivity never resolves anything
    assert len(statuses) == 2


# --- lifecycle and listing over HTTP --------------------------------------------------------


def _as_user(client, github, username="owner"):
    github.set(github_id=uuid.uuid4().int % 10**9 + 7_000_000_000, username=username)
    client.cookies.clear()
    login(client, github)
    return client.get("/auth/me").json()


def test_incident_api_list_status_and_isolation(client, github, db):
    me = _as_user(client, github)
    project = client.post("/projects", json={"project_id": "http-proj"}, headers=csrf()).json()
    pu = uuid.UUID(project["id"])
    valid, _ = telemetry_service.parse_events([make_event("http-proj", message=f"m{i}") for i in range(3)])
    telemetry_service.ingest_events(db, valid, {"http-proj": pu})
    listed = client.get(f"/projects/{project['id']}/incidents?limit=2").json()
    assert len(listed["incidents"]) == 2 and listed["has_more"] is True
    page2 = client.get(f"/projects/{project['id']}/incidents?limit=2&offset=2").json()
    assert len(page2["incidents"]) == 1 and page2["has_more"] is False
    assert client.get(f"/projects/{project['id']}/incidents?status=bogus").status_code == 422
    incident_id = listed["incidents"][0]["id"]
    url = f"/projects/{project['id']}/incidents/{incident_id}/status"
    assert client.post(url, json={"status": "acknowledged"}).status_code == 403  # trusted origin required
    ok = client.post(url, json={"status": "acknowledged", "note": "on it"}, headers=csrf())
    assert ok.status_code == 200 and ok.json()["status"] == "acknowledged"
    assert client.post(url, json={"status": "acknowledged"}, headers=csrf()).status_code == 409
    assert client.post(url, json={"status": "weird"}, headers=csrf()).status_code == 409
    assert client.get(f"/projects/{project['id']}/incidents?status=acknowledged").json()["incidents"][0]["id"] == incident_id
    # Another user can neither see nor change it.
    _as_user(client, github, "intruder")
    assert client.get(f"/projects/{project['id']}/incidents").status_code == 404
    assert client.post(url, json={"status": "resolved"}, headers=csrf()).status_code == 404
    assert me["id"]


# --- concurrency (needs committed data, so only on an isolated database) ---------------------


def _committed_account(url):
    with psycopg.connect(url, autocommit=True, row_factory=dict_row) as c:
        user = c.execute(
            "insert into public.users (github_id, github_username) values (%s, 'concurrent') returning id",
            (uuid.uuid4().int % 10**12 + 1,),
        ).fetchone()
        project = c.execute(
            "insert into public.projects (user_id, project_id, name) values (%s, %s, 'c') returning id, project_id",
            (user["id"], f"conc-{uuid.uuid4().hex[:8]}"),
        ).fetchone()
    return user["id"], project["id"], project["project_id"]


def _run_threads(url, work, count):
    barrier, errors = threading.Barrier(count), []

    def runner(i):
        try:
            with psycopg.connect(url, autocommit=True, row_factory=dict_row) as conn:
                barrier.wait(timeout=20)
                work(conn, i)
        except Exception as exc:  # collected and asserted by the caller
            errors.append(exc)

    threads = [threading.Thread(target=runner, args=(i,)) for i in range(count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    return errors


def test_concurrent_ingestion_of_the_same_error_yields_one_incident(test_database):
    url, throwaway = test_database
    if not throwaway:
        pytest.skip("concurrency test commits data and only runs on an isolated test database")
    user_id, project_uuid, external = _committed_account(url)
    workers = 8

    def work(conn, i):
        valid, rejected = telemetry_service.parse_events([make_event(external, event_id=f"c-{i}-{j}") for j in range(3)])
        telemetry_service.ingest_events(conn, valid, {external: project_uuid})

    try:
        assert _run_threads(url, work, workers) == []
        with psycopg.connect(url, row_factory=dict_row) as conn:
            incidents = conn.execute("select * from public.incidents where project_id = %s", (project_uuid,)).fetchall()
            events = conn.execute("select count(*) as n from public.telemetry_events where project_id = %s", (project_uuid,)).fetchone()["n"]
        assert len(incidents) == 1
        assert incidents[0]["event_count"] == workers * 3 == events  # no lost updates, no duplicates
    finally:
        with psycopg.connect(url, autocommit=True) as conn:
            conn.execute("delete from public.users where id = %s", (user_id,))


def test_concurrent_retries_of_one_event_count_it_once(test_database):
    url, throwaway = test_database
    if not throwaway:
        pytest.skip("concurrency test commits data and only runs on an isolated test database")
    user_id, project_uuid, external = _committed_account(url)

    def work(conn, i):
        valid, _ = telemetry_service.parse_events([make_event(external, event_id="the-one")])
        telemetry_service.ingest_events(conn, valid, {external: project_uuid})

    try:
        assert _run_threads(url, work, 8) == []
        with psycopg.connect(url, row_factory=dict_row) as conn:
            (incident,) = conn.execute("select * from public.incidents where project_id = %s", (project_uuid,)).fetchall()
            n = conn.execute("select count(*) as n from public.telemetry_events where project_id = %s", (project_uuid,)).fetchone()["n"]
        assert n == 1 and incident["event_count"] == 1
    finally:
        with psycopg.connect(url, autocommit=True) as conn:
            conn.execute("delete from public.users where id = %s", (user_id,))


def test_concurrent_batches_with_opposite_fingerprint_order_do_not_deadlock(test_database):
    url, throwaway = test_database
    if not throwaway:
        pytest.skip("concurrency test commits data and only runs on an isolated test database")
    user_id, project_uuid, external = _committed_account(url)

    def work(conn, i):
        for round_ in range(5):
            names = ["alpha", "beta", "gamma"]
            names = names if i % 2 == 0 else list(reversed(names))
            events = [make_event(external, event_id=f"d-{i}-{round_}-{n}", message=n) for n in names]
            valid, _ = telemetry_service.parse_events(events)
            telemetry_service.ingest_events(conn, valid, {external: project_uuid})

    try:
        assert _run_threads(url, work, 6) == []
        with psycopg.connect(url, row_factory=dict_row) as conn:
            incidents = conn.execute(
                "select event_count from public.incidents where project_id = %s order by 1", (project_uuid,)
            ).fetchall()
        assert [i["event_count"] for i in incidents] == [30, 30, 30]
    finally:
        with psycopg.connect(url, autocommit=True) as conn:
            conn.execute("delete from public.users where id = %s", (user_id,))


def test_project_service_is_used_for_account_setup(db):
    a = Account(db)
    assert project_service.get_project(db, a.user["id"], a.project_uuid)["project_id"] == a.external
