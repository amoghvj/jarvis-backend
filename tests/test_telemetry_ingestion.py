"""Persistent, idempotent telemetry ingestion through POST /v1/events."""

import json
from datetime import datetime, timezone

import pytest

from app.services import incident_service, telemetry_service
from tests.conftest import bearer, make_key, make_user
from tests.telemetry_helpers import Account, iso, make_event, startup_event


def _count(db, table, **where):
    clause = " and ".join(f"{k} = %s" for k in where) or "true"
    return db.execute(f"select count(*) as n from public.{table} where {clause}", tuple(where.values())).fetchone()["n"]


# --- persistence and the unchanged contract -----------------------------------


def test_single_event_is_persisted_with_every_field(client, db):
    a = Account(db)
    ev = a.event(event_id="evt-full", service="backend", env="staging", instance="pod-7", trace="trace-1",
                 span="span-1", request="req-1", thread="worker-3", line=42, attributes={"k": "v", "n": 1})
    r = a.post(client, [ev])
    assert r.status_code == 200
    assert r.json() == {"status": "success", "message": "Events received successfully", "received": 1}
    (row,) = a.events()
    assert (row["event_id"], row["event_type"], row["classification"]) == ("evt-full", "error", "abnormal_behaviour")
    assert row["project_id"] == a.project_uuid and row["schema_version"] == "1.0"
    assert (row["service_id"], row["environment"], row["instance_id"]) == ("backend", "staging", "pod-7")
    assert row["runtime"] == {"language": "python", "version": "3.12.2"}
    assert (row["severity"], row["message"], row["logger_name"], row["error_type"]) == (
        "ERROR", "connection refused", "app.db", "ConnectionError")
    assert row["stack_trace"].startswith("Traceback") and row["source_line"] == 42
    assert (row["source_file"], row["source_function"]) == ("app/db.py", "connect")
    assert (row["trace_id"], row["span_id"], row["request_id"], row["thread_name"]) == (
        "trace-1", "span-1", "req-1", "worker-3")
    assert row["attributes"] == {"k": "v", "n": 1}
    assert row["payload"]["event_id"] == "evt-full" and row["payload"]["application"]["project_id"] == a.external
    assert row["application"]["service_id"] == "backend" and row["received_at"] is not None
    assert row["event_timestamp"].astimezone(timezone.utc) == datetime(2026, 11, 9, 8, 30, tzinfo=timezone.utc)
    assert row["incident_id"] is not None


def test_batch_of_distinct_events(client, db):
    a = Account(db)
    r = a.post(client, [a.event(message=f"m{i}") for i in range(5)])
    assert r.json()["received"] == 5 and "accepted" not in r.json()
    assert len(a.events()) == 5


def test_events_without_optional_sections_are_accepted(client, db):
    a = Account(db)
    minimal = {
        "schema_version": "1.0", "event_id": "min-1", "timestamp": iso(), "type": "log",
        "application": {"project_id": a.external}, "details": {"severity": "INFO", "message": "hi"},
    }
    assert a.post(client, [minimal]).status_code == 200
    (row,) = a.events()
    assert row["service_id"] is None and row["trace_id"] is None and row["attributes"] == {}
    # SDK 0.1.0 shape (no classification): falls back to abnormal_behaviour; INFO still makes no incident.
    assert row["classification"] == "abnormal_behaviour" and row["incident_id"] is None


def test_naive_timestamps_are_treated_as_utc(client, db):
    a = Account(db)
    a.post(client, [a.event(ts="2026-11-09T08:30:00")])
    # No offset in the payload means UTC, regardless of the database session time zone.
    assert a.events()[0]["event_timestamp"].astimezone(timezone.utc) == datetime(2026, 11, 9, 8, 30, tzinfo=timezone.utc)


def test_nul_characters_do_not_break_the_batch(client, db):
    a = Account(db)
    assert a.post(client, [a.event(message="bad\x00byte", attributes={"x\x00": "y\x00"})]).status_code == 200
    assert "\x00" not in a.events()[0]["message"]


# --- idempotency ---------------------------------------------------------------


def test_retried_event_is_not_duplicated_and_does_not_double_count(client, db):
    a = Account(db)
    ev = a.event(event_id="retry-me")
    assert a.post(client, [ev]).status_code == 200
    again = a.post(client, [ev])
    assert again.status_code == 200
    body = again.json()
    assert body["received"] == 1 and body["accepted"] == 0 and body["duplicates"] == 1 and body["rejected"] == 0
    assert body["results"] == [{"index": 0, "event_id": "retry-me", "status": "duplicate"}]
    assert len(a.events()) == 1
    (incident,) = a.incidents()
    assert incident["event_count"] == 1
    assert len(a.history(incident["id"])) == 1  # one 'created' entry, no repeated timeline entries


def test_duplicate_inside_one_batch_and_mixed_new_events(client, db):
    a = Account(db)
    dup = a.event(event_id="same")
    r = a.post(client, [dup, dup, a.event(event_id="other", message="different")])
    body = r.json()
    assert body["received"] == 3 and body["accepted"] == 2 and body["duplicates"] == 1
    assert [x["index"] for x in body["results"]] == [1]
    assert len(a.events()) == 2


def test_same_event_id_in_different_projects_is_allowed(client, db):
    a, b = Account(db), Account(db)
    assert a.post(client, [a.event(event_id="shared-id")]).status_code == 200
    assert b.post(client, [b.event(event_id="shared-id")]).status_code == 200
    assert len(a.events()) == 1 and len(b.events()) == 1


def test_retried_startup_event_creates_no_second_registration(client, db):
    a = Account(db)
    ev = startup_event(a.external, event_id="boot-1", repository="https://github.com/acme/webapp")
    a.post(client, [ev])
    first = db.execute("select * from public.project_repositories where project_id = %s", (a.project_uuid,)).fetchall()
    a.post(client, [ev])
    second = db.execute("select * from public.project_repositories where project_id = %s", (a.project_uuid,)).fetchall()
    assert len(first) == len(second) == 1 and first[0]["last_seen_at"] == second[0]["last_seen_at"]
    assert len(a.events()) == 1


# --- validation, mixed batches and limits --------------------------------------


def test_mixed_valid_and_invalid_batch_persists_valid_and_reports_invalid(client, db):
    a = Account(db)
    good = a.event(event_id="good-1")
    bad_missing = {"event_id": "bad-1", "type": "error"}  # no timestamp/application/details
    bad_type = a.event(event_id="bad-2")
    bad_type["timestamp"] = "not-a-time"
    r = a.post(client, [good, bad_missing, "garbage", bad_type])
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "partial" and body["received"] == 4 and body["accepted"] == 1 and body["rejected"] == 3
    rejected = {x["index"]: x for x in body["results"]}
    assert set(rejected) == {1, 2, 3} and all(x["status"] == "rejected" and x["error"] for x in rejected.values())
    assert rejected[1]["event_id"] == "bad-1" and "event_id" not in rejected[2]  # unknown ids are omitted
    assert [e["event_id"] for e in a.events()] == ["good-1"]


def test_rejection_messages_do_not_echo_input_values(client, db):
    a = Account(db)
    secret = "ghp_supersecrettokenvalue1234567890"
    bad = a.event()
    bad["timestamp"] = secret
    r = a.post(client, [a.event(event_id="ok"), bad])
    assert secret not in json.dumps(r.json())


def test_all_invalid_batch_is_rejected_with_details(client, db):
    a = Account(db)
    r = a.post(client, [{"nope": 1}, 5])
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["message"] == "No valid events in the batch" and len(detail["rejected"]) == 2
    assert a.events() == []


def test_oversize_event_and_bad_field_lengths_are_rejected_individually(client, db):
    a = Account(db)
    huge = a.event(event_id="huge", stack="x" * (telemetry_service.MAX_EVENT_BYTES + 10))
    long_id = a.event(event_id="i" * (telemetry_service.MAX_EVENT_ID_LENGTH + 1))
    r = a.post(client, [a.event(event_id="fine"), huge, long_id])
    assert r.json()["rejected"] == 2 and [e["event_id"] for e in a.events()] == ["fine"]


def test_batch_size_limit(client, db):
    a = Account(db)
    r = a.post(client, [a.event() for _ in range(telemetry_service.MAX_BATCH_EVENTS + 1)])
    assert r.status_code == 413 and a.events() == []
    ok = a.post(client, [a.event(message=f"m{i}") for i in range(telemetry_service.MAX_BATCH_EVENTS)])
    assert ok.status_code == 200 and len(a.events()) == telemetry_service.MAX_BATCH_EVENTS


def test_payload_size_limit(client, db, monkeypatch):
    a = Account(db)
    monkeypatch.setattr(telemetry_service, "MAX_PAYLOAD_BYTES", 2000)
    big = a.event(attributes={"blob": "x" * 5000})
    r = a.post(client, [big])
    assert r.status_code == 413 and a.events() == []
    assert a.post(client, [a.event(event_id="small")]).status_code == 200


def test_invalid_bodies_still_422(client, db):
    a = Account(db)
    for body in ({"events": []}, {"events": "x"}, {}, {"other": 1}):
        assert client.post("/v1/events", json=body, headers=bearer(a.key)).status_code == 422
    assert client.post("/v1/events", content="{nope", headers={**bearer(a.key), "Content-Type": "application/json"}).status_code == 422
    assert client.post("/v1/events", json=[1, 2], headers=bearer(a.key)).status_code == 422


# --- ownership and tenant isolation ---------------------------------------------


def test_events_for_a_project_the_key_owner_does_not_own_are_rejected_and_not_stored(client, db):
    a, b = Account(db), Account(db)
    r = client.post("/v1/events", json={"events": [make_event(b.external)]}, headers=bearer(a.key))
    assert r.status_code == 403 and b.events() == [] and a.events() == []


def test_unknown_project_rejected_and_nothing_stored(client, db):
    a = Account(db)
    r = client.post("/v1/events", json={"events": [make_event("ghost-project"), a.event()]}, headers=bearer(a.key))
    assert r.status_code == 404 and a.events() == []


def test_internal_uuid_or_user_id_in_the_event_grants_nothing(client, db):
    a, b = Account(db), Account(db)
    # Claiming another tenant's internal UUID as the project_id is just an unknown project.
    r = client.post("/v1/events", json={"events": [make_event(str(b.project_uuid))]}, headers=bearer(a.key))
    assert r.status_code == 404
    # Extra identity-looking fields are tolerated (SDK evolution) but never trusted.
    sneaky = a.event()
    sneaky["user_id"] = str(b.user["id"])
    sneaky["project_uuid"] = str(b.project_uuid)
    sneaky["application"]["user_id"] = str(b.user["id"])
    assert client.post("/v1/events", json={"events": [sneaky]}, headers=bearer(a.key)).status_code == 200
    assert len(a.events()) == 1 and b.events() == []


def test_same_external_project_id_for_two_users_routes_by_api_key(client, db):
    shared = "same-ext-id"
    a, b = Account(db, shared), Account(db, shared)
    assert a.post(client, [make_event(shared, message="from a")]).status_code == 200
    assert b.post(client, [make_event(shared, message="from b")]).status_code == 200
    assert [e["message"] for e in a.events()] == ["from a"] and [e["message"] for e in b.events()] == ["from b"]


# --- transactionality ------------------------------------------------------------


def test_failure_mid_batch_rolls_back_the_whole_batch(client, db, monkeypatch):
    a = Account(db)
    calls = {"n": 0}
    real = incident_service.attach_event

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated failure")
        return real(*args, **kwargs)

    monkeypatch.setattr(incident_service, "attach_event", flaky)
    r = a.post(client, [a.event(message="one"), a.event(message="two"), a.event(message="three")])
    assert r.status_code == 500
    assert a.events() == [] and a.incidents() == []  # nothing from the failed batch survives
    monkeypatch.setattr(incident_service, "attach_event", real)
    assert a.post(client, [a.event(message="one")]).status_code == 200  # and a retry then succeeds


# --- classification ----------------------------------------------------------------


@pytest.mark.parametrize(
    "kw,expected",
    [
        ({}, "abnormal_behaviour"),  # omitted (SDK 0.1.0): fallback
        ({"severity": "WARNING"}, "abnormal_behaviour"),
        ({"type": "log", "severity": "INFO"}, "abnormal_behaviour"),
        ({"classification": "abnormal_behaviour"}, "abnormal_behaviour"),
        ({"classification": "normal_behaviour"}, "normal_behaviour"),
    ],
)
def test_classification_is_provided_or_falls_back_to_abnormal(client, db, kw, expected):
    a = Account(db)
    a.post(client, [a.event(**kw)])
    assert a.events()[0]["classification"] == expected


def test_startup_event_is_normal_behaviour_even_without_classification(client, db):
    a = Account(db)
    ev = startup_event(a.external)
    del ev["classification"]
    a.post(client, [ev])
    assert a.events()[0]["classification"] == "normal_behaviour"


def test_second_api_key_user_cannot_post_for_first_users_project(client, db):
    a = Account(db)
    intruder = make_user(db)
    key = make_key(db, intruder["id"])
    r = client.post("/v1/events", json={"events": [a.event()]}, headers=bearer(key))
    assert r.status_code == 403 and a.events() == []
