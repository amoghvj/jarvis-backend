"""Classification contract (jarvis-observe 0.1.1): canonical values, fallback, validation, eligibility."""

import json

import pytest

from app.services import incident_service as inc
from app.services import telemetry_service as ts
from tests.conftest import bearer
from tests.telemetry_helpers import Account, iso, make_event, startup_event

NORMAL, ABNORMAL = "normal_behaviour", "abnormal_behaviour"


def _post(client, a, *events):
    return client.post("/v1/events", json={"events": list(events)}, headers=bearer(a.key))


def _row(a, event):
    return next(e for e in a.events() if e["event_id"] == event["event_id"])


# --- unit level: resolution and eligibility -----------------------------------------------------


@pytest.mark.parametrize(
    "payload,event_type,expected",
    [
        ({}, "error", ABNORMAL),                       # SDK 0.1.0: field omitted
        ({}, "log", ABNORMAL),
        ({}, "startup", NORMAL),                       # a lifecycle event is normal by definition
        ({"classification": ABNORMAL}, "error", ABNORMAL),
        ({"classification": NORMAL}, "startup", NORMAL),
        ({"classification": "ABNORMAL_BEHAVIOUR"}, "error", ABNORMAL),   # case-insensitive
        ({"classification": "  Normal_Behaviour "}, "log", NORMAL),      # surrounding whitespace trimmed
        ({"classification": NORMAL}, "error", NORMAL),                    # contradictory but valid: kept as sent
        ({"classification": ABNORMAL}, "startup", ABNORMAL),              # contradictory but valid: kept as sent
    ],
)
def test_resolve_classification_accepts_and_normalizes(payload, event_type, expected):
    assert ts.resolve_classification(payload, event_type) == expected


@pytest.mark.parametrize(
    "value",
    ["", "   ", "unknown", "error", "informational", "normal", "abnormal", "normal-behaviour", "normal_behavior",
     "normal_behaviour;abnormal_behaviour", "x" * 300, None, 0, 1, True, False, 1.5, [], ["normal_behaviour"], {},
     {"v": "normal_behaviour"}],
)
def test_resolve_classification_rejects_everything_else(value):
    with pytest.raises(ts.InvalidClassification, match="normal_behaviour' or 'abnormal_behaviour"):
        ts.resolve_classification({"classification": value}, "error")


@pytest.mark.parametrize(
    "event_type,classification,severity,expected",
    [
        ("error", ABNORMAL, "ERROR", True),
        ("error", ABNORMAL, "CRITICAL", True),
        ("log", ABNORMAL, "ERROR", True),              # SDKs also emit ERROR-level `log` events
        ("error", ABNORMAL, None, True),               # unchanged: an error-type event without severity
        ("log", ABNORMAL, None, False),
        ("error", ABNORMAL, "WARNING", False),
        ("error", ABNORMAL, "INFO", False),
        ("log", ABNORMAL, "DEBUG", False),
        ("error", NORMAL, "ERROR", False),
        ("error", NORMAL, "CRITICAL", False),
        ("startup", NORMAL, "INFO", False),
        ("startup", NORMAL, "CRITICAL", False),
        ("startup", ABNORMAL, "ERROR", False),         # startup never creates an incident, whatever it says
        ("startup", ABNORMAL, "CRITICAL", False),
        ("error", "garbage", "ERROR", False),          # anything that is not abnormal_behaviour is not eligible
    ],
)
def test_incident_eligibility_truth_table(event_type, classification, severity, expected):
    assert inc.is_incident_eligible(event_type, classification, severity) is expected


# --- HTTP level: SDK 0.1.0 and 0.1.1 payloads ----------------------------------------------------


def test_sdk_010_payload_without_classification(client, db):
    a = Account(db)
    err = a.event(message="legacy error")
    info = a.event(message="legacy info", severity="INFO", type="log", error_type=None, stack=None)
    r = _post(client, a, err, info)
    assert r.status_code == 200 and r.json()["received"] == 2 and "rejected" not in r.json()
    assert "classification" not in err and _row(a, err)["classification"] == ABNORMAL
    assert _row(a, err)["incident_id"] is not None  # ERROR still groups into an incident
    assert _row(a, info)["classification"] == ABNORMAL and _row(a, info)["incident_id"] is None
    assert "classification" not in _row(a, err)["payload"]  # the original payload is stored as received


def test_sdk_011_batch_envelope_and_events_are_accepted(client, db):
    """The shape the Python 0.1.1 SDK really sends: extra batch_id/sdk keys plus classified events."""
    a = Account(db)
    startup = startup_event(a.external, repository="https://github.com/acme/webapp", attributes={"capture_source": "sdk_initialization"})
    error = a.event(classification=ABNORMAL, message="db exploded")
    info = a.event(classification=ABNORMAL, severity="INFO", type="log", error_type=None, stack=None, message="hello")
    body = {"batch_id": "b-1", "sdk": {"name": "jarvis-observe", "version": "0.1.1"}, "events": [startup, error, info]}
    r = client.post("/v1/events", json=body, headers=bearer(a.key))
    assert r.status_code == 200 and r.json() == {"status": "success", "message": "Events received successfully", "received": 3}
    assert [_row(a, e)["classification"] for e in (startup, error, info)] == [NORMAL, ABNORMAL, ABNORMAL]
    assert [bool(_row(a, e)["incident_id"]) for e in (startup, error, info)] == [False, True, False]
    assert _row(a, startup)["payload"]["classification"] == NORMAL


def test_startup_event_with_normal_behaviour(client, db):
    a = Account(db)
    ev = startup_event(a.external, repository="https://github.com/acme/webapp")
    assert ev["classification"] == NORMAL and _post(client, a, ev).status_code == 200
    row = _row(a, ev)
    assert (row["event_type"], row["classification"], row["incident_id"]) == ("startup", NORMAL, None)
    assert a.incidents() == []
    assert db.execute("select count(*) as n from public.project_repositories where project_id = %s", (a.project_uuid,)).fetchone()["n"] == 1


def test_startup_event_without_classification_is_normal(client, db):
    a = Account(db)
    ev = startup_event(a.external)
    del ev["classification"]
    _post(client, a, ev)
    assert _row(a, ev)["classification"] == NORMAL and a.incidents() == []


def test_error_event_classified_normal_behaviour_creates_no_incident(client, db):
    a = Account(db)
    ev = a.event(classification=NORMAL, severity="ERROR")
    assert _post(client, a, ev).status_code == 200
    row = _row(a, ev)
    assert row["classification"] == NORMAL and row["payload"]["classification"] == NORMAL  # consistent
    assert row["incident_id"] is None and a.incidents() == []
    crit = a.event(classification=NORMAL, severity="CRITICAL", message="also suppressed")
    _post(client, a, crit)
    assert _row(a, crit)["incident_id"] is None and a.incidents() == []


@pytest.mark.parametrize("severity", ["INFO", "WARNING", "DEBUG"])
@pytest.mark.parametrize("event_type", ["log", "error"])
def test_non_error_severities_never_create_incidents_even_when_abnormal(client, db, severity, event_type):
    a = Account(db)
    ev = a.event(classification=ABNORMAL, severity=severity, type=event_type)
    assert _post(client, a, ev).status_code == 200
    row = _row(a, ev)
    assert row["classification"] == ABNORMAL and row["severity"] == severity
    assert row["incident_id"] is None and a.incidents() == []


@pytest.mark.parametrize("severity", ["ERROR", "CRITICAL"])
def test_abnormal_error_and_critical_events_still_group(client, db, severity):
    a = Account(db)
    first, second = a.event(classification=ABNORMAL, severity=severity, ts=iso(0)), a.event(classification=ABNORMAL, severity=severity, ts=iso(30))
    _post(client, a, first)
    _post(client, a, second)
    (incident,) = a.incidents()
    assert incident["event_count"] == 2 and incident["severity"] == severity
    assert _row(a, first)["incident_id"] == _row(a, second)["incident_id"] == incident["id"]


def test_old_and_new_sdk_events_for_the_same_error_share_one_incident(client, db):
    a = Account(db)
    old, new = a.event(ts=iso(0)), a.event(ts=iso(60), classification=ABNORMAL)  # same fingerprint
    _post(client, a, old)
    _post(client, a, new)
    (incident,) = a.incidents()
    assert incident["event_count"] == 2  # the fingerprint does not depend on classification


# --- casing, malformed and contradictory input ---------------------------------------------------


@pytest.mark.parametrize("raw,stored,incident", [
    ("NORMAL_BEHAVIOUR", NORMAL, False),
    ("Normal_Behaviour", NORMAL, False),
    (" normal_behaviour ", NORMAL, False),
    ("ABNORMAL_BEHAVIOUR", ABNORMAL, True),
])
def test_casing_and_whitespace_are_normalized_and_cannot_bypass_normal_handling(client, db, raw, stored, incident):
    a = Account(db)
    ev = a.event(classification=raw)
    assert _post(client, a, ev).status_code == 200
    row = _row(a, ev)
    assert row["classification"] == stored and bool(row["incident_id"]) is incident
    assert row["payload"]["classification"] == raw  # the original payload is preserved untouched


@pytest.mark.parametrize("bad", ["unknown_value", "", "   ", "error", "informational", None, 123, True, ["abnormal_behaviour"], {"a": 1}])
def test_invalid_classification_is_rejected_not_stored(client, db, bad):
    a = Account(db)
    ev = a.event(classification=bad)
    r = _post(client, a, ev)
    assert r.status_code == 422  # a batch with no valid event
    detail = r.json()["detail"]
    assert detail["message"] == "No valid events in the batch"
    assert "classification must be 'normal_behaviour' or 'abnormal_behaviour'" in detail["rejected"][0]["error"]
    assert detail["rejected"][0]["event_id"] == ev["event_id"]
    assert a.events() == [] and a.incidents() == []


def test_invalid_classification_message_does_not_echo_the_value(client, db):
    a = Account(db)
    secret = "ghp_supersecrettokenvalue1234567890"
    r = _post(client, a, a.event(classification=secret))
    assert secret not in json.dumps(r.json())


def test_invalid_classification_in_a_mixed_batch_rejects_only_that_event(client, db):
    a = Account(db)
    good, bad, startup = a.event(message="good", classification=ABNORMAL), a.event(message="bad", classification="nope"), startup_event(a.external)
    r = _post(client, a, good, bad, startup)
    body = r.json()
    assert r.status_code == 200 and body["status"] == "partial" and body["accepted"] == 2 and body["rejected"] == 1
    assert [x["index"] for x in body["results"]] == [1] and body["results"][0]["status"] == "rejected"
    assert {e["event_id"] for e in a.events()} == {good["event_id"], startup["event_id"]}
    assert len(a.incidents()) == 1


def test_rejected_invalid_event_does_not_count_against_incidents_or_leave_partial_state(client, db):
    a = Account(db)
    _post(client, a, a.event(message="ok", ts=iso(0)))
    before = a.incidents()[0]["event_count"]
    r = _post(client, a, a.event(message="ok", ts=iso(10), classification="nope"))
    assert r.status_code == 422
    assert a.incidents()[0]["event_count"] == before and len(a.events()) == 1


@pytest.mark.parametrize("severity", ["INFO", "ERROR", "CRITICAL"])
@pytest.mark.parametrize("classification", [NORMAL, ABNORMAL, None])
def test_startup_events_never_create_incidents_whatever_they_claim(client, db, severity, classification):
    a = Account(db)
    kwargs = {"severity": severity}
    if classification:
        kwargs["classification"] = classification
    ev = startup_event(a.external, **kwargs)
    if classification is None:
        ev.pop("classification")
    assert _post(client, a, ev).status_code == 200
    row = _row(a, ev)
    assert row["event_type"] == "startup" and row["incident_id"] is None and a.incidents() == []
    assert row["classification"] == (classification or NORMAL)
    # Contradictory-but-valid input is stored exactly as sent (and never reinterpreted).
    if classification:
        assert row["payload"]["classification"] == classification


def test_startup_with_error_severity_does_not_touch_an_existing_incident(client, db):
    a = Account(db)
    _post(client, a, a.event(message="real failure", severity="ERROR"))
    before = a.incidents()[0]
    _post(client, a, startup_event(a.external, severity="CRITICAL", message="real failure"))
    after = a.incidents()[0]
    assert (after["event_count"], after["severity"], after["id"]) == (before["event_count"], "ERROR", before["id"])
    assert len(a.history(after["id"])) == 1  # no escalation recorded


def test_startup_event_type_casing_cannot_bypass_the_startup_rule(client, db):
    a = Account(db)
    ev = startup_event(a.external, severity="ERROR", classification=ABNORMAL)
    ev["type"] = "STARTUP"
    assert _post(client, a, ev).status_code == 200
    assert _row(a, ev)["event_type"] == "startup" and a.incidents() == []


# --- database constraint and idempotency ---------------------------------------------------------


def test_every_persisted_row_has_a_canonical_non_null_classification(client, db):
    a = Account(db)
    events = [
        a.event(), a.event(classification=ABNORMAL, message="b"), a.event(classification=NORMAL, message="c"),
        a.event(classification="ABNORMAL_BEHAVIOUR", message="d"),
        a.event(severity="INFO", type="log", message="e"), startup_event(a.external),
    ]
    assert _post(client, a, *events).status_code == 200
    values = {e["classification"] for e in a.events()}
    assert values <= {NORMAL, ABNORMAL} and None not in values and len(a.events()) == len(events)


def test_classified_duplicates_remain_idempotent(client, db):
    a = Account(db)
    ev = a.event(classification=ABNORMAL, event_id="dup-classified")
    _post(client, a, ev)
    again = _post(client, a, ev)
    assert again.json()["duplicates"] == 1 and len(a.events()) == 1 and a.incidents()[0]["event_count"] == 1


def test_normalize_event_cannot_produce_a_non_canonical_classification(db):
    a = Account(db)
    for payload_kw in ({}, {"classification": "Abnormal_Behaviour"}, {"classification": NORMAL}):
        ev = make_event(a.external, **payload_kw)
        valid, rejected = ts.parse_events([ev])
        assert not rejected
        _, event, payload = valid[0]
        assert ts.normalize_event(event, payload, a.project_uuid)["classification"] in (NORMAL, ABNORMAL)
    bad = make_event(a.external, classification="other")
    event = ts.Event.model_validate(bad)
    with pytest.raises(ts.InvalidClassification):
        ts.normalize_event(event, bad, a.project_uuid)  # defence in depth: never stores a bad value

