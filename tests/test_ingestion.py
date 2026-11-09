from tests.conftest import bearer, make_key, make_project, make_user, sample_event


def _account(db):
    user = make_user(db)
    return user, make_key(db, user["id"]), make_project(db, user["id"])


def test_missing_credentials_rejected(client, db):
    _, _, project = _account(db)
    r = client.post("/v1/events", json=sample_event(project))
    assert r.status_code == 401 and r.headers["www-authenticate"] == "Bearer"


def test_unauthenticated_rejected_even_with_invalid_body(client):
    assert client.post("/v1/events", json={"nope": 1}).status_code == 401


def test_wrong_scheme_or_garbage_key_rejected(client, db):
    _, key, project = _account(db)
    ev = sample_event(project)
    assert client.post("/v1/events", json=ev, headers={"Authorization": f"Basic {key}"}).status_code == 401
    assert client.post("/v1/events", json=ev, headers={"Authorization": key}).status_code == 401
    assert client.post("/v1/events", json=ev, headers=bearer("garbage")).status_code == 401
    assert client.post("/v1/events", json=ev, headers=bearer("jarvis_live_" + "a" * 43)).status_code == 401


def test_valid_key_accepted_with_unchanged_response(client, db):
    _, key, project = _account(db)
    r = client.post("/v1/events", json=sample_event(project), headers=bearer(key))
    assert r.status_code == 200
    assert r.json() == {"status": "success", "message": "Events received successfully", "received": 1}


def test_batch_of_multiple_events(client, db):
    _, key, project = _account(db)
    ev = sample_event(project)
    ev["events"] = ev["events"] * 3
    r = client.post("/v1/events", json=ev, headers=bearer(key))
    assert r.json()["received"] == 3


def test_last_used_updated(client, db):
    user, key, project = _account(db)
    assert db.execute("select last_used_at from public.api_keys where user_id=%s", (user["id"],)).fetchone()["last_used_at"] is None
    client.post("/v1/events", json=sample_event(project), headers=bearer(key))
    assert db.execute("select last_used_at from public.api_keys where user_id=%s", (user["id"],)).fetchone()["last_used_at"] is not None


def test_revoked_and_rotated_keys_rejected(client, db):
    user, key, project = _account(db)
    new = make_key(db, user["id"])
    assert client.post("/v1/events", json=sample_event(project), headers=bearer(key)).status_code == 401
    assert client.post("/v1/events", json=sample_event(project), headers=bearer(new)).status_code == 200
    db.execute("delete from public.api_keys where user_id=%s", (user["id"],))
    assert client.post("/v1/events", json=sample_event(project), headers=bearer(new)).status_code == 401


def test_cross_account_project_rejected(client, db):
    _, key_a, _ = _account(db)
    _, _, project_b = _account(db)
    r = client.post("/v1/events", json=sample_event(project_b), headers=bearer(key_a))
    assert r.status_code == 403


def test_unknown_project_rejected_clearly(client, db):
    _, key, _ = _account(db)
    r = client.post("/v1/events", json=sample_event("does-not-exist-xyz"), headers=bearer(key))
    assert r.status_code == 404 and "does-not-exist-xyz" in r.json()["detail"]


def test_batch_with_one_foreign_project_is_rejected_entirely(client, db):
    _, key_a, project_a = _account(db)
    _, _, project_b = _account(db)
    ev = sample_event(project_a)
    ev["events"].append(sample_event(project_b)["events"][0])
    assert client.post("/v1/events", json=ev, headers=bearer(key_a)).status_code == 403


def test_valid_key_with_invalid_payload_is_422(client, db):
    _, key, _ = _account(db)
    assert client.post("/v1/events", json={"events": []}, headers=bearer(key)).status_code == 422
    r = client.post("/v1/events", content="{not json", headers={**bearer(key), "Content-Type": "application/json"})
    assert r.status_code == 422


def test_project_ids_are_not_shared_between_accounts_via_key(client, db):
    """A key authenticates its own account only, never another account's projects."""
    user_a, key_a, project_a = _account(db)
    user_b, key_b, project_b = _account(db)
    assert client.post("/v1/events", json=sample_event(project_a), headers=bearer(key_b)).status_code == 403
    assert client.post("/v1/events", json=sample_event(project_b), headers=bearer(key_b)).status_code == 200

