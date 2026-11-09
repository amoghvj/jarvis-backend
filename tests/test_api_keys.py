from app.services import api_key_service
from tests.conftest import csrf, login, make_user


def test_rotate_requires_authentication(client):
    assert client.post("/api-keys/rotate", headers=csrf()).status_code == 401
    assert client.get("/api-keys/status").status_code == 401
    assert client.delete("/api-keys", headers=csrf()).status_code == 401


def test_rotate_requires_trusted_origin(client, github):
    login(client, github)
    assert client.post("/api-keys/rotate").status_code == 403
    assert client.delete("/api-keys").status_code == 403


def test_generate_key_format_and_entropy():
    k1, k2 = api_key_service.generate_api_key(), api_key_service.generate_api_key()
    assert k1.startswith("jarvis_live_") and k1 != k2
    assert len(k1) - len("jarvis_live_") >= 43  # >= 32 random bytes, base64url


def test_rotate_returns_key_once_and_stores_only_hash(client, github, db):
    login(client, github)
    r = client.post("/api-keys/rotate", headers=csrf())
    assert r.status_code == 200
    key = r.json()["api_key"]
    assert key.startswith("jarvis_live_") and r.json()["key_prefix"] == key[:16]

    row = db.execute("select key_hash, key_prefix from public.api_keys").fetchall()
    stored = [x for x in row if x["key_hash"] == api_key_service.hash_api_key(key)]
    assert len(stored) == 1
    assert key not in stored[0]["key_hash"] and key != stored[0]["key_prefix"]

    status = client.get("/api-keys/status")
    assert key not in status.text and stored[0]["key_hash"] not in status.text
    assert status.json()["active"] is True and status.json()["key_prefix"] == key[:16]
    assert status.json()["last_used_at"] is None


def test_status_without_key(client, github):
    login(client, github)
    assert client.get("/api-keys/status").json() == {
        "active": False, "key_prefix": None, "created_at": None, "last_used_at": None,
    }


def test_rotation_replaces_single_row_and_invalidates_old_key(client, github, db):
    login(client, github)
    uid = client.get("/auth/me").json()["id"]
    old = client.post("/api-keys/rotate", headers=csrf()).json()["api_key"]
    new = client.post("/api-keys/rotate", headers=csrf()).json()["api_key"]
    assert old != new
    n = db.execute("select count(*) as n from public.api_keys where user_id = %s", (uid,)).fetchone()["n"]
    assert n == 1  # unique user_id respected
    # old key no longer authenticates, new one does
    import pytest
    with pytest.raises(api_key_service.InvalidApiKey):
        api_key_service.authenticate(db, f"Bearer {old}")
    assert api_key_service.authenticate(db, f"Bearer {new}")


def test_revoke_only_affects_current_user(client, github, db):
    other = make_user(db)
    other_key = api_key_service.rotate_api_key(db, other["id"])["api_key"]

    login(client, github)
    client.post("/api-keys/rotate", headers=csrf())
    r = client.delete("/api-keys", headers=csrf())
    assert r.status_code == 200
    assert client.get("/api-keys/status").json()["active"] is False
    assert api_key_service.authenticate(db, f"Bearer {other_key}")  # untouched


def test_revoke_without_key_is_ok(client, github):
    login(client, github)
    r = client.delete("/api-keys", headers=csrf())
    assert r.status_code == 200 and r.json()["message"] == "No active API key"


def test_failed_rotation_keeps_old_key(client, github, db, monkeypatch):
    login(client, github)
    old = client.post("/api-keys/rotate", headers=csrf()).json()["api_key"]

    from app import repositories

    def fail(*a, **k):
        raise RuntimeError("db down")

    monkeypatch.setattr(repositories, "upsert_api_key", fail)
    assert client.post("/api-keys/rotate", headers=csrf()).status_code == 500
    monkeypatch.undo()
    assert api_key_service.authenticate(db, f"Bearer {old}")

