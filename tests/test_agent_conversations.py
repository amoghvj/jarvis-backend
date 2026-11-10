"""API and authorization tests for project-scoped conversations, messages, and runs."""

from uuid import uuid4

import psycopg
import pytest

from app import repositories, repositories_agent
from tests.conftest import csrf, login, make_user


def _login_and_create_project(client, db, github, project_key="proj-alpha"):
    login(client, github, username=f"user-{uuid4().hex[:6]}")
    resp = client.post(
        "/projects",
        json={"project_id": project_key, "name": f"Project {project_key}"},
        headers=csrf(),
    )
    assert resp.status_code == 201
    return resp.json()


def test_conversations_require_authentication(client, db):
    user = make_user(db)
    proj = repositories.insert_project(db, user["id"], "proj-auth-1", "proj-auth-1")
    pid = proj["id"]
    cid = uuid4()
    rid = uuid4()

    assert client.get(f"/projects/{pid}/conversations").status_code == 401
    assert client.post(f"/projects/{pid}/conversations", headers=csrf()).status_code == 401
    assert client.get(f"/projects/{pid}/conversations/{cid}").status_code == 401
    assert client.delete(f"/projects/{pid}/conversations/{cid}", headers=csrf()).status_code == 401
    assert client.post(
        f"/projects/{pid}/conversations/{cid}/messages",
        json={"content": "hello"},
        headers=csrf(),
    ).status_code == 401
    assert client.get(f"/projects/{pid}/conversations/{cid}/runs/{rid}").status_code == 401
    assert client.post(
        f"/projects/{pid}/conversations/{cid}/runs/{rid}/advance",
        headers=csrf(),
    ).status_code == 401


def test_mutating_conversation_routes_enforce_csrf_origin(client, db, github):
    proj = _login_and_create_project(client, db, github)
    pid = proj["id"]

    # Missing Origin header rejected with 403
    assert client.post(f"/projects/{pid}/conversations", json={"title": "x"}).status_code == 403

    created = client.post(f"/projects/{pid}/conversations", json={"title": "x"}, headers=csrf())
    assert created.status_code == 201
    cid = created.json()["id"]

    assert client.post(
        f"/projects/{pid}/conversations/{cid}/messages",
        json={"content": "hello"},
    ).status_code == 403
    assert client.post(
        f"/projects/{pid}/conversations/{cid}/runs/{uuid4()}/advance",
    ).status_code == 403
    assert client.delete(f"/projects/{pid}/conversations/{cid}").status_code == 403


def test_conversation_lifecycle_and_project_isolation(client, db, github):
    proj_a = _login_and_create_project(client, db, github, project_key="proj-a")
    resp_b = client.post(
        "/projects",
        json={"project_id": "proj-b", "name": "Project B"},
        headers=csrf(),
    )
    assert resp_b.status_code == 201
    proj_b = resp_b.json()

    # Create conversation in proj_a without title
    c1_resp = client.post(f"/projects/{proj_a['id']}/conversations", headers=csrf())
    assert c1_resp.status_code == 201
    c1 = c1_resp.json()
    assert c1["project_id"] == proj_a["id"]
    assert c1["title"] is None

    # Create conversation in proj_b with explicit title
    c2_resp = client.post(
        f"/projects/{proj_b['id']}/conversations",
        json={"title": "Investigate checkout latency"},
        headers=csrf(),
    )
    assert c2_resp.status_code == 201
    c2 = c2_resp.json()
    assert c2["project_id"] == proj_b["id"]
    assert c2["title"] == "Investigate checkout latency"

    # Listing proj_a only returns c1; listing proj_b only returns c2
    list_a = client.get(f"/projects/{proj_a['id']}/conversations").json()
    assert [c["id"] for c in list_a["conversations"]] == [c1["id"]]

    list_b = client.get(f"/projects/{proj_b['id']}/conversations").json()
    assert [c["id"] for c in list_b["conversations"]] == [c2["id"]]

    # Submit a message to c1 under proj_a to create a run
    sub_c1 = client.post(
        f"/projects/{proj_a['id']}/conversations/{c1['id']}/messages",
        json={"content": "Check proj_a incidents"},
        headers=csrf(),
    )
    assert sub_c1.status_code == 202
    c1_run_id = sub_c1.json()["run_id"]

    # Cross-project access to c1 and its run under proj_b is rejected with 404
    assert client.get(f"/projects/{proj_b['id']}/conversations/{c1['id']}").status_code == 404
    assert client.get(
        f"/projects/{proj_b['id']}/conversations/{c1['id']}/runs/{c1_run_id}"
    ).status_code == 404
    assert client.delete(
        f"/projects/{proj_b['id']}/conversations/{c1['id']}",
        headers=csrf(),
    ).status_code == 404
    assert client.post(
        f"/projects/{proj_b['id']}/conversations/{c1['id']}/messages",
        json={"content": "cross project message"},
        headers=csrf(),
    ).status_code == 404


def test_cross_user_isolation_rejected(client, db, github):
    # User 1 creates project and conversation
    other_user = make_user(db, username="other-owner")
    other_proj = repositories.insert_project(db, other_user["id"], "other-proj", "other-proj")
    other_conv = repositories_agent.create_conversation(
        db,
        user_id=other_user["id"],
        project_id=other_proj["id"],
        title="Private investigation",
    )
    _, other_run, _ = repositories_agent.submit_user_message_and_create_run(
        db,
        user_id=other_user["id"],
        project_id=other_proj["id"],
        conversation_id=other_conv["id"],
        content="secret question",
    )

    # User 2 logs in and tries to access User 1's project, conversation, and run
    my_proj = _login_and_create_project(client, db, github, project_key="my-proj")

    for pid in (other_proj["id"], my_proj["id"]):
        assert client.get(f"/projects/{pid}/conversations/{other_conv['id']}").status_code == 404
        assert client.delete(
            f"/projects/{pid}/conversations/{other_conv['id']}",
            headers=csrf(),
        ).status_code == 404
        assert client.post(
            f"/projects/{pid}/conversations/{other_conv['id']}/messages",
            json={"content": "intrude"},
            headers=csrf(),
        ).status_code == 404
        assert client.get(
            f"/projects/{pid}/conversations/{other_conv['id']}/runs/{other_run['id']}"
        ).status_code == 404

    assert client.get(f"/projects/{other_proj['id']}/conversations").status_code == 404
    assert client.post(
        f"/projects/{other_proj['id']}/conversations",
        json={"title": "unauthorized"},
        headers=csrf(),
    ).status_code == 404

    # Repository-level ownership enforcement also rejects creating a conversation on another user's project
    intruder = make_user(db, username="intruder-user")
    with pytest.raises(ValueError, match="Project not found for user"):
        repositories_agent.create_conversation(
            db,
            user_id=intruder["id"],
            project_id=other_proj["id"],
            title="Spoofed project owner",
        )


def test_message_submission_idempotency_active_run_guard_and_distinct_request_ids(client, db, github):
    proj = _login_and_create_project(client, db, github)
    pid = proj["id"]
    conv = client.post(f"/projects/{pid}/conversations", headers=csrf()).json()
    cid = conv["id"]

    # 1. Submit first message with client_request_id
    sub1 = client.post(
        f"/projects/{pid}/conversations/{cid}/messages",
        json={
            "content": "Investigate incident 00000000-0000-0000-0000-0000000001a1 in payment-service",
            "client_request_id": "req-001",
        },
        headers=csrf(),
    )
    assert sub1.status_code == 202
    data1 = sub1.json()
    assert data1["status"] == "queued"
    assert data1["deduplicated"] is False
    run_id = data1["run_id"]
    msg_id = data1["user_message_id"]

    # Title was deterministically generated from the first user message
    detail = client.get(f"/projects/{pid}/conversations/{cid}").json()
    assert detail["conversation"]["title"].startswith("Investigate incident")
    assert len(detail["messages"]) == 1
    assert detail["messages"][0]["id"] == msg_id
    assert detail["messages"][0]["sequence_number"] == 1

    # 2. Retry the exact same request with the same client_request_id -> idempotent 202 response
    sub2 = client.post(
        f"/projects/{pid}/conversations/{cid}/messages",
        json={
            "content": "Investigate incident 00000000-0000-0000-0000-0000000001a1 in payment-service",
            "client_request_id": "req-001",
        },
        headers=csrf(),
    )
    assert sub2.status_code == 202
    data2 = sub2.json()
    assert data2["run_id"] == run_id
    assert data2["user_message_id"] == msg_id
    assert data2["deduplicated"] is True

    # 3. Submitting a different client_request_id while the run is still active returns 409 Conflict
    sub3 = client.post(
        f"/projects/{pid}/conversations/{cid}/messages",
        json={
            "content": "Investigate incident 00000000-0000-0000-0000-0000000001a1 in payment-service",
            "client_request_id": "req-002",
        },
        headers=csrf(),
    )
    assert sub3.status_code == 409
    assert sub3.json()["detail"]["active_run_id"] == run_id

    # 4. Complete the first run, then submit with a different client_request_id ("req-002"):
    # must NOT be falsely deduplicated even if the message text is identical!
    repositories_agent.claim_run(db, worker_id="w-idem-1", run_id=run_id)
    repositories_agent.complete_run(db, run_id=run_id, worker_id="w-idem-1")

    sub4 = client.post(
        f"/projects/{pid}/conversations/{cid}/messages",
        json={
            "content": "Investigate incident 00000000-0000-0000-0000-0000000001a1 in payment-service",
            "client_request_id": "req-002",
        },
        headers=csrf(),
    )
    assert sub4.status_code == 202
    data4 = sub4.json()
    assert data4["deduplicated"] is False
    assert data4["run_id"] != run_id
    assert data4["user_message_id"] != msg_id


def test_concurrent_duplicate_submission_unique_violation_recovery(client, db, github, monkeypatch):
    """Verify that if a concurrent request inserts the same (conversation_id, client_request_id)
    between the initial SELECT check and the INSERT into agent_runs, the savepoint rolls back
    the duplicate user message and returns the winning run with HTTP 202 and deduplicated=True.
    """
    proj = _login_and_create_project(client, db, github)
    pid = proj["id"]
    conv = client.post(f"/projects/{pid}/conversations", headers=csrf()).json()
    cid = conv["id"]

    # Create the winning user message and run first
    first_resp = client.post(
        f"/projects/{pid}/conversations/{cid}/messages",
        json={"content": "Concurrent investigation request", "client_request_id": "req-race-1"},
        headers=csrf(),
    )
    assert first_resp.status_code == 202
    first_data = first_resp.json()
    assert first_data["deduplicated"] is False

    # Complete the first run so the active-run check does not short-circuit before INSERT,
    # and simulate a race where the initial pre-check SELECT did not yet see the concurrent
    # winner, while the concurrent winner is already persisted when our savepoint executes!
    repositories_agent.claim_run(db, worker_id="w-race-1", run_id=first_data["run_id"])
    repositories_agent.complete_run(db, run_id=first_data["run_id"], worker_id="w-race-1")

    orig_get_message_by_id = repositories_agent.get_message_by_id
    calls = {"count": 0}

    def _simulate_precheck_miss_then_hit(conn, *, conversation_id, message_id):
        calls["count"] += 1
        return orig_get_message_by_id(
            conn,
            conversation_id=conversation_id,
            message_id=message_id,
        )

    # In submit_user_message_and_create_run, if the initial SELECT (before the savepoint)
    # misses the row because the concurrent transaction committed right after that SELECT,
    # then create_run_for_user_message hits PostgreSQL's UniqueViolation inside the savepoint.
    orig_expire = repositories_agent.expire_stale_runs

    def _expire_and_hide_for_precheck(conn, **kwargs):
        res = orig_expire(conn, **kwargs)
        return res

    # Directly test the savepoint + UniqueViolation recovery in submit_user_message_and_create_run
    # by wrapping conn.execute only for the first pre-check query on client_request_id:
    real_execute = db.execute
    precheck_skipped = {"done": False}

    class _RaceConnProxy:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def execute(self, query, params=None, *args, **kwargs):
            if (
                not precheck_skipped["done"]
                and isinstance(query, str)
                and "WHERE conversation_id = %s AND client_request_id = %s" in query
            ):
                precheck_skipped["done"] = True
                # Simulate empty result on the pre-check SELECT before the concurrent winner commits
                return self._inner.execute(
                    "SELECT 1 FROM public.agent_runs WHERE 1 = 0"
                )
            return self._inner.execute(query, params, *args, **kwargs)

    msg_out, run_out, dedup_out = repositories_agent.submit_user_message_and_create_run(
        _RaceConnProxy(db),
        user_id=db.execute("SELECT user_id FROM public.projects WHERE id = %s", (pid,)).fetchone()["user_id"],
        project_id=pid,
        conversation_id=cid,
        content="Concurrent investigation request",
        client_request_id="req-race-1",
    )
    assert precheck_skipped["done"] is True
    assert dedup_out is True
    assert str(run_out["id"]) == first_data["run_id"]
    assert str(msg_out["id"]) == first_data["user_message_id"]

    # Also verify via the HTTP route
    race_resp = client.post(
        f"/projects/{pid}/conversations/{cid}/messages",
        json={"content": "Concurrent investigation request", "client_request_id": "req-race-1"},
        headers=csrf(),
    )
    assert race_resp.status_code == 202
    race_data = race_resp.json()
    assert race_data["deduplicated"] is True
    assert race_data["run_id"] == first_data["run_id"]
    assert race_data["user_message_id"] == first_data["user_message_id"]

    # Verify the savepoint rolled back the second user message so only 1 message and 1 run exist!
    msg_count = db.execute(
        "SELECT count(*) AS n FROM public.conversation_messages WHERE conversation_id = %s",
        (cid,),
    ).fetchone()["n"]
    run_count = db.execute(
        "SELECT count(*) AS n FROM public.agent_runs WHERE conversation_id = %s",
        (cid,),
    ).fetchone()["n"]
    assert msg_count == 1
    assert run_count == 1


def test_run_polling_incremental_steps_and_visible_transcript_separation(client, db, github):
    proj = _login_and_create_project(client, db, github)
    pid = proj["id"]
    conv = client.post(f"/projects/{pid}/conversations", headers=csrf()).json()
    cid = conv["id"]

    sub = client.post(
        f"/projects/{pid}/conversations/{cid}/messages",
        json={"content": "What caused the database timeout?"},
        headers=csrf(),
    ).json()
    run_id = sub["run_id"]

    # Simulate worker claiming and recording steps + tool messages + final assistant message
    claimed = repositories_agent.claim_run(db, worker_id="test-worker-1", run_id=run_id)
    assert claimed is not None and claimed["status"] == "running"

    repositories_agent.append_run_step(
        db,
        run_id=run_id,
        step_type="model_started",
        status="in_progress",
        input_summary={"iteration": 1, "activity": "Analyzing investigation request"},
    )
    repositories_agent.append_run_step(
        db,
        run_id=run_id,
        step_type="tool_started",
        tool_name="search_events",
        status="in_progress",
        input_summary={"activity": "Searching project telemetry events"},
    )

    # First poll: after_step=0 returns steps 1 and 2
    poll1 = client.get(f"/projects/{pid}/conversations/{cid}/runs/{run_id}").json()
    assert poll1["status"] == "running"
    assert poll1["latest_step_number"] == 2
    assert [s["step_number"] for s in poll1["steps"]] == [1, 2]

    # Persist internal tool_call and tool_result messages (must NOT appear in visible transcript)
    repositories_agent.append_conversation_message(
        db,
        conversation_id=cid,
        role="assistant",
        message_type="tool_call",
        content=None,
        tool_calls=[{"id": "call_1", "name": "search_events", "args": {"severity": "ERROR"}}],
        run_id=run_id,
    )
    repositories_agent.append_conversation_message(
        db,
        conversation_id=cid,
        role="tool",
        message_type="tool_result",
        content='{"ok": true, "operation": "search_events", "data": {"events": []}}',
        tool_call_id="call_1",
        tool_name="search_events",
        run_id=run_id,
    )
    # Persist final visible assistant text message
    final_msg = repositories_agent.append_conversation_message(
        db,
        conversation_id=cid,
        role="assistant",
        message_type="text",
        content="Confirmed Facts: No ERROR events were recorded in the selected window.",
        run_id=run_id,
    )
    repositories_agent.append_run_step(
        db,
        run_id=run_id,
        step_type="tool_completed",
        tool_name="search_events",
        status="completed",
        output_summary={"ok": True, "summary": "Found 0 matching telemetry event(s)."},
    )
    repositories_agent.append_run_step(
        db,
        run_id=run_id,
        step_type="run_completed",
        status="completed",
        output_summary={"activity": "Prepared final investigation response"},
    )
    repositories_agent.complete_run(db, run_id=run_id, worker_id="test-worker-1")

    # Incremental second poll: after_step=2 returns ONLY steps 3 and 4!
    poll2 = client.get(
        f"/projects/{pid}/conversations/{cid}/runs/{run_id}",
        params={"after_step": 2},
    ).json()
    assert poll2["status"] == "completed"
    assert poll2["latest_step_number"] == 4
    assert [s["step_number"] for s in poll2["steps"]] == [3, 4]
    assert len(poll2["assistant_messages"]) == 1
    assert poll2["assistant_messages"][0]["id"] == str(final_msg["id"])

    # Conversation detail returns ONLY the user message and final assistant text message (not tool_call/tool_result)
    conv_detail = client.get(f"/projects/{pid}/conversations/{cid}").json()
    roles_and_types = [(m["role"], m["message_type"]) for m in conv_detail["messages"]]
    assert roles_and_types == [("user", "text"), ("assistant", "text")]
    assert conv_detail["active_run"] is None


def test_advance_endpoint_multi_invocation_continuation_concurrency_and_completed_noop(
    client, db, github, monkeypatch
):
    """Verify:
    1. GET conversation detail returns active_run when a run is queued/running (page refresh recovery).
    2. Sequential POST .../advance invocations each execute bounded work and return 'queued'
       until the final invocation returns 'completed'.
    3. Concurrent POST .../advance while a run is actively leased returns 'running' without
       re-executing or duplicating work.
    4. POST .../advance on a completed run never executes again.
    """
    from langchain_core.messages import AIMessage
    from langchain_core.runnables import RunnableLambda
    import app.routes.conversations as conv_routes
    from app.agent.worker import claim_and_execute_run as real_claim_and_execute

    proj = _login_and_create_project(client, db, github, project_key="proj-advance")
    pid = proj["id"]
    conv = client.post(f"/projects/{pid}/conversations", headers=csrf()).json()
    cid = conv["id"]

    sub = client.post(
        f"/projects/{pid}/conversations/{cid}/messages",
        json={"content": "Investigate multi-step issue across bounded invocations"},
        headers=csrf(),
    ).json()
    run_id = sub["run_id"]
    assert sub["status"] == "queued"

    # Page reload check: GET conversation detail includes active_run while queued
    detail_queued = client.get(f"/projects/{pid}/conversations/{cid}").json()
    assert detail_queued["active_run"] is not None
    assert detail_queued["active_run"]["run_id"] == run_id
    assert detail_queued["active_run"]["status"] == "queued"

    model_calls = {"count": 0}

    def scripted_model(messages):
        model_calls["count"] += 1
        n = model_calls["count"]
        if n == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_step_1",
                        "name": "search_events",
                        "args": {"limit": 5},
                        "type": "tool_call",
                    }
                ],
            )
        if n == 2:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_step_2",
                        "name": "search_events",
                        "args": {"severity": "ERROR", "limit": 5},
                        "type": "tool_call",
                    }
                ],
            )
        return AIMessage(
            content="Confirmed Facts: Completed investigation across 3 bounded invocations."
        )

    def _bounded_claim_and_execute(conn, **kwargs):
        return real_claim_and_execute(
            conn,
            run_id=kwargs.get("run_id"),
            settings=kwargs.get("settings"),
            model=RunnableLambda(scripted_model),
            max_iterations_per_invocation=1,
            commit_step=kwargs.get("commit_step"),
        )

    monkeypatch.setattr(conv_routes, "claim_and_execute_run", _bounded_claim_and_execute)

    # Invocation 1: executes iteration 1 (list_incidents) and yields back to 'queued'
    adv1 = client.post(
        f"/projects/{pid}/conversations/{cid}/runs/{run_id}/advance",
        headers=csrf(),
    )
    assert adv1.status_code == 200
    d1 = adv1.json()
    assert d1["status"] == "queued"
    assert d1["attempt_count"] == 0  # Normal yield does not consume crash retry attempts
    assert model_calls["count"] == 1
    assert any(s["tool_name"] == "search_events" and s["step_type"] == "tool_completed" for s in d1["steps"])

    # Simulate another tab holding an active lease: concurrent /advance must NOT invoke model
    claimed_other = repositories_agent.claim_run(db, worker_id="other-tab-worker", lease_seconds=60, run_id=run_id)
    assert claimed_other is not None and claimed_other["status"] == "running"

    adv_concurrent = client.post(
        f"/projects/{pid}/conversations/{cid}/runs/{run_id}/advance",
        params={"after_step": d1["latest_step_number"]},
        headers=csrf(),
    )
    assert adv_concurrent.status_code == 200
    assert adv_concurrent.json()["status"] == "running"
    assert model_calls["count"] == 1  # Model was NOT called while leased!

    # Simulate lease expiration from the other tab (e.g. function timeout) -> next /advance recovers!
    db.execute(
        "UPDATE public.agent_runs SET lease_expires_at = clock_timestamp() - interval '5 seconds', attempt_count = 0 WHERE id = %s",
        (run_id,),
    )

    # Invocation 2: recovers expired lease, executes iteration 2 (search_events), yields back to 'queued'
    adv2 = client.post(
        f"/projects/{pid}/conversations/{cid}/runs/{run_id}/advance",
        params={"after_step": d1["latest_step_number"]},
        headers=csrf(),
    )
    assert adv2.status_code == 200
    d2 = adv2.json()
    assert d2["status"] == "queued"
    assert model_calls["count"] == 2
    assert any(s["tool_name"] == "search_events" and s["step_type"] == "tool_completed" for s in d2["steps"])

    # Invocation 3: executes iteration 3 (final answer) and transitions run to 'completed'
    adv3 = client.post(
        f"/projects/{pid}/conversations/{cid}/runs/{run_id}/advance",
        params={"after_step": d2["latest_step_number"]},
        headers=csrf(),
    )
    assert adv3.status_code == 200
    d3 = adv3.json()
    assert d3["status"] == "completed"
    assert model_calls["count"] == 3
    assert len(d3["assistant_messages"]) == 1
    assert "3 bounded invocations" in d3["assistant_messages"][0]["content"]

    # Invocation 4 on already-completed run: must never execute again
    adv4 = client.post(
        f"/projects/{pid}/conversations/{cid}/runs/{run_id}/advance",
        headers=csrf(),
    )
    assert adv4.status_code == 200
    assert adv4.json()["status"] == "completed"
    assert model_calls["count"] == 3

    # Verify GET .../runs/{run_id} never calls claim_and_execute_run (strictly read-only polling)
    called_on_get = {"flag": False}
    def _fail_if_called(*args, **kwargs):
        called_on_get["flag"] = True
        raise AssertionError("claim_and_execute_run must never be invoked from GET!")
    monkeypatch.setattr(conv_routes, "claim_and_execute_run", _fail_if_called)

    sub_get_check = client.post(
        f"/projects/{pid}/conversations/{cid}/messages",
        json={"content": "Verify GET does not execute"},
        headers=csrf(),
    ).json()
    poll_ro = client.get(
        f"/projects/{pid}/conversations/{cid}/runs/{sub_get_check['run_id']}"
    )
    assert poll_ro.status_code == 200
    assert poll_ro.json()["status"] == "queued"
    assert called_on_get["flag"] is False

    # Cross-project access to /advance is rejected with 404
    other_proj = client.post(
        "/projects",
        json={"project_id": "proj-other-adv", "name": "Other Adv"},
        headers=csrf(),
    ).json()
    assert client.post(
        f"/projects/{other_proj['id']}/conversations/{cid}/runs/{run_id}/advance",
        headers=csrf(),
    ).status_code == 404
