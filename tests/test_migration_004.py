"""Migration 004 tests: schema, composite FK, role integrity, RLS, timestamps, and cascading deletes."""

import psycopg
import pytest

from app import repositories, repositories_agent
from tests.conftest import MIGRATIONS, make_user


def test_migration_004_tables_and_rls_enabled(db):
    rows = db.execute(
        """
        SELECT c.relname AS table_name, c.relrowsecurity AS rls_enabled
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'public'
          AND c.relname IN ('conversations', 'conversation_messages', 'agent_runs', 'agent_run_steps')
        ORDER BY c.relname
        """
    ).fetchall()
    by_name = {r["table_name"]: r["rls_enabled"] for r in rows}
    assert set(by_name) == {
        "conversations",
        "conversation_messages",
        "agent_runs",
        "agent_run_steps",
    }
    assert all(by_name.values()), f"RLS must be enabled on all 4 tables: {by_name}"

    # Verify backend connection role has bypassrls or appropriate privileges so queries work without permissive policies
    role_row = db.execute(
        """
        SELECT current_user, r.rolsuper, r.rolbypassrls
        FROM pg_roles r
        WHERE r.rolname = current_user
        """
    ).fetchone()
    assert role_row is not None
    assert role_row["rolsuper"] or role_row["rolbypassrls"]

    # Verify no permissive RLS policies were added to the 4 tables
    policy_count = db.execute(
        """
        SELECT count(*) AS n
        FROM pg_policies
        WHERE schemaname = 'public'
          AND tablename IN ('conversations', 'conversation_messages', 'agent_runs', 'agent_run_steps')
        """
    ).fetchone()["n"]
    assert policy_count == 0


def test_migration_004_is_idempotent_and_has_no_redundant_single_column_user_message_fk(db):
    m004_sql = MIGRATIONS[3]
    # Re-applying migration 004 multiple times inside the transaction must succeed cleanly.
    db.execute(m004_sql)
    db.execute(m004_sql)

    # Verify composite unique constraint on conversation_messages(id, conversation_id)
    unique_row = db.execute(
        """
        SELECT conname
        FROM pg_constraint
        WHERE conrelid = 'public.conversation_messages'::regclass
          AND conname = 'conversation_messages_id_conversation_unique'
        """
    ).fetchone()
    assert unique_row is not None

    # Verify composite FK on agent_runs(user_message_id, conversation_id) exists
    # and redundant single-column agent_runs_user_message_id_fkey does NOT exist
    fk_rows = db.execute(
        """
        SELECT conname
        FROM pg_constraint
        WHERE conrelid = 'public.agent_runs'::regclass
          AND contype = 'f'
        ORDER BY conname
        """
    ).fetchall()
    fk_names = {r["conname"] for r in fk_rows}
    assert "agent_runs_user_message_conversation_fkey" in fk_names
    assert "agent_runs_user_message_id_fkey" not in fk_names


def test_migration_004_check_and_unique_constraints(db):
    user = make_user(db)
    proj = repositories.insert_project(db, user["id"], "proj-m004-1", "proj-m004-1")
    conv = repositories_agent.create_conversation(
        db,
        user_id=user["id"],
        project_id=proj["id"],
        title="Test constraints",
    )

    # 1. Invalid role rejected
    with pytest.raises(psycopg.errors.CheckViolation):
        with db.transaction():
            db.execute(
                """
                INSERT INTO public.conversation_messages
                    (conversation_id, sequence_number, role, message_type, content)
                VALUES (%s, 1, 'system', 'text', 'not allowed')
                """,
                (conv["id"],),
            )

    # 2. Invalid message_type rejected
    with pytest.raises(psycopg.errors.CheckViolation):
        with db.transaction():
            db.execute(
                """
                INSERT INTO public.conversation_messages
                    (conversation_id, sequence_number, role, message_type, content)
                VALUES (%s, 1, 'user', 'invalid_type', 'not allowed')
                """,
                (conv["id"],),
            )

    # 3. Duplicate (conversation_id, sequence_number) rejected
    msg = repositories_agent.append_conversation_message(
        db,
        conversation_id=conv["id"],
        role="user",
        message_type="text",
        content="first message",
    )
    with pytest.raises(psycopg.errors.UniqueViolation):
        with db.transaction():
            db.execute(
                """
                INSERT INTO public.conversation_messages
                    (conversation_id, sequence_number, role, message_type, content)
                VALUES (%s, %s, 'user', 'text', 'duplicate seq')
                """,
                (conv["id"], msg["sequence_number"]),
            )

    # 4. At most one active (queued/running) run per conversation enforced by partial unique index
    db.execute(
        """
        INSERT INTO public.agent_runs (conversation_id, user_message_id, status)
        VALUES (%s, %s, 'queued')
        """,
        (conv["id"], msg["id"]),
    )
    with pytest.raises(psycopg.errors.UniqueViolation):
        with db.transaction():
            db.execute(
                """
                INSERT INTO public.agent_runs (conversation_id, user_message_id, status)
                VALUES (%s, %s, 'running')
                """,
                (conv["id"], msg["id"]),
            )


def test_run_cannot_reference_user_message_from_another_conversation(db):
    user = make_user(db)
    proj = repositories.insert_project(db, user["id"], "proj-m004-cross-conv", "proj-m004-cross-conv")
    conv_a = repositories_agent.create_conversation(
        db,
        user_id=user["id"],
        project_id=proj["id"],
        title="Conversation A",
    )
    conv_b = repositories_agent.create_conversation(
        db,
        user_id=user["id"],
        project_id=proj["id"],
        title="Conversation B",
    )
    msg_b = repositories_agent.append_conversation_message(
        db,
        conversation_id=conv_b["id"],
        role="user",
        message_type="text",
        content="User message in conversation B",
    )

    # 1. Repository layer rejects cross-conversation user_message_id
    with pytest.raises(ValueError, match="Initiating message does not belong to this conversation"):
        repositories_agent.create_run_for_user_message(
            db,
            conversation_id=conv_a["id"],
            user_message_id=msg_b["id"],
        )

    # 2. PostgreSQL composite FK (agent_runs_user_message_conversation_fkey) rejects raw SQL insert
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        with db.transaction():
            db.execute(
                """
                INSERT INTO public.agent_runs (conversation_id, user_message_id, status)
                VALUES (%s, %s, 'queued')
                """,
                (conv_a["id"], msg_b["id"]),
            )


def test_run_cannot_use_non_user_message_as_initiating_message(db):
    user = make_user(db)
    proj = repositories.insert_project(db, user["id"], "proj-m004-role-check", "proj-m004-role-check")
    conv = repositories_agent.create_conversation(
        db,
        user_id=user["id"],
        project_id=proj["id"],
        title="Role enforcement test",
    )

    assistant_text = repositories_agent.append_conversation_message(
        db,
        conversation_id=conv["id"],
        role="assistant",
        message_type="text",
        content="Assistant response",
    )
    assistant_tool_call = repositories_agent.append_conversation_message(
        db,
        conversation_id=conv["id"],
        role="assistant",
        message_type="tool_call",
        tool_calls=[{"id": "call_1", "name": "list_incidents", "args": {}}],
    )
    tool_result = repositories_agent.append_conversation_message(
        db,
        conversation_id=conv["id"],
        role="tool",
        message_type="tool_result",
        content='{"ok": true}',
        tool_call_id="call_1",
        tool_name="list_incidents",
    )

    for non_user_msg in (assistant_text, assistant_tool_call, tool_result):
        # 1. Repository layer rejects non-user message
        with pytest.raises(ValueError, match="Agent run can only be initiated by a user text message"):
            repositories_agent.create_run_for_user_message(
                db,
                conversation_id=conv["id"],
                user_message_id=non_user_msg["id"],
            )

        # 2. Database trigger rejects raw SQL insert referencing non-user message
        with pytest.raises(psycopg.errors.CheckViolation):
            with db.transaction():
                db.execute(
                    """
                    INSERT INTO public.agent_runs (conversation_id, user_message_id, status)
                    VALUES (%s, %s, 'queued')
                    """,
                    (conv["id"], non_user_msg["id"]),
                )


def test_conversation_and_run_timestamp_lifecycle(db):
    user = make_user(db)
    proj = repositories.insert_project(db, user["id"], "proj-m004-ts", "proj-m004-ts")
    conv0 = repositories_agent.create_conversation(
        db,
        user_id=user["id"],
        project_id=proj["id"],
        title=None,
    )

    # 1. Visible user message updates both updated_at and last_message_at
    user_msg, run0, _ = repositories_agent.submit_user_message_and_create_run(
        db,
        user_id=user["id"],
        project_id=proj["id"],
        conversation_id=conv0["id"],
        content="Investigate payment-service timeouts",
    )
    conv1 = repositories_agent.get_conversation(
        db,
        user_id=user["id"],
        project_id=proj["id"],
        conversation_id=conv0["id"],
    )
    assert conv1["updated_at"] > conv0["updated_at"]
    assert conv1["last_message_at"] > conv0["last_message_at"]

    # 2. Internal tool_call / tool_result messages update updated_at but preserve last_message_at
    repositories_agent.append_conversation_message(
        db,
        conversation_id=conv0["id"],
        role="assistant",
        message_type="tool_call",
        tool_calls=[{"id": "tc_1", "name": "list_incidents", "args": {}}],
        run_id=run0["id"],
    )
    conv2 = repositories_agent.get_conversation(
        db,
        user_id=user["id"],
        project_id=proj["id"],
        conversation_id=conv0["id"],
    )
    assert conv2["updated_at"] > conv1["updated_at"]
    assert conv2["last_message_at"] == conv1["last_message_at"]

    # 3. Visible assistant text message updates both updated_at and last_message_at
    repositories_agent.append_conversation_message(
        db,
        conversation_id=conv0["id"],
        role="assistant",
        message_type="text",
        content="Found 2 open incidents in payment-service.",
        run_id=run0["id"],
    )
    conv3 = repositories_agent.get_conversation(
        db,
        user_id=user["id"],
        project_id=proj["id"],
        conversation_id=conv0["id"],
    )
    assert conv3["updated_at"] > conv2["updated_at"]
    assert conv3["last_message_at"] > conv2["last_message_at"]

    # 4. Updating conversation title advances updated_at without changing last_message_at
    conv4 = repositories_agent.update_conversation_title(
        db,
        user_id=user["id"],
        project_id=proj["id"],
        conversation_id=conv0["id"],
        title="Payment-service timeout root cause",
    )
    assert conv4 is not None
    assert conv4["title"] == "Payment-service timeout root cause"
    assert conv4["updated_at"] > conv3["updated_at"]
    assert conv4["last_message_at"] == conv3["last_message_at"]

    # 5. Run updated_at strictly advances on every status transition
    claimed1 = repositories_agent.claim_run(db, worker_id="w-1", run_id=run0["id"])
    assert claimed1["status"] == "running"
    assert claimed1["updated_at"] > run0["updated_at"]

    assert repositories_agent.heartbeat_run(db, run_id=run0["id"], worker_id="w-1") is True
    hb_run = repositories_agent.get_run(db, conversation_id=conv0["id"], run_id=run0["id"])
    assert hb_run["updated_at"] > claimed1["updated_at"]

    retried = repositories_agent.fail_or_retry_run(
        db,
        run_id=run0["id"],
        worker_id="w-1",
        error_message="transient error",
        retryable=True,
        retry_delay_seconds=1,
    )
    assert retried["status"] == "queued"
    assert retried["updated_at"] > hb_run["updated_at"]

    # Make retry immediately claimable
    db.execute(
        "UPDATE public.agent_runs SET next_retry_at = clock_timestamp() - interval '1 second' WHERE id = %s",
        (run0["id"],),
    )
    claimed2 = repositories_agent.claim_run(db, worker_id="w-2", run_id=run0["id"])
    assert claimed2["status"] == "running"
    assert claimed2["updated_at"] > retried["updated_at"]

    completed = repositories_agent.complete_run(db, run_id=run0["id"], worker_id="w-2")
    assert completed["status"] == "completed"
    assert completed["updated_at"] > claimed2["updated_at"]
    assert completed["completed_at"] is not None


def test_deleting_conversation_project_and_user_cascades_all_message_and_run_types(db):
    user = make_user(db)
    proj = repositories.insert_project(db, user["id"], "proj-m004-2", "proj-m004-2")

    # Create an incident and telemetry event in this project
    inc = db.execute(
        """
        INSERT INTO public.incidents (project_id, title, status, severity, fingerprint, event_count)
        VALUES (%s, 'ValueError: boom', 'open', 'ERROR', 'fp-1', 1)
        RETURNING id
        """,
        (proj["id"],),
    ).fetchone()
    evt = db.execute(
        """
        INSERT INTO public.telemetry_events (
            project_id, incident_id, event_id, schema_version, event_type,
            classification, event_timestamp, severity, message, payload
        )
        VALUES (%s, %s, 'evt-keep-1', '1.0', 'error', 'abnormal_behaviour', now(), 'ERROR', 'boom', '{}'::jsonb)
        RETURNING id
        """,
        (proj["id"], inc["id"]),
    ).fetchone()

    conv = repositories_agent.create_conversation(
        db,
        user_id=user["id"],
        project_id=proj["id"],
        title="Investigate ValueError",
    )

    # Turn 1: User message -> claimed run -> tool_call -> tool_result -> assistant text -> completed run
    _, run1, _ = repositories_agent.submit_user_message_and_create_run(
        db,
        user_id=user["id"],
        project_id=proj["id"],
        conversation_id=conv["id"],
        content="Why did ValueError happen?",
        client_request_id="req-cascade-1",
    )
    repositories_agent.claim_run(db, worker_id="w-cascade-1", run_id=run1["id"])
    step1 = repositories_agent.append_run_step(
        db,
        run_id=run1["id"],
        step_type="model_started",
        status="in_progress",
        input_summary={"iteration": 1},
    )
    repositories_agent.append_conversation_message(
        db,
        conversation_id=conv["id"],
        role="assistant",
        message_type="tool_call",
        tool_calls=[{"id": "call_1", "name": "get_incident_detail", "args": {"incident_id": str(inc["id"])}}],
        run_id=run1["id"],
    )
    repositories_agent.append_conversation_message(
        db,
        conversation_id=conv["id"],
        role="tool",
        message_type="tool_result",
        content='{"ok": true, "data": {}}',
        tool_call_id="call_1",
        tool_name="get_incident_detail",
        run_id=run1["id"],
    )
    repositories_agent.append_conversation_message(
        db,
        conversation_id=conv["id"],
        role="assistant",
        message_type="text",
        content="Confirmed root cause in parser.",
        run_id=run1["id"],
    )
    repositories_agent.complete_run(db, run_id=run1["id"], worker_id="w-cascade-1")

    # Turn 2: Second user message -> claimed run -> failed run + failed step
    _, run2, _ = repositories_agent.submit_user_message_and_create_run(
        db,
        user_id=user["id"],
        project_id=proj["id"],
        conversation_id=conv["id"],
        content="Follow-up question that fails",
        client_request_id="req-cascade-2",
    )
    repositories_agent.claim_run(db, worker_id="w-cascade-2", run_id=run2["id"])
    step2 = repositories_agent.append_run_step(
        db,
        run_id=run2["id"],
        step_type="run_failed",
        status="failed",
        output_summary={"message": "Model provider unavailable"},
    )
    repositories_agent.fail_or_retry_run(
        db,
        run_id=run2["id"],
        worker_id="w-cascade-2",
        error_message="Model provider unavailable",
        retryable=False,
    )

    # Verify all 5 messages, 2 runs (completed + failed), and 2 steps exist before deletion
    assert db.execute(
        "SELECT count(*) AS n FROM public.conversation_messages WHERE conversation_id = %s",
        (conv["id"],),
    ).fetchone()["n"] == 5
    assert db.execute(
        "SELECT count(*) AS n FROM public.agent_runs WHERE conversation_id = %s",
        (conv["id"],),
    ).fetchone()["n"] == 2

    # 1. Delete conversation -> cleans up all messages, runs, and steps without FK violations
    assert repositories_agent.delete_conversation(
        db,
        user_id=user["id"],
        project_id=proj["id"],
        conversation_id=conv["id"],
    ) is True

    assert db.execute(
        "SELECT count(*) AS n FROM public.conversation_messages WHERE conversation_id = %s",
        (conv["id"],),
    ).fetchone()["n"] == 0
    assert db.execute(
        "SELECT count(*) AS n FROM public.agent_runs WHERE id IN (%s, %s)",
        (run1["id"], run2["id"]),
    ).fetchone()["n"] == 0
    assert db.execute(
        "SELECT count(*) AS n FROM public.agent_run_steps WHERE id IN (%s, %s)",
        (step1["id"], step2["id"]),
    ).fetchone()["n"] == 0

    # Project, incident, and telemetry event remain completely intact
    assert repositories.get_project_for_user(db, user["id"], proj["id"]) is not None
    assert db.execute("SELECT id FROM public.incidents WHERE id = %s", (inc["id"],)).fetchone() is not None
    assert db.execute("SELECT id FROM public.telemetry_events WHERE id = %s", (evt["id"],)).fetchone() is not None

    # 2. Verify deleting a project cascades through its conversations, messages, runs, and steps
    conv_proj = repositories_agent.create_conversation(
        db,
        user_id=user["id"],
        project_id=proj["id"],
        title="Project cascade conversation",
    )
    _, run_proj, _ = repositories_agent.submit_user_message_and_create_run(
        db,
        user_id=user["id"],
        project_id=proj["id"],
        conversation_id=conv_proj["id"],
        content="Will be deleted with project",
    )
    step_proj = repositories_agent.append_run_step(
        db,
        run_id=run_proj["id"],
        step_type="model_started",
        status="in_progress",
    )
    db.execute("DELETE FROM public.projects WHERE id = %s", (proj["id"],))
    assert db.execute("SELECT count(*) AS n FROM public.conversations WHERE id = %s", (conv_proj["id"],)).fetchone()["n"] == 0
    assert db.execute("SELECT count(*) AS n FROM public.agent_runs WHERE id = %s", (run_proj["id"],)).fetchone()["n"] == 0
    assert db.execute("SELECT count(*) AS n FROM public.agent_run_steps WHERE id = %s", (step_proj["id"],)).fetchone()["n"] == 0

    # 3. Verify deleting a user cascades through projects -> conversations -> messages -> runs -> steps
    proj_u = repositories.insert_project(db, user["id"], "proj-m004-u", "proj-m004-u")
    conv_u = repositories_agent.create_conversation(
        db,
        user_id=user["id"],
        project_id=proj_u["id"],
        title="User cascade conversation",
    )
    _, run_u, _ = repositories_agent.submit_user_message_and_create_run(
        db,
        user_id=user["id"],
        project_id=proj_u["id"],
        conversation_id=conv_u["id"],
        content="Will be deleted with user",
    )
    db.execute("DELETE FROM public.users WHERE id = %s", (user["id"],))
    assert db.execute("SELECT count(*) AS n FROM public.conversations WHERE id = %s", (conv_u["id"],)).fetchone()["n"] == 0
    assert db.execute("SELECT count(*) AS n FROM public.agent_runs WHERE id = %s", (run_u["id"],)).fetchone()["n"] == 0
