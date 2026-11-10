"""Migration 005 tests: running lease invariant, pre-005 malformed row defense & remediation,
idempotency, CHECK constraint enforcement, and real multi-connection claim concurrency.
"""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import threading
import uuid

import psycopg
import pytest
from psycopg.rows import dict_row

from app import repositories, repositories_agent
from tests.conftest import BASE_SCHEMA, MIGRATIONS

M001, M002, M003, M004, M005 = MIGRATIONS[:5]


@pytest.fixture
def make_db(test_database):
    """Create isolated databases on the throwaway local Postgres server so tests can commit and upgrade cleanly."""
    server_url, throwaway = test_database
    if not throwaway:
        pytest.skip("isolated migration/concurrency tests only run on a throwaway test database")
    created = []

    @contextmanager
    def build(*scripts):
        name = f"mig005_{uuid.uuid4().hex[:10]}"
        with psycopg.connect(server_url, autocommit=True) as admin:
            admin.execute(f'create database "{name}"')
        created.append(name)
        url = server_url.rsplit("/", 1)[0] + "/" + name
        with psycopg.connect(url, autocommit=True, row_factory=dict_row, prepare_threshold=None) as conn:
            for script in scripts:
                conn.execute(script)
            yield conn, url

    yield build
    with psycopg.connect(server_url, autocommit=True) as admin:
        for name in created:
            admin.execute(f'drop database if exists "{name}" with (force)')


def _seed_conv(conn, *, key_suffix: str):
    user = repositories.upsert_user(
        conn,
        github_id=100000 + abs(hash(key_suffix)) % 800000,
        username=f"user-{key_suffix}",
        email=None,
        avatar_url=None,
    )
    proj = repositories.insert_project(conn, user["id"], f"proj-{key_suffix}", f"Proj {key_suffix}")
    conv = repositories_agent.create_conversation(
        conn,
        user_id=user["id"],
        project_id=proj["id"],
        title=f"Conv {key_suffix}",
    )
    return user, proj, conv


def test_pre_005_malformed_null_lease_rejected_for_writes_and_safely_reclaimed(make_db):
    """Even against a pre-Migration-005 schema where a malformed running row has NULL lease_expires_at:
    - require_run_lease raises LeaseLostError
    - heartbeat_run returns False without granting a future lease
    - append_conversation_message and append_run_step both reject writes cleanly
    - claim_run can safely recover a malformed running row for a new worker (setting non-null future lease)
    - expire_stale_runs can safely fail an exhausted malformed running row
    """
    with make_db(BASE_SCHEMA, M001, M002, M003, M004) as (conn, _):
        _, _, conv = _seed_conv(conn, key_suffix="malformed-rec")
        u_msg = repositories_agent.append_conversation_message(
            conn,
            conversation_id=conv["id"],
            role="user",
            message_type="text",
            content="Recoverable malformed run",
        )
        run_rec = conn.execute(
            """
            INSERT INTO public.agent_runs (
                conversation_id, user_message_id, status, worker_id, lease_expires_at, attempt_count, max_attempts
            )
            VALUES (%s, %s, 'running', 'old-worker', NULL, 1, 3)
            RETURNING id
            """,
            (conv["id"], u_msg["id"]),
        ).fetchone()
        run_id = run_rec["id"]

        # 1. Null lease_expires_at fails require_run_lease
        with pytest.raises(repositories_agent.LeaseLostError):
            repositories_agent.require_run_lease(conn, run_id=run_id, worker_id="old-worker")

        # 2. Stale worker cannot heartbeat a null lease into a valid lease
        assert repositories_agent.heartbeat_run(conn, run_id=run_id, worker_id="old-worker", lease_seconds=90) is False
        row_after_hb = repositories_agent.get_run(conn, conversation_id=conv["id"], run_id=run_id)
        assert row_after_hb["lease_expires_at"] is None
        assert row_after_hb["status"] == "running"

        # 3. Stale worker cannot persist messages or steps when lease_expires_at is NULL
        with pytest.raises(repositories_agent.LeaseLostError):
            repositories_agent.append_conversation_message(
                conn,
                conversation_id=conv["id"],
                role="assistant",
                message_type="text",
                content="Should not persist on null lease",
                run_id=run_id,
                expected_worker_id="old-worker",
            )
        with pytest.raises(repositories_agent.LeaseLostError):
            repositories_agent.append_run_step(
                conn,
                run_id=run_id,
                step_type="run_completed",
                status="completed",
                expected_worker_id="old-worker",
            )
        assert repositories_agent.list_assistant_messages_for_run(conn, conversation_id=conv["id"], run_id=run_id) == []
        steps, _ = repositories_agent.list_run_steps(conn, run_id=run_id, after_step=0)
        assert steps == []

        # 4. Another worker safely reclaims the malformed running row and gets a non-null future lease
        claimed = repositories_agent.claim_run(conn, worker_id="new-worker", lease_seconds=90, run_id=run_id)
        assert claimed is not None
        assert claimed["status"] == "running"
        assert claimed["worker_id"] == "new-worker"
        assert claimed["lease_expires_at"] is not None
        assert claimed["attempt_count"] == 2
        clock_check = conn.execute(
            "SELECT lease_expires_at > clock_timestamp() AS valid FROM public.agent_runs WHERE id = %s",
            (run_id,),
        ).fetchone()
        assert clock_check["valid"] is True

        # Old worker still cannot write; new worker passes require_run_lease
        with pytest.raises(repositories_agent.LeaseLostError):
            repositories_agent.require_run_lease(conn, run_id=run_id, worker_id="old-worker")
        repositories_agent.require_run_lease(conn, run_id=run_id, worker_id="new-worker")

        # 5. Exhausted malformed row (attempt_count >= max_attempts, NULL lease) is failed cleanly by expire_stale_runs
        _, _, conv2 = _seed_conv(conn, key_suffix="malformed-exh")
        u_msg2 = repositories_agent.append_conversation_message(
            conn,
            conversation_id=conv2["id"],
            role="user",
            message_type="text",
            content="Exhausted malformed run",
        )
        run_exh = conn.execute(
            """
            INSERT INTO public.agent_runs (
                conversation_id, user_message_id, status, worker_id, lease_expires_at, attempt_count, max_attempts
            )
            VALUES (%s, %s, 'running', 'dead-worker', NULL, 3, 3)
            RETURNING id
            """,
            (conv2["id"], u_msg2["id"]),
        ).fetchone()
        expired = repositories_agent.expire_stale_runs(conn, run_id=run_exh["id"])
        assert len(expired) == 1
        assert expired[0]["status"] == "failed"
        assert expired[0]["worker_id"] is None
        assert expired[0]["lease_expires_at"] is None


def test_migration_005_remediates_malformed_rows_enforces_constraint_and_is_idempotent(make_db):
    """Verify upgrading a database containing valid and malformed rows:
    1. Leaves valid unrelated runs intact.
    2. Safely transitions recoverable malformed running rows to queued and exhausted ones to failed.
    3. Adds agent_runs_running_requires_lease_check rejecting NULL lease_expires_at or worker_id on running rows.
    4. Succeeds cleanly on repeat execution.
    """
    with make_db(BASE_SCHEMA, M001, M002, M003, M004) as (conn, _):
        # Conv 1: valid completed run (must remain completely untouched, even with legacy worker_id)
        _, _, c1 = _seed_conv(conn, key_suffix="valid-completed")
        m1 = repositories_agent.append_conversation_message(
            conn, conversation_id=c1["id"], role="user", message_type="text", content="Valid completed"
        )
        r_completed = conn.execute(
            """
            INSERT INTO public.agent_runs (
                conversation_id, user_message_id, status, worker_id, lease_expires_at, completed_at
            )
            VALUES (%s, %s, 'completed', 'legacy-completed-worker', NULL, clock_timestamp())
            RETURNING *
            """,
            (c1["id"], m1["id"]),
        ).fetchone()

        # Conv 2: valid active running run (must remain running with identical worker_id and lease_expires_at)
        _, _, c2 = _seed_conv(conn, key_suffix="valid-running")
        m2 = repositories_agent.append_conversation_message(
            conn, conversation_id=c2["id"], role="user", message_type="text", content="Valid running"
        )
        r_valid_running = conn.execute(
            """
            INSERT INTO public.agent_runs (
                conversation_id, user_message_id, status, worker_id, lease_expires_at, attempt_count, max_attempts
            )
            VALUES (%s, %s, 'running', 'active-worker', clock_timestamp() + interval '120 seconds', 1, 3)
            RETURNING *
            """,
            (c2["id"], m2["id"]),
        ).fetchone()

        # Conv 3: recoverable malformed running run (NULL lease_expires_at, attempt_count=1 < max_attempts=3)
        _, _, c3 = _seed_conv(conn, key_suffix="malformed-mig-rec")
        m3 = repositories_agent.append_conversation_message(
            conn, conversation_id=c3["id"], role="user", message_type="text", content="Malformed recoverable"
        )
        r_mal_rec = conn.execute(
            """
            INSERT INTO public.agent_runs (
                conversation_id, user_message_id, status, worker_id, lease_expires_at, attempt_count, max_attempts
            )
            VALUES (%s, %s, 'running', 'zombie-worker', NULL, 1, 3)
            RETURNING id
            """,
            (c3["id"], m3["id"]),
        ).fetchone()

        # Conv 4: exhausted malformed running run (NULL worker_id, attempt_count=3 == max_attempts=3)
        _, _, c4 = _seed_conv(conn, key_suffix="malformed-mig-exh")
        m4 = repositories_agent.append_conversation_message(
            conn, conversation_id=c4["id"], role="user", message_type="text", content="Malformed exhausted"
        )
        r_mal_exh = conn.execute(
            """
            INSERT INTO public.agent_runs (
                conversation_id, user_message_id, status, worker_id, lease_expires_at, attempt_count, max_attempts
            )
            VALUES (%s, %s, 'running', NULL, clock_timestamp() + interval '30 seconds', 3, 3)
            RETURNING id
            """,
            (c4["id"], m4["id"]),
        ).fetchone()

        # Apply Migration 005
        conn.execute(M005)

        # Unrelated runs were not modified
        after_completed = repositories_agent.get_run(conn, conversation_id=c1["id"], run_id=r_completed["id"])
        assert after_completed["status"] == "completed"
        assert after_completed["worker_id"] == "legacy-completed-worker"
        assert after_completed["updated_at"] == r_completed["updated_at"]

        after_valid_running = repositories_agent.get_run(conn, conversation_id=c2["id"], run_id=r_valid_running["id"])
        assert after_valid_running["status"] == "running"
        assert after_valid_running["worker_id"] == "active-worker"
        assert after_valid_running["lease_expires_at"] == r_valid_running["lease_expires_at"]

        # Recoverable malformed row -> queued with cleared lease fields, and claimable by a worker
        after_rec = repositories_agent.get_run(conn, conversation_id=c3["id"], run_id=r_mal_rec["id"])
        assert after_rec["status"] == "queued"
        assert after_rec["worker_id"] is None
        assert after_rec["lease_expires_at"] is None
        assert after_rec["attempt_count"] == 1
        reclaimed = repositories_agent.claim_run(conn, worker_id="post-mig-worker", lease_seconds=90, run_id=r_mal_rec["id"])
        assert reclaimed is not None and reclaimed["status"] == "running"
        assert reclaimed["lease_expires_at"] is not None

        # Exhausted malformed row -> failed with cleared lease fields + run_failed step
        after_exh = repositories_agent.get_run(conn, conversation_id=c4["id"], run_id=r_mal_exh["id"])
        assert after_exh["status"] == "failed"
        assert after_exh["worker_id"] is None
        assert after_exh["lease_expires_at"] is None
        assert after_exh["completed_at"] is not None
        exh_steps, _ = repositories_agent.list_run_steps(conn, run_id=r_mal_exh["id"], after_step=0)
        assert len(exh_steps) == 1 and exh_steps[0]["step_type"] == "run_failed"

        # DB CHECK constraint rejects NULL lease_expires_at or NULL worker_id when status='running'
        _, _, c5 = _seed_conv(conn, key_suffix="constraint-check")
        m5 = repositories_agent.append_conversation_message(
            conn, conversation_id=c5["id"], role="user", message_type="text", content="Constraint test"
        )
        with pytest.raises(psycopg.errors.CheckViolation):
            conn.execute(
                """
                INSERT INTO public.agent_runs (conversation_id, user_message_id, status, worker_id, lease_expires_at)
                VALUES (%s, %s, 'running', 'w-no-lease', NULL)
                """,
                (c5["id"], m5["id"]),
            )
        with pytest.raises(psycopg.errors.CheckViolation):
            conn.execute(
                """
                INSERT INTO public.agent_runs (conversation_id, user_message_id, status, worker_id, lease_expires_at)
                VALUES (%s, %s, 'running', NULL, clock_timestamp() + interval '60 seconds')
                """,
                (c5["id"], m5["id"]),
            )

        # Updating an existing queued run to running without lease_expires_at is also rejected
        valid_queued = repositories_agent.create_run_for_user_message(
            conn, conversation_id=c5["id"], user_message_id=m5["id"]
        )
        with pytest.raises(psycopg.errors.CheckViolation):
            conn.execute(
                "UPDATE public.agent_runs SET status = 'running', worker_id = 'w-bad' WHERE id = %s",
                (valid_queued["id"],),
            )

        # Re-applying Migration 005 is completely idempotent
        conn.execute(M005)
        conn.execute(M005)


def test_concurrent_claim_run_allows_only_one_worker_across_real_connections(make_db):
    """Two concurrent PostgreSQL sessions racing `claim_run` on the same queued run:
    `FOR UPDATE SKIP LOCKED` must guarantee strictly one winner receives a valid future lease
    while the losing worker receives None.
    """
    with make_db(BASE_SCHEMA, M001, M002, M003, M004, M005) as (conn, db_url):
        user, proj, conv = _seed_conv(conn, key_suffix="concurrent-claim")
        _, run, _ = repositories_agent.submit_user_message_and_create_run(
            conn,
            user_id=user["id"],
            project_id=proj["id"],
            conversation_id=conv["id"],
            content="Concurrent claim race test",
        )
        run_id = run["id"]

        barrier = threading.Barrier(2)

        def worker_attempt(worker_name: str):
            with psycopg.connect(db_url, row_factory=dict_row, autocommit=True, prepare_threshold=None) as wconn:
                barrier.wait(timeout=5.0)
                return repositories_agent.claim_run(
                    wconn,
                    worker_id=worker_name,
                    lease_seconds=90,
                    run_id=run_id,
                )

        with ThreadPoolExecutor(max_workers=2) as pool:
            f1 = pool.submit(worker_attempt, "concurrent-w1")
            f2 = pool.submit(worker_attempt, "concurrent-w2")
            r1 = f1.result(timeout=10.0)
            r2 = f2.result(timeout=10.0)

        winners = [r for r in (r1, r2) if r is not None]
        losers = [r for r in (r1, r2) if r is None]
        assert len(winners) == 1
        assert len(losers) == 1
        winner = winners[0]
        assert winner["status"] == "running"
        assert winner["worker_id"] in ("concurrent-w1", "concurrent-w2")
        assert winner["lease_expires_at"] is not None
        assert winner["attempt_count"] == 1

