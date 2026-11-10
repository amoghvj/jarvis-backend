"""Durable PostgreSQL-backed worker and background execution coordinator for agent runs.

Supports three complementary execution modes using the same atomic `FOR UPDATE SKIP LOCKED`
database queue (`public.agent_runs`):

1. **Async Background Worker Thread (`trigger_background_run`)**:
   Spawned immediately after `POST .../messages` commits a queued run so local/container
   deployments process runs asynchronously without blocking the HTTP response.
2. **Poll-Assisted Lease Recovery (`recover_or_trigger_on_poll`)**:
   On Vercel serverless functions, if the `POST .../messages` container freezes after
   returning its HTTP response (leaving the run `queued` or with an expired `lease_expires_at`),
   subsequent `GET .../runs/{run_id}` polls detect the unclaimed/expired run and spawn or
   advance recovery.
3. **Standalone Persistent Worker Loop (`run_worker_loop` / `scripts/run_agent_worker.py`)**:
   Can run as a dedicated background worker process polling PostgreSQL via `FOR UPDATE SKIP LOCKED`.
"""

import logging
import os
import socket
import threading
import time
import uuid
from typing import Any, Callable, Optional
from uuid import UUID

from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable

from app import repositories_agent
from app.agent.history import FullHistoryManager, HistoryManager
from app.agent.model import create_groq_chat_model
from app.agent.orchestrator import AgentRunOutcome, run_agent
from app.config import Settings, get_settings
from app.database import open_connection
from app.tools.definitions import get_default_registry
from app.tools.errors import redact
from app.tools.registry import ToolRegistry
from app.tools.schemas import ToolContext

logger = logging.getLogger("jarvis.agent.worker")

_ACTIVE_LOCAL_RUNS_LOCK = threading.Lock()
_ACTIVE_LOCAL_RUNS: set[UUID] = set()


def generate_worker_id(prefix: str = "worker") -> str:
    host = socket.gethostname()[:32] or "host"
    pid = os.getpid()
    short = uuid.uuid4().hex[:8]
    return f"{prefix}:{host}:{pid}:{short}"


def claim_and_execute_run(
    conn: Any,
    *,
    run_id: Optional[UUID] = None,
    worker_id: Optional[str] = None,
    model: Optional[Runnable[Any, AIMessage]] = None,
    history_manager: Optional[HistoryManager] = None,
    tool_registry: Optional[ToolRegistry] = None,
    settings: Optional[Settings] = None,
    commit_step: Optional[Callable[[], None]] = None,
) -> Optional[AgentRunOutcome]:
    """Atomically claim the specified (or next eligible) run and execute it to completion."""
    cfg = settings or get_settings()
    wid = worker_id or generate_worker_id()
    commit = commit_step or (lambda: None)

    claimed = repositories_agent.claim_run(
        conn,
        worker_id=wid,
        lease_seconds=cfg.agent_lease_seconds,
        run_id=run_id,
    )
    commit()
    if claimed is None:
        return None

    claimed_run_id: UUID = claimed["id"]
    conversation_id: UUID = claimed["conversation_id"]
    user_id: UUID = claimed["user_id"]
    project_id: UUID = claimed["project_id"]

    reg = tool_registry or get_default_registry()
    hist = history_manager or FullHistoryManager()

    try:
        llm = model or create_groq_chat_model(cfg, registry=reg, bind_default_tools=True)
    except Exception as exc:
        safe_msg = redact(f"Failed to initialize language model: {exc}")
        repositories_agent.fail_or_retry_run(
            conn,
            run_id=claimed_run_id,
            worker_id=wid,
            error_message=safe_msg,
            retryable=False,
        )
        repositories_agent.append_run_step(
            conn,
            run_id=claimed_run_id,
            step_type="run_failed",
            status="failed",
            output_summary={"error_code": "model_init_error", "message": safe_msg},
        )
        commit()
        return AgentRunOutcome(
            run_id=claimed_run_id,
            conversation_id=conversation_id,
            status="failed",
            iterations=0,
            tool_calls_executed=0,
            error_message=safe_msg,
        )

    ctx = ToolContext(
        user_id=user_id,
        project_id=project_id,
        conn=conn,
    )

    return run_agent(
        conversation_id=conversation_id,
        run_id=claimed_run_id,
        worker_id=wid,
        history_manager=hist,
        model=llm,
        execution_context=ctx,
        tool_registry=reg,
        max_iterations=cfg.agent_max_iterations,
        run_timeout_seconds=cfg.agent_run_timeout_seconds,
        lease_seconds=cfg.agent_lease_seconds,
        max_tool_result_chars=cfg.agent_max_tool_result_chars,
        commit_step=commit,
    )


def _background_worker_entry(run_id: UUID) -> None:
    """Thread target that opens an independent DB connection and executes `run_id`."""
    try:
        cfg = get_settings()
        if not cfg.database_url:
            return
        with open_connection() as conn:
            claim_and_execute_run(
                conn,
                run_id=run_id,
                settings=cfg,
            )
    except Exception as exc:
        logger.error(
            "Background worker thread for run %s failed: %s",
            run_id,
            redact(f"{type(exc).__name__}: {exc}"),
        )
    finally:
        with _ACTIVE_LOCAL_RUNS_LOCK:
            _ACTIVE_LOCAL_RUNS.discard(run_id)


def trigger_background_run(run_id: UUID, *, enabled: bool = True) -> bool:
    """Spawn a non-blocking daemon thread to claim and execute `run_id` if not already running locally."""
    if not enabled:
        return False
    try:
        cfg = get_settings()
        if not cfg.database_url:
            return False
    except Exception:
        return False

    with _ACTIVE_LOCAL_RUNS_LOCK:
        if run_id in _ACTIVE_LOCAL_RUNS:
            return False
        _ACTIVE_LOCAL_RUNS.add(run_id)

    thread = threading.Thread(
        target=_background_worker_entry,
        args=(run_id,),
        name=f"jarvis-agent-run-{str(run_id)[:8]}",
        daemon=True,
    )
    thread.start()
    return True


def run_worker_loop(
    *,
    poll_interval_seconds: float = 1.5,
    max_runs: Optional[int] = None,
    stop_event: Optional[threading.Event] = None,
) -> int:
    """Standalone persistent worker loop polling `public.agent_runs` via `FOR UPDATE SKIP LOCKED`."""
    processed = 0
    worker_prefix = generate_worker_id("daemon")
    logger.info("Starting J.A.R.V.I.S. agent worker loop (%s)", worker_prefix)

    while True:
        if stop_event is not None and stop_event.is_set():
            break
        if max_runs is not None and processed >= max_runs:
            break

        outcome: Optional[AgentRunOutcome] = None
        try:
            cfg = get_settings()
            with open_connection() as conn:
                outcome = claim_and_execute_run(
                    conn,
                    worker_id=generate_worker_id("daemon"),
                    settings=cfg,
                )
        except Exception as exc:
            logger.error("Worker loop error: %s", redact(f"{type(exc).__name__}: {exc}"))

        if outcome is not None:
            processed += 1
            continue

        time.sleep(max(0.1, float(poll_interval_seconds)))

    return processed
