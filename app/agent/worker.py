"""Durable PostgreSQL-backed worker and execution coordinator for agent runs.

Supports two complementary execution modes using the same atomic `FOR UPDATE SKIP LOCKED`
database queue (`public.agent_runs`):

1. **Bounded Serverless Continuation (`claim_and_execute_run`)**:
   Invoked by `POST .../runs/{run_id}/advance` (including on Vercel serverless functions)
   to atomically claim or resume a run for a bounded time slice and persist progress
   before yielding cleanly back to `queued` when continuation is needed.
2. **Standalone Persistent Worker Loop (`run_worker_loop` / `scripts/run_agent_worker.py`)**:
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
    invocation_timeout_seconds: Optional[float] = None,
    max_iterations_per_invocation: Optional[int] = None,
    commit_step: Optional[Callable[[], None]] = None,
) -> Optional[AgentRunOutcome]:
    """Atomically claim the specified (or next eligible) run and execute it for one bounded invocation or to completion."""
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

    def _model_heartbeat() -> bool:
        ok = repositories_agent.heartbeat_run(
            conn,
            run_id=claimed_run_id,
            worker_id=wid,
            lease_seconds=cfg.agent_lease_seconds,
        )
        commit()
        return ok

    try:
        llm = model or create_groq_chat_model(
            cfg,
            registry=reg,
            bind_default_tools=True,
            heartbeat_callback=_model_heartbeat,
        )
    except Exception as exc:
        safe_msg = redact(f"Failed to initialize language model: {exc}")
        with conn.transaction():
            updated_run = repositories_agent.fail_or_retry_run(
                conn,
                run_id=claimed_run_id,
                worker_id=wid,
                error_message=safe_msg,
                retryable=False,
            )
            if updated_run is not None and updated_run["status"] == "failed":
                repositories_agent.append_run_step(
                    conn,
                    run_id=claimed_run_id,
                    step_type="run_failed",
                    status="failed",
                    output_summary={"error_code": "model_init_error", "message": safe_msg},
                )
        commit()
        if updated_run is None:
            return AgentRunOutcome(
                run_id=claimed_run_id,
                conversation_id=conversation_id,
                status="running",
                iterations=0,
                tool_calls_executed=0,
            )
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

    effective_run_timeout: float = float(cfg.agent_run_timeout_seconds)
    if claimed.get("remaining_timeout_seconds") is not None:
        effective_run_timeout = max(
            0.0,
            min(effective_run_timeout, float(claimed["remaining_timeout_seconds"])),
        )

    effective_invocation_timeout: Optional[float] = (
        float(invocation_timeout_seconds)
        if invocation_timeout_seconds is not None
        else float(cfg.agent_invocation_timeout_seconds)
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
        run_timeout_seconds=effective_run_timeout,
        invocation_timeout_seconds=effective_invocation_timeout,
        invocation_reserve_seconds=float(cfg.agent_invocation_reserve_seconds),
        model_timeout_seconds=float(cfg.agent_model_timeout_seconds),
        max_iterations_per_invocation=max_iterations_per_invocation,
        lease_seconds=cfg.agent_lease_seconds,
        max_tool_result_chars=cfg.agent_max_tool_result_chars,
        commit_step=commit,
    )


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
