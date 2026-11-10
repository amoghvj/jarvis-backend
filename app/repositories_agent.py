"""Database queries for project-scoped conversations, messages, agent runs, and run steps."""

from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

import psycopg
from psycopg.types.json import Jsonb

from app.tools.errors import redact

MAX_TITLE_CHARS = 80
MAX_SUMMARY_STR_CHARS = 500


class ActiveRunConflictError(Exception):
    """Raised when a conversation already has an active (queued or running) run."""

    def __init__(self, active_run_id: UUID) -> None:
        super().__init__("A run is already active for this conversation")
        self.active_run_id = active_run_id


def derive_conversation_title(content: str) -> str:
    """Deterministically generate a concise conversation title from the first user message."""
    cleaned = " ".join((content or "").strip().split())
    if not cleaned:
        return "New investigation"
    if len(cleaned) <= MAX_TITLE_CHARS:
        return cleaned
    return cleaned[: MAX_TITLE_CHARS - 1].rstrip() + "…"


def _sanitize_summary_obj(value: Any, depth: int = 0) -> Any:
    """Redact secrets and bound strings/collections before persisting to agent_run_steps."""
    if depth > 4:
        return "[truncated]"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        redacted = redact(value)
        if len(redacted) > MAX_SUMMARY_STR_CHARS:
            return redacted[:MAX_SUMMARY_STR_CHARS] + "…"
        return redacted
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_sanitize_summary_obj(item, depth + 1) for item in list(value)[:20]]
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for k, v in list(value.items())[:25]:
            key_str = str(k)[:64]
            lower = key_str.lower()
            if any(s in lower for s in ("secret", "token", "password", "api_key", "private_key", "authorization", "cookie")):
                out[key_str] = "[REDACTED]"
            else:
                out[key_str] = _sanitize_summary_obj(v, depth + 1)
        return out
    return redact(str(value))[:MAX_SUMMARY_STR_CHARS]


# ============================================================================
# CONVERSATIONS
# ============================================================================


def create_conversation(
    conn: Any,
    *,
    user_id: UUID,
    project_id: UUID,
    title: Optional[str] = None,
) -> Dict[str, Any]:
    proj = conn.execute(
        "SELECT id FROM public.projects WHERE id = %s AND user_id = %s",
        (project_id, user_id),
    ).fetchone()
    if proj is None:
        raise ValueError("Project not found for user")

    clean_title = title.strip()[:200] if isinstance(title, str) and title.strip() else None
    row = conn.execute(
        """
        INSERT INTO public.conversations (
            user_id,
            project_id,
            title,
            created_at,
            updated_at,
            last_message_at
        )
        VALUES (%s, %s, %s, clock_timestamp(), clock_timestamp(), clock_timestamp())
        RETURNING id, user_id, project_id, title, created_at, updated_at, last_message_at
        """,
        (user_id, project_id, clean_title),
    ).fetchone()
    return dict(row)


def list_conversations(
    conn: Any,
    *,
    user_id: UUID,
    project_id: UUID,
    limit: int = 50,
    offset: int = 0,
) -> Tuple[List[Dict[str, Any]], bool]:
    rows = conn.execute(
        """
        SELECT
            c.id,
            c.user_id,
            c.project_id,
            c.title,
            c.created_at,
            c.updated_at,
            c.last_message_at
        FROM public.conversations c
        JOIN public.projects p
          ON p.id = c.project_id
         AND p.user_id = c.user_id
        WHERE c.user_id = %s AND c.project_id = %s
        ORDER BY c.last_message_at DESC, c.id DESC
        LIMIT %s OFFSET %s
        """,
        (user_id, project_id, limit + 1, offset),
    ).fetchall()
    has_more = len(rows) > limit
    return [dict(r) for r in rows[:limit]], has_more


def get_conversation(
    conn: Any,
    *,
    user_id: UUID,
    project_id: UUID,
    conversation_id: UUID,
) -> Optional[Dict[str, Any]]:
    row = conn.execute(
        """
        SELECT
            c.id,
            c.user_id,
            c.project_id,
            c.title,
            c.created_at,
            c.updated_at,
            c.last_message_at
        FROM public.conversations c
        JOIN public.projects p
          ON p.id = c.project_id
         AND p.user_id = c.user_id
        WHERE c.id = %s AND c.user_id = %s AND c.project_id = %s
        """,
        (conversation_id, user_id, project_id),
    ).fetchone()
    return dict(row) if row else None


def update_conversation_title(
    conn: Any,
    *,
    user_id: UUID,
    project_id: UUID,
    conversation_id: UUID,
    title: Optional[str],
) -> Optional[Dict[str, Any]]:
    """Update a conversation's title and advance updated_at without altering last_message_at."""
    clean_title = title.strip()[:200] if isinstance(title, str) and title.strip() else None
    row = conn.execute(
        """
        UPDATE public.conversations c
        SET title = %s,
            updated_at = clock_timestamp()
        FROM public.projects p
        WHERE c.id = %s
          AND c.user_id = %s
          AND c.project_id = %s
          AND p.id = c.project_id
          AND p.user_id = c.user_id
        RETURNING
            c.id,
            c.user_id,
            c.project_id,
            c.title,
            c.created_at,
            c.updated_at,
            c.last_message_at
        """,
        (clean_title, conversation_id, user_id, project_id),
    ).fetchone()
    return dict(row) if row else None


def delete_conversation(
    conn: Any,
    *,
    user_id: UUID,
    project_id: UUID,
    conversation_id: UUID,
) -> bool:
    row = conn.execute(
        """
        DELETE FROM public.conversations c
        USING public.projects p
        WHERE c.id = %s
          AND c.user_id = %s
          AND c.project_id = %s
          AND p.id = c.project_id
          AND p.user_id = c.user_id
        RETURNING c.id
        """,
        (conversation_id, user_id, project_id),
    ).fetchone()
    return row is not None


# ============================================================================
# MESSAGES
# ============================================================================


def append_conversation_message(
    conn: Any,
    *,
    conversation_id: UUID,
    role: str,
    message_type: str,
    content: Optional[str] = None,
    tool_calls: Optional[List[Dict[str, Any]]] = None,
    tool_call_id: Optional[str] = None,
    tool_name: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    run_id: Optional[UUID] = None,
    set_title_if_empty: bool = False,
) -> Dict[str, Any]:
    """Atomically allocate the next sequence_number for a conversation and persist a message."""
    with conn.transaction():
        conv = conn.execute(
            "SELECT id, title FROM public.conversations WHERE id = %s FOR UPDATE",
            (conversation_id,),
        ).fetchone()
        if conv is None:
            raise ValueError(f"Conversation {conversation_id} does not exist")

        seq_row = conn.execute(
            """
            SELECT COALESCE(MAX(sequence_number), 0) + 1 AS next_seq
            FROM public.conversation_messages
            WHERE conversation_id = %s
            """,
            (conversation_id,),
        ).fetchone()
        next_seq = int(seq_row["next_seq"])

        msg_row = conn.execute(
            """
            INSERT INTO public.conversation_messages (
                conversation_id,
                sequence_number,
                role,
                message_type,
                content,
                tool_calls,
                tool_call_id,
                tool_name,
                metadata,
                run_id,
                created_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, clock_timestamp())
            RETURNING
                id,
                conversation_id,
                sequence_number,
                role,
                message_type,
                content,
                tool_calls,
                tool_call_id,
                tool_name,
                metadata,
                run_id,
                created_at
            """,
            (
                conversation_id,
                next_seq,
                role,
                message_type,
                content,
                Jsonb(tool_calls) if tool_calls is not None else None,
                tool_call_id,
                tool_name,
                Jsonb(metadata or {}),
                run_id,
            ),
        ).fetchone()

        new_title = conv["title"]
        if set_title_if_empty and not new_title and content:
            new_title = derive_conversation_title(content)

        is_visible_message = role in ("user", "assistant") and message_type == "text"
        if is_visible_message:
            conn.execute(
                """
                UPDATE public.conversations
                SET title = %s,
                    updated_at = clock_timestamp(),
                    last_message_at = clock_timestamp()
                WHERE id = %s
                """,
                (new_title, conversation_id),
            )
        else:
            conn.execute(
                """
                UPDATE public.conversations
                SET title = %s,
                    updated_at = clock_timestamp()
                WHERE id = %s
                """,
                (new_title, conversation_id),
            )
        return dict(msg_row)


def list_visible_messages(
    conn: Any,
    *,
    conversation_id: UUID,
    limit: int = 100,
    offset: int = 0,
) -> Tuple[List[Dict[str, Any]], bool]:
    """Return user and final assistant text messages for the visible chat transcript."""
    rows = conn.execute(
        """
        SELECT
            id,
            conversation_id,
            sequence_number,
            role,
            message_type,
            content,
            metadata,
            run_id,
            created_at
        FROM public.conversation_messages
        WHERE conversation_id = %s
          AND role IN ('user', 'assistant')
          AND message_type = 'text'
        ORDER BY sequence_number ASC
        LIMIT %s OFFSET %s
        """,
        (conversation_id, limit + 1, offset),
    ).fetchall()
    has_more = len(rows) > limit
    return [dict(r) for r in rows[:limit]], has_more


def list_all_messages_for_history(
    conn: Any,
    *,
    conversation_id: UUID,
) -> List[Dict[str, Any]]:
    """Return the complete ordered structured message history for HistoryManager."""
    rows = conn.execute(
        """
        SELECT
            id,
            conversation_id,
            sequence_number,
            role,
            message_type,
            content,
            tool_calls,
            tool_call_id,
            tool_name,
            metadata,
            run_id,
            created_at
        FROM public.conversation_messages
        WHERE conversation_id = %s
        ORDER BY sequence_number ASC
        """,
        (conversation_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_message_by_id(
    conn: Any,
    *,
    conversation_id: UUID,
    message_id: UUID,
) -> Optional[Dict[str, Any]]:
    row = conn.execute(
        """
        SELECT
            id,
            conversation_id,
            sequence_number,
            role,
            message_type,
            content,
            tool_calls,
            tool_call_id,
            tool_name,
            metadata,
            run_id,
            created_at
        FROM public.conversation_messages
        WHERE id = %s AND conversation_id = %s
        """,
        (message_id, conversation_id),
    ).fetchone()
    return dict(row) if row else None


def list_assistant_messages_for_run(
    conn: Any,
    *,
    conversation_id: UUID,
    run_id: UUID,
) -> List[Dict[str, Any]]:
    """Return visible assistant text messages produced by a specific run."""
    rows = conn.execute(
        """
        SELECT
            id,
            conversation_id,
            sequence_number,
            role,
            message_type,
            content,
            metadata,
            run_id,
            created_at
        FROM public.conversation_messages
        WHERE conversation_id = %s
          AND run_id = %s
          AND role = 'assistant'
          AND message_type = 'text'
        ORDER BY sequence_number ASC
        """,
        (conversation_id, run_id),
    ).fetchall()
    return [dict(r) for r in rows]


# ============================================================================
# AGENT RUNS & DURABLE QUEUE
# ============================================================================


_RUN_COLUMNS = """
    id,
    conversation_id,
    user_message_id,
    status,
    error_message,
    client_request_id,
    worker_id,
    claimed_at,
    lease_expires_at,
    heartbeat_at,
    attempt_count,
    max_attempts,
    next_retry_at,
    timeout_at,
    created_at,
    started_at,
    completed_at,
    updated_at
"""


def create_run_for_user_message(
    conn: Any,
    *,
    conversation_id: UUID,
    user_message_id: UUID,
    client_request_id: Optional[str] = None,
    max_attempts: int = 3,
    timeout_seconds: int = 120,
) -> Dict[str, Any]:
    """Validate that user_message_id is a user text message in conversation_id and create a queued run."""
    msg = conn.execute(
        """
        SELECT id, conversation_id, role, message_type
        FROM public.conversation_messages
        WHERE id = %s AND conversation_id = %s
        """,
        (user_message_id, conversation_id),
    ).fetchone()
    if msg is None:
        raise ValueError("Initiating message does not belong to this conversation")
    if msg["role"] != "user" or msg["message_type"] != "text":
        raise ValueError("Agent run can only be initiated by a user text message")

    clean_req_id = (
        client_request_id.strip()[:128]
        if isinstance(client_request_id, str) and client_request_id.strip()
        else None
    )

    run_row = conn.execute(
        f"""
        INSERT INTO public.agent_runs (
            conversation_id,
            user_message_id,
            status,
            client_request_id,
            max_attempts,
            timeout_at,
            created_at,
            updated_at
        )
        VALUES (
            %s,
            %s,
            'queued',
            %s,
            %s,
            clock_timestamp() + make_interval(secs => %s),
            clock_timestamp(),
            clock_timestamp()
        )
        RETURNING {_RUN_COLUMNS}
        """,
        (
            conversation_id,
            user_message_id,
            clean_req_id,
            max(1, int(max_attempts)),
            max(10, int(timeout_seconds)),
        ),
    ).fetchone()

    conn.execute(
        "UPDATE public.conversation_messages SET run_id = %s WHERE id = %s AND conversation_id = %s",
        (run_row["id"], user_message_id, conversation_id),
    )
    return dict(run_row)


def expire_stale_runs(
    conn: Any,
    *,
    conversation_id: Optional[UUID] = None,
    run_id: Optional[UUID] = None,
) -> List[Dict[str, Any]]:
    """Fail runs that have exceeded their overall execution timeout or exhausted retries after lease expiry."""
    filters = ["status IN ('queued', 'running')"]
    params: List[Any] = []
    if conversation_id is not None:
        filters.append("conversation_id = %s")
        params.append(conversation_id)
    if run_id is not None:
        filters.append("id = %s")
        params.append(run_id)

    where_sql = " AND ".join(filters)
    stale_rows = conn.execute(
        f"""
        UPDATE public.agent_runs
        SET status = 'failed',
            error_message = CASE
                WHEN timeout_at IS NOT NULL AND timeout_at <= clock_timestamp()
                    THEN 'Investigation run timed out before completion.'
                ELSE COALESCE(error_message, 'Investigation worker lease expired and maximum retry attempts were reached.')
            END,
            completed_at = COALESCE(completed_at, clock_timestamp()),
            lease_expires_at = NULL,
            updated_at = clock_timestamp()
        WHERE {where_sql}
          AND (
              (timeout_at IS NOT NULL AND timeout_at <= clock_timestamp())
              OR (
                  status = 'running'
                  AND lease_expires_at IS NOT NULL
                  AND lease_expires_at <= clock_timestamp()
                  AND attempt_count >= max_attempts
              )
          )
        RETURNING {_RUN_COLUMNS}
        """,
        tuple(params),
    ).fetchall()

    expired = [dict(r) for r in stale_rows]
    for r in expired:
        append_run_step(
            conn,
            run_id=r["id"],
            step_type="run_failed",
            status="failed",
            output_summary={"message": r["error_message"] or "Run timed out or exhausted retries"},
        )
    return expired


def submit_user_message_and_create_run(
    conn: Any,
    *,
    user_id: UUID,
    project_id: UUID,
    conversation_id: UUID,
    content: str,
    client_request_id: Optional[str] = None,
    max_attempts: int = 3,
    timeout_seconds: int = 120,
) -> Tuple[Dict[str, Any], Dict[str, Any], bool]:
    """Persist a user message and create a queued agent run atomically.

    Returns (user_message, run, deduplicated).
    Raises ActiveRunConflictError if another run is currently active for the conversation.
    """
    with conn.transaction():
        conv = conn.execute(
            """
            SELECT c.id, c.user_id, c.project_id, c.title
            FROM public.conversations c
            JOIN public.projects p
              ON p.id = c.project_id
             AND p.user_id = c.user_id
            WHERE c.id = %s AND c.user_id = %s AND c.project_id = %s
            FOR UPDATE OF c
            """,
            (conversation_id, user_id, project_id),
        ).fetchone()
        if conv is None:
            raise ValueError("Conversation not found")

        # Clean up any timed-out or exhausted runs on this conversation first.
        expire_stale_runs(conn, conversation_id=conversation_id)

        clean_req_id = (
            client_request_id.strip()[:128]
            if isinstance(client_request_id, str) and client_request_id.strip()
            else None
        )

        # 1. Explicit client_request_id idempotency check
        if clean_req_id is not None:
            existing_run = conn.execute(
                f"""
                SELECT {_RUN_COLUMNS}
                FROM public.agent_runs
                WHERE conversation_id = %s AND client_request_id = %s
                """,
                (conversation_id, clean_req_id),
            ).fetchone()
            if existing_run is not None:
                existing_msg = get_message_by_id(
                    conn,
                    conversation_id=conversation_id,
                    message_id=existing_run["user_message_id"],
                )
                return dict(existing_msg), dict(existing_run), True

        # 2. Check for an active run on this conversation
        active_run = conn.execute(
            f"""
            SELECT {_RUN_COLUMNS}
            FROM public.agent_runs
            WHERE conversation_id = %s AND status IN ('queued', 'running')
            LIMIT 1
            """,
            (conversation_id,),
        ).fetchone()
        if active_run is not None:
            raise ActiveRunConflictError(active_run["id"])

        try:
            with conn.transaction():
                user_msg = append_conversation_message(
                    conn,
                    conversation_id=conversation_id,
                    role="user",
                    message_type="text",
                    content=content,
                    metadata={},
                    set_title_if_empty=True,
                )

                run_row = create_run_for_user_message(
                    conn,
                    conversation_id=conversation_id,
                    user_message_id=user_msg["id"],
                    client_request_id=clean_req_id,
                    max_attempts=max_attempts,
                    timeout_seconds=timeout_seconds,
                )
                user_msg["run_id"] = run_row["id"]
                return dict(user_msg), dict(run_row), False
        except psycopg.errors.UniqueViolation:
            # Handle concurrent duplicate client_request_id or active-run unique index conflicts
            if clean_req_id is not None:
                existing_run = conn.execute(
                    f"""
                    SELECT {_RUN_COLUMNS}
                    FROM public.agent_runs
                    WHERE conversation_id = %s AND client_request_id = %s
                    """,
                    (conversation_id, clean_req_id),
                ).fetchone()
                if existing_run is not None:
                    existing_msg = get_message_by_id(
                        conn,
                        conversation_id=conversation_id,
                        message_id=existing_run["user_message_id"],
                    )
                    return dict(existing_msg), dict(existing_run), True

            active_run = conn.execute(
                f"""
                SELECT {_RUN_COLUMNS}
                FROM public.agent_runs
                WHERE conversation_id = %s AND status IN ('queued', 'running')
                LIMIT 1
                """,
                (conversation_id,),
            ).fetchone()
            if active_run is not None:
                raise ActiveRunConflictError(active_run["id"])
            raise


def get_run(
    conn: Any,
    *,
    conversation_id: UUID,
    run_id: UUID,
) -> Optional[Dict[str, Any]]:
    row = conn.execute(
        f"""
        SELECT {_RUN_COLUMNS}
        FROM public.agent_runs
        WHERE id = %s AND conversation_id = %s
        """,
        (run_id, conversation_id),
    ).fetchone()
    return dict(row) if row else None


def claim_run(
    conn: Any,
    *,
    worker_id: str,
    lease_seconds: int = 45,
    run_id: Optional[UUID] = None,
) -> Optional[Dict[str, Any]]:
    """Atomically claim a queued run or recover a running run whose worker lease expired.

    Uses PostgreSQL `FOR UPDATE SKIP LOCKED` so concurrent workers never claim the same run.
    """
    with conn.transaction():
        expire_stale_runs(conn, run_id=run_id)

        target_filter = "AND r.id = %s" if run_id is not None else ""
        row = conn.execute(
            f"""
            WITH candidate AS (
                SELECT r.id
                FROM public.agent_runs r
                WHERE (
                    (
                        r.status = 'queued'
                        AND (r.next_retry_at IS NULL OR r.next_retry_at <= clock_timestamp())
                    )
                    OR (
                        r.status = 'running'
                        AND r.lease_expires_at IS NOT NULL
                        AND r.lease_expires_at <= clock_timestamp()
                        AND r.attempt_count < r.max_attempts
                    )
                )
                AND (r.timeout_at IS NULL OR r.timeout_at > clock_timestamp())
                {target_filter}
                ORDER BY r.created_at ASC
                FOR UPDATE SKIP LOCKED
                LIMIT 1
            )
            UPDATE public.agent_runs ar
            SET status = 'running',
                worker_id = %s,
                claimed_at = clock_timestamp(),
                heartbeat_at = clock_timestamp(),
                lease_expires_at = clock_timestamp() + make_interval(secs => %s),
                started_at = COALESCE(ar.started_at, clock_timestamp()),
                attempt_count = ar.attempt_count + 1,
                next_retry_at = NULL,
                updated_at = clock_timestamp()
            FROM candidate c
            WHERE ar.id = c.id
            RETURNING
                ar.id,
                ar.conversation_id,
                ar.user_message_id,
                ar.status,
                ar.error_message,
                ar.client_request_id,
                ar.worker_id,
                ar.claimed_at,
                ar.lease_expires_at,
                ar.heartbeat_at,
                ar.attempt_count,
                ar.max_attempts,
                ar.next_retry_at,
                ar.timeout_at,
                ar.created_at,
                ar.started_at,
                ar.completed_at,
                ar.updated_at
            """,
            (*([run_id] if run_id is not None else []), worker_id, max(10, int(lease_seconds))),
        ).fetchone()
        if row is None:
            return None

        run_dict = dict(row)
        conv_row = conn.execute(
            "SELECT user_id, project_id FROM public.conversations WHERE id = %s",
            (run_dict["conversation_id"],),
        ).fetchone()
        if conv_row:
            run_dict["user_id"] = conv_row["user_id"]
            run_dict["project_id"] = conv_row["project_id"]
        return run_dict


def heartbeat_run(
    conn: Any,
    *,
    run_id: UUID,
    worker_id: str,
    lease_seconds: int = 45,
) -> bool:
    """Refresh the worker lease and heartbeat timestamp while a run is actively executing."""
    row = conn.execute(
        """
        UPDATE public.agent_runs
        SET heartbeat_at = clock_timestamp(),
            lease_expires_at = clock_timestamp() + make_interval(secs => %s),
            updated_at = clock_timestamp()
        WHERE id = %s
          AND worker_id = %s
          AND status = 'running'
        RETURNING id
        """,
        (max(10, int(lease_seconds)), run_id, worker_id),
    ).fetchone()
    return row is not None


def complete_run(
    conn: Any,
    *,
    run_id: UUID,
    worker_id: str,
) -> Optional[Dict[str, Any]]:
    """Transition a running agent run to completed."""
    row = conn.execute(
        f"""
        UPDATE public.agent_runs
        SET status = 'completed',
            error_message = NULL,
            lease_expires_at = NULL,
            heartbeat_at = clock_timestamp(),
            completed_at = clock_timestamp(),
            updated_at = clock_timestamp()
        WHERE id = %s
          AND worker_id = %s
          AND status = 'running'
        RETURNING {_RUN_COLUMNS}
        """,
        (run_id, worker_id),
    ).fetchone()
    return dict(row) if row else None


def fail_or_retry_run(
    conn: Any,
    *,
    run_id: UUID,
    worker_id: str,
    error_message: str,
    retryable: bool = False,
    retry_delay_seconds: int = 2,
) -> Optional[Dict[str, Any]]:
    """Transition a run to queued (for bounded retry) or failed (terminal)."""
    safe_error = redact(error_message or "Agent execution failed")[:500]
    row = conn.execute(
        f"""
        UPDATE public.agent_runs
        SET status = CASE
                WHEN %s = true
                     AND attempt_count < max_attempts
                     AND (timeout_at IS NULL OR timeout_at > clock_timestamp())
                THEN 'queued'
                ELSE 'failed'
            END,
            error_message = %s,
            worker_id = NULL,
            lease_expires_at = NULL,
            next_retry_at = CASE
                WHEN %s = true
                     AND attempt_count < max_attempts
                     AND (timeout_at IS NULL OR timeout_at > clock_timestamp())
                THEN clock_timestamp() + make_interval(secs => %s)
                ELSE NULL
            END,
            completed_at = CASE
                WHEN %s = true
                     AND attempt_count < max_attempts
                     AND (timeout_at IS NULL OR timeout_at > clock_timestamp())
                THEN NULL
                ELSE clock_timestamp()
            END,
            updated_at = clock_timestamp()
        WHERE id = %s
          AND worker_id = %s
          AND status = 'running'
        RETURNING {_RUN_COLUMNS}
        """,
        (
            bool(retryable),
            safe_error,
            bool(retryable),
            max(1, int(retry_delay_seconds)),
            bool(retryable),
            run_id,
            worker_id,
        ),
    ).fetchone()
    return dict(row) if row else None


# ============================================================================
# AGENT RUN STEPS (Structured user-visible activity)
# ============================================================================


def append_run_step(
    conn: Any,
    *,
    run_id: UUID,
    step_type: str,
    status: str,
    tool_name: Optional[str] = None,
    input_summary: Optional[Dict[str, Any]] = None,
    output_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Atomically allocate the next step_number for a run and insert a sanitized step record."""
    with conn.transaction():
        conn.execute(
            "SELECT id FROM public.agent_runs WHERE id = %s FOR UPDATE",
            (run_id,),
        )
        seq_row = conn.execute(
            """
            SELECT COALESCE(MAX(step_number), 0) + 1 AS next_step
            FROM public.agent_run_steps
            WHERE run_id = %s
            """,
            (run_id,),
        ).fetchone()
        next_step = int(seq_row["next_step"])

        safe_in = _sanitize_summary_obj(input_summary or {})
        safe_out = _sanitize_summary_obj(output_summary or {})

        row = conn.execute(
            """
            INSERT INTO public.agent_run_steps (
                run_id,
                step_number,
                step_type,
                tool_name,
                status,
                input_summary,
                output_summary,
                created_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, clock_timestamp())
            RETURNING
                id,
                run_id,
                step_number,
                step_type,
                tool_name,
                status,
                input_summary,
                output_summary,
                created_at
            """,
            (
                run_id,
                next_step,
                step_type,
                tool_name,
                status,
                Jsonb(safe_in if isinstance(safe_in, dict) else {"value": safe_in}),
                Jsonb(safe_out if isinstance(safe_out, dict) else {"value": safe_out}),
            ),
        ).fetchone()
        return dict(row)


def list_run_steps(
    conn: Any,
    *,
    run_id: UUID,
    after_step: int = 0,
    limit: int = 100,
) -> Tuple[List[Dict[str, Any]], int]:
    """Return incremental run steps with step_number > after_step and the latest_step_number."""
    rows = conn.execute(
        """
        SELECT
            id,
            run_id,
            step_number,
            step_type,
            tool_name,
            status,
            input_summary,
            output_summary,
            created_at
        FROM public.agent_run_steps
        WHERE run_id = %s
          AND step_number > %s
        ORDER BY step_number ASC
        LIMIT %s
        """,
        (run_id, max(0, int(after_step)), max(1, min(int(limit), 500))),
    ).fetchall()

    max_row = conn.execute(
        "SELECT COALESCE(MAX(step_number), 0) AS max_step FROM public.agent_run_steps WHERE run_id = %s",
        (run_id,),
    ).fetchone()
    latest_step = int(max_row["max_step"]) if max_row else 0
    return [dict(r) for r in rows], latest_step
