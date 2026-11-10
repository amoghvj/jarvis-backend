"""Replaceable message-history management layer.

Translates between framework-independent `conversation_messages` database rows and
`langchain_core.messages` objects (`HumanMessage`, `AIMessage`, `ToolMessage`).
The agent orchestrator interacts exclusively with `HistoryManager` and never queries
the `conversation_messages` table directly.
"""

from abc import ABC, abstractmethod
import json
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import UUID

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from app import repositories_agent


class HistoryManager(ABC):
    """Abstract interface for retrieving and appending conversation messages.

    Future history strategies (recent-N, token-budget window, sliding window,
    summarized or hybrid history) can implement this interface without changing
    the agent orchestration loop, provided they preserve complete tool-interaction
    cycles (an AIMessage with tool_calls must always remain paired with all of its
    matching ToolMessages).
    """

    @abstractmethod
    def get_history(
        self,
        conn: Any,
        conversation_id: UUID,
        *,
        active_run_id: Optional[UUID] = None,
    ) -> List[BaseMessage]:
        """Return the ordered sequence of LangChain messages for model invocation."""

    @abstractmethod
    def append_message(
        self,
        conn: Any,
        conversation_id: UUID,
        message: BaseMessage,
        *,
        run_id: Optional[UUID] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Persist a LangChain message as a framework-independent database record."""

    @abstractmethod
    def get_pending_tool_calls_for_run(
        self,
        conn: Any,
        conversation_id: UUID,
        run_id: UUID,
    ) -> Tuple[Optional[AIMessage], List[Dict[str, Any]]]:
        """Return (last_ai_message, unexecuted_tool_calls) if the active run crashed mid-tool-batch."""


def _normalize_stored_tool_calls(raw_calls: Any) -> List[Dict[str, Any]]:
    """Convert stored JSONB tool_calls into LangChain AIMessage.tool_calls dicts."""
    if not isinstance(raw_calls, list):
        return []
    normalized: List[Dict[str, Any]] = []
    for idx, item in enumerate(raw_calls):
        if not isinstance(item, dict):
            continue
        call_id = str(item.get("id") or f"call_{idx}")
        name = str(item.get("name") or "")
        args = item.get("args")
        if not isinstance(args, dict):
            if isinstance(args, str):
                try:
                    parsed = json.loads(args)
                    args = parsed if isinstance(parsed, dict) else {"_raw_args": args}
                except Exception:
                    args = {"_raw_args": args}
            else:
                args = {}
        normalized.append(
            {
                "id": call_id,
                "name": name,
                "args": args,
                "type": "tool_call",
            }
        )
    return normalized


def row_to_langchain_message(row: Dict[str, Any]) -> BaseMessage:
    """Reconstruct a single LangChain message from a `conversation_messages` row."""
    role = row.get("role")
    msg_type = row.get("message_type")
    content = row.get("content") or ""
    meta = dict(row.get("metadata") or {})
    meta.setdefault("db_message_id", str(row.get("id")))
    meta.setdefault("sequence_number", row.get("sequence_number"))

    if role == "user":
        return HumanMessage(content=content, additional_kwargs={"metadata": meta})

    if role == "assistant":
        tool_calls = _normalize_stored_tool_calls(row.get("tool_calls"))
        if msg_type == "tool_call" or tool_calls:
            return AIMessage(
                content=content,
                tool_calls=tool_calls,
                additional_kwargs={"metadata": meta},
            )
        return AIMessage(content=content, additional_kwargs={"metadata": meta})

    if role == "tool":
        tool_call_id = str(row.get("tool_call_id") or "unknown_call_id")
        tool_name = str(row.get("tool_name") or "unknown_tool")
        return ToolMessage(
            content=content,
            tool_call_id=tool_call_id,
            name=tool_name,
            additional_kwargs={"metadata": meta},
        )

    raise ValueError(f"Unsupported conversation message role '{role}'")


class FullHistoryManager(HistoryManager):
    """MVP history manager that retrieves the complete conversation history in sequence order.

    Also guarantees tool-call cycle integrity: if an earlier run terminated abnormally after
    persisting an assistant `tool_call` message without all matching `tool_result` messages,
    synthetic error `ToolMessage` records are persisted before reconstructing history so the
    LLM never receives dangling tool calls.
    """

    def get_pending_tool_calls_for_run(
        self,
        conn: Any,
        conversation_id: UUID,
        run_id: UUID,
    ) -> Tuple[Optional[AIMessage], List[Dict[str, Any]]]:
        """Inspect persisted messages for `run_id` to find any unexecuted tool calls from the latest AIMessage.

        This prevents duplicate tool execution when a worker recovers a run that crashed
        mid-way through a multi-tool batch: tools that already have a persisted `tool_result`
        row are skipped, and only the remaining tool calls are executed.
        """
        rows = repositories_agent.list_all_messages_for_history(
            conn,
            conversation_id=conversation_id,
        )
        if not rows:
            return None, []

        # Find the last assistant tool_call message belonging to this run_id
        last_tc_idx: Optional[int] = None
        for idx in range(len(rows) - 1, -1, -1):
            r = rows[idx]
            if r.get("role") == "user":
                break
            if (
                r.get("run_id") == run_id
                and r.get("role") == "assistant"
                and r.get("message_type") == "tool_call"
            ):
                last_tc_idx = idx
                break

        if last_tc_idx is None:
            return None, []

        tc_row = rows[last_tc_idx]
        all_calls = _normalize_stored_tool_calls(tc_row.get("tool_calls"))
        if not all_calls:
            return None, []

        completed_ids: Set[str] = set()
        for subsequent in rows[last_tc_idx + 1 :]:
            if subsequent.get("role") == "tool" and subsequent.get("tool_call_id"):
                completed_ids.add(str(subsequent["tool_call_id"]))

        pending = [c for c in all_calls if c["id"] not in completed_ids]
        if not pending:
            return None, []
        return row_to_langchain_message(tc_row), pending  # type: ignore[return-value]

    def _repair_dangling_tool_calls(
        self,
        conn: Any,
        conversation_id: UUID,
        rows: List[Dict[str, Any]],
        active_run_id: Optional[UUID] = None,
    ) -> List[Dict[str, Any]]:
        """Ensure every persisted assistant tool_call message from non-active turns has matching ToolMessages."""
        if not rows:
            return rows

        existing_tool_result_ids: Set[str] = {
            str(r["tool_call_id"])
            for r in rows
            if r.get("role") == "tool" and r.get("tool_call_id")
        }

        repaired = False
        for idx, r in enumerate(rows):
            if r.get("role") != "assistant" or r.get("message_type") != "tool_call":
                continue
            # If this is the very last assistant tool_call message and belongs to the active_run_id,
            # get_pending_tool_calls_for_run handles resuming it unless another user message followed.
            has_later_non_tool = any(
                later.get("role") in ("user", "assistant") for later in rows[idx + 1 :]
            )
            if (
                active_run_id is not None
                and r.get("run_id") == active_run_id
                and not has_later_non_tool
            ):
                continue

            calls = _normalize_stored_tool_calls(r.get("tool_calls"))
            for call in calls:
                cid = call["id"]
                if cid not in existing_tool_result_ids:
                    err_payload = json.dumps(
                        {
                            "ok": False,
                            "operation": call["name"] or "unknown_tool",
                            "error": {
                                "code": "interrupted_run",
                                "message": "Tool execution was interrupted before completion in a prior run.",
                                "retryable": True,
                            },
                        }
                    )
                    repositories_agent.append_conversation_message(
                        conn,
                        conversation_id=conversation_id,
                        role="tool",
                        message_type="tool_result",
                        content=err_payload,
                        tool_call_id=cid,
                        tool_name=call["name"] or "unknown_tool",
                        metadata={"repaired_dangling_tool_call": True},
                        run_id=r.get("run_id"),
                    )
                    existing_tool_result_ids.add(cid)
                    repaired = True

        if repaired:
            return repositories_agent.list_all_messages_for_history(
                conn,
                conversation_id=conversation_id,
            )
        return rows

    def get_history(
        self,
        conn: Any,
        conversation_id: UUID,
        *,
        active_run_id: Optional[UUID] = None,
    ) -> List[BaseMessage]:
        rows = repositories_agent.list_all_messages_for_history(
            conn,
            conversation_id=conversation_id,
        )
        rows = self._repair_dangling_tool_calls(
            conn,
            conversation_id,
            rows,
            active_run_id=active_run_id,
        )
        return [row_to_langchain_message(r) for r in rows]

    def append_message(
        self,
        conn: Any,
        conversation_id: UUID,
        message: BaseMessage,
        *,
        run_id: Optional[UUID] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        meta = dict(metadata or {})
        if isinstance(message, HumanMessage):
            return repositories_agent.append_conversation_message(
                conn,
                conversation_id=conversation_id,
                role="user",
                message_type="text",
                content=str(message.content or ""),
                metadata=meta,
                run_id=run_id,
                set_title_if_empty=True,
            )

        if isinstance(message, AIMessage):
            raw_tool_calls = getattr(message, "tool_calls", None) or []
            if raw_tool_calls:
                stored_calls = [
                    {
                        "id": str(tc.get("id") or f"call_{i}"),
                        "name": str(tc.get("name") or ""),
                        "args": tc.get("args") if isinstance(tc.get("args"), dict) else {},
                        "type": "tool_call",
                    }
                    for i, tc in enumerate(raw_tool_calls)
                    if isinstance(tc, dict)
                ]
                return repositories_agent.append_conversation_message(
                    conn,
                    conversation_id=conversation_id,
                    role="assistant",
                    message_type="tool_call",
                    content=str(message.content or "") if message.content else None,
                    tool_calls=stored_calls,
                    metadata=meta,
                    run_id=run_id,
                )
            return repositories_agent.append_conversation_message(
                conn,
                conversation_id=conversation_id,
                role="assistant",
                message_type="text",
                content=str(message.content or ""),
                metadata=meta,
                run_id=run_id,
            )

        if isinstance(message, ToolMessage):
            tool_name = getattr(message, "name", None) or meta.get("tool_name") or "unknown_tool"
            return repositories_agent.append_conversation_message(
                conn,
                conversation_id=conversation_id,
                role="tool",
                message_type="tool_result",
                content=str(message.content or ""),
                tool_call_id=str(message.tool_call_id),
                tool_name=str(tool_name),
                metadata=meta,
                run_id=run_id,
            )

        raise TypeError(f"Unsupported LangChain message type: {type(message).__name__}")

