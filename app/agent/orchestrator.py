"""Explicit Python agent orchestration loop (`run_agent`).

Coordinates:
- History loading and persistence exclusively through `HistoryManager`
- System prompt injection (`SystemMessage`, never persisted as a conversation message)
- Native tool calling via `GroqChatModel` (`langchain-core`)
- Validated tool execution via `dispatch_tool_call` -> `app.tools.executor.execute_tool`
- Incremental user-visible progress logging in `agent_run_steps`
- Heartbeat / lease renewal, iteration limits, and execution timeouts
"""

from dataclasses import dataclass
import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
from uuid import UUID

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import Runnable

from app import repositories as db_repo
from app import repositories_agent
from app.agent.history import HistoryManager
from app.agent.model import AgentModelError, GroqChatModel, strip_reasoning_tags
from app.agent.prompt import build_system_message
from app.agent.tools_adapter import activity_label_for_tool, dispatch_tool_call
from app.tools.definitions import get_default_registry
from app.tools.errors import redact
from app.tools.registry import ToolRegistry
from app.tools.schemas import ToolContext

logger = logging.getLogger("jarvis.agent")


@dataclass(frozen=True)
class AgentRunOutcome:
    run_id: UUID
    conversation_id: UUID
    status: str  # "completed" | "failed" | "queued"
    iterations: int
    tool_calls_executed: int
    final_message: Optional[ str] = None
    error_message: Optional[str] = None


def _lookup_tool_timeout(tool_registry: ToolRegistry, tool_name: str) -> float:
    """Return the configured per-tool timeout in seconds, defaulting conservatively."""
    try:
        return float(tool_registry.get(tool_name).constraints.timeout_seconds)
    except Exception:
        return 28.0


def _execute_tool_batch(
    *,
    conn: Any,
    conversation_id: UUID,
    run_id: UUID,
    worker_id: str,
    tool_calls: List[Dict[str, Any]],
    history_manager: HistoryManager,
    tool_registry: ToolRegistry,
    execution_context: ToolContext,
    lease_seconds: int,
    max_tool_result_chars: int,
    deadline: float,
    invocation_deadline: Optional[float] = None,
    total_work_budget: Optional[float] = None,
    made_invocation_progress: bool = False,
    commit: Callable[[], None],
) -> Tuple[List[str], int, bool, bool]:
    """Execute a list of tool calls sequentially, persisting each ToolMessage and run step.

    If the overall run deadline expires mid-batch, any remaining tool calls in the batch still
    receive persisted `ToolMessage` records explaining the timeout so the persisted
    `AIMessage` never has unresolved tool_call_ids.

    If the per-invocation time slice (`invocation_deadline`) expires or lacks headroom for the
    next tool while time remains on the overall run deadline, execution stops cleanly before
    starting the next tool so the next `/advance` invocation can resume the remaining pending
    tool calls via `HistoryManager.get_pending_tool_calls_for_run`.
    """
    summaries: List[str] = []
    executed_count = 0
    timed_out = False
    yielded = False

    for idx, tc in enumerate(tool_calls):
        call_id = str(tc.get("id") or f"call_{idx}")
        tool_name = str(tc.get("name") or "unknown_tool")
        raw_args = tc.get("args")

        now = time.monotonic()
        if now >= deadline:
            timed_out = True
            timeout_summary = f"Skipped {tool_name} because the investigation run reached its time limit."
            timeout_env = json.dumps(
                {
                    "ok": False,
                    "operation": tool_name,
                    "summary": timeout_summary,
                    "error": {
                        "code": "run_timeout",
                        "message": "The investigation run reached its maximum execution time limit.",
                        "retryable": True,
                    },
                }
            )
            with conn.transaction():
                history_manager.append_message(
                    conn,
                    conversation_id,
                    ToolMessage(content=timeout_env, tool_call_id=call_id, name=tool_name),
                    run_id=run_id,
                    metadata={"skipped_due_to_timeout": True},
                    worker_id=worker_id,
                )
                repositories_agent.append_run_step(
                    conn,
                    run_id=run_id,
                    step_type="tool_failed",
                    tool_name=tool_name,
                    status="failed",
                    input_summary={"activity": activity_label_for_tool(tool_name)},
                    output_summary={"ok": False, "summary": timeout_summary, "error_code": "run_timeout"},
                    expected_worker_id=worker_id,
                )
            summaries.append(timeout_summary)
            commit()
            continue

        if invocation_deadline is not None:
            if now >= invocation_deadline:
                yielded = True
                break
            if made_invocation_progress or executed_count > 0:
                tool_timeout = _lookup_tool_timeout(tool_registry, tool_name)
                if (
                    total_work_budget is not None
                    and total_work_budget >= tool_timeout
                    and (invocation_deadline - now) < tool_timeout
                ):
                    yielded = True
                    break

        # Verify + refresh lease atomically alongside `tool_started` before running the tool.
        repositories_agent.append_run_step(
            conn,
            run_id=run_id,
            step_type="tool_started",
            tool_name=tool_name,
            status="in_progress",
            input_summary={
                "activity": activity_label_for_tool(tool_name),
                "args": raw_args if isinstance(raw_args, dict) else {},
            },
            output_summary={},
            expected_worker_id=worker_id,
            refresh_lease_seconds=lease_seconds,
        )
        commit()

        outcome = dispatch_tool_call(
            tool_name=tool_name,
            raw_args=raw_args,
            tool_call_id=call_id,
            context=execution_context,
            registry=tool_registry,
            max_result_chars=max_tool_result_chars,
        )

        # Persist ToolMessage and tool_completed/tool_failed step atomically under active lease fencing.
        with conn.transaction():
            history_manager.append_message(
                conn,
                conversation_id,
                outcome.tool_message,
                run_id=run_id,
                metadata={
                    "ok": outcome.ok,
                    "summary": outcome.summary,
                },
                worker_id=worker_id,
            )
            repositories_agent.append_run_step(
                conn,
                run_id=run_id,
                step_type="tool_completed" if outcome.ok else "tool_failed",
                tool_name=outcome.tool_name,
                status="completed" if outcome.ok else "failed",
                input_summary=outcome.step_input_summary,
                output_summary=outcome.step_output_summary,
                expected_worker_id=worker_id,
                refresh_lease_seconds=lease_seconds,
            )
        executed_count += 1
        summaries.append(outcome.summary)
        commit()

    return summaries, executed_count, timed_out, yielded


def _build_graceful_limit_response(
    *,
    reason: str,
    gathered_summaries: List[str],
) -> str:
    """Construct an evidence-backed fallback response when iteration or timeout limits are reached."""
    bullets = "\n".join(f"- {s}" for s in gathered_summaries[-10:]) if gathered_summaries else "- No tool outputs were completed before the limit was reached."
    return (
        f"I reached the {reason} while investigating your request. Here is a summary of the "
        f"confirmed observations gathered so far:\n\n"
        f"{bullets}\n\n"
        f"You can send a follow-up message to continue the investigation from these findings."
    )


def _invoke_final_synthesis(
    model: Runnable[Any, AIMessage],
    messages: List[BaseMessage],
    reason: str,
    gathered_summaries: List[str],
    call_deadline: Optional[float] = None,
) -> str:
    """Attempt a final tool-free model synthesis when the iteration limit is reached."""
    synthesis_prompt = HumanMessage(
        content=(
            f"[System Note: Maximum tool-calling iterations reached ({reason}). "
            f"Do not call any more tools. Synthesize a clear, evidence-backed response "
            f"using the tool results gathered above, separating confirmed facts, root-cause "
            f"hypotheses, uncertainties, and recommended remediation steps.]"
        )
    )
    try:
        if isinstance(model, GroqChatModel):
            unbound = model.model_copy(
                update={
                    "bound_tools": [],
                    "tool_choice": None,
                    "call_deadline": call_deadline,
                }
            )
            resp = unbound.invoke([*messages, synthesis_prompt])
        else:
            resp = model.invoke([*messages, synthesis_prompt], tool_choice="none")
        text = strip_reasoning_tags(str(getattr(resp, "content", "") or ""))
        if text and not getattr(resp, "tool_calls", None):
            return text
    except repositories_agent.LeaseLostError:
        raise
    except AgentModelError as exc:
        if exc.code == "lease_lost":
            raise
    except Exception:
        pass
    return _build_graceful_limit_response(reason=reason, gathered_summaries=gathered_summaries)


def run_agent(
    *,
    conversation_id: UUID,
    run_id: UUID,
    worker_id: str,
    history_manager: HistoryManager,
    model: Runnable[Any, AIMessage],
    execution_context: ToolContext,
    tool_registry: Optional[ToolRegistry] = None,
    max_iterations: int = 8,
    run_timeout_seconds: float = 120,
    invocation_timeout_seconds: Optional[float] = None,
    invocation_reserve_seconds: Optional[float] = None,
    model_timeout_seconds: Optional[float] = None,
    max_iterations_per_invocation: Optional[int] = None,
    lease_seconds: int = 90,
    max_tool_result_chars: int = 16000,
    commit_step: Optional[Callable[[], None]] = None,
) -> AgentRunOutcome:
    """Execute one claimed agent run end-to-end or for a bounded invocation slice."""
    conn = execution_context.conn
    registry = tool_registry or get_default_registry()
    commit = commit_step or (lambda: None)
    now_start = time.monotonic()
    deadline = now_start + max(0.0, float(run_timeout_seconds))

    invocation_deadline: Optional[float] = None
    work_invocation_deadline: Optional[float] = None
    total_work_budget: Optional[float] = None
    if invocation_timeout_seconds is not None:
        raw_inv = max(0.05, float(invocation_timeout_seconds))
        invocation_deadline = now_start + raw_inv
        target_reserve = (
            float(invocation_reserve_seconds)
            if invocation_reserve_seconds is not None
            else 8.0
        )
        effective_reserve = min(max(0.0, target_reserve), raw_inv * 0.15)
        total_work_budget = max(0.01, raw_inv - effective_reserve)
        work_invocation_deadline = now_start + total_work_budget

    effective_model_timeout: float = (
        float(model_timeout_seconds)
        if model_timeout_seconds is not None
        else float(getattr(model, "timeout_seconds", 45.0) or 45.0)
    )

    if execution_context.project_id is None:
        raise ValueError("ToolContext.project_id is required for agent execution")

    project_row = db_repo.get_project_for_user(
        conn,
        execution_context.user_id,
        execution_context.project_id,
    )
    if project_row is None:
        repositories_agent.fail_or_retry_run(
            conn,
            run_id=run_id,
            worker_id=worker_id,
            error_message="Project not found or not owned by user.",
            retryable=False,
        )
        commit()
        return AgentRunOutcome(
            run_id=run_id,
            conversation_id=conversation_id,
            status="failed",
            iterations=0,
            tool_calls_executed=0,
            error_message="Project not found or not owned by user.",
        )

    system_msg = build_system_message(
        project_uuid=execution_context.project_id,
        project_key=project_row.get("project_id"),
        project_name=project_row.get("name"),
    )

    prior_iterations = repositories_agent.count_completed_model_iterations_for_run(
        conn,
        run_id=run_id,
    )
    all_tool_summaries: List[str] = repositories_agent.list_completed_tool_summaries_for_run(
        conn,
        run_id=run_id,
    )
    total_tools_executed = 0
    iterations_completed = prior_iterations
    iterations_this_invocation = 0
    made_invocation_progress = False

    def _has_model_headroom(now: float) -> bool:
        if work_invocation_deadline is None:
            return True
        remaining_work = work_invocation_deadline - now
        if remaining_work <= 0:
            return False
        if not made_invocation_progress:
            return True
        if total_work_budget is not None and total_work_budget >= effective_model_timeout:
            return remaining_work >= effective_model_timeout
        return True

    def _prepare_model_for_call() -> Runnable[Any, AIMessage]:
        call_deadline = (
            min(deadline, work_invocation_deadline)
            if work_invocation_deadline is not None
            else deadline
        )
        if isinstance(model, GroqChatModel):
            return model.model_copy(update={"call_deadline": call_deadline})
        return model

    def _lost_lease_outcome() -> AgentRunOutcome:
        return AgentRunOutcome(
            run_id=run_id,
            conversation_id=conversation_id,
            status="running",
            iterations=iterations_completed,
            tool_calls_executed=total_tools_executed,
        )

    def _yield_for_continuation() -> AgentRunOutcome:
        updated = repositories_agent.yield_run_for_continuation(
            conn,
            run_id=run_id,
            worker_id=worker_id,
        )
        commit()
        if updated is None:
            return _lost_lease_outcome()
        return AgentRunOutcome(
            run_id=run_id,
            conversation_id=conversation_id,
            status="queued",
            iterations=iterations_completed,
            tool_calls_executed=total_tools_executed,
        )

    def _complete_on_timeout(iter_num: int) -> AgentRunOutcome:
        final_text = _build_graceful_limit_response(
            reason="maximum execution time limit",
            gathered_summaries=all_tool_summaries,
        )
        with conn.transaction():
            history_manager.append_message(
                conn,
                conversation_id,
                AIMessage(content=final_text),
                run_id=run_id,
                metadata={"terminated_by": "run_timeout", "iteration": iter_num},
                worker_id=worker_id,
            )
            repositories_agent.append_run_step(
                conn,
                run_id=run_id,
                step_type="run_completed",
                status="completed",
                output_summary={
                    "activity": "Completed with partial findings (time limit reached)",
                    "iterations": iter_num,
                    "tool_calls_executed": total_tools_executed,
                },
                expected_worker_id=worker_id,
            )
            completed_row = repositories_agent.complete_run(
                conn,
                run_id=run_id,
                worker_id=worker_id,
            )
            if completed_row is None:
                raise repositories_agent.LeaseLostError(run_id, worker_id)
        commit()
        return AgentRunOutcome(
            run_id=run_id,
            conversation_id=conversation_id,
            status="completed",
            iterations=iter_num,
            tool_calls_executed=total_tools_executed,
            final_message=final_text,
        )

    try:
        # 1. Crash / continuation recovery check: if an earlier worker crashed or yielded
        # mid-tool-batch for this run, finish only the remaining unexecuted tool calls
        # before calling the model again.
        _, pending_calls = history_manager.get_pending_tool_calls_for_run(
            conn,
            conversation_id,
            run_id,
        )
        if pending_calls:
            batch_summaries, count, timed_out, yielded = _execute_tool_batch(
                conn=conn,
                conversation_id=conversation_id,
                run_id=run_id,
                worker_id=worker_id,
                tool_calls=pending_calls,
                history_manager=history_manager,
                tool_registry=registry,
                execution_context=execution_context,
                lease_seconds=lease_seconds,
                max_tool_result_chars=max_tool_result_chars,
                deadline=deadline,
                invocation_deadline=work_invocation_deadline,
                total_work_budget=total_work_budget,
                made_invocation_progress=made_invocation_progress,
                commit=commit,
            )
            all_tool_summaries.extend(batch_summaries)
            total_tools_executed += count
            if count > 0:
                made_invocation_progress = True
            if timed_out or time.monotonic() >= deadline:
                return _complete_on_timeout(max(1, iterations_completed))
            if yielded or (
                work_invocation_deadline is not None
                and time.monotonic() >= work_invocation_deadline
            ):
                return _yield_for_continuation()

        # 2. Main model -> tool -> model loop
        bounded_iterations = max(1, int(max_iterations))
        for iteration in range(prior_iterations + 1, bounded_iterations + 1):
            now_iter = time.monotonic()
            if now_iter >= deadline:
                return _complete_on_timeout(iteration)

            if (
                (not _has_model_headroom(now_iter))
                or (
                    max_iterations_per_invocation is not None
                    and iterations_this_invocation >= max(1, int(max_iterations_per_invocation))
                )
            ):
                return _yield_for_continuation()

            if not repositories_agent.heartbeat_run(
                conn,
                run_id=run_id,
                worker_id=worker_id,
                lease_seconds=lease_seconds,
            ):
                return _lost_lease_outcome()

            iterations_completed = iteration
            iterations_this_invocation += 1

            history_messages = history_manager.get_history(
                conn,
                conversation_id,
                active_run_id=run_id,
                worker_id=worker_id,
            )

            repositories_agent.append_run_step(
                conn,
                run_id=run_id,
                step_type="model_started",
                status="in_progress",
                input_summary={
                    "iteration": iteration,
                    "activity": (
                        "Analyzing investigation request"
                        if iteration == 1 and total_tools_executed == 0 and not all_tool_summaries
                        else "Correlating tool evidence and planning next step"
                    ),
                },
                output_summary={},
                expected_worker_id=worker_id,
                refresh_lease_seconds=lease_seconds,
            )
            commit()

            raw_response = _prepare_model_for_call().invoke([system_msg, *history_messages])
            made_invocation_progress = True
            # Fencing check: ensure another worker did not reclaim an expired lease while model was running.
            if not repositories_agent.heartbeat_run(
                conn,
                run_id=run_id,
                worker_id=worker_id,
                lease_seconds=lease_seconds,
            ):
                return _lost_lease_outcome()

            if not isinstance(raw_response, AIMessage):
                ai_message = AIMessage(content=strip_reasoning_tags(str(getattr(raw_response, "content", raw_response))))
            else:
                ai_message = raw_response.model_copy(
                    update={"content": strip_reasoning_tags(str(raw_response.content or ""))}
                )

            tool_calls = list(getattr(ai_message, "tool_calls", None) or [])

            # Case A: Model returned a final response with no tool calls
            if not tool_calls:
                final_content = str(ai_message.content or "").strip()
                if not final_content:
                    final_content = _build_graceful_limit_response(
                        reason="completion of tool analysis",
                        gathered_summaries=all_tool_summaries,
                    )
                    ai_message = AIMessage(
                        content=final_content,
                        response_metadata=getattr(ai_message, "response_metadata", {}),
                    )

                with conn.transaction():
                    history_manager.append_message(
                        conn,
                        conversation_id,
                        ai_message,
                        run_id=run_id,
                        metadata={"iteration": iteration},
                        worker_id=worker_id,
                    )
                    repositories_agent.append_run_step(
                        conn,
                        run_id=run_id,
                        step_type="model_completed",
                        status="completed",
                        input_summary={"iteration": iteration},
                        output_summary={
                            "iteration": iteration,
                            "has_tool_calls": False,
                            "tool_call_count": 0,
                            "tools_requested": [],
                        },
                        expected_worker_id=worker_id,
                    )
                    repositories_agent.append_run_step(
                        conn,
                        run_id=run_id,
                        step_type="run_completed",
                        status="completed",
                        output_summary={
                            "activity": "Prepared final investigation response",
                            "iterations": iteration,
                            "tool_calls_executed": total_tools_executed,
                        },
                        expected_worker_id=worker_id,
                    )
                    completed_row = repositories_agent.complete_run(
                        conn,
                        run_id=run_id,
                        worker_id=worker_id,
                    )
                    if completed_row is None:
                        raise repositories_agent.LeaseLostError(run_id, worker_id)
                commit()
                return AgentRunOutcome(
                    run_id=run_id,
                    conversation_id=conversation_id,
                    status="completed",
                    iterations=iteration,
                    tool_calls_executed=total_tools_executed,
                    final_message=final_content,
                )

            # Case B: Model returned tool calls.
            # Persist AIMessage(tool_calls) and `model_completed` atomically BEFORE executing any tools.
            with conn.transaction():
                history_manager.append_message(
                    conn,
                    conversation_id,
                    ai_message,
                    run_id=run_id,
                    metadata={"iteration": iteration},
                    worker_id=worker_id,
                )
                repositories_agent.append_run_step(
                    conn,
                    run_id=run_id,
                    step_type="model_completed",
                    status="completed",
                    input_summary={"iteration": iteration},
                    output_summary={
                        "iteration": iteration,
                        "has_tool_calls": True,
                        "tool_call_count": len(tool_calls),
                        "tools_requested": [str(tc.get("name") or "") for tc in tool_calls[:10]],
                    },
                    expected_worker_id=worker_id,
                    refresh_lease_seconds=lease_seconds,
                )
            commit()

            if (
                work_invocation_deadline is not None
                and time.monotonic() >= work_invocation_deadline
                and time.monotonic() < deadline
            ):
                return _yield_for_continuation()

            batch_summaries, count, timed_out, yielded = _execute_tool_batch(
                conn=conn,
                conversation_id=conversation_id,
                run_id=run_id,
                worker_id=worker_id,
                tool_calls=tool_calls,
                history_manager=history_manager,
                tool_registry=registry,
                execution_context=execution_context,
                lease_seconds=lease_seconds,
                max_tool_result_chars=max_tool_result_chars,
                deadline=deadline,
                invocation_deadline=work_invocation_deadline,
                total_work_budget=total_work_budget,
                made_invocation_progress=made_invocation_progress,
                commit=commit,
            )
            all_tool_summaries.extend(batch_summaries)
            total_tools_executed += count

            if timed_out or time.monotonic() >= deadline:
                return _complete_on_timeout(iteration)

            if yielded:
                return _yield_for_continuation()

            if (
                (work_invocation_deadline is not None and time.monotonic() >= work_invocation_deadline)
                or (
                    iteration < bounded_iterations
                    and max_iterations_per_invocation is not None
                    and iterations_this_invocation >= max(1, int(max_iterations_per_invocation))
                )
            ):
                return _yield_for_continuation()

        # 3. Reached max_iterations after completing all tool calls and persisting all ToolMessages.
        # Synthesize and persist a final assistant response so the conversation never ends on a tool_result.
        now_synth = time.monotonic()
        if now_synth >= deadline:
            return _complete_on_timeout(bounded_iterations)
        if not _has_model_headroom(now_synth):
            return _yield_for_continuation()
        if not repositories_agent.heartbeat_run(
            conn,
            run_id=run_id,
            worker_id=worker_id,
            lease_seconds=lease_seconds,
        ):
            return _lost_lease_outcome()

        updated_history = history_manager.get_history(
            conn,
            conversation_id,
            active_run_id=run_id,
            worker_id=worker_id,
        )
        call_deadline = (
            min(deadline, work_invocation_deadline)
            if work_invocation_deadline is not None
            else deadline
        )
        final_text = _invoke_final_synthesis(
            model=model,
            messages=[system_msg, *updated_history],
            reason=f"maximum iteration limit ({bounded_iterations})",
            gathered_summaries=all_tool_summaries,
            call_deadline=call_deadline,
        )
        with conn.transaction():
            history_manager.append_message(
                conn,
                conversation_id,
                AIMessage(content=final_text),
                run_id=run_id,
                metadata={"terminated_by": "max_iterations", "iterations": bounded_iterations},
                worker_id=worker_id,
            )
            repositories_agent.append_run_step(
                conn,
                run_id=run_id,
                step_type="run_completed",
                status="completed",
                output_summary={
                    "activity": "Completed investigation (iteration limit reached)",
                    "iterations": bounded_iterations,
                    "tool_calls_executed": total_tools_executed,
                },
                expected_worker_id=worker_id,
            )
            completed_row = repositories_agent.complete_run(
                conn,
                run_id=run_id,
                worker_id=worker_id,
            )
            if completed_row is None:
                raise repositories_agent.LeaseLostError(run_id, worker_id)
        commit()
        return AgentRunOutcome(
            run_id=run_id,
            conversation_id=conversation_id,
            status="completed",
            iterations=bounded_iterations,
            tool_calls_executed=total_tools_executed,
            final_message=final_text,
        )

    except repositories_agent.LeaseLostError:
        logger.warning(
            "Agent run %s aborted immediately: worker %s lost lease ownership.",
            run_id,
            worker_id,
        )
        return _lost_lease_outcome()

    except AgentModelError as exc:
        if exc.code == "lease_lost":
            logger.warning(
                "Agent run %s aborted during model call: worker %s lost lease ownership.",
                run_id,
                worker_id,
            )
            return _lost_lease_outcome()
        logger.warning("Agent run %s model error (%s): %s", run_id, exc.code, exc.message)
        with conn.transaction():
            updated_run = repositories_agent.fail_or_retry_run(
                conn,
                run_id=run_id,
                worker_id=worker_id,
                error_message=exc.message,
                retryable=exc.retryable,
            )
            final_status = (updated_run or {}).get("status", "failed")
            if updated_run is not None and final_status == "failed":
                repositories_agent.append_run_step(
                    conn,
                    run_id=run_id,
                    step_type="run_failed",
                    status="failed",
                    output_summary={
                        "error_code": exc.code,
                        "message": exc.message,
                    },
                )
        commit()
        if updated_run is None:
            return _lost_lease_outcome()
        return AgentRunOutcome(
            run_id=run_id,
            conversation_id=conversation_id,
            status=final_status,
            iterations=iterations_completed,
            tool_calls_executed=total_tools_executed,
            error_message=exc.message,
        )

    except Exception as exc:
        safe_msg = redact(f"Unexpected agent execution error: {type(exc).__name__}")
        logger.error("Agent run %s unexpected error: %s", run_id, redact(f"{type(exc).__name__}: {exc}"))
        with conn.transaction():
            updated_run = repositories_agent.fail_or_retry_run(
                conn,
                run_id=run_id,
                worker_id=worker_id,
                error_message=safe_msg,
                retryable=True,
            )
            final_status = (updated_run or {}).get("status", "failed")
            if updated_run is not None and final_status == "failed":
                repositories_agent.append_run_step(
                    conn,
                    run_id=run_id,
                    step_type="run_failed",
                    status="failed",
                    output_summary={
                        "error_code": "internal_error",
                        "message": safe_msg,
                    },
                )
        commit()
        if updated_run is None:
            return _lost_lease_outcome()
        return AgentRunOutcome(
            run_id=run_id,
            conversation_id=conversation_id,
            status=final_status,
            iterations=iterations_completed,
            tool_calls_executed=total_tools_executed,
            error_message=safe_msg,
        )
