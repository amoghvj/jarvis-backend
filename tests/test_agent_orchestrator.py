"""Unit and integration tests for HistoryManager, GroqChatModel, tools adapter, orchestrator, and worker queue."""

import json
from typing import Any, List
from uuid import uuid4

import httpx
import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from pydantic import SecretStr

from app import repositories, repositories_agent
from app.agent.history import FullHistoryManager
from app.agent.model import (
    AgentModelError,
    GroqChatModel,
    create_groq_chat_model,
    langchain_messages_to_openai,
    parse_openai_chat_response,
)
from app.agent.orchestrator import run_agent
from app.agent.tools_adapter import (
    build_langchain_tools,
    build_openai_tool_schemas,
    dispatch_tool_call,
)
from app.agent.worker import claim_and_execute_run
from app.config import get_settings
from app.tools.definitions import get_default_registry
from app.tools.schemas import ToolContext
from tests.conftest import make_project, make_user


def _seed_project_and_incident(db):
    user = make_user(db)
    proj = repositories.insert_project(db, user["id"], f"proj-{uuid4().hex[:8]}", "Checkout Project")

    inc = db.execute(
        """
        INSERT INTO public.incidents (
            project_id, title, status, severity, service_id, environment,
            fingerprint, event_count, first_seen_at, last_seen_at
        )
        VALUES (
            %s, 'KeyError: user_id in checkout', 'open', 'ERROR',
            'checkout-api', 'production', 'fp-checkout-1', 2, now(), now()
        )
        RETURNING id, title
        """,
        (proj["id"],),
    ).fetchone()

    evt = db.execute(
        """
        INSERT INTO public.telemetry_events (
            project_id, incident_id, event_id, schema_version, event_type,
            classification, event_timestamp, service_id, environment,
            severity, message, error_type, stack_trace,
            source_file, source_function, source_line, trace_id, payload
        )
        VALUES (
            %s, %s, 'evt-checkout-101', '1.0', 'error',
            'abnormal_behaviour', now(), 'checkout-api', 'production',
            'ERROR', 'KeyError: user_id', 'KeyError',
            'Traceback (most recent call last):\n  File "app/checkout.py", line 42, in process_order\n    uid = payload["user_id"]\nKeyError: ''user_id''',
            'app/checkout.py', 'process_order', 42, 'trace-abc-123',
            '{"details": {"error_type": "KeyError"}}'::jsonb
        )
        RETURNING id, event_id
        """,
        (proj["id"], inc["id"]),
    ).fetchone()

    return user, proj, inc, evt


def test_tool_adapter_exposes_all_24_canonical_tools_and_hides_context_inputs(db):
    user, proj, inc, _ = _seed_project_and_incident(db)
    reg = get_default_registry()
    schemas = build_openai_tool_schemas(reg)
    assert len(schemas) == 24

    by_name = {s["function"]["name"]: s["function"] for s in schemas}
    assert set(by_name) == set(reg.names())

    # Every telemetry tool must hide `project_id` from the LLM-facing parameters
    for telemetry_tool in (
        "get_incident",
        "get_related_events",
        "get_incident_timeline",
        "search_events",
        "get_event_details",
        "get_trace",
        "get_service_context",
    ):
        props = by_name[telemetry_tool]["parameters"]["properties"]
        assert "project_id" not in props
        assert "user_id" not in props

    # StructuredTool adaptation works and dispatches through execute_tool
    ctx = ToolContext(user_id=user["id"], project_id=proj["id"], conn=db)
    lc_tools = build_langchain_tools(ctx, reg)
    assert len(lc_tools) == 24
    get_inc_tool = next(t for t in lc_tools if t.name == "get_incident")
    out_json = get_inc_tool.invoke({"incident_id": str(inc["id"]), "include_events": True})
    parsed = json.loads(out_json)
    assert parsed["ok"] is True
    assert parsed["operation"] == "get_incident"
    assert parsed["data"]["incident"]["id"] == str(inc["id"])


def test_dispatch_tool_call_handles_invalid_names_args_and_context_smuggling(db):
    user, proj, inc, _ = _seed_project_and_incident(db)
    ctx = ToolContext(user_id=user["id"], project_id=proj["id"], conn=db)

    # 1. Unknown tool name
    unknown = dispatch_tool_call(
        tool_name="delete_production_database",
        raw_args={},
        tool_call_id="tc_1",
        context=ctx,
    )
    assert unknown.ok is False
    assert unknown.envelope["error"]["code"] == "unknown_tool"
    assert unknown.tool_message.tool_call_id == "tc_1"

    # 2. Malformed non-JSON string arguments
    malformed = dispatch_tool_call(
        tool_name="get_incident",
        raw_args="{not-valid-json",
        tool_call_id="tc_2",
        context=ctx,
    )
    assert malformed.ok is False
    assert malformed.envelope["error"]["code"] == "invalid_input"

    # 3. Attempt by LLM to override project_id in tool arguments is rejected by executor
    smuggled = dispatch_tool_call(
        tool_name="get_incident",
        raw_args={"incident_id": str(inc["id"]), "project_id": str(uuid4())},
        tool_call_id="tc_3",
        context=ctx,
    )
    assert smuggled.ok is False
    assert smuggled.envelope["error"]["code"] == "invalid_input"
    assert "project_id" in smuggled.envelope["error"]["message"]


def test_groq_chat_model_serialization_parsing_and_env_config(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_fake_key_do_not_expose_12345")
    monkeypatch.setenv("GROQ_ENDPOINT", "https://api.groq.com/openai/v1")
    monkeypatch.setenv("GROQ_MODEL", "qwen/qwen3.8-27b")
    get_settings.cache_clear()

    captured_requests: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        body = json.loads(request.content.decode("utf-8"))
        # First call returns a tool call; second call returns final text with <think> tag stripped
        if len(captured_requests) == 1:
            assert body["model"] == "qwen/qwen3.8-27b"
            assert len(body["tools"]) == 24
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-1",
                    "model": "qwen/qwen3.8-27b",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "tool_calls",
                            "message": {
                                "role": "assistant",
                                "content": "<think>private reasoning</think>",
                                "tool_calls": [
                                    {
                                        "id": "call_abc",
                                        "type": "function",
                                        "function": {
                                            "name": "get_incident",
                                            "arguments": '{"incident_id": "00000000-0000-0000-0000-0000000001a1"}',
                                        },
                                    }
                                ],
                            },
                        }
                    ],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
                },
            )
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-2",
                "model": "qwen/qwen3.8-27b",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": "<think>internal thought</think>Confirmed root cause: missing user_id key.",
                        },
                    }
                ],
                "usage": {"prompt_tokens": 150, "completion_tokens": 30, "total_tokens": 180},
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    model = create_groq_chat_model(get_settings(), http_client=client)

    msg1 = model.invoke([SystemMessage(content="sys"), HumanMessage(content="investigate")])
    assert msg1.content == ""  # <think> block stripped
    assert msg1.tool_calls == [
        {
            "id": "call_abc",
            "name": "get_incident",
            "args": {"incident_id": "00000000-0000-0000-0000-0000000001a1"},
            "type": "tool_call",
        }
    ]

    tool_msg = ToolMessage(
        content='{"ok": true, "operation": "get_incident"}',
        tool_call_id="call_abc",
        name="get_incident",
    )
    msg2 = model.invoke([SystemMessage(content="sys"), HumanMessage(content="investigate"), msg1, tool_msg])
    assert msg2.content == "Confirmed root cause: missing user_id key."
    assert msg2.tool_calls == []

    # Verify OpenAI wire format sent on second call included assistant tool_calls and tool role message
    second_payload = json.loads(captured_requests[1].content.decode("utf-8"))
    roles = [m["role"] for m in second_payload["messages"]]
    assert roles == ["system", "user", "assistant", "tool"]
    assert second_payload["messages"][3]["tool_call_id"] == "call_abc"


def test_history_manager_reconstructs_and_repairs_dangling_tool_calls(db):
    user, proj, _, _ = _seed_project_and_incident(db)
    conv = repositories_agent.create_conversation(db, user_id=user["id"], project_id=proj["id"])
    cid = conv["id"]
    hm = FullHistoryManager()

    # Append HumanMessage and an AIMessage with 2 tool calls, but only 1 ToolMessage (simulating a prior crashed run)
    hm.append_message(db, cid, HumanMessage(content="First question"))
    hm.append_message(
        db,
        cid,
        AIMessage(
            content="",
            tool_calls=[
                {"id": "tc_done", "name": "list_installations", "args": {}, "type": "tool_call"},
                {"id": "tc_dangling", "name": "search_events", "args": {"severity": "ERROR"}, "type": "tool_call"},
            ],
        ),
    )
    hm.append_message(
        db,
        cid,
        ToolMessage(
            content='{"ok": true, "operation": "list_installations"}',
            tool_call_id="tc_done",
            name="list_installations",
        ),
    )
    hm.append_message(db, cid, HumanMessage(content="Follow-up question"))

    # When get_history is called, the dangling `tc_dangling` tool call from the earlier turn is automatically paired
    # with a synthetic interrupted_run ToolMessage so the LLM never receives an invalid sequence.
    history = hm.get_history(db, cid)
    tool_msgs = [m for m in history if isinstance(m, ToolMessage)]
    assert {m.tool_call_id for m in tool_msgs} == {"tc_done", "tc_dangling"}


def test_orchestrator_multi_turn_tool_execution_and_activity_steps(db):
    user, proj, inc, evt = _seed_project_and_incident(db)
    conv = repositories_agent.create_conversation(db, user_id=user["id"], project_id=proj["id"])
    cid = conv["id"]

    _, run, _ = repositories_agent.submit_user_message_and_create_run(
        db,
        user_id=user["id"],
        project_id=proj["id"],
        conversation_id=cid,
        content=f"Investigate incident {inc['id']}",
    )

    call_step = {"n": 0}

    def fake_llm(messages: List[BaseMessage]) -> AIMessage:
        call_step["n"] += 1
        # Verify SystemMessage is prepended in memory
        assert isinstance(messages[0], SystemMessage)
        if call_step["n"] == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_inc_1",
                        "name": "get_incident",
                        "args": {"incident_id": str(inc["id"]), "include_events": True},
                        "type": "tool_call",
                    }
                ],
            )
        if call_step["n"] == 2:
            # Verify the ToolMessage from get_incident was passed back to the model
            last_msg = messages[-1]
            assert isinstance(last_msg, ToolMessage)
            assert last_msg.tool_call_id == "call_inc_1"
            env = json.loads(str(last_msg.content))
            assert env["ok"] is True
            assert env["data"]["incident"]["title"] == "KeyError: user_id in checkout"
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_evt_2",
                        "name": "get_event_details",
                        "args": {"event_id": evt["event_id"]},
                        "type": "tool_call",
                    }
                ],
            )
        last_msg = messages[-1]
        assert isinstance(last_msg, ToolMessage)
        assert last_msg.tool_call_id == "call_evt_2"
        env = json.loads(str(last_msg.content))
        assert "app/checkout.py" in env["data"]["event"]["stack_trace"]
        return AIMessage(
            content=(
                "### Confirmed Facts\n"
                "- Incident `KeyError: user_id in checkout` occurred in `app/checkout.py:42` (`process_order`).\n\n"
                "### Root-Cause Hypothesis\n"
                "- `payload['user_id']` is accessed directly without checking whether the key is present.\n\n"
                "### Recommended Remediation\n"
                "- Validate `user_id` before indexing `payload`."
            )
        )

    outcome = claim_and_execute_run(
        db,
        run_id=run["id"],
        worker_id="worker-test-1",
        model=RunnableLambda(fake_llm),
    )
    assert outcome is not None
    assert outcome.status == "completed"
    assert outcome.iterations == 3
    assert outcome.tool_calls_executed == 2
    assert "app/checkout.py:42" in (outcome.final_message or "")

    # Verify persisted run status and steps
    stored_run = repositories_agent.get_run(db, conversation_id=cid, run_id=run["id"])
    assert stored_run["status"] == "completed"
    assert stored_run["completed_at"] is not None

    steps, latest_step = repositories_agent.list_run_steps(db, run_id=run["id"], after_step=0)
    step_types = [s["step_type"] for s in steps]
    assert step_types == [
        "model_started",
        "model_completed",
        "tool_started",
        "tool_completed",
        "model_started",
        "model_completed",
        "tool_started",
        "tool_completed",
        "model_started",
        "model_completed",
        "run_completed",
    ]
    assert latest_step == 11

    # Verify visible messages vs full history
    visible, _ = repositories_agent.list_visible_messages(db, conversation_id=cid)
    assert [(m["role"], m["message_type"]) for m in visible] == [
        ("user", "text"),
        ("assistant", "text"),
    ]
    all_rows = repositories_agent.list_all_messages_for_history(db, conversation_id=cid)
    assert [(r["role"], r["message_type"]) for r in all_rows] == [
        ("user", "text"),
        ("assistant", "tool_call"),
        ("tool", "tool_result"),
        ("assistant", "tool_call"),
        ("tool", "tool_result"),
        ("assistant", "text"),
    ]


def test_iteration_limit_terminates_gracefully_without_dangling_tool_calls(db):
    user, proj, inc, _ = _seed_project_and_incident(db)
    conv = repositories_agent.create_conversation(db, user_id=user["id"], project_id=proj["id"])
    cid = conv["id"]

    _, run, _ = repositories_agent.submit_user_message_and_create_run(
        db,
        user_id=user["id"],
        project_id=proj["id"],
        conversation_id=cid,
        content="Loop forever",
    )
    claimed = repositories_agent.claim_run(db, worker_id="worker-loop", run_id=run["id"])
    assert claimed is not None

    counter = {"n": 0}

    def endless_tool_model(messages: List[BaseMessage]) -> AIMessage:
        counter["n"] += 1
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "id": f"call_loop_{counter['n']}",
                    "name": "get_incident",
                    "args": {"incident_id": str(inc["id"]), "include_events": False},
                    "type": "tool_call",
                }
            ],
        )

    outcome = run_agent(
        conversation_id=cid,
        run_id=run["id"],
        worker_id="worker-loop",
        history_manager=FullHistoryManager(),
        model=RunnableLambda(endless_tool_model),
        execution_context=ToolContext(user_id=user["id"], project_id=proj["id"], conn=db),
        max_iterations=2,
        run_timeout_seconds=60,
    )
    assert outcome.status == "completed"
    assert outcome.iterations == 2
    assert outcome.tool_calls_executed == 2
    assert outcome.final_message is not None

    # Every persisted tool_call message MUST have its matching tool_result message,
    # and the conversation MUST end with a visible assistant text message.
    all_rows = repositories_agent.list_all_messages_for_history(db, conversation_id=cid)
    assert [(r["role"], r["message_type"]) for r in all_rows] == [
        ("user", "text"),
        ("assistant", "tool_call"),
        ("tool", "tool_result"),
        ("assistant", "tool_call"),
        ("tool", "tool_result"),
        ("assistant", "text"),
    ]


def test_crash_recovery_resumes_mid_batch_without_reexecuting_completed_tools(db):
    user, proj, inc, evt = _seed_project_and_incident(db)
    conv = repositories_agent.create_conversation(db, user_id=user["id"], project_id=proj["id"])
    cid = conv["id"]

    _, run, _ = repositories_agent.submit_user_message_and_create_run(
        db,
        user_id=user["id"],
        project_id=proj["id"],
        conversation_id=cid,
        content="Check incident and event",
    )

    # Worker 1 claims the run and crashes after persisting AIMessage(2 tool_calls) and only the 1st ToolMessage
    claimed1 = repositories_agent.claim_run(db, worker_id="worker-1", lease_seconds=10, run_id=run["id"])
    assert claimed1 is not None and claimed1["attempt_count"] == 1

    hm = FullHistoryManager()
    hm.append_message(
        db,
        cid,
        AIMessage(
            content="",
            tool_calls=[
                {"id": "call_already_done", "name": "get_incident", "args": {"incident_id": str(inc["id"])}, "type": "tool_call"},
                {"id": "call_still_pending", "name": "get_event_details", "args": {"event_id": evt["event_id"]}, "type": "tool_call"},
            ],
        ),
        run_id=run["id"],
    )
    hm.append_message(
        db,
        cid,
        ToolMessage(
            content='{"ok": true, "operation": "get_incident", "summary": "Already executed by worker-1"}',
            tool_call_id="call_already_done",
            name="get_incident",
        ),
        run_id=run["id"],
    )

    # While worker-1's lease is still valid, worker-2 cannot claim the run
    assert repositories_agent.claim_run(db, worker_id="worker-2", run_id=run["id"]) is None

    # Simulate worker-1 lease expiration
    db.execute(
        "UPDATE public.agent_runs SET lease_expires_at = now() - interval '5 seconds' WHERE id = %s",
        (run["id"],),
    )

    # Worker-2 now recovers the expired lease and finishes ONLY `call_still_pending` before invoking the LLM
    def recovery_model(messages: List[BaseMessage]) -> AIMessage:
        tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
        assert [m.tool_call_id for m in tool_msgs] == ["call_already_done", "call_still_pending"]
        return AIMessage(content="Recovered and completed investigation.")

    outcome = claim_and_execute_run(
        db,
        run_id=run["id"],
        worker_id="worker-2",
        model=RunnableLambda(recovery_model),
    )
    assert outcome is not None
    assert outcome.status == "completed"
    # Only the 1 pending tool call was executed by worker-2!
    assert outcome.tool_calls_executed == 1

    all_rows = repositories_agent.list_all_messages_for_history(db, conversation_id=cid)
    tool_result_ids = [r["tool_call_id"] for r in all_rows if r["role"] == "tool"]
    assert tool_result_ids == ["call_already_done", "call_still_pending"]


def test_tool_failure_is_returned_to_model_as_structured_error_without_crashing_run(db):
    user, proj, _, _ = _seed_project_and_incident(db)
    conv = repositories_agent.create_conversation(db, user_id=user["id"], project_id=proj["id"])
    cid = conv["id"]

    _, run, _ = repositories_agent.submit_user_message_and_create_run(
        db,
        user_id=user["id"],
        project_id=proj["id"],
        conversation_id=cid,
        content="Investigate non-existent incident",
    )

    step = {"n": 0}
    missing_id = str(uuid4())

    def model_recovering_from_tool_error(messages: List[BaseMessage]) -> AIMessage:
        step["n"] += 1
        if step["n"] == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_missing_inc",
                        "name": "get_incident",
                        "args": {"incident_id": missing_id},
                        "type": "tool_call",
                    }
                ],
            )
        last_msg = messages[-1]
        assert isinstance(last_msg, ToolMessage)
        env = json.loads(str(last_msg.content))
        assert env["ok"] is False
        assert env["error"]["code"] == "incident_not_found"
        return AIMessage(content=f"Incident {missing_id} was not found in this project.")

    outcome = claim_and_execute_run(
        db,
        run_id=run["id"],
        worker_id="worker-tool-err",
        model=RunnableLambda(model_recovering_from_tool_error),
    )
    assert outcome is not None
    assert outcome.status == "completed"
    assert "not found in this project" in (outcome.final_message or "")

    steps, _ = repositories_agent.list_run_steps(db, run_id=run["id"], after_step=0)
    assert any(s["step_type"] == "tool_failed" and s["tool_name"] == "get_incident" for s in steps)


def test_model_error_retries_when_recoverable_and_fails_when_terminal(db):
    user, proj, _, _ = _seed_project_and_incident(db)
    conv = repositories_agent.create_conversation(db, user_id=user["id"], project_id=proj["id"])
    cid = conv["id"]

    _, run, _ = repositories_agent.submit_user_message_and_create_run(
        db,
        user_id=user["id"],
        project_id=proj["id"],
        conversation_id=cid,
        content="Test model failure handling",
        max_attempts=2,
    )

    # 1. First attempt fails with a retryable model timeout -> run transitions back to 'queued'
    def failing_retryable_model(messages: List[BaseMessage]) -> AIMessage:
        raise AgentModelError("model_timeout", "The language model request timed out.", retryable=True)

    outcome1 = claim_and_execute_run(
        db,
        run_id=run["id"],
        worker_id="worker-err-1",
        model=RunnableLambda(failing_retryable_model),
    )
    assert outcome1 is not None
    assert outcome1.status == "queued"

    stored1 = repositories_agent.get_run(db, conversation_id=cid, run_id=run["id"])
    assert stored1["status"] == "queued"
    assert stored1["attempt_count"] == 1
    assert stored1["next_retry_at"] is not None

    # Make next_retry_at immediately eligible and fail the 2nd (final) attempt -> terminal 'failed'
    db.execute("UPDATE public.agent_runs SET next_retry_at = now() - interval '1 second' WHERE id = %s", (run["id"],))

    outcome2 = claim_and_execute_run(
        db,
        run_id=run["id"],
        worker_id="worker-err-2",
        model=RunnableLambda(failing_retryable_model),
    )
    assert outcome2 is not None
    assert outcome2.status == "failed"

    stored2 = repositories_agent.get_run(db, conversation_id=cid, run_id=run["id"])
    assert stored2["status"] == "failed"
    assert stored2["completed_at"] is not None
    steps, _ = repositories_agent.list_run_steps(db, run_id=run["id"], after_step=0)
    assert steps[-1]["step_type"] == "run_failed"


def test_invocation_timeout_yields_mid_tool_batch_and_resumes_cleanly(db):
    """Verify that when invocation_timeout_seconds expires mid-tool-batch (while overall
    run_timeout_seconds still has time remaining), the orchestrator yields the run back to
    'queued' without writing fake timeout ToolMessages, and the next invocation resumes the
    remaining pending tool calls seamlessly.
    """
    import time

    user, proj, inc, evt = _seed_project_and_incident(db)
    conv = repositories_agent.create_conversation(db, user_id=user["id"], project_id=proj["id"])
    cid = conv["id"]

    _, run, _ = repositories_agent.submit_user_message_and_create_run(
        db,
        user_id=user["id"],
        project_id=proj["id"],
        conversation_id=cid,
        content="Investigate incident and event with short invocation slice",
    )

    model_calls = {"n": 0}

    def two_tool_then_final_model(messages: List[BaseMessage]) -> AIMessage:
        model_calls["n"] += 1
        if model_calls["n"] == 1:
            # Sleep briefly so invocation_timeout_seconds (0.05s) expires before the 2nd tool in the batch
            time.sleep(0.06)
            return AIMessage(
                content="",
                tool_calls=[
                    {"id": "call_1", "name": "get_incident", "args": {"incident_id": str(inc["id"])}, "type": "tool_call"},
                    {"id": "call_2", "name": "get_event_details", "args": {"event_id": evt["event_id"]}, "type": "tool_call"},
                ],
            )
        tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
        assert [m.tool_call_id for m in tool_msgs] == ["call_1", "call_2"]
        return AIMessage(content="Finished both tools across two invocations.")

    # Slice 1: model returns 2 tool calls, invocation deadline already reached -> yields back to 'queued'
    outcome1 = claim_and_execute_run(
        db,
        run_id=run["id"],
        worker_id="worker-slice-1",
        model=RunnableLambda(two_tool_then_final_model),
        invocation_timeout_seconds=0.05,
    )
    assert outcome1 is not None
    assert outcome1.status == "queued"
    assert outcome1.tool_calls_executed == 0

    stored_mid = repositories_agent.get_run(db, conversation_id=cid, run_id=run["id"])
    assert stored_mid["status"] == "queued"
    assert stored_mid["worker_id"] is None

    # Slice 2: with generous invocation_timeout_seconds, executes pending call_1 and call_2, then model final answer
    outcome2 = claim_and_execute_run(
        db,
        run_id=run["id"],
        worker_id="worker-slice-2",
        model=RunnableLambda(two_tool_then_final_model),
        invocation_timeout_seconds=60.0,
    )
    assert outcome2 is not None
    assert outcome2.status == "completed"
    assert outcome2.tool_calls_executed == 2
    assert outcome2.final_message == "Finished both tools across two invocations."


def test_overall_timeout_preserved_across_slices_and_enforced_from_db(db):
    """Verify timeout_at is set once on creation, never extended on claim/yield,
    and remaining_timeout_seconds limits subsequent slices even below 5s.
    """
    import time

    user, proj, inc, _ = _seed_project_and_incident(db)
    conv = repositories_agent.create_conversation(db, user_id=user["id"], project_id=proj["id"])
    cid = conv["id"]

    _, run, _ = repositories_agent.submit_user_message_and_create_run(
        db,
        user_id=user["id"],
        project_id=proj["id"],
        conversation_id=cid,
        content="Test overall timeout budget across invocations",
        timeout_seconds=900,
    )
    initial_timeout_at = run["timeout_at"]
    assert initial_timeout_at is not None

    # Slice 1 yields cleanly; timeout_at must remain completely unchanged.
    model_calls = {"n": 0}

    def yielding_then_slow_model(messages: List[BaseMessage]) -> AIMessage:
        model_calls["n"] += 1
        if model_calls["n"] == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_budget_1",
                        "name": "get_incident",
                        "args": {"incident_id": str(inc["id"]), "include_events": False},
                        "type": "tool_call",
                    }
                ],
            )
        if model_calls["n"] == 2:
            time.sleep(0.25)
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_budget_2",
                        "name": "get_incident",
                        "args": {"incident_id": str(inc["id"]), "include_events": False},
                        "type": "tool_call",
                    }
                ],
            )
        raise AssertionError("Model must not be called a 3rd time after overall timeout expires!")

    outcome1 = claim_and_execute_run(
        db,
        run_id=run["id"],
        worker_id="worker-budget-1",
        model=RunnableLambda(yielding_then_slow_model),
        max_iterations_per_invocation=1,
        invocation_timeout_seconds=60.0,
    )
    assert outcome1 is not None and outcome1.status == "queued"

    mid_run = repositories_agent.get_run(db, conversation_id=cid, run_id=run["id"])
    assert mid_run["timeout_at"] == initial_timeout_at

    # Shrink timeout_at to 0.15s from now (< 5s!), verifying worker respects remaining DB budget
    db.execute(
        "UPDATE public.agent_runs SET timeout_at = clock_timestamp() + interval '0.15 seconds' WHERE id = %s",
        (run["id"],),
    )
    outcome2 = claim_and_execute_run(
        db,
        run_id=run["id"],
        worker_id="worker-budget-2",
        model=RunnableLambda(yielding_then_slow_model),
        invocation_timeout_seconds=240.0,
    )
    assert outcome2 is not None
    assert outcome2.status == "completed"
    # Ran iteration 2, saw overall deadline expired (< 0.15s) instead of running a fresh 900s / 240s budget
    assert model_calls["n"] == 2
    assert "maximum execution time limit" in (outcome2.final_message or "")

    # Also verify a queued run whose timeout_at elapsed between slices expires cleanly and cannot be claimed again
    _, run_expired, _ = repositories_agent.submit_user_message_and_create_run(
        db,
        user_id=user["id"],
        project_id=proj["id"],
        conversation_id=cid,
        content="Expired between slices",
        timeout_seconds=900,
    )
    db.execute(
        "UPDATE public.agent_runs SET timeout_at = clock_timestamp() - interval '1 second' WHERE id = %s",
        (run_expired["id"],),
    )
    assert claim_and_execute_run(db, run_id=run_expired["id"], worker_id="w-expired", model=RunnableLambda(yielding_then_slow_model)) is None
    expired_row = repositories_agent.get_run(db, conversation_id=cid, run_id=run_expired["id"])
    assert expired_row["status"] == "failed"
    assert "timed out" in (expired_row["error_message"] or "").lower()


def test_tool_call_recovery_across_all_three_interruption_points_and_lease_fencing(db):
    """Verify tool-call recovery and lease protection:
    1. Invocation ends BEFORE tool execution (AIMessage persisted, 0 tools run).
    2. Invocation ends DURING tool execution (tool_started persisted, crash before ToolMessage).
    3. Invocation ends AFTER 1st ToolMessage persisted -> 2nd tool recovered, 1st NOT repeated.
    4. Active retrying GroqChatModel refreshes lease via heartbeat_callback; superseded worker
       is fenced off by heartbeat_run returning False.
    """
    user, proj, inc, evt = _seed_project_and_incident(db)
    conv = repositories_agent.create_conversation(db, user_id=user["id"], project_id=proj["id"])
    cid = conv["id"]

    _, run, _ = repositories_agent.submit_user_message_and_create_run(
        db,
        user_id=user["id"],
        project_id=proj["id"],
        conversation_id=cid,
        content="Test 3 interruption points",
    )

    executed_calls: List[str] = []
    from app.tools.definitions import get_default_registry
    from app.tools.executor import execute_tool

    real_reg = get_default_registry()

    class _TrackingModelAndCrash:
        pass

    # Phase 1: Claim run-1 with worker-a, model emits 2 tool calls, and simulate termination BEFORE tc_1 runs
    # by pre-invoking or using invocation_timeout_seconds=0.05 with slight sleep in model
    hm = FullHistoryManager()
    claimed_a = repositories_agent.claim_run(db, worker_id="worker-a", lease_seconds=60, run_id=run["id"])
    assert claimed_a is not None
    hm.append_message(
        db,
        cid,
        AIMessage(
            content="",
            tool_calls=[
                {"id": "tc_before_1", "name": "get_incident", "args": {"incident_id": str(inc["id"]), "include_events": False}, "type": "tool_call"},
                {"id": "tc_after_2", "name": "get_event_details", "args": {"event_id": evt["event_id"]}, "type": "tool_call"},
            ],
        ),
        run_id=run["id"],
    )
    repositories_agent.append_run_step(db, run_id=run["id"], step_type="model_completed", status="completed", output_summary={"iteration": 1})
    repositories_agent.yield_run_for_continuation(db, run_id=run["id"], worker_id="worker-a")

    # Phase 2: worker-b claims and simulates crash DURING tc_before_1 (`tool_started` written, no ToolMessage)
    claimed_b = repositories_agent.claim_run(db, worker_id="worker-b", lease_seconds=60, run_id=run["id"])
    assert claimed_b is not None
    repositories_agent.append_run_step(
        db,
        run_id=run["id"],
        step_type="tool_started",
        tool_name="get_incident",
        status="in_progress",
        input_summary={"activity": "Simulated crash during tool call"},
    )
    # While lease is active, worker-c CANNOT steal it prematurely!
    assert repositories_agent.claim_run(db, worker_id="worker-c", lease_seconds=60, run_id=run["id"]) is None

    # Expire worker-b's lease (simulating ungraceful serverless termination)
    db.execute(
        "UPDATE public.agent_runs SET lease_expires_at = clock_timestamp() - interval '5 seconds' WHERE id = %s",
        (run["id"],),
    )

    # Track actual tool calls executed via wrapper around get_incident and get_event_details
    from dataclasses import replace
    orig_tools = {name: real_reg._tools[name] for name in ("get_incident", "get_event_details")}

    def make_spy(name):
        orig_handler = orig_tools[name].handler
        def spy(inp, ctx):
            executed_calls.append(name)
            return orig_handler(inp, ctx)
        return spy

    try:
        # Phase 3: Persist ToolMessage for tc_before_1 BEFORE worker-c runs, simulating interruption AFTER 1st tool result
        _, pending_before = hm.get_pending_tool_calls_for_run(db, cid, run["id"])
        assert [c["id"] for c in pending_before] == ["tc_before_1", "tc_after_2"]

        # Simulate tc_before_1 finished and persisted its ToolMessage right before interruption
        hm.append_message(
            db,
            cid,
            ToolMessage(
                content='{"ok": true, "operation": "get_incident", "summary": "Persisted tc_before_1"}',
                tool_call_id="tc_before_1",
                name="get_incident",
            ),
            run_id=run["id"],
        )

        # Verify GroqChatModel refreshes active lease on retry via heartbeat_callback
        attempts = {"count": 0}
        def flaky_groq(request: httpx.Request) -> httpx.Response:
            attempts["count"] += 1
            if attempts["count"] == 1:
                return httpx.Response(503, json={"error": {"message": "transient upstream error"}})
            return httpx.Response(
                200,
                json={
                    "id": "resp-ok",
                    "model": "qwen/qwen3.8-27b",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": "Recovered after single remaining tool."},
                        }
                    ],
                },
            )

        for k in orig_tools:
            real_reg._tools[k] = replace(orig_tools[k], handler=make_spy(k))

        hb_calls = {"count": 0}
        def hb_cb():
            hb_calls["count"] += 1

        groq = create_groq_chat_model(
            get_settings(),
            bind_default_tools=False,
            http_client=httpx.Client(transport=httpx.MockTransport(flaky_groq)),
            heartbeat_callback=hb_cb,
        )
        groq = groq.model_copy(update={"api_key": SecretStr("test-key")})

        outcome_c = claim_and_execute_run(
            db,
            run_id=run["id"],
            worker_id="worker-c",
            model=groq,
            tool_registry=real_reg,
        )
        assert outcome_c is not None
        assert outcome_c.status == "completed"
        # ONLY tc_after_2 (get_event_details) ran! tc_before_1 (get_incident) was NEVER repeated!
        assert executed_calls == ["get_event_details"]
        assert outcome_c.tool_calls_executed == 1
        assert hb_calls["count"] == 1

        # Phase 4: Verify fencing guard stops superseded worker from persisting messages if lease is taken mid-model-call
        _, fenced_run, _ = repositories_agent.submit_user_message_and_create_run(
            db,
            user_id=user["id"],
            project_id=proj["id"],
            conversation_id=cid,
            content="Test superseded worker fencing",
        )

        def hijacking_model(messages: List[BaseMessage]) -> AIMessage:
            # Simulate another worker reclaiming an expired lease while this worker was stalled in model.invoke
            db.execute(
                "UPDATE public.agent_runs SET worker_id = 'new-winner-worker' WHERE id = %s",
                (fenced_run["id"],),
            )
            return AIMessage(content="STALE WORKER OUTPUT THAT MUST NOT BE PERSISTED")

        fenced_outcome = claim_and_execute_run(
            db,
            run_id=fenced_run["id"],
            worker_id="stale-worker",
            model=RunnableLambda(hijacking_model),
            tool_registry=real_reg,
        )
        assert fenced_outcome is not None
        assert fenced_outcome.status == "running"
        asst_msgs = repositories_agent.list_assistant_messages_for_run(
            db,
            conversation_id=cid,
            run_id=fenced_run["id"],
        )
        assert asst_msgs == []
    finally:
        for k, orig in orig_tools.items():
            real_reg._tools[k] = orig


def test_stale_worker_fenced_inside_persistence_transactions_across_model_tool_and_synthesis(db, monkeypatch):
    """Simulate losing lease BETWEEN a successful heartbeat_run and subsequent DB writes
    (during model completion, during tool batch execution, during yield, and during final
    synthesis). Inside-transaction fencing must abort immediately with zero writes.
    """
    user, proj, inc, evt = _seed_project_and_incident(db)
    conv = repositories_agent.create_conversation(db, user_id=user["id"], project_id=proj["id"])
    cid = conv["id"]

    # Case A: Lease lost AFTER post-model heartbeat_run succeeds, right BEFORE persistence transaction
    _, run_a, _ = repositories_agent.submit_user_message_and_create_run(
        db,
        user_id=user["id"],
        project_id=proj["id"],
        conversation_id=cid,
        content="Case A: steal lease between heartbeat and message insert",
    )
    orig_hb = repositories_agent.heartbeat_run
    hb_count_a = {"n": 0}
    hijacked_a = {"done": False}

    def hijack_after_second_heartbeat(conn, **kwargs):
        ok = orig_hb(conn, **kwargs)
        if ok and kwargs.get("run_id") == run_a["id"] and kwargs.get("worker_id") == "worker-stale-a":
            hb_count_a["n"] += 1
            if hb_count_a["n"] == 2:
                hijacked_a["done"] = True
                conn.execute(
                    "UPDATE public.agent_runs SET worker_id = 'worker-winner-a' WHERE id = %s",
                    (run_a["id"],),
                )
        return ok

    monkeypatch.setattr(repositories_agent, "heartbeat_run", hijack_after_second_heartbeat)
    outcome_a = claim_and_execute_run(
        db,
        run_id=run_a["id"],
        worker_id="worker-stale-a",
        model=RunnableLambda(lambda _msgs: AIMessage(content="STALE FINAL MESSAGE")),
    )
    monkeypatch.undo()

    assert hijacked_a["done"] is True
    assert outcome_a is not None and outcome_a.status == "running"
    assert repositories_agent.list_assistant_messages_for_run(db, conversation_id=cid, run_id=run_a["id"]) == []
    run_a_row = repositories_agent.get_run(db, conversation_id=cid, run_id=run_a["id"])
    assert run_a_row["status"] == "running" and run_a_row["worker_id"] == "worker-winner-a"

    # Yield winner cleanly so next subtests can have active runs in this conversation
    repositories_agent.complete_run(db, run_id=run_a["id"], worker_id="worker-winner-a")

    # Case B: Lease lost DURING tool 1 execution inside _execute_tool_batch:
    # tool 1's ToolMessage + tool_completed must be rolled back, and tool 2 must never execute!
    _, run_b, _ = repositories_agent.submit_user_message_and_create_run(
        db,
        user_id=user["id"],
        project_id=proj["id"],
        conversation_id=cid,
        content="Case B: steal lease during tool 1",
    )
    real_reg = get_default_registry()
    from dataclasses import replace

    orig_inc = real_reg._tools["get_incident"]
    orig_evt = real_reg._tools["get_event_details"]
    tool2_called = {"flag": False}

    def hijack_in_tool1(ctx, inp):
        db.execute(
            "UPDATE public.agent_runs SET worker_id = 'worker-winner-b' WHERE id = %s",
            (run_b["id"],),
        )
        return orig_inc.handler(ctx, inp)

    def spy_tool2(ctx, inp):
        tool2_called["flag"] = True
        return orig_evt.handler(ctx, inp)

    try:
        real_reg._tools["get_incident"] = replace(orig_inc, handler=hijack_in_tool1)
        real_reg._tools["get_event_details"] = replace(orig_evt, handler=spy_tool2)

        def two_tools_model(_msgs):
            return AIMessage(
                content="",
                tool_calls=[
                    {"id": "tc_b1", "name": "get_incident", "args": {"incident_id": str(inc["id"])}, "type": "tool_call"},
                    {"id": "tc_b2", "name": "get_event_details", "args": {"event_id": evt["event_id"]}, "type": "tool_call"},
                ],
            )

        outcome_b = claim_and_execute_run(
            db,
            run_id=run_b["id"],
            worker_id="worker-stale-b",
            model=RunnableLambda(two_tools_model),
            tool_registry=real_reg,
        )
        assert outcome_b is not None and outcome_b.status == "running"
        assert tool2_called["flag"] is False
        b_rows = repositories_agent.list_all_messages_for_history(db, conversation_id=cid)
        assert not any(r.get("run_id") == run_b["id"] and r["role"] == "tool" for r in b_rows)
        b_steps, _ = repositories_agent.list_run_steps(db, run_id=run_b["id"], after_step=0)
        assert not any(s["step_type"] in ("tool_completed", "tool_failed") for s in b_steps)
    finally:
        real_reg._tools["get_incident"] = orig_inc
        real_reg._tools["get_event_details"] = orig_evt

    repositories_agent.complete_run(db, run_id=run_b["id"], worker_id="worker-winner-b")

    # Case C: Lease lost during final synthesis (_invoke_final_synthesis) after max_iterations=1
    _, run_c, _ = repositories_agent.submit_user_message_and_create_run(
        db,
        user_id=user["id"],
        project_id=proj["id"],
        conversation_id=cid,
        content="Case C: steal lease during Step 3 final synthesis",
    )
    repositories_agent.claim_run(db, worker_id="worker-stale-c", lease_seconds=90, run_id=run_c["id"])
    synth_calls = {"n": 0}

    def loop_then_hijack_synthesis(_msgs, **_kwargs):
        synth_calls["n"] += 1
        if synth_calls["n"] == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {"id": "tc_c1", "name": "get_incident", "args": {"incident_id": str(inc["id"])}, "type": "tool_call"},
                ],
            )
        # During Step 3 synthesis call, another worker claims the lease
        db.execute(
            "UPDATE public.agent_runs SET worker_id = 'worker-winner-c' WHERE id = %s",
            (run_c["id"],),
        )
        return AIMessage(content="STALE SYNTHESIS MUST NOT PERSIST")

    outcome_c = run_agent(
        conversation_id=cid,
        run_id=run_c["id"],
        worker_id="worker-stale-c",
        history_manager=FullHistoryManager(),
        model=RunnableLambda(loop_then_hijack_synthesis),
        execution_context=ToolContext(user_id=user["id"], project_id=proj["id"], conn=db),
        max_iterations=1,
        run_timeout_seconds=120,
    )
    assert outcome_c.status == "running"
    assert repositories_agent.list_assistant_messages_for_run(db, conversation_id=cid, run_id=run_c["id"]) == []
    run_c_row = repositories_agent.get_run(db, conversation_id=cid, run_id=run_c["id"])
    assert run_c_row["status"] == "running" and run_c_row["worker_id"] == "worker-winner-c"


def test_atomic_model_response_and_step_persistence_rolls_back_together(db, monkeypatch):
    """If persistence fails between inserting AIMessage and inserting `model_completed`,
    the entire pair (plus `run_completed` / `complete_run` for final responses) must roll back atomically.
    """
    user, proj, inc, _ = _seed_project_and_incident(db)
    conv = repositories_agent.create_conversation(db, user_id=user["id"], project_id=proj["id"])
    cid = conv["id"]

    # 1. Tool-call turn failure between AIMessage and `model_completed`
    _, run_tc, _ = repositories_agent.submit_user_message_and_create_run(
        db,
        user_id=user["id"],
        project_id=proj["id"],
        conversation_id=cid,
        content="Trigger split failure on model tool_call turn",
    )

    orig_append_step = repositories_agent.append_run_step

    def fail_on_model_completed(conn, **kwargs):
        if kwargs.get("step_type") == "model_completed":
            raise RuntimeError("Injected DB failure on model_completed")
        return orig_append_step(conn, **kwargs)

    monkeypatch.setattr(repositories_agent, "append_run_step", fail_on_model_completed)
    claim_and_execute_run(
        db,
        run_id=run_tc["id"],
        worker_id="w-atomic-m1",
        model=RunnableLambda(
            lambda _m: AIMessage(
                content="",
                tool_calls=[{"id": "tc_atom_1", "name": "get_incident", "args": {"incident_id": str(inc["id"])}, "type": "tool_call"}],
            )
        ),
    )
    monkeypatch.undo()

    assert repositories_agent.count_completed_model_iterations_for_run(db, run_id=run_tc["id"]) == 0
    rows_tc = [r for r in repositories_agent.list_all_messages_for_history(db, conversation_id=cid) if r.get("run_id") == run_tc["id"]]
    assert [(r["role"], r["message_type"]) for r in rows_tc] == [("user", "text")]

    # Finish run_tc cleanly before testing final-response turn (clear next_retry_at set by retryable failure)
    db.execute("UPDATE public.agent_runs SET next_retry_at = NULL WHERE id = %s", (run_tc["id"],))
    repositories_agent.claim_run(db, worker_id="w-atomic-m1-cleanup", run_id=run_tc["id"])
    repositories_agent.complete_run(db, run_id=run_tc["id"], worker_id="w-atomic-m1-cleanup")

    # 2. Final answer turn failure between AIMessage and `model_completed`
    _, run_final, _ = repositories_agent.submit_user_message_and_create_run(
        db,
        user_id=user["id"],
        project_id=proj["id"],
        conversation_id=cid,
        content="Trigger split failure on model final turn",
    )
    monkeypatch.setattr(repositories_agent, "append_run_step", fail_on_model_completed)
    claim_and_execute_run(
        db,
        run_id=run_final["id"],
        worker_id="w-atomic-m2",
        model=RunnableLambda(lambda _m: AIMessage(content="Final text that must roll back with model_completed")),
    )
    monkeypatch.undo()

    assert repositories_agent.list_assistant_messages_for_run(db, conversation_id=cid, run_id=run_final["id"]) == []
    assert repositories_agent.count_completed_model_iterations_for_run(db, run_id=run_final["id"]) == 0


def test_atomic_tool_result_and_step_persistence_rolls_back_together(db, monkeypatch):
    """If persistence fails between inserting ToolMessage and inserting `tool_completed`,
    both must roll back together so pending tool calls remain accurately recoverable.
    """
    user, proj, inc, _ = _seed_project_and_incident(db)
    conv = repositories_agent.create_conversation(db, user_id=user["id"], project_id=proj["id"])
    cid = conv["id"]

    _, run, _ = repositories_agent.submit_user_message_and_create_run(
        db,
        user_id=user["id"],
        project_id=proj["id"],
        conversation_id=cid,
        content="Trigger split failure on tool persistence",
    )

    orig_append_step = repositories_agent.append_run_step

    def fail_on_tool_completed(conn, **kwargs):
        if kwargs.get("step_type") == "tool_completed":
            raise RuntimeError("Injected DB failure on tool_completed")
        return orig_append_step(conn, **kwargs)

    monkeypatch.setattr(repositories_agent, "append_run_step", fail_on_tool_completed)
    claim_and_execute_run(
        db,
        run_id=run["id"],
        worker_id="w-atomic-t1",
        model=RunnableLambda(
            lambda _m: AIMessage(
                content="",
                tool_calls=[
                    {"id": "tc_atom_tool", "name": "get_incident", "args": {"incident_id": str(inc["id"])}, "type": "tool_call"}
                ],
            )
        ),
    )
    monkeypatch.undo()

    # ToolMessage MUST have been rolled back alongside `tool_completed`, leaving `tc_atom_tool` pending!
    all_rows = repositories_agent.list_all_messages_for_history(db, conversation_id=cid)
    assert not any(r["role"] == "tool" and r.get("run_id") == run["id"] for r in all_rows)
    hm = FullHistoryManager()
    _, pending = hm.get_pending_tool_calls_for_run(db, cid, run["id"])
    assert [c["id"] for c in pending] == ["tc_atom_tool"]


def test_invocation_budget_headroom_and_model_deadline_clamping(db):
    """Verify:
    1. Orchestrator yields BEFORE starting a new model call when remaining invocation work
       budget is smaller than required model headroom.
    2. Orchestrator yields BEFORE starting a tool call when remaining invocation work budget
       is smaller than that tool's configured timeout.
    3. GroqChatModel clamps per-attempt HTTP timeout to remaining `call_deadline` and aborts
       retries instead of sleeping/retrying past `call_deadline`.
    """
    import time
    from pydantic import BaseModel
    from app.tools.registry import ToolRegistry
    from app.tools.schemas import ToolConstraints, ToolDefinition

    user, proj, inc, _ = _seed_project_and_incident(db)
    conv = repositories_agent.create_conversation(db, user_id=user["id"], project_id=proj["id"])
    cid = conv["id"]

    # Subtest 1: Model headroom yield before Iteration 2
    _, run1, _ = repositories_agent.submit_user_message_and_create_run(
        db,
        user_id=user["id"],
        project_id=proj["id"],
        conversation_id=cid,
        content="Test model headroom yield",
    )
    repositories_agent.claim_run(db, worker_id="w-headroom-1", lease_seconds=90, run_id=run1["id"])
    calls = {"n": 0}

    def slow_first_model(_msgs):
        calls["n"] += 1
        if calls["n"] == 1:
            time.sleep(0.32)
            return AIMessage(
                content="",
                tool_calls=[
                    {"id": "tc_hr_1", "name": "get_incident", "args": {"incident_id": str(inc["id"])}, "type": "tool_call"}
                ],
            )
        return AIMessage(content="Completed on Slice 2 after headroom yield.")

    # Total inv=0.85, reserve=0.10 -> work=0.75. Sleep 0.32 -> remaining ~0.43 < model_timeout=0.55 => yields before call #2!
    out1 = run_agent(
        conversation_id=cid,
        run_id=run1["id"],
        worker_id="w-headroom-1",
        history_manager=FullHistoryManager(),
        model=RunnableLambda(slow_first_model),
        execution_context=ToolContext(user_id=user["id"], project_id=proj["id"], conn=db),
        invocation_timeout_seconds=0.85,
        invocation_reserve_seconds=0.10,
        model_timeout_seconds=0.55,
        run_timeout_seconds=60.0,
    )
    assert out1.status == "queued"
    assert calls["n"] == 1

    # Slice 2 finishes Iteration 2 cleanly
    out2 = claim_and_execute_run(
        db,
        run_id=run1["id"],
        worker_id="w-headroom-2",
        model=RunnableLambda(slow_first_model),
        invocation_timeout_seconds=60.0,
    )
    assert out2 is not None and out2.status == "completed"
    assert calls["n"] == 2

    # Subtest 2: Tool headroom yield inside _execute_tool_batch before starting tool lacking budget
    class _EmptyIn(BaseModel):
        pass

    class _EmptyOut(BaseModel):
        ok: bool = True

    custom_reg = ToolRegistry()
    tool_runs = {"count": 0}

    def _custom_handler(_ctx, _inp):
        tool_runs["count"] += 1
        return _EmptyOut()

    custom_reg.register(
        ToolDefinition(
            name="slow_bounded_tool",
            description="Test tool with 0.55s timeout",
            input_model=_EmptyIn,
            output_model=_EmptyOut,
            handler=_custom_handler,
            constraints=ToolConstraints(timeout_seconds=0.55),
        )
    )

    _, run2, _ = repositories_agent.submit_user_message_and_create_run(
        db,
        user_id=user["id"],
        project_id=proj["id"],
        conversation_id=cid,
        content="Test tool headroom yield",
    )
    repositories_agent.claim_run(db, worker_id="w-tool-hr", lease_seconds=90, run_id=run2["id"])

    def model_emitting_slow_tool(_msgs):
        time.sleep(0.28)
        return AIMessage(
            content="",
            tool_calls=[
                {"id": "tc_slow_tool", "name": "slow_bounded_tool", "args": {}, "type": "tool_call"}
            ],
        )

    out_tool_yield = run_agent(
        conversation_id=cid,
        run_id=run2["id"],
        worker_id="w-tool-hr",
        history_manager=FullHistoryManager(),
        model=RunnableLambda(model_emitting_slow_tool),
        execution_context=ToolContext(user_id=user["id"], project_id=proj["id"], conn=db),
        tool_registry=custom_reg,
        invocation_timeout_seconds=0.75,
        invocation_reserve_seconds=0.10,
        model_timeout_seconds=0.10,
        run_timeout_seconds=60.0,
    )
    # Remaining work (~0.65 - 0.28 = 0.37) < slow_bounded_tool timeout (0.55) => yields BEFORE running tool!
    assert out_tool_yield.status == "queued"
    assert tool_runs["count"] == 0
    repositories_agent.claim_run(db, worker_id="w-tool-hr-done", run_id=run2["id"])
    repositories_agent.complete_run(db, run_id=run2["id"], worker_id="w-tool-hr-done")

    # Subtest 3: GroqChatModel clamps request timeout and aborts retry past call_deadline
    seen_timeouts: List[Any] = []
    started = time.monotonic()

    def retry_transport(request: httpx.Request) -> httpx.Response:
        seen_timeouts.append(request.extensions.get("timeout"))
        return httpx.Response(503, json={"error": {"message": "upstream unavailable"}})

    clamped_model = GroqChatModel(
        api_key=SecretStr("gsk_test_key"),
        timeout_seconds=45.0,
        max_http_retries=2,
        call_deadline=time.monotonic() + 0.35,
        http_client=httpx.Client(transport=httpx.MockTransport(retry_transport)),
    )
    with pytest.raises(AgentModelError) as exc_info:
        clamped_model.invoke([HumanMessage(content="test clamp")])
    elapsed = time.monotonic() - started
    assert exc_info.value.code == "model_upstream_error"
    assert len(seen_timeouts) == 1  # Aborted retry instead of sleeping 2s past 0.35s deadline
    assert elapsed < 1.0
    t_ext = seen_timeouts[0]
    if isinstance(t_ext, dict):
        assert max(v for v in t_ext.values() if isinstance(v, (int, float))) <= 0.36


def test_expired_worker_cannot_persist_even_before_reclaim_and_next_worker_recovers(db):
    """Once a run's lease expires in PostgreSQL, the expired worker must be rejected on every
    write operation even before a second worker calls `claim_run`, and the next worker must recover.
    """
    user, proj, inc, _ = _seed_project_and_incident(db)
    conv = repositories_agent.create_conversation(db, user_id=user["id"], project_id=proj["id"])
    cid = conv["id"]

    _, run, _ = repositories_agent.submit_user_message_and_create_run(
        db,
        user_id=user["id"],
        project_id=proj["id"],
        conversation_id=cid,
        content="Test expired lease rejection before reclaim",
    )
    claimed1 = repositories_agent.claim_run(db, worker_id="worker-exp-1", lease_seconds=90, run_id=run["id"])
    assert claimed1 is not None and claimed1["attempt_count"] == 1

    # Simulate worker-exp-1's lease expiring (without another worker claiming it yet)
    db.execute(
        "UPDATE public.agent_runs SET lease_expires_at = clock_timestamp() - interval '2 seconds' WHERE id = %s",
        (run["id"],),
    )

    assert repositories_agent.heartbeat_run(db, run_id=run["id"], worker_id="worker-exp-1", lease_seconds=90) is False
    with pytest.raises(repositories_agent.LeaseLostError):
        repositories_agent.append_conversation_message(
            db,
            conversation_id=cid,
            role="assistant",
            message_type="text",
            content="Expired write must fail",
            run_id=run["id"],
            expected_worker_id="worker-exp-1",
        )
    with pytest.raises(repositories_agent.LeaseLostError):
        repositories_agent.append_run_step(
            db,
            run_id=run["id"],
            step_type="run_completed",
            status="completed",
            expected_worker_id="worker-exp-1",
        )
    assert repositories_agent.complete_run(db, run_id=run["id"], worker_id="worker-exp-1") is None
    assert repositories_agent.yield_run_for_continuation(db, run_id=run["id"], worker_id="worker-exp-1") is None
    assert repositories_agent.fail_or_retry_run(db, run_id=run["id"], worker_id="worker-exp-1", error_message="err") is None

    # Next worker cleanly recovers and finishes
    outcome2 = claim_and_execute_run(
        db,
        run_id=run["id"],
        worker_id="worker-rec-2",
        model=RunnableLambda(lambda _m: AIMessage(content="Recovered after expiry cleanly.")),
    )
    assert outcome2 is not None and outcome2.status == "completed"
    row = repositories_agent.get_run(db, conversation_id=cid, run_id=run["id"])
    assert row["status"] == "completed"
    assert row["attempt_count"] == 2


def test_late_groq_model_response_and_retry_abort_after_lease_loss(db):
    """Verify both:
    1. When GroqChatModel retries after a 503 and heartbeat detects lost lease, the retry
       attempt is aborted immediately (zero extra Groq calls, zero DB writes).
    2. When GroqChatModel returns HTTP 200 late after lease loss, no stale assistant message
       is persisted and the winning worker completes cleanly.
    """
    user, proj, _, _ = _seed_project_and_incident(db)
    conv = repositories_agent.create_conversation(db, user_id=user["id"], project_id=proj["id"])
    cid = conv["id"]

    # Case 1: Abort retry immediately if lease lost during Attempt 1 (503 -> lost lease)
    _, run1, _ = repositories_agent.submit_user_message_and_create_run(
        db,
        user_id=user["id"],
        project_id=proj["id"],
        conversation_id=cid,
        content="Abort retry on lost lease",
    )
    http_attempts = {"count": 0}

    def hijack_on_503(_request: httpx.Request) -> httpx.Response:
        http_attempts["count"] += 1
        db.execute(
            "UPDATE public.agent_runs SET worker_id = 'winner-on-retry' WHERE id = %s",
            (run1["id"],),
        )
        return httpx.Response(503, json={"error": {"message": "transient error"}})

    def _hb1() -> bool:
        return repositories_agent.heartbeat_run(db, run_id=run1["id"], worker_id="stale-retry-w", lease_seconds=90)

    groq1 = create_groq_chat_model(
        get_settings(),
        bind_default_tools=False,
        http_client=httpx.Client(transport=httpx.MockTransport(hijack_on_503)),
        heartbeat_callback=_hb1,
    ).model_copy(update={"api_key": SecretStr("test-key")})

    out1 = claim_and_execute_run(
        db,
        run_id=run1["id"],
        worker_id="stale-retry-w",
        model=groq1,
    )
    assert out1 is not None and out1.status == "running"
    assert http_attempts["count"] == 1  # Retry Attempt 2 never fired after heartbeat returned False!
    repositories_agent.complete_run(db, run_id=run1["id"], worker_id="winner-on-retry")

    # Case 2: Attempt 1 returns 200 late after lease stolen -> fenced off, winner finishes
    _, run2, _ = repositories_agent.submit_user_message_and_create_run(
        db,
        user_id=user["id"],
        project_id=proj["id"],
        conversation_id=cid,
        content="Late 200 after lease loss",
    )

    def hijack_on_200(_request: httpx.Request) -> httpx.Response:
        db.execute(
            "UPDATE public.agent_runs SET worker_id = 'winner-after-200' WHERE id = %s",
            (run2["id"],),
        )
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-late",
                "model": "qwen/qwen3.8-27b",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "LATE RESPONSE MUST NOT PERSIST"},
                    }
                ],
            },
        )

    groq2 = create_groq_chat_model(
        get_settings(),
        bind_default_tools=False,
        http_client=httpx.Client(transport=httpx.MockTransport(hijack_on_200)),
    ).model_copy(update={"api_key": SecretStr("test-key")})

    out2 = claim_and_execute_run(
        db,
        run_id=run2["id"],
        worker_id="stale-200-w",
        model=groq2,
    )
    assert out2 is not None and out2.status == "running"
    assert repositories_agent.list_assistant_messages_for_run(db, conversation_id=cid, run_id=run2["id"]) == []

    # Expire winner-after-200 and let winner finish via claim_and_execute_run
    db.execute(
        "UPDATE public.agent_runs SET lease_expires_at = clock_timestamp() - interval '2 seconds' WHERE id = %s",
        (run2["id"],),
    )
    win2 = claim_and_execute_run(
        db,
        run_id=run2["id"],
        worker_id="winner-final",
        model=RunnableLambda(lambda _m: AIMessage(content="Clean output from winning worker.")),
    )
    assert win2 is not None and win2.status == "completed"
    msgs2 = repositories_agent.list_assistant_messages_for_run(db, conversation_id=cid, run_id=run2["id"])
    assert len(msgs2) == 1 and msgs2[0]["content"] == "Clean output from winning worker."


def test_running_lease_invariant_lifecycle_validation_and_field_clearing(db):
    """Verify the strict running lease invariant across all transitions and validations:
    - Queued run claim sets status='running', non-null worker_id, and future lease_expires_at.
    - Future lease + matching worker_id passes require_run_lease & heartbeat_run; wrong worker fails.
    - Expired lease fails require_run_lease & heartbeat_run (stale worker cannot regain lease).
    - Completion, retry/failure, and yielding ALL clear BOTH worker_id and lease_expires_at cleanly.
    """
    user, proj, _inc, _ = _seed_project_and_incident(db)
    conv = repositories_agent.create_conversation(db, user_id=user["id"], project_id=proj["id"])
    cid = conv["id"]

    # 1. Queued run claim sets non-null worker_id and future lease_expires_at; passing validates
    _, run1, _ = repositories_agent.submit_user_message_and_create_run(
        db,
        user_id=user["id"],
        project_id=proj["id"],
        conversation_id=cid,
        content="Lifecycle 1: yield and complete",
    )
    assert run1["status"] == "queued"
    assert run1["worker_id"] is None
    assert run1["lease_expires_at"] is None

    claimed1 = repositories_agent.claim_run(db, worker_id="worker-inv-1", lease_seconds=90, run_id=run1["id"])
    assert claimed1 is not None
    assert claimed1["status"] == "running"
    assert claimed1["worker_id"] == "worker-inv-1"
    assert claimed1["lease_expires_at"] is not None
    is_future = db.execute(
        "SELECT lease_expires_at > clock_timestamp() AS ok FROM public.agent_runs WHERE id = %s",
        (run1["id"],),
    ).fetchone()["ok"]
    assert is_future is True

    # Matching worker with future lease passes require_run_lease; wrong worker fails
    repositories_agent.require_run_lease(db, run_id=run1["id"], worker_id="worker-inv-1")
    with pytest.raises(repositories_agent.LeaseLostError):
        repositories_agent.require_run_lease(db, run_id=run1["id"], worker_id="wrong-worker")

    # Heartbeat extends unexpired lease
    old_expiry = claimed1["lease_expires_at"]
    assert repositories_agent.heartbeat_run(db, run_id=run1["id"], worker_id="worker-inv-1", lease_seconds=120) is True
    after_hb = repositories_agent.get_run(db, conversation_id=cid, run_id=run1["id"])
    assert after_hb["lease_expires_at"] > old_expiry

    # Yielding clears BOTH worker_id and lease_expires_at
    yielded = repositories_agent.yield_run_for_continuation(db, run_id=run1["id"], worker_id="worker-inv-1")
    assert yielded is not None
    assert yielded["status"] == "queued"
    assert yielded["worker_id"] is None
    assert yielded["lease_expires_at"] is None

    # Re-claim and complete -> clears BOTH worker_id and lease_expires_at
    claimed_again = repositories_agent.claim_run(db, worker_id="worker-inv-2", lease_seconds=90, run_id=run1["id"])
    assert claimed_again is not None
    completed = repositories_agent.complete_run(db, run_id=run1["id"], worker_id="worker-inv-2")
    assert completed is not None
    assert completed["status"] == "completed"
    assert completed["worker_id"] is None
    assert completed["lease_expires_at"] is None

    # 2. Expired lease fails require_run_lease and cannot heartbeat to regain ownership
    _, run2, _ = repositories_agent.submit_user_message_and_create_run(
        db,
        user_id=user["id"],
        project_id=proj["id"],
        conversation_id=cid,
        content="Lifecycle 2: expiry and fail clearing",
    )
    repositories_agent.claim_run(db, worker_id="worker-exp-check", lease_seconds=90, run_id=run2["id"])
    db.execute(
        "UPDATE public.agent_runs SET lease_expires_at = clock_timestamp() - interval '1 second' WHERE id = %s",
        (run2["id"],),
    )
    with pytest.raises(repositories_agent.LeaseLostError):
        repositories_agent.require_run_lease(db, run_id=run2["id"], worker_id="worker-exp-check")
    assert repositories_agent.heartbeat_run(db, run_id=run2["id"], worker_id="worker-exp-check", lease_seconds=90) is False

    # Recover run2, then transition to terminal failure -> clears BOTH worker_id and lease_expires_at
    repositories_agent.claim_run(db, worker_id="worker-fail-check", lease_seconds=90, run_id=run2["id"])
    failed = repositories_agent.fail_or_retry_run(
        db,
        run_id=run2["id"],
        worker_id="worker-fail-check",
        error_message="Permanent failure check",
        retryable=False,
    )
    assert failed is not None
    assert failed["status"] == "failed"
    assert failed["worker_id"] is None
    assert failed["lease_expires_at"] is None



