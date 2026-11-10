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

