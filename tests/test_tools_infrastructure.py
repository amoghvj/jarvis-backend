"""Registry, executor, validation and redaction tests (no GitHub involved)."""

import logging
import time
from uuid import uuid4

import pytest
from pydantic import BaseModel

from app.tools.definitions import build_default_registry
from app.tools.errors import ToolError, redact
from app.tools.executor import execute_tool
from app.tools.registry import ToolRegistry
from app.tools.schemas import ToolConstraints, ToolContext, ToolDefinition
from tests.conftest import make_user


class _In(BaseModel):
    value: int = 1


class _Out(BaseModel):
    doubled: int


def _tool(name="demo_tool", handler=None, requires_user=True, constraints=ToolConstraints()):
    return ToolDefinition(
        name=name,
        description="demo",
        input_model=_In,
        output_model=_Out,
        handler=handler or (lambda ctx, inp: _Out(doubled=inp.value * 2)),
        requires_user=requires_user,
        source="test",
        constraints=constraints,
    )


def _registry(*tools):
    r = ToolRegistry()
    for t in tools:
        r.register(t)
    return r


# --- registry ---------------------------------------------------------------

def test_registry_rejects_duplicates_and_invalid_names():
    r = _registry(_tool())
    with pytest.raises(ValueError, match="already registered"):
        r.register(_tool())
    for bad in ("Bad-Name", "x", "has space", "../etc"):
        with pytest.raises(ValueError):
            r.register(_tool(name=bad))


def test_registry_does_not_execute_tools():
    calls = []
    r = _registry(_tool(handler=lambda c, i: calls.append(1) or _Out(doubled=0)))
    r.get("demo_tool")
    r.describe()
    assert calls == []


def test_default_registry_exposes_all_tools_machine_readably():
    described = {t["name"]: t for t in build_default_registry().describe()}
    assert set(described) == {
        "list_installations", "list_repositories", "get_repository_metadata",
        "get_repository_tree", "get_file_contents", "search_repository",
        "get_commit", "get_commit_diff", "get_file_history",
        # code-analysis category
        "find_symbol", "find_references", "get_function_context", "analyze_dependencies",
        "find_callers", "find_related_files", "inspect_configuration", "inspect_tests",
        # telemetry / incident category
        "get_incident", "get_related_events", "search_events", "get_event_details", "get_trace",
        "get_service_context", "get_incident_timeline",
    }
    for t in described.values():
        assert t["description"] and t["requires_user"] is True
        assert t["input_schema"]["type"] == "object" and t["output_schema"]["type"] == "object"
        assert t["input_schema"].get("additionalProperties") is False  # unknown fields rejected
        assert "user_id" not in t["input_schema"].get("properties", {})
        assert t["constraints"]["timeout_seconds"] > 0


def test_unknown_tool_returns_safe_error():
    result = execute_tool("rm_rf", {}, None, registry=_registry(_tool()))
    assert result.status == "error" and result.error.code == "unknown_tool"
    assert result.data is None


# --- validation -------------------------------------------------------------

@pytest.mark.parametrize(
    "tool,payload",
    [
        ("get_file_contents", {"installation_id": 1, "owner": "../x", "repo": "r", "path": "a"}),
        ("get_file_contents", {"installation_id": 1, "owner": "o", "repo": "..", "path": "a"}),
        ("get_file_contents", {"installation_id": 1, "owner": "o", "repo": "r", "path": "../../etc/passwd"}),
        ("get_file_contents", {"installation_id": 1, "owner": "o", "repo": "r", "path": "a/./b"}),
        ("get_file_contents", {"installation_id": 1, "owner": "o", "repo": "r", "path": "a\\b"}),
        ("get_file_contents", {"installation_id": 1, "owner": "o", "repo": "r", "path": "a", "ref": "x..y"}),
        ("get_file_contents", {"installation_id": 0, "owner": "o", "repo": "r", "path": "a"}),
        ("get_file_contents", {"installation_id": "1; DROP", "owner": "o", "repo": "r", "path": "a"}),
        ("get_commit", {"installation_id": 1, "owner": "o", "repo": "r", "sha": "not-hex"}),
        ("list_repositories", {"installation_id": 1, "per_page": 1000}),
        ("search_repository", {"installation_id": 1, "owner": "o", "repo": "r", "query": "x"}),
        ("search_repository", {"installation_id": 1, "owner": "o", "repo": "r", "query": "ab", "language": "cobol"}),
    ],
)
def test_invalid_input_is_rejected_before_any_work(tool, payload):
    # No context at all: validation must fail first, so nothing else is touched.
    result = execute_tool(tool, payload, None)
    assert result.status == "error" and result.error.code == "invalid_input"
    assert result.metadata.github_requests == 0


def test_identity_cannot_be_supplied_through_input():
    payload = {"installation_id": 1, "owner": "o", "repo": "r", "path": "a", "user_id": str(uuid4())}
    result = execute_tool("get_file_contents", payload, None)
    assert result.error.code == "invalid_input" and "user_id" in result.error.message


def test_validation_errors_do_not_echo_input_values():
    secret = "ghs_supersecrettokenvalue1234567890"
    result = execute_tool(
        "get_file_contents",
        {"installation_id": 1, "owner": secret, "repo": "r", "path": "a"},
        None,
    )
    assert result.error.code == "invalid_input" and secret not in result.model_dump_json()


def test_non_object_input_rejected():
    assert execute_tool("get_commit", ["x"], None).error.code == "invalid_input"  # type: ignore[arg-type]


# --- authorization context --------------------------------------------------

def test_missing_or_malformed_context_is_unauthorized():
    reg = _registry(_tool())
    for ctx in (None, "user", object()):
        assert execute_tool("demo_tool", {}, ctx, registry=reg).error.code == "unauthorized"
    assert execute_tool("demo_tool", {}, ToolContext(user_id="not-a-uuid", conn=object()), registry=reg).error.code == "unauthorized"  # type: ignore[arg-type]


def test_context_for_nonexistent_user_is_unauthorized(db):
    result = execute_tool("demo_tool", {}, ToolContext(user_id=uuid4(), conn=db), registry=_registry(_tool()))
    assert result.error.code == "unauthorized"


def test_valid_context_executes_and_reports_metadata(db):
    user = make_user(db)
    result = execute_tool("demo_tool", {"value": 21}, ToolContext(user_id=user["id"], conn=db), registry=_registry(_tool()))
    assert result.status == "success" and result.data == {"doubled": 42} and result.error is None
    assert result.metadata.source == "test" and result.metadata.duration_ms >= 0
    assert result.tool_name == "demo_tool"


# --- executor safety --------------------------------------------------------

def test_timeout_is_enforced(db):
    user = make_user(db)
    slow = _tool(handler=lambda c, i: (time.sleep(1.0), _Out(doubled=1))[1],
                 constraints=ToolConstraints(timeout_seconds=0.2))
    result = execute_tool("demo_tool", {}, ToolContext(user_id=user["id"], conn=db), registry=_registry(slow))
    assert result.error.code == "timeout" and result.error.retryable is True


def test_unexpected_exceptions_are_hidden_and_redacted_in_logs(db, caplog):
    user = make_user(db)
    secret = "ghs_leakytoken1234567890abcdef"

    def boom(ctx, inp):
        raise RuntimeError(f"upstream said Bearer {secret} and postgresql://u:p@host/db")

    with caplog.at_level(logging.ERROR, logger="jarvis.tools"):
        result = execute_tool("demo_tool", {}, ToolContext(user_id=user["id"], conn=db), registry=_registry(_tool(handler=boom)))
    assert result.error.code == "internal_error" and result.error.message == "Tool execution failed"
    assert secret not in result.model_dump_json()
    assert "RuntimeError" in caplog.text and secret not in caplog.text and "u:p@host" not in caplog.text


def test_tool_errors_pass_through_with_safe_codes(db):
    user = make_user(db)

    def denied(ctx, inp):
        raise ToolError("repository_not_accessible", "nope", retryable=False)

    result = execute_tool("demo_tool", {}, ToolContext(user_id=user["id"], conn=db), registry=_registry(_tool(handler=denied)))
    assert result.error.code == "repository_not_accessible"


def test_oversized_output_is_rejected(db):
    user = make_user(db)
    tiny = _tool(constraints=ToolConstraints(max_output_bytes=5))
    result = execute_tool("demo_tool", {}, ToolContext(user_id=user["id"], conn=db), registry=_registry(tiny))
    assert result.error.code == "result_too_large"


def test_wrong_output_type_is_internal_error(db):
    user = make_user(db)
    bad = _tool(handler=lambda c, i: {"doubled": 1})
    result = execute_tool("demo_tool", {}, ToolContext(user_id=user["id"], conn=db), registry=_registry(bad))
    assert result.error.code == "internal_error"


# --- redaction --------------------------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        "token ghs_abcdefghijklmnop1234",
        "gho_abcdefghijklmnop1234 and ghu_abcdefghijklmnop1234",
        "github_pat_11ABCDEFG0abcdefghijklmnop",
        "Authorization: Bearer abcdefgh12345678",
        "key jarvis_live_AbCdEfGh1234567890",
        "jwt eyJhbGciOiJSUzI1NiJ9.eyJpc3MiOiIxIn0.c2lnbmF0dXJl",
        "-----BEGIN PRIVATE KEY-----\nMIIEvQ\n-----END PRIVATE KEY-----",
        "postgresql://user:pass@db.example.com:6543/postgres",
    ],
)
def test_redaction_removes_credentials(text):
    out = redact(text)
    assert "[REDACTED]" in out
    for fragment in ("abcdefghijklmnop", "abcdefgh12345678", "AbCdEfGh", "MIIEvQ", "pass@", "c2lnbmF0dXJl"):
        assert fragment not in out

