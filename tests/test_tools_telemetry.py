"""Project-context injection and the seven telemetry/incident tools."""

import json
import uuid
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from app.services import incident_service
from app.tools.definitions import build_default_registry
from app.tools.executor import execute_tool
from app.tools.registry import ToolRegistry
from app.tools.schemas import ToolConstraints, ToolContext, ToolDefinition
from app.tools.telemetry.common import like_pattern, redact_json, redact_text
from tests.conftest import make_user
from tests.telemetry_helpers import Account, add_verified_repository, iso, link_installation, startup_event

TELEMETRY_TOOLS = (
    "get_incident", "get_related_events", "search_events", "get_event_details", "get_trace",
    "get_service_context", "get_incident_timeline",
)


def ok(result):
    assert result.status == "success", result.error
    return result.data


def err(result, code):
    assert result.status == "error" and result.error.code == code, (result.status, result.error)
    return result.error


@pytest.fixture
def world(db):
    a, b = Account(db), Account(db)
    a.ingest([
        a.event(event_id="a1", ts=iso(0), message="db down", trace="trace-1", span="s1", request="req-1"),
        a.event(event_id="a2", ts=iso(60), message="db down", trace="trace-1", span="s2", request="req-2"),
        a.event(event_id="a3", ts=iso(120), message="db down", trace="trace-1", span="s2", severity="CRITICAL"),
        a.event(event_id="a4", ts=iso(130), type="log", severity="WARNING", message="slow query 100%_done",
                error_type=None, stack=None),
        a.event(event_id="a5", ts=iso(140), message="timeout talking to cache", error_type="TimeoutError",
                service="worker", file="app/cache.py", func="get", env="staging", request="req-9"),
        startup_event(a.external, event_id="a6", ts=iso(-100), repository="https://github.com/acme/webapp"),
        a.event(event_id="a7", ts=iso(7200), message="db down"),  # 2h later: a second incident, same fingerprint
    ])
    b.ingest([
        b.event(event_id="a1", ts=iso(0), message="db down", trace="trace-1", span="b1"),  # same ids as A's
        b.event(event_id="b2", ts=iso(5), message="only b sees this", service="b-svc", trace="trace-b"),
    ])
    db_down = [i for i in a.incidents() if i["title"] == "ConnectionError: db down"]  # ordered by first_seen_at
    w = SimpleNamespace(a=a, b=b, db=db)
    w.inc, w.inc2 = db_down[0], db_down[1]                   # a1-a3, then the recurrence a7
    w.cache = next(i for i in a.incidents() if i["title"] == "TimeoutError: timeout talking to cache")
    w.b_inc = b.incidents()[0]
    w.run = lambda tool, who=None, **payload: execute_tool(tool, payload, (who or a).ctx())
    return w


# --- project context injection --------------------------------------------------------------------


def test_llm_facing_schema_and_description_omit_project_id():
    described = {t["name"]: t for t in build_default_registry().describe()}
    for name in TELEMETRY_TOOLS:
        tool = described[name]
        assert "project_id" not in tool["input_schema"]["properties"]
        assert "project_id" not in tool["input_schema"].get("required", [])
        assert "project_id" not in json.dumps(tool["input_schema"])
        assert "project_id" not in tool["description"]
        assert tool["input_schema"]["additionalProperties"] is False and tool["requires_user"] is True


def test_internal_schema_still_requires_project_id():
    internal = {t["name"]: t for t in build_default_registry().describe(include_internal=True)}
    for name in TELEMETRY_TOOLS:
        schema = internal[name]["input_schema"]
        assert "project_id" in schema["properties"] and "project_id" in schema["required"]


def test_existing_tools_keep_their_explicit_inputs_and_gain_no_project_id():
    described = {t["name"]: t for t in build_default_registry().describe()}
    internal = {t["name"]: t for t in build_default_registry().describe(include_internal=True)}
    for name, tool in described.items():
        if name in TELEMETRY_TOOLS:
            continue
        assert tool["input_schema"] == internal[name]["input_schema"]  # nothing is hidden
        assert "project_id" not in tool["input_schema"]["properties"]
    props = described["get_file_contents"]["input_schema"]["properties"]
    assert {"installation_id", "owner", "repo", "path", "ref"} <= set(props)
    assert {"installation_id", "owner", "repo"} <= set(described["find_symbol"]["input_schema"]["properties"])


def test_github_and_code_tools_work_with_a_project_in_context_and_still_reject_project_id(world):
    result = execute_tool("list_installations", {}, world.a.ctx())
    assert ok(result)["count"] == 0
    err(execute_tool("list_repositories", {"installation_id": 1, "project_id": str(world.a.project_uuid)}, world.a.ctx()),
        "invalid_input")
    err(execute_tool("find_symbol", {"installation_id": 1, "owner": "o", "repo": "r", "symbol_name": "x",
                                      "project_id": str(world.a.project_uuid)}, world.a.ctx()), "invalid_input")


def test_project_id_cannot_be_supplied_by_the_caller(world):
    for tool, payload in (("search_events", {}), ("get_incident", {"incident_id": str(world.inc["id"])})):
        for supplied in (str(world.a.project_uuid), str(world.b.project_uuid), "anything", None):
            e = err(execute_tool(tool, {**payload, "project_id": supplied}, world.a.ctx()), "invalid_input")
            assert "supplied by the server" in e.message
    assert world.b.project_uuid != world.a.project_uuid


def test_injection_requires_a_trusted_context(world, db):
    err(execute_tool("search_events", {}, None), "unauthorized")
    err(execute_tool("search_events", {}, "not-a-context"), "unauthorized")
    err(execute_tool("search_events", {}, ToolContext(user_id=world.a.user["id"], conn=db)), "project_context_required")
    err(execute_tool("search_events", {}, ToolContext(user_id=world.a.user["id"], conn=db, project_id="a-string")),  # type: ignore[arg-type]
        "project_context_required")
    err(execute_tool("search_events", {}, ToolContext(user_id=uuid.uuid4(), conn=db, project_id=world.a.project_uuid)),
        "unauthorized")


def test_project_must_belong_to_the_authenticated_user(world, db):
    foreign = ToolContext(user_id=world.a.user["id"], conn=db, project_id=world.b.project_uuid)
    e = err(execute_tool("search_events", {}, foreign), "project_not_found")
    missing = err(execute_tool("search_events", {}, ToolContext(user_id=world.a.user["id"], conn=db, project_id=uuid.uuid4())),
                  "project_not_found")
    assert (e.code, e.message) == (missing.code, missing.message)  # not-yours and nonexistent look the same
    assert "only b sees this" not in json.dumps(e.model_dump())


def test_the_executor_injects_the_selected_project_into_the_handler(world):
    seen = {}

    class In(BaseModel):
        model_config = {"extra": "forbid"}
        project_id: uuid.UUID
        value: int = 1

    class Out(BaseModel):
        project: str

    def handler(ctx, inp):
        seen["inp"] = inp
        return Out(project=str(inp.project_id))

    tool = ToolDefinition(name="demo_scoped", description="d", input_model=In, output_model=Out, handler=handler,
                          context_inputs=("project_id",))
    registry = ToolRegistry()
    registry.register(tool)
    assert "project_id" not in ToolRegistry.input_schema(tool)["properties"]
    assert "project_id" in ToolRegistry.input_schema(tool, include_internal=True)["properties"]
    result = execute_tool("demo_scoped", {"value": 3}, world.b.ctx(), registry=registry)
    assert ok(result) == {"project": str(world.b.project_uuid)} and seen["inp"].value == 3
    err(execute_tool("demo_scoped", {"project_id": str(world.a.project_uuid)}, world.b.ctx(), registry=registry), "invalid_input")
    err(execute_tool("demo_scoped", {}, world.b.ctx(project_uuid=world.a.project_uuid), registry=registry), "project_not_found")


def test_registry_rejects_unsupported_context_inputs():
    class In(BaseModel):
        user_id: uuid.UUID

    class Out(BaseModel):
        x: int = 1

    r = ToolRegistry()
    with pytest.raises(ValueError, match="unsupported context input"):
        r.register(ToolDefinition(name="bad_tool", description="d", input_model=In, output_model=Out,
                                  handler=lambda c, i: Out(), context_inputs=("user_id",)))
    with pytest.raises(ValueError, match="unsupported context input"):
        r.register(ToolDefinition(name="bad_tool2", description="d", input_model=In, output_model=Out,
                                  handler=lambda c, i: Out(), context_inputs=("project_id",)))  # not a field


# --- get_incident ---------------------------------------------------------------------------------


def test_get_incident_returns_incident_project_and_empty_repository_list(world):
    data = ok(world.run("get_incident", incident_id=str(world.inc["id"])))
    inc = data["incident"]
    assert (inc["title"], inc["status"], inc["severity"], inc["event_count"]) == ("ConnectionError: db down", "open", "CRITICAL", 3)
    assert inc["service_id"] == "backend" and inc["environment"] == "production" and len(inc["fingerprint"]) == 64
    assert inc["summary"]["error_type"] == "ConnectionError"
    assert data["project_context"] == {"project_id": world.a.external, "project_uuid": str(world.a.project_uuid),
                                       "project_name": "Project " + world.a.external}
    assert data["repository_context"]["repositories"] == []  # explicit empty list, nothing guessed
    assert data["events"] is None and data["complete"] is True


def test_get_incident_lists_only_verified_repositories_with_service_relevance(world, db):
    a = world.a
    iid = link_installation(db, a.user["id"])
    add_verified_repository(db, a, iid, "acme", "backend-repo", "backend", "main")
    add_verified_repository(db, a, iid, "acme", "worker-repo", "worker", "develop")
    db.execute("insert into public.project_repositories (project_id, service_id, repository_url, owner, repo) "
               "values (%s, 'backend', 'https://github.com/acme/unverified-secret-repo', 'acme', 'unverified-secret-repo')",
               (a.project_uuid,))
    data = ok(world.run("get_incident", incident_id=str(world.inc["id"])))
    repos = data["repository_context"]["repositories"]
    assert [(r["repo"], r["matches_service"]) for r in repos] == [("backend-repo", True), ("worker-repo", False)]
    first = repos[0]
    assert first == {**first, "installation_id": iid, "owner": "acme", "default_branch": "main", "service_id": "backend",
                     "verification_status": "verified"}
    assert data["repository_context"]["unverified_repository_count"] == 1
    assert "unverified-secret-repo" not in json.dumps(data)  # unverified repos are never exposed as targets
    cache = ok(world.run("get_incident", incident_id=str(world.cache["id"])))
    assert [(r["repo"], r["matches_service"]) for r in cache["repository_context"]["repositories"]] == [
        ("worker-repo", True), ("backend-repo", False)]


def test_get_incident_repository_context_follows_installation_state(world, db):
    a = world.a
    iid = link_installation(db, a.user["id"])
    add_verified_repository(db, a, iid, "acme", "backend-repo")
    db.execute("update public.github_app_installations set suspended_at = now() where installation_id = %s", (iid,))
    data = ok(world.run("get_incident", incident_id=str(world.inc["id"])))
    assert data["repository_context"]["repositories"] == [] and data["repository_context"]["unverified_repository_count"] == 1


def test_get_incident_is_project_scoped(world):
    err(world.run("get_incident", incident_id=str(world.b_inc["id"])), "incident_not_found")
    err(world.run("get_incident", incident_id=str(uuid.uuid4())), "incident_not_found")
    ok(world.run("get_incident", who=world.b, incident_id=str(world.b_inc["id"])))
    err(world.run("get_incident", who=world.b, incident_id=str(world.inc["id"])), "incident_not_found")


def test_get_incident_with_events_is_paginated_and_deterministic(world):
    page1 = ok(world.run("get_incident", incident_id=str(world.inc["id"]), include_events=True, limit=2))
    assert [e["event_id"] for e in page1["events"]] == ["a3", "a2"]  # newest first
    assert page1["events_pagination"]["has_more"] is True and page1["events_pagination"]["next_offset"] == 2
    assert page1["complete"] is False and page1["limits_hit"] == ["page_limit"]
    page2 = ok(world.run("get_incident", incident_id=str(world.inc["id"]), include_events=True, limit=2, offset=2))
    assert [e["event_id"] for e in page2["events"]] == ["a1"] and page2["complete"] is True


# --- get_related_events ------------------------------------------------------------------------------


def test_related_events_by_incident_and_by_fingerprint(world):
    own = ok(world.run("get_related_events", incident_id=str(world.inc["id"])))
    assert [e["event_id"] for e in own["events"]] == ["a3", "a2", "a1"] and own["related_incident_ids"] == [str(world.inc["id"])]
    both = ok(world.run("get_related_events", incident_id=str(world.inc["id"]), scope="fingerprint"))
    assert [e["event_id"] for e in both["events"]] == ["a7", "a3", "a2", "a1"]  # the recurrence is included
    assert set(both["related_incident_ids"]) == {str(world.inc["id"]), str(world.inc2["id"])}
    assert both["incident"]["id"] == str(world.inc["id"])
    assert all(e["incident_id"] for e in both["events"])


def test_related_events_filters_and_validation(world):
    since = ok(world.run("get_related_events", incident_id=str(world.inc["id"]), since=iso(30)))
    assert [e["event_id"] for e in since["events"]] == ["a3", "a2"]
    until = ok(world.run("get_related_events", incident_id=str(world.inc["id"]), until=iso(90)))
    assert [e["event_id"] for e in until["events"]] == ["a2", "a1"]
    crit = ok(world.run("get_related_events", incident_id=str(world.inc["id"]), min_severity="CRITICAL"))
    assert [e["event_id"] for e in crit["events"]] == ["a3"]
    err(world.run("get_related_events", incident_id=str(world.inc["id"]), since=iso(100), until=iso(10)), "invalid_input")
    err(world.run("get_related_events", incident_id=str(world.inc["id"]), since="2026-11-09T08:30:00"), "invalid_input")  # naive
    err(world.run("get_related_events", incident_id=str(world.inc["id"]), scope="everything"), "invalid_input")


def test_related_events_never_cross_projects(world):
    err(world.run("get_related_events", incident_id=str(world.b_inc["id"]), scope="fingerprint"), "incident_not_found")
    data = ok(world.run("get_related_events", incident_id=str(world.inc["id"]), scope="fingerprint"))
    assert all(e["trace_id"] != "trace-b" for e in data["events"]) and len(data["events"]) == 4


# --- search_events -------------------------------------------------------------------------------------


def _ids(data):
    return [e["event_id"] for e in data["events"]]


def test_search_events_filters(world):
    r = world.run
    assert _ids(ok(r("search_events"))) == ["a7", "a5", "a4", "a3", "a2", "a1", "a6"]  # newest first, everything
    assert _ids(ok(r("search_events", service_id="worker"))) == ["a5"]
    assert _ids(ok(r("search_events", environment="staging"))) == ["a5"]
    assert _ids(ok(r("search_events", severity="WARNING"))) == ["a4"]
    assert _ids(ok(r("search_events", min_severity="ERROR"))) == ["a7", "a5", "a3", "a2", "a1"]
    assert _ids(ok(r("search_events", event_type="startup"))) == ["a6"]
    assert _ids(ok(r("search_events", error_type="TimeoutError"))) == ["a5"]
    assert _ids(ok(r("search_events", source_file="app/cache.py"))) == ["a5"]
    assert _ids(ok(r("search_events", source_function="get"))) == ["a5"]
    assert _ids(ok(r("search_events", trace_id="trace-1"))) == ["a3", "a2", "a1"]
    assert _ids(ok(r("search_events", request_id="req-9"))) == ["a5"]
    assert _ids(ok(r("search_events", message_contains="TIMEOUT"))) == ["a5"]  # case-insensitive
    assert _ids(ok(r("search_events", since=iso(125), until=iso(135)))) == ["a4"]
    combined = ok(r("search_events", service_id="backend", min_severity="ERROR", message_contains="db", until=iso(100)))
    assert _ids(combined) == ["a2", "a1"] and combined["applied_filters"]["service_id"] == "backend"


def test_search_message_text_is_matched_literally(world):
    assert _ids(ok(world.run("search_events", message_contains="100%_done"))) == ["a4"]
    assert _ids(ok(world.run("search_events", message_contains="%"))) == ["a4"]
    assert ok(world.run("search_events", message_contains="1%0"))["events"] == []   # '%' is not a wildcard
    assert ok(world.run("search_events", message_contains="db_down"))["events"] == []  # nor is '_'
    assert like_pattern("a%b_c\\d") == "%a\\%b\\_c\\\\d%"


def test_search_pagination_and_limits(world):
    p1 = ok(world.run("search_events", limit=3))
    assert _ids(p1) == ["a7", "a5", "a4"] and p1["pagination"] == {"limit": 3, "offset": 0, "returned": 3, "has_more": True, "next_offset": 3}
    assert p1["complete"] is False
    p2 = ok(world.run("search_events", limit=3, offset=3))
    assert _ids(p2) == ["a3", "a2", "a1"] and p2["pagination"]["has_more"] is True
    p3 = ok(world.run("search_events", limit=3, offset=6))
    assert _ids(p3) == ["a6"] and p3["complete"] is True and p3["pagination"]["next_offset"] is None
    assert set(_ids(p1)) | set(_ids(p2)) | set(_ids(p3)) == {"a1", "a2", "a3", "a4", "a5", "a6", "a7"}
    for bad in ({"limit": 0}, {"limit": 101}, {"offset": -1}, {"offset": 10_001}):
        err(world.run("search_events", **bad), "invalid_input")


def test_search_validation_and_tenant_isolation(world):
    err(world.run("search_events", severity="ERROR", min_severity="WARNING"), "invalid_input")
    err(world.run("search_events", severity="LOUD"), "invalid_input")
    err(world.run("search_events", message_contains="x" * 201), "invalid_input")
    err(world.run("search_events", message_contains="bad\x00char"), "invalid_input")
    err(world.run("search_events", trace_id="has space"), "invalid_input")
    err(world.run("search_events", unknown_filter="x"), "invalid_input")
    assert ok(world.run("search_events", message_contains="only b sees this"))["events"] == []
    assert _ids(ok(world.run("search_events", who=world.b, message_contains="only b sees this"))) == ["b2"]
    assert _ids(ok(world.run("search_events", who=world.b))) == ["b2", "a1"]  # B's own "a1", never A's
    assert ok(world.run("search_events", who=world.b, service_id="backend"))["events"][0]["trace_id"] == "trace-1"


def test_search_results_are_deterministic_for_identical_timestamps(world):
    a = world.a
    a.ingest([a.event(event_id=f"tie-{i}", ts=iso(500), message=f"tie {i}", file=f"f{i}.py") for i in range(6)])
    first = _ids(ok(world.run("search_events", since=iso(500), until=iso(500))))
    assert len(first) == 6 and first == _ids(ok(world.run("search_events", since=iso(500), until=iso(500))))
    paged = _ids(ok(world.run("search_events", since=iso(500), until=iso(500), limit=4))) + \
        _ids(ok(world.run("search_events", since=iso(500), until=iso(500), limit=4, offset=4)))
    assert paged == first


# --- get_event_details ----------------------------------------------------------------------------------


def test_get_event_details_returns_the_complete_event(world):
    data = ok(world.run("get_event_details", event_id="a1"))
    e = data["event"]
    assert (e["event_id"], e["event_type"], e["severity"], e["error_type"]) == ("a1", "error", "ERROR", "ConnectionError")
    assert e["stack_trace"].startswith("Traceback") and e["stack_trace_truncated"] is False
    assert e["source"] == {"file": "app/db.py", "function": "connect", "line": 10}
    assert (e["trace_id"], e["span_id"], e["request_id"], e["thread_name"]) == ("trace-1", "s1", "req-1", "MainThread")
    assert e["application"]["project_id"] == world.a.external and e["runtime"]["language"] == "python"
    assert e["payload"]["event_id"] == "a1" and e["attributes"] == {"mechanism": "logging"}
    assert e["incident"]["id"] == str(world.inc["id"]) and e["incident"]["title"] == "ConnectionError: db down"
    by_uuid = ok(world.run("get_event_details", event_uuid=e["id"]))
    assert by_uuid["event"]["event_id"] == "a1"


def test_get_event_details_startup_event_has_no_incident(world):
    e = ok(world.run("get_event_details", event_id="a6"))["event"]
    assert e["event_type"] == "startup" and e["classification"] == "normal_behaviour" and e["incident"] is None


def test_get_event_details_is_project_scoped_and_validated(world):
    err(world.run("get_event_details", event_id="b2"), "event_not_found")
    b_event_uuid = world.db.execute("select id from public.telemetry_events where project_id = %s and event_id = 'b2'",
                                    (world.b.project_uuid,)).fetchone()["id"]
    err(world.run("get_event_details", event_uuid=str(b_event_uuid)), "event_not_found")
    assert ok(world.run("get_event_details", who=world.b, event_id="a1"))["event"]["trace_id"] == "trace-1"
    assert ok(world.run("get_event_details", who=world.b, event_id="a1"))["event"]["span_id"] == "b1"
    err(world.run("get_event_details"), "invalid_input")
    err(world.run("get_event_details", event_id="a1", event_uuid=str(uuid.uuid4())), "invalid_input")


def test_event_details_redact_secrets_everywhere(world):
    a = world.a
    token = "ghp_abcdefghijklmnopqrstuvwxyz0123456789"
    a.ingest([a.event(
        event_id="leaky", ts=iso(900), message=f"auth failed with Bearer {token} password=hunter2hunter2",
        stack=f"File x.py\nDATABASE_URL=postgresql://user:pw@db.internal:5432/app\nkey {token}",
        attributes={"api_key": "sk_live_abcdefghijklmnop1234", "nested": {"password": "pw123", "note": f"tok {token}"},
                    "items": [f"Bearer {token}"]},
    )])
    blob = json.dumps(ok(world.run("get_event_details", event_id="leaky")))
    for secret in (token, "hunter2hunter2", "user:pw@", "sk_live_abcdefghijklmnop1234", "pw123"):
        assert secret not in blob, secret
    assert "[REDACTED]" in blob
    for tool, payload in (("search_events", {"message_contains": "auth failed"}), ("get_trace", {"trace_id": "trace-1"})):
        assert token not in json.dumps(ok(world.run(tool, **payload)))
    assert token not in json.dumps(ok(world.run("get_incident", incident_id=str(a.incidents()[-1]["id"]), include_events=True)))
    assert token not in json.dumps(ok(world.run("get_incident_timeline", incident_id=str(a.incidents()[-1]["id"]))))


def test_redaction_helpers():
    assert redact_text("Bearer abcdefghijklmnop12345") == "[REDACTED]"
    assert redact_text(None) is None and redact_text("plain text") == "plain text"
    out = redact_json({"password": "x", "ok": 1, "empty_secret": "", "list": [{"token": "t"}], "s": "postgresql://u:p@h/db"})
    assert out["password"] == "[REDACTED]" and out["ok"] == 1 and out["empty_secret"] == "" and out["list"][0]["token"] == "[REDACTED]"
    assert "u:p@" not in out["s"]
    deep = {"a": {}}
    node = deep["a"]
    for _ in range(20):
        node["a"] = {}
        node = node["a"]
    assert "[TRUNCATED]" in json.dumps(redact_json(deep))


def test_oversized_stack_trace_is_flagged_not_silently_cut(world):
    a = world.a
    a.ingest([a.event(event_id="big-stack", ts=iso(950), stack="x" * 60_000, message="big")])
    data = ok(world.run("get_event_details", event_id="big-stack"))
    assert data["event"]["stack_trace_truncated"] is True and len(data["event"]["stack_trace"]) == 50_000
    assert data["complete"] is False and "stack_trace_truncated" in data["limits_hit"]


# --- get_trace --------------------------------------------------------------------------------------------


def test_get_trace_orders_events_and_groups_spans_without_inventing_hierarchy(world):
    data = ok(world.run("get_trace", trace_id="trace-1"))
    assert [e["event_id"] for e in data["events"]] == ["a1", "a2", "a3"]  # oldest first
    assert {(s["span_id"], s["event_count"]) for s in data["spans"]} == {("s1", 1), ("s2", 2)}
    assert [(s["service_id"], s["event_count"]) for s in data["services"]] == [("backend", 3)]
    assert data["total_events"] == 3 and data["first_event_at"] < data["last_event_at"] and data["complete"] is True
    dumped = json.dumps(data)
    assert "parent" not in dumped.replace("Parent/child span relationships are not recorded", "")
    assert any("not a full distributed-trace reconstruction" in n for n in data["notes"])


def test_trace_ids_are_isolated_between_projects(world):
    mine = ok(world.run("get_trace", trace_id="trace-1"))
    theirs = ok(world.run("get_trace", who=world.b, trace_id="trace-1"))
    assert mine["total_events"] == 3 and theirs["total_events"] == 1
    assert [e["span_id"] for e in theirs["events"]] == ["b1"] and {s["span_id"] for s in mine["spans"]}.isdisjoint({"b1"})
    err(world.run("get_trace", trace_id="trace-b"), "trace_not_found")
    ok(world.run("get_trace", who=world.b, trace_id="trace-b"))
    err(world.run("get_trace", trace_id="no-such-trace"), "trace_not_found")


def test_get_trace_pagination_and_validation(world):
    a = world.a
    a.ingest([a.event(event_id=f"t-{i}", ts=iso(1000 + i), message=f"traced {i}", trace="long-trace", file=f"t{i}.py")
              for i in range(5)])
    p1 = ok(world.run("get_trace", trace_id="long-trace", limit=3))
    assert [e["event_id"] for e in p1["events"]] == ["t-0", "t-1", "t-2"] and p1["complete"] is False
    assert p1["total_events"] == 5 and p1["pagination"]["next_offset"] == 3
    p2 = ok(world.run("get_trace", trace_id="long-trace", limit=3, offset=3))
    assert [e["event_id"] for e in p2["events"]] == ["t-3", "t-4"] and p2["complete"] is True
    for bad in ("", "x" * 129, "has space", "semi;colon", "../x"):
        err(world.run("get_trace", trace_id=bad), "invalid_input")
    err(world.run("get_trace", trace_id="long-trace", limit=201), "invalid_input")


# --- get_service_context ----------------------------------------------------------------------------------


def test_service_context_summarises_what_telemetry_shows(world, db):
    a = world.a
    iid = link_installation(db, a.user["id"])
    add_verified_repository(db, a, iid, "acme", "webapp", "backend")
    data = ok(world.run("get_service_context", service_id="backend", window_hours=168))
    assert data["service_id"] == "backend" and data["first_event_at"] < data["last_event_at"]
    assert {e["environment"] for e in data["environments"]} == {"production"}
    assert data["instances"][0]["instance_id"] == "host-1" and data["instances"][0]["event_count"] > 0
    assert data["runtimes"] == [{**data["runtimes"][0], "language": "python", "version": "3.12.2"}]
    assert data["last_startup_event"]["event_id"] == "a6" and data["last_startup_event"]["event_type"] == "startup"
    assert [r["repo"] for r in data["repository_context"]["repositories"]] == ["webapp"]
    assert {i["title"] for i in data["recent_incidents"]} == {"ConnectionError: db down"} and len(data["recent_incidents"]) == 2
    assert not {"deployment", "deployments", "release", "version", "deployed_at"} & set(data)  # nothing fabricated
    assert any("No deployment" in n for n in data["notes"])


def test_service_context_error_frequency_uses_the_window(world, db):
    a = world.a
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)
    a.ingest([a.event(event_id=f"recent-{i}", ts=(now - timedelta(minutes=10 + i)).isoformat(), message="recent failure",
                      service="fresh-svc", file="app/fresh.py") for i in range(4)] +
             [a.event(event_id="recent-warn", ts=now.isoformat(), severity="WARNING", message="w", service="fresh-svc",
                      error_type=None)] +
             [a.event(event_id="old-err", ts=(now - timedelta(hours=30)).isoformat(), message="old failure",
                      service="fresh-svc", file="app/old.py")])
    day = ok(world.run("get_service_context", service_id="fresh-svc", window_hours=24))["recent_activity"]
    assert day["total_events"] == 5 and day["error_events"] == 4 and day["errors_per_hour"] == round(4 / 24, 3)
    assert day["events_by_severity"] == {"ERROR": 4, "WARNING": 1}
    wide = ok(world.run("get_service_context", service_id="fresh-svc", window_hours=48))["recent_activity"]
    assert wide["error_events"] == 5 and wide["window_hours"] == 48


def test_service_context_filters_and_isolation(world):
    staging = ok(world.run("get_service_context", service_id="worker", environment="staging"))
    assert staging["environment_filter"] == "staging" and staging["environments"][0]["environment"] == "staging"
    assert ok(world.run("get_service_context", service_id="backend"))["recent_incidents"]
    err(world.run("get_service_context", service_id="worker", environment="production"), "service_not_found")
    err(world.run("get_service_context", service_id="b-svc"), "service_not_found")  # B's service is invisible to A
    assert ok(world.run("get_service_context", who=world.b, service_id="b-svc"))["service_id"] == "b-svc"
    err(world.run("get_service_context", service_id="ghost"), "service_not_found")
    for bad in ({"window_hours": 0}, {"window_hours": 169}):
        err(world.run("get_service_context", service_id="backend", **bad), "invalid_input")


# --- get_incident_timeline ----------------------------------------------------------------------------------


def test_timeline_interleaves_recorded_events_and_lifecycle_transitions(world, db):
    a = world.a
    incident_service.change_status(db, a.project_uuid, world.inc["id"], "acknowledged", a.user["id"], "on it")
    incident_service.change_status(db, a.project_uuid, world.inc["id"], "resolved", a.user["id"])
    data = ok(world.run("get_incident_timeline", incident_id=str(world.inc["id"])))
    entries = data["timeline"]
    lifecycle = [e for e in entries if e["entry_type"] == "lifecycle"]
    events = [e for e in entries if e["entry_type"] == "event"]
    assert [e["event"]["event_id"] for e in events] == ["a1", "a2", "a3"]
    assert [e["change_type"] for e in lifecycle] == ["created", "severity_escalated", "status_changed", "status_changed"]
    assert [e["to_status"] for e in lifecycle if e["change_type"] == "status_changed"] == ["acknowledged", "resolved"]
    assert (lifecycle[1]["from_severity"], lifecycle[1]["to_severity"]) == ("ERROR", "CRITICAL")
    assert all(e["source"] == "recorded_event" and e["timestamp_basis"] == "event_time" for e in events)
    assert all(e["source"] == "recorded_lifecycle_transition" and e["timestamp_basis"] == "recorded_time" for e in lifecycle)
    assert lifecycle[2]["note"] == "on it" and lifecycle[2]["from_status"] == "open"
    stamps = [(e["timestamp"]) for e in entries]
    assert stamps == sorted(stamps)  # chronological
    s = data["derived_summary"]
    assert s["derived"] is True and s["recorded_event_count"] == 3 and s["duration_seconds"] == 120.0
    assert (s["peak_severity"], s["current_status"]) == ("CRITICAL", "resolved")
    assert any("derived_summary" in n for n in data["notes"]) and data["complete"] is True


def test_timeline_is_paginated_and_project_scoped(world):
    p1 = ok(world.run("get_incident_timeline", incident_id=str(world.inc["id"]), limit=2))
    assert [e["event"]["event_id"] for e in p1["timeline"] if e["entry_type"] == "event"] == ["a1", "a2"]
    assert p1["pagination"]["has_more"] is True and p1["complete"] is False
    p2 = ok(world.run("get_incident_timeline", incident_id=str(world.inc["id"]), limit=2, offset=2))
    assert [e["event"]["event_id"] for e in p2["timeline"] if e["entry_type"] == "event"] == ["a3"]
    assert p2["complete"] is True
    err(world.run("get_incident_timeline", incident_id=str(world.b_inc["id"])), "incident_not_found")
    err(world.run("get_incident_timeline", incident_id=str(world.inc["id"]), limit=201), "invalid_input")


def test_timeline_for_an_incident_without_extra_history(world):
    data = ok(world.run("get_incident_timeline", incident_id=str(world.cache["id"])))
    assert [e["entry_type"] for e in data["timeline"]].count("event") == 1
    assert [e["change_type"] for e in data["timeline"] if e["entry_type"] == "lifecycle"] == ["created"]


# --- shared behaviour ------------------------------------------------------------------------------------------


def test_every_telemetry_result_reports_completeness_and_uses_the_standard_envelope(world):
    calls = [
        ("get_incident", {"incident_id": str(world.inc["id"])}), ("get_related_events", {"incident_id": str(world.inc["id"])}),
        ("search_events", {}), ("get_event_details", {"event_id": "a1"}), ("get_trace", {"trace_id": "trace-1"}),
        ("get_service_context", {"service_id": "backend"}), ("get_incident_timeline", {"incident_id": str(world.inc["id"])}),
    ]
    for tool, payload in calls:
        result = world.run(tool, **payload)
        assert result.status == "success" and result.tool_name == tool and result.error is None
        assert result.metadata.source == "database" and result.metadata.github_requests == 0
        for field in ("complete", "limits_hit", "notes"):
            assert field in result.data, (tool, field)


def test_no_telemetry_tool_writes_anything(world, db):
    def snapshot():
        return [db.execute(f"select count(*) as n from public.{t}").fetchone()["n"] for t in
                ("telemetry_events", "incidents", "incident_history", "project_repositories", "projects")]

    before = snapshot()
    for tool, payload in (("search_events", {}), ("get_incident", {"incident_id": str(world.inc["id"]), "include_events": True}),
                          ("get_trace", {"trace_id": "trace-1"}), ("get_incident_timeline", {"incident_id": str(world.inc["id"])})):
        ok(world.run(tool, **payload))
    assert snapshot() == before


def test_user_without_projects_context_cannot_use_other_users_data(world, db):
    stranger = make_user(db)
    err(execute_tool("search_events", {}, ToolContext(user_id=stranger["id"], conn=db, project_id=world.a.project_uuid)),
        "project_not_found")


def test_tool_constraints_are_bounded():
    for t in build_default_registry().describe():
        if t["name"] in TELEMETRY_TOOLS:
            assert 0 < t["constraints"]["timeout_seconds"] <= 30 and t["constraints"]["max_output_bytes"] <= 1_000_000
    assert ToolConstraints().max_output_bytes == 1_000_000

