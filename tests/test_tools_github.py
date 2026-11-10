"""GitHub tool tests. GitHub is mocked; the database is the real one, rolled back."""

import json
import secrets

import httpx
import pytest

from app import repositories as db_repo
from app.integrations.github import client as app_tokens
from app.tools.executor import execute_tool
from app.tools.github import api
from app.tools.schemas import ToolContext
from tests.conftest import make_user
from tests.fake_github_api import (
    SHA_FEAT,
    SHA_MAIN,
    SHA_OLD,
    FakeGitHubAPI,
    TokenSource,
    repo_entry,
    sha_of,
)

TOKEN = "ghs_installation_token_AAAA1111BBBB2222"
BASE = {"owner": "acme", "repo": "webapp"}


def link_installation(db, user, installation_id=None, **kw):
    installation_id = installation_id or secrets.randbelow(10**9) + 5_000_000_000
    db_repo.upsert_github_app_installation(
        db,
        user_id=user["id"],
        installation_id=installation_id,
        github_account_id=kw.get("account_id", 4242),
        github_account_login="acme",
        github_account_type="Organization",
        repository_selection="selected",
        permissions={"contents": "read", "metadata": "read"},
        suspended_at=kw.get("suspended_at"),
    )
    return installation_id


@pytest.fixture
def fake(monkeypatch):
    f = FakeGitHubAPI()
    monkeypatch.setattr(api, "_TRANSPORT", f.transport)
    return f


@pytest.fixture
def tokens(monkeypatch):
    source = TokenSource([TOKEN])
    monkeypatch.setattr(app_tokens, "get_installation_access_token", source)
    return source


@pytest.fixture
def env(db, fake, tokens):
    user = make_user(db)
    iid = link_installation(db, user)
    ctx = ToolContext(user_id=user["id"], conn=db)

    class Env:
        pass

    e = Env()
    e.db, e.user, e.iid, e.ctx, e.fake, e.tokens = db, user, iid, ctx, fake, tokens

    def run(tool, **payload):
        payload.setdefault("installation_id", iid)
        return execute_tool(tool, payload, ctx)

    e.run = run
    return e


def ok(result):
    assert result.status == "success", result.error
    return result.data


def err(result, code):
    assert result.status == "error" and result.error.code == code, (result.status, result.error)
    return result.error


# --- list_installations ------------------------------------------------------

def test_list_installations_returns_only_own(env):
    other = make_user(env.db)
    link_installation(env.db, other, account_id=777)
    data = ok(execute_tool("list_installations", {}, env.ctx))
    assert data["count"] == 1
    inst = data["installations"][0]
    assert inst["installation_id"] == env.iid and inst["account_login"] == "acme"
    assert inst["account_type"] == "Organization" and inst["suspended"] is False
    assert inst["repository_selection"] == "selected"
    assert env.fake.requests == []  # database only; GitHub is not contacted


# --- ownership / authorization boundaries -----------------------------------

def test_cross_user_installation_is_rejected_before_token_or_github(env, db):
    attacker = make_user(db)
    ctx = ToolContext(user_id=attacker["id"], conn=db)
    result = execute_tool("list_repositories", {"installation_id": env.iid}, ctx)
    err(result, "installation_not_found")
    assert env.tokens.calls == [] and env.fake.requests == []
    # Nonexistent and foreign installations look identical (no probing).
    missing = execute_tool("list_repositories", {"installation_id": 1}, ctx)
    assert (missing.error.code, missing.error.message) == (result.error.code, result.error.message)


@pytest.mark.parametrize(
    "tool,extra",
    [
        ("list_repositories", {}),
        ("get_repository_metadata", BASE),
        ("get_repository_tree", BASE),
        ("get_file_contents", {**BASE, "path": "src/app.py"}),
        ("search_repository", {**BASE, "query": "ConnectionError"}),
        ("get_commit", {**BASE, "sha": SHA_MAIN}),
        ("get_commit_diff", {**BASE, "sha": SHA_MAIN}),
        ("get_file_history", {**BASE, "path": "src/app.py"}),
    ],
)
def test_every_repo_tool_enforces_installation_ownership(env, db, tool, extra):
    attacker = make_user(db)
    result = execute_tool(tool, {"installation_id": env.iid, **extra}, ToolContext(user_id=attacker["id"], conn=db))
    err(result, "installation_not_found")
    assert env.tokens.calls == [] and env.fake.requests == []


def test_suspended_installation_is_rejected(db, fake, tokens):
    user = make_user(db)
    from datetime import datetime, timezone

    iid = link_installation(db, user, suspended_at=datetime.now(timezone.utc))
    ctx = ToolContext(user_id=user["id"], conn=db)
    assert ok(execute_tool("list_installations", {}, ctx))["installations"][0]["suspended"] is True
    err(execute_tool("list_repositories", {"installation_id": iid}, ctx), "installation_suspended")
    assert tokens.calls == [] and fake.requests == []


def test_token_service_reporting_revoked_installation(env, monkeypatch):
    def revoked(*a, **k):
        raise app_tokens.GitHubInstallationRevokedError("gone")

    monkeypatch.setattr(app_tokens, "get_installation_access_token", revoked)
    err(env.run("list_repositories"), "installation_unavailable")


def test_repository_outside_installation_is_rejected_even_if_public(env):
    result = env.run("get_repository_metadata", owner="octocat", repo="public-repo")
    err(result, "repository_not_accessible")
    # The public repo is readable with any token, so we must never have asked for it.
    assert "/repos/octocat/public-repo" not in env.fake.paths()


def test_repository_not_in_installation_cannot_be_read_via_other_tools(env):
    for tool, extra in [
        ("get_file_contents", {"path": "README.md"}),
        ("get_repository_tree", {}),
        ("get_commit", {"sha": SHA_MAIN}),
    ]:
        err(env.run(tool, owner="octocat", repo="public-repo", **extra), "repository_not_accessible")
    assert not any(p.startswith("/repos/") for p in env.fake.paths())


def test_repository_lookup_scans_all_pages_and_is_case_insensitive(env):
    env.fake.repos = [repo_entry("acme", f"filler{i}", 100 + i) for i in range(119)] + [repo_entry("acme", "webapp", 1)]
    data = ok(env.run("get_repository_metadata", owner="ACME", repo="WebApp"))
    assert data["full_name"] == "acme/webapp"
    pages = [r["params"]["page"] for r in env.fake.requests if r["path"] == "/installation/repositories"]
    assert pages == ["1", "2"]


def test_incomplete_repository_lookup_fails_closed(env, monkeypatch):
    from app.tools.github import access

    monkeypatch.setattr(access, "MAX_REPO_SCAN_PAGES", 1)
    env.fake.repos = [repo_entry("acme", f"filler{i}", 100 + i) for i in range(150)] + [repo_entry("acme", "webapp", 1)]
    err(env.run("get_repository_metadata", **BASE), "repository_lookup_incomplete")


# --- repositories / metadata -------------------------------------------------

def test_list_repositories_pagination(env):
    env.fake.repos = [repo_entry("acme", f"r{i}", i + 1) for i in range(5)]
    first = ok(env.run("list_repositories", per_page=2))
    assert [r["name"] for r in first["repositories"]] == ["r0", "r1"]
    assert first["pagination"] == {"page": 1, "per_page": 2, "has_next_page": True, "next_page": 2}
    assert first["total_count"] == 5
    last = ok(env.run("list_repositories", per_page=2, page=3))
    assert [r["name"] for r in last["repositories"]] == ["r4"]
    assert last["pagination"]["has_next_page"] is False and last["pagination"]["next_page"] is None
    repo = first["repositories"][0]
    assert set(repo) == {"id", "name", "full_name", "owner", "default_branch", "private", "html_url"}
    assert repo["owner"] == "acme" and repo["private"] is True


def test_get_repository_metadata(env):
    data = ok(env.run("get_repository_metadata", **BASE))
    assert data["full_name"] == "acme/webapp" and data["default_branch"] == "main"
    assert data["visibility"] == "private" and data["language"] == "Python"
    assert data["html_url"].startswith("https://github.com/acme/webapp")
    assert data["pushed_at"] == "2026-01-02T00:00:00Z"


# --- references --------------------------------------------------------------

def test_default_branch_is_used_and_reported(env):
    ref = ok(env.run("get_repository_tree", **BASE))["reference"]
    assert ref["kind"] == "default_branch" and ref["name"] == "main"
    assert ref["commit_sha"] == SHA_MAIN and ref["default_branch_used"] is True
    assert "may differ" in ref["note"]


def test_explicit_branch_sha_and_short_sha_references(env):
    branch = ok(env.run("get_repository_tree", **BASE, ref="feature/x"))["reference"]
    assert branch["kind"] == "ref" and branch["commit_sha"] == SHA_FEAT and branch["default_branch_used"] is False
    full = ok(env.run("get_repository_tree", **BASE, ref=SHA_OLD.upper()))["reference"]
    assert full["kind"] == "commit_sha" and full["commit_sha"] == SHA_OLD and full["note"] is None
    short = ok(env.run("get_repository_tree", **BASE, ref=SHA_FEAT[:8]))["reference"]
    assert short["commit_sha"] == SHA_FEAT and short["requested"] == SHA_FEAT[:8]


def test_unknown_reference(env):
    err(env.run("get_repository_tree", **BASE, ref="no-such-branch"), "reference_not_found")
    err(env.run("get_repository_tree", **BASE, ref="deadbeef"), "reference_not_found")


def test_file_is_read_at_resolved_commit_not_moving_branch(env):
    ok(env.run("get_file_contents", **BASE, path="src/app.py"))
    content_calls = [r for r in env.fake.requests if r["path"].startswith("/repos/acme/webapp/contents/")]
    assert content_calls[0]["params"]["ref"] == SHA_MAIN  # pinned to the commit, not "main"


# --- tree --------------------------------------------------------------------

def test_tree_root_and_directory_listing(env):
    root = ok(env.run("get_repository_tree", **BASE))
    kinds = {e["path"]: e["type"] for e in root["entries"]}
    assert kinds["src"] == "directory" and kinds["README.md"] == "file" and root["truncated"] is False
    src = ok(env.run("get_repository_tree", **BASE, path="src"))
    assert [e["path"] for e in src["entries"]] == ["src/app.py", "src/util.py"]
    app = src["entries"][0]
    assert app["size"] == len(env.fake.files["src/app.py"]) and app["sha"] == sha_of("src/app.py")
    err(env.run("get_repository_tree", **BASE, path="missing/dir"), "path_not_found")


def test_tree_recursive(env):
    data = ok(env.run("get_repository_tree", **BASE, path="src", recursive=True))
    assert data["recursive"] is True and {e["path"] for e in data["entries"]} == {"src/app.py", "src/util.py"}


def test_tree_output_limit_and_offset(env):
    page1 = ok(env.run("get_repository_tree", **BASE, recursive=True, limit=3))
    assert page1["returned_entries"] == 3 and page1["truncated"] is True and page1["next_offset"] == 3
    page2 = ok(env.run("get_repository_tree", **BASE, recursive=True, limit=3, offset=3))
    assert page2["entries"][0]["path"] != page1["entries"][0]["path"]
    last = ok(env.run("get_repository_tree", **BASE, recursive=True, limit=1000))
    assert last["next_offset"] is None and last["truncated"] is False


def test_tree_truncated_by_github_is_reported(env):
    env.fake.truncate_tree = True
    data = ok(env.run("get_repository_tree", **BASE, recursive=True))
    assert data["github_truncated"] is True and data["truncated"] is True
    assert any("truncated" in n for n in data["notes"])


# --- file contents -----------------------------------------------------------

def test_get_file_contents(env):
    data = ok(env.run("get_file_contents", **BASE, path="src/app.py"))
    assert data["content"] == "def main():\n    raise ConnectionError('db')\n"
    assert data["encoding"] == "utf-8" and data["complete"] is True
    assert data["size_bytes"] == len(env.fake.files["src/app.py"]) and data["line_count"] == 2
    assert data["line_endings"] == "lf" and data["sha"] == sha_of("src/app.py")
    assert data["reference"]["commit_sha"] == SHA_MAIN and data["path"] == "src/app.py"


def test_crlf_line_endings_preserved(env):
    data = ok(env.run("get_file_contents", **BASE, path="crlf.txt"))
    assert data["content"] == "a\r\nb\r\n" and data["line_endings"] == "crlf"


def test_binary_and_non_utf8_files_are_refused(env):
    err(env.run("get_file_contents", **BASE, path="bin/logo.png"), "unsupported_content")
    err(env.run("get_file_contents", **BASE, path="latin1.txt"), "unsupported_content")


def test_large_file_is_refused_never_truncated(env):
    error = err(env.run("get_file_contents", **BASE, path="big.txt"), "file_too_large")
    assert "not truncated" in error.message


def test_directory_and_missing_paths(env):
    err(env.run("get_file_contents", **BASE, path="src"), "path_is_directory")
    err(env.run("get_file_contents", **BASE, path="nope.py"), "not_found")


# --- search ------------------------------------------------------------------

def test_search_finds_matches_with_line_numbers(env):
    data = ok(env.run("search_repository", **BASE, query="connectionerror"))  # case-insensitive
    found = {(m["path"], m["line_number"]) for m in data["matches"]}
    assert found == {("src/app.py", 2), ("src/util.py", 3)}
    assert data["complete"] is True and data["limits_hit"] == []
    assert data["reference"]["commit_sha"] == SHA_MAIN
    assert data["skipped"]["excluded_directory"] == 1  # node_modules is skipped, and reported
    assert all(m["blob_sha"] for m in data["matches"])
    assert any("ConnectionError" in m["snippet"] for m in data["matches"])


def test_search_filters(env):
    only_src = ok(env.run("search_repository", **BASE, query="ConnectionError", path_prefix="src/"))
    assert {m["path"] for m in only_src["matches"]} == {"src/app.py", "src/util.py"}
    assert ok(env.run("search_repository", **BASE, query="ConnectionError", language="markdown"))["matches"] == []
    sensitive = ok(env.run("search_repository", **BASE, query="connectionerror", case_sensitive=True))
    assert sensitive["matches"] == []


def test_search_reports_incomplete_when_file_limit_hit(env):
    for i in range(10):
        env.fake.files[f"pkg/m{i}.py"] = b"x = 1\n"
    data = ok(env.run("search_repository", **BASE, query="ConnectionError", max_files=3))
    assert "max_files" in data["limits_hit"] and data["complete"] is False and data["files_scanned"] == 3
    assert any("INCOMPLETE" in n for n in data["notes"])


def test_search_result_limit(env):
    env.fake.files["many.py"] = b"hit\n" * 10
    data = ok(env.run("search_repository", **BASE, query="hit", max_results=2))
    assert len(data["matches"]) == 2 and "max_results" in data["limits_hit"] and data["complete"] is False


def test_search_flags_truncated_tree(env):
    env.fake.truncate_tree = True
    data = ok(env.run("search_repository", **BASE, query="ConnectionError"))
    assert "tree_truncated" in data["limits_hit"] and data["complete"] is False


def test_search_skips_oversized_and_binary_files(env):
    env.fake.files["blob.dat"] = b"ConnectionError\x00\x01binary"
    data = ok(env.run("search_repository", **BASE, query="ConnectionError"))
    assert data["skipped"]["too_large"] == 1  # big.txt
    assert data["skipped"]["binary_or_non_utf8"] >= 1
    assert not any(m["path"] in ("blob.dat", "big.txt") for m in data["matches"])


def test_search_never_uses_github_code_search(env):
    ok(env.run("search_repository", **BASE, query="ConnectionError"))
    assert not any(p.startswith("/search") for p in env.fake.paths())


def test_search_requires_minimum_query(env):
    err(env.run("search_repository", **BASE, query="a"), "invalid_input")


# --- commits -----------------------------------------------------------------

def test_get_commit_without_emails(env):
    data = ok(env.run("get_commit", **BASE, sha=SHA_MAIN))
    assert data["sha"] == SHA_MAIN and data["message"].startswith("Fix db retry")
    assert data["author"] == {"name": "Dev Eloper", "login": "developer", "date": "2026-11-01T10:00:00Z"}
    assert data["parents"] == [SHA_OLD] and data["additions"] == 2 and data["deletions"] == 1
    assert "dev@example.com" not in json.dumps(data)
    assert ok(env.run("get_commit", **BASE, sha=SHA_MAIN[:7]))["sha"] == SHA_MAIN


def test_get_commit_unknown(env):
    err(env.run("get_commit", **BASE, sha="d" * 40), "not_found")


def test_commit_diff_bounds_patches(env):
    huge = "+" + "x" * 50_000
    env.fake.commit_files[SHA_MAIN] = [
        {"filename": "a.py", "status": "modified", "additions": 1, "deletions": 0, "changes": 1, "sha": "e" * 40, "patch": huge},
        {"filename": "logo.png", "status": "added", "additions": 0, "deletions": 0, "changes": 0, "sha": "f" * 40},
    ]
    data = ok(env.run("get_commit_diff", **BASE, sha=SHA_MAIN))
    first, second = data["files"]
    assert first["patch_truncated"] is True and len(first["patch"]) == 20_000 and first["patch_available"] is True
    assert second["patch"] is None and second["patch_available"] is False
    assert data["truncated"] is True
    assert any("truncated" in n for n in data["notes"]) and any("did not provide" in n for n in data["notes"])


def test_commit_diff_total_budget_and_pagination(env):
    env.fake.commit_files[SHA_MAIN] = [
        {"filename": f"f{i}.py", "status": "modified", "additions": 1, "deletions": 0, "changes": 1,
         "sha": "e" * 40, "patch": "+" + "y" * 19_000}
        for i in range(10)
    ]
    data = ok(env.run("get_commit_diff", **BASE, sha=SHA_MAIN, per_page=10))
    assert sum(len(f["patch"] or "") for f in data["files"]) <= 120_000
    assert data["truncated"] is True and data["files"][-1]["patch_truncated"] is True
    paged = ok(env.run("get_commit_diff", **BASE, sha=SHA_MAIN, per_page=4))
    assert len(paged["files"]) == 4 and paged["pagination"]["has_next_page"] is True
    assert paged["pagination"]["next_page"] == 2 and paged["truncated"] is True
    no_patch = ok(env.run("get_commit_diff", **BASE, sha=SHA_MAIN, per_page=2, include_patch=False))
    assert all(f["patch"] is None for f in no_patch["files"])


def test_file_history(env):
    data = ok(env.run("get_file_history", **BASE, path="src/app.py"))
    assert [c["sha"] for c in data["commits"]] == [SHA_MAIN, SHA_OLD]
    assert data["reference"]["default_branch_used"] is True and data["reference"]["commit_sha"] == SHA_MAIN
    assert data["commits"][0]["author"]["login"] == "developer" and "email" not in json.dumps(data)
    history_call = [r for r in env.fake.requests if r["params"].get("path") == "src/app.py"][0]
    assert history_call["params"]["sha"] == SHA_MAIN  # walks back from the resolved commit
    paged = ok(env.run("get_file_history", **BASE, path="src/app.py", per_page=1))
    assert len(paged["commits"]) == 1 and paged["pagination"]["has_next_page"] is True
    explicit = ok(env.run("get_file_history", **BASE, path="src/app.py", ref=SHA_OLD))
    assert explicit["reference"]["default_branch_used"] is False and len(explicit["commits"]) == 1


# --- GitHub failure handling -------------------------------------------------

def test_rate_limits_are_normalized(env):
    env.fake.overrides["/repos/acme/webapp/contents"] = (
        403, {"x-ratelimit-remaining": "0", "retry-after": "30"}, {"message": "API rate limit exceeded"})
    e = err(env.run("get_file_contents", **BASE, path="src/app.py"), "github_rate_limited")
    assert e.retryable is True and "30" in e.message
    env.fake.overrides.clear()
    env.fake.overrides["/repos/acme/webapp/contents"] = (429, {}, {"message": "slow down"})
    assert err(env.run("get_file_contents", **BASE, path="src/app.py"), "github_rate_limited").retryable


def test_secondary_rate_limit_without_headers(env):
    env.fake.overrides["/repos/acme/webapp/contents"] = (403, {}, {"message": "You have exceeded a secondary rate limit"})
    err(env.run("get_file_contents", **BASE, path="src/app.py"), "github_rate_limited")


def test_other_github_failures(env):
    env.fake.overrides["/repos/acme/webapp/contents"] = (500, {}, {"message": "oops internal SQL detail"})
    e = err(env.run("get_file_contents", **BASE, path="src/app.py"), "github_unavailable")
    assert e.retryable is True and "SQL" not in e.message
    env.fake.overrides["/repos/acme/webapp/contents"] = (403, {}, {"message": "Resource not accessible"})
    err(env.run("get_file_contents", **BASE, path="src/app.py"), "github_forbidden")
    env.fake.overrides["/repos/acme/webapp/contents"] = (301, {"location": "https://api.github.com/x"}, None)
    err(env.run("get_file_contents", **BASE, path="src/app.py"), "not_found")
    env.fake.overrides["/repos/acme/webapp/contents"] = (418, {}, {"message": "teapot"})
    err(env.run("get_file_contents", **BASE, path="src/app.py"), "github_error")


def test_malformed_responses_fail_closed(env):
    env.fake.raw_overrides["/repos/acme/webapp/contents"] = (200, b"<html>not json</html>")
    err(env.run("get_file_contents", **BASE, path="src/app.py"), "github_bad_response")
    env.fake.raw_overrides.clear()
    env.fake.overrides["/repos/acme/webapp/commits/"] = (200, {}, ["not", "a", "dict"])
    err(env.run("get_commit", **BASE, sha=SHA_MAIN), "github_bad_response")
    env.fake.overrides.clear()
    env.fake.overrides["/installation/repositories"] = (200, {}, {"repositories": "oops"})
    err(env.run("list_repositories"), "github_bad_response")
    env.fake.overrides["/installation/repositories"] = (200, {}, {"repositories": [{"id": "x"}]})
    err(env.run("list_repositories"), "github_bad_response")


def test_network_failure_is_retryable(env):
    env.fake.raise_on = httpx.ConnectError("boom", request=httpx.Request("GET", "https://api.github.com/"))
    e = err(env.run("list_repositories"), "github_unavailable")
    assert e.retryable is True and "boom" not in e.message


def test_response_size_limit(env, monkeypatch):
    monkeypatch.setattr(api, "MAX_RESPONSE_BYTES", 200)
    err(env.run("list_repositories"), "github_response_too_large")


def test_hostile_github_shas_are_not_used_in_urls(env):
    env.fake.overrides["/repos/acme/webapp/commits"] = (
        200, {}, [{"sha": "../../../etc/passwd"}])
    err(env.run("get_repository_tree", **BASE), "github_bad_response")
    assert not any("passwd" in p for p in env.fake.paths())


def test_expired_deadline_stops_requests(env):
    ctx = ToolContext(user_id=env.user["id"], conn=env.db)
    from app.tools.github.repositories import ListRepositoriesInput, list_repositories
    import time

    ctx.deadline = time.monotonic() - 1
    from app.tools.errors import ToolError

    with pytest.raises(ToolError) as exc:
        list_repositories(ctx, ListRepositoriesInput(installation_id=env.iid))
    assert exc.value.code == "timeout" and env.fake.requests == []


# --- token handling ----------------------------------------------------------

def test_stale_token_is_refreshed_exactly_once(db, fake, monkeypatch):
    user = make_user(db)
    iid = link_installation(db, user)
    source = TokenSource(["ghs_stale_token_000000000000", "ghs_fresh_token_111111111111"])
    monkeypatch.setattr(app_tokens, "get_installation_access_token", source)
    monkeypatch.setattr(api, "_TRANSPORT", fake.transport)
    fake.reject_tokens = {"ghs_stale_token_000000000000"}
    result = execute_tool("list_repositories", {"installation_id": iid}, ToolContext(user_id=user["id"], conn=db))
    assert result.status == "success"
    assert source.calls == [False, True]  # normal fetch, then one forced refresh
    assert [r["auth"] for r in fake.requests] == [
        "Bearer ghs_stale_token_000000000000", "Bearer ghs_fresh_token_111111111111"]


def test_persistent_auth_failure_stops_after_one_refresh(db, fake, monkeypatch):
    user = make_user(db)
    iid = link_installation(db, user)
    source = TokenSource(["ghs_bad_one_0000000000000", "ghs_bad_two_0000000000000"])
    monkeypatch.setattr(app_tokens, "get_installation_access_token", source)
    monkeypatch.setattr(api, "_TRANSPORT", fake.transport)
    fake.reject_tokens = {"ghs_bad_one_0000000000000", "ghs_bad_two_0000000000000"}
    result = execute_tool("list_repositories", {"installation_id": iid}, ToolContext(user_id=user["id"], conn=db))
    err(result, "github_auth_error")
    assert len(source.calls) == 2 and len(fake.requests) == 2


@pytest.mark.parametrize(
    "exc",
    [app_tokens.GitHubAppApiError("x"), app_tokens.GitHubRateLimitError("y")],
)
def test_token_service_failures_are_normalized(env, monkeypatch, exc):
    def fail(*a, **k):
        raise exc

    monkeypatch.setattr(app_tokens, "get_installation_access_token", fail)
    result = env.run("list_repositories")
    assert result.status == "error" and result.error.code in ("github_auth_error", "github_rate_limited")
    assert env.fake.requests == []


def test_tokens_are_never_exposed_and_requests_are_read_only(env, caplog):
    results = [
        env.run("list_repositories"),
        env.run("get_repository_metadata", **BASE),
        env.run("get_repository_tree", **BASE, recursive=True),
        env.run("get_file_contents", **BASE, path="src/app.py"),
        env.run("search_repository", **BASE, query="ConnectionError"),
        env.run("get_commit", **BASE, sha=SHA_MAIN),
        env.run("get_commit_diff", **BASE, sha=SHA_MAIN),
        env.run("get_file_history", **BASE, path="src/app.py"),
        env.run("get_file_contents", **BASE, path="missing.py"),  # an error result too
    ]
    blob = " ".join(r.model_dump_json() for r in results)
    assert TOKEN not in blob and "Bearer" not in blob and "ghs_" not in blob
    assert TOKEN not in caplog.text
    assert env.fake.requests, "token should have been used against GitHub"
    assert {r["auth"] for r in env.fake.requests} == {f"Bearer {TOKEN}"}
    assert {r["method"] for r in env.fake.requests} == {"GET"}
    assert {r["host"] for r in env.fake.requests} == {"api.github.com"}


def test_github_urls_are_built_from_validated_segments(env):
    ok(env.run("get_file_contents", **BASE, path="src/app.py"))
    assert "/repos/acme/webapp/contents/src/app.py" in env.fake.paths()

    # Special characters in a path must stay inside one encoded path segment: they must not
    # be able to start a query string, a fragment or a new path component on the wire.
    env.run("get_file_contents", **BASE, path="src/a b#?.py")
    targets = [r["raw_target"] for r in env.fake.requests if "/contents/" in r["raw_target"]]
    assert targets[-1].split("?", 1)[0] == "/repos/acme/webapp/contents/src/a%20b%23%3F.py"
    assert targets[-1].split("?", 1)[1] == f"ref={SHA_MAIN}"  # the only query parameter is ours
    assert all("#" not in r["raw_target"] for r in env.fake.requests)

