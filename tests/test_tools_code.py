"""Code-analysis tool tests. GitHub is mocked; the database is the real one, rolled back."""

import json
import re
import secrets

import pytest

from app import repositories as db_repo
from app.integrations.github import client as app_tokens
from app.tools.code import source as code_source
from app.tools.executor import execute_tool
from app.tools.github import api
from app.tools.schemas import ToolContext
from tests.conftest import make_user
from tests.fake_github_api import SHA_FEAT, SHA_MAIN, FakeGitHubAPI, TokenSource, sha_of

TOKEN = "ghs_installation_token_AAAA1111BBBB2222"
BASE = {"owner": "acme", "repo": "webapp"}

REPO_FILES = {
    "app/__init__.py": b"",
    "app/config.py": b'''import os

DATABASE_URL = os.environ["DATABASE_URL"]
DEBUG = os.getenv("APP_DEBUG", "0")
TIMEOUT = os.environ.get("APP_TIMEOUT")
''',
    "app/db.py": b'''import os
import psycopg
from .config import DATABASE_URL

MAX_RETRIES = 3


class Database:
    """Wraps a connection."""

    def __init__(self, url=DATABASE_URL):
        self.url = url

    def connect(self, retries: int = MAX_RETRIES):
        for _ in range(retries):
            try:
                return psycopg.connect(self.url)
            except Exception:
                continue
        raise ConnectionError("db")

    @staticmethod
    def make():
        def inner():
            return 1
        return inner()


def get_connection():
    return Database().connect()
''',
    "app/service.py": b'''from app.db import get_connection, Database
from app import db as database


def load_user(user_id):
    conn = get_connection()
    return conn


def reload():
    d = Database()
    d.connect()
    return database.get_connection()
''',
    "app/other.py": b'''def run(obj):
    return obj.get_connection()
''',
    "app/notes.py": b'''# calls get_connection dynamically via getattr
HANDLER = "get_connection"
''',
    "app/jobs.py": b'''def process(item):
    return item
''',
    "app/tasks.py": b'''def process(x):
    return x


class Worker:
    def process(self, item):
        return item

    def run(self):
        return self.process(1)
''',
    "app/big.py": b"def big():\n" + b"".join(b"    value_%d = %d  # padding padding\n" % (i, i) for i in range(30)),
    "app/broken.py": b"def oops(:\n    pass\n",
    "tests/test_db.py": b'''import unittest

import pytest
from app import db
from app.db import Database, get_connection


def test_connect_calls_get_connection():
    assert get_connection() is not None


def test_module_attribute():
    assert db.get_connection()


def test_imports_only():
    assert Database is not None


class TestDatabase:
    def test_connect(self):
        conn = Database().connect()
        assert conn


class LegacyTests(unittest.TestCase):
    def test_legacy(self):
        self.assertEqual(get_connection(), 1)
        with pytest.raises(ValueError):
            pass
''',
    "tests/test_service.py": b'''def test_nothing():
    assert True
''',
    "tests/test_unrelated.py": b'''def test_x():
    assert 1
''',
    "web/src/utils.ts": b'''export interface User { id: number; name: string }
export type UserId = number;
export enum Role { Admin, User }
export function formatDate(d: string): string { return d; }
export const DEFAULT_ROLE: Role = Role.User;
''',
    "web/src/api.js": b'''import axios from 'axios';
import { formatDate } from './utils';
const fs = require('fs');
const lodash = require('lodash');

export const API_URL = process.env.API_URL;
export function fetchUser(id) {
  return axios.get(`${API_URL}/users/${id}`).then(r => formatDate(r.data));
}
export class ApiClient {
  constructor(base) { this.base = base; }
  async get(path) { return fetchUser(path); }
}
const helper = async (x) => { return fetchUser(x); };
''',
    "web/src/App.tsx": b'''import React from 'react';
import { fetchUser } from './api';
import type { User } from './utils';
export function App({ id }: { id: number }) {
  const user = fetchUser(id);
  return <div><Profile user={user} /></div>;
}
function Profile(props: { user: User }) { return <span>{props.user.name}</span>; }
export * from './utils';
''',
    "web/src/api.test.js": b'''import { fetchUser } from './api';
import * as utils from './utils';
describe('api', () => {
  it('fetches', () => {
    expect(fetchUser(1)).toBeDefined();
  });
  test('formats', () => { expect(utils.formatDate('x')).toBe('x'); });
});
''',
    "web/src/broken.ts": b"export function ( {\n",
    "web/src/huge.js": None,  # too large to analyze (served with a size but no inline content)
    "web/lib/native.go": b"package main\n",
    "README.md": b"# demo\n",
    ".env.example": b'''# Example environment
DATABASE_URL=postgresql://user:pass@db.example.com:5432/app
API_URL=http://localhost:3000
APP_DEBUG=1
SESSION_SECRET=
STRIPE_KEY=sk_live_abcdefghijklmnop1234
LOG_LEVEL=info
OPAQUE=aGVsbG8gd29ybGQgdGhpcyBpcyBhIGxvbmcgdG9rZW4xMjM0NTY3ODkw
# OPTIONAL_FLAG=true
''',
    ".env": b"REAL_SECRET=hunter2-super-secret\n",
    ".env.local": b"LOCAL_SECRET=hunter3-super-secret\n",
    ".env.production": b"PROD_SECRET=hunter4-super-secret\n",
    "keys/server.pem": b"-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBg\n-----END PRIVATE KEY-----\n",
    "package.json": b'''{
  "name": "webapp",
  "version": "1.2.3",
  "scripts": {
    "build": "vite build",
    "start": "API_TOKEN=abcd1234abcd1234 node server.js"
  },
  "dependencies": {
    "axios": "^1.6.0",
    "private-lib": "git+https://user:ghp_abcdefghijklmnop1234@github.com/o/r.git"
  },
  "devDependencies": {
    "vitest": "^1.0.0"
  }
}
''',
    "tsconfig.json": b'''{
  // comments and trailing commas are legal in tsconfig
  "compilerOptions": {
    "target": "ES2020", /* inline */
    "strict": true,
    "paths": { "@/*": ["src/*"] },
  },
  "include": ["src"],
}
''',
    "requirements.txt": b'''fastapi>=0.110
--index-url https://user:secret@pypi.example.com/simple
psycopg[binary]>=3.2  # db
-r requirements-dev.txt
''',
    "pyproject.toml": b'''[project]
name = "webapp"
requires-python = ">=3.10"
dependencies = ["httpx>=0.27", "pydantic>=2.6"]

[tool.pytest.ini_options]
addopts = "-q"
''',
    "vite.config.ts": b'''import { defineConfig } from 'vite';
export default defineConfig({ define: { v: process.env.VITE_API } });
const x = import.meta.env.VITE_PUBLIC;
''',
}


def link_installation(db, user):
    iid = secrets.randbelow(10**9) + 5_000_000_000
    db_repo.upsert_github_app_installation(
        db, user_id=user["id"], installation_id=iid, github_account_id=4242, github_account_login="acme",
        github_account_type="Organization", repository_selection="selected",
        permissions={"contents": "read", "metadata": "read"}, suspended_at=None,
    )
    return iid


@pytest.fixture
def env(db, monkeypatch):
    fake = FakeGitHubAPI()
    fake.files = dict(REPO_FILES)
    monkeypatch.setattr(api, "_TRANSPORT", fake.transport)
    tokens = TokenSource([TOKEN])
    monkeypatch.setattr(app_tokens, "get_installation_access_token", tokens)
    user = make_user(db)
    iid = link_installation(db, user)
    ctx = ToolContext(user_id=user["id"], conn=db)

    class Env:
        pass

    e = Env()
    e.db, e.user, e.iid, e.ctx, e.fake, e.tokens = db, user, iid, ctx, fake, tokens

    def run(tool, **payload):
        payload.setdefault("installation_id", iid)
        return execute_tool(tool, {**BASE, **payload}, ctx)

    e.run = run
    return e


def ok(result):
    assert result.status == "success", result.error
    return result.data


def err(result, code):
    assert result.status == "error" and result.error.code == code, (result.status, result.error)
    return result.error


ALL_TOOLS = [
    ("find_symbol", {"symbol_name": "get_connection"}),
    ("find_references", {"symbol_name": "get_connection"}),
    ("get_function_context", {"symbol_name": "get_connection", "path": "app/db.py"}),
    ("analyze_dependencies", {}),
    ("find_callers", {"symbol_name": "get_connection"}),
    ("find_related_files", {"path": "app/db.py"}),
    ("inspect_configuration", {}),
    ("inspect_tests", {"path": "app/db.py"}),
]


# --- registration / contracts ------------------------------------------------


def test_all_eight_tools_registered_with_contracts():
    from app.tools.definitions import build_default_registry

    described = {t["name"]: t for t in build_default_registry().describe()}
    names = {"find_symbol", "find_references", "get_function_context", "analyze_dependencies",
             "find_callers", "find_related_files", "inspect_configuration", "inspect_tests"}
    assert names <= set(described)
    for name in names:
        t = described[name]
        assert t["description"] and t["requires_user"] is True and t["source"] == "github"
        assert t["input_schema"].get("additionalProperties") is False
        assert "user_id" not in t["input_schema"]["properties"]
        assert 0 < t["constraints"]["timeout_seconds"] <= 30
        props = t["output_schema"]["properties"]
        for field in ("resolved_commit_sha", "complete", "files_scanned", "files_skipped",
                      "limits_hit", "parse_errors", "unsupported_files"):
            assert field in props, (name, field)


# --- authorization boundaries ------------------------------------------------


@pytest.mark.parametrize("tool,extra", ALL_TOOLS)
def test_cross_user_installation_denied(env, db, tool, extra):
    attacker = make_user(db)
    result = execute_tool(tool, {"installation_id": env.iid, **BASE, **extra}, ToolContext(user_id=attacker["id"], conn=db))
    err(result, "installation_not_found")
    assert env.tokens.calls == [] and env.fake.requests == []


@pytest.mark.parametrize("tool,extra", ALL_TOOLS)
def test_repository_membership_enforced(env, tool, extra):
    result = execute_tool(
        tool, {"installation_id": env.iid, "owner": "octocat", "repo": "public-repo", **extra}, env.ctx
    )
    err(result, "repository_not_accessible")
    assert not any("/git/" in p or "/contents" in p for p in env.fake.paths())  # nothing was read


@pytest.mark.parametrize("tool,extra", ALL_TOOLS)
def test_identity_and_urls_cannot_be_supplied(env, tool, extra):
    for bad in ({"user_id": "x"}, {"url": "https://evil.example/x"}):
        result = env.run(tool, **extra, **bad)
        err(result, "invalid_input")
    assert env.fake.requests == []


# --- commit consistency / read-only ------------------------------------------


def test_analysis_is_pinned_to_one_commit(env):
    data = ok(env.run("find_symbol", symbol_name="get_connection"))
    assert data["resolved_commit_sha"] == SHA_MAIN == data["reference"]["commit_sha"]
    assert data["reference"]["default_branch_used"] is True
    commit_lookups = [r for r in env.fake.requests if r["path"] == "/repos/acme/webapp/commits"]
    assert len(commit_lookups) == 1  # the reference is resolved once, never re-resolved mid-analysis
    trees = [r["path"] for r in env.fake.requests if "/git/trees/" in r["path"]]
    assert trees == [f"/repos/acme/webapp/git/trees/{SHA_MAIN}"]
    blob_paths = [p for p in env.fake.paths() if "/git/blobs/" in p]
    assert blob_paths and len(blob_paths) == len(set(blob_paths))  # each blob fetched once (cached)
    assert not any("/contents/" in p for p in env.fake.paths())  # blobs come from the pinned tree only
    assert {r["method"] for r in env.fake.requests} == {"GET"}
    assert {r["host"] for r in env.fake.requests} == {"api.github.com"}


def test_explicit_ref_is_resolved_and_reported(env):
    data = ok(env.run("find_symbol", symbol_name="get_connection", ref="feature/x"))
    assert data["resolved_commit_sha"] == SHA_FEAT and data["reference"]["requested"] == "feature/x"
    err(env.run("find_symbol", symbol_name="x", ref="no-such-branch"), "reference_not_found")


# --- find_symbol -------------------------------------------------------------


def test_find_symbol_functions_methods_classes_and_constants(env):
    data = ok(env.run("find_symbol", symbol_name="connect"))
    (match,) = data["matches"]
    assert (match["path"], match["symbol_type"], match["qualified_name"]) == ("app/db.py", "method", "Database.connect")
    assert match["parent"] == "Database" and match["parent_type"] == "class"
    assert match["signature"] == "def connect(self, retries: int=MAX_RETRIES)"
    assert match["start_line"] < match["end_line"] and data["ambiguous"] is False
    const = ok(env.run("find_symbol", symbol_name="MAX_RETRIES"))["matches"][0]
    assert const["symbol_type"] == "constant" and const["parent_type"] == "module"
    cls = ok(env.run("find_symbol", symbol_name="ApiClient"))["matches"][0]
    assert cls["symbol_type"] == "class" and cls["language"] == "javascript"
    ts = {(m["symbol_type"], m["name"]) for n in ("User", "UserId", "Role") for m in ok(env.run("find_symbol", symbol_name=n))["matches"]}
    assert ts == {("interface", "User"), ("type", "UserId"), ("enum", "Role")}
    arrow = ok(env.run("find_symbol", symbol_name="helper"))["matches"][0]
    assert arrow["symbol_type"] == "function" and arrow["signature"] == "async helper(x)"


def test_find_symbol_nested_definitions(env):
    inner = ok(env.run("find_symbol", symbol_name="inner"))["matches"][0]
    assert inner["nested"] is True and inner["parent"] == "Database.make" and inner["parent_type"] == "function"
    by_qualified = ok(env.run("find_symbol", symbol_name="Database.make"))["matches"][0]
    assert by_qualified["decorators"] == ["staticmethod"]


def test_duplicate_symbol_names_are_all_returned_and_flagged(env):
    data = ok(env.run("find_symbol", symbol_name="process"))
    assert data["match_count"] == 3 and data["ambiguous"] is True
    assert {(m["path"], m["symbol_type"]) for m in data["matches"]} == {
        ("app/jobs.py", "function"), ("app/tasks.py", "function"), ("app/tasks.py", "method")}
    assert "none is preferred" in data["ambiguity_note"]
    assert ok(env.run("find_symbol", symbol_name="process", path="app/jobs.py"))["ambiguous"] is False
    assert ok(env.run("find_symbol", symbol_name="process", symbol_type="method"))["match_count"] == 1
    assert ok(env.run("find_symbol", symbol_name="Worker.process"))["match_count"] == 1


def test_find_symbol_typescript_and_jsx_files(env):
    data = ok(env.run("find_symbol", symbol_name="Profile"))
    assert data["matches"][0]["path"] == "web/src/App.tsx" and data["matches"][0]["language"] == "tsx"


def test_find_symbol_not_found_complete_vs_incomplete(env):
    complete = ok(env.run("find_symbol", symbol_name="doesNotExist", max_files=150))
    assert complete["matches"] == [] and complete["match_count"] == 0
    # Parse errors and skipped files exist in this repo, so absence must not be presented as proof.
    assert complete["complete"] is False and any("INCOMPLETE" in n for n in complete["notes"])
    scoped = ok(env.run("find_symbol", symbol_name="doesNotExist", path_prefix="app/jobs.py"))
    assert scoped["complete"] is True and scoped["matches"] == []


# --- limits, completeness and parse errors -----------------------------------


def test_file_count_limit_marks_result_incomplete(env):
    data = ok(env.run("find_symbol", symbol_name="get_connection", max_files=2))
    assert "max_files" in data["limits_hit"] and data["complete"] is False and data["files_scanned"] == 2
    assert any("INCOMPLETE" in n for n in data["notes"])


def test_parse_errors_unsupported_and_large_files_are_reported(env):
    data = ok(env.run("find_symbol", symbol_name="get_connection", max_files=150))
    errors = {e["path"]: e for e in data["parse_errors"]}
    assert "app/broken.py" in errors and "SyntaxError" in errors["app/broken.py"]["message"]
    assert errors["app/broken.py"]["partial_results"] is False
    assert errors["web/src/broken.ts"]["partial_results"] is True and errors["web/src/broken.ts"]["line"] == 1
    assert data["unsupported_files"] == ["web/lib/native.go"] and data["unsupported_file_count"] == 1
    assert data["files_skipped"]["too_large"] >= 1  # web/src/huge.js
    assert data["complete"] is False
    assert any("without a parser" in n for n in data["notes"])


def test_total_byte_limit(env, monkeypatch):
    monkeypatch.setattr(code_source, "MAX_TOTAL_BYTES", 400)
    data = ok(env.run("find_symbol", symbol_name="get_connection"))
    assert "max_total_bytes" in data["limits_hit"] and data["complete"] is False


def test_time_budget_limit(env, monkeypatch):
    monkeypatch.setattr(code_source, "TIME_RESERVE_SECONDS", 10**9)
    data = ok(env.run("find_symbol", symbol_name="get_connection"))
    assert "time_budget" in data["limits_hit"] and data["files_scanned"] == 0 and data["complete"] is False


def test_truncated_github_tree_is_reported(env):
    env.fake.truncate_tree = True
    data = ok(env.run("find_symbol", symbol_name="get_connection"))
    assert "tree_truncated" in data["limits_hit"] and data["complete"] is False


def test_rate_limit_midway_yields_partial_incomplete_result(env):
    env.fake.overrides["/repos/acme/webapp/git/blobs/"] = (429, {}, {"message": "slow down"})
    data = ok(env.run("find_symbol", symbol_name="get_connection"))
    assert "github_rate_limited" in data["limits_hit"] and data["complete"] is False and data["files_scanned"] == 0


# --- input validation and file errors ----------------------------------------


@pytest.mark.parametrize("path", ["../etc/passwd", "a/./b.py", "a\\b.py", ""])
def test_invalid_paths_rejected_before_any_github_call(env, path):
    err(env.run("find_symbol", symbol_name="x", path=path), "invalid_input")
    assert env.fake.requests == []


def test_invalid_symbol_names_rejected(env):
    for bad in ("a b", "x;y", "../x", "a..b", "", "x" * 300):
        err(env.run("find_symbol", symbol_name=bad), "invalid_input")
    assert env.fake.requests == []


def test_missing_and_unsupported_files(env):
    err(env.run("find_symbol", symbol_name="x", path="app/nope.py"), "path_not_found")
    e = err(env.run("find_symbol", symbol_name="x", path="web/lib/native.go"), "unsupported_language")
    assert "No parser" in e.message
    err(env.run("find_symbol", symbol_name="x", path="README.md"), "unsupported_language")
    err(env.run("get_function_context", symbol_name="x", path="web/src/huge.js"), "file_unavailable")


def test_single_file_parse_error_is_structured(env):
    data = ok(env.run("find_symbol", symbol_name="oops", path="app/broken.py"))
    assert data["matches"] == [] and data["complete"] is False
    assert data["parse_errors"][0]["path"] == "app/broken.py"


# --- get_function_context ----------------------------------------------------


def test_function_context_full_source_and_relations(env):
    data = ok(env.run("get_function_context", symbol_name="Database.connect", path="app/db.py"))
    fn = data["function"]
    assert data["resolved"] is True and data["resolved_commit_sha"] == SHA_MAIN
    assert fn["qualified_name"] == "Database.connect" and fn["symbol_type"] == "method"
    assert fn["source"].startswith("    def connect") and fn["source"].rstrip().endswith('raise ConnectionError("db")')
    assert fn["source"].count("\n") + 1 == fn["end_line"] - fn["start_line"] + 1
    assert fn["containing_class"]["name"] == "Database"
    assert [i["text"] for i in fn["imports_used"]] == ["import psycopg"]
    assert {(r["name"], r["relation"]) for r in fn["related_definitions"]} == {("MAX_RETRIES", "module_level")}
    assert fn["source_bytes"] == len(fn["source"].encode())


def test_function_context_decorators_and_start_line(env):
    fn = ok(env.run("get_function_context", symbol_name="Database.make", path="app/db.py"))["function"]
    assert fn["decorators"] == ["staticmethod"] and fn["source"].lstrip().startswith("@staticmethod")


def test_function_context_javascript(env):
    fn = ok(env.run("get_function_context", symbol_name="fetchUser", path="web/src/api.js"))["function"]
    assert fn["signature"] == "function fetchUser(id)" and "axios.get" in fn["source"]
    assert [i["line"] for i in fn["imports_used"]] == [1, 2]  # axios and formatDate imports


def test_function_context_ambiguous_returns_candidates_not_a_guess(env):
    data = ok(env.run("get_function_context", symbol_name="process"))
    assert data["resolved"] is False and data["ambiguous"] is True and data["function"] is None
    assert len(data["candidates"]) == 3 and any("path" in n for n in data["notes"])
    assert ok(env.run("get_function_context", symbol_name="process", path="app/jobs.py"))["resolved"] is True


def test_function_context_refuses_oversized_source_instead_of_truncating(env):
    e = err(env.run("get_function_context", symbol_name="big", path="app/big.py", max_source_bytes=200), "source_too_large")
    assert "never truncated" in e.message
    full = ok(env.run("get_function_context", symbol_name="big", path="app/big.py", max_source_bytes=5000))["function"]
    assert "value_29" in full["source"]


def test_function_context_not_found(env):
    data = ok(env.run("get_function_context", symbol_name="nope", path="app/db.py"))
    assert data["resolved"] is False and data["candidates"] == []


# --- find_references ---------------------------------------------------------


def _refs(data):
    return {(r["path"], r["category"], r["resolution"]) for r in data["references"]}


def test_find_references_classifies_resolution(env):
    data = ok(env.run("find_references", symbol_name="get_connection", max_files=150))
    refs = _refs(data)
    assert ("app/service.py", "import", "confirmed_import_binding") in refs
    assert ("app/service.py", "call", "confirmed_import_binding") in refs
    assert ("app/service.py", "method_call", "confirmed_import_binding") in refs  # database.get_connection()
    assert ("tests/test_db.py", "call", "confirmed_import_binding") in refs
    assert ("app/db.py", "call", "same_file_name_match") not in refs  # not called by bare name in db.py
    assert ("app/other.py", "method_call", "name_match_unresolved") in refs  # obj.get_connection(): unknown receiver
    assert data["definitions"][0]["path"] == "app/db.py" and data["definition_ambiguous"] is False
    assert data["counts_by_resolution"]["confirmed_import_binding"] >= 4
    first = data["references"][0]
    assert first["match_type"] == "ast" and first["excerpt"] and first["line"] > 0
    assert any("no type inference" in n.lower() for n in data["notes"])
    ranks = [r["resolution"] for r in data["references"]]
    assert ranks == sorted(ranks, key=lambda r: ["confirmed_import_binding", "same_class_self_access",
                                                 "same_file_name_match", "star_import_binding",
                                                 "name_match_unresolved", "textual_heuristic"].index(r))


def test_find_references_textual_matches_are_separate_and_labelled(env):
    without = ok(env.run("find_references", symbol_name="get_connection", max_files=150))
    assert not any(r["match_type"] == "textual" for r in without["references"])
    data = ok(env.run("find_references", symbol_name="get_connection", include_textual=True, max_files=150))
    textual = [r for r in data["references"] if r["match_type"] == "textual"]
    assert {(r["path"], r["line"]) for r in textual} == {("app/notes.py", 1), ("app/notes.py", 2)}
    assert all(r["resolution"] == "textual_heuristic" and r["category"] == "textual" for r in textual)
    assert any("plain text" in n for n in data["notes"])


def test_find_references_javascript_and_typescript(env):
    data = ok(env.run("find_references", symbol_name="fetchUser", max_files=150))
    refs = _refs(data)
    assert ("web/src/App.tsx", "import", "confirmed_import_binding") in refs
    assert ("web/src/App.tsx", "call", "confirmed_import_binding") in refs
    assert ("web/src/api.test.js", "call", "confirmed_import_binding") in refs
    assert ("web/src/api.js", "call", "same_file_name_match") in refs
    jsx = ok(env.run("find_references", symbol_name="Profile", max_files=150))
    assert ("web/src/App.tsx", "identifier", "same_file_name_match") in _refs(jsx)  # <Profile /> usage
    namespace = ok(env.run("find_references", symbol_name="formatDate", max_files=150))
    assert ("web/src/api.test.js", "method_call", "confirmed_import_binding") in _refs(namespace)  # utils.formatDate


def test_find_references_without_definition_is_all_unresolved(env):
    data = ok(env.run("find_references", symbol_name="psycopg", max_files=150))
    assert data["definitions"] == []
    assert data["references"] and all(r["resolution"] == "name_match_unresolved" for r in data["references"])
    assert any("No definition" in n for n in data["notes"])


def test_find_references_path_disambiguation_and_errors(env):
    err(env.run("find_references", symbol_name="process", path="app/db.py"), "symbol_not_found")
    data = ok(env.run("find_references", symbol_name="process", max_files=150))
    assert data["definition_ambiguous"] is True and len(data["definitions"]) == 3
    one = ok(env.run("find_references", symbol_name="process", path="app/jobs.py", max_files=150))
    assert one["definition_ambiguous"] is False


def test_find_references_result_limit(env):
    data = ok(env.run("find_references", symbol_name="get_connection", max_results=2, max_files=150))
    assert len(data["references"]) == 2 and data["reference_count"] > 2
    assert "max_results" in data["limits_hit"] and data["complete"] is False


# --- find_callers ------------------------------------------------------------


def test_find_callers_with_resolution_and_callers(env):
    data = ok(env.run("find_callers", symbol_name="get_connection", max_files=150))
    callers = {(c["caller_path"], c["caller_symbol"], c["resolution"]) for c in data["callers"]}
    assert ("app/service.py", "load_user", "confirmed_import_binding") in callers
    assert ("app/service.py", "reload", "confirmed_import_binding") in callers  # via module alias
    assert ("tests/test_db.py", "test_connect_calls_get_connection", "confirmed_import_binding") in callers
    assert ("app/other.py", "run", "name_match_unresolved") in callers
    call = next(c for c in data["callers"] if c["caller_symbol"] == "load_user")
    assert call["call_excerpt"] == "conn = get_connection()" and call["call_kind"] == "function_call"
    assert call["caller_start_line"] < call["call_line"] <= call["caller_end_line"] and call["target"] == "get_connection"
    assert data["resolved_commit_sha"] == SHA_MAIN
    assert any("dynamic dispatch" in n for n in data["notes"])
    # The textual mention in app/notes.py is not a call and must not appear.
    assert not any(c["caller_path"] == "app/notes.py" for c in data["callers"])


def test_find_callers_can_exclude_unresolved(env):
    data = ok(env.run("find_callers", symbol_name="get_connection", include_unresolved=False, max_files=150))
    assert data["callers"] and all(c["resolution"] != "name_match_unresolved" for c in data["callers"])


def test_find_callers_self_calls_and_methods(env):
    data = ok(env.run("find_callers", symbol_name="Worker.process", max_files=150))
    assert [(c["caller_symbol"], c["resolution"], c["call_kind"]) for c in data["callers"]] == [
        ("Worker.run", "same_class_self_access", "method_call")]
    # A method called on an arbitrary object is matched by name only and says so.
    conn = ok(env.run("find_callers", symbol_name="Database.connect", max_files=150))
    assert {c["resolution"] for c in conn["callers"]} == {"name_match_unresolved"}


def test_find_callers_javascript_enclosing_symbols(env):
    data = ok(env.run("find_callers", symbol_name="fetchUser", max_files=150))
    callers = {(c["caller_path"], c["caller_symbol"]) for c in data["callers"]}
    assert {("web/src/api.js", "ApiClient.get"), ("web/src/api.js", "helper"),
            ("web/src/App.tsx", "App"), ("web/src/api.test.js", "<module>")} <= callers


def test_find_callers_constructor_calls(env):
    data = ok(env.run("find_callers", symbol_name="Database", max_files=150))
    assert {c["caller_symbol"] for c in data["callers"]} >= {"get_connection", "reload"}


# --- analyze_dependencies ----------------------------------------------------


def test_dependencies_for_one_file(env):
    data = ok(env.run("analyze_dependencies", path="web/src/App.tsx"))
    by_target = {(e["target"], e["import_kind"]): e for e in data["edges"]}
    assert by_target[("react", "es_import")]["target_kind"] == "external"
    assert by_target[("web/src/api.js", "es_import")]["target_kind"] == "internal"
    assert by_target[("web/src/utils.ts", "es_import")]["type_only"] is True
    assert by_target[("web/src/utils.ts", "es_export_from")]["names"] == ["*"]
    assert data["internal_adjacency"] == {"web/src/App.tsx": ["web/src/api.js", "web/src/utils.ts"]}
    assert [p["name"] for p in data["external_packages"]] == ["react"]
    assert data["complete"] is True and data["files_analyzed"] == ["web/src/App.tsx"]


def test_dependencies_commonjs_and_builtins(env):
    data = ok(env.run("analyze_dependencies", path="web/src/api.js"))
    kinds = {(e["target"], e["target_kind"], e["import_kind"]) for e in data["edges"]}
    assert ("fs", "builtin", "require") in kinds and ("lodash", "external", "require") in kinds
    assert ("axios", "external", "es_import") in kinds and ("web/src/utils.ts", "internal", "es_import") in kinds


def test_dependencies_python_relative_absolute_and_stdlib(env):
    data = ok(env.run("analyze_dependencies", path="app/db.py"))
    kinds = {(e["target"], e["target_kind"]) for e in data["edges"]}
    assert ("os", "stdlib") in kinds and ("psycopg", "external") in kinds
    assert ("app/config.py", "internal") in kinds  # `from .config import ...`
    svc = ok(env.run("analyze_dependencies", path="app/service.py"))
    internal = {e["target"] for e in svc["edges"] if e["target_kind"] == "internal"}
    assert internal == {"app/db.py", "app/__init__.py"}  # `from app import db` also executes the package itself


def test_dependency_graph_whole_repository_and_dependents(env):
    graph = ok(env.run("analyze_dependencies", max_files=150))
    adj = graph["internal_adjacency"]
    assert adj["app/service.py"] == ["app/__init__.py", "app/db.py"] and adj["app/db.py"] == ["app/config.py"]
    assert "app/service.py" in graph["reverse_adjacency"]["app/db.py"]
    assert {"axios", "react", "lodash", "psycopg"} <= {p["name"] for p in graph["external_packages"] if p["kind"] == "external"}
    assert graph["complete"] is False  # broken files exist
    dependents = ok(env.run("analyze_dependencies", path="app/db.py", include_dependents=True, max_files=150))
    assert {"app/service.py", "tests/test_db.py"} <= set(dependents["reverse_adjacency"]["app/db.py"])


def test_dependency_edge_limit(env):
    data = ok(env.run("analyze_dependencies", max_files=150, max_edges=3))
    assert len(data["edges"]) == 3 and data["edge_count"] > 3 and "max_edges" in data["limits_hit"]


# --- find_related_files ------------------------------------------------------


def test_related_files_rank_imports_above_proximity(env):
    data = ok(env.run("find_related_files", path="app/db.py", max_files=150))
    related = {r["path"]: r for r in data["related"]}
    assert related["app/config.py"]["relationships"][0]["type"] == "imports"
    assert related["app/service.py"]["relationships"][0]["type"] == "imported_by"
    assert related["tests/test_db.py"]["is_test_file_by_name"] is True
    weak = related["app/jobs.py"]
    assert [r["type"] for r in weak["relationships"]] == ["same_directory"] and weak["relationships"][0]["strength"] == "weak"
    assert "does NOT show a dependency" in weak["explanation"]
    assert weak["score"] < related["app/service.py"]["score"] and weak["score"] < related["app/config.py"]["score"]
    order = [r["path"] for r in data["related"]]
    assert order.index("app/service.py") < order.index("app/jobs.py")
    assert any("heuristic" in n for n in data["notes"])


def test_related_files_symbol_references_strong_vs_name_only(env):
    data = ok(env.run("find_related_files", path="app/db.py", symbol_name="get_connection", max_files=150))
    related = {r["path"]: {x["type"]: x for x in r["relationships"]} for r in data["related"]}
    assert related["app/service.py"]["references_symbol"]["strength"] == "strong"
    assert related["app/other.py"]["references_symbol_by_name_only"]["strength"] == "moderate"
    assert "may be an unrelated symbol" in related["app/other.py"]["references_symbol_by_name_only"]["detail"]
    err(env.run("find_related_files", path="app/db.py", symbol_name="missing"), "symbol_not_found")


def test_related_files_js_reexports_and_test_naming(env):
    data = ok(env.run("find_related_files", path="web/src/utils.ts", max_files=150))
    related = {r["path"]: {x["type"] for x in r["relationships"]} for r in data["related"]}
    assert related["web/src/App.tsx"] == {"imported_by", "re_exported_by", "same_directory"}
    assert "imported_by" in related["web/src/api.test.js"]
    svc = ok(env.run("find_related_files", path="app/service.py", max_files=150))
    naming = {r["path"]: {x["type"] for x in r["relationships"]} for r in svc["related"]}
    assert "test_naming_convention" in naming["tests/test_service.py"]


# --- inspect_configuration ---------------------------------------------------


def test_configuration_never_fetches_secret_bearing_files(env):
    data = ok(env.run("inspect_configuration"))
    excluded = {e["path"] for e in data["excluded_files"]}
    assert {".env", ".env.local", ".env.production", "keys/server.pem"} <= excluded
    requested = " ".join(env.fake.paths())
    for name in (".env", ".env.local", ".env.production", "keys/server.pem"):
        assert sha_of(name) not in requested  # no blob request was ever made for these
    blob = json.dumps(data)
    for secret in ("hunter2", "hunter3", "hunter4", "BEGIN PRIVATE KEY", "MIIEvQ"):
        assert secret not in blob
    assert {f["path"] for f in data["files"]} >= {".env.example", "package.json", "tsconfig.json",
                                                  "requirements.txt", "pyproject.toml", "vite.config.ts"}


def test_env_example_is_inspected_and_redacted_not_trusted(env):
    data = ok(env.run("inspect_configuration"))
    template = next(f for f in data["files"] if f["path"] == ".env.example")
    keys = {k["key"]: k for k in template["keys"]}
    assert keys["LOG_LEVEL"]["value"] == "info" and keys["APP_DEBUG"]["value"] == "1"
    assert keys["API_URL"]["value"] == "http://localhost:3000" and keys["API_URL"]["line"] == 3
    for secret_key in ("DATABASE_URL", "STRIPE_KEY", "OPAQUE"):
        assert keys[secret_key]["value"] is None and keys[secret_key]["value_redacted"] is True, secret_key
    assert keys["SESSION_SECRET"]["value"] == ""  # unset is not a secret
    assert keys["OPTIONAL_FLAG"]["commented_out"] is True and keys["OPTIONAL_FLAG"]["value"] == "true"
    assert {e["name"] for e in template["env_vars"]} >= {"DATABASE_URL", "API_URL", "SESSION_SECRET"}
    blob = json.dumps(data)
    for secret in ("user:pass@", "sk_live_abcdefghijklmnop1234", "aGVsbG8gd29ybGQ", "db.example.com"):
        assert secret not in blob
    assert template["redaction_count"] == 3 and data["total_redactions"] >= 3


def test_package_json_dependencies_scripts_and_redaction(env):
    data = ok(env.run("inspect_configuration"))
    pkg = next(f for f in data["files"] if f["path"] == "package.json")
    deps = {d["name"]: d for d in pkg["dependencies"]}
    assert deps["axios"]["spec"] == "^1.6.0" and deps["axios"]["group"] == "dependencies" and deps["axios"]["line"] == 9
    assert deps["vitest"]["group"] == "devDependencies"
    assert deps["private-lib"]["spec_redacted"] is True and "ghp_" not in json.dumps(pkg)
    scripts = {k["key"]: k for k in pkg["keys"] if k["key"].startswith("scripts.")}
    assert scripts["scripts.build"]["value"] == "vite build"
    assert scripts["scripts.start"]["value"] == "API_TOKEN=[REDACTED] node server.js"
    assert "abcd1234abcd1234" not in json.dumps(data)


def test_tsconfig_requirements_pyproject_and_bundler_config(env):
    data = ok(env.run("inspect_configuration"))
    files = {f["path"]: f for f in data["files"]}
    ts = {k["key"]: k["value"] for k in files["tsconfig.json"]["keys"]}
    assert files["tsconfig.json"]["parse_error"] is None
    assert ts["compilerOptions.target"] == "ES2020" and ts["compilerOptions.strict"] == "True"
    reqs = {d["name"]: d for d in files["requirements.txt"]["dependencies"]}
    assert reqs["fastapi"]["spec"] == ">=0.110" and reqs["fastapi"]["line"] == 1
    assert reqs["psycopg"]["spec"] == ">=3.2" and "secret" not in json.dumps(files["requirements.txt"])
    option = next(k for k in files["requirements.txt"]["keys"] if k["key"] == "option" and "index-url" in str(k["value"] or "index-url"))
    assert option["value_redacted"] is True
    py = {d["name"]: d["group"] for d in files["pyproject.toml"]["dependencies"]}
    assert py == {"httpx": "project", "pydantic": "project"}
    assert any(k["key"] == "project.name" and k["value"] == "webapp" for k in files["pyproject.toml"]["keys"])
    vite = files["vite.config.ts"]
    assert {e["name"] for e in vite["env_vars"]} == {"VITE_API", "VITE_PUBLIC"}
    assert [d["name"] for d in vite["dependencies"]] == ["vite"]


def test_configuration_env_var_summary_with_source_references(env):
    data = ok(env.run("inspect_configuration", include_source_env_refs=True, max_files=150))
    summary = {e["name"]: e for e in data["env_vars"]}
    assert summary["DATABASE_URL"]["declared_in"] == [".env.example:2"]
    assert "app/config.py:3" in summary["DATABASE_URL"]["referenced_in"]
    assert "app/config.py:4" in summary["APP_DEBUG"]["referenced_in"] and "APP_TIMEOUT" in summary
    assert "web/src/api.js:6" in summary["API_URL"]["referenced_in"]


def test_configuration_malformed_file_is_a_parse_error_not_a_crash(env):
    env.fake.files["package.json"] = b"{ not json"
    data = ok(env.run("inspect_configuration"))
    pkg = next(f for f in data["files"] if f["path"] == "package.json")
    assert pkg["parse_error"] and data["complete"] is False
    assert any(e["path"] == "package.json" for e in data["parse_errors"])


def test_configuration_secret_in_example_file_with_innocent_key_is_still_redacted(env):
    env.fake.files[".env.example"] = (
        b"INNOCENT_NAME=ghp_abcdefghijklmnopqrstuvwxyz0123456789\nHOST=https://admin:pw@internal.example\n")
    data = ok(env.run("inspect_configuration"))
    blob = json.dumps(data)
    assert "ghp_abcdefghijklmnop" not in blob and "admin:pw" not in blob


def test_configuration_prefix_scoping(env):
    data = ok(env.run("inspect_configuration", path_prefix="app"))
    assert data["files"] == [] and any("No supported" in n for n in data["notes"])


# --- inspect_tests -----------------------------------------------------------


def _test_file(data, path):
    return next(f for f in data["test_files"] if f["path"] == path)


def test_inspect_tests_distinguishes_import_from_invocation(env):
    data = ok(env.run("inspect_tests", path="app/db.py", symbol_name="get_connection", max_files=150))
    f = _test_file(data, "tests/test_db.py")
    assert f["relation"] == "invokes_target" and f["imports_target"] is True and f["imported_names"] == ["get_connection"]
    cases = {c["qualified_name"]: c for c in f["test_cases"]}
    assert cases["test_connect_calls_get_connection"]["invokes_target"] is True
    assert cases["test_connect_calls_get_connection"]["invocation_kind"] == "direct_call"
    assert cases["test_module_attribute"]["invocation_kind"] == "module_attribute_call"
    assert cases["test_imports_only"]["invokes_target"] is False and cases["test_imports_only"]["invocation_kind"] is None
    assert cases["LegacyTests.test_legacy"]["invokes_target"] is True
    legacy_kinds = {a["kind"] for a in cases["LegacyTests.test_legacy"]["assertions"]}
    assert legacy_kinds == {"assertX"} or "assertX" in legacy_kinds
    first = cases["test_connect_calls_get_connection"]["assertions"][0]
    assert first["kind"] == "assert" and "get_connection()" in first["excerpt"] and first["line"] > 0
    assert f["framework"] == "pytest+unittest" and "imports pytest" in f["framework_evidence"]
    assert data["target_definitions"][0]["path"] == "app/db.py"


def test_inspect_tests_method_target_is_matched_by_name_only(env):
    data = ok(env.run("inspect_tests", path="app/db.py", symbol_name="Database.connect", max_files=150))
    f = _test_file(data, "tests/test_db.py")
    assert f["relation"] == "calls_method_by_name"
    case = next(c for c in f["test_cases"] if c["qualified_name"] == "TestDatabase.test_connect")
    assert case["invokes_target"] is False and case["invocation_kind"] == "method_name_match"


def test_inspect_tests_module_level_and_naming_only(env):
    module = ok(env.run("inspect_tests", path="app/db.py", max_files=150))
    assert _test_file(module, "tests/test_db.py")["relation"] == "invokes_target"
    assert all(f["path"] != "tests/test_unrelated.py" for f in module["test_files"])  # no evidence, not listed
    svc = ok(env.run("inspect_tests", path="app/service.py", max_files=150))
    only = _test_file(svc, "tests/test_service.py")
    assert only["relation"] == "naming_convention_only" and only["imports_target"] is False
    assert any("file name" in n for n in only["notes"])


def test_inspect_tests_javascript_frameworks_and_assertions(env):
    data = ok(env.run("inspect_tests", path="web/src/api.js", symbol_name="fetchUser", max_files=150))
    f = _test_file(data, "web/src/api.test.js")
    assert f["relation"] == "invokes_target" and f["framework"] == "unknown"
    cases = {c["qualified_name"]: c for c in f["test_cases"]}
    assert cases["api > fetches"]["invokes_target"] is True
    assert cases["api > fetches"]["assertions"][0]["kind"] == "expect"
    assert cases["api > formats"]["invokes_target"] is False
    utils = ok(env.run("inspect_tests", path="web/src/utils.ts", symbol_name="formatDate", max_files=150))
    case = next(c for c in _test_file(utils, "web/src/api.test.js")["test_cases"] if c["name"] == "formats")
    assert case["invocation_kind"] == "module_attribute_call"


def test_inspect_tests_framework_detection(env):
    env.fake.files["web/src/v.test.ts"] = (
        b"import { describe, it, expect, vi } from 'vitest';\nimport { fetchUser } from './api';\n"
        b"describe('v', () => { it('works', () => { vi.fn(); expect(fetchUser(1)).toBe(1); }); });\n")
    env.fake.files["web/src/m.spec.js"] = (
        b"const { fetchUser } = require('./api');\nconst assert = require('assert');\n"
        b"describe('m', function () { it('works', function () { assert.equal(fetchUser(1), 1); }); });\n")
    data = ok(env.run("inspect_tests", path="web/src/api.js", symbol_name="fetchUser", max_files=150))
    assert _test_file(data, "web/src/v.test.ts")["framework"] == "vitest"
    mocha = _test_file(data, "web/src/m.spec.js")
    assert mocha["relation"] == "invokes_target" and mocha["test_cases"][0]["assertions"][0]["kind"] == "assert"


def test_inspect_tests_unknown_symbol_and_unsupported_target(env):
    err(env.run("inspect_tests", path="app/db.py", symbol_name="nope"), "symbol_not_found")
    err(env.run("inspect_tests", path="web/lib/native.go"), "unsupported_language")
    err(env.run("inspect_tests", path="app/missing.py"), "path_not_found")


def test_inspect_tests_reports_statically_only(env):
    data = ok(env.run("inspect_tests", path="app/db.py", max_files=150))
    assert any("never executed" in n for n in data["notes"]) and data["resolved_commit_sha"] == SHA_MAIN
    assert {r["method"] for r in env.fake.requests} == {"GET"}


# --- output safety -----------------------------------------------------------


def test_results_never_contain_tokens(env, caplog):
    results = [env.run(tool, **extra) for tool, extra in ALL_TOOLS]
    blob = " ".join(r.model_dump_json() for r in results)
    assert TOKEN not in blob and "Bearer" not in blob and TOKEN not in caplog.text
    assert re.search(r"ghs_", blob) is None


def test_every_tool_reports_completeness_fields(env):
    for tool, extra in ALL_TOOLS:
        data = ok(env.run(tool, **extra))
        for field in ("resolved_commit_sha", "complete", "files_scanned", "files_skipped", "limits_hit",
                      "parse_errors", "unsupported_files", "notes"):
            assert field in data, (tool, field)
        assert data["resolved_commit_sha"] == SHA_MAIN
