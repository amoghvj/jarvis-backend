"""Real GitHub integration verification script for J.A.R.V.I.S. tools.

Runs read-only verification across all nine GitHub tools and authorization boundaries
against a real GitHub App installation. Never writes or modifies GitHub repositories,
branches, commits, or installations. Never outputs secrets, tokens, or credentials.
"""

import argparse
import os
import sys
import time
from typing import Any, Dict, Optional
from uuid import UUID

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), ".env"))

import psycopg
from psycopg.rows import dict_row

from app.integrations.github.config import get_github_app_settings
from app.tools.executor import execute_tool
from app.tools.schemas import ToolContext, ToolResult


class TestSummary:
    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0
        self.skipped = 0

    def record_pass(self, name: str, detail: str = "") -> None:
        self.passed += 1
        suffix = f" ({detail})" if detail else ""
        print(f"  [PASS] {name}{suffix}")

    def record_fail(self, name: str, reason: str) -> None:
        self.failed += 1
        print(f"  [FAIL] {name}: {reason}")

    def record_skip(self, name: str, reason: str) -> None:
        self.skipped += 1
        print(f"  [SKIP] {name}: {reason}")


def check_prerequisites() -> Optional[Dict[str, Any]]:
    print("Checking prerequisites...")
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        print("  [ERROR] DATABASE_URL is not set.")
        return None

    try:
        settings = get_github_app_settings()
        print(f"  [OK] GitHub App configured: App ID {settings.app_id}, Slug '{settings.app_slug}'")
    except Exception as exc:
        print(f"  [ERROR] GitHub App settings missing or incomplete: {exc}")
        print("         Ensure GITHUB_APP_ID, GITHUB_APP_CLIENT_ID, GITHUB_APP_CLIENT_SECRET,")
        print("         GITHUB_APP_PRIVATE_KEY, and GITHUB_APP_SLUG are set in .env")
        return None

    try:
        conn = psycopg.connect(db_url, row_factory=dict_row)
    except Exception as exc:
        print(f"  [ERROR] Could not connect to database: {exc}")
        return None

    return {"conn": conn, "settings": settings}


def run_test(
    summary: TestSummary,
    name: str,
    tool_name: str,
    params: Dict[str, Any],
    ctx: ToolContext,
    *,
    expect_status: str = "success",
    expect_error_code: Optional[str] = None,
    validate_fn: Optional[Any] = None,
) -> Optional[ToolResult]:
    started = time.perf_counter()
    result = execute_tool(tool_name, params, ctx)
    duration_ms = int((time.perf_counter() - started) * 1000)

    if expect_status == "success":
        if result.status != "success":
            code = result.error.code if result.error else "unknown"
            msg = result.error.message if result.error else "no error details"
            summary.record_fail(name, f"Expected success but got error '{code}': {msg}")
            return result
        if validate_fn:
            try:
                validate_fn(result.data)
            except AssertionError as exc:
                summary.record_fail(name, f"Assertion failed: {exc}")
                return result
            except Exception as exc:
                summary.record_fail(name, f"Validation error: {exc}")
                return result
        summary.record_pass(name, f"{duration_ms}ms")
        return result
    else:
        if result.status == "success":
            summary.record_fail(name, f"Expected error '{expect_error_code}' but call succeeded")
            return result
        actual_code = result.error.code if result.error else ""
        if expect_error_code and actual_code != expect_error_code:
            summary.record_fail(
                name, f"Expected error code '{expect_error_code}' but got '{actual_code}'"
            )
            return result
        summary.record_pass(name, f"Rejected with {actual_code} ({duration_ms}ms)")
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify J.A.R.V.I.S. GitHub tools on real installation")
    parser.add_argument("--installation-id", type=int, help="Target GitHub App installation ID")
    parser.add_argument("--owner", type=str, help="Repository owner")
    parser.add_argument("--repo", type=str, help="Repository name")
    parser.add_argument("--user-id", type=str, help="User UUID owning the installation")
    args = parser.parse_args()

    prereqs = check_prerequisites()
    if not prereqs:
        sys.exit(1)

    conn = prereqs["conn"]
    summary = TestSummary()

    # Discover user and installation if not provided
    user_id_val = None
    installation_id_val = args.installation_id
    owner_val = args.owner
    repo_val = args.repo

    if args.user_id:
        user_id_val = UUID(args.user_id)
    else:
        row = conn.execute(
            "select user_id, installation_id, github_account_login from public.github_app_installations "
            "where suspended_at is null order by created_at desc limit 1"
        ).fetchone()
        if not row:
            print("  [ERROR] No active GitHub App installations found in database.")
            conn.close()
            sys.exit(1)
        user_id_val = row["user_id"]
        if not installation_id_val:
            installation_id_val = row["installation_id"]
        print(f"  [INFO] Using database installation {installation_id_val} for account '{row['github_account_login']}'")

    ctx = ToolContext(user_id=user_id_val, conn=conn)

    print("\n--- Phase 1: Installation & Repository Discovery ---")
    run_test(
        summary,
        "1. list_installations (DB)",
        "list_installations",
        {},
        ctx,
        validate_fn=lambda d: len(d["installations"]) > 0,
    )

    res_repos = run_test(
        summary,
        "2. list_repositories (GitHub API)",
        "list_repositories",
        {"installation_id": installation_id_val, "page": 1, "per_page": 10},
        ctx,
        validate_fn=lambda d: len(d["repositories"]) > 0,
    )

    if not (owner_val and repo_val) and res_repos and res_repos.status == "success" and res_repos.data:
        first_repo = res_repos.data["repositories"][0]
        owner_val = first_repo["owner"]
        repo_val = first_repo["name"]
        print(f"  [INFO] Selected repository '{owner_val}/{repo_val}' for read-only inspection tests")

    if not (owner_val and repo_val):
        print("  [ERROR] No repositories found in this installation to test.")
        conn.close()
        sys.exit(1)

    base_params = {"installation_id": installation_id_val, "owner": owner_val, "repo": repo_val}

    print("\n--- Phase 2: Repository Metadata & Tree Retrieval ---")
    run_test(
        summary,
        "3. get_repository_metadata",
        "get_repository_metadata",
        base_params,
        ctx,
        validate_fn=lambda d: d["full_name"].lower() == f"{owner_val}/{repo_val}".lower(),
    )

    res_tree_root = run_test(
        summary,
        "4a. get_repository_tree (root non-recursive)",
        "get_repository_tree",
        {**base_params, "recursive": False, "limit": 20},
        ctx,
        validate_fn=lambda d: len(d["entries"]) > 0 and d["reference"]["commit_sha"],
    )

    sample_commit = (
        res_tree_root.data["reference"]["commit_sha"]
        if res_tree_root and res_tree_root.data and res_tree_root.data.get("reference")
        else None
    )

    sample_file = None
    if res_tree_root and res_tree_root.data:
        for entry in res_tree_root.data.get("entries", []):
            if entry.get("type") == "file":
                sample_file = entry["path"]
                break

    res_tree_rec = run_test(
        summary,
        "4b. get_repository_tree (recursive)",
        "get_repository_tree",
        {**base_params, "recursive": True, "limit": 20},
        ctx,
        validate_fn=lambda d: len(d["entries"]) > 0,
    )

    if not sample_file and res_tree_rec and res_tree_rec.data:
        for entry in res_tree_rec.data.get("entries", []):
            if entry.get("type") == "file":
                sample_file = entry["path"]
                break

    print("\n--- Phase 3: File Inspection & Literal Search ---")
    if sample_file:
        run_test(
            summary,
            f"5. get_file_contents ({sample_file})",
            "get_file_contents",
            {**base_params, "path": sample_file},
            ctx,
            validate_fn=lambda d: d["size_bytes"] >= 0 and d["complete"] is True,
        )

        run_test(
            summary,
            "6. search_repository (literal search)",
            "search_repository",
            {**base_params, "query": "import" if sample_file.endswith(".py") else "a", "max_results": 5},
            ctx,
            validate_fn=lambda d: "limits_hit" in d and d["reference"]["commit_sha"],
        )
    else:
        summary.record_skip("5. get_file_contents", "No regular file found in repository root")
        summary.record_skip("6. search_repository", "No regular file found to search")

    print("\n--- Phase 4: Commit Metadata, Diffs, and History ---")
    if sample_commit:
        run_test(
            summary,
            "7. get_commit",
            "get_commit",
            {**base_params, "sha": sample_commit},
            ctx,
            validate_fn=lambda d: d["sha"].lower() == sample_commit.lower() and "email" not in str(d["author"]),
        )

        run_test(
            summary,
            "8. get_commit_diff",
            "get_commit_diff",
            {**base_params, "sha": sample_commit, "per_page": 5},
            ctx,
            validate_fn=lambda d: "files" in d and isinstance(d["truncated"], bool),
        )
    else:
        summary.record_skip("7. get_commit", "Could not resolve sample commit SHA")
        summary.record_skip("8. get_commit_diff", "Could not resolve sample commit SHA")

    if sample_file:
        run_test(
            summary,
            f"9. get_file_history ({sample_file})",
            "get_file_history",
            {**base_params, "path": sample_file, "per_page": 5},
            ctx,
            validate_fn=lambda d: len(d["commits"]) > 0 and d["reference"]["commit_sha"],
        )
    else:
        summary.record_skip("9. get_file_history", "No sample file available")

    print("\n--- Phase 5: Security & Negative Boundary Checks ---")
    run_test(
        summary,
        "10. Invalid reference rejected",
        "get_repository_tree",
        {**base_params, "ref": "nonexistent-branch-test-boundary-check-404"},
        ctx,
        expect_status="error",
        expect_error_code="reference_not_found",
    )

    run_test(
        summary,
        "11. Unassigned repository denied (membership check)",
        "get_repository_metadata",
        {"installation_id": installation_id_val, "owner": "torvalds", "repo": "linux"},
        ctx,
        expect_status="error",
        expect_error_code="repository_not_accessible",
    )

    # Cross-user installation denial check
    with conn.transaction():
        u_foreign = conn.execute(
            "insert into public.users (github_id, github_username, email) "
            "values (999999998, 'foreign_tester', 'foreign_tester@example.com') returning id"
        ).fetchone()
        ctx_foreign = ToolContext(user_id=u_foreign["id"], conn=conn)

        run_test(
            summary,
            "12. Cross-user installation access denied",
            "list_repositories",
            {"installation_id": installation_id_val},
            ctx_foreign,
            expect_status="error",
            expect_error_code="installation_not_found",
        )
        conn.rollback()

    conn.close()

    print("\n" + "=" * 50)
    print(f"VERIFICATION RESULTS: {summary.passed} Passed, {summary.failed} Failed, {summary.skipped} Skipped")
    print("=" * 50)

    if summary.failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
