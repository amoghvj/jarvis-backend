"""Authorization boundary for repository tools.

Trust boundary: `GitHubRepoClient` is a low-level helper that will happily use any
installation ID it is given. Everything application-facing must obtain it through
`open_installation`, which proves the installation is linked to the authenticated
J.A.R.V.I.S. user *before* any installation token is generated or used.
"""

from typing import Any, Dict, Optional, Tuple

from app import repositories as db_repo
from app.integrations.github.config import get_github_app_settings
from app.tools.errors import ToolError
from app.tools.github.api import GitHubRepoClient, repo_path
from app.tools.github.common import (
    FULL_SHA_RE,
    ReferenceInfo,
    as_dict,
    as_list,
    require_sha,
)
from app.tools.schemas import ToolContext

MAX_REPO_SCAN_PAGES = 10  # up to 1,000 repositories per installation
DEFAULT_REF_NOTE = (
    "No reference was supplied, so the head of the default branch was used. "
    "This may differ from the code that was running when an incident occurred."
)


def open_installation(ctx: ToolContext, installation_id: int) -> GitHubRepoClient:
    """Return a client for an installation owned by the authenticated user.

    Another user's installation and a nonexistent one produce the same error, so
    installation IDs cannot be probed.
    """
    installation = db_repo.get_github_app_installation_for_user(
        ctx.conn, ctx.user_id, installation_id
    )
    if not installation:
        raise ToolError("installation_not_found", "Installation not found for this account")
    if installation["suspended_at"] is not None:
        raise ToolError("installation_suspended", "This GitHub App installation is suspended")
    try:
        settings = get_github_app_settings()
    except RuntimeError:
        raise ToolError("github_app_not_configured", "The GitHub App is not configured on the server")
    return GitHubRepoClient(ctx, settings, installation_id)


def find_repository(client: GitHubRepoClient, owner: str, repo: str) -> Dict[str, Any]:
    """Prove the repository belongs to this installation and return its listing entry.

    An installation token can read *any public repository*, so a successful
    `GET /repos/{owner}/{repo}` does not show the repo is part of the installation.
    Membership is therefore checked against `/installation/repositories`.
    """
    wanted = f"{owner}/{repo}".lower()
    for page in range(1, MAX_REPO_SCAN_PAGES + 1):
        resp = client.get("/installation/repositories", {"per_page": 100, "page": page})
        data = as_dict(resp.data)
        for entry in as_list(data.get("repositories")):
            if isinstance(entry, dict) and str(entry.get("full_name", "")).lower() == wanted:
                return entry
        if not resp.has_next:
            raise ToolError(
                "repository_not_accessible",
                "The repository is not accessible through this installation",
            )
    raise ToolError(
        "repository_lookup_incomplete",
        "The installation has too many repositories to verify access safely; "
        "restrict the installation to selected repositories",
    )


def open_repository(
    ctx: ToolContext, installation_id: int, owner: str, repo: str
) -> Tuple[GitHubRepoClient, Dict[str, Any]]:
    client = open_installation(ctx, installation_id)
    return client, find_repository(client, owner, repo)


def _head_of(client: GitHubRepoClient, owner: str, repo: str, name: str) -> str:
    try:
        resp = client.get(repo_path(owner, repo, "commits"), {"sha": name, "per_page": 1})
    except ToolError as exc:
        if exc.code in ("not_found", "invalid_request"):
            raise ToolError("reference_not_found", "The reference was not found in this repository")
        raise
    commits = as_list(resp.data)
    if not commits:
        raise ToolError("reference_not_found", "The reference was not found in this repository")
    return require_sha(as_dict(commits[0]).get("sha"))


def resolve_reference(
    client: GitHubRepoClient,
    owner: str,
    repo: str,
    repo_entry: Dict[str, Any],
    ref: Optional[str],
) -> ReferenceInfo:
    """Resolve an optional branch/tag/SHA to a commit SHA and report what was used.

    Hex strings of 7-40 characters are treated as commit SHAs.
    """
    if ref is None:
        branch = repo_entry.get("default_branch")
        if not isinstance(branch, str) or not branch:
            raise ToolError("github_bad_response", "GitHub returned an unexpected response shape")
        return ReferenceInfo(
            requested=None,
            kind="default_branch",
            name=branch,
            commit_sha=_head_of(client, owner, repo, branch),
            default_branch_used=True,
            note=DEFAULT_REF_NOTE,
        )
    if FULL_SHA_RE.fullmatch(ref):
        return ReferenceInfo(
            requested=ref, kind="commit_sha", commit_sha=ref.lower(), default_branch_used=False
        )
    if all(c in "0123456789abcdefABCDEF" for c in ref) and 7 <= len(ref) < 40:
        try:
            data = as_dict(client.get(repo_path(owner, repo, "commits", ref)).data)
        except ToolError as exc:
            if exc.code in ("not_found", "invalid_request"):
                raise ToolError("reference_not_found", "The reference was not found in this repository")
            raise
        return ReferenceInfo(
            requested=ref,
            kind="commit_sha",
            commit_sha=require_sha(data.get("sha")),
            default_branch_used=False,
        )
    return ReferenceInfo(
        requested=ref,
        kind="ref",
        name=ref,
        commit_sha=_head_of(client, owner, repo, ref),
        default_branch_used=False,
    )

