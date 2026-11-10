"""Commit, commit-diff and file-history tools."""

from typing import List, Optional

from pydantic import BaseModel, Field

from app.tools.github import access
from app.tools.github.api import repo_path
from app.tools.github.common import (
    CommitSha,
    FilePath,
    GitRef,
    Pagination,
    ReferenceInfo,
    RepositoryInput,
    as_dict,
    as_list,
    build,
    require_sha,
    truncate_text,
)
from app.tools.schemas import ToolConstraints, ToolContext, ToolDefinition

MAX_MESSAGE_CHARS = 10_000
MAX_PATCH_CHARS_PER_FILE = 20_000
MAX_TOTAL_PATCH_CHARS = 120_000


class CommitPerson(BaseModel):
    name: Optional[str] = None
    login: Optional[str] = None
    date: Optional[str] = None


def _person(git_person: object, account: object) -> CommitPerson:
    p = git_person if isinstance(git_person, dict) else {}
    a = account if isinstance(account, dict) else {}
    # Email addresses are deliberately not returned.
    return CommitPerson(name=p.get("name"), login=a.get("login"), date=p.get("date"))


# --- get_commit --------------------------------------------------------------

class CommitInput(RepositoryInput):
    sha: CommitSha = Field(description="Commit SHA (7-40 hex characters)")


class CommitOutput(BaseModel):
    sha: str
    message: str
    message_truncated: bool
    author: CommitPerson
    committer: CommitPerson
    parents: List[str]
    html_url: Optional[str] = None
    total_changes: Optional[int] = None
    additions: Optional[int] = None
    deletions: Optional[int] = None


def get_commit(ctx: ToolContext, inp: CommitInput) -> CommitOutput:
    client, _ = access.open_repository(ctx, inp.installation_id, inp.owner, inp.repo)
    data = as_dict(client.get(repo_path(inp.owner, inp.repo, "commits", inp.sha)).data)
    commit = as_dict(data.get("commit"))
    message, truncated = truncate_text(str(commit.get("message", "")), MAX_MESSAGE_CHARS)
    stats = data.get("stats") if isinstance(data.get("stats"), dict) else {}
    return build(
        CommitOutput,
        sha=require_sha(data.get("sha")),
        message=message,
        message_truncated=truncated,
        author=_person(commit.get("author"), data.get("author")),
        committer=_person(commit.get("committer"), data.get("committer")),
        parents=[require_sha(as_dict(p).get("sha")) for p in as_list(data.get("parents"))],
        html_url=data.get("html_url"),
        total_changes=stats.get("total"),
        additions=stats.get("additions"),
        deletions=stats.get("deletions"),
    )


# --- get_commit_diff ---------------------------------------------------------

class CommitDiffInput(CommitInput):
    page: int = Field(default=1, ge=1, le=1000, description="Page of changed files")
    per_page: int = Field(default=30, ge=1, le=100, description="Changed files per page")
    include_patch: bool = True


class ChangedFile(BaseModel):
    filename: str
    previous_filename: Optional[str] = None
    status: str
    additions: int = 0
    deletions: int = 0
    changes: int = 0
    sha: Optional[str] = None
    patch: Optional[str] = None
    patch_available: bool
    patch_truncated: bool = False


class CommitDiffOutput(BaseModel):
    commit_sha: str
    files: List[ChangedFile]
    pagination: Pagination
    truncated: bool
    notes: List[str]


def get_commit_diff(ctx: ToolContext, inp: CommitDiffInput) -> CommitDiffOutput:
    client, _ = access.open_repository(ctx, inp.installation_id, inp.owner, inp.repo)
    resp = client.get(
        repo_path(inp.owner, inp.repo, "commits", inp.sha),
        {"page": inp.page, "per_page": inp.per_page},
    )
    data = as_dict(resp.data)

    files: List[ChangedFile] = []
    budget = MAX_TOTAL_PATCH_CHARS
    any_truncated = False
    any_missing = False
    for raw in as_list(data.get("files")):
        f = as_dict(raw)
        patch = f.get("patch") if isinstance(f.get("patch"), str) else None
        patch_truncated = False
        if not inp.include_patch:
            kept = None
        elif patch is None:
            kept = None
            any_missing = True  # GitHub omits patches for binary or very large changes
        else:
            kept, patch_truncated = truncate_text(patch, min(MAX_PATCH_CHARS_PER_FILE, max(budget, 0)))
            budget -= len(kept)
            if len(patch) > len(kept):
                patch_truncated = True
        any_truncated = any_truncated or patch_truncated
        files.append(
            build(
                ChangedFile,
                filename=f.get("filename"),
                previous_filename=f.get("previous_filename"),
                status=f.get("status"),
                additions=f.get("additions", 0),
                deletions=f.get("deletions", 0),
                changes=f.get("changes", 0),
                sha=f.get("sha"),
                patch=kept,
                patch_available=patch is not None,
                patch_truncated=patch_truncated,
            )
        )

    notes = []
    if any_truncated:
        notes.append("Some patches were truncated to fit size limits; they are marked patch_truncated.")
    if any_missing:
        notes.append("GitHub did not provide a patch for some files (binary or too large).")
    if resp.has_next:
        notes.append("More changed files are available; request the next page.")
    return CommitDiffOutput(
        commit_sha=require_sha(data.get("sha")),
        files=files,
        pagination=Pagination(
            page=inp.page, per_page=inp.per_page, has_next_page=resp.has_next, next_page=resp.next_page
        ),
        truncated=any_truncated or resp.has_next,
        notes=notes,
    )


# --- get_file_history --------------------------------------------------------

class FileHistoryInput(RepositoryInput):
    path: FilePath
    ref: Optional[GitRef] = Field(default=None, description="Branch, tag or commit SHA to walk back from; defaults to the default branch")
    page: int = Field(default=1, ge=1, le=1000)
    per_page: int = Field(default=30, ge=1, le=100)


class HistoryCommit(BaseModel):
    sha: str
    author: CommitPerson
    message: str
    message_truncated: bool = False
    html_url: Optional[str] = None


class FileHistoryOutput(BaseModel):
    path: str
    reference: ReferenceInfo
    commits: List[HistoryCommit]
    pagination: Pagination


def get_file_history(ctx: ToolContext, inp: FileHistoryInput) -> FileHistoryOutput:
    client, repo_entry = access.open_repository(ctx, inp.installation_id, inp.owner, inp.repo)
    reference = access.resolve_reference(client, inp.owner, inp.repo, repo_entry, inp.ref)
    resp = client.get(
        repo_path(inp.owner, inp.repo, "commits"),
        {"sha": reference.commit_sha, "path": inp.path, "page": inp.page, "per_page": inp.per_page},
    )
    commits = []
    for raw in as_list(resp.data):
        item = as_dict(raw)
        commit = as_dict(item.get("commit"))
        message, truncated = truncate_text(str(commit.get("message", "")), 2000)
        commits.append(
            build(
                HistoryCommit,
                sha=require_sha(item.get("sha")),
                author=_person(commit.get("author"), item.get("author")),
                message=message,
                message_truncated=truncated,
                html_url=item.get("html_url"),
            )
        )
    return FileHistoryOutput(
        path=inp.path,
        reference=reference,
        commits=commits,
        pagination=Pagination(
            page=inp.page, per_page=inp.per_page, has_next_page=resp.has_next, next_page=resp.next_page
        ),
    )


TOOLS = [
    ToolDefinition(
        name="get_commit",
        description=(
            "Purpose: Fetch metadata for one commit sha in owner/repo (author, committer, timestamps, commit message, parent SHAs, addition/deletion stats). "
            "When to use: After get_file_history or when inspecting a specific commit SHA to see who authored it, when, and why. "
            "When not to use: To see the actual line-by-line code changes in that commit, use get_commit_diff. "
            "Scope & limitations: Requires a commit SHA (7-40 hex chars) under installation_id."
        ),
        input_model=CommitInput,
        output_model=CommitOutput,
        handler=get_commit,
    ),
    ToolDefinition(
        name="get_commit_diff",
        description=(
            "Purpose: Retrieve changed files and unified diff patches for a commit sha in owner/repo (optionally filtered to a single path). "
            "When to use: When checking whether a recent commit introduced a regression or modified a suspect file/function. "
            "When not to use: Do not use before identifying a relevant commit SHA (use get_file_history first to find commits touching a file). "
            "Scope & limitations: Bounded file/patch sizes; explicitly flags truncated patches."
        ),
        input_model=CommitDiffInput,
        output_model=CommitDiffOutput,
        handler=get_commit_diff,
        constraints=ToolConstraints(timeout_seconds=25.0),
    ),
    ToolDefinition(
        name="get_file_history",
        description=(
            "Purpose: List commits that modified a specific file path in owner/repo walking backward from ref (newest first, paginated). "
            "When to use: When investigating recent code changes to a file implicated by a stack trace or incident. "
            "When not to use: Do not use to read current file contents (use get_file_contents) or diff a known commit SHA (use get_commit_diff). "
            "Scope & limitations: Returns commit SHAs, authors, dates, and messages; follow up with get_commit_diff on suspect SHAs."
        ),
        input_model=FileHistoryInput,
        output_model=FileHistoryOutput,
        handler=get_file_history,
    ),
]

