"""Installation, repository listing, metadata and tree tools."""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field

from app import repositories as db_repo
from app.tools.errors import ToolError
from app.tools.github import access
from app.tools.github.api import repo_path, path_segments
from app.tools.github.common import (
    DirectoryPath,
    GitRef,
    InstallationInput,
    Pagination,
    ReferenceInfo,
    RepositoryInput,
    ToolInput,
    as_dict,
    as_list,
    build,
    require_sha,
)
from app.tools.schemas import ToolConstraints, ToolContext, ToolDefinition


# --- list_installations ------------------------------------------------------

class ListInstallationsInput(ToolInput):
    pass


class InstallationSummary(BaseModel):
    installation_id: int
    github_account_id: int
    account_login: str
    account_type: str
    repository_selection: Optional[str] = None
    suspended: bool
    suspended_at: Optional[datetime] = None


class ListInstallationsOutput(BaseModel):
    installations: List[InstallationSummary]
    count: int


def list_installations(ctx: ToolContext, _: ListInstallationsInput) -> ListInstallationsOutput:
    """Only the authenticated user's own linked installations; no GitHub call is made."""
    rows = db_repo.list_github_app_installations_for_user(ctx.conn, ctx.user_id)
    items = [
        InstallationSummary(
            installation_id=int(r["installation_id"]),
            github_account_id=int(r["github_account_id"]),
            account_login=r["github_account_login"],
            account_type=r["github_account_type"],
            repository_selection=r["repository_selection"],
            suspended=r["suspended_at"] is not None,
            suspended_at=r["suspended_at"],
        )
        for r in rows
    ]
    return ListInstallationsOutput(installations=items, count=len(items))


# --- list_repositories -------------------------------------------------------

class ListRepositoriesInput(InstallationInput):
    page: int = Field(default=1, ge=1, le=1000)
    per_page: int = Field(default=30, ge=1, le=100)


class RepositorySummary(BaseModel):
    id: int
    name: str
    full_name: str
    owner: str
    default_branch: Optional[str] = None
    private: bool
    html_url: str


class ListRepositoriesOutput(BaseModel):
    installation_id: int
    repositories: List[RepositorySummary]
    total_count: Optional[int] = None
    pagination: Pagination


def _summarize(entry: object) -> RepositorySummary:
    e = as_dict(entry)
    return build(
        RepositorySummary,
        id=e.get("id"),
        name=e.get("name"),
        full_name=e.get("full_name"),
        owner=as_dict(e.get("owner")).get("login"),
        default_branch=e.get("default_branch"),
        private=e.get("private"),
        html_url=e.get("html_url"),
    )


def list_repositories(ctx: ToolContext, inp: ListRepositoriesInput) -> ListRepositoriesOutput:
    client = access.open_installation(ctx, inp.installation_id)
    resp = client.get("/installation/repositories", {"per_page": inp.per_page, "page": inp.page})
    data = as_dict(resp.data)
    return ListRepositoriesOutput(
        installation_id=inp.installation_id,
        repositories=[_summarize(r) for r in as_list(data.get("repositories"))],
        total_count=data.get("total_count") if isinstance(data.get("total_count"), int) else None,
        pagination=Pagination(
            page=inp.page, per_page=inp.per_page, has_next_page=resp.has_next, next_page=resp.next_page
        ),
    )


# --- get_repository_metadata -------------------------------------------------

class RepositoryMetadataOutput(BaseModel):
    id: int
    name: str
    full_name: str
    owner: str
    owner_type: Optional[str] = None
    description: Optional[str] = None
    default_branch: Optional[str] = None
    private: bool
    visibility: Optional[str] = None
    language: Optional[str] = None
    html_url: str
    clone_url: Optional[str] = None
    archived: bool = False
    fork: bool = False
    size_kb: Optional[int] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    pushed_at: Optional[str] = None


def get_repository_metadata(ctx: ToolContext, inp: RepositoryInput) -> RepositoryMetadataOutput:
    client, _ = access.open_repository(ctx, inp.installation_id, inp.owner, inp.repo)
    r = as_dict(client.get(repo_path(inp.owner, inp.repo)).data)
    owner = as_dict(r.get("owner"))
    return build(
        RepositoryMetadataOutput,
        id=r.get("id"),
        name=r.get("name"),
        full_name=r.get("full_name"),
        owner=owner.get("login"),
        owner_type=owner.get("type"),
        description=r.get("description"),
        default_branch=r.get("default_branch"),
        private=r.get("private"),
        visibility=r.get("visibility"),
        language=r.get("language"),
        html_url=r.get("html_url"),
        clone_url=r.get("clone_url"),
        archived=bool(r.get("archived")),
        fork=bool(r.get("fork")),
        size_kb=r.get("size") if isinstance(r.get("size"), int) else None,
        created_at=r.get("created_at"),
        updated_at=r.get("updated_at"),
        pushed_at=r.get("pushed_at"),
    )


# --- get_repository_tree -----------------------------------------------------

_TYPE_MAP = {"blob": "file", "tree": "directory", "commit": "submodule"}


class RepositoryTreeInput(RepositoryInput):
    ref: Optional[GitRef] = Field(default=None, description="Branch, tag or commit SHA; defaults to the default branch")
    path: Optional[DirectoryPath] = Field(default=None, description="Directory to list; defaults to the repository root")
    recursive: bool = Field(default=False, description="List everything below the directory")
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=200, ge=1, le=1000)


class TreeEntry(BaseModel):
    path: str
    type: str
    size: Optional[int] = None
    sha: str
    mode: Optional[str] = None


class RepositoryTreeOutput(BaseModel):
    reference: ReferenceInfo
    directory: str
    recursive: bool
    entries: List[TreeEntry]
    total_entries: int
    returned_entries: int
    offset: int
    next_offset: Optional[int] = None
    truncated: bool
    github_truncated: bool
    notes: List[str] = []


def tree_entries(raw_tree: object) -> List[dict]:
    return [e for e in as_list(raw_tree) if isinstance(e, dict)]


def fetch_directory_tree(client, owner: str, repo: str, commit_sha: str, directory: str, recursive: bool):
    """Walk down to `directory` one level at a time, then list it. Returns (entries, github_truncated)."""
    sha = commit_sha
    for segment in path_segments(directory):
        data = as_dict(client.get(repo_path(owner, repo, "git", "trees", sha)).data)
        match = next(
            (
                e
                for e in tree_entries(data.get("tree"))
                if e.get("type") == "tree" and e.get("path") == segment
            ),
            None,
        )
        if match is None:
            raise ToolError("path_not_found", "Directory not found at the given reference")
        sha = require_sha(match.get("sha"))
    data = as_dict(
        client.get(
            repo_path(owner, repo, "git", "trees", sha), {"recursive": "1"} if recursive else None
        ).data
    )
    return tree_entries(data.get("tree")), bool(data.get("truncated"))


def get_repository_tree(ctx: ToolContext, inp: RepositoryTreeInput) -> RepositoryTreeOutput:
    client, repo_entry = access.open_repository(ctx, inp.installation_id, inp.owner, inp.repo)
    reference = access.resolve_reference(client, inp.owner, inp.repo, repo_entry, inp.ref)
    directory = inp.path or ""
    raw, github_truncated = fetch_directory_tree(
        client, inp.owner, inp.repo, reference.commit_sha, directory, inp.recursive
    )

    prefix = f"{directory}/" if directory else ""
    entries = []
    for e in raw:
        size = e.get("size")
        entries.append(
            build(
                TreeEntry,
                path=prefix + str(e.get("path", "")),
                type=_TYPE_MAP.get(e.get("type"), "other"),
                size=size if isinstance(size, int) else None,
                sha=require_sha(e.get("sha")),
                mode=e.get("mode"),
            )
        )

    page = entries[inp.offset : inp.offset + inp.limit]
    has_more = inp.offset + inp.limit < len(entries)
    notes = []
    if github_truncated:
        notes.append("GitHub truncated this tree; the listing is incomplete. Browse subdirectories individually.")
    if has_more:
        notes.append("More entries are available; request again with next_offset.")
    return RepositoryTreeOutput(
        reference=reference,
        directory=directory,
        recursive=inp.recursive,
        entries=page,
        total_entries=len(entries),
        returned_entries=len(page),
        offset=inp.offset,
        next_offset=inp.offset + inp.limit if has_more else None,
        truncated=github_truncated or has_more,
        github_truncated=github_truncated,
        notes=notes,
    )


TOOLS = [
    ToolDefinition(
        name="list_installations",
        description="List the GitHub App installations linked to the authenticated J.A.R.V.I.S. user.",
        input_model=ListInstallationsInput,
        output_model=ListInstallationsOutput,
        handler=list_installations,
        source="database",
    ),
    ToolDefinition(
        name="list_repositories",
        description="List repositories accessible through one of the user's GitHub App installations (paginated).",
        input_model=ListRepositoriesInput,
        output_model=ListRepositoriesOutput,
        handler=list_repositories,
    ),
    ToolDefinition(
        name="get_repository_metadata",
        description="Get identity, visibility, language, URLs and timestamps for a repository in an installation.",
        input_model=RepositoryInput,
        output_model=RepositoryMetadataOutput,
        handler=get_repository_metadata,
    ),
    ToolDefinition(
        name="get_repository_tree",
        description="List files and directories at a reference. Bounded; reports truncation and the exact reference used.",
        input_model=RepositoryTreeInput,
        output_model=RepositoryTreeOutput,
        handler=get_repository_tree,
        constraints=ToolConstraints(timeout_seconds=25.0),
    ),
]

