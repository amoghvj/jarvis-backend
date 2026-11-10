"""Input validation and response helpers shared by all GitHub tools."""

import re
from typing import Annotated, Any, List, Optional, Type, TypeVar

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, ValidationError

from app.tools.errors import ToolError

_OWNER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9-]{0,38}")
_REPO_RE = re.compile(r"[A-Za-z0-9._-]{1,100}")
_REF_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,254}")
_HEX_RE = re.compile(r"[0-9a-fA-F]{7,40}")
FULL_SHA_RE = re.compile(r"[0-9a-fA-F]{40}")


def _owner(v: str) -> str:
    if not _OWNER_RE.fullmatch(v):
        raise ValueError("owner must be a valid GitHub account name")
    return v


def _repo(v: str) -> str:
    if v in (".", "..") or not _REPO_RE.fullmatch(v):
        raise ValueError("repo must be a valid repository name")
    return v


def _ref(v: str) -> str:
    if not _REF_RE.fullmatch(v) or ".." in v or "//" in v or v.endswith("/"):
        raise ValueError("ref must be a valid branch, tag or commit SHA")
    return v


def _sha(v: str) -> str:
    if not _HEX_RE.fullmatch(v):
        raise ValueError("sha must be a 7-40 character hexadecimal commit SHA")
    return v.lower()


def clean_path(value: str, allow_empty: bool) -> str:
    """Normalize a repository path; reject traversal, control characters and odd segments."""
    path = value.strip().strip("/")
    if not path:
        if allow_empty:
            return ""
        raise ValueError("path must not be empty")
    if len(path) > 1024:
        raise ValueError("path is too long")
    if "\\" in path or any(ord(c) < 32 or ord(c) == 127 for c in path):
        raise ValueError("path contains invalid characters")
    if any(seg in ("", ".", "..") for seg in path.split("/")):
        raise ValueError("path must not contain empty, '.' or '..' segments")
    return path


OwnerName = Annotated[str, AfterValidator(_owner)]
RepoName = Annotated[str, AfterValidator(_repo)]
GitRef = Annotated[str, AfterValidator(_ref)]
CommitSha = Annotated[str, AfterValidator(_sha)]
FilePath = Annotated[str, AfterValidator(lambda v: clean_path(v, False))]
DirectoryPath = Annotated[str, AfterValidator(lambda v: clean_path(v, True))]


class ToolInput(BaseModel):
    # Unknown fields (for example a smuggled `user_id`) are rejected, not ignored.
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class InstallationInput(ToolInput):
    installation_id: int = Field(gt=0, le=2**63 - 1, description="J.A.R.V.I.S.-linked GitHub App installation ID")


class RepositoryInput(InstallationInput):
    owner: OwnerName = Field(description="Repository owner (user or organization login)")
    repo: RepoName = Field(description="Repository name")


class ReferenceInfo(BaseModel):
    """Which version of the code a result describes."""

    requested: Optional[str] = None
    kind: str  # "default_branch" | "ref" | "commit_sha"
    name: Optional[str] = None
    commit_sha: str
    default_branch_used: bool
    note: Optional[str] = None


class Pagination(BaseModel):
    page: int
    per_page: int
    has_next_page: bool
    next_page: Optional[int] = None


_M = TypeVar("_M", bound=BaseModel)


def build(model: Type[_M], **fields: Any) -> _M:
    """Build an output model, treating a malformed GitHub payload as a safe error."""
    try:
        return model(**fields)
    except ValidationError:
        raise ToolError("github_bad_response", "GitHub returned an unexpected response shape")


def as_dict(value: Any) -> dict:
    if not isinstance(value, dict):
        raise ToolError("github_bad_response", "GitHub returned an unexpected response shape")
    return value


def as_list(value: Any) -> List[Any]:
    if not isinstance(value, list):
        raise ToolError("github_bad_response", "GitHub returned an unexpected response shape")
    return value


def require_sha(value: Any) -> str:
    """GitHub-supplied SHAs are untrusted too; validate before using them in a URL."""
    if not isinstance(value, str) or not FULL_SHA_RE.fullmatch(value):
        raise ToolError("github_bad_response", "GitHub returned an unexpected response shape")
    return value.lower()


def truncate_text(text: str, limit: int) -> tuple:
    if len(text) <= limit:
        return text, False
    return text[:limit], True

