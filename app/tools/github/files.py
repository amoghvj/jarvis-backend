"""File content and repository search tools."""

import base64
import binascii
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from app.tools.errors import ToolError
from app.tools.github import access
from app.tools.github.api import path_segments, repo_path
from app.tools.github.common import (
    DirectoryPath,
    FilePath,
    GitRef,
    ReferenceInfo,
    RepositoryInput,
    as_dict,
    build,
    require_sha,
)
from app.tools.github.repositories import fetch_directory_tree
from app.tools.schemas import ToolConstraints, ToolContext, ToolDefinition

MAX_FILE_BYTES = 512 * 1024  # larger files are refused, never truncated
MAX_SEARCH_FILE_BYTES = 200 * 1024
MAX_SEARCH_TOTAL_BYTES = 2 * 1024 * 1024
MAX_MATCHES_PER_FILE = 5
SNIPPET_LINE_CHARS = 300
SEARCH_TIME_RESERVE_SECONDS = 4.0


# --- get_file_contents -------------------------------------------------------

class FileContentsInput(RepositoryInput):
    path: FilePath = Field(description="File path within the repository")
    ref: Optional[GitRef] = Field(default=None, description="Branch, tag or commit SHA; defaults to the default branch")


class FileContentsOutput(BaseModel):
    path: str
    content: str
    encoding: str
    size_bytes: int
    line_count: int
    line_endings: str
    sha: str
    reference: ReferenceInfo
    complete: bool = True


def _line_endings(text: str) -> str:
    crlf = text.count("\r\n")
    lf = text.count("\n") - crlf
    cr = text.count("\r") - crlf
    kinds = [name for name, n in (("crlf", crlf), ("lf", lf), ("cr", cr)) if n]
    if not kinds:
        return "none"
    return kinds[0] if len(kinds) == 1 else "mixed"


def _decode_text(raw: bytes) -> str:
    if b"\x00" in raw:
        raise ToolError("unsupported_content", "The file is binary and cannot be returned as text")
    try:
        return raw.decode("utf-8")  # no newline translation: line endings are preserved
    except UnicodeDecodeError:
        raise ToolError("unsupported_content", "The file is not valid UTF-8 text")


def _decode_base64(value: object) -> bytes:
    if not isinstance(value, str):
        raise ToolError("github_bad_response", "GitHub returned an unexpected response shape")
    try:
        return base64.b64decode("".join(value.split()), validate=True)
    except (binascii.Error, ValueError):
        raise ToolError("github_bad_response", "GitHub returned undecodable file content")


def get_file_contents(ctx: ToolContext, inp: FileContentsInput) -> FileContentsOutput:
    client, repo_entry = access.open_repository(ctx, inp.installation_id, inp.owner, inp.repo)
    reference = access.resolve_reference(client, inp.owner, inp.repo, repo_entry, inp.ref)
    resp = client.get(
        repo_path(inp.owner, inp.repo, "contents", *path_segments(inp.path)),
        {"ref": reference.commit_sha},
    )
    if isinstance(resp.data, list):
        raise ToolError("path_is_directory", "The path is a directory; use get_repository_tree")
    data = as_dict(resp.data)
    kind = data.get("type")
    if kind != "file":
        raise ToolError("unsupported_content", f"The path is a {str(kind)[:20]}, not a regular file")

    size = data.get("size")
    if not isinstance(size, int):
        raise ToolError("github_bad_response", "GitHub returned an unexpected response shape")
    if size > MAX_FILE_BYTES:
        raise ToolError(
            "file_too_large",
            f"The file is {size} bytes; the limit is {MAX_FILE_BYTES}. It is not truncated or returned.",
        )
    if data.get("encoding") != "base64":
        raise ToolError("unsupported_content", "GitHub did not return inline file content")

    text = _decode_text(_decode_base64(data.get("content")))
    line_count = text.count("\n") + (1 if text and not text.endswith("\n") else 0)
    return build(
        FileContentsOutput,
        path=inp.path,
        content=text,
        encoding="utf-8",
        size_bytes=size,
        line_count=line_count,
        line_endings=_line_endings(text),
        sha=require_sha(data.get("sha")),
        reference=reference,
    )


# --- search_repository -------------------------------------------------------

LANGUAGE_EXTENSIONS: Dict[str, tuple] = {
    "python": (".py",),
    "javascript": (".js", ".jsx", ".mjs", ".cjs"),
    "typescript": (".ts", ".tsx"),
    "java": (".java",),
    "kotlin": (".kt", ".kts"),
    "go": (".go",),
    "ruby": (".rb",),
    "rust": (".rs",),
    "c": (".c", ".h"),
    "cpp": (".cc", ".cpp", ".cxx", ".hpp", ".hh"),
    "csharp": (".cs",),
    "php": (".php",),
    "swift": (".swift",),
    "sql": (".sql",),
    "shell": (".sh", ".bash"),
    "yaml": (".yml", ".yaml"),
    "json": (".json",),
    "markdown": (".md",),
}
SKIPPED_DIRECTORIES = {
    "node_modules", "vendor", ".git", "dist", "build", "__pycache__", ".venv",
    "venv", "target", ".next", "coverage",
}
BINARY_EXTENSIONS = (
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".zip", ".gz", ".tar", ".jar",
    ".woff", ".woff2", ".ttf", ".eot", ".mp3", ".mp4", ".mov", ".exe", ".dll", ".so",
    ".pyc", ".class", ".lock", ".svg", ".bin", ".wasm",
)


class SearchRepositoryInput(RepositoryInput):
    query: str = Field(min_length=2, max_length=200, description="Literal text to find (not a regular expression)")
    ref: Optional[GitRef] = None
    path_prefix: Optional[DirectoryPath] = Field(default=None, description="Only search below this directory")
    language: Optional[Literal[tuple(LANGUAGE_EXTENSIONS)]] = None  # type: ignore[valid-type]
    case_sensitive: bool = False
    max_results: int = Field(default=20, ge=1, le=50)
    max_files: int = Field(default=40, ge=1, le=100, description="Maximum files fetched and scanned")
    context_lines: int = Field(default=1, ge=0, le=3)

    @field_validator("query")
    @classmethod
    def _printable(cls, v: str) -> str:
        if any(ord(c) < 32 for c in v):
            raise ValueError("query must not contain control characters")
        return v


class SearchMatch(BaseModel):
    path: str
    line_number: int
    snippet: str
    blob_sha: str


class SearchRepositoryOutput(BaseModel):
    reference: ReferenceInfo
    query: str
    matches: List[SearchMatch]
    candidate_files: int
    files_scanned: int
    skipped: Dict[str, int]
    limits_hit: List[str]
    complete: bool
    notes: List[str]


def _is_candidate(entry: dict, prefix: str, extensions: Optional[tuple]) -> bool:
    path = entry.get("path")
    if entry.get("type") != "blob" or not isinstance(path, str):
        return False
    if prefix and not (path == prefix or path.startswith(prefix + "/")):
        return False
    lower = path.lower()
    if extensions is not None:
        return lower.endswith(extensions)
    return not lower.endswith(BINARY_EXTENSIONS)


def search_repository(ctx: ToolContext, inp: SearchRepositoryInput) -> SearchRepositoryOutput:
    """Bounded literal search over one exact reference.

    GitHub's code-search API is intentionally not used: it indexes only the default
    branch and cannot be pinned to a commit, so results could describe different
    code than the requested reference. This instead walks that reference's tree and
    reads a bounded number of files. Results say explicitly when they are incomplete.
    """
    client, repo_entry = access.open_repository(ctx, inp.installation_id, inp.owner, inp.repo)
    reference = access.resolve_reference(client, inp.owner, inp.repo, repo_entry, inp.ref)
    entries, tree_truncated = fetch_directory_tree(
        client, inp.owner, inp.repo, reference.commit_sha, "", True
    )

    limits: List[str] = []
    skipped = {"too_large": 0, "binary_or_non_utf8": 0, "excluded_directory": 0, "fetch_errors": 0}
    if tree_truncated:
        limits.append("tree_truncated")

    prefix = inp.path_prefix or ""
    extensions = LANGUAGE_EXTENSIONS[inp.language] if inp.language else None
    candidates = []
    for e in entries:
        if not _is_candidate(e, prefix, extensions):
            continue
        if SKIPPED_DIRECTORIES.intersection(e["path"].split("/")[:-1]):
            skipped["excluded_directory"] += 1
            continue
        if not isinstance(e.get("size"), int) or e["size"] > MAX_SEARCH_FILE_BYTES:
            skipped["too_large"] += 1
            continue
        candidates.append(e)
    if len(candidates) > inp.max_files:
        limits.append("max_files")

    needle = inp.query if inp.case_sensitive else inp.query.lower()
    matches: List[SearchMatch] = []
    scanned = 0
    total_bytes = 0
    for e in candidates[: inp.max_files]:
        if ctx.time_remaining() < SEARCH_TIME_RESERVE_SECONDS:
            limits.append("time_budget")
            break
        if total_bytes + e["size"] > MAX_SEARCH_TOTAL_BYTES:
            limits.append("max_total_bytes")
            break
        sha = require_sha(e.get("sha"))
        try:
            blob = as_dict(client.get(repo_path(inp.owner, inp.repo, "git", "blobs", sha)).data)
            raw = _decode_base64(blob.get("content"))
        except ToolError as exc:
            if exc.code in ("not_found", "github_bad_response", "invalid_request"):
                skipped["fetch_errors"] += 1
                continue
            raise  # rate limits, auth and outages abort the whole search
        scanned += 1
        total_bytes += len(raw)
        try:
            text = _decode_text(raw)
        except ToolError:
            skipped["binary_or_non_utf8"] += 1
            continue

        lines = text.split("\n")  # not splitlines(): keeps line numbers aligned with editors
        in_file = 0
        for index, line in enumerate(lines):
            haystack = line if inp.case_sensitive else line.lower()
            if needle not in haystack:
                continue
            lo, hi = max(0, index - inp.context_lines), min(len(lines), index + inp.context_lines + 1)
            snippet = "\n".join(l.rstrip("\r")[:SNIPPET_LINE_CHARS] for l in lines[lo:hi])
            matches.append(SearchMatch(path=e["path"], line_number=index + 1, snippet=snippet, blob_sha=sha))
            in_file += 1
            if len(matches) >= inp.max_results:
                limits.append("max_results")
                break
            if in_file >= MAX_MATCHES_PER_FILE:
                limits.append("max_matches_per_file")
                break
        if "max_results" in limits:
            break

    limits = sorted(set(limits))
    incomplete = bool(limits) or skipped["fetch_errors"] > 0
    notes = [
        "Literal substring search over the resolved reference; GitHub code search is not used "
        "because it only indexes the default branch.",
        "Files over the size limit, binary files and common vendor/build directories are skipped.",
    ]
    if incomplete:
        notes.append("Results are INCOMPLETE; the absence of a match does not prove the text is absent.")
    return SearchRepositoryOutput(
        reference=reference,
        query=inp.query,
        matches=matches,
        candidate_files=len(candidates),
        files_scanned=scanned,
        skipped=skipped,
        limits_hit=limits,
        complete=not incomplete,
        notes=notes,
    )


TOOLS = [
    ToolDefinition(
        name="get_file_contents",
        description=(
            "Read one text file at a reference. Files over 512 KB, binary files and non-UTF-8 files are "
            "refused rather than truncated."
        ),
        input_model=FileContentsInput,
        output_model=FileContentsOutput,
        handler=get_file_contents,
    ),
    ToolDefinition(
        name="search_repository",
        description=(
            "Bounded literal text search at one reference using the git tree and blobs. Reports exactly which "
            "limits were hit and whether results are complete."
        ),
        input_model=SearchRepositoryInput,
        output_model=SearchRepositoryOutput,
        handler=search_repository,
        constraints=ToolConstraints(timeout_seconds=28.0),
    ),
]

