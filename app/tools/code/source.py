"""One analysis session = one repository at one resolved commit.

All access goes through the existing GitHub tool layer:

* `access.open_repository` verifies installation ownership and repository membership,
* `access.resolve_reference` pins the analysis to a commit SHA,
* `fetch_directory_tree` lists that commit's tree once, and
* blobs are fetched *by the SHA listed in that tree*, so every file read in the
  session belongs to the same commit even if a branch moves mid-analysis.

Fetched files are cached for the session, and every limit that is hit is recorded so
results can state honestly whether they are complete.
"""

from typing import Any, Dict, List, Optional, Set

from app.tools.code import model as m
from app.tools.code.common import AnalysisResult, ParseErrorInfo
from app.tools.code.js_parser import parse_js_family
from app.tools.code.languages import PYTHON, detect_language, is_unsupported_source
from app.tools.code.py_parser import parse_python
from app.tools.errors import ToolError
from app.tools.github import access
from app.tools.github.api import repo_path
from app.tools.github.common import as_dict, require_sha
from app.tools.github.files import (
    MAX_SEARCH_FILE_BYTES,
    SKIPPED_DIRECTORIES,
    _decode_base64,
    _decode_text,
)
from app.tools.github.repositories import fetch_directory_tree
from app.tools.schemas import ToolContext

MAX_TOTAL_BYTES = 3 * 1024 * 1024
TIME_RESERVE_SECONDS = 3.0
MAX_UNSUPPORTED_LISTED = 25
_STOP_CODES = ("github_rate_limited", "github_unavailable", "timeout")


class AnalysisSession:
    def __init__(self, ctx: ToolContext, inp: Any):
        self.ctx = ctx
        self.owner, self.repo = inp.owner, inp.repo
        self.client, repo_entry = access.open_repository(ctx, inp.installation_id, inp.owner, inp.repo)
        self.reference = access.resolve_reference(self.client, inp.owner, inp.repo, repo_entry, inp.ref)
        entries, tree_truncated = fetch_directory_tree(
            self.client, inp.owner, inp.repo, self.reference.commit_sha, "", True
        )
        self.blobs: Dict[str, dict] = {
            e["path"]: e for e in entries if e.get("type") == "blob" and isinstance(e.get("path"), str)
        }
        self.limits: Set[str] = {"tree_truncated"} if tree_truncated else set()
        self.skipped: Dict[str, int] = {
            "too_large": 0, "binary_or_non_utf8": 0, "excluded_directory": 0, "fetch_errors": 0,
        }
        self.parse_errors: List[ParseErrorInfo] = []
        self.unsupported: Set[str] = set()
        self.notes: List[str] = []
        self._texts: Dict[str, Optional[str]] = {}
        self._parsed: Dict[str, Optional[m.ParsedFile]] = {}
        self._scanned: Set[str] = set()
        self._total_bytes = 0
        self._stopped = False

    # -- tree -----------------------------------------------------------------

    @property
    def commit_sha(self) -> str:
        return self.reference.commit_sha

    def has_path(self, path: str) -> bool:
        return path in self.blobs

    def all_paths(self) -> List[str]:
        return sorted(self.blobs)

    def source_candidates(self, prefix: str = "", language: Optional[str] = None) -> List[str]:
        """Supported-language files under `prefix`, in deterministic order.

        Skipped/unsupported files are tallied here so they show up in the result.
        """
        out: List[str] = []
        for path in sorted(self.blobs):
            if prefix and not (path == prefix or path.startswith(prefix + "/")):
                continue
            lang = detect_language(path)
            if lang is None:
                if is_unsupported_source(path):
                    self.unsupported.add(path)
                continue
            if language and lang != language:
                continue
            if SKIPPED_DIRECTORIES.intersection(path.split("/")[:-1]):
                self.skipped["excluded_directory"] += 1
                continue
            size = self.blobs[path].get("size")
            if not isinstance(size, int) or size > MAX_SEARCH_FILE_BYTES:
                self.skipped["too_large"] += 1
                continue
            out.append(path)
        return out

    # -- fetching and parsing -------------------------------------------------

    def text(self, path: str) -> Optional[str]:
        """File text at the pinned commit, or None if it cannot be provided (tallied)."""
        if path in self._texts:
            return self._texts[path]
        entry = self.blobs.get(path)
        if entry is None or self._stopped:
            return None
        if self.ctx.time_remaining() < TIME_RESERVE_SECONDS:
            self._stop("time_budget")
            return None
        size = entry.get("size") if isinstance(entry.get("size"), int) else 0
        if size > MAX_SEARCH_FILE_BYTES:
            self.skipped["too_large"] += 1
            self._texts[path] = None
            return None
        if self._total_bytes + size > MAX_TOTAL_BYTES:
            self._stop("max_total_bytes")
            return None
        text: Optional[str] = None
        try:
            blob = as_dict(
                self.client.get(repo_path(self.owner, self.repo, "git", "blobs", require_sha(entry.get("sha")))).data
            )
            raw = _decode_base64(blob.get("content"))
            self._total_bytes += len(raw)
            text = _decode_text(raw).lstrip("\ufeff")
        except ToolError as exc:
            if exc.code == "unsupported_content":
                self.skipped["binary_or_non_utf8"] += 1
            elif exc.code in _STOP_CODES:
                self._stop(exc.code)
            elif exc.code in ("not_found", "github_bad_response", "invalid_request"):
                self.skipped["fetch_errors"] += 1
            else:
                raise
        self._texts[path] = text
        return text

    def read(self, path: str) -> Optional[str]:
        """Like `text`, but counts the file as analyzed (for non-source files such as configs)."""
        text = self.text(path)
        if text is not None:
            self._scanned.add(path)
        return text

    def _stop(self, limit: str) -> None:
        self.limits.add(limit)
        self._stopped = True

    def parsed(self, path: str) -> Optional[m.ParsedFile]:
        if path in self._parsed:
            return self._parsed[path]
        language = detect_language(path)
        result: Optional[m.ParsedFile] = None
        if language is not None:
            text = self.text(path)
            if text is not None:
                self._scanned.add(path)
                result = (
                    parse_python(path, text) if language == PYTHON else parse_js_family(path, text, language)
                )
                result.lines = [ln.rstrip("\r") for ln in result.lines]
                if result.issue is not None:
                    self.parse_errors.append(
                        ParseErrorInfo(
                            path=path, line=result.issue.line, message=result.issue.message,
                            partial_results=result.issue.partial,
                        )
                    )
        self._parsed[path] = result
        return result

    def scan(self, paths: List[str], max_files: int) -> List[m.ParsedFile]:
        """Parse up to `max_files` of `paths`; record `max_files` if some were left out."""
        results: List[m.ParsedFile] = []
        for path in paths:
            if len(self._scanned) >= max_files and path not in self._parsed:
                self.limits.add("max_files")
                break
            parsed = self.parsed(path)
            if parsed is not None:
                results.append(parsed)
            if self._stopped:
                break  # the stop reason is already recorded in `limits`
        return results

    # -- reporting ------------------------------------------------------------

    def result_fields(self, extra_notes: Optional[List[str]] = None) -> Dict[str, Any]:
        """Keyword arguments for `AnalysisResult` (and subclasses)."""
        notes = list(self.notes) + list(extra_notes or [])
        incomplete = bool(self.limits) or bool(self.parse_errors) or self.skipped["fetch_errors"] > 0
        if incomplete:
            notes.append(
                "Results are INCOMPLETE: the absence of a match does not prove that something does not exist."
            )
        if self.unsupported:
            notes.append(
                f"{len(self.unsupported)} file(s) in languages without a parser were not analyzed "
                "(see unsupported_files). Only Python, JavaScript, TypeScript, JSX and TSX are parsed."
            )
        return dict(
            resolved_commit_sha=self.commit_sha,
            reference=self.reference,
            complete=not incomplete,
            files_scanned=len(self._scanned),
            files_skipped=dict(self.skipped),
            limits_hit=sorted(self.limits),
            parse_errors=self.parse_errors[:50],
            unsupported_files=sorted(self.unsupported)[:MAX_UNSUPPORTED_LISTED],
            unsupported_file_count=len(self.unsupported),
            notes=notes,
        )


__all__ = ["AnalysisSession", "AnalysisResult"]
