"""Shared input/output contracts for the code-analysis tools."""

from typing import Annotated, Dict, List, Optional
import re

from pydantic import AfterValidator, BaseModel, Field

from app.tools.github.common import DirectoryPath, GitRef, RepositoryInput, ReferenceInfo

_SYMBOL_RE = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*(\.[A-Za-z_$][A-Za-z0-9_$]*){0,3}")


def _symbol_name(value: str) -> str:
    if not _SYMBOL_RE.fullmatch(value):
        raise ValueError("symbol name must be an identifier, optionally qualified like 'Class.method'")
    return value


SymbolName = Annotated[str, Field(min_length=1, max_length=200), AfterValidator(_symbol_name)]


class CodeAnalysisInput(RepositoryInput):
    """Common inputs. Identity is never an input; the commit is resolved and pinned."""

    ref: Optional[GitRef] = Field(
        default=None,
        description="Branch, tag or commit SHA. It is resolved once to a commit SHA and every file "
        "read in this analysis comes from that single commit. Defaults to the default branch head.",
    )
    path_prefix: Optional[DirectoryPath] = Field(
        default=None, description="Restrict repository-wide scanning to files below this directory"
    )
    max_files: int = Field(
        default=60, ge=1, le=150, description="Maximum number of source files fetched and parsed"
    )


class ParseErrorInfo(BaseModel):
    path: str
    line: Optional[int] = None
    message: str
    partial_results: bool = False


class AnalysisResult(BaseModel):
    """Fields every analysis result carries so that completeness is never implicit."""

    resolved_commit_sha: str
    reference: ReferenceInfo
    complete: bool
    files_scanned: int
    files_skipped: Dict[str, int]
    limits_hit: List[str]
    parse_errors: List[ParseErrorInfo]
    unsupported_files: List[str]
    unsupported_file_count: int = 0
    notes: List[str]


def excerpt(text: str, limit: int = 200) -> str:
    return text.strip()[:limit]
