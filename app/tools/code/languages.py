"""Language detection from file paths. Only extensions with a real parser are 'supported'."""

import posixpath
from typing import Optional

PYTHON = "python"
JAVASCRIPT = "javascript"  # also covers JSX: the JavaScript grammar parses JSX
TYPESCRIPT = "typescript"
TSX = "tsx"

_EXTENSIONS = {
    ".py": PYTHON,
    ".js": JAVASCRIPT,
    ".jsx": JAVASCRIPT,
    ".mjs": JAVASCRIPT,
    ".cjs": JAVASCRIPT,
    ".ts": TYPESCRIPT,
    ".mts": TYPESCRIPT,
    ".cts": TYPESCRIPT,
    ".tsx": TSX,
}

# Source-like files that exist in many repositories but have no parser here. They are
# reported as unsupported (and make a repository-wide result incomplete) rather than
# being scanned with regular expressions.
UNSUPPORTED_SOURCE_EXTENSIONS = (
    ".go", ".java", ".kt", ".kts", ".rb", ".rs", ".c", ".h", ".cc", ".cpp", ".cxx",
    ".hpp", ".cs", ".php", ".swift", ".scala", ".m", ".mm", ".vue", ".svelte", ".dart",
    ".ex", ".exs", ".lua", ".pl", ".r", ".sh", ".bash", ".ps1",
)

MINIFIED_SUFFIXES = (".min.js", ".min.mjs", ".min.cjs", ".bundle.js")


def detect_language(path: str) -> Optional[str]:
    lower = path.lower()
    if lower.endswith(MINIFIED_SUFFIXES):
        return None
    return _EXTENSIONS.get(posixpath.splitext(lower)[1])


def is_unsupported_source(path: str) -> bool:
    return path.lower().endswith(UNSUPPORTED_SOURCE_EXTENSIONS)


def is_js_family(language: Optional[str]) -> bool:
    return language in (JAVASCRIPT, TYPESCRIPT, TSX)


_TEST_DIRS = {"test", "tests", "__tests__", "spec", "specs", "e2e"}


def is_test_path(path: str) -> bool:
    """Naming-convention test: never proof that a file contains tests."""
    lower = path.lower()
    parts = lower.split("/")
    base = parts[-1]
    language = detect_language(path)
    if language == PYTHON:
        return base.startswith("test_") or base.endswith("_test.py") or (
            base != "conftest.py" and bool(_TEST_DIRS.intersection(parts[:-1])) and base != "__init__.py"
        )
    if is_js_family(language):
        stem_parts = base.split(".")
        return (
            "test" in stem_parts[1:-1] or "spec" in stem_parts[1:-1]
            or bool({"__tests__"}.intersection(parts[:-1]))
            or bool(_TEST_DIRS.intersection(parts[:-1]))
        )
    return False
