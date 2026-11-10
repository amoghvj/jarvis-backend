"""Language-neutral, normalized representation of one parsed source file.

Both parsers (Python `ast`, tree-sitter for JS/TS) emit these records, so every
analysis tool is written once against this model instead of once per language.
All line numbers are 1-based and refer to the file at the resolved commit.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# Symbol kinds
FUNCTION, METHOD, CLASS, VARIABLE, CONSTANT = "function", "method", "class", "variable", "constant"
INTERFACE, TYPE, ENUM = "interface", "type", "enum"
SYMBOL_KINDS = (FUNCTION, METHOD, CLASS, VARIABLE, CONSTANT, INTERFACE, TYPE, ENUM)
CALLABLE_KINDS = (FUNCTION, METHOD, CLASS)


@dataclass
class Symbol:
    name: str
    qualified_name: str
    kind: str
    path: str
    start_line: int  # first line including decorators
    end_line: int
    def_line: int  # line of the `def` / `class` / declaration keyword itself
    parent: Optional[str] = None  # qualified name of the enclosing class/function; None = module
    parent_kind: str = "module"  # "module" | "class" | "function"
    signature: Optional[str] = None
    decorators: List[str] = field(default_factory=list)
    bases: Tuple[str, ...] = ()
    exported: bool = False
    nested: bool = False  # defined inside a function body
    is_async: bool = False


@dataclass
class Use:
    """An identifier occurrence. Scope/shadowing is NOT resolved."""

    name: str
    line: int
    kind: str  # "name" | "attribute" | "import"
    ctx: str = "load"  # "load" | "store"
    receiver: Optional[str] = None  # for attributes: source text of the object, truncated
    enclosing: Optional[str] = None  # qualified name of the enclosing function, if any


@dataclass
class Call:
    name: str  # last segment of the callee
    line: int
    receiver: Optional[str] = None  # None => bare call `foo()`
    enclosing: Optional[str] = None
    construct: bool = False  # `new Foo()`


@dataclass
class ImportedName:
    name: Optional[str]  # name in the target module; None = the module itself; "*" = star/namespace
    local: Optional[str]  # name bound in this file (None for side-effect imports)
    type_only: bool = False


@dataclass
class ImportRec:
    kind: str  # "import" | "from" | "es_import" | "es_export_from" | "require" | "dynamic_import"
    module: str  # as written, without leading dots for Python
    line: int
    level: int = 0  # Python relative-import depth
    names: List[ImportedName] = field(default_factory=list)


@dataclass
class Assertion:
    line: int
    kind: str  # "assert" | "assertX" | "expect" | "raises"
    enclosing: Optional[str] = None


@dataclass
class TestCase:
    name: str
    qualified_name: str
    kind: str  # "test" | "suite"
    start_line: int
    end_line: int
    skipped: bool = False


@dataclass
class ParseIssue:
    line: Optional[int]
    message: str
    partial: bool = False  # True: tree-sitter recovered and partial results exist


@dataclass
class ParsedFile:
    path: str
    language: str
    lines: List[str]
    symbols: List[Symbol] = field(default_factory=list)
    uses: List[Use] = field(default_factory=list)
    calls: List[Call] = field(default_factory=list)
    imports: List[ImportRec] = field(default_factory=list)
    assertions: List[Assertion] = field(default_factory=list)
    test_cases: List[TestCase] = field(default_factory=list)  # JS describe/it/test
    env_refs: List[Tuple[str, int]] = field(default_factory=list)
    issue: Optional[ParseIssue] = None

    def line_text(self, line: int, limit: int = 200) -> str:
        if 1 <= line <= len(self.lines):
            return self.lines[line - 1].strip()[:limit]
        return ""

    def symbol_at(self, qualified_name: Optional[str]) -> Optional[Symbol]:
        if qualified_name is None:
            return None
        return next((s for s in self.symbols if s.qualified_name == qualified_name), None)

