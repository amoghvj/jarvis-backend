"""`find_symbol` and `get_function_context`."""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from app.tools.code import model as m
from app.tools.code.common import AnalysisResult, CodeAnalysisInput, SymbolName
from app.tools.code.languages import detect_language, is_unsupported_source
from app.tools.code.source import AnalysisSession
from app.tools.errors import ToolError
from app.tools.github.common import FilePath
from app.tools.schemas import ToolConstraints, ToolContext, ToolDefinition

SymbolType = Literal["function", "method", "class", "variable", "constant", "interface", "type", "enum"]
MAX_CONTEXT_BYTES = 100_000
_ANALYSIS_TIMEOUT = ToolConstraints(timeout_seconds=28.0)


class SymbolMatch(BaseModel):
    name: str
    qualified_name: str
    symbol_type: str
    path: str
    start_line: int
    end_line: int
    parent: Optional[str] = None
    parent_type: str
    signature: Optional[str] = None
    decorators: List[str] = []
    exported: bool = False
    nested: bool = False
    language: str


def to_match(sym: m.Symbol, language: str) -> SymbolMatch:
    return SymbolMatch(
        name=sym.name, qualified_name=sym.qualified_name, symbol_type=sym.kind, path=sym.path,
        start_line=sym.start_line, end_line=sym.end_line, parent=sym.parent, parent_type=sym.parent_kind,
        signature=sym.signature, decorators=sym.decorators, exported=sym.exported, nested=sym.nested,
        language=language,
    )


def name_matches(sym: m.Symbol, wanted: str) -> bool:
    if "." in wanted:
        return sym.qualified_name == wanted or sym.qualified_name.endswith("." + wanted)
    return sym.name == wanted


def single_file(session: AnalysisSession, path: str) -> m.ParsedFile:
    """Parse one named file, with precise structured errors for the common failures."""
    if not session.has_path(path):
        raise ToolError("path_not_found", "That file does not exist at the resolved commit")
    if detect_language(path) is None:
        code = "unsupported_language"
        hint = "only Python, JavaScript, TypeScript, JSX and TSX can be analyzed"
        if not is_unsupported_source(path):
            hint = "this file type is not recognised as analyzable source; " + hint
        raise ToolError(code, f"No parser is available for this file: {hint}")
    parsed = session.parsed(path)
    if parsed is None:
        raise ToolError(
            "file_unavailable",
            "The file could not be read (too large, binary or non-UTF-8, or a GitHub error)",
        )
    return parsed


# --- find_symbol -------------------------------------------------------------


class FindSymbolInput(CodeAnalysisInput):
    symbol_name: SymbolName = Field(description="Exact symbol name, or 'Class.method'")
    path: Optional[FilePath] = Field(default=None, description="Only look in this file")
    symbol_type: Optional[SymbolType] = None
    max_results: int = Field(default=25, ge=1, le=100)

class FindSymbolOutput(AnalysisResult):
    symbol_name: str
    matches: List[SymbolMatch]
    match_count: int
    ambiguous: bool
    ambiguity_note: Optional[str] = None


def find_symbol(ctx: ToolContext, inp: FindSymbolInput) -> FindSymbolOutput:
    session = AnalysisSession(ctx, inp)
    if inp.path:
        parsed_files = [single_file(session, inp.path)]
    else:
        parsed_files = session.scan(session.source_candidates(inp.path_prefix or ""), inp.max_files)

    found: List[SymbolMatch] = []
    for parsed in parsed_files:
        for sym in parsed.symbols:
            if name_matches(sym, inp.symbol_name) and (inp.symbol_type is None or sym.kind == inp.symbol_type):
                found.append(to_match(sym, parsed.language))
    found.sort(key=lambda s: (s.path, s.start_line))
    if len(found) > inp.max_results:
        session.limits.add("max_results")
    matches = found[: inp.max_results]

    ambiguous = len(found) > 1
    note = None
    if ambiguous:
        files = len({s.path for s in found})
        note = (
            f"{len(found)} definitions named '{inp.symbol_name}' were found in {files} file(s). "
            "They are distinct definitions; none is preferred. Narrow with path, symbol_type or a qualified name."
        )
    return FindSymbolOutput(
        symbol_name=inp.symbol_name, matches=matches, match_count=len(found),
        ambiguous=ambiguous, ambiguity_note=note, **session.result_fields(),
    )


# --- get_function_context ----------------------------------------------------


class FunctionContextInput(CodeAnalysisInput):
    symbol_name: SymbolName = Field(description="Function or method name, or 'Class.method'")
    path: Optional[FilePath] = Field(default=None, description="File containing the function (recommended)")
    max_source_bytes: int = Field(default=20_000, ge=200, le=MAX_CONTEXT_BYTES)


class ContainingClass(BaseModel):
    name: str
    signature: Optional[str] = None
    start_line: int
    end_line: int


class ContextImport(BaseModel):
    line: int
    text: str


class RelatedDefinition(BaseModel):
    name: str
    symbol_type: str
    relation: str  # "module_level" | "sibling_method"
    start_line: int
    end_line: int
    signature: Optional[str] = None


class FunctionContext(BaseModel):
    name: str
    qualified_name: str
    symbol_type: str
    signature: Optional[str] = None
    path: str
    language: str
    start_line: int
    end_line: int
    source: str
    source_bytes: int
    decorators: List[str]
    containing_class: Optional[ContainingClass] = None
    imports_used: List[ContextImport]
    related_definitions: List[RelatedDefinition]


class FunctionContextOutput(AnalysisResult):
    symbol_name: str
    resolved: bool
    ambiguous: bool
    candidates: List[SymbolMatch]
    function: Optional[FunctionContext] = None


def _build_context(parsed: m.ParsedFile, sym: m.Symbol, inp: FunctionContextInput) -> FunctionContext:
    source = "\n".join(parsed.lines[sym.start_line - 1 : sym.end_line])
    size = len(source.encode("utf-8"))
    if size > inp.max_source_bytes:
        raise ToolError(
            "source_too_large",
            f"{sym.qualified_name} is {size} bytes; the limit is {inp.max_source_bytes} "
            f"(maximum {MAX_CONTEXT_BYTES}). It is never truncated.",
        )
    used = {u.name for u in parsed.uses if sym.start_line <= u.line <= sym.end_line and u.kind == "name"}
    sibling_calls = {
        u.name for u in parsed.uses
        if sym.start_line <= u.line <= sym.end_line and u.kind == "attribute" and u.receiver in ("self", "cls", "this")
    }

    imports_used: List[ContextImport] = []
    for rec in parsed.imports:
        locals_ = {n.local for n in rec.names if n.local}
        if locals_ & used:
            imports_used.append(ContextImport(line=rec.line, text=parsed.line_text(rec.line, 240)))

    related: List[RelatedDefinition] = []
    for other in parsed.symbols:
        if other is sym or other.qualified_name == sym.qualified_name:
            continue
        relation = None
        if other.parent is None and other.name in used:
            relation = "module_level"
        elif sym.parent_kind == "class" and other.parent == sym.parent and other.name in sibling_calls:
            relation = "sibling_method"
        if relation:
            related.append(RelatedDefinition(
                name=other.name, symbol_type=other.kind, relation=relation,
                start_line=other.start_line, end_line=other.end_line, signature=other.signature))
    containing = None
    if sym.parent_kind == "class":
        cls = parsed.symbol_at(sym.parent)
        if cls is not None:
            containing = ContainingClass(
                name=cls.name, signature=cls.signature, start_line=cls.start_line, end_line=cls.end_line)
    return FunctionContext(
        name=sym.name, qualified_name=sym.qualified_name, symbol_type=sym.kind, signature=sym.signature,
        path=parsed.path, language=parsed.language, start_line=sym.start_line, end_line=sym.end_line,
        source=source, source_bytes=size, decorators=sym.decorators, containing_class=containing,
        imports_used=imports_used[:30], related_definitions=related[:15],
    )


def get_function_context(ctx: ToolContext, inp: FunctionContextInput) -> FunctionContextOutput:
    session = AnalysisSession(ctx, inp)
    if inp.path:
        parsed_files = [single_file(session, inp.path)]
    else:
        parsed_files = session.scan(session.source_candidates(inp.path_prefix or ""), inp.max_files)

    hits = [
        (parsed, sym)
        for parsed in parsed_files
        for sym in parsed.symbols
        if sym.kind in (m.FUNCTION, m.METHOD) and name_matches(sym, inp.symbol_name)
    ]
    hits.sort(key=lambda h: (h[0].path, h[1].start_line))
    candidates = [to_match(sym, parsed.language) for parsed, sym in hits[:25]]
    extra: List[str] = []
    function = None
    if len(hits) == 1:
        function = _build_context(hits[0][0], hits[0][1], inp)
    elif len(hits) > 1:
        extra.append(
            f"{len(hits)} functions named '{inp.symbol_name}' were found; no context was returned. "
            "Repeat the request with path and/or a qualified name such as 'Class.method'."
        )
    else:
        extra.append("No function or method with that name was found in the analyzed files.")
    return FunctionContextOutput(
        symbol_name=inp.symbol_name, resolved=function is not None, ambiguous=len(hits) > 1,
        candidates=candidates, function=function, **session.result_fields(extra),
    )


TOOLS = [
    ToolDefinition(
        name="find_symbol",
        description=(
            "Locate definitions of functions, classes, methods, variables/constants (and TS interfaces, types, "
            "enums) by exact name in Python, JavaScript, TypeScript, JSX and TSX at one pinned commit. Multiple "
            "definitions are all returned and flagged ambiguous. Parser-based; bounded; reports completeness."
        ),
        input_model=FindSymbolInput,
        output_model=FindSymbolOutput,
        handler=find_symbol,
        constraints=_ANALYSIS_TIMEOUT,
    ),
    ToolDefinition(
        name="get_function_context",
        description=(
            "Return the full source of one function or method (never truncated; refused if over the size limit) "
            "with signature, decorators, containing class, the imports it uses and related same-file definitions."
        ),
        input_model=FunctionContextInput,
        output_model=FunctionContextOutput,
        handler=get_function_context,
        constraints=_ANALYSIS_TIMEOUT,
    ),
]
