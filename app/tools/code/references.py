"""`find_references` and `find_callers`.

Both are *name-based static analyses* over parsed syntax trees. They do not run type
inference, resolve shadowing, follow re-export chains, or model dynamic dispatch, so
every result carries an explicit `resolution` class instead of a claim of certainty:

* ``confirmed_import_binding`` - the file imports this symbol (or the module that
  defines it) from the file that defines it, and the use matches the imported name.
* ``same_class_self_access`` - ``self.x`` / ``this.x`` inside the class defining ``x``.
* ``same_file_name_match`` - same identifier in the file that defines it.
* ``star_import_binding`` - the name may come in through a ``*`` import of the defining file.
* ``name_match_unresolved`` - the identifier matches but nothing ties it to the definition.
* ``textual_heuristic`` - plain text match (only with ``include_textual``); not AST-level.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

from app.tools.code import model as m
from app.tools.code.common import AnalysisResult, CodeAnalysisInput, SymbolName, excerpt
from app.tools.code.imports import PathIndex, resolve_import
from app.tools.code.source import AnalysisSession
from app.tools.code.symbols import SymbolMatch, name_matches, single_file, to_match
from app.tools.errors import ToolError
from app.tools.github.common import FilePath
from app.tools.schemas import ToolConstraints, ToolContext, ToolDefinition

CONFIRMED = "confirmed_import_binding"
SELF_ACCESS = "same_class_self_access"
SAME_FILE = "same_file_name_match"
STAR = "star_import_binding"
UNRESOLVED = "name_match_unresolved"
TEXTUAL = "textual_heuristic"
_RANK = {CONFIRMED: 0, SELF_ACCESS: 1, SAME_FILE: 2, STAR: 3, UNRESOLVED: 4, TEXTUAL: 5}
_SELF_NAMES = ("self", "cls", "this")
_LIMITATIONS = (
    "Name-based static analysis: no type inference, no shadowing resolution, no re-export (barrel file) "
    "following, and no modelling of dynamic dispatch, reflection, getattr/eval or computed property access."
)
_TIMEOUT = ToolConstraints(timeout_seconds=28.0)


def _name_of(wanted: str) -> str:
    return wanted.rsplit(".", 1)[-1]


@dataclass
class Binding:
    local_names: Set[str] = field(default_factory=set)
    star: bool = False
    module_aliases: Set[str] = field(default_factory=set)
    # (line, imported name) -> resolution class, for the import statement's own reference
    import_lines: Dict[Tuple[int, str], str] = field(default_factory=dict)


def compute_binding(parsed: m.ParsedFile, def_paths: Set[str], name: str, index: PathIndex) -> Binding:
    binding = Binding()
    for rec in parsed.imports:
        resolved = resolve_import(rec, parsed.path, parsed.language, index)
        base_internal = resolved.base.kind == "internal" and resolved.base.target in def_paths
        for imported in rec.names:
            if imported.name == name and imported.local and base_internal:
                binding.local_names.add(imported.local)
                binding.import_lines[(rec.line, name)] = CONFIRMED
            elif imported.name == "*" and base_internal:
                if imported.local:  # JS `import * as ns from ...`
                    binding.module_aliases.add(imported.local)
                else:
                    binding.star = True
            elif imported.name is None and base_internal and imported.local:  # Python `import pkg.mod`
                binding.module_aliases.add(imported.local)
            elif imported.name and resolved.submodules.get(imported.name) in def_paths and imported.local:
                binding.module_aliases.add(imported.local)  # `from pkg import mod`
        if rec.kind == "es_export_from" and base_internal:
            for imported in rec.names:
                if imported.name == name:
                    binding.import_lines[(rec.line, name)] = CONFIRMED
    return binding


@dataclass
class Prepared:
    session: AnalysisSession
    parsed_files: List[m.ParsedFile]
    definitions: List[Tuple[m.ParsedFile, m.Symbol]]
    index: PathIndex
    def_paths: Set[str]


def prepare(session: AnalysisSession, inp, kinds: Tuple[str, ...]) -> Prepared:
    name = inp.symbol_name
    first: List[str] = []
    if inp.path:
        parsed = single_file(session, inp.path)
        if not any(name_matches(s, name) and s.kind in kinds for s in parsed.symbols):
            raise ToolError("symbol_not_found", "No matching definition was found in that file")
        first = [inp.path]
    candidates = session.source_candidates(inp.path_prefix or "")
    ordered = first + [p for p in candidates if p not in first]
    parsed_files = session.scan(ordered, inp.max_files)
    definitions = [
        (p, s) for p in parsed_files for s in p.symbols
        if name_matches(s, name) and s.kind in kinds and (not inp.path or p.path == inp.path)
    ]
    return Prepared(session, parsed_files, definitions, PathIndex(session.all_paths()), {p.path for p, _ in definitions})


# --- find_references ---------------------------------------------------------


class FindReferencesInput(CodeAnalysisInput):
    symbol_name: SymbolName = Field(description="Symbol name, or 'Class.method'")
    path: Optional[FilePath] = Field(
        default=None, description="File that defines the symbol; disambiguates duplicate names"
    )
    include_textual: bool = Field(
        default=False, description="Also report plain-text matches on lines with no AST reference (heuristic)"
    )
    max_results: int = Field(default=50, ge=1, le=200)


class ReferenceMatch(BaseModel):
    path: str
    line: int
    excerpt: str
    category: str  # import | export | call | method_call | identifier | attribute_access | assignment | textual
    match_type: str  # "ast" | "textual"
    resolution: str
    enclosing_symbol: Optional[str] = None


class FindReferencesOutput(AnalysisResult):
    symbol_name: str
    definitions: List[SymbolMatch]
    definition_ambiguous: bool
    references: List[ReferenceMatch]
    reference_count: int
    counts_by_resolution: Dict[str, int]


def _classify_use(
    use: m.Use, parsed: m.ParsedFile, binding: Binding, name: str,
    defined_here: List[m.Symbol], call_keys: Set[Tuple[int, str, Optional[str]]],
) -> Optional[Tuple[str, str]]:
    """Return (category, resolution) or None if this use is unrelated to the symbol."""
    if use.kind == "import":
        if use.name != name:
            return None
        resolution = binding.import_lines.get((use.line, name), UNRESOLVED)
        return "import", resolution
    if use.kind == "export":
        if use.name != name:
            return None
        return "export", (SAME_FILE if defined_here else UNRESOLVED)
    if use.kind == "name":
        if use.name == name or use.name in binding.local_names:
            if use.ctx == "store" and any(s.def_line == use.line for s in defined_here):
                return None  # the definition itself
            if use.name in binding.local_names:
                resolution = CONFIRMED
            elif defined_here:
                resolution = SAME_FILE
            elif binding.star:
                resolution = STAR
            else:
                resolution = UNRESOLVED
            if (use.line, use.name, None) in call_keys:
                return "call", resolution
            return ("assignment" if use.ctx == "store" else "identifier"), resolution
        return None
    if use.kind == "attribute":
        if use.name != name:
            return None
        if use.receiver in binding.module_aliases:
            resolution = CONFIRMED
        elif use.receiver in _SELF_NAMES and any(
            s.kind == m.METHOD and s.parent is not None and use.enclosing is not None
            and use.enclosing.rsplit(".", 1)[0] == s.parent
            for s in defined_here
        ):
            resolution = SELF_ACCESS
        else:
            resolution = UNRESOLVED
        if (use.line, use.name, use.receiver) in call_keys:
            return "method_call", resolution
        return "attribute_access", resolution
    return None


def find_references(ctx: ToolContext, inp: FindReferencesInput) -> FindReferencesOutput:
    session = AnalysisSession(ctx, inp)
    prep = prepare(session, inp, m.SYMBOL_KINDS)
    name = _name_of(inp.symbol_name)

    found: List[ReferenceMatch] = []
    for parsed in prep.parsed_files:
        binding = compute_binding(parsed, prep.def_paths, name, prep.index)
        defined_here = [s for p, s in prep.definitions if p.path == parsed.path]
        call_keys = {(c.line, c.name, c.receiver) for c in parsed.calls}
        seen_lines: Set[int] = set()
        for use in parsed.uses:
            classified = _classify_use(use, parsed, binding, name, defined_here, call_keys)
            if classified is None:
                continue
            category, resolution = classified
            seen_lines.add(use.line)
            found.append(ReferenceMatch(
                path=parsed.path, line=use.line, excerpt=excerpt(parsed.line_text(use.line)),
                category=category, match_type="ast", resolution=resolution,
                enclosing_symbol=use.enclosing,
            ))
        if inp.include_textual:
            pattern = re.compile(r"(?<![\w$])" + re.escape(name) + r"(?![\w$])")
            def_lines = {s.def_line for s in defined_here}
            for number, text in enumerate(parsed.lines, start=1):
                if number in seen_lines or number in def_lines or not pattern.search(text):
                    continue
                found.append(ReferenceMatch(
                    path=parsed.path, line=number, excerpt=excerpt(text), category="textual",
                    match_type="textual", resolution=TEXTUAL))
    # Collapse exact duplicates (e.g. attribute + name uses on one line) deterministically.
    unique = {(r.path, r.line, r.category, r.resolution): r for r in found}
    ordered = sorted(unique.values(), key=lambda r: (_RANK[r.resolution], r.path, r.line, r.category))
    counts: Dict[str, int] = {}
    for r in ordered:
        counts[r.resolution] = counts.get(r.resolution, 0) + 1
    if len(ordered) > inp.max_results:
        session.limits.add("max_results")
    extra = [_LIMITATIONS]
    if not prep.definitions:
        extra.append("No definition of this symbol was found in the analyzed files; all matches are unresolved.")
    if inp.include_textual:
        extra.append("Matches with match_type 'textual' are plain text (strings, comments, docs) - not AST references.")
    return FindReferencesOutput(
        symbol_name=inp.symbol_name,
        definitions=[to_match(s, p.language) for p, s in prep.definitions[:25]],
        definition_ambiguous=len(prep.definitions) > 1,
        references=ordered[: inp.max_results], reference_count=len(ordered), counts_by_resolution=counts,
        **session.result_fields(extra),
    )


# --- find_callers ------------------------------------------------------------


class FindCallersInput(CodeAnalysisInput):
    symbol_name: SymbolName = Field(description="Function, method or class name, or 'Class.method'")
    path: Optional[FilePath] = Field(default=None, description="File that defines the target")
    include_unresolved: bool = Field(
        default=True, description="Include calls whose target is only matched by name (flagged as such)"
    )
    max_results: int = Field(default=50, ge=1, le=200)


class CallerMatch(BaseModel):
    caller_symbol: str  # qualified name, or "<module>" for top-level code
    caller_type: Optional[str] = None
    caller_path: str
    caller_start_line: Optional[int] = None
    caller_end_line: Optional[int] = None
    call_line: int
    call_kind: str  # function_call | method_call | constructor
    target: str
    call_excerpt: str
    resolution: str


class FindCallersOutput(AnalysisResult):
    symbol_name: str
    targets: List[SymbolMatch]
    target_ambiguous: bool
    callers: List[CallerMatch]
    caller_count: int
    counts_by_resolution: Dict[str, int]


def find_callers(ctx: ToolContext, inp: FindCallersInput) -> FindCallersOutput:
    session = AnalysisSession(ctx, inp)
    prep = prepare(session, inp, m.CALLABLE_KINDS)
    name = _name_of(inp.symbol_name)

    found: List[CallerMatch] = []
    for parsed in prep.parsed_files:
        binding = compute_binding(parsed, prep.def_paths, name, prep.index)
        defined_here = [s for p, s in prep.definitions if p.path == parsed.path]
        for call in parsed.calls:
            if call.name != name and not (call.receiver is None and call.name in binding.local_names):
                continue
            if call.receiver is None:
                if call.name in binding.local_names:
                    resolution = CONFIRMED
                elif call.name == name and defined_here:
                    resolution = SAME_FILE
                elif call.name == name and binding.star:
                    resolution = STAR
                else:
                    resolution = UNRESOLVED
                kind = "constructor" if call.construct else "function_call"
            else:
                if call.name != name:
                    continue
                kind = "constructor" if call.construct else "method_call"
                if call.receiver in binding.module_aliases:
                    resolution = CONFIRMED
                elif call.receiver in _SELF_NAMES and any(
                    s.kind == m.METHOD and call.enclosing is not None
                    and call.enclosing.rsplit(".", 1)[0] == s.parent
                    for s in defined_here
                ):
                    resolution = SELF_ACCESS
                else:
                    resolution = UNRESOLVED
            if resolution == UNRESOLVED and not inp.include_unresolved:
                continue
            caller = parsed.symbol_at(call.enclosing)
            found.append(CallerMatch(
                caller_symbol=caller.qualified_name if caller else "<module>",
                caller_type=caller.kind if caller else None, caller_path=parsed.path,
                caller_start_line=caller.start_line if caller else None,
                caller_end_line=caller.end_line if caller else None,
                call_line=call.line, call_kind=kind, target=inp.symbol_name,
                call_excerpt=excerpt(parsed.line_text(call.line)), resolution=resolution,
            ))
    unique = {(c.caller_path, c.call_line, c.caller_symbol, c.call_kind): c for c in found}
    ordered = sorted(unique.values(), key=lambda c: (_RANK[c.resolution], c.caller_path, c.call_line))
    counts: Dict[str, int] = {}
    for c in ordered:
        counts[c.resolution] = counts.get(c.resolution, 0) + 1
    if len(ordered) > inp.max_results:
        session.limits.add("max_results")
    extra = [
        _LIMITATIONS,
        "Only direct, statically visible calls are reported. Calls through callbacks, higher-order functions, "
        "decorators, dependency injection, dynamic dispatch (subclass overrides, duck typing) or reflection are "
        "NOT detected; passing a function as a value is not a call (use find_references for that).",
    ]
    if not prep.definitions:
        extra.append("No definition of this target was found in the analyzed files; all matches are unresolved.")
    return FindCallersOutput(
        symbol_name=inp.symbol_name, targets=[to_match(s, p.language) for p, s in prep.definitions[:25]],
        target_ambiguous=len(prep.definitions) > 1, callers=ordered[: inp.max_results],
        caller_count=len(ordered), counts_by_resolution=counts, **session.result_fields(extra),
    )


TOOLS = [
    ToolDefinition(
        name="find_references",
        description=(
            "Find references to a symbol (identifiers, imports, calls, attribute accesses) across a repository "
            "at one pinned commit. Each match states whether it is bound to the definition through an import "
            "or only matches by name; optional plain-text matches are labelled textual. Not semantic resolution."
        ),
        input_model=FindReferencesInput,
        output_model=FindReferencesOutput,
        handler=find_references,
        constraints=_TIMEOUT,
    ),
    ToolDefinition(
        name="find_callers",
        description=(
            "Find direct, statically visible call sites of a function, method or class and the function that "
            "contains each call, with a resolution class per result. Does not detect dynamic or indirect calls."
        ),
        input_model=FindCallersInput,
        output_model=FindCallersOutput,
        handler=find_callers,
        constraints=_TIMEOUT,
    ),
]
