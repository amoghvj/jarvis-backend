"""`analyze_dependencies` and `find_related_files`: import-graph based analyses."""

import posixpath
from typing import Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

from app.tools.code import model as m
from app.tools.code.common import AnalysisResult, CodeAnalysisInput, SymbolName
from app.tools.code.imports import PathIndex, ResolvedImport, resolve_import
from app.tools.code.languages import detect_language, is_test_path
from app.tools.code.references import CONFIRMED, _classify_use, compute_binding
from app.tools.code.source import AnalysisSession
from app.tools.code.symbols import name_matches, single_file
from app.tools.errors import ToolError
from app.tools.github.common import FilePath
from app.tools.schemas import ToolConstraints, ToolContext, ToolDefinition

_TIMEOUT = ToolConstraints(timeout_seconds=28.0)
_GRAPH_NOTE = (
    "Static import analysis only: path aliases, tsconfig `paths`, re-export chains through barrel files, "
    "conditional/dynamic imports with computed specifiers and runtime module loading are not resolved."
)


class DependencyEdge(BaseModel):
    source: str
    target: str  # repository path for internal; package or module name otherwise
    target_kind: str  # internal | external | stdlib | builtin | unresolved
    import_kind: str  # import | from | es_import | es_export_from | require | dynamic_import
    specifier: str
    line: int
    names: List[str] = []
    type_only: bool = False
    detail: Optional[str] = None


def _display_specifier(rec: m.ImportRec) -> str:
    return ("." * rec.level) + rec.module


def edges_for(parsed: m.ParsedFile, index: PathIndex) -> List[DependencyEdge]:
    edges: List[DependencyEdge] = []
    for rec in parsed.imports:
        resolved: ResolvedImport = resolve_import(rec, parsed.path, parsed.language, index)
        names = sorted({n.name for n in rec.names if n.name} | {n.local for n in rec.names if n.name is None and n.local})
        type_only = bool(rec.names) and all(n.type_only for n in rec.names)
        base = resolved.base
        edges.append(DependencyEdge(
            source=parsed.path, target=base.target, target_kind=base.kind, import_kind=rec.kind,
            specifier=_display_specifier(rec), line=rec.line, names=names, type_only=type_only, detail=base.detail,
        ))
        for imported, path in sorted(resolved.submodules.items()):
            edges.append(DependencyEdge(
                source=parsed.path, target=path, target_kind="internal", import_kind=rec.kind,
                specifier=f"{_display_specifier(rec)}.{imported}".lstrip("."), line=rec.line,
                names=[imported], detail="imported name is itself a module in the repository",
            ))
    return edges


# --- analyze_dependencies ----------------------------------------------------


class DependencyInput(CodeAnalysisInput):
    path: Optional[FilePath] = Field(default=None, description="Analyze only this file's imports")
    include_dependents: bool = Field(
        default=False, description="With `path`: also scan the repository for files that import it"
    )
    max_edges: int = Field(default=500, ge=1, le=2000)


class ExternalPackage(BaseModel):
    name: str
    kind: str
    import_count: int
    imported_by: List[str]


class DependencyOutput(AnalysisResult):
    path: Optional[str] = None
    files_analyzed: List[str]
    edges: List[DependencyEdge]
    edge_count: int
    internal_adjacency: Dict[str, List[str]]
    reverse_adjacency: Dict[str, List[str]]
    external_packages: List[ExternalPackage]
    unresolved_imports: List[DependencyEdge]


def analyze_dependencies(ctx: ToolContext, inp: DependencyInput) -> DependencyOutput:
    session = AnalysisSession(ctx, inp)
    index = PathIndex(session.all_paths())
    if inp.path:
        target = single_file(session, inp.path)
        parsed_files = [target]
        if inp.include_dependents:
            others = [p for p in session.source_candidates(inp.path_prefix or "") if p != inp.path]
            parsed_files += session.scan(others, inp.max_files)
    else:
        parsed_files = session.scan(session.source_candidates(inp.path_prefix or ""), inp.max_files)

    all_edges: List[DependencyEdge] = []
    for parsed in parsed_files:
        all_edges.extend(edges_for(parsed, index))
    if inp.path and inp.include_dependents:
        all_edges = [e for e in all_edges if e.source == inp.path or e.target == inp.path]
    elif inp.path:
        all_edges = [e for e in all_edges if e.source == inp.path]
    all_edges.sort(key=lambda e: (e.source, e.line, e.target))

    if len(all_edges) > inp.max_edges:
        session.limits.add("max_edges")
    edges = all_edges[: inp.max_edges]  # the graph below is built from the returned edges only

    adjacency: Dict[str, Set[str]] = {}
    reverse: Dict[str, Set[str]] = {}
    external: Dict[Tuple[str, str], Set[str]] = {}
    counts: Dict[Tuple[str, str], int] = {}
    unresolved: List[DependencyEdge] = []
    for e in edges:
        if e.target_kind == "internal":
            if e.target != e.source:
                adjacency.setdefault(e.source, set()).add(e.target)
                reverse.setdefault(e.target, set()).add(e.source)
        elif e.target_kind == "unresolved":
            unresolved.append(e)
        else:
            key = (e.target, e.target_kind)
            external.setdefault(key, set()).add(e.source)
            counts[key] = counts.get(key, 0) + 1
    packages = [
        ExternalPackage(name=name, kind=kind, import_count=counts[(name, kind)], imported_by=sorted(files)[:10])
        for (name, kind), files in sorted(external.items())
    ]
    return DependencyOutput(
        path=inp.path,
        files_analyzed=sorted(p.path for p in parsed_files),
        edges=edges, edge_count=len(all_edges),
        internal_adjacency={k: sorted(v) for k, v in sorted(adjacency.items())},
        reverse_adjacency={k: sorted(v) for k, v in sorted(reverse.items())},
        external_packages=packages, unresolved_imports=unresolved[:100],
        **session.result_fields([
            _GRAPH_NOTE,
            "`external` means the specifier did not resolve to a file in this repository; it is not verified "
            "against any package registry or manifest.",
        ]),
    )


# --- find_related_files ------------------------------------------------------


class RelatedFilesInput(CodeAnalysisInput):
    path: FilePath = Field(description="Source file to find relatives of")
    symbol_name: Optional[SymbolName] = Field(
        default=None, description="Also rank files that reference this symbol defined in `path`"
    )
    max_results: int = Field(default=20, ge=1, le=50)


class Relationship(BaseModel):
    type: str
    strength: str  # strong | moderate | weak
    detail: str
    lines: List[int] = []


class RelatedFile(BaseModel):
    path: str
    score: int
    is_test_file_by_name: bool
    relationships: List[Relationship]
    explanation: str


class RelatedFilesOutput(AnalysisResult):
    path: str
    symbol_name: Optional[str] = None
    related: List[RelatedFile]
    related_count: int


_SCORES = {
    "imports": 100, "imported_by": 90, "re_exports": 85, "re_exported_by": 80,
    "references_symbol": 70, "references_symbol_by_name_only": 30,
    "test_naming_convention": 15, "same_directory": 10,
}


def _stem(path: str) -> str:
    return posixpath.splitext(posixpath.basename(path))[0].lower()


def find_related_files(ctx: ToolContext, inp: RelatedFilesInput) -> RelatedFilesOutput:
    session = AnalysisSession(ctx, inp)
    index = PathIndex(session.all_paths())
    target = single_file(session, inp.path)
    defs: List[m.Symbol] = []
    if inp.symbol_name:
        defs = [s for s in target.symbols if name_matches(s, inp.symbol_name)]
        if not defs:
            raise ToolError("symbol_not_found", "No matching definition was found in that file")
    sym_name = inp.symbol_name.rsplit(".", 1)[-1] if inp.symbol_name else None

    rels: Dict[str, List[Relationship]] = {}

    def add(path: str, rtype: str, strength: str, detail: str, lines: Optional[List[int]] = None) -> None:
        rels.setdefault(path, []).append(Relationship(type=rtype, strength=strength, detail=detail, lines=lines or []))

    for e in edges_for(target, index):
        if e.target_kind == "internal" and e.target != inp.path:
            rtype = "re_exports" if e.import_kind == "es_export_from" else "imports"
            names = f" ({', '.join(e.names[:5])})" if e.names else ""
            add(e.target, rtype, "strong", f"{inp.path} {'re-exports from' if rtype == 're_exports' else 'imports'} "
                                           f"this file{names}", [e.line])

    candidates = [p for p in session.source_candidates(inp.path_prefix or "") if p != inp.path]
    for parsed in session.scan(candidates, inp.max_files):
        for e in edges_for(parsed, index):
            if e.target_kind == "internal" and e.target == inp.path:
                rtype = "re_exported_by" if e.import_kind == "es_export_from" else "imported_by"
                names = f" ({', '.join(e.names[:5])})" if e.names else ""
                add(parsed.path, rtype, "strong",
                    f"this file {'re-exports' if rtype == 're_exported_by' else 'imports'} {inp.path}{names}", [e.line])
        if sym_name:
            binding = compute_binding(parsed, {inp.path}, sym_name, index)
            call_keys = {(c.line, c.name, c.receiver) for c in parsed.calls}
            confirmed: List[int] = []
            by_name: List[int] = []
            for use in parsed.uses:
                result = _classify_use(use, parsed, binding, sym_name, [], call_keys)
                if result is None:
                    continue
                (confirmed if result[1] == CONFIRMED else by_name).append(use.line)
            if confirmed:
                add(parsed.path, "references_symbol", "strong",
                    f"references {sym_name} bound to the definition in {inp.path} through an import",
                    sorted(set(confirmed))[:10])
            elif by_name:
                add(parsed.path, "references_symbol_by_name_only", "moderate",
                    f"uses an identifier named {sym_name}; it is not tied to the definition by an import "
                    "(may be an unrelated symbol with the same name)", sorted(set(by_name))[:10])
        if is_test_path(parsed.path) and _stem(parsed.path).replace("test_", "").replace("_test", "") \
                .replace(".test", "").replace(".spec", "") == _stem(inp.path):
            add(parsed.path, "test_naming_convention", "weak",
                "file name follows the test naming convention for this file (a naming guess, not a dependency)")

    # Directory proximity needs only the tree listing, so it works even when scanning was limited.
    directory = posixpath.dirname(inp.path)
    siblings = [
        p for p in session.all_paths()
        if p != inp.path and posixpath.dirname(p) == directory and detect_language(p) is not None
    ][:8]
    for p in siblings:
        add(p, "same_directory", "weak",
            "lives in the same directory; directory proximity is a weak hint and does NOT show a dependency")

    related: List[RelatedFile] = []
    for path, items in rels.items():
        score = 0
        seen: Set[str] = set()
        for r in items:
            if r.type not in seen:
                seen.add(r.type)
                score += _SCORES[r.type]
        items.sort(key=lambda r: -_SCORES[r.type])
        related.append(RelatedFile(
            path=path, score=score, is_test_file_by_name=is_test_path(path), relationships=items,
            explanation="; ".join(r.detail for r in items)[:600],
        ))
    related.sort(key=lambda r: (-r.score, r.path))
    if len(related) > inp.max_results:
        session.limits.add("max_results")
    return RelatedFilesOutput(
        path=inp.path, symbol_name=inp.symbol_name, related=related[: inp.max_results],
        related_count=len(related),
        **session.result_fields([
            _GRAPH_NOTE,
            "Ranking is a heuristic: import/export relationships are strong evidence; name-only symbol matches "
            "are moderate; directory proximity and test naming are weak hints only.",
        ]),
    )


TOOLS = [
    ToolDefinition(
        name="analyze_dependencies",
        description=(
            "Build the import dependency graph for one file or a bounded set of repository files at a pinned "
            "commit (Python import/from; JS/TS ES imports, re-exports, require, dynamic import with literal "
            "specifiers). Separates internal repository dependencies from external, stdlib/builtin and "
            "unresolved ones. Nothing is installed or executed."
        ),
        input_model=DependencyInput,
        output_model=DependencyOutput,
        handler=analyze_dependencies,
        constraints=_TIMEOUT,
    ),
    ToolDefinition(
        name="find_related_files",
        description=(
            "Rank files related to a source file (and optionally a symbol in it) using import/export "
            "relationships and symbol references; directory proximity and test naming are included only as "
            "clearly-labelled weak hints, each with an explanation."
        ),
        input_model=RelatedFilesInput,
        output_model=RelatedFilesOutput,
        handler=find_related_files,
        constraints=_TIMEOUT,
    ),
]

