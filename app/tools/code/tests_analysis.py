"""`inspect_tests`: statically find tests relevant to a file or symbol (tests are never run).

Evidence levels are kept strictly apart, strongest first:

* ``invokes_target``        - the test calls the imported symbol (or a module attribute) directly.
* ``calls_method_by_name``  - a method with that name is called on some object in a file that imports
                              the target module; the receiver's type is NOT resolved.
* ``references_target``     - the imported symbol is mentioned but not called.
* ``imports_target_module_only`` - the test file imports the module but nothing ties a test to the symbol.
* ``name_match_only``       - the identifier appears without any import linking it to the target.
* ``naming_convention_only``- only the file name suggests a relationship.
"""

import posixpath
from typing import Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

from app.tools.code import model as m
from app.tools.code.common import AnalysisResult, CodeAnalysisInput, SymbolName, excerpt
from app.tools.code.imports import PathIndex, resolve_import
from app.tools.code.languages import PYTHON, is_js_family, is_test_path
from app.tools.code.source import AnalysisSession
from app.tools.code.symbols import SymbolMatch, name_matches, single_file, to_match
from app.tools.errors import ToolError
from app.tools.github.common import FilePath
from app.tools.schemas import ToolConstraints, ToolContext, ToolDefinition

_TIMEOUT = ToolConstraints(timeout_seconds=28.0)
MAX_CASES_PER_FILE = 40
MAX_ASSERTIONS_PER_CASE = 5
_RELATION_RANK = {
    "invokes_target": 0, "calls_method_by_name": 1, "references_target": 2,
    "imports_target_module_only": 3, "name_match_only": 4, "naming_convention_only": 5,
}
_DIRECT = ("direct_call", "module_attribute_call")


class InspectTestsInput(CodeAnalysisInput):
    path: FilePath = Field(description="Source file under test")
    symbol_name: Optional[SymbolName] = Field(
        default=None, description="Function, method or class under test (defined in `path`)"
    )
    max_results: int = Field(default=20, ge=1, le=50, description="Maximum test files returned")


class AssertionInfo(BaseModel):
    line: int
    kind: str
    excerpt: str


class TestCaseInfo(BaseModel):
    __test__ = False  # not a pytest class

    name: str
    qualified_name: str
    start_line: int
    end_line: int
    skipped: bool = False
    invokes_target: bool
    invocation_kind: Optional[str] = None  # direct_call | module_attribute_call | method_name_match | name_match_only
    target_call_lines: List[int] = []
    references_target: bool = False
    assertions: List[AssertionInfo] = []


class RelevantTestFile(BaseModel):
    path: str
    framework: str
    framework_evidence: str
    relation: str
    imports_target: bool
    imported_names: List[str]
    test_case_count: int
    test_cases: List[TestCaseInfo]
    notes: List[str] = []


class InspectTestsOutput(AnalysisResult):
    path: str
    symbol_name: Optional[str] = None
    target_definitions: List[SymbolMatch]
    test_files: List[RelevantTestFile]
    test_file_count: int


# --- helpers -----------------------------------------------------------------


def _bindings(
    parsed: m.ParsedFile, target: str, index: PathIndex, only_name: Optional[str]
) -> Tuple[Dict[str, str], Set[str], bool, List[str]]:
    """(local -> imported name, module aliases, imports target at all?, imported names)."""
    locals_map: Dict[str, str] = {}
    aliases: Set[str] = set()
    imports = False
    names: Set[str] = set()
    for rec in parsed.imports:
        resolved = resolve_import(rec, parsed.path, parsed.language, index)
        base_internal = resolved.base.kind == "internal" and resolved.base.target == target
        for n in rec.names:
            if n.name and n.name != "*" and resolved.submodules.get(n.name) == target and n.local:
                aliases.add(n.local)
                imports = True
            elif base_internal:
                imports = True
                if n.name is None or n.name == "*":
                    if n.local:
                        aliases.add(n.local)
                elif n.local and (only_name is None or n.name == only_name):
                    locals_map[n.local] = n.name
                    names.add(n.name)
        if base_internal and not rec.names:
            imports = True  # side-effect import
    return locals_map, aliases, imports, sorted(names)


def _classify_call(
    call: m.Call, locals_map: Dict[str, str], aliases: Set[str], sym_name: Optional[str],
    is_method: bool, imports_target: bool,
) -> Optional[str]:
    if call.receiver is None and call.name in locals_map and (sym_name is None or locals_map[call.name] == sym_name):
        return "direct_call"
    if call.receiver in aliases and (sym_name is None or call.name == sym_name):
        return "module_attribute_call"
    if sym_name and call.name == sym_name and call.receiver is not None:
        return "method_name_match" if (is_method and imports_target) else None
    if sym_name and call.name == sym_name and call.receiver is None and not imports_target and call.name not in locals_map:
        return "name_match_only"
    return None


def _python_cases(parsed: m.ParsedFile) -> List[Tuple[str, str, int, int, bool, Optional[str]]]:
    """(name, qualified, start, end, skipped, enclosing-prefix) for test functions/methods."""
    out = []
    for s in parsed.symbols:
        if s.kind not in (m.FUNCTION, m.METHOD) or s.nested or not s.name.startswith("test"):
            continue
        if s.parent_kind == "class":
            cls = parsed.symbol_at(s.parent)
            ok = cls is not None and (
                cls.name.startswith("Test") or any(b.endswith("TestCase") for b in cls.bases)
            )
            if not ok:
                continue
        elif s.parent_kind != "module":
            continue
        skipped = any("skip" in d or "xfail" in d for d in s.decorators)
        out.append((s.name, s.qualified_name, s.start_line, s.end_line, skipped, s.qualified_name))
    return out


def _framework(parsed: m.ParsedFile) -> Tuple[str, str]:
    modules = {r.module.split(".")[0] if parsed.language == PYTHON else r.module for r in parsed.imports}
    if parsed.language == PYTHON:
        has_unittest = "unittest" in modules or any(
            b.endswith("TestCase") for s in parsed.symbols if s.kind == m.CLASS for b in s.bases
        )
        has_pytest = "pytest" in modules
        if has_pytest and has_unittest:
            return "pytest+unittest", "imports pytest and unittest"
        if has_pytest:
            return "pytest", "imports pytest"
        if has_unittest:
            return "unittest", "imports unittest / subclasses TestCase"
        return "pytest", "inferred from test_* naming only (no framework import found)"
    for fw, evidence in (("vitest", "vitest"), ("jest", "@jest/globals"), ("jest", "jest"), ("mocha", "mocha"),
                         ("playwright", "@playwright/test")):
        if evidence in modules:
            return fw, f"imports {evidence}"
    receivers = {c.receiver for c in parsed.calls if c.receiver}
    if "vi" in receivers:
        return "vitest", "uses the `vi` mock object"
    if "jest" in receivers:
        return "jest", "uses the `jest` object"
    return "unknown", "describe/it/test globals are shared by Jest, Vitest and Mocha; no framework import found"


def inspect_tests(ctx: ToolContext, inp: InspectTestsInput) -> InspectTestsOutput:
    session = AnalysisSession(ctx, inp)
    index = PathIndex(session.all_paths())
    target = single_file(session, inp.path)
    defs: List[m.Symbol] = []
    sym_name = None
    if inp.symbol_name:
        defs = [s for s in target.symbols if name_matches(s, inp.symbol_name)]
        if not defs:
            raise ToolError("symbol_not_found", "No matching definition was found in that file")
        sym_name = inp.symbol_name.rsplit(".", 1)[-1]
    is_method = any(s.kind == m.METHOD for s in defs)
    target_stem = posixpath.splitext(posixpath.basename(inp.path))[0].lower()

    candidates = [p for p in session.source_candidates(inp.path_prefix or "") if is_test_path(p) and p != inp.path]
    results: List[Tuple[int, RelevantTestFile]] = []
    for parsed in session.scan(candidates, inp.max_files):
        locals_map, aliases, imports, imported = _bindings(parsed, inp.path, index, sym_name)
        if parsed.language == PYTHON:
            cases = [(n, q, s, e, sk, q) for n, q, s, e, sk, q in _python_cases(parsed)]
        elif is_js_family(parsed.language):
            cases = [(t.name, t.qualified_name, t.start_line, t.end_line, t.skipped, None)
                     for t in parsed.test_cases if t.kind == "test"]
        else:
            cases = []

        def within(line: int, enclosing: Optional[str], case) -> bool:
            _, qual, start, end, _, prefix = case
            if prefix is not None:
                return enclosing is not None and (enclosing == qual or enclosing.startswith(qual + "."))
            return start <= line <= end

        infos: List[TestCaseInfo] = []
        file_kinds: Set[str] = set()
        for case in cases:
            kinds: List[Tuple[int, str]] = []
            for call in parsed.calls:
                if not within(call.line, call.enclosing, case):
                    continue
                kind = _classify_call(call, locals_map, aliases, sym_name, is_method, imports)
                if kind:
                    kinds.append((call.line, kind))
            refs = any(
                u.kind == "name" and u.name in locals_map and within(u.line, u.enclosing, case)
                and (u.line, u.name, None) not in {(c.line, c.name, c.receiver) for c in parsed.calls}
                for u in parsed.uses
            )
            asserts = [
                AssertionInfo(line=a.line, kind=a.kind, excerpt=excerpt(parsed.line_text(a.line)))
                for a in parsed.assertions if within(a.line, a.enclosing, case)
            ][:MAX_ASSERTIONS_PER_CASE]
            best = None
            for pref in ("direct_call", "module_attribute_call", "method_name_match", "name_match_only"):
                if any(k == pref for _, k in kinds):
                    best = pref
                    break
            file_kinds.update(k for _, k in kinds)
            infos.append(TestCaseInfo(
                name=case[0], qualified_name=case[1], start_line=case[2], end_line=case[3], skipped=case[4],
                invokes_target=best in _DIRECT, invocation_kind=best,
                target_call_lines=sorted({ln for ln, _ in kinds})[:10], references_target=refs, assertions=asserts,
            ))
        naming = is_test_path(parsed.path) and target_stem in posixpath.basename(parsed.path).lower()
        if file_kinds & set(_DIRECT):
            relation = "invokes_target"
        elif "method_name_match" in file_kinds:
            relation = "calls_method_by_name"
        elif any(i.references_target for i in infos):
            relation = "references_target"
        elif imports:
            relation = "imports_target_module_only"
        elif "name_match_only" in file_kinds:
            relation = "name_match_only"
        elif naming:
            relation = "naming_convention_only"
        else:
            continue
        infos.sort(key=lambda i: (not i.invokes_target, i.invocation_kind is None, i.start_line))
        framework, evidence = _framework(parsed)
        notes: List[str] = []
        if relation == "imports_target_module_only":
            notes.append("Imports the module, but no test in this file was shown to call or reference the symbol.")
        if relation == "naming_convention_only":
            notes.append("Only the file name suggests a relationship; nothing in the code links it to the target.")
        if relation == "name_match_only":
            notes.append("The identifier is called, but no import ties it to the target file; may be unrelated.")
        if len(infos) > MAX_CASES_PER_FILE:
            notes.append(f"Only the first {MAX_CASES_PER_FILE} of {len(infos)} test cases are listed.")
        results.append((_RELATION_RANK[relation], RelevantTestFile(
            path=parsed.path, framework=framework, framework_evidence=evidence, relation=relation,
            imports_target=imports, imported_names=imported, test_case_count=len(infos),
            test_cases=infos[:MAX_CASES_PER_FILE], notes=notes,
        )))

    results.sort(key=lambda r: (r[0], r[1].path))
    files = [r[1] for r in results]
    if len(files) > inp.max_results:
        session.limits.add("max_results")
    return InspectTestsOutput(
        path=inp.path, symbol_name=inp.symbol_name, target_definitions=[to_match(s, target.language) for s in defs],
        test_files=files[: inp.max_results], test_file_count=len(files),
        **session.result_fields([
            "Tests are located statically and are never executed. A test that imports a module is reported "
            "separately from one that demonstrably calls the symbol (relation / invocation_kind).",
            "Fixtures, parametrization, monkeypatching, dynamic imports and indirect calls through helpers are "
            "not followed, so a real test may be missed or reported at a weaker evidence level.",
        ]),
    )


TOOLS = [
    ToolDefinition(
        name="inspect_tests",
        description=(
            "Find tests relevant to a source file or symbol (pytest/unittest; Jest/Vitest/Mocha-style) at a "
            "pinned commit: test files and cases, imported names, direct invocations, assertions and locations. "
            "Distinguishes importing a module from demonstrably calling a function. Never executes tests."
        ),
        input_model=InspectTestsInput,
        output_model=InspectTestsOutput,
        handler=inspect_tests,
        constraints=_TIMEOUT,
    ),
]

