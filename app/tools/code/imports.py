"""Import resolution against the repository tree of one resolved commit.

Resolution is purely path-based (no packages are installed, no code is run). It can
tell an import of a file in the repository from an external package; it deliberately
does not evaluate tsconfig `paths`, bundler aliases, `sys.path` manipulation or
namespace-package tricks, and reports such imports as `unresolved`.
"""

import posixpath
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from app.tools.code import model as m
from app.tools.code.languages import PYTHON, is_js_family

_JS_EXTENSIONS = (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".mts", ".cts", ".json")
_JS_REMAP = {".js": (".ts", ".tsx"), ".jsx": (".tsx",), ".mjs": (".mts",), ".cjs": (".cts",)}
_NODE_BUILTINS = frozenset(
    "assert async_hooks buffer child_process cluster console constants crypto dgram dns domain "
    "events fs http http2 https inspector module net os path perf_hooks process punycode "
    "querystring readline repl stream string_decoder sys timers tls trace_events tty url util v8 "
    "vm wasi worker_threads zlib".split()
)
_STDLIB = getattr(sys, "stdlib_module_names", frozenset())


@dataclass(frozen=True)
class Resolution:
    kind: str  # "internal" | "external" | "stdlib" | "builtin" | "unresolved"
    target: str  # repository path for internal; package / module name otherwise
    detail: Optional[str] = None


@dataclass
class ResolvedImport:
    record: m.ImportRec
    base: Resolution
    # For `from pkg import name`: names that are themselves repository modules.
    submodules: Dict[str, str]


class PathIndex:
    def __init__(self, paths: Iterable[str]):
        self.paths = set(paths)

    def has(self, path: str) -> bool:
        return path in self.paths


def _join(*parts: str) -> str:
    return posixpath.normpath(posixpath.join(*[p for p in parts if p])) if any(parts) else ""


def _py_roots(importer: str) -> List[str]:
    roots = ["", "src"]
    directory = posixpath.dirname(importer)
    ancestors = []
    while directory:
        ancestors.append(directory)
        directory = posixpath.dirname(directory)
    for a in ancestors:  # deepest first
        if a not in roots:
            roots.append(a)
    return roots


def _py_module_file(index: PathIndex, base: str, parts: List[str]) -> Optional[str]:
    joined = posixpath.join(base, *parts) if base else posixpath.join(*parts) if parts else base
    if not joined:
        return None
    for candidate in (f"{joined}.py", f"{joined}/__init__.py"):
        if index.has(candidate):
            return candidate
    return None


def _resolve_python(rec: m.ImportRec, importer: str, index: PathIndex) -> ResolvedImport:
    parts = [p for p in rec.module.split(".") if p]
    submodules: Dict[str, str] = {}
    base_res: Optional[Resolution] = None

    if rec.level > 0:
        directory = posixpath.dirname(importer)
        for _ in range(rec.level - 1):
            directory = posixpath.dirname(directory)
        if parts:
            found = _py_module_file(index, directory, parts)
        else:
            found = f"{directory}/__init__.py" if directory else "__init__.py"
            found = found if index.has(found) else None
        if found:
            base_res = Resolution("internal", found)
        else:
            base_res = Resolution("unresolved", "." * rec.level + rec.module, "relative import target not found")
        search_bases = [directory]
    else:
        search_bases = _py_roots(importer)
        for root in search_bases:
            found = _py_module_file(index, root, parts) if parts else None
            if found:
                base_res = Resolution("internal", found)
                break
        if base_res is None:
            top = parts[0] if parts else ""
            if top in _STDLIB:
                base_res = Resolution("stdlib", top)
            else:
                base_res = Resolution("external", top or rec.module)

    if rec.kind == "from":
        for imported in rec.names:
            if not imported.name or imported.name == "*":
                continue
            for root in search_bases:
                if rec.level > 0:
                    sub = _py_module_file(index, root, parts + [imported.name])
                else:
                    sub = _py_module_file(index, root, parts + [imported.name]) if parts else None
                if sub:
                    submodules[imported.name] = sub
                    break
    return ResolvedImport(rec, base_res, submodules)


def _js_package_name(spec: str) -> str:
    segments = spec.split("/")
    return "/".join(segments[:2]) if spec.startswith("@") and len(segments) > 1 else segments[0]


def _js_candidates(base: str) -> List[str]:
    stem, ext = posixpath.splitext(base)
    candidates = [base]
    if ext in _JS_REMAP:  # TypeScript ESM: `./x.js` really means x.ts / x.tsx
        candidates += [stem + e for e in _JS_REMAP[ext]]
    candidates += [base + e for e in _JS_EXTENSIONS]
    candidates += [f"{base}/index{e}" for e in _JS_EXTENSIONS]
    return candidates


def _resolve_js(rec: m.ImportRec, importer: str, index: PathIndex) -> ResolvedImport:
    spec = rec.module
    if spec.startswith(("./", "../")) or spec in (".", ".."):
        base = _join(posixpath.dirname(importer), spec)
        for candidate in _js_candidates(base):
            if index.has(candidate):
                return ResolvedImport(rec, Resolution("internal", candidate), {})
        return ResolvedImport(rec, Resolution("unresolved", spec, "relative import target not found"), {})
    if spec.startswith("node:") or spec in _NODE_BUILTINS:
        return ResolvedImport(rec, Resolution("builtin", spec.removeprefix("node:")), {})
    if spec.startswith(("@/", "~/", "#", "$", "/")):
        return ResolvedImport(
            rec, Resolution("unresolved", spec, "path alias or absolute path; tsconfig/bundler aliases are not evaluated"), {}
        )
    return ResolvedImport(rec, Resolution("external", _js_package_name(spec)), {})


def resolve_import(rec: m.ImportRec, importer: str, language: str, index: PathIndex) -> ResolvedImport:
    if language == PYTHON:
        return _resolve_python(rec, importer, index)
    if is_js_family(language):
        return _resolve_js(rec, importer, index)
    return ResolvedImport(rec, Resolution("unresolved", rec.module, "unsupported language"), {})

