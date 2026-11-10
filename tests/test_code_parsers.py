"""Parser-level tests for the shared analysis components (no database, no GitHub)."""

import pytest

from app.tools.code import model as m
from app.tools.code.imports import PathIndex, resolve_import
from app.tools.code.js_parser import parse_js_family
from app.tools.code.languages import (
    JAVASCRIPT, PYTHON, TSX, TYPESCRIPT, detect_language, is_test_path, is_unsupported_source,
)
from app.tools.code.py_parser import parse_python

PY = '''\
import os
import psycopg
from .config import DATABASE_URL, TIMEOUT as T

MAX_RETRIES = 3
lowercase_var = 1


@decorator("x")
class Database(Base):
    """doc"""
    LIMIT = 5

    def __init__(self, url=DATABASE_URL):
        self.url = url

    async def connect(self, retries: int = MAX_RETRIES) -> "Conn":
        def inner():
            return helper()
        return inner()

    @staticmethod
    def make():
        return Database()


def helper():
    value = os.environ["APP_KEY"]
    return os.getenv("APP_MODE", "x") + os.environ.get("APP_OTHER", "")
'''

JS = '''\
import axios, { get as g, post } from 'axios';
import * as utils from './utils';
const fs = require('fs');
export const API_URL = process.env.API_URL;
export function fetchUser(id) { return axios.get(id); }
export class Client extends Base {
  constructor(b) { this.b = b; }
  async load(p) { return fetchUser(p); }
  handle = () => { return fetchUser(1); };
}
const helper = async (x) => fetchUser(x);
let counter = 0;
export { helper } from './helpers';
export * from './more';
const lazy = import('./lazy');
const el = <div className="a"><Child /></div>;
'''


def test_language_detection():
    assert detect_language("a/b.py") == PYTHON
    assert detect_language("a.js") == JAVASCRIPT and detect_language("a.jsx") == JAVASCRIPT
    assert detect_language("a.ts") == TYPESCRIPT and detect_language("a.TSX") == TSX
    assert detect_language("a.min.js") is None and detect_language("a.go") is None
    assert is_unsupported_source("a.go") and not is_unsupported_source("a.py")
    assert is_test_path("tests/test_db.py") and is_test_path("src/a.test.ts") and is_test_path("src/__tests__/a.js")
    assert not is_test_path("app/db.py") and not is_test_path("src/a.ts")


def test_python_symbols_nesting_and_signatures():
    parsed = parse_python("app/db.py", PY)
    assert parsed.issue is None
    by_q = {s.qualified_name: s for s in parsed.symbols}
    assert by_q["Database"].kind == m.CLASS and by_q["Database"].bases == ("Base",)
    assert by_q["Database"].decorators == ["decorator('x')"]
    assert by_q["Database"].start_line == 9  # includes the decorator line
    connect = by_q["Database.connect"]
    assert connect.kind == m.METHOD and connect.parent == "Database" and connect.is_async
    assert connect.signature == "async def connect(self, retries: int=MAX_RETRIES) -> 'Conn'"
    assert by_q["Database.connect.inner"].nested is True and by_q["Database.connect.inner"].kind == m.FUNCTION
    assert by_q["Database.make"].decorators == ["staticmethod"]
    assert by_q["MAX_RETRIES"].kind == m.CONSTANT and by_q["lowercase_var"].kind == m.VARIABLE
    assert by_q["Database.LIMIT"].parent_kind == "class"
    assert by_q["helper"].parent_kind == "module" and by_q["helper"].exported


def test_python_imports_calls_env_refs():
    parsed = parse_python("app/db.py", PY)
    kinds = [(r.kind, r.module, r.level, [(n.name, n.local) for n in r.names]) for r in parsed.imports]
    assert ("import", "os", 0, [(None, "os")]) in kinds
    assert ("from", "config", 1, [("DATABASE_URL", "DATABASE_URL"), ("TIMEOUT", "T")]) in kinds
    calls = {(c.name, c.receiver, c.enclosing) for c in parsed.calls}
    assert ("helper", None, "Database.connect.inner") in calls
    assert ("connect", "psycopg", None) not in calls  # not called in this file
    assert ("getenv", "os", "helper") in calls
    assert {n for n, _ in parsed.env_refs} == {"APP_KEY", "APP_MODE", "APP_OTHER"}


def test_python_syntax_error_is_structured_not_partial():
    parsed = parse_python("bad.py", "def broken(:\n    pass\n")
    assert parsed.issue is not None and parsed.issue.line == 1 and not parsed.symbols
    assert "SyntaxError" in parsed.issue.message and parsed.issue.partial is False


def test_python_null_bytes_do_not_crash():
    parsed = parse_python("x.py", "a = 1\x00\n")
    assert parsed.issue is not None


def test_python_assertions_and_unittest():
    src = (
        "import unittest\nimport pytest\n"
        "class T(unittest.TestCase):\n"
        "    def test_a(self):\n        self.assertEqual(1, 1)\n        assert True\n"
        "def test_b():\n    with pytest.raises(ValueError):\n        pass\n"
    )
    parsed = parse_python("t.py", src)
    kinds = {(a.kind, a.enclosing) for a in parsed.assertions}
    assert ("assertX", "T.test_a") in kinds and ("assert", "T.test_a") in kinds and ("raises", "test_b") in kinds


def test_javascript_symbols_exports_and_imports():
    parsed = parse_js_family("web/api.js", JS, JAVASCRIPT)
    assert parsed.issue is None, parsed.issue
    by_q = {s.qualified_name: s for s in parsed.symbols}
    assert by_q["fetchUser"].kind == m.FUNCTION and by_q["fetchUser"].exported
    assert by_q["fetchUser"].signature == "function fetchUser(id)"
    assert by_q["Client"].kind == m.CLASS and by_q["Client"].bases == ("extends Base",)
    assert by_q["Client.load"].kind == m.METHOD and by_q["Client.load"].signature == "async load(p)"
    assert by_q["Client.handle"].kind == m.METHOD  # arrow-function class field
    assert by_q["helper"].kind == m.FUNCTION and by_q["helper"].signature == "async helper(x)"
    assert by_q["API_URL"].kind == m.CONSTANT and by_q["counter"].kind == m.VARIABLE
    assert not by_q["counter"].exported
    imports = {(r.kind, r.module): [(n.name, n.local, n.type_only) for n in r.names] for r in parsed.imports}
    assert imports[("es_import", "axios")] == [("default", "axios", False), ("get", "g", False), ("post", "post", False)]
    assert imports[("es_import", "./utils")] == [("*", "utils", False)]
    assert imports[("require", "fs")] == [(None, "fs", False)]
    assert imports[("es_export_from", "./helpers")] == [("helper", "helper", False)]
    assert imports[("es_export_from", "./more")] == [("*", None, False)]
    assert ("dynamic_import", "./lazy") in imports
    assert [n for n, _ in parsed.env_refs] == ["API_URL"]


def test_javascript_calls_enclosing_and_jsx():
    parsed = parse_js_family("web/api.js", JS, JAVASCRIPT)
    calls = {(c.name, c.receiver, c.enclosing) for c in parsed.calls}
    assert ("fetchUser", None, "Client.load") in calls and ("fetchUser", None, "Client.handle") in calls
    assert ("fetchUser", None, "helper") in calls and ("get", "axios", "fetchUser") in calls
    names = {u.name for u in parsed.uses if u.kind == "name"}
    assert "Child" in names and "div" not in names  # JSX component is a use; intrinsic element is not


def test_typescript_declarations():
    src = (
        "export interface User { id: number }\nexport type Id = number;\nexport enum Role { A, B }\n"
        "export abstract class Repo<T> { abstract find(id: Id): T; save(x: T): void {} }\n"
        "export const f = (a: number): string => String(a);\n"
        "export default function main(): void {}\n"
        "import x = require('legacy');\n"
        "import type { T } from './types';\n"
    )
    parsed = parse_js_family("a.ts", src, TYPESCRIPT)
    assert parsed.issue is None, parsed.issue
    kinds = {s.qualified_name: s.kind for s in parsed.symbols}
    assert kinds["User"] == m.INTERFACE and kinds["Id"] == m.TYPE and kinds["Role"] == m.ENUM
    assert kinds["Repo"] == m.CLASS and kinds["Repo.find"] == m.METHOD and kinds["Repo.save"] == m.METHOD
    assert kinds["f"] == m.FUNCTION and kinds["main"] == m.FUNCTION
    sig = next(s.signature for s in parsed.symbols if s.name == "f")
    assert sig == "f(a: number): string"
    assert any(r.kind == "require" and r.module == "legacy" for r in parsed.imports)
    type_only = next(r for r in parsed.imports if r.module == "./types")
    assert [(n.name, n.type_only) for n in type_only.names] == [("T", True)]


def test_tsx_parsing():
    src = (
        "import React from 'react';\n"
        "export function App({ id }: { id: number }) { return <div><Profile id={id} /></div>; }\n"
        "const Profile = (p: { id: number }) => <span>{p.id}</span>;\n"
    )
    parsed = parse_js_family("App.tsx", src, TSX)
    assert parsed.issue is None, parsed.issue
    assert {s.name for s in parsed.symbols} == {"App", "Profile"}
    assert any(c.name == "Profile" for c in parsed.uses if c.kind == "name") or any(
        u.name == "Profile" for u in parsed.uses)


def test_jsx_in_javascript_grammar():
    parsed = parse_js_family("a.jsx", "export const A = () => <b>hi</b>;\n", JAVASCRIPT)
    assert parsed.issue is None and parsed.symbols[0].name == "A"


def test_syntax_error_in_js_is_flagged_partial():
    parsed = parse_js_family("bad.js", "export function ok() { return 1; }\nfunction ( {\n", JAVASCRIPT)
    assert parsed.issue is not None and parsed.issue.partial is True and parsed.issue.line is not None
    assert any(s.name == "ok" for s in parsed.symbols)  # recoverable parts are kept, but flagged partial


def test_typescript_syntax_is_not_valid_javascript():
    parsed = parse_js_family("a.js", "const x: number = 1;\n", JAVASCRIPT)
    assert parsed.issue is not None  # parse errors surface instead of being papered over


def test_js_test_blocks_and_assertions():
    src = (
        "import { fetchUser } from './api';\n"
        "describe('api', () => {\n"
        "  it('fetches', () => { expect(fetchUser(1)).toBe(1); });\n"
        "  it.skip('later', () => {});\n"
        "  test.only('x', () => { assert.equal(1, 1); });\n"
        "});\n"
    )
    parsed = parse_js_family("a.test.js", src, JAVASCRIPT)
    cases = {(t.qualified_name, t.kind, t.skipped) for t in parsed.test_cases}
    assert ("api", "suite", False) in cases and ("api > fetches", "test", False) in cases
    assert ("api > later", "test", True) in cases and ("api > x", "test", False) in cases
    assert {a.kind for a in parsed.assertions} == {"expect", "assert"}


def test_deeply_nested_js_does_not_blow_the_stack():
    src = "x = " + "(" * 3000 + "1" + ")" * 3000 + ";\n"
    parsed = parse_js_family("deep.js", src, JAVASCRIPT)
    assert parsed.path == "deep.js"  # completes without RecursionError


# --- import resolution -------------------------------------------------------


def _resolve(importer, language, rec, paths):
    return resolve_import(rec, importer, language, PathIndex(paths))


def test_python_import_resolution():
    paths = ["app/__init__.py", "app/db.py", "app/config.py", "app/util/__init__.py", "app/util/x.py", "src/lib/y.py"]
    p = parse_python("app/db.py", "from .config import X\nfrom . import config\nfrom .util import x\nimport os\n"
                                "import requests\nfrom app.util import x as xx\nimport lib.y\nfrom .nope import z\n")
    out = [_resolve("app/db.py", PYTHON, r, paths) for r in p.imports]
    assert (out[0].base.kind, out[0].base.target) == ("internal", "app/config.py")
    assert (out[1].base.kind, out[1].base.target) == ("internal", "app/__init__.py")
    assert out[1].submodules == {"config": "app/config.py"}
    assert out[2].base.target == "app/util/__init__.py" and out[2].submodules == {"x": "app/util/x.py"}
    assert (out[3].base.kind, out[3].base.target) == ("stdlib", "os")
    assert (out[4].base.kind, out[4].base.target) == ("external", "requests")
    assert out[5].base.target == "app/util/__init__.py"
    assert (out[6].base.kind, out[6].base.target) == ("internal", "src/lib/y.py")  # `src` layout root
    assert out[7].base.kind == "unresolved"


def test_js_import_resolution():
    paths = ["web/src/api.js", "web/src/utils.ts", "web/src/ui/index.tsx", "web/src/lib.ts"]
    src = (
        "import a from './utils';\nimport b from './ui';\nimport c from './lib.js';\nimport d from 'react';\n"
        "import e from '@scope/pkg/deep';\nimport f from 'node:fs';\nimport g from './missing';\n"
        "import h from '@/alias';\nconst i = require('lodash/fp');\nimport j from '../src/api';\n"
    )
    parsed = parse_js_family("web/src/app.ts", src, TYPESCRIPT)
    out = [_resolve("web/src/app.ts", TYPESCRIPT, r, paths).base for r in parsed.imports]
    assert [(o.kind, o.target) for o in out] == [
        ("internal", "web/src/utils.ts"), ("internal", "web/src/ui/index.tsx"), ("internal", "web/src/lib.ts"),
        ("external", "react"), ("external", "@scope/pkg"), ("builtin", "fs"), ("unresolved", "./missing"),
        ("unresolved", "@/alias"), ("external", "lodash"), ("internal", "web/src/api.js"),
    ]


@pytest.mark.parametrize("text", ["", "\n", "# only a comment\n"])
def test_empty_python_files(text):
    parsed = parse_python("e.py", text)
    assert parsed.issue is None and parsed.symbols == []


def test_commonjs_destructured_require_binds_names():
    parsed = parse_js_family(
        "a.js", "const { a, b: c } = require('./m');\nconst d = require('./n');\nrequire('./side');\n", JAVASCRIPT)
    by_module = {r.module: [(n.name, n.local) for n in r.names] for r in parsed.imports}
    assert by_module["./m"] == [("a", "a"), ("b", "c")]
    assert by_module["./n"] == [(None, "d")] and by_module["./side"] == [(None, None)]
