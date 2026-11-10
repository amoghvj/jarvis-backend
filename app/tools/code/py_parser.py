"""Python source analysis with the standard-library `ast` module (no code is executed)."""

import ast
from typing import List, Optional

from app.tools.code import model as m
from app.tools.code.languages import PYTHON

_RECEIVER_LIMIT = 80


def _unparse(node: Optional[ast.AST], limit: int = 200) -> str:
    if node is None:
        return ""
    try:
        return ast.unparse(node)[:limit]
    except Exception:  # unparse can fail on exotic trees; this is cosmetic only
        return ""


def _signature(node) -> str:
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    sig = f"{prefix} {node.name}({_unparse(node.args, 300)})"
    if node.returns is not None:
        sig += f" -> {_unparse(node.returns, 80)}"
    return sig


def _is_constant_name(name: str) -> bool:
    return name.isupper() or (name.upper() == name and any(c.isalpha() for c in name))


class _Visitor(ast.NodeVisitor):
    def __init__(self, parsed: m.ParsedFile):
        self.p = parsed
        # Stack of (kind, qualified_name); kind is "class" or "function".
        self.scope: List[tuple] = []

    # -- scope helpers --------------------------------------------------------

    def _qual(self, name: str) -> str:
        return f"{self.scope[-1][1]}.{name}" if self.scope else name

    def _parent(self):
        return self.scope[-1] if self.scope else None

    def _enclosing_function(self) -> Optional[str]:
        for kind, qual in reversed(self.scope):
            if kind == "function":
                return qual
        return None

    def _in_function(self) -> bool:
        return any(kind == "function" for kind, _ in self.scope)

    # -- definitions ----------------------------------------------------------

    def _function(self, node) -> None:
        parent = self._parent()
        in_class = parent is not None and parent[0] == "class"
        qual = self._qual(node.name)
        decorators = [_unparse(d, 120) for d in node.decorator_list]
        start = min([node.lineno] + [d.lineno for d in node.decorator_list])
        self.p.symbols.append(
            m.Symbol(
                name=node.name,
                qualified_name=qual,
                kind=m.METHOD if in_class else m.FUNCTION,
                path=self.p.path,
                start_line=start,
                end_line=node.end_lineno or node.lineno,
                def_line=node.lineno,
                parent=parent[1] if parent else None,
                parent_kind=(parent[0] if parent else "module"),
                signature=_signature(node),
                decorators=decorators,
                exported=parent is None and not node.name.startswith("_"),
                nested=self._in_function(),
                is_async=isinstance(node, ast.AsyncFunctionDef),
            )
        )
        for d in node.decorator_list:
            self.visit(d)
        # Defaults and annotations are evaluated in the enclosing scope.
        self.visit(node.args)
        if node.returns is not None:
            self.visit(node.returns)
        self.scope.append(("function", qual))
        for stmt in node.body:
            self.visit(stmt)
        self.scope.pop()

    visit_FunctionDef = _function
    visit_AsyncFunctionDef = _function

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        parent = self._parent()
        qual = self._qual(node.name)
        self.p.symbols.append(
            m.Symbol(
                name=node.name,
                qualified_name=qual,
                kind=m.CLASS,
                path=self.p.path,
                start_line=min([node.lineno] + [d.lineno for d in node.decorator_list]),
                end_line=node.end_lineno or node.lineno,
                def_line=node.lineno,
                parent=parent[1] if parent else None,
                parent_kind=(parent[0] if parent else "module"),
                signature=f"class {node.name}({', '.join(_unparse(b, 60) for b in node.bases)})"
                if node.bases
                else f"class {node.name}",
                decorators=[_unparse(d, 120) for d in node.decorator_list],
                bases=tuple(_unparse(b, 60) for b in node.bases),
                exported=parent is None and not node.name.startswith("_"),
                nested=self._in_function(),
            )
        )
        for d in node.decorator_list:
            self.visit(d)
        for b in node.bases:
            self.visit(b)
        for kw in node.keywords:
            self.visit(kw.value)
        self.scope.append(("class", qual))
        for stmt in node.body:
            self.visit(stmt)
        self.scope.pop()

    def _record_variable(self, target: ast.AST, node: ast.AST) -> None:
        if self._in_function() or not isinstance(target, ast.Name):
            return
        parent = self._parent()
        self.p.symbols.append(
            m.Symbol(
                name=target.id,
                qualified_name=self._qual(target.id),
                kind=m.CONSTANT if _is_constant_name(target.id) else m.VARIABLE,
                path=self.p.path,
                start_line=node.lineno,
                end_line=node.end_lineno or node.lineno,
                def_line=node.lineno,
                parent=parent[1] if parent else None,
                parent_kind=(parent[0] if parent else "module"),
                exported=parent is None and not target.id.startswith("_"),
            )
        )

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if isinstance(target, (ast.Tuple, ast.List)):
                for elt in target.elts:
                    self._record_variable(elt, node)
            else:
                self._record_variable(target, node)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._record_variable(node.target, node)
        self.generic_visit(node)

    # -- uses -----------------------------------------------------------------

    def visit_Name(self, node: ast.Name) -> None:
        self.p.uses.append(
            m.Use(
                name=node.id,
                line=node.lineno,
                kind="name",
                ctx="store" if isinstance(node.ctx, (ast.Store, ast.Del)) else "load",
                enclosing=self._enclosing_function(),
            )
        )

    def visit_Attribute(self, node: ast.Attribute) -> None:
        self.p.uses.append(
            m.Use(
                name=node.attr,
                line=node.lineno,
                kind="attribute",
                ctx="store" if isinstance(node.ctx, (ast.Store, ast.Del)) else "load",
                receiver=_unparse(node.value, _RECEIVER_LIMIT),
                enclosing=self._enclosing_function(),
            )
        )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        enclosing = self._enclosing_function()
        if isinstance(func, ast.Name):
            self.p.calls.append(m.Call(name=func.id, line=node.lineno, enclosing=enclosing))
        elif isinstance(func, ast.Attribute):
            receiver = _unparse(func.value, _RECEIVER_LIMIT)
            self.p.calls.append(
                m.Call(name=func.attr, line=node.lineno, receiver=receiver, enclosing=enclosing)
            )
            self._maybe_assertion_or_env(node, func, receiver, enclosing)
        self._maybe_env_getenv(node, func)
        self.generic_visit(node)

    def _maybe_assertion_or_env(self, node, func, receiver, enclosing) -> None:
        if func.attr.startswith("assert") and receiver in ("self", "cls"):
            self.p.assertions.append(m.Assertion(node.lineno, "assertX", enclosing))
        elif func.attr in ("raises", "warns") and receiver in ("pytest",):
            self.p.assertions.append(m.Assertion(node.lineno, "raises", enclosing))
        elif func.attr == "get" and receiver in ("os.environ", "environ"):
            self._env_literal(node)

    def _maybe_env_getenv(self, node: ast.Call, func) -> None:
        name = func.id if isinstance(func, ast.Name) else (func.attr if isinstance(func, ast.Attribute) else "")
        if name == "getenv":
            self._env_literal(node)

    def _env_literal(self, node: ast.Call) -> None:
        if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
            self.p.env_refs.append((node.args[0].value, node.lineno))

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if _unparse(node.value, 30) in ("os.environ", "environ"):
            key = node.slice
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                self.p.env_refs.append((key.value, node.lineno))
        self.generic_visit(node)

    def visit_Assert(self, node: ast.Assert) -> None:
        self.p.assertions.append(m.Assertion(node.lineno, "assert", self._enclosing_function()))
        self.generic_visit(node)

    def visit_With(self, node: ast.With) -> None:
        self._with(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        self._with(node)

    def _with(self, node) -> None:
        for item in node.items:
            ctx = item.context_expr
            if isinstance(ctx, ast.Call) and isinstance(ctx.func, ast.Attribute):
                if ctx.func.attr.startswith("assert") and _unparse(ctx.func.value, 10) in ("self", "cls"):
                    self.p.assertions.append(m.Assertion(node.lineno, "assertX", self._enclosing_function()))
        self.generic_visit(node)

    # -- imports --------------------------------------------------------------

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            local = alias.asname or alias.name.split(".")[0]
            self.p.imports.append(
                m.ImportRec(
                    kind="import",
                    module=alias.name,
                    line=node.lineno,
                    names=[m.ImportedName(name=None, local=local)],
                )
            )
            self.p.uses.append(m.Use(name=alias.name.split(".")[-1], line=node.lineno, kind="import"))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        names = []
        for alias in node.names:
            names.append(
                m.ImportedName(
                    name=alias.name,
                    local=None if alias.name == "*" else (alias.asname or alias.name),
                )
            )
            self.p.uses.append(m.Use(name=alias.name, line=node.lineno, kind="import"))
        self.p.imports.append(
            m.ImportRec(
                kind="from",
                module=node.module or "",
                line=node.lineno,
                level=node.level or 0,
                names=names,
            )
        )


def parse_python(path: str, text: str) -> m.ParsedFile:
    parsed = m.ParsedFile(path=path, language=PYTHON, lines=text.split("\n"))
    try:
        tree = ast.parse(text, filename=path)
    except SyntaxError as exc:
        parsed.issue = m.ParseIssue(exc.lineno, f"SyntaxError: {str(exc.msg)[:120]}")
        return parsed
    except (ValueError, RecursionError, MemoryError) as exc:
        parsed.issue = m.ParseIssue(None, f"{type(exc).__name__}: file could not be parsed")
        return parsed
    try:
        _Visitor(parsed).visit(tree)
    except RecursionError:
        parsed.issue = m.ParseIssue(None, "RecursionError: syntax tree too deeply nested", partial=True)
    return parsed
