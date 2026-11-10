"""JavaScript / TypeScript / JSX / TSX analysis using tree-sitter (no code is executed).

The walk is iterative so adversarially deep syntax trees cannot exhaust the Python
stack. tree-sitter is error-tolerant: when a file has syntax errors the parser still
returns a tree, but the result is flagged `partial` so callers never treat it as complete.
"""

import functools
import re
from typing import List, Optional, Tuple

from app.tools.code import model as m
from app.tools.code.languages import JAVASCRIPT, TSX, TYPESCRIPT

_FUNCTION_VALUES = {"arrow_function", "function_expression", "function", "generator_function"}
_TEST_SUITES = {"describe", "suite", "context"}
_TEST_CASES = {"it", "test", "specify"}
_TEST_MODIFIERS = {"only", "skip", "todo", "concurrent", "each"}
_PARAM_PARENTS = {"formal_parameters", "required_parameter", "optional_parameter", "rest_pattern", "arrow_function"}
_DECL_PARENTS = {
    "function_declaration", "generator_function_declaration", "class_declaration",
    "abstract_class_declaration", "variable_declarator", "method_definition",
    "interface_declaration", "type_alias_declaration", "enum_declaration",
    "function_expression", "public_field_definition", "field_definition",
    "abstract_method_signature", "method_signature", "type_parameter", "class",
}
_STRING_TYPES = {"string", "template_string"}
_SPACE = re.compile(r"\s+")
_RECEIVER_LIMIT = 80


@functools.lru_cache(maxsize=None)
def _parser(language: str):
    from tree_sitter import Language, Parser

    if language == JAVASCRIPT:
        import tree_sitter_javascript as grammar

        lang = grammar.language()
    else:
        import tree_sitter_typescript as grammar

        lang = grammar.language_tsx() if language == TSX else grammar.language_typescript()
    return Parser(Language(lang))


def _txt(node, limit: int = 0) -> str:
    text = node.text.decode("utf-8", "replace") if node is not None and node.text is not None else ""
    return text[:limit] if limit else text


def _line(node) -> int:
    return node.start_point[0] + 1


def _end_line(node) -> int:
    # A node ending at column 0 of the next row does not really include that row.
    row, col = node.end_point
    return row if col == 0 and row > node.start_point[0] else row + 1


def _string_value(node) -> Optional[str]:
    """The literal value of a plain string / template without substitutions."""
    if node is None or node.type not in _STRING_TYPES:
        return None
    if node.type == "template_string" and any(c.type == "template_substitution" for c in node.children):
        return None
    raw = _txt(node)
    return raw[1:-1] if len(raw) >= 2 else None


def _field_is(node, name: str) -> bool:
    parent = node.parent
    if parent is None:
        return False
    target = parent.child_by_field_name(name)
    return target is not None and target.id == node.id


def _compact(text: str, limit: int = 300) -> str:
    return _SPACE.sub(" ", text).strip()[:limit]


def _has_keyword(node, keyword: str) -> bool:
    return any(c.type == keyword for c in node.children)


def _decorators(node) -> List[str]:
    found = [_compact(_txt(c), 120) for c in node.children if c.type == "decorator"]
    parent = node.parent
    if parent is not None and parent.type == "export_statement":
        found = [_compact(_txt(c), 120) for c in parent.children if c.type == "decorator"] + found
    return found


class _Walker:
    def __init__(self, parsed: m.ParsedFile):
        self.p = parsed
        self.error_line: Optional[int] = None

    # scope = tuple of (kind, qualified_name); kinds: class, function, suite, test, var

    @staticmethod
    def _qual(scope: tuple, name: str) -> str:
        for kind, qual in reversed(scope):
            if kind in ("class", "function"):
                return f"{qual}.{name}"
        return name

    @staticmethod
    def _enclosing(scope: tuple) -> Optional[str]:
        for kind, qual in reversed(scope):
            if kind == "function":
                return qual
        return None

    @staticmethod
    def _in_function(scope: tuple) -> bool:
        return any(kind == "function" for kind, _ in scope)

    @staticmethod
    def _parent_info(scope: tuple):
        for kind, qual in reversed(scope):
            if kind in ("class", "function"):
                return qual, kind
        return None, "module"

    def _exported(self, node) -> bool:
        parent = node.parent
        if parent is not None and parent.type in ("lexical_declaration", "variable_declaration"):
            parent = parent.parent
        return parent is not None and parent.type == "export_statement"

    def _outer(self, node):
        """Topmost statement for range purposes: declarator -> declaration -> export."""
        cur = node
        while cur.parent is not None and cur.parent.type in (
            "lexical_declaration", "variable_declaration", "export_statement"
        ):
            cur = cur.parent
        return cur

    def run(self, root) -> None:
        stack: List[Tuple[object, tuple]] = [(root, ())]
        while stack:
            node, scope = stack.pop()
            if node.type == "ERROR" or node.is_missing:
                if self.error_line is None:
                    self.error_line = _line(node)
            child_scope = self._handle(node, scope)
            for child in reversed(node.children):
                stack.append((child, child_scope))

    # -- node dispatch --------------------------------------------------------

    def _handle(self, node, scope: tuple) -> tuple:  # noqa: C901 - flat dispatch is clearer here
        t = node.type
        if t in ("function_declaration", "generator_function_declaration"):
            return self._function_decl(node, scope)
        if t in ("class_declaration", "abstract_class_declaration"):
            return self._class_decl(node, scope)
        if t in ("method_definition", "abstract_method_signature"):
            return self._method(node, scope)
        if t == "variable_declarator":
            return self._declarator(node, scope)
        if t in ("public_field_definition", "field_definition"):
            return self._field(node, scope)
        if t in ("interface_declaration", "type_alias_declaration", "enum_declaration"):
            self._type_decl(node, scope)
            return scope
        if t == "import_statement":
            self._import(node)
            return scope
        if t == "export_statement":
            self._export_from(node)
            return scope
        if t == "call_expression":
            return self._call(node, scope)
        if t == "new_expression":
            self._new(node, scope)
            return scope
        if t == "member_expression":
            self._member(node, scope)
            return scope
        if t == "subscript_expression":
            self._subscript_env(node)
            return scope
        if t in ("identifier", "type_identifier", "shorthand_property_identifier"):
            self._identifier(node, scope)
            return scope
        return scope

    # -- definitions ----------------------------------------------------------

    def _add_symbol(self, node, name: str, kind: str, scope: tuple, *, outer=None, **extra) -> m.Symbol:
        outer = outer or node
        parent_q, parent_kind = self._parent_info(scope)
        sym = m.Symbol(
            name=name,
            qualified_name=self._qual(scope, name),
            kind=kind,
            path=self.p.path,
            start_line=_line(outer),
            end_line=_end_line(outer),
            def_line=_line(node),
            parent=parent_q,
            parent_kind=parent_kind,
            exported=self._exported(node) and parent_q is None,
            nested=self._in_function(scope),
            **extra,
        )
        self.p.symbols.append(sym)
        return sym

    def _signature(self, prefix: str, name: str, node) -> str:
        params = node.child_by_field_name("parameters") or node.child_by_field_name("parameter")
        params_text = _txt(params) if params is not None else "()"
        if params is not None and params.type == "identifier":
            params_text = f"({params_text})"
        ret = node.child_by_field_name("return_type")
        return _compact(f"{prefix}{name}{params_text}{_txt(ret) if ret is not None else ''}")

    def _function_decl(self, node, scope: tuple) -> tuple:
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return scope
        name = _txt(name_node)
        prefix = "async function " if _has_keyword(node, "async") else "function "
        sym = self._add_symbol(
            node, name, m.FUNCTION, scope,
            outer=self._outer(node), signature=self._signature(prefix, name, node),
            decorators=_decorators(node), is_async=_has_keyword(node, "async"),
        )
        return scope + (("function", sym.qualified_name),)

    def _class_decl(self, node, scope: tuple) -> tuple:
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return scope
        name = _txt(name_node)
        heritage = next((c for c in node.children if c.type == "class_heritage"), None)
        bases = (_compact(_txt(heritage), 120),) if heritage is not None else ()
        sym = self._add_symbol(
            node, name, m.CLASS, scope,
            outer=self._outer(node),
            signature=_compact(f"class {name} {bases[0] if bases else ''}"),
            decorators=_decorators(node), bases=bases,
        )
        return scope + (("class", sym.qualified_name),)

    def _method(self, node, scope: tuple) -> tuple:
        name_node = node.child_by_field_name("name")
        in_class = node.parent is not None and node.parent.type == "class_body"
        name = _compact(_txt(name_node), 80) if name_node is not None else ""
        if not name or not in_class:
            return scope  # object-literal methods are not catalogued as symbols
        prefix = "async " if _has_keyword(node, "async") else ""
        sym = self._add_symbol(
            node, name, m.METHOD, scope,
            signature=self._signature(prefix, name, node),
            decorators=_decorators(node), is_async=bool(prefix),
        )
        return scope + (("function", sym.qualified_name),)

    def _declarator(self, node, scope: tuple) -> tuple:
        name_node = node.child_by_field_name("name")
        value = node.child_by_field_name("value")
        if name_node is None or name_node.type != "identifier":
            return scope  # destructuring patterns are not catalogued
        name = _txt(name_node)
        decl = node.parent
        outer = self._outer(node)
        if value is not None and value.type in _FUNCTION_VALUES:
            prefix = "async " if _has_keyword(value, "async") else ""
            sym = self._add_symbol(
                node, name, m.FUNCTION, scope, outer=outer,
                signature=self._signature(prefix, name, value), is_async=bool(prefix),
            )
            return scope + (("function", sym.qualified_name),)
        if self._in_function(scope):
            return scope
        if value is not None and value.type == "class":
            sym = self._add_symbol(node, name, m.CLASS, scope, outer=outer, signature=f"class {name}")
            return scope + (("class", sym.qualified_name),)
        is_const = decl is not None and _has_keyword(decl, "const")
        self._add_symbol(node, name, m.CONSTANT if is_const else m.VARIABLE, scope, outer=outer)
        return scope

    def _field(self, node, scope: tuple) -> tuple:
        name_node = node.child_by_field_name("name") or node.child_by_field_name("property")
        value = node.child_by_field_name("value")
        if name_node is None:
            return scope
        name = _compact(_txt(name_node), 80)
        if value is not None and value.type in _FUNCTION_VALUES:
            sym = self._add_symbol(
                node, name, m.METHOD, scope,
                signature=self._signature("", name, value),
                is_async=_has_keyword(value, "async"),
            )
            return scope + (("function", sym.qualified_name),)
        return scope

    def _type_decl(self, node, scope: tuple) -> None:
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return
        kind = {"interface_declaration": m.INTERFACE, "type_alias_declaration": m.TYPE}.get(node.type, m.ENUM)
        name = _txt(name_node)
        self._add_symbol(node, name, kind, scope, outer=self._outer(node), signature=f"{kind} {name}")

    # -- imports / exports ----------------------------------------------------

    def _import(self, node) -> None:
        source = _string_value(node.child_by_field_name("source"))
        type_only = any(c.type == "type" for c in node.children)
        if source is None:
            req = next((c for c in node.children if c.type == "import_require_clause"), None)
            if req is not None:
                module = _string_value(next((c for c in req.children if c.type in _STRING_TYPES), None))
                local = next((_txt(c) for c in req.children if c.type == "identifier"), None)
                if module:
                    self.p.imports.append(m.ImportRec(
                        "require", module, _line(node), names=[m.ImportedName(None, local)]))
            return
        names: List[m.ImportedName] = []
        clause = next((c for c in node.children if c.type == "import_clause"), None)
        if clause is not None:
            for c in clause.children:
                if c.type == "identifier":
                    names.append(m.ImportedName("default", _txt(c), type_only))
                elif c.type == "namespace_import":
                    ident = next((x for x in c.children if x.type == "identifier"), None)
                    names.append(m.ImportedName("*", _txt(ident) if ident is not None else None, type_only))
                elif c.type == "named_imports":
                    for spec in c.children:
                        if spec.type != "import_specifier":
                            continue
                        n = spec.child_by_field_name("name")
                        alias = spec.child_by_field_name("alias")
                        spec_type = type_only or any(x.type == "type" for x in spec.children)
                        names.append(m.ImportedName(_txt(n), _txt(alias) if alias is not None else _txt(n), spec_type))
        self.p.imports.append(m.ImportRec("es_import", source, _line(node), names=names))
        for n in names:
            if n.name not in (None, "*", "default"):
                self.p.uses.append(m.Use(name=n.name, line=_line(node), kind="import"))

    def _export_from(self, node) -> None:
        source = _string_value(node.child_by_field_name("source"))
        if source is None:
            return
        names: List[m.ImportedName] = []
        clause = next((c for c in node.children if c.type == "export_clause"), None)
        if clause is not None:
            for spec in clause.children:
                if spec.type == "export_specifier":
                    n = spec.child_by_field_name("name")
                    alias = spec.child_by_field_name("alias")
                    names.append(m.ImportedName(_txt(n), _txt(alias) if alias is not None else _txt(n)))
        else:
            names.append(m.ImportedName("*", None))
        self.p.imports.append(m.ImportRec("es_export_from", source, _line(node), names=names))

    # -- calls, members, identifiers -----------------------------------------

    def _call(self, node, scope: tuple) -> tuple:
        func = node.child_by_field_name("function")
        args = node.child_by_field_name("arguments")
        first = next((c for c in args.named_children), None) if args is not None else None
        enclosing = self._enclosing(scope)
        line = _line(node)
        if func is None:
            return scope
        if func.type == "import":
            module = _string_value(first)
            if module:
                self.p.imports.append(m.ImportRec("dynamic_import", module, line))
            return scope
        if func.type == "identifier":
            name = _txt(func)
            if name == "require":
                module = _string_value(first)
                if module:
                    self.p.imports.append(m.ImportRec(
                        "require", module, line, names=self._require_names(node)))
                return scope
            self.p.calls.append(m.Call(name=name, line=line, enclosing=enclosing))
            if name in ("expect", "assert"):
                self.p.assertions.append(m.Assertion(line, name if name == "expect" else "assert", enclosing))
            return self._maybe_test_block(node, name, None, first, scope)
        if func.type == "member_expression":
            prop = func.child_by_field_name("property")
            obj = func.child_by_field_name("object")
            name = _txt(prop)
            receiver = _compact(_txt(obj), _RECEIVER_LIMIT)
            self.p.calls.append(m.Call(name=name, line=line, receiver=receiver, enclosing=enclosing))
            if receiver == "assert":
                self.p.assertions.append(m.Assertion(line, "assert", enclosing))
            return self._maybe_test_block(node, name, receiver, first, scope)
        return scope

    @staticmethod
    def _require_names(call) -> List[m.ImportedName]:
        """Bindings created by `const x = require(..)` or `const { a, b: c } = require(..)`."""
        parent = call.parent
        if parent is None or parent.type != "variable_declarator":
            return [m.ImportedName(None, None)]
        target = parent.child_by_field_name("name")
        if target is None:
            return [m.ImportedName(None, None)]
        if target.type == "identifier":
            return [m.ImportedName(None, _txt(target))]
        names: List[m.ImportedName] = []
        if target.type == "object_pattern":
            for child in target.children:
                if child.type == "shorthand_property_identifier_pattern":
                    names.append(m.ImportedName(_txt(child), _txt(child)))
                elif child.type == "pair_pattern":
                    key = child.child_by_field_name("key")
                    value = child.child_by_field_name("value")
                    if key is not None and value is not None and value.type == "identifier":
                        names.append(m.ImportedName(_txt(key), _txt(value)))
        return names or [m.ImportedName(None, None)]

    def _maybe_test_block(self, node, name: str, receiver: Optional[str], first, scope: tuple) -> tuple:
        base = receiver if receiver in (_TEST_SUITES | _TEST_CASES) and name in _TEST_MODIFIERS else name
        if receiver is not None and base == name:
            return scope  # some other `obj.method(...)`
        if base not in (_TEST_SUITES | _TEST_CASES):
            return scope
        title = _string_value(first)
        if title is None:
            return scope
        suite = base in _TEST_SUITES
        qual = " > ".join([q for k, q in scope if k in ("suite",)] + [title])
        self.p.test_cases.append(m.TestCase(
            name=title, qualified_name=qual, kind="suite" if suite else "test",
            start_line=_line(node), end_line=_end_line(node), skipped=name in ("skip", "todo"),
        ))
        return scope + ((("suite" if suite else "test"), qual),)

    def _new(self, node, scope: tuple) -> None:
        ctor = node.child_by_field_name("constructor")
        if ctor is None:
            return
        enclosing = self._enclosing(scope)
        if ctor.type == "identifier":
            self.p.calls.append(m.Call(name=_txt(ctor), line=_line(node), enclosing=enclosing, construct=True))
        elif ctor.type == "member_expression":
            prop = ctor.child_by_field_name("property")
            obj = ctor.child_by_field_name("object")
            self.p.calls.append(m.Call(
                name=_txt(prop), line=_line(node), receiver=_compact(_txt(obj), _RECEIVER_LIMIT),
                enclosing=enclosing, construct=True))

    def _member(self, node, scope: tuple) -> None:
        prop = node.child_by_field_name("property")
        obj = node.child_by_field_name("object")
        if prop is None or obj is None:
            return
        receiver = _compact(_txt(obj), _RECEIVER_LIMIT)
        if prop.type in ("property_identifier", "private_property_identifier"):
            self.p.uses.append(m.Use(
                name=_txt(prop), line=_line(prop), kind="attribute", receiver=receiver,
                enclosing=self._enclosing(scope)))
            if receiver in ("process.env", "import.meta.env"):
                self.p.env_refs.append((_txt(prop), _line(prop)))

    def _subscript_env(self, node) -> None:
        obj = node.child_by_field_name("object")
        index = node.child_by_field_name("index")
        if obj is not None and _txt(obj, 30) in ("process.env", "import.meta.env"):
            key = _string_value(index)
            if key:
                self.p.env_refs.append((key, _line(node)))

    def _identifier(self, node, scope: tuple) -> None:
        parent = node.parent
        name = _txt(node)
        if parent is None:
            return
        ptype = parent.type
        if ptype in _DECL_PARENTS and _field_is(node, "name"):
            return
        if ptype in _PARAM_PARENTS and (
            ptype in ("formal_parameters",) or _field_is(node, "pattern") or _field_is(node, "parameter")
        ):
            return
        if ptype in ("import_specifier", "import_clause", "namespace_import", "import_require_clause"):
            return  # recorded through the import record
        if ptype in ("jsx_opening_element", "jsx_self_closing_element", "jsx_closing_element"):
            if name[:1].islower():
                return  # intrinsic HTML element, not a component reference
        kind = "export" if ptype == "export_specifier" else "name"
        ctx = "store" if (ptype == "assignment_expression" and _field_is(node, "left")) else "load"
        self.p.uses.append(m.Use(name=name, line=_line(node), kind=kind, ctx=ctx, enclosing=self._enclosing(scope)))


def parse_js_family(path: str, text: str, language: str) -> m.ParsedFile:
    parsed = m.ParsedFile(path=path, language=language, lines=text.split("\n"))
    try:
        tree = _parser(language).parse(text.encode("utf-8"))
    except Exception as exc:  # the native parser failing must not take the whole analysis down
        parsed.issue = m.ParseIssue(None, f"{type(exc).__name__}: file could not be parsed")
        return parsed
    walker = _Walker(parsed)
    walker.run(tree.root_node)
    if tree.root_node.has_error:
        parsed.issue = m.ParseIssue(walker.error_line, "Syntax error; results for this file are partial", partial=True)
    return parsed


__all__ = ["parse_js_family", "TYPESCRIPT"]

