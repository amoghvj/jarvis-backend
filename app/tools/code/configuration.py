"""`inspect_configuration`: allowlisted, redacted inspection of configuration files.

Security model:
* Only files matching an explicit allowlist are ever fetched.
* Secret-bearing files (`.env`, `.env.local`, private keys, credential stores, ...) are
  never fetched; they are listed by path only, with the reason.
* Allowlisted files are still scanned: every value passes through `redaction.safe_value`
  / `scrub_text`, so a real secret pasted into `.env.example` is not returned.
"""

import json
import posixpath
import re
from typing import Any, Dict, Iterator, List, Optional, Tuple

from pydantic import BaseModel, Field

from app.tools.code import model as m
from app.tools.code.common import AnalysisResult, CodeAnalysisInput, ParseErrorInfo
from app.tools.code.js_parser import parse_js_family
from app.tools.code.languages import detect_language
from app.tools.code.redaction import REDACTED, safe_value, scrub_text
from app.tools.code.source import AnalysisSession
from app.tools.schemas import ToolConstraints, ToolContext, ToolDefinition

MAX_ITEMS_PER_FILE = 200
MAX_EXCLUDED_LISTED = 50
_ENV_TEMPLATES = {".env.example", ".env.sample", ".env.template"}
_BUNDLER = re.compile(r"(vite|next|nuxt|webpack|rollup|astro|svelte)\.config\.(js|mjs|cjs|ts|mts|cts)$")
_SECRET_SUFFIXES = (".pem", ".key", ".p12", ".pfx", ".keystore", ".jks", ".tfvars", ".htpasswd")
_SECRET_NAMES = (
    "id_rsa", "id_dsa", "id_ecdsa", "id_ed25519", ".npmrc", ".pypirc", ".netrc", ".git-credentials",
    "credentials", "secrets", "service-account", "serviceaccount", ".dockercfg",
)
_REQ_LINE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)\s*(\[[^\]]*\])?\s*([^;#]*)")
_PEP508 = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(\[[^\]]*\])?\s*([^;]*)")
_ENV_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=(.*)$")
_COMMENTED_ENV = re.compile(r"^\s*#\s*(?:export\s+)?([A-Z][A-Z0-9_]{2,})\s*=(.*)$")
_TIMEOUT = ToolConstraints(timeout_seconds=28.0)


# --- classification ----------------------------------------------------------


def secret_reason(path: str) -> Optional[str]:
    base = posixpath.basename(path).lower()
    if base in _ENV_TEMPLATES:
        return None
    if base == ".env" or base.startswith(".env.") or base.endswith(".env"):
        return "environment file that may contain real secrets; never retrieved"
    if base.endswith(_SECRET_SUFFIXES):
        return "private key / credential file type; never retrieved"
    if any(base == n or base.startswith(n + ".") or base.startswith(n + "-") for n in _SECRET_NAMES):
        return "credential-bearing file name; never retrieved"
    return None


def config_kind(path: str) -> Optional[str]:
    base = posixpath.basename(path).lower()
    if secret_reason(path):
        return None
    if base in _ENV_TEMPLATES:
        return "env_template"
    if base == "pyproject.toml":
        return "pyproject"
    if base == "package.json":
        return "package_json"
    if base == "requirements.txt" or (base.startswith("requirements") and base.endswith(".txt")):
        return "requirements"
    if base.endswith(".txt") and "requirements" in path.lower().split("/")[:-1]:
        return "requirements"
    if (base.startswith("tsconfig") and base.endswith(".json")) or base == "jsconfig.json":
        return "tsconfig"
    if base == "vercel.json":
        return "vercel_json"
    if _BUNDLER.fullmatch(base):
        return "bundler_config"
    return None


# --- output models -----------------------------------------------------------


class ConfigKey(BaseModel):
    key: str
    line: Optional[int] = None
    value: Optional[str] = None
    value_redacted: bool = False
    commented_out: bool = False


class EnvVarRef(BaseModel):
    name: str
    line: Optional[int] = None
    source: str  # env_template | config_file | source_code


class DependencyDecl(BaseModel):
    name: str
    spec: Optional[str] = None
    group: str
    line: Optional[int] = None
    spec_redacted: bool = False


class ConfigFileInfo(BaseModel):
    path: str
    kind: str
    size_bytes: int
    keys: List[ConfigKey] = []
    env_vars: List[EnvVarRef] = []
    dependencies: List[DependencyDecl] = []
    redaction_count: int = 0
    truncated: bool = False
    parse_error: Optional[str] = None


class ExcludedFile(BaseModel):
    path: str
    reason: str


class EnvVarSummary(BaseModel):
    name: str
    declared_in: List[str]
    referenced_in: List[str]


class InspectConfigurationInput(CodeAnalysisInput):
    include_source_env_refs: bool = Field(
        default=False,
        description="Also scan source files for environment variable reads (os.environ, process.env, ...)",
    )


class InspectConfigurationOutput(AnalysisResult):
    files: List[ConfigFileInfo]
    excluded_files: List[ExcludedFile]
    env_vars: List[EnvVarSummary]
    total_redactions: int


# --- parsing helpers ---------------------------------------------------------


def strip_jsonc(text: str) -> str:
    """Remove // and /* */ comments and trailing commas (tsconfig-style JSON) outside strings."""
    out: List[str] = []
    i, n = 0, len(text)
    in_string = False
    while i < n:
        c = text[i]
        if in_string:
            out.append(c)
            if c == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 1
            elif c == '"':
                in_string = False
        elif c == '"':
            in_string = True
            out.append(c)
        elif text.startswith("//", i):
            while i < n and text[i] != "\n":
                i += 1
            continue
        elif text.startswith("/*", i):
            end = text.find("*/", i + 2)
            i = n if end == -1 else end + 2
            continue
        else:
            out.append(c)
        i += 1
    return re.sub(r",(\s*[}\]])", r"\1", "".join(out))


def _key_line(lines: List[str], key: str, start: int = 0) -> Optional[int]:
    pattern = re.compile(r'^\s*"' + re.escape(key) + r'"\s*:')
    for i in range(start, len(lines)):
        if pattern.match(lines[i]):
            return i + 1
    return None


def _text_line(lines: List[str], needle: str, start: int = 0) -> Optional[int]:
    for i in range(start, len(lines)):
        if needle in lines[i]:
            return i + 1
    return None


def _flatten(obj: Any, prefix: str = "", depth: int = 0) -> Iterator[Tuple[str, Any]]:
    if isinstance(obj, dict) and depth < 3:
        for k, v in obj.items():
            yield from _flatten(v, f"{prefix}.{k}" if prefix else str(k), depth + 1)
    elif isinstance(obj, list):
        if all(isinstance(x, (str, int, float, bool)) for x in obj) and len(obj) <= 12:
            yield prefix, ", ".join(str(x) for x in obj)
        else:
            yield prefix, f"[{len(obj)} items]"
    elif isinstance(obj, dict):
        yield prefix, f"{{{len(obj)} keys}}"
    else:
        yield prefix, obj


class _Collector:
    def __init__(self) -> None:
        self.info_keys: List[ConfigKey] = []
        self.env_vars: List[EnvVarRef] = []
        self.deps: List[DependencyDecl] = []
        self.redactions = 0
        self.truncated = False

    def key(self, key: str, value: Optional[Any], line: Optional[int], commented: bool = False) -> None:
        if len(self.info_keys) >= MAX_ITEMS_PER_FILE:
            self.truncated = True
            return
        if value is None:
            self.info_keys.append(ConfigKey(key=key, line=line, commented_out=commented))
            return
        shown, redacted = safe_value(key.rsplit(".", 1)[-1], str(value))
        if shown is not None:
            shown, scrubbed = scrub_text(shown)
            redacted = redacted or scrubbed
        self.redactions += 1 if redacted else 0
        self.info_keys.append(ConfigKey(key=key, line=line, value=shown, value_redacted=redacted, commented_out=commented))

    def env(self, name: str, line: Optional[int], source: str) -> None:
        if len(self.env_vars) < MAX_ITEMS_PER_FILE:
            self.env_vars.append(EnvVarRef(name=name, line=line, source=source))
        else:
            self.truncated = True

    def dep(self, name: str, spec: Optional[str], group: str, line: Optional[int]) -> None:
        if len(self.deps) >= 300:
            self.truncated = True
            return
        redacted = False
        if spec:
            spec, redacted = scrub_text(spec)
            if redacted:
                self.redactions += 1
        self.deps.append(DependencyDecl(name=name, spec=spec or None, group=group, line=line, spec_redacted=redacted))


def _parse_env_template(text: str, c: _Collector) -> None:
    for number, raw in enumerate(text.split("\n"), start=1):
        raw = raw.rstrip("\r")
        match = _ENV_LINE.match(raw)
        commented = False
        if not match:
            match = _COMMENTED_ENV.match(raw)
            commented = True
        if not match:
            continue
        key, value = match.group(1), match.group(2).strip()
        if not value.startswith(("'", '"')):
            value = re.split(r"\s+#", value, maxsplit=1)[0].strip()
        c.env(key, number, "env_template")
        c.key(key, value, number, commented)


def _parse_requirements(text: str, c: _Collector) -> None:
    for number, raw in enumerate(text.split("\n"), start=1):
        line = raw.strip().rstrip("\r")
        if not line or line.startswith("#"):
            continue
        if line.startswith("-"):
            c.key("option", line.split("#", 1)[0].strip(), number)  # value is scrubbed (index URLs etc.)
            continue
        if "://" in line.split("#", 1)[0] and "@" in line:
            c.dep(line.split("#", 1)[0].strip().split("@")[0].strip() or "url-dependency", line, "requirements", number)
            continue
        match = _REQ_LINE.match(line)
        if match:
            c.dep(match.group(1), (match.group(3) or "").strip() or None, "requirements", number)


def _toml_loads(text: str) -> Any:
    try:
        import tomllib  # Python 3.11+
    except ImportError:
        try:
            import tomli as tomllib
        except ImportError:
            raise ValueError("no TOML parser available")
    return tomllib.loads(text)


def _parse_pyproject(text: str, lines: List[str], c: _Collector) -> None:
    data = _toml_loads(text)
    project = data.get("project", {}) if isinstance(data, dict) else {}
    for k, v in _flatten({k: v for k, v in project.items() if k not in ("dependencies", "optional-dependencies")}, "project"):
        c.key(k, v, _text_line(lines, k.split(".")[-1]))
    for group, specs in [("project", project.get("dependencies", []))] + [
        (f"optional:{g}", s) for g, s in (project.get("optional-dependencies") or {}).items()
    ]:
        for spec in specs if isinstance(specs, list) else []:
            match = _PEP508.match(str(spec))
            if match:
                c.dep(match.group(1), (match.group(3) or "").strip() or None, group, _text_line(lines, str(spec)))
    build = data.get("build-system", {}) if isinstance(data, dict) else {}
    for spec in build.get("requires", []) if isinstance(build, dict) else []:
        match = _PEP508.match(str(spec))
        if match:
            c.dep(match.group(1), (match.group(3) or "").strip() or None, "build-system", _text_line(lines, str(spec)))
    if isinstance(build, dict) and "build-backend" in build:
        c.key("build-system.build-backend", build["build-backend"], _text_line(lines, "build-backend"))
    for k, v in _flatten(data.get("tool", {}) if isinstance(data, dict) else {}, "tool"):
        c.key(k, v, _text_line(lines, k.split(".")[-1]))


def _parse_package_json(text: str, lines: List[str], c: _Collector) -> None:
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("package.json is not an object")
    for group in ("dependencies", "devDependencies", "peerDependencies", "optionalDependencies"):
        section = data.get(group)
        start = (_key_line(lines, group) or 1) - 1
        if isinstance(section, dict):
            for name, spec in section.items():
                c.dep(str(name), str(spec), group, _key_line(lines, str(name), start))
    scripts = data.get("scripts")
    start = (_key_line(lines, "scripts") or 1) - 1
    if isinstance(scripts, dict):
        for name, command in scripts.items():
            c.key(f"scripts.{name}", str(command), _key_line(lines, str(name), start))
    skip = {"dependencies", "devDependencies", "peerDependencies", "optionalDependencies", "scripts"}
    for k, v in _flatten({k: v for k, v in data.items() if k not in skip}):
        c.key(k, v, _key_line(lines, k.split(".")[0]))


def _parse_tsconfig(text: str, lines: List[str], c: _Collector) -> None:
    data = json.loads(strip_jsonc(text))
    if not isinstance(data, dict):
        raise ValueError("tsconfig is not an object")
    for k, v in _flatten(data):
        c.key(k, v, _key_line(lines, k.split(".")[-1]))


def _parse_vercel(text: str, lines: List[str], c: _Collector) -> None:
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("vercel.json is not an object")
    for scope, obj in (("env", data.get("env")), ("build.env", (data.get("build") or {}).get("env"))):
        if isinstance(obj, dict):
            for name in obj:
                c.env(str(name), _key_line(lines, str(name)), "config_file")  # values are never shown
                c.key(f"{scope}.{name}", None, _key_line(lines, str(name)))
    rest = {k: v for k, v in data.items() if k != "env"}
    if isinstance(rest.get("build"), dict):
        rest["build"] = {k: v for k, v in rest["build"].items() if k != "env"}
    for k, v in _flatten(rest):
        c.key(k, v, _key_line(lines, k.split(".")[-1]))


def _parse_bundler(path: str, text: str, c: _Collector) -> Optional[str]:
    language = detect_language(path)
    if language is None:
        return None
    parsed: m.ParsedFile = parse_js_family(path, text, language)
    for name, line in parsed.env_refs:
        c.env(name, line, "config_file")
    for rec in parsed.imports:
        c.dep(rec.module, None, "imports", rec.line)
    return parsed.issue.message if parsed.issue else None


def _inspect_file(path: str, kind: str, text: str) -> Tuple[ConfigFileInfo, Optional[str]]:
    c = _Collector()
    lines = text.split("\n")
    error: Optional[str] = None
    try:
        if kind == "env_template":
            _parse_env_template(text, c)
        elif kind == "requirements":
            _parse_requirements(text, c)
        elif kind == "pyproject":
            _parse_pyproject(text, lines, c)
        elif kind == "package_json":
            _parse_package_json(text, lines, c)
        elif kind == "tsconfig":
            _parse_tsconfig(text, lines, c)
        elif kind == "vercel_json":
            _parse_vercel(text, lines, c)
        elif kind == "bundler_config":
            error = _parse_bundler(path, text, c)
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        error = f"{type(exc).__name__}: could not parse this configuration file"
        if isinstance(exc, ValueError) and "TOML" in str(exc):
            error = "No TOML parser is available on the server; pyproject.toml was not inspected"
    info = ConfigFileInfo(
        path=path, kind=kind, size_bytes=len(text.encode("utf-8")), keys=c.info_keys, env_vars=c.env_vars,
        dependencies=c.deps, redaction_count=c.redactions, truncated=c.truncated, parse_error=error,
    )
    return info, error


def inspect_configuration(ctx: ToolContext, inp: InspectConfigurationInput) -> InspectConfigurationOutput:
    session = AnalysisSession(ctx, inp)
    prefix = inp.path_prefix or ""

    def in_scope(p: str) -> bool:
        return not prefix or p == prefix or p.startswith(prefix + "/")

    excluded: List[ExcludedFile] = []
    wanted: List[Tuple[str, str]] = []
    for path in session.all_paths():
        if not in_scope(path):
            continue
        reason = secret_reason(path)
        if reason:
            excluded.append(ExcludedFile(path=path, reason=reason))
            continue
        kind = config_kind(path)
        if kind:
            wanted.append((path, kind))
    wanted.sort(key=lambda w: (w[0].count("/"), w[0]))
    if len(wanted) > inp.max_files:
        session.limits.add("max_files")
    files: List[ConfigFileInfo] = []
    for path, kind in wanted[: inp.max_files]:
        text = session.read(path)
        if text is None:
            continue
        info, error = _inspect_file(path, kind, text)
        if error:
            session.parse_errors.append(ParseErrorInfo(path=path, message=error))
        if info.truncated:
            session.limits.add("per_file_items")
        files.append(info)

    summary: Dict[str, Dict[str, List[str]]] = {}
    for f in files:
        for ref in f.env_vars:
            slot = summary.setdefault(ref.name, {"declared": [], "referenced": []})
            where = f"{f.path}:{ref.line}" if ref.line else f.path
            slot["declared" if ref.source == "env_template" else "referenced"].append(where)
    if inp.include_source_env_refs:
        for parsed in session.scan(session.source_candidates(prefix), inp.max_files):
            for name, line in parsed.env_refs:
                slot = summary.setdefault(name, {"declared": [], "referenced": []})
                slot["referenced"].append(f"{parsed.path}:{line}")
    env_summary = [
        EnvVarSummary(name=n, declared_in=sorted(set(v["declared"]))[:10], referenced_in=sorted(set(v["referenced"]))[:10])
        for n, v in sorted(summary.items())
    ][:500]

    total = sum(f.redaction_count for f in files)
    extra = [
        "Only allowlisted configuration files are read. Secret-bearing files (.env, .env.*, private keys, "
        "credential files) are never retrieved; they are listed in excluded_files by path only.",
        f"Values that look like credentials, or belong to sensitive-looking keys, are replaced with {REDACTED}; "
        "`.env.example`-style files are scanned too and are not assumed to be safe.",
    ]
    if not files:
        extra.append("No supported configuration files were found in scope.")
    return InspectConfigurationOutput(
        files=files, excluded_files=excluded[:MAX_EXCLUDED_LISTED], env_vars=env_summary, total_redactions=total,
        **session.result_fields(extra),
    )


TOOLS = [
    ToolDefinition(
        name="inspect_configuration",
        description=(
            "Inspect allowlisted configuration files (.env.example/.sample/.template, pyproject.toml, "
            "requirements*.txt, package.json, tsconfig*.json, vite/next/webpack config, vercel.json) at a pinned "
            "commit: keys with line numbers, dependency declarations and environment-variable references. "
            "Never retrieves .env or key files; credential-like values are redacted."
        ),
        input_model=InspectConfigurationInput,
        output_model=InspectConfigurationOutput,
        handler=inspect_configuration,
        constraints=_TIMEOUT,
    ),
]
