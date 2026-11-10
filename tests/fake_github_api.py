"""In-memory mock of the parts of the GitHub REST API that the tools use.

It never touches the network. It deliberately models GitHub behaviors that matter for
security: an installation token can read ANY public repository, trees are truncated
for huge repos, and Link headers drive pagination.
"""

import base64
import hashlib
import re
from typing import Any, Dict, List, Optional, Tuple

import httpx

SHA_MAIN = "a" * 40
SHA_OLD = "b" * 40
SHA_FEAT = "c" * 40
BIG_SIZE = 600_000
BRANCHES = {"main": SHA_MAIN, "feature/x": SHA_FEAT}
ANCESTORS = {
    SHA_MAIN: [SHA_MAIN, SHA_OLD],
    SHA_OLD: [SHA_OLD],
    SHA_FEAT: [SHA_FEAT, SHA_MAIN, SHA_OLD],
}


def sha_of(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()


def default_files() -> Dict[str, Optional[bytes]]:
    return {
        "src/app.py": b"def main():\n    raise ConnectionError('db')\n",
        "src/util.py": b"def helper():\n    return 1\n# TODO: ConnectionError retry\n",
        "README.md": b"# Hello\n",
        "bin/logo.png": b"\x89PNG\x00\x01\x02",
        "big.txt": None,  # served with a size but never inline (too large)
        "crlf.txt": b"a\r\nb\r\n",
        "latin1.txt": b"caf\xe9\n",
        "node_modules/dep/index.js": b"ConnectionError in vendored code\n",
    }


def repo_entry(owner: str, name: str, rid: int, private: bool = True) -> Dict[str, Any]:
    return {
        "id": rid,
        "name": name,
        "full_name": f"{owner}/{name}",
        "owner": {"login": owner, "type": "Organization"},
        "default_branch": "main",
        "private": private,
        "visibility": "private" if private else "public",
        "html_url": f"https://github.com/{owner}/{name}",
        "clone_url": f"https://github.com/{owner}/{name}.git",
        "description": "demo",
        "language": "Python",
        "archived": False,
        "fork": False,
        "size": 123,
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
        "pushed_at": "2026-01-02T00:00:00Z",
    }


def commit_obj(sha: str, message: str, parents: List[str]) -> Dict[str, Any]:
    return {
        "sha": sha,
        "html_url": f"https://github.com/acme/webapp/commit/{sha}",
        "commit": {
            "message": message,
            "author": {"name": "Dev Eloper", "email": "dev@example.com", "date": "2026-11-01T10:00:00Z"},
            "committer": {"name": "Dev Eloper", "email": "dev@example.com", "date": "2026-11-01T10:00:00Z"},
        },
        "author": {"login": "developer"},
        "committer": {"login": "developer"},
        "parents": [{"sha": p} for p in parents],
    }


def _resp(status: int, body: Any = None, link: Optional[str] = None, headers: Optional[dict] = None):
    h = dict(headers or {})
    if link:
        h["link"] = link
    return httpx.Response(status, json=body, headers=h)


class FakeGitHubAPI:
    def __init__(self) -> None:
        self.repos: List[Dict[str, Any]] = [
            repo_entry("acme", "webapp", 1),
            repo_entry("acme", "lib", 2),
        ]
        self.public_repos = {"octocat/public-repo": repo_entry("octocat", "public-repo", 99, private=False)}
        self.files = default_files()
        self.commits = {
            SHA_MAIN: commit_obj(SHA_MAIN, "Fix db retry\n\nlong body", [SHA_OLD]),
            SHA_OLD: commit_obj(SHA_OLD, "Initial", []),
            SHA_FEAT: commit_obj(SHA_FEAT, "Feature work", [SHA_MAIN]),
        }
        self.commit_files: Dict[str, List[Dict[str, Any]]] = {
            SHA_MAIN: [
                {"filename": "src/app.py", "status": "modified", "additions": 2, "deletions": 1,
                 "changes": 3, "sha": sha_of("src/app.py"), "patch": "@@ -1 +1,2 @@\n-old\n+new\n+newer"},
            ]
        }
        self.file_history = {"src/app.py": [SHA_MAIN, SHA_OLD]}
        self.truncate_tree = False
        self.reject_tokens: set = set()
        self.overrides: Dict[str, Tuple[int, dict, Any]] = {}
        self.raw_overrides: Dict[str, Tuple[int, bytes]] = {}
        self.raise_on: Optional[Exception] = None
        self.requests: List[Dict[str, Any]] = []

    # -- helpers ------------------------------------------------------------

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def paths(self) -> List[str]:
        return [r["path"] for r in self.requests]

    def _paginate(self, items: List[Any], params: dict) -> Tuple[List[Any], Optional[str]]:
        page, per = int(params.get("page", 1)), int(params.get("per_page", 30))
        chunk = items[(page - 1) * per : page * per]
        has_next = page * per < len(items)
        link = None
        if has_next:
            link = f'<https://api.github.com/x?page={page + 1}&per_page={per}>; rel="next"'
        return chunk, link

    def _dir_entries(self, directory: str, recursive: bool) -> List[Dict[str, Any]]:
        prefix = f"{directory}/" if directory else ""
        entries: List[Dict[str, Any]] = []
        seen_dirs = set()
        for path, content in sorted(self.files.items()):
            if not path.startswith(prefix):
                continue
            rest = path[len(prefix):]
            size = BIG_SIZE if content is None else len(content)
            if recursive:
                parts = rest.split("/")
                for i in range(1, len(parts)):
                    d = "/".join(parts[:i])
                    if d not in seen_dirs:
                        seen_dirs.add(d)
                        entries.append({"path": d, "mode": "040000", "type": "tree", "sha": sha_of("tree:" + prefix + d)})
                entries.append({"path": rest, "mode": "100644", "type": "blob", "sha": sha_of(path), "size": size})
            else:
                head = rest.split("/")[0]
                if "/" in rest:
                    if head not in seen_dirs:
                        seen_dirs.add(head)
                        entries.append({"path": head, "mode": "040000", "type": "tree", "sha": sha_of("tree:" + prefix + head)})
                else:
                    entries.append({"path": rest, "mode": "100644", "type": "blob", "sha": sha_of(path), "size": size})
        return entries

    def _dir_for_tree_sha(self, sha: str) -> Optional[str]:
        if sha in self.commits:
            return ""
        dirs = {""}
        for path in self.files:
            parts = path.split("/")
            for i in range(1, len(parts)):
                dirs.add("/".join(parts[:i]))
        for d in dirs:
            if d and sha_of("tree:" + d) == sha:
                return d
        return None

    # -- request handling ---------------------------------------------------

    def handle(self, request: httpx.Request) -> httpx.Response:
        if self.raise_on is not None:
            raise self.raise_on
        path = request.url.path
        params = dict(request.url.params)
        auth = request.headers.get("authorization", "")
        self.requests.append(
            {"method": request.method, "path": path, "params": params,
             "auth": auth, "host": request.url.host,
             "raw_target": request.url.raw_path.decode()}
        )
        if auth.removeprefix("Bearer ") in self.reject_tokens:
            return _resp(401, {"message": "Bad credentials"})
        for prefix, (status, body) in self.raw_overrides.items():
            if path.startswith(prefix):
                return httpx.Response(status, content=body)
        for prefix, (status, headers, body) in self.overrides.items():
            if path.startswith(prefix):
                return _resp(status, body, headers=headers)

        if path == "/installation/repositories":
            chunk, link = self._paginate(self.repos, params)
            return _resp(200, {"total_count": len(self.repos), "repositories": chunk}, link)

        m = re.fullmatch(r"/repos/([^/]+)/([^/]+)(/.*)?", path)
        if not m:
            return _resp(404, {"message": "Not Found"})
        owner, name, rest = m.group(1), m.group(2), m.group(3) or ""
        full = f"{owner}/{name}".lower()  # GitHub resolves owner/repo case-insensitively
        known = {r["full_name"].lower(): r for r in self.repos}
        repo = known.get(full) or {k.lower(): v for k, v in self.public_repos.items()}.get(full)
        if repo is None:
            return _resp(404, {"message": "Not Found"})

        if rest == "":
            return _resp(200, repo)

        if rest == "/commits":
            sha_param = params.get("sha")
            if sha_param in BRANCHES:
                chain = list(ANCESTORS[BRANCHES[sha_param]])
            elif sha_param in self.commits:
                chain = list(ANCESTORS[sha_param])
            else:
                return _resp(404, {"message": "No commit found for SHA"})
            if "path" in params:
                allowed = self.file_history.get(params["path"], [])
                chain = [c for c in chain if c in allowed]
            chunk, link = self._paginate(chain, params)
            return _resp(200, [self.commits[c] for c in chunk], link)

        m = re.fullmatch(r"/commits/([0-9a-fA-F]+)", rest)
        if m:
            prefix = m.group(1).lower()
            found = next((s for s in self.commits if s.startswith(prefix)), None)
            if not found:
                return _resp(404, {"message": "Not Found"})
            body = dict(self.commits[found])
            files = self.commit_files.get(found, [])
            chunk, link = self._paginate(files, params)
            body["files"] = chunk
            body["stats"] = {"total": 3, "additions": 2, "deletions": 1}
            return _resp(200, body, link)

        m = re.fullmatch(r"/git/trees/([0-9a-fA-F]+)", rest)
        if m:
            directory = self._dir_for_tree_sha(m.group(1).lower())
            if directory is None:
                return _resp(404, {"message": "Not Found"})
            entries = self._dir_entries(directory, params.get("recursive") == "1")
            return _resp(200, {"sha": m.group(1), "tree": entries, "truncated": self.truncate_tree})

        m = re.fullmatch(r"/git/blobs/([0-9a-fA-F]+)", rest)
        if m:
            for path_, content in self.files.items():
                if sha_of(path_) == m.group(1).lower() and content is not None:
                    return _resp(200, {"sha": m.group(1), "size": len(content), "encoding": "base64",
                                       "content": base64.encodebytes(content).decode()})
            return _resp(404, {"message": "Not Found"})

        if rest.startswith("/contents/"):
            file_path = rest[len("/contents/"):]
            if file_path in self.files:
                content = self.files[file_path]
                if content is None:
                    return _resp(200, {"type": "file", "size": BIG_SIZE, "encoding": "none",
                                       "content": "", "sha": sha_of(file_path)})
                return _resp(200, {"type": "file", "size": len(content), "encoding": "base64",
                                   "sha": sha_of(file_path), "content": base64.encodebytes(content).decode()})
            if any(p.startswith(file_path + "/") for p in self.files):
                return _resp(200, [{"type": "file", "path": "x"}])
            return _resp(404, {"message": "Not Found"})

        return _resp(404, {"message": "Not Found"})


class TokenSource:
    """Stands in for the App token service; records refresh behavior."""

    def __init__(self, tokens: Optional[List[str]] = None):
        self.tokens = tokens or ["ghs_installation_token_AAAA1111BBBB2222"]
        self.calls: List[bool] = []

    def __call__(self, settings, installation_id, force_refresh: bool = False):
        self.calls.append(force_refresh)

        class _Tok:
            pass

        tok = _Tok()
        index = min(len(self.calls) - 1, len(self.tokens) - 1)
        tok.token = self.tokens[index]
        return tok
