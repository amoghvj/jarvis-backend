"""Read-only GitHub REST client for tools.

Only GET requests are issued, only to api.github.com, and every URL is built from
validated segments. Installation tokens come from the existing App token service
and stay inside this module: they are never returned, logged, or put in results.
"""

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, quote, urlsplit

import httpx

from app.integrations.github import client as app_tokens
from app.integrations.github.config import GitHubAppSettings
from app.integrations.github.jwt_service import GitHubAppJwtError
from app.tools.errors import ToolError
from app.tools.schemas import ToolContext

API_BASE = "https://api.github.com"
API_VERSION = "2022-11-28"
REQUEST_TIMEOUT = 10.0
MAX_RESPONSE_BYTES = 5 * 1024 * 1024

# Test hook only: lets tests route requests to a mock without real network access.
_TRANSPORT: Optional[httpx.BaseTransport] = None


@dataclass(frozen=True)
class GitHubResponse:
    status: int
    data: Any
    has_next: bool = False
    next_page: Optional[int] = None


def repo_path(owner: str, repo: str, *segments: str) -> str:
    """Build `/repos/{owner}/{repo}/...`, percent-encoding every segment."""
    parts = ["repos", owner, repo, *segments]
    return "/" + "/".join(quote(p, safe="") for p in parts)


def path_segments(path: str) -> list:
    return [s for s in path.split("/") if s]


def _retry_after(headers: httpx.Headers) -> Optional[int]:
    value = headers.get("retry-after")
    if value and value.isdigit():
        return int(value)
    reset = headers.get("x-ratelimit-reset")
    if reset and reset.isdigit():
        return max(0, int(reset) - int(time.time()))
    return None


def _raise_for_status(status: int, headers: httpx.Headers, data: Any) -> None:
    if 200 <= status < 300:
        return
    if status in (301, 302, 307, 308):
        raise ToolError("not_found", "The repository or resource was moved or renamed")
    if status == 401:
        raise ToolError("github_auth_error", "GitHub rejected the installation credentials")
    message = str(data.get("message", "")).lower() if isinstance(data, dict) else ""
    if status == 429 or (
        status == 403
        and (
            headers.get("x-ratelimit-remaining") == "0"
            or "retry-after" in headers
            or "rate limit" in message
        )
    ):
        wait = _retry_after(headers)
        suffix = f"; retry after about {wait}s" if wait is not None else ""
        raise ToolError("github_rate_limited", f"GitHub rate limit reached{suffix}", retryable=True)
    if status == 403:
        raise ToolError("github_forbidden", "GitHub denied access to this resource")
    if status in (404, 410):
        raise ToolError("not_found", "Not found, or not accessible through this installation")
    if status == 409:
        raise ToolError("repository_empty", "The repository is empty or in a conflicting state")
    if status == 422:
        raise ToolError("invalid_request", "GitHub rejected the request parameters")
    if status >= 500:
        raise ToolError("github_unavailable", "GitHub is temporarily unavailable", retryable=True)
    raise ToolError("github_error", f"GitHub returned an unexpected status ({status})")


class GitHubRepoClient:
    def __init__(self, ctx: ToolContext, settings: GitHubAppSettings, installation_id: int):
        self._ctx = ctx
        self._settings = settings
        self._installation_id = installation_id

    def _token(self, force: bool = False) -> str:
        try:
            return app_tokens.get_installation_access_token(
                self._settings, self._installation_id, force_refresh=force
            ).token
        except app_tokens.GitHubRateLimitError:
            raise ToolError("github_rate_limited", "GitHub rate limit reached", retryable=True)
        except app_tokens.GitHubInstallationRevokedError:
            raise ToolError(
                "installation_unavailable",
                "The GitHub App installation is suspended, uninstalled or no longer accessible",
            )
        except (app_tokens.GitHubAppApiError, GitHubAppJwtError):
            raise ToolError("github_auth_error", "Could not obtain GitHub installation credentials")

    def _send(self, path: str, params: Optional[Dict[str, Any]], token: str) -> GitHubResponse:
        if not path.startswith("/") or "://" in path:
            raise ToolError("invalid_request", "Invalid GitHub API path")
        remaining = self._ctx.time_remaining()
        if remaining <= 0:
            raise ToolError("timeout", "The tool exceeded its time limit", retryable=True)
        self._ctx.count_request()
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": API_VERSION,
        }
        try:
            with httpx.Client(
                transport=_TRANSPORT, timeout=min(REQUEST_TIMEOUT, remaining), follow_redirects=False
            ) as http:
                with http.stream("GET", API_BASE + path, params=params, headers=headers) as resp:
                    body = bytearray()
                    for chunk in resp.iter_bytes():
                        body.extend(chunk)
                        if len(body) > MAX_RESPONSE_BYTES:
                            raise ToolError(
                                "github_response_too_large",
                                "GitHub's response exceeded the size limit; narrow the request",
                            )
                    status, resp_headers = resp.status_code, resp.headers
                    next_link = resp.links.get("next")
        except ToolError:
            raise
        except httpx.HTTPError:
            raise ToolError("github_unavailable", "Could not reach GitHub", retryable=True)

        data: Any = None
        if body:
            try:
                data = json.loads(bytes(body))
            except ValueError:
                if 200 <= status < 300:
                    raise ToolError("github_bad_response", "GitHub returned invalid JSON")
        _raise_for_status(status, resp_headers, data)

        next_page = None
        if next_link:
            values = parse_qs(urlsplit(next_link.get("url", "")).query).get("page")
            if values and values[0].isdigit():
                next_page = int(values[0])
        return GitHubResponse(status, data, has_next=bool(next_link), next_page=next_page)

    def get(self, path: str, params: Optional[Dict[str, Any]] = None) -> GitHubResponse:
        token = self._token()
        try:
            return self._send(path, params, token)
        except ToolError as exc:
            if exc.code != "github_auth_error":
                raise
        # The cached token may have been revoked or expired early: refresh exactly once.
        return self._send(path, params, self._token(force=True))

