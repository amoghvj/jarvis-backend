"""GitHub App API client and on-demand installation access-token management."""

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import httpx

from app.integrations.github.config import GitHubAppSettings
from app.integrations.github.jwt_service import generate_app_jwt

API_BASE = "https://api.github.com"
API_VERSION = "2022-11-28"
TIMEOUT = 10.0
EXPIRY_BUFFER_SECONDS = 60


class GitHubAppApiError(Exception):
    """Base error for GitHub App API failures."""


class GitHubRateLimitError(GitHubAppApiError):
    """GitHub API rate limit exceeded."""


class GitHubInstallationRevokedError(GitHubAppApiError):
    """The GitHub App installation was uninstalled, suspended, or is inaccessible."""


@dataclass(frozen=True)
class InstallationToken:
    token: str
    expires_at: datetime
    expires_at_epoch: float
    permissions: Dict[str, Any]
    repository_selection: Optional[str] = None


# Ephemeral per-process optimization cache; serverless correctness does not rely on it.
_TOKEN_CACHE: Dict[int, InstallationToken] = {}


def clear_token_cache(installation_id: Optional[int] = None) -> None:
    if installation_id is None:
        _TOKEN_CACHE.clear()
    else:
        _TOKEN_CACHE.pop(installation_id, None)


def _parse_iso8601(value: str) -> datetime:
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    dt = datetime.fromisoformat(normalized)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _headers(bearer_token: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {bearer_token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": API_VERSION,
    }


def _check_error_response(resp: httpx.Response, context: str) -> None:
    if resp.status_code in (200, 201):
        return
    if resp.status_code == 429 or (
        resp.status_code == 403 and resp.headers.get("x-ratelimit-remaining") == "0"
    ):
        raise GitHubRateLimitError(f"GitHub rate limit exceeded during {context}")
    if resp.status_code == 404:
        raise GitHubInstallationRevokedError(f"GitHub installation not found during {context}")
    if resp.status_code == 403:
        raise GitHubInstallationRevokedError(
            f"GitHub installation access forbidden or suspended during {context}"
        )
    if resp.status_code == 401:
        raise GitHubAppApiError(f"GitHub authentication rejected during {context}")
    raise GitHubAppApiError(f"GitHub API error (HTTP {resp.status_code}) during {context}")


def create_installation_access_token(
    settings: GitHubAppSettings, installation_id: int
) -> InstallationToken:
    """Generate an App JWT and exchange it for a short-lived installation access token."""
    if not isinstance(installation_id, int) or installation_id <= 0:
        raise GitHubAppApiError("Invalid installation_id")

    app_jwt = generate_app_jwt(settings)
    url = f"{API_BASE}/app/installations/{installation_id}/access_tokens"
    try:
        resp = httpx.post(url, headers=_headers(app_jwt), timeout=TIMEOUT)
    except httpx.HTTPError as exc:
        raise GitHubAppApiError("Could not reach GitHub to create installation token") from exc

    _check_error_response(resp, "installation token creation")

    try:
        data = resp.json()
    except ValueError as exc:
        raise GitHubAppApiError("Invalid JSON from GitHub installation token endpoint") from exc

    token = data.get("token") if isinstance(data, dict) else None
    expires_raw = data.get("expires_at") if isinstance(data, dict) else None
    if not isinstance(token, str) or not token or not isinstance(expires_raw, str):
        raise GitHubAppApiError("Malformed installation token response from GitHub")

    expires_dt = _parse_iso8601(expires_raw)
    inst_token = InstallationToken(
        token=token,
        expires_at=expires_dt,
        expires_at_epoch=expires_dt.timestamp(),
        permissions=data.get("permissions") or {},
        repository_selection=data.get("repository_selection"),
    )
    _TOKEN_CACHE[installation_id] = inst_token
    return inst_token


def get_installation_access_token(
    settings: GitHubAppSettings,
    installation_id: int,
    *,
    force_refresh: bool = False,
) -> InstallationToken:
    """Return a valid installation access token, refreshing when missing or near expiry."""
    if not force_refresh:
        cached = _TOKEN_CACHE.get(installation_id)
        if cached and (cached.expires_at_epoch - time.time()) > EXPIRY_BUFFER_SECONDS:
            return cached
    return create_installation_access_token(settings, installation_id)


def installation_request(
    settings: GitHubAppSettings,
    installation_id: int,
    method: str,
    path: str,
    **kwargs: Any,
) -> httpx.Response:
    """Perform an authenticated GitHub API request on behalf of an installation."""
    inst_token = get_installation_access_token(settings, installation_id)
    url = path if path.startswith("https://") else f"{API_BASE}/{path.lstrip('/')}"
    extra_headers = kwargs.pop("headers", None) or {}

    try:
        resp = httpx.request(
            method,
            url,
            headers={**_headers(inst_token.token), **extra_headers},
            timeout=TIMEOUT,
            **kwargs,
        )
        if resp.status_code == 401:
            # Token may have been revoked early; refresh once and retry.
            clear_token_cache(installation_id)
            inst_token = get_installation_access_token(
                settings, installation_id, force_refresh=True
            )
            resp = httpx.request(
                method,
                url,
                headers={**_headers(inst_token.token), **extra_headers},
                timeout=TIMEOUT,
                **kwargs,
            )
    except httpx.HTTPError as exc:
        raise GitHubAppApiError("Could not reach GitHub API") from exc

    _check_error_response(resp, f"{method.upper()} {path}")
    return resp

