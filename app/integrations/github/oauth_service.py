"""GitHub App user-authorization (OAuth) flow, state signing, and user installation lookup."""

import hashlib
import hmac
import secrets
import time
import urllib.parse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from uuid import UUID

import httpx
from fastapi import Response

from app.integrations.github.config import PENDING_FLOW_TTL_SECONDS, GitHubAppSettings

AUTHORIZE_URL = "https://github.com/login/oauth/authorize"
TOKEN_URL = "https://github.com/login/oauth/access_token"
API_BASE = "https://api.github.com"
API_VERSION = "2022-11-28"
TIMEOUT = 10.0

INSTALL_FLOW_COOKIE = "jarvis_gh_install_flow"
OAUTH_STATE_COOKIE = "jarvis_gh_oauth_state"
COOKIE_PATH = "/integrations/github"


class GitHubAppOAuthError(Exception):
    """GitHub App OAuth exchange or user API request failed."""


@dataclass(frozen=True)
class GitHubAppUser:
    github_id: int
    login: str


def generate_random_token() -> str:
    return secrets.token_urlsafe(32)


def hash_state_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def build_app_install_url(settings: GitHubAppSettings) -> str:
    slug = urllib.parse.quote(settings.app_slug.strip("/"), safe="")
    return f"https://github.com/apps/{slug}/installations/new"


def build_app_oauth_authorize_url(settings: GitHubAppSettings, state: str) -> str:
    query = urllib.parse.urlencode(
        {
            "client_id": settings.client_id,
            "redirect_uri": settings.callback_url,
            "state": state,
        }
    )
    return f"{AUTHORIZE_URL}?{query}"


def _sign(settings: GitHubAppSettings, message: str) -> str:
    return hmac.new(
        settings.session_secret.encode("utf-8"),
        message.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def set_install_flow_cookie(
    response: Response, settings: GitHubAppSettings, flow_token: str, user_id: UUID
) -> None:
    expires = int(time.time()) + PENDING_FLOW_TTL_SECONDS
    payload = f"{flow_token}.{user_id}.{expires}"
    sig = _sign(settings, f"gh-install-flow:{payload}")
    response.set_cookie(
        INSTALL_FLOW_COOKIE,
        f"{payload}.{sig}",
        max_age=PENDING_FLOW_TTL_SECONDS,
        httponly=True,
        secure=settings.cookie_secure,
        samesite="lax",
        path=COOKIE_PATH,
    )


def verify_install_flow_cookie(
    settings: GitHubAppSettings, cookie_value: Optional[str], expected_user_id: UUID
) -> Optional[str]:
    if not cookie_value:
        return None
    parts = cookie_value.split(".")
    if len(parts) != 4:
        return None
    flow_token, uid_str, expires_str, sig = parts
    expected_sig = _sign(settings, f"gh-install-flow:{flow_token}.{uid_str}.{expires_str}")
    if not hmac.compare_digest(sig, expected_sig):
        return None
    if not expires_str.isdigit() or int(expires_str) < time.time():
        return None
    if uid_str != str(expected_user_id):
        return None
    return flow_token


def set_oauth_state_cookie(
    response: Response,
    settings: GitHubAppSettings,
    state: str,
    user_id: UUID,
    installation_id: int,
) -> None:
    expires = int(time.time()) + PENDING_FLOW_TTL_SECONDS
    payload = f"{state}.{user_id}.{installation_id}.{expires}"
    sig = _sign(settings, f"gh-app-state:{payload}")
    response.set_cookie(
        OAUTH_STATE_COOKIE,
        f"{payload}.{sig}",
        max_age=PENDING_FLOW_TTL_SECONDS,
        httponly=True,
        secure=settings.cookie_secure,
        samesite="lax",
        path=COOKIE_PATH,
    )


def verify_oauth_state_cookie(
    settings: GitHubAppSettings,
    state_param: Optional[str],
    cookie_value: Optional[str],
) -> Optional[tuple[UUID, int]]:
    """Verify the signed browser cookie matches `state_param` and has not expired.

    Returns `(user_id, installation_id)` if valid, else `None`.
    """
    if not state_param or not cookie_value:
        return None
    parts = cookie_value.split(".")
    if len(parts) != 5:
        return None
    state, uid_str, inst_str, expires_str, sig = parts
    expected_sig = _sign(
        settings, f"gh-app-state:{state}.{uid_str}.{inst_str}.{expires_str}"
    )
    if not hmac.compare_digest(sig, expected_sig):
        return None
    if not hmac.compare_digest(state, state_param):
        return None
    if not expires_str.isdigit() or int(expires_str) < time.time():
        return None
    if not inst_str.isdigit():
        return None
    try:
        return UUID(uid_str), int(inst_str)
    except ValueError:
        return None


def clear_flow_cookies(response: Response, settings: GitHubAppSettings) -> None:
    for name in (INSTALL_FLOW_COOKIE, OAUTH_STATE_COOKIE):
        response.delete_cookie(
            name,
            httponly=True,
            secure=settings.cookie_secure,
            samesite="lax",
            path=COOKIE_PATH,
        )


def exchange_code_for_user_token(settings: GitHubAppSettings, code: str) -> str:
    """Exchange a GitHub App OAuth authorization code for a user access token."""
    try:
        resp = httpx.post(
            TOKEN_URL,
            data={
                "client_id": settings.client_id,
                "client_secret": settings.client_secret,
                "code": code,
                "redirect_uri": settings.callback_url,
            },
            headers={"Accept": "application/json"},
            timeout=TIMEOUT,
        )
        data = resp.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise GitHubAppOAuthError(
            "Could not reach GitHub to exchange the authorization code"
        ) from exc

    token = data.get("access_token") if isinstance(data, dict) else None
    if resp.status_code != 200 or not isinstance(token, str) or not token:
        raise GitHubAppOAuthError("GitHub rejected the authorization code")
    return token


def _user_headers(user_access_token: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {user_access_token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": API_VERSION,
    }


def fetch_authenticated_user(user_access_token: str) -> GitHubAppUser:
    """Retrieve the authenticated GitHub user via GET /user."""
    try:
        resp = httpx.get(
            f"{API_BASE}/user", headers=_user_headers(user_access_token), timeout=TIMEOUT
        )
        if resp.status_code != 200:
            raise GitHubAppOAuthError("Could not fetch authenticated GitHub user")
        data = resp.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise GitHubAppOAuthError("Could not fetch authenticated GitHub user") from exc

    if not isinstance(data, dict) or "id" not in data or "login" not in data:
        raise GitHubAppOAuthError("Unexpected GitHub user profile response")
    return GitHubAppUser(github_id=int(data["id"]), login=str(data["login"]))


def fetch_user_installations(user_access_token: str) -> List[Dict[str, Any]]:
    """Retrieve the installations accessible to the authenticated GitHub user."""
    try:
        resp = httpx.get(
            f"{API_BASE}/user/installations",
            params={"per_page": 100},
            headers=_user_headers(user_access_token),
            timeout=TIMEOUT,
        )
        if resp.status_code != 200:
            raise GitHubAppOAuthError("Could not fetch accessible GitHub App installations")
        data = resp.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise GitHubAppOAuthError(
            "Could not fetch accessible GitHub App installations"
        ) from exc

    installations = data.get("installations") if isinstance(data, dict) else None
    if not isinstance(installations, list):
        raise GitHubAppOAuthError("Unexpected GitHub installations response")
    return installations

