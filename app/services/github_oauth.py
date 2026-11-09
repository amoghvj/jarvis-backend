"""GitHub OAuth authorization-code flow (identity only; no repository access)."""

import urllib.parse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx

from app.config import Settings

AUTHORIZE_URL = "https://github.com/login/oauth/authorize"
TOKEN_URL = "https://github.com/login/oauth/access_token"
API_BASE = "https://api.github.com"
SCOPES = "read:user user:email"
TIMEOUT = 10.0


class GitHubOAuthError(Exception):
    """GitHub rejected a request or was unreachable."""


@dataclass(frozen=True)
class GitHubProfile:
    github_id: int
    username: str
    avatar_url: Optional[str]
    email: Optional[str]


def build_authorize_url(settings: Settings, state: str) -> str:
    query = urllib.parse.urlencode(
        {
            "client_id": settings.github_client_id,
            "redirect_uri": settings.github_callback_url,
            "scope": SCOPES,
            "state": state,
        }
    )
    return f"{AUTHORIZE_URL}?{query}"


def exchange_code_for_token(settings: Settings, code: str) -> str:
    try:
        resp = httpx.post(
            TOKEN_URL,
            data={
                "client_id": settings.github_client_id,
                "client_secret": settings.github_client_secret,
                "code": code,
                "redirect_uri": settings.github_callback_url,
            },
            headers={"Accept": "application/json"},
            timeout=TIMEOUT,
        )
        data = resp.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise GitHubOAuthError("Could not reach GitHub to exchange the authorization code") from exc

    token = data.get("access_token") if isinstance(data, dict) else None
    if resp.status_code != 200 or not token:
        # Log nothing from the body: it may echo request values.
        raise GitHubOAuthError("GitHub rejected the authorization code")
    return token


def pick_verified_email(emails: List[Dict[str, Any]]) -> Optional[str]:
    """Prefer the primary verified address, else any verified one."""
    verified = [e for e in emails if e.get("verified") and e.get("email")]
    for e in verified:
        if e.get("primary"):
            return e["email"]
    return verified[0]["email"] if verified else None


def fetch_profile(access_token: str) -> GitHubProfile:
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    try:
        user_resp = httpx.get(f"{API_BASE}/user", headers=headers, timeout=TIMEOUT)
        if user_resp.status_code != 200:
            raise GitHubOAuthError("Could not fetch the GitHub profile")
        user = user_resp.json()

        email: Optional[str] = None
        try:
            email_resp = httpx.get(f"{API_BASE}/user/emails", headers=headers, timeout=TIMEOUT)
            if email_resp.status_code == 200:
                email = pick_verified_email(email_resp.json())
        except (httpx.HTTPError, ValueError):
            email = None  # Email is optional; never block login on it.
    except (httpx.HTTPError, ValueError) as exc:
        raise GitHubOAuthError("Could not fetch the GitHub profile") from exc

    if not isinstance(user, dict) or "id" not in user or "login" not in user:
        raise GitHubOAuthError("Unexpected GitHub profile response")
    return GitHubProfile(
        github_id=int(user["id"]),
        username=user["login"],
        avatar_url=user.get("avatar_url"),
        email=email,
    )

