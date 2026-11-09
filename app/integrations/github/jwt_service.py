"""GitHub App RS256 JWT generation."""

import time
from typing import Optional

import jwt

from app.integrations.github.config import GitHubAppSettings

CLOCK_SKEW_SECONDS = 60
JWT_LIFETIME_SECONDS = 9 * 60  # 9 minutes (GitHub maximum is 10 minutes)


class GitHubAppJwtError(Exception):
    """Failed to sign a GitHub App JWT."""


def generate_app_jwt(settings: GitHubAppSettings, now: Optional[int] = None) -> str:
    """Create a short-lived RS256 JWT authenticating as the GitHub App."""
    if not isinstance(settings.app_id, int) or settings.app_id <= 0:
        raise GitHubAppJwtError("GitHub App ID must be a positive integer")

    current_time = int(time.time()) if now is None else int(now)
    payload = {
        "iat": current_time - CLOCK_SKEW_SECONDS,
        "exp": current_time + JWT_LIFETIME_SECONDS,
        "iss": str(settings.app_id),
    }
    try:
        token = jwt.encode(payload, settings.private_key, algorithm="RS256")
    except Exception as exc:
        # Never expose the private key in error messages or logs.
        raise GitHubAppJwtError("Failed to sign GitHub App JWT with configured private key") from exc
    return token if isinstance(token, str) else token.decode("utf-8")

