"""Centralised, validated configuration read from environment variables."""

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional
from urllib.parse import urlsplit

from dotenv import load_dotenv

# Local development only; on Vercel real environment variables are used.
# Existing environment variables are never overridden.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

REQUIRED_VARS = (
    "DATABASE_URL",
    "GITHUB_CLIENT_ID",
    "GITHUB_CLIENT_SECRET",
    "GITHUB_CALLBACK_URL",
    "FRONTEND_URL",
    "SESSION_SECRET",
)

SESSION_TTL_SECONDS = 7 * 24 * 3600
OAUTH_STATE_TTL_SECONDS = 10 * 60


def _origin(url: str) -> str:
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}"


@dataclass(frozen=True)
class Settings:
    database_url: str
    github_client_id: str
    github_client_secret: str
    github_callback_url: str
    frontend_url: str
    session_secret: str
    post_login_redirect: bool
    samesite_override: Optional[str]

    @property
    def frontend_origin(self) -> str:
        return _origin(self.frontend_url)

    @property
    def backend_origin(self) -> str:
        return _origin(self.github_callback_url)

    @property
    def cookie_secure(self) -> bool:
        """Secure cookies whenever the backend is served over HTTPS (production)."""
        return self.github_callback_url.lower().startswith("https://")

    @property
    def cookie_samesite(self) -> Literal["lax", "strict", "none"]:
        """Cookie SameSite policy.

        - Production, frontend on a different origin: "none" (+Secure) so a browser
          app can call this API with credentials.
        - Same origin, or plain-HTTP local development: "lax".
        - COOKIE_SAMESITE can force a value (e.g. "lax" when frontend and backend
          share a registrable domain).
        """
        if self.samesite_override:
            return self.samesite_override  # type: ignore[return-value]
        if self.cookie_secure and self.frontend_origin != self.backend_origin:
            return "none"
        return "lax"

    @property
    def allowed_origins(self) -> set:
        """Origins allowed to make state-changing, cookie-authenticated requests."""
        return {self.frontend_origin, self.backend_origin}


@lru_cache
def get_settings() -> Settings:
    missing = [name for name in REQUIRED_VARS if not os.environ.get(name)]
    if missing:
        # Names only; values are never included in errors or logs.
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

    secret = os.environ["SESSION_SECRET"]
    if len(secret) < 32:
        raise RuntimeError("SESSION_SECRET must be at least 32 characters long")

    samesite = os.environ.get("COOKIE_SAMESITE", "").strip().lower() or None
    if samesite not in (None, "lax", "strict", "none"):
        raise RuntimeError("COOKIE_SAMESITE must be one of: lax, strict, none")

    return Settings(
        database_url=os.environ["DATABASE_URL"],
        github_client_id=os.environ["GITHUB_CLIENT_ID"],
        github_client_secret=os.environ["GITHUB_CLIENT_SECRET"],
        github_callback_url=os.environ["GITHUB_CALLBACK_URL"],
        frontend_url=os.environ["FRONTEND_URL"].rstrip("/"),
        session_secret=secret,
        post_login_redirect=os.environ.get("POST_LOGIN_REDIRECT", "false").strip().lower()
        in ("1", "true", "yes"),
        samesite_override=samesite,
    )

