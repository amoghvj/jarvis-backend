"""Reusable FastAPI authentication dependencies."""

from typing import Optional

import psycopg
from fastapi import Depends, Header, HTTPException, Request

from app.config import Settings, get_settings
from app.database import get_db
from app.services import api_key_service, session_service


def get_current_user(request: Request, conn: psycopg.Connection = Depends(get_db)) -> dict:
    """Require a valid browser session cookie."""
    user = session_service.get_user_for_token(conn, request.cookies.get(session_service.SESSION_COOKIE))
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


def require_trusted_origin(request: Request, settings: Settings = Depends(get_settings)) -> None:
    """CSRF defence for cookie-authenticated state-changing requests.

    Browsers always send `Origin` on cross-origin and non-GET requests and
    scripts cannot forge it, so the request must come from the configured
    frontend (or this backend). Combined with a non-simple CORS policy this
    blocks cross-site form posts even when the cookie is SameSite=None.
    """
    if request.headers.get("origin") not in settings.allowed_origins:
        raise HTTPException(status_code=403, detail="Untrusted or missing Origin header")


def get_api_key_account(
    authorization: Optional[str] = Header(default=None),
    conn: psycopg.Connection = Depends(get_db),
) -> api_key_service.ApiKeyAccount:
    """Require `Authorization: Bearer <API key>` (used by telemetry ingestion)."""
    try:
        return api_key_service.authenticate(conn, authorization)
    except api_key_service.InvalidApiKey as exc:
        raise HTTPException(
            status_code=401, detail=str(exc), headers={"WWW-Authenticate": "Bearer"}
        )

