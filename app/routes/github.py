"""GitHub App installation, OAuth verification, status, and disconnect routes."""

import urllib.parse
from typing import Optional

import psycopg
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse

from app.database import get_db
from app.dependencies.authentication import get_current_user, require_trusted_origin
from app.integrations.github import installation_service, oauth_service
from app.integrations.github.config import GitHubAppSettings, get_github_app_settings
from app.schemas import GitHubDisconnectOut, GitHubIntegrationStatusOut
from app.services import session_service

router = APIRouter(prefix="/integrations/github", tags=["github-integration"])

NO_STORE = {"Cache-Control": "no-store"}


def _frontend_redirect_url(base_url: str, **params: str) -> str:
    parts = urllib.parse.urlsplit(base_url)
    existing = dict(urllib.parse.parse_qsl(parts.query, keep_blank_values=True))
    existing.update(params)
    new_query = urllib.parse.urlencode(existing)
    return urllib.parse.urlunsplit(
        (parts.scheme, parts.netloc, parts.path, new_query, parts.fragment)
    )


@router.get("/install", summary="Initiate GitHub App installation")
def install_github_app(
    user: dict = Depends(get_current_user),
    app_settings: GitHubAppSettings = Depends(get_github_app_settings),
    conn: psycopg.Connection = Depends(get_db),
) -> RedirectResponse:
    flow_token = installation_service.initiate_install_flow(conn, user["id"])
    response = RedirectResponse(
        oauth_service.build_app_install_url(app_settings), status_code=302
    )
    response.headers.update(NO_STORE)
    oauth_service.set_install_flow_cookie(response, app_settings, flow_token, user["id"])
    return response


@router.get("/setup", summary="Handle GitHub App post-installation setup redirect")
def setup_github_app(
    request: Request,
    installation_id: Optional[str] = None,
    setup_action: Optional[str] = None,
    user: dict = Depends(get_current_user),
    app_settings: GitHubAppSettings = Depends(get_github_app_settings),
    conn: psycopg.Connection = Depends(get_db),
) -> RedirectResponse:
    if not installation_id or not installation_id.strip().isdigit():
        raise HTTPException(status_code=400, detail="Invalid or missing installation_id")
    parsed_installation_id = int(installation_id.strip())
    if parsed_installation_id <= 0:
        raise HTTPException(status_code=400, detail="Invalid or missing installation_id")

    try:
        oauth_state = installation_service.initiate_setup_oauth_flow(
            conn,
            app_settings,
            user_id=user["id"],
            installation_id=parsed_installation_id,
            setup_action=setup_action,
            install_flow_cookie=request.cookies.get(oauth_service.INSTALL_FLOW_COOKIE),
        )
    except installation_service.InvalidInstallationIdError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    authorize_url = oauth_service.build_app_oauth_authorize_url(app_settings, oauth_state)
    response = RedirectResponse(authorize_url, status_code=302)
    response.headers.update(NO_STORE)
    oauth_service.set_oauth_state_cookie(
        response, app_settings, oauth_state, user["id"], parsed_installation_id
    )
    return response


@router.get("/callback", summary="Complete GitHub App OAuth verification and link installation")
def github_app_callback(
    request: Request,
    code: Optional[str] = None,
    state: Optional[str] = None,
    error: Optional[str] = None,
    app_settings: GitHubAppSettings = Depends(get_github_app_settings),
    conn: psycopg.Connection = Depends(get_db),
) -> RedirectResponse:
    session_user = session_service.get_user_for_token(
        conn, request.cookies.get(session_service.SESSION_COOKIE)
    )
    session_user_id = session_user["id"] if session_user else None

    try:
        saved = installation_service.complete_installation_callback(
            conn,
            app_settings,
            code=code,
            state=state,
            error=error,
            oauth_state_cookie=request.cookies.get(oauth_service.OAUTH_STATE_COOKIE),
            session_user_id=session_user_id,
        )
        target_url = _frontend_redirect_url(
            app_settings.frontend_url,
            github_app="connected",
            installation_id=str(saved["installation_id"]),
        )
    except installation_service.GitHubInstallationError as exc:
        target_url = _frontend_redirect_url(
            app_settings.frontend_url,
            github_app="error",
            reason=exc.reason_code,
        )

    response = RedirectResponse(target_url, status_code=302)
    response.headers.update(NO_STORE)
    oauth_service.clear_flow_cookies(response, app_settings)
    return response


@router.get(
    "/status",
    response_model=GitHubIntegrationStatusOut,
    summary="List the authenticated user's linked GitHub App installations",
)
def github_integration_status(
    user: dict = Depends(get_current_user),
    conn: psycopg.Connection = Depends(get_db),
) -> JSONResponse:
    status_data = installation_service.get_user_installations_status(conn, user["id"])
    body = GitHubIntegrationStatusOut(**status_data)
    return JSONResponse(content=body.model_dump(mode="json"), headers=NO_STORE)


@router.delete(
    "/installations/{installation_id}",
    response_model=GitHubDisconnectOut,
    dependencies=[Depends(require_trusted_origin)],
    summary="Remove the local J.A.R.V.I.S. association for a GitHub App installation",
)
def disconnect_github_installation(
    installation_id: int,
    user: dict = Depends(get_current_user),
    conn: psycopg.Connection = Depends(get_db),
) -> JSONResponse:
    if installation_id <= 0:
        raise HTTPException(status_code=404, detail="Installation not found")

    deleted = installation_service.disconnect_user_installation(
        conn, user["id"], installation_id
    )
    if not deleted:
        raise HTTPException(status_code=404, detail="Installation not found")

    body = GitHubDisconnectOut(
        status="success",
        message=(
            "Installation disconnected from J.A.R.V.I.S. To completely uninstall the "
            "GitHub App, remove it in your GitHub account or organization settings."
        ),
        installation_id=installation_id,
    )
    return JSONResponse(content=body.model_dump(mode="json"), headers=NO_STORE)

