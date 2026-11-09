from typing import Optional

import psycopg
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse, Response

from app import repositories
from app.config import Settings, get_settings
from app.database import get_db
from app.dependencies.authentication import get_current_user, require_trusted_origin
from app.schemas import LoginSuccess, MessageOut, UserOut
from app.services import github_oauth, session_service

router = APIRouter(prefix="/auth", tags=["auth"])

NO_STORE = {"Cache-Control": "no-store"}


@router.get("/github", summary="Start GitHub login (redirects to GitHub)")
def github_login(settings: Settings = Depends(get_settings)) -> RedirectResponse:
    state = session_service.new_state()
    response = RedirectResponse(github_oauth.build_authorize_url(settings, state), status_code=302)
    response.headers.update(NO_STORE)
    session_service.set_state_cookie(response, settings, state)
    return response


@router.get("/github/callback", summary="GitHub OAuth callback", include_in_schema=True)
def github_callback(
    request: Request,
    code: Optional[str] = None,
    state: Optional[str] = None,
    error: Optional[str] = None,
    settings: Settings = Depends(get_settings),
    conn: psycopg.Connection = Depends(get_db),
):
    def fail(status: int, detail: str) -> JSONResponse:
        response = JSONResponse(status_code=status, content={"detail": detail}, headers=NO_STORE)
        session_service.clear_state_cookie(response, settings)
        return response

    if not session_service.verify_state(
        settings, state, request.cookies.get(session_service.STATE_COOKIE)
    ):
        return fail(400, "Invalid or expired OAuth state. Please restart the login.")
    if error:
        # Only GitHub's short error code (e.g. access_denied), not free text.
        return fail(400, f"GitHub authorization failed: {error[:50]}")
    if not code:
        return fail(400, "Missing authorization code")

    try:
        token = github_oauth.exchange_code_for_token(settings, code)
        profile = github_oauth.fetch_profile(token)
    except github_oauth.GitHubOAuthError as exc:
        return fail(502, str(exc))

    with conn.transaction():
        user = repositories.upsert_user(
            conn,
            github_id=profile.github_id,
            username=profile.username,
            email=profile.email,
            avatar_url=profile.avatar_url,
        )
    session_token = session_service.create_session(conn, user["id"])

    if settings.post_login_redirect:
        response: Response = RedirectResponse(settings.frontend_url, status_code=302)
    else:
        # Temporary success response until the React frontend exists.
        body = LoginSuccess(new_account=user["created"], user=UserOut(**user))
        response = JSONResponse(content=body.model_dump(mode="json"))
    response.headers.update(NO_STORE)
    session_service.set_session_cookie(response, settings, session_token)
    session_service.clear_state_cookie(response, settings)
    return response


@router.get("/me", response_model=UserOut, summary="Current authenticated user")
def me(user: dict = Depends(get_current_user)) -> JSONResponse:
    return JSONResponse(content=UserOut(**user).model_dump(mode="json"), headers=NO_STORE)


@router.post(
    "/logout",
    response_model=MessageOut,
    dependencies=[Depends(require_trusted_origin)],
    summary="Invalidate the session and clear the cookie",
)
def logout(
    request: Request,
    settings: Settings = Depends(get_settings),
    conn: psycopg.Connection = Depends(get_db),
) -> JSONResponse:
    session_service.revoke_session(conn, request.cookies.get(session_service.SESSION_COOKIE))
    response = JSONResponse(content={"status": "success", "message": "Logged out"}, headers=NO_STORE)
    session_service.clear_session_cookie(response, settings)
    return response

