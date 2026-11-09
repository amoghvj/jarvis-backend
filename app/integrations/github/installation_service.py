"""GitHub App installation flow orchestration, verification, and persistence."""

from typing import Any, Dict, Optional
from uuid import UUID

import psycopg

from app import repositories
from app.integrations.github import oauth_service
from app.integrations.github.config import PENDING_FLOW_TTL_SECONDS, GitHubAppSettings

ALLOWED_SETUP_ACTIONS = {"install", "update"}


class GitHubInstallationError(Exception):
    """Base exception for installation flow errors with a safe error code."""

    reason_code: str = "installation_error"

    def __init__(self, message: str, reason_code: Optional[str] = None):
        super().__init__(message)
        if reason_code:
            self.reason_code = reason_code


class InvalidInstallationIdError(GitHubInstallationError):
    reason_code = "invalid_installation_id"


class InvalidPendingStateError(GitHubInstallationError):
    reason_code = "invalid_state"


class OAuthCancelledError(GitHubInstallationError):
    reason_code = "cancelled"


class InvalidCallbackParamsError(GitHubInstallationError):
    reason_code = "missing_params"


class IdentityMismatchError(GitHubInstallationError):
    reason_code = "identity_mismatch"


class InstallationAccessDeniedError(GitHubInstallationError):
    reason_code = "installation_not_accessible"


class InstallationSuspendedError(GitHubInstallationError):
    reason_code = "installation_suspended"


class InstallationConflictError(GitHubInstallationError):
    reason_code = "installation_already_linked"


def initiate_install_flow(conn: psycopg.Connection, user_id: UUID) -> str:
    """Create a short-lived pending installation flow record and return the raw token."""
    flow_token = oauth_service.generate_random_token()
    state_hash = oauth_service.hash_state_token(flow_token)
    with conn.transaction():
        repositories.delete_expired_pending_installations(conn, user_id)
        repositories.create_pending_installation(
            conn,
            user_id=user_id,
            state_hash=state_hash,
            ttl_seconds=PENDING_FLOW_TTL_SECONDS,
            stage="install",
        )
    return flow_token


def initiate_setup_oauth_flow(
    conn: psycopg.Connection,
    settings: GitHubAppSettings,
    *,
    user_id: UUID,
    installation_id: Optional[int],
    setup_action: Optional[str] = None,
    install_flow_cookie: Optional[str] = None,
) -> str:
    """Validate setup parameters, create a pending OAuth record, and return the raw state."""
    if installation_id is None or not isinstance(installation_id, int) or installation_id <= 0:
        raise InvalidInstallationIdError("installation_id must be a positive integer")

    normalized_action = (setup_action or "install").strip().lower()
    if normalized_action not in ALLOWED_SETUP_ACTIONS:
        raise InvalidInstallationIdError(f"Unsupported setup_action '{normalized_action[:30]}'")

    flow_token = oauth_service.verify_install_flow_cookie(
        settings, install_flow_cookie, user_id
    )
    oauth_state = oauth_service.generate_random_token()
    state_hash = oauth_service.hash_state_token(oauth_state)

    with conn.transaction():
        if flow_token:
            repositories.consume_install_flow_by_hash(
                conn, oauth_service.hash_state_token(flow_token), user_id
            )
        repositories.delete_expired_pending_installations(conn, user_id)
        repositories.create_pending_installation(
            conn,
            user_id=user_id,
            installation_id=installation_id,
            setup_action=normalized_action,
            stage="oauth",
            state_hash=state_hash,
            ttl_seconds=PENDING_FLOW_TTL_SECONDS,
        )
    return oauth_state


def complete_installation_callback(
    conn: psycopg.Connection,
    settings: GitHubAppSettings,
    *,
    code: Optional[str],
    state: Optional[str],
    error: Optional[str],
    oauth_state_cookie: Optional[str],
    session_user_id: Optional[UUID] = None,
) -> Dict[str, Any]:
    """Verify OAuth state, user identity, and installation ownership, then persist link."""
    if not state or not state.strip():
        raise InvalidPendingStateError("Missing OAuth state", reason_code="invalid_state")

    state_hash = oauth_service.hash_state_token(state)
    pending = repositories.get_pending_installation_by_hash(conn, state_hash)
    if not pending or pending["stage"] != "oauth" or pending["installation_id"] is None:
        raise InvalidPendingStateError("Invalid OAuth state", reason_code="invalid_state")

    if pending["consumed_at"] is not None:
        raise InvalidPendingStateError(
            "OAuth state has already been used", reason_code="replayed_state"
        )

    if pending["is_expired"]:
        raise InvalidPendingStateError("OAuth state has expired", reason_code="expired_state")

    # Single-use consumption committed before any external calls so state cannot be replayed.
    with conn.transaction():
        consumed = repositories.consume_pending_installation(conn, pending["id"])
    if not consumed:
        raise InvalidPendingStateError(
            "OAuth state has already been used", reason_code="replayed_state"
        )

    cookie_binding = oauth_service.verify_oauth_state_cookie(
        settings, state, oauth_state_cookie
    )
    if not cookie_binding:
        raise InvalidPendingStateError(
            "Missing or invalid OAuth state cookie", reason_code="invalid_state"
        )

    cookie_user_id, cookie_installation_id = cookie_binding
    expected_user_id: UUID = pending["user_id"]
    expected_installation_id: int = int(pending["installation_id"])

    if cookie_user_id != expected_user_id or cookie_installation_id != expected_installation_id:
        raise InvalidPendingStateError(
            "OAuth state binding mismatch", reason_code="invalid_state"
        )

    if session_user_id is not None and session_user_id != expected_user_id:
        raise IdentityMismatchError(
            "Active J.A.R.V.I.S. session does not match the user who initiated installation"
        )

    if error:
        raise OAuthCancelledError(f"GitHub authorization failed: {error[:50]}")

    if not code or not code.strip():
        raise InvalidCallbackParamsError("Missing authorization code")

    jarvis_user = repositories.get_user_by_id(conn, expected_user_id)
    if not jarvis_user:
        raise IdentityMismatchError("Authenticated J.A.R.V.I.S. user not found")

    try:
        user_access_token = oauth_service.exchange_code_for_user_token(settings, code)
        gh_user = oauth_service.fetch_authenticated_user(user_access_token)
        accessible_installations = oauth_service.fetch_user_installations(user_access_token)
    except oauth_service.GitHubAppOAuthError as exc:
        raise GitHubInstallationError(str(exc), reason_code="github_oauth_error") from exc

    # Verify immutable GitHub numeric user ID matches the J.A.R.V.I.S. user's github_id.
    if int(gh_user.github_id) != int(jarvis_user["github_id"]):
        raise IdentityMismatchError(
            "Authenticated GitHub account does not match the J.A.R.V.I.S. user's GitHub identity"
        )

    # Locate the requested installation in the user's accessible installations.
    matched_inst: Optional[Dict[str, Any]] = None
    for inst in accessible_installations:
        if isinstance(inst, dict) and inst.get("id") is not None:
            try:
                if int(inst["id"]) == expected_installation_id:
                    matched_inst = inst
                    break
            except (TypeError, ValueError):
                continue

    if not matched_inst:
        raise InstallationAccessDeniedError(
            "Installation was not found in the authenticated user's accessible installations"
        )

    if matched_inst.get("app_id") is not None:
        try:
            if int(matched_inst["app_id"]) != settings.app_id:
                raise InstallationAccessDeniedError(
                    "Installation does not belong to the configured J.A.R.V.I.S. GitHub App"
                )
        except (TypeError, ValueError) as exc:
            raise InstallationAccessDeniedError("Invalid app_id on installation") from exc

    if matched_inst.get("suspended_at") is not None:
        raise InstallationSuspendedError("GitHub App installation is currently suspended")

    account = matched_inst.get("account")
    if not isinstance(account, dict) or "id" not in account or not account.get("login"):
        raise InstallationAccessDeniedError("Installation is missing GitHub account metadata")

    github_account_id = int(account["id"])
    github_account_login = str(account["login"])
    github_account_type = str(
        account.get("type") or matched_inst.get("target_type") or "User"
    )

    if github_account_type.lower() == "user" and github_account_id != int(
        jarvis_user["github_id"]
    ):
        raise InstallationAccessDeniedError(
            "Personal GitHub installation belongs to a different GitHub user"
        )

    repository_selection = matched_inst.get("repository_selection")
    permissions = matched_inst.get("permissions")
    if not isinstance(permissions, dict):
        permissions = {}

    with conn.transaction():
        saved = repositories.upsert_github_app_installation(
            conn,
            user_id=expected_user_id,
            installation_id=expected_installation_id,
            github_account_id=github_account_id,
            github_account_login=github_account_login,
            github_account_type=github_account_type,
            repository_selection=repository_selection,
            permissions=permissions,
            suspended_at=None,
        )

    if not saved:
        raise InstallationConflictError(
            "This GitHub App installation is already associated with another J.A.R.V.I.S. user"
        )

    return saved


def get_user_installations_status(conn: psycopg.Connection, user_id: UUID) -> Dict[str, Any]:
    rows = repositories.list_github_app_installations_for_user(conn, user_id)
    installations = [
        {
            "installation_id": int(r["installation_id"]),
            "account_id": int(r["github_account_id"]),
            "account_login": r["github_account_login"],
            "account_type": r["github_account_type"],
            "repository_selection": r["repository_selection"],
            "permissions": r["permissions"] or {},
            "suspended_at": r["suspended_at"],
            "created_at": r["created_at"],
            "updated_at": r["updated_at"],
        }
        for r in rows
    ]
    return {
        "connected": len(installations) > 0,
        "installations": installations,
    }


def disconnect_user_installation(
    conn: psycopg.Connection, user_id: UUID, installation_id: int
) -> bool:
    with conn.transaction():
        return repositories.delete_github_app_installation(conn, user_id, installation_id)

