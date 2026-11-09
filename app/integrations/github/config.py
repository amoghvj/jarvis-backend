"""GitHub App configuration and credential validation."""

import os
from dataclasses import dataclass

from app.config import Settings, get_settings

REQUIRED_GITHUB_APP_VARS = (
    "GITHUB_APP_ID",
    "GITHUB_APP_CLIENT_ID",
    "GITHUB_APP_CLIENT_SECRET",
    "GITHUB_APP_PRIVATE_KEY",
    "GITHUB_APP_SLUG",
    "GITHUB_APP_CALLBACK_URL",
)

PENDING_FLOW_TTL_SECONDS = 10 * 60


def normalize_private_key(raw_key: str) -> str:
    """Normalize multiline PEM or escaped-newline PEM private key strings."""
    cleaned = raw_key.strip()
    if (cleaned.startswith('"') and cleaned.endswith('"')) or (
        cleaned.startswith("'") and cleaned.endswith("'")
    ):
        cleaned = cleaned[1:-1].strip()
    cleaned = cleaned.replace("\\r\\n", "\n").replace("\\n", "\n").strip()
    if not cleaned.startswith("-----BEGIN ") or "PRIVATE KEY-----" not in cleaned:
        raise RuntimeError("GITHUB_APP_PRIVATE_KEY is not a valid PEM private key")
    return cleaned + "\n"


@dataclass(frozen=True)
class GitHubAppSettings:
    app_id: int
    client_id: str
    client_secret: str
    private_key: str
    app_slug: str
    callback_url: str
    base_settings: Settings

    @property
    def frontend_url(self) -> str:
        return self.base_settings.frontend_url

    @property
    def session_secret(self) -> str:
        return self.base_settings.session_secret

    @property
    def cookie_secure(self) -> bool:
        return self.base_settings.cookie_secure


def get_github_app_settings() -> GitHubAppSettings:
    base = get_settings()
    values = {
        "GITHUB_APP_ID": base.github_app_id or os.environ.get("GITHUB_APP_ID"),
        "GITHUB_APP_CLIENT_ID": base.github_app_client_id or os.environ.get("GITHUB_APP_CLIENT_ID"),
        "GITHUB_APP_CLIENT_SECRET": base.github_app_client_secret
        or os.environ.get("GITHUB_APP_CLIENT_SECRET"),
        "GITHUB_APP_PRIVATE_KEY": base.github_app_private_key
        or os.environ.get("GITHUB_APP_PRIVATE_KEY"),
        "GITHUB_APP_SLUG": base.github_app_slug or os.environ.get("GITHUB_APP_SLUG"),
        "GITHUB_APP_CALLBACK_URL": base.github_app_callback_url
        or os.environ.get("GITHUB_APP_CALLBACK_URL"),
    }
    missing = [name for name, val in values.items() if not val or not str(val).strip()]
    if missing:
        raise RuntimeError(
            f"Missing required GitHub App environment variables: {', '.join(missing)}"
        )

    raw_app_id = str(values["GITHUB_APP_ID"]).strip()
    client_id = str(values["GITHUB_APP_CLIENT_ID"]).strip()
    if raw_app_id == client_id:
        raise RuntimeError("GITHUB_APP_ID must not be set to GITHUB_APP_CLIENT_ID")
    if not raw_app_id.isdigit() or int(raw_app_id) <= 0:
        raise RuntimeError("GITHUB_APP_ID must be a positive integer")

    private_key = normalize_private_key(str(values["GITHUB_APP_PRIVATE_KEY"]))

    return GitHubAppSettings(
        app_id=int(raw_app_id),
        client_id=client_id,
        client_secret=str(values["GITHUB_APP_CLIENT_SECRET"]).strip(),
        private_key=private_key,
        app_slug=str(values["GITHUB_APP_SLUG"]).strip(),
        callback_url=str(values["GITHUB_APP_CALLBACK_URL"]).strip(),
        base_settings=base,
    )

