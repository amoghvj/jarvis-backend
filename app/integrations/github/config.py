"""GitHub App configuration and credential validation."""

import base64
import binascii
import os
import re
from dataclasses import dataclass

from cryptography.exceptions import UnsupportedAlgorithm
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

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


_PEM_BLOCK_RE = re.compile(
    r"-----BEGIN ([A-Z0-9 ]+)-----(.*?)-----END ([A-Z0-9 ]+)-----", re.DOTALL
)
# PKCS#1 ("BEGIN RSA PRIVATE KEY") and PKCS#8 ("BEGIN PRIVATE KEY") are accepted.
_SUPPORTED_PEM_LABELS = {"RSA PRIVATE KEY", "PRIVATE KEY"}
_MIN_RSA_KEY_BITS = 2048


class PrivateKeyConfigError(RuntimeError):
    """GITHUB_APP_PRIVATE_KEY is unusable. Messages never contain key material."""

    def __init__(self, category: str):
        super().__init__(f"GITHUB_APP_PRIVATE_KEY is invalid: {category}")
        self.category = category


def normalize_private_key(raw_key: str) -> str:
    """Return a clean, validated PEM for an RSA private key (PKCS#1 or PKCS#8).

    Accepts real multiline values, values using literal ``\\n`` escapes, CRLF/CR line
    endings, surrounding whitespace, a matching pair of surrounding quotes, and values
    whose newlines were collapsed to spaces. The base64 body is re-wrapped, never
    altered. The key is then loaded with ``cryptography`` to prove it can sign RS256.

    Raises ``PrivateKeyConfigError`` with a non-sensitive ``category``.
    """
    text = (raw_key or "").replace("\ufeff", "").strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        text = text[1:-1].strip()  # base64/PEM never contains quotes, so this is unambiguous
    text = text.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\r", "\n")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if not text:
        raise PrivateKeyConfigError("empty_value")

    if "-----BEGIN" not in text:
        raise PrivateKeyConfigError("missing_pem_header")
    if "-----END" not in text:
        raise PrivateKeyConfigError("missing_pem_footer")
    match = _PEM_BLOCK_RE.search(text)
    if not match:
        raise PrivateKeyConfigError("malformed_pem_block")

    label, body, end_label = match.group(1), match.group(2), match.group(3)
    if label != end_label:
        raise PrivateKeyConfigError("mismatched_pem_header_footer")
    if label not in _SUPPORTED_PEM_LABELS:
        if label.endswith("PRIVATE KEY"):
            raise PrivateKeyConfigError("unsupported_pem_format")  # EC / encrypted / OpenSSH
        raise PrivateKeyConfigError("not_a_private_key")
    if ":" in body:
        raise PrivateKeyConfigError("encrypted_pem_unsupported")  # legacy Proc-Type headers

    compact = re.sub(r"\s+", "", body)
    try:
        if not compact:
            raise ValueError("empty body")
        base64.b64decode(compact, validate=True)
    except (binascii.Error, ValueError):
        raise PrivateKeyConfigError("invalid_base64_encoding") from None

    wrapped = "\n".join(compact[i : i + 64] for i in range(0, len(compact), 64))
    pem = f"-----BEGIN {label}-----\n{wrapped}\n-----END {label}-----\n"

    try:
        key = serialization.load_pem_private_key(pem.encode("ascii"), password=None)
    except (ValueError, TypeError, UnsupportedAlgorithm):
        raise PrivateKeyConfigError("key_load_failed") from None
    if not isinstance(key, rsa.RSAPrivateKey):
        raise PrivateKeyConfigError("not_an_rsa_key")
    if key.key_size < _MIN_RSA_KEY_BITS:
        raise PrivateKeyConfigError("rsa_key_too_small")
    return pem


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

