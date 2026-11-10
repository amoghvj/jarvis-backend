"""Secret detection and redaction for configuration inspection.

Policy (deliberately conservative): a value is shown only if its *key* does not look
sensitive AND the *value* does not look like a credential. Everything else is replaced
by a marker. File names are never trusted: an `.env.example` is still scanned.
"""

import re
from typing import Optional, Tuple

from app.tools.errors import redact

REDACTED = "[REDACTED]"
MAX_SHOWN_VALUE = 120

_SENSITIVE_KEY = re.compile(
    r"(SECRET|TOKEN|PASSWORD|PASSWD|PASSPHRASE|(^|_)PWD($|_)|PRIVATE|CREDENTIAL|API[_-]?KEY|ACCESS[_-]?KEY|"
    r"(^|[_-])KEY($|[_-])|[_-]KEY$|AUTH|SALT|SIGNING|ENCRYPT|DSN|DATABASE_URL|DB_URL|CONNECTION|COOKIE|"
    r"SESSION|JWT|BEARER|WEBHOOK|CERT|LICENSE_KEY|OTP|PIN$)",
    re.IGNORECASE,
)
# The lookbehind anchors the scheme at a word start; without it every letter in a long run would
# start a scan to the end of the run (quadratic time on large stack traces).
_URL_CREDENTIALS = re.compile(r"(?<![A-Za-z0-9+.\-])([a-zA-Z][a-zA-Z0-9+.\-]{0,31}://)[^/\s:@]+:[^/\s@]*@")
_URL_TOKEN_USER = re.compile(r"(?<![A-Za-z0-9+.\-])([a-zA-Z][a-zA-Z0-9+.\-]{0,31}://)[^/\s:@]{20,}@")
_KNOWN_SECRET = re.compile(
    r"(-----BEGIN [A-Z ]+-----|\bgh[pousr]_[A-Za-z0-9]{10,}|\bgithub_pat_[A-Za-z0-9_]{10,}|\bsk[-_](live|test)?[-_]?[A-Za-z0-9]{10,}|"
    r"\b[rp]k_(live|test)_[A-Za-z0-9]{8,}|\bxox[abprs]-[A-Za-z0-9-]{8,}|\bAKIA[0-9A-Z]{12,}|\bAIza[0-9A-Za-z_\-]{20,}|"
    r"\beyJ[A-Za-z0-9_\-]{5,}\.[A-Za-z0-9_\-]{5,}|\bjarvis_live_[A-Za-z0-9_\-]{6,}|\bnpm_[A-Za-z0-9]{20,})"
)
_LONG_TOKEN = re.compile(r"[A-Za-z0-9+/=_\-]{32,}")
_ASSIGNMENT = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]{2,})=(\"[^\"]*\"|'[^']*'|\S+)")


def is_sensitive_key(key: str) -> bool:
    return bool(_SENSITIVE_KEY.search(key))


def looks_secret(value: str) -> bool:
    if _KNOWN_SECRET.search(value) or _URL_CREDENTIALS.search(value) or _URL_TOKEN_USER.search(value):
        return True
    for token in _LONG_TOKEN.findall(value):
        if any(c.isdigit() for c in token) and any(c.isalpha() for c in token):
            return True  # long mixed alphanumeric strings are treated as credentials
    return False


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def safe_value(key: str, value: str) -> Tuple[Optional[str], bool]:
    """Return (value to show, redacted?). `None` means the value is withheld."""
    value = _unquote(value)
    if value == "":
        return "", False  # empty means "unset": useful, and cannot be a secret
    if is_sensitive_key(key) or looks_secret(value) or len(value) > MAX_SHOWN_VALUE:
        return None, True
    return value, False


def scrub_text(text: str) -> Tuple[str, bool]:
    """Mask credentials inside free text (commands, URLs, version specs)."""
    original = text
    text = _URL_CREDENTIALS.sub(lambda m: m.group(1) + REDACTED + "@", text)
    text = _URL_TOKEN_USER.sub(lambda m: m.group(1) + REDACTED + "@", text)

    def mask(match: "re.Match[str]") -> str:
        key, val = match.group(1), match.group(2)
        if is_sensitive_key(key) or looks_secret(val):
            return f"{key}={REDACTED}"
        return match.group(0)

    text = _ASSIGNMENT.sub(mask, text)
    text = _KNOWN_SECRET.sub(REDACTED, text)
    text = redact(text)
    return text, text != original

