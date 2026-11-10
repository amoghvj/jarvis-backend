"""Safe, normalized tool errors and secret redaction for logs and results."""

import re

_REDACTIONS = [
    re.compile(r"-----BEGIN [A-Z ]+-----.*?-----END [A-Z ]+-----", re.DOTALL),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{10,}"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{10,}"),
    re.compile(r"\bjarvis_live_[A-Za-z0-9_\-]{6,}"),
    re.compile(r"eyJ[A-Za-z0-9_\-]{5,}\.[A-Za-z0-9_\-]{5,}\.[A-Za-z0-9_\-]*"),
    re.compile(r"(?i)\b(bearer|token)\s+[A-Za-z0-9._\-]{8,}"),
    re.compile(r"(?i)postgres(?:ql)?://\S+"),
]


def redact(text: str) -> str:
    """Remove anything resembling a credential before text is logged or returned."""
    for pattern in _REDACTIONS:
        text = pattern.sub("[REDACTED]", text)
    return text


class ToolError(Exception):
    """A failure whose code and message are safe to return to callers.

    Messages are written by this codebase and never include GitHub tokens, SQL,
    stack traces or upstream response bodies.
    """

    def __init__(self, code: str, message: str, retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable

