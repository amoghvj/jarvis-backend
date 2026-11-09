"""GITHUB_APP_PRIVATE_KEY normalization and validation tests (no database needed)."""

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec, rsa

from app.integrations.github import jwt_service
from app.integrations.github.config import (
    PrivateKeyConfigError,
    get_github_app_settings,
    normalize_private_key,
)

_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
PUBLIC_PEM = _KEY.public_key().public_bytes(
    serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
).decode()


def _pem(key, fmt):
    return key.private_bytes(
        serialization.Encoding.PEM, fmt, serialization.NoEncryption()
    ).decode()


PKCS8 = _pem(_KEY, serialization.PrivateFormat.PKCS8)
PKCS1 = _pem(_KEY, serialization.PrivateFormat.TraditionalOpenSSL)


def _body(pem: str) -> str:
    return "".join(pem.strip().splitlines()[1:-1])


def _assert_signs_rs256(pem: str, monkeypatch):
    monkeypatch.setenv("GITHUB_APP_PRIVATE_KEY", pem)
    settings = get_github_app_settings()
    token = jwt_service.generate_app_jwt(settings, now=1_700_000_000)
    claims = jwt.decode(token, PUBLIC_PEM, algorithms=["RS256"], options={"verify_exp": False})
    assert claims["iss"] == "987654"


@pytest.mark.parametrize("pem,header", [(PKCS1, "RSA PRIVATE KEY"), (PKCS8, "PRIVATE KEY")])
def test_valid_multiline_keys(pem, header, monkeypatch):
    out = normalize_private_key(pem)
    assert out.startswith(f"-----BEGIN {header}-----\n") and out.endswith(f"-----END {header}-----\n")
    assert _body(out) == _body(pem)  # key material preserved
    _assert_signs_rs256(pem, monkeypatch)


@pytest.mark.parametrize("pem", [PKCS1, PKCS8])
def test_valid_escaped_newline_keys(pem, monkeypatch):
    escaped = pem.strip().replace("\n", "\\n")
    assert normalize_private_key(escaped) == normalize_private_key(pem)
    _assert_signs_rs256(escaped, monkeypatch)
    _assert_signs_rs256('"' + escaped + '"', monkeypatch)  # accidental quotes


@pytest.mark.parametrize("pem", [PKCS1, PKCS8])
def test_crlf_cr_whitespace_and_collapsed_newlines(pem):
    expected = normalize_private_key(pem)
    assert normalize_private_key(pem.replace("\n", "\r\n")) == expected
    assert normalize_private_key(pem.replace("\n", "\r")) == expected
    assert normalize_private_key("  \n\t" + pem + "\n  \t ") == expected
    assert normalize_private_key("'" + pem + "'") == expected
    # Single-line env fields often turn newlines into spaces.
    assert normalize_private_key(pem.strip().replace("\n", " ")) == expected


def _category(value: str) -> str:
    with pytest.raises(PrivateKeyConfigError) as exc:
        normalize_private_key(value)
    message = str(exc.value)
    body = _body(PKCS8)
    assert body[:20] not in message and "BEGIN" not in message  # never leaks key data
    return exc.value.category


def test_missing_header_and_footer():
    assert _category("\n".join(PKCS8.strip().splitlines()[1:])) == "missing_pem_header"
    assert _category("\n".join(PKCS8.strip().splitlines()[:-1])) == "missing_pem_footer"


def test_invalid_contents():
    assert _category("-----BEGIN PRIVATE KEY-----\nnot*valid*base64!!\n-----END PRIVATE KEY-----") == "invalid_base64_encoding"
    assert _category("-----BEGIN PRIVATE KEY-----\nQUJDREVGR0g=\n-----END PRIVATE KEY-----") == "key_load_failed"
    assert _category("-----BEGIN PRIVATE KEY-----\n-----END PRIVATE KEY-----") == "invalid_base64_encoding"


def test_empty_values():
    assert _category("") == "empty_value"
    assert _category("   \n ") == "empty_value"
    assert _category('""') == "empty_value"


def test_empty_environment_variable_is_reported_as_missing(monkeypatch):
    monkeypatch.setenv("GITHUB_APP_PRIVATE_KEY", "")
    with pytest.raises(RuntimeError, match="GITHUB_APP_PRIVATE_KEY"):
        get_github_app_settings()


def test_unsupported_key_types():
    ec_key = ec.generate_private_key(ec.SECP256R1())
    assert _category(_pem(ec_key, serialization.PrivateFormat.PKCS8)) == "not_an_rsa_key"
    assert _category(_pem(ec_key, serialization.PrivateFormat.TraditionalOpenSSL)) == "unsupported_pem_format"
    encrypted = _KEY.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.BestAvailableEncryption(b"pw"),
    ).decode()
    assert _category(encrypted) == "unsupported_pem_format"
    assert _category(PUBLIC_PEM) == "not_a_private_key"


def test_small_rsa_key_rejected():
    small = rsa.generate_private_key(public_exponent=65537, key_size=1024)
    assert _category(_pem(small, serialization.PrivateFormat.PKCS8)) == "rsa_key_too_small"


def test_settings_error_names_variable_without_key(monkeypatch):
    monkeypatch.setenv("GITHUB_APP_PRIVATE_KEY", "-----BEGIN PRIVATE KEY-----\nQUJD\n-----END PRIVATE KEY-----")
    with pytest.raises(PrivateKeyConfigError) as exc:
        get_github_app_settings()
    assert "GITHUB_APP_PRIVATE_KEY" in str(exc.value) and "QUJD" not in str(exc.value)
