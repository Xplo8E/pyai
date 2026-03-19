"""
Tests for OpenAI Codex OAuth logic (no network required).
"""

import base64
import json
import time

import pytest

from pyai.oauth.openai_codex import (
    extract_account_id,
    _decode_jwt_payload,
    _build_auth_url,
    _parse_authorization_input,
    CLIENT_ID,
    AUTHORIZE_URL,
)
from pyai.oauth.types import OAuthCredentials


# ------------------------------------------------------------------ #
# JWT decoding                                                        #
# ------------------------------------------------------------------ #

def _make_jwt(payload: dict) -> str:
    """Build a fake JWT with the given payload."""
    header = base64.urlsafe_b64encode(b'{"alg":"RS256"}').rstrip(b"=").decode()
    payload_bytes = json.dumps(payload).encode()
    payload_b64 = base64.urlsafe_b64encode(payload_bytes).rstrip(b"=").decode()
    sig = base64.urlsafe_b64encode(b"fakesig").rstrip(b"=").decode()
    return f"{header}.{payload_b64}.{sig}"


def test_decode_jwt_payload_returns_dict():
    token = _make_jwt({"sub": "user123", "exp": 9999999999})
    result = _decode_jwt_payload(token)
    assert result is not None
    assert result["sub"] == "user123"


def test_decode_jwt_payload_invalid_returns_none():
    assert _decode_jwt_payload("not.a.jwt.at.all") is None
    assert _decode_jwt_payload("onlytwoparts") is None


def test_extract_account_id_success():
    jwt_claim_path = "https://api.openai.com/auth"
    payload = {jwt_claim_path: {"chatgpt_account_id": "acct-abc123"}}
    token = _make_jwt(payload)
    assert extract_account_id(token) == "acct-abc123"


def test_extract_account_id_missing_raises():
    token = _make_jwt({"sub": "user123"})
    with pytest.raises(ValueError, match="No chatgpt_account_id"):
        extract_account_id(token)


# ------------------------------------------------------------------ #
# Authorization URL                                                   #
# ------------------------------------------------------------------ #

def test_build_auth_url_contains_required_params():
    url = _build_auth_url(challenge="abc123", state="xyz456")
    assert AUTHORIZE_URL in url
    assert f"client_id={CLIENT_ID}" in url
    assert "code_challenge=abc123" in url
    assert "state=xyz456" in url
    assert "code_challenge_method=S256" in url
    assert "codex_cli_simplified_flow=true" in url
    assert "originator=pi" in url


def test_build_auth_url_custom_originator():
    url = _build_auth_url(challenge="c", state="s", originator="myapp")
    assert "originator=myapp" in url


# ------------------------------------------------------------------ #
# Input parsing                                                       #
# ------------------------------------------------------------------ #

def test_parse_full_redirect_url():
    url = "http://localhost:1455/auth/callback?code=AUTH_CODE&state=MY_STATE"
    code, state = _parse_authorization_input(url)
    assert code == "AUTH_CODE"
    assert state == "MY_STATE"


def test_parse_query_string():
    code, state = _parse_authorization_input("code=AUTH_CODE&state=MY_STATE")
    assert code == "AUTH_CODE"
    assert state == "MY_STATE"


def test_parse_bare_code():
    code, state = _parse_authorization_input("BARE_CODE_123")
    assert code == "BARE_CODE_123"
    assert state is None


def test_parse_empty_string():
    code, state = _parse_authorization_input("")
    assert code is None
    assert state is None


# ------------------------------------------------------------------ #
# OAuthCredentials                                                    #
# ------------------------------------------------------------------ #

def test_credentials_round_trip():
    original = {
        "refresh": "refresh_token_abc",
        "access": "access_token_xyz",
        "expires": 9999999999000,
        "accountId": "acct-abc123",
    }
    creds = OAuthCredentials.from_dict(original)
    assert creds.refresh == "refresh_token_abc"
    assert creds.access == "access_token_xyz"
    assert creds.expires == 9999999999000
    assert creds.get_extra("accountId") == "acct-abc123"

    # Round-trip back to dict
    serialized = creds.to_dict()
    assert serialized["refresh"] == "refresh_token_abc"
    assert serialized["accountId"] == "acct-abc123"
    assert "expires" in serialized


def test_credentials_is_expired_past():
    creds = OAuthCredentials(
        refresh="r", access="a",
        expires=int(time.time() * 1000) - 1000,  # 1 second ago
    )
    assert creds.is_expired(buffer_ms=0)


def test_credentials_not_expired_future():
    creds = OAuthCredentials(
        refresh="r", access="a",
        expires=int(time.time() * 1000) + 999999,  # far future
    )
    assert not creds.is_expired()
