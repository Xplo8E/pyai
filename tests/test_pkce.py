"""
Tests for PKCE implementation.

Verifies our Python output matches the expected base64url format
that the JS SDK produces.
"""

import base64
import hashlib

from pyai.oauth.pkce import generate_pkce, _base64url_encode


def test_generate_pkce_returns_strings():
    verifier, challenge = generate_pkce()
    assert isinstance(verifier, str)
    assert isinstance(challenge, str)


def test_verifier_is_base64url_no_padding():
    verifier, _ = generate_pkce()
    assert "+" not in verifier
    assert "/" not in verifier
    assert "=" not in verifier


def test_challenge_is_base64url_no_padding():
    _, challenge = generate_pkce()
    assert "+" not in challenge
    assert "/" not in challenge
    assert "=" not in challenge


def test_challenge_is_sha256_of_verifier():
    verifier, challenge = generate_pkce()
    # Re-derive challenge from verifier
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    expected = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    assert challenge == expected


def test_verifier_is_32_bytes_encoded():
    """32 random bytes → base64url = 43 chars (ceil(32*8/6) = 43, no padding)."""
    verifier, _ = generate_pkce()
    assert len(verifier) == 43


def test_base64url_encode_no_padding():
    data = b"\xff\xfe\xfd"
    result = _base64url_encode(data)
    assert "=" not in result
    assert "+" not in result
    assert "/" not in result


def test_each_call_produces_unique_verifier():
    v1, _ = generate_pkce()
    v2, _ = generate_pkce()
    assert v1 != v2
