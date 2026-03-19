"""
PKCE (RFC 7636) code verifier and challenge generation.

Mirrors src/utils/oauth/pkce.ts — must produce identical output
so auth.json credentials are cross-compatible with the JS SDK.
"""

import base64
import hashlib
import os


def generate_pkce() -> tuple[str, str]:
    """
    Generate a PKCE (code_verifier, code_challenge) pair.

    Returns:
        (verifier, challenge) — both base64url-encoded, no padding.

    JS equivalent:
        const verifierBytes = new Uint8Array(32);
        crypto.getRandomValues(verifierBytes);
        const verifier = base64urlEncode(verifierBytes);
        const hashBuffer = await crypto.subtle.digest("SHA-256", encoder.encode(verifier));
        const challenge = base64urlEncode(new Uint8Array(hashBuffer));
    """
    # 32 random bytes → base64url verifier (no padding)
    verifier_bytes = os.urandom(32)
    verifier = _base64url_encode(verifier_bytes)

    # SHA-256(verifier as UTF-8 bytes) → base64url challenge (no padding)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = _base64url_encode(digest)

    return verifier, challenge


def _base64url_encode(data: bytes) -> str:
    """
    Encode bytes as base64url with no padding.

    JS equivalent:
        btoa(binary).replace(/\\+/g, "-").replace(/\\//g, "_").replace(/=/g, "")
    """
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")
