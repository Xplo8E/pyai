"""
auth.json credential storage.

Mirrors credential persistence from src/cli.ts.

File format is identical to the JS SDK so credentials are cross-compatible:
    {
        "openai-codex": {
            "refresh": "...",
            "access": "...",
            "expires": 1234567890000,
            "accountId": "..."
        }
    }
"""

from __future__ import annotations

import json
from pathlib import Path

from .types import OAuthCredentials

# Default: auth.json in current working directory — same as JS CLI
_DEFAULT_AUTH_FILE = Path("auth.json")


def _auth_file() -> Path:
    return _DEFAULT_AUTH_FILE


def load_all_credentials() -> dict[str, dict]:
    """Load the full auth.json dict. Returns {} if file doesn't exist."""
    f = _auth_file()
    if not f.exists():
        return {}
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def get_provider_credentials(provider_id: str) -> OAuthCredentials | None:
    """Return credentials for a provider, or None if not logged in."""
    all_creds = load_all_credentials()
    raw = all_creds.get(provider_id)
    if not raw:
        return None
    return OAuthCredentials.from_dict(raw)


def save_credentials(provider_id: str, creds: OAuthCredentials) -> None:
    """Persist credentials for a provider to auth.json."""
    all_creds = load_all_credentials()
    all_creds[provider_id] = creds.to_dict()
    _auth_file().write_text(
        json.dumps(all_creds, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def delete_credentials(provider_id: str) -> None:
    """Remove credentials for a provider from auth.json."""
    all_creds = load_all_credentials()
    all_creds.pop(provider_id, None)
    _auth_file().write_text(
        json.dumps(all_creds, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
