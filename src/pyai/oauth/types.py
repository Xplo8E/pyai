"""
OAuth credential types and provider interface.

Mirrors src/utils/oauth/types.ts.

auth.json compatibility note:
    Keys must remain camelCase (accountId, not account_id) in serialized form
    so credentials are interoperable with the JS SDK's auth.json format.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class OAuthCredentials:
    """
    Stored OAuth credentials for a provider.

    JS equivalent:
        type OAuthCredentials = {
            refresh: string;
            access: string;
            expires: number;   // Unix ms timestamp
            [key: string]: unknown;
        }
    """

    refresh: str
    access: str
    expires: int  # Unix millisecond timestamp — same as JS Date.now()
    # Provider-specific extras stored as flat dict (e.g. accountId for openai-codex)
    extras: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Serialization — must match JS auth.json structure exactly           #
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        """Serialize to auth.json-compatible dict (camelCase keys)."""
        d: dict[str, Any] = {
            "refresh": self.refresh,
            "access": self.access,
            "expires": self.expires,
        }
        d.update(self.extras)  # e.g. accountId, projectId, ...
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> OAuthCredentials:
        """Deserialize from auth.json dict."""
        known = {"refresh", "access", "expires"}
        extras = {k: v for k, v in d.items() if k not in known}
        return cls(
            refresh=d["refresh"],
            access=d["access"],
            expires=d["expires"],
            extras=extras,
        )

    # ------------------------------------------------------------------ #
    # Helpers                                                             #
    # ------------------------------------------------------------------ #

    def is_expired(self, buffer_ms: int = 5 * 60 * 1000) -> bool:
        """True if the access token expires within buffer_ms milliseconds."""
        return int(time.time() * 1000) >= self.expires - buffer_ms

    def get_extra(self, key: str) -> Any:
        return self.extras.get(key)


@dataclass
class OAuthPrompt:
    message: str
    placeholder: str | None = None
    allow_empty: bool = False


@dataclass
class OAuthAuthInfo:
    url: str
    instructions: str | None = None


class OAuthLoginCallbacks:
    """
    Callback bundle passed to OAuthProviderInterface.login().

    Mirrors OAuthLoginCallbacks interface from types.ts.
    """

    def __init__(
        self,
        on_auth: Any,  # callable(OAuthAuthInfo) -> None
        on_prompt: Any,  # async callable(OAuthPrompt) -> str
        on_progress: Any | None = None,  # callable(str) -> None
        on_manual_code_input: Any | None = None,  # async callable() -> str
    ):
        self.on_auth = on_auth
        self.on_prompt = on_prompt
        self.on_progress = on_progress
        self.on_manual_code_input = on_manual_code_input


class OAuthProviderInterface(ABC):
    """
    Abstract base for OAuth providers.

    Mirrors OAuthProviderInterface from types.ts.
    """

    id: str
    name: str
    uses_callback_server: bool = False

    @abstractmethod
    async def login(self, callbacks: OAuthLoginCallbacks) -> OAuthCredentials:
        """Run the full login flow, return credentials to persist."""
        ...

    @abstractmethod
    async def refresh_token(self, credentials: OAuthCredentials) -> OAuthCredentials:
        """Refresh expired credentials, return updated credentials to persist."""
        ...

    @abstractmethod
    def get_api_key(self, credentials: OAuthCredentials) -> str:
        """Convert credentials to the API key string used in Authorization header."""
        ...
