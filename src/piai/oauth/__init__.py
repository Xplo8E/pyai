"""
OAuth provider registry and high-level API.

Mirrors src/utils/oauth/index.ts.
"""

from __future__ import annotations

from .openai_codex import OpenAICodexOAuthProvider
from .storage import get_provider_credentials, save_credentials
from .types import OAuthCredentials, OAuthLoginCallbacks, OAuthProviderInterface

# ------------------------------------------------------------------ #
# Registry                                                            #
# ------------------------------------------------------------------ #

_registry: dict[str, OAuthProviderInterface] = {}


def register_oauth_provider(provider: OAuthProviderInterface) -> None:
    _registry[provider.id] = provider


def get_oauth_provider(provider_id: str) -> OAuthProviderInterface | None:
    return _registry.get(provider_id)


def get_oauth_providers() -> list[OAuthProviderInterface]:
    return list(_registry.values())


def unregister_oauth_provider(provider_id: str) -> None:
    _registry.pop(provider_id, None)


# Register built-ins
register_oauth_provider(OpenAICodexOAuthProvider())


# ------------------------------------------------------------------ #
# High-level API                                                      #
# ------------------------------------------------------------------ #


async def get_oauth_api_key(
    provider_id: str,
    credentials: OAuthCredentials,
) -> tuple[OAuthCredentials, str]:
    """
    Return (updated_credentials, api_key), auto-refreshing if expired.

    Mirrors getOAuthApiKey() from index.ts.

    The 5-minute buffer prevents race conditions where the token expires
    between credential check and the actual API call.
    """
    provider = get_oauth_provider(provider_id)
    if not provider:
        raise ValueError(f"Unknown OAuth provider: {provider_id}")

    if credentials.is_expired(buffer_ms=5 * 60 * 1000):
        credentials = await provider.refresh_token(credentials)

    api_key = provider.get_api_key(credentials)
    return credentials, api_key


__all__ = [
    "OAuthCredentials",
    "OAuthLoginCallbacks",
    "OAuthProviderInterface",
    "get_oauth_api_key",
    "get_oauth_provider",
    "get_oauth_providers",
    "get_provider_credentials",
    "register_oauth_provider",
    "save_credentials",
    "unregister_oauth_provider",
]
