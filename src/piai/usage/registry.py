"""
Provider registry for usage fetchers.

To add a new provider:
  1. Create src/piai/usage/<provider_id>.py with an async fetch(token, account_id) -> UsageReport
  2. Add an entry to _FETCHERS below

The CLI never needs to change.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..oauth.types import OAuthCredentials
    from .report import UsageReport

# Map provider_id → module path (lazy import to avoid loading all providers at startup)
_FETCHERS: dict[str, str] = {
    "openai-codex": "piai.usage.openai_codex",
}


async def get_provider_usage(provider_id: str, creds: "OAuthCredentials") -> "UsageReport":
    """
    Fetch usage for a provider using its stored credentials.

    Raises ValueError if provider has no usage fetcher registered.
    """
    module_path = _FETCHERS.get(provider_id)
    if module_path is None:
        from .report import UsageReport
        return UsageReport(
            provider_id=provider_id,
            plan="unknown",
            error=f"Usage reporting not supported for provider: {provider_id}",
        )

    import importlib
    module = importlib.import_module(module_path)
    token = creds.access
    account_id = creds.get_extra("accountId") or ""
    return await module.fetch(token=token, account_id=account_id)
