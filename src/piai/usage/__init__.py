"""
Usage reporting subpackage.

Each provider module exposes:
    async def fetch(token, account_id, base_url=None) -> UsageReport

The CLI calls get_provider_usage(provider_id, creds) and passes the result
to render(report) for display.
"""

from .report import UsageReport, UsageWindow
from .render import render
from .registry import get_provider_usage

__all__ = ["UsageReport", "UsageWindow", "render", "get_provider_usage"]
