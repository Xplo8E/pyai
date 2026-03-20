"""Typed data structures for usage reports."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class UsageWindow:
    """A single rate-limit window (e.g. 5-hour or weekly)."""
    name: str               # "5-hour", "weekly", etc.
    group: str              # display group: "Codex", "Code review", etc.
    used_percent: int       # 0-100
    resets_str: str         # human-readable time until reset, e.g. "2h 15m"
    resets_at: float | None = None  # Unix seconds


@dataclass
class Subscription:
    active: bool
    plan_label: str = ""         # "chatgptplusplan" etc.
    billing_period: str = ""     # "monthly", "yearly"
    billing_currency: str = ""
    renews_at: str = ""          # ISO 8601 string
    discount_pct: float = 0.0
    discount_expires_at: str = ""


@dataclass
class UsageReport:
    """Normalized usage report returned by any provider's fetch()."""
    provider_id: str
    plan: str                           # "plus", "pro", "free", "unknown"
    windows: list[UsageWindow] = field(default_factory=list)
    subscription: Subscription | None = None
    error: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)  # full raw API response
