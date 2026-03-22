"""
Usage fetcher for the openai-codex (ChatGPT Plus/Pro) provider.

Hits:
  /backend-api/accounts/check/v4-2023-04-27  → plan, subscription info
  /backend-api/wham/usage                     → 5-hour + weekly counters
"""

from __future__ import annotations

import datetime
import time
from typing import TYPE_CHECKING, Any

import httpx

from ..providers.openai_codex import DEFAULT_CODEX_BASE_URL, _user_agent
from .report import Subscription, UsageReport, UsageWindow

if TYPE_CHECKING:
    from ..oauth.types import OAuthCredentials


async def fetch(
    token: str,
    account_id: str,
    base_url: str | None = None,
) -> UsageReport:
    base = (base_url or DEFAULT_CODEX_BASE_URL).rstrip("/")
    headers = {
        "Authorization": f"Bearer {token}",
        "chatgpt-account-id": account_id,
        "User-Agent": _user_agent(),
        "accept": "application/json",
        "content-type": "application/json",
        "originator": "pi",
        "OpenAI-Beta": "responses=experimental",
    }

    report = UsageReport(provider_id="openai-codex", plan="unknown")

    async with httpx.AsyncClient(timeout=httpx.Timeout(20.0), follow_redirects=True) as client:
        # ── accounts/check ────────────────────────────────────────────
        try:
            resp = await client.get(f"{base}/accounts/check/v4-2023-04-27", headers=headers)
            data = resp.json()
            report.raw["check"] = data
            if resp.is_success:
                _apply_accounts_check(report, data)
            else:
                report.error = f"accounts/check HTTP {resp.status_code}"
        except Exception as e:
            report.error = f"accounts/check failed: {e}"

        # ── wham/usage ────────────────────────────────────────────────
        try:
            resp = await client.get(f"{base}/wham/usage", headers=headers)
            if resp.is_success:
                data = resp.json()
                report.raw["wham"] = data
                report.windows = _parse_wham(data)
            else:
                report.raw["wham"] = {"status_code": resp.status_code}
        except Exception as e:
            existing = report.error or ""
            report.error = (existing + f" | wham/usage failed: {e}").lstrip(" | ")

    return report


# ── parsers ───────────────────────────────────────────────────────────


def _apply_accounts_check(report: UsageReport, data: dict[str, Any]) -> None:
    accounts_raw = data.get("accounts") or {}
    if isinstance(accounts_raw, dict):
        ordering = data.get("account_ordering") or list(accounts_raw.keys())
        primary_id = ordering[0] if ordering else None
        acc = accounts_raw.get(primary_id) or next(iter(accounts_raw.values()), {})
    elif isinstance(accounts_raw, list) and accounts_raw:
        acc = accounts_raw[0]
    else:
        return

    # Plan
    plan_raw = (
        (acc.get("account") or {}).get("plan_type")
        or (acc.get("entitlement") or {}).get("subscription_plan")
        or "unknown"
    )
    plan_clean = plan_raw.lower().replace("chatgpt", "").replace("plan", "").strip()
    report.plan = plan_clean or plan_raw.lower()

    # Subscription
    ent = acc.get("entitlement") or {}
    if ent.get("has_active_subscription"):
        discount = ent.get("discount") or {}
        report.subscription = Subscription(
            active=True,
            plan_label=ent.get("subscription_plan", ""),
            billing_period=ent.get("billing_period", ""),
            billing_currency=ent.get("billing_currency", ""),
            renews_at=ent.get("renews_at") or ent.get("expires_at") or "",
            discount_pct=float(discount.get("amount") or 0),
            discount_expires_at=discount.get("discount_expires_at") or "",
        )
    else:
        report.subscription = Subscription(active=False)


def _build_window(group: str, name: str, w: dict[str, Any]) -> UsageWindow:
    pct = w.get("used_percent", 0)
    reset_at = w.get("reset_at")
    reset_after = w.get("reset_after_seconds")
    if reset_at:
        resets_str = _fmt_reset(reset_at)
    elif reset_after:
        resets_str = _fmt_seconds(reset_after)
    else:
        resets_str = "unknown"
    return UsageWindow(
        name=name,
        group=group,
        used_percent=pct,
        resets_str=resets_str,
        resets_at=reset_at,
    )


def _parse_wham(data: dict[str, Any]) -> list[UsageWindow]:
    entries: list[UsageWindow] = []

    rl = data.get("rate_limit") or {}
    if rl.get("primary_window"):
        entries.append(_build_window("Codex", "5-hour", rl["primary_window"]))
    if rl.get("secondary_window"):
        entries.append(_build_window("Codex", "weekly", rl["secondary_window"]))

    crl = data.get("code_review_rate_limit") or {}
    if crl.get("primary_window"):
        entries.append(_build_window("Code review", "weekly", crl["primary_window"]))
    if crl.get("secondary_window"):
        entries.append(_build_window("Code review", "secondary", crl["secondary_window"]))

    return entries


# ── time helpers ──────────────────────────────────────────────────────


def _fmt_reset(unix_seconds: float) -> str:
    total = int(unix_seconds - time.time())
    return _fmt_seconds(max(0, total))


def _fmt_seconds(total: int) -> str:
    if total <= 0:
        return "now"
    if total < 60:
        return f"{total}s"
    if total < 3600:
        return f"{total // 60}m"
    if total < 86400:
        h, rem = divmod(total, 3600)
        return f"{h}h {rem // 60}m" if rem >= 60 else f"{h}h"
    d, rem = divmod(total, 86400)
    return f"{d}d {rem // 3600}h" if rem >= 3600 else f"{d}d"


def fmt_date(iso_str: str) -> str:
    """'2026-04-07T06:28:14+00:00'  →  'Apr 7, 2026'"""
    try:
        dt = datetime.datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return dt.strftime("%b %-d, %Y")
    except Exception:
        return iso_str
