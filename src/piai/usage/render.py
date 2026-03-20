"""
Terminal renderer for UsageReport.

All display logic lives here — CLI just calls render(report).
Adding a new provider never requires touching this file unless
you need new display features.
"""

from __future__ import annotations

import datetime

from .report import UsageReport



_BAR_LEN = 22
_BOX_W   = 52  # inner width (between │ chars)


def render(report: UsageReport) -> None:
    """Print a formatted usage report to stdout."""
    if report.error:
        _warn(f"Warning: {report.error}")

    _render_header(report)
    _render_windows(report)


# ── header box ────────────────────────────────────────────────────────


def _render_header(report: UsageReport) -> None:
    _PLAN_LABELS = {
        "plus": "ChatGPT Plus",
        "pro": "ChatGPT Pro",
        "free": "Free",
    }
    plan_label = _PLAN_LABELS.get(report.plan, report.plan.title())
    title = f" {report.provider_id} "

    print(f"\n╭─{title}{'─' * (_BOX_W - len(title))}╮")
    _row("Plan", f"{report.plan}  ({plan_label})")

    sub = report.subscription
    if sub and sub.active:
        renews_fmt = _fmt_date(sub.renews_at) if sub.renews_at else "—"
        billing_str = f"{sub.billing_period}  ·  renews {renews_fmt}"
        if sub.billing_currency:
            billing_str += f"  ({sub.billing_currency})"
        _row("Billing", billing_str)

        if sub.discount_pct > 0 and sub.discount_expires_at:
            try:
                exp_dt = datetime.datetime.fromisoformat(
                    sub.discount_expires_at.replace("Z", "+00:00")
                )
                if exp_dt > datetime.datetime.now(tz=datetime.timezone.utc):
                    _row("Discount", f"{int(sub.discount_pct)}% off  until {_fmt_date(sub.discount_expires_at)}")
            except Exception:
                pass

    print(f"╰{'─' * (_BOX_W + 2)}╯")


def _row(label: str, value: str) -> None:
    line = f"  {label:<10}{value}"
    # Truncate if too wide for the box
    if len(line) > _BOX_W:
        line = line[:_BOX_W - 1] + "…"
    print(f"│{line:<{_BOX_W + 2}}│")


# ── usage windows ─────────────────────────────────────────────────────


def _render_windows(report: UsageReport) -> None:
    if not report.windows:
        print("\n  No usage data available.\n")
        return

    # Group preserving insertion order
    groups: dict[str, list] = {}
    for w in report.windows:
        groups.setdefault(w.group, []).append(w)

    print()
    for group_name, entries in groups.items():
        print(f"  {group_name}")
        for i, w in enumerate(entries):
            pct = w.used_percent
            filled = round(_BAR_LEN * pct / 100)
            bar_inner = "█" * filled + "░" * (_BAR_LEN - filled)
            bar = f"[{bar_inner}]"

            if i == 0 and len(entries) > 1:
                prefix = "┌"
            elif i == len(entries) - 1:
                prefix = "└"
            else:
                prefix = "├"

            pct_str = f"{pct}%".rjust(4)
            print(f"  {prefix} {w.name:<8}  {bar}  {pct_str}   resets in {w.resets_str}")
        print()


# ── helpers ───────────────────────────────────────────────────────────


def _fmt_date(iso_str: str) -> str:
    try:
        dt = datetime.datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return dt.strftime("%b %-d, %Y")
    except Exception:
        return iso_str


def _warn(msg: str) -> None:
    import sys
    print(msg, file=sys.stderr)
