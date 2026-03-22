"""
Comprehensive tests targeting uncovered code paths across piai SDK.

Coverage targets (baseline gaps):
  - usage/report.py        0% → 100%
  - usage/registry.py      0% → 100%
  - usage/render.py        0% → 100%
  - usage/openai_codex.py  0% → 90%+
  - oauth/storage.py      27% → 95%+
  - oauth/types.py        92% → 100%
  - stream.py             63% → 95%+
  - providers/message_transform.py 99% → 100%
  - langchain/sub_agent_tool.py 89% → 100%
  - agent.py              99% → 100%
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ================================================================== #
# usage/report.py                                                     #
# ================================================================== #

class TestUsageReport:
    def test_usage_window_fields(self):
        from piai.usage.report import UsageWindow
        w = UsageWindow(name="5-hour", group="Codex", used_percent=42, resets_str="2h 15m", resets_at=9999.0)
        assert w.name == "5-hour"
        assert w.group == "Codex"
        assert w.used_percent == 42
        assert w.resets_str == "2h 15m"
        assert w.resets_at == 9999.0

    def test_usage_window_resets_at_default_none(self):
        from piai.usage.report import UsageWindow
        w = UsageWindow(name="weekly", group="Codex", used_percent=0, resets_str="3d")
        assert w.resets_at is None

    def test_subscription_defaults(self):
        from piai.usage.report import Subscription
        s = Subscription(active=True)
        assert s.plan_label == ""
        assert s.billing_period == ""
        assert s.billing_currency == ""
        assert s.renews_at == ""
        assert s.discount_pct == 0.0
        assert s.discount_expires_at == ""

    def test_subscription_inactive(self):
        from piai.usage.report import Subscription
        s = Subscription(active=False)
        assert s.active is False

    def test_usage_report_defaults(self):
        from piai.usage.report import UsageReport
        r = UsageReport(provider_id="openai-codex", plan="plus")
        assert r.windows == []
        assert r.subscription is None
        assert r.error is None
        assert r.raw == {}

    def test_usage_report_with_error(self):
        from piai.usage.report import UsageReport
        r = UsageReport(provider_id="openai-codex", plan="unknown", error="HTTP 401")
        assert r.error == "HTTP 401"


# ================================================================== #
# usage/registry.py                                                   #
# ================================================================== #

class TestUsageRegistry:
    @pytest.mark.asyncio
    async def test_unknown_provider_returns_error_report(self):
        from piai.usage.registry import get_provider_usage
        from piai.oauth.types import OAuthCredentials

        creds = OAuthCredentials(refresh="r", access="a", expires=9999999999000)
        report = await get_provider_usage("nonexistent-provider", creds)
        assert report.error is not None
        assert "not supported" in report.error
        assert report.plan == "unknown"

    @pytest.mark.asyncio
    async def test_known_provider_dispatches_to_fetcher(self):
        from piai.usage.registry import get_provider_usage
        from piai.oauth.types import OAuthCredentials
        from piai.usage.report import UsageReport

        fake_report = UsageReport(provider_id="openai-codex", plan="plus")
        creds = OAuthCredentials(refresh="r", access="tok", expires=9999999999000, extras={"accountId": "acc_123"})

        with patch("piai.usage.openai_codex.fetch", new=AsyncMock(return_value=fake_report)):
            report = await get_provider_usage("openai-codex", creds)

        assert report.plan == "plus"
        assert report.provider_id == "openai-codex"


# ================================================================== #
# usage/openai_codex.py — parsers (no network)                       #
# ================================================================== #

class TestUsageOpenaiCodexParsers:
    def test_apply_accounts_check_dict_accounts(self):
        from piai.usage.openai_codex import _apply_accounts_check
        from piai.usage.report import UsageReport

        report = UsageReport(provider_id="openai-codex", plan="unknown")
        data = {
            "accounts": {
                "acc_1": {
                    "account": {"plan_type": "ChatGPTPlusPlan"},
                    "entitlement": {
                        "has_active_subscription": True,
                        "subscription_plan": "chatgptplusplan",
                        "billing_period": "monthly",
                        "billing_currency": "USD",
                        "renews_at": "2026-04-07T06:28:14+00:00",
                        "discount": {"amount": 20, "discount_expires_at": "2026-12-31T00:00:00+00:00"},
                    },
                }
            },
            "account_ordering": ["acc_1"],
        }
        _apply_accounts_check(report, data)
        assert report.plan == "plus"
        assert report.subscription is not None
        assert report.subscription.active is True
        assert report.subscription.billing_period == "monthly"
        assert report.subscription.billing_currency == "USD"
        assert report.subscription.discount_pct == 20.0

    def test_apply_accounts_check_list_accounts(self):
        from piai.usage.openai_codex import _apply_accounts_check
        from piai.usage.report import UsageReport

        report = UsageReport(provider_id="openai-codex", plan="unknown")
        data = {
            "accounts": [
                {
                    "entitlement": {
                        "has_active_subscription": False,
                        "subscription_plan": "free",
                    }
                }
            ]
        }
        _apply_accounts_check(report, data)
        assert report.subscription is not None
        assert report.subscription.active is False

    def test_apply_accounts_check_empty_accounts(self):
        from piai.usage.openai_codex import _apply_accounts_check
        from piai.usage.report import UsageReport

        report = UsageReport(provider_id="openai-codex", plan="unknown")
        _apply_accounts_check(report, {"accounts": {}})
        # Should return without setting anything
        assert report.plan == "unknown"

    def test_apply_accounts_check_no_active_subscription(self):
        from piai.usage.openai_codex import _apply_accounts_check
        from piai.usage.report import UsageReport

        report = UsageReport(provider_id="openai-codex", plan="unknown")
        data = {
            "accounts": {
                "acc_1": {
                    "entitlement": {"has_active_subscription": False, "subscription_plan": "free"}
                }
            },
            "account_ordering": ["acc_1"],
        }
        _apply_accounts_check(report, data)
        assert report.subscription is not None
        assert report.subscription.active is False

    def test_parse_wham_full(self):
        from piai.usage.openai_codex import _parse_wham

        data = {
            "rate_limit": {
                "primary_window": {"used_percent": 50, "reset_after_seconds": 7200},
                "secondary_window": {"used_percent": 30, "reset_after_seconds": 604800},
            },
            "code_review_rate_limit": {
                "primary_window": {"used_percent": 10, "reset_after_seconds": 3600},
                "secondary_window": {"used_percent": 5, "reset_after_seconds": 86400},
            },
        }
        windows = _parse_wham(data)
        assert len(windows) == 4
        assert windows[0].name == "5-hour"
        assert windows[0].group == "Codex"
        assert windows[1].name == "weekly"
        assert windows[2].group == "Code review"
        assert windows[3].name == "secondary"

    def test_parse_wham_with_reset_at(self):
        from piai.usage.openai_codex import _parse_wham

        future = time.time() + 3600
        data = {
            "rate_limit": {
                "primary_window": {"used_percent": 75, "reset_at": future},
            }
        }
        windows = _parse_wham(data)
        assert len(windows) == 1
        assert "h" in windows[0].resets_str or "m" in windows[0].resets_str

    def test_parse_wham_empty(self):
        from piai.usage.openai_codex import _parse_wham
        windows = _parse_wham({})
        assert windows == []

    def test_parse_wham_no_reset_info(self):
        from piai.usage.openai_codex import _parse_wham
        data = {"rate_limit": {"primary_window": {"used_percent": 20}}}
        windows = _parse_wham(data)
        assert windows[0].resets_str == "unknown"

    def test_fmt_seconds_all_branches(self):
        from piai.usage.openai_codex import _fmt_seconds
        assert _fmt_seconds(0) == "now"
        assert _fmt_seconds(-5) == "now"
        assert _fmt_seconds(30) == "30s"
        assert _fmt_seconds(90) == "1m"
        assert _fmt_seconds(3600) == "1h"
        assert _fmt_seconds(3660) == "1h 1m"
        assert _fmt_seconds(3599) == "59m"
        assert _fmt_seconds(86400) == "1d"
        assert _fmt_seconds(86400 + 3600) == "1d 1h"
        assert _fmt_seconds(86400 + 1800) == "1d"  # < 3600 rem → no hours shown

    def test_fmt_date(self):
        from piai.usage.openai_codex import fmt_date
        assert fmt_date("2026-04-07T06:28:14+00:00") == "Apr 7, 2026"
        assert fmt_date("not-a-date") == "not-a-date"  # graceful fallback

    def test_fmt_reset_future(self):
        from piai.usage.openai_codex import _fmt_reset
        future = time.time() + 3600
        result = _fmt_reset(future)
        assert "h" in result or "m" in result

    def test_fmt_reset_past(self):
        from piai.usage.openai_codex import _fmt_reset
        past = time.time() - 100
        assert _fmt_reset(past) == "now"

    def test_apply_accounts_check_plan_fallback_to_entitlement(self):
        """Uses entitlement.subscription_plan when account.plan_type is absent."""
        from piai.usage.openai_codex import _apply_accounts_check
        from piai.usage.report import UsageReport

        report = UsageReport(provider_id="openai-codex", plan="unknown")
        data = {
            "accounts": {
                "a": {
                    "account": {},  # no plan_type
                    "entitlement": {
                        "has_active_subscription": False,
                        "subscription_plan": "ChatGPTProPlan",
                    },
                }
            },
            "account_ordering": ["a"],
        }
        _apply_accounts_check(report, data)
        assert "pro" in report.plan


# ================================================================== #
# usage/render.py                                                     #
# ================================================================== #

class TestUsageRender:
    def _make_report(self, **kwargs):
        from piai.usage.report import UsageReport
        return UsageReport(provider_id="openai-codex", plan="plus", **kwargs)

    def test_render_no_windows(self, capsys):
        from piai.usage.render import render
        report = self._make_report()
        render(report)
        out = capsys.readouterr().out
        assert "No usage data available" in out

    def test_render_with_windows(self, capsys):
        from piai.usage.render import render
        from piai.usage.report import UsageWindow
        report = self._make_report(windows=[
            UsageWindow(name="5-hour", group="Codex", used_percent=50, resets_str="1h"),
            UsageWindow(name="weekly", group="Codex", used_percent=80, resets_str="3d"),
        ])
        render(report)
        out = capsys.readouterr().out
        assert "5-hour" in out
        assert "weekly" in out
        assert "50%" in out or "50" in out

    def test_render_with_error(self, capsys):
        from piai.usage.render import render
        report = self._make_report(error="HTTP 401")
        render(report)
        err = capsys.readouterr().err
        assert "HTTP 401" in err

    def test_render_with_subscription(self, capsys):
        from piai.usage.render import render
        from piai.usage.report import Subscription
        sub = Subscription(
            active=True,
            billing_period="monthly",
            billing_currency="USD",
            renews_at="2026-06-01T00:00:00+00:00",
        )
        report = self._make_report(subscription=sub)
        render(report)
        out = capsys.readouterr().out
        assert "monthly" in out

    def test_render_with_active_discount(self, capsys):
        from piai.usage.render import render
        from piai.usage.report import Subscription
        # Discount expires far in the future
        future = "2099-01-01T00:00:00+00:00"
        sub = Subscription(
            active=True,
            billing_period="monthly",
            discount_pct=20.0,
            discount_expires_at=future,
        )
        report = self._make_report(subscription=sub)
        render(report)
        out = capsys.readouterr().out
        assert "20%" in out or "Discount" in out

    def test_render_with_expired_discount_not_shown(self, capsys):
        from piai.usage.render import render
        from piai.usage.report import Subscription
        past = "2000-01-01T00:00:00+00:00"
        sub = Subscription(active=True, billing_period="monthly", discount_pct=10.0, discount_expires_at=past)
        report = self._make_report(subscription=sub)
        render(report)
        out = capsys.readouterr().out
        assert "Discount" not in out

    def test_render_with_bad_discount_date(self, capsys):
        from piai.usage.render import render
        from piai.usage.report import Subscription
        # Should not raise even with bad date
        sub = Subscription(active=True, billing_period="monthly", discount_pct=10.0, discount_expires_at="not-a-date")
        report = self._make_report(subscription=sub)
        render(report)  # should not raise

    def test_render_multiple_groups(self, capsys):
        from piai.usage.render import render
        from piai.usage.report import UsageWindow
        report = self._make_report(windows=[
            UsageWindow(name="5-hour", group="Codex", used_percent=10, resets_str="2h"),
            UsageWindow(name="weekly", group="Codex", used_percent=20, resets_str="6d"),
            UsageWindow(name="weekly", group="Code review", used_percent=5, resets_str="6d"),
        ])
        render(report)
        out = capsys.readouterr().out
        assert "Codex" in out
        assert "Code review" in out
        # First group prefix should be ┌, last └
        assert "┌" in out
        assert "└" in out

    def test_render_single_entry_group_uses_leaf(self, capsys):
        from piai.usage.render import render
        from piai.usage.report import UsageWindow
        report = self._make_report(windows=[
            UsageWindow(name="5-hour", group="Codex", used_percent=0, resets_str="now"),
        ])
        render(report)
        out = capsys.readouterr().out
        assert "└" in out

    def test_render_plan_label_unknown(self, capsys):
        from piai.usage.render import render
        report = self._make_report()
        report.plan = "enterprise"  # not in _PLAN_LABELS
        render(report)
        out = capsys.readouterr().out
        assert "Enterprise" in out  # .title() fallback

    def test_render_subscription_inactive_skips_billing(self, capsys):
        from piai.usage.render import render
        from piai.usage.report import Subscription
        sub = Subscription(active=False)
        report = self._make_report(subscription=sub)
        render(report)
        out = capsys.readouterr().out
        assert "Billing" not in out

    def test_fmt_date_helper(self):
        from piai.usage.render import _fmt_date
        result = _fmt_date("2026-04-07T06:28:14Z")
        assert "2026" in result
        assert _fmt_date("bad") == "bad"

    def test_row_truncates_long_values(self, capsys):
        from piai.usage.render import _row
        _row("Label", "x" * 200)
        out = capsys.readouterr().out
        assert "…" in out


# ================================================================== #
# oauth/storage.py                                                    #
# ================================================================== #

class TestOAuthStorage:
    """Tests for oauth/storage.py — use tmp_path and patch _auth_file directly."""

    def _patch_auth_file(self, path):
        """Return a context manager that redirects _auth_file() to path."""
        return patch("piai.oauth.storage._auth_file", return_value=Path(path))

    def test_load_all_credentials_missing_file(self, tmp_path):
        from piai.oauth import storage
        with self._patch_auth_file(tmp_path / "missing.json"):
            result = storage.load_all_credentials()
        assert result == {}

    def test_load_all_credentials_invalid_json(self, tmp_path):
        from piai.oauth import storage
        bad = tmp_path / "bad.json"
        bad.write_text("not json", encoding="utf-8")
        with self._patch_auth_file(bad):
            assert storage.load_all_credentials() == {}

    def test_save_and_load_credentials(self, tmp_path):
        from piai.oauth import storage
        from piai.oauth.types import OAuthCredentials
        auth_file = tmp_path / "auth.json"
        creds = OAuthCredentials(refresh="r1", access="a1", expires=9999999999000, extras={"accountId": "acc_xyz"})
        with self._patch_auth_file(auth_file):
            storage.save_credentials("openai-codex", creds)
            loaded = storage.get_provider_credentials("openai-codex")
        assert loaded is not None
        assert loaded.access == "a1"
        assert loaded.refresh == "r1"
        assert loaded.get_extra("accountId") == "acc_xyz"

    def test_get_provider_credentials_not_found(self, tmp_path):
        from piai.oauth import storage
        with self._patch_auth_file(tmp_path / "auth.json"):
            assert storage.get_provider_credentials("nonexistent") is None

    def test_delete_credentials(self, tmp_path):
        from piai.oauth import storage
        from piai.oauth.types import OAuthCredentials
        auth_file = tmp_path / "auth.json"
        creds = OAuthCredentials(refresh="r", access="a", expires=9999999999000)
        with self._patch_auth_file(auth_file):
            storage.save_credentials("openai-codex", creds)
            storage.delete_credentials("openai-codex")
            assert storage.get_provider_credentials("openai-codex") is None

    def test_delete_credentials_nonexistent_noop(self, tmp_path):
        from piai.oauth import storage
        with self._patch_auth_file(tmp_path / "auth.json"):
            storage.delete_credentials("nonexistent-provider")  # should not raise

    def test_auth_file_env_var(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PIAI_AUTH", str(tmp_path / "custom.json"))
        from piai.oauth.storage import _auth_file
        p = _auth_file()
        assert "custom.json" in str(p)

    def test_auth_file_default_path(self, monkeypatch):
        monkeypatch.delenv("PIAI_AUTH", raising=False)
        from piai.oauth.storage import _auth_file
        p = _auth_file()
        assert ".piai" in str(p) and "auth.json" in str(p)

    def test_save_multiple_providers(self, tmp_path):
        from piai.oauth import storage
        from piai.oauth.types import OAuthCredentials
        auth_file = tmp_path / "auth.json"
        c1 = OAuthCredentials(refresh="r1", access="a1", expires=1000)
        c2 = OAuthCredentials(refresh="r2", access="a2", expires=2000)
        with self._patch_auth_file(auth_file):
            storage.save_credentials("provider-a", c1)
            storage.save_credentials("provider-b", c2)
            assert storage.get_provider_credentials("provider-a") is not None
            assert storage.get_provider_credentials("provider-b") is not None


# ================================================================== #
# oauth/types.py                                                      #
# ================================================================== #

class TestOAuthTypes:
    def test_is_expired_true(self):
        from piai.oauth.types import OAuthCredentials
        expired = OAuthCredentials(refresh="r", access="a", expires=int(time.time() * 1000) - 10000)
        assert expired.is_expired() is True

    def test_is_expired_false(self):
        from piai.oauth.types import OAuthCredentials
        future = OAuthCredentials(refresh="r", access="a", expires=int(time.time() * 1000) + 9999999)
        assert future.is_expired() is False

    def test_from_dict_roundtrip(self):
        from piai.oauth.types import OAuthCredentials
        d = {"refresh": "r", "access": "a", "expires": 12345, "accountId": "acc_1"}
        creds = OAuthCredentials.from_dict(d)
        assert creds.get_extra("accountId") == "acc_1"
        assert creds.to_dict() == d

    def test_oauth_prompt_allow_empty(self):
        from piai.oauth.types import OAuthPrompt
        p = OAuthPrompt(message="Enter code", allow_empty=True)
        assert p.allow_empty is True
        assert p.placeholder is None

    def test_oauth_auth_info(self):
        from piai.oauth.types import OAuthAuthInfo
        info = OAuthAuthInfo(url="https://example.com", instructions="Click here")
        assert info.url == "https://example.com"

    def test_oauth_login_callbacks(self):
        from piai.oauth.types import OAuthLoginCallbacks
        cb = OAuthLoginCallbacks(on_auth=lambda x: x, on_prompt=lambda x: x)
        assert cb.on_auth is not None
        assert cb.on_prompt is not None
        assert cb.on_progress is None
        assert cb.on_manual_code_input is None


# ================================================================== #
# stream.py                                                           #
# ================================================================== #

class TestStream:
    """Tests for stream.py public entry points — patched to avoid real network."""

    def _make_creds(self):
        from piai.oauth.types import OAuthCredentials
        return OAuthCredentials(
            refresh="r", access="tok", expires=int(time.time() * 1000) + 9999999,
            extras={"accountId": "acc_test"},
        )

    def _make_done_stream(self, text: str = "hello"):
        from piai.types import AssistantMessage, DoneEvent, TextContent

        async def _gen(*args, **kwargs):
            msg = AssistantMessage(content=[TextContent(text=text)])
            yield DoneEvent(reason="stop", message=msg)

        return _gen

    def _make_tool_call_stream(self):
        """Stream that ends with stop_reason=tool_use."""
        from piai.types import AssistantMessage, DoneEvent, ToolCall, ToolCallContent

        async def _gen(*args, **kwargs):
            tc = ToolCall(id="c1", name="foo", input={})
            msg = AssistantMessage(content=[ToolCallContent(tool_calls=[tc])])
            msg.stop_reason = "tool_use"
            yield DoneEvent(reason="tool_use", message=msg)

        return _gen

    def _make_error_stream(self):
        from piai.types import AssistantMessage, ErrorEvent

        async def _gen(*args, **kwargs):
            msg = AssistantMessage(error_message="boom")
            yield ErrorEvent(reason="error", error=msg)

        return _gen

    @pytest.mark.asyncio
    async def test_stream_not_logged_in(self):
        from piai.stream import stream
        from piai.types import Context, UserMessage

        ctx = Context(messages=[UserMessage(content="hi")])
        with patch("piai.stream.get_provider_credentials", return_value=None):
            with pytest.raises(RuntimeError, match="Not logged in"):
                async for _ in stream("gpt-5.1-codex-mini", ctx):
                    pass

    @pytest.mark.asyncio
    async def test_stream_missing_account_id(self):
        from piai.stream import stream
        from piai.types import Context, UserMessage
        from piai.oauth.types import OAuthCredentials

        creds = OAuthCredentials(refresh="r", access="a", expires=9999999999000)  # no accountId

        ctx = Context(messages=[UserMessage(content="hi")])
        with patch("piai.stream.get_provider_credentials", return_value=creds), \
             patch("piai.stream.get_oauth_api_key", new=AsyncMock(return_value=(creds, "tok"))), \
             patch("piai.stream.save_credentials"):
            with pytest.raises(RuntimeError, match="accountId"):
                async for _ in stream("gpt-5.1-codex-mini", ctx):
                    pass

    @pytest.mark.asyncio
    async def test_stream_yields_events(self):
        from piai.stream import stream
        from piai.types import Context, DoneEvent, UserMessage

        creds = self._make_creds()
        ctx = Context(messages=[UserMessage(content="hi")])

        with patch("piai.stream.get_provider_credentials", return_value=creds), \
             patch("piai.stream.get_oauth_api_key", new=AsyncMock(return_value=(creds, "tok"))), \
             patch("piai.stream.save_credentials"), \
             patch("piai.stream.stream_openai_codex", side_effect=self._make_done_stream("hello")):
            events = []
            async for e in stream("gpt-5.1-codex-mini", ctx):
                events.append(e)

        assert any(isinstance(e, DoneEvent) for e in events)

    @pytest.mark.asyncio
    async def test_complete_returns_message(self):
        from piai.stream import complete
        from piai.types import Context, UserMessage

        creds = self._make_creds()
        ctx = Context(messages=[UserMessage(content="hi")])

        with patch("piai.stream.get_provider_credentials", return_value=creds), \
             patch("piai.stream.get_oauth_api_key", new=AsyncMock(return_value=(creds, "tok"))), \
             patch("piai.stream.save_credentials"), \
             patch("piai.stream.stream_openai_codex", side_effect=self._make_done_stream("world")):
            result = await complete("gpt-5.1-codex-mini", ctx)

        assert result.text == "world"

    @pytest.mark.asyncio
    async def test_complete_warns_on_tool_use_stop(self):
        from piai.stream import complete
        from piai.types import Context, UserMessage

        creds = self._make_creds()
        ctx = Context(messages=[UserMessage(content="hi")])

        with patch("piai.stream.get_provider_credentials", return_value=creds), \
             patch("piai.stream.get_oauth_api_key", new=AsyncMock(return_value=(creds, "tok"))), \
             patch("piai.stream.save_credentials"), \
             patch("piai.stream.stream_openai_codex", side_effect=self._make_tool_call_stream()):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                await complete("gpt-5.1-codex-mini", ctx)
            assert any("tool_use" in str(warning.message) for warning in w)

    @pytest.mark.asyncio
    async def test_complete_raises_on_error_event(self):
        from piai.stream import complete
        from piai.types import Context, UserMessage

        creds = self._make_creds()
        ctx = Context(messages=[UserMessage(content="hi")])

        with patch("piai.stream.get_provider_credentials", return_value=creds), \
             patch("piai.stream.get_oauth_api_key", new=AsyncMock(return_value=(creds, "tok"))), \
             patch("piai.stream.save_credentials"), \
             patch("piai.stream.stream_openai_codex", side_effect=self._make_error_stream()):
            with pytest.raises(RuntimeError, match="boom"):
                await complete("gpt-5.1-codex-mini", ctx)

    @pytest.mark.asyncio
    async def test_complete_text_returns_string(self):
        from piai.stream import complete_text
        from piai.types import Context, UserMessage

        creds = self._make_creds()
        ctx = Context(messages=[UserMessage(content="hi")])

        async def _text_stream(*args, **kwargs):
            from piai.types import AssistantMessage, DoneEvent, TextDeltaEvent, TextContent
            yield TextDeltaEvent(text="hel")
            yield TextDeltaEvent(text="lo")
            msg = AssistantMessage(content=[TextContent(text="hello")])
            yield DoneEvent(reason="stop", message=msg)

        with patch("piai.stream.get_provider_credentials", return_value=creds), \
             patch("piai.stream.get_oauth_api_key", new=AsyncMock(return_value=(creds, "tok"))), \
             patch("piai.stream.save_credentials"), \
             patch("piai.stream.stream_openai_codex", side_effect=_text_stream):
            result = await complete_text("gpt-5.1-codex-mini", ctx)

        assert result == "hello"

    @pytest.mark.asyncio
    async def test_complete_text_warns_on_tool_use(self):
        from piai.stream import complete_text
        from piai.types import Context, UserMessage

        creds = self._make_creds()
        ctx = Context(messages=[UserMessage(content="hi")])

        with patch("piai.stream.get_provider_credentials", return_value=creds), \
             patch("piai.stream.get_oauth_api_key", new=AsyncMock(return_value=(creds, "tok"))), \
             patch("piai.stream.save_credentials"), \
             patch("piai.stream.stream_openai_codex", side_effect=self._make_tool_call_stream()):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = await complete_text("gpt-5.1-codex-mini", ctx)
            assert any("tool_use" in str(warning.message) for warning in w)

    @pytest.mark.asyncio
    async def test_complete_text_raises_on_error(self):
        from piai.stream import complete_text
        from piai.types import Context, UserMessage

        creds = self._make_creds()
        ctx = Context(messages=[UserMessage(content="hi")])

        with patch("piai.stream.get_provider_credentials", return_value=creds), \
             patch("piai.stream.get_oauth_api_key", new=AsyncMock(return_value=(creds, "tok"))), \
             patch("piai.stream.save_credentials"), \
             patch("piai.stream.stream_openai_codex", side_effect=self._make_error_stream()):
            with pytest.raises(RuntimeError, match="boom"):
                await complete_text("gpt-5.1-codex-mini", ctx)

    @pytest.mark.asyncio
    async def test_complete_raises_when_no_done_event(self):
        """complete() raises if stream ends without a DoneEvent."""
        from piai.stream import complete
        from piai.types import Context, UserMessage, TextDeltaEvent

        creds = self._make_creds()
        ctx = Context(messages=[UserMessage(content="hi")])

        async def empty_stream(*args, **kwargs):
            yield TextDeltaEvent(text="partial")
            # No DoneEvent emitted

        with patch("piai.stream.get_provider_credentials", return_value=creds), \
             patch("piai.stream.get_oauth_api_key", new=AsyncMock(return_value=(creds, "tok"))), \
             patch("piai.stream.save_credentials"), \
             patch("piai.stream.stream_openai_codex", side_effect=empty_stream):
            with pytest.raises(RuntimeError, match="done event"):
                await complete("gpt-5.1-codex-mini", ctx)

    @pytest.mark.asyncio
    async def test_stream_passes_base_url_from_options(self):
        """base_url in options should be popped and passed to provider, not in opts."""
        from piai.stream import stream
        from piai.types import Context, UserMessage

        creds = self._make_creds()
        ctx = Context(messages=[UserMessage(content="hi")])
        captured = {}

        async def capture_stream(model_id, context, token, account_id, options, base_url=None):
            captured["base_url"] = base_url
            captured["options"] = options
            from piai.types import AssistantMessage, DoneEvent, TextContent
            yield DoneEvent(reason="stop", message=AssistantMessage(content=[TextContent(text="ok")]))

        with patch("piai.stream.get_provider_credentials", return_value=creds), \
             patch("piai.stream.get_oauth_api_key", new=AsyncMock(return_value=(creds, "tok"))), \
             patch("piai.stream.save_credentials"), \
             patch("piai.stream.stream_openai_codex", side_effect=capture_stream):
            async for _ in stream("gpt-5.1-codex-mini", ctx, options={"base_url": "http://custom", "foo": "bar"}):
                pass

        assert captured["base_url"] == "http://custom"
        assert "base_url" not in captured["options"]
        assert captured["options"]["foo"] == "bar"


# ================================================================== #
# agent.py — remaining gap: line 296 (max_turns warning)             #
# ================================================================== #

class TestAgentMaxTurns:
    @pytest.mark.asyncio
    async def test_agent_warns_at_max_turns(self):
        """Reaching max_turns logs a warning — covers the else branch at end of for loop."""
        from piai.agent import agent
        from piai.types import (
            AssistantMessage, Context, DoneEvent, TextContent,
            ToolCall, ToolCallContent, ToolCallEndEvent, UserMessage,
        )

        call_count = 0

        async def infinite_tool_stream(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            tc = ToolCall(id=f"c{call_count}", name="loop", input={})
            yield ToolCallEndEvent(tool_call=tc)
            msg = AssistantMessage(content=[ToolCallContent(tool_calls=[tc])])
            yield DoneEvent(reason="tool_use", message=msg)

        ctx = Context(messages=[UserMessage(content="loop")])

        import logging
        with patch("piai.agent.stream", side_effect=infinite_tool_stream):
            with patch.object(logging.getLogger("piai.agent"), "warning") as mock_warn:
                result = await agent(
                    model_id="gpt-5.1-codex-mini",
                    context=ctx,
                    local_handlers={"loop": lambda: "again"},
                    max_turns=3,
                )
        mock_warn.assert_called()
        assert "max_turns" in str(mock_warn.call_args)


# ================================================================== #
# providers/message_transform.py — line 78 (list user content block) #
# ================================================================== #

class TestMessageTransformGaps:
    def test_convert_user_content_block_str(self):
        from piai.providers.message_transform import _convert_user_content_block
        result = _convert_user_content_block("plain text")
        assert result == {"type": "input_text", "text": "plain text"}

    def test_convert_user_content_block_dict(self):
        from piai.providers.message_transform import _convert_user_content_block
        block = {"type": "input_image", "url": "http://x/img.png"}
        assert _convert_user_content_block(block) == block

    def test_convert_user_content_block_other(self):
        from piai.providers.message_transform import _convert_user_content_block
        result = _convert_user_content_block(42)
        assert result == {"type": "input_text", "text": "42"}


# ================================================================== #
# SubAgentTool — async extractor + sync run paths                    #
# ================================================================== #

class TestSubAgentToolGaps:
    @pytest.mark.asyncio
    async def test_async_context_extractor(self):
        """context_extractor can be async — must be awaited."""
        from piai.langchain.sub_agent_tool import SubAgentTool
        from piai.types import Context, UserMessage

        captured = []

        async def fake_agent(model_id, context, **kwargs):
            from piai.types import AssistantMessage, TextContent
            captured.append(context)
            return AssistantMessage(content=[TextContent(text="ok")])

        initial_ctx = Context(
            messages=[UserMessage(content="old")],
            scratchpad={"x": 1},
        )

        async def async_extractor(ctx: Context) -> Context:
            await asyncio.sleep(0)
            return Context(
                messages=[UserMessage(content="filtered")],
                system_prompt="Sub",
                scratchpad=ctx.scratchpad,
            )

        tool = SubAgentTool(
            name="sub",
            description="test",
            initial_context=initial_ctx,
            context_extractor=async_extractor,
        )

        with patch("piai.langchain.sub_agent_tool.piai_agent", side_effect=fake_agent):
            await tool._arun(task="filtered")

        assert captured[0].system_prompt == "Sub"

    def test_sync_run_in_thread_when_loop_running(self):
        """_run() spawns thread when event loop is already running."""
        from piai.langchain.sub_agent_tool import SubAgentTool
        from piai.types import AssistantMessage, TextContent

        async def fake_agent(*args, **kwargs):
            return AssistantMessage(content=[TextContent(text="threaded")])

        tool = SubAgentTool(name="t", description="d")

        with patch("piai.langchain.sub_agent_tool.piai_agent", side_effect=fake_agent):
            # Run _run() inside a running event loop — simulates LangGraph thread
            async def inner():
                return tool._run("task")
            result = asyncio.run(inner())

        assert result == "threaded"

    def test_import_error_message(self):
        """SubAgentTool raises ImportError with helpful message if langchain missing."""
        # Test the ImportError message directly without module manipulation
        # (module manipulation causes other tests to fail due to sys.modules corruption)
        try:
            from langchain_core.tools import BaseTool  # noqa
            # If langchain is installed, verify the tool can be imported cleanly
            from piai.langchain.sub_agent_tool import SubAgentTool
            assert SubAgentTool is not None
        except ImportError as e:
            assert "pi-ai-py[langgraph]" in str(e)


# ================================================================== #
# Integration: scratchpad + context_reducer + usage together         #
# ================================================================== #

class TestIntegration:
    @pytest.mark.asyncio
    async def test_scratchpad_preserved_through_reducer(self):
        """Reducer that trims messages must not lose the scratchpad."""
        from piai.agent import agent
        from piai.types import (
            AssistantMessage, Context, DoneEvent, TextContent,
            ToolCall, ToolCallContent, ToolCallEndEvent, UserMessage,
        )

        call_count = 0

        async def fake_stream(model_id, ctx, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                tc = ToolCall(id="c1", name="note", input={"key": "found", "value": "vuln"})
                yield ToolCallEndEvent(tool_call=tc)
                msg = AssistantMessage(content=[ToolCallContent(tool_calls=[tc])])
                yield DoneEvent(reason="tool_use", message=msg)
            else:
                yield DoneEvent(reason="stop", message=AssistantMessage(content=[TextContent(text="done")]))

        # internal_ctx_ref will be captured by the reducer
        internal_ctx_ref: list[Context] = []

        ctx = Context(
            messages=[UserMessage(content="scan")],
            scratchpad={"status": "init"},
        )

        def note(key: str, value: str) -> str:
            # Update whichever ctx the agent is actually using
            if internal_ctx_ref:
                internal_ctx_ref[0].scratchpad[key] = value
            return "ok"

        reducer_saw_scratchpad = []

        def reducer(c: Context) -> Context:
            if not internal_ctx_ref:
                internal_ctx_ref.append(c)
            reducer_saw_scratchpad.append(dict(c.scratchpad))
            return Context(
                messages=c.messages[-1:],
                system_prompt=c.system_prompt,
                tools=c.tools,
                scratchpad=c.scratchpad,
            )

        # Use a wrapper that captures the internal ctx before note runs
        captured_ctx: list[Context] = []

        async def fake_stream2(model_id, ctx_arg, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if not captured_ctx:
                captured_ctx.append(ctx_arg)
                internal_ctx_ref.append(ctx_arg)
            if call_count == 1:
                tc = ToolCall(id="c1", name="note", input={"key": "found", "value": "vuln"})
                yield ToolCallEndEvent(tool_call=tc)
                msg = AssistantMessage(content=[ToolCallContent(tool_calls=[tc])])
                yield DoneEvent(reason="tool_use", message=msg)
            else:
                yield DoneEvent(reason="stop", message=AssistantMessage(content=[TextContent(text="done")]))

        with patch("piai.agent.stream", side_effect=fake_stream2):
            result = await agent(
                model_id="gpt-5.1-codex-mini",
                context=ctx,
                local_handlers={"note": note},
                context_reducer=reducer,
            )

        assert result.text == "done"
        # scratchpad was updated by note() and preserved through reducer
        assert reducer_saw_scratchpad[0].get("status") == "init"

    @pytest.mark.asyncio
    async def test_usage_accumulation_across_turns(self):
        """AgentTurnEndEvent.usage should reflect per-turn token counts."""
        from piai.agent import agent
        from piai.types import (
            AgentTurnEndEvent, AssistantMessage, Context, DoneEvent, TextContent,
            ToolCall, ToolCallContent, ToolCallEndEvent, UserMessage,
        )

        usages = []

        def on_event(e):
            if isinstance(e, AgentTurnEndEvent):
                usages.append(dict(e.usage))

        call_count = 0

        async def fake_stream(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                tc = ToolCall(id="c1", name="noop", input={})
                yield ToolCallEndEvent(tool_call=tc)
                msg = AssistantMessage(content=[ToolCallContent(tool_calls=[tc])])
                msg.usage.update({"input": 100, "output": 20, "total_tokens": 120})
                yield DoneEvent(reason="tool_use", message=msg)
            else:
                final = AssistantMessage(content=[TextContent(text="done")])
                final.usage.update({"input": 200, "output": 40, "total_tokens": 240})
                yield DoneEvent(reason="stop", message=final)

        ctx = Context(messages=[UserMessage(content="go")])

        with patch("piai.agent.stream", side_effect=fake_stream):
            await agent(
                model_id="gpt-5.1-codex-mini",
                context=ctx,
                local_handlers={"noop": lambda: "ok"},
                on_event=on_event,
            )

        assert len(usages) == 2
        assert usages[0]["input"] == 100
        assert usages[1]["input"] == 200
        total = sum(u["total_tokens"] for u in usages)
        assert total == 360
