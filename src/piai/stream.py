"""
Public stream / complete entry points.

Mirrors src/stream.ts — ties together auth, credential refresh, and inference.
"""

from __future__ import annotations

import warnings
from collections.abc import AsyncGenerator
from typing import Any

from .oauth import get_oauth_api_key, get_provider_credentials, save_credentials
from .providers.openai_codex import stream_openai_codex
from .types import AssistantMessage, Context, DoneEvent, ErrorEvent, StreamEvent, TextDeltaEvent

OPENAI_CODEX_PROVIDER = "openai-codex"


async def stream(
    model_id: str,
    context: Context,
    options: dict[str, Any] | None = None,
    provider_id: str = OPENAI_CODEX_PROVIDER,
) -> AsyncGenerator[StreamEvent, None]:
    """
    Stream model completions as typed events.

    Automatically loads credentials from auth.json, refreshes if expired,
    and persists updated credentials back.

    Args:
        model_id:    e.g. "gpt-5.1-codex-mini", "gpt-5.1", "gpt-5.2", "gpt-5.3-codex"
        context:     Conversation context (messages + optional system_prompt + tools)
        options:     Provider-specific options (session_id, reasoning_effort, etc.)
        provider_id: Defaults to "openai-codex" (ChatGPT Plus)

    Yields:
        StreamEvent instances — TextDeltaEvent, ToolCallStartEvent, DoneEvent, etc.

    Raises:
        RuntimeError: If not logged in or token refresh fails.
    """
    creds = get_provider_credentials(provider_id)
    if not creds:
        raise RuntimeError(
            f"Not logged in for provider '{provider_id}'. "
            f"Run: piai login"
        )

    # Auto-refresh if expired (5-minute buffer)
    creds, api_key = await get_oauth_api_key(provider_id, creds)

    # Persist updated credentials (refresh may have rotated tokens)
    save_credentials(provider_id, creds)

    account_id = creds.get_extra("accountId") or ""
    if not account_id:
        raise RuntimeError("Missing accountId in credentials. Please login again: piai login")

    opts = dict(options or {})
    base_url = opts.pop("base_url", None)

    async for event in stream_openai_codex(
        model_id=model_id,
        context=context,
        token=api_key,
        account_id=account_id,
        options=opts,
        base_url=base_url,
    ):
        yield event


async def complete(
    model_id: str,
    context: Context,
    options: dict[str, Any] | None = None,
    provider_id: str = OPENAI_CODEX_PROVIDER,
) -> AssistantMessage:
    """
    One-shot completion — collects full streaming response and returns AssistantMessage.

    Mirrors complete() from stream.ts.
    """
    final: AssistantMessage | None = None

    async for event in stream(model_id, context, options, provider_id):
        if isinstance(event, DoneEvent):
            final = event.message
        elif isinstance(event, ErrorEvent):
            raise RuntimeError(event.error.error_message or "Stream error")

    if final is None:
        raise RuntimeError("Stream ended without a done event")

    if final.stop_reason == "tool_use":
        warnings.warn(
            "complete() returned a message with tool calls (stop_reason='tool_use'). "
            "The model wants to call tools but complete() does not execute them. "
            "Use agent() to run an agentic loop that executes tools, or inspect "
            "AssistantMessage.content for ToolCallContent blocks to handle them manually.",
            stacklevel=2,
        )

    return final


async def complete_text(
    model_id: str,
    context: Context,
    options: dict[str, Any] | None = None,
    provider_id: str = OPENAI_CODEX_PROVIDER,
) -> str:
    """
    Convenience wrapper — returns the full text response as a string.
    """
    text = ""
    has_tool_calls = False
    async for event in stream(model_id, context, options, provider_id):
        if isinstance(event, TextDeltaEvent):
            text += event.text
        elif isinstance(event, ErrorEvent):
            raise RuntimeError(event.error.error_message or "Stream error")
        elif isinstance(event, DoneEvent):
            has_tool_calls = event.message.stop_reason == "tool_use"

    if has_tool_calls:
        warnings.warn(
            "complete_text() returned empty string because the model responded with tool "
            "calls (stop_reason='tool_use'). Use agent() to execute tools, or complete() "
            "to get the full AssistantMessage including ToolCallContent blocks.",
            stacklevel=2,
        )

    return text
