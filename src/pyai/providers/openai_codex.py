"""
OpenAI Codex (ChatGPT Plus) streaming provider.

Hits https://chatgpt.com/backend-api/codex/responses with SSE streaming.
This is NOT the public OpenAI API — it's the internal ChatGPT backend
that your Plus/Pro subscription grants access to.

Mirrors src/providers/openai-codex-responses.ts.
"""

from __future__ import annotations

import asyncio
import json
import platform
import time
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from ..types import (
    AssistantMessage,
    Context,
    DoneEvent,
    ErrorEvent,
    StreamEvent,
    TextContent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ThinkingContent,
    ThinkingDeltaEvent,
    ToolCall,
    ToolCallContent,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
)
from .message_transform import build_request_body

# ------------------------------------------------------------------ #
# Constants — must match JS headers exactly                           #
# ------------------------------------------------------------------ #

DEFAULT_CODEX_BASE_URL = "https://chatgpt.com/backend-api"
CODEX_PATH = "/codex/responses"
JWT_CLAIM_PATH = "https://api.openai.com/auth"

MAX_RETRIES = 3
BASE_DELAY_S = 1.0

_RETRYABLE_STATUSES = {429, 500, 502, 503, 504}


# ------------------------------------------------------------------ #
# Header building                                                     #
# ------------------------------------------------------------------ #


def _user_agent() -> str:
    """
    Mirrors JS: `pi (${os.platform()} ${os.release()}; ${os.arch()})`
    e.g. "pi (darwin 24.6.0; arm64)"
    """
    system = platform.system().lower()
    release = platform.release()
    machine = platform.machine().lower()
    return f"pi ({system} {release}; {machine})"


def build_headers(
    token: str,
    account_id: str,
    session_id: str | None = None,
    extra_headers: dict[str, str] | None = None,
) -> dict[str, str]:
    """
    Build request headers for the Codex backend.

    Must match JS buildHeaders() exactly — the backend validates several
    of these (account-id, originator, OpenAI-Beta).
    """
    headers = {
        "Authorization": f"Bearer {token}",
        "chatgpt-account-id": account_id,
        "OpenAI-Beta": "responses=experimental",
        "originator": "pi",
        "User-Agent": _user_agent(),
        "accept": "text/event-stream",
        "content-type": "application/json",
    }

    if session_id:
        headers["session_id"] = session_id

    if extra_headers:
        headers.update(extra_headers)

    return headers


# ------------------------------------------------------------------ #
# URL resolution                                                      #
# ------------------------------------------------------------------ #


def _resolve_codex_url(base_url: str | None = None) -> str:
    """
    Mirrors resolveCodexUrl() from openai-codex-responses.ts.
    Handles various base URL forms and appends /codex/responses.
    """
    raw = (base_url or DEFAULT_CODEX_BASE_URL).rstrip("/")
    if raw.endswith("/codex/responses"):
        return raw
    if raw.endswith("/codex"):
        return f"{raw}/responses"
    return f"{raw}{CODEX_PATH}"


# ------------------------------------------------------------------ #
# SSE parsing                                                         #
# ------------------------------------------------------------------ #


async def _parse_sse(response: httpx.Response) -> AsyncGenerator[dict[str, Any], None]:
    """
    Parse Server-Sent Events from an httpx streaming response.

    Event boundary: double newline (\n\n).
    Data lines start with "data:".
    "[DONE]" sentinel is skipped.

    Mirrors parseSSE() from openai-codex-responses.ts.
    """
    buffer = ""
    async for chunk in response.aiter_text():
        buffer += chunk
        while "\n\n" in buffer:
            event_str, buffer = buffer.split("\n\n", 1)
            data_lines = [
                line[5:].strip()
                for line in event_str.split("\n")
                if line.startswith("data:")
            ]
            if not data_lines:
                continue
            data = "\n".join(data_lines).strip()
            if not data or data == "[DONE]":
                continue
            try:
                yield json.loads(data)
            except json.JSONDecodeError:
                pass


# ------------------------------------------------------------------ #
# Event → StreamEvent conversion                                      #
# ------------------------------------------------------------------ #


class _StreamProcessor:
    """
    Convert raw Codex SSE events into typed StreamEvents.

    Tracks in-progress tool calls and text accumulation to emit
    complete events at the right boundaries.

    Mirrors processResponsesStream() from openai-responses-shared.ts.
    """

    def __init__(self, output: AssistantMessage):
        self._output = output
        self._current_text = ""
        self._in_text = False
        self._current_thinking = ""
        self._in_thinking = False
        # tool call id → {name, args_buffer}
        self._tool_calls: dict[str, dict[str, Any]] = {}
        self._tool_call_order: list[str] = []  # preserve insertion order

    async def process(
        self, events: AsyncGenerator[dict[str, Any], None]
    ) -> AsyncGenerator[StreamEvent, None]:
        async for event in events:
            event_type = event.get("type", "")

            if event_type == "error":
                code = event.get("code", "")
                message = event.get("message", "")
                raise RuntimeError(f"Codex error: {message or code or json.dumps(event)}")

            if event_type == "response.failed":
                msg = (event.get("response") or {}).get("error", {}).get("message")
                raise RuntimeError(msg or "Codex response failed")

            # --- Text events ---
            if event_type == "response.output_text.delta":
                delta = event.get("delta", "")
                if delta:
                    if not self._in_text:
                        self._in_text = True
                        self._current_text = ""
                        yield TextStartEvent()
                    self._current_text += delta
                    yield TextDeltaEvent(text=delta)

            elif event_type == "response.output_text.done":
                if self._in_text:
                    self._in_text = False
                    text_block = TextContent(text=self._current_text)
                    self._output.content.append(text_block)
                    yield TextEndEvent(text=self._current_text)
                    self._current_text = ""

            # --- Reasoning/thinking events ---
            elif event_type == "response.reasoning_summary_text.delta":
                delta = event.get("delta", "")
                if delta:
                    self._current_thinking += delta
                    yield ThinkingDeltaEvent(thinking=delta)

            elif event_type == "response.reasoning_summary_text.done":
                if self._current_thinking:
                    thinking_block = ThinkingContent(thinking=self._current_thinking)
                    self._output.content.append(thinking_block)
                    self._current_thinking = ""

            # --- Tool call events ---
            elif event_type == "response.output_item.added":
                item = event.get("item") or {}
                if item.get("type") == "function_call":
                    call_id = item.get("call_id", item.get("id", ""))
                    name = item.get("name", "")
                    if call_id and call_id not in self._tool_calls:
                        self._tool_calls[call_id] = {"name": name, "args": ""}
                        self._tool_call_order.append(call_id)
                        tc = ToolCall(id=call_id, name=name)
                        yield ToolCallStartEvent(tool_call=tc)

            elif event_type == "response.function_call_arguments.delta":
                call_id = event.get("call_id", event.get("item_id", ""))
                delta = event.get("delta", "")
                if call_id in self._tool_calls and delta:
                    self._tool_calls[call_id]["args"] += delta
                    yield ToolCallDeltaEvent(id=call_id, json_delta=delta)

            elif event_type == "response.function_call_arguments.done":
                call_id = event.get("call_id", event.get("item_id", ""))
                if call_id in self._tool_calls:
                    entry = self._tool_calls[call_id]
                    try:
                        input_dict = json.loads(entry["args"]) if entry["args"] else {}
                    except json.JSONDecodeError:
                        input_dict = {}
                    tc = ToolCall(id=call_id, name=entry["name"], input=input_dict)
                    yield ToolCallEndEvent(tool_call=tc)

            # --- Completion event ---
            elif event_type in ("response.completed", "response.done"):
                response_data = event.get("response") or {}
                self._consume_final_response(response_data)

            # --- Output item done (finalize tool calls) ---
            elif event_type == "response.output_item.done":
                item = event.get("item") or {}
                if item.get("type") == "function_call":
                    call_id = item.get("call_id", item.get("id", ""))
                    if call_id in self._tool_calls:
                        entry = self._tool_calls[call_id]
                        try:
                            args_str = item.get("arguments", entry["args"])
                            input_dict = json.loads(args_str) if args_str else {}
                        except json.JSONDecodeError:
                            input_dict = {}
                        entry["name"] = item.get("name", entry["name"])
                        entry["input"] = input_dict

        # Finalize any pending tool calls into content
        if self._tool_call_order:
            tool_calls = []
            for call_id in self._tool_call_order:
                entry = self._tool_calls[call_id]
                tool_calls.append(ToolCall(
                    id=call_id,
                    name=entry["name"],
                    input=entry.get("input", {}),
                ))
            self._output.content.append(ToolCallContent(tool_calls=tool_calls))
            self._output.stop_reason = "tool_use"

    def _consume_final_response(self, response_data: dict[str, Any]) -> None:
        """Extract usage stats and stop reason from the completed response."""
        usage = response_data.get("usage") or {}
        self._output.usage["input"] = usage.get("input_tokens", 0)
        self._output.usage["output"] = usage.get("output_tokens", 0)
        self._output.usage["cache_read"] = usage.get("input_tokens_details", {}).get("cached_tokens", 0)

        status = response_data.get("status", "")
        if status == "incomplete":
            self._output.stop_reason = "length"
        elif status in ("failed", "cancelled"):
            self._output.stop_reason = "error"
        elif self._output.stop_reason != "tool_use":
            self._output.stop_reason = "stop"


# ------------------------------------------------------------------ #
# Error detection                                                     #
# ------------------------------------------------------------------ #


def _is_retryable(status: int, body: str) -> bool:
    if status in _RETRYABLE_STATUSES:
        return True
    import re
    return bool(re.search(
        r"rate.?limit|overloaded|service.?unavailable|upstream.?connect|connection.?refused",
        body, re.IGNORECASE
    ))


def _friendly_error(status: int, body: str) -> str:
    try:
        data = json.loads(body)
        err = data.get("error") or {}
        code = err.get("code", "") or err.get("type", "")
        import re
        if re.search(r"usage_limit_reached|usage_not_included|rate_limit_exceeded", code, re.IGNORECASE) or status == 429:
            plan = f" ({err['plan_type'].lower()} plan)" if err.get("plan_type") else ""
            resets_at = err.get("resets_at")
            when = ""
            if resets_at:
                mins = max(0, round((resets_at * 1000 - time.time() * 1000) / 60000))
                when = f" Try again in ~{mins} min."
            return f"You have hit your ChatGPT usage limit{plan}.{when}".strip()
        if err.get("message"):
            return err["message"]
    except (json.JSONDecodeError, KeyError):
        pass
    return body or f"HTTP {status}"


# ------------------------------------------------------------------ #
# Main stream function                                                #
# ------------------------------------------------------------------ #


async def stream_openai_codex(
    model_id: str,
    context: Context,
    token: str,
    account_id: str,
    options: dict[str, Any] | None = None,
    base_url: str | None = None,
) -> AsyncGenerator[StreamEvent, None]:
    """
    Stream completions from ChatGPT backend.

    Yields typed StreamEvent objects (TextDeltaEvent, ToolCallStartEvent, DoneEvent, etc.).
    Handles retries for rate limits and transient errors.

    Mirrors streamOpenAICodexResponses() from openai-codex-responses.ts.
    """
    opts = options or {}
    body = build_request_body(model_id, context, opts)
    headers = build_headers(
        token=token,
        account_id=account_id,
        session_id=opts.get("session_id"),
        extra_headers=opts.get("headers"),
    )
    url = _resolve_codex_url(base_url)

    output = AssistantMessage(
        model=model_id,
        provider="openai-codex",
        api="openai-codex-responses",
        timestamp=int(time.time() * 1000),
    )

    last_error: Exception | None = None

    for attempt in range(MAX_RETRIES + 1):
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
                async with client.stream(
                    "POST", url, json=body, headers=headers
                ) as response:
                    if response.is_success:
                        processor = _StreamProcessor(output)
                        async for event in processor.process(_parse_sse(response)):
                            yield event

                        yield DoneEvent(
                            reason=output.stop_reason,
                            message=output,
                        )
                        return

                    # Not OK — read body for error detail
                    await response.aread()
                    error_body = response.text
                    if attempt < MAX_RETRIES and _is_retryable(response.status_code, error_body):
                        await asyncio.sleep(BASE_DELAY_S * (2 ** attempt))
                        continue

                    raise RuntimeError(_friendly_error(response.status_code, error_body))

        except (httpx.NetworkError, httpx.TimeoutException) as e:
            last_error = e
            if attempt < MAX_RETRIES:
                await asyncio.sleep(BASE_DELAY_S * (2 ** attempt))
                continue
            break
        except RuntimeError:
            raise
        except Exception as e:
            last_error = e
            break

    error_msg = str(last_error) if last_error else "Failed after retries"
    output.stop_reason = "error"
    output.error_message = error_msg
    yield ErrorEvent(reason="error", error=output)
