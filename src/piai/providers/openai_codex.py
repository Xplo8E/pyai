"""
OpenAI Codex (ChatGPT Plus) streaming provider.

Hits https://chatgpt.com/backend-api/codex/responses with SSE streaming.
This is NOT the public OpenAI API — it's the internal ChatGPT backend
that your Plus/Pro subscription grants access to.

Mirrors src/providers/openai-codex-responses.ts +
        src/providers/openai-responses-shared.ts processResponsesStream().
"""

from __future__ import annotations

import asyncio
import json
import logging
import platform
import re
import time
from collections.abc import AsyncGenerator
from typing import Any

logger = logging.getLogger(__name__)

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
    ThinkingEndEvent,
    ThinkingStartEvent,
    ToolCall,
    ToolCallContent,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
)
from .message_transform import build_request_body

# ------------------------------------------------------------------ #
# Constants                                                           #
# ------------------------------------------------------------------ #

DEFAULT_CODEX_BASE_URL = "https://chatgpt.com/backend-api"
CODEX_PATH = "/codex/responses"

MAX_RETRIES = 3
BASE_DELAY_S = 1.0
MAX_TOOL_CALL_ID_LEN = 64


def _make_tc_id(call_id: str, item_id: str) -> str:
    """Build tool call ID, truncated to API limit of 64 chars."""
    return f"{call_id}|{item_id}"[:MAX_TOOL_CALL_ID_LEN]

_RETRYABLE_STATUSES = {429, 500, 502, 503, 504}
_RETRYABLE_PATTERN = re.compile(
    r"rate.?limit|overloaded|service.?unavailable|upstream.?connect|connection.?refused",
    re.IGNORECASE,
)


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

    Event boundary: double newline (\\n\\n or \\r\\n\\r\\n normalized to \\n\\n).
    Data lines start with "data:".
    "[DONE]" sentinel is skipped.

    Mirrors parseSSE() from openai-codex-responses.ts.
    """
    buffer = ""
    async for chunk in response.aiter_text():
        # Normalize CRLF to LF for consistent parsing
        buffer += chunk.replace("\r\n", "\n").replace("\r", "\n")
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
# Stream processor                                                    #
# ------------------------------------------------------------------ #


class _StreamProcessor:
    """
    Convert raw Codex SSE events into typed StreamEvents.

    Faithful port of processResponsesStream() from openai-responses-shared.ts.

    Uses the same currentItem / currentBlock state machine as JS:
    - current_item tracks the active output item (reasoning | message | function_call)
    - current_block tracks the in-progress content block being built

    Key invariants from JS:
    - reasoning_summary_text.delta only updates if there is an active summary part
    - reasoning_summary_part.done appends "\\n\\n" separator to the thinking text
    - output_text.delta only updates if current_item.content is non-empty and
      last part is output_text (guards against refusal parts coming first)
    - refusal.delta surfaces to caller as TextDeltaEvent (same as text)
    - output_item.done finalises each block and clears current_block
    - usage: cached tokens are subtracted from input_tokens (JS line 437-438)
    """

    def __init__(self, output: AssistantMessage):
        self._output = output
        # current_item mirrors JS currentItem — tracks active output item metadata
        self._current_item: dict[str, Any] | None = None
        # current_block mirrors JS currentBlock — tracks in-progress content block
        # For reasoning: {"type": "thinking", "thinking": "", "summary_parts": []}
        # For text:      {"type": "text", "text": "", "content_parts": []}
        # For tool call: {"type": "function_call", "call_id": str, "item_id": str,
        #                 "name": str, "args": str}
        self._current_block: dict[str, Any] | None = None

    async def process(
        self, events: AsyncGenerator[dict[str, Any], None]
    ) -> AsyncGenerator[StreamEvent, None]:

        async for event in events:
            t = event.get("type", "")

            # -------------------------------------------------------- #
            # Error events — raise immediately                         #
            # -------------------------------------------------------- #

            if t == "error":
                code = event.get("code", "")
                message = event.get("message", "")
                raise RuntimeError(f"Codex error: {message or code or json.dumps(event)}")

            if t == "response.failed":
                # Fix 4: include incomplete_details.reason as fallback
                # Mirrors JS lines 449-453 of openai-responses-shared.ts
                response_data = event.get("response") or {}
                err = response_data.get("error") or {}
                details = response_data.get("incomplete_details") or {}
                if err:
                    msg = f"{err.get('code', 'unknown')}: {err.get('message', 'no message')}"
                elif details.get("reason"):
                    msg = f"incomplete: {details['reason']}"
                else:
                    msg = "Unknown error (no error details in response)"
                raise RuntimeError(msg)

            # -------------------------------------------------------- #
            # output_item.added — initialize new block                 #
            # Mirrors JS lines 286-309                                 #
            # -------------------------------------------------------- #

            elif t == "response.output_item.added":
                item = event.get("item") or {}
                item_type = item.get("type", "")

                if item_type == "reasoning":
                    self._current_item = {"type": "reasoning", "summary": []}
                    self._current_block = {"type": "thinking", "thinking": "", "summary_parts": []}
                    # Don't append to output.content yet — done on output_item.done
                    yield ThinkingStartEvent()

                elif item_type == "message":
                    self._current_item = {"type": "message", "id": item.get("id", ""), "content": []}
                    self._current_block = {"type": "text", "text": "", "content_parts": []}
                    yield TextStartEvent()

                elif item_type == "function_call":
                    call_id = item.get("call_id", "")
                    item_id = item.get("id", "")
                    name = item.get("name", "")
                    self._current_item = {
                        "type": "function_call",
                        "call_id": call_id,
                        "id": item_id,
                        "name": name,
                        "arguments": item.get("arguments", ""),
                    }
                    self._current_block = {
                        "type": "function_call",
                        "call_id": call_id,
                        "item_id": item_id,
                        "name": name,
                        "args": item.get("arguments", ""),
                    }
                    tc = ToolCall(id=_make_tc_id(call_id, item_id), name=name)
                    yield ToolCallStartEvent(tool_call=tc)

            # -------------------------------------------------------- #
            # Reasoning events                                          #
            # Mirrors JS lines 310-344                                 #
            # -------------------------------------------------------- #

            elif t == "response.reasoning_summary_part.added":
                # Initialize a new summary part on the current reasoning item
                if self._current_item and self._current_item.get("type") == "reasoning":
                    part = event.get("part") or {}
                    self._current_item["summary"].append({"type": part.get("type", "summary_text"), "text": ""})

            elif t == "response.reasoning_summary_text.delta":
                # Only update if there is an active last summary part (mirrors JS guard)
                if (
                    self._current_item
                    and self._current_item.get("type") == "reasoning"
                    and self._current_block
                    and self._current_block.get("type") == "thinking"
                ):
                    summary = self._current_item["summary"]
                    if summary:
                        delta = event.get("delta", "")
                        summary[-1]["text"] += delta
                        self._current_block["thinking"] += delta
                        yield ThinkingDeltaEvent(thinking=delta)

            elif t == "response.reasoning_summary_part.done":
                # Append "\n\n" separator between multi-part reasoning blocks
                # Mirrors JS lines 330-344
                if (
                    self._current_item
                    and self._current_item.get("type") == "reasoning"
                    and self._current_block
                    and self._current_block.get("type") == "thinking"
                ):
                    summary = self._current_item["summary"]
                    if summary:
                        separator = "\n\n"
                        summary[-1]["text"] += separator
                        self._current_block["thinking"] += separator
                        yield ThinkingDeltaEvent(thinking=separator)

            # -------------------------------------------------------- #
            # Text/refusal events                                       #
            # Mirrors JS lines 345-385                                 #
            # -------------------------------------------------------- #

            elif t == "response.content_part.added":
                # Track content parts on the current message item
                # Only accept output_text and refusal — filter ReasoningText
                if self._current_item and self._current_item.get("type") == "message":
                    part = event.get("part") or {}
                    part_type = part.get("type", "")
                    if part_type in ("output_text", "refusal"):
                        self._current_item["content"].append({"type": part_type, "text": ""})

            elif t == "response.output_text.delta":
                # Guard: only emit if content is non-empty and last part is output_text
                # Mirrors JS lines 353-368
                if (
                    self._current_item
                    and self._current_item.get("type") == "message"
                    and self._current_block
                    and self._current_block.get("type") == "text"
                    and self._current_item["content"]
                    and self._current_item["content"][-1]["type"] == "output_text"
                ):
                    delta = event.get("delta", "")
                    self._current_item["content"][-1]["text"] += delta
                    self._current_block["text"] += delta
                    yield TextDeltaEvent(text=delta)

            elif t == "response.refusal.delta":
                # Refusals surface to caller as text (same behaviour as JS lines 370-385)
                if (
                    self._current_item
                    and self._current_item.get("type") == "message"
                    and self._current_block
                    and self._current_block.get("type") == "text"
                    and self._current_item["content"]
                    and self._current_item["content"][-1]["type"] == "refusal"
                ):
                    delta = event.get("delta", "")
                    self._current_item["content"][-1]["text"] += delta
                    self._current_block["text"] += delta
                    yield TextDeltaEvent(text=delta)

            # -------------------------------------------------------- #
            # Tool call argument events                                 #
            # Mirrors JS lines 386-401                                 #
            # -------------------------------------------------------- #

            elif t == "response.function_call_arguments.delta":
                if (
                    self._current_item
                    and self._current_item.get("type") == "function_call"
                    and self._current_block
                    and self._current_block.get("type") == "function_call"
                ):
                    delta = event.get("delta", "")
                    self._current_block["args"] += delta
                    self._current_item["arguments"] += delta
                    call_id = self._current_block["call_id"]
                    item_id = self._current_block["item_id"]
                    yield ToolCallDeltaEvent(id=_make_tc_id(call_id, item_id), json_delta=delta)

            elif t == "response.function_call_arguments.done":
                if (
                    self._current_item
                    and self._current_item.get("type") == "function_call"
                    and self._current_block
                    and self._current_block.get("type") == "function_call"
                ):
                    # Use the done event's arguments as the canonical final value
                    self._current_block["args"] = event.get("arguments", self._current_block["args"])
                    self._current_item["arguments"] = self._current_block["args"]

            # -------------------------------------------------------- #
            # output_item.done — finalize block                        #
            # Mirrors JS lines 402-433                                 #
            # -------------------------------------------------------- #

            elif t == "response.output_item.done":
                item = event.get("item") or {}
                item_type = item.get("type", "")

                if item_type == "reasoning" and self._current_block and self._current_block.get("type") == "thinking":
                    # Reconstruct thinking from summary parts (authoritative on done)
                    summary_parts = self._current_item.get("summary", []) if self._current_item else []
                    thinking_text = "\n\n".join(p["text"] for p in summary_parts)
                    thinking_block = ThinkingContent(thinking=thinking_text)
                    self._output.content.append(thinking_block)
                    self._current_block = None
                    yield ThinkingEndEvent(thinking=thinking_text)

                elif item_type == "message" and self._current_block and self._current_block.get("type") == "text":
                    # Join all content parts into final text
                    content_parts = self._current_item.get("content", []) if self._current_item else []
                    final_text = "".join(p["text"] for p in content_parts)
                    text_block = TextContent(text=final_text)
                    self._output.content.append(text_block)
                    yield TextEndEvent(text=final_text)
                    self._current_block = None

                elif item_type == "function_call":
                    # Use final arguments from the done item (authoritative)
                    args_str = item.get("arguments", "")
                    if not args_str and self._current_block and self._current_block.get("type") == "function_call":
                        args_str = self._current_block["args"]
                    try:
                        input_dict = json.loads(args_str) if args_str else {}
                    except json.JSONDecodeError as _exc:
                        logger.warning(
                            "Tool %r: failed to parse arguments JSON %r: %s. Using {}.",
                            name or (self._current_block or {}).get("name", "?"),
                            args_str,
                            _exc,
                        )
                        input_dict = {}

                    call_id = item.get("call_id", "")
                    item_id = item.get("id", "")
                    name = item.get("name", "")
                    if not call_id and self._current_block and self._current_block.get("type") == "function_call":
                        call_id = self._current_block["call_id"]
                        item_id = self._current_block["item_id"]
                        name = self._current_block["name"]

                    tc = ToolCall(id=_make_tc_id(call_id, item_id), name=name, input=input_dict)
                    self._output.content.append(ToolCallContent(tool_calls=[tc]))
                    yield ToolCallEndEvent(tool_call=tc)
                    self._current_block = None

            # -------------------------------------------------------- #
            # Completion event — extract usage + stop reason           #
            # Mirrors JS lines 434-447 of openai-responses-shared.ts  #
            # -------------------------------------------------------- #

            elif t == "response.completed":
                response_data = event.get("response") or {}
                self._consume_final_response(response_data)

    def _consume_final_response(self, response_data: dict[str, Any]) -> None:
        """
        Extract usage and stop reason from the completed response.

        Critical: JS subtracts cached_tokens from input_tokens because the API
        includes cached tokens in input_tokens total (lines 437-440 of shared.ts).
        """
        usage = response_data.get("usage") or {}
        cached = (usage.get("input_tokens_details") or {}).get("cached_tokens", 0)
        input_tokens = usage.get("input_tokens", 0)

        self._output.usage["input"] = input_tokens - cached  # non-cached input only
        self._output.usage["output"] = usage.get("output_tokens", 0)
        self._output.usage["cache_read"] = cached
        self._output.usage["total_tokens"] = usage.get("total_tokens", 0)

        status = response_data.get("status", "")
        has_tool_calls = any(isinstance(b, ToolCallContent) for b in self._output.content)

        if status == "incomplete":
            self._output.stop_reason = "length"
        elif status in ("failed", "cancelled"):
            self._output.stop_reason = "error"
        elif has_tool_calls:
            self._output.stop_reason = "tool_use"
        else:
            self._output.stop_reason = "stop"


# ------------------------------------------------------------------ #
# Error helpers                                                       #
# ------------------------------------------------------------------ #


def _is_retryable(status: int, body: str) -> bool:
    if status in _RETRYABLE_STATUSES:
        return True
    return bool(_RETRYABLE_PATTERN.search(body))


def _friendly_error(status: int, body: str) -> str:
    try:
        data = json.loads(body)
        err = data.get("error") or {}
        code = err.get("code", "") or err.get("type", "")
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


# Usage fetching has moved to piai.usage.openai_codex — see that module.


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

    Yields typed StreamEvent objects.
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
                async with client.stream("POST", url, json=body, headers=headers) as response:
                    if response.is_success:
                        processor = _StreamProcessor(output)
                        async for event in processor.process(_parse_sse(response)):
                            yield event
                        yield DoneEvent(reason=output.stop_reason, message=output)
                        return

                    await response.aread()
                    error_body = response.text
                    if attempt < MAX_RETRIES and _is_retryable(response.status_code, error_body):
                        await asyncio.sleep(BASE_DELAY_S * (2 ** attempt))
                        continue

                    raise RuntimeError(_friendly_error(response.status_code, error_body))

        except (httpx.NetworkError, httpx.TimeoutException) as e:
            last_error = e
            # Fix 2: don't retry if the error message is about usage limits
            # Mirrors JS line 227: !lastError.message.includes("usage limit")
            if attempt < MAX_RETRIES and "usage limit" not in str(e).lower():
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
