"""
Convert piai Context/Message types → OpenAI Responses API request format.

The Responses API (chatgpt.com/backend-api/codex/responses) differs from
the Chat Completions API:
  - System prompt → top-level "instructions" field (not in messages)
  - User messages → {"type": "message", "role": "user", "content": [...]}
  - Assistant text → {"type": "message", "role": "assistant", "content": [{"type": "output_text", ...}]}
  - Tool calls → {"type": "function_call", "name": ..., "arguments": ..., "call_id": ...}
  - Tool results → {"type": "function_call_output", "call_id": ..., "output": ...}

Mirrors src/providers/openai-responses-shared.ts convertResponsesMessages()
and src/providers/transform-messages.ts.
"""

from __future__ import annotations

import json
from typing import Any

from ..types import (
    AssistantMessage,
    Context,
    TextContent,
    ThinkingContent,
    Tool,
    ToolCall,
    ToolCallContent,
    ToolResultMessage,
    UserMessage,
)


# ------------------------------------------------------------------ #
# Message conversion                                                  #
# ------------------------------------------------------------------ #


def convert_messages(context: Context) -> list[dict[str, Any]]:
    """
    Convert Context messages to the Responses API input array.

    System prompt is NOT included here — it goes in the top-level
    "instructions" field of the request body.
    """
    result: list[dict[str, Any]] = []

    for msg in context.messages:
        if isinstance(msg, UserMessage):
            result.append(_convert_user_message(msg))
        elif isinstance(msg, AssistantMessage):
            result.extend(_convert_assistant_message(msg))
        elif isinstance(msg, ToolResultMessage):
            result.append(_convert_tool_result(msg))

    return result


def _convert_user_message(msg: UserMessage) -> dict[str, Any]:
    if isinstance(msg.content, str):
        content = [{"type": "input_text", "text": msg.content}]
    else:
        # List of content blocks (e.g. image + text)
        content = [_convert_user_content_block(b) for b in msg.content]

    return {
        "type": "message",
        "role": "user",
        "content": content,
    }


def _convert_user_content_block(block: Any) -> dict[str, Any]:
    if isinstance(block, str):
        return {"type": "input_text", "text": block}
    if isinstance(block, dict):
        return block
    return {"type": "input_text", "text": str(block)}


def _convert_assistant_message(msg: AssistantMessage) -> list[dict[str, Any]]:
    """
    One AssistantMessage can produce multiple Responses API items
    (text block + tool calls are separate items in the array).
    """
    items: list[dict[str, Any]] = []

    for block in msg.content:
        if isinstance(block, TextContent) and block.text:
            items.append({
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": block.text}],
            })
        elif isinstance(block, ThinkingContent) and block.thinking:
            # Thinking blocks become plain text wrapped in <thinking> tags
            # for cross-provider compatibility
            items.append({
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": f"<thinking>{block.thinking}</thinking>"}],
            })
        elif isinstance(block, ToolCallContent):
            for tc in block.tool_calls:
                items.append({
                    "type": "function_call",
                    "name": tc.name,
                    "arguments": json.dumps(tc.input),
                    "call_id": tc.id,
                })

    return items


def _convert_tool_result(msg: ToolResultMessage) -> dict[str, Any]:
    return {
        "type": "function_call_output",
        "call_id": msg.tool_call_id,
        "output": msg.content,
    }


# ------------------------------------------------------------------ #
# Tool conversion                                                     #
# ------------------------------------------------------------------ #


def convert_tools(tools: list[Tool]) -> list[dict[str, Any]]:
    """Convert piai Tool list → OpenAI Responses API tools array."""
    return [_convert_tool(t) for t in tools]


def _convert_tool(tool: Tool) -> dict[str, Any]:
    return {
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.parameters,
        "strict": None,  # JS passes strict: null for codex provider
    }


# ------------------------------------------------------------------ #
# Full request body                                                   #
# ------------------------------------------------------------------ #


def _clamp_reasoning_effort(model_id: str, effort: str) -> str:
    """
    Clamp reasoning effort to values the model actually supports.

    Mirrors clampReasoningEffort() from openai-codex-responses.ts (lines 328-335).
    """
    # Strip provider prefix e.g. "openai-codex/gpt-5.1" → "gpt-5.1"
    mid = model_id.split("/")[-1] if "/" in model_id else model_id

    if (mid.startswith("gpt-5.2") or mid.startswith("gpt-5.3") or mid.startswith("gpt-5.4")) and effort == "minimal":
        return "low"
    if mid == "gpt-5.1" and effort == "xhigh":
        return "high"
    if mid == "gpt-5.1-codex-mini":
        return "high" if effort in ("high", "xhigh") else "medium"
    return effort


def build_request_body(
    model_id: str,
    context: Context,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build the full JSON body for POST /codex/responses.

    Mirrors buildRequestBody() from openai-codex-responses.ts.
    """
    opts = options or {}
    messages = convert_messages(context)

    body: dict[str, Any] = {
        "model": model_id,
        "store": False,
        "stream": True,
        "instructions": context.system_prompt or "You are a helpful assistant.",
        "input": messages,
        "text": {"verbosity": opts.get("text_verbosity", "medium")},
        "include": ["reasoning.encrypted_content"],
    }

    if context.tools:
        body["tools"] = convert_tools(context.tools)
        # Allow callers to override tool_choice (e.g. "required", {"type": "function", "name": "..."})
        body["tool_choice"] = opts.get("tool_choice", "auto")
        body["parallel_tool_calls"] = True

    if opts.get("session_id"):
        body["prompt_cache_key"] = opts["session_id"]

    reasoning_effort = opts.get("reasoning_effort")
    if reasoning_effort:
        body["reasoning"] = {
            "effort": _clamp_reasoning_effort(model_id, reasoning_effort),
            "summary": opts.get("reasoning_summary", "auto"),
        }

    return body
