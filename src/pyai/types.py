"""
Core message and context types.

Mirrors src/types.ts — simplified to what the openai-codex provider needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


# ------------------------------------------------------------------ #
# Message content blocks                                              #
# ------------------------------------------------------------------ #


@dataclass
class TextContent:
    type: Literal["text"] = "text"
    text: str = ""


@dataclass
class ToolCall:
    """A single tool call emitted by the model."""
    id: str
    name: str
    input: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCallContent:
    type: Literal["tool_use"] = "tool_use"
    tool_calls: list[ToolCall] = field(default_factory=list)


@dataclass
class ThinkingContent:
    """Reasoning/thinking block (for models that support extended reasoning)."""
    type: Literal["thinking"] = "thinking"
    thinking: str = ""


# ------------------------------------------------------------------ #
# Messages                                                            #
# ------------------------------------------------------------------ #


@dataclass
class ToolResultMessage:
    """Tool result sent back to the model after a tool call."""
    role: Literal["tool"] = "tool"
    tool_call_id: str = ""
    content: str = ""


@dataclass
class UserMessage:
    role: Literal["user"] = "user"
    content: str | list[Any] = ""


@dataclass
class AssistantMessage:
    role: Literal["assistant"] = "assistant"
    content: list[TextContent | ToolCallContent | ThinkingContent] = field(default_factory=list)
    model: str = ""
    provider: str = ""
    api: str = ""
    stop_reason: str = "stop"
    error_message: str | None = None
    timestamp: int = 0
    usage: dict[str, Any] = field(default_factory=lambda: {
        "input": 0,
        "output": 0,
        "cache_read": 0,
        "cache_write": 0,
        "total_tokens": 0,
        "cost": {"input": 0.0, "output": 0.0, "cache_read": 0.0, "cache_write": 0.0, "total": 0.0},
    })


Message = UserMessage | AssistantMessage | ToolResultMessage


# ------------------------------------------------------------------ #
# Tool definitions                                                    #
# ------------------------------------------------------------------ #


@dataclass
class Tool:
    """
    A tool the model can call.

    JS equivalent: Tool type using TypeBox schemas.
    Python: plain JSON schema dict for simplicity.
    """
    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)  # JSON Schema object


# ------------------------------------------------------------------ #
# Context                                                             #
# ------------------------------------------------------------------ #


@dataclass
class Context:
    """
    Serializable conversation context.

    Mirrors Context type from types.ts.
    """
    messages: list[Message] = field(default_factory=list)
    system_prompt: str | None = None
    tools: list[Tool] | None = None


# ------------------------------------------------------------------ #
# Streaming events                                                    #
# ------------------------------------------------------------------ #


@dataclass
class TextStartEvent:
    type: Literal["text_start"] = "text_start"


@dataclass
class TextDeltaEvent:
    type: Literal["text_delta"] = "text_delta"
    text: str = ""


@dataclass
class TextEndEvent:
    type: Literal["text_end"] = "text_end"
    text: str = ""


@dataclass
class ThinkingDeltaEvent:
    type: Literal["thinking_delta"] = "thinking_delta"
    thinking: str = ""


@dataclass
class ToolCallStartEvent:
    type: Literal["toolcall_start"] = "toolcall_start"
    tool_call: ToolCall = field(default_factory=lambda: ToolCall(id="", name=""))


@dataclass
class ToolCallDeltaEvent:
    type: Literal["toolcall_delta"] = "toolcall_delta"
    id: str = ""
    json_delta: str = ""


@dataclass
class ToolCallEndEvent:
    type: Literal["toolcall_end"] = "toolcall_end"
    tool_call: ToolCall = field(default_factory=lambda: ToolCall(id="", name=""))


@dataclass
class DoneEvent:
    type: Literal["done"] = "done"
    reason: str = "stop"
    message: AssistantMessage = field(default_factory=AssistantMessage)


@dataclass
class ErrorEvent:
    type: Literal["error"] = "error"
    reason: str = "error"
    error: AssistantMessage = field(default_factory=AssistantMessage)


StreamEvent = (
    TextStartEvent
    | TextDeltaEvent
    | TextEndEvent
    | ThinkingDeltaEvent
    | ToolCallStartEvent
    | ToolCallDeltaEvent
    | ToolCallEndEvent
    | DoneEvent
    | ErrorEvent
)
