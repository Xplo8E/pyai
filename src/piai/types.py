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

    @property
    def text(self) -> str:
        """Concatenated text response (convenience accessor)."""
        return "".join(
            block.text for block in self.content if isinstance(block, TextContent)
        )

    @property
    def thinking(self) -> str | None:
        """
        Full reasoning/thinking text if the model emitted any, else None.

        Concatenates all ThinkingContent blocks in order. Returns None (not "")
        when the model produced no thinking output at all, so callers can
        distinguish "thought nothing" from "thinking not supported".
        """
        parts = [block.thinking for block in self.content if isinstance(block, ThinkingContent)]
        return "\n\n".join(parts) if parts else None


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
class ThinkingStartEvent:
    type: Literal["thinking_start"] = "thinking_start"


@dataclass
class ThinkingDeltaEvent:
    type: Literal["thinking_delta"] = "thinking_delta"
    thinking: str = ""


@dataclass
class ThinkingEndEvent:
    type: Literal["thinking_end"] = "thinking_end"
    thinking: str = ""  # full accumulated thinking text for this block


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
class AgentToolCallEvent:
    """Fired just before agent() executes a tool call — gives visibility into what the model decided to do."""
    type: Literal["agent_tool_call"] = "agent_tool_call"
    turn: int = 0
    tool_name: str = ""
    tool_input: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentToolResultEvent:
    """Fired after agent() receives the tool result — gives visibility into what the tool returned."""
    type: Literal["agent_tool_result"] = "agent_tool_result"
    turn: int = 0
    tool_name: str = ""
    tool_input: dict[str, Any] = field(default_factory=dict)
    result: str = ""
    error: bool = False


@dataclass
class AgentTurnEndEvent:
    """Fired at the end of each agent loop turn with a summary of what happened."""
    type: Literal["agent_turn_end"] = "agent_turn_end"
    turn: int = 0
    thinking: str | None = None   # full thinking text for this turn (if any)
    tool_calls: list[ToolCall] = field(default_factory=list)


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
    | ThinkingStartEvent
    | ThinkingDeltaEvent
    | ThinkingEndEvent
    | ToolCallStartEvent
    | ToolCallDeltaEvent
    | ToolCallEndEvent
    | AgentToolCallEvent
    | AgentToolResultEvent
    | AgentTurnEndEvent
    | DoneEvent
    | ErrorEvent
)
