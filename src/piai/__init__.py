"""
piai — Python port of @mariozechner/pi-ai

Unified LLM API with ChatGPT Plus/Pro OAuth authentication.

Quick start:
    # 1. Login (one-time, opens browser)
    $ piai login

    # 2. Use in code
    import asyncio
    from piai import complete_text, stream
    from piai.types import Context, UserMessage

    ctx = Context(
        system_prompt="You are a helpful assistant.",
        messages=[UserMessage(content="What is 2+2?")]
    )

    async def main():
        async for event in stream("gpt-5.1-codex-mini", ctx):
            from piai.types import TextDeltaEvent
            if isinstance(event, TextDeltaEvent):
                print(event.text, end="", flush=True)

    asyncio.run(main())
"""

from .agent import agent
from .mcp import MCPHub, MCPServer
from .stream import complete, complete_text, stream
from .types import (
    AgentToolCallEvent,
    AgentToolResultEvent,
    AgentTurnEndEvent,
    AssistantMessage,
    Context,
    DoneEvent,
    ErrorEvent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ThinkingStartEvent,
    Tool,
    ToolCall,
    ToolCallContent,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    ToolResultMessage,
    UserMessage,
)

__version__ = "0.2.5"

__all__ = [
    "agent",
    "MCPServer",
    "MCPHub",
    "stream",
    "complete",
    "complete_text",
    "Context",
    "UserMessage",
    "AssistantMessage",
    "ToolResultMessage",
    "Tool",
    "ToolCall",
    "ToolCallContent",
    "TextStartEvent",
    "TextDeltaEvent",
    "TextEndEvent",
    "ThinkingStartEvent",
    "ThinkingDeltaEvent",
    "ThinkingEndEvent",
    "ToolCallStartEvent",
    "ToolCallDeltaEvent",
    "ToolCallEndEvent",
    "AgentToolCallEvent",
    "AgentToolResultEvent",
    "AgentTurnEndEvent",
    "DoneEvent",
    "ErrorEvent",
]
