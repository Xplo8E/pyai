"""
pyai — Python port of @mariozechner/pi-ai

Unified LLM API with ChatGPT Plus/Pro OAuth authentication.

Quick start:
    # 1. Login (one-time, opens browser)
    $ pyai login

    # 2. Use in code
    import asyncio
    from piai import complete_text, stream
    from piai.types import Context, UserMessage

    ctx = Context(
        system_prompt="You are a helpful assistant.",
        messages=[UserMessage(content="What is 2+2?")]
    )

    async def main():
        async for event in stream("gpt-4o", ctx):
            from piai.types import TextDeltaEvent
            if isinstance(event, TextDeltaEvent):
                print(event.text, end="", flush=True)

    asyncio.run(main())
"""

from .stream import complete, complete_text, stream
from .types import (
    AssistantMessage,
    Context,
    DoneEvent,
    ErrorEvent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ThinkingDeltaEvent,
    Tool,
    ToolCall,
    ToolCallContent,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    ToolResultMessage,
    UserMessage,
)

__version__ = "0.1.0"

__all__ = [
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
    "ThinkingDeltaEvent",
    "ToolCallStartEvent",
    "ToolCallDeltaEvent",
    "ToolCallEndEvent",
    "DoneEvent",
    "ErrorEvent",
]
