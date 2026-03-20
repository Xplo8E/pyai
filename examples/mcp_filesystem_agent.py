"""
Example 1: piai native agent + MCP filesystem server

Uses piai's built-in agent() loop with the official MCP filesystem server
to let the model read, explore, and summarize files autonomously.

Demonstrates full observability via on_event:
  - ThinkingStartEvent / ThinkingDeltaEvent / ThinkingEndEvent  — model reasoning
  - AgentToolCallEvent   — what tool the model decided to call + args
  - AgentToolResultEvent — what the tool returned
  - AgentTurnEndEvent    — summary at end of each turn
  - TextDeltaEvent       — streaming final response text

Requirements:
    uv add pi-ai-py

Run:
    python examples/mcp_filesystem_agent.py
"""

import asyncio

from piai import agent
from piai.mcp import MCPServer
from piai.types import (
    AgentToolCallEvent,
    AgentToolResultEvent,
    AgentTurnEndEvent,
    Context,
    TextDeltaEvent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ThinkingStartEvent,
    UserMessage,
)

TARGET_DIR = "/tmp"

# ANSI helpers
DIM   = "\033[2m"
CYAN  = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RESET = "\033[0m"


def on_event(event):
    """Full observability callback — shows thinking, tool calls, results, and text."""

    if isinstance(event, ThinkingStartEvent):
        print(f"\n{DIM}💭 Thinking...{RESET}", flush=True)

    elif isinstance(event, ThinkingDeltaEvent):
        print(f"{DIM}{event.thinking}{RESET}", end="", flush=True)

    elif isinstance(event, ThinkingEndEvent):
        print(f"\n{DIM}[thinking complete]{RESET}\n", flush=True)

    elif isinstance(event, AgentToolCallEvent):
        args_str = ", ".join(f"{k}={v!r}" for k, v in event.tool_input.items())
        print(f"\n{CYAN}🔧 Turn {event.turn} → {event.tool_name}({args_str}){RESET}", flush=True)

    elif isinstance(event, AgentToolResultEvent):
        preview = event.result[:200].replace("\n", " ")
        status = "❌" if event.error else "✅"
        print(f"{GREEN}{status} Result: {preview}{'...' if len(event.result) > 200 else ''}{RESET}", flush=True)

    elif isinstance(event, AgentTurnEndEvent):
        thinking_note = f", thought for {len(event.thinking)} chars" if event.thinking else ""
        print(f"\n{YELLOW}── Turn {event.turn} done: {len(event.tool_calls)} tool call(s){thinking_note} ──{RESET}\n", flush=True)

    elif isinstance(event, TextDeltaEvent):
        print(event.text, end="", flush=True)


async def main():
    ctx = Context(
        system_prompt=(
            "You are a helpful file system explorer. "
            "Use the available tools to explore the directory the user specifies. "
            "Always list files first, then read relevant ones, then summarize your findings."
        ),
        messages=[
            UserMessage(
                content=(
                    f"Explore {TARGET_DIR} and give me a summary of what's in there. "
                    "List the files, read a few interesting ones, and explain what you find."
                )
            )
        ],
    )

    print(f"Starting agent — exploring {TARGET_DIR!r} ...\n")
    print("─" * 60)

    result = await agent(
        model_id="gpt-5.1-codex-mini",
        context=ctx,
        mcp_servers=[
            MCPServer.stdio(f"npx -y @modelcontextprotocol/server-filesystem {TARGET_DIR}"),
        ],
        options={"reasoning_effort": "medium"},
        max_turns=15,
        on_event=on_event,
        require_all_servers=True,
        connect_timeout=30.0,
        tool_result_max_chars=16_000,
    )

    print("\n" + "─" * 60)
    print(f"\nStop reason : {result.stop_reason}")
    if result.thinking:
        print(f"Thinking    : {len(result.thinking)} chars total")
    print(f"Tokens      : in={result.usage.get('input','?')}  out={result.usage.get('output','?')}")


if __name__ == "__main__":
    asyncio.run(main())
