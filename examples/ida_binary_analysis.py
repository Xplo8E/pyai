"""
Example 4: Autonomous binary analysis with piai + IDA Pro MCP

Uses piai's native agent() loop with IDA Pro's MCP server to let the model
autonomously reverse engineer a binary — decompile functions, trace xrefs,
find strings, and produce a full report.

Demonstrates full observability:
  - Model reasoning (ThinkingStartEvent / ThinkingDeltaEvent / ThinkingEndEvent)
  - Tool calls before execution (AgentToolCallEvent)
  - Tool results after execution (AgentToolResultEvent)
  - Per-turn summary (AgentTurnEndEvent)
  - Streamed response text (TextDeltaEvent)

Requirements:
    uv add pi-ai-py
    IDA Pro with ida-mcp plugin installed and running

Make sure:
    1. The target binary is already open in IDA Pro
    2. The IDA MCP server is running (either stdio via ida-mcp or HTTP on :13337)

Run:
    # With ida-mcp stdio (default):
    python examples/ida_binary_analysis.py /path/to/binary.so

    # With IDA HTTP MCP server:
    python examples/ida_binary_analysis.py /path/to/binary.so --http
"""

import asyncio
import sys

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

DIM    = "\033[2m"
CYAN   = "\033[36m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
RESET  = "\033[0m"

SYSTEM_PROMPT = """You are an expert ARM64 Android binary reverse engineer using IDA Pro.

The binary is already open in IDA Pro. Work through these steps:
1. Get a list of all functions — focus on Java_* JNI exports
2. Decompile each JNI function to understand what it does
3. Find cross-references to interesting functions
4. Look for strings — URLs, keys, magic values
5. Check imports / external calls
6. Identify any interesting non-JNI functions called internally

After gathering all data, write a comprehensive markdown report covering:
- All JNI functions with decompiled pseudocode and explanation
- Cross-references and call graph highlights
- Interesting strings found
- External library calls / imports
- Overall assessment: what does this binary do?

Be thorough and autonomous. Use whatever IDA tools are available to get a complete picture."""


def on_event(event):
    if isinstance(event, ThinkingStartEvent):
        print(f"\n{DIM}💭 Thinking...{RESET}", flush=True)

    elif isinstance(event, ThinkingDeltaEvent):
        print(f"{DIM}{event.thinking}{RESET}", end="", flush=True)

    elif isinstance(event, ThinkingEndEvent):
        print(f"\n{DIM}[thinking done]{RESET}\n", flush=True)

    elif isinstance(event, AgentToolCallEvent):
        args_str = ", ".join(
            f"{k}={v[:60]!r}..." if isinstance(v, str) and len(v) > 60 else f"{k}={v!r}"
            for k, v in event.tool_input.items()
        )
        print(f"\n{CYAN}🔧 Turn {event.turn} → {event.tool_name}({args_str}){RESET}", flush=True)

    elif isinstance(event, AgentToolResultEvent):
        preview = event.result[:200].replace("\n", " ")
        status = "❌" if event.error else "✅"
        print(f"{GREEN}{status} {preview}{'...' if len(event.result) > 200 else ''}{RESET}", flush=True)

    elif isinstance(event, AgentTurnEndEvent):
        thinking_note = f", {len(event.thinking)} chars reasoning" if event.thinking else ""
        print(f"\n{YELLOW}── Turn {event.turn}: {len(event.tool_calls)} call(s){thinking_note} ──{RESET}\n", flush=True)

    elif isinstance(event, TextDeltaEvent):
        print(event.text, end="", flush=True)


async def main(lib_path: str, use_http: bool = False):
    if use_http:
        mcp_server = MCPServer.http("http://127.0.0.1:13337/mcp")
        print("[*] Using IDA MCP HTTP server at http://127.0.0.1:13337/mcp")
    else:
        mcp_server = MCPServer.stdio("ida-mcp", name="ida")
        print("[*] Using IDA MCP stdio server (ida-mcp)")

    print(f"[*] Target: {lib_path}")
    print("=" * 60)

    ctx = Context(
        system_prompt=SYSTEM_PROMPT,
        messages=[UserMessage(content=f"Analyze {lib_path} and give me a full report.")],
    )

    result = await agent(
        model_id="gpt-5.1-codex-mini",
        context=ctx,
        mcp_servers=[mcp_server],
        options={"reasoning_effort": "medium"},
        max_turns=40,
        on_event=on_event,
    )

    print("\n" + "=" * 60)
    print("[*] Analysis complete.")

    if result.thinking:
        print(f"[*] Total reasoning: {len(result.thinking)} chars")

    # Print final text (convenience — streaming above should have covered it)
    if result.text:
        print("\n[FINAL REPORT]\n")
        print(result.text)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python examples/ida_binary_analysis.py /path/to/binary.so [--http]")
        sys.exit(1)
    use_http = "--http" in sys.argv
    asyncio.run(main(sys.argv[1], use_http=use_http))
