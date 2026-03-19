"""
Example 2: PiAIChatModel + LangChain + MCP tools

Combines three things:
  - piai: ChatGPT Plus OAuth (no API key, no billing)
  - LangChain: message types, tool binding, chains
  - MCP: auto-discovered tools from any MCP server

Sections in this file:
  1. simple_example()          — invoke (sync) + ainvoke (async)
  2. tool_calling_example()    — bind_tools with local Python functions
  3. piai_native_with_mcp()    — piai native agent() + MCP + full observability
  4. langchain_react_agent()   — LangChain create_agent + MCP (requires: uv add langchain)

Requirements:
    uv add pi-ai-py
    uv add langchain          # only for section 4

Run:
    python examples/langchain_mcp_agent.py
"""

import asyncio

from langchain_core.messages import HumanMessage, SystemMessage

from piai.langchain import PiAIChatModel
from piai.mcp import MCPServer


# ─── Model factory ────────────────────────────────────────────────────────────

def make_llm() -> PiAIChatModel:
    return PiAIChatModel(
        model_name="gpt-5.1-codex-mini",
        options={"reasoning_effort": "medium"},
    )


# ─── 1. Simple invoke (sync) + ainvoke (async) ────────────────────────────────

def simple_example():
    print("=== 1. Simple invoke / ainvoke ===\n")
    llm = make_llm()

    result = llm.invoke([
        SystemMessage(content="You are a concise assistant."),
        HumanMessage(content="What is the capital of France? One sentence."),
    ])
    print("invoke:", result.content)
    # Surface thinking if model reasoned about it
    if result.additional_kwargs.get("thinking"):
        print(f"[thinking: {result.additional_kwargs['thinking'][:100]}...]")
    print()


async def ainvoke_example():
    llm = make_llm()

    result = await llm.ainvoke([HumanMessage(content="Name three planets. One per line.")])
    print("ainvoke:", result.content)

    print("astream: ", end="")
    async for chunk in llm.astream([HumanMessage(content="Count from 1 to 5, comma separated.")]):
        print(chunk.content, end="", flush=True)
    print("\n")


# ─── 2. Tool calling with bind_tools ──────────────────────────────────────────

def tool_calling_example():
    print("=== 2. Tool calling with bind_tools ===\n")

    from langchain_core.tools import tool

    @tool
    def calculator(expression: str) -> str:
        """Evaluate a simple Python math expression and return the result."""
        try:
            return str(eval(expression, {"__builtins__": {}}))  # noqa: S307
        except Exception as e:
            return f"Error: {e}"

    @tool
    def word_count(text: str) -> str:
        """Count the number of words and characters in the given text."""
        return f"{len(text.split())} words, {len(text)} characters"

    llm = make_llm()
    llm_with_tools = llm.bind_tools([calculator, word_count])

    result = llm_with_tools.invoke([
        HumanMessage(content="What is 123 * 456? Also count the words in 'hello world foo bar'.")
    ])

    print("Response:", result.content or "(tool calls made)")
    if result.tool_calls:
        print("Tool calls:")
        for tc in result.tool_calls:
            print(f"  {tc['name']}({tc['args']})")
    print()


# ─── 3. piai native agent() + MCP + full observability ────────────────────────

async def piai_native_with_mcp():
    print("=== 3. piai native agent() + MCP + observability ===\n")

    from piai import agent
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

    DIM, CYAN, GREEN, YELLOW, RESET = "\033[2m", "\033[36m", "\033[32m", "\033[33m", "\033[0m"

    def on_event(event):
        if isinstance(event, ThinkingStartEvent):
            print(f"\n{DIM}💭 Thinking...{RESET}", flush=True)
        elif isinstance(event, ThinkingDeltaEvent):
            print(f"{DIM}{event.thinking}{RESET}", end="", flush=True)
        elif isinstance(event, ThinkingEndEvent):
            print(f"\n{DIM}[thinking done]{RESET}\n", flush=True)
        elif isinstance(event, AgentToolCallEvent):
            args_str = ", ".join(f"{k}={v!r}" for k, v in event.tool_input.items())
            print(f"\n{CYAN}🔧 Turn {event.turn} → {event.tool_name}({args_str}){RESET}", flush=True)
        elif isinstance(event, AgentToolResultEvent):
            preview = event.result[:150].replace("\n", " ")
            status = "❌" if event.error else "✅"
            print(f"{GREEN}{status} {preview}{'...' if len(event.result) > 150 else ''}{RESET}", flush=True)
        elif isinstance(event, AgentTurnEndEvent):
            thinking_note = f", {len(event.thinking)} chars of reasoning" if event.thinking else ""
            print(f"\n{YELLOW}── Turn {event.turn}: {len(event.tool_calls)} tool call(s){thinking_note} ──{RESET}\n", flush=True)
        elif isinstance(event, TextDeltaEvent):
            print(event.text, end="", flush=True)

    ctx = Context(
        system_prompt="You are a helpful assistant with filesystem access.",
        messages=[UserMessage(content="List the top-level entries in /tmp and give me a one-line summary.")],
    )

    result = await agent(
        model_id="gpt-5.1-codex-mini",
        context=ctx,
        mcp_servers=[MCPServer.stdio("npx -y @modelcontextprotocol/server-filesystem /tmp")],
        options={"reasoning_effort": "low"},
        max_turns=8,
        on_event=on_event,
    )

    print(f"\n\nFinal answer: {result.text}")
    if result.thinking:
        print(f"[Model reasoning ({len(result.thinking)} chars): {result.thinking[:200]}...]")
    print(f"Stop reason: {result.stop_reason}\n")


# ─── 4. LangChain create_agent + MCP (modern API) ─────────────────────────────

async def langchain_react_agent():
    print("=== 4. LangChain create_agent + MCP tools ===\n")

    try:
        from langchain.agents import create_agent
        from langchain_core.tools import tool
    except ImportError:
        print("  [skipped] 'langchain' not installed. Run: uv add langchain\n")
        return

    from piai.mcp import MCPHubToolset

    @tool
    def calculator(expression: str) -> str:
        """Evaluate a simple Python math expression."""
        try:
            return str(eval(expression, {"__builtins__": {}}))  # noqa: S307
        except Exception as e:
            return f"Error: {e}"

    async with MCPHubToolset(
        [MCPServer.stdio("npx -y @modelcontextprotocol/server-filesystem /tmp")],
        connect_timeout=30.0,
    ) as mcp_tools:
        all_tools = [calculator] + mcp_tools
        llm = make_llm()

        agent = create_agent(
            model=llm,
            tools=all_tools,
            system_prompt="You are a helpful assistant with access to filesystem and math tools.",
        )

        result = await agent.ainvoke({
            "messages": [HumanMessage(content="List files in /tmp then calculate 17 * 23.")]
        })
        final = result["messages"][-1]
        print("Answer:", final.content)
        # LangGraph surfaces thinking via additional_kwargs when PiAIChatModel is used
        if final.additional_kwargs.get("thinking"):
            print(f"[Model thought: {final.additional_kwargs['thinking'][:150]}...]")
        print()


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    simple_example()
    tool_calling_example()

    async def _async_sections():
        await ainvoke_example()
        await piai_native_with_mcp()
        await langchain_react_agent()

    asyncio.run(_async_sections())


if __name__ == "__main__":
    main()
