"""
Example 2: PiAIChatModel + LangChain + MCP tools

Combines three things:
  - piai: ChatGPT Plus OAuth (no API key, no billing)
  - LangChain: message types, tool binding, chains
  - MCP: auto-discovered tools from any MCP server

Sections in this file:
  1. simple_example()          — invoke (sync) + ainvoke (async)
  2. tool_calling_example()    — bind_tools with local Python functions
  3. piai_native_with_mcp()    — piai native agent() + MCP (recommended)
  4. langchain_react_agent()   — LangChain ReAct agent + MCP (requires: uv add langchain)

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
        options={
            "reasoning_effort": "medium",
        },
    )


# ─── 1. Simple invoke (sync) + ainvoke (async) ────────────────────────────────

def simple_example():
    """Sync invoke — must be called from outside asyncio.run()."""
    print("=== 1. Simple invoke / ainvoke ===\n")
    llm = make_llm()

    # Sync invoke — works from any sync context
    result = llm.invoke([
        SystemMessage(content="You are a concise assistant."),
        HumanMessage(content="What is the capital of France? One sentence."),
    ])
    print("invoke:", result.content)


async def ainvoke_example():
    """Async ainvoke + astream — called from inside asyncio.run()."""
    llm = make_llm()

    # Async ainvoke
    result = await llm.ainvoke([HumanMessage(content="Name three planets. One per line.")])
    print("ainvoke:", result.content)

    # Token-by-token astream
    print("astream: ", end="")
    async for chunk in llm.astream([HumanMessage(content="Count from 1 to 5, comma separated.")]):
        print(chunk.content, end="", flush=True)
    print("\n")


# ─── 2. Tool calling with bind_tools ──────────────────────────────────────────

def tool_calling_example():
    """bind_tools — sync, called outside asyncio.run()."""
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

    print("Response:", result.content or "(tool calls made — execute them to get final answer)")
    if result.tool_calls:
        print("Tool calls:")
        for tc in result.tool_calls:
            print(f"  {tc['name']}({tc['args']})")
    print()


# ─── 3. piai native agent() + MCP (recommended path) ─────────────────────────

async def piai_native_with_mcp():
    print("=== 3. piai native agent() + MCP filesystem ===\n")

    from piai import agent
    from piai.types import Context, TextDeltaEvent, UserMessage

    ctx = Context(
        system_prompt="You are a helpful assistant with filesystem access.",
        messages=[UserMessage(content="List the top-level entries in /tmp and give me a one-line summary.")],
    )

    print("Agent: ", end="")
    result = await agent(
        model_id="gpt-5.1-codex-mini",
        context=ctx,
        mcp_servers=[
            MCPServer.stdio("npx -y @modelcontextprotocol/server-filesystem /tmp"),
        ],
        options={"reasoning_effort": "low"},
        max_turns=8,
        on_event=lambda e: print(e.text, end="", flush=True) if isinstance(e, TextDeltaEvent) else None,
    )

    print(f"\nDone. Stop reason: {result.stop_reason}\n")


# ─── 4. LangChain agent + MCP tools (modern API) ──────────────────────────────

async def langchain_react_agent():
    """LangChain agent with MCP tools using the modern create_agent API (LangGraph v1.x)."""
    print("=== 4. LangChain agent + MCP tools ===\n")

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
        print("\nFinal answer:", result["messages"][-1].content)


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Section 1 (sync part)
    simple_example()

    # Section 2 — sync
    tool_calling_example()

    # Sections 1 (async), 3, and 4 — all async
    async def _async_sections():
        await ainvoke_example()
        await piai_native_with_mcp()
        await langchain_react_agent()

    asyncio.run(_async_sections())


if __name__ == "__main__":
    main()
