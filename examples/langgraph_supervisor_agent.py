"""
Example: LangGraph Supervisor + piai + MCP tools

Shows the full orchestrator pattern using piai as the underlying engine:
  - LangGraph Supervisor as the orchestrator (autonomous, decides which agent to call)
  - PiAIChatModel as the model for supervisor and sub-agents
  - MCP servers exposed as LangChain tools via piai's bridge
  - SubAgentTool to wrap full piai agents (with MCP) as supervisor-callable tools
  - Full thinking/observability via on_event on SubAgentTool

Requirements:
    uv add pi-ai-py langgraph langgraph-supervisor langchain-core

Run:
    python examples/langgraph_supervisor_agent.py
"""

import asyncio

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph_supervisor import create_supervisor

from piai.langchain import PiAIChatModel, SubAgentTool
from piai.mcp import MCPHubToolset, MCPServer
from piai.types import (
    AgentToolCallEvent,
    AgentToolResultEvent,
    AgentTurnEndEvent,
    TextDeltaEvent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ThinkingStartEvent,
)


# ─── ANSI helpers ─────────────────────────────────────────────────────────────

DIM    = "\033[2m"
CYAN   = "\033[36m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
RESET  = "\033[0m"


# ─── Local Python tools (no MCP needed) ───────────────────────────────────────

@tool
def calculator(expression: str) -> str:
    """Evaluate a Python math expression and return the result."""
    try:
        return str(eval(expression, {"__builtins__": {}}))  # noqa: S307
    except Exception as e:
        return f"Error: {e}"


# ─── Observability callback for SubAgentTool ──────────────────────────────────

def make_sub_agent_observer(agent_name: str):
    """Returns an on_event callback that prints inner agent activity with a label."""
    def on_event(event):
        label = f"[{agent_name}]"
        if isinstance(event, ThinkingStartEvent):
            print(f"\n  {DIM}{label} 💭 thinking...{RESET}", flush=True)
        elif isinstance(event, ThinkingDeltaEvent):
            print(f"  {DIM}{event.thinking}{RESET}", end="", flush=True)
        elif isinstance(event, ThinkingEndEvent):
            print(f"\n  {DIM}{label} thinking done{RESET}", flush=True)
        elif isinstance(event, AgentToolCallEvent):
            args_str = ", ".join(f"{k}={v!r}" for k, v in event.tool_input.items())
            print(f"\n  {CYAN}{label} 🔧 {event.tool_name}({args_str}){RESET}", flush=True)
        elif isinstance(event, AgentToolResultEvent):
            preview = event.result[:120].replace("\n", " ")
            status = "❌" if event.error else "✅"
            print(f"  {GREEN}{label} {status} {preview}{'...' if len(event.result) > 120 else ''}{RESET}", flush=True)
        elif isinstance(event, AgentTurnEndEvent):
            thinking_note = f", {len(event.thinking)} chars reasoning" if event.thinking else ""
            print(f"\n  {YELLOW}{label} turn {event.turn}: {len(event.tool_calls)} call(s){thinking_note}{RESET}\n", flush=True)
        elif isinstance(event, TextDeltaEvent):
            print(f"  {event.text}", end="", flush=True)
    return on_event


# ─── Option A: Sub-agents using MCP tools via bridge ──────────────────────────
# Standard LangGraph react agents get MCP tools via MCPHubToolset bridge.
# Thinking is surfaced via AIMessage.additional_kwargs["thinking"] on the final message.

async def run_with_mcp_bridge():
    print("=== Option A: LangGraph Supervisor + MCP bridge ===\n")

    async with MCPHubToolset(
        [MCPServer.stdio("npx -y @modelcontextprotocol/server-filesystem /tmp")],
        connect_timeout=30.0,
    ) as mcp_tools:

        file_agent = create_agent(
            model=PiAIChatModel(model_name="gpt-5.1-codex-mini"),
            tools=mcp_tools,
            system_prompt="You are a filesystem expert. Use the available tools to read and explore files.",
            name="file_agent",
        )

        math_agent = create_agent(
            model=PiAIChatModel(model_name="gpt-5.1-codex-mini"),
            tools=[calculator],
            system_prompt="You are a math expert. Use the calculator tool for all computations.",
            name="math_agent",
        )

        workflow = create_supervisor(
            agents=[file_agent, math_agent],
            model=PiAIChatModel(model_name="gpt-5.1-codex-mini"),
            prompt=(
                "You are a team supervisor with two specialists:\n"
                "- file_agent: for filesystem exploration and file reading\n"
                "- math_agent: for calculations and math problems\n"
                "Delegate tasks to the right specialist. Combine results into a final answer."
            ),
            output_mode="last_message",
        )
        app = workflow.compile()

        result = await app.ainvoke({
            "messages": [HumanMessage(content="List files in /tmp and also calculate 17 * 23.")]
        })
        final = result["messages"][-1]
        print("Result:", final.content)
        if final.additional_kwargs.get("thinking"):
            print(f"{DIM}[supervisor thinking: {final.additional_kwargs['thinking'][:150]}...]{RESET}")


# ─── Option B: SubAgentTool — full piai agent (with MCP) as supervisor tool ───
# Each sub-agent has its own MCP server, model, system prompt, and observability.
# The supervisor treats the whole thing as a single callable tool (black box).

async def run_with_sub_agent_tool():
    print("\n=== Option B: LangGraph Supervisor + SubAgentTool ===\n")

    filesystem_agent = SubAgentTool(
        name="filesystem_explorer",
        description=(
            "Explores the filesystem, reads files, and summarizes their contents. "
            "Pass a task describing what to look for and where."
        ),
        model_id="gpt-5.1-codex-mini",
        system_prompt=(
            "You are a filesystem exploration expert. "
            "Use your tools to explore directories, read files, and provide clear summaries."
        ),
        mcp_servers=[
            MCPServer.stdio("npx -y @modelcontextprotocol/server-filesystem /tmp"),
        ],
        max_turns=8,
        options={"reasoning_effort": "low"},
        # Full inner-loop observability — see thinking, tool calls, results in real time
        on_event=make_sub_agent_observer("filesystem_explorer"),
    )

    math_agent = create_agent(
        model=PiAIChatModel(model_name="gpt-5.1-codex-mini"),
        tools=[calculator],
        system_prompt="You are a math expert.",
        name="math_agent",
    )

    workflow = create_supervisor(
        agents=[math_agent],
        tools=[filesystem_agent],
        model=PiAIChatModel(model_name="gpt-5.1-codex-mini"),
        prompt=(
            "You are a supervisor. You have:\n"
            "- filesystem_explorer: to explore /tmp and read files\n"
            "- math_agent: to do calculations\n"
            "Delegate appropriately and combine results."
        ),
        output_mode="last_message",
    )
    app = workflow.compile()

    result = await app.ainvoke({
        "messages": [HumanMessage(content="What files are in /tmp? Also what is 99 * 88?")]
    })
    print("Result:", result["messages"][-1].content)


# ─── Main ──────────────────────────────────────────────────────────────────────

async def main():
    await run_with_mcp_bridge()
    await run_with_sub_agent_tool()


if __name__ == "__main__":
    asyncio.run(main())
