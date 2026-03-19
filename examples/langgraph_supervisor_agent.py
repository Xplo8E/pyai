"""
Example: LangGraph Supervisor + piai + MCP tools

Shows the full orchestrator pattern using piai as the underlying engine:
  - LangGraph Supervisor as the orchestrator (autonomous, decides which agent to call)
  - PiAIChatModel as the model for supervisor and sub-agents
  - MCP servers exposed as LangChain tools via piai's bridge
  - SubAgentTool to wrap full piai agents (with MCP) as supervisor-callable tools

Requirements:
    uv add pi-ai-py langgraph langgraph-supervisor langchain-core

Run:
    python examples/langgraph_supervisor_agent.py
"""

import asyncio

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor

from piai.langchain import PiAIChatModel, SubAgentTool
from piai.mcp import MCPHubToolset, MCPServer


# ─── Local Python tools (no MCP needed) ───────────────────────────────────────

@tool
def calculator(expression: str) -> str:
    """Evaluate a Python math expression and return the result."""
    try:
        return str(eval(expression, {"__builtins__": {}}))  # noqa: S307
    except Exception as e:
        return f"Error: {e}"


# ─── Option A: Sub-agent using MCP tools via bridge ───────────────────────────
# Use this when you want a standard LangGraph react agent but with MCP tools.
# MCPHubToolset connects to MCP servers and returns LangChain-compatible tools.

async def run_with_mcp_bridge():
    print("=== LangGraph Supervisor + MCP bridge ===\n")

    async with MCPHubToolset(
        [MCPServer.stdio("npx -y @modelcontextprotocol/server-filesystem /tmp")],
        connect_timeout=30.0,
    ) as mcp_tools:

        # Sub-agent: standard LangGraph react agent + MCP filesystem tools
        file_agent = create_react_agent(
            model=PiAIChatModel(model_name="gpt-5.1-codex-mini"),
            tools=mcp_tools,   # MCP tools visible to LangGraph thanks to bridge
            prompt="You are a filesystem expert. Use the available tools to read and explore files.",
            name="file_agent",
        )

        # Sub-agent: local tools only, different model
        math_agent = create_react_agent(
            model=PiAIChatModel(model_name="gpt-5.1-codex-mini"),
            tools=[calculator],
            prompt="You are a math expert. Use the calculator tool for all computations.",
            name="math_agent",
        )

        # Supervisor: orchestrates both agents autonomously
        supervisor_model = PiAIChatModel(model_name="gpt-5.1-codex-mini")
        workflow = create_supervisor(
            agents=[file_agent, math_agent],
            model=supervisor_model,
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
        print("Result:", result["messages"][-1].content)


# ─── Option B: SubAgentTool — full piai agent (with MCP) as supervisor tool ───
# Use this when each sub-agent needs its OWN MCP servers, model, and system prompt.
# SubAgentTool wraps a complete piai agent() so the supervisor treats it as a tool.

async def run_with_sub_agent_tool():
    print("\n=== LangGraph Supervisor + SubAgentTool ===\n")

    # This sub-agent has its own MCP server, own model, own system prompt
    # The supervisor sees it as a single callable tool
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
    )

    # Plain react agent for math (no MCP needed)
    math_agent = create_react_agent(
        model=PiAIChatModel(model_name="gpt-5.1-codex-mini"),
        tools=[calculator],
        prompt="You are a math expert.",
        name="math_agent",
    )

    # Supervisor orchestrates both
    workflow = create_supervisor(
        agents=[math_agent],
        tools=[filesystem_agent],   # SubAgentTool alongside standard LangGraph agents
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
    # Run Option A (MCP bridge with standard LangGraph agents)
    await run_with_mcp_bridge()

    # Run Option B (SubAgentTool — piai agent as supervisor tool)
    await run_with_sub_agent_tool()


if __name__ == "__main__":
    asyncio.run(main())
