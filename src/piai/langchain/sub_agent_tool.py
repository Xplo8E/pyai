"""
SubAgentTool — wrap a piai agent as a LangChain BaseTool.

This lets you drop a full piai agent (with its own model, system prompt,
and MCP servers) into any LangChain/LangGraph orchestrator as a single tool.

The primary use case is LangGraph Supervisor: the supervisor sees each
sub-agent as a tool it can call, but each sub-agent is a complete piai
agent() with its own MCP servers, model, and system prompt.

Usage:
    from piai.langchain import SubAgentTool
    from piai.mcp import MCPServer
    from langgraph_supervisor import create_supervisor
    from langchain.agents import create_agent

    # A piai sub-agent with its own MCP servers
    binary_analyzer = SubAgentTool(
        name="binary_analyzer",
        description="Analyzes native ARM64 binaries using radare2. "
                    "Pass the full path to the binary and the analysis task.",
        model_id="gpt-5.1",
        system_prompt="You are an expert ARM64 reverse engineer...",
        mcp_servers=[MCPServer.stdio("r2pm -r r2mcp")],
        max_turns=30,
        options={"reasoning_effort": "high"},
    )

    # A plain LangGraph sub-agent (no MCP needed)
    reporter = create_agent(
        model=PiAIChatModel(model_name="gpt-5.1-codex-mini"),
        tools=[],
        system_prompt="You write clear, concise security reports.",
        name="reporter",
    )

    # LangGraph Supervisor orchestrates both
    workflow = create_supervisor(
        agents=[reporter],
        tools=[binary_analyzer],   # piai agent looks like a tool to supervisor
        model=PiAIChatModel(model_name="gpt-5.1"),
        prompt="You are a security team supervisor...",
    )
    app = workflow.compile()
    result = await app.ainvoke({"messages": [{"role": "user", "content": "Analyze /lib/target.so"}]})

Requires:
    pip install langchain-core langgraph-supervisor
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional, Type

try:
    from langchain_core.tools import BaseTool
    from pydantic import BaseModel, ConfigDict, Field
except ImportError as e:
    raise ImportError(
        "langchain-core is required for SubAgentTool.\n"
        "Install it with: pip install langchain-core"
    ) from e

from ..agent import agent as piai_agent
from ..mcp.server import MCPServer
from ..types import Context, TextContent, UserMessage


class _SubAgentInput(BaseModel):
    """Input schema for SubAgentTool."""
    task: str = Field(description="The task or question for this agent to handle.")


class SubAgentTool(BaseTool):
    """
    A LangChain BaseTool that runs a full piai agent() as a single tool call.

    Each SubAgentTool has its own:
    - model (can be different from other agents)
    - system prompt (specializes the agent's role)
    - MCP servers (each agent gets its own tools)
    - max_turns and options

    This makes it possible to use piai's MCP-powered agents inside
    LangGraph Supervisor or any other LangChain orchestrator.

    Args:
        name:          Tool name (used by supervisor to identify and call this agent).
                       Keep it short and snake_case: "binary_analyzer", "code_reviewer".
        description:   What this agent does and when to call it.
                       The supervisor LLM reads this to decide which agent to use.
                       Be specific: "Analyzes ARM64 binaries using radare2. Pass the
                       binary path and what you want to know."
        model_id:      piai model to use for this agent. Default: "gpt-5.1-codex-mini"
        system_prompt: System prompt that specializes this agent's behavior.
        mcp_servers:   MCP servers available to this agent. Each SubAgentTool gets
                       its own isolated MCPHub — servers are not shared between agents.
        max_turns:     Max agentic loop iterations. Default: 20.
        options:       piai options (reasoning_effort, session_id, etc.).
        on_event:      Optional callback for streaming events during agent execution.
    """

    name: str
    description: str
    args_schema: Type[BaseModel] = _SubAgentInput

    model_id: str = "gpt-5.1-codex-mini"
    system_prompt: Optional[str] = None
    mcp_servers: list[MCPServer] = []
    max_turns: int = 20
    options: dict[str, Any] = {}
    on_event: Optional[Any] = None  # Callable[[StreamEvent], Any] | None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _run(self, task: str, **kwargs: Any) -> str:
        """Sync run — safely handles running event loop (e.g. inside LangGraph threads)."""
        import concurrent.futures

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, self._arun(task))
                return future.result()
        else:
            return asyncio.run(self._arun(task))

    async def _arun(self, task: str, **kwargs: Any) -> str:
        """Async run — spins up a full piai agent() for this task."""
        ctx = Context(
            system_prompt=self.system_prompt,
            messages=[UserMessage(content=task)],
        )

        result = await piai_agent(
            model_id=self.model_id,
            context=ctx,
            mcp_servers=self.mcp_servers if self.mcp_servers else None,
            options=self.options or None,
            max_turns=self.max_turns,
            on_event=self.on_event,
        )

        # Extract text from the final AssistantMessage
        parts = []
        for block in result.content:
            if isinstance(block, TextContent) and block.text:
                parts.append(block.text)

        return "\n\n".join(parts) if parts else "(agent produced no text output)"
