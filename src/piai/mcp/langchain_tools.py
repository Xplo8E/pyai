"""
MCP → LangChain tool bridge.

Converts piai MCP servers into LangChain BaseTool objects so they can be
used directly inside LangGraph agents, create_supervisor, or any other
LangChain-based orchestrator.

Usage:
    from piai.mcp import MCPServer, to_langchain_tools

    tools, hub = await to_langchain_tools([
        MCPServer.stdio("r2pm -r r2mcp"),
        MCPServer.http("http://localhost:13337/mcp"),
    ])

    # Drop directly into a LangGraph agent (modern API)
    from langchain.agents import create_agent
    agent = create_agent(model=llm, tools=tools)

    # Or use MCPHubToolset as an async context manager
    async with MCPHubToolset([...]) as tools:
        agent = create_agent(model=llm, tools=tools)

Notes:
    - The MCPHub connection stays alive for the lifetime of the returned tools.
      Keep a reference to the hub and close it when done, or use it as an
      async context manager (see MCPHubToolset).
    - Requires: langchain-core
"""

from __future__ import annotations

import asyncio
from typing import Any, Type

from ..types import Tool
from .hub import MCPHub
from .server import MCPServer

try:
    from langchain_core.tools import BaseTool
    from pydantic import BaseModel, ConfigDict, Field, create_model
except ImportError as e:
    raise ImportError(
        "langchain-core is required for MCP → LangChain tool bridge.\n"
        "Install it with: pip install langchain-core"
    ) from e


def _make_input_schema(tool: Tool) -> Type[BaseModel]:
    """
    Build a Pydantic input model from a piai Tool's JSON Schema parameters.

    LangChain uses Pydantic models to validate and document tool inputs.
    We dynamically generate one from the tool's JSON Schema.
    """
    props: dict = tool.parameters.get("properties", {})
    required: list = tool.parameters.get("required", [])

    fields: dict[str, Any] = {}
    for prop_name, prop_schema in props.items():
        description = prop_schema.get("description", "")
        json_type = prop_schema.get("type", "string")

        # Map JSON Schema types → Python types
        type_map = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        py_type = type_map.get(json_type, Any)

        if prop_name in required:
            fields[prop_name] = (py_type, Field(description=description))
        else:
            fields[prop_name] = (py_type | None, Field(default=None, description=description))

    return create_model(f"{tool.name}_input", **fields)


class MCPLangChainTool(BaseTool):
    """
    A LangChain BaseTool that wraps a single MCP tool via MCPHub.

    This is returned by to_langchain_tools() — you don't usually
    instantiate it directly.
    """

    name: str
    description: str
    args_schema: Type[BaseModel]
    hub: Any  # MCPHub — typed as Any to avoid pydantic issues with non-pydantic classes
    mcp_tool_name: str  # actual name used to route call via hub

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _run(self, **kwargs: Any) -> str:
        """Sync run — safely handles running event loop (e.g. inside LangGraph threads)."""
        import concurrent.futures

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, self._arun(**kwargs))
                return future.result()
        else:
            return asyncio.run(self._arun(**kwargs))

    async def _arun(self, **kwargs: Any) -> str:
        """Async run — calls the MCP tool via the hub."""
        # Filter out None values from optional fields
        args = {k: v for k, v in kwargs.items() if v is not None}
        return await self.hub.call_tool(self.mcp_tool_name, args)


class MCPHubToolset:
    """
    Async context manager that connects to MCP servers and returns
    LangChain-compatible tools. Cleans up connections on exit.

    Usage:
        async with MCPHubToolset([MCPServer.stdio("r2pm -r r2mcp")]) as tools:
            agent = create_agent(llm, tools=tools)
            result = await agent.ainvoke({"messages": [...]})
    """

    def __init__(
        self,
        servers: list[MCPServer],
        *,
        require_all: bool = False,
        connect_timeout: float = 60.0,
        tool_result_max_chars: int = 32_000,
    ) -> None:
        self._servers = servers
        self._hub = MCPHub(
            servers,
            require_all=require_all,
            connect_timeout=connect_timeout,
            tool_result_max_chars=tool_result_max_chars,
        )
        self._tools: list[MCPLangChainTool] = []

    async def __aenter__(self) -> list[MCPLangChainTool]:
        await self._hub.connect()
        self._tools = _hub_to_langchain_tools(self._hub)
        return self._tools

    async def __aexit__(self, *args: Any) -> None:
        await self._hub.close()


def _hub_to_langchain_tools(hub: MCPHub) -> list[MCPLangChainTool]:
    """Convert all tools from a connected MCPHub to LangChain tools."""
    lc_tools = []
    for piai_tool in hub.all_tools():
        schema = _make_input_schema(piai_tool)
        lc_tools.append(
            MCPLangChainTool(
                name=piai_tool.name,
                description=piai_tool.description or f"MCP tool: {piai_tool.name}",
                args_schema=schema,
                hub=hub,
                mcp_tool_name=piai_tool.name,
            )
        )
    return lc_tools


async def to_langchain_tools(
    servers: list[MCPServer],
    *,
    require_all: bool = False,
    connect_timeout: float = 60.0,
    tool_result_max_chars: int = 32_000,
) -> tuple[list[MCPLangChainTool], MCPHub]:
    """
    Connect to MCP servers and return LangChain-compatible tools.

    Returns a tuple of (tools, hub). You MUST call hub.close() when done,
    or use MCPHubToolset as an async context manager instead.

    Args:
        servers:               List of MCPServer configs to connect to.
        require_all:           Raise if any server fails to connect. Default False.
        connect_timeout:       Per-server connection timeout in seconds. Default 60.
        tool_result_max_chars: Max chars per tool result. Default 32000.

    Returns:
        (tools, hub) — list of LangChain BaseTool objects + the MCPHub managing them.

    Example:
        tools, hub = await to_langchain_tools([
            MCPServer.stdio("r2pm -r r2mcp"),
            MCPServer.http("http://localhost:13337/mcp"),
        ])
        try:
            agent = create_agent(llm, tools=tools)
            result = await agent.ainvoke({"messages": [...]})
        finally:
            await hub.close()

    Tip:
        Use MCPHubToolset for automatic cleanup:
            async with MCPHubToolset(servers) as tools:
                agent = create_agent(llm, tools=tools)
    """
    hub = MCPHub(
        servers,
        require_all=require_all,
        connect_timeout=connect_timeout,
        tool_result_max_chars=tool_result_max_chars,
    )
    await hub.connect()
    tools = _hub_to_langchain_tools(hub)
    return tools, hub
