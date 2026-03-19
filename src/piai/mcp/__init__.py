"""
piai MCP integration.

Plug-and-play MCP server support for piai. Connect any MCP server
and the agent will auto-discover tools and route calls.

Usage:
    from piai import agent
    from piai.mcp import MCPServer
    from piai.types import Context, UserMessage

    ctx = Context(messages=[UserMessage(content="Analyze /lib/target.so")])

    result = await agent(
        model_id="gpt-5.1-codex-mini",
        context=ctx,
        mcp_servers=[
            MCPServer.stdio("r2pm -r r2mcp"),
            MCPServer.stdio("npx @modelcontextprotocol/server-filesystem /tmp"),
            MCPServer.http("http://localhost:9000/mcp"),
        ],
    )
    print(result)
"""

from .client import MCPClient
from .hub import MCPHub
from .langchain_tools import MCPHubToolset, MCPLangChainTool, to_langchain_tools
from .server import MCPServer

__all__ = [
    "MCPServer",
    "MCPClient",
    "MCPHub",
    "to_langchain_tools",
    "MCPHubToolset",
    "MCPLangChainTool",
]
