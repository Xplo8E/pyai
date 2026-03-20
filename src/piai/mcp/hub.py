"""
MCPHub — manages multiple MCP servers as one unified tool pool.

Connects to all servers concurrently, merges their tools into a flat list,
and routes tool calls to the correct server automatically.

Tool name collision handling:
  If two servers expose a tool with the same name, both tools are namespaced:
  "server1__tool_name" and "server2__tool_name".
  A warning is logged when this happens.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from ..types import Tool
from .client import MCPClient, DEFAULT_CONNECT_TIMEOUT, DEFAULT_TOOL_RESULT_MAX_CHARS
from .server import MCPServer

logger = logging.getLogger(__name__)


class MCPHub:
    """
    Multi-server MCP manager.

    Connects to all servers concurrently on entry, merges their tools into
    a single flat list, and routes tool calls to the right server.

    Usage:
        async with MCPHub([
            MCPServer.stdio("r2pm -r r2mcp"),
            MCPServer.stdio("npx @modelcontextprotocol/server-filesystem /tmp"),
        ]) as hub:
            tools = hub.all_tools()
            result = await hub.call_tool("open_file", {"file_path": "/lib.so"})

    Partial failure handling:
        If a server fails to connect, it is skipped with a warning and the hub
        continues with the remaining servers. Tools from the failed server are
        simply not available. This prevents one bad server from breaking everything.

        To require all servers to connect successfully, set require_all=True.
    """

    def __init__(
        self,
        servers: list[MCPServer],
        *,
        require_all: bool = False,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
        tool_result_max_chars: int = DEFAULT_TOOL_RESULT_MAX_CHARS,
    ) -> None:
        """
        Args:
            servers:              List of MCPServer configs.
            require_all:          If True, raise if any server fails to connect.
                                  Default False — partial connections are allowed.
            connect_timeout:      Per-server connection timeout in seconds. Default 60.
            tool_result_max_chars: Max chars per tool result. Default 32000.
        """
        self._servers = servers
        self._require_all = require_all
        self._connect_timeout = connect_timeout
        self._tool_result_max_chars = tool_result_max_chars

        self._clients: list[MCPClient] = []
        # tool_name -> (MCPClient, original_name)
        # original_name is the name to use when actually calling the tool
        # (differs from registered name when namespaced)
        self._tool_registry: dict[str, tuple[MCPClient, str]] = {}
        # piai Tool objects (merged, namespaced if collision)
        self._tools: list[Tool] = []
        self._connected = False
        # Prevents double-initialization if connect() is called concurrently
        self._connect_lock = asyncio.Lock()

    async def connect(self) -> None:
        """Connect to all servers concurrently and discover their tools."""
        async with self._connect_lock:
            await self._connect_inner()

    async def _connect_inner(self) -> None:
        """Guarded connect — must only be called while holding _connect_lock."""
        if self._connected:
            return

        self._clients = [
            MCPClient(
                s,
                connect_timeout=self._connect_timeout,
                tool_result_max_chars=self._tool_result_max_chars,
            )
            for s in self._servers
        ]

        # Connect all in parallel, capture individual failures
        results = await asyncio.gather(
            *[c.connect() for c in self._clients],
            return_exceptions=True,
        )

        failed = []
        for client, result in zip(self._clients, results):
            if isinstance(result, Exception):
                failed.append((client.server, result))
                logger.warning(
                    "Failed to connect to MCP server %s: %s",
                    client.server,
                    result,
                )

        if failed and self._require_all:
            names = [str(s) for s, _ in failed]
            raise RuntimeError(
                f"Failed to connect to required MCP server(s): {names}. "
                "Set require_all=False to allow partial connections."
            )

        # Discover tools from each successfully connected client
        for client in self._clients:
            if not client.is_connected:
                continue
            try:
                tools = await client.list_tools()
            except Exception as e:
                logger.warning("Failed to list tools from %s: %s", client.server, e)
                continue

            for tool in tools:
                self._register_tool(tool, client)

        self._connected = True
        logger.debug(
            "MCPHub connected: %d/%d server(s), %d tool(s): %s",
            sum(1 for c in self._clients if c.is_connected),
            len(self._clients),
            len(self._tools),
            [t.name for t in self._tools],
        )

    def _register_tool(self, tool: Tool, client: MCPClient) -> None:
        """Register a tool, handling name collisions with namespacing."""
        original_name = tool.name

        if tool.name not in self._tool_registry:
            # No collision — register as-is
            self._tool_registry[tool.name] = (client, original_name)
            self._tools.append(tool)
            return

        # Collision — namespace this tool with server name
        server_name = _safe_name(client.server.name or "server")
        namespaced = f"{server_name}__{tool.name}"

        # Also namespace the existing registration if it hasn't been yet
        existing_client, existing_original = self._tool_registry[tool.name]
        if not any(t.name == namespaced for t in self._tools):
            existing_server_name = _safe_name(existing_client.server.name or "server")
            existing_namespaced = f"{existing_server_name}__{tool.name}"

            # Re-register the existing tool under its namespaced name and remove the
            # ambiguous original-name entry so the model only sees namespaced names.
            if existing_namespaced not in self._tool_registry:
                self._tool_registry[existing_namespaced] = (existing_client, existing_original)
                # Find and update the tool entry in _tools
                for i, t in enumerate(self._tools):
                    if t.name == tool.name:
                        self._tools[i] = Tool(
                            name=existing_namespaced,
                            description=t.description,
                            parameters=t.parameters,
                        )
                        break
                # Remove the unnamespaced key — it's now ambiguous and would silently
                # route to the first server even though the model won't see the name.
                del self._tool_registry[tool.name]

        logger.warning(
            "Tool name collision: %r exists from %s. "
            "Registering new tool as %r from %s.",
            tool.name,
            existing_client.server,
            namespaced,
            client.server,
        )

        self._tool_registry[namespaced] = (client, original_name)
        self._tools.append(Tool(
            name=namespaced,
            description=tool.description,
            parameters=tool.parameters,
        ))

    def all_tools(self) -> list[Tool]:
        """Return merged flat list of all tools from all connected servers."""
        return list(self._tools)

    def tool_names(self) -> list[str]:
        """Return registered tool names (keys in _tool_registry, used for executor coverage checks)."""
        return list(self._tool_registry.keys())

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """
        Route a tool call to the correct server and return the result.

        Args:
            name:      Tool name (possibly namespaced, e.g. "ida__decompile")
            arguments: Tool arguments dict (None is normalized to {})

        Raises:
            KeyError:    If tool name not found in any connected server.
        """
        # Normalize None/non-dict arguments — MCP SDK requires a dict
        if not isinstance(arguments, dict):
            arguments = {}

        entry = self._tool_registry.get(name)
        if entry is None:
            available = list(self._tool_registry.keys())
            raise KeyError(
                f"Tool {name!r} not found. Available tools: {available}"
            )

        client, original_name = entry
        return await client.call_tool(original_name, arguments)

    async def close(self) -> None:
        """Disconnect all servers cleanly."""
        await asyncio.gather(
            *[c.close() for c in self._clients],
            return_exceptions=True,
        )
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def connected_servers(self) -> list[MCPServer]:
        """Return list of successfully connected servers."""
        return [c.server for c in self._clients if c.is_connected]

    # ------------------------------------------------------------------ #
    # Async context manager                                               #
    # ------------------------------------------------------------------ #

    async def __aenter__(self) -> "MCPHub":
        await self.connect()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()


def _safe_name(name: str) -> str:
    """Make a name safe for use as a tool namespace prefix."""
    return name.replace("-", "_").replace(".", "_").replace(" ", "_")
