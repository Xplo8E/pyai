"""
MCPClient — manages a persistent connection to one MCP server.

Uses AsyncExitStack to keep the transport alive across multiple tool calls,
which is critical for stateful servers like r2mcp and ida-mcp where
open_file → analyze → decompile must all hit the same process.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from ..types import Tool
from .server import MCPServer

logger = logging.getLogger(__name__)

# Default timeout for connecting to an MCP server (seconds)
DEFAULT_CONNECT_TIMEOUT = 60.0
# Default max chars returned from a single tool call (prevents context explosion)
DEFAULT_TOOL_RESULT_MAX_CHARS = 32_000


class MCPClient:
    """
    Persistent connection to a single MCP server.

    Usage:
        client = MCPClient(MCPServer.stdio("r2pm -r r2mcp"))
        await client.connect()
        tools = await client.list_tools()
        result = await client.call_tool("open_file", {"file_path": "/lib.so"})
        await client.close()

    Or as async context manager:
        async with MCPClient(server) as client:
            tools = await client.list_tools()
            result = await client.call_tool("open_file", {"file_path": "/lib.so"})
    """

    def __init__(
        self,
        server: MCPServer,
        *,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
        tool_result_max_chars: int = DEFAULT_TOOL_RESULT_MAX_CHARS,
    ) -> None:
        self.server = server
        self._connect_timeout = connect_timeout
        self._tool_result_max_chars = tool_result_max_chars
        self._session: ClientSession | None = None
        self._exit_stack = AsyncExitStack()
        self._connected = False

    async def connect(self) -> None:
        """
        Connect to the MCP server and initialize the session.

        Raises:
            TimeoutError:  If the server doesn't respond within connect_timeout seconds.
            RuntimeError:  If the transport or session initialization fails.
        """
        if self._connected:
            return

        try:
            await asyncio.wait_for(self._connect_inner(), timeout=self._connect_timeout)
        except asyncio.TimeoutError:
            await self._exit_stack.aclose()
            raise TimeoutError(
                f"MCP server {self.server} did not respond within "
                f"{self._connect_timeout}s. Is the server running?"
            )
        except Exception:
            await self._exit_stack.aclose()
            raise

        self._connected = True
        logger.debug("Connected to MCP server: %s", self.server)

    async def _connect_inner(self) -> None:
        if self.server.transport == "stdio":
            await self._connect_stdio()
        elif self.server.transport == "http":
            await self._connect_http()
        elif self.server.transport == "sse":
            await self._connect_sse()
        else:
            raise ValueError(f"Unknown transport: {self.server.transport!r}")

        await self._session.initialize()

    async def _connect_stdio(self) -> None:
        params = StdioServerParameters(
            command=self.server.command,
            args=self.server.args,
            env=self.server.env,
        )
        transport = await self._exit_stack.enter_async_context(stdio_client(params))
        read, write = transport
        self._session = await self._exit_stack.enter_async_context(
            ClientSession(read, write)
        )

    async def _connect_http(self) -> None:
        from mcp.client.streamable_http import streamablehttp_client

        transport = await self._exit_stack.enter_async_context(
            streamablehttp_client(self.server.url, headers=self.server.headers)
        )
        read, write, _ = transport
        self._session = await self._exit_stack.enter_async_context(
            ClientSession(read, write)
        )

    async def _connect_sse(self) -> None:
        from mcp.client.sse import sse_client

        transport = await self._exit_stack.enter_async_context(
            sse_client(self.server.url, headers=self.server.headers)
        )
        read, write = transport
        self._session = await self._exit_stack.enter_async_context(
            ClientSession(read, write)
        )

    async def list_tools(self) -> list[Tool]:
        """
        Discover all tools from this server.

        Returns piai Tool objects ready to pass into Context.tools.
        """
        self._ensure_connected()
        response = await self._session.list_tools()
        tools = []
        for t in response.tools:
            tools.append(Tool(
                name=t.name,
                description=t.description or "",
                parameters=t.inputSchema if isinstance(t.inputSchema, dict) else {},
            ))
        return tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """
        Call a tool on this server and return the result as a string.

        Tool errors are returned as result strings (prefixed with "Error:") rather
        than raised as exceptions, so the agent can report them to the model and
        potentially recover. Only connection-level errors are raised.

        Result is truncated to tool_result_max_chars to prevent context explosion.
        """
        self._ensure_connected()
        result = await self._session.call_tool(name, arguments=arguments)

        # Concatenate all content blocks
        parts = []
        for block in result.content:
            if hasattr(block, "text"):
                parts.append(block.text)
            elif hasattr(block, "data"):
                # Binary/image content — summarize rather than dump bytes
                parts.append(f"[binary data: {len(block.data)} bytes]")
            else:
                parts.append(str(block))

        text = "\n".join(parts)

        if result.isError:
            # Return as error string — don't raise. Let agent handle it.
            text = f"Tool error: {text or 'unknown error'}"

        # Truncate to prevent context explosion
        if len(text) > self._tool_result_max_chars:
            truncated = self._tool_result_max_chars
            text = text[:truncated] + f"\n... [truncated: {len(text) - truncated} more chars]"

        return text

    async def close(self) -> None:
        """Shut down the connection cleanly."""
        if self._connected:
            try:
                await self._exit_stack.aclose()
            except Exception as e:
                logger.debug("Error closing MCP client %s: %s", self.server, e)
            finally:
                self._connected = False
                self._session = None

    def _ensure_connected(self) -> None:
        if not self._connected or self._session is None:
            raise RuntimeError(
                f"MCPClient not connected to {self.server}. Call connect() first."
            )

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------ #
    # Async context manager                                               #
    # ------------------------------------------------------------------ #

    async def __aenter__(self) -> "MCPClient":
        await self.connect()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()
