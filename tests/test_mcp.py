"""
Tests for piai MCP integration.

Tests MCPServer config, MCPClient, MCPHub, and agent() robustness.
Uses a lightweight in-process MCP server (via mcp.server.fastmcp.FastMCP)
so no external processes are needed.
"""

from __future__ import annotations

import asyncio
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from piai.mcp.server import MCPServer, _basename
from piai.mcp.client import MCPClient
from piai.mcp.hub import MCPHub, _safe_name
from piai.types import Tool


# ------------------------------------------------------------------ #
# MCPServer tests                                                     #
# ------------------------------------------------------------------ #

class TestMCPServer:

    def test_stdio_simple(self):
        s = MCPServer.stdio("r2pm -r r2mcp")
        assert s.transport == "stdio"
        assert s.command == "r2pm"
        assert s.args == ["-r", "r2mcp"]
        assert s.name == "r2pm"
        assert s.env is None

    def test_stdio_path_with_spaces(self):
        s = MCPServer.stdio('"/path with spaces/server" --flag value')
        assert s.command == "/path with spaces/server"
        assert s.args == ["--flag", "value"]
        assert s.name == "server"

    def test_stdio_quoted_args(self):
        s = MCPServer.stdio('npx @mcp/server "/my path"')
        assert s.command == "npx"
        assert s.args == ["@mcp/server", "/my path"]

    def test_stdio_custom_name(self):
        s = MCPServer.stdio("ida-mcp", name="ida")
        assert s.name == "ida"

    def test_stdio_env_extra(self):
        s = MCPServer.stdio("my-server", env_extra={"API_KEY": "secret"})
        assert s.env is not None
        assert s.env["API_KEY"] == "secret"
        # Should inherit PATH from parent
        assert "PATH" in s.env

    def test_stdio_env_replace(self):
        s = MCPServer.stdio("my-server", env={"ONLY": "this"})
        assert s.env == {"ONLY": "this"}
        # Should NOT have PATH (full replacement)
        assert "PATH" not in s.env

    def test_stdio_env_extra_overrides_base(self):
        s = MCPServer.stdio("my-server", env={"BASE": "val"}, env_extra={"BASE": "overridden"})
        assert s.env["BASE"] == "overridden"

    def test_stdio_empty_command_raises(self):
        with pytest.raises(ValueError, match="Empty command"):
            MCPServer.stdio("")

    def test_http_basic(self):
        s = MCPServer.http("http://127.0.0.1:13337/mcp")
        assert s.transport == "http"
        assert s.url == "http://127.0.0.1:13337/mcp"
        assert s.name == "127.0.0.1"

    def test_http_custom_name(self):
        s = MCPServer.http("http://127.0.0.1:13337/mcp", name="ida")
        assert s.name == "ida"

    def test_http_bearer_token(self):
        s = MCPServer.http("http://example.com/mcp", bearer_token="mytoken")
        assert s.headers["Authorization"] == "Bearer mytoken"

    def test_http_headers_and_bearer(self):
        s = MCPServer.http(
            "http://example.com/mcp",
            headers={"X-Custom": "val"},
            bearer_token="tok",
        )
        assert s.headers["X-Custom"] == "val"
        assert s.headers["Authorization"] == "Bearer tok"

    def test_sse_basic(self):
        s = MCPServer.sse("http://localhost:9000/sse")
        assert s.transport == "sse"
        assert s.url == "http://localhost:9000/sse"
        assert s.name == "localhost"

    def test_sse_bearer_token(self):
        s = MCPServer.sse("http://example.com/sse", bearer_token="tok")
        assert s.headers["Authorization"] == "Bearer tok"

    def test_from_config_stdio_simple(self):
        s = MCPServer.from_config({"command": "ida-mcp"})
        assert s.transport == "stdio"
        assert s.command == "ida-mcp"
        assert s.args == []

    def test_from_config_stdio_with_args(self):
        s = MCPServer.from_config({
            "command": "r2pm",
            "args": ["-r", "r2mcp"],
        })
        assert s.command == "r2pm"
        assert s.args == ["-r", "r2mcp"]

    def test_from_config_stdio_with_env(self):
        s = MCPServer.from_config({
            "command": "my-server",
            "env": {"KEY": "val"},
        })
        assert s.env == {"KEY": "val"}

    def test_from_config_http(self):
        s = MCPServer.from_config({"url": "http://127.0.0.1:13337/mcp"})
        assert s.transport == "http"
        assert s.url == "http://127.0.0.1:13337/mcp"

    def test_from_config_sse_auto_detect(self):
        s = MCPServer.from_config({"url": "http://localhost:9000/sse"})
        assert s.transport == "sse"

    def test_from_config_explicit_transport(self):
        s = MCPServer.from_config({
            "url": "http://localhost:9000/mcp",
            "transport": "sse",  # override auto-detection
        })
        assert s.transport == "sse"

    def test_from_config_with_name(self):
        s = MCPServer.from_config({"command": "ida-mcp", "name": "ida"})
        assert s.name == "ida"

    def test_from_config_bearer_token(self):
        s = MCPServer.from_config({
            "url": "http://example.com/mcp",
            "bearer_token": "secret",
        })
        assert s.headers["Authorization"] == "Bearer secret"

    def test_from_config_no_command_or_url_raises(self):
        with pytest.raises(ValueError, match="Cannot determine transport"):
            MCPServer.from_config({"name": "broken"})

    def test_from_config_stdio_no_command_raises(self):
        with pytest.raises(ValueError, match="must have 'command'"):
            MCPServer.from_config({"transport": "stdio"})

    def test_from_config_http_no_url_raises(self):
        with pytest.raises(ValueError, match="must have 'url'"):
            MCPServer.from_config({"transport": "http"})

    def test_from_config_unknown_transport_raises(self):
        with pytest.raises(ValueError, match="Unknown transport"):
            MCPServer.from_config({"transport": "websocket", "url": "ws://x"})

    def test_repr_stdio(self):
        s = MCPServer.stdio("r2pm -r r2mcp")
        assert "stdio" in repr(s)
        assert "r2pm" in repr(s)

    def test_repr_http(self):
        s = MCPServer.http("http://localhost/mcp")
        assert "http" in repr(s)
        assert "localhost" in repr(s)

    def test_basename(self):
        assert _basename("r2pm") == "r2pm"
        assert _basename("/usr/local/bin/ida-mcp") == "ida-mcp"
        assert _basename("C:\\Programs\\server.exe") == "server.exe"

    def test_from_toml_stdio(self, tmp_path):
        toml_content = b"""
[mcp_servers.r2]
command = "r2pm"
args = ["-r", "r2mcp"]

[mcp_servers.ida]
command = "ida-mcp"
"""
        config_file = tmp_path / "config.toml"
        config_file.write_bytes(toml_content)

        servers = MCPServer.from_toml(str(config_file))
        assert len(servers) == 2
        names = {s.name for s in servers}
        assert names == {"r2", "ida"}

        r2 = next(s for s in servers if s.name == "r2")
        assert r2.transport == "stdio"
        assert r2.command == "r2pm"
        assert r2.args == ["-r", "r2mcp"]

    def test_from_toml_http(self, tmp_path):
        toml_content = b"""
[mcp_servers.ida-pro]
url = "http://127.0.0.1:13337/mcp"

[mcp_servers.remote]
url = "https://api.example.com/mcp"
bearer_token = "my-token"
"""
        config_file = tmp_path / "config.toml"
        config_file.write_bytes(toml_content)

        servers = MCPServer.from_toml(str(config_file))
        assert len(servers) == 2

        remote = next(s for s in servers if s.name == "remote")
        assert remote.transport == "http"
        assert remote.headers["Authorization"] == "Bearer my-token"

    def test_from_toml_mixed(self, tmp_path):
        toml_content = b"""
[mcp_servers.r2]
command = "r2pm"
args = ["-r", "r2mcp"]

[mcp_servers.ida-http]
url = "http://127.0.0.1:13337/mcp"

[mcp_servers.legacy]
url = "http://localhost:9000/sse"
"""
        config_file = tmp_path / "config.toml"
        config_file.write_bytes(toml_content)

        servers = MCPServer.from_toml(str(config_file))
        assert len(servers) == 3
        transports = {s.name: s.transport for s in servers}
        assert transports["r2"] == "stdio"
        assert transports["ida-http"] == "http"
        assert transports["legacy"] == "sse"

    def test_from_toml_env_extra(self, tmp_path):
        toml_content = b"""
[mcp_servers.my-server]
command = "my-server"

[mcp_servers.my-server.env_extra]
API_KEY = "secret"
"""
        config_file = tmp_path / "config.toml"
        config_file.write_bytes(toml_content)

        servers = MCPServer.from_toml(str(config_file))
        assert len(servers) == 1
        assert servers[0].env is not None
        assert servers[0].env["API_KEY"] == "secret"

    def test_from_toml_custom_section(self, tmp_path):
        toml_content = b"""
[tools.r2]
command = "r2pm"
args = ["-r", "r2mcp"]
"""
        config_file = tmp_path / "config.toml"
        config_file.write_bytes(toml_content)

        servers = MCPServer.from_toml(str(config_file), section="tools")
        assert len(servers) == 1
        assert servers[0].name == "r2"

    def test_from_toml_empty_section_returns_empty(self, tmp_path):
        toml_content = b"""
[other_section]
key = "value"
"""
        config_file = tmp_path / "config.toml"
        config_file.write_bytes(toml_content)

        servers = MCPServer.from_toml(str(config_file))
        assert servers == []

    def test_from_toml_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            MCPServer.from_toml("/nonexistent/path/config.toml")

    def test_from_toml_name_from_key(self, tmp_path):
        toml_content = b"""
[mcp_servers.my-custom-name]
command = "server"
"""
        config_file = tmp_path / "config.toml"
        config_file.write_bytes(toml_content)

        servers = MCPServer.from_toml(str(config_file))
        # Name comes from TOML key, not command basename
        assert servers[0].name == "my-custom-name"


# ------------------------------------------------------------------ #
# MCPHub unit tests (mocked clients)                                  #
# ------------------------------------------------------------------ #

def _make_mock_client(server: MCPServer, tools: list[Tool], fail_connect: bool = False) -> MCPClient:
    """Create a mock MCPClient that returns given tools."""
    client = MagicMock(spec=MCPClient)
    client.server = server
    client.is_connected = not fail_connect

    if fail_connect:
        client.connect = AsyncMock(side_effect=RuntimeError("Connection refused"))
    else:
        client.connect = AsyncMock()

    client.list_tools = AsyncMock(return_value=tools)
    client.call_tool = AsyncMock(return_value="tool result")
    client.close = AsyncMock()
    return client


class TestMCPHub:


    async def test_single_server_tools(self):
        tools = [Tool(name="open_file", description="Open file", parameters={})]
        server = MCPServer.stdio("r2pm -r r2mcp")

        hub = MCPHub([server])
        with patch("piai.mcp.hub.MCPClient", return_value=_make_mock_client(server, tools)):
            async with hub:
                assert hub.tool_names() == ["open_file"]


    async def test_multi_server_merged_tools(self):
        server1 = MCPServer.stdio("r2pm -r r2mcp", name="r2")
        server2 = MCPServer.stdio("ida-mcp", name="ida")
        tools1 = [Tool(name="open_file", description="r2 open", parameters={})]
        tools2 = [Tool(name="decompile", description="ida decompile", parameters={})]

        hub = MCPHub([server1, server2])
        clients = [_make_mock_client(server1, tools1), _make_mock_client(server2, tools2)]
        with patch("piai.mcp.hub.MCPClient", side_effect=clients):
            async with hub:
                names = hub.tool_names()
                assert "open_file" in names
                assert "decompile" in names


    async def test_tool_collision_namespacing(self):
        server1 = MCPServer.stdio("server1", name="s1")
        server2 = MCPServer.stdio("server2", name="s2")
        tools1 = [Tool(name="read_file", description="s1 read", parameters={})]
        tools2 = [Tool(name="read_file", description="s2 read", parameters={})]

        hub = MCPHub([server1, server2])
        clients = [_make_mock_client(server1, tools1), _make_mock_client(server2, tools2)]
        with patch("piai.mcp.hub.MCPClient", side_effect=clients):
            async with hub:
                names = hub.tool_names()
                # Both tools should be namespaced on collision
                assert "s1__read_file" in names
                assert "s2__read_file" in names
                # Unnamespaced name is also kept for backward compat (first-wins)
                # but both namespaced versions are registered in the registry
                assert len([n for n in names if "read_file" in n]) >= 2


    async def test_partial_connect_failure_allowed(self):
        server1 = MCPServer.stdio("good-server", name="good")
        server2 = MCPServer.stdio("bad-server", name="bad")
        tools1 = [Tool(name="tool_a", description="", parameters={})]

        hub = MCPHub([server1, server2], require_all=False)
        clients = [
            _make_mock_client(server1, tools1, fail_connect=False),
            _make_mock_client(server2, [], fail_connect=True),
        ]
        with patch("piai.mcp.hub.MCPClient", side_effect=clients):
            async with hub:
                # Should still have tools from the good server
                assert "tool_a" in hub.tool_names()


    async def test_require_all_raises_on_failure(self):
        server1 = MCPServer.stdio("good-server", name="good")
        server2 = MCPServer.stdio("bad-server", name="bad")
        tools1 = [Tool(name="tool_a", description="", parameters={})]

        hub = MCPHub([server1, server2], require_all=True)
        clients = [
            _make_mock_client(server1, tools1, fail_connect=False),
            _make_mock_client(server2, [], fail_connect=True),
        ]
        with patch("piai.mcp.hub.MCPClient", side_effect=clients):
            with pytest.raises(RuntimeError, match="Failed to connect to required MCP server"):
                await hub.connect()


    async def test_call_tool_routing(self):
        server1 = MCPServer.stdio("r2pm -r r2mcp", name="r2")
        server2 = MCPServer.stdio("ida-mcp", name="ida")
        tools1 = [Tool(name="open_file", description="", parameters={})]
        tools2 = [Tool(name="decompile", description="", parameters={})]

        mock1 = _make_mock_client(server1, tools1)
        mock2 = _make_mock_client(server2, tools2)

        hub = MCPHub([server1, server2])
        with patch("piai.mcp.hub.MCPClient", side_effect=[mock1, mock2]):
            async with hub:
                await hub.call_tool("open_file", {"path": "/lib.so"})
                mock1.call_tool.assert_called_once_with("open_file", {"path": "/lib.so"})
                mock2.call_tool.assert_not_called()

                await hub.call_tool("decompile", {"address": "0x1000"})
                mock2.call_tool.assert_called_once_with("decompile", {"address": "0x1000"})


    async def test_call_tool_not_found_raises(self):
        server = MCPServer.stdio("r2pm -r r2mcp", name="r2")
        tools = [Tool(name="open_file", description="", parameters={})]

        hub = MCPHub([server])
        with patch("piai.mcp.hub.MCPClient", return_value=_make_mock_client(server, tools)):
            async with hub:
                with pytest.raises(KeyError, match="nonexistent"):
                    await hub.call_tool("nonexistent", {})


    async def test_empty_servers(self):
        hub = MCPHub([])
        async with hub:
            assert hub.all_tools() == []


    async def test_connected_servers_property(self):
        server1 = MCPServer.stdio("good-server", name="good")
        server2 = MCPServer.stdio("bad-server", name="bad")

        hub = MCPHub([server1, server2], require_all=False)
        clients = [
            _make_mock_client(server1, [], fail_connect=False),
            _make_mock_client(server2, [], fail_connect=True),
        ]
        with patch("piai.mcp.hub.MCPClient", side_effect=clients):
            async with hub:
                connected = hub.connected_servers
                assert len(connected) == 1
                assert connected[0].name == "good"


# ------------------------------------------------------------------ #
# MCPClient unit tests                                                #
# ------------------------------------------------------------------ #

class TestMCPClient:

    def test_not_connected_initially(self):
        client = MCPClient(MCPServer.stdio("r2pm -r r2mcp"))
        assert not client.is_connected

    async def test_ensure_connected_raises_before_connect(self):
        client = MCPClient(MCPServer.stdio("r2pm -r r2mcp"))
        with pytest.raises(RuntimeError, match="not connected"):
            await client.list_tools()


    async def test_connect_timeout(self):
        # Simulate a server that never responds
        server = MCPServer.stdio("sleep 1000")

        async def slow_connect():
            await asyncio.sleep(999)

        client = MCPClient(server, connect_timeout=0.05)
        with patch.object(client, "_connect_inner", side_effect=slow_connect):
            with pytest.raises(TimeoutError, match="did not respond"):
                await client.connect()


    async def test_tool_result_truncation(self):
        server = MCPServer.stdio("r2pm -r r2mcp")
        client = MCPClient(server, tool_result_max_chars=100)

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.isError = False
        mock_block = MagicMock()
        mock_block.text = "A" * 200
        mock_result.content = [mock_block]
        mock_session.call_tool = AsyncMock(return_value=mock_result)

        client._session = mock_session
        client._connected = True

        result = await client.call_tool("test", {})
        assert len(result) <= 150  # 100 + truncation message
        assert "truncated" in result


    async def test_tool_error_returned_as_string(self):
        server = MCPServer.stdio("r2pm -r r2mcp")
        client = MCPClient(server)

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.isError = True
        mock_block = MagicMock()
        mock_block.text = "Function not found"
        mock_result.content = [mock_block]
        mock_session.call_tool = AsyncMock(return_value=mock_result)

        client._session = mock_session
        client._connected = True

        # Should NOT raise — returns error as string
        result = await client.call_tool("disasm", {})
        assert "Tool error" in result
        assert "Function not found" in result


    async def test_binary_content_summarized(self):
        server = MCPServer.stdio("r2pm -r r2mcp")
        client = MCPClient(server)

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.isError = False
        mock_block = MagicMock(spec=["data"])  # no .text, has .data
        mock_block.data = b"\x00" * 1024
        mock_result.content = [mock_block]
        mock_session.call_tool = AsyncMock(return_value=mock_result)

        client._session = mock_session
        client._connected = True

        result = await client.call_tool("get_bytes", {})
        assert "binary data" in result
        assert "1024 bytes" in result


# ------------------------------------------------------------------ #
# Utility tests                                                       #
# ------------------------------------------------------------------ #

class TestUtils:

    def test_safe_name(self):
        assert _safe_name("my-server") == "my_server"
        assert _safe_name("my.server") == "my_server"
        assert _safe_name("my server") == "my_server"
        assert _safe_name("clean") == "clean"

    def test_basename(self):
        assert _basename("/usr/local/bin/ida-mcp") == "ida-mcp"
        assert _basename("simple") == "simple"
        assert _basename("C:\\tools\\server.exe") == "server.exe"


# ------------------------------------------------------------------ #
# agent() tests (mocked stream + hub)                                 #
# ------------------------------------------------------------------ #

class TestAgent:


    async def test_agent_no_mcp_servers(self):
        from piai import agent
        from piai.types import Context, UserMessage, AssistantMessage, TextContent

        ctx = Context(messages=[UserMessage(content="Hello")])

        final = AssistantMessage(content=[TextContent(text="Hello back")])
        from piai.types import DoneEvent
        done = DoneEvent(reason="stop", message=final)

        async def mock_stream(*args, **kwargs):
            from piai.types import TextDeltaEvent
            yield TextDeltaEvent(text="Hello back")
            yield done

        with patch("piai.agent.stream", side_effect=mock_stream):
            result = await agent(
                model_id="gpt-5.1-codex-mini",
                context=ctx,
                mcp_servers=None,
            )
        assert result == final


    async def test_agent_async_on_event(self):
        from piai import agent
        from piai.types import Context, UserMessage, AssistantMessage, TextContent, TextDeltaEvent

        ctx = Context(messages=[UserMessage(content="Hi")])
        final = AssistantMessage(content=[TextContent(text="Hi")])
        from piai.types import DoneEvent
        done = DoneEvent(reason="stop", message=final)

        events_received = []

        async def async_callback(event):
            events_received.append(event)

        async def mock_stream(*args, **kwargs):
            yield TextDeltaEvent(text="Hi")
            yield done

        with patch("piai.agent.stream", side_effect=mock_stream):
            await agent(
                model_id="gpt-5.1-codex-mini",
                context=ctx,
                on_event=async_callback,
            )

        assert len(events_received) == 2


    async def test_agent_tool_call_loop(self):
        from piai import agent
        from piai.types import (
            Context, UserMessage, AssistantMessage, TextContent,
            TextDeltaEvent, DoneEvent, ToolCall, ToolCallEndEvent,
            ToolCallContent,
        )

        ctx = Context(messages=[UserMessage(content="analyze")])

        tool_call = ToolCall(id="tc_123", name="open_file", input={"path": "/lib.so"})
        assistant_with_tool = AssistantMessage(
            content=[ToolCallContent(tool_calls=[tool_call])]
        )
        final = AssistantMessage(content=[TextContent(text="Done")])

        call_count = 0

        async def mock_stream(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First turn: emit a tool call
                yield ToolCallEndEvent(tool_call=tool_call)
                yield DoneEvent(reason="tool_use", message=assistant_with_tool)
            else:
                # Second turn: emit final text
                yield TextDeltaEvent(text="Done")
                yield DoneEvent(reason="stop", message=final)

        mock_hub = AsyncMock(spec=MCPHub)
        mock_hub.__aenter__ = AsyncMock(return_value=mock_hub)
        mock_hub.__aexit__ = AsyncMock(return_value=False)
        mock_hub.all_tools = MagicMock(return_value=[])
        mock_hub.call_tool = AsyncMock(return_value="file opened successfully")

        with patch("piai.agent.stream", side_effect=mock_stream):
            with patch("piai.agent.MCPHub", return_value=mock_hub):
                result = await agent(
                    model_id="gpt-5.1-codex-mini",
                    context=ctx,
                    mcp_servers=[MCPServer.stdio("r2pm -r r2mcp")],
                )

        assert call_count == 2
        mock_hub.call_tool.assert_called_once_with("open_file", {"path": "/lib.so"})
        assert result == final


    async def test_agent_tool_error_doesnt_crash(self):
        from piai import agent
        from piai.types import (
            Context, UserMessage, AssistantMessage, TextContent,
            DoneEvent, ToolCall, ToolCallEndEvent, ToolCallContent, TextDeltaEvent,
        )

        ctx = Context(messages=[UserMessage(content="analyze")])
        tool_call = ToolCall(id="tc_err", name="bad_tool", input={})
        assistant_with_tool = AssistantMessage(content=[ToolCallContent(tool_calls=[tool_call])])
        final = AssistantMessage(content=[TextContent(text="Handled error")])

        call_count = 0

        async def mock_stream(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                yield ToolCallEndEvent(tool_call=tool_call)
                yield DoneEvent(reason="tool_use", message=assistant_with_tool)
            else:
                yield TextDeltaEvent(text="Handled error")
                yield DoneEvent(reason="stop", message=final)

        mock_hub = AsyncMock(spec=MCPHub)
        mock_hub.__aenter__ = AsyncMock(return_value=mock_hub)
        mock_hub.__aexit__ = AsyncMock(return_value=False)
        mock_hub.all_tools = MagicMock(return_value=[])
        # Tool call raises — agent should NOT crash
        mock_hub.call_tool = AsyncMock(side_effect=KeyError("bad_tool not found"))

        with patch("piai.agent.stream", side_effect=mock_stream):
            with patch("piai.agent.MCPHub", return_value=mock_hub):
                result = await agent(
                    model_id="gpt-5.1-codex-mini",
                    context=ctx,
                    mcp_servers=[MCPServer.stdio("r2pm -r r2mcp")],
                )

        # Should complete without raising
        assert result == final


    async def test_agent_max_turns_returns_last_message(self):
        from piai import agent
        from piai.types import (
            Context, UserMessage, AssistantMessage,
            DoneEvent, ToolCall, ToolCallEndEvent, ToolCallContent,
        )

        ctx = Context(messages=[UserMessage(content="analyze")])
        tool_call = ToolCall(id="tc_loop", name="open_file", input={})
        always_tool_msg = AssistantMessage(content=[ToolCallContent(tool_calls=[tool_call])])

        async def mock_stream(*args, **kwargs):
            yield ToolCallEndEvent(tool_call=tool_call)
            yield DoneEvent(reason="tool_use", message=always_tool_msg)

        mock_hub = AsyncMock(spec=MCPHub)
        mock_hub.__aenter__ = AsyncMock(return_value=mock_hub)
        mock_hub.__aexit__ = AsyncMock(return_value=False)
        mock_hub.all_tools = MagicMock(return_value=[])
        mock_hub.call_tool = AsyncMock(return_value="ok")

        with patch("piai.agent.stream", side_effect=mock_stream):
            with patch("piai.agent.MCPHub", return_value=mock_hub):
                result = await agent(
                    model_id="gpt-5.1-codex-mini",
                    context=ctx,
                    mcp_servers=[MCPServer.stdio("r2pm -r r2mcp")],
                    max_turns=3,
                )

        # Should return last message, not raise
        assert result == always_tool_msg
        assert mock_hub.call_tool.call_count == 3
