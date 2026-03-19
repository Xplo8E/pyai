"""
Tests for LangGraph integration:
  - piai.mcp.to_langchain_tools  (MCP → LangChain tool bridge)
  - piai.mcp.MCPHubToolset       (context manager version)
  - piai.langchain.SubAgentTool  (piai agent as LangChain BaseTool)
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.tools import BaseTool

from piai.langchain import SubAgentTool
from piai.mcp import MCPHubToolset, MCPServer, to_langchain_tools
from piai.mcp.langchain_tools import MCPLangChainTool, _make_input_schema
from piai.types import AssistantMessage, TextContent, Tool


# ─── Fixtures ──────────────────────────────────────────────────────────────────

def make_piai_tool(name="read_file", description="Read a file", props=None, required=None):
    return Tool(
        name=name,
        description=description,
        parameters={
            "type": "object",
            "properties": props or {
                "path": {"type": "string", "description": "File path"},
                "encoding": {"type": "string", "description": "File encoding"},
            },
            "required": required or ["path"],
        },
    )


def make_mock_hub(tools=None):
    hub = MagicMock()
    hub.all_tools.return_value = tools or [make_piai_tool()]
    hub.connect = AsyncMock()
    hub.close = AsyncMock()
    hub.call_tool = AsyncMock(return_value="tool result")
    return hub


# ─── _make_input_schema ────────────────────────────────────────────────────────

def test_make_input_schema_required_field():
    tool = make_piai_tool(
        props={"path": {"type": "string", "description": "File path"}},
        required=["path"],
    )
    schema = _make_input_schema(tool)
    fields = schema.model_fields
    assert "path" in fields
    # Required field has no default
    assert fields["path"].default is None or fields["path"].is_required()


def test_make_input_schema_optional_field():
    tool = make_piai_tool(
        props={"encoding": {"type": "string", "description": "Encoding"}},
        required=[],
    )
    schema = _make_input_schema(tool)
    fields = schema.model_fields
    assert "encoding" in fields
    assert not fields["encoding"].is_required()


def test_make_input_schema_type_mapping():
    tool = make_piai_tool(
        props={
            "count": {"type": "integer", "description": "Count"},
            "ratio": {"type": "number", "description": "Ratio"},
            "flag": {"type": "boolean", "description": "Flag"},
            "items": {"type": "array", "description": "Items"},
            "data": {"type": "object", "description": "Data"},
        },
        required=["count"],
    )
    schema = _make_input_schema(tool)
    # Should not raise — all types mapped
    assert "count" in schema.model_fields
    assert "ratio" in schema.model_fields
    assert "flag" in schema.model_fields


def test_make_input_schema_empty_parameters():
    tool = Tool(name="noop", description="Does nothing", parameters={})
    schema = _make_input_schema(tool)
    assert schema.model_fields == {}


# ─── MCPLangChainTool ──────────────────────────────────────────────────────────

def test_mcp_langchain_tool_is_basetool():
    hub = make_mock_hub()
    tool = make_piai_tool()
    schema = _make_input_schema(tool)
    lc_tool = MCPLangChainTool(
        name=tool.name,
        description=tool.description,
        args_schema=schema,
        hub=hub,
        mcp_tool_name=tool.name,
    )
    assert isinstance(lc_tool, BaseTool)
    assert lc_tool.name == "read_file"
    assert lc_tool.description == "Read a file"


async def test_mcp_langchain_tool_arun_calls_hub():
    hub = make_mock_hub()
    hub.call_tool = AsyncMock(return_value="file contents here")
    tool = make_piai_tool()
    schema = _make_input_schema(tool)
    lc_tool = MCPLangChainTool(
        name=tool.name,
        description=tool.description,
        args_schema=schema,
        hub=hub,
        mcp_tool_name=tool.name,
    )
    result = await lc_tool._arun(path="/etc/hosts")
    hub.call_tool.assert_called_once_with("read_file", {"path": "/etc/hosts"})
    assert result == "file contents here"


async def test_mcp_langchain_tool_arun_filters_none():
    """Optional fields with None values should not be passed to hub.call_tool."""
    hub = make_mock_hub()
    hub.call_tool = AsyncMock(return_value="ok")
    tool = make_piai_tool()
    schema = _make_input_schema(tool)
    lc_tool = MCPLangChainTool(
        name=tool.name,
        description=tool.description,
        args_schema=schema,
        hub=hub,
        mcp_tool_name=tool.name,
    )
    await lc_tool._arun(path="/tmp/foo", encoding=None)
    # encoding=None should be filtered out
    hub.call_tool.assert_called_once_with("read_file", {"path": "/tmp/foo"})


# ─── to_langchain_tools ────────────────────────────────────────────────────────

async def test_to_langchain_tools_returns_tools_and_hub():
    piai_tools = [
        make_piai_tool("read_file", "Read a file"),
        make_piai_tool("write_file", "Write a file"),
    ]
    mock_hub = make_mock_hub(tools=piai_tools)

    with patch("piai.mcp.langchain_tools.MCPHub", return_value=mock_hub):
        tools, hub = await to_langchain_tools([MCPServer.stdio("fake-server")])

    assert len(tools) == 2
    assert all(isinstance(t, MCPLangChainTool) for t in tools)
    assert tools[0].name == "read_file"
    assert tools[1].name == "write_file"
    assert hub is mock_hub
    mock_hub.connect.assert_called_once()


async def test_to_langchain_tools_empty_tools():
    """When servers expose no tools, result should be empty list."""
    mock_hub = MagicMock()
    mock_hub.all_tools.return_value = []
    mock_hub.connect = AsyncMock()
    mock_hub.close = AsyncMock()
    with patch("piai.mcp.langchain_tools.MCPHub", return_value=mock_hub):
        tools, hub = await to_langchain_tools([MCPServer.stdio("empty-server")])
    assert tools == []
    assert hub is mock_hub


async def test_to_langchain_tools_passes_kwargs():
    mock_hub = make_mock_hub()
    with patch("piai.mcp.langchain_tools.MCPHub") as MockHub:
        MockHub.return_value = mock_hub
        await to_langchain_tools(
            [MCPServer.stdio("srv")],
            require_all=True,
            connect_timeout=30.0,
            tool_result_max_chars=8000,
        )
        MockHub.assert_called_once_with(
            [MCPServer.stdio("srv")],
            require_all=True,
            connect_timeout=30.0,
            tool_result_max_chars=8000,
        )


# ─── MCPHubToolset ────────────────────────────────────────────────────────────

async def test_mcp_hub_toolset_context_manager():
    piai_tools = [make_piai_tool("list_dir", "List directory")]
    mock_hub = make_mock_hub(tools=piai_tools)

    with patch("piai.mcp.langchain_tools.MCPHub", return_value=mock_hub):
        async with MCPHubToolset([MCPServer.stdio("fs-server")]) as tools:
            assert len(tools) == 1
            assert tools[0].name == "list_dir"

    mock_hub.connect.assert_called_once()
    mock_hub.close.assert_called_once()


async def test_mcp_hub_toolset_closes_on_exception():
    mock_hub = make_mock_hub()
    with patch("piai.mcp.langchain_tools.MCPHub", return_value=mock_hub):
        with pytest.raises(RuntimeError):
            async with MCPHubToolset([MCPServer.stdio("srv")]):
                raise RuntimeError("something broke")
    mock_hub.close.assert_called_once()


# ─── SubAgentTool ─────────────────────────────────────────────────────────────

def test_sub_agent_tool_is_basetool():
    tool = SubAgentTool(
        name="analyzer",
        description="Analyzes code",
        model_id="gpt-5.1-codex-mini",
        system_prompt="You are a code analyst.",
    )
    assert isinstance(tool, BaseTool)
    assert tool.name == "analyzer"
    assert tool.model_id == "gpt-5.1-codex-mini"


def test_sub_agent_tool_defaults():
    tool = SubAgentTool(name="my_agent", description="Does stuff")
    assert tool.model_id == "gpt-5.1-codex-mini"
    assert tool.max_turns == 20
    assert tool.mcp_servers == []
    assert tool.options == {}
    assert tool.system_prompt is None


def test_sub_agent_tool_args_schema():
    tool = SubAgentTool(name="agent", description="test")
    schema = tool.args_schema
    assert "task" in schema.model_fields


async def test_sub_agent_tool_arun_calls_piai_agent():
    """_arun should call piai agent() with correct params and return text."""
    final_message = AssistantMessage(
        content=[TextContent(text="Analysis complete: found 3 issues.")],
        stop_reason="stop",
    )

    with patch("piai.langchain.sub_agent_tool.piai_agent", new_callable=AsyncMock) as mock_agent:
        mock_agent.return_value = final_message

        tool = SubAgentTool(
            name="analyzer",
            description="Analyzes code",
            model_id="gpt-5.1",
            system_prompt="You are a security expert.",
            max_turns=15,
            options={"reasoning_effort": "high"},
        )

        result = await tool._arun(task="Find SQL injection vulnerabilities in login.py")

    assert result == "Analysis complete: found 3 issues."
    mock_agent.assert_called_once()
    call_kwargs = mock_agent.call_args
    assert call_kwargs.kwargs["model_id"] == "gpt-5.1"
    assert call_kwargs.kwargs["max_turns"] == 15
    assert call_kwargs.kwargs["options"] == {"reasoning_effort": "high"}


async def test_sub_agent_tool_arun_with_mcp_servers():
    """MCP servers should be passed through to piai agent()."""
    servers = [MCPServer.stdio("r2pm -r r2mcp")]
    final_message = AssistantMessage(
        content=[TextContent(text="Binary analysis done.")],
        stop_reason="stop",
    )

    with patch("piai.langchain.sub_agent_tool.piai_agent", new_callable=AsyncMock) as mock_agent:
        mock_agent.return_value = final_message

        tool = SubAgentTool(
            name="re_agent",
            description="Reverse engineers binaries",
            mcp_servers=servers,
        )
        result = await tool._arun(task="Analyze /lib/target.so")

    assert result == "Binary analysis done."
    call_kwargs = mock_agent.call_args.kwargs
    assert call_kwargs["mcp_servers"] == servers


async def test_sub_agent_tool_arun_no_text_output():
    """When agent produces no text, return fallback string."""
    final_message = AssistantMessage(content=[], stop_reason="stop")

    with patch("piai.langchain.sub_agent_tool.piai_agent", new_callable=AsyncMock) as mock_agent:
        mock_agent.return_value = final_message
        tool = SubAgentTool(name="agent", description="test")
        result = await tool._arun(task="Do something")

    assert result == "(agent produced no text output)"


async def test_sub_agent_tool_arun_multiple_text_blocks():
    """Multiple TextContent blocks should be joined with double newline."""
    final_message = AssistantMessage(
        content=[
            TextContent(text="First finding."),
            TextContent(text="Second finding."),
        ],
        stop_reason="stop",
    )

    with patch("piai.langchain.sub_agent_tool.piai_agent", new_callable=AsyncMock) as mock_agent:
        mock_agent.return_value = final_message
        tool = SubAgentTool(name="agent", description="test")
        result = await tool._arun(task="Analyze")

    assert result == "First finding.\n\nSecond finding."


async def test_sub_agent_tool_no_mcp_passes_none():
    """When no MCP servers, agent() should receive mcp_servers=None."""
    final_message = AssistantMessage(
        content=[TextContent(text="Done.")],
        stop_reason="stop",
    )

    with patch("piai.langchain.sub_agent_tool.piai_agent", new_callable=AsyncMock) as mock_agent:
        mock_agent.return_value = final_message
        tool = SubAgentTool(name="agent", description="test")
        await tool._arun(task="Do it")

    call_kwargs = mock_agent.call_args.kwargs
    assert call_kwargs["mcp_servers"] is None


async def test_sub_agent_tool_no_options_passes_none():
    """When no options set, agent() should receive options=None."""
    final_message = AssistantMessage(
        content=[TextContent(text="Done.")],
        stop_reason="stop",
    )

    with patch("piai.langchain.sub_agent_tool.piai_agent", new_callable=AsyncMock) as mock_agent:
        mock_agent.return_value = final_message
        tool = SubAgentTool(name="agent", description="test")
        await tool._arun(task="Do it")

    call_kwargs = mock_agent.call_args.kwargs
    assert call_kwargs["options"] is None
