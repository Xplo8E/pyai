"""
Tests for thinking/reasoning observability features:
  - AssistantMessage.thinking property
  - AssistantMessage.text property
  - ThinkingStartEvent / ThinkingEndEvent emitted by SSE processor
  - AgentToolCallEvent / AgentToolResultEvent / AgentTurnEndEvent in agent loop
  - PiAIChatModel surfaces thinking in additional_kwargs
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from piai.types import (
    AgentToolCallEvent,
    AgentToolResultEvent,
    AgentTurnEndEvent,
    AssistantMessage,
    TextContent,
    ThinkingContent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ThinkingStartEvent,
    ToolCall,
    ToolCallContent,
    ToolCallEndEvent,
    DoneEvent,
    TextDeltaEvent,
    ErrorEvent,
    Context,
    UserMessage,
)


# ─── AssistantMessage properties ──────────────────────────────────────────────

class TestAssistantMessageProperties:
    def test_text_property_concatenates_text_blocks(self):
        msg = AssistantMessage(content=[
            TextContent(text="Hello "),
            TextContent(text="world"),
        ])
        assert msg.text == "Hello world"

    def test_text_property_skips_non_text_blocks(self):
        msg = AssistantMessage(content=[
            ThinkingContent(thinking="I should say hello"),
            TextContent(text="Hello"),
        ])
        assert msg.text == "Hello"

    def test_text_property_empty(self):
        assert AssistantMessage().text == ""

    def test_thinking_property_single_block(self):
        msg = AssistantMessage(content=[
            ThinkingContent(thinking="Step 1: think"),
            TextContent(text="Answer"),
        ])
        assert msg.thinking == "Step 1: think"

    def test_thinking_property_multiple_blocks(self):
        msg = AssistantMessage(content=[
            ThinkingContent(thinking="Part A"),
            ThinkingContent(thinking="Part B"),
        ])
        assert msg.thinking == "Part A\n\nPart B"

    def test_thinking_property_none_when_no_thinking(self):
        msg = AssistantMessage(content=[TextContent(text="Hello")])
        assert msg.thinking is None

    def test_thinking_property_none_on_empty_message(self):
        assert AssistantMessage().thinking is None

    def test_thinking_distinguishes_none_from_empty_string(self):
        """None means no thinking block; never returns empty string."""
        msg_no_thinking = AssistantMessage(content=[TextContent(text="Hi")])
        assert msg_no_thinking.thinking is None  # not ""


# ─── New stream event types ────────────────────────────────────────────────────

class TestNewStreamEventTypes:
    def test_thinking_start_event_defaults(self):
        e = ThinkingStartEvent()
        assert e.type == "thinking_start"

    def test_thinking_end_event_carries_full_text(self):
        e = ThinkingEndEvent(thinking="The answer is 42")
        assert e.type == "thinking_end"
        assert e.thinking == "The answer is 42"

    def test_agent_tool_call_event(self):
        e = AgentToolCallEvent(turn=1, tool_name="read_file", tool_input={"path": "/tmp/foo"})
        assert e.type == "agent_tool_call"
        assert e.turn == 1
        assert e.tool_name == "read_file"
        assert e.tool_input == {"path": "/tmp/foo"}

    def test_agent_tool_result_event(self):
        e = AgentToolResultEvent(turn=1, tool_name="read_file", tool_input={"path": "/tmp/foo"},
                                  result="file contents", error=False)
        assert e.type == "agent_tool_result"
        assert e.result == "file contents"
        assert e.error is False

    def test_agent_tool_result_event_error_flag(self):
        e = AgentToolResultEvent(turn=1, tool_name="bad_tool", result="Tool not found: bad_tool", error=True)
        assert e.error is True

    def test_agent_turn_end_event(self):
        tc = ToolCall(id="tc1", name="my_tool", input={})
        e = AgentTurnEndEvent(turn=2, thinking="thought about it", tool_calls=[tc])
        assert e.type == "agent_turn_end"
        assert e.turn == 2
        assert e.thinking == "thought about it"
        assert len(e.tool_calls) == 1

    def test_agent_turn_end_event_no_thinking(self):
        e = AgentTurnEndEvent(turn=1, thinking=None, tool_calls=[])
        assert e.thinking is None


# ─── Agent loop fires observability events ────────────────────────────────────

class TestAgentObservabilityEvents:

    @pytest.mark.asyncio
    async def test_agent_fires_turn_end_on_simple_response(self):
        """AgentTurnEndEvent fired even when no tool calls made."""
        from piai import agent

        final = AssistantMessage(content=[TextContent(text="Hello")])
        done = DoneEvent(reason="stop", message=final)

        events = []

        async def mock_stream(*a, **kw):
            yield TextDeltaEvent(text="Hello")
            yield done

        with patch("piai.agent.stream", side_effect=mock_stream):
            await agent("gpt-5.1-codex-mini", Context(messages=[UserMessage(content="Hi")]),
                        on_event=lambda e: events.append(e))

        turn_end_events = [e for e in events if isinstance(e, AgentTurnEndEvent)]
        assert len(turn_end_events) == 1
        assert turn_end_events[0].turn == 1
        assert turn_end_events[0].tool_calls == []

    @pytest.mark.asyncio
    async def test_agent_fires_tool_call_and_result_events(self):
        """AgentToolCallEvent + AgentToolResultEvent fired around each tool execution."""
        from piai import agent

        tc = ToolCall(id="call_1", name="list_files", input={"path": "/tmp"})
        turn1_msg = AssistantMessage(content=[ToolCallContent(tool_calls=[tc])])
        turn2_msg = AssistantMessage(content=[TextContent(text="Done")])

        call_count = 0

        async def mock_stream(*a, **kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                yield ToolCallEndEvent(tool_call=tc)
                yield DoneEvent(reason="tool_calls", message=turn1_msg)
            else:
                yield TextDeltaEvent(text="Done")
                yield DoneEvent(reason="stop", message=turn2_msg)

        mock_hub = MagicMock()
        mock_hub.all_tools.return_value = []
        mock_hub.call_tool = AsyncMock(return_value="file1.txt\nfile2.txt")
        mock_hub.__aenter__ = AsyncMock(return_value=mock_hub)
        mock_hub.__aexit__ = AsyncMock(return_value=False)

        events = []

        with patch("piai.agent.stream", side_effect=mock_stream), \
             patch("piai.agent.MCPHub", return_value=mock_hub):
            from piai.mcp import MCPServer
            await agent(
                "gpt-5.1-codex-mini",
                Context(messages=[UserMessage(content="List files")]),
                mcp_servers=[MCPServer.stdio("dummy")],
                on_event=lambda e: events.append(e),
            )

        tool_call_events = [e for e in events if isinstance(e, AgentToolCallEvent)]
        tool_result_events = [e for e in events if isinstance(e, AgentToolResultEvent)]
        turn_end_events = [e for e in events if isinstance(e, AgentTurnEndEvent)]

        assert len(tool_call_events) == 1
        assert tool_call_events[0].tool_name == "list_files"
        assert tool_call_events[0].tool_input == {"path": "/tmp"}
        assert tool_call_events[0].turn == 1

        assert len(tool_result_events) == 1
        assert tool_result_events[0].result == "file1.txt\nfile2.txt"
        assert tool_result_events[0].error is False
        assert tool_result_events[0].turn == 1

        # Two turns → two AgentTurnEndEvents
        assert len(turn_end_events) == 2

    @pytest.mark.asyncio
    async def test_agent_turn_end_includes_thinking(self):
        """AgentTurnEndEvent carries thinking text from that turn."""
        from piai import agent

        final = AssistantMessage(content=[
            ThinkingContent(thinking="Let me reason..."),
            TextContent(text="42"),
        ])
        done = DoneEvent(reason="stop", message=final)

        events = []

        async def mock_stream(*a, **kw):
            yield done

        with patch("piai.agent.stream", side_effect=mock_stream):
            await agent("gpt-5.1-codex-mini", Context(messages=[UserMessage(content="What is 6*7?")]),
                        on_event=lambda e: events.append(e))

        turn_end = next(e for e in events if isinstance(e, AgentTurnEndEvent))
        assert turn_end.thinking == "Let me reason..."

    @pytest.mark.asyncio
    async def test_agent_tool_result_error_flag_set_on_failure(self):
        """AgentToolResultEvent.error=True when tool execution fails."""
        from piai import agent

        tc = ToolCall(id="call_err", name="bad_tool", input={})
        turn1_msg = AssistantMessage(content=[ToolCallContent(tool_calls=[tc])])
        turn2_msg = AssistantMessage(content=[TextContent(text="ok")])

        call_count = 0

        async def mock_stream(*a, **kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                yield ToolCallEndEvent(tool_call=tc)
                yield DoneEvent(reason="tool_calls", message=turn1_msg)
            else:
                yield DoneEvent(reason="stop", message=turn2_msg)

        mock_hub = MagicMock()
        mock_hub.all_tools.return_value = []
        mock_hub.call_tool = AsyncMock(side_effect=KeyError("bad_tool"))
        mock_hub.__aenter__ = AsyncMock(return_value=mock_hub)
        mock_hub.__aexit__ = AsyncMock(return_value=False)

        events = []

        with patch("piai.agent.stream", side_effect=mock_stream), \
             patch("piai.agent.MCPHub", return_value=mock_hub):
            from piai.mcp import MCPServer
            await agent(
                "gpt-5.1-codex-mini",
                Context(messages=[UserMessage(content="run bad tool")]),
                mcp_servers=[MCPServer.stdio("dummy")],
                on_event=lambda e: events.append(e),
            )

        result_events = [e for e in events if isinstance(e, AgentToolResultEvent)]
        assert len(result_events) == 1
        assert result_events[0].error is True


# ─── PiAIChatModel thinking in additional_kwargs ──────────────────────────────

class TestPiAIChatModelThinking:

    @pytest.mark.asyncio
    async def test_thinking_delta_yielded_as_chunk(self):
        """ThinkingDeltaEvent → ChatGenerationChunk with additional_kwargs['thinking_delta']."""
        from piai.langchain import PiAIChatModel
        from langchain_core.messages import HumanMessage

        llm = PiAIChatModel(model_name="gpt-5.1-codex-mini")

        final_msg = AssistantMessage(content=[
            ThinkingContent(thinking="some reasoning"),
            TextContent(text="Answer"),
        ])

        async def mock_stream(*a, **kw):
            yield ThinkingStartEvent()
            yield ThinkingDeltaEvent(thinking="some ")
            yield ThinkingDeltaEvent(thinking="reasoning")
            yield ThinkingEndEvent(thinking="some reasoning")
            yield TextDeltaEvent(text="Answer")
            yield DoneEvent(reason="stop", message=final_msg)

        with patch("piai.langchain.chat_model.piai_stream", side_effect=mock_stream):
            chunks = []
            async for chunk in llm._astream([HumanMessage(content="think about it")]):
                chunks.append(chunk)

        thinking_delta_chunks = [
            c for c in chunks
            if c.message.additional_kwargs.get("thinking_delta")
        ]
        assert len(thinking_delta_chunks) == 2
        assert thinking_delta_chunks[0].message.additional_kwargs["thinking_delta"] == "some "
        assert thinking_delta_chunks[1].message.additional_kwargs["thinking_delta"] == "reasoning"

    @pytest.mark.asyncio
    async def test_done_chunk_carries_full_thinking(self):
        """DoneEvent chunk has additional_kwargs['thinking'] with full accumulated text."""
        from piai.langchain import PiAIChatModel
        from langchain_core.messages import HumanMessage

        llm = PiAIChatModel(model_name="gpt-5.1-codex-mini")

        final_msg = AssistantMessage(content=[
            ThinkingContent(thinking="deep thought"),
            TextContent(text="42"),
        ])

        async def mock_stream(*a, **kw):
            yield ThinkingStartEvent()
            yield ThinkingEndEvent(thinking="deep thought")
            yield TextDeltaEvent(text="42")
            yield DoneEvent(reason="stop", message=final_msg)

        with patch("piai.langchain.chat_model.piai_stream", side_effect=mock_stream):
            chunks = []
            async for chunk in llm._astream([HumanMessage(content="42?")]):
                chunks.append(chunk)

        done_chunk = next(c for c in chunks if c.generation_info)
        assert done_chunk.message.additional_kwargs.get("thinking") == "deep thought"

    @pytest.mark.asyncio
    async def test_no_thinking_no_additional_kwargs(self):
        """When model doesn't think, done chunk has no thinking in additional_kwargs."""
        from piai.langchain import PiAIChatModel
        from langchain_core.messages import HumanMessage

        llm = PiAIChatModel(model_name="gpt-5.1-codex-mini")
        final_msg = AssistantMessage(content=[TextContent(text="Hi")])

        async def mock_stream(*a, **kw):
            yield TextDeltaEvent(text="Hi")
            yield DoneEvent(reason="stop", message=final_msg)

        with patch("piai.langchain.chat_model.piai_stream", side_effect=mock_stream):
            chunks = []
            async for chunk in llm._astream([HumanMessage(content="hi")]):
                chunks.append(chunk)

        done_chunk = next(c for c in chunks if c.generation_info)
        assert "thinking" not in done_chunk.message.additional_kwargs


# ─── Agent coverage: error + no-done paths ────────────────────────────────────

class TestAgentEdgePaths:

    @pytest.mark.asyncio
    async def test_agent_raises_on_error_event(self):
        """ErrorEvent from stream raises RuntimeError in agent."""
        from piai import agent

        error_msg = AssistantMessage(error_message="quota exceeded")

        async def mock_stream(*a, **kw):
            yield ErrorEvent(reason="error", error=error_msg)

        with patch("piai.agent.stream", side_effect=mock_stream):
            with pytest.raises(RuntimeError, match="quota exceeded"):
                await agent("gpt-5.1-codex-mini", Context(messages=[UserMessage(content="hi")]))

    @pytest.mark.asyncio
    async def test_agent_raises_if_stream_ends_without_done(self):
        """Stream ending without DoneEvent raises RuntimeError."""
        from piai import agent

        async def mock_stream(*a, **kw):
            yield TextDeltaEvent(text="partial")
            # No DoneEvent!

        with patch("piai.agent.stream", side_effect=mock_stream):
            with pytest.raises(RuntimeError, match="without a done event"):
                await agent("gpt-5.1-codex-mini", Context(messages=[UserMessage(content="hi")]))

    @pytest.mark.asyncio
    async def test_execute_tool_with_no_hub_returns_message(self):
        """_execute_tool returns (error_string, True) when hub is None (no MCP servers)."""
        from piai.agent import _execute_tool

        tc = ToolCall(id="x", name="my_tool", input={})
        result, is_error = await _execute_tool(None, tc, 1000)
        assert "my_tool" in result
        assert "No MCP servers" in result
        assert is_error is True

    @pytest.mark.asyncio
    async def test_execute_tool_generic_exception(self):
        """_execute_tool catches generic Exception and returns (error_string, True)."""
        from piai.agent import _execute_tool

        hub = MagicMock()
        hub.call_tool = AsyncMock(side_effect=ValueError("unexpected"))

        tc = ToolCall(id="x", name="boom", input={})
        result, is_error = await _execute_tool(hub, tc, 1000)
        assert "boom" in result
        assert "failed" in result
        assert is_error is True


# ─── chat_model coverage: sync _stream + AIMessage list content ───────────────

class TestChatModelEdgePaths:

    def test_lc_messages_to_piai_ai_message_list_content(self):
        """AIMessage with list content extracts text blocks correctly."""
        from piai.langchain.chat_model import _lc_messages_to_piai
        from langchain_core.messages import AIMessage

        msg = AIMessage(content=[
            {"type": "text", "text": "Hello"},
            {"type": "text", "text": " world"},
        ])
        ctx = _lc_messages_to_piai([msg])
        assert len(ctx.messages) == 1
        from piai.types import AssistantMessage as AM
        am = ctx.messages[0]
        assert isinstance(am, AM)
        assert am.text == "Hello world"

    def test_lc_messages_to_piai_ai_message_str_content(self):
        """AIMessage with plain string content works."""
        from piai.langchain.chat_model import _lc_messages_to_piai
        from langchain_core.messages import AIMessage

        msg = AIMessage(content="plain string")
        ctx = _lc_messages_to_piai([msg])
        assert ctx.messages[0].text == "plain string"

    def test_lc_messages_to_piai_ai_message_non_text_content_type(self):
        """AIMessage with list containing non-text dict uses fallback."""
        from piai.langchain.chat_model import _lc_messages_to_piai
        from langchain_core.messages import AIMessage

        msg = AIMessage(content=[{"type": "image_url", "text": "ignored"}])
        ctx = _lc_messages_to_piai([msg])
        # Should not crash; non-text blocks use str fallback

    def test_sync_stream_yields_chunks(self):
        """_stream() (sync) yields ChatGenerationChunk via _run_async."""
        from piai.langchain import PiAIChatModel
        from langchain_core.messages import HumanMessage

        llm = PiAIChatModel(model_name="gpt-5.1-codex-mini")
        final_msg = AssistantMessage(content=[TextContent(text="Hi")])

        async def mock_stream(*a, **kw):
            yield TextDeltaEvent(text="Hi")
            yield DoneEvent(reason="stop", message=final_msg)

        with patch("piai.langchain.chat_model.piai_stream", side_effect=mock_stream):
            chunks = list(llm._stream([HumanMessage(content="hello")]))

        assert len(chunks) >= 1
        assert any(c.message.content == "Hi" for c in chunks)


# ─── SubAgentTool coverage: _run sync + _arun text extraction ─────────────────

class TestSubAgentToolCoverage:

    @pytest.mark.asyncio
    async def test_arun_extracts_text(self):
        """_arun extracts text from AssistantMessage and joins blocks."""
        from piai.langchain import SubAgentTool

        tool = SubAgentTool(name="test_agent", description="test")
        final = AssistantMessage(content=[
            TextContent(text="Part 1"),
            TextContent(text="Part 2"),
        ])

        with patch("piai.langchain.sub_agent_tool.piai_agent", new=AsyncMock(return_value=final)):
            result = await tool._arun("do something")

        assert "Part 1" in result
        assert "Part 2" in result

    @pytest.mark.asyncio
    async def test_arun_returns_fallback_when_no_text(self):
        """_arun returns fallback string when agent produces no text."""
        from piai.langchain import SubAgentTool

        tool = SubAgentTool(name="test_agent", description="test")
        final = AssistantMessage(content=[])

        with patch("piai.langchain.sub_agent_tool.piai_agent", new=AsyncMock(return_value=final)):
            result = await tool._arun("do something")

        assert "no text output" in result

    def test_run_sync_no_event_loop(self):
        """_run() works in a plain sync context (no running event loop)."""
        from piai.langchain import SubAgentTool

        tool = SubAgentTool(name="test_agent", description="test")
        final = AssistantMessage(content=[TextContent(text="sync result")])

        with patch("piai.langchain.sub_agent_tool.piai_agent", new=AsyncMock(return_value=final)):
            result = tool._run("do something")

        assert result == "sync result"


# ─── MCPLangChainTool._run sync path ──────────────────────────────────────────

class TestMCPLangChainToolSync:

    def test_run_sync_dispatches_to_async(self):
        """MCPLangChainTool._run() works outside event loop."""
        from piai.mcp.langchain_tools import MCPLangChainTool
        from piai.types import Tool

        hub = MagicMock()
        hub.call_tool = AsyncMock(return_value="file contents")

        piai_tool = Tool(name="read_file", description="reads a file", parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}},
        })

        from piai.mcp.langchain_tools import _make_input_schema
        schema = _make_input_schema(piai_tool)

        lc_tool = MCPLangChainTool(
            name="read_file",
            description="reads a file",
            hub=hub,
            mcp_tool_name="read_file",
            args_schema=schema,
        )

        with patch.object(lc_tool, "_arun", new=AsyncMock(return_value="file contents")):
            result = lc_tool._run(path="/tmp/test.txt")

        assert result == "file contents"
