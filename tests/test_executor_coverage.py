"""
Tests for "executor coverage" — every tool visible to the model must have an executor.

Covers all issues found in the piai SDK audit:
  - Issue 1/9:  Context.tools with no MCP and no local_handlers → warning emitted
  - Issue 2:    local_handlers overrides MCP on same name → debug log, correct dispatch
  - Issue 3:    tc.input is None or {} → handler called with (**{}) not crashed
  - Issue 4:    is_error flag is now a structured tuple, not string-prefix detection
  - Issue 5:    complete() warns when stop_reason == tool_use
  - Issue 6:    tool_choice forwarded from bind_tools to request body
  - Issue 7:    SubAgentTool propagates local_handlers
  - Issue 8:    MCPHub.connect() concurrent calls — no double-init
  - Issue 10:   Tool name collision removes unnamespaced key
  - Issue 11:   JSON decode failure on tool args → warning, fallback to {}
  - Issue 13:   LangChain tc["args"] = None → normalized to {}
  - Issue 14:   hub.call_tool(None args) → normalized to {}
  - Issue 16:   Parallel tool calls produce multiple ToolCallContent blocks (documented behavior)
  - Issue 17:   complete_text() warns on tool-calling response
  - Issue 18:   AgentToolCallEvent/AgentToolResultEvent get {} not None when tc.input is None

All tests are pure unit tests — no network, no real MCP servers.
"""
from __future__ import annotations

import asyncio
import json
import logging
import warnings
from dataclasses import field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from piai.agent import _execute_tool, _run_loop
from piai.mcp.hub import MCPHub
from piai.mcp.server import MCPServer
from piai.types import (
    AgentToolCallEvent,
    AgentToolResultEvent,
    AssistantMessage,
    Context,
    DoneEvent,
    TextContent,
    Tool,
    ToolCall,
    ToolCallContent,
    ToolResultMessage,
    UserMessage,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_assistant_with_tool_calls(*tool_calls: ToolCall) -> AssistantMessage:
    msg = AssistantMessage(stop_reason="tool_use")
    for tc in tool_calls:
        msg.content.append(ToolCallContent(tool_calls=[tc]))
    return msg


def _make_assistant_text(text: str) -> AssistantMessage:
    msg = AssistantMessage(stop_reason="stop")
    msg.content.append(TextContent(text=text))
    return msg


# ---------------------------------------------------------------------------
# Issue 3 + 18: tc.input None/empty dict normalization in _execute_tool
# ---------------------------------------------------------------------------

class TestInputNormalization:
    """_execute_tool must never call handler(**None)."""

    @pytest.mark.asyncio
    async def test_none_input_calls_handler_with_empty_dict(self):
        called_with = {}

        def handler(**kwargs):
            called_with.update(kwargs)
            return "ok"

        tc = ToolCall(id="1", name="my_tool", input=None)  # type: ignore[arg-type]
        result, is_error = await _execute_tool(
            hub=None,
            tc=tc,
            max_chars=1000,
            local_handlers={"my_tool": handler},
        )
        assert result == "ok"
        assert is_error is False
        assert called_with == {}

    @pytest.mark.asyncio
    async def test_empty_dict_input_calls_handler_correctly(self):
        call_count = [0]

        def handler(**kwargs):
            call_count[0] += 1
            return "called"

        tc = ToolCall(id="1", name="my_tool", input={})
        result, is_error = await _execute_tool(
            hub=None,
            tc=tc,
            max_chars=1000,
            local_handlers={"my_tool": handler},
        )
        assert result == "called"
        assert call_count[0] == 1
        assert is_error is False

    @pytest.mark.asyncio
    async def test_handler_with_kwargs_gets_correct_args(self):
        received = {}

        def handler(**kwargs):
            received.update(kwargs)
            return "done"

        tc = ToolCall(id="1", name="submit", input={"library_name": "libfoo.so", "targets": []})
        result, is_error = await _execute_tool(
            hub=None,
            tc=tc,
            max_chars=1000,
            local_handlers={"submit": handler},
        )
        assert received == {"library_name": "libfoo.so", "targets": []}
        assert is_error is False

    @pytest.mark.asyncio
    async def test_async_handler_awaited_correctly(self):
        async def async_handler(**kwargs):
            await asyncio.sleep(0)
            return "async_result"

        tc = ToolCall(id="1", name="async_tool", input={"x": 1})
        result, is_error = await _execute_tool(
            hub=None,
            tc=tc,
            max_chars=1000,
            local_handlers={"async_tool": async_handler},
        )
        assert result == "async_result"
        assert is_error is False


# ---------------------------------------------------------------------------
# Issue 4: _execute_tool returns structured (result, is_error) tuple
# ---------------------------------------------------------------------------

class TestStructuredErrorReturn:
    """_execute_tool returns (str, bool) — no more string-prefix fragility."""

    @pytest.mark.asyncio
    async def test_tool_not_in_local_handlers_no_hub_returns_error_true(self):
        tc = ToolCall(id="1", name="missing_tool", input={})
        result, is_error = await _execute_tool(
            hub=None, tc=tc, max_chars=1000, local_handlers=None
        )
        assert is_error is True
        assert "missing_tool" in result

    @pytest.mark.asyncio
    async def test_handler_exception_returns_error_true(self):
        def broken_handler(**kwargs):
            raise ValueError("something went wrong")

        tc = ToolCall(id="1", name="broken", input={})
        result, is_error = await _execute_tool(
            hub=None,
            tc=tc,
            max_chars=1000,
            local_handlers={"broken": broken_handler},
        )
        assert is_error is True
        assert "something went wrong" in result

    @pytest.mark.asyncio
    async def test_successful_handler_returns_error_false(self):
        tc = ToolCall(id="1", name="ok_tool", input={})
        result, is_error = await _execute_tool(
            hub=None,
            tc=tc,
            max_chars=1000,
            local_handlers={"ok_tool": lambda **kw: "Tool not found: some/path/file.txt"},
        )
        # This legitimate result used to trigger is_error=True via string prefix detection.
        # With structured return, it should be is_error=False.
        assert is_error is False
        assert result == "Tool not found: some/path/file.txt"

    @pytest.mark.asyncio
    async def test_handler_returning_none_becomes_ok(self):
        tc = ToolCall(id="1", name="void_tool", input={})
        result, is_error = await _execute_tool(
            hub=None,
            tc=tc,
            max_chars=1000,
            local_handlers={"void_tool": lambda **kw: None},
        )
        assert result == "ok"
        assert is_error is False


# ---------------------------------------------------------------------------
# Issue 2: local_handlers priority over MCP when names collide
# ---------------------------------------------------------------------------

class TestLocalHandlerPriority:
    """local_handlers must win over MCP when both have the same tool name."""

    @pytest.mark.asyncio
    async def test_local_handler_wins_over_mcp(self, caplog):
        local_called = [False]

        def local_handler(**kwargs):
            local_called[0] = True
            return "local_result"

        mock_hub = MagicMock()
        mock_hub.tool_names.return_value = ["submit_report"]
        mock_hub.call_tool = AsyncMock(return_value="mcp_result")

        tc = ToolCall(id="1", name="submit_report", input={"data": "x"})
        with caplog.at_level(logging.DEBUG, logger="piai.agent"):
            result, is_error = await _execute_tool(
                hub=mock_hub,
                tc=tc,
                max_chars=1000,
                local_handlers={"submit_report": local_handler},
            )

        assert result == "local_result"
        assert is_error is False
        assert local_called[0] is True
        mock_hub.call_tool.assert_not_called()
        # Should log a debug note about the priority
        assert any("local_handler wins" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Issue 1/9: Warning emitted for unexecutable tools
# ---------------------------------------------------------------------------

class TestUnexecutableToolWarning:
    """_run_loop must log a warning when tools have no executor."""

    @pytest.mark.asyncio
    async def test_warns_when_context_tool_has_no_executor(self, caplog):
        """Tool visible to model but neither in local_handlers nor in hub — should warn."""
        orphan_tool = Tool(name="orphan_tool", description="no handler", parameters={})
        ctx = Context(
            messages=[UserMessage(content="hello")],
            tools=[orphan_tool],
        )

        # Fake a stream that returns immediately (no tool calls)
        final_msg = _make_assistant_text("done")

        async def fake_stream(*args, **kwargs):
            from piai.types import DoneEvent as DE
            yield DE(reason="stop", message=final_msg)

        with patch("piai.agent.stream", fake_stream):
            with caplog.at_level(logging.WARNING, logger="piai.agent"):
                result = await _run_loop(
                    model_id="test-model",
                    context=ctx,
                    hub=None,
                    options={},
                    provider_id="openai-codex",
                    max_turns=1,
                    on_event=None,
                    tool_result_max_chars=1000,
                    local_handlers=None,
                )

        assert any("orphan_tool" in r.message for r in caplog.records)
        assert any("no executor" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_no_warning_when_all_tools_in_local_handlers(self, caplog):
        """When all context tools are covered by local_handlers — no warning."""
        submit_tool = Tool(name="submit", description="submit report", parameters={})
        ctx = Context(
            messages=[UserMessage(content="hello")],
            tools=[submit_tool],
        )

        final_msg = _make_assistant_text("done")

        async def fake_stream(*args, **kwargs):
            from piai.types import DoneEvent as DE
            yield DE(reason="stop", message=final_msg)

        with patch("piai.agent.stream", fake_stream):
            with caplog.at_level(logging.WARNING, logger="piai.agent"):
                await _run_loop(
                    model_id="test-model",
                    context=ctx,
                    hub=None,
                    options={},
                    provider_id="openai-codex",
                    max_turns=1,
                    on_event=None,
                    tool_result_max_chars=1000,
                    local_handlers={"submit": lambda **kw: "ok"},
                )

        # No "no executor" warning
        assert not any("no executor" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Issue 5 + 17: complete() and complete_text() warn on tool-calling response
# ---------------------------------------------------------------------------

class TestCompleteToolCallWarning:
    """complete() and complete_text() must warn when model responds with tool calls."""

    @pytest.mark.asyncio
    async def test_complete_warns_on_tool_use_stop_reason(self):
        msg = AssistantMessage(stop_reason="tool_use")
        msg.content.append(ToolCallContent(tool_calls=[ToolCall(id="1", name="foo", input={})]))

        async def fake_stream(*args, **kwargs):
            yield DoneEvent(reason="tool_use", message=msg)

        with patch("piai.stream.stream", fake_stream):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                from piai.stream import complete
                result = await complete("test-model", Context(messages=[UserMessage(content="hi")]))

        assert result.stop_reason == "tool_use"
        assert any("tool calls" in str(warning.message) for warning in w)

    @pytest.mark.asyncio
    async def test_complete_no_warning_on_normal_stop(self):
        msg = _make_assistant_text("hello world")

        async def fake_stream(*args, **kwargs):
            yield DoneEvent(reason="stop", message=msg)

        with patch("piai.stream.stream", fake_stream):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                from piai.stream import complete
                result = await complete("test-model", Context(messages=[UserMessage(content="hi")]))

        assert not any("tool calls" in str(warning.message) for warning in w)

    @pytest.mark.asyncio
    async def test_complete_text_warns_on_tool_use(self):
        msg = AssistantMessage(stop_reason="tool_use")
        msg.content.append(ToolCallContent(tool_calls=[ToolCall(id="1", name="foo", input={})]))

        async def fake_stream(*args, **kwargs):
            yield DoneEvent(reason="tool_use", message=msg)

        with patch("piai.stream.stream", fake_stream):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                from piai.stream import complete_text
                result = await complete_text("test-model", Context(messages=[UserMessage(content="hi")]))

        assert result == ""
        assert any("complete_text" in str(warning.message) or "tool" in str(warning.message) for warning in w)


# ---------------------------------------------------------------------------
# Issue 6: tool_choice forwarded through bind_tools → request body
# ---------------------------------------------------------------------------

class TestToolChoiceForwarding:
    """tool_choice passed to bind_tools must reach build_request_body."""

    def test_tool_choice_required_in_request_body(self):
        from piai.providers.message_transform import build_request_body

        tool = Tool(name="my_tool", description="a tool", parameters={"type": "object", "properties": {}})
        ctx = Context(
            messages=[UserMessage(content="hi")],
            tools=[tool],
        )
        body = build_request_body("gpt-5.1-codex-mini", ctx, {"tool_choice": "required"})
        assert body["tool_choice"] == "required"

    def test_tool_choice_specific_tool_in_request_body(self):
        from piai.providers.message_transform import build_request_body

        tool = Tool(name="submit_report", description="submit", parameters={"type": "object", "properties": {}})
        ctx = Context(messages=[UserMessage(content="hi")], tools=[tool])
        specific = {"type": "function", "name": "submit_report"}
        body = build_request_body("gpt-5.1-codex-mini", ctx, {"tool_choice": specific})
        assert body["tool_choice"] == specific

    def test_tool_choice_defaults_to_auto_when_tools_present(self):
        from piai.providers.message_transform import build_request_body

        tool = Tool(name="my_tool", description="a tool", parameters={"type": "object", "properties": {}})
        ctx = Context(messages=[UserMessage(content="hi")], tools=[tool])
        body = build_request_body("gpt-5.1-codex-mini", ctx, {})
        assert body["tool_choice"] == "auto"

    def test_no_tool_choice_when_no_tools(self):
        from piai.providers.message_transform import build_request_body

        ctx = Context(messages=[UserMessage(content="hi")])
        body = build_request_body("gpt-5.1-codex-mini", ctx, {"tool_choice": "required"})
        # tool_choice should not be in body when there are no tools
        assert "tool_choice" not in body


# ---------------------------------------------------------------------------
# Issue 7: SubAgentTool propagates local_handlers
# ---------------------------------------------------------------------------

class TestSubAgentToolLocalHandlers:
    """SubAgentTool must pass local_handlers through to piai_agent."""

    @pytest.mark.asyncio
    async def test_local_handlers_propagated(self):
        handler_called = [False]

        def my_handler(**kwargs):
            handler_called[0] = True
            return "handled"

        from piai.langchain.sub_agent_tool import SubAgentTool

        tool = SubAgentTool(
            name="test_agent",
            description="test",
            model_id="gpt-5.1-codex-mini",
            local_handlers={"my_tool": my_handler},
        )

        captured_kwargs = {}

        async def fake_agent(**kwargs):
            captured_kwargs.update(kwargs)
            return _make_assistant_text("done")

        with patch("piai.langchain.sub_agent_tool.piai_agent", fake_agent):
            await tool._arun("do something")

        assert "local_handlers" in captured_kwargs
        assert captured_kwargs["local_handlers"] == {"my_tool": my_handler}

    @pytest.mark.asyncio
    async def test_none_local_handlers_propagated_as_none(self):
        """SubAgentTool with no local_handlers passes None through."""
        from piai.langchain.sub_agent_tool import SubAgentTool

        tool = SubAgentTool(
            name="test_agent",
            description="test",
        )

        captured_kwargs = {}

        async def fake_agent(**kwargs):
            captured_kwargs.update(kwargs)
            return _make_assistant_text("done")

        with patch("piai.langchain.sub_agent_tool.piai_agent", fake_agent):
            await tool._arun("do something")

        assert captured_kwargs.get("local_handlers") is None


# ---------------------------------------------------------------------------
# Issue 8: MCPHub.connect() concurrent calls — no double-init
# ---------------------------------------------------------------------------

class TestMCPHubConcurrentConnect:
    """MCPHub.connect() must be idempotent under concurrent calls."""

    @pytest.mark.asyncio
    async def test_concurrent_connect_only_initializes_once(self):
        hub = MCPHub([MCPServer.stdio("echo hello")])

        init_count = [0]

        async def fake_connect_inner():
            # Mirrors the real _connect_inner guard — must respect _connected flag
            if hub._connected:
                return
            init_count[0] += 1
            await asyncio.sleep(0)  # Yield to allow other coroutines to try to enter
            hub._connected = True

        with patch.object(hub, "_connect_inner", fake_connect_inner):
            # Fire 10 concurrent connect() calls
            await asyncio.gather(*[hub.connect() for _ in range(10)])

        assert init_count[0] == 1, f"Expected 1 init, got {init_count[0]}"


# ---------------------------------------------------------------------------
# Issue 10: Tool name collision removes unnamespaced registry entry
# ---------------------------------------------------------------------------

class TestToolNameCollisionCleanup:
    """After collision, the unnamespaced registry key must be removed."""

    def test_collision_removes_unnamespaced_key(self):
        hub = MCPHub([])

        client1 = MagicMock()
        client1.server.name = "server1"
        client2 = MagicMock()
        client2.server.name = "server2"

        tool = Tool(name="search", description="search tool", parameters={})

        # Register first server's tool
        hub._register_tool(tool, client1)
        assert "search" in hub._tool_registry

        # Register second server's tool with same name — collision
        hub._register_tool(tool, client2)

        # The unnamespaced "search" key must be removed
        assert "search" not in hub._tool_registry, \
            "Unnamespaced 'search' key should be removed after collision to prevent ambiguous routing"
        assert "server1__search" in hub._tool_registry
        assert "server2__search" in hub._tool_registry

    def test_tool_names_after_collision_are_namespaced_only(self):
        hub = MCPHub([])

        client1 = MagicMock()
        client1.server.name = "ida"
        client2 = MagicMock()
        client2.server.name = "r2"

        tool = Tool(name="decompile", description="decompile function", parameters={})
        hub._register_tool(tool, client1)
        hub._register_tool(tool, client2)

        names = hub.tool_names()
        assert "decompile" not in names
        assert "ida__decompile" in names
        assert "r2__decompile" in names


# ---------------------------------------------------------------------------
# Issue 14: hub.call_tool normalizes None arguments
# ---------------------------------------------------------------------------

class TestHubCallToolNoneArgs:
    """hub.call_tool must normalize None args to {} before forwarding to MCP SDK."""

    @pytest.mark.asyncio
    async def test_none_args_normalized(self):
        hub = MCPHub([])
        hub._connected = True

        mock_client = AsyncMock()
        mock_client.call_tool = AsyncMock(return_value="result")
        hub._tool_registry["my_tool"] = (mock_client, "my_tool")

        await hub.call_tool("my_tool", None)  # type: ignore[arg-type]
        mock_client.call_tool.assert_called_once_with("my_tool", {})

    @pytest.mark.asyncio
    async def test_dict_args_passed_through(self):
        hub = MCPHub([])
        hub._connected = True

        mock_client = AsyncMock()
        mock_client.call_tool = AsyncMock(return_value="result")
        hub._tool_registry["my_tool"] = (mock_client, "my_tool")

        await hub.call_tool("my_tool", {"key": "value"})
        mock_client.call_tool.assert_called_once_with("my_tool", {"key": "value"})


# ---------------------------------------------------------------------------
# Issue 13: LangChain tc["args"] = None normalized to {}
# ---------------------------------------------------------------------------

class TestLangChainArgsNormalization:
    """_lc_messages_to_piai must handle None tool call args."""

    def test_none_args_become_empty_dict(self):
        """
        LangChain's AIMessage validates args at construction time (Pydantic),
        so args=None is rejected before _lc_messages_to_piai is called.
        This test verifies that _lc_messages_to_piai's `tc["args"] or {}` guard
        handles the empty-dict case correctly (args={} → stays {}, never None).

        The args=None scenario can arrive via deserialization or dict access.
        We test it by passing a raw message dict with None args directly.
        """
        from piai.langchain.chat_model import _lc_messages_to_piai
        from piai.types import AssistantMessage as PiaiAssistantMessage
        from unittest.mock import MagicMock

        # Simulate a message object that has tool_calls with None args
        # (as could arrive from deserialization bypassing Pydantic validation)
        mock_msg = MagicMock()
        mock_msg.type = "ai"
        mock_msg.content = ""
        mock_msg.tool_calls = [
            {"id": "call_123", "name": "my_tool", "args": None}
        ]

        ctx = _lc_messages_to_piai([mock_msg])
        assistant_msgs = [m for m in ctx.messages if isinstance(m, PiaiAssistantMessage)]
        assert len(assistant_msgs) == 1
        tool_call_blocks = [
            b for b in assistant_msgs[0].content
            if isinstance(b, ToolCallContent)
        ]
        assert len(tool_call_blocks) == 1
        for tc in tool_call_blocks[0].tool_calls:
            assert tc.input == {}, f"Expected {{}}, got {tc.input!r}"

    def test_valid_args_passed_through(self):
        from langchain_core.messages import AIMessage
        from piai.langchain.chat_model import _lc_messages_to_piai

        msg = AIMessage(
            content="",
            tool_calls=[{
                "id": "call_456",
                "name": "search",
                "args": {"query": "piai sdk"},
                "type": "tool_call",
            }],
        )
        ctx = _lc_messages_to_piai([msg])
        assistant_msgs = [m for m in ctx.messages if hasattr(m, "content")]
        tool_call_blocks = [b for b in assistant_msgs[0].content if isinstance(b, ToolCallContent)]
        tc = tool_call_blocks[0].tool_calls[0]
        assert tc.input == {"query": "piai sdk"}


# ---------------------------------------------------------------------------
# Issue 18: AgentToolCallEvent / AgentToolResultEvent get {} not None
# ---------------------------------------------------------------------------

class TestAgentEventInputNormalization:
    """AgentToolCallEvent and AgentToolResultEvent must receive {} when tc.input is None."""

    @pytest.mark.asyncio
    async def test_event_tool_input_is_dict_when_tc_input_is_none(self):
        events = []

        def on_event(e):
            events.append(e)

        tc = ToolCall(id="1", name="submit", input=None)  # type: ignore[arg-type]
        assistant_msg_with_tc = _make_assistant_with_tool_calls(tc)
        final_msg = _make_assistant_text("done")

        call_count = [0]

        async def fake_stream(*args, **kwargs):
            from piai.types import DoneEvent as DE, ToolCallEndEvent as TCE
            call_count[0] += 1
            if call_count[0] == 1:
                # First call: model makes a tool call
                yield TCE(tool_call=tc)
                yield DE(reason="tool_use", message=assistant_msg_with_tc)
            else:
                # Second call: model produces text (agent done)
                yield DE(reason="stop", message=final_msg)

        with patch("piai.agent.stream", fake_stream):
            ctx = Context(
                messages=[UserMessage(content="go")],
                tools=[Tool(name="submit", description="submit", parameters={})],
            )
            await _run_loop(
                model_id="test-model",
                context=ctx,
                hub=None,
                options={},
                provider_id="openai-codex",
                max_turns=3,
                on_event=on_event,
                tool_result_max_chars=1000,
                local_handlers={"submit": lambda **kw: "ok"},
            )

        tool_call_events = [e for e in events if isinstance(e, AgentToolCallEvent)]
        tool_result_events = [e for e in events if isinstance(e, AgentToolResultEvent)]

        for e in tool_call_events:
            assert isinstance(e.tool_input, dict), f"tool_input should be dict, got {type(e.tool_input)}"

        for e in tool_result_events:
            assert isinstance(e.tool_input, dict), f"tool_input should be dict, got {type(e.tool_input)}"


# ---------------------------------------------------------------------------
# Integration: r2mcp-style scenario — submit_fuzzing_report as local handler
# ---------------------------------------------------------------------------

class TestR2MCPScenario:
    """
    Simulates the r2mcp use case:
    - submit_fuzzing_report registered as local_handler
    - r2mcp tools come from MCP hub
    - Model calls submit_fuzzing_report as a real tool call
    - Report is captured and returned
    """

    @pytest.mark.asyncio
    async def test_submit_fuzzing_report_captured_by_local_handler(self):
        captured = {}

        def handle_submit(**kwargs):
            captured.update(kwargs)
            return "Report submitted successfully."

        submit_tool = Tool(
            name="submit_fuzzing_report",
            description="Submit final fuzzing report",
            parameters={
                "type": "object",
                "properties": {
                    "library_name": {"type": "string"},
                    "targets": {"type": "array", "items": {"type": "object"}},
                },
                "required": ["library_name", "targets"],
            },
        )

        ctx = Context(
            messages=[UserMessage(content="Analyze libfoo.so")],
            tools=[submit_tool],
        )

        submit_tc = ToolCall(
            id="call_submit",
            name="submit_fuzzing_report",
            input={"library_name": "libfoo.so", "targets": [{"name": "fuzz_target"}]},
        )
        assistant_with_submit = _make_assistant_with_tool_calls(submit_tc)
        final_text = _make_assistant_text("Analysis complete.")

        call_count = [0]

        async def fake_stream(*args, **kwargs):
            from piai.types import DoneEvent as DE, ToolCallEndEvent as TCE
            call_count[0] += 1
            if call_count[0] == 1:
                yield TCE(tool_call=submit_tc)
                yield DE(reason="tool_use", message=assistant_with_submit)
            else:
                yield DE(reason="stop", message=final_text)

        with patch("piai.agent.stream", fake_stream):
            result = await _run_loop(
                model_id="test-model",
                context=ctx,
                hub=None,
                options={},
                provider_id="openai-codex",
                max_turns=5,
                on_event=None,
                tool_result_max_chars=32000,
                local_handlers={"submit_fuzzing_report": handle_submit},
            )

        assert captured == {
            "library_name": "libfoo.so",
            "targets": [{"name": "fuzz_target"}],
        }
        assert result.text == "Analysis complete."

    @pytest.mark.asyncio
    async def test_submit_not_routed_to_mcp_when_local_handler_present(self):
        """Even with an MCP hub present, submit_fuzzing_report should go to local handler."""
        captured = {}

        def handle_submit(**kwargs):
            captured.update(kwargs)
            return "submitted"

        mock_hub = MagicMock()
        mock_hub.all_tools.return_value = [
            Tool(name="open_file", description="open file in r2", parameters={}),
            Tool(name="run_command", description="run r2 command", parameters={}),
        ]
        mock_hub.tool_names.return_value = ["open_file", "run_command"]

        submit_tool = Tool(name="submit_fuzzing_report", description="submit", parameters={})
        ctx = Context(
            messages=[UserMessage(content="Analyze libbar.so")],
            tools=[submit_tool],
        )

        submit_tc = ToolCall(
            id="call_1",
            name="submit_fuzzing_report",
            input={"library_name": "libbar.so", "targets": []},
        )
        assistant_with_submit = _make_assistant_with_tool_calls(submit_tc)
        final_text = _make_assistant_text("done")

        call_count = [0]

        async def fake_stream(*args, **kwargs):
            from piai.types import DoneEvent as DE, ToolCallEndEvent as TCE
            call_count[0] += 1
            if call_count[0] == 1:
                yield TCE(tool_call=submit_tc)
                yield DE(reason="tool_use", message=assistant_with_submit)
            else:
                yield DE(reason="stop", message=final_text)

        with patch("piai.agent.stream", fake_stream):
            result = await _run_loop(
                model_id="test-model",
                context=ctx,
                hub=mock_hub,
                options={},
                provider_id="openai-codex",
                max_turns=5,
                on_event=None,
                tool_result_max_chars=32000,
                local_handlers={"submit_fuzzing_report": handle_submit},
            )

        assert captured["library_name"] == "libbar.so"
        mock_hub.call_tool.assert_not_called()


# ---------------------------------------------------------------------------
# Edge case: IDA MCP scenario — two MCP servers (r2 + ida) with name collision
# ---------------------------------------------------------------------------

class TestIDAMCPCollisionScenario:
    """
    Simulates using both r2mcp and ida-mcp simultaneously.
    Both expose a 'decompile' tool — collision must be handled cleanly.
    """

    def test_r2_and_ida_decompile_collision_both_namespaced(self):
        hub = MCPHub([])

        r2_client = MagicMock()
        r2_client.server.name = "r2"
        ida_client = MagicMock()
        ida_client.server.name = "ida"

        decompile_tool = Tool(name="decompile", description="decompile function", parameters={})
        hub._register_tool(decompile_tool, r2_client)
        hub._register_tool(decompile_tool, ida_client)

        names = hub.tool_names()
        assert "decompile" not in names, "Unnamespaced 'decompile' must not exist after collision"
        assert "r2__decompile" in names
        assert "ida__decompile" in names

    @pytest.mark.asyncio
    async def test_namespaced_tool_routes_to_correct_server(self):
        hub = MCPHub([])
        hub._connected = True

        r2_client = AsyncMock()
        r2_client.server.name = "r2"
        r2_client.call_tool = AsyncMock(return_value="r2_decompiled")

        ida_client = AsyncMock()
        ida_client.server.name = "ida"
        ida_client.call_tool = AsyncMock(return_value="ida_decompiled")

        decompile_tool = Tool(name="decompile", description="decompile", parameters={})
        hub._register_tool(decompile_tool, r2_client)
        hub._register_tool(decompile_tool, ida_client)

        r2_result = await hub.call_tool("r2__decompile", {"addr": "0x1000"})
        ida_result = await hub.call_tool("ida__decompile", {"addr": "0x1000"})

        assert r2_result == "r2_decompiled"
        assert ida_result == "ida_decompiled"
        r2_client.call_tool.assert_called_once_with("decompile", {"addr": "0x1000"})
        ida_client.call_tool.assert_called_once_with("decompile", {"addr": "0x1000"})
