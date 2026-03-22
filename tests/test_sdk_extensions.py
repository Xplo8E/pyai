"""
Unit tests for the four SDK extension features:

1. context_reducer hook in agent()
2. usage in AgentTurnEndEvent
3. context_extractor / initial_context in SubAgentTool
4. scratchpad in Context (injected into instructions)

All tests are fully offline — no real LLM calls, no network.
The agent loop is driven by injecting fake stream events via a patched stream().
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from piai.providers.message_transform import build_request_body
from piai.types import (
    AgentTurnEndEvent,
    AssistantMessage,
    Context,
    DoneEvent,
    TextContent,
    ToolCall,
    ToolCallContent,
    ToolCallEndEvent,
    ToolResultMessage,
    UserMessage,
)


# ------------------------------------------------------------------ #
# Helpers — fake stream factories                                     #
# ------------------------------------------------------------------ #

def _make_final_message(text: str = "done", usage: dict | None = None) -> AssistantMessage:
    """Build a minimal AssistantMessage that looks like it came from the model."""
    msg = AssistantMessage(content=[TextContent(text=text)])
    if usage:
        msg.usage.update(usage)
    return msg


async def _stream_done_only(text: str = "done", usage: dict | None = None) -> AsyncGenerator:
    """Fake stream that emits just a DoneEvent (no tool calls)."""
    msg = _make_final_message(text, usage)
    yield DoneEvent(reason="stop", message=msg)


async def _stream_one_tool_call(tool_name: str, tool_input: dict) -> AsyncGenerator:
    """Fake stream that emits one tool call then a DoneEvent."""
    tc = ToolCall(id="call_1", name=tool_name, input=tool_input)
    yield ToolCallEndEvent(tool_call=tc)
    msg = AssistantMessage(content=[ToolCallContent(tool_calls=[tc])])
    msg.usage.update({"input": 10, "output": 5, "total_tokens": 15})
    yield DoneEvent(reason="tool_use", message=msg)


# ------------------------------------------------------------------ #
# Feature 4: scratchpad injected into instructions                    #
# ------------------------------------------------------------------ #

class TestScratchpadInjection:
    def test_empty_scratchpad_no_change(self):
        ctx = Context(
            system_prompt="You are helpful.",
            messages=[UserMessage(content="hi")],
            scratchpad={},
        )
        body = build_request_body("gpt-5.1-codex-mini", ctx)
        assert body["instructions"] == "You are helpful."

    def test_scratchpad_appended_to_instructions(self):
        ctx = Context(
            system_prompt="You are a fuzzer.",
            messages=[UserMessage(content="go")],
            scratchpad={"known_vulns": ["0x400A", "0x400B"], "phase": "recon"},
        )
        body = build_request_body("gpt-5.1-codex-mini", ctx)
        instructions = body["instructions"]

        assert instructions.startswith("You are a fuzzer.")
        assert "<scratchpad>" in instructions
        assert "</scratchpad>" in instructions
        parsed = json.loads(instructions.split("<scratchpad>")[1].split("</scratchpad>")[0].strip())
        assert parsed["known_vulns"] == ["0x400A", "0x400B"]
        assert parsed["phase"] == "recon"

    def test_scratchpad_with_no_system_prompt(self):
        ctx = Context(
            messages=[UserMessage(content="hi")],
            scratchpad={"key": "value"},
        )
        body = build_request_body("gpt-5.1-codex-mini", ctx)
        instructions = body["instructions"]
        # Default prompt should still be first
        assert instructions.startswith("You are a helpful assistant.")
        assert "<scratchpad>" in instructions

    def test_scratchpad_injected_on_every_call(self):
        """Mutating the scratchpad between calls produces different instructions."""
        ctx = Context(
            system_prompt="Agent.",
            messages=[UserMessage(content="go")],
            scratchpad={"step": 1},
        )
        body1 = build_request_body("gpt-5.1-codex-mini", ctx)
        ctx.scratchpad["step"] = 2
        body2 = build_request_body("gpt-5.1-codex-mini", ctx)

        assert '"step": 1' in body1["instructions"]
        assert '"step": 2' in body2["instructions"]

    def test_scratchpad_field_defaults_to_empty_dict(self):
        ctx = Context(messages=[UserMessage(content="hi")])
        assert ctx.scratchpad == {}


# ------------------------------------------------------------------ #
# Feature 2: usage in AgentTurnEndEvent                              #
# ------------------------------------------------------------------ #

class TestAgentTurnEndUsage:
    def test_event_has_usage_field(self):
        event = AgentTurnEndEvent(turn=1, usage={"input": 100, "output": 50})
        assert event.usage["input"] == 100
        assert event.usage["output"] == 50

    def test_event_usage_defaults_to_empty_dict(self):
        event = AgentTurnEndEvent(turn=1)
        assert event.usage == {}

    @pytest.mark.asyncio
    async def test_agent_fires_turn_end_with_usage(self):
        """Agent loop fires AgentTurnEndEvent with usage from the model response."""
        from piai.agent import agent

        ctx = Context(messages=[UserMessage(content="hello")])
        events: list = []

        usage_data = {"input": 42, "output": 18, "total_tokens": 60}

        async def fake_stream(*args, **kwargs):
            yield DoneEvent(reason="stop", message=_make_final_message(usage=usage_data))

        with patch("piai.agent.stream", side_effect=fake_stream):
            await agent(
                model_id="gpt-5.1-codex-mini",
                context=ctx,
                on_event=lambda e: events.append(e),
            )

        turn_end_events = [e for e in events if isinstance(e, AgentTurnEndEvent)]
        assert len(turn_end_events) == 1
        evt = turn_end_events[0]
        assert evt.usage["input"] == 42
        assert evt.usage["output"] == 18
        assert evt.usage["total_tokens"] == 60


# ------------------------------------------------------------------ #
# Feature 1: context_reducer hook                                     #
# ------------------------------------------------------------------ #

class TestContextReducer:
    @pytest.mark.asyncio
    async def test_reducer_called_after_tool_results(self):
        """context_reducer is called once after tool results are appended."""
        from piai.agent import agent

        reducer_calls: list[Context] = []

        def reducer(ctx: Context) -> Context:
            reducer_calls.append(ctx)
            return ctx

        call_count = 0

        async def fake_stream(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call: emit a tool call
                tc = ToolCall(id="call_1", name="echo", input={"msg": "hi"})
                yield ToolCallEndEvent(tool_call=tc)
                msg = AssistantMessage(content=[ToolCallContent(tool_calls=[tc])])
                yield DoneEvent(reason="tool_use", message=msg)
            else:
                # Second call: just stop
                yield DoneEvent(reason="stop", message=_make_final_message())

        ctx = Context(messages=[UserMessage(content="go")])

        with patch("piai.agent.stream", side_effect=fake_stream):
            await agent(
                model_id="gpt-5.1-codex-mini",
                context=ctx,
                local_handlers={"echo": lambda msg: f"echoed: {msg}"},
                context_reducer=reducer,
            )

        # Reducer should be called exactly once (after turn 1's tool results)
        assert len(reducer_calls) == 1
        # Context passed to reducer should include the tool result message
        tool_results = [m for m in reducer_calls[0].messages if isinstance(m, ToolResultMessage)]
        assert len(tool_results) == 1

    @pytest.mark.asyncio
    async def test_reducer_can_trim_messages(self):
        """Reducer that drops old messages is respected by subsequent LLM calls."""
        from piai.agent import agent

        captured_msg_counts: list[int] = []
        call_count = 0

        async def fake_stream(model_id, ctx, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            # Snapshot the count at call time (ctx.messages is mutable, don't store ref)
            captured_msg_counts.append(len(ctx.messages))
            if call_count == 1:
                tc = ToolCall(id="call_1", name="noop", input={})
                yield ToolCallEndEvent(tool_call=tc)
                msg = AssistantMessage(content=[ToolCallContent(tool_calls=[tc])])
                yield DoneEvent(reason="tool_use", message=msg)
            else:
                yield DoneEvent(reason="stop", message=_make_final_message())

        def trimming_reducer(ctx: Context) -> Context:
            # Keep only the last message (the tool result)
            return Context(
                messages=ctx.messages[-1:],
                system_prompt=ctx.system_prompt,
                tools=ctx.tools,
                scratchpad=ctx.scratchpad,
            )

        ctx = Context(messages=[UserMessage(content="start")])

        with patch("piai.agent.stream", side_effect=fake_stream):
            await agent(
                model_id="gpt-5.1-codex-mini",
                context=ctx,
                local_handlers={"noop": lambda: "ok"},
                context_reducer=trimming_reducer,
            )

        # Turn 1: 1 initial user message
        assert captured_msg_counts[0] == 1
        # After turn 1: 3 messages appended (user + assistant + tool result).
        # Reducer keeps only last 1 → turn 2 sees exactly 1 message.
        assert captured_msg_counts[1] == 1

    @pytest.mark.asyncio
    async def test_reducer_not_called_when_no_tool_calls(self):
        """Reducer is NOT called on turns where the model makes no tool calls."""
        from piai.agent import agent

        reducer_calls = []

        async def fake_stream(*args, **kwargs):
            yield DoneEvent(reason="stop", message=_make_final_message())

        ctx = Context(messages=[UserMessage(content="hi")])

        with patch("piai.agent.stream", side_effect=fake_stream):
            await agent(
                model_id="gpt-5.1-codex-mini",
                context=ctx,
                context_reducer=lambda c: reducer_calls.append(c) or c,
            )

        assert len(reducer_calls) == 0

    @pytest.mark.asyncio
    async def test_async_reducer_supported(self):
        """context_reducer can be an async function."""
        from piai.agent import agent

        call_count = 0

        async def fake_stream(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                tc = ToolCall(id="c1", name="noop", input={})
                yield ToolCallEndEvent(tool_call=tc)
                msg = AssistantMessage(content=[ToolCallContent(tool_calls=[tc])])
                yield DoneEvent(reason="tool_use", message=msg)
            else:
                yield DoneEvent(reason="stop", message=_make_final_message())

        async def async_reducer(ctx: Context) -> Context:
            await asyncio.sleep(0)  # simulate async work
            return ctx

        ctx = Context(messages=[UserMessage(content="go")])

        with patch("piai.agent.stream", side_effect=fake_stream):
            result = await agent(
                model_id="gpt-5.1-codex-mini",
                context=ctx,
                local_handlers={"noop": lambda: "ok"},
                context_reducer=async_reducer,
            )

        assert result.text == "done"


# ------------------------------------------------------------------ #
# Feature 3: context_extractor in SubAgentTool                       #
# ------------------------------------------------------------------ #

class TestContextExtractor:
    @pytest.mark.asyncio
    async def test_extractor_filters_initial_context(self):
        """context_extractor receives initial_context and its output is used."""
        from piai.langchain.sub_agent_tool import SubAgentTool
        from piai.types import Context, UserMessage

        captured_contexts: list[Context] = []

        async def fake_agent(model_id, context, **kwargs):
            captured_contexts.append(context)
            return _make_final_message("result")

        # A "fat" context simulating a parent orchestrator state
        initial_ctx = Context(
            messages=[
                UserMessage(content="global task ledger with 50 entries..."),
                UserMessage(content="irrelevant history 1"),
                UserMessage(content="irrelevant history 2"),
            ],
            system_prompt="Parent orchestrator",
            scratchpad={"apk_path": "/data/app.apk", "findings": []},
        )

        def extractor(ctx: Context) -> Context:
            # Only pass through the scratchpad's apk_path — strip all history
            return Context(
                messages=[UserMessage(content=f"Analyze: {ctx.scratchpad['apk_path']}")],
                system_prompt="Sub-agent: binary analyzer",
                scratchpad={"apk_path": ctx.scratchpad["apk_path"]},
            )

        tool = SubAgentTool(
            name="binary_analyzer",
            description="Analyzes APK",
            initial_context=initial_ctx,
            context_extractor=extractor,
        )

        with patch("piai.langchain.sub_agent_tool.piai_agent", side_effect=fake_agent):
            result = await tool._arun(task="find vulnerabilities")

        assert len(captured_contexts) == 1
        ctx_used = captured_contexts[0]
        # Extractor stripped the 3-message history down to 1
        # (extractor returned 1 message, then _arun may add task if last isn't user msg)
        assert any("Analyze:" in (m.content if isinstance(m.content, str) else "") for m in ctx_used.messages)
        assert ctx_used.system_prompt == "Sub-agent: binary analyzer"
        assert ctx_used.scratchpad["apk_path"] == "/data/app.apk"
        # The massive parent history should NOT be present
        assert not any("global task ledger" in (m.content if isinstance(m.content, str) else "") for m in ctx_used.messages)

    @pytest.mark.asyncio
    async def test_no_extractor_uses_fresh_context(self):
        """Without context_extractor, SubAgentTool builds a fresh context from the task."""
        from piai.langchain.sub_agent_tool import SubAgentTool

        captured_contexts: list[Context] = []

        async def fake_agent(model_id, context, **kwargs):
            captured_contexts.append(context)
            return _make_final_message("result")

        tool = SubAgentTool(
            name="analyzer",
            description="Test",
            system_prompt="You are an analyzer.",
        )

        with patch("piai.langchain.sub_agent_tool.piai_agent", side_effect=fake_agent):
            await tool._arun(task="analyze this")

        assert len(captured_contexts) == 1
        ctx = captured_contexts[0]
        assert ctx.system_prompt == "You are an analyzer."
        assert len(ctx.messages) == 1
        assert ctx.messages[0].content == "analyze this"

    @pytest.mark.asyncio
    async def test_task_appended_when_extractor_leaves_no_user_message(self):
        """If extractor returns context without a trailing user message, task is appended."""
        from piai.langchain.sub_agent_tool import SubAgentTool

        captured_contexts: list[Context] = []

        async def fake_agent(model_id, context, **kwargs):
            captured_contexts.append(context)
            return _make_final_message("ok")

        initial_ctx = Context(
            messages=[UserMessage(content="some old history")],
            scratchpad={"data": 42},
        )

        def extractor(ctx: Context) -> Context:
            # Return context with NO user message at the end
            return Context(messages=[], system_prompt="Sub-agent", scratchpad=ctx.scratchpad)

        tool = SubAgentTool(
            name="sub",
            description="test",
            initial_context=initial_ctx,
            context_extractor=extractor,
        )

        with patch("piai.langchain.sub_agent_tool.piai_agent", side_effect=fake_agent):
            await tool._arun(task="do the thing")

        ctx = captured_contexts[0]
        # Task should have been appended
        assert any(
            isinstance(m, UserMessage) and "do the thing" in (m.content if isinstance(m.content, str) else "")
            for m in ctx.messages
        )

    @pytest.mark.asyncio
    async def test_extractor_result_replaces_initial_context(self):
        """What the extractor returns IS the context passed to the agent — not a merge."""
        from piai.langchain.sub_agent_tool import SubAgentTool

        captured_contexts: list[Context] = []

        async def fake_agent(model_id, context, **kwargs):
            captured_contexts.append(context)
            return _make_final_message("ok")

        fat_ctx = Context(
            messages=[UserMessage(content=f"msg{i}") for i in range(20)],
            scratchpad={"key": "val"},
        )

        extractor_output = Context(
            messages=[UserMessage(content="only this")],
            system_prompt="Stripped",
            scratchpad={"key": "val"},
        )

        tool = SubAgentTool(
            name="sub",
            description="test",
            initial_context=fat_ctx,
            context_extractor=lambda _: extractor_output,
        )

        with patch("piai.langchain.sub_agent_tool.piai_agent", side_effect=fake_agent):
            await tool._arun(task="only this")

        ctx = captured_contexts[0]
        # Extractor output had "only this" as last msg = user msg, so task not re-appended
        assert len(ctx.messages) == 1
        assert ctx.messages[0].content == "only this"
        assert ctx.system_prompt == "Stripped"
