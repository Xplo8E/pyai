"""
Tests for the LangChain PiAIChatModel adapter.

Covers message conversion helpers, stream event handling, _agenerate / _generate
wrappers, bind_tools, and model metadata properties.  All piai_stream calls are
mocked — no real API calls are made.
"""

from __future__ import annotations

from typing import TypedDict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from piai.langchain.chat_model import (
    PiAIChatModel,
    _lc_messages_to_piai,
    _lc_tools_to_piai,
)
from piai.types import (
    AssistantMessage,
    Context,
    DoneEvent,
    ErrorEvent,
    TextContent,
    TextDeltaEvent,
    ToolCall,
    ToolCallContent,
    ToolCallDeltaEvent,
    ToolCallStartEvent,
    ToolResultMessage,
    UserMessage,
)


# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #

def _make_async_gen(*events):
    """Return an async generator that yields the given events."""
    async def _gen(*args, **kwargs):
        for event in events:
            yield event
    return _gen


# ------------------------------------------------------------------ #
# _lc_messages_to_piai                                               #
# ------------------------------------------------------------------ #

class TestLcMessagesToPiai:

    def test_system_message_becomes_system_prompt(self):
        msgs = [SystemMessage(content="You are helpful.")]
        ctx = _lc_messages_to_piai(msgs)
        assert ctx.system_prompt == "You are helpful."
        assert ctx.messages == []

    def test_human_message_becomes_user_message(self):
        msgs = [HumanMessage(content="Hello")]
        ctx = _lc_messages_to_piai(msgs)
        assert len(ctx.messages) == 1
        assert isinstance(ctx.messages[0], UserMessage)
        assert ctx.messages[0].content == "Hello"

    def test_ai_message_text_only(self):
        msgs = [AIMessage(content="I am an assistant.")]
        ctx = _lc_messages_to_piai(msgs)
        assert len(ctx.messages) == 1
        msg = ctx.messages[0]
        assert isinstance(msg, AssistantMessage)
        assert len(msg.content) == 1
        assert isinstance(msg.content[0], TextContent)
        assert msg.content[0].text == "I am an assistant."

    def test_ai_message_with_tool_calls(self):
        tc = {"id": "call_abc", "name": "get_weather", "args": {"city": "London"}, "type": "tool_call"}
        msgs = [AIMessage(content="", tool_calls=[tc])]
        ctx = _lc_messages_to_piai(msgs)
        assert len(ctx.messages) == 1
        msg = ctx.messages[0]
        assert isinstance(msg, AssistantMessage)
        # No TextContent block because content was empty
        tcc_blocks = [b for b in msg.content if isinstance(b, ToolCallContent)]
        assert len(tcc_blocks) == 1
        tool_calls = tcc_blocks[0].tool_calls
        assert len(tool_calls) == 1
        assert tool_calls[0].id == "call_abc"
        assert tool_calls[0].name == "get_weather"
        assert tool_calls[0].input == {"city": "London"}

    def test_ai_message_text_and_tool_calls(self):
        tc = {"id": "call_xyz", "name": "lookup", "args": {}, "type": "tool_call"}
        msgs = [AIMessage(content="Let me look that up.", tool_calls=[tc])]
        ctx = _lc_messages_to_piai(msgs)
        msg = ctx.messages[0]
        assert isinstance(msg, AssistantMessage)
        text_blocks = [b for b in msg.content if isinstance(b, TextContent)]
        tcc_blocks = [b for b in msg.content if isinstance(b, ToolCallContent)]
        assert len(text_blocks) == 1
        assert text_blocks[0].text == "Let me look that up."
        assert len(tcc_blocks) == 1

    def test_tool_message_becomes_tool_result_message(self):
        msgs = [ToolMessage(content="42 degrees", tool_call_id="call_abc")]
        ctx = _lc_messages_to_piai(msgs)
        assert len(ctx.messages) == 1
        msg = ctx.messages[0]
        assert isinstance(msg, ToolResultMessage)
        assert msg.tool_call_id == "call_abc"
        assert msg.content == "42 degrees"

    def test_no_system_message_gives_none_system_prompt(self):
        msgs = [HumanMessage(content="Hi")]
        ctx = _lc_messages_to_piai(msgs)
        assert ctx.system_prompt is None

    def test_mixed_conversation(self):
        tc = {"id": "tc1", "name": "calc", "args": {"x": 1}, "type": "tool_call"}
        msgs = [
            SystemMessage(content="System instructions."),
            HumanMessage(content="What is 1+1?"),
            AIMessage(content="", tool_calls=[tc]),
            ToolMessage(content="2", tool_call_id="tc1"),
            AIMessage(content="The answer is 2."),
        ]
        ctx = _lc_messages_to_piai(msgs)
        assert ctx.system_prompt == "System instructions."
        assert len(ctx.messages) == 4
        assert isinstance(ctx.messages[0], UserMessage)
        assert isinstance(ctx.messages[1], AssistantMessage)
        assert isinstance(ctx.messages[2], ToolResultMessage)
        assert isinstance(ctx.messages[3], AssistantMessage)

    def test_empty_message_list(self):
        ctx = _lc_messages_to_piai([])
        assert ctx.system_prompt is None
        assert ctx.messages == []

    def test_returns_context_instance(self):
        ctx = _lc_messages_to_piai([HumanMessage(content="x")])
        assert isinstance(ctx, Context)

    def test_ai_message_list_content_extracts_text(self):
        """AIMessage with list content (multi-modal) extracts text blocks."""
        msgs = [AIMessage(content=[{"type": "text", "text": "Hello"}, {"type": "text", "text": " world"}])]
        ctx = _lc_messages_to_piai(msgs)
        msg = ctx.messages[0]
        assert isinstance(msg, AssistantMessage)
        text_blocks = [b for b in msg.content if isinstance(b, TextContent)]
        assert len(text_blocks) == 1
        assert text_blocks[0].text == "Hello world"

    def test_ai_message_empty_list_content_no_text_block(self):
        """AIMessage with empty list content should produce no text block."""
        msgs = [AIMessage(content=[])]
        ctx = _lc_messages_to_piai(msgs)
        msg = ctx.messages[0]
        assert isinstance(msg, AssistantMessage)
        assert msg.content == []  # no text block for empty content


# ------------------------------------------------------------------ #
# _lc_tools_to_piai                                                  #
# ------------------------------------------------------------------ #

class TestLcToolsToPiai:

    def test_flat_dict_format(self):
        tools = [
            {
                "name": "get_weather",
                "description": "Get weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            }
        ]
        result = _lc_tools_to_piai(tools)
        assert len(result) == 1
        assert result[0].name == "get_weather"
        assert result[0].description == "Get weather for a city"
        assert result[0].parameters["properties"]["city"]["type"] == "string"

    def test_nested_function_format(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_web",
                    "description": "Search the internet",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        result = _lc_tools_to_piai(tools)
        assert len(result) == 1
        assert result[0].name == "search_web"
        assert result[0].description == "Search the internet"

    def test_missing_description_defaults_to_empty_string(self):
        tools = [{"name": "no_desc", "parameters": {}}]
        result = _lc_tools_to_piai(tools)
        assert result[0].description == ""

    def test_missing_parameters_defaults_to_empty_dict(self):
        tools = [{"name": "no_params", "description": "No params here"}]
        result = _lc_tools_to_piai(tools)
        assert result[0].parameters == {}

    def test_empty_parameters_dict(self):
        tools = [{"name": "empty", "description": "", "parameters": {}}]
        result = _lc_tools_to_piai(tools)
        assert result[0].parameters == {}

    def test_multiple_tools(self):
        tools = [
            {"name": "tool_a", "description": "A", "parameters": {}},
            {"name": "tool_b", "description": "B", "parameters": {}},
        ]
        result = _lc_tools_to_piai(tools)
        assert len(result) == 2
        assert result[0].name == "tool_a"
        assert result[1].name == "tool_b"

    def test_empty_list(self):
        assert _lc_tools_to_piai([]) == []


# ------------------------------------------------------------------ #
# PiAIChatModel._astream                                             #
# ------------------------------------------------------------------ #

class TestAStream:

    async def test_text_delta_yields_content_chunk(self):
        model = PiAIChatModel()
        events = [TextDeltaEvent(text="Hello, world!"), DoneEvent(reason="stop")]

        with patch("piai.langchain.chat_model.piai_stream", side_effect=_make_async_gen(*events)):
            chunks = [c async for c in model._astream([HumanMessage(content="hi")])]

        text_chunk = chunks[0]
        assert text_chunk.message.content == "Hello, world!"

    async def test_done_event_sets_finish_reason(self):
        model = PiAIChatModel()
        events = [DoneEvent(reason="stop")]

        with patch("piai.langchain.chat_model.piai_stream", side_effect=_make_async_gen(*events)):
            chunks = [c async for c in model._astream([HumanMessage(content="hi")])]

        done_chunk = chunks[-1]
        assert done_chunk.generation_info is not None
        assert done_chunk.generation_info["finish_reason"] == "stop"

    async def test_tool_call_start_event_yields_chunk(self):
        model = PiAIChatModel()
        tc = ToolCall(id="call_123", name="get_weather")
        events = [ToolCallStartEvent(tool_call=tc), DoneEvent(reason="tool_calls")]

        with patch("piai.langchain.chat_model.piai_stream", side_effect=_make_async_gen(*events)):
            chunks = [c async for c in model._astream([HumanMessage(content="weather?")])]

        start_chunk = chunks[0]
        tcc = start_chunk.message.tool_call_chunks
        assert len(tcc) == 1
        assert tcc[0]["name"] == "get_weather"
        assert tcc[0]["id"] == "call_123"
        assert tcc[0]["index"] == 0
        assert tcc[0]["args"] == ""

    async def test_tool_call_delta_event_yields_args_chunk(self):
        model = PiAIChatModel()
        tc = ToolCall(id="call_abc", name="lookup")
        events = [
            ToolCallStartEvent(tool_call=tc),
            ToolCallDeltaEvent(id="call_abc", json_delta='{"q":'),
            ToolCallDeltaEvent(id="call_abc", json_delta='"foo"}'),
            DoneEvent(reason="tool_calls"),
        ]

        with patch("piai.langchain.chat_model.piai_stream", side_effect=_make_async_gen(*events)):
            chunks = [c async for c in model._astream([HumanMessage(content="lookup")])]

        delta_chunks = [c for c in chunks if c.message.tool_call_chunks and c.message.tool_call_chunks[0]["name"] is None]
        assert len(delta_chunks) == 2
        assert delta_chunks[0].message.tool_call_chunks[0]["args"] == '{"q":'
        assert delta_chunks[1].message.tool_call_chunks[0]["args"] == '"foo"}'

    async def test_tool_call_id_truncated_to_64_chars(self):
        model = PiAIChatModel()
        long_id = "x" * 100
        tc = ToolCall(id=long_id, name="big_tool")
        events = [ToolCallStartEvent(tool_call=tc), DoneEvent(reason="tool_calls")]

        with patch("piai.langchain.chat_model.piai_stream", side_effect=_make_async_gen(*events)):
            chunks = [c async for c in model._astream([HumanMessage(content="hi")])]

        start_chunk = chunks[0]
        emitted_id = start_chunk.message.tool_call_chunks[0]["id"]
        assert len(emitted_id) == 64
        assert emitted_id == "x" * 64

    async def test_error_event_raises_runtime_error(self):
        model = PiAIChatModel()
        error_msg = AssistantMessage(error_message="Something went wrong")
        events = [ErrorEvent(error=error_msg)]

        with patch("piai.langchain.chat_model.piai_stream", side_effect=_make_async_gen(*events)):
            with pytest.raises(RuntimeError, match="Something went wrong"):
                async for _ in model._astream([HumanMessage(content="hi")]):
                    pass

    async def test_error_event_with_no_message_raises_generic(self):
        model = PiAIChatModel()
        # error_message defaults to None on AssistantMessage
        error_msg = AssistantMessage()
        events = [ErrorEvent(error=error_msg)]

        with patch("piai.langchain.chat_model.piai_stream", side_effect=_make_async_gen(*events)):
            with pytest.raises(RuntimeError, match="piai stream error"):
                async for _ in model._astream([HumanMessage(content="hi")]):
                    pass

    async def test_tool_call_delta_index_follows_start_index(self):
        model = PiAIChatModel()
        tc0 = ToolCall(id="id_0", name="tool_a")
        tc1 = ToolCall(id="id_1", name="tool_b")
        events = [
            ToolCallStartEvent(tool_call=tc0),
            ToolCallStartEvent(tool_call=tc1),
            ToolCallDeltaEvent(id="id_1", json_delta='{}'),
            DoneEvent(reason="tool_calls"),
        ]

        with patch("piai.langchain.chat_model.piai_stream", side_effect=_make_async_gen(*events)):
            chunks = [c async for c in model._astream([HumanMessage(content="hi")])]

        # Delta for id_1 should have index 1
        delta_chunks = [
            c for c in chunks
            if c.message.tool_call_chunks and c.message.tool_call_chunks[0]["name"] is None
        ]
        assert delta_chunks[0].message.tool_call_chunks[0]["index"] == 1


# ------------------------------------------------------------------ #
# PiAIChatModel._agenerate                                           #
# ------------------------------------------------------------------ #

class TestAGenerate:

    async def test_returns_ai_message_with_text(self):
        model = PiAIChatModel()
        events = [
            TextDeltaEvent(text="Hello "),
            TextDeltaEvent(text="there"),
            DoneEvent(reason="stop"),
        ]

        with patch("piai.langchain.chat_model.piai_stream", side_effect=_make_async_gen(*events)):
            result = await model._agenerate([HumanMessage(content="hi")])

        assert len(result.generations) == 1
        msg = result.generations[0].message
        assert isinstance(msg, AIMessage)
        assert "Hello" in msg.content
        assert "there" in msg.content

    async def test_returns_ai_message_with_tool_calls(self):
        model = PiAIChatModel()
        tc = ToolCall(id="call_99", name="do_thing")
        events = [
            ToolCallStartEvent(tool_call=tc),
            ToolCallDeltaEvent(id="call_99", json_delta='{"a": 1}'),
            DoneEvent(reason="tool_calls"),
        ]

        with patch("piai.langchain.chat_model.piai_stream", side_effect=_make_async_gen(*events)):
            result = await model._agenerate([HumanMessage(content="do something")])

        msg = result.generations[0].message
        assert isinstance(msg, AIMessage)
        assert len(msg.tool_calls) > 0
        assert msg.tool_calls[0]["name"] == "do_thing"

    async def test_empty_stream_returns_empty_ai_message(self):
        model = PiAIChatModel()

        async def _empty_gen(*args, **kwargs):
            return
            yield  # pragma: no cover

        with patch("piai.langchain.chat_model.piai_stream", side_effect=_empty_gen):
            result = await model._agenerate([HumanMessage(content="hi")])

        assert len(result.generations) == 1
        msg = result.generations[0].message
        assert isinstance(msg, AIMessage)
        assert msg.content == ""

    async def test_generation_info_finish_reason_propagated(self):
        model = PiAIChatModel()
        events = [TextDeltaEvent(text="ok"), DoneEvent(reason="length")]

        with patch("piai.langchain.chat_model.piai_stream", side_effect=_make_async_gen(*events)):
            result = await model._agenerate([HumanMessage(content="hi")])

        gen_info = result.generations[0].generation_info
        assert gen_info is not None
        assert gen_info["finish_reason"] == "length"

    async def test_tools_passed_via_kwargs_reach_stream(self):
        model = PiAIChatModel()
        events = [DoneEvent(reason="stop")]

        raw_tools = [{"name": "my_tool", "description": "desc", "parameters": {}}]
        captured = {}

        async def capturing_stream(model_name, ctx, opts, provider_id):
            captured["tools"] = ctx.tools
            yield DoneEvent(reason="stop")

        with patch("piai.langchain.chat_model.piai_stream", side_effect=capturing_stream):
            await model._agenerate([HumanMessage(content="hi")], tools=raw_tools)

        assert captured["tools"] is not None
        assert captured["tools"][0].name == "my_tool"

    async def test_instance_options_merged_with_call_options(self):
        model = PiAIChatModel(options={"temperature": 0.5})
        captured = {}

        async def capturing_stream(model_name, ctx, opts, provider_id):
            captured["opts"] = opts
            yield DoneEvent(reason="stop")

        with patch("piai.langchain.chat_model.piai_stream", side_effect=capturing_stream):
            await model._agenerate([HumanMessage(content="hi")], options={"max_tokens": 100})

        assert captured["opts"]["temperature"] == 0.5
        assert captured["opts"]["max_tokens"] == 100


# ------------------------------------------------------------------ #
# PiAIChatModel._generate (sync wrapper)                             #
# ------------------------------------------------------------------ #

class TestGenerate:

    def test_sync_generate_returns_chat_result(self):
        model = PiAIChatModel()
        events = [TextDeltaEvent(text="Sync works"), DoneEvent(reason="stop")]

        with patch("piai.langchain.chat_model.piai_stream", side_effect=_make_async_gen(*events)):
            result = model._generate([HumanMessage(content="test")])

        msg = result.generations[0].message
        assert isinstance(msg, AIMessage)
        assert "Sync works" in msg.content

    def test_sync_generate_with_tool_calls(self):
        model = PiAIChatModel()
        tc = ToolCall(id="sync_tc", name="sync_tool")
        events = [
            ToolCallStartEvent(tool_call=tc),
            ToolCallDeltaEvent(id="sync_tc", json_delta='{}'),
            DoneEvent(reason="tool_calls"),
        ]

        with patch("piai.langchain.chat_model.piai_stream", side_effect=_make_async_gen(*events)):
            result = model._generate([HumanMessage(content="test")])

        msg = result.generations[0].message
        assert len(msg.tool_calls) > 0
        assert msg.tool_calls[0]["name"] == "sync_tool"


# ------------------------------------------------------------------ #
# PiAIChatModel.bind_tools                                           #
# ------------------------------------------------------------------ #

class TestBindTools:

    def test_bind_tools_returns_runnable_with_tools(self):
        model = PiAIChatModel()

        # Use a plain dict that convert_to_openai_tool accepts — or pass a
        # pre-formatted OpenAI tool dict (bind_tools wraps with convert_to_openai_tool).
        from langchain_core.tools import tool as lc_tool

        @lc_tool
        def add(x: int, y: int) -> int:
            """Add two numbers."""
            return x + y

        bound = model.bind_tools([add])
        # bind() returns a RunnableBinding whose kwargs hold the tools
        assert "tools" in bound.kwargs
        formatted_tools = bound.kwargs["tools"]
        names = [t["function"]["name"] for t in formatted_tools]
        assert "add" in names

    def test_bind_tools_with_tool_choice(self):
        model = PiAIChatModel()

        from langchain_core.tools import tool as lc_tool

        @lc_tool
        def subtract(x: int, y: int) -> int:
            """Subtract y from x."""
            return x - y

        bound = model.bind_tools([subtract], tool_choice="auto")
        assert bound.kwargs.get("tool_choice") == "auto"

    def test_bind_tools_empty_list(self):
        model = PiAIChatModel()
        bound = model.bind_tools([])
        assert bound.kwargs["tools"] == []


# ------------------------------------------------------------------ #
# PiAIChatModel properties                                           #
# ------------------------------------------------------------------ #

class TestModelProperties:

    def test_llm_type(self):
        model = PiAIChatModel()
        assert model._llm_type == "pi-ai"

    def test_identifying_params_default(self):
        model = PiAIChatModel()
        params = model._identifying_params
        assert params["model_name"] == "gpt-5.1-codex-mini"
        assert params["provider_id"] == "openai-codex"

    def test_identifying_params_custom(self):
        model = PiAIChatModel(model_name="gpt-4o", provider_id="custom-provider")
        params = model._identifying_params
        assert params["model_name"] == "gpt-4o"
        assert params["provider_id"] == "custom-provider"

    def test_default_model_name(self):
        model = PiAIChatModel()
        assert model.model_name == "gpt-5.1-codex-mini"

    def test_custom_model_name(self):
        model = PiAIChatModel(model_name="gpt-4o-mini")
        assert model.model_name == "gpt-4o-mini"

    def test_default_provider_id(self):
        model = PiAIChatModel()
        assert model.provider_id == "openai-codex"

    def test_options_default_empty(self):
        model = PiAIChatModel()
        assert model.options == {}

    def test_options_stored(self):
        model = PiAIChatModel(options={"temperature": 0.7})
        assert model.options["temperature"] == 0.7


# ------------------------------------------------------------------ #
# PiAIChatModel.with_structured_output                               #
# ------------------------------------------------------------------ #

class TestWithStructuredOutput:
    """Tests for with_structured_output() — behavior, not just presence."""

    def _make_tool_call_stream(self, schema_name: str, args_json: str):
        """
        Helper: returns an async gen that simulates the model calling
        `schema_name` tool with `args_json` as the arguments string.
        """
        import json

        from piai.types import ToolCall, ToolCallDeltaEvent, ToolCallEndEvent

        tc = ToolCall(id="call_struct_001", name=schema_name, input=json.loads(args_json))
        return _make_async_gen(
            ToolCallStartEvent(tool_call=tc),
            ToolCallDeltaEvent(id="call_struct_001", json_delta=args_json),
            ToolCallEndEvent(tool_call=tc),
            DoneEvent(reason="tool_use"),
        )

    def test_returns_pydantic_instance(self):
        """with_structured_output() should return a Pydantic model instance, not an AIMessage."""
        from pydantic import BaseModel

        class Answer(BaseModel):
            value: int
            explanation: str

        model = PiAIChatModel()
        chain = model.with_structured_output(Answer)

        stream = self._make_tool_call_stream("Answer", '{"value": 42, "explanation": "forty-two"}')
        with patch("piai.langchain.chat_model.piai_stream", side_effect=stream):
            result = chain.invoke([HumanMessage(content="What is the answer?")])

        assert isinstance(result, Answer)
        assert result.value == 42
        assert result.explanation == "forty-two"

    def test_include_raw_single_llm_call(self):
        """
        include_raw=True must NOT invoke the LLM twice.
        The model is called once; output fans to both 'raw' and 'parsed' branches.
        """
        from pydantic import BaseModel

        class Simple(BaseModel):
            result: str

        model = PiAIChatModel()
        chain = model.with_structured_output(Simple, include_raw=True)

        stream = self._make_tool_call_stream("Simple", '{"result": "hello"}')

        call_count = 0

        async def counting_stream(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            async for event in stream(*args, **kwargs):
                yield event

        with patch("piai.langchain.chat_model.piai_stream", side_effect=counting_stream):
            result = chain.invoke([HumanMessage(content="test")])

        # LLM must be called exactly once even with include_raw=True
        assert call_count == 1, f"LLM was called {call_count} times — expected 1"

    def test_include_raw_returns_both_keys(self):
        """include_raw=True should return dict with 'raw' (AIMessage) and 'parsed' (schema)."""
        from pydantic import BaseModel

        class Tag(BaseModel):
            label: str

        model = PiAIChatModel()
        chain = model.with_structured_output(Tag, include_raw=True)

        stream = self._make_tool_call_stream("Tag", '{"label": "critical"}')
        with patch("piai.langchain.chat_model.piai_stream", side_effect=stream):
            result = chain.invoke([HumanMessage(content="tag this")])

        assert isinstance(result, dict), "include_raw=True should return a dict"
        assert "raw" in result, "dict must have 'raw' key"
        assert "parsed" in result, "dict must have 'parsed' key"
        assert isinstance(result["parsed"], Tag)
        assert result["parsed"].label == "critical"
        # raw should be an AIMessage
        assert isinstance(result["raw"], AIMessage)

    def test_json_mode_raises_not_implemented(self):
        """method='json_mode' must raise NotImplementedError — not silently fail."""
        model = PiAIChatModel()
        with pytest.raises(NotImplementedError, match="json_mode"):
            model.with_structured_output(dict, method="json_mode")

    def test_unknown_method_raises_value_error(self):
        """An unrecognised method string must raise ValueError, not AttributeError."""
        model = PiAIChatModel()
        with pytest.raises(ValueError, match="Unknown method"):
            model.with_structured_output(dict, method="xml_mode")

    def test_typed_dict_schema_returns_dict(self):
        """TypedDict schema should parse to dict (not raise parser errors)."""

        class TargetSummary(TypedDict):
            library_name: str
            targets_found: int

        model = PiAIChatModel()
        chain = model.with_structured_output(TargetSummary)

        stream = self._make_tool_call_stream(
            "TargetSummary",
            '{"library_name": "libfoo.so", "targets_found": 2}',
        )
        with patch("piai.langchain.chat_model.piai_stream", side_effect=stream):
            result = chain.invoke([HumanMessage(content="Summarize findings")])

        assert isinstance(result, dict)
        assert result["library_name"] == "libfoo.so"
        assert result["targets_found"] == 2

    def test_openai_tool_dict_schema_returns_dict(self):
        """OpenAI-style tool schema dict should parse to dict output."""
        schema = {
            "name": "FindingSchema",
            "description": "Structured finding summary",
            "parameters": {
                "type": "object",
                "properties": {
                    "library": {"type": "string"},
                    "verdict": {"type": "string"},
                },
                "required": ["library", "verdict"],
            },
        }

        model = PiAIChatModel()
        chain = model.with_structured_output(schema)

        stream = self._make_tool_call_stream(
            "FindingSchema",
            '{"library": "libssl.so", "verdict": "candidate"}',
        )
        with patch("piai.langchain.chat_model.piai_stream", side_effect=stream):
            result = chain.invoke([HumanMessage(content="Return finding schema output")])

        assert isinstance(result, dict)
        assert result["library"] == "libssl.so"
        assert result["verdict"] == "candidate"
