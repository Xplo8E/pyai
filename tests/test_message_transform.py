"""
Tests for Context → OpenAI Responses API message transformation.
"""

import json

from piai.providers.message_transform import build_request_body, convert_messages, convert_tools
from piai.types import (
    AssistantMessage,
    Context,
    TextContent,
    ThinkingContent,
    Tool,
    ToolCall,
    ToolCallContent,
    ToolResultMessage,
    UserMessage,
)


def test_simple_user_message():
    ctx = Context(messages=[UserMessage(content="Hello")])
    msgs = convert_messages(ctx)
    assert len(msgs) == 1
    assert msgs[0]["type"] == "message"
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"][0]["text"] == "Hello"
    assert msgs[0]["content"][0]["type"] == "input_text"


def test_assistant_text_message():
    msg = AssistantMessage(content=[TextContent(text="Hi there")])
    ctx = Context(messages=[msg])
    msgs = convert_messages(ctx)
    assert len(msgs) == 1
    assert msgs[0]["role"] == "assistant"
    assert msgs[0]["content"][0]["type"] == "output_text"
    assert msgs[0]["content"][0]["text"] == "Hi there"


def test_tool_call_becomes_function_call():
    tc = ToolCall(id="call_abc", name="get_weather", input={"city": "London"})
    msg = AssistantMessage(content=[ToolCallContent(tool_calls=[tc])])
    ctx = Context(messages=[msg])
    msgs = convert_messages(ctx)
    assert len(msgs) == 1
    assert msgs[0]["type"] == "function_call"
    assert msgs[0]["name"] == "get_weather"
    assert msgs[0]["call_id"] == "call_abc"
    args = json.loads(msgs[0]["arguments"])
    assert args["city"] == "London"


def test_tool_result_becomes_function_call_output():
    result = ToolResultMessage(tool_call_id="call_abc", content='{"temp": 20}')
    ctx = Context(messages=[result])
    msgs = convert_messages(ctx)
    assert len(msgs) == 1
    assert msgs[0]["type"] == "function_call_output"
    assert msgs[0]["call_id"] == "call_abc"
    assert msgs[0]["output"] == '{"temp": 20}'


def test_thinking_block_wrapped_in_tags():
    msg = AssistantMessage(content=[ThinkingContent(thinking="Let me think...")])
    ctx = Context(messages=[msg])
    msgs = convert_messages(ctx)
    assert "<thinking>Let me think...</thinking>" in msgs[0]["content"][0]["text"]


def test_system_prompt_in_request_body():
    ctx = Context(
        system_prompt="You are helpful.",
        messages=[UserMessage(content="Hi")],
    )
    body = build_request_body("gpt-5.1-codex-mini", ctx)
    assert body["instructions"] == "You are helpful."
    assert body["model"] == "gpt-5.1-codex-mini"
    assert body["stream"] is True
    assert body["store"] is False


def test_tools_converted():
    tool = Tool(
        name="search",
        description="Search the web",
        parameters={"type": "object", "properties": {"query": {"type": "string"}}},
    )
    ctx = Context(messages=[UserMessage(content="Search for X")], tools=[tool])
    body = build_request_body("gpt-5.1-codex-mini", ctx)
    assert "tools" in body
    assert body["tools"][0]["name"] == "search"
    assert body["tools"][0]["type"] == "function"


def test_session_id_becomes_prompt_cache_key():
    ctx = Context(messages=[UserMessage(content="Hi")])
    body = build_request_body("gpt-5.1-codex-mini", ctx, options={"session_id": "sess_abc"})
    assert body["prompt_cache_key"] == "sess_abc"


def test_reasoning_effort_in_body():
    ctx = Context(messages=[UserMessage(content="Hi")])
    body = build_request_body("gpt-5.1-codex-mini", ctx, options={"reasoning_effort": "high"})
    assert body["reasoning"]["effort"] == "high"
    assert body["reasoning"]["summary"] == "auto"
