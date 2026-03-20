"""
LangChain ChatModel wrapper for piai.

Bridges LangChain's BaseChatModel interface to piai's stream() API,
enabling piai (ChatGPT Plus OAuth) to be used anywhere LangChain
accepts a chat model — chains, agents, MCP tool servers, etc.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Iterator, Sequence

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import (
    ChatGeneration,
    ChatGenerationChunk,
    ChatResult,
)

from ..stream import stream as piai_stream
from ..types import (
    AssistantMessage,
    Context,
    DoneEvent,
    ErrorEvent,
    TextContent,
    TextDeltaEvent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    Tool,
    ToolCall,
    ToolCallContent,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    ToolResultMessage,
    UserMessage,
)


def _lc_messages_to_piai(messages: list[BaseMessage]) -> Context:
    """
    Convert LangChain message list to piai Context.

    - SystemMessage  → context.system_prompt
    - HumanMessage   → UserMessage
    - AIMessage      → AssistantMessage (text + tool calls)
    - ToolMessage    → ToolResultMessage
    """
    system_prompt: str | None = None
    piai_msgs = []

    for msg in messages:
        if msg.type == "system":
            system_prompt = msg.content if isinstance(msg.content, str) else str(msg.content)

        elif msg.type == "human":
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            piai_msgs.append(UserMessage(content=content))

        elif msg.type == "ai":
            blocks = []
            if msg.content:
                if isinstance(msg.content, str):
                    text = msg.content
                elif isinstance(msg.content, list):
                    # Extract text from content blocks (e.g. [{"type": "text", "text": "..."}])
                    parts = []
                    for item in msg.content:
                        if isinstance(item, str):
                            parts.append(item)
                        elif isinstance(item, dict) and item.get("type") == "text":
                            parts.append(item.get("text", ""))
                        elif isinstance(item, dict):
                            parts.append(str(item.get("text", item)))
                    text = "".join(parts)
                else:
                    text = str(msg.content)
                if text:
                    blocks.append(TextContent(text=text))
            if msg.tool_calls:
                tcs = [
                    ToolCall(id=tc["id"], name=tc["name"], input=tc["args"] or {})
                    for tc in msg.tool_calls
                ]
                blocks.append(ToolCallContent(tool_calls=tcs))
            piai_msgs.append(AssistantMessage(content=blocks))

        elif msg.type == "tool":
            piai_msgs.append(
                ToolResultMessage(
                    tool_call_id=msg.tool_call_id,
                    content=msg.content if isinstance(msg.content, str) else str(msg.content),
                )
            )

    return Context(system_prompt=system_prompt, messages=piai_msgs)


def _lc_tools_to_piai(tools: list[dict]) -> list[Tool]:
    """Convert OpenAI-format tool dicts (from bind_tools) to piai Tool objects."""
    result = []
    for t in tools:
        fn = t.get("function", t)  # handle both {function: {...}} and flat
        result.append(Tool(
            name=fn["name"],
            description=fn.get("description", ""),
            parameters=fn.get("parameters", {}),
        ))
    return result


class PiAIChatModel(BaseChatModel):
    """
    LangChain ChatModel backed by piai (ChatGPT Plus OAuth).

    Args:
        model_name:        GPT model to use. Default: "gpt-5.1-codex-mini"
        provider_id:       OAuth provider. Default: "openai-codex"
        options:           Extra options passed to piai stream()
                           (reasoning_effort, session_id, temperature, etc.)

    Example:
        llm = PiAIChatModel(model_name="gpt-5.1-codex-mini")
        result = llm.invoke([HumanMessage(content="What is 2+2?")])

        # With tools (LangChain handles MCP, agents, etc.)
        llm_with_tools = llm.bind_tools([my_tool])
        result = llm_with_tools.invoke([HumanMessage(content="What's the weather?")])
    """

    model_name: str = "gpt-5.1-codex-mini"
    provider_id: str = "openai-codex"
    options: dict[str, Any] = {}

    @property
    def _llm_type(self) -> str:
        return "pi-ai"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model_name": self.model_name, "provider_id": self.provider_id}

    # ------------------------------------------------------------------ #
    # Sync interface (runs async in a new event loop)                     #
    # ------------------------------------------------------------------ #

    def _run_async(self, coro: Any) -> Any:
        """
        Run an async coroutine from a sync context, safely handling the case
        where an event loop is already running (e.g. inside LangGraph).

        LangGraph runs its graph execution inside an async event loop, so
        asyncio.run() fails with 'cannot be called from a running event loop'.
        In that case we dispatch to a thread with its own fresh event loop.
        """
        import concurrent.futures

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            # Already inside an event loop (LangGraph, Jupyter, etc.)
            # Run in a thread with its own event loop.
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        else:
            return asyncio.run(coro)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._run_async(self._agenerate(messages, stop=stop, **kwargs))

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        async def _collect():
            chunks = []
            async for chunk in self._astream(messages, stop=stop, **kwargs):
                chunks.append(chunk)
            return chunks

        for chunk in self._run_async(_collect()):
            yield chunk

    # ------------------------------------------------------------------ #
    # Async interface (native — preferred)                                #
    # ------------------------------------------------------------------ #

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        chunks: list[ChatGenerationChunk] = []
        async for chunk in self._astream(messages, stop=stop, **kwargs):
            chunks.append(chunk)

        if not chunks:
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=""))])

        # Merge all chunks into final message
        merged = chunks[0]
        for c in chunks[1:]:
            merged = merged + c

        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content=merged.message.content or "",
                        tool_calls=list(merged.message.tool_calls) if merged.message.tool_calls else [],
                    ),
                    generation_info=merged.generation_info,
                )
            ]
        )

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        ctx = _lc_messages_to_piai(messages)

        # Tools passed via bind_tools → self.bind(tools=...) → arrive in kwargs
        raw_tools = kwargs.get("tools")
        if raw_tools:
            ctx.tools = _lc_tools_to_piai(raw_tools)

        # Merge instance options with call-time options
        opts = {**self.options}
        if "options" in kwargs:
            opts.update(kwargs["options"])

        # Forward tool_choice from bind_tools() into the request options
        raw_tool_choice = kwargs.get("tool_choice")
        if raw_tool_choice:
            opts["tool_choice"] = raw_tool_choice

        # Track tool call index for LangChain chunk merging
        tc_index: dict[str, int] = {}
        # Map piai tool_call.id → truncated id safe for the API (max 64 chars)
        tc_id_map: dict[str, str] = {}
        # Accumulate thinking blocks for this response
        thinking_parts: list[str] = []

        async for event in piai_stream(self.model_name, ctx, opts, self.provider_id):
            if isinstance(event, TextDeltaEvent):
                yield ChatGenerationChunk(message=AIMessageChunk(content=event.text))

            elif isinstance(event, ThinkingDeltaEvent):
                # Stream thinking tokens as additional_kwargs so callers can observe live
                yield ChatGenerationChunk(
                    message=AIMessageChunk(
                        content="",
                        additional_kwargs={"thinking_delta": event.thinking},
                    )
                )

            elif isinstance(event, ThinkingEndEvent):
                # Full thinking block completed — accumulate
                thinking_parts.append(event.thinking)

            elif isinstance(event, ToolCallStartEvent):
                idx = len(tc_index)
                safe_id = event.tool_call.id[:64]
                tc_index[event.tool_call.id] = idx
                tc_id_map[event.tool_call.id] = safe_id
                yield ChatGenerationChunk(
                    message=AIMessageChunk(
                        content="",
                        tool_call_chunks=[{
                            "name": event.tool_call.name,
                            "args": "",
                            "id": safe_id,
                            "index": idx,
                            "type": "tool_call_chunk",
                        }],
                    )
                )

            elif isinstance(event, ToolCallDeltaEvent):
                idx = tc_index.get(event.id, 0)
                yield ChatGenerationChunk(
                    message=AIMessageChunk(
                        content="",
                        tool_call_chunks=[{
                            "name": None,
                            "args": event.json_delta,
                            "id": None,
                            "index": idx,
                            "type": "tool_call_chunk",
                        }],
                    )
                )

            elif isinstance(event, ToolCallEndEvent):
                # Emit a final chunk with the canonical parsed args from the done event.
                # This cross-checks the accumulated delta JSON and handles the edge case
                # where the last delta was empty or missing.
                idx = tc_index.get(event.tool_call.id, 0)
                import json as _json
                canonical_args = _json.dumps(event.tool_call.input or {})
                yield ChatGenerationChunk(
                    message=AIMessageChunk(
                        content="",
                        tool_call_chunks=[{
                            "name": None,
                            "args": "",  # Already accumulated via deltas; this is a no-op for merging
                            "id": None,
                            "index": idx,
                            "type": "tool_call_chunk",
                        }],
                        additional_kwargs={"tool_call_args_canonical": {event.tool_call.id: canonical_args}},
                    )
                )

            elif isinstance(event, DoneEvent):
                extra: dict[str, Any] = {}
                if thinking_parts:
                    extra["thinking"] = "\n\n".join(thinking_parts)
                yield ChatGenerationChunk(
                    message=AIMessageChunk(content="", additional_kwargs=extra),
                    generation_info={"finish_reason": event.reason},
                )

            elif isinstance(event, ErrorEvent):
                raise RuntimeError(event.error.error_message or "piai stream error")

    # ------------------------------------------------------------------ #
    # Tool binding                                                        #
    # ------------------------------------------------------------------ #

    def bind_tools(
        self,
        tools: Sequence[Any],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> "PiAIChatModel":
        """
        Bind tools to this model instance.

        Accepts anything LangChain's convert_to_openai_tool() understands:
        BaseTool instances, dicts, Pydantic models, or plain functions.
        """
        from langchain_core.utils.function_calling import convert_to_openai_tool

        formatted = [convert_to_openai_tool(t) for t in tools]
        bound_kwargs: dict[str, Any] = {"tools": formatted}
        if tool_choice:
            bound_kwargs["tool_choice"] = tool_choice
        return self.bind(**bound_kwargs)  # type: ignore[return-value]
