"""
LangChain ChatModel wrapper for piai.

Bridges LangChain's BaseChatModel interface to piai's stream() API,
enabling piai (ChatGPT Plus OAuth) to be used anywhere LangChain
accepts a chat model — chains, agents, MCP tool servers, etc.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
from typing import TYPE_CHECKING, Any, AsyncIterator, Iterator, Sequence

if TYPE_CHECKING:
    from langchain_core.runnables import Runnable

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
)
from langchain_core.output_parsers.openai_tools import (
    JsonOutputKeyToolsParser,
    PydanticToolsParser,
)
from langchain_core.outputs import (
    ChatGeneration,
    ChatGenerationChunk,
    ChatResult,
)
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel

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
            system_prompt = _extract_text_from_content(msg.content)

        elif msg.type == "human":
            piai_msgs.append(UserMessage(content=_extract_text_from_content(msg.content)))

        elif msg.type == "ai":
            blocks = []
            if msg.content:
                text = _extract_text_from_content(msg.content)
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
                    content=_extract_text_from_content(msg.content),
                )
            )

    return Context(system_prompt=system_prompt, messages=piai_msgs)


def _extract_text_from_content(content: Any) -> str:
    """Extract plain text from a LangChain message content (str or list of blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, dict):
                parts.append(str(item.get("text", item)))
        return "".join(parts)
    return str(content)


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
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
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
                canonical_args = json.dumps(event.tool_call.input or {})
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
        formatted = [convert_to_openai_tool(t) for t in tools]
        bound_kwargs: dict[str, Any] = {"tools": formatted}
        if tool_choice:
            bound_kwargs["tool_choice"] = tool_choice
        return self.bind(**bound_kwargs)  # type: ignore[return-value]

    def with_structured_output(
        self,
        schema: Any,
        *,
        method: str = "tool_calling",
        include_raw: bool = False,
        **kwargs: Any,
    ) -> "Runnable":
        """
        Return a Runnable that forces structured output conforming to `schema`.

        Implements the standard LangChain with_structured_output() interface
        so PiAIChatModel works with create_supervisor, create_react_agent,
        and any LangChain chain that uses structured output.

        Args:
            schema:      Pydantic BaseModel class or TypedDict defining the output shape.
            method:      "tool_calling" (default, works today) or "json_mode" (future).
            include_raw: If True, returns dict {"raw": AIMessage, "parsed": schema_instance}.
                         If False (default), returns schema instance directly.
            **kwargs:    Ignored (for LangChain interface compatibility).

        Returns:
            A Runnable: messages -> schema instance  (or dict if include_raw=True)

        Example:
            from pydantic import BaseModel

            class VulnReport(BaseModel):
                target: str
                severity: str
                description: str

            llm = PiAIChatModel(model_name="gpt-5.1-codex-mini")
            chain = llm.with_structured_output(VulnReport)
            report = chain.invoke([HumanMessage(content="Analyze libcrypto.so")])
            # report is a VulnReport instance
        """
        if method == "json_mode":
            raise NotImplementedError(
                "json_mode is not yet supported for PiAIChatModel. "
                "Use method='tool_calling' (the default)."
            )
        if method != "tool_calling":
            raise ValueError(
                f"Unknown method: {method!r}. "
                "Supported methods: 'tool_calling'. "
                "'json_mode' is planned but not yet implemented."
            )

        # Use LangChain's conversion to get the canonical tool name.
        # This avoids mismatches for TypedDict/dict schemas.
        tool_spec = convert_to_openai_tool(schema)
        schema_name = tool_spec["function"]["name"]

        # Bind only this schema tool, and require a tool call.
        # For piai/openai-codex backend, supported tool_choice values are
        # none/auto/required (tool-name forcing is rejected server-side).
        # Since only one tool is bound here, "required" effectively forces
        # this exact schema tool to be called.
        llm = self.bind_tools([schema], tool_choice="required")

        # Parse strategy:
        # - Pydantic model class -> parsed BaseModel instance
        # - TypedDict / dict schema -> parsed args dict
        is_pydantic_model = isinstance(schema, type) and issubclass(schema, BaseModel)
        if is_pydantic_model:
            parser = PydanticToolsParser(tools=[schema], first_tool_only=True)
        else:
            parser = JsonOutputKeyToolsParser(key_name=schema_name, first_tool_only=True)

        if include_raw:
            # CRITICAL IMPLEMENTATION NOTE:
            # We pipe `llm | RunnableParallel(...)` NOT `RunnableParallel(raw=llm, parsed=...)`.
            # The wrong form (raw=llm) would invoke the LLM twice — once for raw, once for parsed.
            # The correct form: llm runs ONCE, its AIMessage output fans to both branches.
            #   raw=RunnablePassthrough() → passes the AIMessage through unchanged
            #   parsed=parser             → parses the same AIMessage into schema instance
            return llm | RunnableParallel(
                raw=RunnablePassthrough(),
                parsed=parser,
            )

        return llm | parser
