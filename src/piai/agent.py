"""
agent() — autonomous agentic loop with MCP tool support.

Connects to MCP servers, discovers tools, runs the model in a loop,
executes tool calls, and continues until the model stops or max_turns is reached.

No LangChain, no LangGraph, no mcpo required. Just piai + MCP.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Callable, Coroutine
from typing import Any

from .mcp.hub import MCPHub
from .mcp.server import MCPServer
from .stream import stream
from .types import (
    AgentToolCallEvent,
    AgentToolResultEvent,
    AgentTurnEndEvent,
    AssistantMessage,
    Context,
    DoneEvent,
    ErrorEvent,
    StreamEvent,
    TextDeltaEvent,
    ThinkingContent,
    ToolCall,
    ToolCallEndEvent,
    ToolResultMessage,
)

logger = logging.getLogger(__name__)

OPENAI_CODEX_PROVIDER = "openai-codex"


async def agent(
    model_id: str,
    context: Context,
    mcp_servers: list[MCPServer] | None = None,
    options: dict[str, Any] | None = None,
    provider_id: str = OPENAI_CODEX_PROVIDER,
    max_turns: int = 20,
    on_event: Callable[[StreamEvent], Any] | None = None,
    require_all_servers: bool = False,
    connect_timeout: float = 60.0,
    tool_result_max_chars: int = 32_000,
    local_handlers: dict[str, Callable[..., Any]] | None = None,
) -> AssistantMessage:
    """
    Run an autonomous agentic loop, optionally with MCP tool servers.

    Connects to all provided MCP servers concurrently, auto-discovers their tools,
    injects them into the context, and runs the model in a loop — executing tool
    calls and feeding results back — until the model stops or max_turns is reached.

    Can also be used without MCP servers as a simple streaming loop.

    Args:
        model_id:            Model to use, e.g. "gpt-5.1-codex-mini"
        context:             Conversation context (messages + optional system_prompt + tools)
        mcp_servers:         MCP server configs. All connected concurrently.
                             Pass None or [] to run without MCP tools.
        options:             Provider options (reasoning_effort, session_id, etc.)
        provider_id:         Defaults to "openai-codex"
        max_turns:           Safety limit on agentic loop iterations. Default: 20.
        on_event:            Optional callback for every StreamEvent (sync or async).
                             Use this for live streaming output.
        require_all_servers: If True, raise if any MCP server fails to connect.
                             Default False — partial connections allowed.
        connect_timeout:     Per-server connection timeout in seconds. Default 60.
        tool_result_max_chars: Max chars per tool result fed back to model. Default 32000.

    Returns:
        The final AssistantMessage after the model stops.

    Examples:
        # With MCP tools
        from piai import agent
        from piai.mcp import MCPServer
        from piai.types import Context, UserMessage, TextDeltaEvent

        ctx = Context(messages=[UserMessage(content="Analyze /lib/target.so")])

        result = await agent(
            model_id="gpt-5.1-codex-mini",
            context=ctx,
            mcp_servers=[
                MCPServer.stdio("r2pm -r r2mcp"),
                MCPServer.stdio("ida-mcp", name="ida"),
            ],
            options={"reasoning_effort": "medium"},
            max_turns=30,
            on_event=lambda e: print(e.text, end="") if isinstance(e, TextDeltaEvent) else None,
        )

        # Without MCP (simple agentic loop)
        result = await agent(
            model_id="gpt-5.1-codex-mini",
            context=ctx,
        )
    """
    servers = mcp_servers or []

    if servers:
        hub = MCPHub(
            servers,
            require_all=require_all_servers,
            connect_timeout=connect_timeout,
            tool_result_max_chars=tool_result_max_chars,
        )
        async with hub:
            return await _run_loop(
                model_id=model_id,
                context=context,
                hub=hub,
                options=options,
                provider_id=provider_id,
                max_turns=max_turns,
                on_event=on_event,
                tool_result_max_chars=tool_result_max_chars,
                local_handlers=local_handlers,
            )
    else:
        # No MCP servers — run loop without tool injection
        return await _run_loop(
            model_id=model_id,
            context=context,
            hub=None,
            options=options,
            provider_id=provider_id,
            max_turns=max_turns,
            on_event=on_event,
            tool_result_max_chars=tool_result_max_chars,
            local_handlers=local_handlers,
        )


async def _run_loop(
    model_id: str,
    context: Context,
    hub: MCPHub | None,
    options: dict[str, Any] | None,
    provider_id: str,
    max_turns: int,
    on_event: Callable[[StreamEvent], Any] | None,
    tool_result_max_chars: int,
    local_handlers: dict[str, Callable[..., Any]] | None = None,
) -> AssistantMessage:
    """Internal agentic loop."""
    # Build working context — inject MCP tools if available
    if hub is not None:
        mcp_tools = hub.all_tools()
        if not mcp_tools:
            logger.warning("No tools discovered from MCP servers. Running without tools.")
        # Merge MCP tools with any pre-defined context tools (MCP takes priority on name collision)
        existing_names = {t.name for t in mcp_tools}
        extra_tools = [t for t in (context.tools or []) if t.name not in existing_names]
        merged = mcp_tools + extra_tools
        combined_tools = merged if merged else None
        ctx = Context(
            messages=list(context.messages),
            system_prompt=context.system_prompt,
            tools=combined_tools,
        )
    else:
        ctx = Context(
            messages=list(context.messages),
            system_prompt=context.system_prompt,
            tools=context.tools,
        )

    # Warn if any tool is visible to the model but has no executor.
    # Tools are executable if they are in local_handlers OR discoverable via the hub.
    if ctx.tools:
        handled = set(local_handlers.keys()) if local_handlers else set()
        hub_names = set(hub.tool_names()) if hub is not None else set()
        unexecutable = [t.name for t in ctx.tools if t.name not in handled and t.name not in hub_names]
        if unexecutable:
            logger.warning(
                "Tools visible to model but have no executor (not in local_handlers and not in MCP): %s. "
                "The model may call these tools and receive 'Tool not found' errors. "
                "Pass mcp_servers or local_handlers to handle them.",
                unexecutable,
            )

    opts = dict(options or {})
    final_message: AssistantMessage | None = None

    for turn in range(max_turns):
        logger.debug("Agent turn %d/%d", turn + 1, max_turns)

        tool_calls_made: list[ToolCall] = []
        done_event: DoneEvent | None = None

        async for event in stream(model_id, ctx, opts, provider_id):
            await _fire_event(on_event, event)

            if isinstance(event, ToolCallEndEvent):
                tool_calls_made.append(event.tool_call)
            elif isinstance(event, DoneEvent):
                done_event = event
                final_message = event.message
            elif isinstance(event, ErrorEvent):
                raise RuntimeError(
                    event.error.error_message or "piai stream error"
                )

        if done_event is None:
            raise RuntimeError("Stream ended without a done event")

        # Collect thinking from this turn's message
        turn_thinking = final_message.thinking if final_message else None

        # Fire turn-end summary event (useful for observability dashboards)
        await _fire_event(on_event, AgentTurnEndEvent(
            turn=turn + 1,
            thinking=turn_thinking,
            tool_calls=list(tool_calls_made),
        ))

        # No tool calls → model is done
        if not tool_calls_made:
            logger.debug("No tool calls — agent complete after %d turn(s)", turn + 1)
            break

        # Append assistant turn to context
        ctx.messages.append(final_message)

        # Execute all tool calls, firing observability events around each
        for tc in tool_calls_made:
            await _fire_event(on_event, AgentToolCallEvent(
                turn=turn + 1,
                tool_name=tc.name,
                tool_input=tc.input or {},
            ))

            result, is_error = await _execute_tool(hub, tc, tool_result_max_chars, local_handlers)

            await _fire_event(on_event, AgentToolResultEvent(
                turn=turn + 1,
                tool_name=tc.name,
                tool_input=tc.input or {},
                result=result,
                error=is_error,
            ))

            ctx.messages.append(
                ToolResultMessage(
                    tool_call_id=tc.id,
                    content=result,
                )
            )

        logger.debug(
            "Turn %d complete: %d tool call(s) executed, continuing...",
            turn + 1,
            len(tool_calls_made),
        )

    else:
        logger.warning(
            "Agent reached max_turns=%d without stopping. Returning last message.",
            max_turns,
        )

    if final_message is None:
        raise RuntimeError("Agent loop ended without a final message")

    return final_message


async def _execute_tool(
    hub: MCPHub | None,
    tc: ToolCall,
    max_chars: int,
    local_handlers: dict[str, Callable[..., Any]] | None = None,
) -> tuple[str, bool]:
    """Execute a tool call and return (result_string, is_error). Never raises.

    Returns a tuple so callers get a structured error flag instead of relying on
    fragile string-prefix detection.

    local_handlers: optional dict of tool_name -> callable for tools that
    should be handled locally rather than forwarded to an MCP server.
    Callables may be sync or async; they receive the tool input dict as kwargs.
    local_handlers take priority over MCP when names conflict.
    """
    # Normalise input — always pass a dict, never None or other falsy types
    input_args: dict[str, Any] = tc.input if isinstance(tc.input, dict) else {}

    # Check local handlers first (takes priority over MCP on name collision)
    if local_handlers and tc.name in local_handlers:
        handler = local_handlers[tc.name]
        if hub is not None and tc.name in hub.tool_names():
            logger.debug(
                "Tool %r has both a local_handler and an MCP entry; local_handler wins.",
                tc.name,
            )
        try:
            result = handler(**input_args)
            if inspect.isawaitable(result):
                result = await result
            return (str(result) if result is not None else "ok", False)
        except Exception as e:
            msg = f"Tool {tc.name!r} failed: {e}"
            logger.warning(msg)
            return (msg, True)

    if hub is None:
        msg = f"No MCP servers configured — cannot execute tool {tc.name!r}."
        logger.warning(msg)
        return (msg, True)

    logger.debug("Calling tool %r with args: %s", tc.name, input_args)
    try:
        result = await hub.call_tool(tc.name, input_args)
        logger.debug("Tool %r returned %d chars", tc.name, len(result))
        # MCP tool errors are returned as "Tool error: ..." strings from MCPClient
        is_error = result.startswith("Tool error:")
        return (result, is_error)
    except KeyError as e:
        msg = f"Tool not found: {e}"
        logger.warning(msg)
        return (msg, True)
    except Exception as e:
        msg = f"Tool {tc.name!r} failed: {e}"
        logger.warning(msg)
        return (msg, True)


async def _fire_event(
    callback: Callable[[StreamEvent], Any] | None,
    event: StreamEvent,
) -> None:
    """Fire the on_event callback, supporting both sync and async callbacks."""
    if callback is None:
        return
    result = callback(event)
    if inspect.isawaitable(result):
        await result
