"""
Example: Four piai SDK Extensions (offline demo — no real LLM calls)

Demonstrates all four new features using a simulated agent loop:

  1. context_reducer  — trim history to prevent context explosion
  2. AgentTurnEndEvent.usage — token tracking per turn
  3. context_extractor on SubAgentTool — context isolation between tiers
  4. Context.scratchpad — persistent working memory injected into system prompt

Run:
  PYTHONPATH=src .venv/bin/python examples/sdk_extensions_demo.py

This is fully offline — it monkey-patches the internal stream() so you can
see the behavior without ChatGPT credentials or network access.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import patch

from piai.agent import agent
from piai.providers.message_transform import build_request_body
from piai.types import (
    AgentTurnEndEvent,
    AgentToolResultEvent,
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
# Shared fake stream infrastructure                                   #
# ------------------------------------------------------------------ #

def _msg(text: str = "Task complete.", usage: dict | None = None) -> AssistantMessage:
    msg = AssistantMessage(content=[TextContent(text=text)])
    if usage:
        msg.usage.update(usage)
    return msg


def make_tool_then_done_stream(tool_calls: list[tuple[str, dict]], usage: dict | None = None):
    """Returns a factory that emits tool calls on turn 1, then stops on turn 2."""
    call_count = 0

    async def fake_stream(*args, **kwargs) -> AsyncGenerator:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            tcs = [ToolCall(id=f"call_{i}", name=name, input=inp) for i, (name, inp) in enumerate(tool_calls)]
            for tc in tcs:
                yield ToolCallEndEvent(tool_call=tc)
            msg = AssistantMessage(content=[ToolCallContent(tool_calls=tcs)])
            if usage:
                msg.usage.update(usage)
            yield DoneEvent(reason="tool_use", message=msg)
        else:
            yield DoneEvent(reason="stop", message=_msg())

    return fake_stream


# ============================================================ #
# Feature 1: context_reducer                                   #
# ============================================================ #

async def demo_context_reducer() -> None:
    print("\n" + "=" * 60)
    print("FEATURE 1: context_reducer — prevent context explosion")
    print("=" * 60)

    message_counts: list[int] = []
    reducer_triggered = False

    def sliding_window_reducer(ctx: Context) -> Context:
        """Keep only the last 2 messages — simulates a sliding window."""
        nonlocal reducer_triggered
        reducer_triggered = True
        trimmed = ctx.messages[-2:]
        print(f"  [reducer] Context had {len(ctx.messages)} messages → trimmed to {len(trimmed)}")
        return Context(
            messages=trimmed,
            system_prompt=ctx.system_prompt,
            tools=ctx.tools,
            scratchpad=ctx.scratchpad,
        )

    call_count = 0

    async def fake_stream(model_id, ctx, *args, **kwargs) -> AsyncGenerator:
        nonlocal call_count
        call_count += 1
        message_counts.append(len(ctx.messages))
        print(f"  [LLM call {call_count}] Context has {len(ctx.messages)} message(s)")

        if call_count == 1:
            tc = ToolCall(id="c1", name="poll_status", input={"id": "job-42"})
            yield ToolCallEndEvent(tool_call=tc)
            msg = AssistantMessage(content=[ToolCallContent(tool_calls=[tc])])
            yield DoneEvent(reason="tool_use", message=msg)
        elif call_count == 2:
            tc = ToolCall(id="c2", name="poll_status", input={"id": "job-42"})
            yield ToolCallEndEvent(tool_call=tc)
            msg = AssistantMessage(content=[ToolCallContent(tool_calls=[tc])])
            yield DoneEvent(reason="tool_use", message=msg)
        else:
            yield DoneEvent(reason="stop", message=_msg("Job complete."))

    ctx = Context(
        messages=[UserMessage(content="Monitor job-42 until done.")],
        system_prompt="You are a job monitor agent.",
    )

    poll_calls = 0

    def poll_status(id: str) -> str:
        nonlocal poll_calls
        poll_calls += 1
        return f"Job {id}: {'running' if poll_calls < 2 else 'done'}"

    with patch("piai.agent.stream", side_effect=fake_stream):
        result = await agent(
            model_id="gpt-5.1-codex-mini",
            context=ctx,
            local_handlers={"poll_status": poll_status},
            context_reducer=sliding_window_reducer,
        )

    print(f"\n  Result: {result.text}")
    print(f"  LLM calls: {call_count}")
    print(f"  Messages seen per turn: {message_counts}")
    print(f"  Reducer triggered: {reducer_triggered}")

    # Without reducer: turn 3 would see 1 + 1 + 1 (assistant) + 1 (tool) + 1 (assistant) + 1 (tool) = 6 msgs
    # With reducer:    turn 3 sees only 2 messages (last tool result)
    assert message_counts[-1] <= 2, f"Expected reducer to cap messages, got {message_counts[-1]}"
    print("  [OK] Reducer kept context from exploding")


# ============================================================ #
# Feature 2: AgentTurnEndEvent.usage                           #
# ============================================================ #

async def demo_turn_usage() -> None:
    print("\n" + "=" * 60)
    print("FEATURE 2: AgentTurnEndEvent.usage — token tracking")
    print("=" * 60)

    turn_usages: list[dict] = []

    def on_event(event):
        if isinstance(event, AgentTurnEndEvent):
            print(f"  [Turn {event.turn}] usage = {event.usage}")
            turn_usages.append(dict(event.usage))

    call_count = 0

    async def fake_stream(*args, **kwargs) -> AsyncGenerator:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            tc = ToolCall(id="c1", name="check_file", input={"path": "/bin/target"})
            yield ToolCallEndEvent(tool_call=tc)
            msg = AssistantMessage(content=[ToolCallContent(tool_calls=[tc])])
            msg.usage.update({"input": 1200, "output": 80, "cache_read": 400, "total_tokens": 1280})
            yield DoneEvent(reason="tool_use", message=msg)
        else:
            final = _msg("Analysis complete.")
            final.usage.update({"input": 1350, "output": 150, "cache_read": 900, "total_tokens": 1500})
            yield DoneEvent(reason="stop", message=final)

    ctx = Context(messages=[UserMessage(content="Analyze /bin/target")])

    with patch("piai.agent.stream", side_effect=fake_stream):
        await agent(
            model_id="gpt-5.1-codex-mini",
            context=ctx,
            local_handlers={"check_file": lambda path: f"file: {path}, size: 2MB"},
            on_event=on_event,
        )

    print(f"\n  Turn usages captured: {len(turn_usages)}")
    assert len(turn_usages) == 2
    assert turn_usages[0]["input"] == 1200
    assert turn_usages[1]["input"] == 1350

    total_input = sum(u.get("input", 0) for u in turn_usages)
    total_output = sum(u.get("output", 0) for u in turn_usages)
    print(f"  Total input tokens across turns: {total_input}")
    print(f"  Total output tokens across turns: {total_output}")
    print("  [OK] usage surfaced correctly in AgentTurnEndEvent")


# ============================================================ #
# Feature 3: context_extractor in SubAgentTool                 #
# ============================================================ #

async def demo_context_extractor() -> None:
    print("\n" + "=" * 60)
    print("FEATURE 3: context_extractor — context isolation between tiers")
    print("=" * 60)

    from piai.langchain.sub_agent_tool import SubAgentTool

    captured: list[Context] = []

    async def fake_agent(model_id, context, **kwargs) -> AssistantMessage:
        captured.append(context)
        print(f"  [sub-agent] received {len(context.messages)} message(s)")
        print(f"  [sub-agent] system_prompt: {context.system_prompt!r}")
        print(f"  [sub-agent] scratchpad: {context.scratchpad}")
        return _msg("Found 2 vulnerabilities in libssl.so")

    # Simulate a fat parent-orchestrator context
    parent_ctx = Context(
        messages=[
            UserMessage(content="[Global task ledger — 200 entries not shown]"),
            UserMessage(content="Previous scan result for libcrypto.so"),
            UserMessage(content="Previous scan result for libm.so"),
        ],
        system_prompt="Master orchestrator with full state",
        scratchpad={
            "apk_path": "/data/com.example.apk",
            "scan_queue": ["libssl.so", "libxml2.so"],
            "findings": [{"lib": "libcrypto.so", "severity": "high"}],
        },
    )

    def isolating_extractor(ctx: Context) -> Context:
        """Strip orchestrator history; pass only what the sub-agent needs."""
        apk = ctx.scratchpad.get("apk_path", "")
        current_target = ctx.scratchpad.get("scan_queue", ["unknown"])[0]
        print(f"  [extractor] stripping {len(ctx.messages)} parent messages → passing only target info")
        return Context(
            messages=[UserMessage(content=f"Analyze {current_target} from {apk}")],
            system_prompt="Sub-agent: native binary security analyzer",
            scratchpad={"target": current_target, "apk_path": apk},
        )

    tool = SubAgentTool(
        name="binary_analyzer",
        description="Analyzes a native library for vulnerabilities.",
        initial_context=parent_ctx,
        context_extractor=isolating_extractor,
    )

    with patch("piai.langchain.sub_agent_tool.piai_agent", side_effect=fake_agent):
        result = await tool._arun(task="analyze libssl.so")

    print(f"\n  Sub-agent result: {result}")

    ctx_used = captured[0]
    assert len(ctx_used.messages) == 1, f"Expected 1 message, got {len(ctx_used.messages)}"
    assert "libssl.so" in (ctx_used.messages[0].content or "")
    assert ctx_used.system_prompt == "Sub-agent: native binary security analyzer"
    assert ctx_used.scratchpad["target"] == "libssl.so"
    assert "Global task ledger" not in str([m.content for m in ctx_used.messages])
    print("  [OK] Extractor isolated context — parent history not passed to sub-agent")


# ============================================================ #
# Feature 4: Context.scratchpad                                #
# ============================================================ #

async def demo_scratchpad() -> None:
    print("\n" + "=" * 60)
    print("FEATURE 4: Context.scratchpad — persistent working memory")
    print("=" * 60)

    instructions_seen: list[str] = []

    call_count = 0

    async def fake_stream(model_id, ctx, opts=None, provider_id=None, *args, **kwargs) -> AsyncGenerator:
        nonlocal call_count
        call_count += 1

        # Capture what instructions the LLM would see (mirrors what the real stream sends)
        body = build_request_body(model_id, ctx, opts)
        instructions_seen.append(body["instructions"])
        print(f"  [LLM call {call_count}] instructions snippet:")
        for line in body["instructions"].split("\n"):
            print(f"    {line}")

        if call_count == 1:
            tc = ToolCall(id="c1", name="update_scratchpad", input={
                "key": "known_vuln_funcs",
                "value": ["0x400A", "0x400B"],
            })
            yield ToolCallEndEvent(tool_call=tc)
            msg = AssistantMessage(content=[ToolCallContent(tool_calls=[tc])])
            yield DoneEvent(reason="tool_use", message=msg)
        else:
            yield DoneEvent(reason="stop", message=_msg("Scan complete. Found 2 vuln funcs."))

    def update_scratchpad_handler(key: str, value: Any) -> str:
        """Simulates the agent updating its own scratchpad."""
        ctx.scratchpad[key] = value
        return f"scratchpad[{key!r}] updated"

    ctx = Context(
        system_prompt="You are a binary analysis agent.",
        messages=[UserMessage(content="Scan the binary and track vulnerable functions.")],
        scratchpad={"phase": "recon", "known_vuln_funcs": []},
    )

    with patch("piai.agent.stream", side_effect=fake_stream):
        result = await agent(
            model_id="gpt-5.1-codex-mini",
            context=ctx,
            local_handlers={"update_scratchpad": update_scratchpad_handler},
        )

    print(f"\n  Result: {result.text}")
    print(f"  Final scratchpad: {ctx.scratchpad}")

    # Turn 1: scratchpad had initial values
    assert "<scratchpad>" in instructions_seen[0]
    assert '"phase": "recon"' in instructions_seen[0]

    # The handler updated the scratchpad — turn 2 would see the new values
    assert ctx.scratchpad["known_vuln_funcs"] == ["0x400A", "0x400B"]
    print("  [OK] scratchpad injected into instructions each turn; handler updated it successfully")


# ============================================================ #
# Main                                                         #
# ============================================================ #

async def main() -> None:
    print("piai SDK Extensions Demo")
    print("Four new features — fully offline (no LLM calls)")

    await demo_context_reducer()
    await demo_turn_usage()
    await demo_context_extractor()
    await demo_scratchpad()

    print("\n" + "=" * 60)
    print("All four features verified successfully.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
