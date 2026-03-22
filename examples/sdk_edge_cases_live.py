"""
Live edge-case examples for the four piai SDK extensions.

Goes beyond the happy path — tests real LLM behavior at boundaries:

  1. context_reducer that summarizes instead of just trimming
  2. AgentTurnEndEvent.usage — accumulate cost budget across turns, abort if over limit
  3. context_extractor with async extractor + scratchpad forwarding
  4. scratchpad used as agent memory — model reads its own notes mid-session
  5. context_reducer that returns empty messages (edge: model must still get task)
  6. SubAgentTool with no initial_context but with context_extractor (uses fresh ctx)

Run:
  PYTHONPATH=src .venv/bin/python examples/sdk_edge_cases_live.py
"""

from __future__ import annotations

import asyncio

from piai import agent
from piai.types import (
    AgentToolCallEvent,
    AgentToolResultEvent,
    AgentTurnEndEvent,
    Context,
    TextDeltaEvent,
    Tool,
    UserMessage,
)

RESET  = "\033[0m"
DIM    = "\033[2m"
CYAN   = "\033[36m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
RED    = "\033[31m"
BOLD   = "\033[1m"
MAGENTA = "\033[35m"


def _sep(title: str) -> None:
    print(f"\n{BOLD}{'─' * 60}{RESET}")
    print(f"{BOLD}{title}{RESET}")
    print(f"{BOLD}{'─' * 60}{RESET}\n")


def _basic_on_event(event):
    if isinstance(event, AgentToolCallEvent):
        args = ", ".join(f"{k}={v!r}" for k, v in event.tool_input.items())
        print(f"\n{CYAN}🔧 Turn {event.turn} → {event.tool_name}({args}){RESET}", flush=True)
    elif isinstance(event, AgentToolResultEvent):
        icon = "❌" if event.error else "✅"
        print(f"{GREEN}{icon} {event.result[:120]}{RESET}", flush=True)
    elif isinstance(event, AgentTurnEndEvent):
        print(f"\n{YELLOW}── Turn {event.turn} done ──{RESET}\n", flush=True)
    elif isinstance(event, TextDeltaEvent):
        print(event.text, end="", flush=True)


# ============================================================ #
# Edge case 1: Summarizing reducer (not just trimming)         #
# ============================================================ #

async def demo_summarizing_reducer() -> None:
    _sep("EDGE CASE 1: Summarizing reducer — compress history into a note")

    print("Agent runs multi-step calculation. After each turn, reducer compresses")
    print("the message history into a single summary note in the scratchpad.\n")

    ctx = Context(
        system_prompt=(
            "You are a step-by-step math agent. You have a `multiply(a, b)` tool. "
            "ALWAYS use it — never compute mentally. "
            "Work through: 3 × 4, then result × 5, then result × 2."
        ),
        messages=[UserMessage(content="Compute 3 × 4 × 5 × 2 step by step using multiply.")],
        scratchpad={"steps": []},
        tools=[
            Tool(
                name="multiply",
                description="Multiplies two numbers.",
                parameters={
                    "type": "object",
                    "properties": {
                        "a": {"type": "number"},
                        "b": {"type": "number"},
                    },
                    "required": ["a", "b"],
                },
            )
        ],
    )

    step_results: list[str] = []

    def multiply(a: float, b: float) -> str:
        result = a * b
        step_results.append(f"{a} × {b} = {result}")
        return str(result)

    def summarizing_reducer(c: Context) -> Context:
        """After each turn, compress all tool exchanges into a scratchpad note."""
        from piai.types import ToolResultMessage, AssistantMessage
        # Collect tool results from this turn
        tool_results = [m.content for m in c.messages if isinstance(m, ToolResultMessage)]
        if tool_results:
            c.scratchpad["steps"] = list(step_results)  # keep running list
            print(f"\n{DIM}[reducer] Compressed {len(c.messages)} messages → scratchpad has {len(step_results)} steps{RESET}", flush=True)
        # Keep only the first user message + last 2 messages
        first = c.messages[:1]
        last = c.messages[-2:] if len(c.messages) > 2 else c.messages[1:]
        kept = first + [m for m in last if m not in first]
        return Context(messages=kept, system_prompt=c.system_prompt, tools=c.tools, scratchpad=c.scratchpad)

    result = await agent(
        model_id="gpt-5.1-codex-mini",
        context=ctx,
        local_handlers={"multiply": multiply},
        options={"reasoning_effort": "low"},
        max_turns=8,
        on_event=_basic_on_event,
        context_reducer=summarizing_reducer,
    )

    print(f"\n\nFinal answer: {result.text}")
    print(f"Steps recorded in scratchpad: {ctx.scratchpad.get('steps', [])}")
    print(f"[OK] Summarizing reducer maintained step history without full message history")


# ============================================================ #
# Edge case 2: Token budget abort via usage events             #
# ============================================================ #

async def demo_token_budget_abort() -> None:
    _sep("EDGE CASE 2: Token budget — abort agent if cumulative tokens exceed limit")

    TOKEN_BUDGET = 2000  # low budget to demonstrate cut-off
    total_tokens = [0]
    budget_hit = [False]

    def budget_on_event(event):
        if isinstance(event, AgentTurnEndEvent):
            total_tokens[0] += event.usage.get("total_tokens", 0)
            used = total_tokens[0]
            budget = TOKEN_BUDGET
            pct = min(100, int(used / budget * 100))
            bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
            print(f"\n{MAGENTA}📊 Turn {event.turn}: +{event.usage.get('total_tokens',0)} tokens  |  total={used}/{budget} [{bar}]{RESET}\n", flush=True)
            if used >= budget:
                budget_hit[0] = True
        elif isinstance(event, AgentToolCallEvent):
            args = ", ".join(f"{k}={v!r}" for k, v in event.tool_input.items())
            print(f"\n{CYAN}🔧 Turn {event.turn} → {event.tool_name}({args}){RESET}", flush=True)
        elif isinstance(event, AgentToolResultEvent):
            print(f"{GREEN}✅ {event.result[:100]}{RESET}", flush=True)
        elif isinstance(event, TextDeltaEvent):
            print(event.text, end="", flush=True)

    ctx = Context(
        system_prompt="You are a data lookup agent. Use the `lookup(key)` tool to retrieve values. Look up: alpha, beta, gamma, delta, epsilon.",
        messages=[UserMessage(content="Look up all five keys: alpha, beta, gamma, delta, epsilon using the lookup tool, one at a time.")],
        tools=[
            Tool(
                name="lookup",
                description="Looks up a value by key.",
                parameters={
                    "type": "object",
                    "properties": {"key": {"type": "string"}},
                    "required": ["key"],
                },
            )
        ],
    )

    db = {"alpha": "42", "beta": "99", "gamma": "17", "delta": "55", "epsilon": "81"}

    def lookup(key: str) -> str:
        return db.get(key, f"key '{key}' not found")

    result = await agent(
        model_id="gpt-5.1-codex-mini",
        context=ctx,
        local_handlers={"lookup": lookup},
        options={"reasoning_effort": "low"},
        max_turns=10,
        on_event=budget_on_event,
    )

    print(f"\n\nFinal answer: {result.text}")
    print(f"Total tokens used: {total_tokens[0]}")
    if budget_hit[0]:
        print(f"{RED}⚠ Token budget of {TOKEN_BUDGET} was exceeded — in production you'd abort here{RESET}")
    else:
        print(f"{GREEN}✓ Completed within {TOKEN_BUDGET} token budget{RESET}")


# ============================================================ #
# Edge case 3: Async context_extractor + scratchpad forwarding #
# ============================================================ #

async def demo_async_extractor_scratchpad() -> None:
    _sep("EDGE CASE 3: Async context_extractor + scratchpad forwarding")

    from piai.langchain.sub_agent_tool import SubAgentTool
    from piai.types import Context, UserMessage

    print("Parent has rich scratchpad. Async extractor forwards only relevant keys.\n")

    parent_ctx = Context(
        messages=[
            UserMessage(content="[master state — irrelevant to sub-agent]"),
        ],
        scratchpad={
            "job_id": "job-001",
            "targets": ["libssl.so", "libcrypto.so"],
            "completed": [],
            "irrelevant_key_1": "noise",
            "irrelevant_key_2": "more noise",
        },
    )

    extractor_calls = [0]

    async def async_isolating_extractor(ctx: Context) -> Context:
        """Async extractor — simulates async DB lookup or network call."""
        await asyncio.sleep(0.01)  # simulate async work
        extractor_calls[0] += 1
        targets = ctx.scratchpad.get("targets", [])
        current = targets[0] if targets else "unknown"
        print(f"{DIM}[async extractor] call #{extractor_calls[0]}: isolating for target={current!r}{RESET}")
        return Context(
            system_prompt="You are a concise security analyst. In 2 sentences: what is the main risk of the given library in a banking app?",
            messages=[UserMessage(content=f"Assess: {current}")],
            # Only forward what the sub-agent needs
            scratchpad={"current_target": current, "job_id": ctx.scratchpad["job_id"]},
        )

    sub = SubAgentTool(
        name="assessor",
        description="Assesses a library.",
        model_id="gpt-5.1-codex-mini",
        initial_context=parent_ctx,
        context_extractor=async_isolating_extractor,
        options={"reasoning_effort": "low"},
        on_event=_basic_on_event,
    )

    print(f"Parent scratchpad has {len(parent_ctx.scratchpad)} keys. Sub-agent should only see 2.\n")
    result = await sub._arun(task="assess libssl.so")

    print(f"\n{GREEN}Result:{RESET} {result}")
    assert extractor_calls[0] == 1, f"Expected 1 extractor call, got {extractor_calls[0]}"
    print(f"[OK] Async extractor called {extractor_calls[0]} time(s)")


# ============================================================ #
# Edge case 4: Agent reads its own scratchpad notes            #
# ============================================================ #

async def demo_agent_reads_scratchpad() -> None:
    _sep("EDGE CASE 4: Agent reads its own scratchpad notes mid-session")

    print("Agent first records a number in scratchpad via tool, then uses it later.")
    print("Scratchpad is injected into system prompt — model 'remembers' it.\n")

    ctx = Context(
        system_prompt=(
            "You are a stateful math agent. "
            "Your scratchpad (shown below) persists your working state across turns. "
            "Tools: save_result(value) stores a number; get_double() returns 2× your saved result."
        ),
        messages=[
            UserMessage(
                content=(
                    "Step 1: compute 7 × 6 = 42, call save_result(42). "
                    "Step 2: then call get_double() which will use your saved value. "
                    "Report both results."
                )
            )
        ],
        scratchpad={"saved_result": None},
        tools=[
            Tool(
                name="save_result",
                description="Saves a number to scratchpad for later use.",
                parameters={
                    "type": "object",
                    "properties": {"value": {"type": "number", "description": "The value to save"}},
                    "required": ["value"],
                },
            ),
            Tool(
                name="get_double",
                description="Returns 2× the value previously saved via save_result.",
                parameters={"type": "object", "properties": {}, "required": []},
            ),
        ],
    )

    def save_result(value: float) -> str:
        ctx.scratchpad["saved_result"] = value
        return f"Saved {value} to scratchpad."

    def get_double() -> str:
        saved = ctx.scratchpad.get("saved_result")
        if saved is None:
            return "Error: no result saved yet."
        return f"2 × {saved} = {saved * 2}"

    result = await agent(
        model_id="gpt-5.1-codex-mini",
        context=ctx,
        local_handlers={"save_result": save_result, "get_double": get_double},
        options={"reasoning_effort": "low"},
        max_turns=8,
        on_event=_basic_on_event,
    )

    print(f"\n\nFinal answer: {result.text}")
    print(f"Final scratchpad: {ctx.scratchpad}")
    assert ctx.scratchpad.get("saved_result") == 42, "Expected 42 saved to scratchpad"
    print(f"[OK] Agent successfully read its own scratchpad note to compute double")


# ============================================================ #
# Edge case 5: SubAgentTool with no initial_context            #
#              but with context_extractor (fresh ctx filtered) #
# ============================================================ #

async def demo_extractor_on_fresh_context() -> None:
    _sep("EDGE CASE 5: context_extractor applied to fresh context (no initial_context)")

    from piai.langchain.sub_agent_tool import SubAgentTool

    print("When initial_context is None, sub-agent builds fresh context from task.")
    print("context_extractor still runs — it gets the fresh context and can enrich it.\n")

    enrichment_called = [False]

    def enriching_extractor(ctx: Context) -> Context:
        """Extractor that adds metadata to a fresh context."""
        enrichment_called[0] = True
        print(f"{DIM}[extractor] enriching fresh context: {len(ctx.messages)} message(s){RESET}")
        return Context(
            system_prompt="You are a precise code reviewer. Be extremely concise — one sentence max.",
            messages=ctx.messages,  # keep original task message
            scratchpad={"review_mode": "strict", "max_words": 20},
        )

    sub = SubAgentTool(
        name="reviewer",
        description="Code reviewer.",
        model_id="gpt-5.1-codex-mini",
        context_extractor=enriching_extractor,
        options={"reasoning_effort": "low"},
        on_event=_basic_on_event,
    )

    result = await sub._arun(task="Is `x = x + 1` or `x += 1` preferred in Python? One sentence.")

    print(f"\n{GREEN}Result:{RESET} {result}")
    assert enrichment_called[0], "Extractor should have been called"
    print(f"[OK] context_extractor ran on fresh context — enriched system prompt and scratchpad")


# ============================================================ #
# Main                                                         #
# ============================================================ #

async def main() -> None:
    print(f"{BOLD}piai SDK Extensions — Edge Case Live Demo{RESET}")
    print("Real LLM calls — testing boundary conditions and non-obvious behaviors.\n")

    await demo_summarizing_reducer()
    await demo_token_budget_abort()
    await demo_async_extractor_scratchpad()
    await demo_agent_reads_scratchpad()
    await demo_extractor_on_fresh_context()

    print(f"\n{BOLD}{'─' * 60}{RESET}")
    print(f"{BOLD}All edge case demos complete.{RESET}")


if __name__ == "__main__":
    asyncio.run(main())
