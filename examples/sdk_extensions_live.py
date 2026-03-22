"""
Live example: Four piai SDK Extensions — real LLM calls via piai auth.

Demonstrates all four new features against the actual ChatGPT backend:

  1. context_reducer  — sliding-window reducer keeps context from growing
  2. AgentTurnEndEvent.usage — real token counts reported per turn
  3. context_extractor on SubAgentTool — context isolation between tiers
  4. Context.scratchpad — agent notes injected into system prompt every turn

No mocking. No stubs. Real responses.

Run:
  PYTHONPATH=src .venv/bin/python examples/sdk_extensions_live.py

Each demo is independent and can be commented out in main() if needed.
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

# ── ANSI colours ──────────────────────────────────────────────────────────────
RESET  = "\033[0m"
DIM    = "\033[2m"
CYAN   = "\033[36m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
BOLD   = "\033[1m"


def _sep(title: str) -> None:
    print(f"\n{BOLD}{'─' * 60}{RESET}")
    print(f"{BOLD}{title}{RESET}")
    print(f"{BOLD}{'─' * 60}{RESET}\n")


# ============================================================ #
# Shared event handler                                         #
# ============================================================ #

def make_on_event(show_usage: bool = False):
    """Returns an on_event callback. Toggle show_usage to print token counts."""
    def on_event(event):
        if isinstance(event, AgentToolCallEvent):
            args = ", ".join(f"{k}={v!r}" for k, v in event.tool_input.items())
            print(f"\n{CYAN}🔧 Turn {event.turn} → {event.tool_name}({args}){RESET}", flush=True)

        elif isinstance(event, AgentToolResultEvent):
            preview = event.result[:150].replace("\n", " ")
            icon = "❌" if event.error else "✅"
            suffix = "..." if len(event.result) > 150 else ""
            print(f"{GREEN}{icon} {preview}{suffix}{RESET}", flush=True)

        elif isinstance(event, AgentTurnEndEvent):
            msg = f"Turn {event.turn} complete — {len(event.tool_calls)} tool call(s)"
            if show_usage and event.usage:
                inp = event.usage.get("input", 0)
                out = event.usage.get("output", 0)
                cached = event.usage.get("cache_read", 0)
                msg += f"  |  tokens in={inp} out={out} cached={cached}"
            print(f"\n{YELLOW}── {msg} ──{RESET}\n", flush=True)

        elif isinstance(event, TextDeltaEvent):
            print(event.text, end="", flush=True)

    return on_event


# ============================================================ #
# Feature 1: context_reducer                                   #
# ============================================================ #

async def demo_context_reducer() -> None:
    _sep("FEATURE 1: context_reducer — sliding window to prevent context explosion")

    turn_msg_counts: list[int] = []
    reducer_runs = 0

    def sliding_window(ctx: Context) -> Context:
        """Keep only the system context and the last 4 messages."""
        nonlocal reducer_runs
        reducer_runs += 1
        kept = ctx.messages[-4:]
        print(f"\n{DIM}[reducer] {len(ctx.messages)} msgs → trimmed to {len(kept)}{RESET}", flush=True)
        return Context(
            messages=kept,
            system_prompt=ctx.system_prompt,
            tools=ctx.tools,
            scratchpad=ctx.scratchpad,
        )

    # on_event that also tracks message counts (we peek at AgentTurnEndEvent)
    base_on_event = make_on_event(show_usage=True)
    call_n = 0

    def on_event(event):
        base_on_event(event)

    ctx = Context(
        system_prompt=(
            "You are a calculator agent. You have one tool: `add(a, b)` which adds two numbers and returns the result. "
            "The user will give you a chain of additions. You MUST call add() for each step — do not compute in your head. "
            "Show your work step by step, one tool call per addition."
        ),
        messages=[
            UserMessage(
                content=(
                    "Add these numbers in sequence using the add tool, one call per step: "
                    "start with 10 + 20, then take that result and add 30, then add 40, then add 50. "
                    "Call add() four separate times and report the running total after each."
                )
            )
        ],
        tools=[
            Tool(
                name="add",
                description="Adds two numbers and returns the sum.",
                parameters={
                    "type": "object",
                    "properties": {
                        "a": {"type": "number", "description": "First number"},
                        "b": {"type": "number", "description": "Second number"},
                    },
                    "required": ["a", "b"],
                },
            )
        ],
    )

    running_total = [0]

    def add(a: int | float, b: int | float) -> str:
        running_total[0] = a + b
        return f"{a} + {b} = {running_total[0]}"

    print("Running multi-step calculator agent with sliding-window reducer...\n")

    result = await agent(
        model_id="gpt-5.1-codex-mini",
        context=ctx,
        local_handlers={"add": add},
        options={"reasoning_effort": "low"},
        max_turns=10,
        on_event=on_event,
        context_reducer=sliding_window,
    )

    print(f"\n\nFinal answer: {result.text}")
    print(f"Reducer ran {reducer_runs} time(s). Context never exceeded 4 messages after each trim.")


# ============================================================ #
# Feature 2: AgentTurnEndEvent.usage                           #
# ============================================================ #

async def demo_turn_usage() -> None:
    _sep("FEATURE 2: AgentTurnEndEvent.usage — real token counts per turn")

    turn_usages: list[dict] = []

    def on_event(event):
        if isinstance(event, AgentTurnEndEvent):
            turn_usages.append(dict(event.usage))
            inp   = event.usage.get("input", 0)
            out   = event.usage.get("output", 0)
            cache = event.usage.get("cache_read", 0)
            total = event.usage.get("total_tokens", 0)
            print(
                f"\n{MAGENTA}📊 Turn {event.turn} tokens — "
                f"input={inp}  output={out}  cached={cache}  total={total}{RESET}\n",
                flush=True,
            )
        elif isinstance(event, AgentToolCallEvent):
            args = ", ".join(f"{k}={v!r}" for k, v in event.tool_input.items())
            print(f"\n{CYAN}🔧 Turn {event.turn} → {event.tool_name}({args}){RESET}", flush=True)
        elif isinstance(event, AgentToolResultEvent):
            print(f"{GREEN}✅ {event.result[:120]}{RESET}", flush=True)
        elif isinstance(event, TextDeltaEvent):
            print(event.text, end="", flush=True)

    ctx = Context(
        system_prompt=(
            "You are a unit conversion assistant. "
            "You MUST use the convert tool for every conversion — do not calculate yourself. "
            "Call it once per conversion."
        ),
        messages=[
            UserMessage(
                content=(
                    "Convert the following using the convert tool:\n"
                    "1. 100 kilometers to miles\n"
                    "2. 72 Fahrenheit to Celsius\n"
                    "3. 5 kilograms to pounds\n"
                    "Call the tool once for each conversion, then summarize all three results."
                )
            )
        ],
        tools=[
            Tool(
                name="convert",
                description="Converts a value from one unit to another. Supported: kilometers→miles, fahrenheit→celsius, kilograms→pounds.",
                parameters={
                    "type": "object",
                    "properties": {
                        "value":     {"type": "number", "description": "The numeric value to convert"},
                        "from_unit": {"type": "string", "description": "Source unit (e.g. 'kilometers')"},
                        "to_unit":   {"type": "string", "description": "Target unit (e.g. 'miles')"},
                    },
                    "required": ["value", "from_unit", "to_unit"],
                },
            )
        ],
    )

    conversions = {
        ("kilometers", "miles"):    lambda v: round(v * 0.621371, 4),
        ("fahrenheit", "celsius"):  lambda v: round((v - 32) * 5 / 9, 4),
        ("kilograms",  "pounds"):   lambda v: round(v * 2.20462, 4),
    }

    def convert(value: float, from_unit: str, to_unit: str) -> str:
        key = (from_unit.lower(), to_unit.lower())
        if key in conversions:
            result = conversions[key](value)
            return f"{value} {from_unit} = {result} {to_unit}"
        return f"Conversion {from_unit}→{to_unit} not supported"

    print("Running unit conversion agent — watching real token usage per turn...\n")

    result = await agent(
        model_id="gpt-5.1-codex-mini",
        context=ctx,
        local_handlers={"convert": convert},
        options={"reasoning_effort": "low"},
        max_turns=8,
        on_event=on_event,
    )

    print(f"\n\nFinal answer: {result.text}")
    if turn_usages:
        total_in  = sum(u.get("input", 0) for u in turn_usages)
        total_out = sum(u.get("output", 0) for u in turn_usages)
        print(f"\n{MAGENTA}Total across {len(turn_usages)} turn(s): input={total_in}  output={total_out}{RESET}")


# ============================================================ #
# Feature 3: context_extractor in SubAgentTool                 #
# ============================================================ #

async def demo_context_extractor() -> None:
    _sep("FEATURE 3: context_extractor — context isolation between tiers")

    from piai.langchain.sub_agent_tool import SubAgentTool
    from piai.types import Context, UserMessage

    print("Scenario: Tier-1 orchestrator has fat state. It spawns a sub-agent")
    print("that only receives the specific target — not the full orchestrator history.\n")

    # Simulate a fat orchestrator context — lots of history the sub-agent doesn't need
    parent_ctx = Context(
        system_prompt="Master orchestrator managing a multi-library scan pipeline.",
        messages=[
            UserMessage(content="[Scan job started: com.example.banking]"),
            UserMessage(content="[libcrypto.so scan complete: 3 findings]"),
            UserMessage(content="[libm.so scan complete: 0 findings]"),
            UserMessage(content="[Global task ledger: 47 pending items]"),
        ],
        scratchpad={
            "job_id": "scan-2026-001",
            "target_apk": "com.example.banking",
            "scan_queue": ["libssl.so", "libxml2.so"],
            "total_findings": 3,
        },
    )

    def isolating_extractor(ctx: Context) -> Context:
        """Strip orchestrator history. Pass only the current target to the sub-agent."""
        target = ctx.scratchpad.get("scan_queue", ["unknown"])[0]
        apk    = ctx.scratchpad.get("target_apk", "")
        print(f"{DIM}[extractor] Stripping {len(ctx.messages)} parent messages → passing target only{RESET}")
        return Context(
            system_prompt=(
                "You are a native library security pre-screener. "
                "Given a library name and APK, return a brief threat assessment: "
                "likely attack surface, 1-2 risk factors to investigate, severity estimate (low/medium/high)."
            ),
            messages=[
                UserMessage(content=f"Pre-screen {target} from APK {apk}. Be concise — 3-4 sentences max.")
            ],
            scratchpad={"target": target, "apk": apk},
        )

    sub_agent = SubAgentTool(
        name="library_prescreener",
        description="Pre-screens a native library for security risks.",
        model_id="gpt-5.1-codex-mini",
        initial_context=parent_ctx,
        context_extractor=isolating_extractor,
        options={"reasoning_effort": "low"},
        on_event=make_on_event(),
    )

    print(f"Parent context has {len(parent_ctx.messages)} messages + scratchpad: {list(parent_ctx.scratchpad.keys())}")
    print("Calling sub-agent via SubAgentTool._arun()...\n")

    result = await sub_agent._arun(task="Assess the security risk.")

    print(f"\n{GREEN}Sub-agent result:{RESET}\n{result}")
    print(f"\n{DIM}(Sub-agent never saw the parent's {len(parent_ctx.messages)}-message history){RESET}")


# ============================================================ #
# Feature 4: Context.scratchpad                                #
# ============================================================ #

async def demo_scratchpad() -> None:
    _sep("FEATURE 4: Context.scratchpad — working memory injected into every prompt")

    print("Scenario: Agent tracks discovered facts in the scratchpad across turns.")
    print("Even if message history were trimmed, the scratchpad persists in the system prompt.\n")

    scratchpad_updates: list[dict] = []

    def on_event(event):
        if isinstance(event, AgentToolCallEvent):
            args = ", ".join(f"{k}={v!r}" for k, v in event.tool_input.items())
            print(f"\n{CYAN}🔧 Turn {event.turn} → {event.tool_name}({args}){RESET}", flush=True)
        elif isinstance(event, AgentToolResultEvent):
            print(f"{GREEN}✅ {event.result[:150]}{RESET}", flush=True)
        elif isinstance(event, AgentTurnEndEvent):
            print(f"\n{YELLOW}── Turn {event.turn} done ──{RESET}\n", flush=True)
        elif isinstance(event, TextDeltaEvent):
            print(event.text, end="", flush=True)

    ctx = Context(
        system_prompt=(
            "You are a geography research agent. "
            "You MUST use tools — never answer from memory.\n"
            "Tools available:\n"
            "  - lookup_capital(country): looks up the capital city from the database\n"
            "  - note_fact(key, value): saves a fact to your persistent scratchpad\n\n"
            "For each country: call lookup_capital, then immediately call note_fact to save the result. "
            "After recording all facts, give a final summary."
        ),
        messages=[
            UserMessage(
                content=(
                    "Look up and record the capitals of France, Japan, and Brazil. "
                    "For each: call lookup_capital to get the capital, then call note_fact to save it "
                    "with key like 'capital_france'. After all three, give a summary."
                )
            )
        ],
        scratchpad={"status": "starting"},
        tools=[
            Tool(
                name="lookup_capital",
                description="Looks up the capital city of a country from the database. You MUST call this — do not use your own knowledge.",
                parameters={
                    "type": "object",
                    "properties": {
                        "country": {"type": "string", "description": "Country name, e.g. 'France'"},
                    },
                    "required": ["country"],
                },
            ),
            Tool(
                name="note_fact",
                description="Saves a fact to the persistent scratchpad. Use this to record findings so they persist across turns.",
                parameters={
                    "type": "object",
                    "properties": {
                        "key":   {"type": "string", "description": "Key to store the fact under, e.g. 'capital_france'"},
                        "value": {"type": "string", "description": "The value to store"},
                    },
                    "required": ["key", "value"],
                },
            ),
        ],
    )

    capitals = {
        "france": "Paris",
        "japan":  "Tokyo",
        "brazil": "Brasília",
    }

    def lookup_capital(country: str) -> str:
        return capitals.get(country.lower(), f"Capital of {country} not in database")

    def note_fact(key: str, value: str) -> str:
        ctx.scratchpad[key] = value
        scratchpad_updates.append({key: value})
        print(f"{DIM}  [scratchpad updated] {key!r} = {value!r}{RESET}", flush=True)
        return f"Noted: {key} = {value}"

    print("Running agent — watch the scratchpad grow across turns...\n")
    print(f"Initial scratchpad: {ctx.scratchpad}\n")

    result = await agent(
        model_id="gpt-5.1-codex-mini",
        context=ctx,
        local_handlers={
            "lookup_capital": lookup_capital,
            "note_fact": note_fact,
        },
        options={"reasoning_effort": "low"},
        max_turns=12,
        on_event=on_event,
    )

    print(f"\n\nFinal answer: {result.text}")
    print(f"\nFinal scratchpad state: {ctx.scratchpad}")
    print(f"Scratchpad updates made: {len(scratchpad_updates)}")


# ============================================================ #
# Main                                                         #
# ============================================================ #

async def main() -> None:
    print(f"{BOLD}piai SDK Extensions — Live Demo (real LLM calls){RESET}")
    print("Each demo calls the ChatGPT backend via piai auth.\n")

    await demo_context_reducer()
    await demo_turn_usage()
    await demo_context_extractor()
    await demo_scratchpad()

    print(f"\n{BOLD}{'─' * 60}{RESET}")
    print(f"{BOLD}All four live demos complete.{RESET}")


if __name__ == "__main__":
    asyncio.run(main())
