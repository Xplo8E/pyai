# piai — Agent Context

This file is the starting point for any AI-assisted work on piai. Read this first before making any changes. It is kept up to date with every significant modification.

---

## What this project is

**piai** is a Python library that lets you use your ChatGPT Plus/Pro subscription to call GPT models without paying per-token API fees. It authenticates via OAuth (same flow as the ChatGPT web app) and streams responses from ChatGPT's internal backend.

- PyPI package: `pi-ai-py`
- Import: `from piai import stream, complete, complete_text, agent`
- CLI: `piai login`, `piai run "prompt"`, `piai status`
- GitHub: https://github.com/Xplo8E/piai
- Python: 3.12+, built with `uv`

---

## Project layout

```
src/piai/
├── __init__.py              # Public exports: stream, complete, complete_text, agent, MCPServer, MCPHub + all types
├── types.py                 # Context, messages, stream events (all data types)
├── stream.py                # stream() / complete() / complete_text() entry points
├── agent.py                 # agent() — autonomous agentic loop with MCP + observability events
├── cli.py                   # CLI (Click): login, logout, list, status, run
├── oauth/
│   ├── __init__.py          # Provider registry + get_oauth_api_key() with auto-refresh
│   ├── types.py             # OAuthCredentials, OAuthProviderInterface ABC
│   ├── storage.py           # auth.json read/write (~/.piai/auth.json, or PIAI_AUTH env override)
│   ├── pkce.py              # RFC 7636 PKCE: verifier + challenge
│   └── openai_codex.py      # ChatGPT Plus OAuth login + refresh
├── mcp/
│   ├── __init__.py          # exports MCPServer, MCPClient, MCPHub (langchain tools in langchain_tools.py)
│   ├── server.py            # MCPServer config (stdio/http/sse + from_config + from_toml)
│   ├── client.py            # MCPClient — persistent connection to one MCP server
│   ├── hub.py               # MCPHub — manages N servers, merges tools, routes calls
│   └── langchain_tools.py   # MCP → LangChain bridge (to_langchain_tools, MCPHubToolset, MCPLangChainTool)
├── langchain/
│   ├── chat_model.py        # PiAIChatModel — LangChain BaseChatModel adapter (surfaces thinking via additional_kwargs)
│   └── sub_agent_tool.py    # SubAgentTool — full piai agent() as a LangChain BaseTool
└── providers/
    ├── message_transform.py # Context → OpenAI Responses API wire format
    └── openai_codex.py      # SSE streaming + _StreamProcessor state machine
tests/
    test_pkce.py
    test_oauth_codex.py
    test_message_transform.py
    test_stream_processor.py
    test_sse_parser.py
    test_mcp.py
    test_langchain.py
    test_langgraph_integration.py
    test_thinking.py         # thinking/observability events (34 tests)
    test_sdk_extensions.py   # scratchpad, context_reducer, usage, context_extractor (16 tests)
    test_coverage_boost.py   # broad coverage across usage/, oauth/, agent, stream (68+ tests)
docs/
    architecture.md          # Design overview and flow diagrams
    internals.md             # Per-module deep-dive
    mcp.md                   # Full MCP + observability reference (includes scratchpad, context_reducer, usage)
    contributing.md          # Setup and contribution guide
    AGENTS.md                # This file (full feature reference)
examples/
    mcp_filesystem_agent.py          # piai native agent + MCP filesystem
    langchain_mcp_agent.py           # PiAIChatModel + LangChain + MCP
    radare2_binary_analysis.py       # piai agent + r2mcp binary RE
    ida_binary_analysis.py           # piai agent + IDA Pro MCP
    langgraph_supervisor_agent.py    # LangGraph Supervisor + SubAgentTool
    sdk_extensions_demo.py           # 4 features — offline/mocked demo
    sdk_extensions_live.py           # 4 features — real LLM calls
    sdk_edge_cases_live.py           # edge cases: async reducer, token budget, scratchpad memory
```

---

## MCP integration

piai has a native MCP client layer. No LangChain, no mcpo, no manual tool wrappers.

```python
from piai import agent
from piai.mcp import MCPServer
from piai.types import Context, UserMessage

ctx = Context(messages=[UserMessage(content="Analyze /lib/target.so")])

result = await agent(
    model_id="gpt-5.1-codex-mini",
    context=ctx,
    mcp_servers=[
        MCPServer.stdio("r2pm -r r2mcp"),                          # spawns subprocess
        MCPServer.stdio("npx @modelcontextprotocol/server-filesystem /tmp"),
        MCPServer.http("http://localhost:9000/mcp"),                # Streamable HTTP
        MCPServer.sse("http://localhost:9000/sse"),                 # legacy SSE
    ],
    options={"reasoning_effort": "medium"},
    max_turns=20,                                                   # safety limit
    on_event=lambda e: print(e),                                    # optional live output
    require_all_servers=False,                                      # allow partial connect
    connect_timeout=60.0,                                           # per-server timeout
    tool_result_max_chars=32_000,                                   # truncate huge results
)
```

**How it works:**
1. `MCPHub` connects to all servers concurrently (respects `connect_timeout`)
2. Tools are auto-discovered via `list_tools` from each server
3. All tools merged into a flat list **plus** any pre-existing `context.tools`, injected into `Context.tools`
4. `agent()` runs `stream()` in a loop, executing tool calls via `MCPHub.call_tool()`
5. Tool results appended as `ToolResultMessage`, loop continues until model stops or `max_turns` reached

**Tool name collisions:** If two servers expose the same tool name, **both** are namespaced: `server1__toolname` and `server2__toolname`. The unnamespaced key is removed — a warning is logged.

**local_handlers:** Pass pure Python callables instead of (or alongside) MCP servers. `local_handlers` take priority over MCP on name conflicts:

```python
result = await agent(
    model_id="gpt-5.1-codex-mini",
    context=ctx,
    local_handlers={
        "add": lambda a, b: a + b,           # sync
        "fetch_url": my_async_fetch,          # async — awaited automatically
    },
    mcp_servers=[...],                        # mix with MCP — each tool goes to the right handler
)
```

**Key classes:**
- `MCPServer` — config only, no connection. Factory: `.stdio()`, `.http()`, `.sse()`, `.from_config()`, `.from_toml()`
- `MCPClient` — one persistent session (uses `AsyncExitStack` to keep transport alive across calls)
- `MCPHub` — async context manager over N clients, handles connect/discover/route/close

**Loading from TOML config:**
```python
servers = MCPServer.from_toml("~/.piai/config.toml")  # loads [mcp_servers] section
```

---

## Thinking / reasoning observability

Reasoning models emit `ThinkingContent` blocks. piai surfaces these as stream events and convenience properties.

### Stream events (from `stream()` and `agent()`)

| Event | Description |
|---|---|
| `ThinkingStartEvent` | Model started a reasoning block |
| `ThinkingDeltaEvent(thinking: str)` | Incremental reasoning chunk |
| `ThinkingEndEvent(thinking: str)` | Block done — `thinking` is always the full accumulated text |
| `AgentToolCallEvent(turn, tool_name, tool_input)` | Fired just before a tool is executed |
| `AgentToolResultEvent(turn, tool_name, tool_input, result, error)` | Fired after tool execution |
| `AgentTurnEndEvent(turn, thinking, tool_calls)` | End of each agent loop turn |

### AssistantMessage properties

```python
result = await agent(...)
result.text       # str — concatenated TextContent blocks
result.thinking   # str | None — concatenated ThinkingContent blocks, None if no reasoning
```

### How _StreamProcessor handles reasoning

- `response.output_item.added` with `type="reasoning"` → emit `ThinkingStartEvent`
- `response.reasoning_summary_text.delta` → accumulate delta, emit `ThinkingDeltaEvent`
- `response.reasoning_summary_part.done` → append `"\n\n"` separator
- `response.output_item.done` with `type="reasoning"` → reconstruct full text from summary parts, append `ThinkingContent` to message, emit `ThinkingEndEvent(thinking=full_text)`

> Backend sometimes skips streaming deltas for short reasoning — `ThinkingEndEvent.thinking` is always authoritative.

### LangChain thinking surfacing

`PiAIChatModel._astream` accumulates thinking and surfaces it via `additional_kwargs`:

- `ThinkingDeltaEvent` → `AIMessageChunk(additional_kwargs={"thinking_delta": delta})`
- `ThinkingEndEvent` → accumulated in `thinking_parts`
- `DoneEvent` → final chunk gets `additional_kwargs={"thinking": "\n\n".join(thinking_parts)}`

---

## LangChain integration

`PiAIChatModel` is a drop-in LangChain `BaseChatModel` backed by piai:

```python
from piai.langchain import PiAIChatModel
from langchain_core.messages import HumanMessage

llm = PiAIChatModel(model_name="gpt-5.1-codex-mini", options={"reasoning_effort": "medium"})
result = llm.invoke([HumanMessage(content="What is 2+2?")])
print(result.content)
# model reasoning in: result.additional_kwargs.get("thinking")
```

Supports `invoke`, `ainvoke`, `stream`, `astream`, `bind_tools`. Thinking is surfaced via `additional_kwargs["thinking"]` on the final message and `additional_kwargs["thinking_delta"]` on streaming chunks.

---

## Critical invariants (do not break these)

1. **auth.json keys are camelCase** — `accountId`, `expires` (in ms). Must stay compatible with the JS SDK.
2. **`expires` is Unix milliseconds** — `int(time.time() * 1000) + expires_in * 1000`.
3. **JWT base64 padding** — JWT strips `=` padding. Always re-add: `payload + "=" * (4 - len(payload) % 4)`.
4. **Stream processor event order** — `_StreamProcessor` in `providers/openai_codex.py` must handle SSE events in the exact sequence the Responses API sends. See `docs/internals.md` for the full event table.
5. **No retry on usage limit** — Skip retries when `"usage limit"` appears in the error message.
6. **`instructions` always present** — Defaults to `"You are a helpful assistant."` if no system prompt.
7. **PKCE base64url has no padding** — Strip all `=` characters after encoding.
8. **Tool call IDs truncated to 64 chars** — `_make_tc_id(call_id, item_id)` truncates `f"{call_id}|{item_id}"` to 64 chars. The Responses API enforces a 64-char limit.
9. **SSE CRLF normalization** — `_parse_sse` normalizes `\r\n` and `\r` to `\n` before splitting events.
10. **Options dict not mutated** — `stream()` copies `options` before calling `opts.pop("base_url", None)` to avoid mutating the caller's dict.
11. **MCP tool merge order** — MCP tools take priority; user-defined `context.tools` are appended de-duplicated by name.
12. **`asyncio.get_running_loop()`** — All async dispatch uses `get_running_loop()` (not the deprecated `get_event_loop()`). When already inside a running loop (LangGraph, Jupyter), dispatch to a thread with its own `asyncio.run()`.
13. **`ThinkingEndEvent.thinking` is authoritative** — Backend may skip `ThinkingDeltaEvent` deltas for short reasoning blocks. Always use `ThinkingEndEvent.thinking` (reconstructed from summary parts) as the canonical full text.
14. **`AssistantMessage.thinking` returns `None` not `""`** — When no ThinkingContent blocks are present. Callers check `if result.thinking:` not `if result.thinking is not None:`.
15. **`Context.scratchpad` injected only when non-empty** — `build_request_body()` skips the `<scratchpad>` block entirely if `context.scratchpad` is `{}`. Default is `{}`.
16. **`context_reducer` must carry scratchpad forward** — The reducer receives the full Context but must explicitly include `scratchpad=ctx.scratchpad` when constructing the returned Context, or scratchpad state is lost.
17. **`AgentTurnEndEvent.usage` keys match Responses API fields** — `input`, `output`, `cache_read`, `total_tokens`. All default to `0` if the backend doesn't send them.
18. **`SubAgentTool` auto-appends task message** — If `context_extractor` returns a Context whose last message is not a `UserMessage`, the `task` string is automatically appended as one. Prevents the sub-agent from getting a context with no task.
19. **`with_structured_output` uses `tool_choice="required"`** — Backend rejects tool-name forcing; `"required"` is used instead. Only one tool is bound so the model always calls the schema tool. `json_mode` is not yet supported.

---

## How to run tests

```bash
# Always use the local venv directly (avoids pi-mono workspace venv conflict)
.venv/bin/python -m pytest tests/ -v

# Quick count
.venv/bin/python -m pytest tests/ -q

# With coverage
.venv/bin/python -m pytest tests/ --cov=src/piai --cov-report=term-missing
```

Test files and what they cover:
| File | Tests | Coverage |
|------|-------|----------|
| `test_pkce.py` | — | PKCE verifier/challenge generation |
| `test_oauth_codex.py` | — | JWT decoding, auth URL, credential serialization |
| `test_message_transform.py` | — | Context → Responses API conversion, `_clamp_reasoning_effort`, `_make_tc_id` |
| `test_stream_processor.py` | — | `_StreamProcessor` state machine (text, tool calls, thinking, errors, edge cases) |
| `test_sse_parser.py` | — | SSE parser (CRLF normalization, split chunks, multi-event, invalid JSON) |
| `test_mcp.py` | — | MCPServer config, MCPClient, MCPHub, agent loop, `AgentTurnEndEvent` |
| `test_langchain.py` | — | PiAIChatModel, message conversion, streaming, bind_tools |
| `test_langgraph_integration.py` | — | MCP bridge, SubAgentTool, schema generation, MCPHubToolset |
| `test_thinking.py` | 34 | `AssistantMessage.text/thinking`, all new event types, agent observability, LangChain thinking surfacing, edge paths |

**Total: 381 tests**

---

## Supported models

Models are passed directly to the ChatGPT backend. Only models available on ChatGPT Plus/Pro work:

| Model ID | Notes |
|----------|-------|
| `gpt-5.1-codex-mini` | Fast, default |
| `gpt-5.1` | More capable |
| `gpt-5.1-codex-max` | |
| `gpt-5.2`, `gpt-5.2-codex` | |
| `gpt-5.3-codex`, `gpt-5.3-codex-spark` | |
| `gpt-5.4` | |

**Do not use** `gpt-4o`, `o3`, `o4-mini` etc. — those are public API models, not available on this backend.

---

## Options reference

```python
options = {
    "session_id": "my-session",      # → prompt_cache_key (enables caching)
    "reasoning_effort": "high",      # low / medium / high / xhigh (clamped per model)
    "reasoning_summary": "auto",     # auto / concise / detailed / off
    "text_verbosity": "medium",      # low / medium / high
    "base_url": "...",               # Override backend URL (testing only)
}
```

> **Note:** `temperature` is **not supported** by the ChatGPT backend and will cause a 400 error if passed.

---

## Changelog

### 2026-03-22 — Code quality pass
- **Refactor** Moved all inline imports to module level across 9 files; extracted nested functions in `oauth/openai_codex.py` and `usage/openai_codex.py`
- **Bug fix** `providers/openai_codex.py`: `name` referenced before assignment on `function_call` done path
- **Cleanup** Removed unused `tc_id_map` and 3 unused imports in `chat_model.py`; compiled `_USAGE_LIMIT_PATTERN` regex at module level
- **Dedup** `OPENAI_CODEX_PROVIDER` no longer redefined in `agent.py` — imported from `stream.py`
- **Docs** Updated `AGENTS.md`, `mcp.md`, and `CLAUDE.md` with full documentation for all new features

### 2026-03-22 — SDK extensions
- **New** `Context.scratchpad: dict[str, Any]` — injected as `<scratchpad>` JSON block into every LLM call
- **New** `agent(context_reducer=...)` — called after each turn's tool results, before next LLM call; sync and async supported
- **New** `AgentTurnEndEvent.usage` — per-turn token counts: `input`, `output`, `cache_read`, `total_tokens`
- **New** `SubAgentTool(initial_context=..., context_extractor=...)` — filter/transform parent context before sub-agent runs
- **New** `PiAIChatModel.with_structured_output(schema)` — Pydantic models and TypedDicts; `include_raw` mode
- **Fix** `agent.py`: `_run_loop` now correctly copies `context.scratchpad` into internal ctx
- **Examples** `sdk_extensions_demo.py`, `sdk_extensions_live.py`, `sdk_edge_cases_live.py`
- **Tests** `test_sdk_extensions.py` (16 tests), `test_coverage_boost.py` (68+ tests) — **Total: 381**

### 2026-03-20 — Thinking/reasoning observability
- **New** `ThinkingStartEvent`, `ThinkingEndEvent(thinking: str)` in `types.py` — bracket reasoning blocks; `ThinkingEndEvent.thinking` always has full text
- **New** `AgentToolCallEvent(turn, tool_name, tool_input)` — fired just before `_execute_tool()` in `agent.py`
- **New** `AgentToolResultEvent(turn, tool_name, tool_input, result, error)` — fired after `_execute_tool()`
- **New** `AgentTurnEndEvent(turn, thinking, tool_calls)` — fired at end of each loop turn
- **New** `AssistantMessage.text` property — concatenates all `TextContent` blocks
- **New** `AssistantMessage.thinking` property — concatenates all `ThinkingContent` blocks, returns `None` (not `""`) when absent
- **Fix** `providers/openai_codex.py`: emit `ThinkingStartEvent` when reasoning block opens, `ThinkingEndEvent` with full reconstructed text when it closes
- **Fix** `langchain/chat_model.py`: thinking surfaced via `additional_kwargs["thinking_delta"]` on stream chunks and `additional_kwargs["thinking"]` on final message
- **Fix** `langchain/sub_agent_tool.py`: `_run()` uses `get_running_loop()` + thread dispatch, not deprecated `asyncio.get_event_loop().run_until_complete()`
- **Examples** All 5 examples updated with full ANSI-colored observability: thinking, tool calls, results, turn summaries
- **Tests** Added `test_thinking.py` (34 tests): full coverage of new events + edge paths
- **Total tests:** 257

### 2026-03-20 — LangGraph integration
- **New** `src/piai/mcp/langchain_tools.py`: MCP → LangChain tool bridge
  - `to_langchain_tools(servers) → (list[MCPLangChainTool], MCPHub)` — converts MCP servers to LangChain tools
  - `MCPHubToolset` — async context manager version (auto-connect + close)
  - `MCPLangChainTool` — individual MCP tool as a LangChain `BaseTool`
- **New** `src/piai/langchain/sub_agent_tool.py`: `SubAgentTool`
  - Wraps a full piai `agent()` (with its own model + MCP servers) as a single `BaseTool`
  - Designed for use as a sub-agent inside a LangGraph Supervisor
- **Fix** `langchain/chat_model.py`: `_run_async()` helper — detects running event loop and dispatches to a fresh thread, fixing `asyncio.run()` errors inside LangGraph
- **Fix** `langchain/sub_agent_tool.py`: `_run()` now uses `asyncio.get_running_loop()` instead of deprecated `asyncio.get_event_loop()`, with same thread-safe dispatch
- **Fix** `mcp/langchain_tools.py`: `MCPLangChainTool._run()` — same thread-safe async dispatch as above
- **Examples** Added 3 new examples:
  - `examples/radare2_binary_analysis.py` — piai agent + r2mcp for binary RE
  - `examples/ida_binary_analysis.py` — piai agent + IDA Pro MCP
  - `examples/langgraph_supervisor_agent.py` — LangGraph Supervisor with MCP bridge + SubAgentTool
- **Tests** Added `test_langgraph_integration.py` (21 tests): covers MCP bridge, SubAgentTool, schema generation, hub toolset
- **Total tests:** 223

### 2026-03-19 — Examples + temperature note
- **Examples** Added `examples/` directory with two runnable examples:
  - `mcp_filesystem_agent.py` — piai native `agent()` + MCP filesystem server
  - `langchain_mcp_agent.py` — `PiAIChatModel` with invoke/astream/bind_tools/MCP agent patterns
- **Docs** Removed `temperature` from all options references — the ChatGPT backend rejects it with a 400 error
- **Code** Removed `temperature` passthrough from `build_request_body()` in `providers/message_transform.py`

### 2026-03-19 — Autoresearch improvement pass
- **Bug fix** `stream.py`: Copy options dict before `pop("base_url")` — prevents mutating caller's dict
- **Bug fix** `agent.py`: MCP tools now merged with pre-existing `context.tools` (de-duplicated by name, MCP takes priority) instead of silently replacing them
- **Bug fix** `oauth/openai_codex.py`: `asyncio.get_event_loop()` → `asyncio.get_running_loop()` (deprecated in Python 3.10+)
- **Bug fix** `oauth/storage.py`: `delete_credentials()` skips file write if provider wasn't present
- **Bug fix** `langchain/chat_model.py`: `AIMessage` with list content now extracts text from `{"type": "text", "text": "..."}` blocks instead of `str([...])`
- **Robustness** `providers/openai_codex.py`: CRLF/CR line ending normalization in SSE parser
- **Robustness** `providers/openai_codex.py`: Removed misleading empty `ThinkingDeltaEvent(thinking="")` at reasoning block end
- **Robustness** `mcp/client.py`: Improved binary content handling — distinguishes `bytes`/`bytearray` ("N bytes"), `EmbeddedResource` text extraction, MIME type in summary
- **Docs** `mcp/hub.py`: Fixed docstring — both tools get namespaced on collision, not just the second
- **Tests** Added 71 new tests (132 → 203 total):
  - `test_stream_processor.py` (27 tests): `_StreamProcessor` state machine
  - `test_sse_parser.py` (17 tests): SSE parsing and CRLF normalization
  - `test_message_transform.py` (+25 tests): `_clamp_reasoning_effort`, `_make_tc_id`, edge cases
  - `test_langchain.py` (+2 tests): list content handling

### 2026-03-19 — MCP + LangChain + from_toml
- Added native MCP integration: `MCPServer`, `MCPClient`, `MCPHub`, `agent()`
- Added `MCPServer.from_toml()` for piai-native TOML config loading
- Added `PiAIChatModel` LangChain adapter
- Added `test_mcp.py` (59 tests) and `test_langchain.py` (44 tests)
- Fixed tool call ID truncation to 64 chars (`_make_tc_id`)
- Added `require_all_servers`, `connect_timeout`, `tool_result_max_chars` params to `agent()`
- Added `docs/mcp.md` full MCP reference

### 2026-03-19 — Initial port + gap fixes
- Ported entire openai-codex provider from JS SDK (@mariozechner/pi-ai)
- OAuth PKCE flow, token refresh, auth.json storage
- SSE stream processor state machine (`_StreamProcessor`) mirroring JS `processResponsesStream()`
- Added `_clamp_reasoning_effort()` (Fix 1)
- Added usage limit retry exclusion (Fix 2)
- Full `_StreamProcessor` rewrite to faithful JS state machine port (Fix 3)
- `response.failed` extracts `incomplete_details.reason` (Fix 4)
- Fixed usage calculation: subtracts `cached_tokens` from `input_tokens`
- Renamed package from `pyai` → `piai` (PyPI name + module + CLI)
- Created `docs/` folder with architecture, internals, contributing, and AGENTS.md
