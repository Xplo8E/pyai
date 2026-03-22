# MCP Integration

piai has a native MCP (Model Context Protocol) client. Pass any MCP server and `agent()` auto-discovers tools, runs the model in a loop, executes tool calls, and returns when the model stops. No LangChain, no mcpo, no manual wrappers needed.

---

## Quick start

```python
import asyncio
from piai import agent
from piai.mcp import MCPServer
from piai.types import Context, UserMessage

ctx = Context(messages=[UserMessage(content="Analyze /lib/target.so")])

result = await agent(
    model_id="gpt-5.1-codex-mini",
    context=ctx,
    mcp_servers=[MCPServer.stdio("r2pm -r r2mcp")],
    options={"reasoning_effort": "medium"},
    max_turns=30,
)

print(result.text)
if result.thinking:
    print(f"Model reasoned {len(result.thinking)} chars")
```

piai will:
1. Spawn the MCP server subprocess
2. Auto-discover all tools it exposes
3. Run the model in a loop — executing tool calls, feeding results back
4. Return the final `AssistantMessage` when the model stops

`result.text` gives you the full response. `result.thinking` gives you the full reasoning text (or `None` if the model didn't reason).

---

## MCPServer transports

### stdio — local subprocess

Spawns a process and communicates via stdin/stdout. Uses `shlex.split` so paths with spaces work correctly.

```python
MCPServer.stdio("r2pm -r r2mcp")

# With explicit name (used as namespace prefix on tool collisions)
MCPServer.stdio("ida-mcp", name="ida")

# Add env vars on top of parent env (preserves PATH etc.)
MCPServer.stdio("my-server", env_extra={"API_KEY": "secret"})

# Replace env entirely (use carefully)
MCPServer.stdio("my-server", env={"ONLY": "this"})

# Paths with spaces
MCPServer.stdio('"/path with spaces/server" --flag')
```

### http — Streamable HTTP (modern)

Recommended for remote or long-running servers.

```python
MCPServer.http("http://127.0.0.1:13337/mcp")
MCPServer.http("http://127.0.0.1:13337/mcp", name="ida")
MCPServer.http("https://api.example.com/mcp", bearer_token="my-token")
MCPServer.http("https://api.example.com/mcp", headers={"X-Api-Key": "abc"})
```

### sse — Server-Sent Events (legacy)

For older MCP servers that use SSE instead of Streamable HTTP.

```python
MCPServer.sse("http://localhost:9000/sse")
MCPServer.sse("http://localhost:9000/sse", headers={"Authorization": "Bearer token"})
```

---

## Multiple MCP servers

Pass multiple servers — piai connects to all concurrently, merges their tools into a flat list, and routes each tool call to the right server automatically.

```python
result = await agent(
    model_id="gpt-5.1-codex-mini",
    context=ctx,
    mcp_servers=[
        MCPServer.stdio("r2pm -r r2mcp"),
        MCPServer.stdio("npx @modelcontextprotocol/server-filesystem /tmp"),
        MCPServer.http("http://127.0.0.1:13337/mcp", name="ida"),
    ],
)
```

### Tool name collisions

If two servers expose a tool with the same name, both are namespaced:

```
server1: read_file  →  s1__read_file
server2: read_file  →  s2__read_file
```

A warning is logged. Set explicit names to control the prefix:

```python
MCPServer.stdio("r2pm -r r2mcp", name="r2")   # collision → r2__tool_name
MCPServer.stdio("ida-mcp", name="ida")          # collision → ida__tool_name
```

### Loading from config files

```python
# From TOML (e.g. ~/.piai/config.toml)
servers = MCPServer.from_toml("~/.piai/config.toml")
result = await agent(model_id="gpt-5.1-codex-mini", context=ctx, mcp_servers=servers)
```

```toml
# ~/.piai/config.toml
[mcp_servers.r2]
command = "r2pm"
args = ["-r", "r2mcp"]

[mcp_servers.ida]
command = "ida-mcp"

[mcp_servers.ida-http]
url = "http://127.0.0.1:13337/mcp"

[mcp_servers.remote]
url = "https://api.example.com/mcp"
bearer_token = "my-token"
```

Or from a dict directly:

```python
MCPServer.from_config({"command": "ida-mcp", "name": "ida"})
MCPServer.from_config({"url": "http://127.0.0.1:13337/mcp"})
MCPServer.from_config({"command": "r2pm", "args": ["-r", "r2mcp"]})
```

---

## agent() options

```python
result = await agent(
    model_id="gpt-5.1-codex-mini",
    context=ctx,
    mcp_servers=[...],
    options={
        "reasoning_effort": "medium",   # low / medium / high
        "reasoning_summary": "auto",    # auto / concise / detailed / off
        "session_id": "my-session",     # optional session continuity
    },
    max_turns=20,                       # safety limit on loop iterations (default: 20)
    on_event=my_callback,               # sync or async callback for every StreamEvent
    require_all_servers=False,          # True = raise if any server fails to connect
    connect_timeout=60.0,               # per-server connection timeout in seconds
    tool_result_max_chars=32_000,       # max chars per tool result (prevents context explosion)
    context_reducer=my_reducer,         # optional hook to trim/summarize context each turn
    local_handlers={"tool": fn},        # pure Python tool handlers (sync or async)
)
```

### Context.scratchpad — persistent working memory

Pass `scratchpad` in the `Context` to give the agent persistent state across turns. It is serialized as a `<scratchpad>` JSON block and injected into the system prompt on every LLM call — even if message history is trimmed.

```python
ctx = Context(
    system_prompt="You are a stateful analyst.",
    messages=[UserMessage(content="Track findings across scans.")],
    scratchpad={"findings": [], "scanned": 0},
)

def record_finding(severity: str, description: str) -> str:
    ctx.scratchpad["findings"].append({"severity": severity, "description": description})
    ctx.scratchpad["scanned"] += 1
    return "Recorded."

result = await agent(
    model_id="gpt-5.1-codex-mini",
    context=ctx,
    local_handlers={"record_finding": record_finding},
    mcp_servers=[...],
)
# ctx.scratchpad["findings"] has all findings the model recorded
```

### context_reducer — prevent context explosion

`context_reducer` is called after each turn's tool results are appended, before the next LLM call. Return a trimmed or summarized `Context`.

```python
def sliding_window(ctx: Context) -> Context:
    return Context(
        messages=ctx.messages[-6:],     # keep last 6 messages
        system_prompt=ctx.system_prompt,
        tools=ctx.tools,
        scratchpad=ctx.scratchpad,      # always carry scratchpad forward
    )

result = await agent(
    model_id="gpt-5.1-codex-mini",
    context=ctx,
    mcp_servers=[...],
    max_turns=50,
    context_reducer=sliding_window,
)
```

Async reducers are supported — return a coroutine.

---

## Full observability with on_event

`on_event` fires for every `StreamEvent` in the loop — reasoning, text, tool calls, and results. This gives you complete inner-loop visibility: you can see exactly what the model is thinking, what tools it's calling, and what they returned, all in real time.

### Available events

**Stream events** (model output):

| Event | Fields | Description |
|---|---|---|
| `ThinkingStartEvent` | — | Model started a reasoning block |
| `ThinkingDeltaEvent` | `thinking: str` | Incremental reasoning chunk |
| `ThinkingEndEvent` | `thinking: str` | Full reasoning text for this block |
| `TextDeltaEvent` | `text: str` | Incremental response text chunk |
| `ToolCallStartEvent` | `tool_call: ToolCall` | Model started a tool call |
| `ToolCallEndEvent` | `tool_call: ToolCall` | Complete tool call with parsed input |
| `DoneEvent` | `reason: str`, `message: AssistantMessage` | Stream complete |
| `ErrorEvent` | `reason: str`, `error: AssistantMessage` | Stream failed |

**Agent loop events** (agentic execution):

| Event | Fields | Description |
|---|---|---|
| `AgentToolCallEvent` | `turn: int`, `tool_name: str`, `tool_input: dict` | Just before a tool is executed |
| `AgentToolResultEvent` | `turn: int`, `tool_name: str`, `tool_input: dict`, `result: str`, `error: bool` | After tool execution completes |
| `AgentTurnEndEvent` | `turn: int`, `thinking: str \| None`, `tool_calls: list[ToolCall]`, `usage: dict` | Summary at end of each loop turn |

**`AgentTurnEndEvent.usage` keys:**

| Key | Description |
|---|---|
| `input` | Input tokens this turn |
| `output` | Output tokens this turn |
| `cache_read` | Tokens served from prompt cache |
| `total_tokens` | `input + output` (before cache deduction) |

Use `usage` to track cumulative token spend, implement token budgets, or log per-turn cost:

```python
from piai.types import AgentTurnEndEvent

cumulative = 0

def on_event(event):
    global cumulative
    if isinstance(event, AgentTurnEndEvent):
        cumulative += event.usage.get("total_tokens", 0)
        print(f"Turn {event.turn}: {event.usage}  (total so far: {cumulative})")
        if cumulative > 5000:
            raise RuntimeError("Token budget exceeded")
```

> `ThinkingEndEvent.thinking` always has the full text for the block even if `ThinkingDeltaEvent` was sparse or empty — the backend controls delta granularity.

### Full observability callback

```python
from piai.types import (
    ThinkingStartEvent, ThinkingDeltaEvent, ThinkingEndEvent,
    AgentToolCallEvent, AgentToolResultEvent, AgentTurnEndEvent,
    TextDeltaEvent,
)

DIM, CYAN, GREEN, YELLOW, RESET = "\033[2m", "\033[36m", "\033[32m", "\033[33m", "\033[0m"

def on_event(event):
    if isinstance(event, ThinkingStartEvent):
        print(f"\n{DIM}💭 Thinking...{RESET}", flush=True)

    elif isinstance(event, ThinkingDeltaEvent):
        print(f"{DIM}{event.thinking}{RESET}", end="", flush=True)

    elif isinstance(event, ThinkingEndEvent):
        print(f"\n{DIM}[thinking done]{RESET}\n", flush=True)

    elif isinstance(event, AgentToolCallEvent):
        args = ", ".join(
            f"{k}={v[:60]!r}..." if isinstance(v, str) and len(v) > 60 else f"{k}={v!r}"
            for k, v in event.tool_input.items()
        )
        print(f"\n{CYAN}🔧 Turn {event.turn} → {event.tool_name}({args}){RESET}", flush=True)

    elif isinstance(event, AgentToolResultEvent):
        status = "❌" if event.error else "✅"
        preview = event.result[:200].replace("\n", " ")
        print(f"{GREEN}{status} {preview}{'...' if len(event.result) > 200 else ''}{RESET}", flush=True)

    elif isinstance(event, AgentTurnEndEvent):
        note = f", {len(event.thinking)} chars reasoning" if event.thinking else ""
        print(f"\n{YELLOW}── Turn {event.turn}: {len(event.tool_calls)} call(s){note} ──{RESET}\n", flush=True)

    elif isinstance(event, TextDeltaEvent):
        print(event.text, end="", flush=True)

result = await agent(
    model_id="gpt-5.1-codex-mini",
    context=ctx,
    mcp_servers=[MCPServer.stdio("r2pm -r r2mcp")],
    options={"reasoning_effort": "medium"},
    max_turns=30,
    on_event=on_event,
)

print(f"\nDone. {len(result.text)} chars.")
if result.thinking:
    print(f"Total reasoning: {len(result.thinking)} chars")
```

> The `on_event` callback can be async — piai awaits it automatically. You can write directly to websockets, queues, or any async sink.

---

## Real-world examples

### Binary analysis with radare2

```python
from piai import agent
from piai.mcp import MCPServer
from piai.types import Context, UserMessage

ctx = Context(
    system_prompt="You are an expert ARM64 reverse engineer.",
    messages=[UserMessage(content="Analyze /path/to/lib.so and report all JNI functions.")],
)

result = await agent(
    model_id="gpt-5.1-codex-mini",
    context=ctx,
    mcp_servers=[MCPServer.stdio("r2pm -r r2mcp")],
    options={"reasoning_effort": "medium"},
    max_turns=30,
    on_event=on_event,
)
print(result.text)
```

### Binary analysis with IDA Pro (stdio)

```python
# ida-mcp runs headless IDA Pro — no GUI required
result = await agent(
    model_id="gpt-5.1-codex-mini",
    context=ctx,
    mcp_servers=[MCPServer.stdio("ida-mcp", name="ida")],
    options={"reasoning_effort": "medium"},
    max_turns=40,
    on_event=on_event,
)
```

### Binary analysis with IDA Pro (HTTP server)

```python
# Start the IDA MCP HTTP server first: ida-mcp serve-http --port 13337
result = await agent(
    model_id="gpt-5.1-codex-mini",
    context=ctx,
    mcp_servers=[MCPServer.http("http://127.0.0.1:13337/mcp", name="ida")],
    on_event=on_event,
)
```

### Filesystem operations

```python
MCPServer.stdio("npx -y @modelcontextprotocol/server-filesystem /home/user/projects")
```

### Web search + code analysis together

```python
mcp_servers=[
    MCPServer.stdio("npx -y @modelcontextprotocol/server-brave-search"),
    MCPServer.stdio("r2pm -r r2mcp"),
]
```

See the [examples/](../examples/) directory for complete runnable examples.

---

## Architecture

```
agent()
  │
  ├── MCPHub.connect()          ← connects all servers concurrently
  │     ├── MCPClient(server1)  ← persistent stdio/http/sse session
  │     ├── MCPClient(server2)
  │     └── ...
  │
  ├── hub.all_tools()           ← merged flat tool list → Context.tools
  │
  └── loop:
        stream() → tool calls → hub.call_tool() → ToolResultMessage → continue
                → no tool calls → return AssistantMessage
```

**Why MCPClient uses AsyncExitStack:** MCP servers like `r2mcp` and `ida-mcp` are stateful — you open a file in one call, analyze it in the next. `AsyncExitStack` keeps the subprocess alive for the entire agent session. Without it, each tool call would spawn a fresh process and lose all state.

---

## Using MCPClient / MCPHub directly

For lower-level access without the agent loop:

```python
from piai.mcp import MCPClient, MCPHub, MCPServer

# Single server
async with MCPClient(MCPServer.stdio("r2pm -r r2mcp")) as client:
    tools = await client.list_tools()
    result = await client.call_tool("open_file", {"file_path": "/lib/target.so"})
    print(result)

# Multiple servers
async with MCPHub([
    MCPServer.stdio("r2pm -r r2mcp"),
    MCPServer.stdio("ida-mcp", name="ida"),
]) as hub:
    print(hub.all_tools())
    result = await hub.call_tool("open_file", {"file_path": "/lib/target.so"})
```

---

## Using MCP tools with LangGraph

piai provides a bridge to convert MCP servers into LangChain `BaseTool` instances for use in LangGraph agents and Supervisors.

### MCPHubToolset — async context manager (recommended)

```python
from piai.mcp import MCPServer
from piai.mcp.langchain_tools import MCPHubToolset
from piai.langchain import PiAIChatModel
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

llm = PiAIChatModel(model_name="gpt-5.1-codex-mini")
servers = [MCPServer.stdio("npx -y @modelcontextprotocol/server-filesystem /tmp")]

async with MCPHubToolset(servers, connect_timeout=30.0) as tools:
    agent = create_agent(model=llm, tools=tools, system_prompt="You are a file explorer.")
    result = await agent.ainvoke({"messages": [HumanMessage(content="List /tmp")]})
    print(result["messages"][-1].content)
```

### to_langchain_tools — one-shot conversion

```python
from piai.mcp import MCPServer
from piai.mcp.langchain_tools import to_langchain_tools

servers = [MCPServer.stdio("npx -y @modelcontextprotocol/server-filesystem /tmp")]
tools, hub = await to_langchain_tools(servers)

# use tools in any LangChain/LangGraph agent
# ...

await hub.close()  # clean up when done
```
