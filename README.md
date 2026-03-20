# piai

Python port of [@mariozechner/pi-ai](https://github.com/badlogic/pi-mono).

Use your ChatGPT Plus/Pro subscription to access GPT models from Python — no API key, no per-token billing. Authenticates via OAuth using your existing ChatGPT account and streams completions directly from ChatGPT's backend.

---

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- A ChatGPT Plus or Pro subscription

---

## Installation

```bash
uv add pi-ai-py
# or
pip install pi-ai-py
```

From source:

```bash
git clone https://github.com/Xplo8E/piai
cd piai
uv sync
```

---

## Login

Run once to authenticate. Opens a browser for you to log in with your ChatGPT account.

```bash
piai login
```

Credentials are saved to `auth.json` in the current directory. Keep it private — add to `.gitignore`. If you've already logged in with the JS CLI (`npx @mariozechner/pi-ai login openai-codex`), the same `auth.json` works without re-logging in.

---

## CLI

```bash
piai run "Explain async/await in Python"
piai run "What is 2+2?" --model gpt-5.1
piai run "Summarize this" --system "You are a concise assistant"
piai status      # check login state
piai logout
```

---

## Python API

### stream()

Streams the model response as typed events.

```python
import asyncio
from piai import stream
from piai.types import Context, UserMessage, TextDeltaEvent, DoneEvent

async def main():
    ctx = Context(
        system_prompt="You are a helpful assistant.",
        messages=[UserMessage(content="What is the capital of France?")]
    )
    async for event in stream("gpt-5.1-codex-mini", ctx):
        if isinstance(event, TextDeltaEvent):
            print(event.text, end="", flush=True)
        elif isinstance(event, DoneEvent):
            print()
            print(f"Tokens: {event.message.usage['input']} in, {event.message.usage['output']} out")

asyncio.run(main())
```

### complete()

Returns the full `AssistantMessage` after the response finishes.

```python
from piai import complete
from piai.types import Context, UserMessage

msg = await complete("gpt-5.1-codex-mini", ctx)
print(msg.text)          # convenience property — no need to iterate content blocks
print(msg.stop_reason)
print(msg.usage)
```

### complete_text()

Simplest interface — returns the response as a plain string.

```python
from piai import complete_text

text = await complete_text("gpt-5.1-codex-mini", ctx)
print(text)
```

---

## Multi-turn conversations

Append messages to `context.messages` to keep conversation history:

```python
from piai import complete
from piai.types import Context, UserMessage

ctx = Context(system_prompt="You are a helpful assistant.")

ctx.messages.append(UserMessage(content="My name is Vinay."))
response = await complete("gpt-5.1-codex-mini", ctx)
ctx.messages.append(response)

ctx.messages.append(UserMessage(content="What's my name?"))
response = await complete("gpt-5.1-codex-mini", ctx)
print(response.text)
```

---

## Tool calling

Define tools with a JSON Schema `parameters` dict:

```python
import asyncio
from piai import stream
from piai.types import (
    Context, UserMessage, ToolResultMessage, Tool,
    ToolCallEndEvent, TextDeltaEvent, DoneEvent,
)

def get_weather(city: str) -> str:
    return f"The weather in {city} is sunny, 22°C."

async def main():
    ctx = Context(
        system_prompt="You are a helpful assistant with access to weather data.",
        messages=[UserMessage(content="What's the weather in London?")],
        tools=[
            Tool(
                name="get_weather",
                description="Get current weather for a city.",
                parameters={
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            )
        ],
    )

    # First turn — model calls the tool
    tool_calls = []
    async for event in stream("gpt-5.1-codex-mini", ctx):
        if isinstance(event, ToolCallEndEvent):
            tool_calls.append(event.tool_call)
        elif isinstance(event, DoneEvent):
            ctx.messages.append(event.message)

    # Execute tools and feed results back
    for tc in tool_calls:
        result = get_weather(**tc.input)
        ctx.messages.append(ToolResultMessage(tool_call_id=tc.id, content=result))

    # Second turn — final answer
    async for event in stream("gpt-5.1-codex-mini", ctx):
        if isinstance(event, TextDeltaEvent):
            print(event.text, end="", flush=True)
    print()

asyncio.run(main())
```

---

## Model reasoning (thinking)

Reasoning models (gpt-5.x) think through problems before responding. piai surfaces this reasoning in real time via events, and accumulates the full text on the final message.

### How it works

For each reasoning block the model emits:

1. `ThinkingStartEvent` — reasoning block opens
2. `ThinkingDeltaEvent(thinking)` — incremental chunks (may be sparse — backend controls granularity)
3. `ThinkingEndEvent(thinking)` — block closes; `thinking` field always has the full accumulated text

After the run, `result.thinking` returns the full reasoning across all turns (`None` if the model didn't reason).

> The backend sometimes skips streaming deltas for short reasoning — `ThinkingEndEvent.thinking` is always authoritative.

### Controlling reasoning depth

```python
options = {
    "reasoning_effort": "medium",    # low / medium / high
    "reasoning_summary": "auto",     # auto / concise / detailed / off
}
```

### Accessing reasoning after the run

```python
result = await agent(...)

print(result.text)       # full response text
if result.thinking:
    print(f"Model reasoned {len(result.thinking)} chars")
    print(result.thinking[:500])
```

---

## MCP tool servers + agentic loop

piai has a native MCP client. Pass any MCP server — radare2, IDA Pro, filesystem, web search — and `agent()` auto-discovers tools, runs the model in a loop, executes tool calls, and returns when the model stops.

```python
import asyncio
from piai import agent
from piai.mcp import MCPServer
from piai.types import Context, UserMessage

ctx = Context(
    system_prompt="You are an expert ARM64 reverse engineer.",
    messages=[UserMessage(content="Analyze /lib/target.so and report all JNI functions.")],
)

result = await agent(
    model_id="gpt-5.1-codex-mini",
    context=ctx,
    mcp_servers=[
        MCPServer.stdio("r2pm -r r2mcp"),             # radare2
        MCPServer.stdio("ida-mcp", name="ida"),        # IDA Pro headless
        MCPServer.http("http://127.0.0.1:13337/mcp"),  # IDA Pro HTTP server
    ],
    options={"reasoning_effort": "medium"},
    max_turns=30,
)

print(result.text)
```

**Transport types:**

| Transport | Usage |
|---|---|
| `MCPServer.stdio("cmd --args")` | Spawns a local subprocess |
| `MCPServer.http("http://host/mcp")` | Streamable HTTP (modern) |
| `MCPServer.sse("http://host/sse")` | Legacy SSE transport |

**With auth:**

```python
MCPServer.http("https://api.example.com/mcp", bearer_token="my-token")
MCPServer.stdio("my-server", env_extra={"API_KEY": "secret"})
```

**Load from TOML config:**

```toml
# ~/.piai/config.toml
[mcp_servers.r2]
command = "r2pm"
args = ["-r", "r2mcp"]

[mcp_servers.ida]
command = "ida-mcp"

[mcp_servers.ida-http]
url = "http://127.0.0.1:13337/mcp"
```

```python
from piai.mcp import MCPServer
servers = MCPServer.from_toml("~/.piai/config.toml")
result = await agent(model_id="gpt-5.1-codex-mini", context=ctx, mcp_servers=servers)
```

**All agent() options:**

```python
result = await agent(
    model_id="gpt-5.1-codex-mini",
    context=ctx,
    mcp_servers=[...],
    options={"reasoning_effort": "medium"},
    max_turns=20,                    # safety limit on loop iterations
    on_event=my_callback,            # sync or async — called for every StreamEvent
    require_all_servers=False,       # True = raise if any server fails to connect
    connect_timeout=60.0,            # per-server connection timeout in seconds
    tool_result_max_chars=32_000,    # max chars per tool result (prevents context explosion)
)
```

If you pass both `context.tools` and `mcp_servers`, they are merged. MCP tools take priority on name conflicts.

See [docs/mcp.md](docs/mcp.md) for the full MCP reference.

---

## Full observability

`on_event` fires for every event in the stream — text, reasoning, tool calls, and results. Use it to see exactly what the model is thinking and doing in real time.

### Event reference

**Stream events** (from `stream()` and `agent()`):

| Event | Fields | Description |
|---|---|---|
| `TextStartEvent` | — | Model started producing text |
| `TextDeltaEvent` | `text: str` | Incremental text chunk |
| `TextEndEvent` | `text: str` | Full accumulated text for this block |
| `ThinkingStartEvent` | — | Model started a reasoning block |
| `ThinkingDeltaEvent` | `thinking: str` | Incremental reasoning chunk |
| `ThinkingEndEvent` | `thinking: str` | Full reasoning text for this block |
| `ToolCallStartEvent` | `tool_call: ToolCall` | Model started a tool call |
| `ToolCallDeltaEvent` | `id: str`, `json_delta: str` | Partial tool call arguments |
| `ToolCallEndEvent` | `tool_call: ToolCall` | Complete tool call with parsed input |
| `DoneEvent` | `reason: str`, `message: AssistantMessage` | Stream complete |
| `ErrorEvent` | `reason: str`, `error: AssistantMessage` | Stream failed |

**Agent loop events** (from `agent()` only):

| Event | Fields | Description |
|---|---|---|
| `AgentToolCallEvent` | `turn: int`, `tool_name: str`, `tool_input: dict` | Just before a tool is executed |
| `AgentToolResultEvent` | `turn: int`, `tool_name: str`, `tool_input: dict`, `result: str`, `error: bool` | After tool execution completes |
| `AgentTurnEndEvent` | `turn: int`, `thinking: str \| None`, `tool_calls: list[ToolCall]` | Summary at end of each loop turn |

`DoneEvent.reason` values: `"stop"`, `"length"`, `"tool_use"`, `"error"`

### Full observability example

```python
import asyncio
from piai import agent
from piai.mcp import MCPServer
from piai.types import (
    Context, UserMessage,
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
        # .thinking always has the full text even if deltas were sparse
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

async def main():
    ctx = Context(
        system_prompt="You are an expert reverse engineer.",
        messages=[UserMessage(content="Analyze /lib/target.so")],
    )
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

asyncio.run(main())
```

> The `on_event` callback can be async — piai awaits it automatically. You can write to websockets, queues, or any async sink directly.

See the [examples/](examples/) directory for complete runnable examples including radare2, IDA Pro, LangChain, and LangGraph supervisor.

---

## Options

```python
options = {
    "session_id": "my-session",      # enables prompt caching across calls
    "reasoning_effort": "high",      # low / medium / high — how hard the model thinks
    "reasoning_summary": "auto",     # auto / concise / detailed / off — thinking verbosity
    "text_verbosity": "medium",      # low / medium / high
}
```

> `temperature` is **not supported** by the ChatGPT backend — the API returns an error if you pass it.

---

## Supported models

Any model your ChatGPT Plus/Pro subscription can access. Common ones:

- `gpt-5.1-codex-mini` — fast, default
- `gpt-5.1` — more capable
- `gpt-5.1-codex-max`
- `gpt-5.2`, `gpt-5.2-codex`
- `gpt-5.3-codex`, `gpt-5.3-codex-spark`
- `gpt-5.4`

The model ID is passed directly to the backend — use whatever appears in ChatGPT's model picker.

---

## LangChain integration

`PiAIChatModel` is a drop-in LangChain `BaseChatModel` backed by piai. Works anywhere LangChain accepts a chat model.

```bash
pip install "pi-ai-py[langgraph]"
```

```python
from piai.langchain import PiAIChatModel
from langchain_core.messages import HumanMessage

llm = PiAIChatModel(
    model_name="gpt-5.1-codex-mini",
    options={"reasoning_effort": "medium"},
)

# Sync invoke
result = llm.invoke([HumanMessage(content="What is 2+2?")])
print(result.content)

# Reasoning is surfaced via additional_kwargs
if result.additional_kwargs.get("thinking"):
    print(f"Model thought: {result.additional_kwargs['thinking'][:200]}...")

# Async stream — get thinking deltas live
async for chunk in llm.astream([HumanMessage(content="Solve step by step: 23 * 47")]):
    if chunk.additional_kwargs.get("thinking_delta"):
        print(chunk.additional_kwargs["thinking_delta"], end="", flush=True)
    elif chunk.content:
        print(chunk.content, end="", flush=True)

# Tool binding
llm_with_tools = llm.bind_tools([my_tool])
result = llm_with_tools.invoke([HumanMessage(content="Use the tool")])
```

---

## LangGraph integration

piai integrates with [LangGraph](https://langchain-ai.github.io/langgraph/) for multi-agent workflows.

### MCP → LangChain tool bridge

Convert any MCP server into LangChain `BaseTool` instances for use in LangGraph agents:

```python
from piai.mcp import MCPHubToolset, MCPServer
from piai.langchain import PiAIChatModel
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

llm = PiAIChatModel(model_name="gpt-5.1-codex-mini")
servers = [MCPServer.stdio("npx -y @modelcontextprotocol/server-filesystem /tmp")]

async with MCPHubToolset(servers) as tools:
    agent = create_agent(model=llm, tools=tools, system_prompt="You are a file explorer.")
    result = await agent.ainvoke({"messages": [HumanMessage(content="List files in /tmp")]})
    print(result["messages"][-1].content)
```

### SubAgentTool — piai agent as a LangGraph tool

Wrap a full piai `agent()` (with its own model, MCP servers, and observability) as a single callable tool for a supervisor:

```python
from piai.langchain import SubAgentTool, PiAIChatModel
from piai.mcp import MCPServer
from langgraph_supervisor import create_supervisor
from langchain_core.messages import HumanMessage

file_agent = SubAgentTool(
    name="file_agent",
    description="Reads, writes, and analyses files.",
    model_id="gpt-5.1-codex-mini",
    system_prompt="You are a file management specialist.",
    mcp_servers=[MCPServer.stdio("npx -y @modelcontextprotocol/server-filesystem /tmp")],
    max_turns=8,
    on_event=on_event,   # full inner-loop observability on the sub-agent
)

workflow = create_supervisor(
    tools=[file_agent],
    model=PiAIChatModel(model_name="gpt-5.1-codex-mini"),
    prompt="You are a supervisor. Delegate tasks to specialists.",
).compile()

result = await workflow.ainvoke({"messages": [HumanMessage(content="Summarize /tmp/app.py")]})
print(result["messages"][-1].content)
```

See [`examples/langgraph_supervisor_agent.py`](examples/langgraph_supervisor_agent.py) for the full runnable example.

---

## auth.json format

```json
{
  "openai-codex": {
    "refresh": "<refresh_token>",
    "access": "<access_token>",
    "expires": 1234567890000,
    "accountId": "<account_id>"
  }
}
```

**Never commit `auth.json` to version control.**

---

## Project structure

```
src/piai/
├── __init__.py              # Public API: stream, complete, complete_text, agent, MCPServer
├── types.py                 # Context, messages, stream events
├── stream.py                # Entry points with auth handling
├── agent.py                 # Autonomous agentic loop with MCP support
├── cli.py                   # CLI commands
├── mcp/
│   ├── server.py            # MCPServer config (stdio/http/sse + from_toml)
│   ├── client.py            # MCPClient — persistent session per server
│   ├── hub.py               # MCPHub — multi-server manager
│   └── langchain_tools.py   # MCP → LangChain tool bridge (MCPHubToolset)
├── langchain/
│   ├── chat_model.py        # PiAIChatModel — LangChain BaseChatModel adapter
│   └── sub_agent_tool.py    # SubAgentTool — piai agent as LangChain BaseTool
├── oauth/
│   ├── pkce.py              # PKCE verifier/challenge (RFC 7636)
│   ├── types.py             # OAuthCredentials, OAuthProviderInterface
│   ├── storage.py           # auth.json read/write
│   ├── openai_codex.py      # ChatGPT Plus OAuth login + refresh
│   └── __init__.py          # Provider registry + get_oauth_api_key()
└── providers/
    ├── message_transform.py # Context → OpenAI Responses API format
    └── openai_codex.py      # SSE streaming to chatgpt.com/backend-api
```

---

## Running tests

```bash
uv run pytest tests/ -v
```

---

## Thanks

Built on top of [@mariozechner/pi-mono](https://github.com/badlogic/pi-mono) — the original JS SDK that reverse-engineered the ChatGPT OAuth flow and backend API. The Python port follows the same provider architecture and auth design.

LangChain and LangGraph integrations use [LangChain](https://github.com/langchain-ai/langchain) and [LangGraph](https://github.com/langchain-ai/langgraph) by LangChain AI.
