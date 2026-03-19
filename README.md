# piai

Python port of [@mariozechner/pi-ai](https://github.com/badlogic/pi-mono) — use your **ChatGPT Plus/Pro subscription** to access GPT models from Python, without paying per-token API rates.

Authenticates via OAuth using your existing ChatGPT account, then streams completions from ChatGPT's internal backend. No OpenAI API key needed.

---

## How it works

The library logs in to ChatGPT using the same OAuth flow the official web app uses. It stores a refresh token in `auth.json` locally and auto-refreshes it before each request. Your Plus/Pro subscription grants access — no separate API billing.

---

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- A ChatGPT Plus or Pro subscription

---

## Installation

### From source

```bash
git clone https://github.com/Xplo8E/piai
cd piai
uv sync
```

### As a dependency in your project

```bash
uv add pi-ai-py
```

Or with pip:

```bash
pip install pi-ai-py
```

---

## Setup: Login

Run once to authenticate. Opens a browser for you to log in with your ChatGPT account.

```bash
uv run piai login
# or after installing as a package:
piai login
```

Credentials are saved to `auth.json` in your current working directory. Keep this file private — add it to `.gitignore`.

---

## CLI usage

```bash
# Quick one-shot prompt
piai run "Explain async/await in Python"

# Specify a model
piai run "What is 2+2?" --model gpt-5.1

# With a system prompt
piai run "Summarize this" --system "You are a concise assistant"

# Check login status
piai status

# List available OAuth providers
piai list

# Log out
piai logout
```

---

## Python API

### `stream(model_id, context, options?)` → `AsyncGenerator[StreamEvent]`

Streams the model response as typed events. Handles auth and token refresh automatically.

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
            print()  # newline at end
            print(f"Tokens used: {event.message.usage['input']} in, {event.message.usage['output']} out")

asyncio.run(main())
```

### `complete(model_id, context, options?)` → `AssistantMessage`

Collects the full response and returns an `AssistantMessage`.

```python
import asyncio
from piai import complete
from piai.types import Context, UserMessage

async def main():
    ctx = Context(messages=[UserMessage(content="Write a haiku about Python.")])
    msg = await complete("gpt-5.1-codex-mini", ctx)

    for block in msg.content:
        from piai.types import TextContent
        if isinstance(block, TextContent):
            print(block.text)

    print(f"Stop reason: {msg.stop_reason}")
    print(f"Usage: {msg.usage}")

asyncio.run(main())
```

### `complete_text(model_id, context, options?)` → `str`

Simplest interface — returns the full response text as a string.

```python
import asyncio
from piai import complete_text
from piai.types import Context, UserMessage

async def main():
    ctx = Context(messages=[UserMessage(content="What is 2 + 2?")])
    text = await complete_text("gpt-5.1-codex-mini", ctx)
    print(text)

asyncio.run(main())
```

---

## Multi-turn conversations

Append messages to `context.messages` to continue a conversation:

```python
import asyncio
from piai import complete
from piai.types import Context, UserMessage

async def main():
    ctx = Context(system_prompt="You are a helpful assistant.")

    ctx.messages.append(UserMessage(content="My name is Vinay."))
    response = await complete("gpt-5.1-codex-mini", ctx)
    ctx.messages.append(response)  # add assistant reply to history

    ctx.messages.append(UserMessage(content="What's my name?"))
    response = await complete("gpt-5.1-codex-mini", ctx)
    ctx.messages.append(response)

    from piai.types import TextContent
    for block in response.content:
        if isinstance(block, TextContent):
            print(block.text)

asyncio.run(main())
```

---

## Tool calling (function calling)

Define tools with a JSON Schema `parameters` dict:

```python
import asyncio
import json
from piai import stream
from piai.types import (
    Context, UserMessage, ToolResultMessage, Tool,
    ToolCallStartEvent, ToolCallEndEvent, TextDeltaEvent, DoneEvent,
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
                    "properties": {
                        "city": {"type": "string", "description": "City name"}
                    },
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
            ctx.messages.append(event.message)  # add assistant message to history

    # Execute tools and feed results back
    for tc in tool_calls:
        result = get_weather(**tc.input)
        ctx.messages.append(ToolResultMessage(tool_call_id=tc.id, content=result))

    # Second turn — model produces final answer
    async for event in stream("gpt-5.1-codex-mini", ctx):
        if isinstance(event, TextDeltaEvent):
            print(event.text, end="", flush=True)
    print()

asyncio.run(main())
```

---

## Stream events reference

All events yielded by `stream()`:

| Event | Fields | Description |
|---|---|---|
| `TextStartEvent` | — | Model started producing text |
| `TextDeltaEvent` | `text: str` | Incremental text chunk |
| `TextEndEvent` | `text: str` | Full accumulated text for this block |
| `ThinkingDeltaEvent` | `thinking: str` | Incremental reasoning chunk (reasoning models) |
| `ToolCallStartEvent` | `tool_call: ToolCall` | Model started a tool call |
| `ToolCallDeltaEvent` | `id: str`, `json_delta: str` | Partial tool call arguments |
| `ToolCallEndEvent` | `tool_call: ToolCall` | Complete tool call with parsed input |
| `DoneEvent` | `reason: str`, `message: AssistantMessage` | Stream complete |
| `ErrorEvent` | `reason: str`, `error: AssistantMessage` | Stream failed |

`DoneEvent.reason` values: `"stop"`, `"length"`, `"tool_use"`, `"error"`, `"aborted"`

---

## Options

Pass an `options` dict to `stream()` or `complete()`:

```python
options = {
    "session_id": "my-session",        # enables prompt caching across calls
    "reasoning_effort": "high",        # for reasoning models (gpt-5.x): low/medium/high
    "reasoning_summary": "auto",       # auto/concise/detailed/off
    "text_verbosity": "medium",        # low/medium/high
}
```

> **Note:** `temperature` is **not supported** by the ChatGPT backend — the API will return an error if you pass it.

---

## Supported models

Any model your ChatGPT Plus/Pro subscription can access. Common ones:

- `gpt-5.1-codex-mini` — fast, default
- `gpt-5.1` — more capable
- `gpt-5.1-codex-max`
- `gpt-5.2`, `gpt-5.2-codex`
- `gpt-5.3-codex`, `gpt-5.3-codex-spark`
- `gpt-5.4`

The model ID is passed directly to the backend — use whatever ChatGPT shows in its model picker.

---

## `auth.json` format

Credentials are stored as JSON, compatible with the original JS `pi-ai` SDK:

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

If you've already logged in using the JS CLI (`npx @mariozechner/pi-ai login openai-codex`), the same `auth.json` works with `piai` without re-logging in.

**Never commit `auth.json` to version control.**

---

## MCP tool servers

piai has a native MCP (Model Context Protocol) client. Pass any MCP server — radare2, IDA Pro, filesystem, web search, or any custom server — and the agent auto-discovers tools and runs the agentic loop for you.

```python
import asyncio
from piai import agent
from piai.mcp import MCPServer
from piai.types import Context, UserMessage, TextDeltaEvent

async def main():
    ctx = Context(
        system_prompt="You are an expert reverse engineer.",
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
        on_event=lambda e: print(e.text, end="", flush=True) if isinstance(e, TextDeltaEvent) else None,
    )

asyncio.run(main())
```

**Transport types:**
- `MCPServer.stdio("command --args")` — spawns a local subprocess
- `MCPServer.http("http://host/mcp")` — Streamable HTTP (modern)
- `MCPServer.sse("http://host/sse")` — legacy SSE transport

**Auth shorthand:**
```python
MCPServer.http("https://api.example.com/mcp", bearer_token="my-token")
MCPServer.stdio("my-server", env_extra={"API_KEY": "secret"})
```

**Load from a TOML config file:**

Create `~/.piai/config.toml` (or any path you prefer):

```toml
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

[mcp_servers.with-env]
command = "my-server"

[mcp_servers.with-env.env_extra]
API_KEY = "secret"
```

Then load in one line:

```python
from piai.mcp import MCPServer

servers = MCPServer.from_toml("~/.piai/config.toml")
result = await agent(model_id="gpt-5.1-codex-mini", context=ctx, mcp_servers=servers)
```

**agent() options:**
```python
result = await agent(
    model_id="gpt-5.1-codex-mini",
    context=ctx,
    mcp_servers=[...],
    options={"reasoning_effort": "medium"},
    max_turns=20,                    # safety limit on agentic iterations
    on_event=my_callback,            # sync or async callback for every StreamEvent
    require_all_servers=False,       # True = raise if any server fails to connect
    connect_timeout=60.0,            # per-server connection timeout in seconds
    tool_result_max_chars=32_000,    # max chars per tool result (prevents context explosion)
)
```

**Pre-defined tools + MCP:** If you pass both `context.tools` and `mcp_servers`, they are merged. MCP tools take priority on name conflicts; your pre-defined tools are appended de-duplicated.

See [docs/mcp.md](docs/mcp.md) for the full MCP reference.

---

## LangChain integration

`PiAIChatModel` is a drop-in LangChain `BaseChatModel` backed by piai. Use it anywhere LangChain accepts a chat model — chains, agents, tools.

```python
from piai.langchain import PiAIChatModel
from langchain_core.messages import HumanMessage

llm = PiAIChatModel(model_name="gpt-5.1-codex-mini")

# Invoke
result = llm.invoke([HumanMessage(content="What is 2+2?")])
print(result.content)

# Stream
async for chunk in llm.astream([HumanMessage(content="Tell me a joke")]):
    print(chunk.content, end="", flush=True)

# With tools (works with any LangChain agent or tool framework)
llm_with_tools = llm.bind_tools([my_tool])
result = llm_with_tools.invoke([HumanMessage(content="Use the tool")])
```

Install with LangGraph extras:

```bash
pip install "pi-ai-py[langgraph]"
```

---

## LangGraph integration

piai integrates with [LangGraph](https://langchain-ai.github.io/langgraph/) for building multi-agent workflows. Two components are provided:

### MCP → LangChain tool bridge

Convert MCP servers into LangChain `BaseTool` instances so LangGraph agents can use them directly:

```python
from piai.mcp import to_langchain_tools, MCPServer, MCPHubToolset
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from piai.langchain import PiAIChatModel

servers = [MCPServer.stdio("npx -y @modelcontextprotocol/server-filesystem /tmp")]
llm = PiAIChatModel(model_name="gpt-5.1-codex-mini")

async with MCPHubToolset(servers) as tools:
    agent = create_react_agent(llm, tools)
    result = await agent.ainvoke({"messages": [HumanMessage(content="List files in /tmp")]})
    print(result["messages"][-1].content)
```

### SubAgentTool — piai agent as a LangGraph tool

Wrap a full piai `agent()` (with its own model + MCP servers) as a single `BaseTool`. Use as a sub-agent inside a LangGraph Supervisor:

```python
from piai.langchain import SubAgentTool
from piai.mcp import MCPServer

file_agent = SubAgentTool(
    name="file_agent",
    description="Reads, writes, and analyses files using the filesystem MCP server",
    model_id="gpt-5.1-codex-mini",
    system_prompt="You are a file management specialist.",
    mcp_servers=[MCPServer.stdio("npx -y @modelcontextprotocol/server-filesystem /tmp")],
)
```

### LangGraph Supervisor example

```python
from piai.langchain import PiAIChatModel, SubAgentTool
from piai.mcp import MCPServer
from langgraph_supervisor import create_supervisor
from langchain_core.messages import HumanMessage

supervisor_llm = PiAIChatModel(model_name="gpt-5.1-codex-mini")
file_agent = SubAgentTool(name="file_agent", description="...", mcp_servers=[...])
code_agent = SubAgentTool(name="code_agent", description="...", mcp_servers=[...])

workflow = create_supervisor(
    agents=[file_agent, code_agent],
    model=supervisor_llm,
    prompt="You are a supervisor. Delegate tasks to the appropriate specialist.",
).compile()

result = await workflow.ainvoke({"messages": [HumanMessage(content="Analyse the code in /tmp/app.py")]})
```

See [`examples/langgraph_supervisor_agent.py`](examples/langgraph_supervisor_agent.py) for a full runnable example.

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
│   └── langchain_tools.py   # MCP → LangChain tool bridge (to_langchain_tools, MCPHubToolset)
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
