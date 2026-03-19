# MCP Integration

piai has a native MCP (Model Context Protocol) client built in. No LangChain, no mcpo, no manual tool wrappers — just pass your MCP server configs and the agent handles everything automatically.

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
    mcp_servers=[
        MCPServer.stdio("r2pm -r r2mcp"),
    ],
)
print(result)
```

That's it. piai will:
1. Spawn the MCP server subprocess
2. Auto-discover all tools it exposes
3. Run the model in a loop, executing tool calls as requested
4. Return the final `AssistantMessage` when the model stops

---

## MCPServer transports

### stdio — local subprocess

Spawns a process and communicates via stdin/stdout. Works with any MCP server that runs as a subprocess. Uses `shlex.split` so paths with spaces are handled correctly.

```python
# Simple command
MCPServer.stdio("r2pm -r r2mcp")

# Path with spaces — handled correctly via shlex.split
MCPServer.stdio('"/path with spaces/server" --flag')

# With explicit name (used for namespacing on tool collisions)
MCPServer.stdio("ida-mcp", name="ida")

# Add env vars ON TOP of parent env (preserves PATH etc.)
MCPServer.stdio("my-server", env_extra={"API_KEY": "secret"})

# Replace env entirely (use carefully)
MCPServer.stdio("my-server", env={"ONLY": "this"})
```

### http — Streamable HTTP (modern)

Connects to an MCP server over HTTP. The recommended transport for remote or long-running servers.

```python
MCPServer.http("http://127.0.0.1:13337/mcp")
MCPServer.http("http://127.0.0.1:13337/mcp", name="ida")
# bearer_token shorthand
MCPServer.http("https://api.example.com/mcp", bearer_token="my-token")
# explicit headers
MCPServer.http("https://api.example.com/mcp", headers={"X-Api-Key": "abc"})
```

### sse — Server-Sent Events (legacy)

For older MCP servers that use the SSE transport instead of Streamable HTTP.

```python
MCPServer.sse("http://localhost:9000/sse")
MCPServer.sse("http://localhost:9000/sse", headers={"Authorization": "Bearer token"})
```

---

## Multiple MCP servers

Pass an array of servers. piai connects to all of them concurrently, merges their tools into a flat list, and routes each tool call to the correct server automatically. The model picks whatever tool fits the task — it doesn't know or care which server it came from.

```python
result = await agent(
    model_id="gpt-5.1-codex-mini",
    context=ctx,
    mcp_servers=[
        MCPServer.stdio("r2pm -r r2mcp"),                           # radare2 tools
        MCPServer.stdio("npx @modelcontextprotocol/server-filesystem /tmp"),  # filesystem
        MCPServer.http("http://127.0.0.1:13337/mcp", name="ida"),   # IDA Pro
    ],
)
```

### Tool name collisions

If two servers expose a tool with the same name, **both** are namespaced with their server name:

```
server1 exposes: read_file       → registered as: s1__read_file
server2 exposes: read_file       → registered as: s2__read_file
```

A warning is logged. The model sees both namespaced versions and picks the right one from context.
Set explicit names to control the namespace prefix:

```python
MCPServer.stdio("r2pm -r r2mcp", name="r2")    # collision → r2__tool_name
MCPServer.stdio("ida-mcp", name="ida")          # collision → ida__tool_name
```

### Loading servers from config files

`MCPServer.from_config()` accepts the same dict format as Claude/Codex `config.toml`:

```python
# From TOML config (e.g. ~/.codex/config.toml mcp_servers section)
import tomllib

with open("~/.codex/config.toml", "rb") as f:
    config = tomllib.load(f)

servers = [
    MCPServer.from_config({**cfg, "name": name})
    for name, cfg in config.get("mcp_servers", {}).items()
]

result = await agent(model_id="gpt-5.1-codex-mini", context=ctx, mcp_servers=servers)
```

```python
# Direct dict config
MCPServer.from_config({"command": "ida-mcp", "name": "ida"})
MCPServer.from_config({"url": "http://127.0.0.1:13337/mcp"})
MCPServer.from_config({"command": "r2pm", "args": ["-r", "r2mcp"]})
MCPServer.from_config({"url": "https://api.example.com/mcp", "bearer_token": "tok"})
```

---

## agent() options

```python
result = await agent(
    model_id="gpt-5.1-codex-mini",       # any supported model
    context=ctx,                           # Context with messages + optional system_prompt
    mcp_servers=[...],                     # list of MCPServer configs (or None for no MCP)
    options={                              # passed to piai stream()
        "reasoning_effort": "medium",      # low / medium / high
        "session_id": "my-session",        # optional session continuity
    },
    provider_id="openai-codex",            # default, don't need to set
    max_turns=20,                          # safety limit on loop iterations (default: 20)
    on_event=my_callback,                  # optional: sync or async callback for every StreamEvent
    require_all_servers=False,             # if True, raise if any server fails to connect
    connect_timeout=60.0,                  # per-server connection timeout in seconds
    tool_result_max_chars=32_000,          # max chars per tool result (prevents context explosion)
)
```

### Live streaming with on_event

```python
from piai.types import TextDeltaEvent, ToolCallEndEvent

def on_event(event):
    if isinstance(event, TextDeltaEvent):
        print(event.text, end="", flush=True)
    elif isinstance(event, ToolCallEndEvent):
        print(f"\n[tool] {event.tool_call.name}({event.tool_call.input})")

result = await agent(..., on_event=on_event)
```

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
)
```

### Binary analysis with IDA Pro (stdio)

```python
# ida-mcp runs headless IDA Pro, no GUI needed
result = await agent(
    model_id="gpt-5.1-codex-mini",
    context=ctx,
    mcp_servers=[MCPServer.stdio("ida-mcp", name="ida")],
    options={"reasoning_effort": "medium"},
    max_turns=40,
)
```

### Binary analysis with IDA Pro (HTTP server)

```python
# Start IDA MCP HTTP server first:
#   ida-mcp serve-http --port 13337
result = await agent(
    model_id="gpt-5.1-codex-mini",
    context=ctx,
    mcp_servers=[MCPServer.http("http://127.0.0.1:13337/mcp", name="ida")],
)
```

### Filesystem operations

```python
MCPServer.stdio("npx @modelcontextprotocol/server-filesystem /home/user/projects")
```

### Web search + code analysis together

```python
mcp_servers=[
    MCPServer.stdio("npx @modelcontextprotocol/server-brave-search"),
    MCPServer.stdio("r2pm -r r2mcp"),
]
```

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
                → stop → return AssistantMessage
```

**Why MCPClient uses AsyncExitStack:**
MCP servers like `r2mcp` and `ida-mcp` are stateful — you open a file in one call, analyze it in the next. `AsyncExitStack` keeps the subprocess/connection alive for the entire agent session. Without it, each tool call would spawn a fresh process and lose all state.

---

## Using MCPClient / MCPHub directly

If you need lower-level access:

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
