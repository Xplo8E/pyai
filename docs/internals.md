# piai — Internals Reference

Deep-dive into how each module works. Read this before modifying any core file.

---

## oauth/pkce.py

Implements RFC 7636 PKCE (Proof Key for Code Exchange).

**`generate_pkce() -> (verifier, challenge)`**
- Generates 32 random bytes → base64url-encodes (no padding) as the verifier
- SHA-256 hashes the verifier → base64url-encodes as the challenge
- Must match JS exactly: `btoa(...).replace(/\+/g,'-').replace(/\//g,'_').replace(/=/g,'')`

Python gotcha: `base64.urlsafe_b64encode()` adds `=` padding — strip it manually.

---

## oauth/types.py

**`OAuthCredentials`**
- `expires` is a **Unix millisecond timestamp** (matching JS `Date.now()`), not seconds
- `to_dict()` / `from_dict()` use **camelCase** (`accountId`) for JS SDK compatibility
- `is_expired(buffer_ms=300000)` — 5-minute default buffer
- Extra provider-specific fields (like `accountId`) live in `extras` dict

**`OAuthProviderInterface`**
- ABC that every provider must implement
- `get_api_key(credentials)` returns the bearer token string for the Authorization header

---

## oauth/storage.py

- Reads/writes `auth.json` in the **current working directory** (same as JS CLI)
- Full file is loaded, updated, and written back atomically (no partial writes)
- Returns `{}` silently if `auth.json` doesn't exist or is malformed
- `delete_credentials()` skips the file write if the provider wasn't present (no unnecessary I/O)

---

## oauth/openai_codex.py

**Constants**
```python
CLIENT_ID      = "app_EMoamEEZ73f0CkXaXp7hrann"
AUTHORIZE_URL  = "https://auth.openai.com/oauth/authorize"
TOKEN_URL      = "https://auth.openai.com/oauth/token"
REDIRECT_URI   = "http://localhost:1455/auth/callback"
SCOPE          = "openid profile email offline_access"
JWT_CLAIM_PATH = "https://api.openai.com/auth"
```
These must match the JS SDK exactly — they're registered OAuth client values.

**`_CallbackServer`**
- Spins up a minimal `HTTPServer` on `127.0.0.1:1455` in a daemon thread
- Uses `threading.Event` to signal when the auth code arrives
- `cancel()` unblocks `wait_for_code()` on Ctrl+C — prevents the daemon thread from printing a traceback during interpreter shutdown
- Returns `False` from `start()` if the port is already in use → falls back to manual paste

**`extract_account_id(access_token)`**
- Decodes the JWT payload (middle segment) without verification
- JWT strips base64 padding — must re-add `=` characters: `payload + "=" * (4 - len(payload) % 4)`
- Extracts `chatgpt_account_id` from the `https://api.openai.com/auth` claim

**`login_openai_codex()`**
- Full PKCE flow: generate verifier/challenge → build auth URL → start server → open browser → wait for callback → exchange code → return credentials
- If `on_manual_code_input` is provided, races browser callback vs manual paste (whichever comes first wins)
- Fallback: if server can't start (port in use), prompts user to paste the redirect URL
- Uses `asyncio.get_running_loop()` (not deprecated `get_event_loop()`)

**`_parse_authorization_input(raw)`**
- Handles three input formats: full redirect URL, query string (`code=...&state=...`), or bare code
- Used in the manual paste fallback

---

## providers/message_transform.py

Converts piai's `Context` / `Message` types to the OpenAI Responses API wire format.

**Message format differences from Chat Completions:**

| piai type | Responses API format |
|-----------|---------------------|
| `UserMessage` | `{"type": "message", "role": "user", "content": [{"type": "input_text", ...}]}` |
| `AssistantMessage` (text) | `{"type": "message", "role": "assistant", "content": [{"type": "output_text", ...}]}` |
| `AssistantMessage` (tool call) | `{"type": "function_call", "name": ..., "arguments": ..., "call_id": ...}` |
| `ToolResultMessage` | `{"type": "function_call_output", "call_id": ..., "output": ...}` |
| `ThinkingContent` | Wrapped in `<thinking>...</thinking>` tags as `output_text` |

**`_clamp_reasoning_effort(model_id, effort)`**
- Model-specific effort value constraints (mirrors JS `clampReasoningEffort()`):
  - `gpt-5.2/5.3/5.4`: `"minimal"` → `"low"`
  - `gpt-5.1`: `"xhigh"` → `"high"`
  - `gpt-5.1-codex-mini`: only supports `"medium"` or `"high"` (all others → `"medium"`, `"high"`/`"xhigh"` → `"high"`)
- Strips provider prefix: `"openai-codex/gpt-5.1"` → `"gpt-5.1"` before checking

**`build_request_body(model_id, context, options)`**
- Always sets `store: False`, `stream: True`, `include: ["reasoning.encrypted_content"]`
- `instructions` defaults to `"You are a helpful assistant."` if no system prompt
- `prompt_cache_key` is set from `options["session_id"]` if provided
- Empty `TextContent` blocks (empty string) are not emitted to the wire format
- **`temperature` is not passed** — the ChatGPT backend rejects it with a 400 error (`"Unsupported parameter: temperature"`)

---

## providers/openai_codex.py

The core streaming provider. POSTs to `https://chatgpt.com/backend-api/codex/responses` and processes the SSE stream.

**`_make_tc_id(call_id, item_id) -> str`**
- The Responses API sends `call_id` and `item_id` separately for tool calls
- Combined as `f"{call_id}|{item_id}"` truncated to 64 chars (API enforces this limit)
- Applied consistently in `ToolCallStartEvent`, `ToolCallDeltaEvent`, and `ToolCallEndEvent`

**`_parse_sse(response)`**
- Normalizes `\r\n` and `\r` to `\n` before parsing (handles CRLF servers)
- Splits on `\n\n` event boundaries, handles partial chunks across network reads
- Skips `[DONE]` sentinel, empty events, and invalid JSON silently

**`_StreamProcessor`** — state machine mirroring JS `processResponsesStream()`

Tracks two pieces of state:
- `current_item` — the active output item (a reasoning block, a message, or a function call)
- `current_block` — the active content block within a message item

SSE event handling:

| SSE event | Action |
|-----------|--------|
| `response.output_item.added` | Set `current_item`, prepare for content |
| `response.reasoning_summary_part.added` | Begin accumulating reasoning text |
| `response.reasoning_summary_text.delta` | Emit `ThinkingDeltaEvent`, accumulate |
| `response.reasoning_summary_part.done` | Append `"\n\n"` separator to thinking block |
| `response.content_part.added` | Set content part type (`output_text` or `refusal`) |
| `response.output_text.delta` | Emit `TextDeltaEvent` (guarded: only if content block active and last part is `output_text`) |
| `response.refusal.delta` | Emit `TextDeltaEvent` (refusals surface as text) |
| `response.function_call_arguments.delta` | Emit `ToolCallDeltaEvent`, accumulate |
| `response.function_call_arguments.done` | Canonicalize args from the done event |
| `response.output_item.done` (reasoning) | Reconstruct thinking from summary parts, append `ThinkingContent` to output |
| `response.output_item.done` (message) | Join content parts, emit `TextEndEvent`, append `TextContent` to output |
| `response.output_item.done` (function_call) | Parse JSON args, emit `ToolCallEndEvent`, append `ToolCallContent` to output |
| `response.completed` | Extract usage + stop reason, emit (nothing — caller emits `DoneEvent`) |
| `response.failed` | Extract error from `response.error` or `incomplete_details.reason`, raise `RuntimeError` |
| `error` | Raise `RuntimeError` immediately |

**Stop reason mapping:**

| API status | `stop_reason` |
|------------|--------------|
| `"completed"` with tool calls | `"tool_use"` |
| `"completed"` without tool calls | `"stop"` |
| `"incomplete"` | `"length"` |
| `"failed"` / `"cancelled"` | `"error"` |

**Retry logic**
- Max 3 retries on network/transient errors
- Retryable: connection errors, 500/502/503/504, rate limits (429)
- **Not** retried: usage limit errors (`"usage limit"` in error message) — no point retrying an exhausted plan
- Exponential backoff: 1s, 2s, 4s

**Usage calculation**
```python
input_tokens  = usage.get("input_tokens", 0)
cached        = usage.get("input_tokens_details", {}).get("cached_tokens", 0)
output_tokens = usage.get("output_tokens", 0)
# Net input = total input minus what was served from cache
net_input = input_tokens - cached
```

---

## stream.py

Thin orchestration layer:
1. Load credentials from `auth.json`
2. Auto-refresh if within 5-minute expiry buffer
3. Save updated credentials back (rotation)
4. Copy `options` dict before calling provider (avoids mutating caller's dict when `base_url` is popped)
5. Call `stream_openai_codex()` and yield events

`complete()` collects all events and returns the final `AssistantMessage` from `DoneEvent`.
`complete_text()` collects only `TextDeltaEvent.text` and returns a plain string.

---

## agent.py

**`agent(model_id, context, mcp_servers, ...)`**

Entry point for the autonomous agentic loop.

1. If `mcp_servers` provided: creates `MCPHub`, connects all servers, discovers tools
2. Merges MCP tools with `context.tools` (MCP takes priority on name conflicts, user tools appended de-duplicated)
3. Runs `stream()` in a loop up to `max_turns`:
   - Collects `ToolCallEndEvent` items each turn
   - On `DoneEvent`: saves `final_message`
   - On `ErrorEvent`: raises `RuntimeError`
   - After stream ends: collects `turn_thinking = final_message.thinking`, fires `AgentTurnEndEvent(turn, thinking, tool_calls)`
   - If no tool calls: model is done, break
   - For each tool call: fire `AgentToolCallEvent` → `_execute_tool()` → fire `AgentToolResultEvent`
   - Appends `AssistantMessage` + `ToolResultMessage` per tool result to context
   - Continues to next turn
4. Returns final `AssistantMessage`

**`_execute_tool(hub, tc, max_chars)`** — never raises:
- `KeyError` (unknown tool) → returns `"Tool not found: ..."` string
- Any other exception → returns `"Tool {name!r} failed: ..."` string
- Error detection: `AgentToolResultEvent.error` is set if result starts with these prefixes

**`_fire_event(callback, event)`** — supports both sync and async `on_event` callbacks via `inspect.isawaitable`.

---

## mcp/server.py

`MCPServer` is a plain dataclass — no connection logic, just configuration.

Factory methods:
- `.stdio(command)` — uses `shlex.split` for correct handling of paths with spaces
- `.http(url)` — Streamable HTTP transport (modern, recommended)
- `.sse(url)` — legacy SSE transport
- `.from_config(dict)` — auto-detects transport from dict keys (`"command"` → stdio, `"url"` → http/sse)
- `.from_toml(path, section="mcp_servers")` — loads piai TOML config format

`env_extra` merges on top of `os.environ` (preserves PATH etc.). `env` replaces entirely.

---

## mcp/client.py

`MCPClient` manages one persistent MCP server connection.

**`connect()`** — wraps `_connect_inner()` with `asyncio.wait_for(timeout=connect_timeout)`. On timeout, cleans up the `AsyncExitStack` and raises `TimeoutError`.

**Transport setup:**
- `stdio`: `StdioServerParameters` → `stdio_client()` → `ClientSession`
- `http`: `streamablehttp_client()` → unpack `(read, write, _)` → `ClientSession`
- `sse`: `sse_client()` → unpack `(read, write)` → `ClientSession`

**`call_tool(name, arguments)`** — result processing:
- `TextContent` (has `.text`): appended as-is
- `ImageContent` (has `.data`): summarized as `[mime data: N bytes]` (preserves MIME type)
- `EmbeddedResource` (has `.resource`): extracts `.text` if available, else summarizes URI
- Any other block: `str(block)`
- `result.isError`: prepended with `"Tool error: "`
- Truncated to `tool_result_max_chars` with a count of remaining chars

---

## mcp/hub.py

`MCPHub` connects to N servers concurrently and routes tool calls.

**`connect()`**: `asyncio.gather(*[c.connect() for c in clients], return_exceptions=True)`. Failed clients are logged and skipped. If `require_all=True`, raises on any failure.

**`_register_tool(tool, client)`** collision handling:
1. No collision → register as `tool.name` → `(client, original_name)`
2. Collision detected:
   - Re-register existing tool under `existing_server__tool_name`
   - Register new tool under `new_server__tool_name`
   - Original unnamespaced name still routes to first server (backward compat)
3. Warning logged on collision

`_safe_name()` replaces `-`, `.`, ` ` with `_` for namespace prefix safety.

---

## langchain/chat_model.py

`PiAIChatModel(BaseChatModel)` fields:
- `model_name: str` — passed to piai `stream()`
- `provider_id: str` — defaults to `"openai-codex"`
- `options: dict` — merged with call-time options (call-time wins)

**`_lc_messages_to_piai()`** — type mapping:
- `SystemMessage` → `context.system_prompt`
- `HumanMessage` → `UserMessage`
- `AIMessage` (str content) → `AssistantMessage` with `TextContent`
- `AIMessage` (list content) → extracts text from `{"type": "text", "text": "..."}` blocks
- `AIMessage` with `tool_calls` → `ToolCallContent` appended to content
- `ToolMessage` → `ToolResultMessage`

**`_run_async()`** — thread-safe async dispatch:
- If no running loop: `asyncio.run(coro)`
- If running loop detected (LangGraph, Jupyter): dispatches to `ThreadPoolExecutor` with its own `asyncio.run()` — prevents `RuntimeError: cannot be called from a running event loop`

**`_astream()`** event mapping:
- `TextDeltaEvent` → `ChatGenerationChunk(AIMessageChunk(content=delta))`
- `ThinkingDeltaEvent` → chunk with `additional_kwargs={"thinking_delta": delta}`
- `ThinkingEndEvent` → accumulated into `thinking_parts` list
- `ToolCallStartEvent` → chunk with `tool_call_chunks[{name, args:"", id, index}]`
- `ToolCallDeltaEvent` → chunk with `tool_call_chunks[{args:delta, index}]`
- `DoneEvent` → final chunk with `generation_info={"finish_reason": reason}` + `additional_kwargs={"thinking": "\n\n".join(thinking_parts)}` if any thinking accumulated
- `ErrorEvent` → raises `RuntimeError`

**`bind_tools()`** uses `langchain_core.utils.function_calling.convert_to_openai_tool` to accept `BaseTool`, Pydantic models, plain functions, or dicts.

---

## cli.py

Built with [Click](https://click.palletsprojects.com/).

**Thread safety note**: `_quiet_threading_excepthook` is installed at import time to suppress `KeyboardInterrupt` tracebacks from the OAuth callback server's daemon thread during interpreter shutdown. This is cosmetic — the thread is a daemon and exits cleanly, but Python prints a noisy traceback without this hook.

Commands:
- `piai login [PROVIDER]` — OAuth login flow, saves to `auth.json`
- `piai logout [PROVIDER]` — Deletes provider entry from `auth.json`
- `piai list` — Lists registered OAuth providers
- `piai status` — Shows login status and token expiry for all providers
- `piai run PROMPT [--model] [--system] [--provider]` — One-shot completion, streams to stdout
