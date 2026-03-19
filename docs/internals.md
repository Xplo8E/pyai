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

---

## oauth/openai_codex.py

**Constants**
```python
CLIENT_ID    = "app_EMoamEEZ73f0CkXaXp7hrann"
AUTHORIZE_URL = "https://auth.openai.com/oauth/authorize"
TOKEN_URL    = "https://auth.openai.com/oauth/token"
REDIRECT_URI = "http://localhost:1455/auth/callback"
SCOPE        = "openid profile email offline_access"
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
  - `gpt-5.1-codex-mini`: only supports `"medium"` or `"high"`

**`build_request_body(model_id, context, options)`**
- Always sets `store: False`, `stream: True`, `include: ["reasoning.encrypted_content"]`
- `instructions` defaults to `"You are a helpful assistant."` if no system prompt
- `prompt_cache_key` is set from `options["session_id"]` if provided

---

## providers/openai_codex.py

The core streaming provider. POSTs to `https://chatgpt.com/backend-api/codex/responses` and processes the SSE stream.

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
| `response.content_part.added` | Set `current_block` (only for `output_text` / `refusal`) |
| `response.output_text.delta` | Emit `TextDeltaEvent` (guarded: only if content block active) |
| `response.refusal.delta` | Emit `TextDeltaEvent` (refusals surface as text) |
| `response.function_call_arguments.delta` | Emit `ToolCallDeltaEvent`, accumulate |
| `response.function_call_arguments.done` | Parse JSON args, emit `ToolCallEndEvent` |
| `response.output_item.done` | Finalize text/thinking block, emit `TextEndEvent` |
| `response.completed` | Build final `AssistantMessage`, emit `DoneEvent` |
| `response.failed` | Extract `incomplete_details.reason`, emit `ErrorEvent` |

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

**Tool call ID format**
- The Responses API sends `call_id` and `item_id` separately
- Combined as `f"{call_id}|{item_id}"` to form a unique tool call ID

---

## stream.py

Thin orchestration layer:
1. Load credentials from `auth.json`
2. Auto-refresh if within 5-minute expiry buffer
3. Save updated credentials back (rotation)
4. Call `stream_openai_codex()` and yield events

`complete()` collects all events and returns the final `AssistantMessage` from `DoneEvent`.
`complete_text()` collects only `TextDeltaEvent.text` and returns a plain string.

---

## cli.py

Built with [Click](https://click.palletsprojects.com/).

**Thread safety note**: `_quiet_threading_excepthook` is installed at import time to suppress `KeyboardInterrupt` tracebacks from the OAuth callback server's daemon thread during interpreter shutdown. This is cosmetic — the thread is a daemon and exits cleanly, but Python prints a noisy traceback without this hook.

Commands:
- `piai login [PROVIDER]` — OAuth login flow, saves to `auth.json`
- `piai logout [PROVIDER]` — Deletes provider entry from `auth.json`
- `piai list` — Lists registered OAuth providers
- `piai status` — Shows login status and token expiry for all providers
- `piai run PROMPT` — One-shot completion, streams to stdout
