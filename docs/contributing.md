# Contributing to piai

## Setup

```bash
git clone https://github.com/Xplo8E/piai
cd piai
uv sync
```

## Running tests

```bash
# Use the local venv's python directly (avoids workspace venv conflicts)
.venv/bin/python3 -m pytest tests/ -v
```

## Project structure

```
piai/
├── src/piai/         # Library source
├── tests/            # Pytest test suite
├── docs/             # Documentation (you're here)
│   ├── architecture.md   # High-level design and flow
│   ├── internals.md      # Per-module deep-dive
│   ├── contributing.md   # This file
│   └── AGENTS.md          # AI agent context (auto-updated on changes)
├── pyproject.toml    # Package config, deps, scripts
└── README.md         # User-facing docs
```

## Making changes

Before touching any module, read the corresponding section in `docs/internals.md`. Each module has specific invariants (especially around auth.json format, JWT decoding, and the stream processor state machine) that must be preserved.

Key rules:
- **auth.json keys must stay camelCase** (`accountId`, not `account_id`) — JS SDK compatibility
- **`expires` is Unix milliseconds**, not seconds
- **Stream processor order matters** — SSE events must be handled in the exact sequence the Responses API sends them. Check `internals.md` before modifying `_StreamProcessor`.
- **No retry on usage limit errors** — `"usage limit"` in the error message means the plan is exhausted; retrying just wastes time.

## Adding tests

Tests live in `tests/`. Three files currently:
- `test_pkce.py` — PKCE verifier/challenge generation
- `test_oauth_codex.py` — JWT decoding, auth URL, credential serialization
- `test_message_transform.py` — Context → Responses API format conversion

New tests should follow the same pattern (plain pytest, no mocking of the HTTP layer).

## Dependency policy

Keep dependencies minimal. Currently:
- `httpx` — async HTTP + SSE streaming
- `click` — CLI
- `websockets` — future use

Don't add dependencies without a strong reason.

## After making changes

Update `docs/AGENTS.md` with a changelog entry describing what changed and why. This keeps the AI agent context current for future sessions.
