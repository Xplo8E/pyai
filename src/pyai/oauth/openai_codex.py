"""
OpenAI Codex (ChatGPT Plus/Pro) OAuth flow.

Mirrors src/utils/oauth/openai-codex.ts exactly.

Flow:
    1. Generate PKCE verifier + challenge
    2. Build authorization URL
    3. Start local HTTP server on :1455 to catch the redirect
    4. Open browser → user logs in with ChatGPT account
    5. Browser redirects to http://localhost:1455/auth/callback?code=...&state=...
    6. Exchange code for access_token + refresh_token
    7. Extract accountId from JWT payload
    8. Return OAuthCredentials

Token refresh:
    POST https://auth.openai.com/oauth/token
    grant_type=refresh_token
    → new access_token + refresh_token + accountId
"""

from __future__ import annotations

import asyncio
import base64
import json
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlencode, urlparse

import httpx

from .pkce import generate_pkce
from .types import OAuthAuthInfo, OAuthCredentials, OAuthLoginCallbacks, OAuthPrompt, OAuthProviderInterface

# ------------------------------------------------------------------ #
# Constants — must match JS exactly                                   #
# ------------------------------------------------------------------ #

CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
AUTHORIZE_URL = "https://auth.openai.com/oauth/authorize"
TOKEN_URL = "https://auth.openai.com/oauth/token"
REDIRECT_URI = "http://localhost:1455/auth/callback"
SCOPE = "openid profile email offline_access"
JWT_CLAIM_PATH = "https://api.openai.com/auth"
CALLBACK_PORT = 1455
CALLBACK_TIMEOUT_S = 60  # wait up to 60s for browser redirect

SUCCESS_HTML = """\
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Authentication successful</title>
</head>
<body>
  <p>Authentication successful. Return to your terminal to continue.</p>
</body>
</html>"""


# ------------------------------------------------------------------ #
# PKCE + state                                                        #
# ------------------------------------------------------------------ #


def _create_state() -> str:
    """16 random bytes → hex string. JS: randomBytes(16).toString("hex")"""
    return secrets.token_hex(16)


# ------------------------------------------------------------------ #
# JWT decoding                                                        #
# ------------------------------------------------------------------ #


def _decode_jwt_payload(token: str) -> dict | None:
    """
    Decode the payload (middle) section of a JWT without verification.

    JS equivalent:
        const parts = token.split(".");
        const decoded = atob(parts[1]);
        return JSON.parse(decoded);

    Python must manually re-add base64 padding that JWT strips.
    """
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        payload_b64 = parts[1]
        # base64url → standard base64 with padding
        padding = "=" * (4 - len(payload_b64) % 4)
        decoded = base64.urlsafe_b64decode(payload_b64 + padding)
        return json.loads(decoded)
    except Exception:
        return None


def extract_account_id(access_token: str) -> str:
    """
    Extract chatgpt_account_id from the JWT access token.

    JS equivalent:
        const payload = decodeJwt(token);
        return payload?.[JWT_CLAIM_PATH]?.chatgpt_account_id;
    """
    payload = _decode_jwt_payload(access_token)
    if payload is None:
        raise ValueError("Failed to decode JWT payload")
    auth_claims = payload.get(JWT_CLAIM_PATH, {})
    account_id = auth_claims.get("chatgpt_account_id")
    if not account_id:
        raise ValueError("No chatgpt_account_id in JWT claims")
    return account_id


# ------------------------------------------------------------------ #
# Local OAuth callback server                                         #
# ------------------------------------------------------------------ #


class _CallbackServer:
    """
    Minimal HTTP server that listens on localhost:1455 for the OAuth redirect.

    JS equivalent: http.createServer() listening on 127.0.0.1:1455.
    """

    def __init__(self, expected_state: str):
        self._expected_state = expected_state
        self._code_event = threading.Event()
        self._received_code: str | None = None
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> bool:
        """Start server. Returns False if port is already in use (fallback to manual paste)."""
        parent = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                parsed = urlparse(self.path)
                if parsed.path != "/auth/callback":
                    self._send(404, "Not found")
                    return
                params = parse_qs(parsed.query)
                state = params.get("state", [None])[0]
                if state != parent._expected_state:
                    self._send(400, "State mismatch")
                    return
                code = params.get("code", [None])[0]
                if not code:
                    self._send(400, "Missing authorization code")
                    return
                self._send(200, SUCCESS_HTML, "text/html; charset=utf-8")
                parent._received_code = code
                parent._code_event.set()

            def _send(self, status: int, body: str, content_type: str = "text/plain"):
                encoded = body.encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def log_message(self, *args):
                pass  # silence access logs

        try:
            self._server = HTTPServer(("127.0.0.1", CALLBACK_PORT), Handler)
        except OSError:
            # Port in use — JS falls back to manual paste
            return False

        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return True

    def wait_for_code(self, timeout: float = CALLBACK_TIMEOUT_S) -> str | None:
        """Block until code is received or timeout expires."""
        self._code_event.wait(timeout=timeout)
        return self._received_code

    def cancel(self):
        """Unblock wait_for_code without a code (used on Ctrl+C)."""
        self._code_event.set()

    def close(self):
        if self._server:
            self._server.shutdown()
            self._server = None


# ------------------------------------------------------------------ #
# Authorization URL                                                   #
# ------------------------------------------------------------------ #


def _build_auth_url(challenge: str, state: str, originator: str = "pi") -> str:
    params = {
        "response_type": "code",
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "scope": SCOPE,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": state,
        "id_token_add_organizations": "true",
        "codex_cli_simplified_flow": "true",
        "originator": originator,
    }
    return f"{AUTHORIZE_URL}?{urlencode(params)}"


# ------------------------------------------------------------------ #
# Token exchange                                                      #
# ------------------------------------------------------------------ #


async def _exchange_code(code: str, verifier: str) -> dict:
    """
    POST /oauth/token with authorization_code grant.

    JS equivalent: exchangeAuthorizationCode()
    """
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            TOKEN_URL,
            data={
                "grant_type": "authorization_code",
                "client_id": CLIENT_ID,
                "code": code,
                "code_verifier": verifier,
                "redirect_uri": REDIRECT_URI,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        if not resp.is_success:
            raise RuntimeError(f"Token exchange failed: {resp.status_code} {resp.text}")
        return resp.json()


async def _refresh_access_token(refresh_token: str) -> dict:
    """
    POST /oauth/token with refresh_token grant.

    JS equivalent: refreshAccessToken()
    """
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            TOKEN_URL,
            data={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": CLIENT_ID,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        if not resp.is_success:
            raise RuntimeError(f"Token refresh failed: {resp.status_code} {resp.text}")
        return resp.json()


# ------------------------------------------------------------------ #
# Input parsing (manual paste fallback)                               #
# ------------------------------------------------------------------ #


def _parse_authorization_input(raw: str) -> tuple[str | None, str | None]:
    """
    Parse code and state from various input formats:
      - Full redirect URL: http://localhost:1455/auth/callback?code=...&state=...
      - Query string: code=...&state=...
      - Bare code: abc123

    JS equivalent: parseAuthorizationInput()
    """
    value = raw.strip()
    if not value:
        return None, None

    # Try as full URL
    try:
        parsed = urlparse(value)
        if parsed.scheme and parsed.netloc:
            params = parse_qs(parsed.query)
            return params.get("code", [None])[0], params.get("state", [None])[0]
    except Exception:
        pass

    # Try as query string
    if "code=" in value:
        params = parse_qs(value)
        return params.get("code", [None])[0], params.get("state", [None])[0]

    # # separator format
    if "#" in value:
        parts = value.split("#", 1)
        return parts[0] or None, parts[1] or None

    # Bare code
    return value, None


# ------------------------------------------------------------------ #
# Credentials builder                                                 #
# ------------------------------------------------------------------ #


def _build_credentials(token_data: dict) -> OAuthCredentials:
    """Convert token response dict to OAuthCredentials."""
    access = token_data.get("access_token", "")
    refresh = token_data.get("refresh_token", "")
    expires_in = token_data.get("expires_in", 3600)

    account_id = extract_account_id(access)

    return OAuthCredentials(
        access=access,
        refresh=refresh,
        # JS: Date.now() + expires_in * 1000
        expires=int(time.time() * 1000) + expires_in * 1000,
        extras={"accountId": account_id},
    )


# ------------------------------------------------------------------ #
# Public login / refresh functions                                    #
# ------------------------------------------------------------------ #


async def login_openai_codex(
    on_auth,
    on_prompt,
    on_progress=None,
    on_manual_code_input=None,
    originator: str = "pi",
) -> OAuthCredentials:
    """
    Full ChatGPT Plus OAuth login flow.

    Mirrors loginOpenAICodex() from openai-codex.ts.
    """
    verifier, challenge = generate_pkce()
    state = _create_state()
    url = _build_auth_url(challenge, state, originator)

    # Start local callback server
    server = _CallbackServer(expected_state=state)
    server_started = server.start()

    on_auth(OAuthAuthInfo(url=url, instructions="A browser window should open. Complete login to continue."))

    code: str | None = None

    try:
        if server_started:
            if on_manual_code_input:
                # Race: browser callback vs manual paste — whichever wins
                loop = asyncio.get_event_loop()
                manual_future: asyncio.Future = loop.create_future()
                browser_future: asyncio.Future = loop.create_future()

                def _wait_browser():
                    result = server.wait_for_code()
                    if not browser_future.done():
                        loop.call_soon_threadsafe(browser_future.set_result, result)

                browser_thread = threading.Thread(target=_wait_browser, daemon=True)
                browser_thread.start()

                async def _get_manual():
                    try:
                        result = await on_manual_code_input()
                        if not manual_future.done():
                            manual_future.set_result(result)
                    except Exception as e:
                        if not manual_future.done():
                            manual_future.set_exception(e)

                asyncio.ensure_future(_get_manual())

                done, _ = await asyncio.wait(
                    [browser_future, manual_future],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                first = done.pop()
                result = first.result()

                if browser_future in done or (not browser_future.done() and result):
                    # Browser won or manual won
                    if isinstance(result, str):
                        parsed_code, parsed_state = _parse_authorization_input(result)
                        if parsed_state and parsed_state != state:
                            raise RuntimeError("State mismatch in manual input")
                        code = parsed_code
                    else:
                        code = result  # browser returned bare code
                else:
                    code = result
            else:
                # Wait for browser callback only
                code = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: server.wait_for_code(CALLBACK_TIMEOUT_S)
                )
        # else: server not started, fall through to manual paste

        # Fallback: ask user to paste
        if not code:
            pasted = await on_prompt(OAuthPrompt(message="Paste the authorization code (or full redirect URL):"))
            parsed_code, parsed_state = _parse_authorization_input(pasted)
            if parsed_state and parsed_state != state:
                raise RuntimeError("State mismatch")
            code = parsed_code

        if not code:
            raise RuntimeError("No authorization code received")

        token_data = await _exchange_code(code, verifier)
        return _build_credentials(token_data)

    finally:
        server.cancel()  # unblock wait_for_code if still waiting
        server.close()


async def refresh_openai_codex_token(credentials: OAuthCredentials) -> OAuthCredentials:
    """
    Refresh an expired ChatGPT Plus token.

    Mirrors refreshOpenAICodexToken() from openai-codex.ts.
    """
    token_data = await _refresh_access_token(credentials.refresh)
    return _build_credentials(token_data)


# ------------------------------------------------------------------ #
# Provider object                                                     #
# ------------------------------------------------------------------ #


class OpenAICodexOAuthProvider(OAuthProviderInterface):
    """
    OAuthProviderInterface implementation for ChatGPT Plus/Pro.

    id = "openai-codex"  (matches JS SDK provider id)
    """

    id = "openai-codex"
    name = "ChatGPT Plus/Pro (Codex Subscription)"
    uses_callback_server = True

    async def login(self, callbacks: OAuthLoginCallbacks) -> OAuthCredentials:
        return await login_openai_codex(
            on_auth=callbacks.on_auth,
            on_prompt=callbacks.on_prompt,
            on_progress=callbacks.on_progress,
            on_manual_code_input=callbacks.on_manual_code_input,
        )

    async def refresh_token(self, credentials: OAuthCredentials) -> OAuthCredentials:
        return await refresh_openai_codex_token(credentials)

    def get_api_key(self, credentials: OAuthCredentials) -> str:
        return credentials.access
