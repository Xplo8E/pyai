"""
piai CLI — login and provider management.

Usage:
    piai login [PROVIDER]     # OAuth login (default: openai-codex)
    piai logout [PROVIDER]    # Remove saved credentials
    piai list                 # List available OAuth providers
    piai status               # Show login status for all providers
    piai version              # Show installed version

Mirrors src/cli.ts login flow.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time
import webbrowser

import click

# Suppress the noisy threading traceback that Python prints when a daemon
# thread is still alive during interpreter shutdown (e.g. after Ctrl+C).
_original_excepthook = threading.excepthook

def _quiet_threading_excepthook(args):
    if args.exc_type is KeyboardInterrupt:
        return
    _original_excepthook(args)

threading.excepthook = _quiet_threading_excepthook

import json

from . import __version__
from .oauth import get_oauth_provider, get_oauth_providers
from .oauth.storage import delete_credentials, get_provider_credentials, save_credentials
from .oauth.types import OAuthAuthInfo, OAuthLoginCallbacks, OAuthPrompt
from .stream import stream
from .types import Context, DoneEvent, ErrorEvent, TextDeltaEvent, UserMessage
from .usage import get_provider_usage, render


@click.group()
def cli():
    """piai — Python port of pi-ai. ChatGPT Plus OAuth + LLM streaming."""


@cli.command()
def version():
    """Show the installed piai version."""
    click.echo(__version__)


# ------------------------------------------------------------------ #
# login                                                               #
# ------------------------------------------------------------------ #


@cli.command()
@click.argument("provider", default="openai-codex")
def login(provider: str):
    """
    Login with OAuth for a provider.

    PROVIDER defaults to "openai-codex" (ChatGPT Plus/Pro).
    """
    p = get_oauth_provider(provider)
    if p is None:
        available = [x.id for x in get_oauth_providers()]
        click.echo(f"Unknown provider: {provider}", err=True)
        click.echo(f"Available: {', '.join(available)}", err=True)
        sys.exit(1)

    click.echo(f"Logging in with {p.name}...")

    try:
        creds = asyncio.run(_do_login(p))
    except KeyboardInterrupt:
        click.echo("\nCancelled.")
        sys.exit(1)
    except Exception as e:
        click.echo(f"Login failed: {e}", err=True)
        sys.exit(1)

    save_credentials(provider, creds)
    click.echo(f"Logged in successfully. Credentials saved to auth.json")


async def _do_login(provider):
    def on_auth(info: OAuthAuthInfo):
        click.echo(f"\n{info.instructions or 'Opening browser...'}")
        click.echo(f"URL: {info.url}\n")
        webbrowser.open(info.url)

    async def on_prompt(prompt: OAuthPrompt) -> str:
        return click.prompt(prompt.message, default=prompt.placeholder or "")

    def on_progress(message: str):
        click.echo(f"  {message}")

    callbacks = OAuthLoginCallbacks(
        on_auth=on_auth,
        on_prompt=on_prompt,
        on_progress=on_progress,
    )

    return await provider.login(callbacks)


# ------------------------------------------------------------------ #
# logout                                                              #
# ------------------------------------------------------------------ #


@cli.command()
@click.argument("provider", default="openai-codex")
def logout(provider: str):
    """Remove saved credentials for a provider."""
    creds = get_provider_credentials(provider)
    if creds is None:
        click.echo(f"Not logged in for provider: {provider}")
        return
    delete_credentials(provider)
    click.echo(f"Logged out from {provider}.")


# ------------------------------------------------------------------ #
# list                                                                #
# ------------------------------------------------------------------ #


@cli.command("list")
def list_providers():
    """List available OAuth providers."""
    providers = get_oauth_providers()
    click.echo("Available OAuth providers:")
    for p in providers:
        click.echo(f"  {p.id:30s} {p.name}")


# ------------------------------------------------------------------ #
# status                                                              #
# ------------------------------------------------------------------ #


@cli.command()
def status():
    """Show login status for all providers."""
    providers = get_oauth_providers()
    click.echo("Login status:")
    for p in providers:
        creds = get_provider_credentials(p.id)
        if creds is None:
            click.echo(f"  {p.id:30s} not logged in")
        else:
            now_ms = int(time.time() * 1000)
            remaining_ms = creds.expires - now_ms
            if remaining_ms <= 0:
                status_str = "expired"
            else:
                remaining_min = remaining_ms // 60000
                status_str = f"valid (~{remaining_min}m remaining)"
            click.echo(f"  {p.id:30s} {status_str}")


# ------------------------------------------------------------------ #
# usage                                                               #
# ------------------------------------------------------------------ #


@cli.command()
@click.argument("provider", default="openai-codex")
@click.option("--raw", is_flag=True, help="Dump raw API responses as JSON")
def usage(provider: str, raw: bool):
    """Show usage limits and quota reset times for a provider."""
    asyncio.run(_do_usage(provider, raw))


async def _do_usage(provider_id: str, raw: bool):
    creds = get_provider_credentials(provider_id)
    if creds is None:
        click.echo(f"Not logged in for provider: {provider_id}. Run: piai login {provider_id}", err=True)
        sys.exit(1)

    # Auto-refresh if expired
    if creds.is_expired():
        p = get_oauth_provider(provider_id)
        if p is not None:
            try:
                creds = await p.refresh_token(creds)
                save_credentials(provider_id, creds)
            except Exception as e:
                click.echo(f"Token refresh failed: {e}", err=True)
                sys.exit(1)

    report = await get_provider_usage(provider_id, creds)

    if raw:
        click.echo(json.dumps(report.raw, indent=2))
        return

    render(report)


# ------------------------------------------------------------------ #
# run (quick test)                                                    #
# ------------------------------------------------------------------ #


@cli.command()
@click.argument("prompt")
@click.option("--model", "-m", default="gpt-5.1-codex-mini", help="Model ID to use")
@click.option("--system", "-s", default=None, help="System prompt")
@click.option("--provider", default="openai-codex", help="Provider ID")
def run(prompt: str, model: str, system: str | None, provider: str):
    """
    Quick one-shot completion from the command line.

    Example:
        piai run "What is 2+2?"
        piai run "Explain async/await in Python" --model gpt-5.1-codex-mini
    """
    asyncio.run(_do_run(prompt, model, system, provider))


async def _do_run(prompt: str, model: str, system: str | None, provider: str):
    context = Context(
        messages=[UserMessage(content=prompt)],
        system_prompt=system,
    )

    try:
        async for event in stream(model, context, provider_id=provider):
            if isinstance(event, TextDeltaEvent):
                click.echo(event.text, nl=False)
            elif isinstance(event, DoneEvent):
                click.echo()  # final newline
                usage = event.message.usage
                click.echo(
                    f"\n[tokens: in={usage['input']} out={usage['output']} "
                    f"stop={event.message.stop_reason}]",
                    err=True,
                )
            elif isinstance(event, ErrorEvent):
                click.echo(f"\nError: {event.error.error_message}", err=True)
                sys.exit(1)
    except RuntimeError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
