"""
MCPServer — configuration for a single MCP server.

Supports three transports:
  - stdio: spawns a local subprocess (e.g. "r2pm -r r2mcp")
  - http:  connects to a Streamable HTTP MCP server
  - sse:   connects to a legacy SSE MCP server
"""

from __future__ import annotations

import os
import shlex
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class MCPServer:
    """
    Configuration for one MCP server.

    Use the class methods for convenience:
        MCPServer.stdio("r2pm -r r2mcp")
        MCPServer.http("http://localhost:9000/mcp")
        MCPServer.sse("http://localhost:9000/sse")

    Env handling for stdio:
        env=None            → inherit parent process env (default)
        env={"KEY": "val"}  → replace env entirely (use env_extra to extend instead)
        env_extra={"KEY"}   → extend parent env with these extra vars (recommended)
    """

    transport: Literal["stdio", "http", "sse"]

    # stdio fields
    command: str | None = None
    args: list[str] = field(default_factory=list)
    # None = inherit parent env. Set to extend or replace.
    env: dict[str, str] | None = None

    # http / sse fields
    url: str | None = None
    headers: dict[str, str] = field(default_factory=dict)

    # optional human-readable name used for namespacing on tool collisions
    # if not set, derived from command or url
    name: str | None = None

    # ------------------------------------------------------------------ #
    # Factory helpers                                                      #
    # ------------------------------------------------------------------ #

    @classmethod
    def stdio(
        cls,
        command: str,
        *,
        name: str | None = None,
        env: dict[str, str] | None = None,
        env_extra: dict[str, str] | None = None,
    ) -> "MCPServer":
        """
        Spawn a local subprocess as an MCP server.

        Args:
            command:   Full command string, shell-quoted.
                       e.g. "r2pm -r r2mcp"
                       e.g. "npx @modelcontextprotocol/server-filesystem /tmp"
                       e.g. "/path with spaces/server --flag value"
            name:      Optional namespace name for tool collision resolution.
                       Defaults to the command basename.
            env:       Full replacement env dict. Overrides parent env entirely.
                       Use env_extra to extend instead.
            env_extra: Extra env vars merged on top of parent env (os.environ).
                       Useful for adding API keys without losing PATH.

        Examples:
            MCPServer.stdio("r2pm -r r2mcp")
            MCPServer.stdio("ida-mcp", name="ida")
            MCPServer.stdio("my-server", env_extra={"API_KEY": "secret"})
        """
        parts = shlex.split(command)
        if not parts:
            raise ValueError(f"Empty command: {command!r}")

        # Build effective env
        effective_env: dict[str, str] | None = env
        if env_extra:
            base = dict(os.environ) if env is None else dict(env)
            base.update(env_extra)
            effective_env = base

        return cls(
            transport="stdio",
            command=parts[0],
            args=parts[1:],
            env=effective_env,
            name=name or _basename(parts[0]),
        )

    @classmethod
    def http(
        cls,
        url: str,
        *,
        name: str | None = None,
        headers: dict[str, str] | None = None,
        bearer_token: str | None = None,
    ) -> "MCPServer":
        """
        Connect to a Streamable HTTP MCP server (modern transport, recommended).

        Args:
            url:          Server URL, e.g. "http://localhost:9000/mcp"
            name:         Optional namespace name. Defaults to hostname.
            headers:      Optional HTTP headers dict.
            bearer_token: Shorthand for Authorization: Bearer <token> header.

        Examples:
            MCPServer.http("http://127.0.0.1:13337/mcp")
            MCPServer.http("https://api.example.com/mcp", bearer_token="my-token")
            MCPServer.http("https://api.example.com/mcp", headers={"X-Api-Key": "abc"})
        """
        from urllib.parse import urlparse
        hostname = urlparse(url).hostname or url

        hdrs: dict[str, str] = dict(headers or {})
        if bearer_token:
            hdrs["Authorization"] = f"Bearer {bearer_token}"

        return cls(
            transport="http",
            url=url,
            headers=hdrs,
            name=name or hostname,
        )

    @classmethod
    def sse(
        cls,
        url: str,
        *,
        name: str | None = None,
        headers: dict[str, str] | None = None,
        bearer_token: str | None = None,
    ) -> "MCPServer":
        """
        Connect to a legacy SSE MCP server.

        Args:
            url:          SSE endpoint URL, e.g. "http://localhost:9000/sse"
            name:         Optional namespace name. Defaults to hostname.
            headers:      Optional HTTP headers dict.
            bearer_token: Shorthand for Authorization: Bearer <token> header.
        """
        from urllib.parse import urlparse
        hostname = urlparse(url).hostname or url

        hdrs: dict[str, str] = dict(headers or {})
        if bearer_token:
            hdrs["Authorization"] = f"Bearer {bearer_token}"

        return cls(
            transport="sse",
            url=url,
            headers=hdrs,
            name=name or hostname,
        )

    @classmethod
    def from_config(cls, config: dict) -> "MCPServer":
        """
        Build an MCPServer from a dict config (e.g. from a TOML/JSON config file).

        Supports the same format as Claude/Codex config.toml:

            # stdio
            {"command": "ida-mcp"}
            {"command": "/path/to/server", "args": ["--flag"], "env": {"KEY": "val"}}

            # http / sse
            {"url": "http://127.0.0.1:13337/mcp"}
            {"url": "http://server.example.com/sse", "headers": {"Authorization": "Bearer token"}}

        Optional extra fields:
            "name": str   — override server name
            "transport": "stdio" | "http" | "sse"  — override auto-detection

        Examples:
            MCPServer.from_config({"command": "r2pm", "args": ["-r", "r2mcp"]})
            MCPServer.from_config({"url": "http://127.0.0.1:13337/mcp", "name": "ida"})
        """
        name = config.get("name")

        # Auto-detect transport from config keys
        transport = config.get("transport")
        if transport is None:
            if "command" in config:
                transport = "stdio"
            elif "url" in config:
                url = config["url"]
                transport = "sse" if str(url).endswith("/sse") else "http"
            else:
                raise ValueError(
                    f"Cannot determine transport from config: {config}. "
                    "Provide 'command' for stdio or 'url' for http/sse."
                )

        if transport == "stdio":
            command = config.get("command", "")
            args = config.get("args", [])
            env = config.get("env") or None
            env_extra = config.get("env_extra") or None
            if not command:
                raise ValueError("stdio server config must have 'command'")

            effective_env: dict[str, str] | None = env
            if env_extra:
                base = dict(os.environ) if env is None else dict(env)
                base.update(env_extra)
                effective_env = base

            return cls(
                transport="stdio",
                command=command,
                args=list(args),
                env=effective_env,
                name=name or _basename(command),
            )

        elif transport in ("http", "sse"):
            url = config.get("url", "")
            if not url:
                raise ValueError(f"{transport} server config must have 'url'")

            from urllib.parse import urlparse
            hostname = urlparse(url).hostname or url

            hdrs: dict[str, str] = dict(config.get("headers") or {})
            if config.get("bearer_token"):
                hdrs["Authorization"] = f"Bearer {config['bearer_token']}"

            return cls(
                transport=transport,
                url=url,
                headers=hdrs,
                name=name or hostname,
            )

        else:
            raise ValueError(f"Unknown transport: {transport!r}. Must be 'stdio', 'http', or 'sse'.")

    @classmethod
    def from_toml(cls, path: str, section: str = "mcp_servers") -> "list[MCPServer]":
        """
        Load MCP servers from a piai TOML config file.

        The TOML format uses a [mcp_servers] table where each key is the
        server name and the value is the server config:

            [mcp_servers.r2]
            command = "r2pm"
            args = ["-r", "r2mcp"]

            [mcp_servers.ida]
            command = "ida-mcp"

            [mcp_servers.my-http-server]
            url = "http://127.0.0.1:13337/mcp"

            [mcp_servers.remote]
            url = "https://api.example.com/mcp"
            bearer_token = "my-token"

            [mcp_servers.with-env]
            command = "my-server"
            env_extra = { API_KEY = "secret" }

        Args:
            path:    Path to the TOML file. e.g. "~/.piai/config.toml"
            section: Top-level key containing server configs. Default: "mcp_servers"

        Returns:
            List of MCPServer instances, one per entry under [section].

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError:        If a server config is invalid.

        Example:
            servers = MCPServer.from_toml("~/.piai/config.toml")
            result = await agent(model_id="gpt-5.1-codex-mini", context=ctx, mcp_servers=servers)
        """
        import tomllib
        from pathlib import Path

        resolved = Path(path).expanduser()
        if not resolved.exists():
            raise FileNotFoundError(f"piai config not found: {resolved}")

        with open(resolved, "rb") as f:
            data = tomllib.load(f)

        servers_cfg = data.get(section, {})
        if not servers_cfg:
            return []

        servers = []
        for name, cfg in servers_cfg.items():
            if not isinstance(cfg, dict):
                continue
            # Inject name from the TOML key if not explicitly set
            cfg_with_name = {"name": name, **cfg}
            servers.append(cls.from_config(cfg_with_name))
        return servers

    def __repr__(self) -> str:
        if self.transport == "stdio":
            cmd = " ".join([self.command or ""] + self.args)
            return f"MCPServer.stdio({cmd!r}, name={self.name!r})"
        return f"MCPServer.{self.transport}({self.url!r}, name={self.name!r})"


def _basename(path: str) -> str:
    """Extract a clean basename for use as a server name."""
    return path.split("/")[-1].split("\\")[-1]
