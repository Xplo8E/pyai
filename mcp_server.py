"""
piai MCP Server

Exposes piai's complete_text() as an MCP tool so any MCP-compatible client
(Claude Code, Cursor, Windsurf, etc.) can query LLMs through your ChatGPT
Plus/Pro subscription — no API key, no per-token billing.

Usage:
    python mcp_server.py

Register in Claude Code (~/.claude/claude_desktop_config.json):
    {
      "mcpServers": {
        "piai": {
          "command": "/path/to/.venv/bin/python",
          "args": ["/path/to/piai/mcp_server.py"]
        }
      }
    }

Register in ~/.piai/config.toml (for use with piai's own agent()):
    [mcp_servers.piai]
    command = "/path/to/.venv/bin/python"
    args = ["/path/to/piai/mcp_server.py"]
"""

from mcp.server.fastmcp import FastMCP

from piai.stream import complete_text
from piai.types import Context, UserMessage

DEFAULT_MODEL = "gpt-5.1-codex-mini"

mcp = FastMCP("piai")


@mcp.tool()
async def complete(
    query: str,
    model: str = DEFAULT_MODEL,
) -> str:
    """
    Send a query to a ChatGPT model and return the response.

    Args:
        query: The prompt or question to send to the model.
        model: Model ID to use (default: gpt-5.1-codex-mini).
               Other options: gpt-5.1, gpt-5.1-codex-max, gpt-5.2,
               gpt-5.2-codex, gpt-5.3-codex, gpt-5.3-codex-spark, gpt-5.4
    """
    context = Context(messages=[UserMessage(content=query)])
    return await complete_text(model_id=model, context=context)


if __name__ == "__main__":
    mcp.run()
