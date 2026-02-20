"""Local MCP server used for end-to-end MCP integration tests."""
from __future__ import annotations

import os

from mcp.server.fastmcp import FastMCP


PORT = int(os.getenv("MCP_TEST_PORT", "8765"))
HOST = os.getenv("MCP_TEST_HOST", "127.0.0.1")

mcp = FastMCP(
    name="fast-chat-mcp-test-server",
    host=HOST,
    port=PORT,
    json_response=True,
    stateless_http=True,
)


@mcp.tool()
def search_engine(query: str, engine: str = "google") -> dict:
    """Return deterministic content for MCP transport validation."""
    return {
        "provider": "local-mcp-test",
        "engine": engine,
        "query": query,
        "summary": f"MCP_E2E_RESULT::{query}",
    }


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
