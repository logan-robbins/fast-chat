import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "chat-app" / "src"))

ROOT_DIR = Path(__file__).resolve().parents[1]
CHAT_APP_DIR = ROOT_DIR / "chat-app"
MCP_HOST = "127.0.0.1"
MCP_PORT = 8765


def _wait_for_port(host: str, port: int, timeout_seconds: float = 20.0) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            if sock.connect_ex((host, port)) == 0:
                return
        time.sleep(0.2)
    raise TimeoutError(f"Timed out waiting for {host}:{port}")


@pytest.fixture(scope="module")
def local_mcp_server():
    """Spin up the local MCP test server fixture for websearch integration."""
    env = os.environ.copy()
    env["MCP_TEST_HOST"] = MCP_HOST
    env["MCP_TEST_PORT"] = str(MCP_PORT)
    env["PYTHONUNBUFFERED"] = "1"

    process = subprocess.Popen(
        ["uv", "run", "../tests/fixtures/mcp_test_server.py"],
        cwd=str(CHAT_APP_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        _wait_for_port(MCP_HOST, MCP_PORT)
    except Exception:
        output = process.stdout.read() if process.stdout else ""
        process.terminate()
        raise RuntimeError(f"Failed to start local MCP test server. Output:\n{output}")

    try:
        yield {
            "url": f"http://{MCP_HOST}:{MCP_PORT}/mcp",
            "process": process,
        }
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


@pytest.mark.asyncio
async def test_websearch_tool_invokes_local_mcp_server(monkeypatch, local_mcp_server):
    monkeypatch.setenv("WEB_SEARCH_PROVIDER", "mcp")
    monkeypatch.setenv("MCP_WEB_SEARCH_TOOL", "search_engine")
    monkeypatch.setenv(
        "MCP_SERVERS_JSON",
        json.dumps(
            [
                {
                    "server_id": "local-mcp",
                    "url": local_mcp_server["url"],
                    "capabilities": ["web_search"],
                }
            ]
        ),
    )

    from chat_app.tools import tools

    result = await tools.perplexity_search.ainvoke({"query": "mcp integration test"})
    assert "MCP_E2E_RESULT::mcp integration test" in result


@pytest.mark.asyncio
async def test_websearch_tool_supports_brightdata_mcp_when_token_is_available(monkeypatch):
    token = os.getenv("BRIGHTDATA_MCP_TOKEN", "").strip()
    if not token:
        pytest.skip("Set BRIGHTDATA_MCP_TOKEN to run Bright Data MCP integration test")

    monkeypatch.setenv("WEB_SEARCH_PROVIDER", "mcp")
    monkeypatch.setenv("MCP_WEB_SEARCH_TOOL", "search_engine")
    monkeypatch.setenv(
        "MCP_SERVERS_JSON",
        json.dumps(
            [
                {
                    "server_id": "brightdata",
                    "url": f"https://mcp.brightdata.com/mcp?token={token}",
                    "capabilities": ["web_search"],
                }
            ]
        ),
    )

    from chat_app.tools import tools

    result = await tools.perplexity_search.ainvoke({"query": "openai responses api docs"})
    assert len(result.strip()) > 0
