import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "chat-app" / "src"))


def test_mcp_registry_policy(monkeypatch):
    """Confirm MCP registry honors tenants/roles/workspaces in config."""
    monkeypatch.setenv(
        "MCP_SERVERS_JSON",
        '[{"server_id":"m1","url":"http://mcp","capabilities":["retrieval"],"headers":{"x-api-key":"k"},"allowed_tenants":["t1"],"allowed_roles":["analyst"],"allowed_workspaces":["ws1"]}]',
    )
    from chat_app.services import mcp_registry

    importlib.reload(mcp_registry)
    servers = mcp_registry.load_mcp_servers()
    s = servers["m1"]
    assert mcp_registry.list_capabilities_by_server()["m1"] == ("retrieval",)
    assert s.headers == (("x-api-key", "k"),)
    assert mcp_registry.is_mcp_access_allowed(s, "t1", "analyst", "ws1")
    assert not mcp_registry.is_mcp_access_allowed(s, "t2", "analyst", "ws1")


def test_a2a_registry_trust(monkeypatch):
    """Ensure route_agent_for_capability returns a trusted remote agent."""
    monkeypatch.setenv("A2A_TRUSTED_ISSUERS", "issuer-a")
    monkeypatch.setenv(
        "A2A_AGENTS_JSON",
        '[{"agent_id":"ra","endpoint":"http://agent","capabilities":["websearch"],"issuer":"issuer-a","manifest_signature":"sig"}]',
    )
    from chat_app.services import a2a_registry

    importlib.reload(a2a_registry)
    agent = a2a_registry.route_agent_for_capability("websearch")
    assert agent is not None
    assert agent.agent_id == "ra"


def test_graph_registry_pins(monkeypatch):
    """Validate tenant pins override the default graph version."""
    monkeypatch.setenv("GRAPH_DEFAULT_VERSION", "v2")
    monkeypatch.setenv("GRAPH_AVAILABLE_VERSIONS", "v1,v2")
    monkeypatch.setenv("GRAPH_TENANT_PINS_JSON", '{"tenant-1":"v1"}')
    from chat_app.services import graph_registry

    importlib.reload(graph_registry)
    assert graph_registry.resolve_graph_version("tenant-1") == "v1"
    assert graph_registry.resolve_graph_version("tenant-x") == "v2"


def test_mcp_registry_rejects_non_array_json(monkeypatch):
    """Reject MCP config when the payload is not a JSON array."""
    monkeypatch.setenv("MCP_SERVERS_JSON", '{"server_id":"m1"}')
    from chat_app.services import mcp_registry

    importlib.reload(mcp_registry)
    with pytest.raises(RuntimeError, match="MCP_SERVERS_JSON must be a JSON array"):
        mcp_registry.load_mcp_servers()


def test_mcp_registry_rejects_invalid_url(monkeypatch):
    """Reject MCP entries with non-HTTP(S) URLs."""
    monkeypatch.setenv(
        "MCP_SERVERS_JSON",
        '[{"server_id":"m1","url":"not-a-url","capabilities":["web_search"]}]',
    )
    from chat_app.services import mcp_registry

    importlib.reload(mcp_registry)
    with pytest.raises(RuntimeError, match="url must be an absolute http\\(s\\) URL"):
        mcp_registry.load_mcp_servers()


def test_mcp_registry_rejects_duplicate_server_ids(monkeypatch):
    """Reject MCP configs that include duplicate server IDs."""
    monkeypatch.setenv(
        "MCP_SERVERS_JSON",
        (
            '[{"server_id":"m1","url":"http://mcp-a","capabilities":["retrieval"]},'
            '{"server_id":"m1","url":"http://mcp-b","capabilities":["web_search"]}]'
        ),
    )
    from chat_app.services import mcp_registry

    importlib.reload(mcp_registry)
    with pytest.raises(RuntimeError, match="duplicate server_id"):
        mcp_registry.load_mcp_servers()


def test_mcp_registry_requires_capabilities(monkeypatch):
    """Require each MCP server to advertise at least one capability."""
    monkeypatch.setenv(
        "MCP_SERVERS_JSON",
        '[{"server_id":"m1","url":"http://mcp","capabilities":[]}]',
    )
    from chat_app.services import mcp_registry

    importlib.reload(mcp_registry)
    with pytest.raises(RuntimeError, match="capabilities must contain at least one value"):
        mcp_registry.load_mcp_servers()
