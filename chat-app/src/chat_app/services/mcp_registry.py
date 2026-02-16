"""MCP registry with capability mapping and tenant/workspace/role policy checks."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class MCPServer:
    server_id: str
    url: str
    capabilities: tuple[str, ...] = field(default_factory=tuple)
    allowed_tenants: tuple[str, ...] = field(default_factory=tuple)
    allowed_roles: tuple[str, ...] = field(default_factory=tuple)
    allowed_workspaces: tuple[str, ...] = field(default_factory=tuple)


def _tuple(v):
    if not v:
        return ()
    return tuple(str(x) for x in v)


def load_mcp_servers() -> dict[str, MCPServer]:
    raw = os.getenv("MCP_SERVERS_JSON", "[]")
    parsed = json.loads(raw)
    servers: dict[str, MCPServer] = {}
    for item in parsed:
        server = MCPServer(
            server_id=item["server_id"],
            url=item["url"],
            capabilities=_tuple(item.get("capabilities")),
            allowed_tenants=_tuple(item.get("allowed_tenants")),
            allowed_roles=_tuple(item.get("allowed_roles")),
            allowed_workspaces=_tuple(item.get("allowed_workspaces")),
        )
        servers[server.server_id] = server
    return servers


def list_capabilities_by_server() -> dict[str, tuple[str, ...]]:
    return {sid: s.capabilities for sid, s in load_mcp_servers().items()}


def is_mcp_access_allowed(server: MCPServer, tenant_id: str | None, role: str | None, workspace: str | None) -> bool:
    if server.allowed_tenants and tenant_id not in server.allowed_tenants:
        return False
    if server.allowed_roles and role not in server.allowed_roles:
        return False
    if server.allowed_workspaces and workspace not in server.allowed_workspaces:
        return False
    return True
