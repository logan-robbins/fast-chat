"""MCP registry with capability mapping and tenant/workspace/role policy checks."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse


@dataclass(frozen=True)
class MCPServer:
    server_id: str
    url: str
    capabilities: tuple[str, ...] = field(default_factory=tuple)
    headers: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    allowed_tenants: tuple[str, ...] = field(default_factory=tuple)
    allowed_roles: tuple[str, ...] = field(default_factory=tuple)
    allowed_workspaces: tuple[str, ...] = field(default_factory=tuple)


def _field_error(index: int, field_name: str, message: str) -> RuntimeError:
    return RuntimeError(f"MCP_SERVERS_JSON[{index}].{field_name} {message}")


def _parse_string_field(record: dict[str, Any], *, index: int, field_name: str) -> str:
    raw_value = record.get(field_name)
    if not isinstance(raw_value, str) or not raw_value.strip():
        raise _field_error(index, field_name, "must be a non-empty string")
    return raw_value.strip()


def _parse_string_list_field(
    record: dict[str, Any],
    *,
    index: int,
    field_name: str,
    required: bool = False,
) -> tuple[str, ...]:
    raw_value = record.get(field_name)
    if raw_value is None:
        if required:
            raise _field_error(index, field_name, "is required")
        return ()

    if not isinstance(raw_value, list):
        raise _field_error(index, field_name, "must be a JSON array of strings")

    values: list[str] = []
    for item_index, item in enumerate(raw_value):
        if not isinstance(item, str) or not item.strip():
            raise _field_error(index, field_name, f"contains invalid value at index {item_index}")
        values.append(item.strip())

    if required and not values:
        raise _field_error(index, field_name, "must contain at least one value")

    deduped = tuple(dict.fromkeys(values))
    if len(deduped) != len(values):
        raise _field_error(index, field_name, "contains duplicate values")

    return deduped


def _parse_headers_field(record: dict[str, Any], *, index: int) -> tuple[tuple[str, str], ...]:
    raw_headers = record.get("headers")
    if raw_headers is None:
        return ()
    if not isinstance(raw_headers, dict):
        raise _field_error(index, "headers", "must be a JSON object")

    normalized: list[tuple[str, str]] = []
    for key, value in raw_headers.items():
        if not isinstance(key, str) or not key.strip():
            raise _field_error(index, "headers", "contains an empty header key")
        if not isinstance(value, str) or not value.strip():
            raise _field_error(index, "headers", f"for key {key!r} must be a non-empty string")
        normalized.append((key.strip(), value.strip()))

    return tuple(sorted(normalized))


def _validate_server_url(url: str, *, index: int) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise _field_error(index, "url", "must be an absolute http(s) URL")
    return url


def load_mcp_servers() -> dict[str, MCPServer]:
    """Load and validate the MCP server registry from file or environment.

    Attempts to read `MCP_CONFIG_PATH` (default `mcp_servers.json`) and
    supports both the wrapper schema (`{"mcpServers": {...}}`) and the
    direct array schema. Falls back to `MCP_SERVERS_JSON` if no file exists.
    Each server entry is normalized into an `MCPServer` dataclass that
    strictly validates required fields.

    Returns:
        dict[str, MCPServer]: Mapping from server_id to parsed configuration.

    Raises:
        RuntimeError: For invalid JSON, missing required fields, duplicate IDs,
            or invalid URLs.
    """
    # Check for file-based config first (2026 standard)
    config_path = os.getenv("MCP_CONFIG_PATH", "mcp_servers.json")
    raw = "[]"
    
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                file_content = f.read().strip()
                if file_content:
                    # Check if it's the wrapper format {"mcpServers": ...} or direct array
                    loaded = json.loads(file_content)
                    if isinstance(loaded, dict) and "mcpServers" in loaded:
                        # Transform wrapper format to expected array format
                        # Wrapper: {"mcpServers": {"brightdata": {"url": ...}}}
                        # Target: [{"server_id": "brightdata", "url": ..., "capabilities": ["web_search"]}]
                        normalized = []
                        for sid, sconfig in loaded["mcpServers"].items():
                            # Auto-detect capabilities based on server ID for now if not present
                            caps = sconfig.get("capabilities", [])
                            if not caps:
                                if "brightdata" in sid.lower():
                                    caps = ["web_search"]
                                else:
                                    caps = ["tool_use"] # Default
                            
                            normalized.append({
                                "server_id": sid,
                                "url": sconfig["url"],
                                "capabilities": caps,
                                "headers": sconfig.get("headers", {})
                            })
                        raw = json.dumps(normalized)
                    else:
                        raw = file_content
        except Exception as e:
            # Fallback to env var if file fails, but log warning
            print(f"Warning: Failed to load {config_path}: {e}")
            pass
    
    # Fallback to env var
    if raw == "[]":
        raw = os.getenv("MCP_SERVERS_JSON", "[]").strip() or "[]"

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"MCP configuration must be valid JSON: {exc.msg}") from exc

    if not isinstance(parsed, list):
        raise RuntimeError("MCP configuration must be a JSON array of MCP server objects")

    servers: dict[str, MCPServer] = {}
    for index, item in enumerate(parsed):
        if not isinstance(item, dict):
            raise RuntimeError(f"MCP_SERVERS_JSON[{index}] must be a JSON object")

        server_id = _parse_string_field(item, index=index, field_name="server_id")
        if server_id in servers:
            raise RuntimeError(f"MCP_SERVERS_JSON contains duplicate server_id {server_id!r}")

        url = _validate_server_url(
            _parse_string_field(item, index=index, field_name="url"),
            index=index,
        )

        server = MCPServer(
            server_id=server_id,
            url=url,
            capabilities=_parse_string_list_field(item, index=index, field_name="capabilities", required=True),
            headers=_parse_headers_field(item, index=index),
            allowed_tenants=_parse_string_list_field(item, index=index, field_name="allowed_tenants"),
            allowed_roles=_parse_string_list_field(item, index=index, field_name="allowed_roles"),
            allowed_workspaces=_parse_string_list_field(item, index=index, field_name="allowed_workspaces"),
        )
        servers[server.server_id] = server
    return servers


def list_capabilities_by_server() -> dict[str, tuple[str, ...]]:
    """Return the declared capability tuples for each registered MCP server."""
    return {sid: s.capabilities for sid, s in load_mcp_servers().items()}


def is_mcp_access_allowed(server: MCPServer, tenant_id: str | None, role: str | None, workspace: str | None) -> bool:
    """Evaluate tenant/role/workspace eligibility for a specific MCP server.

    Args:
        server: Parsed MCPServer configuration.
        tenant_id: Tenant ID making the request, or None.
        role: Role of the requester, or None.
        workspace: Workspace identifier, or None.

    Returns:
        bool: True when the provided identifiers are allowed or unrestricted.
    """
    if server.allowed_tenants and tenant_id not in server.allowed_tenants:
        return False
    if server.allowed_roles and role not in server.allowed_roles:
        return False
    if server.allowed_workspaces and workspace not in server.allowed_workspaces:
        return False
    return True
