"""MCP client helpers for invoking configured remote MCP tools."""
from __future__ import annotations

import json
import logging
from typing import Any

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from chat_app.services.mcp_registry import MCPServer, is_mcp_access_allowed, load_mcp_servers
from chat_app.services.runtime_config import get_mcp_transport_config, get_web_search_runtime_config

logger = logging.getLogger(__name__)


def _resolve_server(
    *,
    capability: str,
    tenant_id: str | None = None,
    role: str | None = None,
    workspace: str | None = None,
) -> MCPServer:
    for server in load_mcp_servers().values():
        if capability not in server.capabilities:
            continue
        if not is_mcp_access_allowed(server, tenant_id, role, workspace):
            continue
        return server
    raise RuntimeError(
        f"No MCP server configured for capability '{capability}'. "
        "Set MCP_SERVERS_JSON with a matching server entry."
    )


def _extract_tool_text(result: Any) -> str:
    parts: list[str] = []

    structured = getattr(result, "structuredContent", None)
    if structured:
        parts.append(json.dumps(structured))

    for item in getattr(result, "content", []):
        text = getattr(item, "text", None)
        if text:
            parts.append(str(text))
            continue
        if hasattr(item, "model_dump"):
            payload = item.model_dump()
            payload_text = payload.get("text") if isinstance(payload, dict) else None
            if payload_text:
                parts.append(str(payload_text))
            else:
                parts.append(json.dumps(payload))
            continue
        parts.append(str(item))

    content = "\n".join(part for part in parts if part).strip()
    if not content:
        raise RuntimeError("MCP tool returned no text content")
    return content


async def invoke_mcp_tool(
    *,
    capability: str,
    tool_name: str,
    arguments: dict[str, Any],
    tenant_id: str | None = None,
    role: str | None = None,
    workspace: str | None = None,
) -> str:
    """Invoke a tool on the first allowed MCP server with the given capability."""
    server = _resolve_server(
        capability=capability,
        tenant_id=tenant_id,
        role=role,
        workspace=workspace,
    )

    transport_config = get_mcp_transport_config()
    connect_timeout = transport_config.http_timeout_seconds
    read_timeout = transport_config.sse_read_timeout_seconds
    headers = dict(server.headers) if server.headers else None

    logger.info(
        "Invoking MCP tool",
        extra={
            "server_id": server.server_id,
            "url": server.url,
            "tool_name": tool_name,
            "capability": capability,
        },
    )

    async with httpx.AsyncClient(
        headers=headers,
        timeout=httpx.Timeout(
            connect=connect_timeout,
            read=read_timeout,
            write=connect_timeout,
            pool=connect_timeout,
        ),
    ) as http_client:
        async with streamable_http_client(server.url, http_client=http_client) as (
            read_stream,
            write_stream,
            _,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                tools = await session.list_tools()
                available = {tool.name for tool in tools.tools}
                if tool_name not in available:
                    available_tools = ", ".join(sorted(available)) or "<none>"
                    raise RuntimeError(
                        f"MCP tool '{tool_name}' not available on server '{server.server_id}'. "
                        f"Available tools: {available_tools}"
                    )

                result = await session.call_tool(tool_name, arguments=arguments)

    if getattr(result, "isError", False):
        raise RuntimeError(f"MCP tool '{tool_name}' returned an error response")

    return _extract_tool_text(result)


async def search_web_via_mcp(query: str) -> str:
    """Search the web via an MCP server configured with `web_search` capability."""
    if not query or not query.strip():
        raise ValueError("query cannot be empty")

    web_search_config = get_web_search_runtime_config()
    tool_name = web_search_config.mcp_web_search_tool
    engine = web_search_config.mcp_web_search_engine

    if tool_name == "search_engine_batch":
        arguments = {"queries": [{"query": query, "engine": engine}]}
    else:
        arguments = {"query": query, "engine": engine}

    return await invoke_mcp_tool(
        capability="web_search",
        tool_name=tool_name,
        arguments=arguments,
    )
