"""Administrative control-plane endpoints for registry and policy inspection.

Implements SPEC 7.2 control/registry APIs:
- /v1/admin/model-registry
- /v1/admin/retrieval-providers
- /v1/admin/mcp/servers
- /v1/admin/a2a/agents
- /v1/admin/graph-versions
- /v1/admin/policies

These endpoints are intentionally read-only and designed for operational
visibility. Access can be protected with ADMIN_API_KEY.
"""
from __future__ import annotations

import json
import os
from typing import Any

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from src.services.model_policy import get_model_policy
from src.services.model_registry import get_model_objects

router = APIRouter(tags=["admin"])


def _authorize_admin(x_admin_key: str | None) -> None:
    """Validate admin key if configured.

    If ``ADMIN_API_KEY`` is unset, endpoints are open for local/dev ergonomics.
    """
    expected = os.getenv("ADMIN_API_KEY", "").strip()
    if not expected:
        return
    if x_admin_key != expected:
        raise HTTPException(status_code=403, detail="Forbidden: invalid admin key")


def _parse_json_env(name: str, default: Any) -> Any:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"Invalid JSON in {name}") from exc


def _parse_csv_env(name: str, default: list[str] | None = None) -> list[str]:
    value = os.getenv(name, "")
    parsed = [item.strip() for item in value.split(",") if item.strip()]
    if parsed:
        return parsed
    return default or []


class ModelRegistryResponse(BaseModel):
    models: list[dict[str, Any]]


class RetrievalProvidersResponse(BaseModel):
    active_provider: str
    available_providers: list[str]


class MCPServersResponse(BaseModel):
    servers: list[dict[str, Any]]


class A2AAgentsResponse(BaseModel):
    agents: list[dict[str, Any]]
    trusted_issuers: list[str]
    require_signature: bool


class GraphVersionsResponse(BaseModel):
    default_version: str
    available_versions: list[str]
    tenant_pins: dict[str, str] = Field(default_factory=dict)
    hot_reload_enabled: bool


class AdminPoliciesResponse(BaseModel):
    model_policy: dict[str, Any]
    request_policy: dict[str, Any]


@router.get("/v1/admin/model-registry", response_model=ModelRegistryResponse)
async def admin_model_registry(x_admin_key: str | None = Header(default=None)) -> ModelRegistryResponse:
    _authorize_admin(x_admin_key)
    return ModelRegistryResponse(models=get_model_objects())


@router.get("/v1/admin/retrieval-providers", response_model=RetrievalProvidersResponse)
async def admin_retrieval_providers(x_admin_key: str | None = Header(default=None)) -> RetrievalProvidersResponse:
    _authorize_admin(x_admin_key)
    active = os.getenv("VECTOR_STORE_PROVIDER", "chroma").lower()
    available = ["chroma", "memory", "pgvector", "qdrant"]
    return RetrievalProvidersResponse(active_provider=active, available_providers=available)


@router.get("/v1/admin/mcp/servers", response_model=MCPServersResponse)
async def admin_mcp_servers(x_admin_key: str | None = Header(default=None)) -> MCPServersResponse:
    _authorize_admin(x_admin_key)
    servers = _parse_json_env("MCP_SERVERS_JSON", [])
    if not isinstance(servers, list):
        raise HTTPException(status_code=500, detail="MCP_SERVERS_JSON must be a JSON list")
    return MCPServersResponse(servers=servers)


@router.get("/v1/admin/a2a/agents", response_model=A2AAgentsResponse)
async def admin_a2a_agents(x_admin_key: str | None = Header(default=None)) -> A2AAgentsResponse:
    _authorize_admin(x_admin_key)
    agents = _parse_json_env("A2A_AGENTS_JSON", [])
    if not isinstance(agents, list):
        raise HTTPException(status_code=500, detail="A2A_AGENTS_JSON must be a JSON list")
    trusted_issuers = _parse_csv_env("A2A_TRUSTED_ISSUERS")
    require_signature = os.getenv("A2A_REQUIRE_SIGNATURE", "true").lower() == "true"
    return A2AAgentsResponse(
        agents=agents,
        trusted_issuers=trusted_issuers,
        require_signature=require_signature,
    )


@router.get("/v1/admin/graph-versions", response_model=GraphVersionsResponse)
async def admin_graph_versions(x_admin_key: str | None = Header(default=None)) -> GraphVersionsResponse:
    _authorize_admin(x_admin_key)
    default_version = os.getenv("GRAPH_DEFAULT_VERSION", "v1")
    available_versions = _parse_csv_env("GRAPH_AVAILABLE_VERSIONS", default=[default_version])
    tenant_pins = _parse_json_env("GRAPH_TENANT_PINS_JSON", {})
    if not isinstance(tenant_pins, dict):
        raise HTTPException(status_code=500, detail="GRAPH_TENANT_PINS_JSON must be a JSON object")
    hot_reload_enabled = os.getenv("GRAPH_HOT_RELOAD", "false").lower() == "true"
    return GraphVersionsResponse(
        default_version=default_version,
        available_versions=available_versions,
        tenant_pins={str(k): str(v) for k, v in tenant_pins.items()},
        hot_reload_enabled=hot_reload_enabled,
    )


@router.get("/v1/admin/policies", response_model=AdminPoliciesResponse)
async def admin_policies(x_admin_key: str | None = Header(default=None)) -> AdminPoliciesResponse:
    _authorize_admin(x_admin_key)

    model_policy = get_model_policy()

    request_policy = {
        "allowed_tenants": _parse_csv_env("POLICY_ALLOWED_TENANTS"),
        "model_role_map": _parse_csv_env("MODEL_ROLE_MAP"),
        "model_abac_rules": _parse_csv_env("MODEL_ABAC_RULES"),
    }

    return AdminPoliciesResponse(
        model_policy={
            "allowed_families": list(model_policy.allowed_families),
            "max_context_tokens": model_policy.max_context_tokens,
            "max_output_tokens": model_policy.max_output_tokens,
            "fallback_chain": list(model_policy.fallback_chain),
        },
        request_policy=request_policy,
    )
