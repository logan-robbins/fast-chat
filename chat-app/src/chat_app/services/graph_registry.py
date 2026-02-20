"""Graph version pinning and hot-load selection controls."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class GraphRegistryState:
    default_version: str
    available_versions: tuple[str, ...]
    tenant_pins: dict[str, str]


def get_graph_registry_state() -> GraphRegistryState:
    """Construct the graph registry state from environment configuration.

    Returns:
        GraphRegistryState: Contains the default version, the available
            versions list, and any tenant-specific version pins.
    """
    default_version = os.getenv("GRAPH_DEFAULT_VERSION", "v1")
    versions_raw = os.getenv("GRAPH_AVAILABLE_VERSIONS", default_version)
    available_versions = tuple(v.strip() for v in versions_raw.split(",") if v.strip())
    tenant_pins = json.loads(os.getenv("GRAPH_TENANT_PINS_JSON", "{}"))
    return GraphRegistryState(default_version, available_versions, tenant_pins)


def resolve_graph_version(tenant_id: str | None) -> str:
    """Determine which graph version a tenant should use."""
    state = get_graph_registry_state()
    if tenant_id and tenant_id in state.tenant_pins and state.tenant_pins[tenant_id] in state.available_versions:
        return state.tenant_pins[tenant_id]
    return state.default_version


def is_hot_reload_enabled() -> bool:
    """Check whether hot reload is explicitly enabled via env config."""
    return os.getenv("GRAPH_HOT_RELOAD", "false").lower() == "true"
