"""A2A remote agent registry + trust policy checks."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class RemoteAgent:
    agent_id: str
    endpoint: str
    capabilities: tuple[str, ...]
    issuer: str
    manifest_signature: str


def load_remote_agents() -> dict[str, RemoteAgent]:
    """Parse remote agent metadata from the `A2A_AGENTS_JSON` environment variable.

    Returns:
        dict[str, RemoteAgent]: Agent ID to metadata mapping that includes endpoint,
            declared capabilities, issuer, and manifest signature.
    """
    raw = os.getenv("A2A_AGENTS_JSON", "[]")
    parsed = json.loads(raw)
    agents: dict[str, RemoteAgent] = {}
    for item in parsed:
        agents[item["agent_id"]] = RemoteAgent(
            agent_id=item["agent_id"],
            endpoint=item["endpoint"],
            capabilities=tuple(item.get("capabilities", [])),
            issuer=item.get("issuer", ""),
            manifest_signature=item.get("manifest_signature", ""),
        )
    return agents


def is_remote_agent_trusted(agent: RemoteAgent) -> bool:
    """Apply trust checks (issuer whitelist & signature) for a remote agent.

    Args:
        agent: RemoteAgent metadata to evaluate.

    Returns:
        bool: True when the agent meets issuer and signature policies.
    """
    allowed_issuers = {i.strip() for i in os.getenv("A2A_TRUSTED_ISSUERS", "").split(",") if i.strip()}
    if allowed_issuers and agent.issuer not in allowed_issuers:
        return False
    # lightweight signature policy: non-empty or strict disabled env
    strict = os.getenv("A2A_REQUIRE_SIGNATURE", "true").lower() == "true"
    if strict and not agent.manifest_signature:
        return False
    return True


def route_agent_for_capability(capability: str) -> RemoteAgent | None:
    """Find a trusted remote agent that advertises the requested capability."""
    for agent in load_remote_agents().values():
        if capability in agent.capabilities and is_remote_agent_trusted(agent):
            return agent
    return None
