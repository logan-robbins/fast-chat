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
    allowed_issuers = {i.strip() for i in os.getenv("A2A_TRUSTED_ISSUERS", "").split(",") if i.strip()}
    if allowed_issuers and agent.issuer not in allowed_issuers:
        return False
    # lightweight signature policy: non-empty or strict disabled env
    strict = os.getenv("A2A_REQUIRE_SIGNATURE", "true").lower() == "true"
    if strict and not agent.manifest_signature:
        return False
    return True


def route_agent_for_capability(capability: str) -> RemoteAgent | None:
    for agent in load_remote_agents().values():
        if capability in agent.capabilities and is_remote_agent_trusted(agent):
            return agent
    return None
