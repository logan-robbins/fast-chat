"""Runtime model policy controls (tenant/profile aware via env configuration)."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPolicy:
    allowed_families: tuple[str, ...]
    max_context_tokens: int | None
    max_output_tokens: int | None
    fallback_chain: tuple[str, ...]


def _parse_csv(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def get_model_policy() -> ModelPolicy:
    """Read runtime model policy settings from environment variables."""
    allowed_families = _parse_csv(os.getenv("MODEL_POLICY_ALLOWED_FAMILIES", ""))
    max_context_raw = os.getenv("MODEL_POLICY_MAX_CONTEXT_TOKENS", "")
    max_output_raw = os.getenv("MODEL_POLICY_MAX_OUTPUT_TOKENS", "")
    fallback_chain = _parse_csv(os.getenv("MODEL_FALLBACK_CHAIN", ""))

    return ModelPolicy(
        allowed_families=allowed_families,
        max_context_tokens=int(max_context_raw) if max_context_raw.strip() else None,
        max_output_tokens=int(max_output_raw) if max_output_raw.strip() else None,
        fallback_chain=fallback_chain,
    )


def is_model_allowed(model_id: str, allowed_families: tuple[str, ...] | None = None) -> bool:
    """Determine if the requested model belongs to an allowed family."""
    families = allowed_families if allowed_families is not None else get_model_policy().allowed_families
    if not families:
        return True
    return any(model_id.startswith(family) for family in families)


def resolve_fallback_chain(requested_model: str) -> list[str]:
    """Construct a fallback chain prefixed by the requested model."""
    policy = get_model_policy()
    if not policy.fallback_chain:
        return [requested_model]
    return [requested_model, *[m for m in policy.fallback_chain if m != requested_model]]
