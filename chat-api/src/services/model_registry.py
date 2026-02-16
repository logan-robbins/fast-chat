"""
Authoritative model registry for the Chat API.

Single source of truth for all supported models, their capabilities,
context window limits, and metadata. Every router and service imports
from here to guarantee consistency across endpoints.

Architecture:
    - /v1/models      -> get_model_objects()   for listing
    - /v1/chat/completions -> is_model_supported() for validation
    - /v1/responses    -> is_model_supported()  for validation
    - tokens.py        -> get_context_limit()   for context management

Adding a new model:
    1. Add an entry to _MODELS below.
    2. That's it -- all endpoints pick it up automatically.

Last Grunted: 02/05/2026 12:00:00 PM UTC
"""
from __future__ import annotations

import structlog
from dataclasses import dataclass
from typing import Optional

from src.services.model_policy import resolve_fallback_chain

logger = structlog.get_logger(__name__)


# ============================================================================
# Model Specification Dataclass
# ============================================================================

@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Immutable specification for a single supported model.

    Attributes:
        id: Canonical model identifier (e.g. ``"gpt-4o"``).
        owned_by: Organisation that owns the model.
        created: Approximate Unix timestamp of public availability.
        context_window: Maximum *total* context tokens (input + output).
        max_output_tokens: Maximum tokens the model can generate.
        supports_streaming: Whether SSE streaming is supported.
        supports_tools: Whether function / tool calling is supported.
        supports_vision: Whether the model accepts image inputs.
        supports_reasoning: Whether the model is an o-series reasoning model.
        tiktoken_encoding: Encoding name used by ``tiktoken`` for this model.
        description: Short human-readable description.
    """

    id: str
    owned_by: str
    created: int
    context_window: int
    max_output_tokens: int
    supports_streaming: bool = True
    supports_tools: bool = True
    supports_vision: bool = False
    supports_reasoning: bool = False
    tiktoken_encoding: str = "o200k_base"
    description: str = ""


# ============================================================================
# Canonical Model Definitions
# ============================================================================
# Keep alphabetical within each family for easy scanning.

_MODELS: dict[str, ModelSpec] = {
    # --- GPT-3.5 (legacy) ---------------------------------------------------
    "gpt-3.5-turbo": ModelSpec(
        id="gpt-3.5-turbo",
        owned_by="openai",
        created=1677610602,
        context_window=16_385,
        max_output_tokens=4_096,
        supports_vision=False,
        tiktoken_encoding="cl100k_base",
        description="Legacy fast model for simple tasks",
    ),

    # --- GPT-4 Turbo (legacy) -----------------------------------------------
    "gpt-4-turbo": ModelSpec(
        id="gpt-4-turbo",
        owned_by="openai",
        created=1712361441,
        context_window=128_000,
        max_output_tokens=4_096,
        supports_vision=True,
        tiktoken_encoding="cl100k_base",
        description="High-intelligence legacy model with vision",
    ),

    # --- GPT-4o family -------------------------------------------------------
    "gpt-4o": ModelSpec(
        id="gpt-4o",
        owned_by="openai",
        created=1715367049,
        context_window=128_000,
        max_output_tokens=16_384,
        supports_vision=True,
        description="Flagship multimodal model for text, vision, and audio",
    ),
    "gpt-4o-mini": ModelSpec(
        id="gpt-4o-mini",
        owned_by="openai",
        created=1721172741,
        context_window=128_000,
        max_output_tokens=16_384,
        supports_vision=True,
        description="Small, fast, affordable multimodal model",
    ),

    # --- GPT-4.1 family (1 M context) ----------------------------------------
    "gpt-4.1": ModelSpec(
        id="gpt-4.1",
        owned_by="openai",
        created=1744934400,
        context_window=1_000_000,
        max_output_tokens=32_768,
        supports_vision=True,
        description="API-first model with 1M context, best-in-class coding",
    ),
    "gpt-4.1-mini": ModelSpec(
        id="gpt-4.1-mini",
        owned_by="openai",
        created=1744934400,
        context_window=1_000_000,
        max_output_tokens=32_768,
        supports_vision=True,
        description="Fast, affordable GPT-4.1 variant",
    ),
    "gpt-4.1-nano": ModelSpec(
        id="gpt-4.1-nano",
        owned_by="openai",
        created=1744934400,
        context_window=1_000_000,
        max_output_tokens=32_768,
        supports_vision=False,
        description="Fastest and cheapest GPT-4.1 variant",
    ),

    # --- GPT-5 family --------------------------------------------------------
    "gpt-5": ModelSpec(
        id="gpt-5",
        owned_by="openai",
        created=1754006400,
        context_window=400_000,
        max_output_tokens=128_000,
        supports_vision=True,
        description="Latest-generation flagship model",
    ),

    # --- O-series (reasoning) ------------------------------------------------
    "o1": ModelSpec(
        id="o1",
        owned_by="openai",
        created=1726012800,
        context_window=200_000,
        max_output_tokens=100_000,
        supports_vision=True,
        supports_reasoning=True,
        description="Reasoning model for complex multi-step tasks",
    ),
    "o1-mini": ModelSpec(
        id="o1-mini",
        owned_by="openai",
        created=1726012800,
        context_window=128_000,
        max_output_tokens=65_536,
        supports_vision=False,
        supports_reasoning=True,
        description="Small, cost-effective reasoning model",
    ),
    "o1-pro": ModelSpec(
        id="o1-pro",
        owned_by="openai",
        created=1741132800,
        context_window=200_000,
        max_output_tokens=100_000,
        supports_vision=True,
        supports_reasoning=True,
        description="Extended-compute reasoning for highest reliability",
    ),
    "o3": ModelSpec(
        id="o3",
        owned_by="openai",
        created=1745020800,
        context_window=200_000,
        max_output_tokens=100_000,
        supports_vision=True,
        supports_reasoning=True,
        description="Latest reasoning model, excels at coding, math, science",
    ),
    "o3-mini": ModelSpec(
        id="o3-mini",
        owned_by="openai",
        created=1738281600,
        context_window=200_000,
        max_output_tokens=100_000,
        supports_vision=False,
        supports_reasoning=True,
        description="Cost-effective reasoning model optimised for STEM",
    ),
}


# ============================================================================
# Public Query API
# ============================================================================

def get_all_models() -> dict[str, ModelSpec]:
    """Return the full model registry (read-only snapshot).

    Returns:
        Mapping of model ID to ModelSpec.
    """
    return dict(_MODELS)


def get_model(model_id: str) -> Optional[ModelSpec]:
    """Look up a model by exact ID.

    Args:
        model_id: Canonical model name (e.g. ``"gpt-4o"``).

    Returns:
        ModelSpec if found, ``None`` otherwise.
    """
    return _MODELS.get(model_id)


def get_supported_model_ids() -> set[str]:
    """Return the set of all supported model IDs.

    Useful for fast membership checks in validation logic.

    Returns:
        Frozen set of model ID strings.
    """
    return set(_MODELS.keys())


def is_model_supported(model_id: str) -> bool:
    """Check whether *model_id* (or a dated variant) is supported.

    Supports both exact matches (``"gpt-4o"``) and dated variants
    (``"gpt-4o-2024-08-06"``).

    Args:
        model_id: Model identifier from the request.

    Returns:
        ``True`` if the model is supported.
    """
    if model_id in _MODELS:
        return True
    # Allow dated variants like gpt-4o-2024-08-06
    return any(model_id.startswith(base) for base in _MODELS)


def get_context_limit(model_id: str) -> int:
    """Return the context-window token limit for *model_id*.

    Falls back to 128 000 for unknown models.

    Args:
        model_id: Model identifier.

    Returns:
        Maximum context tokens.
    """
    spec = _MODELS.get(model_id)
    if spec:
        return spec.context_window

    # Try prefix match for dated variants
    for base_id, base_spec in _MODELS.items():
        if model_id.startswith(base_id):
            return base_spec.context_window

    return 128_000  # safe default


def get_tiktoken_encoding(model_id: str) -> str:
    """Return the tiktoken encoding name for *model_id*.

    Falls back to ``"o200k_base"`` for unknown models.

    Args:
        model_id: Model identifier.

    Returns:
        Encoding name string.
    """
    spec = _MODELS.get(model_id)
    if spec:
        return spec.tiktoken_encoding

    for base_id, base_spec in _MODELS.items():
        if model_id.startswith(base_id):
            return base_spec.tiktoken_encoding

    return "o200k_base"


def get_model_objects() -> list[dict]:
    """Return model objects in OpenAI ``/v1/models`` list format.

    Returns:
        List of dicts with ``id``, ``object``, ``created``, ``owned_by``.
    """
    return [
        {
            "id": spec.id,
            "object": "model",
            "created": spec.created,
            "owned_by": spec.owned_by,
        }
        for spec in _MODELS.values()
    ]


def get_resolved_model_chain(requested_model: str) -> list[str]:
    """Return declarative fallback chain filtered to supported models."""
    chain = resolve_fallback_chain(requested_model)
    resolved: list[str] = []
    for model in chain:
        if is_model_supported(model) and model not in resolved:
            resolved.append(model)
    return resolved or [requested_model]
