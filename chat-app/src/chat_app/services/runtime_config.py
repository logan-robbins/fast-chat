"""Strict runtime configuration parsing for enterprise extension points.

This module centralizes environment parsing for provider routing and MCP-related
settings so behavior is deterministic and easy to audit.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import cast





@dataclass(frozen=True)
class LLMRuntimeConfig:
    use_responses_api: bool
    base_url: str | None
    api_key_override: str | None


@dataclass(frozen=True)
class WebSearchRuntimeConfig:
    mcp_web_search_tool: str
    mcp_web_search_engine: str


@dataclass(frozen=True)
class MCPTransportConfig:
    http_timeout_seconds: float
    sse_read_timeout_seconds: float


def _normalize_base_url(raw: str | None) -> str | None:
    if not raw:
        return None

    normalized = raw.strip().rstrip("/")
    if not normalized:
        return None

    # Normalize legacy proxy-style OpenAI URL to canonical v1 endpoint.
    if normalized == "https://api.openai.com:18080":
        return "https://api.openai.com/v1"

    return normalized


def _parse_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default

    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"{name} must be a boolean-like value, got: {raw!r}")


def _parse_positive_float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if not raw or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a float, got: {raw!r}") from exc
    if value <= 0:
        raise RuntimeError(f"{name} must be > 0, got: {raw!r}")
    return value


def get_llm_runtime_config() -> LLMRuntimeConfig:
    """Parse OpenAI/LiteLLM runtime routing configuration."""
    litellm_base_url = _normalize_base_url(os.getenv("LITELLM_BASE_URL"))
    openai_base_url = _normalize_base_url(os.getenv("OPENAI_BASE_URL"))
    base_url = litellm_base_url or openai_base_url

    litellm_api_key = os.getenv("LITELLM_API_KEY", "").strip() or None
    if litellm_api_key and not litellm_base_url:
        raise RuntimeError("LITELLM_API_KEY is set but LITELLM_BASE_URL is empty")

    return LLMRuntimeConfig(
        use_responses_api=_parse_bool_env("OPENAI_USE_RESPONSES_API", default=True),
        base_url=base_url,
        api_key_override=litellm_api_key,
    )


def get_web_search_runtime_config() -> WebSearchRuntimeConfig:
    """Parse web search MCP tool args."""
    mcp_web_search_tool = os.getenv("MCP_WEB_SEARCH_TOOL", "search_engine").strip() or "search_engine"
    mcp_web_search_engine = os.getenv("MCP_WEB_SEARCH_ENGINE", "google").strip() or "google"

    return WebSearchRuntimeConfig(
        mcp_web_search_tool=mcp_web_search_tool,
        mcp_web_search_engine=mcp_web_search_engine,
    )


def get_mcp_transport_config() -> MCPTransportConfig:
    """Parse MCP HTTP/SSE timeout controls."""
    return MCPTransportConfig(
        http_timeout_seconds=_parse_positive_float_env("MCP_HTTP_TIMEOUT_SECONDS", 30.0),
        sse_read_timeout_seconds=_parse_positive_float_env("MCP_SSE_READ_TIMEOUT_SECONDS", 300.0),
    )
