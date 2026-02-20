"""Canonical LLM provider configuration for chat-app.

Centralizes model initialization so every agent and utility uses the same
provider routing behavior:
- OpenAI Responses API enabled by default
- LiteLLM routing when LITELLM_BASE_URL is configured
- OpenAI-compatible base URL routing via OPENAI_BASE_URL
"""
from __future__ import annotations

from typing import Any, Sequence

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from chat_app.services.runtime_config import get_llm_runtime_config


def get_llm_runtime_kwargs() -> dict[str, Any]:
    """Return provider routing kwargs for LangChain OpenAI models."""
    runtime_config = get_llm_runtime_config()
    kwargs: dict[str, Any] = {}

    if runtime_config.base_url:
        kwargs["base_url"] = runtime_config.base_url

    if runtime_config.api_key_override:
        kwargs["api_key"] = runtime_config.api_key_override

    return kwargs


def use_responses_api() -> bool:
    """Whether LangChain OpenAI calls should use the Responses API."""
    return get_llm_runtime_config().use_responses_api


def init_openai_chat_model(
    model_name: str,
    *,
    temperature: float = 0.0,
    streaming: bool = True,
    tags: Sequence[str] | None = None,
) -> BaseChatModel:
    """Create a chat model with shared provider/runtime configuration."""
    resolved_model = model_name if ":" in model_name else f"openai:{model_name}"
    return init_chat_model(
        resolved_model,
        temperature=temperature,
        streaming=streaming,
        use_responses_api=use_responses_api(),
        tags=list(tags or []),
        **get_llm_runtime_kwargs(),
    )
