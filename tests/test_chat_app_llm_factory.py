import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "chat-app" / "src"))


def test_litellm_base_url_is_preferred(monkeypatch):
    """Ensure LiteLLM base_url overrides the OpenAI base when provided."""
    monkeypatch.setenv("LITELLM_BASE_URL", "http://litellm:4000")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    from chat_app.services import llm_factory

    importlib.reload(llm_factory)
    assert llm_factory.get_llm_runtime_kwargs()["base_url"] == "http://litellm:4000"


def test_openai_use_responses_api_defaults_true(monkeypatch):
    """Confirm the Responses API flag defaults to True when unset."""
    monkeypatch.delenv("OPENAI_USE_RESPONSES_API", raising=False)

    from chat_app.services import llm_factory

    importlib.reload(llm_factory)
    assert llm_factory.use_responses_api() is True


def test_openai_use_responses_api_validates_boolean(monkeypatch):
    """Validate the OPENAI_USE_RESPONSES_API env var must be boolean-like."""
    monkeypatch.setenv("OPENAI_USE_RESPONSES_API", "not-a-bool")

    from chat_app.services import llm_factory

    importlib.reload(llm_factory)
    with pytest.raises(RuntimeError, match="OPENAI_USE_RESPONSES_API"):
        llm_factory.use_responses_api()


def test_litellm_api_key_requires_litellm_base_url(monkeypatch):
    """Require both LiteLLM API key and base URL to be configured together."""
    monkeypatch.delenv("LITELLM_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("LITELLM_API_KEY", "test-gateway-key")

    from chat_app.services import llm_factory

    importlib.reload(llm_factory)
    with pytest.raises(RuntimeError, match="LITELLM_API_KEY is set but LITELLM_BASE_URL is empty"):
        llm_factory.get_llm_runtime_kwargs()
