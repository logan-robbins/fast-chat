import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "chat-api"))


def test_model_policy_allowed_families(monkeypatch):
    """Confirm allowed model families are loaded and enforced."""
    monkeypatch.setenv("MODEL_POLICY_ALLOWED_FAMILIES", "gpt-4o,o3")
    from src.services import model_policy

    importlib.reload(model_policy)

    policy = model_policy.get_model_policy()
    assert policy.allowed_families == ("gpt-4o", "o3")
    assert model_policy.is_model_allowed("gpt-4o-mini", policy.allowed_families)
    assert not model_policy.is_model_allowed("gpt-3.5-turbo", policy.allowed_families)


def test_model_policy_fallback_chain(monkeypatch):
    """Ensure the fallback chain includes configured alternatives."""
    monkeypatch.setenv("MODEL_FALLBACK_CHAIN", "gpt-4o-mini,gpt-4.1-mini")
    from src.services import model_policy

    importlib.reload(model_policy)

    chain = model_policy.resolve_fallback_chain("gpt-4o")
    assert chain == ["gpt-4o", "gpt-4o-mini", "gpt-4.1-mini"]
