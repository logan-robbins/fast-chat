import importlib
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "chat-api"))


def test_resolved_model_chain_filters_unknown(monkeypatch):
    """Verify the resolved model chain skips entries that fail validation."""
    monkeypatch.setenv("MODEL_FALLBACK_CHAIN", "not-a-model,gpt-4o-mini")

    fake_structlog = types.SimpleNamespace(get_logger=lambda *_args, **_kwargs: types.SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None))
    monkeypatch.setitem(sys.modules, "structlog", fake_structlog)

    from src.services import model_policy
    from src.services import model_registry

    importlib.reload(model_policy)
    importlib.reload(model_registry)

    chain = model_registry.get_resolved_model_chain("gpt-4o")
    assert chain == ["gpt-4o", "gpt-4o-mini"]
