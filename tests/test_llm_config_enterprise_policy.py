import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docproc"))


def test_enterprise_profile_disables_local_models(monkeypatch):
    monkeypatch.setenv("DEPLOYMENT_PROFILE", "enterprise")
    monkeypatch.setenv("USE_LOCAL_MODELS", "true")

    with pytest.raises(RuntimeError, match="USE_LOCAL_MODELS is not allowed"):
        import docproc.utils.llm_config as llm_config
        importlib.reload(llm_config)


def test_litellm_base_url_routing_kwargs(monkeypatch):
    monkeypatch.setenv("DEPLOYMENT_PROFILE", "oss")
    monkeypatch.setenv("USE_LOCAL_MODELS", "false")
    monkeypatch.setenv("LITELLM_BASE_URL", "http://litellm:4000")

    import docproc.utils.llm_config as llm_config
    importlib.reload(llm_config)

    assert llm_config._get_openai_runtime_kwargs() == {"base_url": "http://litellm:4000"}
