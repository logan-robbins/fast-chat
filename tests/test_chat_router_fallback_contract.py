from pathlib import Path


def test_chat_router_uses_resolved_model_chain():
    """Ensure chat router delegates to the resolved model chain helper."""
    src = Path("chat-api/src/routers/chat.py").read_text()
    assert "get_resolved_model_chain(request.model)" in src
    assert "model_chain=model_chain" in src
