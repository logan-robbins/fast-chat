import ast
from pathlib import Path


def _stream_options_fields() -> dict[str, str]:
    src = Path("chat-api/src/routers/chat.py").read_text()
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "StreamOptions":
            fields: dict[str, str] = {}
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    name = item.target.id
                    default_repr = ast.unparse(item.value) if item.value else ""
                    fields[name] = default_repr
            return fields
    raise AssertionError("StreamOptions class not found")


def test_stream_options_has_contract_fields():
    fields = _stream_options_fields()
    assert "include_usage" in fields
    assert "include_status" in fields
    assert "event_envelope" in fields


def test_stream_options_event_envelope_default_is_legacy():
    fields = _stream_options_fields()
    assert fields["event_envelope"] == "'legacy'"
