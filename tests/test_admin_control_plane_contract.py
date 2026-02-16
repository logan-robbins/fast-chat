import ast
from pathlib import Path


def test_admin_router_exposes_required_endpoints():
    src = Path("chat-api/src/routers/admin.py").read_text()
    tree = ast.parse(src)

    paths = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "get" and node.args and isinstance(node.args[0], ast.Constant):
                paths.add(node.args[0].value)

    required = {
        "/v1/admin/model-registry",
        "/v1/admin/retrieval-providers",
        "/v1/admin/mcp/servers",
        "/v1/admin/a2a/agents",
        "/v1/admin/graph-versions",
        "/v1/admin/policies",
    }
    assert required.issubset(paths)


def test_admin_router_has_key_based_guard():
    src = Path("chat-api/src/routers/admin.py").read_text()
    assert "ADMIN_API_KEY" in src
    assert "status_code=403" in src
    assert "x_admin_key" in src
