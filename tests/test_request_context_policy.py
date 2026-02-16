import sys
from pathlib import Path

from starlette.requests import Request

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "chat-api"))

from src.services.request_context import (
    extract_request_context,
    is_tenant_allowed,
    is_role_allowed_for_model,
    is_abac_allowed_for_model,
)


def _request(headers: dict[str, str]) -> Request:
    scope = {"type": "http", "method": "POST", "path": "/", "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()]}
    return Request(scope)


def test_tenant_role_abac_policy(monkeypatch):
    monkeypatch.setenv("POLICY_ALLOWED_TENANTS", "tenant-a,tenant-b")
    monkeypatch.setenv("MODEL_ROLE_MAP", "gpt-4o:admin|analyst")
    monkeypatch.setenv("MODEL_ABAC_RULES", "gpt-4o:department=research")

    req = _request(
        {
            "x-tenant-id": "tenant-a",
            "x-user-role": "analyst",
            "x-user-attrs": "department=research",
        }
    )
    ctx = extract_request_context(req)
    assert is_tenant_allowed(ctx)
    assert is_role_allowed_for_model("gpt-4o-mini", ctx)
    assert is_abac_allowed_for_model("gpt-4o-mini", ctx)


def test_tenant_denied(monkeypatch):
    monkeypatch.setenv("POLICY_ALLOWED_TENANTS", "tenant-a")
    ctx = extract_request_context(_request({"x-tenant-id": "tenant-x"}))
    assert not is_tenant_allowed(ctx)
