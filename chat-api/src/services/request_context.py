"""Tenant/user request context extraction and lightweight RBAC/ABAC checks."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from fastapi import Request


@dataclass(frozen=True)
class RequestContext:
    tenant_id: str | None
    user_id: str | None
    role: str | None
    workspace: str | None
    attributes: dict[str, Any]


def _parse_kv_csv(value: str | None) -> dict[str, str]:
    if not value:
        return {}
    pairs: dict[str, str] = {}
    for part in value.split(","):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        pairs[k.strip()] = v.strip()
    return pairs


def extract_request_context(request: Request) -> RequestContext:
    """Build RequestContext from standard header metadata and fallback parsing."""
    attrs_raw = request.headers.get("x-user-attrs", "")
    attrs: dict[str, Any] = {}
    if attrs_raw:
        try:
            attrs = json.loads(attrs_raw)
        except json.JSONDecodeError:
            attrs = _parse_kv_csv(attrs_raw)

    return RequestContext(
        tenant_id=request.headers.get("x-tenant-id"),
        user_id=request.headers.get("x-user-id"),
        role=request.headers.get("x-user-role"),
        workspace=request.headers.get("x-workspace"),
        attributes=attrs,
    )


def is_tenant_allowed(ctx: RequestContext) -> bool:
    """Verify that the request tenant is in the allowlist configured via env."""
    allowed = os.getenv("POLICY_ALLOWED_TENANTS", "").strip()
    if not allowed:
        return True
    if not ctx.tenant_id:
        return False
    allowed_set = {item.strip() for item in allowed.split(",") if item.strip()}
    return ctx.tenant_id in allowed_set


def is_role_allowed_for_model(model: str, ctx: RequestContext) -> bool:
    """Enforce role-based access by mapping model prefixes to allowed roles."""
    role_map = os.getenv("MODEL_ROLE_MAP", "").strip()
    if not role_map:
        return True
    if not ctx.role:
        return False

    for rule in role_map.split(";"):
        if ":" not in rule:
            continue
        prefix, roles_csv = rule.split(":", 1)
        prefix = prefix.strip()
        if model.startswith(prefix):
            allowed_roles = {r.strip() for r in roles_csv.split("|") if r.strip()}
            return ctx.role in allowed_roles
    return True


def is_abac_allowed_for_model(model: str, ctx: RequestContext) -> bool:
    """MODEL_ABAC_RULES example: gpt-4o:department=research|tier=premium;o3:clearance=high"""
    rules = os.getenv("MODEL_ABAC_RULES", "").strip()
    if not rules:
        return True

    for rule in rules.split(";"):
        if ":" not in rule:
            continue
        prefix, requirements = rule.split(":", 1)
        if not model.startswith(prefix.strip()):
            continue
        for req in requirements.split("|"):
            if "=" not in req:
                continue
            key, expected = req.split("=", 1)
            if str(ctx.attributes.get(key.strip())) != expected.strip():
                return False
    return True
