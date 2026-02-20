"""Real end-to-end tests against running fast-chat services."""
from __future__ import annotations

import os
import uuid

import httpx
import pytest

CHAT_API_URL = os.getenv("CHAT_API_URL", "http://localhost:8000").rstrip("/")
TIMEOUT = httpx.Timeout(connect=10.0, read=240.0, write=60.0, pool=10.0)
MODEL = os.getenv("E2E_MODEL", "gpt-4o-mini")


@pytest.mark.asyncio
async def test_real_health_checks():
    async with httpx.AsyncClient(timeout=TIMEOUT) as api_client:
        health = await api_client.get(f"{CHAT_API_URL}/health")
        assert health.status_code == 200
        assert health.json().get("status") == "ok"

        ready = await api_client.get(f"{CHAT_API_URL}/health/ready")
        assert ready.status_code == 200
        assert ready.json().get("status") == "ready"


@pytest.mark.asyncio
async def test_real_chat_completions_roundtrip():
    payload = {
        "model": MODEL,
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Return one short sentence that includes the token "
                    "REAL_CHAT_E2E_OK."
                ),
            }
        ],
    }

    async with httpx.AsyncClient(timeout=TIMEOUT) as api_client:
        response = await api_client.post(f"{CHAT_API_URL}/v1/chat/completions", json=payload)
        assert response.status_code == 200, response.text

        body = response.json()
        content = body["choices"][0]["message"]["content"]
        assert isinstance(content, str)
        assert content.strip()
        assert body["usage"]["total_tokens"] > 0


@pytest.mark.asyncio
async def test_real_responses_api_lifecycle():
    request_body = {
        "model": MODEL,
        "stream": False,
        "store": True,
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Reply with REAL_RESPONSES_E2E_OK"}],
            }
        ],
        "metadata": {"e2e_run": str(uuid.uuid4())},
    }

    async with httpx.AsyncClient(timeout=TIMEOUT) as api_client:
        create = await api_client.post(f"{CHAT_API_URL}/v1/responses", json=request_body)
        assert create.status_code == 200, create.text
        created = create.json()
        response_id = created["id"]
        assert response_id.startswith("resp_")

        output_text = created["output"][0]["content"][0]["text"]
        assert isinstance(output_text, str)
        assert output_text.strip()
        assert created["usage"]["total_tokens"] > 0

        get_resp = await api_client.get(f"{CHAT_API_URL}/v1/responses/{response_id}")
        assert get_resp.status_code == 200, get_resp.text
        fetched = get_resp.json()
        assert fetched["id"] == response_id
        assert fetched["status"] == "completed"

        input_items = await api_client.get(f"{CHAT_API_URL}/v1/responses/{response_id}/input_items")
        assert input_items.status_code == 200, input_items.text
        input_items_json = input_items.json()
        assert input_items_json["object"] == "list"
        assert len(input_items_json["data"]) >= 1

        delete_resp = await api_client.delete(f"{CHAT_API_URL}/v1/responses/{response_id}")
        assert delete_resp.status_code == 200, delete_resp.text
        assert delete_resp.json().get("deleted") is True

        get_after_delete = await api_client.get(f"{CHAT_API_URL}/v1/responses/{response_id}")
        assert get_after_delete.status_code == 404, get_after_delete.text
