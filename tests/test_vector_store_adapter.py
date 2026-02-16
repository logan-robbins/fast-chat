import asyncio
import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docproc"))


def test_memory_vector_store_similarity_and_filters(monkeypatch):
    monkeypatch.setenv("VECTOR_STORE_PROVIDER", "memory")
    from docproc.utils import vector_store as vs

    importlib.reload(vs)

    async def _run() -> None:
        store = vs.get_vector_store("unit_collection")
        await store.initialize()

        await store.add_document(
            doc_id="doc_a",
            embedding=[1.0, 0.0, 0.0],
            text="alpha document",
            metadata={"filename": "a.txt", "topic": "alpha"},
        )
        await store.add_document(
            doc_id="doc_b",
            embedding=[0.0, 1.0, 0.0],
            text="beta document",
            metadata={"filename": "b.txt", "topic": "beta"},
        )

        results = await store.search_similar([0.9, 0.1, 0.0], limit=2)
        assert len(results) == 2
        assert results[0]["id"] == "doc_a"
        assert results[0]["similarity_score"] > results[1]["similarity_score"]

        filtered = await store.search_similar([0.9, 0.1, 0.0], where={"topic": "beta"}, limit=5)
        assert [r["id"] for r in filtered] == ["doc_b"]

        thresholded = await store.search_similar([1.0, 0.0, 0.0], score_threshold=0.95, limit=5)
        assert [r["id"] for r in thresholded] == ["doc_a"]

    asyncio.run(_run())


def test_memory_vector_store_delete_by_metadata(monkeypatch):
    monkeypatch.setenv("VECTOR_STORE_PROVIDER", "memory")
    from docproc.utils import vector_store as vs

    importlib.reload(vs)

    async def _run() -> None:
        store = vs.get_vector_store("delete_collection")
        await store.initialize()
        await store.add_document("1", [1.0, 0.0], "one", {"kind": "keep"})
        await store.add_document("2", [0.0, 1.0], "two", {"kind": "drop"})

        assert await store.count() == 2
        await store.delete_by_metadata({"kind": "drop"})
        remaining = await store.get_all_documents()
        assert [doc["id"] for doc in remaining] == ["1"]

    asyncio.run(_run())


def test_memory_vector_store_get_and_update(monkeypatch):
    monkeypatch.setenv("VECTOR_STORE_PROVIDER", "memory")
    from docproc.utils import vector_store as vs

    importlib.reload(vs)

    async def _run() -> None:
        store = vs.get_vector_store("update_collection")
        await store.initialize()
        await store.add_document("doc", [1.0, 0.0], "initial", {"v": 1})

        before = await store.get_document("doc")
        assert before is not None and before["text"] == "initial"

        ok = await store.update_document("doc", [0.5, 0.5], "updated", {"v": 2})
        assert ok is True

        after = await store.get_document("doc")
        assert after is not None
        assert after["text"] == "updated"
        assert after["metadata"]["v"] == 2

    asyncio.run(_run())


def test_unsupported_vector_store_provider_raises(monkeypatch):
    monkeypatch.setenv("VECTOR_STORE_PROVIDER", "not-a-provider")
    from docproc.utils import vector_store as vs

    importlib.reload(vs)

    with pytest.raises(ValueError, match="Unsupported VECTOR_STORE_PROVIDER"):
        vs.get_vector_store("any")
