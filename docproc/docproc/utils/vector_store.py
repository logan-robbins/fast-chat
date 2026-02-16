"""
Vector Store - ChromaDB Implementation
"""
import asyncio
import logging
import math
import os
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any

import chromadb

logger = logging.getLogger(__name__)

PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "")
DEFAULT_COLLECTION = os.getenv("CHROMA_COLLECTION", "documents")
VECTOR_STORE_PROVIDER = os.getenv("VECTOR_STORE_PROVIDER", "chroma").lower()

_client: chromadb.ClientAPI | None = None


def _get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        if PERSIST_DIR:
            logger.info(f"ChromaDB: Persistent mode ({PERSIST_DIR})")
            _client = chromadb.PersistentClient(path=PERSIST_DIR)
        else:
            logger.info("ChromaDB: In-memory mode")
            _client = chromadb.Client()
    return _client


class VectorStoreAdapter(ABC):
    @abstractmethod
    async def initialize(self) -> None: ...

    @abstractmethod
    async def add_document(self, doc_id: str, embedding: list[float], text: str, metadata: dict[str, Any]) -> bool: ...

    @abstractmethod
    async def add_documents_batch(self, documents: list[dict[str, Any]]) -> int: ...

    @abstractmethod
    async def get_document(self, doc_id: str) -> dict[str, Any] | None: ...

    @abstractmethod
    async def update_document(
        self,
        doc_id: str,
        embedding: list[float],
        text: str,
        metadata: dict[str, Any],
    ) -> bool: ...

    @abstractmethod
    async def search_similar(
        self,
        query_embedding: list[float],
        limit: int = 5,
        where: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> list[dict[str, Any]]: ...

    @abstractmethod
    async def delete_document(self, doc_id: str) -> bool: ...

    @abstractmethod
    async def delete_documents(self, doc_ids: list[str]) -> bool: ...

    @abstractmethod
    async def delete_by_metadata(self, where: dict[str, Any]) -> bool: ...

    @abstractmethod
    async def get_all_documents(self) -> list[dict[str, Any]]: ...

    @abstractmethod
    async def count(self) -> int: ...

    @abstractmethod
    async def clear_all(self) -> bool: ...

    @abstractmethod
    async def drop_collection(self) -> bool: ...


class InMemoryVectorStore(VectorStoreAdapter):
    _collections: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)

    def __init__(self, collection_name: str = DEFAULT_COLLECTION):
        self.collection_name = collection_name
        self._docs = self._collections[collection_name]

    async def initialize(self) -> None:
        return None

    async def add_document(self, doc_id: str, embedding: list[float], text: str, metadata: dict[str, Any]) -> bool:
        self._docs[doc_id] = {"id": doc_id, "embedding": embedding, "text": text, "metadata": metadata}
        return True

    async def add_documents_batch(self, documents: list[dict[str, Any]]) -> int:
        for item in documents:
            await self.add_document(item["doc_id"], item["embedding"], item["text"], item["metadata"])
        return len(documents)

    async def get_document(self, doc_id: str) -> dict[str, Any] | None:
        doc = self._docs.get(doc_id)
        if not doc:
            return None
        return {"id": doc["id"], "text": doc["text"], "metadata": doc["metadata"]}

    async def update_document(
        self,
        doc_id: str,
        embedding: list[float],
        text: str,
        metadata: dict[str, Any],
    ) -> bool:
        if doc_id not in self._docs:
            return False
        self._docs[doc_id] = {"id": doc_id, "embedding": embedding, "text": text, "metadata": metadata}
        return True

    async def search_similar(
        self,
        query_embedding: list[float],
        limit: int = 5,
        where: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        matches = []
        for doc in self._docs.values():
            if where and not all(doc["metadata"].get(k) == v for k, v in where.items()):
                continue
            similarity = self._cosine_similarity(query_embedding, doc["embedding"])
            if score_threshold is not None and similarity < score_threshold:
                continue
            matches.append(
                {
                    "id": doc["id"],
                    "text": doc["text"],
                    "metadata": doc["metadata"],
                    "distance": 1 - similarity,
                    "similarity_score": similarity,
                }
            )
        matches.sort(key=lambda item: item["similarity_score"], reverse=True)
        return matches[:limit]

    async def delete_document(self, doc_id: str) -> bool:
        self._docs.pop(doc_id, None)
        return True

    async def delete_documents(self, doc_ids: list[str]) -> bool:
        for doc_id in doc_ids:
            self._docs.pop(doc_id, None)
        return True

    async def delete_by_metadata(self, where: dict[str, Any]) -> bool:
        to_delete = [doc_id for doc_id, doc in self._docs.items() if all(doc["metadata"].get(k) == v for k, v in where.items())]
        for doc_id in to_delete:
            self._docs.pop(doc_id, None)
        return True

    async def get_all_documents(self) -> list[dict[str, Any]]:
        return [{"id": d["id"], "text": d["text"], "metadata": d["metadata"]} for d in self._docs.values()]

    async def count(self) -> int:
        return len(self._docs)

    async def clear_all(self) -> bool:
        self._docs.clear()
        return True

    async def drop_collection(self) -> bool:
        self._collections.pop(self.collection_name, None)
        self._docs = self._collections[self.collection_name]
        return True

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        if len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b, strict=False))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


class ChromaVectorStore(VectorStoreAdapter):
    def __init__(self, collection_name: str = DEFAULT_COLLECTION):
        self.collection_name = collection_name
        self._collection = None

    async def initialize(self) -> None:
        client = _get_client()
        self._collection = await asyncio.to_thread(
            client.get_or_create_collection,
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    async def add_document(self, doc_id: str, embedding: list[float], text: str, metadata: dict[str, Any]) -> bool:
        try:
            await asyncio.to_thread(
                self._collection.add,
                ids=[doc_id],
                embeddings=[embedding],
                documents=[text],
                metadatas=[self._clean_metadata(metadata)],
            )
            return True
        except Exception as e:
            logger.error(f"ChromaDB add failed: {e}")
            return False

    async def add_documents_batch(self, documents: list[dict[str, Any]]) -> int:
        if not documents:
            return 0
        try:
            await asyncio.to_thread(
                self._collection.add,
                ids=[d["doc_id"] for d in documents],
                embeddings=[d["embedding"] for d in documents],
                documents=[d["text"] for d in documents],
                metadatas=[self._clean_metadata(d["metadata"]) for d in documents],
            )
            return len(documents)
        except Exception as e:
            logger.error(f"ChromaDB batch add failed: {e}")
            return 0

    async def get_document(self, doc_id: str) -> dict[str, Any] | None:
        try:
            result = await asyncio.to_thread(
                self._collection.get,
                ids=[doc_id],
                include=["documents", "metadatas"],
            )
            if result["ids"]:
                return {
                    "id": result["ids"][0],
                    "text": result["documents"][0] if result["documents"] else "",
                    "metadata": result["metadatas"][0] if result["metadatas"] else {},
                }
            return None
        except Exception as e:
            logger.error(f"ChromaDB get failed: {e}")
            return None

    async def update_document(
        self,
        doc_id: str,
        embedding: list[float],
        text: str,
        metadata: dict[str, Any],
    ) -> bool:
        try:
            await asyncio.to_thread(
                self._collection.update,
                ids=[doc_id],
                embeddings=[embedding],
                documents=[text],
                metadatas=[self._clean_metadata(metadata)],
            )
            return True
        except Exception as e:
            logger.error(f"ChromaDB update failed: {e}")
            return False

    async def search_similar(
        self,
        query_embedding: list[float],
        limit: int = 5,
        where: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        try:
            query_kwargs: dict[str, Any] = {
                "query_embeddings": [query_embedding],
                "n_results": limit,
                "include": ["documents", "metadatas", "distances"],
            }
            if where is not None:
                query_kwargs["where"] = where
            results = await asyncio.to_thread(self._collection.query, **query_kwargs)
            docs = []
            if results["ids"] and results["ids"][0]:
                for i, doc_id in enumerate(results["ids"][0]):
                    distance = results["distances"][0][i] if results["distances"] else 0
                    similarity = 1 - distance
                    if score_threshold is not None and similarity < score_threshold:
                        continue
                    docs.append(
                        {
                            "id": doc_id,
                            "text": results["documents"][0][i] if results["documents"] else "",
                            "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                            "distance": distance,
                            "similarity_score": similarity,
                        }
                    )
            return docs
        except Exception as e:
            logger.error(f"ChromaDB search failed: {e}")
            return []

    async def delete_document(self, doc_id: str) -> bool:
        try:
            await asyncio.to_thread(self._collection.delete, ids=[doc_id])
            return True
        except Exception:
            return False

    async def delete_documents(self, doc_ids: list[str]) -> bool:
        try:
            await asyncio.to_thread(self._collection.delete, ids=doc_ids)
            return True
        except Exception:
            return False

    async def delete_by_metadata(self, where: dict[str, Any]) -> bool:
        try:
            await asyncio.to_thread(self._collection.delete, where=where)
            return True
        except Exception:
            return False

    async def get_all_documents(self) -> list[dict[str, Any]]:
        try:
            result = await asyncio.to_thread(self._collection.get, include=["documents", "metadatas"])
            return [
                {
                    "id": doc_id,
                    "text": result["documents"][i] if result["documents"] else "",
                    "metadata": result["metadatas"][i] if result["metadatas"] else {},
                }
                for i, doc_id in enumerate(result["ids"])
            ]
        except Exception:
            return []

    async def count(self) -> int:
        try:
            return await asyncio.to_thread(self._collection.count)
        except Exception:
            return 0

    async def clear_all(self) -> bool:
        try:
            client = _get_client()
            await asyncio.to_thread(client.delete_collection, self.collection_name)
            await self.initialize()
            return True
        except Exception:
            return False

    async def drop_collection(self) -> bool:
        try:
            client = _get_client()
            await asyncio.to_thread(client.delete_collection, self.collection_name)
            self._collection = None
            return True
        except Exception:
            return False

    def _clean_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        clean = {}
        for key, value in metadata.items():
            if value is None:
                clean[key] = ""
            elif isinstance(value, (str, int, float, bool)):
                clean[key] = value
            else:
                clean[key] = str(value)
        return clean


async def collection_exists(collection_name: str) -> bool:
    try:
        if VECTOR_STORE_PROVIDER == "memory":
            return collection_name in InMemoryVectorStore._collections
        client = _get_client()
        collections = await asyncio.to_thread(client.list_collections)
        return any(c.name == collection_name for c in collections)
    except Exception as e:
        logger.error(f"Vector store collection_exists check failed: {e}")
        return False


async def list_collections() -> list[str]:
    try:
        if VECTOR_STORE_PROVIDER == "memory":
            return list(InMemoryVectorStore._collections.keys())
        client = _get_client()
        collections = await asyncio.to_thread(client.list_collections)
        return [c.name for c in collections]
    except Exception as e:
        logger.error(f"Vector store list_collections failed: {e}")
        return []


def get_vector_store(collection_name: str = DEFAULT_COLLECTION) -> VectorStoreAdapter:
    if VECTOR_STORE_PROVIDER == "memory":
        return InMemoryVectorStore(collection_name=collection_name)
    if VECTOR_STORE_PROVIDER == "chroma":
        return ChromaVectorStore(collection_name=collection_name)
    raise ValueError(
        f"Unsupported VECTOR_STORE_PROVIDER '{VECTOR_STORE_PROVIDER}'. Supported providers: chroma, memory"
    )
