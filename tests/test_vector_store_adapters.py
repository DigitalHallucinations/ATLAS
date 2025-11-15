from __future__ import annotations

import asyncio
from typing import Any, Dict, Iterable, Mapping, Optional

import pytest

from modules.Tools.Base_Tools.vector_store import VectorRecord
from modules.Tools.providers.vector_store.chroma import ChromaVectorStoreAdapter
from modules.Tools.providers.vector_store.faiss import FaissVectorStoreAdapter
from modules.Tools.providers.vector_store.in_memory import InMemoryVectorStoreAdapter
from modules.Tools.providers.vector_store.mongodb import MongoDBVectorStoreAdapter
from modules.Tools.providers.vector_store.pinecone import PineconeVectorStoreAdapter


class _StubChromaCollection:
    def __init__(self, name: str) -> None:
        self.name = name
        self._records: Dict[str, Dict[str, Any]] = {}

    def upsert(
        self,
        *,
        ids: Iterable[str],
        embeddings: Iterable[Iterable[float]],
        metadatas: Iterable[Mapping[str, Any]],
    ) -> None:
        for vector_id, vector_values, metadata in zip(ids, embeddings, metadatas):
            self._records[str(vector_id)] = {
                "values": [float(value) for value in vector_values],
                "metadata": dict(metadata),
            }

    def query(
        self,
        *,
        query_embeddings: Iterable[Iterable[float]],
        n_results: int,
        where: Optional[Mapping[str, Any]],
        include: Iterable[str],
    ) -> Mapping[str, Any]:
        include_embeddings = "embeddings" in set(include)
        ids: list[list[str]] = [[]]
        distances: list[list[float]] = [[]]
        metadatas: list[list[Mapping[str, Any]]] = [[]]
        embeddings: list[list[Iterable[float]]] = [[]] if include_embeddings else []

        for vector_id, payload in list(self._records.items())[:n_results]:
            if where:
                matched = True
                for key, expected in where.items():
                    if payload["metadata"].get(key) != expected:
                        matched = False
                        break
                if not matched:
                    continue
            ids[0].append(vector_id)
            distances[0].append(0.0)
            metadatas[0].append(payload["metadata"])
            if include_embeddings:
                embeddings[0].append(payload["values"])

        result: Dict[str, Any] = {"ids": ids, "distances": distances, "metadatas": metadatas}
        if include_embeddings:
            result["embeddings"] = embeddings
        return result

class _StubChromaClient:
    def __init__(self) -> None:
        self._collections: Dict[str, _StubChromaCollection] = {}

    def get_collection(self, *, name: str) -> _StubChromaCollection:
        return self.get_or_create_collection(name=name)

    def get_or_create_collection(
        self,
        *,
        name: str,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> _StubChromaCollection:
        collection = self._collections.get(name)
        if collection is None:
            collection = _StubChromaCollection(name)
            self._collections[name] = collection
        return collection

    def delete_collection(self, *, name: str) -> bool:
        existed = name in self._collections
        self._collections.pop(name, None)
        return existed


class _StubPineconeIndex:
    def __init__(self) -> None:
        self.namespaces: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.deleted_namespaces: list[str] = []

    def upsert(self, *, vectors: Iterable[Mapping[str, Any]], namespace: str) -> None:
        store = self.namespaces.setdefault(namespace, {})
        for entry in vectors:
            vector_id = str(entry["id"])
            store[vector_id] = {
                "values": list(entry.get("values", ())),
                "metadata": dict(entry.get("metadata", {})),
            }

    def query(
        self,
        *,
        namespace: str,
        vector: Iterable[float],
        top_k: int,
        filter: Optional[Mapping[str, Any]],
        include_values: bool,
        include_metadata: bool,
    ) -> Mapping[str, Any]:
        entries = list(self.namespaces.get(namespace, {}).items())
        matches = []
        for vector_id, payload in entries[:top_k]:
            if filter:
                matched = True
                for key, expected in filter.items():
                    if payload["metadata"].get(key) != expected:
                        matched = False
                        break
                if not matched:
                    continue
            match: Dict[str, Any] = {"id": vector_id, "score": 1.0}
            if include_metadata:
                match["metadata"] = payload["metadata"]
            if include_values:
                match["values"] = payload["values"]
            matches.append(match)
        return {"matches": matches}

    def delete(self, *, namespace: str, delete_all: bool) -> None:
        if delete_all:
            self.deleted_namespaces.append(namespace)
            self.namespaces.pop(namespace, None)


class _StubPineconeClient:
    def __init__(self, index: _StubPineconeIndex) -> None:
        self._index = index

    def Index(self, index_name: str) -> _StubPineconeIndex:  # noqa: N802 - mimics pinecone API
        return self._index


class _StubMongoResult:
    def __init__(self, deleted_count: int) -> None:
        self.deleted_count = deleted_count


class _StubUpdateOne:
    def __init__(self, filter_doc: Mapping[str, Any], update_doc: Mapping[str, Any], *, upsert: bool) -> None:
        self.filter = dict(filter_doc)
        self.update = dict(update_doc)
        self.upsert = upsert


class _StubMongoCollection:
    def __init__(self) -> None:
        self.documents: Dict[tuple[str, str], Dict[str, Any]] = {}

    def bulk_write(self, operations: Iterable[_StubUpdateOne], ordered: bool) -> None:
        for op in operations:
            namespace = str(op.filter.get("namespace", ""))
            vector_id = str(op.filter.get("_id", ""))
            payload = dict(op.update.get("$set", {}))
            payload.setdefault("namespace", namespace)
            payload.setdefault("_id", vector_id)
            self.documents[(namespace, vector_id)] = payload

    def aggregate(self, pipeline: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
        pipeline_list = list(pipeline)
        match_stage = pipeline_list[1]["$match"]
        namespace = match_stage.get("namespace")
        results: list[Dict[str, Any]] = []
        for (doc_namespace, vector_id), payload in self.documents.items():
            if doc_namespace != namespace:
                continue
            include = True
            for key, expected in match_stage.items():
                if key == "namespace":
                    continue
                current: Any = payload
                for part in key.split("."):
                    if isinstance(current, Mapping):
                        current = current.get(part)
                    else:
                        current = None
                        break
                if current != expected:
                    include = False
                    break
            if not include:
                continue
            result = dict(payload)
            result["score"] = 1.0
            results.append(result)
        return results

    def delete_many(self, filter_doc: Mapping[str, Any]) -> _StubMongoResult:
        namespace = filter_doc.get("namespace")
        keys_to_remove = [key for key in self.documents if key[0] == namespace]
        for key in keys_to_remove:
            del self.documents[key]
        return _StubMongoResult(len(keys_to_remove))


class _StubMongoClient(dict):
    pass


def test_in_memory_adapter_roundtrip() -> None:
    async def _run() -> None:
        adapter = InMemoryVectorStoreAdapter(index_name="unit-test")
        records = (
            VectorRecord(id="doc-1", values=(0.1, 0.2), metadata={"tag": "alpha"}),
        )
        upsert = await adapter.upsert_vectors("workspace", records)
        assert upsert.upserted_count == 1

        query = await adapter.query_vectors(
            "workspace",
            (0.1, 0.2),
            top_k=1,
            metadata_filter={"tag": "alpha"},
            include_values=True,
        )
        assert query.matches and query.matches[0].values == records[0].values

        delete = await adapter.delete_namespace("workspace")
        assert delete.deleted is True

    asyncio.run(_run())


def test_chroma_adapter_uses_stub_client() -> None:
    async def _run() -> None:
        client = _StubChromaClient()
        adapter = ChromaVectorStoreAdapter(
            client=client,
            collection_name="atlas",
            metric="cosine",
        )
        records = (
            VectorRecord(id="doc-1", values=(0.1, 0.2), metadata={"tag": "alpha"}),
        )

        await adapter.upsert_vectors("tenant", records)
        response = await adapter.query_vectors(
            "tenant",
            (0.1, 0.2),
            top_k=1,
            metadata_filter={"tag": "alpha"},
            include_values=True,
        )
        assert response.matches and response.matches[0].metadata["tag"] == "alpha"
        assert response.matches[0].values == records[0].values

        delete = await adapter.delete_namespace("tenant")
        assert delete.deleted is True

    asyncio.run(_run())


def test_pinecone_adapter_with_stub_client() -> None:
    async def _run() -> None:
        index = _StubPineconeIndex()
        client = _StubPineconeClient(index)
        adapter = PineconeVectorStoreAdapter(
            index_name="atlas-index",
            client=client,
            namespace_prefix="org-",
        )
        records = (
            VectorRecord(id="doc-1", values=(0.3, 0.4), metadata={"team": "search"}),
        )

        await adapter.upsert_vectors("tenant", records)
        query = await adapter.query_vectors(
            "tenant",
            (0.3, 0.4),
            top_k=1,
            metadata_filter={"team": "search"},
            include_values=True,
        )
        assert query.matches and query.matches[0].metadata["team"] == "search"
        assert query.matches[0].values == records[0].values

        delete = await adapter.delete_namespace("tenant")
        assert delete.deleted is True
        assert index.deleted_namespaces == ["org-tenant"]

    asyncio.run(_run())


def test_mongodb_adapter_roundtrip(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _run() -> None:
        from modules.Tools.providers.vector_store import mongodb as mongodb_module

        monkeypatch.setattr(mongodb_module, "UpdateOne", _StubUpdateOne, raising=False)

        collection = _StubMongoCollection()
        client = _StubMongoClient({"atlas": {"vectors": collection}})

        adapter = MongoDBVectorStoreAdapter(
            client=client,
            database="atlas",
            collection="vectors",
            index_name="vectors-index",
            manage_index=False,
            metadata_field="metadata",
            embedding_field="embedding",
            search_stage="vector_search",
            candidate_multiplier=1,
        )

        records = (
            VectorRecord(id="doc-1", values=(0.5, 0.6), metadata={"category": "docs"}),
        )

        await adapter.upsert_vectors("tenant", records)
        query = await adapter.query_vectors(
            "tenant",
            (0.5, 0.6),
            top_k=1,
            metadata_filter={"category": "docs"},
            include_values=True,
        )

        assert query.matches and query.matches[0].metadata["category"] == "docs"
        assert query.matches[0].values == records[0].values

        delete = await adapter.delete_namespace("tenant")
        assert delete.deleted is True

    asyncio.run(_run())


def test_faiss_adapter_persists_state(tmp_path) -> None:
    async def _run() -> None:
        index_path = tmp_path / "faiss.json"
        adapter = FaissVectorStoreAdapter(index_path=str(index_path), metric="cosine")

        records = (
            VectorRecord(id="doc-1", values=(0.7, 0.8), metadata={"topic": "ai"}),
        )
        await adapter.upsert_vectors("tenant", records)

        query = await adapter.query_vectors(
            "tenant",
            (0.7, 0.8),
            top_k=1,
            metadata_filter=None,
            include_values=True,
        )
        assert query.matches and query.matches[0].values == records[0].values

        delete = await adapter.delete_namespace("tenant")
        assert delete.deleted is True

        adapter_reloaded = FaissVectorStoreAdapter(index_path=str(index_path), metric="cosine")
        query_empty = await adapter_reloaded.query_vectors(
            "tenant",
            (0.7, 0.8),
            top_k=1,
            metadata_filter=None,
            include_values=False,
        )
        assert not query_empty.matches

    asyncio.run(_run())
