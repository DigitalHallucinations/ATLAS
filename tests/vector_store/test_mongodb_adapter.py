"""Unit tests for the MongoDB vector store adapter."""

from __future__ import annotations

import asyncio
import math
import os
from dataclasses import dataclass
from typing import Any, Coroutine, Dict, Iterable, Iterator, Mapping, MutableMapping, Optional

import pytest

from modules.Tools.Base_Tools.vector_store import VectorRecord
from modules.Tools.providers.vector_store import mongodb as mongo_module

USE_STUBS = os.environ.get("ATLAS_TEST_USE_MONGO_STUBS", "0").lower() in {"1", "true", "yes"}

if mongo_module.UpdateOne is None and not USE_STUBS:
    pytest.skip("PyMongo is required for MongoDB adapter tests.", allow_module_level=True)


class _UpdateOneStub:
    """Minimal stand-in for :class:`pymongo.operations.UpdateOne`."""

    def __init__(self, filter: Mapping[str, Any], update: Mapping[str, Any], *, upsert: bool) -> None:
        self.filter = dict(filter)
        self.update = dict(update)
        self.upsert = bool(upsert)


if mongo_module.UpdateOne is None:
    mongo_module.UpdateOne = _UpdateOneStub  # type: ignore[assignment]


@dataclass
class _DeleteResultStub:
    deleted_count: int


class _FakeCollection:
    """In-memory collection that mimics the subset of PyMongo used by the adapter."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._database: Optional["_FakeDatabase"] = None
        self._documents: Dict[str, Dict[str, Any]] = {}
        self._vector_indexes: MutableMapping[str, Mapping[str, Any]] = {}
        self._atlas_indexes: Dict[str, Mapping[str, Any]] = {}
        self.embedding_field = "embedding"
        self.metadata_field = "metadata"

    @property
    def database(self) -> "_FakeDatabase":
        assert self._database is not None, "Collection has not been bound to a database."
        return self._database

    def bind(self, database: "_FakeDatabase") -> None:
        self._database = database

    # Index helpers -----------------------------------------------------
    def index_information(self) -> Mapping[str, Any]:
        return dict(self._vector_indexes)

    def drop_index(self, name: str) -> None:
        self._vector_indexes.pop(name, None)

    def list_search_indexes(self, name: Optional[str] = None) -> Iterable[Mapping[str, Any]]:
        for index_name, payload in self._atlas_indexes.items():
            if name is None or index_name == name:
                yield {"name": index_name, **payload}

    # Write helpers -----------------------------------------------------
    def bulk_write(self, operations: Iterable[Any], ordered: bool = False) -> None:  # noqa: ARG002 - parity with PyMongo
        for op in operations:
            if not isinstance(op, _UpdateOneStub):  # pragma: no cover - guards unexpected subclasses
                raise AssertionError("Unexpected bulk operation type")
            self._apply_update(op)

    def _apply_update(self, operation: _UpdateOneStub) -> None:
        filter_doc = operation.filter
        update_doc = operation.update
        identifier = str(filter_doc.get("_id"))
        namespace = filter_doc.get("namespace")

        existing = self._documents.get(identifier)
        matches_namespace = existing is not None and existing.get("namespace") == namespace

        payload = dict(update_doc.get("$set", {}))
        payload.setdefault("namespace", namespace)
        payload.setdefault("_id", identifier)

        for field, value in payload.items():
            if field == "namespace":
                continue
            if isinstance(value, list):
                self.embedding_field = field
            if isinstance(value, dict) and field != "_id":
                self.metadata_field = field

        if existing and matches_namespace:
            existing.update(payload)
        elif operation.upsert:
            seeded = dict(update_doc.get("$setOnInsert", {}))
            seeded.update(payload)
            self._documents[identifier] = seeded
        else:  # pragma: no cover - defensive guard
            raise AssertionError("Non-upsert update attempted in test stub")

    # Query helpers -----------------------------------------------------
    def aggregate(self, pipeline: Iterable[Mapping[str, Any]]) -> Iterator[Mapping[str, Any]]:
        pipeline_list = list(pipeline)
        if not pipeline_list:
            return iter(())

        query_vector: Optional[list[float]] = None
        limit = len(self._documents)
        match_filter: Mapping[str, Any] = {}

        first_stage = pipeline_list[0]
        if "$vectorSearch" in first_stage:
            stage = first_stage["$vectorSearch"]
            query_vector = [float(v) for v in stage.get("queryVector", [])]
            limit = int(stage.get("limit", limit))
        elif "$search" in first_stage:
            stage = first_stage["$search"].get("knnBeta", {})
            query_vector = [float(v) for v in stage.get("vector", [])]
            limit = int(stage.get("k", limit))

        for entry in pipeline_list:
            if "$match" in entry:
                match_filter = entry["$match"]
            if "$limit" in entry:
                limit = int(entry["$limit"]) or limit

        scored = []
        for document in self._documents.values():
            if not self._matches(document, match_filter):
                continue
            embedding = document.get(self.embedding_field, [])
            if query_vector is None or not embedding:
                similarity = 0.0
            else:
                similarity = self._cosine_similarity(query_vector, embedding)
            scored.append(
                {
                    "_id": document.get("_id"),
                    "score": similarity,
                    self.embedding_field: list(embedding),
                    self.metadata_field: dict(document.get(self.metadata_field, {})),
                }
            )

        scored.sort(key=lambda entry: entry.get("score", 0.0), reverse=True)
        return iter(scored[:limit])

    def _matches(self, document: Mapping[str, Any], match_filter: Mapping[str, Any]) -> bool:
        for key, expected in match_filter.items():
            if key == "namespace":
                if document.get("namespace") != expected:
                    return False
            elif key.startswith(f"{self.metadata_field}."):
                metadata_key = key[len(self.metadata_field) + 1 :]
                metadata = document.get(self.metadata_field, {})
                if metadata.get(metadata_key) != expected:
                    return False
            else:
                if document.get(key) != expected:
                    return False
        return True

    @staticmethod
    def _cosine_similarity(lhs: Iterable[float], rhs: Iterable[float]) -> float:
        left = list(lhs)
        right = list(rhs)
        numerator = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(a * a for a in left))
        right_norm = math.sqrt(sum(b * b for b in right))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return numerator / (left_norm * right_norm)

    # Delete helpers ----------------------------------------------------
    def delete_many(self, filter: Mapping[str, Any]) -> _DeleteResultStub:
        namespace = filter.get("namespace")
        to_delete = [key for key, value in self._documents.items() if value.get("namespace") == namespace]
        for key in to_delete:
            del self._documents[key]
        return _DeleteResultStub(deleted_count=len(to_delete))


class _FakeDatabase:
    def __init__(self, name: str) -> None:
        self.name = name
        self._collections: Dict[str, _FakeCollection] = {}

    def __getitem__(self, collection_name: str) -> _FakeCollection:
        collection = self._collections.get(collection_name)
        if collection is None:
            collection = _FakeCollection(collection_name)
            collection.bind(self)
            self._collections[collection_name] = collection
        return collection

    def command(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        if "createIndexes" in payload:
            collection = self[payload["createIndexes"]]
            for entry in payload.get("indexes", []):
                name = entry.get("name")
                if isinstance(name, str):
                    collection._vector_indexes[name] = dict(entry)
            return {"ok": 1}
        if "createSearchIndexes" in payload:
            collection = self[payload["createSearchIndexes"]]
            for entry in payload.get("indexes", []):
                name = entry.get("name")
                if isinstance(name, str):
                    collection._atlas_indexes[name] = dict(entry)
            return {"ok": 1}
        if "dropSearchIndex" in payload:
            collection = self[payload["dropSearchIndex"]]
            name = payload.get("name")
            if isinstance(name, str):
                collection._atlas_indexes.pop(name, None)
            return {"ok": 1}
        return {"ok": 1}


class _FakeClient:
    def __init__(self) -> None:
        self._databases: Dict[str, _FakeDatabase] = {}

    def __getitem__(self, name: str) -> _FakeDatabase:
        database = self._databases.get(name)
        if database is None:
            database = _FakeDatabase(name)
            self._databases[name] = database
        return database


class _MongoFixture:
    def __init__(self) -> None:
        self.client = _FakeClient()

    def create_client(self) -> _FakeClient:
        return self.client

    @property
    def collection(self) -> _FakeCollection:
        return self.client["atlas_vector_store"]["embeddings"]


@pytest.fixture()
def mongodb_adapter() -> mongo_module.MongoDBVectorStoreAdapter:
    fixture = _MongoFixture()
    adapter = mongo_module.MongoDBVectorStoreAdapter(
        client_factory=fixture.create_client,
        database="atlas_vector_store",
        collection="embeddings",
        index_name="vector_index",
        manage_index=True,
        index_type="vector_search",
        search_stage="vector_search",
    )
    adapter._test_fixture = fixture  # type: ignore[attr-defined]  # aid assertions in tests
    return adapter


def _get_fixture(adapter: mongo_module.MongoDBVectorStoreAdapter) -> _MongoFixture:
    fixture = getattr(adapter, "_test_fixture", None)
    assert isinstance(fixture, _MongoFixture)
    return fixture


def _run(coro: Coroutine[Any, Any, Any]) -> Any:
    return asyncio.run(coro)


def test_upsert_and_query_vectors(mongodb_adapter: mongo_module.MongoDBVectorStoreAdapter) -> None:
    fixture = _get_fixture(mongodb_adapter)

    first = VectorRecord(id="doc-1", values=(0.3, 0.4, 0.5), metadata={"topic": "alpha"})
    second = VectorRecord(id="doc-2", values=(0.2, 0.1, 0.0), metadata={"topic": "beta"})

    response = _run(mongodb_adapter.upsert_vectors("tenant-a", (first, second)))

    assert response.namespace == "tenant-a"
    assert response.upserted_count == 2
    assert fixture.collection.index_information()  # index provisioning executed

    stored = fixture.collection._documents
    assert stored["doc-1"][fixture.collection.metadata_field]["topic"] == "alpha"

    query = _run(
        mongodb_adapter.query_vectors(
            "tenant-a",
            (0.3, 0.4, 0.5),
            top_k=2,
            metadata_filter=None,
            include_values=True,
        )
    )

    assert query.top_k == 2
    assert [match.id for match in query.matches] == ["doc-1", "doc-2"]
    assert query.matches[0].values == first.values

    filtered = _run(
        mongodb_adapter.query_vectors(
            "tenant-a",
            (0.1, 0.1, 0.0),
            top_k=1,
            metadata_filter={"topic": "beta"},
            include_values=False,
        )
    )

    assert len(filtered.matches) == 1
    assert filtered.matches[0].id == "doc-2"
    assert filtered.matches[0].values is None


def test_delete_namespace_removes_documents(mongodb_adapter: mongo_module.MongoDBVectorStoreAdapter) -> None:
    fixture = _get_fixture(mongodb_adapter)

    record = VectorRecord(id="doc-3", values=(1.0, 0.0, 0.0), metadata={})
    _run(mongodb_adapter.upsert_vectors("tenant-b", (record,)))

    assert "doc-3" in fixture.collection._documents

    response = _run(mongodb_adapter.delete_namespace("tenant-b"))
    assert response.deleted is True
    assert response.namespace == "tenant-b"
    assert "doc-3" not in fixture.collection._documents
