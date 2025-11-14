import asyncio
import math
from pathlib import Path

import pytest

from tests.test_vector_store_tool import (
    ROOT,
    _ensure_package,
    _ensure_stub_modules,
    _load_module,
)


_ensure_stub_modules()

_ensure_package("modules", ROOT / "modules")
_ensure_package("modules.Tools", ROOT / "modules" / "Tools")
_ensure_package("modules.Tools.Base_Tools", ROOT / "modules" / "Tools" / "Base_Tools")
_ensure_package("modules.Tools.providers", ROOT / "modules" / "Tools" / "providers")
_ensure_package("modules.Tools.providers.vector_store", ROOT / "modules" / "Tools" / "providers" / "vector_store")

vector_store_module = _load_module(
    "modules.Tools.Base_Tools.vector_store",
    ROOT / "modules" / "Tools" / "Base_Tools" / "vector_store.py",
)
_load_module(
    "modules.Tools.providers.vector_store.chroma",
    ROOT / "modules" / "Tools" / "providers" / "vector_store" / "chroma.py",
)
_load_module(
    "modules.Tools.providers.vector_store.faiss",
    ROOT / "modules" / "Tools" / "providers" / "vector_store" / "faiss.py",
)
_load_module(
    "modules.Tools.providers.vector_store.pinecone",
    ROOT / "modules" / "Tools" / "providers" / "vector_store" / "pinecone.py",
)


def _matches_filter(candidate, expected):
    for key, value in expected.items():
        if key not in candidate:
            return False
        candidate_value = candidate[key]
        if isinstance(value, dict) and isinstance(candidate_value, dict):
            if not _matches_filter(candidate_value, value):
                return False
            continue
        if candidate_value != value:
            return False
    return True


class _FakeChromaCollection:
    def __init__(self) -> None:
        self.records: dict[str, tuple[tuple[float, ...], dict[str, object]]] = {}

    def upsert(self, ids, embeddings, metadatas):
        for vector_id, values, metadata in zip(ids, embeddings, metadatas):
            stored_meta = dict(metadata) if isinstance(metadata, dict) else {}
            self.records[str(vector_id)] = (tuple(float(v) for v in values), stored_meta)

    def query(self, query_embeddings, n_results, where=None, include=None):
        query_vector = tuple(float(v) for v in query_embeddings[0])
        scored = []
        for vector_id, (values, metadata) in self.records.items():
            if where and not _matches_filter(metadata, where):
                continue
            score = _cosine_similarity(values, query_vector)
            scored.append((vector_id, score, values, metadata))
        scored.sort(key=lambda item: (-item[1], item[0]))
        scored = scored[:n_results]
        ids = [[item[0] for item in scored]]
        distances = [[1.0 - item[1] for item in scored]]
        metadatas = [[item[3] for item in scored]]
        response = {"ids": ids, "distances": distances, "metadatas": metadatas}
        if include and "embeddings" in include:
            response["embeddings"] = [[list(item[2]) for item in scored]]
        return response


class _FakeChromaClient:
    def __init__(self) -> None:
        self.collections: dict[str, _FakeChromaCollection] = {}

    def get_or_create_collection(self, name, metadata=None):
        return self.collections.setdefault(name, _FakeChromaCollection())

    def delete_collection(self, name):
        if name in self.collections:
            del self.collections[name]
        else:
            raise KeyError(name)


def _cosine_similarity(lhs, rhs):
    lhs_norm = math.sqrt(sum(value * value for value in lhs))
    rhs_norm = math.sqrt(sum(value * value for value in rhs))
    if lhs_norm == 0 or rhs_norm == 0:
        return 0.0
    dot = sum(a * b for a, b in zip(lhs, rhs))
    return dot / (lhs_norm * rhs_norm)


class _FakePineconeIndex:
    def __init__(self) -> None:
        self._store: dict[str, dict[str, dict[str, object]]] = {}

    def upsert(self, vectors, namespace):
        space = self._store.setdefault(namespace, {})
        for entry in vectors:
            space[entry["id"]] = {
                "values": tuple(float(v) for v in entry.get("values", ())),
                "metadata": dict(entry.get("metadata", {})),
            }

    def query(self, *, namespace, vector, top_k, filter=None, include_values=False, include_metadata=True):
        query_vector = tuple(float(v) for v in vector)
        matches = []
        space = self._store.get(namespace, {})
        for vector_id, record in space.items():
            metadata = record.get("metadata", {})
            if filter and not _matches_filter(metadata, filter):
                continue
            values = record.get("values", ())
            score = _cosine_similarity(values, query_vector)
            payload = {"id": vector_id, "score": score}
            if include_metadata:
                payload["metadata"] = metadata
            if include_values:
                payload["values"] = list(values)
            matches.append(payload)
        matches.sort(key=lambda item: (-item["score"], item["id"]))
        return {"matches": matches[:top_k]}

    def delete(self, *, namespace, delete_all=False):  # noqa: ARG002 - parity with Pinecone
        if namespace in self._store:
            del self._store[namespace]


class _FakePineconeClient:
    def __init__(self) -> None:
        self._index = _FakePineconeIndex()

    def Index(self, name):  # noqa: N802 - mimics Pinecone API
        return self._index


def test_chroma_adapter_roundtrip() -> None:
    async def _run() -> None:
        service = vector_store_module.build_vector_store_service(
            adapter_name="chroma",
            adapter_config={
                "client": _FakeChromaClient(),
                "collection_name": "atlas",
                "metric": "cosine",
                "namespace_separator": "::",
            },
        )

        upsert = await service.upsert_vectors(
            namespace="workspace",
            vectors=[
                {"id": "doc-1", "values": [0.1, 0.2, 0.3], "metadata": {"kind": "note"}},
                {"id": "doc-2", "values": [0.2, 0.3, 0.4], "metadata": {"kind": "note"}},
            ],
        )
        assert upsert["ids"] == ["doc-1", "doc-2"]

        query = await service.query_vectors(
            namespace="workspace",
            query=[0.1, 0.2, 0.3],
            top_k=2,
            filter={"kind": "note"},
            include_values=True,
        )
        assert query["namespace"] == "workspace"
        assert len(query["matches"]) == 2
        assert [match["id"] for match in query["matches"]] == ["doc-1", "doc-2"]

        deleted = await service.delete_namespace(namespace="workspace")
        assert deleted == {
            "namespace": "workspace",
            "removed_ids": ["doc-1", "doc-2"],
            "deleted": True,
        }
        deleted_again = await service.delete_namespace(namespace="workspace")
        assert deleted_again == {
            "namespace": "workspace",
            "removed_ids": [],
            "deleted": False,
        }

    asyncio.run(_run())


def test_faiss_adapter_persistence(tmp_path: Path) -> None:
    async def _run() -> None:
        index_path = tmp_path / "faiss-state.json"
        service = vector_store_module.build_vector_store_service(
            adapter_name="faiss",
            adapter_config={
                "index_path": str(index_path),
                "metric": "cosine",
            },
        )

        await service.upsert_vectors(
            namespace="workspace",
            vectors=[{"id": "doc-1", "values": [0.2, 0.1, 0.3], "metadata": {"category": "alpha"}}],
        )
        assert index_path.exists()

        service_reloaded = vector_store_module.build_vector_store_service(
            adapter_name="faiss",
            adapter_config={
                "index_path": str(index_path),
                "metric": "cosine",
            },
        )

        query = await service_reloaded.query_vectors(
            namespace="workspace",
            query=[0.2, 0.1, 0.3],
            top_k=1,
            filter={"category": "alpha"},
            include_values=True,
        )
        assert query["matches"][0]["id"] == "doc-1"

        deleted = await service_reloaded.delete_namespace(namespace="workspace")
        assert deleted == {
            "namespace": "workspace",
            "removed_ids": ["doc-1"],
            "deleted": True,
        }

    asyncio.run(_run())


def test_pinecone_adapter_roundtrip() -> None:
    async def _run() -> None:
        client = _FakePineconeClient()
        service = vector_store_module.build_vector_store_service(
            adapter_name="pinecone",
            adapter_config={
                "client": client,
                "index_name": "atlas",
                "namespace_prefix": "tenant-",
            },
        )

        await service.upsert_vectors(
            namespace="workspace",
            vectors=[{"id": "doc-1", "values": [0.0, 0.1], "metadata": {"team": "search"}}],
        )
        query = await service.query_vectors(
            namespace="workspace",
            query=[0.0, 0.1],
            top_k=1,
            filter={"team": "search"},
            include_values=True,
        )
        assert query["matches"][0]["id"] == "doc-1"
        assert query["matches"][0]["metadata"]["team"] == "search"

        deleted = await service.delete_namespace(namespace="workspace")
        assert deleted == {
            "namespace": "workspace",
            "removed_ids": ["doc-1"],
            "deleted": True,
        }
        deleted_again = await service.delete_namespace(namespace="workspace")
        assert deleted_again == {
            "namespace": "workspace",
            "removed_ids": [],
            "deleted": False,
        }

    asyncio.run(_run())
