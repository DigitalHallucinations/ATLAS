import asyncio
from datetime import datetime, timezone

import pytest

try:  # pragma: no cover - optional SQLAlchemy dependency in some environments
    from sqlalchemy import create_engine
    from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TSVECTOR, UUID
    from sqlalchemy.ext.compiler import compiles
    from sqlalchemy.orm import sessionmaker
except Exception:  # pragma: no cover - skip tests when SQLAlchemy unavailable
    pytest.skip("SQLAlchemy is required for memory graph tests", allow_module_level=True)

from modules.conversation_store import Base, ConversationStoreRepository
from modules.Tools.Base_Tools.memory_graph import MemoryGraphTool

TENANT = "graph-test"


@compiles(JSONB, "sqlite")
def _compile_jsonb(_type, compiler, **_kwargs):
    return "JSON"


@compiles(UUID, "sqlite")
def _compile_uuid(_type, compiler, **_kwargs):
    return "CHAR(36)"


@compiles(ARRAY, "sqlite")
def _compile_array(_type, compiler, **_kwargs):
    return "BLOB"


@compiles(TSVECTOR, "sqlite")
def _compile_tsvector(_type, compiler, **_kwargs):
    return "TEXT"


@pytest.fixture
def repository(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'graph.sqlite'}", future=True)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, future=True)
    if getattr(factory, "bind", None) is None:  # SQLAlchemy 2.x compatibility
        factory.bind = engine  # type: ignore[attr-defined]
    repo = ConversationStoreRepository(factory)
    repo.create_schema()
    try:
        yield repo
    finally:
        engine.dispose()


def test_graph_repository_operations(repository):
    nodes = repository.upsert_graph_nodes(
        tenant_id=TENANT,
        nodes=[
            {"key": "alpha", "label": "Alpha", "type": "person", "metadata": {"role": "lead"}},
            {"key": "beta", "metadata": {"score": 1}},
        ],
    )
    assert {node["key"] for node in nodes} == {"alpha", "beta"}

    edges = repository.upsert_graph_edges(
        tenant_id=TENANT,
        edges=[
            {
                "source": "alpha",
                "target": "beta",
                "edge_type": "knows",
                "weight": 0.75,
                "metadata": {"since": datetime.now(timezone.utc).isoformat()},
            }
        ],
    )
    assert len(edges) == 1
    assert edges[0]["edge_type"] == "knows"

    subgraph = repository.query_graph(tenant_id=TENANT, node_keys=["alpha"])
    assert len(subgraph["nodes"]) == 2
    assert len(subgraph["edges"]) == 1
    assert subgraph["edges"][0]["source_key"] == "alpha"

    removed_edges = repository.delete_graph_entries(
        tenant_id=TENANT,
        edge_ids=[edges[0]["id"]],
    )
    assert removed_edges["edges"] == 1
    assert repository.query_graph(tenant_id=TENANT)["edges"] == []

    removed_nodes = repository.delete_graph_entries(
        tenant_id=TENANT,
        node_keys=["alpha", "beta"],
    )
    assert removed_nodes["nodes"] == 2
    assert repository.query_graph(tenant_id=TENANT)["nodes"] == []


def test_memory_graph_tool_roundtrip(repository):
    async def _exercise() -> None:
        tool = MemoryGraphTool(repository=repository)

        upserted = await tool.run(
            operation="upsert_nodes",
            tenant_id=TENANT,
            nodes=[
                {"key": "gamma", "label": "Gamma"},
                {"key": "delta", "metadata": {"status": "active"}},
            ],
        )
        assert upserted["operation"] == "upsert_nodes"
        assert {node["key"] for node in upserted["nodes"]} == {"gamma", "delta"}

        edge_result = await tool.run(
            operation="upsert_edges",
            tenant_id=TENANT,
            edges=[{"source": "gamma", "target": "delta", "edge_type": "related"}],
        )
        assert edge_result["operation"] == "upsert_edges"
        assert edge_result["edges"][0]["edge_type"] == "related"

        query = await tool.run(
            operation="query",
            tenant_id=TENANT,
            node_keys=["gamma"],
        )
        assert query["operation"] == "query"
        assert any(node["key"] == "delta" for node in query["nodes"])
        assert len(query["edges"]) == 1

        removal = await tool.run(
            operation="remove",
            tenant_id=TENANT,
            node_keys=["delta"],
        )
        assert removal["operation"] == "remove"
        assert removal["nodes"] == 1

        remaining = repository.query_graph(tenant_id=TENANT)
        assert all(node["key"] != "delta" for node in remaining["nodes"])

    asyncio.run(_exercise())
