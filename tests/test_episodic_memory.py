from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TSVECTOR, UUID
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import sessionmaker

from modules.conversation_store import Base, ConversationStoreRepository
from modules.Tools.Base_Tools.memory_episodic import EpisodicMemoryTool

TENANT = "episodic-test"


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
    engine = create_engine(f"sqlite:///{tmp_path / 'episodic.sqlite'}", future=True)
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


def test_append_and_query_episodic_memory(repository):
    conversation_id = uuid.uuid4()
    base_time = datetime.now(timezone.utc)

    stored = repository.append_episodic_memory(
        tenant_id=TENANT,
        content={"summary": "met project stakeholder"},
        tags=["milestone", "Project"],
        metadata={"mood": "optimistic"},
        occurred_at=base_time,
        conversation_id=conversation_id,
    )

    assert stored["tenant_id"] == TENANT
    assert stored["tags"] == ["milestone", "Project"]
    assert stored["metadata"]["mood"] == "optimistic"

    results = repository.query_episodic_memories(
        tenant_id=TENANT,
        tags_all=["milestone"],
        from_time=base_time - timedelta(minutes=1),
        to_time=base_time + timedelta(minutes=1),
    )

    assert len(results) == 1
    assert results[0]["conversation_id"] == str(conversation_id)


def test_prune_respects_filters(repository):
    now = datetime.now(timezone.utc)
    repository.append_episodic_memory(
        tenant_id=TENANT,
        content={"summary": "early note"},
        tags=["note"],
        occurred_at=now - timedelta(days=3),
    )
    repository.append_episodic_memory(
        tenant_id=TENANT,
        content={"summary": "recent note"},
        tags=["note"],
        occurred_at=now - timedelta(hours=1),
        expires_at=now + timedelta(days=1),
    )
    repository.append_episodic_memory(
        tenant_id=TENANT,
        content={"summary": "expired"},
        tags=["note"],
        occurred_at=now - timedelta(days=1),
        expires_at=now - timedelta(hours=1),
    )

    removed_expired = repository.prune_episodic_memories(
        tenant_id=TENANT,
        expired_only=True,
    )
    assert removed_expired == 1

    removed_before = repository.prune_episodic_memories(
        tenant_id=TENANT,
        before=now - timedelta(days=2),
    )
    assert removed_before == 1

    remaining = repository.query_episodic_memories(tenant_id=TENANT)
    assert len(remaining) == 1
    assert remaining[0]["content"]["summary"] == "recent note"


def test_tool_roundtrip(repository):
    async def _run() -> None:
        tool = EpisodicMemoryTool(repository=repository)
        base_time = datetime.now(timezone.utc)

        stored = await tool.store(
            tenant_id=TENANT,
            content={"summary": "async entry"},
            tags=["async"],
            occurred_at=base_time,
        )
        assert stored["tags"] == ["async"]

        queried = await tool.query(
            tenant_id=TENANT,
            tags_any=["async"],
        )
        assert queried["count"] == 1
        assert queried["episodes"][0]["content"]["summary"] == "async entry"

        pruned = await tool.prune(tenant_id=TENANT)
        assert pruned["deleted"] >= 1

        remaining = repository.query_episodic_memories(tenant_id=TENANT)
        assert remaining == []

    asyncio.run(_run())
