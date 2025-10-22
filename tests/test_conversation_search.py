from __future__ import annotations

import uuid
from typing import Any, Iterable, Tuple

import pytest

sqlalchemy = pytest.importorskip("sqlalchemy")
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from modules.conversation_store import Base, ConversationStoreRepository


TENANT = "search-tenant"


@pytest.fixture
def postgres_repository(postgresql) -> Iterable[Tuple[ConversationStoreRepository, Any]]:
    engine = create_engine(postgresql.dsn(), future=True)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, future=True)
    repository = ConversationStoreRepository(factory)
    repository.create_schema()
    try:
        yield repository, engine
    finally:
        engine.dispose()


@pytest.fixture
def sqlite_repository(tmp_path) -> Iterable[ConversationStoreRepository]:
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'conversation.sqlite'}", future=True)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, future=True)
    repository = ConversationStoreRepository(factory)
    repository.create_schema()
    try:
        yield repository
    finally:
        engine.dispose()


def _seed_messages(repository: ConversationStoreRepository, conversation_id: uuid.UUID) -> str:
    for idx in range(250):
        repository.add_message(
            conversation_id,
            tenant_id=TENANT,
            role="user",
            content={"text": f"filler message {idx}"},
            metadata={"ordinal": idx},
        )
    match = repository.add_message(
        conversation_id,
        tenant_id=TENANT,
        role="assistant",
        content={"text": "This entry hides a synthetic needle in plain sight."},
        metadata={"needle": True},
    )
    return match["id"]


@pytest.mark.parametrize("order", ["asc", "desc"])
def test_full_text_query_uses_index(postgres_repository, order: str) -> None:
    repository, engine = postgres_repository
    conversation_id = uuid.uuid4()
    repository.ensure_conversation(conversation_id, tenant_id=TENANT)
    message_id = _seed_messages(repository, conversation_id)

    results = list(
        repository.query_messages_by_text(
            conversation_ids=[conversation_id],
            tenant_id=TENANT,
            text="needle",
            order=order,
        )
    )
    assert results
    assert any(item["id"] == message_id for item in results)

    with engine.begin() as connection:
        connection.execute(text("ANALYZE messages"))
        plan_rows = connection.execute(
            text(
                """
                EXPLAIN (FORMAT TEXT)
                SELECT id
                  FROM messages
                 WHERE tenant_id = :tenant
                   AND conversation_id = :conversation
                   AND message_text_tsv @@ plainto_tsquery('simple', :term)
                ORDER BY created_at DESC
                LIMIT 5
                """
            ),
            {"tenant": TENANT, "conversation": str(conversation_id), "term": "needle"},
        ).scalars().all()

    plan_text = " ".join(plan_rows)
    assert "ix_messages_message_text_tsv" in plan_text


def test_sqlite_search_falls_back_to_ilike(sqlite_repository) -> None:
    repository = sqlite_repository
    conversation_id = uuid.uuid4()
    repository.ensure_conversation(conversation_id, tenant_id=TENANT)
    repository.add_message(
        conversation_id,
        tenant_id=TENANT,
        role="user",
        content={"text": "A greeting from sqlite"},
        metadata={"source": "sqlite"},
    )
    repository.add_message(
        conversation_id,
        tenant_id=TENANT,
        role="assistant",
        content={"text": "Something entirely unrelated"},
        metadata={},
    )

    results = list(
        repository.query_messages_by_text(
            conversation_ids=[conversation_id],
            tenant_id=TENANT,
            text="greeting",
        )
    )

    assert len(results) == 1
    assert results[0]["metadata"]["source"] == "sqlite"
