from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

from modules.conversation_store import Base, ConversationStoreRepository
from modules.conversation_store.models import Message


@pytest.fixture
def engine(postgresql):
    engine = create_engine(postgresql.dsn(), future=True)
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def repository(engine):
    factory = sessionmaker(bind=engine, future=True)
    repo = ConversationStoreRepository(factory)
    repo.create_schema()
    return repo


def _uuid() -> uuid.UUID:
    return uuid.uuid4()


def test_schema_contains_expected_tables(engine):
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    assert {
        "users",
        "sessions",
        "conversations",
        "messages",
        "message_assets",
        "message_vectors",
        "message_events",
    }.issubset(tables)


def test_recent_messages_are_returned_in_chronological_order(repository):
    conversation_id = _uuid()
    repository.ensure_conversation(conversation_id)
    base_time = datetime.now(timezone.utc)
    for idx in range(5):
        repository.add_message(
            conversation_id,
            role="user" if idx % 2 else "assistant",
            content={"text": f"message-{idx}"},
            timestamp=(base_time + timedelta(seconds=idx)).isoformat(),
            metadata={"idx": idx},
            message_id=f"m-{idx}",
        )

    messages = repository.load_recent_messages(conversation_id, limit=5)
    assert [msg["metadata"]["idx"] for msg in messages] == list(range(5))


def test_idempotent_inserts(repository, engine):
    conversation_id = _uuid()
    repository.ensure_conversation(conversation_id)
    repository.add_message(
        conversation_id,
        role="user",
        content={"text": "hello"},
        metadata={},
        message_id="duplicate",
    )
    repository.add_message(
        conversation_id,
        role="user",
        content={"text": "hello"},
        metadata={},
        message_id="duplicate",
    )

    messages = repository.load_recent_messages(conversation_id)
    assert len(messages) == 1

    with sessionmaker(bind=engine, future=True)() as session:
        stored = session.query(Message).filter_by(conversation_id=conversation_id).all()
        assert len(stored) == 1


def test_delete_workflows(repository, engine):
    conversation_id = _uuid()
    repository.ensure_conversation(conversation_id)
    stored = repository.add_message(
        conversation_id,
        role="assistant",
        content={"text": "to delete"},
        metadata={},
    )

    repository.soft_delete_message(conversation_id, stored["id"], reason="cleanup")

    with sessionmaker(bind=engine, future=True)() as session:
        message = session.get(Message, uuid.UUID(stored["id"]))
        assert message.deleted_at is not None

    repository.hard_delete_conversation(conversation_id)

    with sessionmaker(bind=engine, future=True)() as session:
        assert session.get(Message, uuid.UUID(stored["id"])) is None
