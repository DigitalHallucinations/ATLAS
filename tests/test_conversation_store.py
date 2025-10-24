from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine, inspect, select
from sqlalchemy.orm import sessionmaker

from modules.conversation_store import Base, ConversationStoreRepository
from modules.conversation_store.models import Message, Session, User, Conversation


TENANT = "test-tenant"


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
    repository.ensure_conversation(conversation_id, tenant_id=TENANT)
    base_time = datetime.now(timezone.utc)
    for idx in range(5):
        repository.add_message(
            conversation_id,
            tenant_id=TENANT,
            role="user" if idx % 2 else "assistant",
            content={"text": f"message-{idx}"},
            timestamp=(base_time + timedelta(seconds=idx)).isoformat(),
            metadata={"idx": idx},
            message_id=f"m-{idx}",
        )

    messages = repository.load_recent_messages(
        conversation_id, tenant_id=TENANT, limit=5
    )
    assert [msg["metadata"]["idx"] for msg in messages] == list(range(5))
    assert all(msg["message_type"] == "text" for msg in messages)
    assert all(msg["status"] == "sent" for msg in messages)


def test_idempotent_inserts(repository, engine):
    conversation_id = _uuid()
    repository.ensure_conversation(conversation_id, tenant_id=TENANT)
    repository.add_message(
        conversation_id,
        tenant_id=TENANT,
        role="user",
        content={"text": "hello"},
        metadata={},
        message_id="duplicate",
    )
    repository.add_message(
        conversation_id,
        tenant_id=TENANT,
        role="user",
        content={"text": "hello"},
        metadata={},
        message_id="duplicate",
    )

    messages = repository.load_recent_messages(conversation_id, tenant_id=TENANT)
    assert len(messages) == 1

    with sessionmaker(bind=engine, future=True)() as session:
        stored = session.query(Message).filter_by(conversation_id=conversation_id).all()
        assert len(stored) == 1


def test_iso_z_timestamp_is_normalised(repository):
    conversation_id = _uuid()
    repository.ensure_conversation(conversation_id, tenant_id=TENANT)

    message = repository.add_message(
        conversation_id,
        tenant_id=TENANT,
        role="assistant",
        content={"text": "timestamp"},
        metadata={},
        timestamp=" 2024-05-21T09:00:00Z ",
    )

    assert message["created_at"] == "2024-05-21T09:00:00+00:00"
    assert message["timestamp"] == "2024-05-21 09:00:00"


def test_user_and_session_identity_reuse(repository, engine):
    conversation_id = _uuid()
    repository.ensure_conversation(conversation_id, tenant_id=TENANT)

    first = repository.add_message(
        conversation_id,
        tenant_id=TENANT,
        role="user",
        content={"text": "hello"},
        metadata={"source": "unit"},
        user="tester",
        user_display_name="Test User",
        session="session-xyz",
        session_metadata={"ip": "127.0.0.1"},
    )

    second = repository.add_message(
        conversation_id,
        tenant_id=TENANT,
        role="assistant",
        content={"text": "response"},
        metadata={},
        user="tester",
        session="session-xyz",
    )

    assert first["user_id"] == second["user_id"]
    assert first["session_id"] == second["session_id"]

    with sessionmaker(bind=engine, future=True)() as db_session:
        user_row = db_session.execute(
            select(User).where(User.external_id == "tester")
        ).scalar_one()
        session_row = db_session.execute(
            select(Session).where(Session.external_id == "session-xyz")
        ).scalar_one()
        message_rows = (
            db_session.query(Message)
            .filter_by(conversation_id=conversation_id)
            .order_by(Message.created_at)
            .all()
        )
        conversation_row = db_session.get(Conversation, conversation_id)

    assert str(user_row.id) == first["user_id"]
    assert user_row.display_name == "Test User"
    assert str(session_row.id) == first["session_id"]
    assert session_row.user_id == user_row.id
    assert conversation_row is not None
    assert conversation_row.session_id == session_row.id
    assert {message.user_id for message in message_rows} == {user_row.id}


def test_delete_workflows(repository, engine):
    conversation_id = _uuid()
    repository.ensure_conversation(conversation_id, tenant_id=TENANT)
    stored = repository.add_message(
        conversation_id,
        tenant_id=TENANT,
        role="assistant",
        content={"text": "to delete"},
        metadata={},
    )

    repository.soft_delete_message(
        conversation_id, stored["id"], tenant_id=TENANT, reason="cleanup"
    )

    with sessionmaker(bind=engine, future=True)() as session:
        message = session.get(Message, uuid.UUID(stored["id"]))
        assert message.deleted_at is not None
        assert message.status == "deleted"

    repository.hard_delete_conversation(conversation_id, tenant_id=TENANT)

    with sessionmaker(bind=engine, future=True)() as session:
        assert session.get(Message, uuid.UUID(stored["id"])) is None


def test_vector_upsert_and_fetch(repository):
    conversation_id = _uuid()
    repository.ensure_conversation(conversation_id, tenant_id=TENANT)
    message = repository.add_message(
        conversation_id,
        tenant_id=TENANT,
        role="user",
        content={"text": "vector"},
        metadata={},
    )

    stored_vectors = repository.upsert_message_vectors(
        message["id"],
        [
            {
                "values": [0.1, 0.2, 0.3],
                "provider": "unit-test",
                "model": "tiny",
                "metadata": {"extra": True},
            }
        ],
    )

    assert len(stored_vectors) == 1
    stored_record = stored_vectors[0]
    assert stored_record["embedding_checksum"]
    assert stored_record["metadata"]["conversation_id"] == str(conversation_id)

    fetched = repository.fetch_message_vectors(
        tenant_id=TENANT, conversation_id=conversation_id
    )
    assert len(fetched) == 1
    assert fetched[0]["vector_key"] == stored_record["vector_key"]

    removed = repository.delete_message_vectors(
        tenant_id=TENANT, conversation_id=conversation_id
    )
    assert removed == 1
    assert (
        repository.fetch_message_vectors(
            tenant_id=TENANT, conversation_id=conversation_id
        )
        == []
    )


def test_record_edit_updates_type_and_status(repository):
    conversation_id = _uuid()
    repository.ensure_conversation(conversation_id, tenant_id=TENANT)
    stored = repository.add_message(
        conversation_id,
        tenant_id=TENANT,
        role="assistant",
        content={"text": "draft"},
        metadata={},
    )

    updated = repository.record_edit(
        conversation_id,
        stored["id"],
        tenant_id=TENANT,
        content={"text": "final"},
        message_type="tool",
        status="delivered",
    )

    assert updated["content"] == {"text": "final"}
    assert updated["message_type"] == "tool"
    assert updated["status"] == "delivered"

    with repository._session_scope() as session:  # pylint: disable=protected-access
        message = session.get(Message, uuid.UUID(stored["id"]))
        assert message is not None
        assert message.message_type == "tool"
        assert message.status == "delivered"


def test_fetch_messages_filters_by_type_and_status(repository):
    conversation_id = _uuid()
    repository.ensure_conversation(conversation_id, tenant_id=TENANT)
    first = repository.add_message(
        conversation_id,
        tenant_id=TENANT,
        role="user",
        content={"text": "pending"},
        metadata={},
        message_type="text",
        status="pending",
    )
    second = repository.add_message(
        conversation_id,
        tenant_id=TENANT,
        role="assistant",
        content={"text": "complete"},
        metadata={},
        message_type="summary",
        status="sent",
    )
    repository.add_message(
        conversation_id,
        tenant_id=TENANT,
        role="assistant",
        content={"text": "other"},
        metadata={},
        message_type="tool",
        status="failed",
    )

    filtered = repository.fetch_messages(
        conversation_id,
        tenant_id=TENANT,
        message_types=["summary", "text"],
        statuses=["sent", "pending"],
    )

    assert [msg["id"] for msg in filtered] == [first["id"], second["id"]]
