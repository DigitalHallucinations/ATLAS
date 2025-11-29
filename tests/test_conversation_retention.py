"""Integration tests for conversation store retention policies."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict

import pytest

sqlalchemy = pytest.importorskip("sqlalchemy")
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from modules.Server import AtlasServer, RequestContext
from modules.Server.conversation_routes import ConversationAuthorizationError
from modules.background_tasks import RetentionWorker
from modules.conversation_store import Base, ConversationStoreRepository
from modules.conversation_store.models import Conversation, Message, MessageAsset, MessageEvent


@pytest.fixture
def engine(postgresql):
    engine = create_engine(postgresql.dsn(), future=True)
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def retention_repository(engine):
    factory = sessionmaker(bind=engine, future=True)
    retention: Dict[str, object] = {
        "message_retention_days": 30,
        "soft_delete_after_days": 15,
        "soft_delete_grace_days": 7,
        "conversation_archive_days": 30,
        "archived_conversation_retention_days": 60,
        "tenant_limits": {"alpha": {"max_conversations": 2}},
    }
    repository = ConversationStoreRepository(factory, retention=retention)
    repository.create_schema()
    return repository


def test_prune_expired_messages_removes_old_records(retention_repository, engine):
    repo = retention_repository
    now = datetime(2024, 2, 1, tzinfo=timezone.utc)
    conversation_id = uuid.uuid4()
    repo.ensure_conversation(conversation_id, tenant_id="alpha")

    old_message = repo.add_message(
        conversation_id,
        tenant_id="alpha",
        role="user",
        content={"text": "ancient"},
        timestamp=(now - timedelta(days=45)).isoformat(),
        metadata={},
        assets=[{"asset_type": "file", "uri": "s3://bucket/asset"}],
        events=[{"event_type": "ingested", "metadata": {"source": "test"}}],
    )

    soft_deleted = repo.add_message(
        conversation_id,
        tenant_id="alpha",
        role="assistant",
        content={"text": "trashed"},
        timestamp=(now - timedelta(days=20)).isoformat(),
        metadata={},
    )

    lingering = repo.add_message(
        conversation_id,
        tenant_id="alpha",
        role="system",
        content={"text": "lingering"},
        timestamp=(now - timedelta(days=18)).isoformat(),
        metadata={},
    )

    with sessionmaker(bind=engine, future=True)() as session:
        soft_row = session.get(Message, uuid.UUID(soft_deleted["id"]))
        assert soft_row is not None
        soft_row.deleted_at = now - timedelta(days=10)
        soft_row.status = "deleted"
        session.flush()

    stats = repo.prune_expired_messages(now=now)

    assert stats["hard_deleted"] >= 2
    assert stats["soft_deleted"] >= 1

    with sessionmaker(bind=engine, future=True)() as session:
        old_row = session.get(Message, uuid.UUID(old_message["id"]))
        assert old_row is None
        soft_row = session.get(Message, uuid.UUID(soft_deleted["id"]))
        assert soft_row is None
        lingering_row = session.get(Message, uuid.UUID(lingering["id"]))
        assert lingering_row is not None
        assert lingering_row.deleted_at is not None
        assert lingering_row.status == "deleted"

        asset_rows = session.execute(
            select(MessageAsset).where(
                MessageAsset.message_id == uuid.UUID(old_message["id"])
            )
        ).scalars().all()
        assert asset_rows == []

        event_rows = session.execute(
            select(MessageEvent).where(
                MessageEvent.message_id == uuid.UUID(old_message["id"])
            )
        ).scalars().all()
        assert event_rows == []


def test_prune_archived_conversations_enforces_policies(retention_repository, engine):
    repo = retention_repository
    now = datetime(2024, 2, 1, tzinfo=timezone.utc)

    aging_conversation = repo.ensure_conversation(uuid.uuid4(), tenant_id="alpha")
    tenant_recent_a = repo.ensure_conversation(uuid.uuid4(), tenant_id="alpha")
    tenant_recent_b = repo.ensure_conversation(uuid.uuid4(), tenant_id="alpha")
    tenant_recent_c = repo.ensure_conversation(uuid.uuid4(), tenant_id="alpha")

    archived_to_prune = repo.ensure_conversation(uuid.uuid4(), tenant_id="beta")

    with sessionmaker(bind=engine, future=True)() as session:
        aging = session.get(Conversation, aging_conversation)
        assert aging is not None
        aging.created_at = now - timedelta(days=90)

        recent_a = session.get(Conversation, tenant_recent_a)
        recent_b = session.get(Conversation, tenant_recent_b)
        recent_c = session.get(Conversation, tenant_recent_c)
        assert recent_a and recent_b and recent_c
        recent_a.created_at = now - timedelta(days=10)
        recent_b.created_at = now - timedelta(days=5)
        recent_c.created_at = now - timedelta(days=1)

        archived_row = session.get(Conversation, archived_to_prune)
        assert archived_row is not None
        archived_row.created_at = now - timedelta(days=120)
        archived_row.archived_at = now - timedelta(days=70)
        session.flush()

    stats = repo.prune_archived_conversations(now=now)

    assert stats["archived"] >= 2
    assert stats["deleted"] == 1

    with sessionmaker(bind=engine, future=True)() as session:
        aging = session.get(Conversation, aging_conversation)
        assert aging is not None
        assert aging.archived_at is not None

        recent_a = session.get(Conversation, tenant_recent_a)
        recent_b = session.get(Conversation, tenant_recent_b)
        recent_c = session.get(Conversation, tenant_recent_c)
        assert recent_a is not None
        assert recent_a.archived_at is not None
        assert recent_b is not None
        assert recent_b.archived_at is None
        assert recent_c is not None
        assert recent_c.archived_at is None

        assert session.get(Conversation, archived_to_prune) is None


def test_server_retention_endpoint_requires_admin():
    class DummyRepository:
        def __init__(self) -> None:
            self.calls = 0

        def run_retention(self, *, now: datetime) -> Dict[str, Dict[str, int]]:
            self.calls += 1
            return {"messages": {"hard_deleted": 0, "soft_deleted": 0}, "conversations": {"archived": 0, "deleted": 0}}

    repo = DummyRepository()
    server = AtlasServer(conversation_repository=repo)

    admin_context = RequestContext.from_authenticated_claims(
        tenant_id="admin", roles=("Admin",)
    )
    result = server.run_conversation_retention(context=admin_context)
    assert repo.calls == 1
    assert result["messages"] == {"hard_deleted": 0, "soft_deleted": 0}

    with pytest.raises(ConversationAuthorizationError):
        server.run_conversation_retention(context=RequestContext(tenant_id="tenant", roles=("user",)))


def test_retention_worker_triggers_repository():
    class DummyRepository:
        def __init__(self) -> None:
            self.invocations: list[datetime] = []

        def run_retention(self, *, now: datetime) -> Dict[str, Dict[str, int]]:
            self.invocations.append(now)
            return {"messages": {}, "conversations": {}}

    repo = DummyRepository()
    worker = RetentionWorker(repo, interval_seconds=10)
    moment = datetime(2024, 2, 1, tzinfo=timezone.utc)
    result = worker.run_once(now=moment)
    assert repo.invocations == [moment]
    assert result == {"messages": {}, "conversations": {}}
