from __future__ import annotations

import uuid
from typing import Any, Iterable, Tuple

import pytest

sqlalchemy = pytest.importorskip("sqlalchemy")
try:
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import sessionmaker
except ImportError as exc:  # pragma: no cover - skip when SQLAlchemy is stubbed
    pytest.skip(
        f"SQLAlchemy textual helpers unavailable: {exc}", allow_module_level=True
    )

if getattr(create_engine, "__module__", "").startswith("tests.conftest"):
    pytest.skip(
        "SQLAlchemy runtime is unavailable for conversation search tests",
        allow_module_level=True,
    )

pytest.importorskip(
    "pytest_postgresql",
    reason="PostgreSQL fixture is required for conversation search tests",
)
from modules.conversation_store import Base, ConversationStoreRepository

from ATLAS.services.conversations import ConversationService
from ATLAS.ATLAS import ATLAS as AtlasApplication


TENANT = "search-tenant"


class DummyLogger:
    def __init__(self) -> None:
        self.records: list[tuple[str, tuple[Any, ...]]] = []

    def debug(self, *_args, **_kwargs) -> None:  # pragma: no cover - no-op
        pass

    def info(self, *_args, **_kwargs) -> None:  # pragma: no cover - no-op
        pass

    def warning(self, message: str, *args: Any, **_kwargs) -> None:
        self.records.append(("warning", (message, *args)))

    def error(self, message: str, *args: Any, **_kwargs) -> None:
        self.records.append(("error", (message, *args)))


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


def test_service_search_supports_metadata_and_vectors(postgres_repository) -> None:
    repository, _engine = postgres_repository
    service = ConversationService(repository=repository, logger=DummyLogger(), tenant_id=TENANT)

    conversation_id = uuid.uuid4()
    repository.ensure_conversation(conversation_id, tenant_id=TENANT)

    vector = [0.1, 0.2, 0.3]
    matching = repository.add_message(
        conversation_id,
        tenant_id=TENANT,
        role="assistant",
        content={"text": "Vector needle"},
        metadata={"needle": True, "category": "match"},
        vectors=[
            {
                "values": vector,
                "provider": "unit-test",
                "model": "demo",
                "metadata": {"category": "match"},
            }
        ],
    )

    repository.add_message(
        conversation_id,
        tenant_id=TENANT,
        role="assistant",
        content={"text": "background"},
        metadata={"needle": False, "category": "other"},
        vectors=[
            {
                "values": [component * -1 for component in vector],
                "provider": "unit-test",
                "model": "demo",
                "metadata": {"category": "other"},
            }
        ],
    )

    text_results = service.search_conversations(
        text="needle",
        metadata={"needle": True},
        conversation_ids=[conversation_id],
        limit=5,
    )
    assert text_results["count"] == 1
    assert text_results["items"][0]["message"]["id"] == matching["id"]

    vector_results = service.search_conversations(
        vector={"values": vector},
        metadata={"category": "match"},
        conversation_ids=[conversation_id],
        limit=3,
    )
    assert vector_results["count"] == 1
    assert vector_results["items"][0]["message"]["id"] == matching["id"]
    assert vector_results["items"][0]["score"] > 0


def test_atlas_search_conversations_prefers_server(postgres_repository) -> None:
    repository, _engine = postgres_repository
    service = ConversationService(repository=repository, logger=DummyLogger(), tenant_id=TENANT)

    class StubServer:
        def __init__(self) -> None:
            self.calls: list[tuple[dict[str, Any], dict[str, Any]]] = []

        def search_conversations(self, payload, *, context=None):
            snapshot = (dict(payload), dict(context or {}))
            self.calls.append(snapshot)
            return {"count": 2, "items": ["server"]}

    atlas = AtlasApplication.__new__(AtlasApplication)
    atlas.logger = DummyLogger()
    atlas.server = StubServer()
    atlas.conversation_service = service
    atlas.tenant_id = TENANT

    result = atlas.search_conversations({"text": "hello", "limit": 5})
    assert result == {"count": 2, "items": ["server"]}
    assert atlas.server.calls
    payload, context = atlas.server.calls[0]
    assert payload["text"] == "hello"
    assert context["tenant_id"] == TENANT


def test_atlas_search_conversations_falls_back_to_service(postgres_repository) -> None:
    repository, _engine = postgres_repository
    logger = DummyLogger()
    service = ConversationService(repository=repository, logger=logger, tenant_id=TENANT)

    conversation_id = uuid.uuid4()
    repository.ensure_conversation(conversation_id, tenant_id=TENANT)
    match = repository.add_message(
        conversation_id,
        tenant_id=TENANT,
        role="assistant",
        content={"text": "Fallback needle"},
        metadata={"tag": "fallback"},
    )

    class ExplodingServer:
        def search_conversations(self, *_args, **_kwargs):  # pragma: no cover - deliberate failure path
            raise RuntimeError("boom")

    atlas = AtlasApplication.__new__(AtlasApplication)
    atlas.logger = logger
    atlas.server = ExplodingServer()
    atlas.conversation_service = service
    atlas.tenant_id = TENANT

    response = atlas.search_conversations({"text": "needle", "conversation_ids": [conversation_id]})
    assert response["count"] == 1
    assert response["items"][0]["message"]["id"] == match["id"]
    assert any(level == "warning" for level, _ in logger.records)
