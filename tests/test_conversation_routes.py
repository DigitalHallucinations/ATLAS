from __future__ import annotations

import asyncio
import uuid
from datetime import timezone

import pytest

sqlalchemy = pytest.importorskip("sqlalchemy")
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from modules.Server import AtlasServer, RequestContext
from modules.Server.conversation_routes import (
    ConversationAuthorizationError,
    _decode_cursor,
    _encode_cursor,
)
from modules.conversation_store import Base, ConversationStoreRepository
from modules.conversation_store.models import Message, Session, User
from modules.orchestration.message_bus import InMemoryQueueBackend, MessageBus


@pytest.fixture
def repository(postgresql):
    engine = create_engine(postgresql.dsn(), future=True)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, future=True)
    repo = ConversationStoreRepository(factory)
    repo.create_schema()
    yield repo
    engine.dispose()


@pytest.fixture
def message_bus():
    bus = MessageBus(backend=InMemoryQueueBackend())
    yield bus
    try:
        asyncio.run(bus.close())
    except RuntimeError:
        # When running inside an existing event loop we fall back to a best effort close.
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            return
        if loop.is_running():
            loop.create_task(bus.close())
        else:
            loop.run_until_complete(bus.close())


@pytest.fixture
def server(repository, message_bus):
    return AtlasServer(conversation_repository=repository, message_bus=message_bus)


@pytest.fixture
def tenant_context():
    return RequestContext(
        tenant_id="tenant-1",
        user_id=str(uuid.uuid4()),
        session_id="session-1",
    )


def test_message_crud_and_pagination(
    server: AtlasServer,
    tenant_context: RequestContext,
    repository: ConversationStoreRepository,
):
    conversation_id = str(uuid.uuid4())
    first = server.create_message(
        {
            "conversation_id": conversation_id,
            "role": "user",
            "content": {"text": "hello"},
            "metadata": {"topic": "intro"},
        },
        context=tenant_context,
    )
    assert first["tenant_id"] == tenant_context.tenant_id
    assert first["user_id"] is not None
    assert first["session_id"] is not None

    with repository._session_factory() as db_session:  # type: ignore[attr-defined]
        message_row = db_session.get(Message, uuid.UUID(first["id"]))
        assert message_row is not None
        assert message_row.user_id is not None
        conversation_row = message_row.conversation
        assert conversation_row is not None
        assert conversation_row.session_id is not None
        stored_user = db_session.get(User, message_row.user_id)
        stored_session = db_session.get(Session, conversation_row.session_id)
        assert stored_user is not None
        assert stored_user.external_id == tenant_context.user_id
        assert stored_session is not None
        assert stored_session.external_id == tenant_context.session_id
        assert stored_session.user_id == stored_user.id

    updated = server.update_message(
        first["id"],
        {"conversation_id": conversation_id, "metadata": {"edited": True}},
        context=tenant_context,
    )
    assert updated["metadata"]["edited"]

    deleted = server.delete_message(
        first["id"],
        {"conversation_id": conversation_id, "reason": "cleanup"},
        context=tenant_context,
    )
    assert deleted["status"] == "deleted"
    assert deleted["message"]["deleted_at"]

    for idx in range(4):
        server.create_message(
            {
                "conversation_id": conversation_id,
                "role": "assistant",
                "content": {"text": f"response-{idx}"},
                "metadata": {"sequence": idx},
            },
            context=tenant_context,
        )

    page_one = server.list_messages(
        conversation_id,
        {"page_size": 2, "include_deleted": False},
        context=tenant_context,
    )
    assert len(page_one["items"]) == 2
    assert page_one["page"]["next_cursor"] is not None
    assert first["id"] not in {item["id"] for item in page_one["items"]}

    page_two = server.list_messages(
        conversation_id,
        {"page_size": 2, "cursor": page_one["page"]["next_cursor"]},
        context=tenant_context,
    )
    assert page_two["items"]
    assert page_two["page"]["next_cursor"] is None

    foreign_context = RequestContext(tenant_id="other-tenant")
    with pytest.raises(ConversationAuthorizationError):
        server.list_messages(conversation_id, context=foreign_context)


def test_search_supports_text_and_vector(server: AtlasServer, tenant_context: RequestContext):
    conversation_id = str(uuid.uuid4())
    other_conversation = str(uuid.uuid4())
    foreign_conversation = str(uuid.uuid4())

    vector_message = server.create_message(
        {
            "conversation_id": conversation_id,
            "role": "assistant",
            "content": {"text": "weather forecast"},
            "metadata": {"topic": "weather"},
            "vectors": [
                {
                    "values": [0.1, 0.2, 0.3],
                    "provider": "unit-test",
                    "model": "demo",
                    "metadata": {"topic": "weather"},
                }
            ],
        },
        context=tenant_context,
    )

    server.create_message(
        {
            "conversation_id": other_conversation,
            "role": "assistant",
            "content": {"text": "meeting notes"},
            "metadata": {"topic": "work"},
        },
        context=tenant_context,
    )

    foreign_context = RequestContext(tenant_id="tenant-2")
    server.create_message(
        {
            "conversation_id": foreign_conversation,
            "role": "assistant",
            "content": {"text": "outside tenant"},
            "metadata": {"topic": "weather"},
        },
        context=foreign_context,
    )

    text_results = server.search_conversations(
        {"conversation_ids": [conversation_id], "text": "weather"},
        context=tenant_context,
    )
    assert text_results["count"] == 1
    assert text_results["items"][0]["message"]["id"] == vector_message["id"]

    vector_results = server.search_conversations(
        {
            "conversation_ids": [conversation_id, other_conversation],
            "vector": {"values": [0.1, 0.2, 0.3]},
            "metadata": {"topic": "weather"},
        },
        context=tenant_context,
    )
    assert vector_results["count"] == 1
    assert vector_results["items"][0]["message"]["id"] == vector_message["id"]


    with pytest.raises(ConversationAuthorizationError):
        server.search_conversations(
            {"conversation_ids": [foreign_conversation], "text": "weather"},
            context=tenant_context,
        )


def test_cursor_decoding_accepts_zulu_timestamp() -> None:
    message_id = str(uuid.uuid4())
    cursor = _encode_cursor("2024-01-01T09:00:00Z", message_id)

    decoded_at, decoded_id = _decode_cursor(cursor)

    assert decoded_id == uuid.UUID(message_id)
    assert decoded_at.tzinfo == timezone.utc


def test_search_scans_long_conversations(server: AtlasServer, tenant_context: RequestContext):
    conversation_id = str(uuid.uuid4())
    base_vector = [0.05, 0.1, 0.2]

    early_message = server.create_message(
        {
            "conversation_id": conversation_id,
            "role": "assistant",
            "content": {"text": "needle early"},
            "metadata": {"topic": "long"},
            "vectors": [
                {
                    "values": base_vector,
                    "provider": "unit-test",
                    "model": "demo",
                }
            ],
        },
        context=tenant_context,
    )

    for idx in range(120):
        server.create_message(
            {
                "conversation_id": conversation_id,
                "role": "user",
                "content": {"text": f"filler-{idx}"},
                "metadata": {"topic": "long", "index": idx},
            },
            context=tenant_context,
        )

    late_message = server.create_message(
        {
            "conversation_id": conversation_id,
            "role": "assistant",
            "content": {"text": "needle late"},
            "metadata": {"topic": "long"},
            "vectors": [
                {
                    "values": base_vector,
                    "provider": "unit-test",
                    "model": "demo",
                }
            ],
        },
        context=tenant_context,
    )

    text_results = server.search_conversations(
        {"conversation_ids": [conversation_id], "text": "Needle", "limit": 2},
        context=tenant_context,
    )
    assert [item["message"]["id"] for item in text_results["items"]] == [
        late_message["id"],
        early_message["id"],
    ]

    text_offset = server.search_conversations(
        {
            "conversation_ids": [conversation_id],
            "text": "needle",
            "limit": 1,
            "offset": 1,
        },
        context=tenant_context,
    )
    assert text_offset["count"] == 1
    assert text_offset["items"][0]["message"]["id"] == early_message["id"]

    vector_results = server.search_conversations(
        {
            "conversation_ids": [conversation_id],
            "vector": {"values": base_vector},
            "limit": 2,
        },
        context=tenant_context,
    )
    assert [item["message"]["id"] for item in vector_results["items"]] == [
        late_message["id"],
        early_message["id"],
    ]

    vector_ascending = server.search_conversations(
        {
            "conversation_ids": [conversation_id],
            "vector": {"values": base_vector},
            "limit": 2,
            "order": "asc",
        },
        context=tenant_context,
    )
    assert [item["message"]["id"] for item in vector_ascending["items"]] == [
        early_message["id"],
        late_message["id"],
    ]


def test_vector_search_limits_repository_fetches(
    server: AtlasServer,
    tenant_context: RequestContext,
    monkeypatch: pytest.MonkeyPatch,
):
    conversation_id = str(uuid.uuid4())
    base_vector = [0.2, 0.3, 0.4]

    for filler_index in range(80):
        server.create_message(
            {
                "conversation_id": conversation_id,
                "role": "assistant",
                "content": {"text": f"filler-{filler_index}"},
                "metadata": {"topic": "bulk", "filler": filler_index},
                "vectors": [
                    {
                        "values": [-value for value in base_vector],
                        "provider": "unit-test",
                        "model": "demo",
                    }
                ],
            },
            context=tenant_context,
        )

    matching_ids: list[str] = []
    for match_index in range(40):
        created = server.create_message(
            {
                "conversation_id": conversation_id,
                "role": "assistant",
                "content": {"text": f"match-{match_index}"},
                "metadata": {"topic": "bulk", "match": match_index},
                "vectors": [
                    {
                        "values": base_vector,
                        "provider": "unit-test",
                        "model": "demo",
                    }
                ],
            },
            context=tenant_context,
        )
        matching_ids.append(created["id"])

    calls: list[dict[str, object]] = []
    original = ConversationStoreRepository.query_message_vectors

    def instrumented(self, *args, **kwargs):
        record: dict[str, object] = {"kwargs": dict(kwargs), "yield_count": 0}
        calls.append(record)
        for message, vector in original(self, *args, **kwargs):
            record["yield_count"] = int(record["yield_count"]) + 1
            yield message, vector

    monkeypatch.setattr(ConversationStoreRepository, "query_message_vectors", instrumented)

    expected_desc = list(reversed(matching_ids))

    windowed = server.search_conversations(
        {
            "conversation_ids": [conversation_id],
            "vector": {"values": base_vector},
            "limit": 5,
            "offset": 10,
            "top_k": 50,
        },
        context=tenant_context,
    )
    assert windowed["count"] == 5
    assert [item["message"]["id"] for item in windowed["items"]] == expected_desc[10:15]
    assert len(calls) >= 1
    first_call = calls[0]
    first_kwargs = first_call["kwargs"]
    assert isinstance(first_kwargs, dict)
    assert first_kwargs["limit"] == 15
    assert first_kwargs["offset"] == 10
    assert first_kwargs["top_k"] == 50
    assert first_call["yield_count"] == 15

    bulk = server.search_conversations(
        {
            "conversation_ids": [conversation_id],
            "vector": {"values": base_vector},
            "limit": 20,
            "top_k": 12,
        },
        context=tenant_context,
    )
    assert bulk["count"] == 12
    assert [item["message"]["id"] for item in bulk["items"]] == expected_desc[:12]
    assert len(calls) >= 2
    second_call = calls[1]
    second_kwargs = second_call["kwargs"]
    assert isinstance(second_kwargs, dict)
    assert second_kwargs["limit"] == 20
    assert second_kwargs["offset"] == 0
    assert second_kwargs["top_k"] == 12
    assert second_call["yield_count"] == 12

@pytest.mark.asyncio
async def test_streaming_delivers_message_events(server: AtlasServer, tenant_context: RequestContext):
    conversation_id = str(uuid.uuid4())
    stream = server.stream_conversation_events(conversation_id, context=tenant_context)
    first_event = asyncio.create_task(stream.__anext__())
    await asyncio.sleep(0.05)

    created = server.create_message(
        {
            "conversation_id": conversation_id,
            "role": "user",
            "content": {"text": "stream me"},
        },
        context=tenant_context,
    )

    event = await asyncio.wait_for(first_event, timeout=1.0)
    await stream.aclose()

    assert event["event_type"] == "created"
    assert event["message"]["id"] == created["id"]
    assert event["tenant_id"] == tenant_context.tenant_id
