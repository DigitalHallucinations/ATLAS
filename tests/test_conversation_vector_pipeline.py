from __future__ import annotations

import uuid

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from modules.conversation_store import (
    Base,
    ConversationStoreRepository,
    ConversationVectorCatalog,
    ConversationVectorPipeline,
)
from modules.Tools.Base_Tools.vector_store import build_vector_store_service

# Use a fixed tenant identifier for repository interactions.
TENANT = "vector-test"

# Ensure the in-memory adapter is registered for tests.
import modules.Tools.providers.vector_store.in_memory  # noqa: F401


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


@pytest.mark.asyncio
async def test_pipeline_persists_vectors_and_hydrates(repository):
    conversation_id = _uuid()
    repository.ensure_conversation(conversation_id, tenant_id=TENANT)
    message = repository.add_message(
        conversation_id,
        tenant_id=TENANT,
        role="assistant",
        content={"text": "hello"},
        metadata={},
    )

    catalog = ConversationVectorCatalog(repository)
    service = build_vector_store_service(
        adapter_name="in_memory",
        adapter_config={"index_name": "pipeline-primary"},
        catalog=catalog,
    )
    pipeline = ConversationVectorPipeline(repository, service)

    await pipeline.start()
    try:
        await pipeline.enqueue_message(
            {
                "id": message["id"],
                "conversation_id": str(conversation_id),
            },
            vectors=[
                {
                    "values": [0.1, 0.2, 0.3],
                    "provider": "unit-test",
                    "model": "tiny",
                }
            ],
        )
        await pipeline.wait_for_idle()
    finally:
        await pipeline.stop()

    initial_query = await service.query_vectors(
        namespace=str(conversation_id),
        query=[0.1, 0.2, 0.3],
        top_k=3,
        include_values=True,
    )

    assert initial_query["matches"]
    match = initial_query["matches"][0]
    assert match["id"]
    assert match["metadata"]["conversation_id"] == str(conversation_id)

    # Simulate a restart by constructing a new service with an empty adapter
    restarted_service = build_vector_store_service(
        adapter_name="in_memory",
        adapter_config={"index_name": "pipeline-restart"},
        catalog=catalog,
    )

    restart_query = await restarted_service.query_vectors(
        namespace=str(conversation_id),
        query=[0.1, 0.2, 0.3],
        top_k=3,
        include_values=False,
    )

    assert restart_query["matches"]
    restart_match = restart_query["matches"][0]
    assert restart_match["metadata"]["message_id"] == message["id"]


@pytest.mark.asyncio
async def test_pipeline_uses_embedder_when_vectors_missing(repository):
    conversation_id = _uuid()
    repository.ensure_conversation(conversation_id, tenant_id=TENANT)
    message = repository.add_message(
        conversation_id,
        tenant_id=TENANT,
        role="user",
        content={"text": "embed me"},
        metadata={},
    )

    captured: list[str] = []

    async def embedder(payload):
        captured.append(payload["id"])
        return [0.4, 0.5, 0.6]

    catalog = ConversationVectorCatalog(repository)
    service = build_vector_store_service(
        adapter_name="in_memory",
        adapter_config={"index_name": "pipeline-embedder"},
        catalog=catalog,
    )
    pipeline = ConversationVectorPipeline(repository, service, embedder=embedder)

    await pipeline.start()
    try:
        await pipeline.enqueue_message(
            {
                "id": message["id"],
                "conversation_id": str(conversation_id),
            }
        )
        await pipeline.wait_for_idle()
    finally:
        await pipeline.stop()

    assert captured == [message["id"]]

    stored = repository.fetch_message_vectors(
        tenant_id=TENANT, conversation_id=conversation_id
    )
    assert len(stored) == 1
    assert stored[0]["metadata"]["message_id"] == message["id"]

    query = await service.query_vectors(
        namespace=str(conversation_id),
        query=[0.4, 0.5, 0.6],
        top_k=1,
        include_values=True,
    )
    assert query["matches"]
    assert query["matches"][0]["id"] == stored[0]["vector_key"]
