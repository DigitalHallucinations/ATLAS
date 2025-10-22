from __future__ import annotations

import asyncio
import uuid

import pytest

sqlalchemy = pytest.importorskip("sqlalchemy")
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from modules.Server import AtlasServer, RequestContext
from modules.Server.task_routes import TaskAuthorizationError, TaskNotFoundRouteError
from modules.conversation_store import Base, ConversationStoreRepository
from modules.task_store.repository import TaskStoreRepository
from modules.task_store.service import TaskService


class _BusStub:
    """Lightweight stub capturing published messages for assertions."""

    def __init__(self) -> None:
        self.published: list[tuple[str, dict]] = []

    def publish_from_sync(self, topic: str, payload: dict, **_kwargs) -> str:
        self.published.append((topic, dict(payload)))
        return uuid.uuid4().hex


@pytest.fixture
def session_factory(postgresql):
    engine = create_engine(postgresql.dsn(), future=True)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, future=True)
    yield factory
    engine.dispose()


@pytest.fixture
def repositories(session_factory):
    conversation_repo = ConversationStoreRepository(session_factory)
    conversation_repo.create_schema()
    task_repo = TaskStoreRepository(session_factory)
    task_repo.create_schema()
    return conversation_repo, task_repo


@pytest.fixture
def message_bus_stub():
    return _BusStub()


@pytest.fixture
def server(repositories, message_bus_stub):
    conversation_repo, task_repo = repositories
    service = TaskService(task_repo)
    return AtlasServer(
        conversation_repository=conversation_repo,
        message_bus=message_bus_stub,
        task_service=service,
    )


@pytest.fixture
def tenant_context(repositories):
    conversation_repo, _ = repositories
    tenant_id = "tenant-1"
    user_uuid = conversation_repo.ensure_user(
        "user-ext",
        metadata={"tenant_id": tenant_id},
    )
    session_uuid = conversation_repo.ensure_session(
        user_uuid,
        "session-ext",
        metadata={"tenant_id": tenant_id},
    )
    conversation_id = str(uuid.uuid4())
    conversation_repo.ensure_conversation(
        conversation_id,
        tenant_id=tenant_id,
        session_id=session_uuid,
        metadata={"topic": "demo"},
    )
    context = RequestContext(
        tenant_id=tenant_id,
        user_id=str(user_uuid),
        session_id=str(session_uuid),
    )
    return context, {
        "conversation_id": conversation_id,
        "owner_id": str(user_uuid),
        "session_id": str(session_uuid),
    }


def test_task_crud_and_pagination(server, tenant_context, message_bus_stub):
    context, identity = tenant_context
    created = server.create_task(
        {
            "title": "Draft report",
            "description": "Initial draft",
            "conversation_id": identity["conversation_id"],
            "owner_id": identity["owner_id"],
            "session_id": identity["session_id"],
            "metadata": {"team": "blue"},
        },
        context=context,
    )

    assert created["title"] == "Draft report"
    assert created["tenant_id"] == context.tenant_id
    assert any(
        topic == f"task.events.{created['id']}" for topic, _ in message_bus_stub.published
    )

    updated = server.update_task(
        created["id"],
        {"description": "Updated draft", "priority": 5},
        context=context,
    )
    assert updated["description"] == "Updated draft"
    assert updated["priority"] == 5

    transitioned = server.transition_task(
        created["id"],
        "ready",
        context=context,
        expected_updated_at=updated["updated_at"],
    )
    assert transitioned["status"] == "ready"

    # create additional tasks to exercise pagination
    for idx in range(4):
        server.create_task(
            {
                "title": f"Follow-up {idx}",
                "conversation_id": identity["conversation_id"],
            },
            context=context,
        )

    page_one = server.list_tasks({"page_size": 2}, context=context)
    assert len(page_one["items"]) == 2
    assert page_one["page"]["next_cursor"] is not None

    page_two = server.list_tasks(
        {"cursor": page_one["page"]["next_cursor"]},
        context=context,
    )
    assert page_two["items"]

    search = server.search_tasks({"text": "draft"}, context=context)
    assert any(item["id"] == created["id"] for item in search["items"])


def test_authorization_and_isolation(server, tenant_context):
    context, identity = tenant_context
    record = server.create_task(
        {
            "title": "Private task",
            "conversation_id": identity["conversation_id"],
        },
        context=context,
    )

    with pytest.raises(TaskAuthorizationError):
        server.list_tasks(context=RequestContext(tenant_id=""))

    foreign_context = RequestContext(tenant_id="tenant-2")
    with pytest.raises(TaskNotFoundRouteError):
        server.get_task(record["id"], context=foreign_context)


@pytest.mark.asyncio
async def test_stream_task_events_polling(server, tenant_context):
    context, identity = tenant_context
    record = server.create_task(
        {
            "title": "Long running",
            "conversation_id": identity["conversation_id"],
        },
        context=context,
    )

    stream = server.stream_task_events(record["id"], context=context)
    async def _consume():
        iterator = stream.__aiter__()
        for _ in range(2):
            event = await asyncio.wait_for(iterator.__anext__(), timeout=2)
            if event["event_type"] == "updated":
                return event
    consumer = asyncio.create_task(_consume())
    await asyncio.sleep(0.1)
    server.update_task(
        record["id"],
        {"description": "Progress"},
        context=context,
    )
    result = await consumer
    assert result["event_type"] == "updated"


def test_list_tasks_handles_missing_service(caplog):
    class _ConfigManager:
        def get_task_service(self):
            raise RuntimeError("not available")

    server = AtlasServer(config_manager=_ConfigManager())

    result = server.list_tasks(context=RequestContext(tenant_id="tenant"))

    assert result == {"items": []}
    assert any(
        "Task service is not configured" in record.message for record in caplog.records
    )
