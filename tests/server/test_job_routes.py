import asyncio
import uuid

import pytest

sqlalchemy = pytest.importorskip("sqlalchemy")
pytest.importorskip("pytest_postgresql")

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from modules.Server import AtlasServer, RequestContext
from modules.Server.job_routes import (
    JobAuthorizationError,
    JobNotFoundRouteError,
)
from modules.conversation_store import Base, ConversationStoreRepository
from modules.job_store.repository import JobStoreRepository
from modules.job_store.service import JobService
from modules.task_store.repository import TaskStoreRepository
from modules.task_store.service import TaskService


class _BusMessage:
    def __init__(self, topic: str, payload: dict) -> None:
        self.topic = topic
        self.payload = payload


class _BusStub:
    """In-memory message bus supporting subscribe/publish for tests."""

    def __init__(self) -> None:
        self.published: list[tuple[str, dict]] = []
        self._handlers: dict[str, list] = {}

    def publish_from_sync(self, topic: str, payload: dict, **_kwargs) -> str:
        self.published.append((topic, dict(payload)))
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.get_event_loop()
        message = _BusMessage(topic, dict(payload))
        for handler in list(self._handlers.get(topic, [])):
            loop.call_soon_threadsafe(asyncio.create_task, handler(message))
        return uuid.uuid4().hex

    def subscribe(self, topic: str, handler, **_kwargs):
        self._handlers.setdefault(topic, []).append(handler)

        class _Subscription:
            def cancel(inner_self) -> None:  # pragma: no cover - cleanup guard
                callbacks = self._handlers.get(topic, [])
                if handler in callbacks:
                    callbacks.remove(handler)

        return _Subscription()


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
    job_repo = JobStoreRepository(session_factory)
    job_repo.create_schema()
    return conversation_repo, task_repo, job_repo


@pytest.fixture
def message_bus_stub():
    return _BusStub()


@pytest.fixture
def server(repositories, message_bus_stub):
    conversation_repo, task_repo, job_repo = repositories
    task_service = TaskService(task_repo)
    job_service = JobService(job_repo, event_emitter=lambda *_args, **_kwargs: None)
    return AtlasServer(
        conversation_repository=conversation_repo,
        message_bus=message_bus_stub,
        task_service=task_service,
        job_service=job_service,
    )


@pytest.fixture
def server_without_bus(repositories):
    conversation_repo, task_repo, job_repo = repositories
    task_service = TaskService(task_repo)
    job_service = JobService(job_repo, event_emitter=lambda *_args, **_kwargs: None)
    return AtlasServer(
        conversation_repository=conversation_repo,
        task_service=task_service,
        job_service=job_service,
    )


@pytest.fixture
def tenant_context(repositories):
    conversation_repo, _, _ = repositories
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
        metadata={"topic": "jobs"},
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


def test_job_crud_and_task_linking(server, tenant_context):
    context, identity = tenant_context
    job = server.create_job(
        {
            "name": "Weekly summary",
            "description": "Compile weekly report",
            "conversation_id": identity["conversation_id"],
            "metadata": {"persona": "analyst"},
        },
        context=context,
    )

    assert job["name"] == "Weekly summary"
    assert job["tenant_id"] == context.tenant_id

    updated = server.update_job(
        job["id"],
        {"description": "Compile and distribute"},
        context=context,
    )
    assert updated["description"] == "Compile and distribute"

    # create an additional job to exercise pagination
    server.create_job({"name": "Backlog grooming"}, context=context)
    listing = server.list_jobs({"page_size": 1}, context=context)
    assert listing["items"]
    assert listing["page"]["count"] == 1

    task = server.create_task(
        {
            "title": "Initial research",
            "conversation_id": identity["conversation_id"],
            "owner_id": identity["owner_id"],
            "session_id": identity["session_id"],
        },
        context=context,
    )

    link = server.link_job_task(
        job["id"],
        {"task_id": task["id"], "relationship_type": "blocks"},
        context=context,
    )
    assert link["task"]["id"] == task["id"]

    linked = server.list_job_tasks(job["id"], context=context)
    assert [entry["task"]["id"] for entry in linked] == [task["id"]]

    removed = server.unlink_job_task(
        job["id"],
        context=context,
        task_id=task["id"],
    )
    assert removed["task_id"] == task["id"]
    assert server.list_job_tasks(job["id"], context=context) == []


def test_job_authorization(server, tenant_context):
    context, identity = tenant_context
    job = server.create_job(
        {
            "name": "Confidential",
            "conversation_id": identity["conversation_id"],
        },
        context=context,
    )

    with pytest.raises(JobAuthorizationError):
        server.list_jobs(context=RequestContext(tenant_id=""))

    foreign_context = RequestContext(tenant_id="tenant-2")
    with pytest.raises(JobNotFoundRouteError):
        server.get_job(job["id"], context=foreign_context)


def test_stream_job_events_from_bus(server, tenant_context, message_bus_stub):
    context, identity = tenant_context

    async def _exercise():
        job = server.create_job(
            {
                "name": "Monitor",
                "conversation_id": identity["conversation_id"],
            },
            context=context,
        )

        stream = server.stream_job_events(job["id"], context=context)

        async def _consume():
            iterator = stream.__aiter__()
            return await asyncio.wait_for(iterator.__anext__(), timeout=3)

        consumer = asyncio.create_task(_consume())
        await asyncio.sleep(0.1)
        message_bus_stub.publish_from_sync(
            "jobs.updated",
            {"job_id": job["id"], "tenant_id": context.tenant_id},
        )
        payload = await consumer
        assert payload["job_id"] == job["id"]
        assert payload["tenant_id"] == context.tenant_id
        assert payload["event_type"] == "jobs.updated"

    asyncio.run(_exercise())


def test_stream_job_events_polling(server_without_bus, tenant_context):
    context, identity = tenant_context

    async def _exercise():
        job = server_without_bus.create_job(
            {
                "name": "Long running",
                "conversation_id": identity["conversation_id"],
            },
            context=context,
        )

        stream = server_without_bus.stream_job_events(job["id"], context=context)

        async def _consume():
            iterator = stream.__aiter__()
            return await asyncio.wait_for(iterator.__anext__(), timeout=5)

        consumer = asyncio.create_task(_consume())
        await asyncio.sleep(0.1)
        server_without_bus.update_job(
            job["id"],
            {"description": "Progress"},
            context=context,
        )
        event = await consumer
        assert event["job_id"] == job["id"]
        assert event["event_type"] in {"updated", "job.updated"}

    asyncio.run(_exercise())
