import uuid

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from modules.conversation_store import Base
from modules.conversation_store.models import Conversation, Session as ConversationSession, User
from modules.task_store import TaskStatus
from modules.task_store.models import TaskDependency
from modules.task_store.repository import TaskStoreRepository
from modules.task_store.service import (
    TaskDependencyError,
    TaskService,
    TaskTransitionError,
)


@pytest.fixture()
def engine():
    engine = create_engine("sqlite:///:memory:", future=True)

    @event.listens_for(engine, "connect")
    def _enable_sqlite_foreign_keys(dbapi_connection, connection_record):  # pragma: no cover - event wiring
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture()
def session_factory(engine):
    return sessionmaker(bind=engine, future=True)


@pytest.fixture()
def repository(session_factory):
    repo = TaskStoreRepository(session_factory)
    repo.create_schema()
    return repo


@pytest.fixture()
def identity(session_factory):
    with session_factory() as session:
        user = User(external_id="user-1", display_name="Test User")
        store_session = ConversationSession(user=user)
        conversation = Conversation(session=store_session, tenant_id="tenant-1")
        session.add_all([user, store_session, conversation])
        session.commit()
        return {
            "user_id": user.id,
            "session_id": store_session.id,
            "conversation_id": conversation.id,
            "tenant_id": conversation.tenant_id,
        }


@pytest.fixture()
def event_recorder():
    events: list[tuple[str, dict]] = []

    def emit(event_name: str, payload):
        events.append((event_name, dict(payload)))

    return events, emit


@pytest.fixture()
def service(repository, event_recorder):
    events, emitter = event_recorder
    svc = TaskService(repository, event_emitter=emitter)
    return svc, events


def test_transition_sequence_enforces_rules(service, identity):
    svc, events = service
    record = svc.create_task(
        "Draft report",
        tenant_id=identity["tenant_id"],
        conversation_id=identity["conversation_id"],
    )

    assert events[-1][0] == "task.created"

    with pytest.raises(TaskTransitionError):
        svc.transition_task(
            record["id"],
            tenant_id=identity["tenant_id"],
            target_status=TaskStatus.REVIEW,
        )

    ready = svc.transition_task(
        record["id"], tenant_id=identity["tenant_id"], target_status=TaskStatus.READY
    )
    assert ready["status"] == TaskStatus.READY.value

    in_progress = svc.transition_task(
        record["id"], tenant_id=identity["tenant_id"], target_status=TaskStatus.IN_PROGRESS
    )
    assert in_progress["status"] == TaskStatus.IN_PROGRESS.value

    review = svc.transition_task(
        record["id"], tenant_id=identity["tenant_id"], target_status=TaskStatus.REVIEW
    )
    assert review["status"] == TaskStatus.REVIEW.value

    done = svc.transition_task(
        record["id"], tenant_id=identity["tenant_id"], target_status=TaskStatus.DONE
    )
    assert done["status"] == TaskStatus.DONE.value

    assert any(event for event in events if event[0] == "task.status_changed")


def test_dependency_gating_blocks_progress(service, repository, identity, session_factory):
    svc, events = service
    primary = svc.create_task(
        "Primary",
        tenant_id=identity["tenant_id"],
        conversation_id=identity["conversation_id"],
    )
    blocker = svc.create_task(
        "Blocker",
        tenant_id=identity["tenant_id"],
        conversation_id=identity["conversation_id"],
    )

    with session_factory() as session:
        dependency = TaskDependency(
            task_id=uuid.UUID(primary["id"]),
            depends_on_id=uuid.UUID(blocker["id"]),
        )
        session.add(dependency)
        session.commit()

    with pytest.raises(TaskDependencyError):
        svc.transition_task(
            primary["id"],
            tenant_id=identity["tenant_id"],
            target_status=TaskStatus.READY,
        )

    svc.transition_task(
        blocker["id"], tenant_id=identity["tenant_id"], target_status=TaskStatus.READY
    )
    svc.transition_task(
        blocker["id"], tenant_id=identity["tenant_id"], target_status=TaskStatus.IN_PROGRESS
    )
    svc.transition_task(
        blocker["id"], tenant_id=identity["tenant_id"], target_status=TaskStatus.REVIEW
    )
    svc.transition_task(
        blocker["id"], tenant_id=identity["tenant_id"], target_status=TaskStatus.DONE
    )

    progressed = svc.transition_task(
        primary["id"],
        tenant_id=identity["tenant_id"],
        target_status=TaskStatus.READY,
    )
    assert progressed["status"] == TaskStatus.READY.value
    assert svc.dependencies_complete(primary["id"], tenant_id=identity["tenant_id"])
    assert any(event for event in events if event[0] == "task.status_changed")


def test_update_task_same_owner_no_reassignment(service, identity, monkeypatch):
    svc, events = service
    record = svc.create_task(
        "Owned task",
        tenant_id=identity["tenant_id"],
        conversation_id=identity["conversation_id"],
        owner_id=identity["user_id"],
    )
    initial_event_count = len(events)

    lifecycle_events: list[dict] = []

    def capture_lifecycle_event(**payload):
        lifecycle_events.append(payload)

    monkeypatch.setattr(
        "modules.task_store.service.record_task_lifecycle_event",
        capture_lifecycle_event,
    )

    updated = svc.update_task(
        record["id"],
        tenant_id=identity["tenant_id"],
        changes={"owner_id": identity["user_id"]},
    )

    assert updated["owner_id"] == record["owner_id"]
    assert updated.get("events") == []
    assert len(events) == initial_event_count
    assert lifecycle_events == []
