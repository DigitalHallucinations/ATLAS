import uuid

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from modules.conversation_store.models import Conversation, Session as ConversationSession, User
from modules.task_store import TaskStatus
from modules.task_store.repository import TaskConcurrencyError, TaskStoreRepository
from modules.conversation_store import Base
from modules.task_store.models import TaskDependency


@pytest.fixture()
def engine(postgresql):
    engine = create_engine(postgresql.dsn(), future=True)

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


def test_create_and_list_tasks_enforces_tenant(repository, identity):
    record = repository.create_task(
        "Draft report",
        tenant_id=identity["tenant_id"],
        owner_id=identity["user_id"],
        session_id=identity["session_id"],
        conversation_id=identity["conversation_id"],
    )

    assert record["status"] == TaskStatus.DRAFT.value
    assert record["tenant_id"] == identity["tenant_id"]

    tasks = repository.list_tasks(tenant_id=identity["tenant_id"])
    assert [task["id"] for task in tasks] == [record["id"]]

    with pytest.raises(ValueError):
        repository.create_task(
            "Cross tenant",
            tenant_id="tenant-2",
            owner_id=identity["user_id"],
            session_id=identity["session_id"],
            conversation_id=identity["conversation_id"],
        )

    other_tasks = repository.list_tasks(tenant_id="tenant-2")
    assert other_tasks == []


def test_update_uses_optimistic_concurrency(repository, identity):
    record = repository.create_task(
        "Initial",
        tenant_id=identity["tenant_id"],
        conversation_id=identity["conversation_id"],
    )

    updated = repository.update_task(
        record["id"],
        tenant_id=identity["tenant_id"],
        changes={"title": "Renamed"},
        expected_updated_at=record["updated_at"],
    )

    assert updated["title"] == "Renamed"

    with pytest.raises(TaskConcurrencyError):
        repository.update_task(
            record["id"],
            tenant_id=identity["tenant_id"],
            changes={"priority": 2},
            expected_updated_at=record["updated_at"],
        )


def test_dependency_statuses_reflect_current_state(repository, identity, session_factory):
    primary = repository.create_task(
        "Primary",
        tenant_id=identity["tenant_id"],
        conversation_id=identity["conversation_id"],
    )
    blocker = repository.create_task(
        "Blocker",
        tenant_id=identity["tenant_id"],
        conversation_id=identity["conversation_id"],
        status=TaskStatus.DONE,
    )

    with session_factory() as session:
        dependency = TaskDependency(
            task_id=uuid.UUID(primary["id"]),
            depends_on_id=uuid.UUID(blocker["id"]),
        )
        session.add(dependency)
        session.commit()

    statuses = repository.dependency_statuses(primary["id"], tenant_id=identity["tenant_id"])
    assert statuses == [TaskStatus.DONE.value]
