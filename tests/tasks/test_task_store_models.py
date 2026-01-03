import pytest

sqlalchemy = pytest.importorskip("sqlalchemy", reason="SQLAlchemy is required for task store tests.")
if getattr(sqlalchemy, "__version__", None) is None:
    pytest.skip("SQLAlchemy not installed; skipping task store model tests.", allow_module_level=True)

from sqlalchemy import create_engine, select
from sqlalchemy.exc import StatementError
from sqlalchemy.orm import sessionmaker

from modules.conversation_store import Base
from modules.conversation_store.models import Conversation, Session as ConversationSession, User
from modules.task_store.models import (
    Task,
    TaskAssignment,
    TaskAssignmentStatus,
    TaskDependency,
    TaskEvent,
    TaskEventType,
    TaskStatus,
    ensure_task_schema,
)


@pytest.fixture
def engine(postgresql):
    engine = create_engine(postgresql.dsn(), future=True)

    Base.metadata.create_all(engine)
    ensure_task_schema(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def session(engine):
    factory = sessionmaker(bind=engine, future=True)
    with factory() as session:
        yield session
        session.rollback()


@pytest.fixture
def identity(session):
    user = User(external_id="user-1", display_name="Test User")
    store_session = ConversationSession(user=user)
    conversation = Conversation(session=store_session, tenant_id="tenant-1")
    session.add_all([user, store_session, conversation])
    session.commit()
    return user, store_session, conversation


def test_task_status_enum_enforced(session, identity):
    user, store_session, conversation = identity
    task = Task(
        title="Prepare summary",
        status=TaskStatus.IN_PROGRESS,
        owner=user,
        session=store_session,
        conversation=conversation,
    )
    session.add(task)
    session.commit()

    assert task.status == TaskStatus.IN_PROGRESS

    invalid_task = Task(title="Invalid", status="unknown")
    session.add(invalid_task)
    with pytest.raises((LookupError, ValueError, StatementError)):
        session.flush()
    session.rollback()


def test_task_relationship_cascades(session, identity):
    user, store_session, conversation = identity

    task = Task(title="Coordinate review", owner=user, session=store_session, conversation=conversation)
    task.assignments.append(
        TaskAssignment(
            assignee=user,
            status=TaskAssignmentStatus.ACCEPTED,
            role="owner",
        )
    )
    task.events.append(
        TaskEvent(
            event_type=TaskEventType.CREATED,
            triggered_by=user,
            session=store_session,
            payload={"details": "task created"},
        )
    )

    session.add(task)
    session.commit()

    session.delete(task)
    session.commit()

    remaining_assignments = session.execute(select(TaskAssignment)).scalars().all()
    remaining_events = session.execute(select(TaskEvent)).scalars().all()
    assert remaining_assignments == []
    assert remaining_events == []


def test_dependency_cascade(session, identity):
    user, store_session, conversation = identity

    blocker = Task(title="Draft outline", owner=user, session=store_session, conversation=conversation)
    blocked = Task(
        title="Write report",
        owner=user,
        session=store_session,
        conversation=conversation,
        status=TaskStatus.READY,
    )
    blocked.dependencies.append(TaskDependency(depends_on=blocker))

    session.add_all([blocker, blocked])
    session.commit()

    session.delete(blocker)
    session.commit()

    dependencies = session.execute(select(TaskDependency)).scalars().all()
    assert dependencies == []

    remaining_task = session.execute(select(Task).where(Task.title == "Write report")).scalar_one()
    assert remaining_task.status == TaskStatus.READY
