from datetime import datetime, timezone

import pytest

try:  # pragma: no cover - optional dependency gate
    import sqlalchemy
    from sqlalchemy import create_engine, event
    from sqlalchemy.orm import sessionmaker
except Exception:  # pragma: no cover - skip when SQLAlchemy unavailable
    sqlalchemy = None
    pytestmark = pytest.mark.skip("SQLAlchemy is required for job service tests")
else:  # pragma: no cover - skip when stubbed module detected
    if not getattr(sqlalchemy, "__version__", None):
        pytestmark = pytest.mark.skip("SQLAlchemy runtime is required for job service tests")

from modules.conversation_store import Base
from modules.conversation_store.models import Conversation, Session as ConversationSession, User
from modules.job_store import JobStatus
from modules.job_store.repository import JobStoreRepository
from modules.job_store.service import (
    JobDependencyError,
    JobService,
    JobTransitionError,
)
from modules.task_store import TaskStatus
from modules.task_store.repository import TaskStoreRepository
from modules.task_store.service import TaskService


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
def job_repository(session_factory):
    repo = JobStoreRepository(session_factory)
    repo.create_schema()
    return repo


@pytest.fixture()
def task_repository(session_factory):
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
def job_service(job_repository, event_recorder):
    events, emitter = event_recorder
    svc = JobService(job_repository, event_emitter=emitter)
    return svc, events


@pytest.fixture()
def task_service(task_repository):
    return TaskService(task_repository, event_emitter=lambda *_args, **_kwargs: None)


def _create_job(job_service, identity, job_repository):
    svc, _events = job_service
    record = svc.create_job(
        "Lifecycle Job",
        tenant_id=identity["tenant_id"],
        conversation_id=identity["conversation_id"],
        owner_id=identity["user_id"],
        metadata={"personas": ["Navigator"]},
    )
    job_repository.upsert_schedule(
        record["id"],
        tenant_id=identity["tenant_id"],
        schedule_type="cron",
        expression="0 0 * * *",
        next_run_at=datetime.now(timezone.utc),
    )
    return record


def test_job_transition_sequence_enforces_rules(job_service, job_repository, identity, monkeypatch):
    svc, events = job_service
    metrics: list[dict] = []

    def capture_metric(**payload):
        metrics.append(payload)

    monkeypatch.setattr(
        "modules.job_store.service.record_job_lifecycle_event",
        capture_metric,
    )

    record = _create_job(job_service, identity, job_repository)

    with pytest.raises(JobTransitionError):
        svc.transition_job(
            record["id"],
            tenant_id=identity["tenant_id"],
            target_status=JobStatus.SUCCEEDED,
        )

    scheduled = svc.transition_job(
        record["id"], tenant_id=identity["tenant_id"], target_status=JobStatus.SCHEDULED
    )
    assert scheduled["status"] == JobStatus.SCHEDULED.value

    running = svc.transition_job(
        record["id"], tenant_id=identity["tenant_id"], target_status=JobStatus.RUNNING
    )
    assert running["status"] == JobStatus.RUNNING.value

    completed = svc.transition_job(
        record["id"], tenant_id=identity["tenant_id"], target_status=JobStatus.SUCCEEDED
    )
    assert completed["status"] == JobStatus.SUCCEEDED.value

    assert any(event for event in events if event[0] == "job.status_changed")
    assert any(metric for metric in metrics if metric.get("event") == "completed")


def test_job_dependencies_gate_progress(job_service, job_repository, task_service, identity):
    svc, _events = job_service
    job_record = _create_job(job_service, identity, job_repository)

    blocker = task_service.create_task(
        "Blocking task",
        tenant_id=identity["tenant_id"],
        conversation_id=identity["conversation_id"],
    )

    job_repository.attach_task(
        job_record["id"],
        blocker["id"],
        tenant_id=identity["tenant_id"],
    )

    svc.transition_job(
        job_record["id"], tenant_id=identity["tenant_id"], target_status=JobStatus.SCHEDULED
    )

    with pytest.raises(JobDependencyError):
        svc.transition_job(
            job_record["id"],
            tenant_id=identity["tenant_id"],
            target_status=JobStatus.RUNNING,
        )

    task_service.transition_task(
        blocker["id"], tenant_id=identity["tenant_id"], target_status=TaskStatus.READY
    )
    task_service.transition_task(
        blocker["id"], tenant_id=identity["tenant_id"], target_status=TaskStatus.IN_PROGRESS
    )
    task_service.transition_task(
        blocker["id"], tenant_id=identity["tenant_id"], target_status=TaskStatus.REVIEW
    )
    task_service.transition_task(
        blocker["id"], tenant_id=identity["tenant_id"], target_status=TaskStatus.DONE
    )

    running = svc.transition_job(
        job_record["id"], tenant_id=identity["tenant_id"], target_status=JobStatus.RUNNING
    )
    assert running["status"] == JobStatus.RUNNING.value
    assert svc.dependencies_complete(job_record["id"], tenant_id=identity["tenant_id"])


def test_update_job_no_op_owner_does_not_emit(job_service, identity, monkeypatch):
    svc, events = job_service
    record = svc.create_job(
        "Owned job",
        tenant_id=identity["tenant_id"],
        conversation_id=identity["conversation_id"],
        owner_id=identity["user_id"],
        metadata={"personas": ["Navigator"]},
    )
    initial_event_count = len(events)

    metrics: list[dict] = []

    def capture_metric(**payload):
        metrics.append(payload)

    monkeypatch.setattr(
        "modules.job_store.service.record_job_lifecycle_event",
        capture_metric,
    )

    updated = svc.update_job(
        record["id"],
        tenant_id=identity["tenant_id"],
        changes={"owner_id": identity["user_id"]},
    )

    assert updated["owner_id"] == record["owner_id"]
    assert updated.get("events") == []
    assert len(events) == initial_event_count
    assert metrics == []


def test_roster_required_for_activation(job_service, job_repository, identity):
    svc, _events = job_service
    record = svc.create_job(
        "Rosterless",
        tenant_id=identity["tenant_id"],
        conversation_id=identity["conversation_id"],
    )
    job_repository.upsert_schedule(
        record["id"],
        tenant_id=identity["tenant_id"],
        schedule_type="cron",
        expression="0 0 * * *",
        next_run_at=datetime.now(timezone.utc),
    )

    with pytest.raises(JobTransitionError):
        svc.transition_job(
            record["id"],
            tenant_id=identity["tenant_id"],
            target_status=JobStatus.SCHEDULED,
        )
