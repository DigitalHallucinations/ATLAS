from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TSVECTOR
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import sessionmaker

from modules.conversation_store import Base
from modules.conversation_store.models import Conversation, Session as ConversationSession, User
from modules.job_store.models import JobEventType, JobRunStatus, JobStatus
from modules.job_store.repository import (
    JobConcurrencyError,
    JobStoreRepository,
    JobTransitionError,
)
from modules.task_store.repository import TaskStoreRepository


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
def job_repository(session_factory):
    repo = JobStoreRepository(session_factory)
    repo.create_schema()
    return repo


@pytest.fixture()
def task_repository(session_factory):
    repo = TaskStoreRepository(session_factory)
    repo.create_schema()
    return repo


def test_create_and_list_jobs_with_pagination(job_repository, identity):
    for index in range(3):
        job_repository.create_job(
            f"Job {index}",
            tenant_id=identity["tenant_id"],
            owner_id=identity["user_id"],
            conversation_id=identity["conversation_id"],
        )

    first_page = job_repository.list_jobs(tenant_id=identity["tenant_id"], limit=2)
    assert len(first_page["items"]) == 2
    assert first_page["next_cursor"] is not None

    second_page = job_repository.list_jobs(
        tenant_id=identity["tenant_id"], cursor=first_page["next_cursor"], limit=2
    )
    assert len(second_page["items"]) == 1
    assert second_page["next_cursor"] is None

    other_tenant = job_repository.list_jobs(tenant_id="tenant-2")
    assert other_tenant["items"] == []
    assert other_tenant["next_cursor"] is None


def test_update_job_uses_optimistic_concurrency(job_repository, identity):
    record = job_repository.create_job(
        "Initial Job",
        tenant_id=identity["tenant_id"],
        conversation_id=identity["conversation_id"],
    )

    updated = job_repository.update_job(
        record["id"],
        tenant_id=identity["tenant_id"],
        changes={"name": "Renamed Job", "status": JobStatus.RUNNING},
        expected_updated_at=record["updated_at"],
    )

    assert updated["name"] == "Renamed Job"
    assert updated["status"] == JobStatus.RUNNING.value

    with pytest.raises(JobConcurrencyError):
        job_repository.update_job(
            record["id"],
            tenant_id=identity["tenant_id"],
            changes={"description": "oops"},
            expected_updated_at=record["updated_at"],
        )


def test_invalid_status_transition_raises(job_repository, identity):
    record = job_repository.create_job(
        "Transition",
        tenant_id=identity["tenant_id"],
        conversation_id=identity["conversation_id"],
    )

    running = job_repository.update_job(
        record["id"],
        tenant_id=identity["tenant_id"],
        changes={"status": JobStatus.RUNNING},
        expected_updated_at=record["updated_at"],
    )

    completed = job_repository.update_job(
        record["id"],
        tenant_id=identity["tenant_id"],
        changes={"status": JobStatus.SUCCEEDED},
        expected_updated_at=running["updated_at"],
    )

    assert completed["status"] == JobStatus.SUCCEEDED.value

    with pytest.raises(JobTransitionError):
        job_repository.update_job(
            record["id"],
            tenant_id=identity["tenant_id"],
            changes={"status": JobStatus.RUNNING},
            expected_updated_at=completed["updated_at"],
        )


def test_attach_task_requires_matching_tenant(job_repository, task_repository, identity, session_factory):
    job = job_repository.create_job(
        "Linkable",
        tenant_id=identity["tenant_id"],
        conversation_id=identity["conversation_id"],
    )

    task = task_repository.create_task(
        "Task",
        tenant_id=identity["tenant_id"],
        conversation_id=identity["conversation_id"],
        owner_id=identity["user_id"],
        session_id=identity["session_id"],
    )

    link = job_repository.attach_task(
        job["id"],
        task["id"],
        tenant_id=identity["tenant_id"],
        relationship_type="blocks",
        metadata={"note": "important"},
    )

    assert link["relationship_type"] == "blocks"
    assert link["task"]["id"] == task["id"]

    linked = job_repository.list_linked_tasks(job["id"], tenant_id=identity["tenant_id"])
    assert [item["task"]["id"] for item in linked] == [task["id"]]

    with session_factory() as session:
        other_user = User(external_id="user-2")
        other_session = ConversationSession(user=other_user)
        other_conversation = Conversation(session=other_session, tenant_id="tenant-2")
        session.add_all([other_user, other_session, other_conversation])
        session.commit()
        other_identity = {
            "user_id": other_user.id,
            "session_id": other_session.id,
            "conversation_id": other_conversation.id,
            "tenant_id": other_conversation.tenant_id,
        }

    other_task = task_repository.create_task(
        "Other Task",
        tenant_id=other_identity["tenant_id"],
        conversation_id=other_identity["conversation_id"],
    )

    with pytest.raises(ValueError):
        job_repository.attach_task(
            job["id"],
            other_task["id"],
            tenant_id=identity["tenant_id"],
        )


def test_job_serialization_includes_schedule_runs_events(job_repository, identity):
    record = job_repository.create_job(
        "Schedule",
        tenant_id=identity["tenant_id"],
        conversation_id=identity["conversation_id"],
    )

    job_repository.upsert_schedule(
        record["id"],
        tenant_id=identity["tenant_id"],
        schedule_type="cron",
        expression="0 0 * * *",
        timezone_name="UTC",
        next_run_at=datetime.now(timezone.utc),
        metadata={"window": "nightly"},
    )

    job_repository.create_run(
        record["id"],
        tenant_id=identity["tenant_id"],
        status=JobRunStatus.RUNNING,
        started_at=datetime.now(timezone.utc),
        metadata={"attempt": 1},
    )

    job_repository.record_event(
        record["id"],
        tenant_id=identity["tenant_id"],
        event_type=JobEventType.UPDATED,
        payload={"note": "manual"},
    )

    loaded = job_repository.get_job(
        record["id"],
        tenant_id=identity["tenant_id"],
        with_schedule=True,
        with_runs=True,
        with_events=True,
    )

    assert loaded["schedule"]["expression"] == "0 0 * * *"
    assert loaded["runs"][0]["status"] == JobRunStatus.RUNNING.value
    event_types = {event["event_type"] for event in loaded["events"]}
    assert JobEventType.CREATED.value in event_types
    assert JobEventType.UPDATED.value in event_types


def test_run_transition_rules(job_repository, identity):
    record = job_repository.create_job(
        "Runner",
        tenant_id=identity["tenant_id"],
        conversation_id=identity["conversation_id"],
    )

    run = job_repository.create_run(
        record["id"],
        tenant_id=identity["tenant_id"],
        status=JobRunStatus.SCHEDULED,
    )

    active = job_repository.update_run(
        run["id"],
        tenant_id=identity["tenant_id"],
        changes={"status": JobRunStatus.RUNNING},
        expected_updated_at=run["updated_at"],
    )

    assert active["status"] == JobRunStatus.RUNNING.value

    finished = job_repository.update_run(
        run["id"],
        tenant_id=identity["tenant_id"],
        changes={"status": JobRunStatus.SUCCEEDED},
        expected_updated_at=active["updated_at"],
    )

    assert finished["status"] == JobRunStatus.SUCCEEDED.value

    with pytest.raises(JobTransitionError):
        job_repository.update_run(
            run["id"],
            tenant_id=identity["tenant_id"],
            changes={"status": JobRunStatus.RUNNING},
            expected_updated_at=finished["updated_at"],
        )


@compiles(JSONB, "sqlite")
def _compile_jsonb_sqlite(_type, compiler, **_kwargs):  # pragma: no cover - dialect shim
    return "JSON"


@compiles(TSVECTOR, "sqlite")
def _compile_tsvector_sqlite(_type, compiler, **_kwargs):  # pragma: no cover - dialect shim
    return "TEXT"


@compiles(ARRAY, "sqlite")
def _compile_array_sqlite(_type, compiler, **_kwargs):  # pragma: no cover - dialect shim
    return "BLOB"

