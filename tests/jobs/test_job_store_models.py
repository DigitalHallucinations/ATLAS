from __future__ import annotations

import pytest

sqlalchemy = pytest.importorskip("sqlalchemy", reason="SQLAlchemy is required for job store tests.")
if getattr(sqlalchemy, "__version__", None) is None:
    pytest.skip("SQLAlchemy not installed; skipping job store model tests.", allow_module_level=True)

from sqlalchemy import create_engine, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from modules.conversation_store.models import Conversation, Session as ConversationSession, User, Base
from modules.job_store.models import (
    Job,
    JobAssignment,
    JobAssignmentStatus,
    JobEvent,
    JobEventType,
    JobRun,
    JobRunStatus,
    JobSchedule,
    JobStatus,
    JobTaskLink,
    ensure_job_schema,
)
from modules.task_store.models import Task, ensure_task_schema


@pytest.fixture()
def session_factory(postgresql) -> sessionmaker:
    engine = create_engine(postgresql.dsn(), future=True)

    Base.metadata.create_all(
        engine,
        tables=[
            User.__table__,
            ConversationSession.__table__,
            Conversation.__table__,
        ],
    )
    ensure_task_schema(engine)
    ensure_job_schema(engine)
    factory = sessionmaker(bind=engine, future=True, class_=Session)

    try:
        yield factory
    finally:
        engine.dispose()


def _create_user(session: Session) -> User:
    user = User(display_name="Owner", meta={})
    session.add(user)
    session.flush()
    return user


def _create_conversation(session: Session, tenant_id: str) -> Conversation:
    conversation = Conversation(tenant_id=tenant_id, meta={})
    session.add(conversation)
    session.flush()
    return conversation


def _create_task(session: Session, conversation: Conversation) -> Task:
    task = Task(title="Example Task", conversation_id=conversation.id, meta={})
    session.add(task)
    session.flush()
    return task


def test_job_relationship_cascades(session_factory: sessionmaker) -> None:
    factory = session_factory
    with factory() as session:
        user = _create_user(session)
        conversation = _create_conversation(session, tenant_id="tenant")
        task = _create_task(session, conversation)

        job = Job(
            name="Example Job",
            tenant_id="tenant",
            owner_id=user.id,
            conversation_id=conversation.id,
            meta={"kind": "demo"},
        )
        job.runs.append(JobRun(run_number=1, status=JobRunStatus.RUNNING))
        job.assignments.append(
            JobAssignment(
                assignee_id=user.id,
                role="executor",
                status=JobAssignmentStatus.ACCEPTED,
            )
        )
        job.tasks.append(
            JobTaskLink(task_id=task.id, relationship_type="produces")
        )
        job.schedule = JobSchedule(schedule_type="cron", expression="* * * * *")
        job.events.append(
            JobEvent(
                event_type=JobEventType.CREATED,
                triggered_by_id=user.id,
                payload={"status": JobStatus.DRAFT.value},
            )
        )

        session.add(job)
        session.commit()
        job_id = job.id

        session.delete(job)
        session.commit()

        assert session.execute(select(JobRun).where(JobRun.job_id == job_id)).first() is None
        assert (
            session.execute(select(JobAssignment).where(JobAssignment.job_id == job_id)).first()
            is None
        )
        assert (
            session.execute(select(JobTaskLink).where(JobTaskLink.job_id == job_id)).first()
            is None
        )
        assert (
            session.execute(select(JobSchedule).where(JobSchedule.job_id == job_id)).first()
            is None
        )
        assert (
            session.execute(select(JobEvent).where(JobEvent.job_id == job_id)).first()
            is None
        )


def test_job_task_link_unique_constraint(session_factory: sessionmaker) -> None:
    factory = session_factory
    with factory() as session:
        conversation = _create_conversation(session, tenant_id="tenant")
        task = _create_task(session, conversation)
        job = Job(name="Dup Link", tenant_id="tenant")
        session.add_all([job])
        session.flush()

        session.add(JobTaskLink(job_id=job.id, task_id=task.id))
        session.flush()

        session.add(JobTaskLink(job_id=job.id, task_id=task.id))
        with pytest.raises(IntegrityError):
            session.flush()


def test_job_enum_round_trip(session_factory: sessionmaker) -> None:
    factory = session_factory
    with factory() as session:
        user = _create_user(session)
        job = Job(name="Enum Job", tenant_id="tenant", status=JobStatus.RUNNING)
        session.add(job)
        session.flush()

        run = JobRun(job_id=job.id, status=JobRunStatus.SUCCEEDED)
        assignment = JobAssignment(job_id=job.id, assignee_id=user.id)
        event = JobEvent(job_id=job.id, event_type=JobEventType.STATUS_CHANGED, payload={})
        session.add_all([run, assignment, event])
        session.commit()
        session.refresh(job)

        assert isinstance(job.status, JobStatus)
        assert job.status is JobStatus.RUNNING

        stored_run = session.execute(select(JobRun).where(JobRun.job_id == job.id)).scalar_one()
        assert isinstance(stored_run.status, JobRunStatus)
        assert stored_run.status is JobRunStatus.SUCCEEDED

        stored_assignment = (
            session.execute(select(JobAssignment).where(JobAssignment.job_id == job.id)).scalar_one()
        )
        assert isinstance(stored_assignment.status, JobAssignmentStatus)
        assert stored_assignment.status is JobAssignmentStatus.PENDING

        stored_event = session.execute(select(JobEvent).where(JobEvent.job_id == job.id)).scalar_one()
        assert isinstance(stored_event.event_type, JobEventType)
        assert stored_event.event_type is JobEventType.STATUS_CHANGED
