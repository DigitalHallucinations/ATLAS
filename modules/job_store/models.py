"""SQLAlchemy models for coordinating job workflow entities."""

from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    Column,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    inspect,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.engine import Engine
from sqlalchemy.orm import relationship

from modules.conversation_store.models import Base as ConversationBase, PortableJSON

try:  # pragma: no cover - optional dependency for task relationships
    from modules.task_store.models import Task as _Task  # noqa: F401
except Exception:  # pragma: no cover - fallback when task store unavailable
    _Task = None  # type: ignore[assignment]


Base = ConversationBase


def _uuid() -> uuid.UUID:
    return uuid.uuid4()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class JobStatus(str, enum.Enum):
    """Enumerates supported lifecycle states for a job."""

    DRAFT = "draft"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobAssignmentStatus(str, enum.Enum):
    """Enumerates statuses for job assignments."""

    PENDING = "pending"
    ACCEPTED = "accepted"
    DECLINED = "declined"
    COMPLETED = "completed"


class JobEventType(str, enum.Enum):
    """Enumerates tracked event types for job audit history."""

    CREATED = "created"
    UPDATED = "updated"
    STATUS_CHANGED = "status_changed"
    RUN = "run"
    ASSIGNMENT = "assignment"


class JobRunStatus(str, enum.Enum):
    """Enumerates execution states for job runs."""

    SCHEDULED = "scheduled"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Job(Base):
    __tablename__ = "jobs"
    __table_args__ = (
        Index("ix_jobs_tenant_status", "tenant_id", "status"),
        Index("ix_jobs_owner_status", "owner_id", "status"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    status = Column(
        Enum(JobStatus, name="job_status", validate_strings=True),
        nullable=False,
        default=JobStatus.DRAFT,
    )
    owner_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"))
    conversation_id = Column(
        UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="SET NULL")
    )
    tenant_id = Column(String(255), nullable=False, index=True)
    meta = Column("metadata", PortableJSON(), nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    updated_at = Column(
        DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow
    )

    owner = relationship("User", foreign_keys=[owner_id])
    conversation = relationship("Conversation", foreign_keys=[conversation_id])
    runs = relationship(
        "JobRun",
        back_populates="job",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    tasks = relationship(
        "JobTaskLink",
        back_populates="job",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    assignments = relationship(
        "JobAssignment",
        back_populates="job",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    schedule = relationship(
        "JobSchedule",
        back_populates="job",
        cascade="all, delete-orphan",
        passive_deletes=True,
        uselist=False,
    )
    events = relationship(
        "JobEvent",
        back_populates="job",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class JobRun(Base):
    __tablename__ = "job_runs"
    __table_args__ = (
        UniqueConstraint("job_id", "run_number", name="uq_job_run_number"),
        Index("ix_job_runs_job_started", "job_id", "started_at"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    job_id = Column(UUID(as_uuid=True), ForeignKey("jobs.id", ondelete="CASCADE"))
    run_number = Column(Integer, nullable=False, default=1)
    status = Column(
        Enum(JobRunStatus, name="job_run_status", validate_strings=True),
        nullable=False,
        default=JobRunStatus.SCHEDULED,
    )
    started_at = Column(DateTime(timezone=True), nullable=True)
    finished_at = Column(DateTime(timezone=True), nullable=True)
    meta = Column("metadata", PortableJSON(), nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    updated_at = Column(
        DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow
    )

    job = relationship("Job", back_populates="runs", foreign_keys=[job_id])


class JobTaskLink(Base):
    __tablename__ = "job_task_links"
    __table_args__ = (
        UniqueConstraint("job_id", "task_id", name="uq_job_task_link"),
        Index("ix_job_task_links_task", "task_id"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    job_id = Column(UUID(as_uuid=True), ForeignKey("jobs.id", ondelete="CASCADE"))
    task_id = Column(UUID(as_uuid=True), ForeignKey("tasks.id", ondelete="CASCADE"))
    relationship_type = Column(String(64), nullable=False, default="relates_to")
    meta = Column("metadata", PortableJSON(), nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)

    job = relationship("Job", back_populates="tasks", foreign_keys=[job_id])
    task = relationship("Task", foreign_keys=[task_id])


class JobAssignment(Base):
    __tablename__ = "job_assignments"
    __table_args__ = (
        UniqueConstraint(
            "job_id", "assignee_id", "role", name="uq_job_assignment_unique_role"
        ),
        Index("ix_job_assignments_status", "job_id", "status"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    job_id = Column(UUID(as_uuid=True), ForeignKey("jobs.id", ondelete="CASCADE"))
    assignee_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"))
    role = Column(String(64), nullable=False, default="participant")
    status = Column(
        Enum(JobAssignmentStatus, name="job_assignment_status", validate_strings=True),
        nullable=False,
        default=JobAssignmentStatus.PENDING,
    )
    meta = Column("metadata", PortableJSON(), nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    updated_at = Column(
        DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow
    )

    job = relationship("Job", back_populates="assignments", foreign_keys=[job_id])
    assignee = relationship("User", foreign_keys=[assignee_id])


class JobSchedule(Base):
    __tablename__ = "job_schedules"
    __table_args__ = (
        UniqueConstraint("job_id", name="uq_job_schedule_unique"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    job_id = Column(UUID(as_uuid=True), ForeignKey("jobs.id", ondelete="CASCADE"))
    schedule_type = Column(String(64), nullable=False, default="cron")
    expression = Column(String(255), nullable=False)
    timezone = Column(String(64), nullable=False, default="UTC")
    next_run_at = Column(DateTime(timezone=True), nullable=True)
    meta = Column("metadata", PortableJSON(), nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    updated_at = Column(
        DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow
    )

    job = relationship("Job", back_populates="schedule", foreign_keys=[job_id])


class JobEvent(Base):
    __tablename__ = "job_events"
    __table_args__ = (
        Index("ix_job_events_job_created", "job_id", "created_at"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    job_id = Column(UUID(as_uuid=True), ForeignKey("jobs.id", ondelete="CASCADE"))
    event_type = Column(
        Enum(JobEventType, name="job_event_type", validate_strings=True),
        nullable=False,
    )
    triggered_by_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"))
    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.id", ondelete="SET NULL"))
    payload = Column(PortableJSON(), nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)

    job = relationship("Job", back_populates="events", foreign_keys=[job_id])
    triggered_by = relationship("User", foreign_keys=[triggered_by_id])
    session = relationship("Session", foreign_keys=[session_id])


try:
    _JOB_TABLES = (
        Job.__table__,
        JobRun.__table__,
        JobTaskLink.__table__,
        JobAssignment.__table__,
        JobSchedule.__table__,
        JobEvent.__table__,
    )
except AttributeError:  # pragma: no cover - minimal test stubs without SQLAlchemy
    _JOB_TABLES = ()


def ensure_job_schema(engine: Engine) -> None:
    """Create job-related tables and indexes if they do not exist."""

    inspector = inspect(engine)
    existing = {name for name in inspector.get_table_names()}
    required = {table.name for table in _JOB_TABLES}
    missing = required.difference(existing)

    if missing:
        Base.metadata.create_all(
            engine,
            tables=[table for table in _JOB_TABLES if table.name in missing],
        )

    with engine.begin() as connection:
        for table in _JOB_TABLES:
            for index in table.indexes:
                index.create(bind=connection, checkfirst=True)


__all__ = [
    "Base",
    "Job",
    "JobStatus",
    "JobRun",
    "JobRunStatus",
    "JobTaskLink",
    "JobAssignment",
    "JobAssignmentStatus",
    "JobSchedule",
    "JobEvent",
    "JobEventType",
    "ensure_job_schema",
]
