"""SQLAlchemy models for coordinating task workflow entities."""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Tuple, cast
from uuid import UUID

from sqlalchemy import (
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    String,
    Table,
    Text,
    UniqueConstraint,
    inspect,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Mapped, mapped_column, relationship

from modules.conversation_store.models import Base as ConversationBase
from modules.conversation_store.models import GUID, PortableJSON
from modules.store_common.model_utils import generate_uuid, utcnow

if TYPE_CHECKING:
    from modules.conversation_store.models import Conversation, Session, User

Base = ConversationBase
class TaskStatus(str, enum.Enum):
    """Enumerates supported lifecycle states for a task."""

    DRAFT = "draft"
    READY = "ready"
    IN_PROGRESS = "in_progress"
    REVIEW = "review"
    DONE = "done"
    CANCELLED = "cancelled"


class TaskAssignmentStatus(str, enum.Enum):
    """Enumerates statuses for task assignments."""

    PENDING = "pending"
    ACCEPTED = "accepted"
    DECLINED = "declined"
    COMPLETED = "completed"


class TaskEventType(str, enum.Enum):
    """Enumerates tracked event types for task audit history."""

    CREATED = "created"
    UPDATED = "updated"
    STATUS_CHANGED = "status_changed"
    COMMENT = "comment"
    ASSIGNMENT = "assignment"


class Task(Base):
    __tablename__ = "tasks"
    __table_args__ = (
        Index("ix_tasks_conversation_status", "conversation_id", "status"),
        Index("ix_tasks_owner_status", "owner_id", "status"),
    )

    id: Mapped[UUID] = mapped_column(GUID(), primary_key=True, default=generate_uuid)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    status: Mapped[TaskStatus] = mapped_column(
        Enum(TaskStatus, name="task_status", validate_strings=True),
        nullable=False,
        default=TaskStatus.DRAFT,
    )
    priority: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    owner_id: Mapped[Optional[UUID]] = mapped_column(GUID(), ForeignKey("users.id", ondelete="SET NULL"))
    session_id: Mapped[Optional[UUID]] = mapped_column(GUID(), ForeignKey("sessions.id", ondelete="SET NULL"))
    conversation_id: Mapped[Optional[UUID]] = mapped_column(
        GUID(), ForeignKey("conversations.id", ondelete="SET NULL")
    )
    meta: Mapped[Dict[str, Any]] = mapped_column("metadata", PortableJSON(), nullable=False, default=dict)
    due_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow
    )

    owner: Mapped[Optional["User"]] = relationship("User", foreign_keys=[owner_id])
    session: Mapped[Optional["Session"]] = relationship("Session", foreign_keys=[session_id])
    conversation: Mapped[Optional["Conversation"]] = relationship("Conversation", foreign_keys=[conversation_id])
    assignments: Mapped[List["TaskAssignment"]] = relationship(
        "TaskAssignment",
        back_populates="task",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    dependencies: Mapped[List["TaskDependency"]] = relationship(
        "TaskDependency",
        back_populates="task",
        cascade="all, delete-orphan",
        passive_deletes=True,
        foreign_keys="TaskDependency.task_id",
    )
    dependents: Mapped[List["TaskDependency"]] = relationship(
        "TaskDependency",
        back_populates="depends_on",
        cascade="all",
        passive_deletes=True,
        foreign_keys="TaskDependency.depends_on_id",
    )
    events: Mapped[List["TaskEvent"]] = relationship(
        "TaskEvent",
        back_populates="task",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class TaskAssignment(Base):
    __tablename__ = "task_assignments"
    __table_args__ = (
        UniqueConstraint(
            "task_id", "assignee_id", "role", name="uq_task_assignment_unique_role"
        ),
        Index("ix_task_assignments_status", "task_id", "status"),
    )

    id: Mapped[UUID] = mapped_column(GUID(), primary_key=True, default=generate_uuid)
    task_id: Mapped[Optional[UUID]] = mapped_column(GUID(), ForeignKey("tasks.id", ondelete="CASCADE"))
    assignee_id: Mapped[Optional[UUID]] = mapped_column(GUID(), ForeignKey("users.id", ondelete="SET NULL"))
    role: Mapped[str] = mapped_column(String(64), nullable=False, default="participant")
    status: Mapped[TaskAssignmentStatus] = mapped_column(
        Enum(TaskAssignmentStatus, name="task_assignment_status", validate_strings=True),
        nullable=False,
        default=TaskAssignmentStatus.PENDING,
    )
    meta: Mapped[Dict[str, Any]] = mapped_column("metadata", PortableJSON(), nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow
    )

    task: Mapped[Optional["Task"]] = relationship("Task", back_populates="assignments", foreign_keys=[task_id])
    assignee: Mapped[Optional["User"]] = relationship("User", foreign_keys=[assignee_id])


class TaskDependency(Base):
    __tablename__ = "task_dependencies"
    __table_args__ = (
        UniqueConstraint("task_id", "depends_on_id", name="uq_task_dependency_unique"),
        Index("ix_task_dependencies_depends_on", "depends_on_id"),
    )

    id: Mapped[UUID] = mapped_column(GUID(), primary_key=True, default=generate_uuid)
    task_id: Mapped[Optional[UUID]] = mapped_column(GUID(), ForeignKey("tasks.id", ondelete="CASCADE"))
    depends_on_id: Mapped[Optional[UUID]] = mapped_column(
        GUID(), ForeignKey("tasks.id", ondelete="CASCADE")
    )
    relationship_type: Mapped[str] = mapped_column(String(64), nullable=False, default="blocks")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)

    task: Mapped[Optional["Task"]] = relationship(
        "Task",
        back_populates="dependencies",
        foreign_keys=[task_id],
    )
    depends_on: Mapped[Optional["Task"]] = relationship(
        "Task",
        back_populates="dependents",
        foreign_keys=[depends_on_id],
    )


class TaskEvent(Base):
    __tablename__ = "task_events"
    __table_args__ = (
        Index("ix_task_events_task_created", "task_id", "created_at"),
    )

    id: Mapped[UUID] = mapped_column(GUID(), primary_key=True, default=generate_uuid)
    task_id: Mapped[Optional[UUID]] = mapped_column(GUID(), ForeignKey("tasks.id", ondelete="CASCADE"))
    event_type: Mapped[TaskEventType] = mapped_column(
        Enum(TaskEventType, name="task_event_type", validate_strings=True),
        nullable=False,
    )
    triggered_by_id: Mapped[Optional[UUID]] = mapped_column(
        GUID(), ForeignKey("users.id", ondelete="SET NULL")
    )
    session_id: Mapped[Optional[UUID]] = mapped_column(GUID(), ForeignKey("sessions.id", ondelete="SET NULL"))
    payload: Mapped[Dict[str, Any]] = mapped_column(PortableJSON(), nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)

    task: Mapped[Optional["Task"]] = relationship("Task", back_populates="events", foreign_keys=[task_id])
    triggered_by: Mapped[Optional["User"]] = relationship("User", foreign_keys=[triggered_by_id])
    session: Mapped[Optional["Session"]] = relationship("Session", foreign_keys=[session_id])


try:
    _TASK_TABLES: Tuple[Table, ...] = cast(Tuple[Table, ...], (
        Task.__table__,
        TaskAssignment.__table__,
        TaskDependency.__table__,
        TaskEvent.__table__,
    ))
except AttributeError:  # pragma: no cover - minimal test stubs without SQLAlchemy
    _TASK_TABLES = ()


def ensure_task_schema(engine: Engine) -> None:
    """Create task-related tables and indexes if they do not exist."""

    inspector = inspect(engine)
    existing = {name for name in inspector.get_table_names()}
    required = {table.name for table in _TASK_TABLES}
    missing = required.difference(existing)

    if missing:
        Base.metadata.create_all(
            engine,
            tables=[table for table in _TASK_TABLES if table.name in missing],
        )

    # Ensure indexes exist even if tables were created previously without them.
    with engine.begin() as connection:
        for table in _TASK_TABLES:
            for index in table.indexes:
                index.create(bind=connection, checkfirst=True)


__all__ = [
    "Base",
    "Task",
    "TaskStatus",
    "TaskAssignment",
    "TaskAssignmentStatus",
    "TaskDependency",
    "TaskEvent",
    "TaskEventType",
    "ensure_task_schema",
]
