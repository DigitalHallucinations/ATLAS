"""SQLAlchemy models for linking calendar events to jobs and tasks.

These models define the database schema for tracking relationships between
calendar events and job/task entities, enabling bidirectional navigation
and automatic event updates when jobs/tasks change.

Author: ATLAS Team
Date: Jan 8, 2026
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any, Dict, Optional, TYPE_CHECKING
from uuid import UUID

from sqlalchemy import (
    DateTime,
    Enum,
    ForeignKey,
    Index,
    String,
    Table,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from modules.conversation_store.models import Base as ConversationBase
from modules.conversation_store.models import GUID, PortableJSON
from modules.store_common.model_utils import generate_uuid, utcnow

if TYPE_CHECKING:
    from .models import CalendarEventModel
    from modules.job_store.models import Job
    from modules.task_store.models import Task

Base = ConversationBase


class LinkType(str, enum.Enum):
    """Type of relationship between calendar event and linked entity."""
    
    # Event was auto-created from job/task
    AUTO_CREATED = "auto_created"
    
    # User manually linked existing event to job/task
    MANUAL = "manual"
    
    # Event represents a deadline for the job/task
    DEADLINE = "deadline"
    
    # Event is a scheduled work block for the job/task
    WORK_BLOCK = "work_block"
    
    # Event is a milestone for the job/task
    MILESTONE = "milestone"
    
    # Event is a review/check-in for the job/task
    REVIEW = "review"


class SyncBehavior(str, enum.Enum):
    """How changes should propagate between linked entities."""
    
    # No automatic sync - links are informational only
    NONE = "none"
    
    # Update event when job/task changes (one-way)
    FROM_SOURCE = "from_source"
    
    # Update job/task when event changes (one-way reverse)
    FROM_EVENT = "from_event"
    
    # Sync changes in both directions
    BIDIRECTIONAL = "bidirectional"


class JobEventLink(Base):
    """Links a calendar event to a job.
    
    Enables:
    - Auto-creating calendar events for scheduled jobs
    - Showing job status on calendar events
    - Navigating from calendar to job details
    - Updating events when job status changes
    """
    
    __tablename__ = "calendar_job_links"
    __table_args__ = (
        UniqueConstraint("event_id", "job_id", name="uq_calendar_job_link"),
        Index("ix_calendar_job_links_event", "event_id"),
        Index("ix_calendar_job_links_job", "job_id"),
    )
    
    id: Mapped[UUID] = mapped_column(GUID(), primary_key=True, default=generate_uuid)
    
    event_id: Mapped[UUID] = mapped_column(
        GUID(),
        ForeignKey("calendar_events.id", ondelete="CASCADE"),
        nullable=False,
    )
    job_id: Mapped[UUID] = mapped_column(
        GUID(),
        ForeignKey("jobs.id", ondelete="CASCADE"),
        nullable=False,
    )
    
    # Link metadata
    link_type: Mapped[LinkType] = mapped_column(
        Enum(LinkType, name="calendar_link_type", validate_strings=True),
        nullable=False,
        default=LinkType.AUTO_CREATED,
    )
    sync_behavior: Mapped[SyncBehavior] = mapped_column(
        Enum(SyncBehavior, name="calendar_sync_behavior", validate_strings=True),
        nullable=False,
        default=SyncBehavior.FROM_SOURCE,
    )
    
    # Optional description of why this link exists
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Additional metadata (e.g., which job run this event represents)
    meta: Mapped[Dict[str, Any]] = mapped_column(
        "metadata", PortableJSON(), nullable=False, default=dict
    )
    
    # Audit timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow
    )
    created_by: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True
    )  # user_id or "system"
    
    # Relationships
    event: Mapped["CalendarEventModel"] = relationship(
        "CalendarEventModel",
        foreign_keys=[event_id],
        lazy="joined",
    )
    job: Mapped["Job"] = relationship(
        "Job",
        foreign_keys=[job_id],
        lazy="joined",
    )


class TaskEventLink(Base):
    """Links a calendar event to a task.
    
    Enables:
    - Auto-creating calendar events for task deadlines
    - Showing task status/progress on calendar events
    - Navigating from calendar to task details
    - Scheduling work blocks for tasks
    """
    
    __tablename__ = "calendar_task_links"
    __table_args__ = (
        UniqueConstraint("event_id", "task_id", name="uq_calendar_task_link"),
        Index("ix_calendar_task_links_event", "event_id"),
        Index("ix_calendar_task_links_task", "task_id"),
    )
    
    id: Mapped[UUID] = mapped_column(GUID(), primary_key=True, default=generate_uuid)
    
    event_id: Mapped[UUID] = mapped_column(
        GUID(),
        ForeignKey("calendar_events.id", ondelete="CASCADE"),
        nullable=False,
    )
    task_id: Mapped[UUID] = mapped_column(
        GUID(),
        ForeignKey("tasks.id", ondelete="CASCADE"),
        nullable=False,
    )
    
    # Link metadata
    link_type: Mapped[LinkType] = mapped_column(
        Enum(LinkType, name="calendar_link_type", validate_strings=True, create_constraint=False),
        nullable=False,
        default=LinkType.AUTO_CREATED,
    )
    sync_behavior: Mapped[SyncBehavior] = mapped_column(
        Enum(SyncBehavior, name="calendar_sync_behavior", validate_strings=True, create_constraint=False),
        nullable=False,
        default=SyncBehavior.FROM_SOURCE,
    )
    
    # Optional description of why this link exists
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Additional metadata
    meta: Mapped[Dict[str, Any]] = mapped_column(
        "metadata", PortableJSON(), nullable=False, default=dict
    )
    
    # Audit timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow
    )
    created_by: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True
    )  # user_id or "system"
    
    # Relationships
    event: Mapped["CalendarEventModel"] = relationship(
        "CalendarEventModel",
        foreign_keys=[event_id],
        lazy="joined",
    )
    task: Mapped["Task"] = relationship(
        "Task",
        foreign_keys=[task_id],
        lazy="joined",
    )


def ensure_link_schema(engine) -> None:
    """Create link tables and indexes if they don't exist.
    
    This is idempotent and safe to call multiple times.
    """
    # Cast needed for Pylance - __table__ is FromClause but create_all expects Table
    tables: list[Table] = [
        JobEventLink.__table__,  # type: ignore[list-item]
        TaskEventLink.__table__,  # type: ignore[list-item]
    ]
    Base.metadata.create_all(engine, tables=tables)
