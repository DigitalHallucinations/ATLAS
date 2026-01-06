"""SQLAlchemy models for the ATLAS Master Calendar store.

These models define the database schema for calendar categories, events,
import mappings, and sync state tracking.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import ARRAY as _PG_ARRAY
from sqlalchemy.dialects.postgresql import TSVECTOR as _PG_TSVECTOR
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.schema import DDL
from sqlalchemy import event as sa_event

from modules.conversation_store.models import Base as ConversationBase
from modules.conversation_store.models import GUID, PortableJSON, TextSearchVector
from modules.store_common.model_utils import generate_uuid, utcnow

from .dataclasses import (
    EventStatus,
    EventVisibility,
    BusyStatus,
    SyncDirection,
    SyncStatus,
)

Base = ConversationBase


class PortableArray(PortableJSON):
    """Dialect-aware array type that uses ARRAY for PostgreSQL, JSON elsewhere."""

    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(_PG_ARRAY(String))
        # Fall back to JSON for other dialects
        return super().load_dialect_impl(dialect)

    def process_bind_param(self, value, dialect):
        if value is None:
            return []
        if dialect.name == "postgresql":
            return list(value) if value else []
        # For non-PostgreSQL, store as JSON
        return list(value) if value else []

    def process_result_value(self, value, dialect):
        if value is None:
            return []
        return list(value)


class CalendarCategoryModel(Base):
    """Calendar category model for organizing events."""

    __tablename__ = "calendar_categories"
    __table_args__ = (
        UniqueConstraint("slug", name="uq_calendar_category_slug"),
        Index("ix_calendar_categories_is_default", "is_default"),
        Index("ix_calendar_categories_sort_order", "sort_order"),
    )

    id: Mapped[UUID] = mapped_column(GUID(), primary_key=True, default=generate_uuid)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    slug: Mapped[str] = mapped_column(String(100), nullable=False)
    color: Mapped[str] = mapped_column(String(7), nullable=False, default="#4285F4")
    icon: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    is_builtin: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_visible: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    is_default: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_readonly: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    sort_order: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    sync_direction: Mapped[SyncDirection] = mapped_column(
        Enum(SyncDirection, name="sync_direction", validate_strings=True),
        nullable=False,
        default=SyncDirection.BIDIRECTIONAL,
    )

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow
    )

    # Relationships
    events: Mapped[List["CalendarEventModel"]] = relationship(
        "CalendarEventModel",
        back_populates="category",
        cascade="all",
        passive_deletes=True,
    )
    import_mappings: Mapped[List["CalendarImportMappingModel"]] = relationship(
        "CalendarImportMappingModel",
        back_populates="target_category",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class CalendarEventModel(Base):
    """Calendar event model with full RFC 5545 property support."""

    __tablename__ = "calendar_events"
    __table_args__ = (
        Index("ix_calendar_events_start", "start_time"),
        Index("ix_calendar_events_end", "end_time"),
        Index("ix_calendar_events_category", "category_id"),
        Index(
            "ix_calendar_events_external",
            "external_source",
            "external_id",
        ),
        Index("ix_calendar_events_date_range", "start_time", "end_time"),
        Index("ix_calendar_events_sync_status", "sync_status"),
    )

    id: Mapped[UUID] = mapped_column(GUID(), primary_key=True, default=generate_uuid)

    # External sync tracking
    external_id: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    external_source: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    # Content
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    location: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    # Timing
    start_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    end_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    timezone: Mapped[str] = mapped_column(String(100), nullable=False, default="UTC")
    is_all_day: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # Recurrence (RFC 5545 RRULE)
    recurrence_rule: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    recurrence_id: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    original_start: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Organization
    category_id: Mapped[Optional[UUID]] = mapped_column(
        GUID(),
        ForeignKey("calendar_categories.id", ondelete="SET NULL"),
        nullable=True,
    )
    tags: Mapped[List[str]] = mapped_column(PortableArray(), nullable=False, default=list)
    color_override: Mapped[Optional[str]] = mapped_column(String(7), nullable=True)

    # Status
    status: Mapped[EventStatus] = mapped_column(
        Enum(EventStatus, name="event_status", validate_strings=True),
        nullable=False,
        default=EventStatus.CONFIRMED,
    )
    visibility: Mapped[EventVisibility] = mapped_column(
        Enum(EventVisibility, name="event_visibility", validate_strings=True),
        nullable=False,
        default=EventVisibility.PUBLIC,
    )
    busy_status: Mapped[BusyStatus] = mapped_column(
        Enum(BusyStatus, name="busy_status", validate_strings=True),
        nullable=False,
        default=BusyStatus.BUSY,
    )

    # Attendees (stored as JSONB)
    organizer: Mapped[Optional[Dict[str, Any]]] = mapped_column(PortableJSON(), nullable=True)
    attendees: Mapped[List[Dict[str, Any]]] = mapped_column(PortableJSON(), nullable=False, default=list)

    # Reminders (stored as JSONB array)
    reminders: Mapped[List[Dict[str, Any]]] = mapped_column(PortableJSON(), nullable=False, default=list)

    # Metadata
    url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    attachments: Mapped[List[Dict[str, Any]]] = mapped_column(PortableJSON(), nullable=False, default=list)
    custom_properties: Mapped[Dict[str, Any]] = mapped_column(PortableJSON(), nullable=False, default=dict)

    # Sync tracking
    etag: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    sync_status: Mapped[SyncStatus] = mapped_column(
        Enum(SyncStatus, name="sync_status", validate_strings=True),
        nullable=False,
        default=SyncStatus.SYNCED,
    )
    last_synced_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Full-text search vector (PostgreSQL-specific, computed column)
    search_vector: Mapped[Optional[Any]] = mapped_column(TextSearchVector(), nullable=True)

    # Audit timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow
    )

    # Relationships
    category: Mapped[Optional["CalendarCategoryModel"]] = relationship(
        "CalendarCategoryModel",
        back_populates="events",
        foreign_keys=[category_id],
    )


class CalendarImportMappingModel(Base):
    """Mapping from external calendar sources to ATLAS categories."""

    __tablename__ = "calendar_import_mappings"
    __table_args__ = (
        UniqueConstraint(
            "source_type",
            "source_account",
            "source_calendar",
            name="uq_calendar_import_mapping",
        ),
        Index(
            "ix_calendar_import_mappings_source",
            "source_type",
            "source_account",
        ),
    )

    id: Mapped[UUID] = mapped_column(GUID(), primary_key=True, default=generate_uuid)
    source_type: Mapped[str] = mapped_column(String(50), nullable=False)  # google | outlook | caldav | ics
    source_account: Mapped[str] = mapped_column(String(200), nullable=False)  # Account identifier
    source_calendar: Mapped[str] = mapped_column(String(200), nullable=False)  # External calendar name/id
    target_category_id: Mapped[UUID] = mapped_column(
        GUID(),
        ForeignKey("calendar_categories.id", ondelete="CASCADE"),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)

    # Relationships
    target_category: Mapped["CalendarCategoryModel"] = relationship(
        "CalendarCategoryModel",
        back_populates="import_mappings",
        foreign_keys=[target_category_id],
    )


class CalendarSyncStateModel(Base):
    """Synchronization state tracking for external calendar sources."""

    __tablename__ = "calendar_sync_state"
    __table_args__ = (
        UniqueConstraint(
            "source_type",
            "source_account",
            "source_calendar",
            name="uq_calendar_sync_state",
        ),
        Index("ix_calendar_sync_state_source", "source_type", "source_account"),
        Index("ix_calendar_sync_state_last_sync", "last_sync_at"),
    )

    id: Mapped[UUID] = mapped_column(GUID(), primary_key=True, default=generate_uuid)
    source_type: Mapped[str] = mapped_column(String(50), nullable=False)
    source_account: Mapped[str] = mapped_column(String(200), nullable=False)
    source_calendar: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    sync_token: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # For incremental sync (delta tokens)
    last_sync_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    last_sync_status: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow
    )


# ============================================================================
# Schema creation helpers
# ============================================================================


def _create_search_trigger_ddl() -> DDL:
    """Create PostgreSQL trigger for maintaining search_vector."""
    return DDL(
        """
        CREATE OR REPLACE FUNCTION calendar_events_search_trigger()
        RETURNS trigger AS $$
        BEGIN
            NEW.search_vector :=
                setweight(to_tsvector('english', coalesce(NEW.title, '')), 'A') ||
                setweight(to_tsvector('english', coalesce(NEW.description, '')), 'B') ||
                setweight(to_tsvector('english', coalesce(NEW.location, '')), 'C');
            RETURN NEW;
        END
        $$ LANGUAGE plpgsql;

        DROP TRIGGER IF EXISTS calendar_events_search_update ON calendar_events;
        CREATE TRIGGER calendar_events_search_update
            BEFORE INSERT OR UPDATE ON calendar_events
            FOR EACH ROW EXECUTE FUNCTION calendar_events_search_trigger();
        """
    )


def _create_search_index_ddl() -> DDL:
    """Create GIN index for full-text search."""
    return DDL(
        """
        CREATE INDEX IF NOT EXISTS ix_calendar_events_search
        ON calendar_events USING GIN(search_vector);
        """
    )


def ensure_calendar_schema(engine: Engine) -> None:
    """Ensure PostgreSQL-specific schema elements exist.

    This creates the search trigger and GIN index for full-text search
    when using PostgreSQL.
    """
    if engine.dialect.name != "postgresql":
        return

    with engine.connect() as conn:
        conn.execute(_create_search_trigger_ddl())
        conn.execute(_create_search_index_ddl())
        conn.commit()


__all__ = [
    "Base",
    "CalendarCategoryModel",
    "CalendarEventModel",
    "CalendarImportMappingModel",
    "CalendarSyncStateModel",
    "ensure_calendar_schema",
]
