"""Pure Python dataclasses and enums for the ATLAS Master Calendar.

This module defines the data transfer objects used at the API layer,
separate from SQLAlchemy models to maintain clean architecture.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
from uuid import UUID


# ============================================================================
# Enumerations
# ============================================================================


class EventStatus(str, enum.Enum):
    """Event confirmation status."""

    CONFIRMED = "confirmed"
    TENTATIVE = "tentative"
    CANCELLED = "cancelled"


class EventVisibility(str, enum.Enum):
    """Event visibility level."""

    PUBLIC = "public"
    PRIVATE = "private"
    CONFIDENTIAL = "confidential"


class BusyStatus(str, enum.Enum):
    """Free/busy indicator for events."""

    BUSY = "busy"
    FREE = "free"
    TENTATIVE = "tentative"
    OUT_OF_OFFICE = "out_of_office"


class SyncDirection(str, enum.Enum):
    """Synchronization direction for calendars and categories."""

    PULL_ONLY = "pull_only"
    PUSH_ONLY = "push_only"
    BIDIRECTIONAL = "bidirectional"


class SyncStatus(str, enum.Enum):
    """Event synchronization status."""

    SYNCED = "synced"
    PENDING = "pending"
    CONFLICT = "conflict"
    ERROR = "error"


class ReminderMethod(str, enum.Enum):
    """Notification method for reminders."""

    POPUP = "popup"
    SOUND = "sound"
    EMAIL = "email"
    NOTIFICATION = "notification"


class AttendeeStatus(str, enum.Enum):
    """Attendee response status."""

    NEEDS_ACTION = "needs_action"
    ACCEPTED = "accepted"
    DECLINED = "declined"
    TENTATIVE = "tentative"


class AttendeeRole(str, enum.Enum):
    """Attendee role in an event."""

    REQUIRED = "required"
    OPTIONAL = "optional"
    CHAIR = "chair"


# ============================================================================
# Dataclasses
# ============================================================================


@dataclass(frozen=True)
class Attendee:
    """Event attendee information."""

    email: str
    name: Optional[str] = None
    status: AttendeeStatus = AttendeeStatus.NEEDS_ACTION
    role: AttendeeRole = AttendeeRole.REQUIRED

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        return {
            "email": self.email,
            "name": self.name,
            "status": self.status.value,
            "role": self.role.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Attendee:
        """Deserialize from dictionary."""
        return cls(
            email=data.get("email", ""),
            name=data.get("name"),
            status=AttendeeStatus(data.get("status", "needs_action")),
            role=AttendeeRole(data.get("role", "required")),
        )


@dataclass(frozen=True)
class CalendarReminder:
    """Event reminder configuration."""

    minutes_before: int
    method: ReminderMethod = ReminderMethod.NOTIFICATION

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        return {
            "minutes_before": self.minutes_before,
            "method": self.method.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CalendarReminder:
        """Deserialize from dictionary."""
        return cls(
            minutes_before=int(data.get("minutes_before", 15)),
            method=ReminderMethod(data.get("method", "notification")),
        )


@dataclass
class CalendarCategory:
    """Calendar category / calendar grouping."""

    id: UUID
    name: str
    slug: str
    color: str = "#4285F4"
    icon: Optional[str] = None
    description: Optional[str] = None
    is_builtin: bool = False
    is_visible: bool = True
    is_default: bool = False
    is_readonly: bool = False
    sort_order: int = 0
    sync_direction: SyncDirection = SyncDirection.BIDIRECTIONAL
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass
class CalendarEvent:
    """Calendar event with full properties."""

    id: UUID
    title: str
    start_time: datetime
    end_time: datetime

    # External sync tracking
    external_id: Optional[str] = None
    external_source: Optional[str] = None

    # Content
    description: Optional[str] = None
    location: Optional[str] = None

    # Timing
    timezone: str = "UTC"
    is_all_day: bool = False

    # Recurrence
    recurrence_rule: Optional[str] = None
    recurrence_id: Optional[datetime] = None
    original_start: Optional[datetime] = None

    # Organization
    category_id: Optional[UUID] = None
    tags: list[str] = field(default_factory=list)
    color_override: Optional[str] = None

    # Status
    status: EventStatus = EventStatus.CONFIRMED
    visibility: EventVisibility = EventVisibility.PUBLIC
    busy_status: BusyStatus = BusyStatus.BUSY

    # Attendees
    organizer: Optional[dict[str, Any]] = None
    attendees: list[Attendee] = field(default_factory=list)

    # Reminders
    reminders: list[CalendarReminder] = field(default_factory=list)

    # Metadata
    url: Optional[str] = None
    attachments: list[dict[str, Any]] = field(default_factory=list)
    custom_properties: dict[str, Any] = field(default_factory=dict)

    # Sync tracking
    etag: Optional[str] = None
    sync_status: SyncStatus = SyncStatus.SYNCED
    last_synced_at: Optional[datetime] = None

    # Audit
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass
class ImportMapping:
    """Mapping from external calendar to ATLAS category."""

    id: UUID
    source_type: str  # google | outlook | caldav | ics
    source_account: str  # Account identifier
    source_calendar: str  # External calendar name/id
    target_category_id: UUID
    created_at: Optional[datetime] = None


@dataclass
class SyncState:
    """Synchronization state for an external calendar source."""

    id: UUID
    source_type: str
    source_account: str
    source_calendar: Optional[str] = None
    sync_token: Optional[str] = None  # For incremental sync
    last_sync_at: Optional[datetime] = None
    last_sync_status: Optional[str] = None
    error_message: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


# ============================================================================
# Built-in Category Definitions
# ============================================================================

BUILTIN_CATEGORIES = [
    {
        "name": "Work",
        "slug": "work",
        "color": "#4285F4",
        "icon": "💼",
        "is_builtin": True,
        "is_default": False,
        "is_readonly": False,
        "sort_order": 1,
    },
    {
        "name": "Personal",
        "slug": "personal",
        "color": "#34A853",
        "icon": "👤",
        "is_builtin": True,
        "is_default": True,
        "is_readonly": False,
        "sort_order": 2,
    },
    {
        "name": "Health",
        "slug": "health",
        "color": "#EA4335",
        "icon": "❤️",
        "is_builtin": True,
        "is_default": False,
        "is_readonly": False,
        "sort_order": 3,
    },
    {
        "name": "Family",
        "slug": "family",
        "color": "#FF6D01",
        "icon": "👨‍👩‍👧",
        "is_builtin": True,
        "is_default": False,
        "is_readonly": False,
        "sort_order": 4,
    },
    {
        "name": "Holidays",
        "slug": "holidays",
        "color": "#9334E6",
        "icon": "🎉",
        "is_builtin": True,
        "is_default": False,
        "is_readonly": True,
        "sort_order": 5,
    },
    {
        "name": "Birthdays",
        "slug": "birthdays",
        "color": "#E91E63",
        "icon": "🎂",
        "is_builtin": True,
        "is_default": False,
        "is_readonly": True,
        "sort_order": 6,
    },
]

# Color palette for custom categories
COLOR_PALETTE = [
    "#4285F4",  # Blue
    "#34A853",  # Green
    "#EA4335",  # Red
    "#FBBC05",  # Yellow
    "#FF6D01",  # Orange
    "#9334E6",  # Purple
    "#E91E63",  # Pink
    "#00BCD4",  # Cyan
    "#795548",  # Brown
    "#607D8B",  # Gray
    "#009688",  # Teal
    "#673AB7",  # Indigo
]


__all__ = [
    # Enums
    "EventStatus",
    "EventVisibility",
    "BusyStatus",
    "SyncDirection",
    "SyncStatus",
    "ReminderMethod",
    "AttendeeStatus",
    "AttendeeRole",
    # Dataclasses
    "Attendee",
    "CalendarReminder",
    "CalendarCategory",
    "CalendarEvent",
    "ImportMapping",
    "SyncState",
    # Constants
    "BUILTIN_CATEGORIES",
    "COLOR_PALETTE",
]
