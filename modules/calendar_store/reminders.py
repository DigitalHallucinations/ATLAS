"""Reminder utilities for ATLAS Calendar.

Provides reminder calculation, scheduling, and notification support.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class ReminderMethod(str, Enum):
    """Methods for delivering reminders."""

    POPUP = "popup"           # In-app notification popup
    SOUND = "sound"           # Audio alert
    NOTIFICATION = "notification"  # System notification
    EMAIL = "email"           # Email reminder (requires email config)
    NONE = "none"             # Reminder tracked but not delivered


class ReminderStatus(str, Enum):
    """Status of a scheduled reminder."""

    PENDING = "pending"       # Not yet triggered
    TRIGGERED = "triggered"   # Triggered and delivered
    DISMISSED = "dismissed"   # User dismissed
    SNOOZED = "snoozed"       # User snoozed
    FAILED = "failed"         # Delivery failed


# Default snooze durations in minutes
DEFAULT_SNOOZE_OPTIONS = [5, 10, 15, 30, 60, 1440]  # 5m to 1 day


@dataclass
class Reminder:
    """Represents a reminder for a calendar event.

    Attributes:
        minutes_before: Minutes before event start to trigger
        method: How to deliver the reminder
        message: Optional custom message (defaults to event title)
    """

    minutes_before: int
    method: ReminderMethod = ReminderMethod.NOTIFICATION
    message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON storage."""
        return {
            "minutes_before": self.minutes_before,
            "method": self.method.value,
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Reminder":
        """Create from dictionary."""
        return cls(
            minutes_before=data["minutes_before"],
            method=ReminderMethod(data.get("method", "notification")),
            message=data.get("message"),
        )

    @property
    def timedelta(self) -> timedelta:
        """Get the reminder offset as a timedelta."""
        return timedelta(minutes=self.minutes_before)

    def get_trigger_time(self, event_start: datetime) -> datetime:
        """Calculate when this reminder should trigger.

        Args:
            event_start: The event's start datetime

        Returns:
            Datetime when reminder should fire
        """
        return event_start - self.timedelta

    def is_due(self, event_start: datetime, now: Optional[datetime] = None) -> bool:
        """Check if the reminder is due to trigger.

        Args:
            event_start: The event's start datetime
            now: Current time (default: datetime.now())

        Returns:
            True if the reminder trigger time has passed
        """
        if now is None:
            now = datetime.now()
        return now >= self.get_trigger_time(event_start)

    def describe(self) -> str:
        """Get a human-readable description.

        Returns:
            String like "15 minutes before" or "1 day before"
        """
        return _format_duration(self.minutes_before)


@dataclass
class ScheduledReminder:
    """A reminder instance scheduled for a specific event occurrence.

    Attributes:
        event_id: ID of the event this reminder is for
        event_title: Title of the event
        reminder: The reminder configuration
        trigger_time: When this reminder should fire
        event_start: When the event starts
        status: Current status of this reminder
        snoozed_until: If snoozed, when to remind again
    """

    event_id: str
    event_title: str
    reminder: Reminder
    trigger_time: datetime
    event_start: datetime
    status: ReminderStatus = ReminderStatus.PENDING
    snoozed_until: Optional[datetime] = None

    @property
    def is_pending(self) -> bool:
        """Check if reminder is still pending."""
        return self.status == ReminderStatus.PENDING

    @property
    def is_due(self) -> bool:
        """Check if reminder should fire now."""
        now = datetime.now()
        if self.status == ReminderStatus.SNOOZED and self.snoozed_until:
            return now >= self.snoozed_until
        return self.status == ReminderStatus.PENDING and now >= self.trigger_time

    @property
    def time_until_event(self) -> timedelta:
        """Get time remaining until the event starts."""
        return self.event_start - datetime.now()

    def dismiss(self) -> None:
        """Mark the reminder as dismissed."""
        self.status = ReminderStatus.DISMISSED

    def snooze(self, minutes: int = 5) -> None:
        """Snooze the reminder for specified minutes."""
        self.status = ReminderStatus.SNOOZED
        self.snoozed_until = datetime.now() + timedelta(minutes=minutes)

    def trigger(self) -> None:
        """Mark the reminder as triggered."""
        self.status = ReminderStatus.TRIGGERED


class ReminderScheduler:
    """Manages reminder scheduling and delivery for calendar events.

    This class maintains a queue of upcoming reminders and triggers
    callbacks when reminders are due.
    """

    def __init__(self):
        """Initialize the scheduler."""
        self._scheduled: Dict[str, List[ScheduledReminder]] = {}
        self._callbacks: List[Callable[[ScheduledReminder], None]] = []
        self._running = False

    def register_callback(
        self, callback: Callable[[ScheduledReminder], None]
    ) -> None:
        """Register a callback for when reminders trigger.

        Args:
            callback: Function called with ScheduledReminder when due
        """
        self._callbacks.append(callback)

    def unregister_callback(
        self, callback: Callable[[ScheduledReminder], None]
    ) -> None:
        """Unregister a reminder callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def schedule_event_reminders(
        self,
        event_id: str,
        event_title: str,
        event_start: datetime,
        reminders: List[Reminder],
    ) -> List[ScheduledReminder]:
        """Schedule all reminders for an event.

        Args:
            event_id: Unique event identifier
            event_title: Event title for display
            event_start: Event start datetime
            reminders: List of reminders to schedule

        Returns:
            List of ScheduledReminder instances created
        """
        scheduled = []
        now = datetime.now()

        for reminder in reminders:
            trigger_time = reminder.get_trigger_time(event_start)

            # Don't schedule reminders for past events or already-triggered times
            if trigger_time < now:
                continue

            sr = ScheduledReminder(
                event_id=event_id,
                event_title=event_title,
                reminder=reminder,
                trigger_time=trigger_time,
                event_start=event_start,
            )
            scheduled.append(sr)

        if scheduled:
            self._scheduled[event_id] = scheduled
            logger.debug(
                "Scheduled %d reminders for event %s", len(scheduled), event_id
            )

        return scheduled

    def cancel_event_reminders(self, event_id: str) -> int:
        """Cancel all reminders for an event.

        Args:
            event_id: Event identifier

        Returns:
            Number of reminders cancelled
        """
        if event_id in self._scheduled:
            count = len(self._scheduled[event_id])
            del self._scheduled[event_id]
            logger.debug("Cancelled %d reminders for event %s", count, event_id)
            return count
        return 0

    def get_due_reminders(self) -> List[ScheduledReminder]:
        """Get all reminders that are currently due.

        Returns:
            List of ScheduledReminder instances that should trigger
        """
        due = []
        for reminders in self._scheduled.values():
            for sr in reminders:
                if sr.is_due:
                    due.append(sr)
        return sorted(due, key=lambda r: r.trigger_time)

    def get_upcoming_reminders(
        self, within_minutes: int = 60
    ) -> List[ScheduledReminder]:
        """Get reminders that will trigger within the specified time.

        Args:
            within_minutes: Look-ahead window in minutes

        Returns:
            List of upcoming ScheduledReminder instances
        """
        cutoff = datetime.now() + timedelta(minutes=within_minutes)
        upcoming = []
        for reminders in self._scheduled.values():
            for sr in reminders:
                if sr.is_pending and sr.trigger_time <= cutoff:
                    upcoming.append(sr)
        return sorted(upcoming, key=lambda r: r.trigger_time)

    def process_due_reminders(self) -> int:
        """Process and trigger all due reminders.

        Returns:
            Number of reminders triggered
        """
        count = 0
        for sr in self.get_due_reminders():
            self._trigger_reminder(sr)
            count += 1
        return count

    def _trigger_reminder(self, reminder: ScheduledReminder) -> None:
        """Trigger a specific reminder."""
        reminder.trigger()

        for callback in self._callbacks:
            try:
                callback(reminder)
            except Exception as exc:
                logger.error(
                    "Reminder callback failed for event %s: %s",
                    reminder.event_id,
                    exc,
                )

    def clear_all(self) -> None:
        """Clear all scheduled reminders."""
        self._scheduled.clear()


# ============================================================================
# Helper functions
# ============================================================================


def _format_duration(minutes: int) -> str:
    """Format a duration in minutes as human-readable text.

    Args:
        minutes: Duration in minutes

    Returns:
        Human-readable string like "15 minutes before"
    """
    if minutes == 0:
        return "At event time"
    elif minutes < 60:
        unit = "minute" if minutes == 1 else "minutes"
        return f"{minutes} {unit} before"
    elif minutes < 1440:
        hours = minutes // 60
        unit = "hour" if hours == 1 else "hours"
        return f"{hours} {unit} before"
    elif minutes < 10080:
        days = minutes // 1440
        unit = "day" if days == 1 else "days"
        return f"{days} {unit} before"
    else:
        weeks = minutes // 10080
        unit = "week" if weeks == 1 else "weeks"
        return f"{weeks} {unit} before"


def create_default_reminders(
    event_duration_minutes: int = 60,
    is_all_day: bool = False,
) -> List[Reminder]:
    """Create default reminders based on event type.

    Args:
        event_duration_minutes: Event duration in minutes
        is_all_day: Whether this is an all-day event

    Returns:
        List of default Reminder instances
    """
    if is_all_day:
        # All-day events: remind day before at 9 AM
        return [
            Reminder(minutes_before=540, method=ReminderMethod.NOTIFICATION),  # 9 hours = 9 AM if event is at 6 PM
        ]
    elif event_duration_minutes >= 1440:
        # Multi-day events: remind 1 day and 1 hour before
        return [
            Reminder(minutes_before=1440, method=ReminderMethod.NOTIFICATION),
            Reminder(minutes_before=60, method=ReminderMethod.NOTIFICATION),
        ]
    else:
        # Regular events: remind 15 minutes before
        return [
            Reminder(minutes_before=15, method=ReminderMethod.NOTIFICATION),
        ]


def parse_reminder_from_natural_language(text: str) -> Optional[Reminder]:
    """Parse a natural language reminder string.

    Args:
        text: String like "15 minutes before", "1 hour before", "1 day before"

    Returns:
        Reminder instance, or None if parsing fails
    """
    text = text.lower().strip()

    # Common patterns
    patterns = {
        "at event time": 0,
        "at time of event": 0,
        "5 minutes before": 5,
        "10 minutes before": 10,
        "15 minutes before": 15,
        "30 minutes before": 30,
        "1 hour before": 60,
        "2 hours before": 120,
        "1 day before": 1440,
        "2 days before": 2880,
        "1 week before": 10080,
    }

    if text in patterns:
        return Reminder(minutes_before=patterns[text])

    # Try to parse numeric patterns
    import re

    # Match patterns like "15 minutes", "1 hour", "2 days"
    match = re.match(r"(\d+)\s*(minute|hour|day|week)s?\s*before", text)
    if match:
        value = int(match.group(1))
        unit = match.group(2)

        multipliers = {
            "minute": 1,
            "hour": 60,
            "day": 1440,
            "week": 10080,
        }

        minutes = value * multipliers.get(unit, 1)
        return Reminder(minutes_before=minutes)

    return None


# ============================================================================
# Preset reminder configurations
# ============================================================================


# Common preset options for UI dropdowns
REMINDER_PRESETS: List[Dict[str, Any]] = [
    {"label": "At time of event", "minutes": 0},
    {"label": "5 minutes before", "minutes": 5},
    {"label": "10 minutes before", "minutes": 10},
    {"label": "15 minutes before", "minutes": 15},
    {"label": "30 minutes before", "minutes": 30},
    {"label": "1 hour before", "minutes": 60},
    {"label": "2 hours before", "minutes": 120},
    {"label": "1 day before", "minutes": 1440},
    {"label": "2 days before", "minutes": 2880},
    {"label": "1 week before", "minutes": 10080},
]


def get_reminder_preset_labels() -> List[str]:
    """Get list of preset reminder labels for UI."""
    return [p["label"] for p in REMINDER_PRESETS]


def minutes_from_preset_label(label: str) -> int:
    """Get minutes value from a preset label.

    Args:
        label: Preset label string

    Returns:
        Minutes value, or 15 as default
    """
    for preset in REMINDER_PRESETS:
        if preset["label"] == label:
            return preset["minutes"]
    return 15  # Default
