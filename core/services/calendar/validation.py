"""
Calendar event validation logic.

Provides comprehensive validation for calendar events, moving business
logic from the UI layer to the service layer.

Author: ATLAS Team
Date: Jan 7, 2026
"""

from datetime import datetime, timezone
from typing import List

from core.services.common import OperationResult, ValidationError

from .types import CalendarEvent, CalendarEventCreate, CalendarEventUpdate


class CalendarEventValidator:
    """
    Validates calendar event data and business rules.
    
    Encapsulates all validation logic that was previously scattered
    across UI components, ensuring consistent validation across all
    entry points.
    """
    
    def __init__(self) -> None:
        # Maximum event duration in days
        self.MAX_EVENT_DURATION_DAYS = 365
        
        # Minimum event duration in minutes
        self.MIN_EVENT_DURATION_MINUTES = 1
        
        # Maximum title length
        self.MAX_TITLE_LENGTH = 255
        
        # Maximum description length
        self.MAX_DESCRIPTION_LENGTH = 10000
    
    async def validate_create(
        self, 
        event_data: CalendarEventCreate
    ) -> OperationResult[None]:
        """
        Validate calendar event creation data.
        
        Args:
            event_data: Event data to validate
            
        Returns:
            Success if valid, failure with details if invalid
        """
        errors = []
        
        # Validate title
        title_errors = self._validate_title(event_data.title)
        errors.extend(title_errors)
        
        # Validate description
        if event_data.description:
            desc_errors = self._validate_description(event_data.description)
            errors.extend(desc_errors)
        
        # Validate times
        if event_data.start_time and event_data.end_time:
            time_errors = self._validate_times(
                event_data.start_time,
                event_data.end_time,
                event_data.all_day
            )
            errors.extend(time_errors)
        
        # Validate timezone - always required for create
        tz_errors = self._validate_timezone(event_data.timezone_name or "")
        errors.extend(tz_errors)
        
        # Validate recurrence
        if event_data.is_recurring and event_data.recurrence_pattern:
            recurrence_errors = self._validate_recurrence(event_data.recurrence_pattern)
            errors.extend(recurrence_errors)
        elif event_data.is_recurring and not event_data.recurrence_pattern:
            errors.append("Recurring events must have a recurrence pattern")
        
        # Return errors if any
        if errors:
            return OperationResult.failure(
                "; ".join(errors),
                "VALIDATION_FAILED"
            )
        
        return OperationResult.success(None)
    
    async def validate_update(
        self,
        update_data: CalendarEventUpdate,
        existing_event: CalendarEvent,
    ) -> OperationResult[None]:
        """
        Validate calendar event update data.
        
        Args:
            update_data: Update data to validate
            existing_event: Current event data
            
        Returns:
            Success if valid, failure with details if invalid
        """
        errors = []
        
        # Validate title if being updated
        if update_data.title is not None:
            title_errors = self._validate_title(update_data.title)
            errors.extend(title_errors)
        
        # Validate description if being updated
        if update_data.description is not None:
            desc_errors = self._validate_description(update_data.description)
            errors.extend(desc_errors)
        
        # Validate times if being updated
        start_time = update_data.start_time or existing_event.start_time
        end_time = update_data.end_time or existing_event.end_time
        all_day = update_data.all_day if update_data.all_day is not None else existing_event.all_day
        
        if update_data.start_time or update_data.end_time or update_data.all_day is not None:
            time_errors = self._validate_times(start_time, end_time, all_day)
            errors.extend(time_errors)
        
        # Validate timezone if being updated
        if update_data.timezone_name:
            tz_errors = self._validate_timezone(update_data.timezone_name)
            errors.extend(tz_errors)
        
        # Validate recurrence if being updated
        is_recurring = update_data.is_recurring if update_data.is_recurring is not None else existing_event.is_recurring
        recurrence_pattern = update_data.recurrence_pattern or existing_event.recurrence_pattern
        
        if is_recurring and recurrence_pattern:
            recurrence_errors = self._validate_recurrence(recurrence_pattern)
            errors.extend(recurrence_errors)
        elif is_recurring and not recurrence_pattern:
            errors.append("Recurring events must have a recurrence pattern")
        
        # Return errors if any
        if errors:
            return OperationResult.failure(
                "; ".join(errors),
                "VALIDATION_FAILED"
            )
        
        return OperationResult.success(None)
    
    def _validate_title(self, title: str) -> List[str]:
        """Validate event title."""
        errors = []
        
        if not title:
            errors.append("Title is required")
        elif not title.strip():
            errors.append("Title cannot be empty or whitespace only")
        elif len(title) > self.MAX_TITLE_LENGTH:
            errors.append(f"Title cannot exceed {self.MAX_TITLE_LENGTH} characters")
        
        return errors
    
    def _validate_description(self, description: str) -> List[str]:
        """Validate event description."""
        errors = []
        
        if len(description) > self.MAX_DESCRIPTION_LENGTH:
            errors.append(f"Description cannot exceed {self.MAX_DESCRIPTION_LENGTH} characters")
        
        return errors
    
    def _validate_times(
        self, 
        start_time: datetime, 
        end_time: datetime, 
        all_day: bool
    ) -> List[str]:
        """Validate event start and end times."""
        errors = []
        
        # Basic time validation
        if start_time >= end_time:
            errors.append("Start time must be before end time")
        
        # Duration validation
        duration = end_time - start_time
        duration_days = duration.total_seconds() / (24 * 60 * 60)
        duration_minutes = duration.total_seconds() / 60
        
        if duration_days > self.MAX_EVENT_DURATION_DAYS:
            errors.append(f"Event duration cannot exceed {self.MAX_EVENT_DURATION_DAYS} days")
        
        if not all_day and duration_minutes < self.MIN_EVENT_DURATION_MINUTES:
            errors.append(f"Event duration must be at least {self.MIN_EVENT_DURATION_MINUTES} minute(s)")
        
        # All-day event validation
        if all_day:
            # All-day events should have times at midnight
            if (start_time.hour != 0 or start_time.minute != 0 or start_time.second != 0 or
                end_time.hour != 0 or end_time.minute != 0 or end_time.second != 0):
                errors.append("All-day events should have times set to midnight")
        
        # Past event validation (warn but don't error)
        now = datetime.now(timezone.utc)
        if end_time < now:
            # This is a warning, not an error - users might want to create past events
            pass
        
        return errors
    
    def _validate_timezone(self, timezone_name: str) -> List[str]:
        """Validate timezone name."""
        errors = []
        
        if not timezone_name:
            errors.append("Timezone is required")
        else:
            try:
                # Try to create a timezone object to validate
                from zoneinfo import ZoneInfo
                ZoneInfo(timezone_name)
            except Exception:
                errors.append(f"Invalid timezone: {timezone_name}")
        
        return errors
    
    def _validate_recurrence(self, recurrence_pattern: str) -> List[str]:
        """Validate recurrence pattern."""
        errors = []
        
        if not recurrence_pattern:
            errors.append("Recurrence pattern is required for recurring events")
        else:
            # Basic RRULE validation
            if not recurrence_pattern.startswith("FREQ="):
                errors.append("Recurrence pattern must be a valid RRULE starting with FREQ=")
            else:
                # More detailed validation could be added here
                # For now, just check for basic RRULE structure
                valid_frequencies = ["DAILY", "WEEKLY", "MONTHLY", "YEARLY"]
                freq_part = recurrence_pattern.split(";")[0]
                freq_value = freq_part.split("=")[1] if "=" in freq_part else ""
                
                if freq_value not in valid_frequencies:
                    errors.append(f"Invalid recurrence frequency: {freq_value}")
        
        return errors
    
    def validate_event_conflict_resolution(
        self,
        conflicting_events: List[CalendarEvent],
        resolution_strategy: str,
    ) -> OperationResult[None]:
        """
        Validate conflict resolution strategy.
        
        Args:
            conflicting_events: List of conflicting events
            resolution_strategy: How to handle conflicts
            
        Returns:
            Success if strategy is valid, failure otherwise
        """
        valid_strategies = ["ignore", "warn", "block", "auto_reschedule"]
        
        if resolution_strategy not in valid_strategies:
            return OperationResult.failure(
                f"Invalid conflict resolution strategy: {resolution_strategy}",
                "INVALID_STRATEGY"
            )
        
        if resolution_strategy == "auto_reschedule" and len(conflicting_events) > 5:
            return OperationResult.failure(
                "Auto-reschedule not available when more than 5 conflicts exist",
                "TOO_MANY_CONFLICTS"
            )
        
        return OperationResult.success(None)
    
    def validate_bulk_operation(
        self,
        event_ids: List[str],
        operation: str,
    ) -> OperationResult[None]:
        """
        Validate bulk operations on events.
        
        Args:
            event_ids: List of event IDs
            operation: Type of bulk operation
            
        Returns:
            Success if valid, failure otherwise
        """
        if not event_ids:
            return OperationResult.failure(
                "Event IDs list cannot be empty",
                "EMPTY_EVENT_LIST"
            )
        
        if len(event_ids) > 100:
            return OperationResult.failure(
                "Bulk operations limited to 100 events at a time",
                "TOO_MANY_EVENTS"
            )
        
        valid_operations = ["delete", "update_category", "update_status", "export"]
        if operation not in valid_operations:
            return OperationResult.failure(
                f"Invalid bulk operation: {operation}",
                "INVALID_OPERATION"
            )
        
        return OperationResult.success(None)