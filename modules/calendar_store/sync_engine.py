"""Calendar sync engine for importing from external sources.

Provides a unified interface for syncing calendar data from various
external sources (ICS files, CalDAV servers, etc.) into the ATLAS
Master Calendar.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
from uuid import uuid4

logger = logging.getLogger(__name__)


class SyncDirection(Enum):
    """Direction of sync operation."""
    IMPORT = "import"       # External → ATLAS
    EXPORT = "export"       # ATLAS → External
    BIDIRECTIONAL = "bidirectional"


class SyncStatus(Enum):
    """Status of a sync operation."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    PARTIAL = "partial"     # Some items failed
    FAILED = "failed"
    CANCELLED = "cancelled"


class ConflictResolution(Enum):
    """How to resolve conflicts between local and remote."""
    ASK = "ask"             # Prompt user
    LOCAL_WINS = "local_wins"
    REMOTE_WINS = "remote_wins"
    NEWEST_WINS = "newest_wins"
    MERGE = "merge"         # Attempt to merge changes


@dataclass
class SyncConflict:
    """Represents a conflict between local and remote event."""
    event_id: str
    external_id: str
    local_event: Dict[str, Any]
    remote_event: Dict[str, Any]
    conflict_type: str  # "modified", "deleted_local", "deleted_remote"
    resolution: Optional[ConflictResolution] = None
    resolved_event: Optional[Dict[str, Any]] = None


@dataclass
class SyncResult:
    """Result of a sync operation."""
    status: SyncStatus
    source_type: str
    source_account: str
    source_calendar: Optional[str] = None
    
    # Counts
    events_added: int = 0
    events_updated: int = 0
    events_deleted: int = 0
    events_skipped: int = 0
    
    # Errors and conflicts
    errors: List[str] = field(default_factory=list)
    conflicts: List[SyncConflict] = field(default_factory=list)
    
    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Sync token for incremental sync
    sync_token: Optional[str] = None

    @property
    def total_processed(self) -> int:
        """Total events processed."""
        return self.events_added + self.events_updated + self.events_deleted

    @property
    def has_errors(self) -> bool:
        """Check if sync had errors."""
        return len(self.errors) > 0

    @property
    def has_conflicts(self) -> bool:
        """Check if there are unresolved conflicts."""
        return any(c.resolution is None for c in self.conflicts)


@dataclass
class ExternalEvent:
    """Event data from an external source."""
    external_id: str
    title: str
    start_time: datetime
    end_time: datetime
    
    description: Optional[str] = None
    location: Optional[str] = None
    is_all_day: bool = False
    timezone: str = "UTC"
    
    recurrence_rule: Optional[str] = None
    attendees: List[Dict[str, Any]] = field(default_factory=list)
    reminders: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    etag: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    # Raw data for provider-specific fields
    raw_data: Dict[str, Any] = field(default_factory=dict)


class CalendarSyncProvider(ABC):
    """Abstract base class for calendar sync providers.
    
    Implement this class to add support for syncing from a specific
    calendar source (ICS, CalDAV, Google, Outlook, etc.).
    """
    
    @property
    @abstractmethod
    def provider_type(self) -> str:
        """Return the provider type identifier (e.g., 'ics', 'caldav')."""
        pass
    
    @property
    @abstractmethod
    def display_name(self) -> str:
        """Return human-readable provider name."""
        pass
    
    @abstractmethod
    def connect(self, config: Dict[str, Any]) -> bool:
        """Connect to the calendar source.
        
        Args:
            config: Provider-specific configuration
            
        Returns:
            True if connection successful
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the calendar source."""
        pass
    
    @abstractmethod
    def list_calendars(self) -> List[Dict[str, Any]]:
        """List available calendars from the source.
        
        Returns:
            List of calendar info dicts with keys:
            - id: External calendar ID
            - name: Calendar display name
            - color: Optional color hint
            - readonly: Whether calendar is read-only
        """
        pass
    
    @abstractmethod
    def fetch_events(
        self,
        calendar_id: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        sync_token: Optional[str] = None,
    ) -> Tuple[List[ExternalEvent], Optional[str]]:
        """Fetch events from a calendar.
        
        Args:
            calendar_id: ID of the calendar to fetch from
            start: Optional start date filter
            end: Optional end date filter
            sync_token: Optional token for incremental sync
            
        Returns:
            Tuple of (events list, new sync token)
        """
        pass
    
    def push_event(
        self,
        calendar_id: str,
        event: ExternalEvent,
    ) -> Optional[str]:
        """Push an event to the remote calendar (for bidirectional sync).
        
        Args:
            calendar_id: ID of the target calendar
            event: Event to push
            
        Returns:
            New external_id if created, or None on failure
        """
        raise NotImplementedError("This provider does not support pushing events")
    
    def delete_event(
        self,
        calendar_id: str,
        external_id: str,
    ) -> bool:
        """Delete an event from the remote calendar.
        
        Args:
            calendar_id: ID of the calendar
            external_id: ID of the event to delete
            
        Returns:
            True if deleted successfully
        """
        raise NotImplementedError("This provider does not support deleting events")


class SyncEngine:
    """Main sync engine coordinating calendar synchronization.
    
    Handles the sync workflow:
    1. Connect to provider
    2. Fetch remote events
    3. Compare with local events
    4. Resolve conflicts
    5. Apply changes
    6. Update sync state
    """
    
    def __init__(
        self,
        repository: Any,  # CalendarStoreRepository
        conflict_resolution: ConflictResolution = ConflictResolution.REMOTE_WINS,
    ):
        self._repo = repository
        self._conflict_resolution = conflict_resolution
        self._providers: Dict[str, CalendarSyncProvider] = {}
        self._progress_callback: Optional[Callable[[str, int, int], None]] = None
        self._running_syncs: Dict[str, bool] = {}
    
    def register_provider(self, provider: CalendarSyncProvider) -> None:
        """Register a sync provider."""
        self._providers[provider.provider_type] = provider
        logger.info("Registered sync provider: %s", provider.provider_type)
    
    def get_provider(self, provider_type: str) -> Optional[CalendarSyncProvider]:
        """Get a registered provider by type."""
        return self._providers.get(provider_type)
    
    def list_providers(self) -> List[Dict[str, str]]:
        """List registered providers."""
        return [
            {"type": p.provider_type, "name": p.display_name}
            for p in self._providers.values()
        ]
    
    def set_progress_callback(
        self,
        callback: Callable[[str, int, int], None],
    ) -> None:
        """Set callback for progress updates.
        
        Callback receives (message, current, total).
        """
        self._progress_callback = callback
    
    def _report_progress(self, message: str, current: int, total: int) -> None:
        """Report sync progress."""
        if self._progress_callback:
            self._progress_callback(message, current, total)
    
    def sync_calendar(
        self,
        provider_type: str,
        config: Dict[str, Any],
        calendar_id: str,
        target_category_id: Optional[str] = None,
        direction: SyncDirection = SyncDirection.IMPORT,
        incremental: bool = True,
    ) -> SyncResult:
        """Sync a single calendar.
        
        Args:
            provider_type: Type of provider to use
            config: Provider configuration
            calendar_id: External calendar ID to sync
            target_category_id: Category to assign events to
            direction: Sync direction
            incremental: Use incremental sync if available
            
        Returns:
            SyncResult with operation details
        """
        sync_key = f"{provider_type}:{calendar_id}"
        
        if sync_key in self._running_syncs:
            return SyncResult(
                status=SyncStatus.FAILED,
                source_type=provider_type,
                source_account=config.get("account", "unknown"),
                source_calendar=calendar_id,
                errors=["Sync already in progress for this calendar"],
            )
        
        self._running_syncs[sync_key] = True
        result = SyncResult(
            status=SyncStatus.IN_PROGRESS,
            source_type=provider_type,
            source_account=config.get("account", "unknown"),
            source_calendar=calendar_id,
            started_at=datetime.now(timezone.utc),
        )
        
        try:
            # Get provider
            provider = self.get_provider(provider_type)
            if not provider:
                result.status = SyncStatus.FAILED
                result.errors.append(f"Unknown provider: {provider_type}")
                return result
            
            # Connect
            self._report_progress("Connecting...", 0, 100)
            if not provider.connect(config):
                result.status = SyncStatus.FAILED
                result.errors.append("Failed to connect to calendar source")
                return result
            
            try:
                # Get existing sync state for incremental sync
                sync_token = None
                if incremental:
                    sync_state = self._repo.get_sync_state(
                        provider_type,
                        config.get("account", "unknown"),
                        calendar_id,
                    )
                    if sync_state:
                        sync_token = sync_state.get("sync_token")
                
                # Fetch events
                self._report_progress("Fetching events...", 10, 100)
                events, new_token = provider.fetch_events(
                    calendar_id,
                    sync_token=sync_token,
                )
                
                total_events = len(events)
                logger.info("Fetched %d events from %s", total_events, calendar_id)
                
                # Process events
                for i, ext_event in enumerate(events):
                    self._report_progress(
                        f"Processing event {i+1}/{total_events}",
                        10 + int((i / max(total_events, 1)) * 80),
                        100,
                    )
                    
                    try:
                        self._process_event(
                            ext_event,
                            provider_type,
                            config.get("account", "unknown"),
                            target_category_id,
                            result,
                        )
                    except Exception as e:
                        result.errors.append(f"Error processing event '{ext_event.title}': {e}")
                        result.events_skipped += 1
                
                # Update sync state
                self._report_progress("Updating sync state...", 95, 100)
                self._repo.update_sync_state(
                    source_type=provider_type,
                    source_account=config.get("account", "unknown"),
                    source_calendar=calendar_id,
                    sync_token=new_token,
                    last_sync_status="success" if not result.has_errors else "partial",
                )
                result.sync_token = new_token
                
                # Set final status
                if result.has_errors:
                    result.status = SyncStatus.PARTIAL
                else:
                    result.status = SyncStatus.SUCCESS
                
            finally:
                provider.disconnect()
                
        except Exception as e:
            logger.exception("Sync failed: %s", e)
            result.status = SyncStatus.FAILED
            result.errors.append(str(e))
            
        finally:
            result.completed_at = datetime.now(timezone.utc)
            del self._running_syncs[sync_key]
            self._report_progress("Complete", 100, 100)
        
        return result
    
    def _process_event(
        self,
        ext_event: ExternalEvent,
        source_type: str,
        source_account: str,
        target_category_id: Optional[str],
        result: SyncResult,
    ) -> None:
        """Process a single external event."""
        # Check if event already exists
        existing = self._repo.get_event_by_external_id(
            ext_event.external_id,
            source_type,
        )
        
        if existing:
            # Update existing event
            # Check for conflicts based on etag/updated_at
            if self._has_conflict(existing, ext_event):
                conflict = self._create_conflict(existing, ext_event)
                resolved = self._resolve_conflict(conflict)
                
                if resolved:
                    self._update_event_from_external(existing["id"], resolved, source_type)
                    result.events_updated += 1
                else:
                    result.conflicts.append(conflict)
                    result.events_skipped += 1
            else:
                self._update_event_from_external(existing["id"], ext_event, source_type)
                result.events_updated += 1
        else:
            # Create new event
            self._create_event_from_external(
                ext_event,
                source_type,
                source_account,
                target_category_id,
            )
            result.events_added += 1
    
    def _has_conflict(
        self,
        local: Dict[str, Any],
        remote: ExternalEvent,
    ) -> bool:
        """Check if there's a conflict between local and remote."""
        # If local was modified after last sync, we have a potential conflict
        local_etag = local.get("etag")
        if local_etag and remote.etag and local_etag != remote.etag:
            # Check if local was modified
            local_updated = local.get("updated_at")
            local_synced = local.get("last_synced_at")
            if local_updated and local_synced and local_updated > local_synced:
                return True
        return False
    
    def _create_conflict(
        self,
        local: Dict[str, Any],
        remote: ExternalEvent,
    ) -> SyncConflict:
        """Create a conflict object."""
        return SyncConflict(
            event_id=local["id"],
            external_id=remote.external_id,
            local_event=local,
            remote_event={
                "title": remote.title,
                "start_time": remote.start_time,
                "end_time": remote.end_time,
                "description": remote.description,
                "location": remote.location,
            },
            conflict_type="modified",
        )
    
    def _resolve_conflict(
        self,
        conflict: SyncConflict,
    ) -> Optional[ExternalEvent]:
        """Resolve a conflict based on configured strategy."""
        if self._conflict_resolution == ConflictResolution.ASK:
            return None  # Leave unresolved for user
        
        if self._conflict_resolution == ConflictResolution.REMOTE_WINS:
            # Return remote event data
            return ExternalEvent(
                external_id=conflict.external_id,
                title=conflict.remote_event["title"],
                start_time=conflict.remote_event["start_time"],
                end_time=conflict.remote_event["end_time"],
                description=conflict.remote_event.get("description"),
                location=conflict.remote_event.get("location"),
            )
        
        if self._conflict_resolution == ConflictResolution.LOCAL_WINS:
            conflict.resolution = ConflictResolution.LOCAL_WINS
            return None  # Keep local, skip update
        
        if self._conflict_resolution == ConflictResolution.NEWEST_WINS:
            local_updated = conflict.local_event.get("updated_at")
            remote_updated = conflict.remote_event.get("updated_at")
            
            if remote_updated and local_updated:
                if remote_updated > local_updated:
                    return ExternalEvent(
                        external_id=conflict.external_id,
                        title=conflict.remote_event["title"],
                        start_time=conflict.remote_event["start_time"],
                        end_time=conflict.remote_event["end_time"],
                    )
            return None  # Keep local
        
        return None
    
    def _create_event_from_external(
        self,
        ext_event: ExternalEvent,
        source_type: str,
        source_account: str,
        target_category_id: Optional[str],
    ) -> str:
        """Create a local event from external event."""
        return self._repo.create_event(
            title=ext_event.title,
            start_time=ext_event.start_time,
            end_time=ext_event.end_time,
            description=ext_event.description,
            location=ext_event.location,
            is_all_day=ext_event.is_all_day,
            timezone=ext_event.timezone,
            recurrence_rule=ext_event.recurrence_rule,
            category_id=target_category_id,
            external_id=ext_event.external_id,
            external_source=source_type,
            etag=ext_event.etag,
            attendees=ext_event.attendees,
            reminders=ext_event.reminders,
        )
    
    def _update_event_from_external(
        self,
        event_id: str,
        ext_event: ExternalEvent,
        source_type: str,
    ) -> None:
        """Update a local event from external event."""
        self._repo.update_event(
            event_id,
            title=ext_event.title,
            start_time=ext_event.start_time,
            end_time=ext_event.end_time,
            description=ext_event.description,
            location=ext_event.location,
            is_all_day=ext_event.is_all_day,
            timezone=ext_event.timezone,
            recurrence_rule=ext_event.recurrence_rule,
            etag=ext_event.etag,
            attendees=ext_event.attendees,
            reminders=ext_event.reminders,
            last_synced_at=datetime.now(timezone.utc),
        )
    
    def cancel_sync(self, provider_type: str, calendar_id: str) -> bool:
        """Cancel an in-progress sync."""
        sync_key = f"{provider_type}:{calendar_id}"
        if sync_key in self._running_syncs:
            # Signal cancellation (provider should check this)
            self._running_syncs[sync_key] = False
            return True
        return False
