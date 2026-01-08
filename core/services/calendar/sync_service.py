"""
Calendar synchronization service.

Provides a service layer for synchronizing calendar events with external
sources (ICS files, CalDAV servers, etc.) while emitting domain events
for reactive UI updates and audit logging.

Author: ATLAS Team
Date: Jan 8, 2026
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Union
from uuid import UUID, uuid4

from core.services.common import Actor, DomainEvent, OperationResult
from core.services.common.protocols import EventPublisher

logger = logging.getLogger(__name__)


# ============================================================================
# Sync Types
# ============================================================================


class SyncDirection(str, Enum):
    """Direction of sync operation."""
    IMPORT = "import"           # External → ATLAS
    EXPORT = "export"           # ATLAS → External
    BIDIRECTIONAL = "bidirectional"


class SyncStatus(str, Enum):
    """Status of a sync operation."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    PARTIAL = "partial"         # Some items failed
    FAILED = "failed"
    CANCELLED = "cancelled"


class ConflictResolution(str, Enum):
    """How to resolve conflicts between local and remote."""
    ASK = "ask"                 # Prompt user
    LOCAL_WINS = "local_wins"
    REMOTE_WINS = "remote_wins"
    NEWEST_WINS = "newest_wins"
    MERGE = "merge"             # Attempt to merge changes


@dataclass
class SyncConflict:
    """Represents a conflict between local and remote event."""
    conflict_id: str
    event_id: str
    external_id: str
    local_event: Dict[str, Any]
    remote_event: Dict[str, Any]
    conflict_type: str  # "modified", "deleted_local", "deleted_remote"
    resolution: Optional[ConflictResolution] = None
    resolved_event: Optional[Dict[str, Any]] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class SyncProgress:
    """Progress update for a sync operation."""
    sync_id: str
    provider_type: str
    calendar_id: str
    message: str
    current: int
    total: int
    percent: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class SyncResult:
    """Result of a sync operation."""
    sync_id: str
    status: SyncStatus
    provider_type: str
    source_account: str
    source_calendar: Optional[str] = None
    target_category_id: Optional[str] = None
    
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
    duration_seconds: float = 0.0
    
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
    def has_unresolved_conflicts(self) -> bool:
        """Check if there are unresolved conflicts."""
        return any(c.resolution is None for c in self.conflicts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "sync_id": self.sync_id,
            "status": self.status.value,
            "provider_type": self.provider_type,
            "source_account": self.source_account,
            "source_calendar": self.source_calendar,
            "target_category_id": self.target_category_id,
            "events_added": self.events_added,
            "events_updated": self.events_updated,
            "events_deleted": self.events_deleted,
            "events_skipped": self.events_skipped,
            "total_processed": self.total_processed,
            "errors": self.errors,
            "conflict_count": len(self.conflicts),
            "has_unresolved_conflicts": self.has_unresolved_conflicts,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class SyncConfiguration:
    """Configuration for a sync operation."""
    provider_type: str
    account_name: str
    calendar_id: str
    config: Dict[str, Any]  # Provider-specific configuration
    
    target_category_id: Optional[str] = None
    direction: SyncDirection = SyncDirection.IMPORT
    conflict_resolution: ConflictResolution = ConflictResolution.REMOTE_WINS
    incremental: bool = True
    
    # Schedule settings
    auto_sync_enabled: bool = False
    sync_interval_minutes: int = 60


@dataclass
class ProviderInfo:
    """Information about a registered sync provider."""
    provider_type: str
    display_name: str
    supports_push: bool = False
    supports_incremental: bool = True
    requires_auth: bool = False
    config_schema: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Sync Domain Events
# ============================================================================


@dataclass(frozen=True)
class CalendarSyncStarted(DomainEvent):
    """Published when a calendar sync operation starts."""
    
    sync_id: str = ""
    provider_type: str = ""
    source_account: str = ""
    source_calendar: str = ""
    direction: str = ""
    
    @classmethod
    def for_sync(
        cls,
        sync_id: str,
        tenant_id: str,
        provider_type: str,
        source_account: str,
        source_calendar: str,
        direction: SyncDirection,
    ) -> "CalendarSyncStarted":
        return cls(
            event_type="calendar.sync.started",
            entity_id=UUID(sync_id) if len(sync_id) == 36 else uuid4(),
            tenant_id=tenant_id,
            actor="sync",
            metadata={
                "sync_id": sync_id,
                "provider_type": provider_type,
                "source_account": source_account,
            },
            timestamp=datetime.now(timezone.utc),
            sync_id=sync_id,
            provider_type=provider_type,
            source_account=source_account,
            source_calendar=source_calendar,
            direction=direction.value,
        )


@dataclass(frozen=True)
class CalendarSyncProgress(DomainEvent):
    """Published during sync to report progress."""
    
    sync_id: str = ""
    message: str = ""
    current: int = 0
    total: int = 0
    percent: float = 0.0
    
    @classmethod
    def for_sync(
        cls,
        sync_id: str,
        tenant_id: str,
        message: str,
        current: int,
        total: int,
    ) -> "CalendarSyncProgress":
        percent = (current / max(total, 1)) * 100
        return cls(
            event_type="calendar.sync.progress",
            entity_id=UUID(sync_id) if len(sync_id) == 36 else uuid4(),
            tenant_id=tenant_id,
            actor="sync",
            metadata={},
            timestamp=datetime.now(timezone.utc),
            sync_id=sync_id,
            message=message,
            current=current,
            total=total,
            percent=percent,
        )


@dataclass(frozen=True)
class CalendarSyncCompleted(DomainEvent):
    """Published when a calendar sync operation completes."""
    
    sync_id: str = ""
    status: str = ""
    events_added: int = 0
    events_updated: int = 0
    events_deleted: int = 0
    error_count: int = 0
    conflict_count: int = 0
    duration_seconds: float = 0.0
    
    @classmethod
    def for_sync(
        cls,
        sync_id: str,
        tenant_id: str,
        result: SyncResult,
    ) -> "CalendarSyncCompleted":
        return cls(
            event_type="calendar.sync.completed",
            entity_id=UUID(sync_id) if len(sync_id) == 36 else uuid4(),
            tenant_id=tenant_id,
            actor="sync",
            metadata=result.to_dict(),
            timestamp=datetime.now(timezone.utc),
            sync_id=sync_id,
            status=result.status.value,
            events_added=result.events_added,
            events_updated=result.events_updated,
            events_deleted=result.events_deleted,
            error_count=len(result.errors),
            conflict_count=len(result.conflicts),
            duration_seconds=result.duration_seconds,
        )


@dataclass(frozen=True)
class CalendarSyncConflict(DomainEvent):
    """Published when a sync conflict is detected."""
    
    sync_id: str = ""
    conflict_id: str = ""
    calendar_event_id: str = ""  # Renamed to avoid conflict with base class
    external_id: str = ""
    conflict_type: str = ""
    
    @classmethod
    def for_sync(
        cls,
        sync_id: str,
        tenant_id: str,
        conflict: SyncConflict,
    ) -> "CalendarSyncConflict":
        return cls(
            event_type="calendar.sync.conflict",
            entity_id=UUID(conflict.event_id) if len(conflict.event_id) == 36 else uuid4(),
            tenant_id=tenant_id,
            actor="sync",
            metadata={
                "local_event": conflict.local_event,
                "remote_event": conflict.remote_event,
            },
            timestamp=datetime.now(timezone.utc),
            sync_id=sync_id,
            conflict_id=conflict.conflict_id,
            calendar_event_id=conflict.event_id,
            external_id=conflict.external_id,
            conflict_type=conflict.conflict_type,
        )


@dataclass(frozen=True)
class CalendarSyncFailed(DomainEvent):
    """Published when a sync operation fails."""
    
    sync_id: str = ""
    provider_type: str = ""
    error_message: str = ""
    
    @classmethod
    def for_sync(
        cls,
        sync_id: str,
        tenant_id: str,
        provider_type: str,
        error_message: str,
    ) -> "CalendarSyncFailed":
        return cls(
            event_type="calendar.sync.failed",
            entity_id=UUID(sync_id) if len(sync_id) == 36 else uuid4(),
            tenant_id=tenant_id,
            actor="sync",
            metadata={"error": error_message},
            timestamp=datetime.now(timezone.utc),
            sync_id=sync_id,
            provider_type=provider_type,
            error_message=error_message,
        )


# ============================================================================
# Permission Protocol
# ============================================================================


class SyncPermissionChecker(Protocol):
    """Protocol for checking sync permissions."""
    
    async def can_sync(self, actor: Actor, provider_type: str) -> bool:
        """Check if actor can perform sync operations."""
        ...
    
    async def can_configure_sync(self, actor: Actor) -> bool:
        """Check if actor can configure sync settings."""
        ...


# ============================================================================
# CalendarSyncService
# ============================================================================


class CalendarSyncService:
    """Service for managing calendar synchronization.
    
    Provides a service layer over the SyncEngine, adding:
    - Actor-based permission checking
    - Domain event publishing
    - Async operation support
    - Progress tracking
    - Configuration management
    """
    
    def __init__(
        self,
        sync_engine: Any,  # modules.calendar_store.sync_engine.SyncEngine
        repository: Any,   # CalendarStoreRepository
        event_publisher: Optional[EventPublisher] = None,
        permission_checker: Optional[SyncPermissionChecker] = None,
    ):
        self._engine = sync_engine
        self._repository = repository
        self._publisher = event_publisher
        self._permissions = permission_checker
        
        # Active syncs
        self._active_syncs: Dict[str, SyncResult] = {}
        self._progress_callbacks: Dict[str, List[Callable[[SyncProgress], None]]] = {}
        
        # Configure engine callbacks
        self._engine.set_progress_callback(self._on_engine_progress)
    
    # ========================================================================
    # Provider Management
    # ========================================================================
    
    def list_providers(self) -> List[ProviderInfo]:
        """List all registered sync providers."""
        providers = self._engine.list_providers()
        return [
            ProviderInfo(
                provider_type=p["type"],
                display_name=p["name"],
                supports_push=getattr(
                    self._engine.get_provider(p["type"]),
                    "supports_push",
                    False,
                ),
                supports_incremental=True,
            )
            for p in providers
        ]
    
    def get_provider(self, provider_type: str) -> Optional[ProviderInfo]:
        """Get information about a specific provider."""
        provider = self._engine.get_provider(provider_type)
        if not provider:
            return None
        return ProviderInfo(
            provider_type=provider.provider_type,
            display_name=provider.display_name,
            supports_push=hasattr(provider, "push_event"),
            supports_incremental=True,
        )
    
    def register_provider(self, provider: Any) -> None:
        """Register a new sync provider."""
        self._engine.register_provider(provider)
        logger.info("Registered sync provider: %s", provider.provider_type)
    
    # ========================================================================
    # Sync Operations
    # ========================================================================
    
    async def sync_calendar(
        self,
        actor: Actor,
        config: SyncConfiguration,
    ) -> OperationResult[SyncResult]:
        """Sync a calendar from an external source.
        
        Args:
            actor: The actor performing the sync
            config: Sync configuration
            
        Returns:
            OperationResult containing SyncResult
        """
        # Check permissions
        if self._permissions:
            if not await self._permissions.can_sync(actor, config.provider_type):
                return OperationResult.failure(
                    error=f"PERMISSION_DENIED: Actor lacks permission to sync with {config.provider_type}",
                )
        
        # Generate sync ID
        sync_id = str(uuid4())
        
        # Initialize result
        result = SyncResult(
            sync_id=sync_id,
            status=SyncStatus.IN_PROGRESS,
            provider_type=config.provider_type,
            source_account=config.account_name,
            source_calendar=config.calendar_id,
            target_category_id=config.target_category_id,
            started_at=datetime.now(timezone.utc),
        )
        
        self._active_syncs[sync_id] = result
        
        # Publish start event
        await self._publish_event(
            CalendarSyncStarted.for_sync(
                sync_id=sync_id,
                tenant_id=actor.tenant_id,
                provider_type=config.provider_type,
                source_account=config.account_name,
                source_calendar=config.calendar_id,
                direction=config.direction,
            )
        )
        
        try:
            # Run sync in thread pool (engine is synchronous)
            loop = asyncio.get_event_loop()
            engine_result = await loop.run_in_executor(
                None,
                lambda: self._engine.sync_calendar(
                    provider_type=config.provider_type,
                    config=config.config,
                    calendar_id=config.calendar_id,
                    target_category_id=config.target_category_id,
                    direction=self._convert_direction(config.direction),
                    incremental=config.incremental,
                ),
            )
            
            # Convert engine result to our result type
            result.status = self._convert_status(engine_result.status)
            result.events_added = engine_result.events_added
            result.events_updated = engine_result.events_updated
            result.events_deleted = engine_result.events_deleted
            result.events_skipped = engine_result.events_skipped
            result.errors = engine_result.errors
            result.sync_token = engine_result.sync_token
            
            # Convert conflicts
            for ec in engine_result.conflicts:
                conflict = SyncConflict(
                    conflict_id=str(uuid4()),
                    event_id=ec.event_id,
                    external_id=ec.external_id,
                    local_event=ec.local_event,
                    remote_event=ec.remote_event,
                    conflict_type=ec.conflict_type,
                    resolution=self._convert_resolution(ec.resolution) if ec.resolution else None,
                )
                result.conflicts.append(conflict)
                
                # Publish conflict event
                await self._publish_event(
                    CalendarSyncConflict.for_sync(
                        sync_id=sync_id,
                        tenant_id=actor.tenant_id,
                        conflict=conflict,
                    )
                )
            
            result.completed_at = datetime.now(timezone.utc)
            if result.started_at:
                result.duration_seconds = (
                    result.completed_at - result.started_at
                ).total_seconds()
            
            # Publish completion event
            await self._publish_event(
                CalendarSyncCompleted.for_sync(
                    sync_id=sync_id,
                    tenant_id=actor.tenant_id,
                    result=result,
                )
            )
            
            logger.info(
                "Sync %s completed: +%d ~%d -%d (%.1fs)",
                sync_id,
                result.events_added,
                result.events_updated,
                result.events_deleted,
                result.duration_seconds,
            )
            
            return OperationResult.success(result)
            
        except Exception as e:
            logger.exception("Sync failed: %s", e)
            result.status = SyncStatus.FAILED
            result.errors.append(str(e))
            result.completed_at = datetime.now(timezone.utc)
            if result.started_at:
                result.duration_seconds = (
                    result.completed_at - result.started_at
                ).total_seconds()
            
            # Publish failure event
            await self._publish_event(
                CalendarSyncFailed.for_sync(
                    sync_id=sync_id,
                    tenant_id=actor.tenant_id,
                    provider_type=config.provider_type,
                    error_message=str(e),
                )
            )
            
            return OperationResult.failure(
                error=f"SYNC_FAILED: {e}",
            )
            
        finally:
            del self._active_syncs[sync_id]
    
    async def cancel_sync(
        self,
        actor: Actor,
        sync_id: str,
    ) -> OperationResult[bool]:
        """Cancel an in-progress sync operation.
        
        Args:
            actor: The actor requesting cancellation
            sync_id: ID of the sync to cancel
            
        Returns:
            OperationResult indicating success
        """
        if sync_id not in self._active_syncs:
            return OperationResult.failure(
                error=f"NOT_FOUND: No active sync with ID {sync_id}",
            )
        
        result = self._active_syncs[sync_id]
        
        # Try to cancel via engine
        cancelled = self._engine.cancel_sync(
            result.provider_type,
            result.source_calendar or "",
        )
        
        if cancelled:
            result.status = SyncStatus.CANCELLED
            logger.info("Sync %s cancelled by %s", sync_id, actor.user_id)
            return OperationResult.success(True)
        
        return OperationResult.failure(
            error="CANCEL_FAILED: Unable to cancel sync",
        )
    
    def get_active_syncs(self) -> List[SyncResult]:
        """Get all currently active sync operations."""
        return list(self._active_syncs.values())
    
    def get_sync_status(self, sync_id: str) -> Optional[SyncResult]:
        """Get status of a specific sync operation."""
        return self._active_syncs.get(sync_id)
    
    # ========================================================================
    # Sync History
    # ========================================================================
    
    async def get_sync_history(
        self,
        actor: Actor,
        provider_type: Optional[str] = None,
        calendar_id: Optional[str] = None,
        limit: int = 50,
    ) -> OperationResult[List[Dict[str, Any]]]:
        """Get sync history from stored sync states.
        
        Args:
            actor: The actor requesting history
            provider_type: Optional filter by provider
            calendar_id: Optional filter by calendar
            limit: Maximum results
            
        Returns:
            OperationResult with list of sync history entries
        """
        try:
            # Query sync state from repository
            states = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._repository.list_sync_states(
                    source_type=provider_type,
                    source_calendar=calendar_id,
                    limit=limit,
                ),
            )
            return OperationResult.success(states or [])
        except Exception as e:
            logger.exception("Failed to get sync history: %s", e)
            return OperationResult.failure(
                error=f"QUERY_FAILED: {e}",
            )
    
    async def get_last_sync_time(
        self,
        provider_type: str,
        account: str,
        calendar_id: str,
    ) -> Optional[datetime]:
        """Get the last sync time for a specific calendar."""
        try:
            state = self._repository.get_sync_state(
                provider_type, account, calendar_id
            )
            if state:
                return state.get("last_sync_at")
            return None
        except Exception:
            return None
    
    # ========================================================================
    # Conflict Resolution
    # ========================================================================
    
    async def resolve_conflict(
        self,
        actor: Actor,
        sync_id: str,
        conflict_id: str,
        resolution: ConflictResolution,
        merged_event: Optional[Dict[str, Any]] = None,
    ) -> OperationResult[bool]:
        """Resolve a sync conflict.
        
        Args:
            actor: The actor resolving the conflict
            sync_id: ID of the sync operation
            conflict_id: ID of the conflict to resolve
            resolution: How to resolve the conflict
            merged_event: Optional merged event data (for MERGE resolution)
            
        Returns:
            OperationResult indicating success
        """
        # Find the sync result
        result = self._active_syncs.get(sync_id)
        if not result:
            return OperationResult.failure(
                error=f"NOT_FOUND: No active sync with ID {sync_id}",
            )
        
        # Find the conflict
        conflict = next(
            (c for c in result.conflicts if c.conflict_id == conflict_id),
            None,
        )
        if not conflict:
            return OperationResult.failure(
                error=f"NOT_FOUND: No conflict with ID {conflict_id}",
            )
        
        # Apply resolution
        conflict.resolution = resolution
        
        if resolution == ConflictResolution.LOCAL_WINS:
            # Keep local, no action needed
            pass
        elif resolution == ConflictResolution.REMOTE_WINS:
            # Apply remote changes
            try:
                self._repository.update_event(
                    conflict.event_id,
                    **conflict.remote_event,
                )
            except Exception as e:
                return OperationResult.failure(
                    error=f"UPDATE_FAILED: {e}",
                )
        elif resolution == ConflictResolution.MERGE:
            if not merged_event:
                return OperationResult.failure(
                    error="INVALID_ARGUMENT: Merged event data required for MERGE resolution",
                )
            conflict.resolved_event = merged_event
            try:
                self._repository.update_event(
                    conflict.event_id,
                    **merged_event,
                )
            except Exception as e:
                return OperationResult.failure(
                    error=f"UPDATE_FAILED: {e}",
                )
        
        logger.info(
            "Conflict %s resolved as %s by %s",
            conflict_id,
            resolution.value,
            actor.user_id,
        )
        
        return OperationResult.success(True)
    
    # ========================================================================
    # Progress Tracking
    # ========================================================================
    
    def subscribe_to_progress(
        self,
        sync_id: str,
        callback: Callable[[SyncProgress], None],
    ) -> None:
        """Subscribe to progress updates for a sync operation."""
        if sync_id not in self._progress_callbacks:
            self._progress_callbacks[sync_id] = []
        self._progress_callbacks[sync_id].append(callback)
    
    def unsubscribe_from_progress(
        self,
        sync_id: str,
        callback: Callable[[SyncProgress], None],
    ) -> None:
        """Unsubscribe from progress updates."""
        if sync_id in self._progress_callbacks:
            try:
                self._progress_callbacks[sync_id].remove(callback)
            except ValueError:
                pass
    
    def _on_engine_progress(self, message: str, current: int, total: int) -> None:
        """Handle progress callback from engine."""
        # Find active sync for this progress update
        for sync_id, result in self._active_syncs.items():
            progress = SyncProgress(
                sync_id=sync_id,
                provider_type=result.provider_type,
                calendar_id=result.source_calendar or "",
                message=message,
                current=current,
                total=total,
                percent=(current / max(total, 1)) * 100,
            )
            
            # Notify subscribers
            for callback in self._progress_callbacks.get(sync_id, []):
                try:
                    callback(progress)
                except Exception as e:
                    logger.warning("Progress callback error: %s", e)
    
    # ========================================================================
    # Helpers
    # ========================================================================
    
    async def _publish_event(self, event: DomainEvent) -> None:
        """Publish a domain event."""
        if self._publisher:
            try:
                await self._publisher.publish(event)
            except Exception as e:
                logger.warning("Failed to publish event: %s", e)
    
    def _convert_direction(self, direction: SyncDirection) -> Any:
        """Convert our SyncDirection to engine's SyncDirection."""
        from modules.calendar_store.sync_engine import SyncDirection as EngineDirection
        mapping = {
            SyncDirection.IMPORT: EngineDirection.IMPORT,
            SyncDirection.EXPORT: EngineDirection.EXPORT,
            SyncDirection.BIDIRECTIONAL: EngineDirection.BIDIRECTIONAL,
        }
        return mapping.get(direction, EngineDirection.IMPORT)
    
    def _convert_status(self, status: Any) -> SyncStatus:
        """Convert engine SyncStatus to our SyncStatus."""
        from modules.calendar_store.sync_engine import SyncStatus as EngineStatus
        mapping = {
            EngineStatus.PENDING: SyncStatus.PENDING,
            EngineStatus.IN_PROGRESS: SyncStatus.IN_PROGRESS,
            EngineStatus.SUCCESS: SyncStatus.SUCCESS,
            EngineStatus.PARTIAL: SyncStatus.PARTIAL,
            EngineStatus.FAILED: SyncStatus.FAILED,
            EngineStatus.CANCELLED: SyncStatus.CANCELLED,
        }
        return mapping.get(status, SyncStatus.FAILED)
    
    def _convert_resolution(self, resolution: Any) -> ConflictResolution:
        """Convert engine ConflictResolution to our ConflictResolution."""
        from modules.calendar_store.sync_engine import (
            ConflictResolution as EngineResolution,
        )
        mapping = {
            EngineResolution.ASK: ConflictResolution.ASK,
            EngineResolution.LOCAL_WINS: ConflictResolution.LOCAL_WINS,
            EngineResolution.REMOTE_WINS: ConflictResolution.REMOTE_WINS,
            EngineResolution.NEWEST_WINS: ConflictResolution.NEWEST_WINS,
            EngineResolution.MERGE: ConflictResolution.MERGE,
        }
        return mapping.get(resolution, ConflictResolution.ASK)


# ============================================================================
# Factory
# ============================================================================


def create_sync_service(
    repository: Any,
    event_publisher: Optional[EventPublisher] = None,
    permission_checker: Optional[SyncPermissionChecker] = None,
    conflict_resolution: ConflictResolution = ConflictResolution.REMOTE_WINS,
) -> CalendarSyncService:
    """Create a CalendarSyncService with its dependencies.
    
    Args:
        repository: CalendarStoreRepository instance
        event_publisher: Optional event publisher for domain events
        permission_checker: Optional permission checker
        conflict_resolution: Default conflict resolution strategy
        
    Returns:
        Configured CalendarSyncService
    """
    from modules.calendar_store.sync_engine import (
        SyncEngine,
        ConflictResolution as EngineResolution,
    )
    from modules.calendar_store.providers.ics_provider import ICSProvider
    
    # Map our resolution to engine's
    engine_resolution_map = {
        ConflictResolution.ASK: EngineResolution.ASK,
        ConflictResolution.LOCAL_WINS: EngineResolution.LOCAL_WINS,
        ConflictResolution.REMOTE_WINS: EngineResolution.REMOTE_WINS,
        ConflictResolution.NEWEST_WINS: EngineResolution.NEWEST_WINS,
        ConflictResolution.MERGE: EngineResolution.MERGE,
    }
    
    # Create engine
    engine = SyncEngine(
        repository=repository,
        conflict_resolution=engine_resolution_map.get(
            conflict_resolution, EngineResolution.REMOTE_WINS
        ),
    )
    
    # Register default providers
    engine.register_provider(ICSProvider())
    
    # Try to register CalDAV provider if available
    try:
        from modules.calendar_store.providers.caldav_provider import CalDAVProvider
        engine.register_provider(CalDAVProvider())
    except ImportError:
        logger.debug("CalDAV provider not available")
    
    return CalendarSyncService(
        sync_engine=engine,
        repository=repository,
        event_publisher=event_publisher,
        permission_checker=permission_checker,
    )
