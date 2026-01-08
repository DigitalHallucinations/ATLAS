"""Tests for CalendarSyncService.

This module tests the service layer for calendar synchronization,
including sync operations, conflict resolution, and domain events.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from core.services.common import Actor, OperationResult
from core.services.calendar.sync_service import (
    CalendarSyncService,
    SyncConfiguration,
    SyncConflict,
    SyncDirection,
    SyncProgress,
    SyncResult,
    SyncStatus,
    ConflictResolution,
    ProviderInfo,
    CalendarSyncStarted,
    CalendarSyncProgress,
    CalendarSyncCompleted,
    CalendarSyncConflict,
    CalendarSyncFailed,
    create_sync_service,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_sync_engine():
    """Create a mock SyncEngine."""
    engine = MagicMock()
    engine.register_provider = MagicMock()
    engine.get_provider = MagicMock(return_value=None)
    engine.list_providers = MagicMock(return_value={})
    engine.sync = MagicMock(return_value=MagicMock(
        success=True,
        events_added=5,
        events_updated=2,
        events_deleted=1,
        errors=[],
        conflicts=[],
        sync_token="test-token-123",
    ))
    return engine


@pytest.fixture
def mock_repository():
    """Create a mock repository."""
    repo = MagicMock()
    repo.get_category = MagicMock(return_value={"id": str(uuid4()), "name": "Test"})
    repo.create_event = MagicMock(return_value=str(uuid4()))
    repo.update_event = MagicMock()
    repo.delete_event = MagicMock()
    return repo


@pytest.fixture
def mock_event_publisher():
    """Create a mock event publisher."""
    publisher = MagicMock()
    publisher.publish = AsyncMock()
    return publisher


@pytest.fixture
def mock_permission_checker():
    """Create a mock permission checker."""
    checker = MagicMock()
    checker.can_sync = AsyncMock(return_value=True)
    checker.can_configure_sync = AsyncMock(return_value=True)
    return checker


@pytest.fixture
def sync_service(mock_sync_engine, mock_repository, mock_event_publisher, mock_permission_checker):
    """Create a CalendarSyncService with mocked dependencies."""
    return CalendarSyncService(
        sync_engine=mock_sync_engine,
        repository=mock_repository,
        event_publisher=mock_event_publisher,
        permission_checker=mock_permission_checker,
    )


@pytest.fixture
def actor():
    """Create a test actor."""
    return Actor(
        type="user",
        id="user-123",
        tenant_id="tenant-456",
        permissions=["calendar:sync", "calendar:read", "calendar:write"],
    )


@pytest.fixture
def sync_config():
    """Create a test sync configuration."""
    return SyncConfiguration(
        provider_type="ics",
        account_name="test-calendar",
        calendar_id="cal-123",
        config={"url": "https://example.com/calendar.ics"},
        target_category_id=str(uuid4()),
        direction=SyncDirection.IMPORT,
        conflict_resolution=ConflictResolution.REMOTE_WINS,
    )


# ============================================================================
# SyncConfiguration Tests
# ============================================================================


class TestSyncConfiguration:
    """Tests for SyncConfiguration dataclass."""

    def test_create_minimal_config(self):
        """Test creating a config with minimal required fields."""
        config = SyncConfiguration(
            provider_type="ics",
            account_name="my-calendar",
            calendar_id="default",
            config={"url": "http://example.com/cal.ics"},
        )
        
        assert config.provider_type == "ics"
        assert config.account_name == "my-calendar"
        assert config.direction == SyncDirection.IMPORT  # default
        assert config.conflict_resolution == ConflictResolution.REMOTE_WINS  # default
        assert config.incremental is True  # default
        assert config.auto_sync_enabled is False  # default

    def test_create_full_config(self):
        """Test creating a config with all fields."""
        config = SyncConfiguration(
            provider_type="caldav",
            account_name="work",
            calendar_id="cal-uuid",
            config={"server": "https://caldav.example.com", "username": "user"},
            target_category_id="category-123",
            direction=SyncDirection.BIDIRECTIONAL,
            conflict_resolution=ConflictResolution.NEWEST_WINS,
            incremental=False,
            auto_sync_enabled=True,
            sync_interval_minutes=30,
        )
        
        assert config.direction == SyncDirection.BIDIRECTIONAL
        assert config.conflict_resolution == ConflictResolution.NEWEST_WINS
        assert config.incremental is False
        assert config.auto_sync_enabled is True
        assert config.sync_interval_minutes == 30


# ============================================================================
# SyncResult Tests
# ============================================================================


class TestSyncResult:
    """Tests for SyncResult dataclass."""

    def test_create_success_result(self):
        """Test creating a successful sync result."""
        result = SyncResult(
            sync_id="sync-123",
            status=SyncStatus.SUCCESS,
            provider_type="ics",
            source_account="test",
            events_added=10,
            events_updated=5,
            events_deleted=2,
        )
        
        assert result.status == SyncStatus.SUCCESS
        assert result.total_processed == 17
        assert result.has_errors is False
        assert result.has_unresolved_conflicts is False

    def test_result_with_errors(self):
        """Test sync result with errors."""
        result = SyncResult(
            sync_id="sync-123",
            status=SyncStatus.PARTIAL,
            provider_type="ics",
            source_account="test",
            errors=["Failed to parse event", "Network timeout"],
        )
        
        assert result.has_errors is True
        assert len(result.errors) == 2

    def test_result_with_unresolved_conflicts(self):
        """Test sync result with unresolved conflicts."""
        conflict = SyncConflict(
            conflict_id="conf-1",
            event_id="evt-1",
            external_id="ext-1",
            local_event={"title": "Local Meeting"},
            remote_event={"title": "Remote Meeting"},
            conflict_type="modified",
            resolution=None,  # Unresolved
        )
        
        result = SyncResult(
            sync_id="sync-123",
            status=SyncStatus.PARTIAL,
            provider_type="ics",
            source_account="test",
            conflicts=[conflict],
        )
        
        assert result.has_unresolved_conflicts is True

    def test_result_to_dict(self):
        """Test converting result to dictionary."""
        result = SyncResult(
            sync_id="sync-123",
            status=SyncStatus.SUCCESS,
            provider_type="ics",
            source_account="test",
            events_added=5,
            started_at=datetime(2026, 1, 8, 10, 0, 0, tzinfo=timezone.utc),
            completed_at=datetime(2026, 1, 8, 10, 0, 30, tzinfo=timezone.utc),
            duration_seconds=30.0,
        )
        
        data = result.to_dict()
        
        assert data["sync_id"] == "sync-123"
        assert data["status"] == "success"
        assert data["events_added"] == 5
        assert data["duration_seconds"] == 30.0
        assert "2026-01-08" in data["started_at"]


# ============================================================================
# SyncConflict Tests
# ============================================================================


class TestSyncConflict:
    """Tests for SyncConflict dataclass."""

    def test_create_conflict(self):
        """Test creating a sync conflict."""
        conflict = SyncConflict(
            conflict_id="conf-123",
            event_id="event-456",
            external_id="ext-789",
            local_event={"title": "Team Standup", "start": "2026-01-08T09:00:00Z"},
            remote_event={"title": "Daily Standup", "start": "2026-01-08T09:30:00Z"},
            conflict_type="modified",
        )
        
        assert conflict.conflict_id == "conf-123"
        assert conflict.conflict_type == "modified"
        assert conflict.resolution is None
        assert conflict.resolved_event is None

    def test_conflict_with_resolution(self):
        """Test conflict with resolution applied."""
        conflict = SyncConflict(
            conflict_id="conf-123",
            event_id="event-456",
            external_id="ext-789",
            local_event={"title": "Team Standup"},
            remote_event={"title": "Daily Standup"},
            conflict_type="modified",
            resolution=ConflictResolution.LOCAL_WINS,
            resolved_event={"title": "Team Standup"},
        )
        
        assert conflict.resolution == ConflictResolution.LOCAL_WINS
        assert conflict.resolved_event == {"title": "Team Standup"}


# ============================================================================
# ProviderInfo Tests
# ============================================================================


class TestProviderInfo:
    """Tests for ProviderInfo dataclass."""

    def test_create_provider_info(self):
        """Test creating provider info."""
        info = ProviderInfo(
            provider_type="caldav",
            display_name="CalDAV",
            supports_push=True,
            supports_incremental=True,
            requires_auth=True,
        )
        
        assert info.provider_type == "caldav"
        assert info.display_name == "CalDAV"
        assert info.supports_push is True
        assert info.requires_auth is True


# ============================================================================
# Domain Event Tests
# ============================================================================


class TestSyncDomainEvents:
    """Tests for sync domain events."""

    def test_sync_started_event(self):
        """Test CalendarSyncStarted event creation."""
        event = CalendarSyncStarted.for_sync(
            sync_id="sync-123",
            tenant_id="tenant-456",
            provider_type="ics",
            source_account="test",
            source_calendar="default",
            direction=SyncDirection.IMPORT,
        )
        
        assert event.sync_id == "sync-123"
        assert event.tenant_id == "tenant-456"
        assert event.provider_type == "ics"
        assert event.direction == "import"

    def test_sync_progress_event(self):
        """Test CalendarSyncProgress event creation."""
        event = CalendarSyncProgress.for_sync(
            sync_id="sync-123",
            tenant_id="tenant-456",
            current=50,
            total=100,
            message="Processing events...",
        )
        
        assert event.sync_id == "sync-123"
        assert event.current == 50
        assert event.total == 100
        assert event.percent == 50.0

    def test_sync_completed_event(self):
        """Test CalendarSyncCompleted event creation."""
        result = SyncResult(
            sync_id="sync-123",
            status=SyncStatus.SUCCESS,
            provider_type="ics",
            source_account="test",
            events_added=10,
        )
        
        event = CalendarSyncCompleted.for_sync(
            sync_id="sync-123",
            tenant_id="tenant-456",
            result=result,
        )
        
        assert event.sync_id == "sync-123"
        assert event.status == "success"
        assert event.events_added == 10

    def test_sync_conflict_event(self):
        """Test CalendarSyncConflict event creation."""
        conflict = SyncConflict(
            conflict_id="conf-1",
            event_id="evt-1",
            external_id="ext-1",
            local_event={"title": "Local"},
            remote_event={"title": "Remote"},
            conflict_type="modified",
        )
        
        event = CalendarSyncConflict.for_sync(
            sync_id="sync-123",
            tenant_id="tenant-456",
            conflict=conflict,
        )
        
        assert event.sync_id == "sync-123"
        assert event.conflict_id == "conf-1"
        assert event.conflict_type == "modified"

    def test_sync_failed_event(self):
        """Test CalendarSyncFailed event creation."""
        event = CalendarSyncFailed.for_sync(
            sync_id="sync-123",
            tenant_id="tenant-456",
            provider_type="caldav",
            error_message="Connection refused",
        )
        
        assert event.sync_id == "sync-123"
        assert event.provider_type == "caldav"
        assert event.error_message == "Connection refused"


# ============================================================================
# CalendarSyncService Tests
# ============================================================================


class TestCalendarSyncService:
    """Tests for CalendarSyncService."""

    def test_register_provider(self, sync_service):
        """Test registering a sync provider."""
        mock_provider = MagicMock()
        mock_provider.provider_type = "google"
        mock_provider.display_name = "Google Calendar"
        
        sync_service.register_provider(mock_provider)
        
        # Verify the engine's register_provider was called
        sync_service._engine.register_provider.assert_called_once_with(mock_provider)

    def test_list_providers(self, sync_service, mock_sync_engine):
        """Test listing registered providers."""
        # Setup mock
        mock_sync_engine.list_providers.return_value = [
            {"type": "ics", "name": "ICS Files"}
        ]
        mock_provider = MagicMock()
        mock_provider.provider_type = "ics"
        mock_provider.display_name = "ICS Files"
        mock_sync_engine.get_provider.return_value = mock_provider
        
        result = sync_service.list_providers()
        
        assert len(result) == 1
        assert result[0].provider_type == "ics"

    @pytest.mark.asyncio
    async def test_sync_calendar_success(self, sync_service, actor, sync_config, mock_sync_engine, mock_event_publisher):
        """Test successful calendar sync."""
        from modules.calendar_store.sync_engine import SyncStatus as EngineStatus
        
        # Setup mock to return proper values (not MagicMock)
        mock_engine_result = MagicMock()
        mock_engine_result.status = EngineStatus.SUCCESS
        mock_engine_result.events_added = 5
        mock_engine_result.events_updated = 2
        mock_engine_result.events_deleted = 1
        mock_engine_result.events_skipped = 0
        mock_engine_result.errors = []
        mock_engine_result.conflicts = []
        mock_engine_result.sync_token = "token-123"
        mock_sync_engine.sync_calendar.return_value = mock_engine_result
        
        result = await sync_service.sync_calendar(actor, sync_config)
        
        assert result.success is True
        sync_result = result.value
        assert sync_result.status in [SyncStatus.SUCCESS, SyncStatus.PARTIAL]
        
        # Verify events were published
        assert mock_event_publisher.publish.call_count >= 2  # Started + Completed

    @pytest.mark.asyncio
    async def test_sync_calendar_permission_denied(
        self, sync_service, actor, sync_config, mock_permission_checker
    ):
        """Test sync denied due to permissions."""
        mock_permission_checker.can_sync.return_value = False
        
        result = await sync_service.sync_calendar(actor, sync_config)
        
        assert result.success is False
        assert "PERMISSION_DENIED" in result.error

    @pytest.mark.asyncio
    async def test_sync_calendar_tracks_active_sync(self, sync_service, actor, sync_config, mock_sync_engine):
        """Test that active syncs are tracked."""
        # Setup mock to return proper values
        mock_engine_result = MagicMock()
        mock_engine_result.events_added = 1
        mock_engine_result.events_updated = 0
        mock_engine_result.events_deleted = 0
        mock_engine_result.events_skipped = 0
        mock_engine_result.errors = []
        mock_engine_result.conflicts = []
        mock_engine_result.sync_token = "token"
        mock_sync_engine.sync_calendar.return_value = mock_engine_result
        
        # Start sync
        result = await sync_service.sync_calendar(actor, sync_config)
        
        assert result.success is True
        # After completion, active sync should be cleared
        assert len(sync_service._active_syncs) == 0

    def test_get_sync_status(self, sync_service):
        """Test getting current sync status."""
        # Create an active sync
        sync_id = "test-sync-123"
        sync_result = SyncResult(
            sync_id=sync_id,
            status=SyncStatus.IN_PROGRESS,
            provider_type="ics",
            source_account="test",
        )
        sync_service._active_syncs[sync_id] = sync_result
        
        result = sync_service.get_sync_status(sync_id)
        
        assert result is not None
        assert result.sync_id == sync_id

    def test_get_sync_status_not_found(self, sync_service):
        """Test getting status for non-existent sync."""
        result = sync_service.get_sync_status("nonexistent")
        
        assert result is None

    @pytest.mark.asyncio
    async def test_cancel_sync(self, sync_service, actor, mock_sync_engine):
        """Test cancelling an active sync."""
        sync_id = "test-sync-123"
        sync_result = SyncResult(
            sync_id=sync_id,
            status=SyncStatus.IN_PROGRESS,
            provider_type="ics",
            source_account="test",
            source_calendar="cal-1",
        )
        sync_service._active_syncs[sync_id] = sync_result
        mock_sync_engine.cancel_sync.return_value = True
        
        result = await sync_service.cancel_sync(actor, sync_id)
        
        assert result.success is True


class TestSyncConflictResolution:
    """Tests for conflict resolution in CalendarSyncService."""

    @pytest.mark.asyncio
    async def test_resolve_conflict_local_wins(self, sync_service, actor):
        """Test resolving conflict with local wins."""
        conflict_id = "conflict-123"
        sync_id = "sync-456"
        
        # Setup active sync with conflict
        conflict = SyncConflict(
            conflict_id=conflict_id,
            event_id="evt-1",
            external_id="ext-1",
            local_event={"title": "Local Meeting"},
            remote_event={"title": "Remote Meeting"},
            conflict_type="modified",
        )
        sync_result = SyncResult(
            sync_id=sync_id,
            status=SyncStatus.IN_PROGRESS,
            provider_type="ics",
            source_account="test",
            conflicts=[conflict],
        )
        sync_service._active_syncs[sync_id] = sync_result
        
        result = await sync_service.resolve_conflict(
            actor,
            sync_id,
            conflict_id,
            ConflictResolution.LOCAL_WINS,
        )
        
        assert result.success is True

    @pytest.mark.asyncio
    async def test_resolve_conflict_remote_wins(self, sync_service, actor, mock_repository):
        """Test resolving conflict with remote wins."""
        conflict_id = "conflict-123"
        sync_id = "sync-456"
        
        conflict = SyncConflict(
            conflict_id=conflict_id,
            event_id="evt-1",
            external_id="ext-1",
            local_event={"title": "Local Meeting"},
            remote_event={"title": "Remote Meeting"},
            conflict_type="modified",
        )
        sync_result = SyncResult(
            sync_id=sync_id,
            status=SyncStatus.IN_PROGRESS,
            provider_type="ics",
            source_account="test",
            conflicts=[conflict],
        )
        sync_service._active_syncs[sync_id] = sync_result
        
        result = await sync_service.resolve_conflict(
            actor,
            sync_id,
            conflict_id,
            ConflictResolution.REMOTE_WINS,
        )
        
        assert result.success is True
        mock_repository.update_event.assert_called()

    @pytest.mark.asyncio
    async def test_resolve_conflict_not_found(self, sync_service, actor):
        """Test resolving non-existent conflict."""
        result = await sync_service.resolve_conflict(
            actor,
            "sync-1",
            "nonexistent",
            ConflictResolution.LOCAL_WINS,
        )
        
        assert result.success is False
        assert "NOT_FOUND" in result.error


class TestSyncServiceFactory:
    """Tests for create_sync_service factory function."""

    def test_create_sync_service_with_mocks(self, mock_repository):
        """Test creating sync service with factory."""
        with patch("modules.calendar_store.sync_engine.SyncEngine") as MockEngine:
            with patch("modules.calendar_store.providers.ics_provider.ICSProvider") as MockICS:
                mock_engine = MagicMock()
                MockEngine.return_value = mock_engine
                MockICS.return_value = MagicMock()
                
                service = create_sync_service(
                    repository=mock_repository,
                    conflict_resolution=ConflictResolution.LOCAL_WINS,
                )
                
                assert service is not None
                assert isinstance(service, CalendarSyncService)


# ============================================================================
# Integration-Style Tests
# ============================================================================


class TestSyncServiceIntegration:
    """Integration-style tests for sync scenarios."""

    @pytest.mark.asyncio
    async def test_full_sync_workflow(
        self, sync_service, actor, sync_config, mock_sync_engine, mock_event_publisher
    ):
        """Test complete sync workflow from start to finish."""
        # Configure mock engine to return realistic result
        mock_result = MagicMock()
        mock_result.events_added = 10
        mock_result.events_updated = 3
        mock_result.events_deleted = 1
        mock_result.events_skipped = 0
        mock_result.errors = []
        mock_result.conflicts = []
        mock_result.sync_token = "new-token-abc"
        mock_sync_engine.sync_calendar.return_value = mock_result
        
        # Execute sync
        result = await sync_service.sync_calendar(actor, sync_config)
        
        # Verify success
        assert result.success is True
        sync_result = result.value
        assert sync_result.events_added == 10
        assert sync_result.events_updated == 3
        assert sync_result.events_deleted == 1
        assert sync_result.sync_token == "new-token-abc"
        
        # Verify events published
        published_events = [
            call.args[0] for call in mock_event_publisher.publish.call_args_list
        ]
        event_types = [type(e).__name__ for e in published_events]
        
        assert "CalendarSyncStarted" in event_types
        assert "CalendarSyncCompleted" in event_types

    @pytest.mark.asyncio
    async def test_sync_with_conflicts(
        self, sync_service, actor, sync_config, mock_sync_engine, mock_event_publisher
    ):
        """Test sync workflow with conflicts."""
        # Configure mock to return conflicts
        mock_conflict = MagicMock()
        mock_conflict.conflict_id = "conf-1"
        mock_conflict.event_id = "evt-1"
        mock_conflict.external_id = "ext-1"
        mock_conflict.local_event = {"title": "Local"}
        mock_conflict.remote_event = {"title": "Remote"}
        mock_conflict.conflict_type = "modified"
        mock_conflict.resolution = None
        
        mock_result = MagicMock()
        mock_result.events_added = 5
        mock_result.events_updated = 2
        mock_result.events_deleted = 0
        mock_result.events_skipped = 0
        mock_result.errors = []
        mock_result.conflicts = [mock_conflict]
        mock_result.sync_token = "token"
        mock_sync_engine.sync_calendar.return_value = mock_result
        
        # Execute sync
        result = await sync_service.sync_calendar(actor, sync_config)
        
        # Verify success (may have conflicts)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_sync_failure_publishes_event(
        self, sync_service, actor, sync_config, mock_sync_engine, mock_event_publisher
    ):
        """Test that sync failure publishes failure event."""
        # Configure mock to raise exception
        mock_sync_engine.sync_calendar.side_effect = Exception("Network error")
        
        # Execute sync
        result = await sync_service.sync_calendar(actor, sync_config)
        
        # Verify failure
        assert result.success is False
        
        # Verify failure event published
        published_events = [
            call.args[0] for call in mock_event_publisher.publish.call_args_list
        ]
        event_types = [type(e).__name__ for e in published_events]
        
        assert "CalendarSyncFailed" in event_types
