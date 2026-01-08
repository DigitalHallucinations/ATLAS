"""
Tests for calendar permission checking.

Tests the calendar-specific permission logic including hierarchical
permissions and calendar context-aware access control.

Author: ATLAS Team
Date: Jan 7, 2026
"""

import pytest
from unittest.mock import Mock, AsyncMock

from core.services.common import Actor, PermissionDeniedError
from core.services.calendar.permissions import CalendarPermissionChecker
from core.services.calendar.types import CalendarEvent, EventVisibility, EventStatus
from modules.calendar_store.dataclasses import BusyStatus
from datetime import datetime, timezone


def make_test_event(
    event_id: str = "evt-123",
    title: str = "Test Event",
    visibility: EventVisibility = EventVisibility.PRIVATE,
    created_by: str = "user-123",
    **kwargs
) -> CalendarEvent:
    """Helper to create CalendarEvent instances with all required fields."""
    defaults = {
        "event_id": event_id,
        "title": title,
        "description": "Test description",
        "start_time": datetime(2026, 2, 1, 10, 0, tzinfo=timezone.utc),
        "end_time": datetime(2026, 2, 1, 11, 0, tzinfo=timezone.utc),
        "timezone_name": "UTC",
        "location": None,
        "status": EventStatus.CONFIRMED,
        "visibility": visibility,
        "busy_status": BusyStatus.BUSY,
        "all_day": True,
        "category_id": None,
        "created_by": created_by,
        "tenant_id": "tenant-1",
        "created_at": datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc),
        "updated_at": datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc),
        "is_recurring": False,
        "recurrence_pattern": None,
    }
    defaults.update(kwargs)
    return CalendarEvent(**defaults)


@pytest.fixture
def base_checker():
    """Mock base permission checker."""
    mock_checker = Mock()
    mock_checker.has_permission = AsyncMock()
    mock_checker.require = AsyncMock()
    return mock_checker


@pytest.fixture
def calendar_checker(base_checker):
    """Create a CalendarPermissionChecker for testing."""
    return CalendarPermissionChecker(base_checker)


@pytest.fixture
def user_actor():
    """Standard user actor."""
    return Actor(type="user", id="user-123", tenant_id="tenant-1", permissions={"calendar:read", "calendar:write"})


@pytest.fixture
def admin_actor():
    """Admin user actor."""
    return Actor(type="user", id="admin-456", tenant_id="tenant-1", permissions={"calendar:admin", "calendar:write", "calendar:read"})


@pytest.fixture
def readonly_actor():
    """Read-only user actor."""
    return Actor(type="user", id="readonly-789", tenant_id="tenant-1", permissions={"calendar:read"})


@pytest.fixture
def public_event():
    """A public calendar event."""
    return make_test_event(
        event_id="evt-public",
        title="Public Meeting",
        description="Open to all",
        visibility=EventVisibility.PUBLIC,
        created_by="user-123",
    )


@pytest.fixture
def private_event():
    """A private calendar event."""
    return make_test_event(
        event_id="evt-private",
        title="Private Meeting",
        description="Sensitive content",
        visibility=EventVisibility.PRIVATE,
        created_by="user-123",
    )


class TestBasicPermissionChecking:
    """Test basic calendar permission operations."""
    
    @pytest.mark.asyncio
    async def test_can_read_calendar(self, calendar_checker, base_checker, user_actor):
        """Users with calendar.read should be able to read calendar."""
        base_checker.has_permission.return_value = True
        
        result = await calendar_checker.can_read_calendar(user_actor)
        
        assert result is True
        base_checker.has_permission.assert_called_once_with(user_actor, "calendar:read")
    
    @pytest.mark.asyncio
    async def test_cannot_read_calendar_without_permission(self, calendar_checker, base_checker):
        """Users without calendar.read should not be able to read."""
        base_checker.has_permission.return_value = False
        actor = Actor(type="user", id="user-no-perms", tenant_id="tenant-1", permissions=set())
        
        result = await calendar_checker.can_read_calendar(actor)
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_can_write_calendar(self, calendar_checker, base_checker, user_actor):
        """Users with calendar.write should be able to write calendar."""
        base_checker.has_permission.return_value = True
        
        result = await calendar_checker.can_write_calendar(user_actor)
        
        assert result is True
        base_checker.has_permission.assert_called_once_with(user_actor, "calendar:write")
    
    @pytest.mark.asyncio
    async def test_can_admin_calendar(self, calendar_checker, base_checker, admin_actor):
        """Users with calendar.admin should be able to admin calendar."""
        base_checker.has_permission.return_value = True
        
        result = await calendar_checker.can_admin_calendar(admin_actor)
        
        assert result is True
        base_checker.has_permission.assert_called_once_with(admin_actor, "calendar:admin")


class TestEventSpecificPermissions:
    """Test event-specific permission logic."""
    
    @pytest.mark.asyncio
    async def test_can_read_public_event_with_read_permission(
        self, 
        calendar_checker, 
        base_checker, 
        user_actor, 
        public_event
    ):
        """Users with calendar.read can read public events."""
        # Return True only for calendar:read, False for admin
        base_checker.has_permission.side_effect = lambda actor, perm: perm == "calendar:read"
        
        result = await calendar_checker.can_read_event(user_actor, public_event)
        
        assert result is True
        # Implementation first checks admin, then read for public events
        assert base_checker.has_permission.call_count >= 1
    
    @pytest.mark.asyncio
    async def test_cannot_read_public_event_without_read_permission(
        self,
        calendar_checker,
        base_checker,
        public_event
    ):
        """Users without calendar.read cannot read even public events."""
        base_checker.has_permission.return_value = False
        actor = Actor(type="user", id="no-perms", tenant_id="tenant-1", permissions=[])
        
        result = await calendar_checker.can_read_event(actor, public_event)
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_owner_can_read_own_private_event(
        self,
        calendar_checker,
        base_checker,
        private_event
    ):
        """Event owners can read their own private events."""
        # Create actor who owns the event
        owner_actor = Actor(type="user", id="user-123", tenant_id="tenant-1", permissions=["calendar:read"])
        base_checker.has_permission.return_value = True
        
        result = await calendar_checker.can_read_event(owner_actor, private_event)
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_non_owner_cannot_read_private_event(
        self,
        calendar_checker,
        base_checker,
        private_event
    ):
        """Non-owners cannot read private events even with calendar.read."""
        other_actor = Actor(type="user", id="other-user", tenant_id="tenant-1", permissions=["calendar:read"])
        # Has read permission but NOT admin - still should be denied for private event they don't own
        base_checker.has_permission.side_effect = lambda actor, perm: perm == "calendar:read"
        
        result = await calendar_checker.can_read_event(other_actor, private_event)
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_admin_can_read_any_private_event(
        self,
        calendar_checker,
        base_checker,
        private_event
    ):
        """Admins can read any private event."""
        admin_actor = Actor(type="user", id="admin-user", tenant_id="tenant-1", permissions=["calendar:admin"])
        base_checker.has_permission.side_effect = lambda actor, perm: perm == "calendar:admin"
        
        result = await calendar_checker.can_read_event(admin_actor, private_event)
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_can_edit_own_event(
        self,
        calendar_checker,
        base_checker,
        public_event
    ):
        """Users can edit events they created."""
        owner_actor = Actor(type="user", id="user-123", tenant_id="tenant-1", permissions=["calendar:write"])
        base_checker.has_permission.return_value = True
        
        result = await calendar_checker.can_edit_event(owner_actor, public_event)
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_cannot_edit_others_event_without_admin(
        self,
        calendar_checker,
        base_checker,
        public_event
    ):
        """Non-owners cannot edit events without admin permission."""
        other_actor = Actor(type="user", id="other-user", tenant_id="tenant-1", permissions=["calendar:write"])
        # has_permission returns True for calendar.write, False for calendar.admin
        base_checker.has_permission.side_effect = lambda actor, perm: perm == "calendar:write"
        
        result = await calendar_checker.can_edit_event(other_actor, public_event)
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_admin_can_edit_any_event(
        self,
        calendar_checker,
        base_checker,
        public_event
    ):
        """Admins can edit any event."""
        admin_actor = Actor(type="user", id="admin-user", tenant_id="tenant-1", permissions=["calendar:admin"])
        base_checker.has_permission.side_effect = lambda actor, perm: perm == "calendar:admin"
        
        result = await calendar_checker.can_edit_event(admin_actor, public_event)
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_can_delete_own_event(
        self,
        calendar_checker,
        base_checker,
        public_event
    ):
        """Users can delete events they created."""
        owner_actor = Actor(type="user", id="user-123", tenant_id="tenant-1", permissions=["calendar:write"])
        base_checker.has_permission.return_value = True
        
        result = await calendar_checker.can_delete_event(owner_actor, public_event)
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_cannot_delete_others_event_without_admin(
        self,
        calendar_checker,
        base_checker,
        public_event
    ):
        """Non-owners cannot delete events without admin permission."""
        other_actor = Actor(type="user", id="other-user", tenant_id="tenant-1", permissions=["calendar:write"])
        base_checker.has_permission.side_effect = lambda actor, perm: perm == "calendar:write"
        
        result = await calendar_checker.can_delete_event(other_actor, public_event)
        
        assert result is False


class TestRequirePermissions:
    """Test permission requirement enforcement."""
    
    @pytest.mark.asyncio
    async def test_require_read_permission_passes(
        self,
        calendar_checker,
        base_checker,
        user_actor
    ):
        """require_read_permission should pass for users with permission."""
        base_checker.has_permission.return_value = True  # Has permission
        
        # Should not raise an exception
        await calendar_checker.require_read_permission(user_actor)
        
        base_checker.has_permission.assert_called_once_with(user_actor, "calendar:read")
    
    @pytest.mark.asyncio
    async def test_require_read_permission_raises(
        self,
        calendar_checker,
        base_checker,
        readonly_actor
    ):
        """require_read_permission should raise for users without permission."""
        base_checker.has_permission.return_value = False  # No permission
        
        with pytest.raises(PermissionDeniedError):
            await calendar_checker.require_read_permission(readonly_actor)
    
    @pytest.mark.asyncio
    async def test_require_event_read_permission_for_public_event(
        self,
        calendar_checker,
        base_checker,
        user_actor,
        public_event
    ):
        """Public events should only require calendar.read permission."""
        base_checker.has_permission.return_value = True
        
        await calendar_checker.require_event_read_permission(user_actor, public_event)
        
        # Can read event involves checking admin first, then read permission
        assert base_checker.has_permission.call_count >= 1
    
    @pytest.mark.asyncio
    async def test_require_event_read_permission_for_private_event_owner(
        self,
        calendar_checker,
        base_checker,
        private_event
    ):
        """Private event owners should be allowed to read their events."""
        owner_actor = Actor(type="user", id="user-123", tenant_id="tenant-1", permissions=["calendar:read"])
        # Not admin, but is owner with read permission
        base_checker.has_permission.side_effect = lambda actor, perm: perm == "calendar:read"
        
        # Should not raise an exception
        await calendar_checker.require_event_read_permission(owner_actor, private_event)
    
    @pytest.mark.asyncio
    async def test_require_event_read_permission_for_private_event_non_owner(
        self,
        calendar_checker,
        base_checker,
        private_event
    ):
        """Non-owners should be denied access to private events."""
        other_actor = Actor(type="user", id="other-user", tenant_id="tenant-1", permissions=["calendar:read"])
        # Not admin, not owner, has read permission but still should be denied for private event
        base_checker.has_permission.side_effect = lambda actor, perm: perm == "calendar:read"
        
        with pytest.raises(PermissionDeniedError) as exc_info:
            await calendar_checker.require_event_read_permission(other_actor, private_event)
        
        assert "private event" in str(exc_info.value).lower()
    
    @pytest.mark.asyncio
    async def test_require_event_edit_permission_for_owner(
        self,
        calendar_checker,
        base_checker,
        public_event
    ):
        """Event owners should be allowed to edit their events."""
        owner_actor = Actor(type="user", id="user-123", tenant_id="tenant-1", permissions=["calendar:write"])
        # Has write permission but not admin
        base_checker.has_permission.side_effect = lambda actor, perm: perm == "calendar:write"
        
        await calendar_checker.require_event_edit_permission(owner_actor, public_event)
    
    @pytest.mark.asyncio  
    async def test_require_event_edit_permission_for_non_owner_fails(
        self,
        calendar_checker,
        base_checker,
        public_event
    ):
        """Non-owners should be denied edit access without admin permission."""
        other_actor = Actor(type="user", id="other-user", tenant_id="tenant-1", permissions=["calendar:write"])
        base_checker.require.return_value = None
        base_checker.has_permission.return_value = False  # No admin permission
        
        with pytest.raises(PermissionDeniedError) as exc_info:
            await calendar_checker.require_event_edit_permission(other_actor, public_event)
        
        assert "edit this event" in str(exc_info.value)


class TestPermissionHelpers:
    """Test permission helper methods."""
    
    @pytest.mark.asyncio
    async def test_get_event_visibility_filter(
        self,
        calendar_checker,
        base_checker,
        user_actor
    ):
        """Event visibility filters should vary by user permissions."""
        base_checker.has_permission.side_effect = lambda actor, perm: perm == "calendar:read"
        
        visibility_filter = await calendar_checker.get_event_visibility_filter(user_actor)
        
        # Regular users should see public events and their own private events
        expected_filter = {
            "OR": [
                {"visibility": "PUBLIC"},
                {"AND": [{"visibility": "PRIVATE"}, {"created_by": user_actor.user_id}]}
            ]
        }
        
        assert visibility_filter == expected_filter
    
    @pytest.mark.asyncio
    async def test_get_event_visibility_filter_admin(
        self,
        calendar_checker,
        base_checker,
        admin_actor
    ):
        """Admins should see all events."""
        base_checker.has_permission.side_effect = lambda actor, perm: perm == "calendar:admin"
        
        visibility_filter = await calendar_checker.get_event_visibility_filter(admin_actor)
        
        # Admins should see all events (no filter)
        assert visibility_filter is None
    
    @pytest.mark.asyncio
    async def test_filter_events_by_permissions(
        self,
        calendar_checker,
        base_checker,
        public_event,
        private_event
    ):
        """Events should be filtered based on user permissions."""
        user_actor = Actor(type="user", id="other-user", tenant_id="tenant-1", permissions=["calendar:read"])
        # Has read permission but not admin - should see public but not others' private events
        base_checker.has_permission.side_effect = lambda actor, perm: perm == "calendar:read"
        
        events = [public_event, private_event]
        
        filtered = await calendar_checker.filter_events_by_permissions(user_actor, events)
        
        # Should only return public event (private event belongs to different user)
        assert len(filtered) == 1
        assert filtered[0].event_id == "evt-public"