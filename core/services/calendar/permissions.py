"""
Calendar-specific permission checking.

Defines calendar domain permissions and implements permission
checking logic specific to calendar operations.

Author: ATLAS Team
Date: Jan 7, 2026
"""

from typing import List, Set

from core.services.common import (
    Actor,
    PermissionChecker,
    PermissionDeniedError,
    PermissionProvider,
)

from .types import CalendarEvent, EventVisibility


class CalendarPermissionChecker(PermissionChecker):
    """
    Permission checker for calendar operations.
    
    Extends the base PermissionChecker with calendar-specific
    permission definitions and hierarchical permission logic.
    """
    
    # Calendar permission definitions
    PERMISSIONS = {
        # Basic permissions
        "calendar:read": "View calendar events and categories",
        "calendar:write": "Create and update calendar events",
        "calendar:delete": "Delete calendar events", 
        "calendar:admin": "Full administrative access to calendar",
        
        # Category permissions
        "calendar:categories:read": "View calendar categories",
        "calendar:categories:write": "Create and update categories",
        "calendar:categories:delete": "Delete categories",
        "calendar:categories:admin": "Full category administrative access",
        
        # Reminder permissions
        "calendar:reminders:read": "View event reminders",
        "calendar:reminders:write": "Create and update reminders",
        "calendar:reminders:delete": "Delete reminders",
        
        # Sync permissions
        "calendar:sync:read": "View sync status and settings",
        "calendar:sync:write": "Configure calendar synchronization",
        "calendar:sync:execute": "Execute calendar sync operations",
        
        # Advanced permissions
        "calendar:conflicts:read": "View scheduling conflicts",
        "calendar:conflicts:resolve": "Resolve scheduling conflicts",
        "calendar:analytics:read": "View calendar analytics and reports",
        
        # Cross-system integration permissions
        "calendar:jobs:link": "Link calendar events to jobs",
        "calendar:tasks:link": "Link calendar events to tasks",
        "calendar:agent:schedule": "Allow agents to schedule events",
        "calendar:agent:modify": "Allow agents to modify events",
        
        # Bulk operations
        "calendar:bulk:import": "Import calendar events in bulk",
        "calendar:bulk:export": "Export calendar events in bulk",
        "calendar:bulk:delete": "Delete multiple events at once",
    }
    
    def __init__(self, provider: PermissionProvider | None = None) -> None:
        super().__init__(provider)
        
    async def _is_permission_implied(
        self, 
        held_permission: str, 
        required_permission: str
    ) -> bool:
        """
        Check if a held permission implies the required permission.
        
        Calendar permission hierarchy:
        - calendar:admin → all calendar permissions
        - calendar:write → calendar:read
        - calendar:categories:admin → all category permissions
        - calendar:categories:write → calendar:categories:read
        """
        # System permissions
        if held_permission == "*":
            return True
            
        # Split permissions into parts
        held_parts = held_permission.split(":")
        required_parts = required_permission.split(":")
        
        # Must be calendar domain
        if (len(held_parts) < 1 or len(required_parts) < 1 or
            held_parts[0] != "calendar" or required_parts[0] != "calendar"):
            return False
            
        # Global calendar admin
        if held_permission == "calendar:admin":
            return required_permission.startswith("calendar:")
            
        # Same subdomain check
        if len(held_parts) >= 2 and len(required_parts) >= 2:
            # Category admin permissions
            if (held_parts[1] == "categories" and required_parts[1] == "categories" and
                len(held_parts) >= 3 and held_parts[2] == "admin"):
                return required_permission.startswith("calendar:categories:")
            
            # General write → read implications
            if (len(held_parts) >= 3 and len(required_parts) >= 3 and
                held_parts[:-1] == required_parts[:-1]):  # Same path except action
                
                held_action = held_parts[-1]
                required_action = required_parts[-1]
                
                # write implies read
                if held_action == "write" and required_action == "read":
                    return True
                    
                # admin implies all actions in same domain
                if held_action == "admin":
                    return True
        
        return False
    
    async def require_calendar_access(
        self, 
        actor, 
        operation: str = "read"
    ) -> None:
        """
        Convenience method to check basic calendar access.
        
        Args:
            actor: Actor attempting the operation
            operation: Type of operation (read, write, delete, admin)
        """
        await self.require(actor, f"calendar:{operation}")
    
    async def require_category_access(
        self, 
        actor, 
        operation: str = "read"
    ) -> None:
        """
        Convenience method to check calendar category access.
        
        Args:
            actor: Actor attempting the operation
            operation: Type of operation (read, write, delete, admin)
        """
        await self.require(actor, f"calendar:categories:{operation}")
    
    async def require_reminder_access(
        self, 
        actor, 
        operation: str = "read"
    ) -> None:
        """
        Convenience method to check reminder access.
        
        Args:
            actor: Actor attempting the operation
            operation: Type of operation (read, write, delete)
        """
        await self.require(actor, f"calendar:reminders:{operation}")
    
    async def require_sync_access(
        self, 
        actor, 
        operation: str = "read"
    ) -> None:
        """
        Convenience method to check sync access.
        
        Args:
            actor: Actor attempting the operation
            operation: Type of operation (read, write, execute)
        """
        await self.require(actor, f"calendar:sync:{operation}")
    
    async def can_agent_schedule(self, actor) -> bool:
        """
        Check if an agent can schedule calendar events.
        
        Args:
            actor: Actor to check (should be agent type)
            
        Returns:
            True if agent can schedule events
        """
        return await self.has_permission(actor, "calendar:agent:schedule")
    
    async def can_agent_modify(self, actor) -> bool:
        """
        Check if an agent can modify existing calendar events.
        
        Args:
            actor: Actor to check (should be agent type)
            
        Returns:
            True if agent can modify events
        """
        return await self.has_permission(actor, "calendar:agent:modify")
    
    async def require_cross_system_access(
        self, 
        actor, 
        system: str
    ) -> None:
        """
        Require permission to link calendar with other systems.
        
        Args:
            actor: Actor attempting the operation
            system: System to link with (jobs, tasks)
        """
        await self.require(actor, f"calendar:{system}:link")
    
    async def can_read_calendar(self, actor: Actor) -> bool:
        """Return True when the actor can read calendar data."""
        return await self.has_permission(actor, "calendar:read")

    async def can_write_calendar(self, actor: Actor) -> bool:
        """Return True when the actor can create or update events."""
        return await self.has_permission(actor, "calendar:write")

    async def can_admin_calendar(self, actor: Actor) -> bool:
        """Return True when the actor has calendar administrative access."""
        return await self.has_permission(actor, "calendar:admin")

    async def require_read_permission(self, actor: Actor) -> None:
        """Ensure the actor can read calendar data."""
        await self.require(actor, "calendar:read")

    async def require_write_permission(self, actor: Actor) -> None:
        """Ensure the actor can write calendar data."""
        await self.require(actor, "calendar:write")

    async def require_admin_permission(self, actor: Actor) -> None:
        """Ensure the actor has administrative calendar rights."""
        await self.require(actor, "calendar:admin")

    async def can_read_event(self, actor: Actor, event: CalendarEvent) -> bool:
        """Check whether the actor can read a specific event."""
        if await self.can_admin_calendar(actor):
            return True

        if event.visibility == EventVisibility.PUBLIC:
            return await self.can_read_calendar(actor)

        # Private events require ownership or admin rights
        if event.created_by == actor.user_id:
            return await self.can_read_calendar(actor)

        return False

    async def can_edit_event(self, actor: Actor, event: CalendarEvent) -> bool:
        """Check whether the actor can edit a specific event."""
        if await self.can_admin_calendar(actor):
            return True

        return event.created_by == actor.user_id and await self.can_write_calendar(actor)

    async def can_delete_event(self, actor: Actor, event: CalendarEvent) -> bool:
        """Check whether the actor can delete a specific event."""
        if await self.can_admin_calendar(actor):
            return True

        return event.created_by == actor.user_id and await self.can_write_calendar(actor)

    async def require_event_read_permission(
        self,
        actor: Actor,
        event: CalendarEvent,
    ) -> None:
        """Ensure the actor may read the supplied event."""
        if not await self.can_read_event(actor, event):
            raise PermissionDeniedError(
                "Access denied to private event",
                "EVENT_READ_DENIED",
                {"event_id": event.event_id, "actor_id": actor.user_id},
            )

    async def require_event_edit_permission(
        self,
        actor: Actor,
        event: CalendarEvent,
    ) -> None:
        """Ensure the actor may edit the supplied event."""
        if not await self.can_edit_event(actor, event):
            raise PermissionDeniedError(
                "You do not have permission to edit this event",
                "EVENT_EDIT_DENIED",
                {"event_id": event.event_id, "actor_id": actor.user_id},
            )

    async def require_event_delete_permission(
        self,
        actor: Actor,
        event: CalendarEvent,
    ) -> None:
        """Ensure the actor may delete the supplied event."""
        if not await self.can_delete_event(actor, event):
            raise PermissionDeniedError(
                "You do not have permission to delete this event",
                "EVENT_DELETE_DENIED",
                {"event_id": event.event_id, "actor_id": actor.user_id},
            )

    async def get_event_visibility_filter(self, actor: Actor) -> dict | None:
        """Return a filter dict representing the actor's visibility scope."""
        if await self.can_admin_calendar(actor):
            return None

        return {
            "OR": [
                {"visibility": EventVisibility.PUBLIC.name},
                {
                    "AND": [
                        {"visibility": EventVisibility.PRIVATE.name},
                        {"created_by": actor.user_id},
                    ]
                },
            ]
        }

    async def filter_events_by_permissions(
        self,
        actor: Actor,
        events: List[CalendarEvent],
    ) -> List[CalendarEvent]:
        """Filter a list of events down to those visible to the actor."""
        result: List[CalendarEvent] = []
        for event in events:
            if await self.can_read_event(actor, event):
                result.append(event)
        return result
    
    def get_all_calendar_permissions(self) -> Set[str]:
        """Get all defined calendar permissions."""
        return set(self.PERMISSIONS.keys())
    
    def get_permission_description(self, permission: str) -> str | None:
        """Get human-readable description of a permission."""
        return self.PERMISSIONS.get(permission)