"""
Task permission checking.

Enforces access control for task operations with strict
tenant isolation by default.

Author: ATLAS Team
Date: Jan 10, 2026
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Optional

from core.services.common import Actor, PermissionDeniedError


logger = logging.getLogger(__name__)


class TaskPermissionChecker:
    """
    Permission checker for task operations.
    
    Enforces strict tenant isolation by default:
    - Actors can only access tasks where task.tenant_id == actor.tenant_id
    - System actors bypass tenant restrictions
    - Admin permissions grant full access
    
    Permission hierarchy:
    - task:admin - Full access to all task operations
    - task:write - Create, update, delete tasks within tenant
    - task:read - View tasks within tenant
    
    Example:
        checker = TaskPermissionChecker()
        
        # Check if user can read tasks
        checker.require_read(actor)
        
        # Check if user can modify a specific task
        checker.require_task_write(actor, task_dict)
    """
    
    # Permission constants
    PERMISSION_ADMIN = "task:admin"
    PERMISSION_WRITE = "task:write"
    PERMISSION_READ = "task:read"
    
    def __init__(self) -> None:
        pass
    
    def _is_system_actor(self, actor: Actor) -> bool:
        """Check if actor is a system-level actor."""
        return actor.type in ("system", "job", "sync", "task")
    
    def _has_permission(self, actor: Actor, permission: str) -> bool:
        """Check if actor has a specific permission."""
        permissions = set(actor.permissions) if isinstance(actor.permissions, list) else actor.permissions
        
        # Wildcard grants all
        if "*" in permissions:
            return True
        
        # Direct check
        if permission in permissions:
            return True
        
        # Admin implies all task permissions
        if self.PERMISSION_ADMIN in permissions:
            return True
        
        # Write implies read
        if permission == self.PERMISSION_READ and self.PERMISSION_WRITE in permissions:
            return True
        
        return False
    
    def _can_access_tenant(self, actor: Actor, tenant_id: Optional[str]) -> bool:
        """Check if actor can access resources in the given tenant."""
        # System actors bypass tenant restrictions
        if self._is_system_actor(actor):
            return True
        
        # Global scope requires system actor
        if tenant_id is None:
            return False
        
        # Strict tenant isolation for all non-system actors
        return actor.tenant_id == tenant_id
    
    def _can_access_task(self, actor: Actor, task: Mapping[str, Any]) -> bool:
        """Check if actor can access a specific task."""
        task_tenant_id = task.get("tenant_id")
        
        # Must be able to access the tenant
        if not self._can_access_tenant(actor, task_tenant_id):
            return False
        
        # Owner can always access their own tasks
        task_owner_id = task.get("owner_id")
        if task_owner_id is not None and str(task_owner_id) == actor.id:
            return True
        
        # Must have at least read permission
        return self._has_permission(actor, self.PERMISSION_READ)
    
    # =========================================================================
    # Permission requirement methods
    # =========================================================================
    
    def require_read(self, actor: Actor, tenant_id: Optional[str] = None) -> None:
        """Require read permission for listing/viewing tasks."""
        if not self._has_permission(actor, self.PERMISSION_READ):
            raise PermissionDeniedError(
                f"Actor {actor.id} lacks permission: {self.PERMISSION_READ}",
                error_code="TASK_READ_DENIED",
                details={"actor_id": actor.id, "required": self.PERMISSION_READ},
            )
        
        if tenant_id and not self._can_access_tenant(actor, tenant_id):
            raise PermissionDeniedError(
                f"Actor {actor.id} cannot access tenant: {tenant_id}",
                error_code="TASK_TENANT_DENIED",
                details={"actor_id": actor.id, "tenant_id": tenant_id},
            )
    
    def require_write(self, actor: Actor, tenant_id: Optional[str] = None) -> None:
        """Require write permission for creating/updating tasks."""
        if not self._has_permission(actor, self.PERMISSION_WRITE):
            raise PermissionDeniedError(
                f"Actor {actor.id} lacks permission: {self.PERMISSION_WRITE}",
                error_code="TASK_WRITE_DENIED",
                details={"actor_id": actor.id, "required": self.PERMISSION_WRITE},
            )
        
        if tenant_id and not self._can_access_tenant(actor, tenant_id):
            raise PermissionDeniedError(
                f"Actor {actor.id} cannot access tenant: {tenant_id}",
                error_code="TASK_TENANT_DENIED",
                details={"actor_id": actor.id, "tenant_id": tenant_id},
            )
    
    def require_admin(self, actor: Actor) -> None:
        """Require admin permission."""
        if not self._has_permission(actor, self.PERMISSION_ADMIN):
            raise PermissionDeniedError(
                f"Actor {actor.id} lacks permission: {self.PERMISSION_ADMIN}",
                error_code="TASK_ADMIN_DENIED",
                details={"actor_id": actor.id, "required": self.PERMISSION_ADMIN},
            )
    
    def require_task_read(self, actor: Actor, task: Mapping[str, Any]) -> None:
        """Require read access to a specific task."""
        if not self._can_access_task(actor, task):
            raise PermissionDeniedError(
                f"Actor {actor.id} cannot access task: {task.get('id')}",
                error_code="TASK_ACCESS_DENIED",
                details={"actor_id": actor.id, "task_id": str(task.get("id"))},
            )
    
    def require_task_write(self, actor: Actor, task: Mapping[str, Any]) -> None:
        """Require write access to a specific task."""
        if not self._has_permission(actor, self.PERMISSION_WRITE):
            raise PermissionDeniedError(
                f"Actor {actor.id} lacks permission: {self.PERMISSION_WRITE}",
                error_code="TASK_WRITE_DENIED",
                details={"actor_id": actor.id, "required": self.PERMISSION_WRITE},
            )
        
        if not self._can_access_task(actor, task):
            raise PermissionDeniedError(
                f"Actor {actor.id} cannot modify task: {task.get('id')}",
                error_code="TASK_ACCESS_DENIED",
                details={"actor_id": actor.id, "task_id": str(task.get("id"))},
            )
    
    def require_task_delete(self, actor: Actor, task: Mapping[str, Any]) -> None:
        """Require delete access to a specific task."""
        self.require_task_write(actor, task)
        
        # Only owners and admins can delete
        task_owner_id = task.get("owner_id")
        is_owner = task_owner_id is not None and str(task_owner_id) == actor.id
        is_admin = self._has_permission(actor, self.PERMISSION_ADMIN)
        
        if not (is_owner or is_admin or self._is_system_actor(actor)):
            raise PermissionDeniedError(
                f"Actor {actor.id} cannot delete task: {task.get('id')} (not owner)",
                error_code="TASK_DELETE_DENIED",
                details={"actor_id": actor.id, "task_id": str(task.get("id"))},
            )
    
    # =========================================================================
    # Check methods (return bool instead of raising)
    # =========================================================================
    
    def can_read(self, actor: Actor, tenant_id: Optional[str] = None) -> bool:
        """Check if actor can read tasks."""
        try:
            self.require_read(actor, tenant_id)
            return True
        except PermissionDeniedError:
            return False
    
    def can_write(self, actor: Actor, tenant_id: Optional[str] = None) -> bool:
        """Check if actor can write tasks."""
        try:
            self.require_write(actor, tenant_id)
            return True
        except PermissionDeniedError:
            return False
    
    def can_access_task(self, actor: Actor, task: Mapping[str, Any]) -> bool:
        """Check if actor can access a specific task."""
        return self._can_access_task(actor, task)


__all__ = [
    "TaskPermissionChecker",
]
