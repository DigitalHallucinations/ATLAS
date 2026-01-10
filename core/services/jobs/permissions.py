"""
Job permission checking.

Enforces access control for job operations with strict
tenant isolation by default.

Author: ATLAS Team
Date: Jan 10, 2026
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Optional

from core.services.common import Actor, PermissionDeniedError


logger = logging.getLogger(__name__)


class JobPermissionChecker:
    """
    Permission checker for job operations.
    
    Enforces strict tenant isolation by default:
    - Actors can only access jobs where job.tenant_id == actor.tenant_id
    - System actors bypass tenant restrictions
    - Admin permissions grant full access
    
    Permission hierarchy:
    - job:admin - Full access to all job operations
    - job:write - Create, update, delete jobs within tenant
    - job:read - View jobs within tenant
    - job:execute - Start, cancel, retry jobs
    
    Example:
        checker = JobPermissionChecker()
        
        # Check if user can read jobs
        checker.require_read(actor)
        
        # Check if user can modify a specific job
        checker.require_job_write(actor, job_dict)
    """
    
    # Permission constants
    PERMISSION_ADMIN = "job:admin"
    PERMISSION_WRITE = "job:write"
    PERMISSION_READ = "job:read"
    PERMISSION_EXECUTE = "job:execute"
    
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
        
        # Admin implies all job permissions
        if self.PERMISSION_ADMIN in permissions:
            return True
        
        # Write implies read
        if permission == self.PERMISSION_READ and self.PERMISSION_WRITE in permissions:
            return True
        
        # Execute implies read
        if permission == self.PERMISSION_READ and self.PERMISSION_EXECUTE in permissions:
            return True
        
        return False
    
    def _can_access_tenant(self, actor: Actor, tenant_id: Optional[str]) -> bool:
        """Check if actor can access resources in the given tenant."""
        # System actors bypass tenant restrictions
        if self._is_system_actor(actor):
            return True
        
        # Admin can access any tenant
        if self._has_permission(actor, self.PERMISSION_ADMIN):
            return True
        
        # Global scope requires admin
        if tenant_id is None:
            return False
        
        # Strict tenant isolation
        return actor.tenant_id == tenant_id
    
    def _can_access_job(self, actor: Actor, job: Mapping[str, Any]) -> bool:
        """Check if actor can access a specific job."""
        job_tenant_id = job.get("tenant_id")
        
        # Must be able to access the tenant
        if not self._can_access_tenant(actor, job_tenant_id):
            return False
        
        # Owner can always access their own jobs
        job_owner_id = job.get("owner_id")
        if job_owner_id is not None and str(job_owner_id) == actor.id:
            return True
        
        # Must have at least read permission
        return self._has_permission(actor, self.PERMISSION_READ)
    
    # =========================================================================
    # Permission requirement methods
    # =========================================================================
    
    def require_read(self, actor: Actor, tenant_id: Optional[str] = None) -> None:
        """Require read permission for listing/viewing jobs."""
        if not self._has_permission(actor, self.PERMISSION_READ):
            raise PermissionDeniedError(
                f"Actor {actor.id} lacks permission: {self.PERMISSION_READ}",
                error_code="JOB_READ_DENIED",
                details={"actor_id": actor.id, "required": self.PERMISSION_READ},
            )
        
        if tenant_id and not self._can_access_tenant(actor, tenant_id):
            raise PermissionDeniedError(
                f"Actor {actor.id} cannot access tenant: {tenant_id}",
                error_code="JOB_TENANT_DENIED",
                details={"actor_id": actor.id, "tenant_id": tenant_id},
            )
    
    def require_write(self, actor: Actor, tenant_id: Optional[str] = None) -> None:
        """Require write permission for creating/updating jobs."""
        if not self._has_permission(actor, self.PERMISSION_WRITE):
            raise PermissionDeniedError(
                f"Actor {actor.id} lacks permission: {self.PERMISSION_WRITE}",
                error_code="JOB_WRITE_DENIED",
                details={"actor_id": actor.id, "required": self.PERMISSION_WRITE},
            )
        
        if tenant_id and not self._can_access_tenant(actor, tenant_id):
            raise PermissionDeniedError(
                f"Actor {actor.id} cannot access tenant: {tenant_id}",
                error_code="JOB_TENANT_DENIED",
                details={"actor_id": actor.id, "tenant_id": tenant_id},
            )
    
    def require_execute(self, actor: Actor, tenant_id: Optional[str] = None) -> None:
        """Require execute permission for starting/cancelling jobs."""
        if not self._has_permission(actor, self.PERMISSION_EXECUTE):
            raise PermissionDeniedError(
                f"Actor {actor.id} lacks permission: {self.PERMISSION_EXECUTE}",
                error_code="JOB_EXECUTE_DENIED",
                details={"actor_id": actor.id, "required": self.PERMISSION_EXECUTE},
            )
        
        if tenant_id and not self._can_access_tenant(actor, tenant_id):
            raise PermissionDeniedError(
                f"Actor {actor.id} cannot access tenant: {tenant_id}",
                error_code="JOB_TENANT_DENIED",
                details={"actor_id": actor.id, "tenant_id": tenant_id},
            )
    
    def require_admin(self, actor: Actor) -> None:
        """Require admin permission."""
        if not self._has_permission(actor, self.PERMISSION_ADMIN):
            raise PermissionDeniedError(
                f"Actor {actor.id} lacks permission: {self.PERMISSION_ADMIN}",
                error_code="JOB_ADMIN_DENIED",
                details={"actor_id": actor.id, "required": self.PERMISSION_ADMIN},
            )
    
    def require_job_read(self, actor: Actor, job: Mapping[str, Any]) -> None:
        """Require read access to a specific job."""
        if not self._can_access_job(actor, job):
            raise PermissionDeniedError(
                f"Actor {actor.id} cannot access job: {job.get('id')}",
                error_code="JOB_ACCESS_DENIED",
                details={"actor_id": actor.id, "job_id": str(job.get("id"))},
            )
    
    def require_job_write(self, actor: Actor, job: Mapping[str, Any]) -> None:
        """Require write access to a specific job."""
        if not self._has_permission(actor, self.PERMISSION_WRITE):
            raise PermissionDeniedError(
                f"Actor {actor.id} lacks permission: {self.PERMISSION_WRITE}",
                error_code="JOB_WRITE_DENIED",
                details={"actor_id": actor.id, "required": self.PERMISSION_WRITE},
            )
        
        if not self._can_access_job(actor, job):
            raise PermissionDeniedError(
                f"Actor {actor.id} cannot modify job: {job.get('id')}",
                error_code="JOB_ACCESS_DENIED",
                details={"actor_id": actor.id, "job_id": str(job.get("id"))},
            )
    
    def require_job_execute(self, actor: Actor, job: Mapping[str, Any]) -> None:
        """Require execute access to a specific job."""
        if not self._has_permission(actor, self.PERMISSION_EXECUTE):
            raise PermissionDeniedError(
                f"Actor {actor.id} lacks permission: {self.PERMISSION_EXECUTE}",
                error_code="JOB_EXECUTE_DENIED",
                details={"actor_id": actor.id, "required": self.PERMISSION_EXECUTE},
            )
        
        if not self._can_access_job(actor, job):
            raise PermissionDeniedError(
                f"Actor {actor.id} cannot execute job: {job.get('id')}",
                error_code="JOB_ACCESS_DENIED",
                details={"actor_id": actor.id, "job_id": str(job.get("id"))},
            )
    
    def require_job_delete(self, actor: Actor, job: Mapping[str, Any]) -> None:
        """Require delete access to a specific job."""
        self.require_job_write(actor, job)
        
        # Only owners and admins can delete
        job_owner_id = job.get("owner_id")
        is_owner = job_owner_id is not None and str(job_owner_id) == actor.id
        is_admin = self._has_permission(actor, self.PERMISSION_ADMIN)
        
        if not (is_owner or is_admin or self._is_system_actor(actor)):
            raise PermissionDeniedError(
                f"Actor {actor.id} cannot delete job: {job.get('id')} (not owner)",
                error_code="JOB_DELETE_DENIED",
                details={"actor_id": actor.id, "job_id": str(job.get("id"))},
            )
    
    # =========================================================================
    # Check methods (return bool instead of raising)
    # =========================================================================
    
    def can_read(self, actor: Actor, tenant_id: Optional[str] = None) -> bool:
        """Check if actor can read jobs."""
        try:
            self.require_read(actor, tenant_id)
            return True
        except PermissionDeniedError:
            return False
    
    def can_write(self, actor: Actor, tenant_id: Optional[str] = None) -> bool:
        """Check if actor can write jobs."""
        try:
            self.require_write(actor, tenant_id)
            return True
        except PermissionDeniedError:
            return False
    
    def can_execute(self, actor: Actor, tenant_id: Optional[str] = None) -> bool:
        """Check if actor can execute jobs."""
        try:
            self.require_execute(actor, tenant_id)
            return True
        except PermissionDeniedError:
            return False
    
    def can_access_job(self, actor: Actor, job: Mapping[str, Any]) -> bool:
        """Check if actor can access a specific job."""
        return self._can_access_job(actor, job)


__all__ = [
    "JobPermissionChecker",
]
