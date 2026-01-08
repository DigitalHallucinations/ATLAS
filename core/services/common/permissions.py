"""
Permission checking system for ATLAS services.

Provides standardized permission validation that all services
should use to enforce access control. Designed to be async-first
to support future database-backed permission lookups.

Author: ATLAS Team  
Date: Jan 7, 2026
"""

import logging
from typing import List, Set, Protocol

from .exceptions import PermissionDeniedError
from .types import Actor


logger = logging.getLogger(__name__)


class PermissionProvider(Protocol):
    """
    Protocol for permission data sources.
    
    Allows pluggable permission storage (in-memory, database, etc.)
    without changing the PermissionChecker interface.
    """
    
    async def get_permissions(self, actor: Actor) -> Set[str]:
        """Get all permissions for an actor."""
        ...
    
    async def has_permission(self, actor: Actor, permission: str) -> bool:
        """Check if actor has a specific permission."""
        ...


class InMemoryPermissionProvider:
    """
    Simple in-memory permission provider.
    
    Suitable for development and simple deployments.
    For production, consider a database-backed provider.
    """
    
    def __init__(self) -> None:
        # In real implementation, this would be loaded from config
        self._role_permissions: dict[str, set[str]] = {
            "admin": {"*"},  # Admin has all permissions
            "user": {
                "conversations:read",
                "conversations:write", 
                "personas:read",
                "tools:read",
            },
            "readonly": {
                "conversations:read",
                "personas:read", 
                "tools:read",
            },
            "system": {"*"},  # System actors have all permissions
        }
    
    async def get_permissions(self, actor: Actor) -> Set[str]:
        """Get permissions from actor's permission set."""
        return actor.permissions
    
    async def has_permission(self, actor: Actor, permission: str) -> bool:
        """Check if actor has permission (including wildcard check)."""
        # System actors and wildcard permissions
        if "*" in actor.permissions or actor.is_system():
            return True
            
        # Direct permission check
        if permission in actor.permissions:
            return True
            
        # Check for hierarchical permissions (e.g., conversations:admin includes conversations:write)
        for actor_perm in actor.permissions:
            if self._is_permission_implied(actor_perm, permission):
                return True
                
        return False
    
    def _is_permission_implied(self, held_permission: str, required_permission: str) -> bool:
        """
        Check if a held permission implies the required permission.
        
        Examples:
        - "conversations:admin" implies "conversations:write"
        - "conversations:write" implies "conversations:read"  
        """
        if held_permission == "*":
            return True
            
        # Split permissions into parts
        held_parts = held_permission.split(":")
        required_parts = required_permission.split(":")
        
        # Must be same domain
        if len(held_parts) < 1 or len(required_parts) < 1:
            return False
        
        if held_parts[0] != required_parts[0]:
            return False
            
        # Check permission hierarchy
        if len(held_parts) >= 2 and len(required_parts) >= 2:
            held_action = held_parts[1]
            required_action = required_parts[1]
            
            # Admin implies all other actions
            if held_action == "admin":
                return True
                
            # Write implies read
            if held_action == "write" and required_action == "read":
                return True
                
        return False


class PermissionChecker:
    """
    Async permission checker for service operations.
    
    Validates that actors have required permissions before
    allowing operations to proceed. Designed to be injectable
    into services for testing and different permission providers.
    
    Example:
        checker = PermissionChecker(InMemoryPermissionProvider())
        
        # This will raise PermissionDeniedError if user lacks permission
        await checker.require(user_actor, "conversations:write")
        
        # This returns a boolean
        can_read = await checker.has_permission(user_actor, "conversations:read")
    """
    
    def __init__(self, provider: PermissionProvider | None = None) -> None:
        self._provider = provider or InMemoryPermissionProvider()
    
    async def require(self, actor: Actor, permission: str) -> None:
        """
        Require that actor has permission, raise exception if not.
        
        Args:
            actor: The actor attempting the operation
            permission: Required permission (e.g., "conversations:write")
            
        Raises:
            PermissionDeniedError: If actor lacks the required permission
        """
        if not await self.has_permission(actor, permission):
            logger.warning(
                f"Permission denied: {actor.type}:{actor.id} lacks {permission}",
                extra={
                    "actor_type": actor.type,
                    "actor_id": actor.id,
                    "tenant_id": actor.tenant_id,
                    "required_permission": permission,
                }
            )
            raise PermissionDeniedError(
                f"Insufficient permissions: {permission} required",
                "INSUFFICIENT_PERMISSIONS",
                {
                    "required_permission": permission,
                    "actor_type": actor.type,
                    "tenant_id": actor.tenant_id,
                }
            )
    
    async def has_permission(self, actor: Actor, permission: str) -> bool:
        """
        Check if actor has the specified permission.
        
        Args:
            actor: The actor to check
            permission: Permission to check for
            
        Returns:
            True if actor has permission, False otherwise
        """
        try:
            return await self._provider.has_permission(actor, permission)
        except Exception as e:
            logger.error(
                f"Error checking permissions for {actor.id}: {e}",
                extra={
                    "actor_type": actor.type,
                    "actor_id": actor.id,
                    "permission": permission,
                    "error": str(e),
                }
            )
            # Fail closed - deny permission if there's an error checking
            return False
    
    async def require_any(self, actor: Actor, permissions: List[str]) -> None:
        """
        Require that actor has at least one of the specified permissions.
        
        Args:
            actor: The actor attempting the operation
            permissions: List of permissions (actor needs any one)
            
        Raises:
            PermissionDeniedError: If actor lacks all permissions
        """
        for permission in permissions:
            if await self.has_permission(actor, permission):
                return
        
        logger.warning(
            f"Permission denied: {actor.type}:{actor.id} lacks any of {permissions}",
            extra={
                "actor_type": actor.type,
                "actor_id": actor.id, 
                "tenant_id": actor.tenant_id,
                "required_permissions": permissions,
            }
        )
        raise PermissionDeniedError(
            f"Insufficient permissions: one of {permissions} required",
            "INSUFFICIENT_PERMISSIONS",
            {
                "required_permissions": permissions,
                "actor_type": actor.type,
                "tenant_id": actor.tenant_id,
            }
        )
    
    async def require_all(self, actor: Actor, permissions: List[str]) -> None:
        """
        Require that actor has all of the specified permissions.
        
        Args:
            actor: The actor attempting the operation
            permissions: List of permissions (actor needs all)
            
        Raises:
            PermissionDeniedError: If actor lacks any permission
        """
        missing_permissions = []
        
        for permission in permissions:
            if not await self.has_permission(actor, permission):
                missing_permissions.append(permission)
        
        if missing_permissions:
            logger.warning(
                f"Permission denied: {actor.type}:{actor.id} lacks {missing_permissions}",
                extra={
                    "actor_type": actor.type,
                    "actor_id": actor.id,
                    "tenant_id": actor.tenant_id,
                    "missing_permissions": missing_permissions,
                }
            )
            raise PermissionDeniedError(
                f"Insufficient permissions: {missing_permissions} required",
                "INSUFFICIENT_PERMISSIONS",
                {
                    "required_permissions": permissions,
                    "missing_permissions": missing_permissions,
                    "actor_type": actor.type,
                    "tenant_id": actor.tenant_id,
                }
            )
    
    async def get_permissions(self, actor: Actor) -> Set[str]:
        """
        Get all permissions for an actor.
        
        Args:
            actor: The actor to get permissions for
            
        Returns:
            Set of permission strings
        """
        try:
            return await self._provider.get_permissions(actor)
        except Exception as e:
            logger.error(
                f"Error getting permissions for {actor.id}: {e}",
                extra={
                    "actor_type": actor.type,
                    "actor_id": actor.id,
                    "error": str(e),
                }
            )
            # Return empty set if error occurs
            return set()