"""
Budget permission checking.

Enforces access control for budget operations with strict
tenant isolation by default.

Author: ATLAS Team
Date: Jan 8, 2026
"""

from __future__ import annotations

import logging
from typing import Optional

from core.services.common import Actor, PermissionDeniedError

from modules.budget.models import BudgetPolicy, BudgetScope


logger = logging.getLogger(__name__)


class BudgetPermissionChecker:
    """
    Permission checker for budget operations.
    
    Enforces strict tenant isolation by default:
    - Actors can only access policies where policy.scope_id == actor.tenant_id
    - System actors bypass tenant restrictions
    - GLOBAL scope policies require admin permissions
    
    Permission hierarchy:
    - budget:admin - Full access to all budget operations
    - budget:write - Create, update, delete policies within tenant
    - budget:read - View policies and check budgets within tenant
    
    Example:
        checker = BudgetPermissionChecker()
        
        # Check if user can read budgets
        await checker.require_read(actor)
        
        # Check if user can modify a specific policy
        await checker.require_policy_write(actor, policy)
    """
    
    # Permission constants
    PERMISSION_ADMIN = "budget:admin"
    PERMISSION_WRITE = "budget:write"
    PERMISSION_READ = "budget:read"
    
    def __init__(self) -> None:
        pass
    
    def _is_system_actor(self, actor: Actor) -> bool:
        """Check if actor is a system-level actor."""
        return actor.type in ("system", "job", "sync")
    
    def _has_permission(self, actor: Actor, permission: str) -> bool:
        """Check if actor has a specific permission."""
        permissions = set(actor.permissions) if isinstance(actor.permissions, list) else actor.permissions
        
        # Wildcard grants all
        if "*" in permissions:
            return True
        
        # Direct check
        if permission in permissions:
            return True
        
        # Admin implies all budget permissions
        if self.PERMISSION_ADMIN in permissions:
            return True
        
        # Write implies read
        if permission == self.PERMISSION_READ and self.PERMISSION_WRITE in permissions:
            return True
        
        return False
    
    def _can_access_tenant(self, actor: Actor, scope_id: Optional[str]) -> bool:
        """Check if actor can access resources in the given scope.
        
        Strict tenant isolation:
        - System actors can access any tenant
        - Users can only access their own tenant
        - None scope_id (global) requires admin or system actor
        """
        # System actors bypass tenant restrictions
        if self._is_system_actor(actor):
            return True
        
        # Admin can access any tenant
        if self._has_permission(actor, self.PERMISSION_ADMIN):
            return True
        
        # Global scope requires admin
        if scope_id is None:
            return False
        
        # Strict tenant isolation
        return actor.tenant_id == scope_id
    
    async def require_read(self, actor: Actor) -> None:
        """Require read permission for budget operations.
        
        Args:
            actor: Actor attempting the operation
            
        Raises:
            PermissionDeniedError: If actor lacks read permission
        """
        if not self._has_permission(actor, self.PERMISSION_READ):
            logger.warning(
                "Permission denied: actor %s lacks %s",
                actor.id, self.PERMISSION_READ
            )
            raise PermissionDeniedError(
                f"Permission denied: {self.PERMISSION_READ} required",
                error_code="BUDGET_READ_DENIED",
                details={"actor_id": actor.id, "required": self.PERMISSION_READ},
            )
    
    async def require_write(self, actor: Actor) -> None:
        """Require write permission for budget modifications.
        
        Args:
            actor: Actor attempting the operation
            
        Raises:
            PermissionDeniedError: If actor lacks write permission
        """
        if not self._has_permission(actor, self.PERMISSION_WRITE):
            logger.warning(
                "Permission denied: actor %s lacks %s",
                actor.id, self.PERMISSION_WRITE
            )
            raise PermissionDeniedError(
                f"Permission denied: {self.PERMISSION_WRITE} required",
                error_code="BUDGET_WRITE_DENIED",
                details={"actor_id": actor.id, "required": self.PERMISSION_WRITE},
            )
    
    async def require_admin(self, actor: Actor) -> None:
        """Require admin permission for administrative operations.
        
        Args:
            actor: Actor attempting the operation
            
        Raises:
            PermissionDeniedError: If actor lacks admin permission
        """
        if not self._has_permission(actor, self.PERMISSION_ADMIN):
            logger.warning(
                "Permission denied: actor %s lacks %s",
                actor.id, self.PERMISSION_ADMIN
            )
            raise PermissionDeniedError(
                f"Permission denied: {self.PERMISSION_ADMIN} required",
                error_code="BUDGET_ADMIN_DENIED",
                details={"actor_id": actor.id, "required": self.PERMISSION_ADMIN},
            )
    
    async def require_policy_read(self, actor: Actor, policy: BudgetPolicy) -> None:
        """Require permission to read a specific policy.
        
        Args:
            actor: Actor attempting the operation
            policy: Policy to access
            
        Raises:
            PermissionDeniedError: If actor cannot access this policy
        """
        await self.require_read(actor)
        
        # Check tenant access based on policy scope
        if policy.scope == BudgetScope.GLOBAL:
            # Global policies are readable by all with read permission
            return
        
        if not self._can_access_tenant(actor, policy.scope_id):
            logger.warning(
                "Tenant access denied: actor %s (tenant %s) cannot access policy in scope %s",
                actor.id, actor.tenant_id, policy.scope_id
            )
            raise PermissionDeniedError(
                "Permission denied: cannot access policy outside your tenant",
                error_code="BUDGET_TENANT_DENIED",
                details={
                    "actor_id": actor.id,
                    "actor_tenant": actor.tenant_id,
                    "policy_scope_id": policy.scope_id,
                },
            )
    
    async def require_policy_write(self, actor: Actor, policy: BudgetPolicy) -> None:
        """Require permission to modify a specific policy.
        
        Args:
            actor: Actor attempting the operation
            policy: Policy to modify
            
        Raises:
            PermissionDeniedError: If actor cannot modify this policy
        """
        await self.require_write(actor)
        
        # Global policies require admin
        if policy.scope == BudgetScope.GLOBAL:
            await self.require_admin(actor)
            return
        
        if not self._can_access_tenant(actor, policy.scope_id):
            logger.warning(
                "Tenant write denied: actor %s (tenant %s) cannot modify policy in scope %s",
                actor.id, actor.tenant_id, policy.scope_id
            )
            raise PermissionDeniedError(
                "Permission denied: cannot modify policy outside your tenant",
                error_code="BUDGET_TENANT_WRITE_DENIED",
                details={
                    "actor_id": actor.id,
                    "actor_tenant": actor.tenant_id,
                    "policy_scope_id": policy.scope_id,
                },
            )
    
    async def require_scope_access(
        self,
        actor: Actor,
        scope: BudgetScope,
        scope_id: Optional[str],
        write: bool = False,
    ) -> None:
        """Require permission to access a specific scope.
        
        Used when creating policies or listing by scope.
        
        Args:
            actor: Actor attempting the operation
            scope: Budget scope
            scope_id: Scope identifier
            write: Whether write access is required
            
        Raises:
            PermissionDeniedError: If actor cannot access this scope
        """
        if write:
            await self.require_write(actor)
        else:
            await self.require_read(actor)
        
        # Global scope requires admin for writes
        if scope == BudgetScope.GLOBAL and write:
            await self.require_admin(actor)
            return
        
        # Global scope is readable by all with read permission
        if scope == BudgetScope.GLOBAL:
            return
        
        if not self._can_access_tenant(actor, scope_id):
            action = "modify" if write else "access"
            raise PermissionDeniedError(
                f"Permission denied: cannot {action} scope outside your tenant",
                error_code="BUDGET_SCOPE_DENIED",
                details={
                    "actor_id": actor.id,
                    "actor_tenant": actor.tenant_id,
                    "scope": scope.value,
                    "scope_id": scope_id,
                },
            )
    
    def filter_policies_for_actor(
        self,
        actor: Actor,
        policies: list[BudgetPolicy],
    ) -> list[BudgetPolicy]:
        """Filter a list of policies to only those accessible by the actor.
        
        Args:
            actor: Actor requesting the policies
            policies: Full list of policies
            
        Returns:
            Filtered list of accessible policies
        """
        if self._is_system_actor(actor):
            return policies
        
        if self._has_permission(actor, self.PERMISSION_ADMIN):
            return policies
        
        accessible = []
        for policy in policies:
            # Global policies are always visible
            if policy.scope == BudgetScope.GLOBAL:
                accessible.append(policy)
                continue
            
            # Check tenant match
            if self._can_access_tenant(actor, policy.scope_id):
                accessible.append(policy)
        
        return accessible
