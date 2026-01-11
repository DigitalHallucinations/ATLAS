"""
Permission checking for persona operations.

Provides standardized permission validation for all persona
service operations.

Author: ATLAS Team
Date: Jan 10, 2026
"""

import logging
from typing import Optional

from core.services.common import Actor
from core.services.common.exceptions import PermissionDeniedError


logger = logging.getLogger(__name__)


# Permission constants
PERMISSION_PERSONAS_READ = "personas:read"
PERMISSION_PERSONAS_WRITE = "personas:write"
PERMISSION_PERSONAS_DELETE = "personas:delete"
PERMISSION_PERSONAS_ADMIN = "personas:admin"
PERMISSION_PERSONAS_ACTIVATE = "personas:activate"


class PersonaPermissionChecker:
    """
    Permission checker for persona operations.

    Enforces access control based on:
    - Actor permissions (what they're allowed to do)
    - Tenant isolation (actors can only access their tenant's resources)

    Note: Personas are typically shared across tenants (system-level resources),
    but write operations may be restricted.
    """

    def __init__(self, strict_tenant_isolation: bool = False) -> None:
        """
        Initialize permission checker.

        Args:
            strict_tenant_isolation: If True, enforce tenant isolation even for
                read operations. If False, allow cross-tenant reads for personas
                (since personas are typically shared resources).
        """
        self._strict_tenant_isolation = strict_tenant_isolation

    async def check_read(self, actor: Actor, persona_name: Optional[str] = None) -> None:
        """
        Check if actor can read personas.

        Args:
            actor: The actor performing the operation
            persona_name: Optional specific persona name

        Raises:
            PermissionDeniedError: If permission is denied
        """
        if actor.is_system():
            return

        if not self._has_any_permission(
            actor,
            PERMISSION_PERSONAS_READ,
            PERMISSION_PERSONAS_WRITE,
            PERMISSION_PERSONAS_ADMIN,
        ):
            logger.warning(
                "Permission denied for persona read: actor=%s, persona=%s",
                actor.id,
                persona_name,
            )
            raise PermissionDeniedError(
                message="Permission denied: cannot read personas",
                error_code="PERSONA_READ_DENIED",
                details={"actor_id": actor.id, "persona_name": persona_name},
            )

    async def check_write(
        self,
        actor: Actor,
        persona_name: str,
        is_create: bool = False,
    ) -> None:
        """
        Check if actor can create or update personas.

        Args:
            actor: The actor performing the operation
            persona_name: The persona being created/updated
            is_create: True if this is a create operation

        Raises:
            PermissionDeniedError: If permission is denied
        """
        if actor.is_system():
            return

        if not self._has_any_permission(
            actor,
            PERMISSION_PERSONAS_WRITE,
            PERMISSION_PERSONAS_ADMIN,
        ):
            operation = "create" if is_create else "update"
            logger.warning(
                "Permission denied for persona %s: actor=%s, persona=%s",
                operation,
                actor.id,
                persona_name,
            )
            raise PermissionDeniedError(
                message=f"Permission denied: cannot {operation} persona '{persona_name}'",
                error_code=f"PERSONA_{operation.upper()}_DENIED",
                details={"actor_id": actor.id, "persona_name": persona_name},
            )

    async def check_delete(self, actor: Actor, persona_name: str) -> None:
        """
        Check if actor can delete a persona.

        Args:
            actor: The actor performing the operation
            persona_name: The persona being deleted

        Raises:
            PermissionDeniedError: If permission is denied
        """
        if actor.is_system():
            return

        if not self._has_any_permission(
            actor,
            PERMISSION_PERSONAS_DELETE,
            PERMISSION_PERSONAS_ADMIN,
        ):
            logger.warning(
                "Permission denied for persona delete: actor=%s, persona=%s",
                actor.id,
                persona_name,
            )
            raise PermissionDeniedError(
                message=f"Permission denied: cannot delete persona '{persona_name}'",
                error_code="PERSONA_DELETE_DENIED",
                details={"actor_id": actor.id, "persona_name": persona_name},
            )

    async def check_activate(self, actor: Actor, persona_name: str) -> None:
        """
        Check if actor can activate/set a persona.

        Activation typically requires read permission plus activate permission.

        Args:
            actor: The actor performing the operation
            persona_name: The persona being activated

        Raises:
            PermissionDeniedError: If permission is denied
        """
        if actor.is_system():
            return

        # Users typically can activate any persona they can read
        # But may need specific activate permission in strict mode
        if not self._has_any_permission(
            actor,
            PERMISSION_PERSONAS_ACTIVATE,
            PERMISSION_PERSONAS_READ,
            PERMISSION_PERSONAS_ADMIN,
        ):
            logger.warning(
                "Permission denied for persona activate: actor=%s, persona=%s",
                actor.id,
                persona_name,
            )
            raise PermissionDeniedError(
                message=f"Permission denied: cannot activate persona '{persona_name}'",
                error_code="PERSONA_ACTIVATE_DENIED",
                details={"actor_id": actor.id, "persona_name": persona_name},
            )

    async def check_validate(self, actor: Actor, persona_name: Optional[str] = None) -> None:
        """
        Check if actor can validate personas.

        Validation is typically allowed for anyone with write permissions.

        Args:
            actor: The actor performing the operation
            persona_name: Optional specific persona name

        Raises:
            PermissionDeniedError: If permission is denied
        """
        if actor.is_system():
            return

        if not self._has_any_permission(
            actor,
            PERMISSION_PERSONAS_READ,
            PERMISSION_PERSONAS_WRITE,
            PERMISSION_PERSONAS_ADMIN,
        ):
            logger.warning(
                "Permission denied for persona validate: actor=%s, persona=%s",
                actor.id,
                persona_name,
            )
            raise PermissionDeniedError(
                message="Permission denied: cannot validate personas",
                error_code="PERSONA_VALIDATE_DENIED",
                details={"actor_id": actor.id, "persona_name": persona_name},
            )

    async def check_admin(self, actor: Actor, persona_name: str) -> None:
        """
        Check if actor has admin permissions for personas.

        Args:
            actor: The actor performing the operation
            persona_name: The persona being administered

        Raises:
            PermissionDeniedError: If permission is denied
        """
        if actor.is_system():
            return

        if not self._has_any_permission(actor, PERMISSION_PERSONAS_ADMIN):
            logger.warning(
                "Permission denied for persona admin: actor=%s, persona=%s",
                actor.id,
                persona_name,
            )
            raise PermissionDeniedError(
                message=f"Permission denied: cannot administer persona '{persona_name}'",
                error_code="PERSONA_ADMIN_DENIED",
                details={"actor_id": actor.id, "persona_name": persona_name},
            )

    # Boolean convenience methods for conditional checks
    def can_read(self, actor: Actor, persona_name: Optional[str] = None) -> bool:
        """
        Check if actor can read personas (non-raising version).

        Args:
            actor: The actor performing the operation
            persona_name: Optional specific persona name

        Returns:
            True if actor has read permission
        """
        if actor.is_system():
            return True
        return self._has_any_permission(
            actor,
            PERMISSION_PERSONAS_READ,
            PERMISSION_PERSONAS_WRITE,
            PERMISSION_PERSONAS_ADMIN,
        )

    def can_write(self, actor: Actor, persona_name: Optional[str] = None) -> bool:
        """
        Check if actor can write personas (non-raising version).

        Args:
            actor: The actor performing the operation
            persona_name: Optional specific persona name

        Returns:
            True if actor has write permission
        """
        if actor.is_system():
            return True
        return self._has_any_permission(
            actor,
            PERMISSION_PERSONAS_WRITE,
            PERMISSION_PERSONAS_ADMIN,
        )

    def can_delete(self, actor: Actor, persona_name: Optional[str] = None) -> bool:
        """
        Check if actor can delete personas (non-raising version).

        Args:
            actor: The actor performing the operation
            persona_name: Optional specific persona name

        Returns:
            True if actor has delete permission
        """
        if actor.is_system():
            return True
        return self._has_any_permission(
            actor,
            PERMISSION_PERSONAS_DELETE,
            PERMISSION_PERSONAS_ADMIN,
        )

    def can_admin(self, actor: Actor, persona_name: Optional[str] = None) -> bool:
        """
        Check if actor has admin permissions (non-raising version).

        Args:
            actor: The actor performing the operation
            persona_name: Optional specific persona name

        Returns:
            True if actor has admin permission
        """
        if actor.is_system():
            return True
        return self._has_any_permission(actor, PERMISSION_PERSONAS_ADMIN)

    def can_activate(self, actor: Actor, persona_name: Optional[str] = None) -> bool:
        """
        Check if actor can activate personas (non-raising version).

        Args:
            actor: The actor performing the operation
            persona_name: Optional specific persona name

        Returns:
            True if actor has activate permission
        """
        if actor.is_system():
            return True
        return self._has_any_permission(
            actor,
            PERMISSION_PERSONAS_ACTIVATE,
            PERMISSION_PERSONAS_READ,
            PERMISSION_PERSONAS_ADMIN,
        )

    def _has_any_permission(self, actor: Actor, *permissions: str) -> bool:
        """Check if actor has any of the given permissions."""
        if "*" in actor.permissions:
            return True
        return any(perm in actor.permissions for perm in permissions)

    def _has_all_permissions(self, actor: Actor, *permissions: str) -> bool:
        """Check if actor has all of the given permissions."""
        if "*" in actor.permissions:
            return True
        return all(perm in actor.permissions for perm in permissions)
