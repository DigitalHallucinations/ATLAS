"""
Permission checking for provider operations.

Provides standardized permission validation for all provider
service operations.

Author: ATLAS Team
Date: Jan 11, 2026
"""

import logging
from typing import Optional

from core.services.common import Actor
from core.services.common.exceptions import PermissionDeniedError


logger = logging.getLogger(__name__)


# Permission constants
PERMISSION_PROVIDERS_READ = "providers:read"
PERMISSION_PROVIDERS_WRITE = "providers:write"
PERMISSION_PROVIDERS_ADMIN = "providers:admin"


class ProviderPermissionChecker:
    """
    Permission checker for provider operations.
    
    Enforces access control for provider management.
    Providers are generally system-wide resources.
    """

    def __init__(self, strict_isolation: bool = False) -> None:
        self._strict_isolation = strict_isolation

    def check_read_permission(self, actor: Actor, provider_tenant_id: Optional[str] = None) -> None:
        """
        Check if actor has permission to read provider details.
        
        Args:
            actor: The actor attempting the operation.
            provider_tenant_id: The tenant ID owning the provider config (if any).
        
        Raises:
            PermissionDeniedError: If permission is denied.
        """
        # System actors always skip checks
        if actor.is_system:
            return

        # Check explicit permission
        if not actor.has_permission(PERMISSION_PROVIDERS_READ) and not actor.has_permission(PERMISSION_PROVIDERS_ADMIN):
            raise PermissionDeniedError(
                f"Actor '{actor.id}' lacks '{PERMISSION_PROVIDERS_READ}' permission."
            )

    def check_write_permission(self, actor: Actor, provider_tenant_id: Optional[str] = None) -> None:
        """
        Check if actor has permission to modify provider configuration.
        
        Args:
            actor: The actor attempting the operation.
            provider_tenant_id: The tenant ID owning the provider config (if any).
        
        Raises:
            PermissionDeniedError: If permission is denied.
        """
        if actor.is_system:
            return

        if not actor.has_permission(PERMISSION_PROVIDERS_WRITE) and not actor.has_permission(PERMISSION_PROVIDERS_ADMIN):
            raise PermissionDeniedError(
                f"Actor '{actor.id}' lacks '{PERMISSION_PROVIDERS_WRITE}' permission."
            )

    def check_credential_access(self, actor: Actor) -> None:
        """
        Check if actor has critical permission to manage credentials.
        
        Args:
            actor: The actor attempting the operation.
        
        Raises:
            PermissionDeniedError: If permission is denied.
        """
        if actor.is_system:
            return

        if not actor.has_permission(PERMISSION_PROVIDERS_ADMIN):
            raise PermissionDeniedError(
                f"Actor '{actor.id}' lacks '{PERMISSION_PROVIDERS_ADMIN}' permission."
            )
