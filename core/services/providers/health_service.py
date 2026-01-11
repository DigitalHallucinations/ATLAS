"""
Provider Health Service.

Monitors the health and availability of configured providers.
Performs periodic checks and emits health change events.

Author: ATLAS Team
Date: Jan 11, 2026
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from core.services.common import Actor
from core.services.providers.types import (
    ProviderHealth,
    ProviderStatus,
    ProviderHealthEvent,
    ProviderErrorEvent,
)
from core.services.providers.config_service import ProviderConfigService
from core.services.providers.permissions import ProviderPermissionChecker


logger = logging.getLogger(__name__)


class ProviderHealthService:
    """
    Service for monitoring provider health.
    """

    def __init__(
        self,
        config_service: ProviderConfigService,
        message_bus: Any,
        permission_checker: Optional[ProviderPermissionChecker] = None
    ) -> None:
        self._config_service = config_service
        self._bus = message_bus
        self._permissions = permission_checker or ProviderPermissionChecker()
        
        # In-memory monitoring state
        self._latest_health: Dict[str, ProviderHealth] = {}
        
        # Initialize statuses as UNKNOWN
        self._initialize_health_cache()

    def _initialize_health_cache(self) -> None:
        # This assumes we have a system actor or bypass permissions for internal init
        # For now, we'll lazily load on first check or rely on explicit calls
        pass

    def _now_utc(self) -> datetime:
        return datetime.now(timezone.utc)

    async def check_health(self, actor: Actor, provider_id: str) -> ProviderHealth:
        """
        Perform an immediate health check on a specific provider.
        """
        self._permissions.check_read_permission(actor)
        
        # 1. Get config
        provider = self._config_service.get_provider(actor, provider_id)
        if not provider:
            # Maybe raise NotFound? For now return unknown/error
            return ProviderHealth(
                provider_id=provider_id,
                status=ProviderStatus.UNKNOWN,
                last_check=self._now_utc(),
                error_message="Provider not found"
            )

        if not provider.enabled:
             # If disabled, status is DISABLED (not necessarily an error)
            health = ProviderHealth(
                provider_id=provider_id,
                status=ProviderStatus.DISABLED,
                last_check=self._now_utc()
            )
            self._update_health_state(provider_id, health)
            return health

        # 2. Perform actual check (Mocked for now)
        # In reality, this would likely delegate to the specific ProviderAdapter or 
        # ProviderManager to make a 'ping' or 'models' call.
        
        # Simulating a check
        try:
            # Mock check logic
            # await self.provider_manager.ping(provider_id)
            latency = 100.0 # ms
            status = ProviderStatus.ENABLED
            error = None
        except Exception as e:
            latency = None
            status = ProviderStatus.ERROR
            error = str(e)
            
            # Emit error event
            error_event = ProviderErrorEvent(
                provider_id=provider_id,
                error_code="HEALTH_CHECK_FAILED",
                error_message=str(e)
            )
            self._bus.publish(error_event)

        health = ProviderHealth(
            provider_id=provider_id,
            status=status,
            last_check=self._now_utc(),
            latency_ms=latency,
            error_message=error
        )
        
        self._update_health_state(provider_id, health)
        return health

    async def check_all_health(self, actor: Actor) -> List[ProviderHealth]:
        """
        Check health for all configured providers.
        """
        self._permissions.check_read_permission(actor)
        
        providers = self._config_service.list_providers(actor)
        results = []
        
        # Run checks in parallel
        tasks = [self.check_health(actor, p.provider_id) for p in providers]
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions from gather if any (though check_health catches most)
            valid_results = []
            for r in results:
                if isinstance(r, ProviderHealth):
                    valid_results.append(r)
                else:
                    logger.error(f"Health check task failed: {r}")
            return valid_results
            
        return []

    def get_status(self, actor: Actor, provider_id: str) -> Optional[ProviderHealth]:
        """
        Get the last known health status from cache without performing a check.
        """
        self._permissions.check_read_permission(actor)
        return self._latest_health.get(provider_id)

    def get_all_statuses(self, actor: Actor) -> List[ProviderHealth]:
        """
        Get all cached health statuses.
        """
        self._permissions.check_read_permission(actor)
        return list(self._latest_health.values())

    def _update_health_state(self, provider_id: str, new_health: ProviderHealth) -> None:
        """
        Update local cache and emit events if status changed.
        """
        old_health = self._latest_health.get(provider_id)
        
        self._latest_health[provider_id] = new_health
        
        # Check for status change
        old_status = old_health.status if old_health else ProviderStatus.UNKNOWN
        if old_status != new_health.status:
            # Emit change event
            event = ProviderHealthEvent(
                provider_id=provider_id,
                old_status=old_status,
                new_status=new_health.status,
                latency_ms=new_health.latency_ms
            )
            self._bus.publish(event)
