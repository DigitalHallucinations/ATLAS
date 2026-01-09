"""
Budget Service Factory.

Provides factory functions to create and access budget service instances.
This module serves as the primary entry point for GTKUI and other callers
to obtain budget services.

Author: ATLAS Team
Date: Jan 8, 2026
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional, TYPE_CHECKING

from .policy_service import BudgetPolicyService, BudgetRepository
from .tracking_service import BudgetTrackingService, UsageRepository, PolicyRepository
from .alert_service import BudgetAlertService, AlertRepository, AlertConfigRepository
from .permissions import BudgetPermissionChecker

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# =============================================================================
# Global Service Instances (lazy-initialized)
# =============================================================================

_policy_service: Optional[BudgetPolicyService] = None
_tracking_service: Optional[BudgetTrackingService] = None
_alert_service: Optional[BudgetAlertService] = None
_initialized: bool = False


def _get_budget_store_backend() -> Any:
    """Get the budget store backend directly from BudgetStore.
    
    This provides direct access to the persistence layer.
    """
    try:
        from modules.budget.persistence import BudgetStore
        store = BudgetStore.get_instance()
        if store and hasattr(store, '_backend'):
            return store._backend
    except Exception as exc:
        logger.warning("Failed to get budget store backend: %s", exc)
    return None


def _get_pricing_registry() -> Any:
    """Get the pricing registry."""
    try:
        from modules.budget.pricing import get_pricing_registry
        return get_pricing_registry()
    except Exception as exc:
        logger.debug("Pricing registry not available: %s", exc)
    return None


# =============================================================================
# Repository Adapters
# =============================================================================


class StoreBackendPolicyAdapter:
    """Adapts BudgetStoreBackend to BudgetRepository protocol."""
    
    def __init__(self, store: Any) -> None:
        self._store = store
    
    async def get_policy(self, policy_id: str) -> Optional[Any]:
        """Get policy by ID."""
        if self._store:
            return await self._store.get_policy(policy_id)
        return None
    
    async def save_policy(self, policy: Any) -> Any:
        """Save a policy."""
        if self._store:
            return await self._store.save_policy(policy)
        return policy
    
    async def delete_policy(self, policy_id: str) -> bool:
        """Delete a policy."""
        if self._store:
            return await self._store.delete_policy(policy_id)
        return False
    
    async def list_policies(
        self,
        scope: Optional[Any] = None,
        scope_id: Optional[str] = None,
        enabled_only: bool = True,
        limit: int = 100,
        offset: int = 0,
    ) -> list:
        """List policies with filtering."""
        if self._store:
            policies = await self._store.get_policies(
                scope=scope,
                scope_id=scope_id,
                enabled_only=enabled_only,
            )
            return list(policies)[offset:offset + limit]
        return []
    
    async def get_policies(
        self,
        scope: Optional[Any] = None,
        scope_id: Optional[str] = None,
        enabled_only: bool = True,
    ) -> list:
        """Get policies (alias for tracking service)."""
        return await self.list_policies(scope=scope, scope_id=scope_id, enabled_only=enabled_only)
    
    async def get_current_spend(
        self,
        scope: Any,
        scope_id: Optional[str],
        period: Any,
    ) -> Any:
        """Get current spending for a scope/period."""
        from decimal import Decimal
        if self._store:
            try:
                return await self._store.get_aggregate_spend(scope, scope_id, period)
            except Exception:
                pass
        return Decimal("0")


class StoreBackendUsageAdapter:
    """Adapts BudgetStoreBackend to UsageRepository protocol."""
    
    def __init__(self, store: Any) -> None:
        self._store = store
    
    async def save_usage_record(self, record: Any) -> str:
        """Save a usage record."""
        if self._store:
            await self._store.save_usage_records([record])
            return record.id
        return ""
    
    async def save_usage_records(self, records: list) -> int:
        """Batch save usage records."""
        if self._store:
            await self._store.save_usage_records(records)
            return len(records)
        return 0
    
    async def get_usage_records(
        self,
        start_date: Any,
        end_date: Any,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        limit: int = 1000,
    ) -> list:
        """Retrieve usage records."""
        if self._store:
            records = await self._store.get_usage_records(
                start_date=start_date,
                end_date=end_date,
                user_id=user_id,
                tenant_id=tenant_id,
            )
            return list(records)[:limit]
        return []
    
    async def get_aggregate_spend(
        self,
        start_date: Any,
        end_date: Any,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Any:
        """Get aggregate spend."""
        from decimal import Decimal
        if self._store:
            try:
                return await self._store.get_aggregate_spend(
                    start_date=start_date,
                    end_date=end_date,
                    user_id=user_id,
                    tenant_id=tenant_id,
                )
            except Exception:
                pass
        return Decimal("0")
    
    async def get_spend_by_dimension(
        self,
        dimension: str,
        start_date: Any,
        end_date: Any,
        tenant_id: Optional[str] = None,
    ) -> dict:
        """Get spend by dimension."""
        # Fallback implementation
        return {}


class StoreBackendAlertAdapter:
    """Adapts BudgetStoreBackend to AlertRepository protocol."""
    
    def __init__(self, store: Any) -> None:
        self._store = store
    
    async def save_alert(self, alert: Any) -> str:
        """Save an alert."""
        if self._store:
            await self._store.save_alert(alert)
            return alert.id
        return ""
    
    async def get_alert(self, alert_id: str) -> Optional[Any]:
        """Get alert by ID."""
        if self._store:
            alerts = await self._store.get_alerts()
            for alert in alerts:
                if alert.id == alert_id:
                    return alert
        return None
    
    async def get_alerts(
        self,
        policy_id: Optional[str] = None,
        severity: Optional[str] = None,
        active_only: bool = True,
        limit: int = 100,
        offset: int = 0,
    ) -> list:
        """Get alerts with filtering."""
        if self._store:
            alerts = await self._store.get_alerts()
            if policy_id:
                alerts = [a for a in alerts if a.policy_id == policy_id]
            if active_only:
                alerts = [a for a in alerts if not getattr(a, 'acknowledged', False)]
            return list(alerts)[offset:offset + limit]
        return []
    
    async def update_alert(self, alert: Any) -> bool:
        """Update an alert."""
        if self._store:
            await self._store.save_alert(alert)
            return True
        return False
    
    async def delete_alert(self, alert_id: str) -> bool:
        """Delete an alert."""
        # Most stores don't support deletion, acknowledge instead
        return False


# =============================================================================
# Service Factory Functions
# =============================================================================


async def get_policy_service() -> BudgetPolicyService:
    """Get or create the budget policy service instance."""
    global _policy_service, _initialized
    
    if _policy_service is None:
        store = _get_budget_store_backend()
        pricing = _get_pricing_registry()
        
        repository = StoreBackendPolicyAdapter(store)
        _policy_service = BudgetPolicyService(
            repository=repository,
            permission_checker=BudgetPermissionChecker(),
            pricing_registry=pricing,
        )
        await _policy_service.initialize()
        logger.debug("BudgetPolicyService created and initialized")
    
    return _policy_service


async def get_tracking_service() -> BudgetTrackingService:
    """Get or create the budget tracking service instance."""
    global _tracking_service
    
    if _tracking_service is None:
        store = _get_budget_store_backend()
        pricing = _get_pricing_registry()
        
        usage_repo = StoreBackendUsageAdapter(store)
        policy_repo = StoreBackendPolicyAdapter(store)
        
        _tracking_service = BudgetTrackingService(
            usage_repository=usage_repo,
            policy_repository=policy_repo,
            pricing=pricing,
        )
        await _tracking_service.initialize()
        logger.debug("BudgetTrackingService created and initialized")
    
    return _tracking_service


async def get_alert_service() -> BudgetAlertService:
    """Get or create the budget alert service instance."""
    global _alert_service
    
    if _alert_service is None:
        store = _get_budget_store_backend()
        
        alert_repo = StoreBackendAlertAdapter(store)
        policy_repo = StoreBackendPolicyAdapter(store)
        
        _alert_service = BudgetAlertService(
            alert_repository=alert_repo,
            policy_repository=policy_repo,
        )
        await _alert_service.initialize()
        logger.debug("BudgetAlertService created and initialized")
    
    return _alert_service


async def get_all_services() -> tuple:
    """Get all budget service instances as a tuple.
    
    Returns:
        Tuple of (policy_service, tracking_service, alert_service)
    """
    policy = await get_policy_service()
    tracking = await get_tracking_service()
    alert = await get_alert_service()
    return policy, tracking, alert


async def shutdown_services() -> None:
    """Shutdown all budget services."""
    global _policy_service, _tracking_service, _alert_service, _initialized
    
    if _tracking_service:
        await _tracking_service.shutdown()
        _tracking_service = None
    
    if _alert_service:
        await _alert_service.shutdown()
        _alert_service = None
    
    if _policy_service:
        await _policy_service.cleanup()
        _policy_service = None
    
    _initialized = False
    logger.info("Budget services shut down")


def reset_services() -> None:
    """Reset all service instances (for testing)."""
    global _policy_service, _tracking_service, _alert_service, _initialized
    _policy_service = None
    _tracking_service = None
    _alert_service = None
    _initialized = False
