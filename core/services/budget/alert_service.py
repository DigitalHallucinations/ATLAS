"""
Budget Alert Service implementation.

Provides alert configuration, threshold monitoring, and notifications
for budget policies. Part of the ATLAS budget service layer.

Author: ATLAS Team
Date: Jan 8, 2026
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    List,
    Optional,
    Protocol,
    runtime_checkable,
)

from modules.budget.models import (
    AlertSeverity,
    AlertTriggerType,
    BudgetAlert,
    BudgetPolicy,
    BudgetScope,
    BudgetPeriod,
    SpendSummary,
)

from .types import (
    # Events
    BudgetAlertTriggered,
    BudgetAlertAcknowledged,
    BudgetLimitExceeded,
    BudgetApproachingLimit,
    # DTOs
    AlertConfigCreate,
    AlertConfigUpdate,
    AlertConfig,
    ActiveAlert,
    AlertListRequest,
    _now_utc,
    _generate_uuid,
)
from .exceptions import (
    BudgetError,
    BudgetValidationError,
)


logger = logging.getLogger(__name__)


# =============================================================================
# Severity ordering for sorting
# =============================================================================

SEVERITY_ORDER = {
    "emergency": 0,
    "critical": 1,
    "warning": 2,
    "info": 3,
}


# =============================================================================
# Repository Protocols
# =============================================================================


@runtime_checkable
class AlertRepository(Protocol):
    """Protocol for alert persistence operations."""

    async def save_alert(self, alert: BudgetAlert) -> str:
        """Save an alert, return its ID."""
        ...

    async def get_alert(self, alert_id: str) -> Optional[BudgetAlert]:
        """Get an alert by ID."""
        ...

    async def get_alerts(
        self,
        policy_id: Optional[str] = None,
        severity: Optional[str] = None,
        active_only: bool = True,
        limit: int = 100,
        offset: int = 0,
    ) -> List[BudgetAlert]:
        """Get alerts with filtering."""
        ...

    async def update_alert(self, alert: BudgetAlert) -> bool:
        """Update an existing alert."""
        ...

    async def delete_alert(self, alert_id: str) -> bool:
        """Delete an alert."""
        ...


@runtime_checkable
class AlertConfigRepository(Protocol):
    """Protocol for alert configuration persistence."""

    async def save_config(self, config: AlertConfig) -> str:
        """Save an alert configuration."""
        ...

    async def get_config(self, config_id: str) -> Optional[AlertConfig]:
        """Get config by ID."""
        ...

    async def get_configs_for_policy(self, policy_id: str) -> List[AlertConfig]:
        """Get all configs for a policy."""
        ...

    async def delete_config(self, config_id: str) -> bool:
        """Delete an alert configuration."""
        ...


@runtime_checkable
class PolicyRepository(Protocol):
    """Protocol for budget policy lookups."""

    async def get_policy(self, policy_id: str) -> Optional[BudgetPolicy]:
        """Get policy by ID."""
        ...

    async def get_policies(
        self,
        scope: Optional[BudgetScope] = None,
        scope_id: Optional[str] = None,
        enabled_only: bool = True,
    ) -> List[BudgetPolicy]:
        """Get policies with filtering."""
        ...


@runtime_checkable
class SpendingRepository(Protocol):
    """Protocol for spending data lookups."""

    async def get_current_spend(
        self,
        scope: BudgetScope,
        scope_id: Optional[str],
        period: BudgetPeriod,
    ) -> SpendSummary:
        """Get current spending summary."""
        ...


# =============================================================================
# Event Publisher Type
# =============================================================================

AlertEventPublisher = Callable[
    [BudgetAlertTriggered | BudgetAlertAcknowledged | BudgetLimitExceeded | BudgetApproachingLimit],
    Coroutine[Any, Any, None],
]


# =============================================================================
# Budget Alert Service
# =============================================================================


class BudgetAlertService:
    """Service for managing budget alerts and threshold monitoring.

    This service handles:
    - Alert configuration (thresholds, channels, cooldowns)
    - Active alert management (create, acknowledge, resolve)
    - Background threshold evaluation
    - Alert event publishing

    Example:
        service = BudgetAlertService(
            alert_repository=alert_repo,
            config_repository=config_repo,
            policy_repository=policy_repo,
            spending_repository=spend_repo,
        )
        await service.initialize()

        # Configure an alert threshold
        config = await service.configure_alert(
            actor=user_actor,
            config=AlertConfigCreate(
                policy_id="policy_123",
                threshold_percent=0.80,
                severity="warning",
            ),
        )

        # Evaluate all thresholds (typically run in background)
        await service.evaluate_alerts()

        # Get active alerts
        alerts = await service.get_active_alerts(
            actor=user_actor,
            request=AlertListRequest(active_only=True),
        )
    """

    def __init__(
        self,
        alert_repository: Optional[AlertRepository] = None,
        config_repository: Optional[AlertConfigRepository] = None,
        policy_repository: Optional[PolicyRepository] = None,
        spending_repository: Optional[SpendingRepository] = None,
        event_publisher: Optional[AlertEventPublisher] = None,
        *,
        evaluation_interval_seconds: int = 300,  # 5 minutes
        default_thresholds: Optional[List[float]] = None,
    ):
        """Initialize the alert service.

        Args:
            alert_repository: Repository for alert persistence.
            config_repository: Repository for alert config persistence.
            policy_repository: Repository for policy lookups.
            spending_repository: Repository for spending data.
            event_publisher: Callback for publishing domain events.
            evaluation_interval_seconds: Interval for background evaluation.
            default_thresholds: Default alert thresholds (e.g., [0.5, 0.8, 0.9, 1.0]).
        """
        self._alert_repo = alert_repository
        self._config_repo = config_repository
        self._policy_repo = policy_repository
        self._spending_repo = spending_repository
        self._event_publisher = event_publisher
        
        self._evaluation_interval = evaluation_interval_seconds
        self._default_thresholds = default_thresholds or [0.5, 0.8, 0.9, 1.0]
        
        # In-memory state
        self._alerts: List[BudgetAlert] = []
        self._alert_lock = asyncio.Lock()
        self._configs: Dict[str, AlertConfig] = {}  # config_id -> config
        
        # Threshold tracking to avoid duplicate alerts
        self._triggered_thresholds: Dict[str, float] = {}  # policy_id -> last_threshold
        
        # Background task
        self._evaluation_task: Optional[asyncio.Task] = None
        self._enabled = True
        
        # Lifecycle state
        self._initialized = False
        self._shutting_down = False
        
        self.logger = logger

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def initialize(self) -> None:
        """Initialize the alert service."""
        if self._initialized:
            return

        self.logger.info("Initializing BudgetAlertService")

        # Load active alerts from repository
        if self._alert_repo:
            try:
                alerts = await self._alert_repo.get_alerts(active_only=True)
                async with self._alert_lock:
                    self._alerts = list(alerts)
                self.logger.debug("Loaded %d active alerts", len(alerts))
            except Exception as exc:
                self.logger.warning("Failed to load alerts: %s", exc)

        # Start background evaluation task
        if self._evaluation_interval > 0:
            self._evaluation_task = asyncio.create_task(
                self._background_evaluation_loop()
            )

        self._initialized = True
        self.logger.info("BudgetAlertService initialized")

    async def shutdown(self) -> None:
        """Shutdown the alert service."""
        if self._shutting_down:
            return

        self._shutting_down = True
        self.logger.info("Shutting down BudgetAlertService")

        # Cancel background task
        if self._evaluation_task and not self._evaluation_task.done():
            self._evaluation_task.cancel()
            try:
                await self._evaluation_task
            except asyncio.CancelledError:
                pass

        self.logger.info("BudgetAlertService shutdown complete")

    async def _background_evaluation_loop(self) -> None:
        """Background task to periodically evaluate alert thresholds."""
        while not self._shutting_down:
            try:
                await asyncio.sleep(self._evaluation_interval)
                await self.evaluate_alerts()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self.logger.warning("Background evaluation error: %s", exc)

    # =========================================================================
    # Alert Configuration
    # =========================================================================

    async def configure_alert(
        self,
        actor: Any,
        config: AlertConfigCreate,
    ) -> AlertConfig:
        """Configure an alert threshold for a policy.

        Args:
            actor: The actor configuring the alert.
            config: Alert configuration data.

        Returns:
            The created AlertConfig.

        Raises:
            BudgetValidationError: If configuration is invalid.
        """
        # Validate policy exists
        if self._policy_repo:
            policy = await self._policy_repo.get_policy(config.policy_id)
            if not policy:
                raise BudgetValidationError(f"Policy not found: {config.policy_id}")

        # Create config
        alert_config = AlertConfig(
            id=_generate_uuid(),
            policy_id=config.policy_id,
            threshold_percent=config.threshold_percent,
            severity=config.severity,
            notification_channels=config.notification_channels,
            cooldown_minutes=config.cooldown_minutes,
            enabled=config.enabled,
            metadata=config.metadata,
        )

        # Persist if repository available
        if self._config_repo:
            try:
                await self._config_repo.save_config(alert_config)
            except Exception as exc:
                self.logger.warning("Failed to persist alert config: %s", exc)

        # Cache in memory
        self._configs[alert_config.id] = alert_config

        self.logger.info(
            "Alert configured for policy %s at %.0f%% threshold",
            config.policy_id,
            config.threshold_percent * 100,
        )

        return alert_config

    async def update_alert_config(
        self,
        actor: Any,
        config_id: str,
        updates: AlertConfigUpdate,
    ) -> Optional[AlertConfig]:
        """Update an alert configuration.

        Args:
            actor: The actor updating the config.
            config_id: Configuration ID to update.
            updates: Fields to update.

        Returns:
            Updated AlertConfig or None if not found.
        """
        config = self._configs.get(config_id)
        if not config:
            if self._config_repo:
                config = await self._config_repo.get_config(config_id)
            if not config:
                return None

        # Apply updates
        if updates.threshold_percent is not None:
            config.threshold_percent = updates.threshold_percent
        if updates.severity is not None:
            config.severity = updates.severity
        if updates.notification_channels is not None:
            config.notification_channels = updates.notification_channels
        if updates.cooldown_minutes is not None:
            config.cooldown_minutes = updates.cooldown_minutes
        if updates.enabled is not None:
            config.enabled = updates.enabled
        if updates.metadata is not None:
            config.metadata = updates.metadata

        config.updated_at = _now_utc()

        # Persist
        if self._config_repo:
            try:
                await self._config_repo.save_config(config)
            except Exception as exc:
                self.logger.warning("Failed to persist config update: %s", exc)

        self._configs[config_id] = config
        return config

    async def remove_alert_config(
        self,
        actor: Any,
        config_id: str,
    ) -> bool:
        """Remove an alert configuration.

        Args:
            actor: The actor removing the config.
            config_id: Configuration ID to remove.

        Returns:
            True if removed, False if not found.
        """
        if config_id in self._configs:
            del self._configs[config_id]

        if self._config_repo:
            try:
                return await self._config_repo.delete_config(config_id)
            except Exception as exc:
                self.logger.warning("Failed to delete config: %s", exc)

        return True

    async def list_alert_configs(
        self,
        actor: Any,
        policy_id: Optional[str] = None,
    ) -> List[AlertConfig]:
        """List alert configurations.

        Args:
            actor: The actor requesting the list.
            policy_id: Optional filter by policy.

        Returns:
            List of AlertConfig objects.
        """
        configs = list(self._configs.values())

        if policy_id:
            configs = [c for c in configs if c.policy_id == policy_id]

        # Also load from repository if available
        if self._config_repo and policy_id:
            try:
                repo_configs = await self._config_repo.get_configs_for_policy(policy_id)
                # Merge, preferring in-memory versions
                existing_ids = {c.id for c in configs}
                for rc in repo_configs:
                    if rc.id not in existing_ids:
                        configs.append(rc)
            except Exception as exc:
                self.logger.warning("Failed to load configs from repo: %s", exc)

        return configs

    # =========================================================================
    # Active Alert Management
    # =========================================================================

    async def get_active_alerts(
        self,
        actor: Any,
        request: Optional[AlertListRequest] = None,
    ) -> List[ActiveAlert]:
        """Get active (unresolved) alerts.

        Args:
            actor: The actor requesting alerts.
            request: Optional filter criteria.

        Returns:
            List of active alerts.
        """
        request = request or AlertListRequest()

        async with self._alert_lock:
            alerts = list(self._alerts)

        # Apply filters
        if request.active_only:
            alerts = [a for a in alerts if not a.resolved]

        if request.policy_id:
            alerts = [a for a in alerts if a.policy_id == request.policy_id]

        if request.severity:
            severity_val = request.severity.lower()
            alerts = [a for a in alerts if a.severity.value == severity_val]

        if request.unacknowledged_only:
            alerts = [a for a in alerts if not a.acknowledged]

        # Sort by severity (critical first) then by time (newest first)
        alerts.sort(
            key=lambda a: (
                SEVERITY_ORDER.get(a.severity.value, 99),
                -a.triggered_at.timestamp(),
            )
        )

        # Apply pagination
        alerts = alerts[request.offset : request.offset + request.limit]

        # Convert to ActiveAlert DTOs
        active_alerts = []
        for alert in alerts:
            # Get policy name
            policy_name = ""
            if self._policy_repo:
                policy = await self._policy_repo.get_policy(alert.policy_id)
                if policy:
                    policy_name = policy.name

            active_alerts.append(
                ActiveAlert(
                    id=alert.id,
                    policy_id=alert.policy_id,
                    policy_name=policy_name,
                    severity=alert.severity.value,
                    trigger_type=alert.trigger_type.value,
                    threshold_percent=alert.threshold_percent or 0.0,
                    current_spend=alert.current_spend,
                    limit_amount=alert.limit_amount,
                    message=alert.message,
                    triggered_at=alert.triggered_at,
                    acknowledged=alert.acknowledged,
                    acknowledged_at=alert.acknowledged_at,
                    acknowledged_by=alert.acknowledged_by,
                )
            )

        return active_alerts

    async def acknowledge_alert(
        self,
        actor: Any,
        alert_id: str,
    ) -> bool:
        """Acknowledge a budget alert.

        Args:
            actor: The actor acknowledging the alert.
            alert_id: ID of the alert to acknowledge.

        Returns:
            True if acknowledged, False if not found.
        """
        actor_id = getattr(actor, "user_id", None) or getattr(actor, "id", "system")

        async with self._alert_lock:
            for alert in self._alerts:
                if alert.id == alert_id:
                    alert.acknowledge(actor_id)

                    # Persist update
                    if self._alert_repo:
                        try:
                            await self._alert_repo.update_alert(alert)
                        except Exception as exc:
                            self.logger.warning("Failed to persist acknowledgment: %s", exc)

                    # Publish event
                    await self._publish_acknowledged_event(actor, alert)

                    self.logger.info("Alert %s acknowledged by %s", alert_id, actor_id)
                    return True

        return False

    async def resolve_alert(
        self,
        actor: Any,
        alert_id: str,
    ) -> bool:
        """Resolve (close) a budget alert.

        Args:
            actor: The actor resolving the alert.
            alert_id: ID of the alert to resolve.

        Returns:
            True if resolved, False if not found.
        """
        async with self._alert_lock:
            for alert in self._alerts:
                if alert.id == alert_id:
                    alert.resolve()

                    # Persist update
                    if self._alert_repo:
                        try:
                            await self._alert_repo.update_alert(alert)
                        except Exception as exc:
                            self.logger.warning("Failed to persist resolution: %s", exc)

                    self.logger.info("Alert %s resolved", alert_id)
                    return True

        return False

    # =========================================================================
    # Alert Evaluation
    # =========================================================================

    async def evaluate_alerts(self) -> int:
        """Evaluate all policy thresholds and trigger alerts as needed.

        This method should be called periodically (e.g., every 5 minutes)
        to check if any budget thresholds have been crossed.

        Returns:
            Number of new alerts triggered.
        """
        if not self._enabled:
            return 0

        if not self._policy_repo or not self._spending_repo:
            return 0

        alerts_triggered = 0

        try:
            # Get all enabled policies
            policies = await self._policy_repo.get_policies(enabled_only=True)

            for policy in policies:
                try:
                    # Get current spending
                    summary = await self._spending_repo.get_current_spend(
                        scope=policy.scope,
                        scope_id=policy.scope_id,
                        period=policy.period,
                    )

                    # Check thresholds
                    triggered = await self._check_policy_thresholds(policy, summary)
                    alerts_triggered += triggered

                except Exception as exc:
                    self.logger.warning(
                        "Failed to evaluate policy %s: %s", policy.id, exc
                    )

        except Exception as exc:
            self.logger.error("Alert evaluation failed: %s", exc)

        if alerts_triggered > 0:
            self.logger.info("Alert evaluation triggered %d new alerts", alerts_triggered)

        return alerts_triggered

    async def _check_policy_thresholds(
        self,
        policy: BudgetPolicy,
        summary: SpendSummary,
    ) -> int:
        """Check thresholds for a single policy.

        Args:
            policy: Budget policy to check.
            summary: Current spending summary.

        Returns:
            Number of alerts triggered.
        """
        percent_used = summary.percent_used
        thresholds = self._get_thresholds_for_policy(policy)
        alerts_triggered = 0

        for threshold in thresholds:
            if percent_used >= threshold:
                # Check if we've already alerted for this threshold
                last_threshold = self._triggered_thresholds.get(policy.id, 0.0)
                if threshold <= last_threshold:
                    continue

                # Create alert
                created = await self._create_alert(policy, summary, threshold)
                if created:
                    alerts_triggered += 1
                    self._triggered_thresholds[policy.id] = threshold

        # Reset tracking if spending drops below all thresholds
        if percent_used < min(thresholds, default=0.5):
            self._triggered_thresholds.pop(policy.id, None)

        return alerts_triggered

    def _get_thresholds_for_policy(self, policy: BudgetPolicy) -> List[float]:
        """Get alert thresholds for a policy.

        Uses configured thresholds or defaults, always including
        the policy's soft limit.
        """
        thresholds = list(self._default_thresholds)

        # Always include the soft limit
        if policy.soft_limit_percent not in thresholds:
            thresholds.append(policy.soft_limit_percent)

        return sorted(set(thresholds))

    async def _create_alert(
        self,
        policy: BudgetPolicy,
        summary: SpendSummary,
        threshold: float,
    ) -> bool:
        """Create an alert for a threshold crossing.

        Args:
            policy: Budget policy that was exceeded.
            summary: Current spending summary.
            threshold: Threshold that was crossed.

        Returns:
            True if alert was created, False if already exists.
        """
        # Check if alert already exists for this threshold
        async with self._alert_lock:
            for existing in self._alerts:
                if (
                    existing.policy_id == policy.id
                    and existing.threshold_percent == threshold
                    and not existing.resolved
                ):
                    return False

        # Determine severity and trigger type
        if threshold >= 1.0:
            severity = AlertSeverity.CRITICAL
            trigger_type = AlertTriggerType.LIMIT_EXCEEDED
            message = f"Budget exceeded: ${summary.total_spent:.2f} / ${policy.limit_amount:.2f}"
        elif threshold >= policy.soft_limit_percent:
            severity = AlertSeverity.WARNING
            trigger_type = AlertTriggerType.THRESHOLD_REACHED
            message = f"Budget warning: {summary.percent_used:.1%} used (${summary.total_spent:.2f} / ${policy.limit_amount:.2f})"
        else:
            severity = AlertSeverity.INFO
            trigger_type = AlertTriggerType.THRESHOLD_REACHED
            message = f"Budget checkpoint: {summary.percent_used:.1%} used"

        alert = BudgetAlert(
            policy_id=policy.id,
            severity=severity,
            trigger_type=trigger_type,
            threshold_percent=threshold,
            current_spend=summary.total_spent,
            limit_amount=policy.limit_amount,
            message=message,
        )

        # Add to in-memory list
        async with self._alert_lock:
            self._alerts.append(alert)

        # Persist
        if self._alert_repo:
            try:
                await self._alert_repo.save_alert(alert)
            except Exception as exc:
                self.logger.warning("Failed to persist alert: %s", exc)

        self.logger.warning("Budget alert created: %s", message)

        # Publish events
        await self._publish_alert_event(policy, alert, summary)

        return True

    # =========================================================================
    # Event Publishing
    # =========================================================================

    async def _publish_alert_event(
        self,
        policy: BudgetPolicy,
        alert: BudgetAlert,
        summary: SpendSummary,
    ) -> None:
        """Publish alert triggered event."""
        if not self._event_publisher:
            return

        tenant_id = (
            policy.scope_id if policy.scope == BudgetScope.TEAM and policy.scope_id
            else "global"
        )

        # Publish main alert event
        event = BudgetAlertTriggered(
            alert_id=alert.id,
            policy_id=policy.id,
            policy_name=policy.name,
            severity=alert.severity.value,
            trigger_type=alert.trigger_type.value,
            threshold_percent=alert.threshold_percent or 0.0,
            current_spend=alert.current_spend,
            limit_amount=alert.limit_amount,
            message=alert.message,
            tenant_id=tenant_id,
            scope=policy.scope.value,
            scope_id=policy.scope_id,
        )

        try:
            await self._event_publisher(event)
        except Exception as exc:
            self.logger.warning("Failed to publish alert event: %s", exc)

        # Publish specific limit/approaching events
        if alert.trigger_type == AlertTriggerType.LIMIT_EXCEEDED:
            overage = summary.total_spent - policy.limit_amount
            exceeded_event = BudgetLimitExceeded(
                policy_id=policy.id,
                policy_name=policy.name,
                current_spend=summary.total_spent,
                limit_amount=policy.limit_amount,
                overage_amount=max(Decimal("0"), overage),
                tenant_id=tenant_id,
                scope=policy.scope.value,
                scope_id=policy.scope_id,
                action_taken=policy.hard_limit_action.value,
            )
            try:
                await self._event_publisher(exceeded_event)
            except Exception as exc:
                self.logger.warning("Failed to publish limit exceeded event: %s", exc)

        elif alert.trigger_type == AlertTriggerType.THRESHOLD_REACHED:
            approaching_event = BudgetApproachingLimit(
                policy_id=policy.id,
                policy_name=policy.name,
                current_percent=summary.percent_used,
                threshold_percent=alert.threshold_percent or 0.0,
                current_spend=summary.total_spent,
                limit_amount=policy.limit_amount,
                remaining=summary.remaining,
                tenant_id=tenant_id,
                scope=policy.scope.value,
                scope_id=policy.scope_id,
            )
            try:
                await self._event_publisher(approaching_event)
            except Exception as exc:
                self.logger.warning("Failed to publish approaching limit event: %s", exc)

    async def _publish_acknowledged_event(
        self,
        actor: Any,
        alert: BudgetAlert,
    ) -> None:
        """Publish alert acknowledged event."""
        if not self._event_publisher:
            return

        actor_id = getattr(actor, "user_id", None) or getattr(actor, "id", "system")
        actor_type = getattr(actor, "actor_type", "user")
        tenant_id = getattr(actor, "tenant_id", "default")

        event = BudgetAlertAcknowledged(
            alert_id=alert.id,
            policy_id=alert.policy_id,
            severity=alert.severity.value,
            tenant_id=tenant_id,
            actor_id=actor_id,
            actor_type=actor_type,
        )

        try:
            await self._event_publisher(event)
        except Exception as exc:
            self.logger.warning("Failed to publish acknowledged event: %s", exc)

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable alert evaluation."""
        self._enabled = enabled
        self.logger.info("Alert evaluation %s", "enabled" if enabled else "disabled")

    def get_alert_count(self, active_only: bool = True) -> int:
        """Get count of alerts."""
        if active_only:
            return len([a for a in self._alerts if not a.resolved])
        return len(self._alerts)
