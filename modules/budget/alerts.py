"""Alert engine for budget monitoring.

Handles threshold detection, alert creation, acknowledgment,
and integration with the MessageBus for notifications.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from modules.logging.logger import setup_logger

from .models import (
    AlertSeverity,
    AlertTriggerType,
    BudgetAlert,
    BudgetPolicy,
    SpendSummary,
)

if TYPE_CHECKING:
    from core.config import ConfigManager
    from .manager import BudgetManager

logger = setup_logger(__name__)

# Module-level singleton
_alert_engine_instance: Optional["AlertEngine"] = None
_alert_engine_lock: Optional[asyncio.Lock] = None


async def get_alert_engine(
    config_manager: Optional["ConfigManager"] = None,
) -> "AlertEngine":
    """Get the global AlertEngine singleton.

    Args:
        config_manager: Configuration manager (required on first call).

    Returns:
        Initialized AlertEngine instance.
    """
    global _alert_engine_instance, _alert_engine_lock

    if _alert_engine_instance is not None:
        return _alert_engine_instance

    if _alert_engine_lock is None:
        _alert_engine_lock = asyncio.Lock()

    async with _alert_engine_lock:
        if _alert_engine_instance is None:
            _alert_engine_instance = AlertEngine(config_manager)
            await _alert_engine_instance.initialize()
            logger.info("AlertEngine singleton created")

    return _alert_engine_instance


class AlertRule:
    """Defines a rule for triggering budget alerts.

    Attributes:
        threshold_percent: Percentage threshold (0.0-1.0).
        severity: Alert severity when triggered.
        trigger_type: Type of trigger.
        cooldown_minutes: Minimum time between alerts of same type.
        notification_channels: Channels to notify.
    """

    def __init__(
        self,
        threshold_percent: float,
        severity: AlertSeverity,
        trigger_type: AlertTriggerType = AlertTriggerType.THRESHOLD_REACHED,
        cooldown_minutes: int = 60,
        notification_channels: Optional[List[str]] = None,
    ):
        """Initialize an alert rule.

        Args:
            threshold_percent: Percentage threshold.
            severity: Alert severity.
            trigger_type: Type of trigger.
            cooldown_minutes: Cooldown period.
            notification_channels: Notification channels.
        """
        self.threshold_percent = threshold_percent
        self.severity = severity
        self.trigger_type = trigger_type
        self.cooldown_minutes = cooldown_minutes
        self.notification_channels = notification_channels or ["in_app"]


# Default alert rules
DEFAULT_ALERT_RULES = [
    AlertRule(
        threshold_percent=0.50,
        severity=AlertSeverity.INFO,
        cooldown_minutes=1440,  # 24 hours
    ),
    AlertRule(
        threshold_percent=0.75,
        severity=AlertSeverity.INFO,
        cooldown_minutes=720,  # 12 hours
    ),
    AlertRule(
        threshold_percent=0.90,
        severity=AlertSeverity.WARNING,
        cooldown_minutes=120,  # 2 hours
    ),
    AlertRule(
        threshold_percent=1.0,
        severity=AlertSeverity.CRITICAL,
        trigger_type=AlertTriggerType.LIMIT_EXCEEDED,
        cooldown_minutes=30,
    ),
    AlertRule(
        threshold_percent=1.25,
        severity=AlertSeverity.EMERGENCY,
        trigger_type=AlertTriggerType.LIMIT_EXCEEDED,
        cooldown_minutes=15,
    ),
]


class AlertEngine:
    """Engine for monitoring budget thresholds and generating alerts.

    Provides:
    - Threshold monitoring
    - Alert rule management
    - Alert lifecycle (create, acknowledge, resolve)
    - MessageBus integration for notifications
    - Anomaly detection hooks

    Usage::

        engine = await get_alert_engine(config_manager)

        # Check thresholds
        alerts = await engine.evaluate_thresholds(policy, summary)

        # Get active alerts
        active = await engine.get_active_alerts()

        # Acknowledge an alert
        await engine.acknowledge_alert(alert_id, user_id)
    """

    def __init__(self, config_manager: Optional["ConfigManager"] = None):
        """Initialize the alert engine.

        Args:
            config_manager: Optional configuration manager.
        """
        self.config_manager = config_manager
        self.logger = setup_logger(__name__)

        # Alert storage (in-memory, synced with persistence)
        self._alerts: Dict[str, BudgetAlert] = {}
        self._alert_lock = asyncio.Lock()

        # Alert rules
        self._rules: List[AlertRule] = list(DEFAULT_ALERT_RULES)

        # Last alert timestamps by (policy_id, threshold)
        self._last_alert_times: Dict[tuple[str, float], datetime] = {}

        # Notification callbacks
        self._notification_handlers: Dict[str, List[Callable]] = {
            "in_app": [],
            "email": [],
            "webhook": [],
            "message_bus": [],
        }

        # Configuration
        self._enabled = True
        self._anomaly_detection_enabled = False
        self._anomaly_threshold_stddev = 2.0

    async def initialize(self) -> None:
        """Initialize the alert engine."""
        self._load_config()
        await self._load_alerts()
        await self._setup_message_bus()
        self.logger.info(
            "AlertEngine initialized with %d rules, %d active alerts",
            len(self._rules),
            len([a for a in self._alerts.values() if not a.resolved]),
        )

    def _load_config(self) -> None:
        """Load alert configuration."""
        if self.config_manager is None:
            return

        try:
            alert_config = self.config_manager.get_config("BUDGET_ALERTS")
            if isinstance(alert_config, dict):
                self._enabled = alert_config.get("enabled", True)
                self._anomaly_detection_enabled = alert_config.get(
                    "anomaly_detection", False
                )
                self._anomaly_threshold_stddev = alert_config.get(
                    "anomaly_threshold_stddev", 2.0
                )

                # Load custom rules
                custom_rules = alert_config.get("rules", [])
                if custom_rules:
                    self._rules = []
                    for rule_data in custom_rules:
                        self._rules.append(
                            AlertRule(
                                threshold_percent=rule_data.get("threshold", 0.8),
                                severity=AlertSeverity(
                                    rule_data.get("severity", "warning")
                                ),
                                trigger_type=AlertTriggerType(
                                    rule_data.get("trigger", "threshold_reached")
                                ),
                                cooldown_minutes=rule_data.get("cooldown_minutes", 60),
                                notification_channels=rule_data.get("channels"),
                            )
                        )
        except Exception as exc:
            self.logger.warning("Failed to load alert config: %s", exc)

    async def _load_alerts(self) -> None:
        """Load active alerts from persistence."""
        try:
            from modules.budget import get_budget_manager_sync

            manager = get_budget_manager_sync()
            if manager and manager._persistence:
                alerts = await manager._persistence.get_alerts(active_only=True)
                async with self._alert_lock:
                    self._active_alerts = {a.id: a for a in alerts}
                self.logger.debug("Loaded %d active alerts from storage", len(alerts))
        except Exception as exc:
            self.logger.warning("Failed to load alerts from persistence: %s", exc)

    async def _setup_message_bus(self) -> None:
        """Set up MessageBus integration for alert notifications."""
        try:
            from core.messaging import get_message_bus

            bus = get_message_bus()
            if bus:
                self._notification_handlers["message_bus"].append(
                    lambda alert: bus.publish("budget.alert", alert.as_dict())
                )
        except Exception:
            self.logger.debug("MessageBus not available for alert notifications")

    # =========================================================================
    # Threshold Evaluation
    # =========================================================================

    async def evaluate_thresholds(
        self,
        policy: BudgetPolicy,
        summary: SpendSummary,
    ) -> List[BudgetAlert]:
        """Evaluate alert thresholds for a policy.

        Args:
            policy: Budget policy to check.
            summary: Current spending summary.

        Returns:
            List of newly created alerts.
        """
        if not self._enabled or not policy.enabled:
            return []

        new_alerts: List[BudgetAlert] = []
        percent_used = summary.percent_used

        for rule in self._rules:
            if percent_used >= rule.threshold_percent:
                alert = await self._maybe_create_alert(
                    policy=policy,
                    summary=summary,
                    rule=rule,
                )
                if alert:
                    new_alerts.append(alert)

        return new_alerts

    async def _maybe_create_alert(
        self,
        policy: BudgetPolicy,
        summary: SpendSummary,
        rule: AlertRule,
    ) -> Optional[BudgetAlert]:
        """Create an alert if cooldown has passed.

        Args:
            policy: Budget policy.
            summary: Spending summary.
            rule: Alert rule that triggered.

        Returns:
            New BudgetAlert if created, None otherwise.
        """
        cache_key = (policy.id, rule.threshold_percent)
        now = datetime.now(timezone.utc)

        # Check cooldown
        last_alert_time = self._last_alert_times.get(cache_key)
        if last_alert_time:
            minutes_since = (now - last_alert_time).total_seconds() / 60
            if minutes_since < rule.cooldown_minutes:
                return None

        # Check if unresolved alert already exists
        async with self._alert_lock:
            for alert in self._alerts.values():
                if (
                    alert.policy_id == policy.id
                    and alert.threshold_percent == rule.threshold_percent
                    and not alert.resolved
                ):
                    return None

        # Create the alert
        if rule.trigger_type == AlertTriggerType.LIMIT_EXCEEDED:
            message = (
                f"Budget limit exceeded for '{policy.name}': "
                f"${summary.total_spent:.2f} / ${policy.limit_amount:.2f} "
                f"({summary.percent_used:.1%})"
            )
        else:
            message = (
                f"Budget threshold reached for '{policy.name}': "
                f"{summary.percent_used:.1%} used "
                f"(${summary.total_spent:.2f} / ${policy.limit_amount:.2f})"
            )

        alert = BudgetAlert(
            policy_id=policy.id,
            severity=rule.severity,
            trigger_type=rule.trigger_type,
            threshold_percent=rule.threshold_percent,
            current_spend=summary.total_spent,
            limit_amount=policy.limit_amount,
            message=message,
        )

        # Store alert
        async with self._alert_lock:
            self._alerts[alert.id] = alert

        # Update last alert time
        self._last_alert_times[cache_key] = now

        # Send notifications
        await self._send_notifications(alert, rule.notification_channels)

        self.logger.warning("Budget alert created: %s", message)

        return alert

    # =========================================================================
    # Anomaly Detection
    # =========================================================================

    async def check_for_anomalies(
        self,
        policy: BudgetPolicy,
        recent_costs: List[Decimal],
        current_cost: Decimal,
    ) -> Optional[BudgetAlert]:
        """Check for spending anomalies.

        Args:
            policy: Budget policy.
            recent_costs: Recent cost values for comparison.
            current_cost: Current cost to check.

        Returns:
            BudgetAlert if anomaly detected, None otherwise.
        """
        if not self._anomaly_detection_enabled:
            return None

        if len(recent_costs) < 10:
            return None  # Not enough data

        # Calculate mean and standard deviation
        costs_float = [float(c) for c in recent_costs]
        mean = sum(costs_float) / len(costs_float)
        variance = sum((x - mean) ** 2 for x in costs_float) / len(costs_float)
        stddev = variance ** 0.5

        if stddev == 0:
            return None

        # Check if current cost is an anomaly
        z_score = (float(current_cost) - mean) / stddev
        if abs(z_score) < self._anomaly_threshold_stddev:
            return None

        # Create anomaly alert
        message = (
            f"Unusual spending detected for '{policy.name}': "
            f"${current_cost:.4f} is {z_score:.1f} standard deviations "
            f"from the mean (${mean:.4f})"
        )

        alert = BudgetAlert(
            policy_id=policy.id,
            severity=AlertSeverity.WARNING,
            trigger_type=AlertTriggerType.ANOMALY_DETECTED,
            current_spend=current_cost,
            limit_amount=policy.limit_amount,
            message=message,
            metadata={"z_score": z_score, "mean": mean, "stddev": stddev},
        )

        async with self._alert_lock:
            self._alerts[alert.id] = alert

        await self._send_notifications(alert, ["in_app", "message_bus"])

        self.logger.warning("Spending anomaly detected: %s", message)

        return alert

    # =========================================================================
    # Alert Management
    # =========================================================================

    async def get_active_alerts(
        self,
        policy_id: Optional[str] = None,
        severity: Optional[AlertSeverity] = None,
        unacknowledged_only: bool = False,
    ) -> List[BudgetAlert]:
        """Get active (unresolved) alerts.

        Args:
            policy_id: Filter by policy ID.
            severity: Filter by severity.
            unacknowledged_only: Only return unacknowledged alerts.

        Returns:
            List of matching alerts.
        """
        async with self._alert_lock:
            alerts = [a for a in self._alerts.values() if not a.resolved]

        if policy_id:
            alerts = [a for a in alerts if a.policy_id == policy_id]

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        if unacknowledged_only:
            alerts = [a for a in alerts if not a.acknowledged]

        # Sort by severity then time
        severity_order = {
            AlertSeverity.EMERGENCY: 0,
            AlertSeverity.CRITICAL: 1,
            AlertSeverity.WARNING: 2,
            AlertSeverity.INFO: 3,
        }
        alerts.sort(
            key=lambda a: (
                severity_order.get(a.severity, 99),
                -a.triggered_at.timestamp(),
            )
        )

        return alerts

    async def get_alert(self, alert_id: str) -> Optional[BudgetAlert]:
        """Get a specific alert by ID.

        Args:
            alert_id: Alert identifier.

        Returns:
            BudgetAlert if found, None otherwise.
        """
        async with self._alert_lock:
            return self._alerts.get(alert_id)

    async def acknowledge_alert(
        self,
        alert_id: str,
        user_id: Optional[str] = None,
    ) -> bool:
        """Acknowledge an alert.

        Args:
            alert_id: Alert identifier.
            user_id: User acknowledging the alert.

        Returns:
            True if acknowledged, False if not found.
        """
        async with self._alert_lock:
            alert = self._alerts.get(alert_id)
            if alert is None:
                return False

            alert.acknowledge(user_id)

        self.logger.info("Alert acknowledged: %s by %s", alert_id, user_id)
        return True

    async def resolve_alert(self, alert_id: str) -> bool:
        """Resolve (close) an alert.

        Args:
            alert_id: Alert identifier.

        Returns:
            True if resolved, False if not found.
        """
        async with self._alert_lock:
            alert = self._alerts.get(alert_id)
            if alert is None:
                return False

            alert.resolve()

        self.logger.info("Alert resolved: %s", alert_id)
        return True

    async def resolve_alerts_for_policy(self, policy_id: str) -> int:
        """Resolve all alerts for a policy.

        Args:
            policy_id: Policy identifier.

        Returns:
            Number of alerts resolved.
        """
        count = 0
        async with self._alert_lock:
            for alert in self._alerts.values():
                if alert.policy_id == policy_id and not alert.resolved:
                    alert.resolve()
                    count += 1

        self.logger.info("Resolved %d alerts for policy %s", count, policy_id)
        return count

    # =========================================================================
    # Notifications
    # =========================================================================

    async def _send_notifications(
        self,
        alert: BudgetAlert,
        channels: List[str],
    ) -> None:
        """Send alert notifications to specified channels.

        Args:
            alert: The alert to notify about.
            channels: Notification channels to use.
        """
        for channel in channels:
            handlers = self._notification_handlers.get(channel, [])
            for handler in handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(alert)
                    else:
                        handler(alert)
                    alert.notification_sent = True
                except Exception as exc:
                    self.logger.warning(
                        "Failed to send notification via %s: %s", channel, exc
                    )

    def register_notification_handler(
        self,
        channel: str,
        handler: Callable[[BudgetAlert], Any],
    ) -> None:
        """Register a notification handler.

        Args:
            channel: Notification channel name.
            handler: Handler function.
        """
        if channel not in self._notification_handlers:
            self._notification_handlers[channel] = []
        self._notification_handlers[channel].append(handler)

    # =========================================================================
    # Rule Management
    # =========================================================================

    def add_rule(self, rule: AlertRule) -> None:
        """Add an alert rule.

        Args:
            rule: Alert rule to add.
        """
        self._rules.append(rule)
        self._rules.sort(key=lambda r: r.threshold_percent)

    def remove_rule(self, threshold_percent: float) -> bool:
        """Remove an alert rule by threshold.

        Args:
            threshold_percent: Threshold of rule to remove.

        Returns:
            True if removed, False if not found.
        """
        for i, rule in enumerate(self._rules):
            if rule.threshold_percent == threshold_percent:
                self._rules.pop(i)
                return True
        return False

    def get_rules(self) -> List[AlertRule]:
        """Get all alert rules.

        Returns:
            List of alert rules.
        """
        return list(self._rules)

    @property
    def enabled(self) -> bool:
        """Whether alerting is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable or disable alerting."""
        self._enabled = value
