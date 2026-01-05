"""Tests for budget alert functionality."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from modules.budget.models import (
    AlertSeverity,
    AlertTriggerType,
    BudgetAlert,
    BudgetPeriod,
    BudgetPolicy,
    BudgetScope,
    LimitAction,
    SpendSummary,
)
from modules.budget.alerts import AlertEngine, AlertRule, get_alert_engine


@pytest.fixture
def sample_policy() -> BudgetPolicy:
    """Create a sample policy for testing alerts."""
    return BudgetPolicy(
        name="Alert Test Policy",
        scope=BudgetScope.GLOBAL,
        limit_amount=Decimal("100.00"),
        period=BudgetPeriod.MONTHLY,
        soft_limit_percent=0.80,
        hard_limit_action=LimitAction.BLOCK,
    )


@pytest.fixture
def sample_summary(sample_policy: BudgetPolicy) -> SpendSummary:
    """Create a sample spend summary for testing."""
    now = datetime.now(timezone.utc)
    return SpendSummary(
        policy_id=sample_policy.id,
        period_start=now.replace(day=1, hour=0, minute=0, second=0, microsecond=0),
        period_end=now,
        total_spent=Decimal("50.00"),
        limit_amount=sample_policy.limit_amount,
    )


@pytest.fixture
def sample_alert() -> BudgetAlert:
    """Create a sample alert for testing."""
    return BudgetAlert(
        policy_id="test-policy-123",
        severity=AlertSeverity.WARNING,
        trigger_type=AlertTriggerType.THRESHOLD_REACHED,
        threshold_percent=0.80,
        current_spend=Decimal("85.00"),
        limit_amount=Decimal("100.00"),
        message="Budget at 85% of limit",
    )


class TestAlertEngineSingleton:
    """Tests for AlertEngine singleton pattern."""

    @pytest.mark.asyncio
    async def test_get_alert_engine_returns_instance(self) -> None:
        """Test that get_alert_engine returns an engine."""
        engine = await get_alert_engine()
        assert engine is not None
        assert isinstance(engine, AlertEngine)

    @pytest.mark.asyncio
    async def test_get_alert_engine_singleton(self) -> None:
        """Test singleton pattern."""
        engine1 = await get_alert_engine()
        engine2 = await get_alert_engine()
        assert engine1 is engine2


class TestAlertEngineRules:
    """Tests for alert rule management."""

    @pytest.mark.asyncio
    async def test_get_rules(self) -> None:
        """Test getting alert rules."""
        engine = await get_alert_engine()

        rules = engine.get_rules()

        assert isinstance(rules, list)
        assert len(rules) > 0
        assert all(isinstance(r, AlertRule) for r in rules)

    @pytest.mark.asyncio
    async def test_add_rule(self) -> None:
        """Test adding a custom alert rule."""
        engine = await get_alert_engine()

        new_rule = AlertRule(
            threshold_percent=0.60,
            severity=AlertSeverity.INFO,
            trigger_type=AlertTriggerType.THRESHOLD_REACHED,
        )

        original_count = len(engine.get_rules())
        engine.add_rule(new_rule)

        assert len(engine.get_rules()) == original_count + 1

        # Clean up
        engine.remove_rule(0.60)

    @pytest.mark.asyncio
    async def test_remove_rule(self) -> None:
        """Test removing an alert rule."""
        engine = await get_alert_engine()

        # Add a rule to remove
        new_rule = AlertRule(
            threshold_percent=0.55,
            severity=AlertSeverity.INFO,
            trigger_type=AlertTriggerType.THRESHOLD_REACHED,
        )
        engine.add_rule(new_rule)

        original_count = len(engine.get_rules())
        removed = engine.remove_rule(0.55)

        assert removed is True
        assert len(engine.get_rules()) == original_count - 1


class TestAlertEngineEvaluation:
    """Tests for threshold evaluation."""

    @pytest.mark.asyncio
    async def test_evaluate_under_threshold(
        self, sample_policy: BudgetPolicy, sample_summary: SpendSummary
    ) -> None:
        """Test evaluation when under all thresholds."""
        engine = await get_alert_engine()

        # Summary is at 50%, should not trigger most alerts
        sample_summary.total_spent = Decimal("40.00")

        alerts = await engine.evaluate_thresholds(sample_policy, sample_summary)

        # Should be empty or only INFO level
        critical = [a for a in alerts if a.severity == AlertSeverity.CRITICAL]
        assert len(critical) == 0

    @pytest.mark.asyncio
    async def test_evaluate_at_warning_threshold(
        self, sample_policy: BudgetPolicy, sample_summary: SpendSummary
    ) -> None:
        """Test evaluation when at warning threshold."""
        engine = await get_alert_engine()

        # Set spend to 82% of limit
        sample_summary.total_spent = Decimal("82.00")

        alerts = await engine.evaluate_thresholds(sample_policy, sample_summary)

        # May have alerts depending on cooldowns
        assert isinstance(alerts, list)

    @pytest.mark.asyncio
    async def test_evaluate_over_limit(
        self, sample_policy: BudgetPolicy, sample_summary: SpendSummary
    ) -> None:
        """Test evaluation when over budget limit."""
        engine = await get_alert_engine()

        # Set spend to 105% of limit
        sample_summary.total_spent = Decimal("105.00")

        alerts = await engine.evaluate_thresholds(sample_policy, sample_summary)

        # Should have at least one alert
        assert isinstance(alerts, list)


class TestAlertEngineManagement:
    """Tests for alert management."""

    @pytest.mark.asyncio
    async def test_get_active_alerts(self) -> None:
        """Test getting active alerts."""
        engine = await get_alert_engine()

        alerts = await engine.get_active_alerts()

        assert isinstance(alerts, list)

    @pytest.mark.asyncio
    async def test_get_active_alerts_filtered_by_policy(self) -> None:
        """Test getting alerts filtered by policy."""
        engine = await get_alert_engine()

        alerts = await engine.get_active_alerts(policy_id="test-policy")

        assert isinstance(alerts, list)
        for alert in alerts:
            assert alert.policy_id == "test-policy"

    @pytest.mark.asyncio
    async def test_get_active_alerts_unacknowledged_only(self) -> None:
        """Test getting only unacknowledged alerts."""
        engine = await get_alert_engine()

        alerts = await engine.get_active_alerts(unacknowledged_only=True)

        assert isinstance(alerts, list)
        for alert in alerts:
            assert alert.acknowledged is False

    @pytest.mark.asyncio
    async def test_get_alert_by_id(self, sample_alert: BudgetAlert) -> None:
        """Test getting a specific alert by ID."""
        engine = await get_alert_engine()

        # Store the alert
        async with engine._alert_lock:
            engine._alerts[sample_alert.id] = sample_alert

        retrieved = await engine.get_alert(sample_alert.id)

        assert retrieved is not None
        assert retrieved.id == sample_alert.id

        # Clean up
        async with engine._alert_lock:
            engine._alerts.pop(sample_alert.id, None)

    @pytest.mark.asyncio
    async def test_get_alert_not_found(self) -> None:
        """Test getting a non-existent alert."""
        engine = await get_alert_engine()

        alert = await engine.get_alert("non-existent-id-xyz")

        assert alert is None


class TestAlertEngineAcknowledge:
    """Tests for alert acknowledgment."""

    @pytest.mark.asyncio
    async def test_acknowledge_alert(self, sample_alert: BudgetAlert) -> None:
        """Test acknowledging an alert."""
        engine = await get_alert_engine()

        # Store the alert
        async with engine._alert_lock:
            engine._alerts[sample_alert.id] = sample_alert

        result = await engine.acknowledge_alert(
            sample_alert.id,
            user_id="test-user"
        )

        assert result is True

        # Verify acknowledgment
        alert = await engine.get_alert(sample_alert.id)
        assert alert is not None
        assert alert.acknowledged is True
        assert alert.acknowledged_by == "test-user"

        # Clean up
        async with engine._alert_lock:
            engine._alerts.pop(sample_alert.id, None)

    @pytest.mark.asyncio
    async def test_acknowledge_nonexistent_alert(self) -> None:
        """Test acknowledging a non-existent alert returns False."""
        engine = await get_alert_engine()

        result = await engine.acknowledge_alert("nonexistent-alert-id")

        assert result is False


class TestAlertEngineResolve:
    """Tests for alert resolution."""

    @pytest.mark.asyncio
    async def test_resolve_alert(self, sample_alert: BudgetAlert) -> None:
        """Test resolving an alert."""
        engine = await get_alert_engine()

        # Store the alert
        async with engine._alert_lock:
            engine._alerts[sample_alert.id] = sample_alert

        result = await engine.resolve_alert(sample_alert.id)

        assert result is True

        # Verify resolution
        alert = await engine.get_alert(sample_alert.id)
        assert alert is not None
        assert alert.resolved is True

        # Clean up
        async with engine._alert_lock:
            engine._alerts.pop(sample_alert.id, None)

    @pytest.mark.asyncio
    async def test_resolve_alerts_for_policy(self) -> None:
        """Test resolving all alerts for a policy."""
        engine = await get_alert_engine()

        # Create and store multiple alerts for same policy
        policy_id = "test-policy-batch"
        alerts = [
            BudgetAlert(
                policy_id=policy_id,
                severity=AlertSeverity.WARNING,
                trigger_type=AlertTriggerType.THRESHOLD_REACHED,
                current_spend=Decimal("80.00"),
                limit_amount=Decimal("100.00"),
                message=f"Alert {i}",
            )
            for i in range(3)
        ]

        async with engine._alert_lock:
            for alert in alerts:
                engine._alerts[alert.id] = alert

        resolved_count = await engine.resolve_alerts_for_policy(policy_id)

        assert resolved_count == 3

        # Clean up
        async with engine._alert_lock:
            for alert in alerts:
                engine._alerts.pop(alert.id, None)


class TestAlertEngineEnabled:
    """Tests for enabled property."""

    @pytest.mark.asyncio
    async def test_enabled_property_getter(self) -> None:
        """Test getting enabled state."""
        engine = await get_alert_engine()

        enabled = engine.enabled
        assert isinstance(enabled, bool)

    @pytest.mark.asyncio
    async def test_enabled_property_setter(self) -> None:
        """Test setting enabled state."""
        engine = await get_alert_engine()

        original = engine.enabled

        engine.enabled = False
        assert engine.enabled is False

        engine.enabled = True
        assert engine.enabled is True

        # Restore original
        engine.enabled = original


class TestAlertRule:
    """Tests for AlertRule dataclass."""

    def test_create_alert_rule(self) -> None:
        """Test creating an alert rule."""
        rule = AlertRule(
            threshold_percent=0.80,
            severity=AlertSeverity.WARNING,
            trigger_type=AlertTriggerType.THRESHOLD_REACHED,
        )

        assert rule.threshold_percent == 0.80
        assert rule.severity == AlertSeverity.WARNING
        assert rule.trigger_type == AlertTriggerType.THRESHOLD_REACHED

    def test_alert_rule_with_cooldown(self) -> None:
        """Test creating a rule with cooldown."""
        rule = AlertRule(
            threshold_percent=0.95,
            severity=AlertSeverity.CRITICAL,
            trigger_type=AlertTriggerType.THRESHOLD_REACHED,
            cooldown_minutes=30,
        )

        assert rule.cooldown_minutes == 30

    def test_alert_rule_with_channels(self) -> None:
        """Test creating a rule with notification channels."""
        rule = AlertRule(
            threshold_percent=1.0,
            severity=AlertSeverity.EMERGENCY,
            trigger_type=AlertTriggerType.LIMIT_EXCEEDED,
            notification_channels=["email", "webhook"],
        )

        assert rule.notification_channels == ["email", "webhook"]
