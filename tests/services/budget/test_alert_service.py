"""
Unit tests for BudgetAlertService.

Tests cover:
- Alert configuration (create, update, delete)
- Active alert management
- Alert acknowledgment
- Threshold evaluation
- Event publishing

Author: ATLAS Team
Date: Jan 8, 2026
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional
from uuid import uuid4

import pytest
import pytest_asyncio

from core.services.budget.alert_service import BudgetAlertService
from core.services.budget.types import (
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
)
from modules.budget.models import (
    AlertSeverity,
    AlertTriggerType,
    BudgetAlert,
    BudgetPolicy,
    BudgetScope,
    BudgetPeriod,
    LimitAction,
    SpendSummary,
)


# =============================================================================
# Mock Actor
# =============================================================================


@dataclass
class MockActor:
    """Mock actor for testing."""
    
    user_id: str = "user_123"
    tenant_id: str = "tenant_abc"
    actor_type: str = "user"
    is_system: bool = False
    is_admin: bool = False
    
    @property
    def id(self) -> str:
        return self.user_id


# =============================================================================
# Mock Repositories
# =============================================================================


class MockAlertRepository:
    """Mock alert repository for testing."""
    
    def __init__(self):
        self.alerts: List[BudgetAlert] = []
        self.save_called = 0
        self.update_called = 0
    
    async def save_alert(self, alert: BudgetAlert) -> str:
        self.alerts.append(alert)
        self.save_called += 1
        return alert.id
    
    async def get_alert(self, alert_id: str) -> Optional[BudgetAlert]:
        for alert in self.alerts:
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
    ) -> List[BudgetAlert]:
        result = list(self.alerts)
        if active_only:
            result = [a for a in result if not a.resolved]
        if policy_id:
            result = [a for a in result if a.policy_id == policy_id]
        if severity:
            result = [a for a in result if a.severity.value == severity]
        return result[offset:offset + limit]
    
    async def update_alert(self, alert: BudgetAlert) -> bool:
        self.update_called += 1
        for i, a in enumerate(self.alerts):
            if a.id == alert.id:
                self.alerts[i] = alert
                return True
        return False
    
    async def delete_alert(self, alert_id: str) -> bool:
        for i, a in enumerate(self.alerts):
            if a.id == alert_id:
                del self.alerts[i]
                return True
        return False


class MockAlertConfigRepository:
    """Mock alert config repository for testing."""
    
    def __init__(self):
        self.configs: Dict[str, AlertConfig] = {}
    
    async def save_config(self, config: AlertConfig) -> str:
        self.configs[config.id] = config
        return config.id
    
    async def get_config(self, config_id: str) -> Optional[AlertConfig]:
        return self.configs.get(config_id)
    
    async def get_configs_for_policy(self, policy_id: str) -> List[AlertConfig]:
        return [c for c in self.configs.values() if c.policy_id == policy_id]
    
    async def delete_config(self, config_id: str) -> bool:
        if config_id in self.configs:
            del self.configs[config_id]
            return True
        return False


class MockPolicyRepository:
    """Mock policy repository for testing."""
    
    def __init__(self):
        self.policies: Dict[str, BudgetPolicy] = {}
    
    async def get_policy(self, policy_id: str) -> Optional[BudgetPolicy]:
        return self.policies.get(policy_id)
    
    async def get_policies(
        self,
        scope: Optional[BudgetScope] = None,
        scope_id: Optional[str] = None,
        enabled_only: bool = True,
    ) -> List[BudgetPolicy]:
        result = list(self.policies.values())
        if scope:
            result = [p for p in result if p.scope == scope]
        if scope_id:
            result = [p for p in result if p.scope_id == scope_id]
        if enabled_only:
            result = [p for p in result if p.enabled]
        return result
    
    def add_policy(self, policy: BudgetPolicy) -> None:
        self.policies[policy.id] = policy


class MockSpendingRepository:
    """Mock spending repository for testing."""
    
    def __init__(self):
        self.summaries: Dict[str, SpendSummary] = {}
    
    async def get_current_spend(
        self,
        scope: BudgetScope,
        scope_id: Optional[str],
        period: BudgetPeriod,
    ) -> SpendSummary:
        key = f"{scope.value}:{scope_id or 'all'}:{period.value}"
        if key in self.summaries:
            return self.summaries[key]
        
        # Return empty summary
        now = datetime.now(timezone.utc)
        return SpendSummary(
            policy_id="",
            period_start=now.replace(day=1),
            period_end=now,
            total_spent=Decimal("0"),
            limit_amount=Decimal("100"),
            currency="USD",
        )
    
    def set_spend(
        self,
        scope: BudgetScope,
        scope_id: Optional[str],
        period: BudgetPeriod,
        total_spent: Decimal,
        limit_amount: Decimal,
        policy_id: str = "",
    ) -> None:
        key = f"{scope.value}:{scope_id or 'all'}:{period.value}"
        now = datetime.now(timezone.utc)
        self.summaries[key] = SpendSummary(
            policy_id=policy_id,
            period_start=now.replace(day=1),
            period_end=now,
            total_spent=total_spent,
            limit_amount=limit_amount,
            currency="USD",
        )


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def actor() -> MockActor:
    return MockActor()


@pytest.fixture
def alert_repo() -> MockAlertRepository:
    return MockAlertRepository()


@pytest.fixture
def config_repo() -> MockAlertConfigRepository:
    return MockAlertConfigRepository()


@pytest.fixture
def policy_repo() -> MockPolicyRepository:
    return MockPolicyRepository()


@pytest.fixture
def spending_repo() -> MockSpendingRepository:
    return MockSpendingRepository()


@pytest.fixture
def alert_service(
    alert_repo: MockAlertRepository,
    config_repo: MockAlertConfigRepository,
    policy_repo: MockPolicyRepository,
    spending_repo: MockSpendingRepository,
) -> BudgetAlertService:
    return BudgetAlertService(
        alert_repository=alert_repo,
        config_repository=config_repo,
        policy_repository=policy_repo,
        spending_repository=spending_repo,
        evaluation_interval_seconds=0,  # Disable background task
    )


@pytest_asyncio.fixture
async def initialized_service(
    alert_service: BudgetAlertService,
):
    await alert_service.initialize()
    try:
        yield alert_service
    finally:
        await alert_service.shutdown()


# =============================================================================
# Alert Configuration Tests
# =============================================================================


class TestConfigureAlert:
    """Tests for alert configuration."""
    
    @pytest.mark.asyncio
    async def test_configure_alert_creates_config(
        self,
        initialized_service: BudgetAlertService,
        policy_repo: MockPolicyRepository,
        actor: MockActor,
    ):
        """Configure alert should create an alert config."""
        # Add a policy
        policy = BudgetPolicy(
            id="policy_123",
            name="Test Policy",
            scope=BudgetScope.TEAM,
            limit_amount=Decimal("100.00"),
        )
        policy_repo.add_policy(policy)
        
        config_data = AlertConfigCreate(
            policy_id="policy_123",
            threshold_percent=0.80,
            severity="warning",
        )
        
        config = await initialized_service.configure_alert(actor, config_data)
        
        assert config is not None
        assert config.policy_id == "policy_123"
        assert config.threshold_percent == 0.80
        assert config.severity == "warning"
        assert config.enabled is True
    
    @pytest.mark.asyncio
    async def test_configure_alert_with_channels(
        self,
        initialized_service: BudgetAlertService,
        policy_repo: MockPolicyRepository,
        actor: MockActor,
    ):
        """Configure alert should support notification channels."""
        policy = BudgetPolicy(
            id="policy_456",
            name="Test Policy",
            scope=BudgetScope.GLOBAL,
            limit_amount=Decimal("500.00"),
        )
        policy_repo.add_policy(policy)
        
        config_data = AlertConfigCreate(
            policy_id="policy_456",
            threshold_percent=0.90,
            severity="critical",
            notification_channels=["email", "slack"],
            cooldown_minutes=30,
        )
        
        config = await initialized_service.configure_alert(actor, config_data)
        
        assert config.notification_channels == ["email", "slack"]
        assert config.cooldown_minutes == 30
    
    @pytest.mark.asyncio
    async def test_update_alert_config(
        self,
        initialized_service: BudgetAlertService,
        policy_repo: MockPolicyRepository,
        actor: MockActor,
    ):
        """Update alert config should modify existing config."""
        policy = BudgetPolicy(
            id="policy_789",
            name="Test Policy",
            scope=BudgetScope.GLOBAL,
            limit_amount=Decimal("200.00"),
        )
        policy_repo.add_policy(policy)
        
        # Create initial config
        config_data = AlertConfigCreate(
            policy_id="policy_789",
            threshold_percent=0.75,
            severity="info",
        )
        config = await initialized_service.configure_alert(actor, config_data)
        
        # Update it
        updates = AlertConfigUpdate(
            threshold_percent=0.85,
            severity="warning",
        )
        updated = await initialized_service.update_alert_config(actor, config.id, updates)
        
        assert updated is not None
        assert updated.threshold_percent == 0.85
        assert updated.severity == "warning"
    
    @pytest.mark.asyncio
    async def test_remove_alert_config(
        self,
        initialized_service: BudgetAlertService,
        policy_repo: MockPolicyRepository,
        actor: MockActor,
    ):
        """Remove alert config should delete the configuration."""
        policy = BudgetPolicy(
            id="policy_del",
            name="Delete Test",
            scope=BudgetScope.GLOBAL,
            limit_amount=Decimal("100.00"),
        )
        policy_repo.add_policy(policy)
        
        config_data = AlertConfigCreate(
            policy_id="policy_del",
            threshold_percent=0.50,
        )
        config = await initialized_service.configure_alert(actor, config_data)
        
        result = await initialized_service.remove_alert_config(actor, config.id)
        
        assert result is True
        
        # Should not be in the list anymore
        configs = await initialized_service.list_alert_configs(actor, "policy_del")
        assert len(configs) == 0


class TestListAlertConfigs:
    """Tests for listing alert configurations."""
    
    @pytest.mark.asyncio
    async def test_list_all_configs(
        self,
        initialized_service: BudgetAlertService,
        policy_repo: MockPolicyRepository,
        actor: MockActor,
    ):
        """List configs should return all configurations."""
        policy = BudgetPolicy(
            id="policy_list",
            name="List Test",
            scope=BudgetScope.GLOBAL,
            limit_amount=Decimal("100.00"),
        )
        policy_repo.add_policy(policy)
        
        # Create multiple configs
        for threshold in [0.50, 0.75, 0.90]:
            config_data = AlertConfigCreate(
                policy_id="policy_list",
                threshold_percent=threshold,
            )
            await initialized_service.configure_alert(actor, config_data)
        
        configs = await initialized_service.list_alert_configs(actor, "policy_list")
        
        assert len(configs) == 3


# =============================================================================
# Active Alert Tests
# =============================================================================


class TestGetActiveAlerts:
    """Tests for getting active alerts."""
    
    @pytest.mark.asyncio
    async def test_get_active_alerts_empty(
        self,
        initialized_service: BudgetAlertService,
        actor: MockActor,
    ):
        """Get active alerts should return empty list when no alerts."""
        alerts = await initialized_service.get_active_alerts(actor)
        
        assert alerts == []
    
    @pytest.mark.asyncio
    async def test_get_active_alerts_with_filters(
        self,
        initialized_service: BudgetAlertService,
        alert_repo: MockAlertRepository,
        actor: MockActor,
    ):
        """Get active alerts should apply filters."""
        # Add some alerts directly
        alert1 = BudgetAlert(
            policy_id="policy_1",
            severity=AlertSeverity.WARNING,
            trigger_type=AlertTriggerType.THRESHOLD_REACHED,
            current_spend=Decimal("80"),
            limit_amount=Decimal("100"),
            message="Warning alert",
        )
        alert2 = BudgetAlert(
            policy_id="policy_2",
            severity=AlertSeverity.CRITICAL,
            trigger_type=AlertTriggerType.LIMIT_EXCEEDED,
            current_spend=Decimal("110"),
            limit_amount=Decimal("100"),
            message="Critical alert",
        )
        
        initialized_service._alerts = [alert1, alert2]
        
        # Filter by severity
        request = AlertListRequest(severity="critical")
        alerts = await initialized_service.get_active_alerts(actor, request)
        
        assert len(alerts) == 1
        assert alerts[0].severity == "critical"
    
    @pytest.mark.asyncio
    async def test_get_active_alerts_sorted_by_severity(
        self,
        initialized_service: BudgetAlertService,
        actor: MockActor,
    ):
        """Active alerts should be sorted by severity (critical first)."""
        alert_info = BudgetAlert(
            policy_id="p1",
            severity=AlertSeverity.INFO,
            trigger_type=AlertTriggerType.THRESHOLD_REACHED,
            current_spend=Decimal("50"),
            limit_amount=Decimal("100"),
            message="Info",
        )
        alert_critical = BudgetAlert(
            policy_id="p2",
            severity=AlertSeverity.CRITICAL,
            trigger_type=AlertTriggerType.LIMIT_EXCEEDED,
            current_spend=Decimal("110"),
            limit_amount=Decimal("100"),
            message="Critical",
        )
        alert_warning = BudgetAlert(
            policy_id="p3",
            severity=AlertSeverity.WARNING,
            trigger_type=AlertTriggerType.THRESHOLD_REACHED,
            current_spend=Decimal("85"),
            limit_amount=Decimal("100"),
            message="Warning",
        )
        
        initialized_service._alerts = [alert_info, alert_critical, alert_warning]
        
        alerts = await initialized_service.get_active_alerts(actor)
        
        assert len(alerts) == 3
        assert alerts[0].severity == "critical"
        assert alerts[1].severity == "warning"
        assert alerts[2].severity == "info"


# =============================================================================
# Acknowledge Alert Tests
# =============================================================================


class TestAcknowledgeAlert:
    """Tests for acknowledging alerts."""
    
    @pytest.mark.asyncio
    async def test_acknowledge_alert_success(
        self,
        initialized_service: BudgetAlertService,
        alert_repo: MockAlertRepository,
        actor: MockActor,
    ):
        """Acknowledge alert should mark alert as acknowledged."""
        alert = BudgetAlert(
            policy_id="policy_ack",
            severity=AlertSeverity.WARNING,
            trigger_type=AlertTriggerType.THRESHOLD_REACHED,
            current_spend=Decimal("80"),
            limit_amount=Decimal("100"),
            message="Test alert",
        )
        alert_id = alert.id
        initialized_service._alerts = [alert]
        
        result = await initialized_service.acknowledge_alert(actor, alert_id)
        
        assert result is True
        assert alert.acknowledged is True
        assert alert.acknowledged_by == actor.user_id
    
    @pytest.mark.asyncio
    async def test_acknowledge_nonexistent_alert(
        self,
        initialized_service: BudgetAlertService,
        actor: MockActor,
    ):
        """Acknowledge should return False for nonexistent alert."""
        result = await initialized_service.acknowledge_alert(actor, "nonexistent_id")
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_acknowledge_publishes_event(
        self,
        alert_repo: MockAlertRepository,
        config_repo: MockAlertConfigRepository,
        policy_repo: MockPolicyRepository,
        spending_repo: MockSpendingRepository,
        actor: MockActor,
    ):
        """Acknowledge should publish an event."""
        events: List[Any] = []
        
        async def capture_event(event):
            events.append(event)
        
        service = BudgetAlertService(
            alert_repository=alert_repo,
            config_repository=config_repo,
            policy_repository=policy_repo,
            spending_repository=spending_repo,
            event_publisher=capture_event,
            evaluation_interval_seconds=0,
        )
        await service.initialize()
        
        alert = BudgetAlert(
            policy_id="policy_event",
            severity=AlertSeverity.WARNING,
            trigger_type=AlertTriggerType.THRESHOLD_REACHED,
            current_spend=Decimal("80"),
            limit_amount=Decimal("100"),
            message="Event test",
        )
        service._alerts = [alert]
        
        await service.acknowledge_alert(actor, alert.id)
        await service.shutdown()
        
        ack_events = [e for e in events if isinstance(e, BudgetAlertAcknowledged)]
        assert len(ack_events) == 1
        assert ack_events[0].alert_id == alert.id


# =============================================================================
# Resolve Alert Tests
# =============================================================================


class TestResolveAlert:
    """Tests for resolving alerts."""
    
    @pytest.mark.asyncio
    async def test_resolve_alert_success(
        self,
        initialized_service: BudgetAlertService,
        actor: MockActor,
    ):
        """Resolve alert should mark alert as resolved."""
        alert = BudgetAlert(
            policy_id="policy_resolve",
            severity=AlertSeverity.CRITICAL,
            trigger_type=AlertTriggerType.LIMIT_EXCEEDED,
            current_spend=Decimal("110"),
            limit_amount=Decimal("100"),
            message="Resolve test",
        )
        initialized_service._alerts = [alert]
        
        result = await initialized_service.resolve_alert(actor, alert.id)
        
        assert result is True
        assert alert.resolved is True
    
    @pytest.mark.asyncio
    async def test_resolved_alerts_not_in_active(
        self,
        initialized_service: BudgetAlertService,
        actor: MockActor,
    ):
        """Resolved alerts should not appear in active alerts."""
        alert = BudgetAlert(
            policy_id="policy_res",
            severity=AlertSeverity.WARNING,
            trigger_type=AlertTriggerType.THRESHOLD_REACHED,
            current_spend=Decimal("80"),
            limit_amount=Decimal("100"),
            message="Test",
        )
        alert.resolve()
        initialized_service._alerts = [alert]
        
        alerts = await initialized_service.get_active_alerts(actor)
        
        assert len(alerts) == 0


# =============================================================================
# Evaluate Alerts Tests
# =============================================================================


class TestEvaluateAlerts:
    """Tests for alert evaluation."""
    
    @pytest.mark.asyncio
    async def test_evaluate_creates_alert_over_threshold(
        self,
        initialized_service: BudgetAlertService,
        policy_repo: MockPolicyRepository,
        spending_repo: MockSpendingRepository,
    ):
        """Evaluate should create alert when threshold is crossed."""
        # Add a policy
        policy = BudgetPolicy(
            id="eval_policy",
            name="Evaluation Test",
            scope=BudgetScope.GLOBAL,
            limit_amount=Decimal("100.00"),
            soft_limit_percent=0.80,
        )
        policy_repo.add_policy(policy)
        
        # Set spending at 85%
        spending_repo.set_spend(
            scope=BudgetScope.GLOBAL,
            scope_id=None,
            period=BudgetPeriod.MONTHLY,
            total_spent=Decimal("85.00"),
            limit_amount=Decimal("100.00"),
            policy_id="eval_policy",
        )
        
        # Evaluate
        alerts_triggered = await initialized_service.evaluate_alerts()
        
        assert alerts_triggered >= 1
        assert len(initialized_service._alerts) >= 1
    
    @pytest.mark.asyncio
    async def test_evaluate_creates_critical_when_exceeded(
        self,
        initialized_service: BudgetAlertService,
        policy_repo: MockPolicyRepository,
        spending_repo: MockSpendingRepository,
    ):
        """Evaluate should create critical alert when limit is exceeded."""
        policy = BudgetPolicy(
            id="exceed_policy",
            name="Exceed Test",
            scope=BudgetScope.GLOBAL,
            limit_amount=Decimal("100.00"),
        )
        policy_repo.add_policy(policy)
        
        # Set spending at 105%
        spending_repo.set_spend(
            scope=BudgetScope.GLOBAL,
            scope_id=None,
            period=BudgetPeriod.MONTHLY,
            total_spent=Decimal("105.00"),
            limit_amount=Decimal("100.00"),
            policy_id="exceed_policy",
        )
        
        await initialized_service.evaluate_alerts()
        
        # Should have critical alert
        critical_alerts = [
            a for a in initialized_service._alerts
            if a.severity == AlertSeverity.CRITICAL
        ]
        assert len(critical_alerts) >= 1
    
    @pytest.mark.asyncio
    async def test_evaluate_no_duplicate_alerts(
        self,
        initialized_service: BudgetAlertService,
        policy_repo: MockPolicyRepository,
        spending_repo: MockSpendingRepository,
    ):
        """Evaluate should not create duplicate alerts for same threshold."""
        policy = BudgetPolicy(
            id="dup_policy",
            name="Duplicate Test",
            scope=BudgetScope.GLOBAL,
            limit_amount=Decimal("100.00"),
        )
        policy_repo.add_policy(policy)
        
        spending_repo.set_spend(
            scope=BudgetScope.GLOBAL,
            scope_id=None,
            period=BudgetPeriod.MONTHLY,
            total_spent=Decimal("90.00"),
            limit_amount=Decimal("100.00"),
            policy_id="dup_policy",
        )
        
        # Evaluate twice
        await initialized_service.evaluate_alerts()
        first_count = len(initialized_service._alerts)
        
        await initialized_service.evaluate_alerts()
        second_count = len(initialized_service._alerts)
        
        assert second_count == first_count


# =============================================================================
# Event Publishing Tests
# =============================================================================


class TestEventPublishing:
    """Tests for domain event publishing."""
    
    @pytest.mark.asyncio
    async def test_alert_triggered_event_published(
        self,
        alert_repo: MockAlertRepository,
        policy_repo: MockPolicyRepository,
        spending_repo: MockSpendingRepository,
    ):
        """Alert triggered event should be published."""
        events: List[Any] = []
        
        async def capture_event(event):
            events.append(event)
        
        service = BudgetAlertService(
            alert_repository=alert_repo,
            policy_repository=policy_repo,
            spending_repository=spending_repo,
            event_publisher=capture_event,
            evaluation_interval_seconds=0,
        )
        await service.initialize()
        
        policy = BudgetPolicy(
            id="event_policy",
            name="Event Policy",
            scope=BudgetScope.GLOBAL,
            limit_amount=Decimal("100.00"),
        )
        policy_repo.add_policy(policy)
        
        spending_repo.set_spend(
            scope=BudgetScope.GLOBAL,
            scope_id=None,
            period=BudgetPeriod.MONTHLY,
            total_spent=Decimal("85.00"),
            limit_amount=Decimal("100.00"),
            policy_id="event_policy",
        )
        
        await service.evaluate_alerts()
        await service.shutdown()
        
        triggered_events = [e for e in events if isinstance(e, BudgetAlertTriggered)]
        assert len(triggered_events) >= 1
    
    @pytest.mark.asyncio
    async def test_limit_exceeded_event_published(
        self,
        alert_repo: MockAlertRepository,
        policy_repo: MockPolicyRepository,
        spending_repo: MockSpendingRepository,
    ):
        """Limit exceeded event should be published when over budget."""
        events: List[Any] = []
        
        async def capture_event(event):
            events.append(event)
        
        service = BudgetAlertService(
            alert_repository=alert_repo,
            policy_repository=policy_repo,
            spending_repository=spending_repo,
            event_publisher=capture_event,
            evaluation_interval_seconds=0,
        )
        await service.initialize()
        
        policy = BudgetPolicy(
            id="exceeded_policy",
            name="Exceeded Policy",
            scope=BudgetScope.GLOBAL,
            limit_amount=Decimal("100.00"),
        )
        policy_repo.add_policy(policy)
        
        spending_repo.set_spend(
            scope=BudgetScope.GLOBAL,
            scope_id=None,
            period=BudgetPeriod.MONTHLY,
            total_spent=Decimal("120.00"),
            limit_amount=Decimal("100.00"),
            policy_id="exceeded_policy",
        )
        
        await service.evaluate_alerts()
        await service.shutdown()
        
        exceeded_events = [e for e in events if isinstance(e, BudgetLimitExceeded)]
        assert len(exceeded_events) >= 1
        assert exceeded_events[0].overage_amount == Decimal("20.00")


# =============================================================================
# Service Lifecycle Tests
# =============================================================================


class TestServiceLifecycle:
    """Tests for service initialization and shutdown."""
    
    @pytest.mark.asyncio
    async def test_initialize_starts_service(
        self,
        alert_service: BudgetAlertService,
    ):
        """Initialize should start the service."""
        assert not alert_service._initialized
        
        await alert_service.initialize()
        
        assert alert_service._initialized
        
        await alert_service.shutdown()
    
    @pytest.mark.asyncio
    async def test_double_initialize_is_safe(
        self,
        alert_service: BudgetAlertService,
    ):
        """Multiple initialize calls should be safe."""
        await alert_service.initialize()
        await alert_service.initialize()  # Should not error
        
        assert alert_service._initialized
        
        await alert_service.shutdown()
    
    @pytest.mark.asyncio
    async def test_double_shutdown_is_safe(
        self,
        alert_service: BudgetAlertService,
    ):
        """Multiple shutdown calls should be safe."""
        await alert_service.initialize()
        await alert_service.shutdown()
        await alert_service.shutdown()  # Should not error
        
        assert alert_service._shutting_down
    
    @pytest.mark.asyncio
    async def test_set_enabled(
        self,
        initialized_service: BudgetAlertService,
    ):
        """Set enabled should control alert evaluation."""
        assert initialized_service._enabled is True
        
        initialized_service.set_enabled(False)
        
        assert initialized_service._enabled is False
    
    @pytest.mark.asyncio
    async def test_get_alert_count(
        self,
        initialized_service: BudgetAlertService,
    ):
        """Get alert count should return correct count."""
        alert1 = BudgetAlert(
            policy_id="p1",
            severity=AlertSeverity.WARNING,
            trigger_type=AlertTriggerType.THRESHOLD_REACHED,
            current_spend=Decimal("80"),
            limit_amount=Decimal("100"),
            message="Alert 1",
        )
        alert2 = BudgetAlert(
            policy_id="p2",
            severity=AlertSeverity.INFO,
            trigger_type=AlertTriggerType.THRESHOLD_REACHED,
            current_spend=Decimal("50"),
            limit_amount=Decimal("100"),
            message="Alert 2",
        )
        alert2.resolve()
        
        initialized_service._alerts = [alert1, alert2]
        
        assert initialized_service.get_alert_count(active_only=True) == 1
        assert initialized_service.get_alert_count(active_only=False) == 2
