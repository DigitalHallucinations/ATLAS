"""Tests for BudgetManager orchestrator."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from modules.budget.models import (
    BudgetCheckResult,
    BudgetPeriod,
    BudgetPolicy,
    BudgetScope,
    LimitAction,
    OperationType,
    UsageRecord,
)
from modules.budget.manager import BudgetManager


@pytest.fixture
def mock_config_manager() -> MagicMock:
    """Create a mock ConfigManager."""
    config = MagicMock()
    config.get_config.return_value = {
        "enabled": True,
        "default_limit": 100.0,
        "default_period": "monthly",
        "alert_thresholds": [0.5, 0.8, 0.95],
    }
    return config


@pytest.fixture
def sample_policy() -> BudgetPolicy:
    """Create a sample budget policy."""
    return BudgetPolicy(
        name="Test Policy",
        scope=BudgetScope.GLOBAL,
        limit_amount=Decimal("100.00"),
        period=BudgetPeriod.MONTHLY,
        soft_limit_percent=0.80,
        hard_limit_action=LimitAction.WARN,
    )


@pytest.fixture
def sample_usage() -> UsageRecord:
    """Create a sample usage record."""
    return UsageRecord(
        provider="openai",
        model="gpt-4o",
        operation_type=OperationType.CHAT_COMPLETION,
        input_tokens=1000,
        output_tokens=500,
        cost_usd=Decimal("0.0075"),
    )


class TestBudgetManagerCreation:
    """Tests for BudgetManager creation."""

    def test_create_budget_manager(self, mock_config_manager: MagicMock) -> None:
        """Test creating a BudgetManager instance."""
        manager = BudgetManager(mock_config_manager)

        assert manager is not None
        assert manager.config_manager is mock_config_manager

    def test_manager_initial_state(self, mock_config_manager: MagicMock) -> None:
        """Test manager initial state."""
        manager = BudgetManager(mock_config_manager)

        # Should have empty policies initially
        assert hasattr(manager, "_policies")


class TestBudgetManagerPolicies:
    """Tests for policy management."""

    @pytest.mark.asyncio
    async def test_set_policy(
        self,
        mock_config_manager: MagicMock,
        sample_policy: BudgetPolicy
    ) -> None:
        """Test setting a budget policy."""
        manager = BudgetManager(mock_config_manager)
        await manager._initialize()

        result = await manager.set_budget_policy(sample_policy)

        assert result is not None
        assert result.id == sample_policy.id

    @pytest.mark.asyncio
    async def test_get_policy_by_id(
        self,
        mock_config_manager: MagicMock,
        sample_policy: BudgetPolicy
    ) -> None:
        """Test retrieving a policy by ID."""
        manager = BudgetManager(mock_config_manager)
        await manager._initialize()
        await manager.set_budget_policy(sample_policy)

        retrieved = await manager.get_policy(sample_policy.id)

        assert retrieved is not None
        assert retrieved.id == sample_policy.id
        assert retrieved.name == sample_policy.name

    @pytest.mark.asyncio
    async def test_get_policy_not_found(
        self,
        mock_config_manager: MagicMock
    ) -> None:
        """Test getting a non-existent policy returns None."""
        manager = BudgetManager(mock_config_manager)
        await manager._initialize()

        retrieved = await manager.get_policy("non-existent-id")

        assert retrieved is None

    @pytest.mark.asyncio
    async def test_delete_policy(
        self,
        mock_config_manager: MagicMock,
        sample_policy: BudgetPolicy
    ) -> None:
        """Test deleting a policy."""
        manager = BudgetManager(mock_config_manager)
        await manager._initialize()
        await manager.set_budget_policy(sample_policy)

        result = await manager.delete_policy(sample_policy.id)

        assert result is True

        retrieved = await manager.get_policy(sample_policy.id)
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_update_policy(
        self,
        mock_config_manager: MagicMock,
        sample_policy: BudgetPolicy
    ) -> None:
        """Test updating an existing policy."""
        manager = BudgetManager(mock_config_manager)
        await manager._initialize()
        await manager.set_budget_policy(sample_policy)

        # Update the policy
        sample_policy.limit_amount = Decimal("200.00")
        await manager.set_budget_policy(sample_policy)

        retrieved = await manager.get_policy(sample_policy.id)
        assert retrieved is not None
        assert retrieved.limit_amount == Decimal("200.000000")


class TestBudgetManagerUsageRecording:
    """Tests for usage recording."""

    @pytest.mark.asyncio
    async def test_record_usage(
        self,
        mock_config_manager: MagicMock,
        sample_usage: UsageRecord
    ) -> None:
        """Test recording a usage event."""
        manager = BudgetManager(mock_config_manager)
        await manager._initialize()

        await manager.record_usage(sample_usage)

        # Verify usage was recorded
        assert len(manager._usage_records) > 0

    @pytest.mark.asyncio
    async def test_record_multiple_usages(
        self,
        mock_config_manager: MagicMock
    ) -> None:
        """Test recording multiple usage events."""
        manager = BudgetManager(mock_config_manager)
        await manager._initialize()

        usages = [
            UsageRecord(
                provider="openai",
                model="gpt-4o",
                operation_type=OperationType.CHAT_COMPLETION,
                cost_usd=Decimal("0.01"),
            ),
            UsageRecord(
                provider="anthropic",
                model="claude-3-sonnet",
                operation_type=OperationType.CHAT_COMPLETION,
                cost_usd=Decimal("0.015"),
            ),
        ]

        for usage in usages:
            await manager.record_usage(usage)

        assert len(manager._usage_records) >= 2


class TestBudgetManagerLifecycle:
    """Tests for manager lifecycle."""

    @pytest.mark.asyncio
    async def test_initialize(self, mock_config_manager: MagicMock) -> None:
        """Test manager initialization."""
        manager = BudgetManager(mock_config_manager)

        # Should complete without error
        await manager._initialize()

        # Verify pricing was loaded
        assert manager._pricing is not None

    @pytest.mark.asyncio
    async def test_shutdown(self, mock_config_manager: MagicMock) -> None:
        """Test manager shutdown."""
        manager = BudgetManager(mock_config_manager)
        await manager._initialize()

        # Shutdown should complete without error
        await manager.shutdown()


class TestBudgetManagerEnabled:
    """Tests for enabled property."""

    def test_enabled_property(self, mock_config_manager: MagicMock) -> None:
        """Test getting enabled state."""
        manager = BudgetManager(mock_config_manager)

        # Should default to True
        assert isinstance(manager.enabled, bool)

    def test_enabled_setter(self, mock_config_manager: MagicMock) -> None:
        """Test setting enabled state."""
        manager = BudgetManager(mock_config_manager)

        manager.enabled = False
        assert manager.enabled is False

        manager.enabled = True
        assert manager.enabled is True


class TestBudgetManagerRollover:
    """Tests for rollover functionality."""

    @pytest.fixture
    def rollover_policy(self) -> BudgetPolicy:
        """Create a policy with rollover enabled."""
        return BudgetPolicy(
            name="Rollover Policy",
            scope=BudgetScope.GLOBAL,
            limit_amount=Decimal("100.00"),
            period=BudgetPeriod.MONTHLY,
            rollover_enabled=True,
            rollover_max_percent=0.25,
        )

    @pytest.mark.asyncio
    async def test_calculate_rollover_disabled(
        self,
        mock_config_manager: MagicMock,
        sample_policy: BudgetPolicy
    ) -> None:
        """Test rollover returns 0 when disabled."""
        manager = BudgetManager(mock_config_manager)
        await manager._initialize()

        # sample_policy has rollover_enabled=False by default
        rollover = await manager.calculate_rollover(sample_policy)

        assert rollover == Decimal("0")

    @pytest.mark.asyncio
    async def test_calculate_rollover_enabled(
        self,
        mock_config_manager: MagicMock,
        rollover_policy: BudgetPolicy
    ) -> None:
        """Test rollover calculation with enabled policy."""
        manager = BudgetManager(mock_config_manager)
        await manager._initialize()

        # With no previous period spending, rollover should be max (25% of $100 = $25)
        rollover = await manager.calculate_rollover(rollover_policy)

        # Since there's no spending, unused = $100, capped at 25% = $25
        assert rollover == Decimal("25.00")

    @pytest.mark.asyncio
    async def test_rollover_capped_at_max_percent(
        self,
        mock_config_manager: MagicMock
    ) -> None:
        """Test that rollover is capped at max percent."""
        policy = BudgetPolicy(
            name="Low Cap Policy",
            scope=BudgetScope.GLOBAL,
            limit_amount=Decimal("100.00"),
            period=BudgetPeriod.MONTHLY,
            rollover_enabled=True,
            rollover_max_percent=0.10,  # Only 10%
        )

        manager = BudgetManager(mock_config_manager)
        await manager._initialize()

        rollover = await manager.calculate_rollover(policy)

        # Max rollover is 10% of $100 = $10
        assert rollover == Decimal("10.00")

    @pytest.mark.asyncio
    async def test_get_rollover_amount(
        self,
        mock_config_manager: MagicMock,
        rollover_policy: BudgetPolicy
    ) -> None:
        """Test getting rollover amount for a policy."""
        manager = BudgetManager(mock_config_manager)
        await manager._initialize()

        # Initially should be 0
        amount = await manager.get_rollover_amount(rollover_policy.id)
        assert amount == Decimal("0")

        # After processing period end, should have rollover
        await manager.set_budget_policy(rollover_policy)
        await manager.process_period_end(rollover_policy)

        amount = await manager.get_rollover_amount(rollover_policy.id)
        assert amount > Decimal("0")

    @pytest.mark.asyncio
    async def test_process_period_end(
        self,
        mock_config_manager: MagicMock,
        rollover_policy: BudgetPolicy
    ) -> None:
        """Test processing period end."""
        manager = BudgetManager(mock_config_manager)
        await manager._initialize()
        await manager.set_budget_policy(rollover_policy)

        rollover = await manager.process_period_end(rollover_policy)

        # Should return rollover amount
        assert rollover >= Decimal("0")
        # Should be stored
        stored = await manager.get_rollover_amount(rollover_policy.id)
        assert stored == rollover

    @pytest.mark.asyncio
    async def test_rollover_applied_to_effective_limit(
        self,
        mock_config_manager: MagicMock,
        rollover_policy: BudgetPolicy
    ) -> None:
        """Test that rollover is applied to effective limit in summaries."""
        manager = BudgetManager(mock_config_manager)
        await manager._initialize()
        await manager.set_budget_policy(rollover_policy)

        # Process period end to set rollover
        await manager.process_period_end(rollover_policy)
        rollover_amount = await manager.get_rollover_amount(rollover_policy.id)

        # Get spend summary
        summary = await manager.get_current_spend(
            scope=BudgetScope.GLOBAL,
            period=BudgetPeriod.MONTHLY,
        )

        # Effective limit should include rollover
        expected_effective = rollover_policy.limit_amount + rollover_amount
        assert summary.effective_limit == expected_effective

    @pytest.mark.asyncio
    async def test_rollover_not_applied_to_rolling_periods(
        self,
        mock_config_manager: MagicMock
    ) -> None:
        """Test that rollover is not applied to rolling period types."""
        policy = BudgetPolicy(
            name="Rolling Policy",
            scope=BudgetScope.GLOBAL,
            limit_amount=Decimal("100.00"),
            period=BudgetPeriod.ROLLING_30D,
            rollover_enabled=True,  # Even if enabled
            rollover_max_percent=0.25,
        )

        manager = BudgetManager(mock_config_manager)
        await manager._initialize()

        rollover = await manager.calculate_rollover(policy)

        # Rolling periods don't support rollover
        assert rollover == Decimal("0")
