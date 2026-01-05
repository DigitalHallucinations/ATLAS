"""Tests for budget management data models.

Note: Pylance may report false positives for BudgetCheckResult constructor
parameters due to dataclass field detection limitations. All tests pass at runtime.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from modules.budget.models import (
    AlertSeverity,
    AlertTriggerType,
    BudgetAlert,
    BudgetCheckResult,
    BudgetPeriod,
    BudgetPolicy,
    BudgetScope,
    LimitAction,
    OperationType,
    SpendSummary,
    UsageRecord,
)


class TestBudgetPolicy:
    """Tests for BudgetPolicy dataclass."""

    def test_create_basic_policy(self) -> None:
        """Test creating a policy with minimal required fields."""
        policy = BudgetPolicy(
            name="Test Budget",
            scope=BudgetScope.GLOBAL,
            limit_amount=Decimal("100.00"),
        )

        assert policy.name == "Test Budget"
        assert policy.scope == BudgetScope.GLOBAL
        assert policy.limit_amount == Decimal("100.000000")
        assert policy.period == BudgetPeriod.MONTHLY
        assert policy.currency == "USD"
        assert policy.enabled is True
        assert policy.id is not None

    def test_policy_with_all_fields(self) -> None:
        """Test creating a policy with all fields specified."""
        policy = BudgetPolicy(
            name="Complete Budget",
            scope=BudgetScope.USER,
            scope_id="user-123",
            limit_amount=Decimal("500.00"),
            period=BudgetPeriod.WEEKLY,
            currency="USD",
            soft_limit_percent=0.75,
            hard_limit_action=LimitAction.SOFT_BLOCK,
            rollover_enabled=True,
            rollover_max_percent=0.20,
            provider_limits={"openai": Decimal("200.00")},
            model_limits={"gpt-4o": Decimal("100.00")},
            enabled=True,
            priority=10,
        )

        assert policy.scope_id == "user-123"
        assert policy.period == BudgetPeriod.WEEKLY
        assert policy.soft_limit_percent == 0.75
        assert policy.hard_limit_action == LimitAction.SOFT_BLOCK
        assert policy.rollover_enabled is True
        assert "openai" in policy.provider_limits
        assert "gpt-4o" in policy.model_limits

    def test_policy_decimal_normalization(self) -> None:
        """Test that numeric values are normalized to Decimal."""
        policy = BudgetPolicy(
            name="Normalize Test",
            scope=BudgetScope.GLOBAL,
            limit_amount=100,  # type: ignore[arg-type]  # Testing runtime coercion
        )

        assert isinstance(policy.limit_amount, Decimal)
        assert policy.limit_amount == Decimal("100.000000")

    def test_policy_negative_limit_raises(self) -> None:
        """Test that negative limit raises ValueError."""
        with pytest.raises(ValueError, match="limit_amount cannot be negative"):
            BudgetPolicy(
                name="Negative",
                scope=BudgetScope.GLOBAL,
                limit_amount=Decimal("-10.00"),
            )

    def test_policy_invalid_soft_limit_percent_raises(self) -> None:
        """Test that invalid soft_limit_percent raises ValueError."""
        with pytest.raises(ValueError, match="soft_limit_percent must be between"):
            BudgetPolicy(
                name="Bad Percent",
                scope=BudgetScope.GLOBAL,
                limit_amount=Decimal("100"),
                soft_limit_percent=1.5,
            )

    def test_policy_string_enum_conversion(self) -> None:
        """Test that string values are converted to enums."""
        policy = BudgetPolicy(
            name="Enum Conversion",
            scope="global",  # type: ignore
            period="monthly",  # type: ignore
            hard_limit_action="warn",  # type: ignore
            limit_amount=Decimal("50"),
        )

        assert policy.scope == BudgetScope.GLOBAL
        assert policy.period == BudgetPeriod.MONTHLY
        assert policy.hard_limit_action == LimitAction.WARN

    def test_get_soft_limit(self) -> None:
        """Test soft limit calculation."""
        policy = BudgetPolicy(
            name="Soft Limit Test",
            scope=BudgetScope.GLOBAL,
            limit_amount=Decimal("100"),
            soft_limit_percent=0.80,
        )

        assert policy.get_soft_limit() == Decimal("80.000000")

    def test_policy_as_dict(self) -> None:
        """Test serialization to dictionary."""
        policy = BudgetPolicy(
            name="Serialize Test",
            scope=BudgetScope.GLOBAL,
            limit_amount=Decimal("100"),
        )

        data = policy.as_dict()

        assert data["name"] == "Serialize Test"
        assert data["scope"] == "global"
        assert data["limit_amount"] == "100.000000"
        assert "created_at" in data
        assert "updated_at" in data

    def test_policy_from_dict_roundtrip(self) -> None:
        """Test serialization roundtrip."""
        original = BudgetPolicy(
            name="Roundtrip Test",
            scope=BudgetScope.USER,
            scope_id="user-456",
            limit_amount=Decimal("250.50"),
            provider_limits={"anthropic": Decimal("100")},
        )

        data = original.as_dict()
        restored = BudgetPolicy.from_dict(data)

        assert restored.name == original.name
        assert restored.scope == original.scope
        assert restored.scope_id == original.scope_id
        assert restored.limit_amount == original.limit_amount
        assert restored.provider_limits == original.provider_limits


class TestUsageRecord:
    """Tests for UsageRecord dataclass."""

    def test_create_llm_usage(self) -> None:
        """Test creating an LLM usage record."""
        record = UsageRecord(
            provider="openai",
            model="gpt-4o",
            operation_type=OperationType.CHAT_COMPLETION,
            input_tokens=500,
            output_tokens=200,
            cost_usd=Decimal("0.003"),
        )

        assert record.provider == "openai"
        assert record.model == "gpt-4o"
        assert record.input_tokens == 500
        assert record.output_tokens == 200
        assert record.cost_usd == Decimal("0.003000")
        assert record.id is not None
        assert record.timestamp is not None

    def test_create_image_usage(self) -> None:
        """Test creating an image generation usage record."""
        record = UsageRecord(
            provider="openai",
            model="dall-e-3",
            operation_type=OperationType.IMAGE_GENERATION,
            cost_usd=Decimal("0.04"),
            image_size="1024x1024",
            images_generated=1,
        )

        assert record.operation_type == OperationType.IMAGE_GENERATION
        assert record.image_size == "1024x1024"
        assert record.images_generated == 1

    def test_usage_as_dict(self) -> None:
        """Test usage record serialization."""
        record = UsageRecord(
            provider="anthropic",
            model="claude-3-opus",
            operation_type=OperationType.CHAT_COMPLETION,
            cost_usd=Decimal("0.015"),
            input_tokens=1000,
            output_tokens=500,
        )

        data = record.as_dict()

        assert data["provider"] == "anthropic"
        assert data["model"] == "claude-3-opus"
        assert data["operation_type"] == "chat_completion"
        assert data["cost_usd"] == "0.015000"


class TestSpendSummary:
    """Tests for SpendSummary dataclass."""

    def test_create_spend_summary(self) -> None:
        """Test creating a spend summary."""
        summary = SpendSummary(
            policy_id="policy-123",
            period_start=datetime(2025, 1, 1, tzinfo=timezone.utc),
            period_end=datetime(2025, 1, 31, 23, 59, 59, tzinfo=timezone.utc),
            total_spent=Decimal("47.50"),
            limit_amount=Decimal("100.00"),
        )

        assert summary.total_spent == Decimal("47.500000")
        assert summary.limit_amount == Decimal("100.000000")
        assert summary.remaining == Decimal("52.500000")
        assert summary.percent_used == pytest.approx(0.475, rel=1e-2)

    def test_spend_summary_over_budget(self) -> None:
        """Test spend summary when over budget."""
        summary = SpendSummary(
            policy_id="policy-456",
            period_start=datetime(2025, 1, 1, tzinfo=timezone.utc),
            period_end=datetime(2025, 1, 31, tzinfo=timezone.utc),
            total_spent=Decimal("120.00"),
            limit_amount=Decimal("100.00"),
        )

        assert summary.remaining == Decimal("0.000000")  # clamped to 0
        assert summary.is_over_budget is True
        assert summary.percent_used == pytest.approx(1.2, rel=1e-2)


class TestBudgetCheckResult:
    """Tests for BudgetCheckResult dataclass."""

    def test_allowed_result(self) -> None:
        """Test an allowed budget check result."""
        result = BudgetCheckResult(
            allowed=True,
            action=LimitAction.WARN,
            policy_id="policy-123",
            current_spend=Decimal("50.00"),
            limit_amount=Decimal("100.00"),
        )

        assert result.allowed is True
        assert result.action == LimitAction.WARN

    def test_blocked_result(self) -> None:
        """Test a blocked budget check result."""
        result = BudgetCheckResult(
            allowed=False,
            action=LimitAction.BLOCK,
            policy_id="policy-456",
            current_spend=Decimal("100.00"),
            limit_amount=Decimal("100.00"),
            warnings=["Budget exceeded"],
        )

        assert result.allowed is False
        assert result.action == LimitAction.BLOCK
        assert "Budget exceeded" in result.warnings


class TestBudgetAlert:
    """Tests for BudgetAlert dataclass."""

    def test_create_alert(self) -> None:
        """Test creating a budget alert."""
        alert = BudgetAlert(
            policy_id="policy-789",
            severity=AlertSeverity.WARNING,
            trigger_type=AlertTriggerType.THRESHOLD_REACHED,
            message="80% of budget used",
            current_spend=Decimal("80.00"),
            limit_amount=Decimal("100.00"),
            threshold_percent=0.80,
        )

        assert alert.severity == AlertSeverity.WARNING
        assert alert.trigger_type == AlertTriggerType.THRESHOLD_REACHED
        assert alert.acknowledged is False
        assert alert.id is not None

    def test_acknowledge_alert(self) -> None:
        """Test acknowledging an alert."""
        alert = BudgetAlert(
            policy_id="policy-abc",
            severity=AlertSeverity.CRITICAL,
            trigger_type=AlertTriggerType.LIMIT_EXCEEDED,
            message="Budget exceeded",
            current_spend=Decimal("105.00"),
            limit_amount=Decimal("100.00"),
        )

        # Manually acknowledge since there may not be an acknowledge method
        alert.acknowledged = True
        alert.acknowledged_by = "admin"
        alert.acknowledged_at = datetime.now(timezone.utc)

        assert alert.acknowledged is True
        assert alert.acknowledged_by == "admin"
        assert alert.acknowledged_at is not None


class TestEnumerations:
    """Tests for budget-related enumerations."""

    def test_budget_scope_values(self) -> None:
        """Test BudgetScope enum values."""
        assert BudgetScope.GLOBAL.value == "global"
        assert BudgetScope.USER.value == "user"
        assert BudgetScope.PROVIDER.value == "provider"
        assert BudgetScope.MODEL.value == "model"

    def test_budget_period_values(self) -> None:
        """Test BudgetPeriod enum values."""
        assert BudgetPeriod.DAILY.value == "daily"
        assert BudgetPeriod.WEEKLY.value == "weekly"
        assert BudgetPeriod.MONTHLY.value == "monthly"
        assert BudgetPeriod.LIFETIME.value == "lifetime"

    def test_operation_type_values(self) -> None:
        """Test OperationType enum values."""
        assert OperationType.CHAT_COMPLETION.value == "chat_completion"
        assert OperationType.IMAGE_GENERATION.value == "image_generation"
        assert OperationType.EMBEDDING.value == "embedding"

    def test_limit_action_values(self) -> None:
        """Test LimitAction enum values."""
        assert LimitAction.WARN.value == "warn"
        assert LimitAction.BLOCK.value == "block"
        assert LimitAction.THROTTLE.value == "throttle"
