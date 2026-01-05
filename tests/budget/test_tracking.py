"""Tests for usage tracking functionality."""

from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from modules.budget.models import BudgetCheckResult, OperationType
from modules.budget.tracking import UsageTracker, get_usage_tracker


class TestUsageTrackerSingleton:
    """Tests for UsageTracker singleton pattern."""

    @pytest.mark.asyncio
    async def test_get_usage_tracker_returns_instance(self) -> None:
        """Test that get_usage_tracker returns a tracker."""
        tracker = await get_usage_tracker()
        assert tracker is not None
        assert isinstance(tracker, UsageTracker)

    @pytest.mark.asyncio
    async def test_get_usage_tracker_singleton(self) -> None:
        """Test singleton pattern."""
        tracker1 = await get_usage_tracker()
        tracker2 = await get_usage_tracker()
        assert tracker1 is tracker2


class TestUsageTrackerRecording:
    """Tests for recording usage via record_usage method."""

    @pytest.mark.asyncio
    async def test_record_llm_usage(self) -> None:
        """Test recording LLM usage event."""
        tracker = await get_usage_tracker()

        # Enable tracking for test
        tracker.enabled = True

        result = await tracker.record_usage(
            provider="openai",
            model="gpt-4o",
            operation_type=OperationType.CHAT_COMPLETION,
            cost_usd=Decimal("0.0075"),
            input_tokens=1000,
            output_tokens=500,
        )

        # Should complete without error (may return None if no manager)
        assert result is None or hasattr(result, "provider")

    @pytest.mark.asyncio
    async def test_record_image_usage(self) -> None:
        """Test recording image generation usage."""
        tracker = await get_usage_tracker()
        tracker.enabled = True

        result = await tracker.record_usage(
            provider="openai",
            model="dall-e-3",
            operation_type=OperationType.IMAGE_GENERATION,
            cost_usd=Decimal("0.040"),
            images_generated=1,
            metadata={"size": "1024x1024", "quality": "standard"},
        )

        assert result is None or hasattr(result, "provider")

    @pytest.mark.asyncio
    async def test_record_embedding_usage(self) -> None:
        """Test recording embedding usage."""
        tracker = await get_usage_tracker()
        tracker.enabled = True

        result = await tracker.record_usage(
            provider="openai",
            model="text-embedding-3-small",
            operation_type=OperationType.EMBEDDING,
            cost_usd=Decimal("0.0002"),
            input_tokens=10000,
        )

        assert result is None or hasattr(result, "provider")

    @pytest.mark.asyncio
    async def test_record_with_user_context(self) -> None:
        """Test recording usage with user/tenant context."""
        tracker = await get_usage_tracker()
        tracker.enabled = True

        result = await tracker.record_usage(
            provider="anthropic",
            model="claude-3-5-sonnet",
            operation_type=OperationType.CHAT_COMPLETION,
            cost_usd=Decimal("0.015"),
            input_tokens=2000,
            output_tokens=1000,
            user_id="user-123",
            tenant_id="tenant-abc",
            persona="helpful-assistant",
        )

        assert result is None or hasattr(result, "user_id")

    @pytest.mark.asyncio
    async def test_record_disabled_returns_none(self) -> None:
        """Test that recording when disabled returns None."""
        tracker = await get_usage_tracker()
        tracker.enabled = False

        result = await tracker.record_usage(
            provider="openai",
            model="gpt-4o",
            operation_type=OperationType.CHAT_COMPLETION,
            cost_usd=Decimal("0.01"),
        )

        assert result is None


class TestUsageTrackerBudgetCheck:
    """Tests for pre-request budget checking."""

    @pytest.mark.asyncio
    async def test_check_budget_returns_result(self) -> None:
        """Test that check_budget returns a BudgetCheckResult."""
        tracker = await get_usage_tracker()

        result = await tracker.check_budget(
            provider="openai",
            model="gpt-4o",
            estimated_input_tokens=1000,
            estimated_output_tokens=500,
        )

        assert isinstance(result, BudgetCheckResult)
        assert hasattr(result, "allowed")

    @pytest.mark.asyncio
    async def test_check_budget_for_images(self) -> None:
        """Test budget check for image generation."""
        tracker = await get_usage_tracker()

        result = await tracker.check_budget(
            provider="openai",
            model="dall-e-3",
            image_count=2,
        )

        assert isinstance(result, BudgetCheckResult)

    @pytest.mark.asyncio
    async def test_check_budget_with_user_context(self) -> None:
        """Test budget check with user/tenant context."""
        tracker = await get_usage_tracker()

        result = await tracker.check_budget(
            provider="openai",
            model="gpt-4o",
            estimated_input_tokens=5000,
            estimated_output_tokens=2000,
            user_id="user-123",
            tenant_id="tenant-abc",
        )

        assert isinstance(result, BudgetCheckResult)


class TestUsageTrackerDecorator:
    """Tests for the track_llm_call decorator."""

    @pytest.mark.asyncio
    async def test_track_llm_call_decorator(self) -> None:
        """Test track_llm_call decorator on async function."""
        tracker = await get_usage_tracker()
        call_count = 0

        @tracker.track_llm_call("openai", "gpt-4o")
        async def mock_llm_call() -> dict[str, Any]:
            nonlocal call_count
            call_count += 1
            # Return mock response with usage info
            return {
                "choices": [{"message": {"content": "Hello"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }

        result = await mock_llm_call()

        assert result is not None
        assert call_count == 1
        assert "choices" in result

    @pytest.mark.asyncio
    async def test_decorator_preserves_return_value(self) -> None:
        """Test that decorator preserves function return value."""
        tracker = await get_usage_tracker()

        @tracker.track_llm_call("anthropic", "claude-3-sonnet")
        async def return_dict() -> dict[str, Any]:
            return {"key": "value", "count": 42}

        result = await return_dict()

        assert result == {"key": "value", "count": 42}

    @pytest.mark.asyncio
    async def test_decorator_handles_exceptions(self) -> None:
        """Test that decorator handles exceptions properly."""
        tracker = await get_usage_tracker()

        @tracker.track_llm_call("openai", "gpt-4o")
        async def raises_error() -> None:
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            await raises_error()

    @pytest.mark.asyncio
    async def test_decorator_with_user_context(self) -> None:
        """Test decorator with user context parameters."""
        tracker = await get_usage_tracker()

        @tracker.track_llm_call(
            provider="openai",
            model="gpt-4o",
            user_id="user-123",
            tenant_id="tenant-abc",
            persona="test-persona",
        )
        async def contextual_call() -> dict[str, Any]:
            return {"response": "test"}

        result = await contextual_call()
        assert result == {"response": "test"}


class TestUsageTrackerEnabled:
    """Tests for enabled property."""

    @pytest.mark.asyncio
    async def test_enabled_property_getter(self) -> None:
        """Test getting enabled state."""
        tracker = await get_usage_tracker()

        enabled = tracker.enabled
        assert isinstance(enabled, bool)

    @pytest.mark.asyncio
    async def test_enabled_property_setter(self) -> None:
        """Test setting enabled state."""
        tracker = await get_usage_tracker()

        original = tracker.enabled

        tracker.enabled = False
        assert tracker.enabled is False

        tracker.enabled = True
        assert tracker.enabled is True

        # Restore original
        tracker.enabled = original
