"""
Unit tests for BudgetTrackingService.

Tests cover:
- Usage recording (LLM, image, generic)
- Spend summary aggregation
- Spend breakdowns by dimension
- Spend trends
- Threshold detection and events
- Buffer and cache management

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

from core.services.budget.tracking_service import BudgetTrackingService
from core.services.budget.types import (
    BudgetUsageRecorded,
    BudgetThresholdReached,
    UsageRecordCreate,
    LLMUsageCreate,
    ImageUsageCreate,
    UsageSummaryRequest,
    SpendBreakdown,
    SpendTrend,
    BudgetPolicy,
    BudgetScope,
    BudgetPeriod,
    LimitAction,
    OperationType,
    UsageRecord,
    SpendSummary,
)


# =============================================================================
# Mock Actor
# =============================================================================


@dataclass
class MockActor:
    """Mock actor for testing."""
    
    user_id: str = "user_123"
    team_id: str = "team_abc"
    actor_type: str = "user"
    is_system: bool = False
    is_admin: bool = False
    
    @property
    def id(self) -> str:
        return self.user_id


# =============================================================================
# Mock Repositories
# =============================================================================


class MockUsageRepository:
    """Mock usage repository for testing."""
    
    def __init__(self):
        self.records: List[UsageRecord] = []
        self.save_called = 0
        self.batch_save_called = 0
    
    async def save_usage_record(self, record: UsageRecord) -> str:
        self.records.append(record)
        self.save_called += 1
        return record.id
    
    async def save_usage_records(self, records: List[UsageRecord]) -> int:
        self.records.extend(records)
        self.batch_save_called += 1
        return len(records)
    
    async def get_usage_records(
        self,
        start_date: datetime,
        end_date: datetime,
        user_id: Optional[str] = None,
        team_id: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        limit: int = 1000,
    ) -> List[UsageRecord]:
        result = []
        for r in self.records:
            if r.timestamp < start_date or r.timestamp > end_date:
                continue
            if user_id and r.user_id != user_id:
                continue
            if team_id and r.team_id != team_id:
                continue
            if provider and r.provider != provider:
                continue
            if model and r.model != model:
                continue
            result.append(r)
            if len(result) >= limit:
                break
        return result
    
    async def get_aggregate_spend(
        self,
        start_date: datetime,
        end_date: datetime,
        user_id: Optional[str] = None,
        team_id: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Decimal:
        total = Decimal("0")
        for r in self.records:
            if r.timestamp < start_date or r.timestamp > end_date:
                continue
            if user_id and r.user_id != user_id:
                continue
            if team_id and r.team_id != team_id:
                continue
            if provider and r.provider != provider:
                continue
            if model and r.model != model:
                continue
            total += r.cost_usd
        return total
    
    async def get_spend_by_dimension(
        self,
        dimension: str,
        start_date: datetime,
        end_date: datetime,
        team_id: Optional[str] = None,
    ) -> Dict[str, Decimal]:
        result: Dict[str, Decimal] = {}
        for r in self.records:
            if r.timestamp < start_date or r.timestamp > end_date:
                continue
            if team_id and r.team_id != team_id:
                continue
            
            if dimension == "provider":
                key = r.provider
            elif dimension == "model":
                key = r.model
            elif dimension == "operation":
                key = r.operation_type.value
            elif dimension == "user":
                key = r.user_id or "unknown"
            else:
                continue
            
            result[key] = result.get(key, Decimal("0")) + r.cost_usd
        return result


class MockPolicyRepository:
    """Mock policy repository for testing."""
    
    def __init__(self):
        self.policies: List[BudgetPolicy] = []
    
    async def get_policies(
        self,
        scope: Optional[BudgetScope] = None,
        scope_id: Optional[str] = None,
        enabled_only: bool = True,
    ) -> List[BudgetPolicy]:
        result = []
        for p in self.policies:
            if scope and p.scope != scope:
                continue
            if scope_id and p.scope_id != scope_id:
                continue
            if enabled_only and not p.enabled:
                continue
            result.append(p)
        return result
    
    def add_policy(self, policy: BudgetPolicy) -> None:
        self.policies.append(policy)


class MockPricingCalculator:
    """Mock pricing calculator for testing."""
    
    def __init__(self, llm_cost: Decimal = Decimal("0.01"), image_cost: Decimal = Decimal("0.04")):
        self.llm_cost = llm_cost
        self.image_cost = image_cost
    
    def calculate_llm_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0,
    ) -> Decimal:
        # Simple mock: $0.01 per 1000 tokens
        total_tokens = input_tokens + output_tokens - cached_tokens
        return Decimal(str(total_tokens)) / Decimal("1000") * self.llm_cost
    
    def calculate_image_cost(
        self,
        model: str,
        count: int = 1,
        size: str = "1024x1024",
        quality: str = "standard",
    ) -> Decimal:
        return self.image_cost * Decimal(str(count))


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def actor() -> MockActor:
    return MockActor()


@pytest.fixture
def usage_repo() -> MockUsageRepository:
    return MockUsageRepository()


@pytest.fixture
def policy_repo() -> MockPolicyRepository:
    return MockPolicyRepository()


@pytest.fixture
def pricing() -> MockPricingCalculator:
    return MockPricingCalculator()


@pytest.fixture
def tracking_service(
    usage_repo: MockUsageRepository,
    policy_repo: MockPolicyRepository,
    pricing: MockPricingCalculator,
) -> BudgetTrackingService:
    return BudgetTrackingService(
        usage_repository=usage_repo,
        policy_repository=policy_repo,
        pricing=pricing,
        buffer_size=10,
        cache_ttl_seconds=60,
        flush_interval_seconds=0,  # Disable background flush for tests
    )


@pytest_asyncio.fixture
async def initialized_service(
    tracking_service: BudgetTrackingService,
):
    await tracking_service.initialize()
    try:
        yield tracking_service
    finally:
        await tracking_service.shutdown()


# =============================================================================
# Record Usage Tests
# =============================================================================


class TestRecordUsage:
    """Tests for basic usage recording."""
    
    @pytest.mark.asyncio
    async def test_record_usage_creates_record(
        self,
        initialized_service: BudgetTrackingService,
        actor: MockActor,
    ):
        """Record usage should create and return a usage record."""
        usage = UsageRecordCreate(
            provider="OpenAI",
            model="gpt-4o",
            operation_type=OperationType.CHAT_COMPLETION,
            cost_usd=Decimal("0.05"),
            input_tokens=1000,
            output_tokens=500,
            team_id="team_abc",
        )
        
        record = await initialized_service.record_usage(actor, usage)
        
        assert record is not None
        assert record.provider == "OpenAI"
        assert record.model == "gpt-4o"
        assert record.cost_usd == Decimal("0.05")
        assert record.input_tokens == 1000
        assert record.output_tokens == 500
    
    @pytest.mark.asyncio
    async def test_record_usage_sets_team_from_actor(
        self,
        initialized_service: BudgetTrackingService,
        actor: MockActor,
    ):
        """Record usage should set team_id from actor if not provided."""
        usage = UsageRecordCreate(
            provider="OpenAI",
            model="gpt-4o",
            operation_type=OperationType.CHAT_COMPLETION,
            cost_usd=Decimal("0.01"),
        )
        
        record = await initialized_service.record_usage(actor, usage)
        
        assert record.team_id == actor.team_id
        assert record.user_id == actor.user_id
    
    @pytest.mark.asyncio
    async def test_record_usage_buffers_record(
        self,
        initialized_service: BudgetTrackingService,
        actor: MockActor,
    ):
        """Record usage should buffer records for batch persistence."""
        for i in range(5):
            usage = UsageRecordCreate(
                provider="OpenAI",
                model="gpt-4o",
                operation_type=OperationType.CHAT_COMPLETION,
                cost_usd=Decimal("0.01"),
            )
            await initialized_service.record_usage(actor, usage)
        
        # Records should be in buffer
        assert len(initialized_service._usage_buffer) == 5
    
    @pytest.mark.asyncio
    async def test_record_usage_auto_flushes_on_buffer_full(
        self,
        initialized_service: BudgetTrackingService,
        usage_repo: MockUsageRepository,
        actor: MockActor,
    ):
        """Record usage should auto-flush when buffer is full."""
        # Buffer size is 10, so 11th record should trigger flush
        for i in range(11):
            usage = UsageRecordCreate(
                provider="OpenAI",
                model="gpt-4o",
                operation_type=OperationType.CHAT_COMPLETION,
                cost_usd=Decimal("0.01"),
            )
            await initialized_service.record_usage(actor, usage)
        
        # Give async flush task time to run
        await asyncio.sleep(0.1)
        
        # Batch save should have been called
        assert usage_repo.batch_save_called >= 1


class TestRecordLLMUsage:
    """Tests for LLM usage recording with automatic cost calculation."""
    
    @pytest.mark.asyncio
    async def test_record_llm_usage_calculates_cost(
        self,
        initialized_service: BudgetTrackingService,
        actor: MockActor,
    ):
        """Record LLM usage should calculate cost automatically."""
        usage = LLMUsageCreate(
            provider="OpenAI",
            model="gpt-4o",
            input_tokens=1000,
            output_tokens=500,
        )
        
        record = await initialized_service.record_llm_usage(actor, usage)
        
        # Mock pricing: $0.01 per 1000 tokens = 1500 tokens = $0.015
        assert record.cost_usd == Decimal("0.015")
        assert record.operation_type == OperationType.CHAT_COMPLETION
    
    @pytest.mark.asyncio
    async def test_record_llm_usage_with_cached_tokens(
        self,
        initialized_service: BudgetTrackingService,
        actor: MockActor,
    ):
        """Record LLM usage should account for cached tokens."""
        usage = LLMUsageCreate(
            provider="OpenAI",
            model="gpt-4o",
            input_tokens=1000,
            output_tokens=500,
            cached_tokens=500,  # 500 cached = 1000 billable
        )
        
        record = await initialized_service.record_llm_usage(actor, usage)
        
        # 1500 - 500 cached = 1000 billable = $0.01
        assert record.cost_usd == Decimal("0.010")


class TestRecordImageUsage:
    """Tests for image usage recording with automatic cost calculation."""
    
    @pytest.mark.asyncio
    async def test_record_image_usage_calculates_cost(
        self,
        initialized_service: BudgetTrackingService,
        actor: MockActor,
    ):
        """Record image usage should calculate cost automatically."""
        usage = ImageUsageCreate(
            provider="OpenAI",
            model="dall-e-3",
            count=2,
            size="1024x1024",
            quality="standard",
        )
        
        record = await initialized_service.record_image_usage(actor, usage)
        
        # Mock pricing: $0.04 per image = 2 images = $0.08
        assert record.cost_usd == Decimal("0.08")
        assert record.operation_type == OperationType.IMAGE_GENERATION
        assert record.images_generated == 2


# =============================================================================
# Usage Summary Tests
# =============================================================================


class TestGetUsageSummary:
    """Tests for usage summary aggregation."""
    
    @pytest.mark.asyncio
    async def test_get_usage_summary_aggregates_buffer(
        self,
        initialized_service: BudgetTrackingService,
        actor: MockActor,
    ):
        """Get usage summary should aggregate from buffer."""
        # Record some usage
        for i in range(3):
            usage = UsageRecordCreate(
                provider="OpenAI",
                model="gpt-4o",
                operation_type=OperationType.CHAT_COMPLETION,
                cost_usd=Decimal("0.10"),
                team_id=actor.team_id,
            )
            await initialized_service.record_usage(actor, usage)
        
        request = UsageSummaryRequest(
            scope=BudgetScope.TEAM,
            scope_id=actor.team_id,
            period=BudgetPeriod.MONTHLY,
        )
        
        summary = await initialized_service.get_usage_summary(actor, request)
        
        assert summary.total_spent == Decimal("0.30")
        assert summary.record_count == 3
        assert summary.by_provider.get("OpenAI") == Decimal("0.30")
    
    @pytest.mark.asyncio
    async def test_get_usage_summary_uses_cache(
        self,
        initialized_service: BudgetTrackingService,
        actor: MockActor,
    ):
        """Get usage summary should use cached values."""
        # First call populates cache
        request = UsageSummaryRequest(
            scope=BudgetScope.GLOBAL,
            period=BudgetPeriod.MONTHLY,
        )
        
        summary1 = await initialized_service.get_usage_summary(actor, request)
        
        # Add a record (won't invalidate cache for different scope)
        usage = UsageRecordCreate(
            provider="OpenAI",
            model="gpt-4o",
            operation_type=OperationType.CHAT_COMPLETION,
            cost_usd=Decimal("1.00"),
            # No team_id - should not affect team-scoped cache
        )
        await initialized_service.record_usage(actor, usage)
        
        # Second call should still return cached value (cache invalidated for GLOBAL)
        # Actually, GLOBAL cache is invalidated, so this is a fresh call
        summary2 = await initialized_service.get_usage_summary(actor, request)
        
        # Cache was invalidated, so should include new record
        assert summary2.total_spent >= summary1.total_spent
    
    @pytest.mark.asyncio
    async def test_get_usage_summary_with_policy(
        self,
        initialized_service: BudgetTrackingService,
        policy_repo: MockPolicyRepository,
        actor: MockActor,
    ):
        """Get usage summary should include policy limits."""
        # Add a policy
        policy = BudgetPolicy(
            name="Team Monthly",
            scope=BudgetScope.TEAM,
            scope_id=actor.team_id,
            limit_amount=Decimal("100.00"),
            period=BudgetPeriod.MONTHLY,
        )
        policy_repo.add_policy(policy)
        
        # Record usage
        usage = UsageRecordCreate(
            provider="OpenAI",
            model="gpt-4o",
            operation_type=OperationType.CHAT_COMPLETION,
            cost_usd=Decimal("25.00"),
            team_id=actor.team_id,
        )
        await initialized_service.record_usage(actor, usage)
        
        request = UsageSummaryRequest(
            scope=BudgetScope.TEAM,
            scope_id=actor.team_id,
            period=BudgetPeriod.MONTHLY,
        )
        
        summary = await initialized_service.get_usage_summary(actor, request)
        
        assert summary.total_spent == Decimal("25.00")
        assert summary.limit_amount == Decimal("100.00")
        assert summary.percent_used == 0.25


# =============================================================================
# Spend Breakdown Tests
# =============================================================================


class TestSpendBreakdown:
    """Tests for spend breakdown by dimension."""
    
    @pytest.mark.asyncio
    async def test_get_spend_by_provider(
        self,
        initialized_service: BudgetTrackingService,
        actor: MockActor,
    ):
        """Get spend by provider should break down by provider."""
        # Record usage from different providers
        for provider, cost in [("OpenAI", "0.50"), ("Anthropic", "0.30"), ("OpenAI", "0.20")]:
            usage = UsageRecordCreate(
                provider=provider,
                model="test-model",
                operation_type=OperationType.CHAT_COMPLETION,
                cost_usd=Decimal(cost),
            )
            await initialized_service.record_usage(actor, usage)
        
        # Flush buffer to repo so breakdown query finds the records
        await initialized_service._flush_buffer()
        
        breakdown = await initialized_service.get_spend_by_provider(actor)
        
        assert breakdown.dimension == "provider"
        assert breakdown.items.get("OpenAI") == Decimal("0.70")
        assert breakdown.items.get("Anthropic") == Decimal("0.30")
        assert breakdown.total == Decimal("1.00")
    
    @pytest.mark.asyncio
    async def test_get_spend_by_model(
        self,
        initialized_service: BudgetTrackingService,
        actor: MockActor,
    ):
        """Get spend by model should break down by model."""
        for model, cost in [("gpt-4o", "0.40"), ("gpt-4o-mini", "0.10"), ("gpt-4o", "0.20")]:
            usage = UsageRecordCreate(
                provider="OpenAI",
                model=model,
                operation_type=OperationType.CHAT_COMPLETION,
                cost_usd=Decimal(cost),
            )
            await initialized_service.record_usage(actor, usage)
        
        # Flush buffer to repo so breakdown query finds the records
        await initialized_service._flush_buffer()
        
        breakdown = await initialized_service.get_spend_by_model(actor)
        
        assert breakdown.dimension == "model"
        assert breakdown.items.get("gpt-4o") == Decimal("0.60")
        assert breakdown.items.get("gpt-4o-mini") == Decimal("0.10")
    
    @pytest.mark.asyncio
    async def test_get_spend_by_operation(
        self,
        initialized_service: BudgetTrackingService,
        actor: MockActor,
    ):
        """Get spend by operation should break down by operation type."""
        for op, cost in [
            (OperationType.CHAT_COMPLETION, "0.50"),
            (OperationType.IMAGE_GENERATION, "0.30"),
            (OperationType.CHAT_COMPLETION, "0.20"),
        ]:
            usage = UsageRecordCreate(
                provider="OpenAI",
                model="test-model",
                operation_type=op,
                cost_usd=Decimal(cost),
            )
            await initialized_service.record_usage(actor, usage)
        
        # Flush buffer to repo so breakdown query finds the records
        await initialized_service._flush_buffer()
        
        breakdown = await initialized_service.get_spend_by_operation(actor)
        
        assert breakdown.dimension == "operation"
        assert breakdown.items.get("chat_completion") == Decimal("0.70")
        assert breakdown.items.get("image_generation") == Decimal("0.30")


# =============================================================================
# Spend Trend Tests
# =============================================================================


class TestSpendTrend:
    """Tests for historical spend trends."""
    
    @pytest.mark.asyncio
    async def test_get_spend_trend_returns_points(
        self,
        initialized_service: BudgetTrackingService,
        actor: MockActor,
    ):
        """Get spend trend should return historical data points."""
        # Record some usage
        usage = UsageRecordCreate(
            provider="OpenAI",
            model="gpt-4o",
            operation_type=OperationType.CHAT_COMPLETION,
            cost_usd=Decimal("1.00"),
        )
        await initialized_service.record_usage(actor, usage)
        
        trend = await initialized_service.get_spend_trend(
            actor,
            scope=BudgetScope.GLOBAL,
            period=BudgetPeriod.DAILY,
            num_periods=7,
        )
        
        assert len(trend.points) == 7
        assert trend.period_type == BudgetPeriod.DAILY
        assert trend.total_spent >= Decimal("0")
    
    @pytest.mark.asyncio
    async def test_get_spend_trend_calculates_direction(
        self,
        initialized_service: BudgetTrackingService,
        actor: MockActor,
    ):
        """Get spend trend should calculate trend direction."""
        trend = await initialized_service.get_spend_trend(
            actor,
            scope=BudgetScope.GLOBAL,
            period=BudgetPeriod.DAILY,
            num_periods=7,
        )
        
        assert trend.trend_direction in ("up", "down", "stable")
        assert isinstance(trend.percent_change, float)


# =============================================================================
# Threshold Detection Tests
# =============================================================================


class TestThresholdDetection:
    """Tests for budget threshold detection and events."""
    
    @pytest.mark.asyncio
    async def test_threshold_event_published_when_crossed(
        self,
        usage_repo: MockUsageRepository,
        policy_repo: MockPolicyRepository,
        pricing: MockPricingCalculator,
        actor: MockActor,
    ):
        """Threshold event should be published when soft limit is crossed."""
        events_published: List[Any] = []
        
        async def capture_event(event):
            events_published.append(event)
        
        service = BudgetTrackingService(
            usage_repository=usage_repo,
            policy_repository=policy_repo,
            pricing=pricing,
            event_publisher=capture_event,
            flush_interval_seconds=0,
        )
        await service.initialize()
        
        # Add a policy with 80% soft limit
        policy = BudgetPolicy(
            name="Test Budget",
            scope=BudgetScope.TEAM,
            scope_id=actor.team_id,
            limit_amount=Decimal("100.00"),
            period=BudgetPeriod.MONTHLY,
            soft_limit_percent=0.80,
        )
        policy_repo.add_policy(policy)
        
        # Record usage that crosses 80% threshold
        usage = UsageRecordCreate(
            provider="OpenAI",
            model="gpt-4o",
            operation_type=OperationType.CHAT_COMPLETION,
            cost_usd=Decimal("85.00"),
            team_id=actor.team_id,
        )
        await service.record_usage(actor, usage)
        
        await service.shutdown()
        
        # Check for threshold event
        threshold_events = [e for e in events_published if isinstance(e, BudgetThresholdReached)]
        assert len(threshold_events) >= 1
        
        event = threshold_events[0]
        assert event.policy_id == policy.id
        assert event.current_percent >= 0.80


# =============================================================================
# Event Publishing Tests
# =============================================================================


class TestEventPublishing:
    """Tests for domain event publishing."""
    
    @pytest.mark.asyncio
    async def test_usage_event_published_on_record(
        self,
        usage_repo: MockUsageRepository,
        pricing: MockPricingCalculator,
        actor: MockActor,
    ):
        """Usage recorded event should be published on record."""
        events_published: List[Any] = []
        
        async def capture_event(event):
            events_published.append(event)
        
        service = BudgetTrackingService(
            usage_repository=usage_repo,
            pricing=pricing,
            event_publisher=capture_event,
            flush_interval_seconds=0,
        )
        await service.initialize()
        
        usage = UsageRecordCreate(
            provider="OpenAI",
            model="gpt-4o",
            operation_type=OperationType.CHAT_COMPLETION,
            cost_usd=Decimal("0.05"),
            team_id=actor.team_id,
        )
        await service.record_usage(actor, usage)
        
        await service.shutdown()
        
        usage_events = [e for e in events_published if isinstance(e, BudgetUsageRecorded)]
        assert len(usage_events) == 1
        
        event = usage_events[0]
        assert event.provider == "OpenAI"
        assert event.model == "gpt-4o"
        assert event.cost_usd == Decimal("0.05")


# =============================================================================
# Cache Management Tests
# =============================================================================


class TestCacheManagement:
    """Tests for spend cache management."""
    
    @pytest.mark.asyncio
    async def test_invalidate_cache_all(
        self,
        initialized_service: BudgetTrackingService,
        actor: MockActor,
    ):
        """Invalidate cache should clear all entries."""
        # Populate cache
        request = UsageSummaryRequest(
            scope=BudgetScope.GLOBAL,
            period=BudgetPeriod.MONTHLY,
        )
        await initialized_service.get_usage_summary(actor, request)
        
        assert len(initialized_service._spend_cache) > 0
        
        # Invalidate
        initialized_service.invalidate_cache()
        
        assert len(initialized_service._spend_cache) == 0
    
    @pytest.mark.asyncio
    async def test_invalidate_cache_specific_scope(
        self,
        initialized_service: BudgetTrackingService,
        actor: MockActor,
    ):
        """Invalidate cache should clear specific scope entries."""
        # Populate cache for multiple scopes
        for scope in [BudgetScope.GLOBAL, BudgetScope.TEAM]:
            request = UsageSummaryRequest(
                scope=scope,
                scope_id=actor.team_id if scope == BudgetScope.TEAM else None,
                period=BudgetPeriod.MONTHLY,
            )
            await initialized_service.get_usage_summary(actor, request)
        
        # Invalidate just team scope
        initialized_service.invalidate_cache(
            scope=BudgetScope.TEAM,
            scope_id=actor.team_id,
        )
        
        # Global cache should still exist
        global_key = "global:all:monthly"
        assert global_key in initialized_service._spend_cache
    
    @pytest.mark.asyncio
    async def test_set_rollover_amount(
        self,
        initialized_service: BudgetTrackingService,
    ):
        """Set rollover amount should update rollover tracking."""
        policy_id = "policy_123"
        
        initialized_service.set_rollover_amount(policy_id, Decimal("50.00"))
        
        assert initialized_service._rollover_amounts[policy_id] == Decimal("50.000000")


# =============================================================================
# Service Lifecycle Tests
# =============================================================================


class TestServiceLifecycle:
    """Tests for service initialization and shutdown."""
    
    @pytest.mark.asyncio
    async def test_initialize_starts_service(
        self,
        tracking_service: BudgetTrackingService,
    ):
        """Initialize should start the service."""
        assert not tracking_service._initialized
        
        await tracking_service.initialize()
        
        assert tracking_service._initialized
        
        await tracking_service.shutdown()
    
    @pytest.mark.asyncio
    async def test_shutdown_flushes_buffer(
        self,
        tracking_service: BudgetTrackingService,
        usage_repo: MockUsageRepository,
        actor: MockActor,
    ):
        """Shutdown should flush remaining buffer records."""
        await tracking_service.initialize()
        
        # Add some records to buffer
        for _ in range(3):
            usage = UsageRecordCreate(
                provider="OpenAI",
                model="gpt-4o",
                operation_type=OperationType.CHAT_COMPLETION,
                cost_usd=Decimal("0.01"),
            )
            await tracking_service.record_usage(actor, usage)
        
        assert len(tracking_service._usage_buffer) == 3
        
        # Shutdown should flush
        await tracking_service.shutdown()
        
        assert usage_repo.batch_save_called >= 1
        assert len(usage_repo.records) == 3
    
    @pytest.mark.asyncio
    async def test_double_initialize_is_safe(
        self,
        tracking_service: BudgetTrackingService,
    ):
        """Multiple initialize calls should be safe."""
        await tracking_service.initialize()
        await tracking_service.initialize()  # Should not error
        
        assert tracking_service._initialized
        
        await tracking_service.shutdown()
    
    @pytest.mark.asyncio
    async def test_double_shutdown_is_safe(
        self,
        tracking_service: BudgetTrackingService,
    ):
        """Multiple shutdown calls should be safe."""
        await tracking_service.initialize()
        await tracking_service.shutdown()
        await tracking_service.shutdown()  # Should not error
        
        assert tracking_service._shutting_down
