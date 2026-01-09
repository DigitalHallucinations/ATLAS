"""
Budget Tracking Service.

High-performance service for recording usage and generating spend summaries.
Part of the ATLAS budget service layer.

Author: ATLAS Team
Date: Jan 8, 2026
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    Tuple,
    runtime_checkable,
)

from modules.logging.logger import setup_logger

from .types import (
    # Models
    BudgetPolicy,
    BudgetPeriod,
    BudgetScope,
    LimitAction,
    OperationType,
    UsageRecord,
    SpendSummary,
    # Events
    BudgetUsageRecorded,
    BudgetThresholdReached,
    # DTOs
    UsageRecordCreate,
    LLMUsageCreate,
    ImageUsageCreate,
    UsageSummaryRequest,
    SpendBreakdown,
    SpendTrendPoint,
    SpendTrend,
)
from .exceptions import (
    BudgetError,
    BudgetValidationError,
)

logger = setup_logger(__name__)


def _quantize_decimal(value: Decimal, places: int = 6) -> Decimal:
    """Quantize decimal to specified decimal places."""
    quantizer = Decimal(10) ** -places
    return value.quantize(quantizer, rounding=ROUND_HALF_UP)


# =============================================================================
# Repository Protocol
# =============================================================================


@runtime_checkable
class UsageRepository(Protocol):
    """Protocol for usage record storage operations."""

    async def save_usage_record(self, record: UsageRecord) -> str:
        """Save a usage record, return its ID."""
        ...

    async def save_usage_records(self, records: List[UsageRecord]) -> int:
        """Batch save usage records, return count saved."""
        ...

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
        """Retrieve usage records with filters."""
        ...

    async def get_aggregate_spend(
        self,
        start_date: datetime,
        end_date: datetime,
        user_id: Optional[str] = None,
        team_id: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Decimal:
        """Get aggregate spend for a time period."""
        ...

    async def get_spend_by_dimension(
        self,
        dimension: str,  # "provider", "model", "operation", "user"
        start_date: datetime,
        end_date: datetime,
        team_id: Optional[str] = None,
    ) -> Dict[str, Decimal]:
        """Get spend broken down by a dimension."""
        ...


@runtime_checkable
class PolicyRepository(Protocol):
    """Protocol for budget policy lookups."""

    async def get_policies(
        self,
        scope: Optional[BudgetScope] = None,
        scope_id: Optional[str] = None,
        enabled_only: bool = True,
    ) -> List[BudgetPolicy]:
        """Retrieve budget policies matching criteria."""
        ...


@runtime_checkable
class PricingCalculator(Protocol):
    """Protocol for cost calculation."""

    def calculate_llm_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0,
    ) -> Decimal:
        """Calculate cost for LLM usage."""
        ...

    def calculate_image_cost(
        self,
        model: str,
        count: int = 1,
        size: str = "1024x1024",
        quality: str = "standard",
    ) -> Decimal:
        """Calculate cost for image generation."""
        ...


# =============================================================================
# Event Publisher Type
# =============================================================================

EventPublisher = Callable[
    [BudgetUsageRecorded | BudgetThresholdReached],
    Coroutine[Any, Any, None],
]


# =============================================================================
# Budget Tracking Service
# =============================================================================


class BudgetTrackingService:
    """Service for recording usage and generating spend summaries.

    This service is designed to be high-performance as it sits on the
    critical path for all LLM and media generation operations.

    Responsibilities:
    - Record usage events (LLM, image, audio)
    - Aggregate spending by various dimensions
    - Generate spending trends and reports
    - Emit events for threshold notifications

    Thread Safety:
    - Uses async locks for in-memory buffer operations
    - Repository operations assumed to be async-safe

    Usage::

        service = BudgetTrackingService(
            usage_repository=repo,
            policy_repository=policy_repo,
            pricing=pricing_registry,
        )
        await service.initialize()

        # Record LLM usage
        record = await service.record_llm_usage(
            actor=actor,
            usage=LLMUsageCreate(
                provider="OpenAI",
                model="gpt-4o",
                input_tokens=1000,
                output_tokens=500,
            ),
        )

        # Get spending summary
        summary = await service.get_usage_summary(
            actor=actor,
            request=UsageSummaryRequest(
                scope=BudgetScope.TEAM,
                period=BudgetPeriod.MONTHLY,
            ),
        )
    """

    def __init__(
        self,
        usage_repository: Optional[UsageRepository] = None,
        policy_repository: Optional[PolicyRepository] = None,
        pricing: Optional[PricingCalculator] = None,
        event_publisher: Optional[EventPublisher] = None,
        *,
        buffer_size: int = 100,
        cache_ttl_seconds: int = 60,
        flush_interval_seconds: int = 30,
    ):
        """Initialize the tracking service.

        Args:
            usage_repository: Repository for usage record persistence.
            policy_repository: Repository for policy lookups.
            pricing: Calculator for cost estimation.
            event_publisher: Callback for publishing domain events.
            buffer_size: Max records to buffer before flush.
            cache_ttl_seconds: TTL for spend cache entries.
            flush_interval_seconds: Interval for background flush.
        """
        self._usage_repo = usage_repository
        self._policy_repo = policy_repository
        self._pricing = pricing
        self._event_publisher = event_publisher
        
        # Configuration
        self._buffer_size = buffer_size
        self._cache_ttl_seconds = cache_ttl_seconds
        self._flush_interval_seconds = flush_interval_seconds
        
        # In-memory buffer for high-performance recording
        self._usage_buffer: List[UsageRecord] = []
        self._buffer_lock = asyncio.Lock()
        
        # Spend cache for fast lookups
        self._spend_cache: Dict[str, SpendSummary] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
        
        # Rollover amounts (loaded from policies)
        self._rollover_amounts: Dict[str, Decimal] = {}
        
        # Threshold tracking to avoid duplicate events
        self._threshold_notified: Dict[str, float] = {}
        
        # Background tasks
        self._flush_task: Optional[asyncio.Task] = None
        self._enabled = True
        
        # Lifecycle state
        self._initialized = False
        self._shutting_down = False
        
        self.logger = logger

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def initialize(self) -> None:
        """Initialize the tracking service."""
        if self._initialized:
            return

        self.logger.info("Initializing BudgetTrackingService")
        
        # Start background flush task
        if self._flush_interval_seconds > 0:
            self._flush_task = asyncio.create_task(
                self._background_flush_loop()
            )
        
        self._initialized = True
        self.logger.info("BudgetTrackingService initialized")

    async def shutdown(self) -> None:
        """Shutdown the tracking service, flushing pending records."""
        if self._shutting_down:
            return

        self._shutting_down = True
        self.logger.info("Shutting down BudgetTrackingService")
        
        # Cancel background task
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        
        # Flush remaining records
        await self._flush_buffer()
        
        self.logger.info("BudgetTrackingService shutdown complete")

    async def _background_flush_loop(self) -> None:
        """Background task to periodically flush the usage buffer."""
        while not self._shutting_down:
            try:
                await asyncio.sleep(self._flush_interval_seconds)
                await self._flush_buffer()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self.logger.warning("Background flush error: %s", exc)

    async def _flush_buffer(self) -> int:
        """Flush buffered usage records to storage.

        Returns:
            Number of records flushed.
        """
        if not self._usage_repo:
            return 0

        async with self._buffer_lock:
            if not self._usage_buffer:
                return 0
            
            records_to_flush = self._usage_buffer.copy()
            self._usage_buffer.clear()

        try:
            saved = await self._usage_repo.save_usage_records(records_to_flush)
            self.logger.debug("Flushed %d usage records", saved)
            return saved
        except Exception as exc:
            self.logger.error("Failed to flush usage records: %s", exc)
            # Put records back in buffer for retry
            async with self._buffer_lock:
                self._usage_buffer = records_to_flush + self._usage_buffer
            return 0

    # =========================================================================
    # Usage Recording
    # =========================================================================

    async def record_usage(
        self,
        actor: Any,
        usage: UsageRecordCreate,
    ) -> UsageRecord:
        """Record a usage event.

        This is the primary entry point for recording usage. It:
        1. Creates the usage record
        2. Buffers for batch persistence
        3. Updates the spend cache
        4. Checks for threshold crossings
        5. Emits usage_recorded event

        Args:
            actor: The actor performing the operation.
            usage: The usage data to record.

        Returns:
            The created UsageRecord with generated ID.
        """
        if not self._enabled:
            return usage.to_record()

        # Create the record
        record = usage.to_record()
        
        # Set team from actor if not provided
        if not record.team_id and hasattr(actor, "team_id"):
            record.team_id = actor.team_id
        if not record.user_id and hasattr(actor, "user_id"):
            record.user_id = actor.user_id

        # Buffer the record
        async with self._buffer_lock:
            self._usage_buffer.append(record)
            
            # Auto-flush if buffer is full
            if len(self._usage_buffer) >= self._buffer_size:
                asyncio.create_task(self._flush_buffer())

        # Update spend cache
        await self._update_spend_cache(record)

        # Check thresholds and emit events
        await self._check_thresholds(actor, record)

        # Publish event
        await self._publish_usage_event(actor, record)

        return record

    async def record_llm_usage(
        self,
        actor: Any,
        usage: LLMUsageCreate,
    ) -> UsageRecord:
        """Record LLM API usage with automatic cost calculation.

        Args:
            actor: The actor performing the operation.
            usage: LLM usage details.

        Returns:
            The created UsageRecord.
        """
        # Calculate cost
        cost = Decimal("0")
        if self._pricing:
            cost = self._pricing.calculate_llm_cost(
                model=usage.model,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cached_tokens=usage.cached_tokens,
            )

        # Create full usage record
        full_usage = UsageRecordCreate(
            provider=usage.provider,
            model=usage.model,
            operation_type=OperationType.CHAT_COMPLETION,
            cost_usd=cost,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            user_id=usage.user_id,
            team_id=usage.team_id,
            project_id=usage.project_id,
            job_id=usage.job_id,
            task_id=usage.task_id,
            agent_id=usage.agent_id,
            session_id=usage.session_id,
            conversation_id=usage.conversation_id,
            tool_id=usage.tool_id,
            skill_id=usage.skill_id,
            request_id=usage.request_id,
            success=usage.success,
            metadata=usage.metadata,
        )

        return await self.record_usage(actor, full_usage)

    async def record_image_usage(
        self,
        actor: Any,
        usage: ImageUsageCreate,
    ) -> UsageRecord:
        """Record image generation usage with automatic cost calculation.

        Args:
            actor: The actor performing the operation.
            usage: Image usage details.

        Returns:
            The created UsageRecord.
        """
        # Calculate cost
        cost = Decimal("0")
        if self._pricing:
            cost = self._pricing.calculate_image_cost(
                model=usage.model,
                count=usage.count,
                size=usage.size,
                quality=usage.quality,
            )

        # Create full usage record
        full_usage = UsageRecordCreate(
            provider=usage.provider,
            model=usage.model,
            operation_type=OperationType.IMAGE_GENERATION,
            cost_usd=cost,
            images_generated=usage.count,
            image_size=usage.size,
            image_quality=usage.quality,
            user_id=usage.user_id,
            team_id=usage.team_id,
            project_id=usage.project_id,
            job_id=usage.job_id,
            task_id=usage.task_id,
            agent_id=usage.agent_id,
            session_id=usage.session_id,
            conversation_id=usage.conversation_id,
            tool_id=usage.tool_id,
            skill_id=usage.skill_id,
            request_id=usage.request_id,
            success=usage.success,
            metadata=usage.metadata,
        )

        return await self.record_usage(actor, full_usage)

    # =========================================================================
    # Spending Queries
    # =========================================================================

    async def get_usage_summary(
        self,
        actor: Any,
        request: UsageSummaryRequest,
    ) -> SpendSummary:
        """Get a spending summary for a scope and period.

        Args:
            actor: The actor requesting the summary.
            request: Summary request parameters.

        Returns:
            SpendSummary with aggregated data.
        """
        # Use scope_id from request or actor
        scope_id = request.scope_id
        if not scope_id:
            if request.scope == BudgetScope.TEAM and hasattr(actor, "team_id"):
                scope_id = actor.team_id
            elif request.scope == BudgetScope.USER and hasattr(actor, "user_id"):
                scope_id = actor.user_id

        # Check cache
        cache_key = self._get_cache_key(request.scope, scope_id, request.period)
        if cache_key in self._spend_cache:
            cached_at = self._cache_timestamps.get(cache_key)
            if cached_at:
                age = (datetime.now(timezone.utc) - cached_at).total_seconds()
                if age < self._cache_ttl_seconds:
                    return self._spend_cache[cache_key]

        # Calculate period boundaries
        period_start, period_end = self._get_period_boundaries(
            request.period,
            request.start_date,
            request.end_date,
        )

        # Find applicable policy
        policy = await self._find_policy(request.scope, scope_id)

        # Aggregate spending
        summary = await self._aggregate_spending(
            scope=request.scope,
            scope_id=scope_id,
            period_start=period_start,
            period_end=period_end,
            policy=policy,
        )

        # Cache result
        self._spend_cache[cache_key] = summary
        self._cache_timestamps[cache_key] = datetime.now(timezone.utc)

        return summary

    async def get_spend_by_provider(
        self,
        actor: Any,
        scope: BudgetScope = BudgetScope.GLOBAL,
        scope_id: Optional[str] = None,
        period: BudgetPeriod = BudgetPeriod.MONTHLY,
    ) -> SpendBreakdown:
        """Get spending breakdown by provider.

        Args:
            actor: The actor requesting the breakdown.
            scope: Budget scope.
            scope_id: Scope identifier.
            period: Time period.

        Returns:
            SpendBreakdown by provider.
        """
        return await self._get_spend_breakdown(
            actor=actor,
            dimension="provider",
            scope=scope,
            scope_id=scope_id,
            period=period,
        )

    async def get_spend_by_model(
        self,
        actor: Any,
        scope: BudgetScope = BudgetScope.GLOBAL,
        scope_id: Optional[str] = None,
        period: BudgetPeriod = BudgetPeriod.MONTHLY,
    ) -> SpendBreakdown:
        """Get spending breakdown by model.

        Args:
            actor: The actor requesting the breakdown.
            scope: Budget scope.
            scope_id: Scope identifier.
            period: Time period.

        Returns:
            SpendBreakdown by model.
        """
        return await self._get_spend_breakdown(
            actor=actor,
            dimension="model",
            scope=scope,
            scope_id=scope_id,
            period=period,
        )

    async def get_spend_by_operation(
        self,
        actor: Any,
        scope: BudgetScope = BudgetScope.GLOBAL,
        scope_id: Optional[str] = None,
        period: BudgetPeriod = BudgetPeriod.MONTHLY,
    ) -> SpendBreakdown:
        """Get spending breakdown by operation type.

        Args:
            actor: The actor requesting the breakdown.
            scope: Budget scope.
            scope_id: Scope identifier.
            period: Time period.

        Returns:
            SpendBreakdown by operation type.
        """
        return await self._get_spend_breakdown(
            actor=actor,
            dimension="operation",
            scope=scope,
            scope_id=scope_id,
            period=period,
        )

    async def get_spend_trend(
        self,
        actor: Any,
        scope: BudgetScope = BudgetScope.GLOBAL,
        scope_id: Optional[str] = None,
        period: BudgetPeriod = BudgetPeriod.DAILY,
        num_periods: int = 30,
    ) -> SpendTrend:
        """Get historical spending trend.

        Args:
            actor: The actor requesting the trend.
            scope: Budget scope.
            scope_id: Scope identifier.
            period: Period granularity (daily, weekly, monthly).
            num_periods: Number of periods to include.

        Returns:
            SpendTrend with historical data points.
        """
        # Use scope_id from actor if not provided
        if not scope_id:
            if scope == BudgetScope.TEAM and hasattr(actor, "team_id"):
                scope_id = actor.team_id
            elif scope == BudgetScope.USER and hasattr(actor, "user_id"):
                scope_id = actor.user_id

        # Calculate period boundaries for each point
        points: List[SpendTrendPoint] = []
        now = datetime.now(timezone.utc)
        
        for i in range(num_periods - 1, -1, -1):
            period_start, period_end = self._get_period_at_offset(period, i)
            
            # Get spend for this period
            summary = await self._aggregate_spending(
                scope=scope,
                scope_id=scope_id,
                period_start=period_start,
                period_end=min(period_end, now),
                policy=None,
            )
            
            points.append(SpendTrendPoint(
                period_start=period_start,
                period_end=period_end,
                total_spent=summary.total_spent,
                record_count=summary.record_count,
                by_provider=summary.by_provider,
            ))

        # Calculate trend statistics
        total_spent = sum((p.total_spent for p in points), Decimal("0"))
        average = total_spent / Decimal(str(num_periods)) if points else Decimal("0")
        
        # Determine trend direction
        if len(points) >= 2:
            first_half = sum((p.total_spent for p in points[:len(points)//2]), Decimal("0"))
            second_half = sum((p.total_spent for p in points[len(points)//2:]), Decimal("0"))
            
            if first_half > 0:
                change_pct = float((second_half - first_half) / first_half * 100)
            else:
                change_pct = 100.0 if second_half > 0 else 0.0
            
            if change_pct > 10:
                direction: Literal["up", "down", "stable"] = "up"
            elif change_pct < -10:
                direction = "down"
            else:
                direction = "stable"
        else:
            direction = "stable"
            change_pct = 0.0

        return SpendTrend(
            scope=scope,
            scope_id=scope_id,
            period_type=period,
            points=points,
            total_spent=total_spent,
            average_per_period=_quantize_decimal(average),
            trend_direction=direction,
            percent_change=change_pct,
        )

    async def _get_spend_breakdown(
        self,
        actor: Any,
        dimension: str,
        scope: BudgetScope,
        scope_id: Optional[str],
        period: BudgetPeriod,
    ) -> SpendBreakdown:
        """Get spending breakdown by a dimension.

        Args:
            actor: The actor requesting the breakdown.
            dimension: Dimension to break down by.
            scope: Budget scope.
            scope_id: Scope identifier.
            period: Time period.

        Returns:
            SpendBreakdown with items by dimension.
        """
        # Use scope_id from actor if not provided
        if not scope_id:
            if scope == BudgetScope.TEAM and hasattr(actor, "team_id"):
                scope_id = actor.team_id
            elif scope == BudgetScope.USER and hasattr(actor, "user_id"):
                scope_id = actor.user_id

        period_start, period_end = self._get_period_boundaries(period)

        # Get from repository if available
        if self._usage_repo:
            try:
                items = await self._usage_repo.get_spend_by_dimension(
                    dimension=dimension,
                    start_date=period_start,
                    end_date=period_end,
                    team_id=scope_id if scope == BudgetScope.TEAM else None,
                )
                total = sum(items.values(), Decimal("0"))
                return SpendBreakdown(
                    dimension=dimension,
                    items=items,
                    total=total,
                )
            except Exception as exc:
                self.logger.warning(
                    "Failed to get breakdown from repo: %s", exc
                )

        # Fall back to buffer aggregation
        items: Dict[str, Decimal] = defaultdict(Decimal)
        
        async with self._buffer_lock:
            records = list(self._usage_buffer)

        for record in records:
            if record.timestamp < period_start or record.timestamp > period_end:
                continue
            if not self._record_matches_scope(record, scope, scope_id):
                continue

            if dimension == "provider":
                items[record.provider] += record.cost_usd
            elif dimension == "model":
                items[record.model] += record.cost_usd
            elif dimension == "operation":
                items[record.operation_type.value] += record.cost_usd
            elif dimension == "user":
                if record.user_id:
                    items[record.user_id] += record.cost_usd

        total = sum(items.values(), Decimal("0"))
        return SpendBreakdown(
            dimension=dimension,
            items=dict(items),
            total=total,
        )

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    async def _update_spend_cache(self, record: UsageRecord) -> None:
        """Update cached spend summaries with a new record."""
        # Invalidate caches that this record affects
        scopes_to_invalidate: List[Tuple[BudgetScope, Optional[str]]] = [
            (BudgetScope.GLOBAL, None),
        ]
        
        if record.team_id:
            scopes_to_invalidate.append((BudgetScope.TEAM, record.team_id))
        if record.user_id:
            scopes_to_invalidate.append((BudgetScope.USER, record.user_id))
        if record.provider:
            scopes_to_invalidate.append((BudgetScope.PROVIDER, record.provider))
        if record.model:
            scopes_to_invalidate.append((BudgetScope.MODEL, record.model))

        for scope, scope_id in scopes_to_invalidate:
            # Invalidate all period caches for this scope
            for period in BudgetPeriod:
                cache_key = self._get_cache_key(scope, scope_id, period)
                self._spend_cache.pop(cache_key, None)
                self._cache_timestamps.pop(cache_key, None)

    async def _check_thresholds(self, actor: Any, record: UsageRecord) -> None:
        """Check if any budget thresholds have been crossed."""
        if not self._policy_repo:
            return

        # Get applicable policies
        policies: List[BudgetPolicy] = []
        
        if record.team_id:
            team_policies = await self._policy_repo.get_policies(
                scope=BudgetScope.TEAM,
                scope_id=record.team_id,
            )
            policies.extend(team_policies)
        
        if record.user_id:
            user_policies = await self._policy_repo.get_policies(
                scope=BudgetScope.USER,
                scope_id=record.user_id,
            )
            policies.extend(user_policies)

        # Also check global policies
        global_policies = await self._policy_repo.get_policies(
            scope=BudgetScope.GLOBAL,
        )
        policies.extend(global_policies)

        for policy in policies:
            await self._check_policy_threshold(actor, record, policy)

    async def _check_policy_threshold(
        self,
        actor: Any,
        record: UsageRecord,
        policy: BudgetPolicy,
    ) -> None:
        """Check if a specific policy threshold has been crossed."""
        # Get current spend for this policy
        period_start, period_end = self._get_period_boundaries(policy.period)
        
        summary = await self._aggregate_spending(
            scope=policy.scope,
            scope_id=policy.scope_id,
            period_start=period_start,
            period_end=period_end,
            policy=policy,
        )
        
        current_percent = summary.percent_used
        threshold_key = policy.id
        
        # Check soft limit threshold
        if current_percent >= policy.soft_limit_percent:
            last_notified = self._threshold_notified.get(threshold_key, 0.0)
            
            # Only notify if we crossed a new 10% boundary
            current_decile = int(current_percent * 10)
            last_decile = int(last_notified * 10)
            
            if current_decile > last_decile:
                self._threshold_notified[threshold_key] = current_percent
                
                # Publish threshold event
                await self._publish_threshold_event(
                    policy=policy,
                    threshold_percent=policy.soft_limit_percent,
                    current_percent=current_percent,
                    current_spend=summary.total_spent,
                )

    async def _find_policy(
        self,
        scope: BudgetScope,
        scope_id: Optional[str],
    ) -> Optional[BudgetPolicy]:
        """Find the applicable budget policy for a scope."""
        if not self._policy_repo:
            return None

        policies = await self._policy_repo.get_policies(
            scope=scope,
            scope_id=scope_id,
            enabled_only=True,
        )
        
        # Return highest priority policy
        if policies:
            return sorted(policies, key=lambda p: -p.priority)[0]
        return None

    async def _aggregate_spending(
        self,
        scope: BudgetScope,
        scope_id: Optional[str],
        period_start: datetime,
        period_end: datetime,
        policy: Optional[BudgetPolicy],
    ) -> SpendSummary:
        """Aggregate spending from buffered and persisted records."""
        total_spent = Decimal("0")
        by_provider: Dict[str, Decimal] = defaultdict(Decimal)
        by_model: Dict[str, Decimal] = defaultdict(Decimal)
        by_operation: Dict[str, Decimal] = defaultdict(Decimal)
        record_count = 0

        # Aggregate from buffer
        async with self._buffer_lock:
            records = list(self._usage_buffer)

        for record in records:
            if record.timestamp < period_start or record.timestamp > period_end:
                continue
            if not self._record_matches_scope(record, scope, scope_id):
                continue

            total_spent += record.cost_usd
            by_provider[record.provider] += record.cost_usd
            by_model[record.model] += record.cost_usd
            by_operation[record.operation_type.value] += record.cost_usd
            record_count += 1

        # Aggregate from repository
        if self._usage_repo:
            try:
                persisted_spend = await self._usage_repo.get_aggregate_spend(
                    start_date=period_start,
                    end_date=period_end,
                    user_id=scope_id if scope == BudgetScope.USER else None,
                    team_id=scope_id if scope == BudgetScope.TEAM else None,
                    provider=scope_id if scope == BudgetScope.PROVIDER else None,
                    model=scope_id if scope == BudgetScope.MODEL else None,
                )
                total_spent += persisted_spend
            except Exception as exc:
                self.logger.warning("Failed to aggregate from repo: %s", exc)

        # Get rollover amount
        rollover_amount = Decimal("0")
        if policy and policy.rollover_enabled:
            rollover_amount = self._rollover_amounts.get(policy.id, Decimal("0"))

        return SpendSummary(
            policy_id=policy.id if policy else "",
            period_start=period_start,
            period_end=period_end,
            total_spent=total_spent,
            limit_amount=policy.limit_amount if policy else Decimal("0"),
            currency=policy.currency if policy else "USD",
            by_provider=dict(by_provider),
            by_model=dict(by_model),
            by_operation=dict(by_operation),
            record_count=record_count,
            rollover_amount=rollover_amount,
        )

    def _record_matches_scope(
        self,
        record: UsageRecord,
        scope: BudgetScope,
        scope_id: Optional[str],
    ) -> bool:
        """Check if a record matches the given scope."""
        if scope == BudgetScope.GLOBAL:
            return True
        elif scope == BudgetScope.TEAM:
            return record.team_id == scope_id
        elif scope == BudgetScope.USER:
            return record.user_id == scope_id
        elif scope == BudgetScope.PROJECT:
            return record.project_id == scope_id
        elif scope == BudgetScope.JOB:
            return record.job_id == scope_id
        elif scope == BudgetScope.TASK:
            return record.task_id == scope_id
        elif scope == BudgetScope.AGENT:
            return record.agent_id == scope_id
        elif scope == BudgetScope.SESSION:
            return record.session_id == scope_id
        elif scope == BudgetScope.PROVIDER:
            return record.provider == scope_id
        elif scope == BudgetScope.MODEL:
            return record.model == scope_id
        elif scope == BudgetScope.TOOL:
            return record.tool_id == scope_id
        elif scope == BudgetScope.SKILL:
            return record.skill_id == scope_id
        return False

    def _get_cache_key(
        self,
        scope: BudgetScope,
        scope_id: Optional[str],
        period: BudgetPeriod,
    ) -> str:
        """Generate a cache key for spending data."""
        return f"{scope.value}:{scope_id or 'all'}:{period.value}"

    def _get_period_boundaries(
        self,
        period: BudgetPeriod,
        start_override: Optional[datetime] = None,
        end_override: Optional[datetime] = None,
    ) -> Tuple[datetime, datetime]:
        """Get start and end timestamps for a budget period."""
        if start_override and end_override:
            return start_override, end_override

        now = datetime.now(timezone.utc)

        if period == BudgetPeriod.HOURLY:
            start = now.replace(minute=0, second=0, microsecond=0)
            end = start + timedelta(hours=1)
        elif period == BudgetPeriod.DAILY:
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(days=1)
        elif period == BudgetPeriod.WEEKLY:
            start = now - timedelta(days=now.weekday())
            start = start.replace(hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(weeks=1)
        elif period == BudgetPeriod.MONTHLY:
            start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            if now.month == 12:
                end = start.replace(year=now.year + 1, month=1)
            else:
                end = start.replace(month=now.month + 1)
        elif period == BudgetPeriod.QUARTERLY:
            quarter_month = ((now.month - 1) // 3) * 3 + 1
            start = now.replace(
                month=quarter_month, day=1, hour=0, minute=0, second=0, microsecond=0
            )
            next_quarter = quarter_month + 3
            if next_quarter > 12:
                end = start.replace(year=now.year + 1, month=next_quarter - 12)
            else:
                end = start.replace(month=next_quarter)
        elif period == BudgetPeriod.YEARLY:
            start = now.replace(
                month=1, day=1, hour=0, minute=0, second=0, microsecond=0
            )
            end = start.replace(year=now.year + 1)
        elif period == BudgetPeriod.ROLLING_24H:
            start = now - timedelta(hours=24)
            end = now
        elif period == BudgetPeriod.ROLLING_7D:
            start = now - timedelta(days=7)
            end = now
        elif period == BudgetPeriod.ROLLING_30D:
            start = now - timedelta(days=30)
            end = now
        elif period == BudgetPeriod.LIFETIME:
            start = datetime(2020, 1, 1, tzinfo=timezone.utc)
            end = now + timedelta(days=365 * 100)
        else:
            # Default to monthly
            start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            if now.month == 12:
                end = start.replace(year=now.year + 1, month=1)
            else:
                end = start.replace(month=now.month + 1)

        return start, end

    def _get_period_at_offset(
        self,
        period: BudgetPeriod,
        offset: int,
    ) -> Tuple[datetime, datetime]:
        """Get period boundaries at a given offset from current.

        Args:
            period: Period type.
            offset: Number of periods back (0 = current).

        Returns:
            Tuple of (start, end) for the period.
        """
        now = datetime.now(timezone.utc)

        if period == BudgetPeriod.DAILY:
            base = now - timedelta(days=offset)
            start = base.replace(hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(days=1)
        elif period == BudgetPeriod.WEEKLY:
            base = now - timedelta(weeks=offset)
            start = base - timedelta(days=base.weekday())
            start = start.replace(hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(weeks=1)
        elif period == BudgetPeriod.MONTHLY:
            # Calculate month offset
            year = now.year
            month = now.month - offset
            while month <= 0:
                month += 12
                year -= 1
            start = datetime(year, month, 1, tzinfo=timezone.utc)
            if month == 12:
                end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
            else:
                end = datetime(year, month + 1, 1, tzinfo=timezone.utc)
        else:
            # Default to daily
            base = now - timedelta(days=offset)
            start = base.replace(hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(days=1)

        return start, end

    # =========================================================================
    # Event Publishing
    # =========================================================================

    async def _publish_usage_event(
        self,
        actor: Any,
        record: UsageRecord,
    ) -> None:
        """Publish a usage recorded event."""
        if not self._event_publisher:
            return

        actor_id = getattr(actor, "user_id", None) or getattr(actor, "id", "system")
        actor_type = getattr(actor, "actor_type", "user")

        event = BudgetUsageRecorded(
            record_id=record.id,
            provider=record.provider,
            model=record.model,
            operation_type=record.operation_type.value,
            cost_usd=record.cost_usd,
            actor_id=actor_id,
            actor_type=actor_type,
            input_tokens=record.input_tokens,
            output_tokens=record.output_tokens,
            images_generated=record.images_generated,
            audio_seconds=record.audio_seconds,
            user_id=record.user_id,
            team_id=record.team_id,
            project_id=record.project_id,
            job_id=record.job_id,
            task_id=record.task_id,
            agent_id=record.agent_id,
            session_id=record.session_id,
            conversation_id=record.conversation_id,
            tool_id=record.tool_id,
            skill_id=record.skill_id,
        )

        try:
            await self._event_publisher(event)
        except Exception as exc:
            self.logger.warning("Failed to publish usage event: %s", exc)

    async def _publish_threshold_event(
        self,
        policy: BudgetPolicy,
        threshold_percent: float,
        current_percent: float,
        current_spend: Decimal,
    ) -> None:
        """Publish a threshold reached event."""
        if not self._event_publisher:
            return

        event = BudgetThresholdReached(
            policy_id=policy.id,
            policy_name=policy.name,
            threshold_percent=threshold_percent,
            current_percent=current_percent,
            current_spend=current_spend,
            limit_amount=policy.limit_amount,
            scope=policy.scope.value,
            scope_id=policy.scope_id,
        )

        try:
            await self._event_publisher(event)
            self.logger.info(
                "Budget threshold reached: %s at %.1f%% (limit: %s)",
                policy.name,
                current_percent * 100,
                policy.limit_amount,
            )
        except Exception as exc:
            self.logger.warning("Failed to publish threshold event: %s", exc)

    # =========================================================================
    # Cache Management
    # =========================================================================

    def invalidate_cache(
        self,
        scope: Optional[BudgetScope] = None,
        scope_id: Optional[str] = None,
    ) -> None:
        """Invalidate spend cache entries.

        Args:
            scope: Specific scope to invalidate, or None for all.
            scope_id: Specific scope_id to invalidate.
        """
        if scope is None:
            self._spend_cache.clear()
            self._cache_timestamps.clear()
        else:
            keys_to_remove = []
            for key in self._spend_cache:
                if key.startswith(f"{scope.value}:{scope_id or ''}"):
                    keys_to_remove.append(key)
            for key in keys_to_remove:
                self._spend_cache.pop(key, None)
                self._cache_timestamps.pop(key, None)

    def set_rollover_amount(self, policy_id: str, amount: Decimal) -> None:
        """Set the rollover amount for a policy.

        Args:
            policy_id: Policy ID.
            amount: Rollover amount.
        """
        self._rollover_amounts[policy_id] = _quantize_decimal(amount)
