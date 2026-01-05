"""BudgetManager - Singleton orchestrator for budget tracking and enforcement.

Provides centralized budget management following the same patterns as
ProviderManager and MediaProviderManager:
- Async singleton factory
- Policy management
- Usage tracking
- Spending summaries
- Budget enforcement
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from modules.logging.logger import setup_logger

from .models import (
    BudgetAlert,
    AlertSeverity,
    AlertTriggerType,
    BudgetCheckResult,
    BudgetPeriod,
    BudgetPolicy,
    BudgetScope,
    LimitAction,
    OperationType,
    SpendSummary,
    UsageRecord,
)
from .pricing import get_pricing_registry, PricingRegistry

if TYPE_CHECKING:
    from ATLAS.config import ConfigManager

logger = setup_logger(__name__)

# Module-level singleton state
_budget_manager_instance: Optional["BudgetManager"] = None
_budget_manager_lock: Optional[asyncio.Lock] = None


async def get_budget_manager(
    config_manager: Optional["ConfigManager"] = None,
) -> "BudgetManager":
    """Get the global BudgetManager singleton.

    Creates and initializes the manager on first call.
    Subsequent calls return the same instance.

    Args:
        config_manager: Configuration manager (required on first call).

    Returns:
        Initialized BudgetManager instance.

    Raises:
        ValueError: If config_manager not provided on first call.
    """
    global _budget_manager_instance, _budget_manager_lock

    if _budget_manager_instance is not None:
        return _budget_manager_instance

    if _budget_manager_lock is None:
        _budget_manager_lock = asyncio.Lock()

    async with _budget_manager_lock:
        if _budget_manager_instance is None:
            if config_manager is None:
                raise ValueError(
                    "config_manager required for first BudgetManager initialization"
                )
            _budget_manager_instance = await BudgetManager.create(config_manager)

    return _budget_manager_instance


def get_budget_manager_sync() -> Optional["BudgetManager"]:
    """Get BudgetManager if already initialized (non-async).

    Returns None if not yet initialized. Use get_budget_manager()
    for guaranteed initialization.
    """
    return _budget_manager_instance


async def reset_budget_manager() -> None:
    """Reset the singleton instance (primarily for testing)."""
    global _budget_manager_instance
    if _budget_manager_instance is not None:
        await _budget_manager_instance.shutdown()
        _budget_manager_instance = None


class BudgetManager:
    """Manages budget policies, usage tracking, and enforcement.

    Provides:
    - Singleton pattern with async factory
    - Budget policy CRUD operations
    - Real-time usage recording
    - Spending summaries and analytics
    - Pre-request budget checks
    - Alert generation

    Usage::

        manager = await get_budget_manager(config_manager)

        # Set a budget policy
        policy = BudgetPolicy(
            name="Monthly Global",
            scope=BudgetScope.GLOBAL,
            limit_amount=Decimal("100.00"),
        )
        await manager.set_budget_policy(policy)

        # Record usage
        await manager.record_usage(usage_record)

        # Check budget before request
        result = await manager.check_budget_available(
            provider="OpenAI",
            model="gpt-4o",
            estimated_cost=Decimal("0.05"),
        )
    """

    _instance: Optional["BudgetManager"] = None
    _lock: Optional[asyncio.Lock] = None

    def __init__(self, config_manager: "ConfigManager"):
        """Private constructor - use create() factory method.

        Args:
            config_manager: ATLAS configuration manager.
        """
        self.config_manager = config_manager
        self.logger = setup_logger(__name__)

        # Budget policies by ID
        self._policies: Dict[str, BudgetPolicy] = {}

        # Usage records (in-memory buffer, periodically flushed to storage)
        self._usage_records: List[UsageRecord] = []
        self._usage_lock = asyncio.Lock()

        # Spending aggregates (cached for performance)
        self._spend_cache: Dict[str, SpendSummary] = {}
        self._cache_ttl_seconds = 60
        self._cache_timestamps: Dict[str, datetime] = {}

        # Alerts
        self._alerts: List[BudgetAlert] = []
        self._alert_lock = asyncio.Lock()

        # Rollover amounts by policy ID (persisted separately)
        self._rollover_amounts: Dict[str, Decimal] = {}

        # Pricing registry
        self._pricing: Optional[PricingRegistry] = None

        # Persistence layer (lazy initialized)
        self._persistence: Optional[Any] = None

        # Event callbacks
        self._on_budget_warning: List[Callable] = []
        self._on_budget_exceeded: List[Callable] = []
        self._on_usage_recorded: List[Callable] = []

        # Configuration
        self._enabled = True
        self._flush_interval = 300  # 5 minutes
        self._flush_task: Optional[asyncio.Task] = None

    @classmethod
    async def create(cls, config_manager: "ConfigManager") -> "BudgetManager":
        """Async factory method to create or retrieve singleton instance.

        Args:
            config_manager: ATLAS configuration manager.

        Returns:
            Initialized BudgetManager singleton.
        """
        if cls._lock is None:
            cls._lock = asyncio.Lock()

        async with cls._lock:
            if cls._instance is None:
                instance = cls(config_manager)
                await instance._initialize()
                cls._instance = instance
                logger.info("BudgetManager singleton created")
            return cls._instance

    async def _initialize(self) -> None:
        """Initialize the manager (called once during creation)."""
        # Load configuration
        self._load_config()

        # Initialize persistence layer
        await self._init_persistence()

        # Initialize pricing registry
        self._pricing = await get_pricing_registry(self.config_manager)

        # Load policies from storage
        await self._load_policies()

        # Load rollover amounts from storage
        await self._load_rollovers()

        # Load existing alerts
        await self._load_alerts()

        # Start background flush task
        self._start_flush_task()

        self.logger.info(
            "BudgetManager initialized with %d policies, enabled=%s",
            len(self._policies),
            self._enabled,
        )

    async def _init_persistence(self) -> None:
        """Initialize the persistence layer via StorageManager."""
        try:
            from modules.storage import get_storage_manager_sync
            from modules.storage.adapters import create_budget_store

            storage = get_storage_manager_sync()
            if storage is not None and storage.is_initialized:
                self._persistence = create_budget_store(storage)
                await self._persistence.initialize()
                self.logger.debug("Persistence layer initialized via StorageManager")
            else:
                # StorageManager not available, use in-memory fallback
                from .persistence import InMemoryBudgetStore
                self._persistence = InMemoryBudgetStore()
                await self._persistence.initialize()
                self.logger.info("Using in-memory budget store (StorageManager not available)")
        except Exception as exc:
            self.logger.warning("Failed to initialize persistence: %s", exc)
            # Fall back to in-memory store
            try:
                from .persistence import InMemoryBudgetStore
                self._persistence = InMemoryBudgetStore()
                await self._persistence.initialize()
            except Exception:
                self._persistence = None

    def _load_config(self) -> None:
        """Load budget configuration from config manager."""
        try:
            budget_config = self.config_manager.get_config("BUDGET")
            if isinstance(budget_config, dict):
                self._enabled = budget_config.get("enabled", True)
                self._flush_interval = budget_config.get("flush_interval", 300)
                self._cache_ttl_seconds = budget_config.get("cache_ttl", 60)
        except Exception as exc:
            self.logger.warning("Failed to load budget config: %s", exc)

    async def _load_policies(self) -> None:
        """Load budget policies from storage."""
        # First, load from persistence layer
        if self._persistence is not None:
            try:
                persisted_policies = await self._persistence.get_policies()
                for policy in persisted_policies:
                    self._policies[policy.id] = policy
                self.logger.debug("Loaded %d policies from storage", len(persisted_policies))
            except Exception as exc:
                self.logger.warning("Failed to load policies from storage: %s", exc)

        # Then, load from config (these override/supplement persisted policies)
        try:
            policies_config = self.config_manager.get_config("BUDGET_POLICIES")
            if isinstance(policies_config, list):
                for policy_data in policies_config:
                    if isinstance(policy_data, dict):
                        policy = BudgetPolicy.from_dict(policy_data)
                        self._policies[policy.id] = policy
        except Exception as exc:
            self.logger.warning("Failed to load budget policies from config: %s", exc)

    async def _load_alerts(self) -> None:
        """Load unresolved alerts from storage."""
        if self._persistence is not None:
            try:
                alerts = await self._persistence.get_alerts(active_only=True)
                async with self._alert_lock:
                    self._alerts = list(alerts)
                self.logger.debug("Loaded %d active alerts from storage", len(alerts))
            except Exception as exc:
                self.logger.warning("Failed to load alerts from storage: %s", exc)

    async def _load_rollovers(self) -> None:
        """Load rollover amounts from storage."""
        # Rollovers are currently managed in-memory per session
        # Future enhancement: persist rollover amounts to budget_rollovers table
        pass

    async def _save_rollover(self, policy_id: str, amount: Decimal) -> None:
        """Save rollover amount for a policy.

        Args:
            policy_id: Policy identifier.
            amount: Rollover amount.
        """
        self._rollover_amounts[policy_id] = amount
        # Rollover amounts stored in-memory for this session
        # Future enhancement: persist to budget_rollovers table

    def _start_flush_task(self) -> None:
        """Start background task to periodically flush usage records."""
        if self._flush_task is not None:
            return

        async def flush_loop():
            while True:
                await asyncio.sleep(self._flush_interval)
                try:
                    await self._flush_usage_records()
                except Exception as exc:
                    self.logger.error("Failed to flush usage records: %s", exc)

        self._flush_task = asyncio.create_task(flush_loop())

    async def _flush_usage_records(self) -> None:
        """Flush buffered usage records to storage."""
        async with self._usage_lock:
            if not self._usage_records:
                return

            records_to_flush = self._usage_records.copy()
            self._usage_records.clear()

        # Write to persistence layer
        if self._persistence is not None:
            try:
                saved = await self._persistence.save_usage_records(records_to_flush)
                self.logger.debug("Flushed %d usage records to storage", saved)
            except Exception as exc:
                self.logger.error("Failed to flush usage records: %s", exc)
                # Re-add records on failure so they're not lost
                async with self._usage_lock:
                    self._usage_records = records_to_flush + self._usage_records
        else:
            self.logger.debug("Flushed %d usage records (in-memory)", len(records_to_flush))

    async def shutdown(self) -> None:
        """Shutdown the manager and flush pending data."""
        if self._flush_task is not None:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
            self._flush_task = None

        await self._flush_usage_records()
        self.logger.info("BudgetManager shutdown complete")

    # =========================================================================
    # Policy Management
    # =========================================================================

    async def set_budget_policy(self, policy: BudgetPolicy) -> BudgetPolicy:
        """Create or update a budget policy.

        Args:
            policy: The policy to create or update.

        Returns:
            The saved policy.
        """
        # Update timestamp
        policy = replace(policy, updated_at=datetime.now(timezone.utc))

        self._policies[policy.id] = policy

        # Invalidate related cache entries
        self._invalidate_cache_for_policy(policy)

        # Persist to storage
        if self._persistence is not None:
            try:
                await self._persistence.save_policy(policy)
            except Exception as exc:
                self.logger.warning("Failed to persist policy: %s", exc)

        self.logger.info("Saved budget policy: %s (%s)", policy.name, policy.id)

        return policy

    async def get_policy(self, policy_id: str) -> Optional[BudgetPolicy]:
        """Get a budget policy by ID.

        Args:
            policy_id: Policy identifier.

        Returns:
            BudgetPolicy if found, None otherwise.
        """
        return self._policies.get(policy_id)

    async def get_policies(
        self,
        scope: Optional[BudgetScope] = None,
        scope_id: Optional[str] = None,
        enabled_only: bool = True,
    ) -> List[BudgetPolicy]:
        """Get budget policies matching criteria.

        Args:
            scope: Filter by scope level.
            scope_id: Filter by scope identifier.
            enabled_only: Only return enabled policies.

        Returns:
            List of matching policies, ordered by priority.
        """
        policies = list(self._policies.values())

        if scope is not None:
            policies = [p for p in policies if p.scope == scope]

        if scope_id is not None:
            policies = [p for p in policies if p.scope_id == scope_id]

        if enabled_only:
            policies = [p for p in policies if p.enabled]

        # Sort by priority (higher first)
        policies.sort(key=lambda p: p.priority, reverse=True)

        return policies

    async def delete_policy(self, policy_id: str) -> bool:
        """Delete a budget policy.

        Args:
            policy_id: Policy identifier.

        Returns:
            True if deleted, False if not found.
        """
        if policy_id not in self._policies:
            return False

        policy = self._policies.pop(policy_id)
        self._invalidate_cache_for_policy(policy)

        # Delete from storage
        if self._persistence is not None:
            try:
                await self._persistence.delete_policy(policy_id)
            except Exception as exc:
                self.logger.warning("Failed to delete policy from storage: %s", exc)

        self.logger.info("Deleted budget policy: %s", policy_id)

        return True

    def _invalidate_cache_for_policy(self, policy: BudgetPolicy) -> None:
        """Invalidate cached spending data for a policy."""
        cache_key = self._get_cache_key(policy.scope, policy.scope_id)
        self._spend_cache.pop(cache_key, None)
        self._cache_timestamps.pop(cache_key, None)

    # =========================================================================
    # Usage Recording
    # =========================================================================

    async def record_usage(self, record: UsageRecord) -> None:
        """Record a usage event.

        Args:
            record: The usage record to store.
        """
        if not self._enabled:
            return

        async with self._usage_lock:
            self._usage_records.append(record)

        # Update cached spend summaries
        await self._update_spend_cache(record)

        # Check thresholds and generate alerts
        await self._check_thresholds(record)

        # Notify listeners
        for callback in self._on_usage_recorded:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(record)
                else:
                    callback(record)
            except Exception as exc:
                self.logger.warning("Usage callback error: %s", exc)

    async def record_llm_usage(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        persona: Optional[str] = None,
        conversation_id: Optional[str] = None,
        request_id: Optional[str] = None,
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> UsageRecord:
        """Convenience method to record LLM usage with cost calculation.

        Args:
            provider: Provider name.
            model: Model identifier.
            input_tokens: Input token count.
            output_tokens: Output token count.
            cached_tokens: Cached token count.
            user_id: User identifier.
            tenant_id: Tenant identifier.
            persona: Persona name.
            conversation_id: Conversation identifier.
            request_id: Request identifier.
            success: Whether request succeeded.
            metadata: Additional metadata.

        Returns:
            The recorded UsageRecord.
        """
        # Calculate cost
        cost = Decimal("0")
        if self._pricing:
            cost = self._pricing.calculate_llm_cost(
                model, input_tokens, output_tokens, cached_tokens
            )

        record = UsageRecord(
            provider=provider,
            model=model,
            operation_type=OperationType.CHAT_COMPLETION,
            cost_usd=cost,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            user_id=user_id,
            tenant_id=tenant_id,
            persona=persona,
            conversation_id=conversation_id,
            request_id=request_id,
            success=success,
            metadata=metadata or {},
        )

        await self.record_usage(record)
        return record

    async def record_image_usage(
        self,
        provider: str,
        model: str,
        count: int = 1,
        size: str = "1024x1024",
        quality: str = "standard",
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        persona: Optional[str] = None,
        conversation_id: Optional[str] = None,
        request_id: Optional[str] = None,
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> UsageRecord:
        """Convenience method to record image generation usage.

        Args:
            provider: Provider name.
            model: Model identifier.
            count: Number of images generated.
            size: Image size.
            quality: Image quality.
            user_id: User identifier.
            tenant_id: Tenant identifier.
            persona: Persona name.
            conversation_id: Conversation identifier.
            request_id: Request identifier.
            success: Whether request succeeded.
            metadata: Additional metadata.

        Returns:
            The recorded UsageRecord.
        """
        # Calculate cost
        cost = Decimal("0")
        if self._pricing:
            cost = self._pricing.calculate_image_cost(model, count, size, quality)

        record = UsageRecord(
            provider=provider,
            model=model,
            operation_type=OperationType.IMAGE_GENERATION,
            cost_usd=cost,
            images_generated=count,
            image_size=size,
            image_quality=quality,
            user_id=user_id,
            tenant_id=tenant_id,
            persona=persona,
            conversation_id=conversation_id,
            request_id=request_id,
            success=success,
            metadata=metadata or {},
        )

        await self.record_usage(record)
        return record

    # =========================================================================
    # Spending Queries
    # =========================================================================

    async def get_current_spend(
        self,
        scope: BudgetScope = BudgetScope.GLOBAL,
        scope_id: Optional[str] = None,
        period: BudgetPeriod = BudgetPeriod.MONTHLY,
    ) -> SpendSummary:
        """Get current spending for a budget scope.

        Args:
            scope: Budget scope level.
            scope_id: Scope identifier.
            period: Time period for aggregation.

        Returns:
            SpendSummary with current spending data.
        """
        cache_key = self._get_cache_key(scope, scope_id, period)

        # Check cache
        if cache_key in self._spend_cache:
            cached_at = self._cache_timestamps.get(cache_key)
            if cached_at and (datetime.now(timezone.utc) - cached_at).total_seconds() < self._cache_ttl_seconds:
                return self._spend_cache[cache_key]

        # Calculate period boundaries
        period_start, period_end = self._get_period_boundaries(period)

        # Find applicable policy
        policies = await self.get_policies(scope=scope, scope_id=scope_id)
        policy = policies[0] if policies else None

        # Aggregate spending
        summary = await self._aggregate_spending(
            scope=scope,
            scope_id=scope_id,
            period_start=period_start,
            period_end=period_end,
            policy=policy,
        )

        # Cache result
        self._spend_cache[cache_key] = summary
        self._cache_timestamps[cache_key] = datetime.now(timezone.utc)

        return summary

    async def _aggregate_spending(
        self,
        scope: BudgetScope,
        scope_id: Optional[str],
        period_start: datetime,
        period_end: datetime,
        policy: Optional[BudgetPolicy],
    ) -> SpendSummary:
        """Aggregate spending from usage records.

        Args:
            scope: Budget scope.
            scope_id: Scope identifier.
            period_start: Start of period.
            period_end: End of period.
            policy: Applicable budget policy.

        Returns:
            SpendSummary with aggregated data.
        """
        total_spent = Decimal("0")
        by_provider: Dict[str, Decimal] = defaultdict(Decimal)
        by_model: Dict[str, Decimal] = defaultdict(Decimal)
        by_operation: Dict[str, Decimal] = defaultdict(Decimal)
        record_count = 0

        # Filter records for this scope and period
        async with self._usage_lock:
            records = list(self._usage_records)

        for record in records:
            # Check time period
            if record.timestamp < period_start or record.timestamp > period_end:
                continue

            # Check scope
            if not self._record_matches_scope(record, scope, scope_id):
                continue

            # Aggregate
            total_spent += record.cost_usd
            by_provider[record.provider] += record.cost_usd
            by_model[record.model] += record.cost_usd
            by_operation[record.operation_type.value] += record.cost_usd
            record_count += 1

        # Also aggregate from persisted records
        if self._persistence is not None:
            try:
                # Build filters based on scope
                user_id = scope_id if scope == BudgetScope.USER else None
                tenant_id = scope_id if scope == BudgetScope.TENANT else None
                provider = scope_id if scope == BudgetScope.PROVIDER else None
                model = scope_id if scope == BudgetScope.MODEL else None

                # Get aggregate spend from persistence
                persisted_spend = await self._persistence.get_aggregate_spend(
                    start_date=period_start,
                    end_date=period_end,
                    user_id=user_id,
                    tenant_id=tenant_id,
                    provider=provider,
                    model=model,
                )
                total_spent += persisted_spend
            except Exception as exc:
                self.logger.warning("Failed to aggregate from persisted records: %s", exc)

        # Get rollover amount for this policy
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
        elif scope == BudgetScope.TENANT:
            return record.tenant_id == scope_id
        elif scope == BudgetScope.USER:
            return record.user_id == scope_id
        elif scope == BudgetScope.PERSONA:
            return record.persona == scope_id
        elif scope == BudgetScope.PROVIDER:
            return record.provider == scope_id
        elif scope == BudgetScope.MODEL:
            return record.model == scope_id
        return False

    def _get_cache_key(
        self,
        scope: BudgetScope,
        scope_id: Optional[str],
        period: Optional[BudgetPeriod] = None,
    ) -> str:
        """Generate a cache key for spending data."""
        parts = [scope.value, scope_id or "all"]
        if period:
            parts.append(period.value)
        return ":".join(parts)

    def _get_period_boundaries(
        self,
        period: BudgetPeriod,
    ) -> tuple[datetime, datetime]:
        """Get start and end timestamps for a budget period."""
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
            # First day of next month
            if now.month == 12:
                end = start.replace(year=now.year + 1, month=1)
            else:
                end = start.replace(month=now.month + 1)
        elif period == BudgetPeriod.QUARTERLY:
            quarter_month = ((now.month - 1) // 3) * 3 + 1
            start = now.replace(month=quarter_month, day=1, hour=0, minute=0, second=0, microsecond=0)
            next_quarter = quarter_month + 3
            if next_quarter > 12:
                end = start.replace(year=now.year + 1, month=next_quarter - 12)
            else:
                end = start.replace(month=next_quarter)
        elif period == BudgetPeriod.YEARLY:
            start = now.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
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
            end = now + timedelta(days=365 * 100)  # Far future
        else:
            # Default to monthly
            start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            if now.month == 12:
                end = start.replace(year=now.year + 1, month=1)
            else:
                end = start.replace(month=now.month + 1)

        return start, end

    def _get_previous_period_boundaries(
        self,
        period: BudgetPeriod,
    ) -> tuple[datetime, datetime]:
        """Get start and end timestamps for the previous budget period.

        Args:
            period: The budget period type.

        Returns:
            Tuple of (previous_start, previous_end) datetimes.
        """
        current_start, current_end = self._get_period_boundaries(period)

        if period == BudgetPeriod.HOURLY:
            prev_start = current_start - timedelta(hours=1)
            prev_end = current_start
        elif period == BudgetPeriod.DAILY:
            prev_start = current_start - timedelta(days=1)
            prev_end = current_start
        elif period == BudgetPeriod.WEEKLY:
            prev_start = current_start - timedelta(weeks=1)
            prev_end = current_start
        elif period == BudgetPeriod.MONTHLY:
            # Go back one month
            if current_start.month == 1:
                prev_start = current_start.replace(year=current_start.year - 1, month=12)
            else:
                prev_start = current_start.replace(month=current_start.month - 1)
            prev_end = current_start
        elif period == BudgetPeriod.QUARTERLY:
            # Go back 3 months
            prev_end = current_start
            if current_start.month <= 3:
                prev_start = current_start.replace(
                    year=current_start.year - 1,
                    month=current_start.month + 9
                )
            else:
                prev_start = current_start.replace(month=current_start.month - 3)
        elif period == BudgetPeriod.YEARLY:
            prev_start = current_start.replace(year=current_start.year - 1)
            prev_end = current_start
        else:
            # Rolling periods don't have distinct previous periods
            # Return the same as current (no rollover applicable)
            prev_start = current_start
            prev_end = current_end

        return prev_start, prev_end

    async def calculate_rollover(
        self,
        policy: BudgetPolicy,
    ) -> Decimal:
        """Calculate rollover amount from the previous period.

        Calculates unused budget from the previous period, capped at
        the policy's rollover_max_percent.

        Args:
            policy: Budget policy with rollover settings.

        Returns:
            Rollover amount (0 if rollover disabled or no unused budget).
        """
        if not policy.rollover_enabled:
            return Decimal("0")

        # Rolling periods don't support rollover
        if policy.period in (
            BudgetPeriod.ROLLING_24H,
            BudgetPeriod.ROLLING_7D,
            BudgetPeriod.ROLLING_30D,
            BudgetPeriod.LIFETIME,
        ):
            return Decimal("0")

        # Get previous period spending
        prev_start, prev_end = self._get_previous_period_boundaries(policy.period)

        prev_summary = await self._aggregate_spending(
            scope=policy.scope,
            scope_id=policy.scope_id,
            period_start=prev_start,
            period_end=prev_end,
            policy=policy,
        )

        # Calculate unused amount
        unused = policy.limit_amount - prev_summary.total_spent

        if unused <= 0:
            return Decimal("0")

        # Cap at max rollover percent
        max_rollover = policy.limit_amount * Decimal(str(policy.rollover_max_percent))
        rollover = min(unused, max_rollover)

        self.logger.debug(
            "Calculated rollover for policy %s: $%.2f (unused: $%.2f, max: $%.2f)",
            policy.id, rollover, unused, max_rollover
        )

        return rollover

    async def get_rollover_amount(self, policy_id: str) -> Decimal:
        """Get the current rollover amount for a policy.

        Args:
            policy_id: Policy identifier.

        Returns:
            Rollover amount or 0 if none.
        """
        return self._rollover_amounts.get(policy_id, Decimal("0"))

    async def process_period_end(self, policy: BudgetPolicy) -> Decimal:
        """Process end of period and calculate rollover for next period.

        Call this method when a budget period ends to finalize spending
        and calculate rollover for the next period.

        Args:
            policy: Budget policy to process.

        Returns:
            Rollover amount for next period.
        """
        rollover = await self.calculate_rollover(policy)
        await self._save_rollover(policy.id, rollover)

        if rollover > 0:
            self.logger.info(
                "Period end processed for policy %s: $%.2f rollover to next period",
                policy.name, rollover
            )

        # Invalidate cache for this policy
        self._invalidate_cache_for_policy(policy)

        return rollover

    async def _update_spend_cache(self, record: UsageRecord) -> None:
        """Update cached spending summaries with a new record."""
        # Update global cache
        global_key = self._get_cache_key(BudgetScope.GLOBAL, None)
        if global_key in self._spend_cache:
            summary = self._spend_cache[global_key]
            # Increment totals (simplified - full update would recalculate)
            new_summary = SpendSummary(
                policy_id=summary.policy_id,
                period_start=summary.period_start,
                period_end=summary.period_end,
                total_spent=summary.total_spent + record.cost_usd,
                limit_amount=summary.limit_amount,
                currency=summary.currency,
                by_provider={
                    **summary.by_provider,
                    record.provider: summary.by_provider.get(record.provider, Decimal("0")) + record.cost_usd,
                },
                by_model={
                    **summary.by_model,
                    record.model: summary.by_model.get(record.model, Decimal("0")) + record.cost_usd,
                },
                by_operation={
                    **summary.by_operation,
                    record.operation_type.value: summary.by_operation.get(record.operation_type.value, Decimal("0")) + record.cost_usd,
                },
                record_count=summary.record_count + 1,
            )
            self._spend_cache[global_key] = new_summary

    # =========================================================================
    # Budget Checks
    # =========================================================================

    async def check_budget_available(
        self,
        provider: str,
        model: str,
        estimated_cost: Optional[Decimal] = None,
        estimated_input_tokens: int = 0,
        estimated_output_tokens: int = 0,
        image_count: int = 0,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> BudgetCheckResult:
        """Check if budget is available for a request.

        Args:
            provider: Provider name.
            model: Model identifier.
            estimated_cost: Pre-calculated estimated cost.
            estimated_input_tokens: Expected input tokens (for LLM).
            estimated_output_tokens: Expected output tokens (for LLM).
            image_count: Number of images to generate.
            user_id: User identifier.
            tenant_id: Tenant identifier.

        Returns:
            BudgetCheckResult indicating if request is allowed.
        """
        if not self._enabled:
            return BudgetCheckResult(
                allowed=True,
                action=LimitAction.WARN,
            )

        # Calculate estimated cost if not provided
        if estimated_cost is None and self._pricing:
            estimated_cost = self._pricing.estimate_request_cost(
                provider=provider,
                model=model,
                estimated_input_tokens=estimated_input_tokens,
                estimated_output_tokens=estimated_output_tokens,
                image_count=image_count,
            )

        estimated_cost = estimated_cost or Decimal("0")

        # Check all applicable policies
        warnings: List[str] = []
        most_restrictive_action = LimitAction.WARN
        blocking_policy: Optional[BudgetPolicy] = None

        # Check policies in priority order
        scopes_to_check = [
            (BudgetScope.GLOBAL, None),
            (BudgetScope.PROVIDER, provider),
            (BudgetScope.MODEL, model),
        ]

        if tenant_id:
            scopes_to_check.append((BudgetScope.TENANT, tenant_id))
        if user_id:
            scopes_to_check.append((BudgetScope.USER, user_id))

        for scope, scope_id in scopes_to_check:
            policies = await self.get_policies(scope=scope, scope_id=scope_id)

            for policy in policies:
                summary = await self.get_current_spend(
                    scope=policy.scope,
                    scope_id=policy.scope_id,
                    period=policy.period,
                )

                remaining = summary.remaining
                percent_used = summary.percent_used

                # Check if over soft limit
                if percent_used >= policy.soft_limit_percent:
                    warnings.append(
                        f"{policy.name}: {percent_used:.1%} of budget used "
                        f"(${summary.total_spent:.2f} / ${policy.limit_amount:.2f})"
                    )

                # Check if request would exceed limit
                if estimated_cost > remaining:
                    if policy.hard_limit_action.value > most_restrictive_action.value:
                        most_restrictive_action = policy.hard_limit_action
                        blocking_policy = policy

        # Determine if allowed
        allowed = most_restrictive_action in (LimitAction.WARN, LimitAction.THROTTLE)

        # Find cheaper alternative if blocked
        alternative_model = None
        if not allowed and self._pricing:
            alt = self._pricing.get_cheaper_alternative(model, provider)
            if alt:
                alternative_model = alt[0]

        # Get current spend for response
        current_summary = await self.get_current_spend()

        return BudgetCheckResult(
            allowed=allowed,
            action=most_restrictive_action,
            policy_id=blocking_policy.id if blocking_policy else None,
            current_spend=current_summary.total_spent,
            limit_amount=current_summary.limit_amount,
            estimated_cost=estimated_cost,
            remaining_after=current_summary.remaining - estimated_cost,
            warnings=warnings,
            alternative_model=alternative_model,
        )

    # =========================================================================
    # Alert Management
    # =========================================================================

    async def _check_thresholds(self, record: UsageRecord) -> None:
        """Check budget thresholds after recording usage."""
        for policy in self._policies.values():
            if not policy.enabled:
                continue

            summary = await self.get_current_spend(
                scope=policy.scope,
                scope_id=policy.scope_id,
                period=policy.period,
            )

            percent_used = summary.percent_used

            # Check for threshold alerts
            thresholds = [0.5, policy.soft_limit_percent, 0.9, 1.0]
            for threshold in thresholds:
                if percent_used >= threshold:
                    await self._maybe_create_alert(policy, summary, threshold)

    async def _maybe_create_alert(
        self,
        policy: BudgetPolicy,
        summary: SpendSummary,
        threshold: float,
    ) -> None:
        """Create an alert if one doesn't already exist for this threshold."""
        # Check if alert already exists
        async with self._alert_lock:
            for alert in self._alerts:
                if (
                    alert.policy_id == policy.id
                    and alert.threshold_percent == threshold
                    and not alert.resolved
                ):
                    return  # Alert already exists

        # Determine severity
        if threshold >= 1.0:
            severity = AlertSeverity.CRITICAL
            trigger = AlertTriggerType.LIMIT_EXCEEDED
            message = f"Budget exceeded: ${summary.total_spent:.2f} / ${policy.limit_amount:.2f}"
        elif threshold >= policy.soft_limit_percent:
            severity = AlertSeverity.WARNING
            trigger = AlertTriggerType.THRESHOLD_REACHED
            message = f"Budget warning: {summary.percent_used:.1%} used (${summary.total_spent:.2f} / ${policy.limit_amount:.2f})"
        else:
            severity = AlertSeverity.INFO
            trigger = AlertTriggerType.THRESHOLD_REACHED
            message = f"Budget checkpoint: {summary.percent_used:.1%} used"

        alert = BudgetAlert(
            policy_id=policy.id,
            severity=severity,
            trigger_type=trigger,
            threshold_percent=threshold,
            current_spend=summary.total_spent,
            limit_amount=policy.limit_amount,
            message=message,
        )

        async with self._alert_lock:
            self._alerts.append(alert)

        self.logger.warning("Budget alert created: %s", message)

        # Notify listeners
        callbacks = self._on_budget_exceeded if severity == AlertSeverity.CRITICAL else self._on_budget_warning
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert)
                else:
                    callback(alert)
            except Exception as exc:
                self.logger.warning("Alert callback error: %s", exc)

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
            alerts = [a for a in self._alerts if not a.resolved]

        if policy_id:
            alerts = [a for a in alerts if a.policy_id == policy_id]

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        if unacknowledged_only:
            alerts = [a for a in alerts if not a.acknowledged]

        # Sort by severity (critical first) then by time (newest first)
        severity_order = {
            AlertSeverity.EMERGENCY: 0,
            AlertSeverity.CRITICAL: 1,
            AlertSeverity.WARNING: 2,
            AlertSeverity.INFO: 3,
        }
        alerts.sort(key=lambda a: (severity_order.get(a.severity, 99), -a.triggered_at.timestamp()))

        return alerts

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
            for alert in self._alerts:
                if alert.id == alert_id:
                    alert.acknowledge(user_id)
                    return True
        return False

    async def resolve_alert(self, alert_id: str) -> bool:
        """Resolve (close) an alert.

        Args:
            alert_id: Alert identifier.

        Returns:
            True if resolved, False if not found.
        """
        async with self._alert_lock:
            for alert in self._alerts:
                if alert.id == alert_id:
                    alert.resolve()
                    return True
        return False

    # =========================================================================
    # Event Hooks
    # =========================================================================

    def on_budget_warning(self, callback: Callable) -> None:
        """Register callback for budget warning events."""
        self._on_budget_warning.append(callback)

    def on_budget_exceeded(self, callback: Callable) -> None:
        """Register callback for budget exceeded events."""
        self._on_budget_exceeded.append(callback)

    def on_usage_recorded(self, callback: Callable) -> None:
        """Register callback for usage recorded events."""
        self._on_usage_recorded.append(callback)

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def enabled(self) -> bool:
        """Whether budget tracking is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable or disable budget tracking."""
        self._enabled = value
        self.logger.info("Budget tracking %s", "enabled" if value else "disabled")

    @property
    def pricing_registry(self) -> Optional[PricingRegistry]:
        """Get the pricing registry."""
        return self._pricing
