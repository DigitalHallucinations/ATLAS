"""Public API for the budget module.

Provides high-level functions for budget management that can be
used by other modules without directly accessing internal classes.

Usage::

    from modules.budget import (
        record_llm_usage,
        record_image_usage,
        check_budget,
        get_spend_summary,
        get_usage_report,
        get_alerts,
        acknowledge_alert,
        set_budget_policy,
    )

    # Record usage
    await record_llm_usage(
        provider="openai",
        model="gpt-4o",
        input_tokens=1000,
        output_tokens=500,
        user_id="user_123",
    )

    # Check budget before expensive operation
    result = await check_budget(
        user_id="user_123",
        estimated_cost=Decimal("0.50"),
    )
    if result.allowed:
        # Proceed with operation
        pass

    # Get spending summary
    summary = await get_spend_summary(user_id="user_123")
    print(f"This month: ${summary.total_cost:.2f}")
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from .alerts import AlertEngine, AlertRule, get_alert_engine
from .manager import BudgetManager, get_budget_manager, get_budget_manager_sync
from .models import (
    AlertSeverity,
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
from .pricing import PricingRegistry, get_pricing_registry
from .reports import (
    ExportFormat,
    ReportGenerator,
    ReportGrouping,
    UsageReport,
)
from .tracking import UsageTracker, TrackingContext, get_usage_tracker


# =============================================================================
# Usage Recording API
# =============================================================================


async def record_llm_usage(
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
    """Record LLM token usage.

    Args:
        provider: LLM provider name.
        model: Model identifier.
        input_tokens: Number of input tokens.
        output_tokens: Number of output tokens.
        cached_tokens: Cached tokens (if any).
        user_id: User identifier.
        tenant_id: Tenant identifier.
        persona: Persona name.
        conversation_id: Conversation identifier.
        request_id: Unique request identifier.
        success: Whether the request succeeded.
        metadata: Additional metadata.

    Returns:
        Created UsageRecord.
    """
    manager = await get_budget_manager()
    return await manager.record_llm_usage(
        provider=provider,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_tokens=cached_tokens,
        user_id=user_id,
        tenant_id=tenant_id,
        persona=persona,
        conversation_id=conversation_id,
        request_id=request_id,
        success=success,
        metadata=metadata,
    )


async def record_image_usage(
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
    """Record image generation usage.

    Args:
        provider: Image provider name.
        model: Model/engine identifier.
        count: Number of images generated.
        size: Image size (e.g., "1024x1024").
        quality: Quality setting (e.g., "hd", "standard").
        user_id: User identifier.
        tenant_id: Tenant identifier.
        persona: Persona name.
        conversation_id: Conversation identifier.
        request_id: Unique request identifier.
        success: Whether the request succeeded.
        metadata: Additional metadata.

    Returns:
        Created UsageRecord.
    """
    manager = await get_budget_manager()
    return await manager.record_image_usage(
        provider=provider,
        model=model,
        count=count,
        size=size,
        quality=quality,
        user_id=user_id,
        tenant_id=tenant_id,
        persona=persona,
        conversation_id=conversation_id,
        request_id=request_id,
        success=success,
        metadata=metadata,
    )


async def record_audio_usage(
    provider: str,
    model: str,
    duration_seconds: float,
    operation: str = "transcription",
    user_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
    request_id: Optional[str] = None,
) -> UsageRecord:
    """Record audio usage (TTS/STT).

    Note: This creates a UsageRecord using the base record_usage method.
    Cost should be calculated by the caller or will be zero.

    Args:
        provider: Audio provider name.
        model: Model identifier.
        duration_seconds: Audio duration in seconds.
        operation: "transcription" or "synthesis".
        user_id: User identifier.
        tenant_id: Tenant identifier.
        request_id: Unique request identifier.

    Returns:
        Created UsageRecord.
    """
    manager = await get_budget_manager()
    
    op_type = (
        OperationType.SPEECH_TO_TEXT if operation == "transcription"
        else OperationType.TEXT_TO_SPEECH
    )
    
    record = UsageRecord(
        provider=provider,
        model=model,
        operation_type=op_type,
        cost_usd=Decimal("0"),  # Cost calculation TBD
        audio_seconds=duration_seconds,
        user_id=user_id,
        tenant_id=tenant_id,
        request_id=request_id,
    )
    
    await manager.record_usage(record)
    return record


async def record_embedding_usage(
    provider: str,
    model: str,
    input_tokens: int,
    user_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
    request_id: Optional[str] = None,
) -> UsageRecord:
    """Record embedding usage.

    Note: This creates a UsageRecord using the base record_usage method.
    Cost calculation uses the pricing registry.

    Args:
        provider: Embedding provider name.
        model: Model identifier.
        input_tokens: Number of tokens embedded.
        user_id: User identifier.
        tenant_id: Tenant identifier.
        request_id: Unique request identifier.

    Returns:
        Created UsageRecord.
    """
    manager = await get_budget_manager()
    
    # Calculate cost for embeddings
    cost = calculate_llm_cost(
        model=model,
        input_tokens=input_tokens,
        output_tokens=0,
    )
    
    record = UsageRecord(
        provider=provider,
        model=model,
        operation_type=OperationType.EMBEDDING,
        cost_usd=cost,
        input_tokens=input_tokens,
        user_id=user_id,
        tenant_id=tenant_id,
        request_id=request_id,
    )
    
    await manager.record_usage(record)
    return record


# =============================================================================
# Budget Check API
# =============================================================================


async def check_budget(
    provider: str,
    model: str,
    estimated_cost: Optional[Decimal] = None,
    estimated_input_tokens: int = 0,
    estimated_output_tokens: int = 0,
    image_count: int = 0,
    user_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
) -> BudgetCheckResult:
    """Check if budget is available for an operation.

    Args:
        provider: Provider to check.
        model: Model to check.
        estimated_cost: Pre-calculated estimated cost.
        estimated_input_tokens: Expected input tokens (for LLM).
        estimated_output_tokens: Expected output tokens (for LLM).
        image_count: Number of images to generate.
        user_id: User identifier.
        tenant_id: Tenant identifier.

    Returns:
        BudgetCheckResult with allowed status and details.
    """
    manager = await get_budget_manager()
    return await manager.check_budget_available(
        provider=provider,
        model=model,
        estimated_cost=estimated_cost,
        estimated_input_tokens=estimated_input_tokens,
        estimated_output_tokens=estimated_output_tokens,
        image_count=image_count,
        user_id=user_id,
        tenant_id=tenant_id,
    )


async def get_current_spend(
    scope: BudgetScope = BudgetScope.GLOBAL,
    scope_id: Optional[str] = None,
    period: BudgetPeriod = BudgetPeriod.MONTHLY,
) -> SpendSummary:
    """Get current spending summary for a scope.

    Args:
        scope: Budget scope.
        scope_id: Scope identifier.
        period: Budget period.

    Returns:
        SpendSummary with current spending data.
    """
    manager = await get_budget_manager()
    return await manager.get_current_spend(
        scope=scope,
        scope_id=scope_id,
        period=period,
    )


# =============================================================================
# Spending Summary API
# =============================================================================


async def get_spend_summary(
    user_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
    period: BudgetPeriod = BudgetPeriod.MONTHLY,
) -> SpendSummary:
    """Get a spending summary for a user or tenant.

    This is a convenience wrapper around get_current_spend that
    determines the appropriate scope based on user_id/tenant_id.

    Args:
        user_id: User identifier.
        tenant_id: Tenant identifier.
        period: Budget period.

    Returns:
        SpendSummary with spending data and breakdown.
    """
    # Determine scope
    if user_id:
        scope = BudgetScope.USER
        scope_id = user_id
    elif tenant_id:
        scope = BudgetScope.TENANT
        scope_id = tenant_id
    else:
        scope = BudgetScope.GLOBAL
        scope_id = None

    return await get_current_spend(
        scope=scope,
        scope_id=scope_id,
        period=period,
    )


# =============================================================================
# Reporting API
# =============================================================================


async def get_usage_report(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    group_by: Optional[List[ReportGrouping]] = None,
    user_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
    provider: Optional[str] = None,
) -> UsageReport:
    """Generate a usage report.

    Args:
        start_date: Report start date (default: 30 days ago).
        end_date: Report end date (default: now).
        group_by: Grouping options.
        user_id: Filter by user.
        tenant_id: Filter by tenant.
        provider: Filter by provider.

    Returns:
        UsageReport with aggregated data.
    """
    now = datetime.now(timezone.utc)
    start_date = start_date or (now - timedelta(days=30))
    end_date = end_date or now
    group_by = group_by or [ReportGrouping.PROVIDER]

    manager = await get_budget_manager()
    generator = ReportGenerator(manager)

    return await generator.generate_report(
        start_date=start_date,
        end_date=end_date,
        group_by=group_by,
        user_id=user_id,
        tenant_id=tenant_id,
        provider=provider,
    )


async def export_usage_report(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    format: ExportFormat = ExportFormat.JSON,
    group_by: Optional[List[ReportGrouping]] = None,
) -> str:
    """Export a usage report to a string format.

    Args:
        start_date: Report start date.
        end_date: Report end date.
        format: Export format (JSON, CSV, Markdown).
        group_by: Grouping options.

    Returns:
        Formatted report string.
    """
    report = await get_usage_report(
        start_date=start_date,
        end_date=end_date,
        group_by=group_by,
    )

    manager = await get_budget_manager()
    generator = ReportGenerator(manager)
    return generator.export_report(report, format)


async def get_cost_projection(
    days_to_project: int = 30,
) -> Dict[str, Any]:
    """Get a cost projection report.

    Args:
        days_to_project: Number of days to project forward.

    Returns:
        Projection data dictionary.
    """
    manager = await get_budget_manager()
    generator = ReportGenerator(manager)
    return await generator.generate_projection_report(days_to_project)


# =============================================================================
# Alert API
# =============================================================================


async def get_alerts(
    policy_id: Optional[str] = None,
    severity: Optional[AlertSeverity] = None,
    unacknowledged_only: bool = False,
) -> List[BudgetAlert]:
    """Get active (unresolved) budget alerts.

    Args:
        policy_id: Filter by policy ID.
        severity: Filter by severity.
        unacknowledged_only: Only return unacknowledged alerts.

    Returns:
        List of BudgetAlerts.
    """
    manager = await get_budget_manager()
    return await manager.get_active_alerts(
        policy_id=policy_id,
        severity=severity,
        unacknowledged_only=unacknowledged_only,
    )


async def acknowledge_alert(
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
    manager = await get_budget_manager()
    return await manager.acknowledge_alert(
        alert_id=alert_id,
        user_id=user_id,
    )


async def add_alert_rule(
    name: str,
    threshold_percent: float,
    severity: AlertSeverity = AlertSeverity.WARNING,
    scope: Optional[BudgetScope] = None,
    cooldown_minutes: int = 60,
) -> None:
    """Add a custom alert rule.

    Args:
        name: Rule name.
        threshold_percent: Threshold percentage (0-200).
        severity: Alert severity.
        scope: Scope to apply rule to.
        cooldown_minutes: Cooldown between alerts.
    """
    engine = await get_alert_engine()

    rule = AlertRule(
        threshold_percent=threshold_percent,
        severity=severity,
        cooldown_minutes=cooldown_minutes,
    )
    engine.add_rule(rule)


# =============================================================================
# Policy Management API
# =============================================================================


async def set_budget_policy(
    name: str,
    limit_amount: Decimal,
    period: BudgetPeriod = BudgetPeriod.MONTHLY,
    scope: BudgetScope = BudgetScope.GLOBAL,
    scope_id: Optional[str] = None,
    soft_limit_percent: float = 80.0,
    actions: Optional[List[LimitAction]] = None,
    enabled: bool = True,
) -> BudgetPolicy:
    """Set a budget policy.

    Args:
        name: Policy name.
        limit_amount: Budget limit in USD.
        period: Budget period.
        scope: Policy scope.
        scope_id: Scope identifier.
        soft_limit_percent: Percentage for soft limit warning.
        actions: Actions when limit reached.
        enabled: Whether policy is active.

    Returns:
        Created BudgetPolicy.
    """
    manager = await get_budget_manager()

    policy = BudgetPolicy(
        name=name,
        scope=scope,
        scope_id=scope_id,
        period=period,
        limit_amount=limit_amount,
        soft_limit_percent=soft_limit_percent / 100.0,  # Convert to 0.0-1.0 range
        hard_limit_action=actions[0] if actions else LimitAction.WARN,
        enabled=enabled,
    )

    await manager.set_budget_policy(policy)
    return policy


async def get_policies(
    scope: Optional[BudgetScope] = None,
    scope_id: Optional[str] = None,
    enabled_only: bool = False,
) -> List[BudgetPolicy]:
    """Get budget policies.

    Args:
        scope: Filter by scope.
        scope_id: Filter by scope ID.
        enabled_only: Only return enabled policies.

    Returns:
        List of BudgetPolicies.
    """
    manager = await get_budget_manager()
    return await manager.get_policies(
        scope=scope,
        scope_id=scope_id,
        enabled_only=enabled_only,
    )


async def delete_policy(policy_id: str) -> bool:
    """Delete a budget policy.

    Args:
        policy_id: Policy identifier.

    Returns:
        True if deleted.
    """
    manager = await get_budget_manager()
    return await manager.delete_policy(policy_id)


# =============================================================================
# Pricing API
# =============================================================================


def get_model_price(
    model: str,
) -> Optional[Dict[str, Any]]:
    """Get pricing data for a model.

    Note: This only retrieves stored pricing data. Model availability
    and configuration should be obtained through ProviderManager.

    Args:
        model: Model identifier (as known to ProviderManager).

    Returns:
        Pricing dictionary or None if no pricing data available.
    """
    registry = PricingRegistry()
    pricing = registry.get_model_pricing(model)
    if pricing is None:
        return None
    return {
        "model": pricing.model,
        "provider": pricing.provider,
        "input_price": str(pricing.input_price),
        "output_price": str(pricing.output_price) if pricing.output_price else None,
        "unit": pricing.unit.value,
        "cached_input_price": str(pricing.cached_input_price) if pricing.cached_input_price else None,
    }


def calculate_llm_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int = 0,
) -> Decimal:
    """Calculate LLM cost based on token counts.

    Note: Model identifier should come from ProviderManager.
    This only performs cost calculation using stored pricing data.

    Args:
        model: Model identifier (as configured in ProviderManager).
        input_tokens: Input tokens.
        output_tokens: Output tokens.
        cached_tokens: Cached tokens (may have discounted rate).

    Returns:
        Cost in USD.
    """
    registry = PricingRegistry()
    return registry.calculate_llm_cost(
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_tokens=cached_tokens,
    )


def calculate_image_cost(
    model: str,
    count: int = 1,
    size: str = "1024x1024",
    quality: str = "standard",
) -> Decimal:
    """Calculate image generation cost.

    Note: Model identifier should come from MediaProviderManager.
    This only performs cost calculation using stored pricing data.

    Args:
        model: Model/engine identifier (as configured in MediaProviderManager).
        count: Number of images.
        size: Image size/resolution.
        quality: Quality setting.

    Returns:
        Cost in USD.
    """
    registry = PricingRegistry()
    return registry.calculate_image_cost(
        model=model,
        count=count,
        size=size,
        quality=quality,
    )


def get_cheaper_alternative(
    model: str,
) -> Optional[Tuple[str, str]]:
    """Suggest a cheaper model alternative based on pricing data.

    Note: Model identifier should come from ProviderManager.
    Alternatives should be validated against ProviderManager
    for actual availability before use.

    Args:
        model: Current model identifier.

    Returns:
        Tuple of (model_id, provider) if cheaper alternative found, None otherwise.
    """
    registry = PricingRegistry()
    return registry.get_cheaper_alternative(model)


# =============================================================================
# Tracking Context API
# =============================================================================


async def track_request(
    provider: str,
    model: str,
    operation_type: OperationType = OperationType.CHAT_COMPLETION,
    user_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
    persona: Optional[str] = None,
) -> TrackingContext:
    """Create a tracking context for a request.

    Usage::

        async with await track_request("openai", "gpt-4o") as ctx:
            # Make API call
            response = await client.chat.completions.create(...)

            # Record tokens from response
            ctx.record_tokens(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            )

    Args:
        provider: Provider name.
        model: Model identifier.
        operation_type: Type of operation.
        user_id: User identifier.
        tenant_id: Tenant identifier.
        conversation_id: Conversation identifier.
        persona: Persona name.

    Returns:
        TrackingContext that can be used as async context manager.
    """
    tracker = await get_usage_tracker()
    return tracker.track_request(
        provider=provider,
        model=model,
        operation_type=operation_type,
        user_id=user_id,
        tenant_id=tenant_id,
        conversation_id=conversation_id,
        persona=persona,
    )


# =============================================================================
# Rollover API
# =============================================================================


async def get_rollover_amount(policy_id: str) -> Decimal:
    """Get the current rollover amount for a policy.

    Args:
        policy_id: Budget policy identifier.

    Returns:
        Rollover amount in USD (Decimal).
    """
    manager = await get_budget_manager()
    return await manager.get_rollover_amount(policy_id)


async def calculate_rollover(policy_id: str) -> Decimal:
    """Calculate potential rollover from previous period.

    This calculates what the rollover would be based on unused budget
    from the previous period, without actually applying it.

    Args:
        policy_id: Budget policy identifier.

    Returns:
        Calculated rollover amount (0 if policy not found or rollover disabled).
    """
    manager = await get_budget_manager()
    policy = await manager.get_policy(policy_id)
    if not policy:
        return Decimal("0")
    return await manager.calculate_rollover(policy)


async def process_period_end(policy_id: str) -> Decimal:
    """Process end of budget period and apply rollover.

    Call this when a budget period ends to finalize spending and
    calculate rollover for the next period.

    Args:
        policy_id: Budget policy identifier.

    Returns:
        Rollover amount for next period (0 if policy not found).
    """
    manager = await get_budget_manager()
    policy = await manager.get_policy(policy_id)
    if not policy:
        return Decimal("0")
    return await manager.process_period_end(policy)


# =============================================================================
# Lifecycle API
# =============================================================================


async def initialize_budget_manager(
    config_manager: Optional[Any] = None,
) -> BudgetManager:
    """Initialize the budget management system.

    Should be called during application startup with a ConfigManager.

    Args:
        config_manager: Configuration manager instance.

    Returns:
        Initialized BudgetManager.
    """
    manager = await get_budget_manager(config_manager)
    return manager


async def shutdown_budget_manager() -> None:
    """Shutdown the budget management system.

    Should be called during application shutdown.
    Flushes pending usage records and closes resources.
    """
    manager = get_budget_manager_sync()
    if manager:
        await manager.shutdown()
