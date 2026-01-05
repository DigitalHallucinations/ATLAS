"""Budget management module for ATLAS.

Provides centralized budget tracking, cost monitoring, and spending
controls across all LLM and media generation providers.

Key Components:
- BudgetManager: Singleton orchestrator for budget operations
- PricingRegistry: Provider/model pricing database
- UsageTracker: Real-time cost capture from provider calls
- AlertEngine: Threshold monitoring and notifications
- ReportGenerator: Usage analytics and export

Usage::

    from modules.budget import get_budget_manager, BudgetPolicy, BudgetScope

    manager = await get_budget_manager()
    
    # Set a monthly budget
    policy = BudgetPolicy(
        name="Monthly Limit",
        scope=BudgetScope.GLOBAL,
        period=BudgetPeriod.MONTHLY,
        limit_amount=Decimal("100.00"),
    )
    await manager.set_budget_policy(policy)
    
    # Check current spend
    summary = await manager.get_current_spend()
    print(f"Spent: ${summary.total_spent} / ${summary.limit}")
"""

from .models import (
    BudgetPolicy,
    BudgetPeriod,
    BudgetScope,
    LimitAction,
    UsageRecord,
    OperationType,
    SpendSummary,
    BudgetCheckResult,
    BudgetAlert,
    AlertSeverity,
    AlertTriggerType,
)
from .pricing import PricingRegistry, get_pricing_registry
from .manager import BudgetManager, get_budget_manager
from .tracking import UsageTracker, get_usage_tracker
from .alerts import AlertEngine, get_alert_engine
from .reports import ReportGenerator, UsageReport, ReportGrouping
from .api import (
    record_llm_usage,
    record_image_usage,
    record_audio_usage,
    record_embedding_usage,
    check_budget,
    get_spend_summary,
    get_usage_report,
    get_alerts,
    acknowledge_alert,
    set_budget_policy,
    get_policies,
    delete_policy,
    get_rollover_amount,
    calculate_rollover,
    process_period_end,
    initialize_budget_manager,
    shutdown_budget_manager,
)
from .integration import (
    setup_budget_integration,
    shutdown_budget_integration,
    record_llm_usage_from_response,
    record_image_usage_from_result,
    check_budget_before_request,
)

__all__ = [
    # Core manager
    "BudgetManager",
    "get_budget_manager",
    # Models
    "BudgetPolicy",
    "BudgetPeriod",
    "BudgetScope",
    "LimitAction",
    "UsageRecord",
    "OperationType",
    "SpendSummary",
    "BudgetCheckResult",
    "BudgetAlert",
    "AlertSeverity",
    "AlertTriggerType",
    # Pricing
    "PricingRegistry",
    "get_pricing_registry",
    # Tracking
    "UsageTracker",
    "get_usage_tracker",
    # Alerts
    "AlertEngine",
    "get_alert_engine",
    # Reports
    "ReportGenerator",
    "UsageReport",
    "ReportGrouping",
    # Public API functions
    "record_llm_usage",
    "record_image_usage",
    "record_audio_usage",
    "record_embedding_usage",
    "check_budget",
    "get_spend_summary",
    "get_usage_report",
    "get_alerts",
    "acknowledge_alert",
    "set_budget_policy",
    "get_policies",
    "delete_policy",
    "get_rollover_amount",
    "calculate_rollover",
    "process_period_end",
    "initialize_budget_manager",
    "shutdown_budget_manager",
    # Integration functions
    "setup_budget_integration",
    "shutdown_budget_integration",
    "record_llm_usage_from_response",
    "record_image_usage_from_result",
    "check_budget_before_request",
]
