"""Budget management module for ATLAS.

Provides centralized budget tracking, cost monitoring, and spending
controls across all LLM and media generation providers.

Key Components:
- Budget Services: Policy, Tracking, and Alert services from core.services.budget
- PricingRegistry: Provider/model pricing database
- UsageTracker: Real-time cost capture from provider calls
- AlertEngine: Threshold monitoring and notifications
- ReportGenerator: Usage analytics and export

Usage::

    from modules.budget import set_budget_policy, check_budget, BudgetPolicy, BudgetScope

    # Set a monthly budget
    policy = await set_budget_policy(
        name="Monthly Limit",
        scope=BudgetScope.GLOBAL,
        period=BudgetPeriod.MONTHLY,
        limit_amount=Decimal("100.00"),
    )
    
    # Check current budget status
    result = await check_budget()
    print(f"Allowed: {result.allowed}, Spend: {result.current_spend}")
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
from .tracking import UsageTracker, get_usage_tracker
from .alerts import AlertEngine, get_alert_engine
from .reports import ReportGenerator, UsageReport, ReportGrouping
from .scope_hierarchy import (
    ScopeHierarchyResolver,
    UsageContext,
    get_scope_resolver,
    reset_scope_resolver,
    get_scope_priority,
    is_hierarchical_scope,
    is_resource_scope,
    HIERARCHY_ORDER,
    RESOURCE_ORDER,
)
from .policy_matcher import (
    PolicyMatcher,
    PolicyMatch,
    PolicyMatchResult,
    build_scopes_to_check,
)
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
    initialize_budget_services,
    shutdown_budget_services,
)
from .integration import (
    setup_budget_integration,
    shutdown_budget_integration,
    record_llm_usage_from_response,
    record_image_usage_from_result,
    check_budget_before_request,
)

__all__ = [
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
    # Scope hierarchy
    "ScopeHierarchyResolver",
    "UsageContext",
    "get_scope_resolver",
    "reset_scope_resolver",
    "get_scope_priority",
    "is_hierarchical_scope",
    "is_resource_scope",
    "HIERARCHY_ORDER",
    "RESOURCE_ORDER",
    # Policy matching
    "PolicyMatcher",
    "PolicyMatch",
    "PolicyMatchResult",
    "build_scopes_to_check",
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
    "initialize_budget_services",
    "shutdown_budget_services",
    # Integration functions
    "setup_budget_integration",
    "shutdown_budget_integration",
    "record_llm_usage_from_response",
    "record_image_usage_from_result",
    "check_budget_before_request",
]
