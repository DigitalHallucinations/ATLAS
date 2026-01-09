"""
ATLAS Budget Services

Provides budget policy management, usage tracking, and alert services
following the ATLAS service pattern.

This package provides three focused services:
- BudgetPolicyService: Policy CRUD, enforcement, pre-flight checks
- BudgetTrackingService: Usage recording, aggregation, reporting
- BudgetAlertService: Alert configuration, threshold monitoring, notifications

Author: ATLAS Team
Date: Jan 8, 2026
"""

from .types import (
    # Policy Events
    BudgetPolicyCreated,
    BudgetPolicyUpdated,
    BudgetPolicyDeleted,
    BudgetCheckRequested,
    # Tracking Events
    BudgetUsageRecorded,
    BudgetThresholdReached,
    # Alert Events
    BudgetAlertTriggered,
    BudgetAlertAcknowledged,
    BudgetLimitExceeded,
    BudgetApproachingLimit,
    # Policy DTOs
    BudgetPolicyCreate,
    BudgetPolicyUpdate,
    BudgetCheckRequest,
    BudgetCheckResponse,
    # Tracking DTOs
    UsageRecordCreate,
    LLMUsageCreate,
    ImageUsageCreate,
    UsageSummaryRequest,
    SpendBreakdown,
    SpendTrendPoint,
    SpendTrend,
    # Alert DTOs
    AlertConfigCreate,
    AlertConfigUpdate,
    AlertConfig,
    ActiveAlert,
    AlertListRequest,
    # Re-exported models
    BudgetPolicy,
    BudgetScope,
    BudgetPeriod,
    LimitAction,
    OperationType,
    UsageRecord,
    SpendSummary,
)
from .exceptions import (
    BudgetError,
    BudgetPolicyNotFoundError,
    BudgetPolicyConflictError,
    BudgetExceededError,
    BudgetValidationError,
)
from .permissions import BudgetPermissionChecker
from .policy_service import BudgetPolicyService
from .tracking_service import BudgetTrackingService
from .alert_service import BudgetAlertService
from .factory import (
    get_policy_service,
    get_tracking_service,
    get_alert_service,
    get_all_services,
    shutdown_services,
    reset_services,
)

__all__ = [
    # Services
    "BudgetPolicyService",
    "BudgetTrackingService",
    "BudgetAlertService",
    "BudgetPermissionChecker",
    
    # Policy Events
    "BudgetPolicyCreated",
    "BudgetPolicyUpdated",
    "BudgetPolicyDeleted",
    "BudgetCheckRequested",
    
    # Tracking Events
    "BudgetUsageRecorded",
    "BudgetThresholdReached",
    
    # Alert Events
    "BudgetAlertTriggered",
    "BudgetAlertAcknowledged",
    "BudgetLimitExceeded",
    "BudgetApproachingLimit",
    
    # Policy DTOs
    "BudgetPolicyCreate",
    "BudgetPolicyUpdate",
    "BudgetCheckRequest",
    "BudgetCheckResponse",
    
    # Tracking DTOs
    "UsageRecordCreate",
    "LLMUsageCreate",
    "ImageUsageCreate",
    "UsageSummaryRequest",
    "SpendBreakdown",
    "SpendTrendPoint",
    "SpendTrend",
    
    # Alert DTOs
    "AlertConfigCreate",
    "AlertConfigUpdate",
    "AlertConfig",
    "ActiveAlert",
    "AlertListRequest",
    
    # Re-exported models
    "BudgetPolicy",
    "BudgetScope",
    "BudgetPeriod",
    "LimitAction",
    "OperationType",
    "UsageRecord",
    "SpendSummary",
    
    # Exceptions
    "BudgetError",
    "BudgetPolicyNotFoundError",
    "BudgetPolicyConflictError",
    "BudgetExceededError",
    "BudgetValidationError",
    
    # Factory Functions
    "get_policy_service",
    "get_tracking_service",
    "get_alert_service",
    "get_all_services",
    "shutdown_services",
    "reset_services",
]
