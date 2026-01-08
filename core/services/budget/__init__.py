"""
ATLAS Budget Services

Provides budget policy management, usage tracking, and alert services
following the ATLAS service pattern.

This package splits the monolithic BudgetManager into three focused services:
- BudgetPolicyService: Policy CRUD, enforcement, pre-flight checks
- BudgetTrackingService: Usage recording, aggregation, reporting
- BudgetAlertService: Alert configuration, threshold monitoring (Phase 3)

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

__all__ = [
    # Services
    "BudgetPolicyService",
    "BudgetTrackingService",
    "BudgetPermissionChecker",
    
    # Policy Events
    "BudgetPolicyCreated",
    "BudgetPolicyUpdated",
    "BudgetPolicyDeleted",
    "BudgetCheckRequested",
    
    # Tracking Events
    "BudgetUsageRecorded",
    "BudgetThresholdReached",
    
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
]
