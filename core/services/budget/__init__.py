"""
ATLAS Budget Services

Provides budget policy management, usage tracking, and alert services
following the ATLAS service pattern.

This package splits the monolithic BudgetManager into three focused services:
- BudgetPolicyService: Policy CRUD, enforcement, pre-flight checks
- BudgetTrackingService: Usage recording, aggregation, reporting (Phase 2)
- BudgetAlertService: Alert configuration, threshold monitoring (Phase 3)

Author: ATLAS Team
Date: Jan 8, 2026
"""

from .types import (
    # Events
    BudgetPolicyCreated,
    BudgetPolicyUpdated,
    BudgetPolicyDeleted,
    BudgetCheckRequested,
    # DTOs
    BudgetPolicyCreate,
    BudgetPolicyUpdate,
    BudgetCheckRequest,
    BudgetCheckResponse,
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

__all__ = [
    # Services
    "BudgetPolicyService",
    "BudgetPermissionChecker",
    
    # Events
    "BudgetPolicyCreated",
    "BudgetPolicyUpdated",
    "BudgetPolicyDeleted",
    "BudgetCheckRequested",
    
    # DTOs
    "BudgetPolicyCreate",
    "BudgetPolicyUpdate",
    "BudgetCheckRequest",
    "BudgetCheckResponse",
    
    # Exceptions
    "BudgetError",
    "BudgetPolicyNotFoundError",
    "BudgetPolicyConflictError",
    "BudgetExceededError",
    "BudgetValidationError",
]
