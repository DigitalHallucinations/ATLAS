"""
Budget service exceptions.

Provides specific exception types for budget-related errors,
extending the common service exception hierarchy.

Author: ATLAS Team
Date: Jan 8, 2026
"""

from typing import Any, Dict, Optional
from decimal import Decimal

from core.services.common import (
    ServiceError,
    NotFoundError,
    ConflictError,
    ValidationError,
)


class BudgetError(ServiceError):
    """Base exception for all budget-related errors."""
    
    pass


class BudgetPolicyNotFoundError(NotFoundError, BudgetError):
    """Raised when a requested budget policy does not exist."""
    
    def __init__(
        self,
        policy_id: str,
        message: Optional[str] = None,
    ) -> None:
        super().__init__(
            message or f"Budget policy not found: {policy_id}",
            error_code="BUDGET_POLICY_NOT_FOUND",
            details={"policy_id": policy_id},
        )
        self.policy_id = policy_id


class BudgetPolicyConflictError(ConflictError, BudgetError):
    """Raised when a policy operation conflicts with existing state."""
    
    def __init__(
        self,
        message: str,
        conflicting_policy_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message,
            error_code="BUDGET_POLICY_CONFLICT",
            details=details or {},
        )
        if conflicting_policy_id:
            self.details["conflicting_policy_id"] = conflicting_policy_id


class BudgetExceededError(BudgetError):
    """Raised when an operation would exceed budget limits."""
    
    def __init__(
        self,
        policy_id: str,
        policy_name: str,
        current_spend: Decimal,
        limit_amount: Decimal,
        estimated_cost: Decimal,
        message: Optional[str] = None,
    ) -> None:
        remaining = limit_amount - current_spend
        default_message = (
            f"Budget exceeded for '{policy_name}': "
            f"${current_spend:.2f} spent of ${limit_amount:.2f} limit, "
            f"estimated cost ${estimated_cost:.2f} exceeds remaining ${remaining:.2f}"
        )
        super().__init__(
            message or default_message,
            error_code="BUDGET_EXCEEDED",
            details={
                "policy_id": policy_id,
                "policy_name": policy_name,
                "current_spend": str(current_spend),
                "limit_amount": str(limit_amount),
                "estimated_cost": str(estimated_cost),
                "remaining": str(remaining),
            },
        )
        self.policy_id = policy_id
        self.policy_name = policy_name
        self.current_spend = current_spend
        self.limit_amount = limit_amount
        self.estimated_cost = estimated_cost


class BudgetValidationError(ValidationError, BudgetError):
    """Raised when budget policy validation fails."""
    
    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        error_details = details or {}
        if field:
            error_details["field"] = field
        super().__init__(
            message,
            error_code="BUDGET_VALIDATION_ERROR",
            details=error_details,
        )
        self.field = field
