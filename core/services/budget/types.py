"""
Budget service types and domain events.

Defines DTOs for service operations and domain events for
integration with the ATLAS messaging system.

Author: ATLAS Team
Date: Jan 8, 2026
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Literal, Optional
from uuid import UUID

# Re-export core budget models for convenience
from modules.budget.models import (
    BudgetPolicy,
    BudgetScope,
    BudgetPeriod,
    LimitAction,
    BudgetCheckResult,
)


def _now_utc() -> datetime:
    """Get current UTC timestamp."""
    return datetime.now(timezone.utc)


def _generate_uuid() -> str:
    """Generate a new UUID string."""
    return str(uuid.uuid4())


# =============================================================================
# Domain Events (standalone frozen dataclasses, not inheriting from DomainEvent)
# =============================================================================


@dataclass(frozen=True)
class BudgetPolicyCreated:
    """Emitted when a budget policy is created."""
    
    policy_id: str
    policy_name: str
    scope: str
    limit_amount: Decimal
    period: str
    tenant_id: str
    actor_id: str
    scope_id: Optional[str] = None
    actor_type: str = "user"
    event_type: str = "budget.policy_created"
    entity_id: str = field(default_factory=_generate_uuid)
    timestamp: datetime = field(default_factory=_now_utc)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_type": self.event_type,
            "entity_id": self.entity_id,
            "tenant_id": self.tenant_id,
            "actor_id": self.actor_id,
            "actor_type": self.actor_type,
            "policy_id": self.policy_id,
            "policy_name": self.policy_name,
            "scope": self.scope,
            "scope_id": self.scope_id,
            "limit_amount": str(self.limit_amount),
            "period": self.period,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass(frozen=True)
class BudgetPolicyUpdated:
    """Emitted when a budget policy is updated."""
    
    policy_id: str
    policy_name: str
    changed_fields: tuple[str, ...]
    tenant_id: str
    actor_id: str
    actor_type: str = "user"
    event_type: str = "budget.policy_updated"
    entity_id: str = field(default_factory=_generate_uuid)
    timestamp: datetime = field(default_factory=_now_utc)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_type": self.event_type,
            "entity_id": self.entity_id,
            "tenant_id": self.tenant_id,
            "actor_id": self.actor_id,
            "actor_type": self.actor_type,
            "policy_id": self.policy_id,
            "policy_name": self.policy_name,
            "changed_fields": list(self.changed_fields),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass(frozen=True)
class BudgetPolicyDeleted:
    """Emitted when a budget policy is deleted."""
    
    policy_id: str
    policy_name: str
    tenant_id: str
    actor_id: str
    actor_type: str = "user"
    event_type: str = "budget.policy_deleted"
    entity_id: str = field(default_factory=_generate_uuid)
    timestamp: datetime = field(default_factory=_now_utc)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_type": self.event_type,
            "entity_id": self.entity_id,
            "tenant_id": self.tenant_id,
            "actor_id": self.actor_id,
            "actor_type": self.actor_type,
            "policy_id": self.policy_id,
            "policy_name": self.policy_name,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass(frozen=True)
class BudgetCheckRequested:
    """Emitted when a budget pre-flight check is performed."""
    
    tenant_id: str
    actor_id: str
    estimated_cost: Decimal
    allowed: bool
    action: str
    policy_id: Optional[str] = None
    warnings: tuple[str, ...] = ()
    actor_type: str = "user"
    event_type: str = "budget.check_requested"
    entity_id: str = field(default_factory=_generate_uuid)
    timestamp: datetime = field(default_factory=_now_utc)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_type": self.event_type,
            "entity_id": self.entity_id,
            "tenant_id": self.tenant_id,
            "actor_id": self.actor_id,
            "actor_type": self.actor_type,
            "policy_id": self.policy_id,
            "estimated_cost": str(self.estimated_cost),
            "allowed": self.allowed,
            "action": self.action,
            "warnings": list(self.warnings),
            "timestamp": self.timestamp.isoformat(),
        }


# =============================================================================
# Data Transfer Objects (DTOs)
# =============================================================================


@dataclass
class BudgetPolicyCreate:
    """DTO for creating a new budget policy."""
    
    name: str
    scope: BudgetScope
    limit_amount: Decimal
    scope_id: Optional[str] = None
    period: BudgetPeriod = BudgetPeriod.MONTHLY
    currency: str = "USD"
    soft_limit_percent: float = 0.80
    hard_limit_action: LimitAction = LimitAction.WARN
    rollover_enabled: bool = False
    rollover_max_percent: float = 0.25
    provider_limits: Dict[str, Decimal] = field(default_factory=dict)
    model_limits: Dict[str, Decimal] = field(default_factory=dict)
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_policy(self) -> BudgetPolicy:
        """Convert DTO to BudgetPolicy domain model."""
        return BudgetPolicy(
            name=self.name,
            scope=self.scope,
            limit_amount=self.limit_amount,
            scope_id=self.scope_id,
            period=self.period,
            currency=self.currency,
            soft_limit_percent=self.soft_limit_percent,
            hard_limit_action=self.hard_limit_action,
            rollover_enabled=self.rollover_enabled,
            rollover_max_percent=self.rollover_max_percent,
            provider_limits=self.provider_limits,
            model_limits=self.model_limits,
            priority=self.priority,
            metadata=self.metadata,
        )


@dataclass
class BudgetPolicyUpdate:
    """DTO for updating an existing budget policy.
    
    All fields are optional - only provided fields will be updated.
    """
    
    name: Optional[str] = None
    limit_amount: Optional[Decimal] = None
    soft_limit_percent: Optional[float] = None
    hard_limit_action: Optional[LimitAction] = None
    rollover_enabled: Optional[bool] = None
    rollover_max_percent: Optional[float] = None
    provider_limits: Optional[Dict[str, Decimal]] = None
    model_limits: Optional[Dict[str, Decimal]] = None
    enabled: Optional[bool] = None
    priority: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def get_changed_fields(self) -> List[str]:
        """Return list of fields that have been set."""
        return [
            field_name for field_name in [
                "name", "limit_amount", "soft_limit_percent", "hard_limit_action",
                "rollover_enabled", "rollover_max_percent", "provider_limits",
                "model_limits", "enabled", "priority", "metadata"
            ]
            if getattr(self, field_name) is not None
        ]


@dataclass
class BudgetCheckRequest:
    """DTO for requesting a budget pre-flight check."""
    
    provider: str
    model: str
    estimated_cost: Optional[Decimal] = None
    estimated_input_tokens: int = 0
    estimated_output_tokens: int = 0
    image_count: int = 0
    user_id: Optional[str] = None
    tenant_id: Optional[str] = None


@dataclass
class BudgetCheckResponse:
    """Response from a budget pre-flight check."""
    
    allowed: bool
    action: LimitAction
    policy_id: Optional[str] = None
    current_spend: Decimal = Decimal("0")
    limit_amount: Decimal = Decimal("0")
    estimated_cost: Decimal = Decimal("0")
    remaining_after: Decimal = Decimal("0")
    warnings: List[str] = field(default_factory=list)
    alternative_model: Optional[str] = None
    
    @classmethod
    def from_check_result(cls, result: BudgetCheckResult) -> "BudgetCheckResponse":
        """Create response from legacy BudgetCheckResult."""
        return cls(
            allowed=result.allowed,
            action=result.action,
            policy_id=result.policy_id,
            current_spend=result.current_spend or Decimal("0"),
            limit_amount=result.limit_amount or Decimal("0"),
            estimated_cost=result.estimated_cost or Decimal("0"),
            remaining_after=result.remaining_after or Decimal("0"),
            warnings=result.warnings or [],
            alternative_model=result.alternative_model,
        )
    
    @property
    def percent_used(self) -> float:
        """Calculate percentage of budget used."""
        if self.limit_amount <= 0:
            return 0.0
        return float(self.current_spend / self.limit_amount)
    
    @property
    def is_near_limit(self) -> bool:
        """Check if approaching budget limit (>80% used)."""
        return self.percent_used >= 0.80
