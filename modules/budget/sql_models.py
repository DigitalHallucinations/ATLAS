"""SQLAlchemy models for budget persistence.

Provides ORM models for storing budget data in PostgreSQL/SQLite.
Follows the same patterns as conversation_store.models.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, Optional, cast

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import declarative_base

from .models import (
    AlertSeverity,
    AlertTriggerType,
    BudgetAlert,
    BudgetPeriod,
    BudgetPolicy,
    BudgetScope,
    LimitAction,
    OperationType,
    UsageRecord,
)

# Use a separate Base to avoid coupling with conversation_store
BudgetBase = declarative_base()


class BudgetPolicyTable(BudgetBase):
    """SQLAlchemy model for budget policies."""

    __tablename__ = "budget_policies"

    id = Column(String(36), primary_key=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    scope = Column(String(32), nullable=False, default="global")
    scope_id = Column(String(255), nullable=True)
    period = Column(String(32), nullable=False, default="monthly")
    limit_amount = Column(Numeric(20, 8), nullable=False)
    soft_limit_percent = Column(Integer, nullable=False, default=80)
    action = Column(String(32), nullable=False, default="warn")
    enabled = Column(Boolean, nullable=False, default=True)
    priority = Column(Integer, nullable=False, default=0)
    rollover_enabled = Column(Boolean, nullable=False, default=False)
    rollover_max_percent = Column(Integer, nullable=False, default=100)
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    metadata_ = Column("metadata", JSONB, nullable=True)

    __table_args__ = (
        Index("ix_budget_policies_scope", "scope", "scope_id"),
        Index("ix_budget_policies_enabled", "enabled"),
    )

    def to_domain(self) -> BudgetPolicy:
        """Convert to domain model."""
        row = cast(Any, self)  # Runtime: columns resolve to values
        return BudgetPolicy(
            id=row.id,
            name=row.name,
            scope=BudgetScope(row.scope),
            scope_id=row.scope_id,
            period=BudgetPeriod(row.period),
            limit_amount=Decimal(str(row.limit_amount)),
            soft_limit_percent=float(row.soft_limit_percent) / 100.0,
            hard_limit_action=LimitAction(row.action),
            enabled=row.enabled,
            priority=row.priority,
            rollover_enabled=row.rollover_enabled,
            rollover_max_percent=float(row.rollover_max_percent) / 100.0,
            created_at=row.created_at,
            updated_at=row.updated_at,
            metadata=dict(row.metadata_) if row.metadata_ else {},
        )

    @classmethod
    def from_domain(cls, policy: BudgetPolicy) -> "BudgetPolicyTable":
        """Create from domain model."""
        return cls(
            id=policy.id,
            name=policy.name,
            description=None,  # Domain model doesn't have description
            scope=policy.scope.value,
            scope_id=policy.scope_id,
            period=policy.period.value,
            limit_amount=policy.limit_amount,
            soft_limit_percent=int(policy.soft_limit_percent * 100),  # Store as int percentage
            action=policy.hard_limit_action.value,
            enabled=policy.enabled,
            priority=policy.priority,
            rollover_enabled=policy.rollover_enabled,
            rollover_max_percent=int(policy.rollover_max_percent * 100),  # Store as int percentage
            created_at=policy.created_at,
            updated_at=policy.updated_at,
            metadata_=policy.metadata or {},
        )


class UsageRecordTable(BudgetBase):
    """SQLAlchemy model for usage records."""

    __tablename__ = "budget_usage_records"

    id = Column(String(36), primary_key=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    provider = Column(String(64), nullable=False, index=True)
    model = Column(String(128), nullable=False)
    operation_type = Column(String(32), nullable=False, default="inference")
    input_tokens = Column(Integer, nullable=False, default=0)
    output_tokens = Column(Integer, nullable=False, default=0)
    cached_tokens = Column(Integer, nullable=False, default=0)
    cost_usd = Column(Numeric(20, 10), nullable=False)
    user_id = Column(String(255), nullable=True, index=True)
    tenant_id = Column(String(255), nullable=True, index=True)
    session_id = Column(String(255), nullable=True)
    conversation_id = Column(String(255), nullable=True)
    request_id = Column(String(255), nullable=True)
    metadata_ = Column("metadata", JSONB, nullable=True)

    __table_args__ = (
        Index("ix_budget_usage_timestamp_range", "timestamp", "provider"),
        Index("ix_budget_usage_user_tenant", "user_id", "tenant_id"),
    )

    def to_domain(self) -> UsageRecord:
        """Convert to domain model."""
        row = cast(Any, self)  # Runtime: columns resolve to values
        return UsageRecord(
            id=row.id,
            timestamp=row.timestamp,
            provider=row.provider,
            model=row.model,
            operation_type=OperationType(row.operation_type),
            input_tokens=row.input_tokens,
            output_tokens=row.output_tokens,
            cost_usd=Decimal(str(row.cost_usd)),
            user_id=row.user_id,
            tenant_id=row.tenant_id,
            conversation_id=row.conversation_id,
            request_id=row.request_id,
            metadata=dict(row.metadata_) if row.metadata_ else {},
        )

    @classmethod
    def from_domain(cls, record: UsageRecord) -> "UsageRecordTable":
        """Create from domain model."""
        return cls(
            id=record.id,
            timestamp=record.timestamp,
            provider=record.provider,
            model=record.model,
            operation_type=record.operation_type.value,
            input_tokens=record.input_tokens or 0,
            output_tokens=record.output_tokens or 0,
            cached_tokens=0,  # Domain model doesn't track cached tokens
            cost_usd=record.cost_usd,
            user_id=record.user_id,
            tenant_id=record.tenant_id,
            session_id=None,  # Domain model doesn't have session_id
            conversation_id=record.conversation_id,
            request_id=record.request_id,
            metadata_=record.metadata or {},
        )


class BudgetAlertTable(BudgetBase):
    """SQLAlchemy model for budget alerts."""

    __tablename__ = "budget_alerts"

    id = Column(String(36), primary_key=True)
    policy_id = Column(String(36), nullable=False, index=True)
    trigger_type = Column(String(32), nullable=False)
    severity = Column(String(16), nullable=False, index=True)
    message = Column(Text, nullable=False)
    current_spend = Column(Numeric(20, 8), nullable=False)
    limit_amount = Column(Numeric(20, 8), nullable=False)
    threshold_percent = Column(Integer, nullable=False)
    triggered_at = Column(DateTime(timezone=True), nullable=False, index=True)
    acknowledged = Column(Boolean, nullable=False, default=False, index=True)
    acknowledged_at = Column(DateTime(timezone=True), nullable=True)
    acknowledged_by = Column(String(255), nullable=True)
    metadata_ = Column("metadata", JSONB, nullable=True)

    __table_args__ = (
        Index("ix_budget_alerts_active", "acknowledged", "triggered_at"),
    )

    def to_domain(self) -> BudgetAlert:
        """Convert to domain model."""
        row = cast(Any, self)  # Runtime: columns resolve to values
        return BudgetAlert(
            id=row.id,
            policy_id=row.policy_id,
            trigger_type=AlertTriggerType(row.trigger_type),
            severity=AlertSeverity(row.severity),
            message=row.message,
            current_spend=Decimal(str(row.current_spend)),
            limit_amount=Decimal(str(row.limit_amount)),
            threshold_percent=float(row.threshold_percent) / 100.0,
            triggered_at=row.triggered_at,
            acknowledged=row.acknowledged,
            acknowledged_at=row.acknowledged_at,
            acknowledged_by=row.acknowledged_by,
            metadata=dict(row.metadata_) if row.metadata_ else {},
        )

    @classmethod
    def from_domain(cls, alert: BudgetAlert) -> "BudgetAlertTable":
        """Create from domain model."""
        return cls(
            id=alert.id,
            policy_id=alert.policy_id,
            trigger_type=alert.trigger_type.value,
            severity=alert.severity.value,
            message=alert.message,
            current_spend=alert.current_spend,
            limit_amount=alert.limit_amount,
            threshold_percent=alert.threshold_percent,
            triggered_at=alert.triggered_at,
            acknowledged=alert.acknowledged,
            acknowledged_at=alert.acknowledged_at,
            acknowledged_by=alert.acknowledged_by,
            metadata_=alert.metadata or {},
        )


class BudgetRolloverTable(BudgetBase):
    """SQLAlchemy model for policy rollover amounts."""

    __tablename__ = "budget_rollovers"

    id = Column(String(36), primary_key=True)
    policy_id = Column(String(36), nullable=False, unique=True, index=True)
    amount = Column(Numeric(20, 8), nullable=False)
    period_start = Column(DateTime(timezone=True), nullable=False)
    period_end = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, default=func.now())


def create_budget_schema(engine, *, logger=None) -> None:
    """Create the budget tables if they don't exist.

    Args:
        engine: SQLAlchemy engine.
        logger: Optional logger for output.
    """
    try:
        BudgetBase.metadata.create_all(engine)
        if logger:
            logger.info("Budget schema created successfully")
    except Exception as exc:
        if logger:
            logger.error("Failed to create budget schema: %s", exc)
        raise
