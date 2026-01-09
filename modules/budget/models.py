"""Budget management data models.

Defines core data structures for budget policies, usage tracking,
spending summaries, and alerts.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional


class BudgetScope(Enum):
    """Scope level for budget policies.

    Scopes are organized hierarchically for policy resolution:
    - GLOBAL: Instance-wide budget (top level)
    - TEAM: Group of users/agents with shared budget
    - PROJECT: Project-specific budget allocation
    - JOB: Background job budget allocation
    - TASK: Individual task budget
    - AGENT: Individual agent budget (for multi-agent systems)
    - USER: Per-user budget
    - SESSION: Single conversation/session budget
    - PROVIDER: Per-provider (OpenAI, Anthropic, etc.)
    - MODEL: Per-model (gpt-4o, claude-3-opus, etc.)
    - TOOL: Per-tool budget (function calling costs)
    - SKILL: Per-skill budget (skill execution costs)

    Hierarchy for policy resolution (most specific wins):
    GLOBAL > TEAM > PROJECT > JOB > TASK > AGENT > USER > SESSION
    PROVIDER, MODEL, TOOL, SKILL are orthogonal and apply as additional constraints.
    """

    # Hierarchical scopes (ordered from broadest to most specific)
    GLOBAL = "global"  # Applies to entire ATLAS instance
    TEAM = "team"  # Team/department budget
    PROJECT = "project"  # Project-specific budget allocation
    JOB = "job"  # Background job budget
    TASK = "task"  # Individual task budget
    AGENT = "agent"  # Individual agent in multi-agent orchestration
    USER = "user"  # Per-user budget
    SESSION = "session"  # Single conversation/session budget

    # Orthogonal resource-based scopes (apply as additional constraints)
    PROVIDER = "provider"  # Per-provider (OpenAI, Anthropic, etc.)
    MODEL = "model"  # Per-model (gpt-4o, claude-3-opus, etc.)
    TOOL = "tool"  # Per-tool (function calling)
    SKILL = "skill"  # Per-skill (skill execution)


class BudgetPeriod(Enum):
    """Time period for budget cycles."""

    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"
    ROLLING_24H = "rolling_24h"
    ROLLING_7D = "rolling_7d"
    ROLLING_30D = "rolling_30d"
    LIFETIME = "lifetime"  # No reset, cumulative


class LimitAction(Enum):
    """Action to take when budget limit is reached."""

    WARN = "warn"  # Log warning, send notification, allow request
    THROTTLE = "throttle"  # Rate limit requests
    SOFT_BLOCK = "soft_block"  # Block but allow override
    BLOCK = "block"  # Hard block all requests
    DEGRADE = "degrade"  # Switch to cheaper model automatically


class OperationType(Enum):
    """Type of billable operation."""

    CHAT_COMPLETION = "chat_completion"
    TEXT_COMPLETION = "text_completion"
    EMBEDDING = "embedding"
    IMAGE_GENERATION = "image_generation"
    IMAGE_EDIT = "image_edit"
    IMAGE_VARIATION = "image_variation"
    SPEECH_TO_TEXT = "speech_to_text"
    TEXT_TO_SPEECH = "text_to_speech"
    FINE_TUNING = "fine_tuning"
    MODERATION = "moderation"
    ASSISTANT = "assistant"
    VISION = "vision"
    VIDEO_GENERATION = "video_generation"


class AlertSeverity(Enum):
    """Severity level for budget alerts."""

    INFO = "info"  # Informational (e.g., 50% used)
    WARNING = "warning"  # Approaching limit (e.g., 80% used)
    CRITICAL = "critical"  # At or over limit
    EMERGENCY = "emergency"  # Significant overage


class AlertTriggerType(Enum):
    """What triggered a budget alert."""

    THRESHOLD_REACHED = "threshold_reached"  # Hit configured percentage
    LIMIT_EXCEEDED = "limit_exceeded"  # Over budget
    ANOMALY_DETECTED = "anomaly_detected"  # Unusual spending pattern
    RATE_SPIKE = "rate_spike"  # Sudden increase in spend rate
    NEW_PERIOD = "new_period"  # Budget period reset
    POLICY_CHANGED = "policy_changed"  # Policy was modified


def _generate_id() -> str:
    """Generate a unique identifier."""
    return str(uuid.uuid4())


def _now_utc() -> datetime:
    """Get current UTC timestamp."""
    return datetime.now(timezone.utc)


def _quantize_decimal(value: Decimal, places: int = 6) -> Decimal:
    """Quantize decimal to specified decimal places."""
    quantizer = Decimal(10) ** -places
    return value.quantize(quantizer, rounding=ROUND_HALF_UP)


@dataclass
class BudgetPolicy:
    """Defines a budget limit and its configuration.

    Attributes:
        id: Unique identifier for the policy.
        name: Human-readable policy name.
        scope: What level this budget applies to.
        scope_id: Identifier within scope (user_id, tenant_id, etc.).
        period: Time period for budget cycles.
        limit_amount: Maximum spend allowed in period.
        currency: Currency code (default USD).
        soft_limit_percent: Percentage for warning threshold (0.0-1.0).
        hard_limit_action: Action when limit is exceeded.
        rollover_enabled: Whether unused budget carries forward.
        rollover_max_percent: Maximum rollover as percent of limit.
        provider_limits: Per-provider spend limits within this budget.
        model_limits: Per-model spend limits within this budget.
        enabled: Whether this policy is active.
        priority: Higher priority policies are checked first.
        created_at: When policy was created.
        updated_at: When policy was last modified.
        metadata: Additional custom data.
    """

    name: str
    scope: BudgetScope
    limit_amount: Decimal
    id: str = field(default_factory=_generate_id)
    scope_id: Optional[str] = None
    period: BudgetPeriod = BudgetPeriod.MONTHLY
    currency: str = "USD"
    soft_limit_percent: float = 0.80
    hard_limit_action: LimitAction = LimitAction.WARN
    rollover_enabled: bool = False
    rollover_max_percent: float = 0.25
    provider_limits: Dict[str, Decimal] = field(default_factory=dict)
    model_limits: Dict[str, Decimal] = field(default_factory=dict)
    enabled: bool = True
    priority: int = 0
    created_at: datetime = field(default_factory=_now_utc)
    updated_at: datetime = field(default_factory=_now_utc)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and normalize policy fields."""
        if isinstance(self.limit_amount, (int, float)):
            self.limit_amount = Decimal(str(self.limit_amount))
        self.limit_amount = _quantize_decimal(self.limit_amount)

        if self.limit_amount < Decimal("0"):
            raise ValueError("limit_amount cannot be negative")

        if not 0.0 <= self.soft_limit_percent <= 1.0:
            raise ValueError("soft_limit_percent must be between 0.0 and 1.0")

        if not 0.0 <= self.rollover_max_percent <= 1.0:
            raise ValueError("rollover_max_percent must be between 0.0 and 1.0")

        # Normalize provider limits
        normalized_provider = {}
        for provider, limit in self.provider_limits.items():
            if isinstance(limit, (int, float)):
                limit = Decimal(str(limit))
            normalized_provider[provider] = _quantize_decimal(limit)
        self.provider_limits = normalized_provider

        # Normalize model limits
        normalized_model = {}
        for model, limit in self.model_limits.items():
            if isinstance(limit, (int, float)):
                limit = Decimal(str(limit))
            normalized_model[model] = _quantize_decimal(limit)
        self.model_limits = normalized_model

        if isinstance(self.scope, str):
            self.scope = BudgetScope(self.scope)
        if isinstance(self.period, str):
            self.period = BudgetPeriod(self.period)
        if isinstance(self.hard_limit_action, str):
            self.hard_limit_action = LimitAction(self.hard_limit_action)

    def get_soft_limit(self) -> Decimal:
        """Calculate the soft limit amount."""
        return _quantize_decimal(self.limit_amount * Decimal(str(self.soft_limit_percent)))

    def as_dict(self) -> Dict[str, Any]:
        """Serialize policy to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "scope": self.scope.value,
            "scope_id": self.scope_id,
            "period": self.period.value,
            "limit_amount": str(self.limit_amount),
            "currency": self.currency,
            "soft_limit_percent": self.soft_limit_percent,
            "hard_limit_action": self.hard_limit_action.value,
            "rollover_enabled": self.rollover_enabled,
            "rollover_max_percent": self.rollover_max_percent,
            "provider_limits": {k: str(v) for k, v in self.provider_limits.items()},
            "model_limits": {k: str(v) for k, v in self.model_limits.items()},
            "enabled": self.enabled,
            "priority": self.priority,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BudgetPolicy":
        """Create policy from dictionary."""
        data = dict(data)  # Don't mutate input

        # Parse timestamps
        for ts_field in ("created_at", "updated_at"):
            if ts_field in data and isinstance(data[ts_field], str):
                ts = data[ts_field]
                if ts.endswith("Z"):
                    ts = ts[:-1] + "+00:00"
                data[ts_field] = datetime.fromisoformat(ts)

        # Parse decimals
        if "limit_amount" in data:
            data["limit_amount"] = Decimal(str(data["limit_amount"]))

        for limit_field in ("provider_limits", "model_limits"):
            if limit_field in data:
                data[limit_field] = {
                    k: Decimal(str(v)) for k, v in data[limit_field].items()
                }

        return cls(**data)


@dataclass
class UsageRecord:
    """Records a single billable usage event.

    Attributes:
        id: Unique record identifier.
        timestamp: When the usage occurred.
        provider: Provider name (OpenAI, Anthropic, etc.).
        model: Model identifier used.
        operation_type: Type of operation performed.
        input_tokens: Number of input/prompt tokens.
        output_tokens: Number of output/completion tokens.
        total_tokens: Combined token count.
        images_generated: Number of images generated.
        image_size: Image dimensions if applicable.
        audio_seconds: Duration for audio operations.
        cost_usd: Calculated cost in USD.

        Context fields for granular budget attribution:
        user_id: User who initiated the request.
        team_id: Team/department identifier.
        project_id: Project identifier.
        job_id: Background job identifier.
        task_id: Task identifier.
        agent_id: Agent identifier (for multi-agent orchestration).
        session_id: Session/workflow identifier.
        conversation_id: Associated conversation.
        tool_id: Tool/function that was called.
        skill_id: Skill that was executed.
        request_id: Unique request identifier for tracing.
        parent_request_id: Parent request for nested/delegated calls.
        success: Whether the operation succeeded.
        metadata: Additional context.
    """

    provider: str
    model: str
    operation_type: OperationType
    cost_usd: Decimal
    id: str = field(default_factory=_generate_id)
    timestamp: datetime = field(default_factory=_now_utc)
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    images_generated: Optional[int] = None
    image_size: Optional[str] = None
    image_quality: Optional[str] = None
    audio_seconds: Optional[float] = None
    # Granular attribution context
    user_id: Optional[str] = None
    team_id: Optional[str] = None
    project_id: Optional[str] = None
    job_id: Optional[str] = None
    task_id: Optional[str] = None
    agent_id: Optional[str] = None
    session_id: Optional[str] = None
    conversation_id: Optional[str] = None
    tool_id: Optional[str] = None
    skill_id: Optional[str] = None
    request_id: Optional[str] = None
    parent_request_id: Optional[str] = None
    success: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and normalize record fields."""
        if isinstance(self.cost_usd, (int, float)):
            self.cost_usd = Decimal(str(self.cost_usd))
        self.cost_usd = _quantize_decimal(self.cost_usd)

        if self.cost_usd < Decimal("0"):
            raise ValueError("cost_usd cannot be negative")

        if isinstance(self.operation_type, str):
            self.operation_type = OperationType(self.operation_type)

        # Calculate total_tokens if not provided
        if self.total_tokens is None and self.input_tokens and self.output_tokens:
            self.total_tokens = self.input_tokens + self.output_tokens

    def as_dict(self) -> Dict[str, Any]:
        """Serialize record to dictionary."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "provider": self.provider,
            "model": self.model,
            "operation_type": self.operation_type.value,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "images_generated": self.images_generated,
            "image_size": self.image_size,
            "image_quality": self.image_quality,
            "audio_seconds": self.audio_seconds,
            "cost_usd": str(self.cost_usd),
            "user_id": self.user_id,
            "team_id": self.team_id,
            "project_id": self.project_id,
            "job_id": self.job_id,
            "task_id": self.task_id,
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "conversation_id": self.conversation_id,
            "tool_id": self.tool_id,
            "skill_id": self.skill_id,
            "request_id": self.request_id,
            "parent_request_id": self.parent_request_id,
            "success": self.success,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UsageRecord":
        """Create record from dictionary."""
        data = dict(data)

        if "timestamp" in data and isinstance(data["timestamp"], str):
            ts = data["timestamp"]
            if ts.endswith("Z"):
                ts = ts[:-1] + "+00:00"
            data["timestamp"] = datetime.fromisoformat(ts)

        if "cost_usd" in data:
            data["cost_usd"] = Decimal(str(data["cost_usd"]))

        return cls(**data)


@dataclass
class SpendSummary:
    """Summary of spending within a budget period.

    Attributes:
        policy_id: Associated budget policy.
        period_start: Start of the budget period.
        period_end: End of the budget period.
        total_spent: Total amount spent.
        limit_amount: Budget limit.
        remaining: Amount remaining in budget.
        percent_used: Percentage of budget used.
        by_provider: Breakdown by provider.
        by_model: Breakdown by model.
        by_operation: Breakdown by operation type.
        record_count: Number of usage records.
        currency: Currency code.
    """

    policy_id: str
    period_start: datetime
    period_end: datetime
    total_spent: Decimal
    limit_amount: Decimal
    currency: str = "USD"
    by_provider: Dict[str, Decimal] = field(default_factory=dict)
    by_model: Dict[str, Decimal] = field(default_factory=dict)
    by_operation: Dict[str, Decimal] = field(default_factory=dict)
    record_count: int = 0
    rollover_amount: Decimal = field(default_factory=lambda: Decimal("0"))

    def __post_init__(self) -> None:
        """Calculate derived fields."""
        for attr in ("total_spent", "limit_amount", "rollover_amount"):
            val = getattr(self, attr)
            if isinstance(val, (int, float)):
                setattr(self, attr, Decimal(str(val)))
            setattr(self, attr, _quantize_decimal(getattr(self, attr)))

    @property
    def effective_limit(self) -> Decimal:
        """Limit including any rollover."""
        return self.limit_amount + self.rollover_amount

    @property
    def remaining(self) -> Decimal:
        """Amount remaining in budget."""
        return max(Decimal("0"), self.effective_limit - self.total_spent)

    @property
    def percent_used(self) -> float:
        """Percentage of budget used (0.0-1.0+)."""
        if self.effective_limit <= 0:
            return 1.0 if self.total_spent > 0 else 0.0
        return float(self.total_spent / self.effective_limit)

    @property
    def is_over_budget(self) -> bool:
        """Check if spending exceeds limit."""
        return self.total_spent > self.effective_limit

    @property
    def overage_amount(self) -> Decimal:
        """Amount over budget (0 if under)."""
        return max(Decimal("0"), self.total_spent - self.effective_limit)

    def as_dict(self) -> Dict[str, Any]:
        """Serialize summary to dictionary."""
        return {
            "policy_id": self.policy_id,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "total_spent": str(self.total_spent),
            "limit_amount": str(self.limit_amount),
            "rollover_amount": str(self.rollover_amount),
            "effective_limit": str(self.effective_limit),
            "remaining": str(self.remaining),
            "percent_used": self.percent_used,
            "is_over_budget": self.is_over_budget,
            "overage_amount": str(self.overage_amount),
            "by_provider": {k: str(v) for k, v in self.by_provider.items()},
            "by_model": {k: str(v) for k, v in self.by_model.items()},
            "by_operation": {k: str(v) for k, v in self.by_operation.items()},
            "record_count": self.record_count,
            "currency": self.currency,
        }


@dataclass
class BudgetCheckResult:
    """Result of a budget availability check.

    Attributes:
        allowed: Whether the request is allowed.
        action: Recommended action if not fully allowed.
        policy_id: Policy that was checked.
        current_spend: Current spending in period.
        limit_amount: Budget limit.
        estimated_cost: Estimated cost of pending request.
        remaining_after: Remaining budget after request.
        warnings: Warning messages to display.
        alternative_model: Suggested cheaper alternative.
    """

    allowed: bool
    action: LimitAction
    policy_id: Optional[str] = None
    current_spend: Decimal = field(default_factory=lambda: Decimal("0"))
    limit_amount: Decimal = field(default_factory=lambda: Decimal("0"))
    estimated_cost: Decimal = field(default_factory=lambda: Decimal("0"))
    remaining_after: Decimal = field(default_factory=lambda: Decimal("0"))
    warnings: List[str] = field(default_factory=list)
    alternative_model: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize decimal fields."""
        for attr in ("current_spend", "limit_amount", "estimated_cost", "remaining_after"):
            val = getattr(self, attr)
            if isinstance(val, (int, float)):
                setattr(self, attr, Decimal(str(val)))

    def as_dict(self) -> Dict[str, Any]:
        """Serialize result to dictionary."""
        return {
            "allowed": self.allowed,
            "action": self.action.value,
            "policy_id": self.policy_id,
            "current_spend": str(self.current_spend),
            "limit_amount": str(self.limit_amount),
            "estimated_cost": str(self.estimated_cost),
            "remaining_after": str(self.remaining_after),
            "warnings": self.warnings,
            "alternative_model": self.alternative_model,
            "metadata": self.metadata,
        }


@dataclass
class BudgetAlert:
    """Budget alert notification.

    Attributes:
        id: Unique alert identifier.
        policy_id: Associated budget policy.
        severity: Alert severity level.
        trigger_type: What caused the alert.
        threshold_percent: Threshold that was triggered.
        current_spend: Spending when alert was triggered.
        limit_amount: Budget limit at time of alert.
        message: Human-readable alert message.
        triggered_at: When alert was created.
        acknowledged: Whether alert has been acknowledged.
        acknowledged_at: When alert was acknowledged.
        acknowledged_by: Who acknowledged the alert.
        notification_sent: Whether notification was dispatched.
        metadata: Additional context.
    """

    policy_id: str
    severity: AlertSeverity
    trigger_type: AlertTriggerType
    current_spend: Decimal
    limit_amount: Decimal
    message: str
    id: str = field(default_factory=_generate_id)
    threshold_percent: Optional[float] = None
    triggered_at: datetime = field(default_factory=_now_utc)
    acknowledged: bool = False
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    notification_sent: bool = False
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize fields."""
        if isinstance(self.current_spend, (int, float)):
            self.current_spend = Decimal(str(self.current_spend))
        if isinstance(self.limit_amount, (int, float)):
            self.limit_amount = Decimal(str(self.limit_amount))

        if isinstance(self.severity, str):
            self.severity = AlertSeverity(self.severity)
        if isinstance(self.trigger_type, str):
            self.trigger_type = AlertTriggerType(self.trigger_type)

    @property
    def percent_used(self) -> float:
        """Percentage of budget used when alert triggered."""
        if self.limit_amount <= 0:
            return 1.0 if self.current_spend > 0 else 0.0
        return float(self.current_spend / self.limit_amount)

    def acknowledge(self, user_id: Optional[str] = None) -> None:
        """Mark alert as acknowledged."""
        self.acknowledged = True
        self.acknowledged_at = _now_utc()
        self.acknowledged_by = user_id

    def resolve(self) -> None:
        """Mark alert as resolved."""
        self.resolved = True
        self.resolved_at = _now_utc()

    def as_dict(self) -> Dict[str, Any]:
        """Serialize alert to dictionary."""
        return {
            "id": self.id,
            "policy_id": self.policy_id,
            "severity": self.severity.value,
            "trigger_type": self.trigger_type.value,
            "threshold_percent": self.threshold_percent,
            "current_spend": str(self.current_spend),
            "limit_amount": str(self.limit_amount),
            "percent_used": self.percent_used,
            "message": self.message,
            "triggered_at": self.triggered_at.isoformat(),
            "acknowledged": self.acknowledged,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "acknowledged_by": self.acknowledged_by,
            "notification_sent": self.notification_sent,
            "resolved": self.resolved,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BudgetAlert":
        """Create alert from dictionary."""
        data = dict(data)

        for ts_field in ("triggered_at", "acknowledged_at", "resolved_at"):
            if ts_field in data and isinstance(data[ts_field], str):
                ts = data[ts_field]
                if ts.endswith("Z"):
                    ts = ts[:-1] + "+00:00"
                data[ts_field] = datetime.fromisoformat(ts)

        for dec_field in ("current_spend", "limit_amount"):
            if dec_field in data:
                data[dec_field] = Decimal(str(data[dec_field]))

        return cls(**data)
