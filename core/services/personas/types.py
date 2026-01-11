"""
Persona service types and domain events.

Defines DTOs for service operations and domain events for
integration with the ATLAS messaging system.

Author: ATLAS Team
Date: Jan 10, 2026
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence
from uuid import UUID


def _now_utc() -> datetime:
    """Get current UTC timestamp."""
    return datetime.now(timezone.utc)


def _generate_uuid() -> str:
    """Generate a new UUID string."""
    return str(uuid.uuid4())


# =============================================================================
# Domain Events
# =============================================================================


@dataclass(frozen=True)
class PersonaCreated:
    """Emitted when a persona is created."""

    persona_name: str
    tenant_id: str
    actor_id: str
    provider: Optional[str] = None
    model: Optional[str] = None
    actor_type: str = "user"
    event_type: str = "persona.created"
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
            "persona_name": self.persona_name,
            "provider": self.provider,
            "model": self.model,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass(frozen=True)
class PersonaUpdated:
    """Emitted when a persona is updated."""

    persona_name: str
    changed_fields: tuple[str, ...]
    tenant_id: str
    actor_id: str
    actor_type: str = "user"
    event_type: str = "persona.updated"
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
            "persona_name": self.persona_name,
            "changed_fields": list(self.changed_fields),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass(frozen=True)
class PersonaDeleted:
    """Emitted when a persona is deleted."""

    persona_name: str
    tenant_id: str
    actor_id: str
    actor_type: str = "user"
    event_type: str = "persona.deleted"
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
            "persona_name": self.persona_name,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass(frozen=True)
class PersonaActivated:
    """Emitted when a persona is activated for a user/session."""

    persona_name: str
    tenant_id: str
    actor_id: str
    previous_persona: Optional[str] = None
    actor_type: str = "user"
    event_type: str = "persona.activated"
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
            "persona_name": self.persona_name,
            "previous_persona": self.previous_persona,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass(frozen=True)
class PersonaDeactivated:
    """Emitted when a persona is deactivated."""

    persona_name: str
    tenant_id: str
    actor_id: str
    actor_type: str = "user"
    event_type: str = "persona.deactivated"
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
            "persona_name": self.persona_name,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass(frozen=True)
class PersonaValidated:
    """Emitted when persona validation is performed."""

    persona_name: str
    tenant_id: str
    actor_id: str
    is_valid: bool
    error_count: int = 0
    warning_count: int = 0
    actor_type: str = "user"
    event_type: str = "persona.validated"
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
            "persona_name": self.persona_name,
            "is_valid": self.is_valid,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "timestamp": self.timestamp.isoformat(),
        }


# =============================================================================
# DTOs - Input Types
# =============================================================================


@dataclass
class PersonaCreate:
    """DTO for creating a new persona."""

    name: str
    content: Dict[str, str]  # start_locked, editable_content, end_locked
    meaning: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    speech_provider: Optional[str] = None
    voice: Optional[str] = None
    sys_info_enabled: bool = False
    user_profile_enabled: bool = False
    allowed_tools: Optional[List[str]] = None
    allowed_skills: Optional[List[str]] = None
    persona_type: Optional[Dict[str, Any]] = None  # Type flags (personal_assistant, etc.)
    image_generation: Optional[Dict[str, Any]] = None
    ui_state: Optional[Dict[str, Any]] = None


@dataclass
class PersonaUpdate:
    """DTO for updating an existing persona."""

    # All fields optional - only provided fields are updated
    name: Optional[str] = None  # For rename operations
    meaning: Optional[str] = None
    content: Optional[Dict[str, str]] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    speech_provider: Optional[str] = None
    voice: Optional[str] = None
    sys_info_enabled: Optional[bool] = None
    user_profile_enabled: Optional[bool] = None
    allowed_tools: Optional[List[str]] = None
    allowed_skills: Optional[List[str]] = None
    persona_type: Optional[Dict[str, Any]] = None
    image_generation: Optional[Dict[str, Any]] = None
    ui_state: Optional[Dict[str, Any]] = None


@dataclass
class PersonaFilters:
    """Filters for listing personas."""

    name_pattern: Optional[str] = None  # Glob or substring match
    provider: Optional[str] = None
    has_tool: Optional[str] = None  # Filter by allowed tool
    has_skill: Optional[str] = None  # Filter by allowed skill
    persona_type: Optional[str] = None  # Filter by type flag (e.g., "personal_assistant")
    limit: int = 100
    offset: int = 0


# =============================================================================
# DTOs - Output Types
# =============================================================================


@dataclass
class PersonaCapabilities:
    """Summary of persona capabilities."""

    tools: List[str]
    skills: List[str]
    has_calendar_access: bool = False
    has_calendar_write: bool = False
    has_terminal_read: bool = False
    has_terminal_write: bool = False
    has_image_generation: bool = False
    persona_types: List[str] = field(default_factory=list)  # Enabled type flags


@dataclass
class PersonaResponse:
    """Full persona response with all details."""

    name: str
    meaning: Optional[str]
    content: Dict[str, str]
    provider: Optional[str]
    model: Optional[str]
    speech_provider: Optional[str]
    voice: Optional[str]
    sys_info_enabled: bool
    user_profile_enabled: bool
    allowed_tools: List[str]
    allowed_skills: List[str]
    persona_type: Dict[str, Any]
    capabilities: PersonaCapabilities
    image_generation: Optional[Dict[str, Any]] = None
    ui_state: Optional[Dict[str, Any]] = None

    @classmethod
    def from_persona_dict(cls, persona: Dict[str, Any]) -> "PersonaResponse":
        """Create a PersonaResponse from a raw persona dictionary."""
        content = persona.get("content", {})
        if isinstance(content, str):
            content = {"editable_content": content, "start_locked": "", "end_locked": ""}

        persona_type = persona.get("type", {})
        allowed_tools = persona.get("allowed_tools", [])
        if isinstance(allowed_tools, str):
            allowed_tools = [allowed_tools]
        allowed_skills = persona.get("allowed_skills", [])
        if isinstance(allowed_skills, str):
            allowed_skills = [allowed_skills]

        # Build capabilities
        personal_assistant = persona_type.get("personal_assistant", {})
        image_gen = persona.get("image_generation", {})

        capabilities = PersonaCapabilities(
            tools=list(allowed_tools),
            skills=list(allowed_skills),
            has_calendar_access=_coerce_bool(personal_assistant.get("access_to_calendar", False)),
            has_calendar_write=_coerce_bool(personal_assistant.get("calendar_write_enabled", False)),
            has_terminal_read=_coerce_bool(personal_assistant.get("terminal_read_enabled", False)),
            has_terminal_write=_coerce_bool(personal_assistant.get("terminal_write_enabled", False)),
            has_image_generation=_coerce_bool(image_gen.get("enabled", False)),
            persona_types=[k for k, v in persona_type.items() if _coerce_bool(v.get("enabled", False)) if isinstance(v, dict)],
        )

        return cls(
            name=persona.get("name", ""),
            meaning=persona.get("meaning"),
            content=content,
            provider=persona.get("provider"),
            model=persona.get("model"),
            speech_provider=persona.get("Speech_provider"),
            voice=persona.get("voice"),
            sys_info_enabled=_coerce_bool(persona.get("sys_info_enabled", False)),
            user_profile_enabled=_coerce_bool(persona.get("user_profile_enabled", False)),
            allowed_tools=list(allowed_tools),
            allowed_skills=list(allowed_skills),
            persona_type=persona_type,
            capabilities=capabilities,
            image_generation=image_gen if image_gen else None,
            ui_state=persona.get("ui_state"),
        )


@dataclass
class PersonaListResponse:
    """Paginated list of personas."""

    personas: List[PersonaSummary]
    total: int
    limit: int
    offset: int
    has_more: bool


@dataclass
class PersonaSummary:
    """Lightweight persona summary for list operations."""

    name: str
    meaning: Optional[str]
    provider: Optional[str]
    model: Optional[str]
    tool_count: int = 0
    skill_count: int = 0


@dataclass
class ValidationResult:
    """Result of persona validation."""

    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    persona_name: Optional[str] = None


# =============================================================================
# Helpers
# =============================================================================


def _coerce_bool(value: Any) -> bool:
    """Convert serialized persona truthy values into booleans."""
    if isinstance(value, str):
        lowered = value.strip().lower()
        return lowered in {"true", "1", "yes", "on", "enabled"}
    return bool(value)


# =============================================================================
# SOTA Enhancement Types - Safety & Guardrails
# =============================================================================


class RiskLevel:
    """Risk level classifications for actions."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PIIHandlingMode:
    """How to handle PII (Personally Identifiable Information)."""
    ALLOW = "allow"  # PII allowed in outputs
    MASK = "mask"  # Replace PII with placeholders
    BLOCK = "block"  # Block responses containing PII
    WARN = "warn"  # Allow but log warning


class ApprovalStatus:
    """Status of an approval request."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    AUTO_APPROVED = "auto_approved"


@dataclass
class ContentFilterRule:
    """A rule for filtering content."""
    
    rule_id: str
    name: str
    pattern: str  # Regex pattern or keyword
    action: str  # "block", "warn", "mask", "allow"
    applies_to: str  # "input", "output", "both"
    priority: int = 100
    enabled: bool = True
    message: Optional[str] = None  # Custom message when triggered


@dataclass
class PIIPolicy:
    """Policy for PII handling."""
    
    mode: str = PIIHandlingMode.MASK
    detect_names: bool = True
    detect_emails: bool = True
    detect_phones: bool = True
    detect_addresses: bool = True
    detect_ssn: bool = True
    detect_credit_cards: bool = True
    custom_patterns: List[str] = field(default_factory=list)
    exempted_fields: List[str] = field(default_factory=list)


@dataclass
class EscalationPath:
    """Defines how to escalate from one persona to another or to human."""
    
    path_id: str = field(default_factory=_generate_uuid)
    from_persona: str = ""  # Empty = any persona
    to_persona: Optional[str] = None  # None = escalate to human
    triggers: List[str] = field(default_factory=list)  # Conditions that trigger escalation
    auto_escalate: bool = False  # Auto-escalate without user confirmation
    timeout_seconds: int = 300  # Timeout for escalation response
    priority: int = 100
    enabled: bool = True


@dataclass
class HITLTrigger:
    """Trigger for Human-in-the-Loop approval."""
    
    trigger_id: str = field(default_factory=_generate_uuid)
    name: str = ""
    action_pattern: str = ""  # Pattern matching action names
    risk_threshold: str = RiskLevel.HIGH  # Trigger when risk >= threshold
    require_reason: bool = True  # Require human to provide reason
    timeout_seconds: int = 600
    auto_approve_after_timeout: bool = False
    notify_channels: List[str] = field(default_factory=list)  # Email, slack, etc.
    enabled: bool = True


@dataclass
class PersonaSafetyPolicy:
    """Safety configuration for a persona."""
    
    persona_id: str
    
    # Action controls
    allowed_actions: List[str] = field(default_factory=list)  # Whitelist (empty = all allowed)
    blocked_actions: List[str] = field(default_factory=list)  # Blacklist
    require_approval_actions: List[str] = field(default_factory=list)  # HITL required
    
    # Content policies
    input_filters: List[ContentFilterRule] = field(default_factory=list)
    output_filters: List[ContentFilterRule] = field(default_factory=list)
    pii_policy: PIIPolicy = field(default_factory=PIIPolicy)
    
    # Rate limits
    max_actions_per_minute: int = 60
    max_tokens_per_conversation: int = 100000
    max_tool_calls_per_turn: int = 10
    
    # Risk tolerance
    risk_threshold: float = 0.7  # 0.0 - 1.0, actions above this need review
    auto_escalation_threshold: float = 0.9  # Above this, auto-escalate
    
    # HITL configuration
    hitl_triggers: List[HITLTrigger] = field(default_factory=list)
    escalation_paths: List[EscalationPath] = field(default_factory=list)
    
    # Audit
    log_all_actions: bool = True
    log_retention_days: int = 90
    
    created_at: datetime = field(default_factory=_now_utc)
    updated_at: datetime = field(default_factory=_now_utc)


@dataclass
class ActionCheckResult:
    """Result of checking an action against safety policy."""
    
    allowed: bool
    action: str
    persona_id: str
    risk_level: str = RiskLevel.LOW
    risk_score: float = 0.0
    requires_approval: bool = False
    approval_id: Optional[str] = None
    blocked_reason: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    matched_rules: List[str] = field(default_factory=list)
    checked_at: datetime = field(default_factory=_now_utc)


@dataclass
class ApprovalRequest:
    """Request for human approval of an action."""
    
    approval_id: str = field(default_factory=_generate_uuid)
    persona_id: str = ""
    actor_id: str = ""
    action: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    risk_score: float = 0.0
    reason_required: str = ""  # Why approval is needed
    status: str = ApprovalStatus.PENDING
    approver_id: Optional[str] = None
    approver_reason: Optional[str] = None
    created_at: datetime = field(default_factory=_now_utc)
    expires_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None


# =============================================================================
# SOTA Enhancement Types - Memory
# =============================================================================


@dataclass
class LearnedFact:
    """A fact learned by a persona from interactions."""
    
    fact_id: str = field(default_factory=_generate_uuid)
    subject: str = ""  # What the fact is about
    predicate: str = ""  # The relationship
    object: str = ""  # The value
    confidence: float = 1.0  # 0.0 - 1.0
    source: str = ""  # Where this was learned (conversation_id, etc.)
    learned_at: datetime = field(default_factory=_now_utc)
    last_verified: Optional[datetime] = None
    verification_count: int = 0


@dataclass
class PersonaMemoryContext:
    """Working memory scoped to a persona."""
    
    persona_id: str
    user_id: str  # Memory is per-user-per-persona
    scratchpad: Dict[str, Any] = field(default_factory=dict)  # Key-value working memory
    active_goals: List[str] = field(default_factory=list)  # Current objectives
    task_context: Optional[str] = None  # Current task description
    conversation_summary: Optional[str] = None  # Summary of recent conversation
    pending_actions: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=_now_utc)
    last_accessed: datetime = field(default_factory=_now_utc)
    expires_at: Optional[datetime] = None


@dataclass
class PersonaKnowledge:
    """Semantic knowledge specific to a persona-user pair."""
    
    persona_id: str
    user_id: str
    learned_facts: List[LearnedFact] = field(default_factory=list)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    domain_expertise: Dict[str, float] = field(default_factory=dict)  # Topic -> confidence
    entity_mentions: Dict[str, int] = field(default_factory=dict)  # Entity -> mention count
    last_updated: datetime = field(default_factory=_now_utc)


@dataclass
class EpisodicMemory:
    """A single episode (interaction) for episodic memory."""
    
    episode_id: str = field(default_factory=_generate_uuid)
    persona_id: str = ""
    user_id: str = ""
    conversation_id: Optional[str] = None
    summary: str = ""  # What happened in this episode
    outcome: Optional[str] = None  # How it ended (success, failure, etc.)
    key_entities: List[str] = field(default_factory=list)
    key_topics: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None  # Vector embedding for similarity search
    timestamp: datetime = field(default_factory=_now_utc)


# =============================================================================
# SOTA Enhancement Types - Analytics & Self-Improvement
# =============================================================================


@dataclass
class TokenUsage:
    """Token usage statistics."""
    
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0


@dataclass
class PersonaPerformanceMetrics:
    """Track persona effectiveness."""
    
    persona_id: str
    period_start: datetime
    period_end: datetime
    
    # Usage metrics
    total_interactions: int = 0
    total_conversations: int = 0
    avg_response_time_ms: float = 0.0
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    
    # Quality metrics
    task_success_rate: float = 0.0
    user_satisfaction_score: Optional[float] = None  # From feedback, 0-5
    escalation_rate: float = 0.0  # How often it hands off
    retry_rate: float = 0.0  # How often responses need retry
    error_rate: float = 0.0
    
    # Capability metrics
    tools_used: Dict[str, int] = field(default_factory=dict)  # Tool -> usage count
    skills_invoked: Dict[str, int] = field(default_factory=dict)
    capability_gaps: List[str] = field(default_factory=list)  # Requested but unavailable
    
    # Guardrails metrics
    actions_blocked: int = 0
    approvals_requested: int = 0
    approvals_granted: int = 0
    content_filtered: int = 0


@dataclass
class PersonaVariant:
    """A variation of a persona for A/B testing."""
    
    variant_id: str = field(default_factory=_generate_uuid)
    base_persona_id: str = ""
    name: str = ""
    description: str = ""
    modifications: Dict[str, Any] = field(default_factory=dict)  # What's different
    traffic_percentage: float = 0.0  # 0-100, percentage of traffic
    metrics: Optional[PersonaPerformanceMetrics] = None
    is_active: bool = True
    created_at: datetime = field(default_factory=_now_utc)
    created_by: Optional[str] = None


@dataclass
class PromptRefinement:
    """A suggested or applied refinement to persona prompts."""
    
    refinement_id: str = field(default_factory=_generate_uuid)
    persona_id: str = ""
    field_path: str = ""  # e.g., "content.editable_content"
    original_value: str = ""
    suggested_value: str = ""
    reason: str = ""  # Why this refinement was suggested
    source: str = ""  # "ai_suggested", "user_modified", "a/b_winner"
    status: str = "pending"  # pending, applied, rejected, rolled_back
    impact_score: Optional[float] = None  # Measured improvement
    created_at: datetime = field(default_factory=_now_utc)
    applied_at: Optional[datetime] = None
    applied_by: Optional[str] = None


# =============================================================================
# SOTA Enhancement Types - Switching & Handoff
# =============================================================================


class HandoffStatus:
    """Status of a persona handoff."""
    PENDING = "pending"
    ACCEPTED = "accepted"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SwitchTriggerType:
    """Types of triggers for persona switching."""
    EXPLICIT = "explicit"  # User explicitly requested
    CAPABILITY_MISMATCH = "capability_mismatch"  # Persona can't handle request
    DOMAIN_CHANGE = "domain_change"  # Conversation topic changed
    ESCALATION = "escalation"  # Complexity exceeds scope
    RULE_BASED = "rule_based"  # Matched a switching rule
    PERFORMANCE = "performance"  # Current persona underperforming


@dataclass
class HandoffContext:
    """What gets transferred during handoff."""
    
    conversation_summary: str = ""
    active_goals: List[str] = field(default_factory=list)
    working_memory_snapshot: Dict[str, Any] = field(default_factory=dict)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    pending_actions: List[Dict[str, Any]] = field(default_factory=list)
    recent_messages: List[Dict[str, Any]] = field(default_factory=list)  # Last N messages
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PersonaHandoff:
    """Context transfer between personas."""
    
    handoff_id: str = field(default_factory=_generate_uuid)
    from_persona: str = ""
    to_persona: str = ""
    user_id: str = ""
    conversation_id: Optional[str] = None
    context: HandoffContext = field(default_factory=HandoffContext)
    trigger_type: str = SwitchTriggerType.EXPLICIT
    reason: str = ""
    status: str = HandoffStatus.PENDING
    initiated_at: datetime = field(default_factory=_now_utc)
    accepted_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None


@dataclass
class SwitchingRule:
    """Rule for conditional persona switching."""
    
    rule_id: str = field(default_factory=_generate_uuid)
    name: str = ""
    description: str = ""
    trigger_pattern: str = ""  # Regex or intent pattern
    from_persona: Optional[str] = None  # None = any persona
    to_persona: str = ""
    priority: int = 100  # Lower = higher priority
    conditions: List[Dict[str, Any]] = field(default_factory=list)  # Additional conditions
    require_confirmation: bool = True  # Ask user before switching
    preserve_context: bool = True  # Transfer context on switch
    enabled: bool = True
    created_at: datetime = field(default_factory=_now_utc)


@dataclass
class FallbackChain:
    """Chain of fallback personas if primary fails."""
    
    chain_id: str = field(default_factory=_generate_uuid)
    name: str = ""
    primary_persona: str = ""
    fallback_personas: List[str] = field(default_factory=list)  # Ordered list
    trigger_on_error: bool = True
    trigger_on_timeout: bool = True
    trigger_on_capability_gap: bool = True
    max_fallbacks: int = 3
    enabled: bool = True


# =============================================================================
# SOTA Enhancement Types - Multi-Agent Coordination
# =============================================================================


class OrchestratorRole:
    """Role of a persona in multi-agent orchestration."""
    WORKER = "worker"  # Can only execute assigned tasks
    ORCHESTRATOR = "orchestrator"  # Can delegate and coordinate
    HYBRID = "hybrid"  # Can do both


@dataclass
class PersonaAgentManifest:
    """Persona capabilities for multi-agent orchestration."""
    
    persona_id: str
    capabilities: List[str] = field(default_factory=list)
    specializations: List[str] = field(default_factory=list)  # Domains of expertise
    max_parallel_tasks: int = 1
    can_delegate: bool = False
    can_be_delegated_to: bool = True
    orchestration_role: str = OrchestratorRole.WORKER
    estimated_response_time_ms: int = 5000
    reliability_score: float = 1.0  # 0.0 - 1.0


@dataclass
class DelegatedTask:
    """A task delegated to a persona in multi-agent context."""
    
    task_id: str = field(default_factory=_generate_uuid)
    parent_task_id: Optional[str] = None
    delegated_to: str = ""  # persona_id
    delegated_by: str = ""  # persona_id
    task_description: str = ""
    expected_output: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"  # pending, in_progress, completed, failed
    result: Optional[Dict[str, Any]] = None
    created_at: datetime = field(default_factory=_now_utc)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    timeout_seconds: int = 300


@dataclass
class PersonaLedgerEntry:
    """Entry in a persona's task/progress ledger (Magentic-One style)."""
    
    entry_id: str = field(default_factory=_generate_uuid)
    persona_id: str = ""
    ledger_type: str = ""  # "task" or "progress"
    content: str = ""
    status: Optional[str] = None
    related_task_id: Optional[str] = None
    created_at: datetime = field(default_factory=_now_utc)


# =============================================================================
# SOTA Domain Events
# =============================================================================


@dataclass(frozen=True)
class PersonaSafetyViolation:
    """Emitted when a safety policy is violated."""
    
    persona_id: str
    actor_id: str
    action: str
    violation_type: str  # "blocked", "filtered", "escalated"
    risk_score: float
    details: str
    tenant_id: str = ""
    event_type: str = "persona.safety_violation"
    entity_id: str = field(default_factory=_generate_uuid)
    timestamp: datetime = field(default_factory=_now_utc)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "entity_id": self.entity_id,
            "persona_id": self.persona_id,
            "actor_id": self.actor_id,
            "action": self.action,
            "violation_type": self.violation_type,
            "risk_score": self.risk_score,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass(frozen=True)
class PersonaHandoffInitiated:
    """Emitted when a persona handoff begins."""
    
    handoff_id: str
    from_persona: str
    to_persona: str
    user_id: str
    reason: str
    trigger_type: str
    tenant_id: str = ""
    event_type: str = "persona.handoff_initiated"
    entity_id: str = field(default_factory=_generate_uuid)
    timestamp: datetime = field(default_factory=_now_utc)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "entity_id": self.entity_id,
            "handoff_id": self.handoff_id,
            "from_persona": self.from_persona,
            "to_persona": self.to_persona,
            "user_id": self.user_id,
            "reason": self.reason,
            "trigger_type": self.trigger_type,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass(frozen=True)
class PersonaHandoffCompleted:
    """Emitted when a persona handoff completes."""
    
    handoff_id: str
    from_persona: str
    to_persona: str
    user_id: str
    success: bool
    duration_ms: int
    tenant_id: str = ""
    event_type: str = "persona.handoff_completed"
    entity_id: str = field(default_factory=_generate_uuid)
    timestamp: datetime = field(default_factory=_now_utc)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "entity_id": self.entity_id,
            "handoff_id": self.handoff_id,
            "from_persona": self.from_persona,
            "to_persona": self.to_persona,
            "user_id": self.user_id,
            "success": self.success,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass(frozen=True)
class PersonaMemoryUpdated:
    """Emitted when persona memory is updated."""
    
    persona_id: str
    user_id: str
    memory_type: str  # "working", "episodic", "semantic"
    operation: str  # "set", "clear", "learn"
    key: Optional[str] = None
    tenant_id: str = ""
    event_type: str = "persona.memory_updated"
    entity_id: str = field(default_factory=_generate_uuid)
    timestamp: datetime = field(default_factory=_now_utc)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "entity_id": self.entity_id,
            "persona_id": self.persona_id,
            "user_id": self.user_id,
            "memory_type": self.memory_type,
            "operation": self.operation,
            "key": self.key,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass(frozen=True)
class PersonaMetricsRecorded:
    """Emitted when persona performance metrics are recorded."""
    
    persona_id: str
    interaction_count: int
    success_rate: float
    avg_response_time_ms: float
    tenant_id: str = ""
    event_type: str = "persona.metrics_recorded"
    entity_id: str = field(default_factory=_generate_uuid)
    timestamp: datetime = field(default_factory=_now_utc)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "entity_id": self.entity_id,
            "persona_id": self.persona_id,
            "interaction_count": self.interaction_count,
            "success_rate": self.success_rate,
            "avg_response_time_ms": self.avg_response_time_ms,
            "timestamp": self.timestamp.isoformat(),
        }

