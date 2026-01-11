"""
Persona service package for ATLAS.

Provides a clean service layer for persona management with
permission checks, validation, and event publishing.

Includes SOTA enhancements:
- Safety/Guardrails per persona
- Memory (working, episodic, semantic)
- Analytics and self-improvement
- Switching and handoff protocols

Author: ATLAS Team
Date: Jan 10, 2026
"""

from .exceptions import (
    PersonaError,
    PersonaNotFoundError,
    PersonaValidationError,
    PersonaAlreadyExistsError,
    PersonaDeleteError,
    PersonaActiveError,
    PersonaIOError,
    PersonaSchemaError,
)
from .permissions import (
    PersonaPermissionChecker,
    PERMISSION_PERSONAS_READ,
    PERMISSION_PERSONAS_WRITE,
    PERMISSION_PERSONAS_DELETE,
    PERMISSION_PERSONAS_ADMIN,
    PERMISSION_PERSONAS_ACTIVATE,
)
from .service import PersonaService
from .safety import (
    PersonaSafetyService,
    PersonaSafetyError,
    ActionBlockedError,
    ApprovalRequiredError,
)
from .memory import (
    PersonaMemoryService,
    PersonaMemoryError,
)
from .analytics import (
    PersonaAnalyticsService,
    PersonaAnalyticsError,
)
from .switching import (
    PersonaSwitchingService,
    PersonaSwitchingError,
    HandoffFailedError,
)
from .types import (
    # Events
    PersonaCreated,
    PersonaUpdated,
    PersonaDeleted,
    PersonaActivated,
    PersonaDeactivated,
    PersonaValidated,
    # SOTA Events
    PersonaSafetyViolation,
    PersonaHandoffInitiated,
    PersonaHandoffCompleted,
    PersonaMemoryUpdated,
    PersonaMetricsRecorded,
    # DTOs - Input
    PersonaCreate,
    PersonaUpdate,
    PersonaFilters,
    # DTOs - Output
    PersonaResponse,
    PersonaListResponse,
    PersonaSummary,
    PersonaCapabilities,
    ValidationResult,
    # Safety types
    PersonaSafetyPolicy,
    ContentFilterRule,
    PIIPolicy,
    PIIHandlingMode,
    EscalationPath,
    HITLTrigger,
    ActionCheckResult,
    ApprovalRequest,
    ApprovalStatus,
    RiskLevel,
    # Memory types
    PersonaMemoryContext,
    PersonaKnowledge,
    LearnedFact,
    EpisodicMemory,
    # Analytics types
    PersonaPerformanceMetrics,
    TokenUsage,
    PersonaVariant,
    PromptRefinement,
    # Switching types
    PersonaHandoff,
    HandoffContext,
    HandoffStatus,
    SwitchingRule,
    SwitchTriggerType,
    FallbackChain,
    # Multi-agent types
    OrchestratorRole,
    PersonaAgentManifest,
    DelegatedTask,
    PersonaLedgerEntry,
)
from .validation import PersonaValidator, create_validator

__all__ = [
    # Core Service
    "PersonaService",
    # SOTA Services
    "PersonaSafetyService",
    "PersonaMemoryService",
    "PersonaAnalyticsService",
    "PersonaSwitchingService",
    # Exceptions
    "PersonaError",
    "PersonaNotFoundError",
    "PersonaValidationError",
    "PersonaAlreadyExistsError",
    "PersonaDeleteError",
    "PersonaActiveError",
    "PersonaIOError",
    "PersonaSchemaError",
    "PersonaSafetyError",
    "ActionBlockedError",
    "ApprovalRequiredError",
    "PersonaMemoryError",
    "PersonaAnalyticsError",
    "PersonaSwitchingError",
    "HandoffFailedError",
    # Permissions
    "PersonaPermissionChecker",
    "PERMISSION_PERSONAS_READ",
    "PERMISSION_PERSONAS_WRITE",
    "PERMISSION_PERSONAS_DELETE",
    "PERMISSION_PERSONAS_ADMIN",
    "PERMISSION_PERSONAS_ACTIVATE",
    # Core Events
    "PersonaCreated",
    "PersonaUpdated",
    "PersonaDeleted",
    "PersonaActivated",
    "PersonaDeactivated",
    "PersonaValidated",
    # SOTA Events
    "PersonaSafetyViolation",
    "PersonaHandoffInitiated",
    "PersonaHandoffCompleted",
    "PersonaMemoryUpdated",
    "PersonaMetricsRecorded",
    # DTOs - Input
    "PersonaCreate",
    "PersonaUpdate",
    "PersonaFilters",
    # DTOs - Output
    "PersonaResponse",
    "PersonaListResponse",
    "PersonaSummary",
    "PersonaCapabilities",
    "ValidationResult",
    # Safety types
    "PersonaSafetyPolicy",
    "ContentFilterRule",
    "PIIPolicy",
    "PIIHandlingMode",
    "EscalationPath",
    "HITLTrigger",
    "ActionCheckResult",
    "ApprovalRequest",
    "ApprovalStatus",
    "RiskLevel",
    # Memory types
    "PersonaMemoryContext",
    "PersonaKnowledge",
    "LearnedFact",
    "EpisodicMemory",
    # Analytics types
    "PersonaPerformanceMetrics",
    "TokenUsage",
    "PersonaVariant",
    "PromptRefinement",
    # Switching types
    "PersonaHandoff",
    "HandoffContext",
    "HandoffStatus",
    "SwitchingRule",
    "SwitchTriggerType",
    "FallbackChain",
    # Multi-agent types
    "OrchestratorRole",
    "PersonaAgentManifest",
    "DelegatedTask",
    "PersonaLedgerEntry",
    # Validation
    "PersonaValidator",
    "create_validator",
]
