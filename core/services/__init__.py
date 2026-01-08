"""ATLAS services layer.

High-level service facades that coordinate multiple subsystems:
- RAGService: Retrieval-Augmented Generation pipeline
- CalendarEventService: Calendar event management
- NotificationService: Notification delivery
- BudgetPolicyService: Budget policy management and enforcement
- BudgetTrackingService: Usage recording and spend aggregation
- ConversationService: Conversation management (future)
- ProviderService: LLM provider orchestration (future)

Common service patterns and types are available from the .common submodule.
"""

from core.services.rag import RAGService, RAGServiceStatus

# Calendar services
from core.services.calendar import CalendarEventService, CalendarPermissionChecker

# Budget services
from core.services.budget import (
    # Services
    BudgetPolicyService,
    BudgetTrackingService,
    BudgetPermissionChecker,
    # Policy events
    BudgetPolicyCreated,
    BudgetPolicyUpdated,
    BudgetPolicyDeleted,
    # Tracking events
    BudgetUsageRecorded,
    BudgetThresholdReached,
    # Policy DTOs
    BudgetCheckRequest,
    BudgetCheckResponse,
    BudgetPolicyCreate,
    BudgetPolicyUpdate,
    # Tracking DTOs
    UsageRecordCreate,
    LLMUsageCreate,
    ImageUsageCreate,
    UsageSummaryRequest,
    SpendBreakdown,
    SpendTrend,
    # Exceptions
    BudgetError,
    BudgetPolicyNotFoundError,
    BudgetExceededError,
    BudgetValidationError,
)

# Notification services
from core.services.notifications import (
    NotificationService,
    DesktopNotificationService,
    DummyNotificationService,
    CompositeNotificationService,
)

# Re-export common service types for convenient access
from core.services.common import (
    Actor,
    BusinessRuleError,
    ConfigurationError,
    ConflictError,
    DomainEvent,
    DomainEventPublisher,
    ExternalServiceError,
    NotFoundError,
    OperationResult,
    PermissionChecker,
    PermissionDeniedError,
    RateLimitError,
    Repository,
    SearchableRepository,
    Service,
    ServiceError,
    ValidationError,
    create_domain_event_publisher,
)

__all__ = [
    # Existing services
    "RAGService",
    "RAGServiceStatus",
    
    # Calendar services
    "CalendarEventService",
    "CalendarPermissionChecker",
    
    # Budget services
    "BudgetPolicyService",
    "BudgetTrackingService",
    "BudgetPermissionChecker",
    "BudgetPolicyCreated",
    "BudgetPolicyUpdated",
    "BudgetPolicyDeleted",
    "BudgetUsageRecorded",
    "BudgetThresholdReached",
    "BudgetCheckRequest",
    "BudgetCheckResponse",
    "BudgetPolicyCreate",
    "BudgetPolicyUpdate",
    "UsageRecordCreate",
    "LLMUsageCreate",
    "ImageUsageCreate",
    "UsageSummaryRequest",
    "SpendBreakdown",
    "SpendTrend",
    "BudgetError",
    "BudgetPolicyNotFoundError",
    "BudgetExceededError",
    "BudgetValidationError",
    
    # Notification services
    "NotificationService",
    "DesktopNotificationService",
    "DummyNotificationService",
    "CompositeNotificationService",
    
    # Common types (most frequently used)
    "OperationResult",
    "Actor", 
    "DomainEvent",
    "PermissionChecker",
    
    # Common exceptions
    "ServiceError",
    "ValidationError",
    "NotFoundError", 
    "ConflictError",
    "PermissionDeniedError",
    "ConfigurationError",
    "ExternalServiceError",
    "RateLimitError",
    "BusinessRuleError",
    
    # Common protocols
    "Service",
    "Repository", 
    "SearchableRepository",
    
    # Messaging
    "DomainEventPublisher",
    "create_domain_event_publisher",
]
