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

# Job services
from core.services.jobs import (
    JobService,
    JobPermissionChecker,
    # Events
    JobCreated,
    JobUpdated,
    JobDeleted,
    JobStatusChanged,
    JobScheduled,
    JobStarted,
    JobSucceeded,
    JobFailed,
    JobCancelled,
    JobCheckpointed,
    JobAgentAssigned,
    # DTOs
    JobCreate,
    JobUpdate,
    JobFilters,
    JobCheckpoint,
    JobResponse,
    JobListResponse,
    # Enums
    CancellationMode,
    DependencyType as JobDependencyType,
    # Exceptions
    JobError,
    JobNotFoundError,
    JobTransitionError,
    JobDependencyError,
    JobConcurrencyError,
    JobValidationError,
    JobTimeoutError,
    JobBudgetExceededError,
)

# Task services
from core.services.tasks import (
    TaskService,
    TaskPermissionChecker,
    # Events
    TaskCreated,
    TaskUpdated,
    TaskDeleted,
    TaskStatusChanged,
    TaskAssigned,
    TaskCompleted,
    TaskCancelled,
    TaskAgentAssigned,
    SubtaskCreated,
    # DTOs
    TaskCreate,
    TaskUpdate,
    TaskFilters,
    TaskResult,
    DependencyCreate,
    TaskResponse,
    TaskListResponse,
    # Enums
    TaskStatus,
    TaskPriority,
    DependencyType as TaskDependencyType,
    # Exceptions
    TaskError,
    TaskNotFoundError,
    TaskTransitionError,
    TaskDependencyError,
    TaskConcurrencyError,
    TaskValidationError,
    TaskTimeoutError,
    TaskCircularDependencyError,
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
    
    # Job services
    "JobService",
    "JobPermissionChecker",
    "JobCreated",
    "JobUpdated",
    "JobDeleted",
    "JobStatusChanged",
    "JobScheduled",
    "JobStarted",
    "JobSucceeded",
    "JobFailed",
    "JobCancelled",
    "JobCheckpointed",
    "JobAgentAssigned",
    "JobCreate",
    "JobUpdate",
    "JobFilters",
    "JobCheckpoint",
    "JobResponse",
    "JobListResponse",
    "CancellationMode",
    "JobDependencyType",
    "JobError",
    "JobNotFoundError",
    "JobTransitionError",
    "JobDependencyError",
    "JobConcurrencyError",
    "JobValidationError",
    "JobTimeoutError",
    "JobBudgetExceededError",
    
    # Task services
    "TaskService",
    "TaskPermissionChecker",
    "TaskCreated",
    "TaskUpdated",
    "TaskDeleted",
    "TaskStatusChanged",
    "TaskAssigned",
    "TaskCompleted",
    "TaskCancelled",
    "TaskAgentAssigned",
    "SubtaskCreated",
    "TaskCreate",
    "TaskUpdate",
    "TaskFilters",
    "TaskResult",
    "DependencyCreate",
    "TaskResponse",
    "TaskListResponse",
    "TaskStatus",
    "TaskPriority",
    "TaskDependencyType",
    "TaskError",
    "TaskNotFoundError",
    "TaskTransitionError",
    "TaskDependencyError",
    "TaskConcurrencyError",
    "TaskValidationError",
    "TaskTimeoutError",
    "TaskCircularDependencyError",
    
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
