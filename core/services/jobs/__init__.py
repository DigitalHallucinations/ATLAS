"""
ATLAS Job Service

Provides job lifecycle management following the ATLAS service pattern.

This package provides:
- JobService: Job CRUD, lifecycle transitions, SOTA features
- Permission checking with tenant isolation
- MessageBus event emission

Author: ATLAS Team
Date: Jan 10, 2026
"""

from .types import (
    # Re-exported models
    Job,
    JobStatus,
    JobRun,
    JobRunStatus,
    JobTaskLink,
    JobAssignment,
    JobAssignmentStatus,
    JobSchedule,
    JobEvent,
    JobEventType,
    
    # SOTA enums
    CancellationMode,
    DependencyType,
    
    # Domain events
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
    ScheduleCreate,
    JobResult,
    TaskLinkCreate,
    JobCheckpoint,
    
    # Response DTOs
    JobResponse,
    JobRunResponse,
    JobListResponse,
)
from .exceptions import (
    JobError,
    JobNotFoundError,
    JobTransitionError,
    JobDependencyError,
    JobConcurrencyError,
    JobValidationError,
    JobTimeoutError,
    JobBudgetExceededError,
)
from .permissions import JobPermissionChecker
from .service import JobService


__all__ = [
    # Service
    "JobService",
    "JobPermissionChecker",
    
    # Models
    "Job",
    "JobStatus",
    "JobRun",
    "JobRunStatus",
    "JobTaskLink",
    "JobAssignment",
    "JobAssignmentStatus",
    "JobSchedule",
    "JobEvent",
    "JobEventType",
    
    # SOTA enums
    "CancellationMode",
    "DependencyType",
    
    # Domain Events
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
    
    # DTOs
    "JobCreate",
    "JobUpdate",
    "JobFilters",
    "ScheduleCreate",
    "JobResult",
    "TaskLinkCreate",
    "JobCheckpoint",
    
    # Response DTOs
    "JobResponse",
    "JobRunResponse",
    "JobListResponse",
    
    # Exceptions
    "JobError",
    "JobNotFoundError",
    "JobTransitionError",
    "JobDependencyError",
    "JobConcurrencyError",
    "JobValidationError",
    "JobTimeoutError",
    "JobBudgetExceededError",
]
