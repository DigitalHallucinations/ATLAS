"""
ATLAS Tasks Service Package.

Provides task lifecycle management with permission checks,
domain events, and OperationResult returns.

Usage:
    from core.services.tasks import (
        TaskService,
        TaskCreate,
        TaskUpdate,
        TaskResponse,
        TaskStatus,
        TaskPriority,
    )
    
    service = TaskService(repository, message_bus)
    result = await service.create_task(actor, TaskCreate(
        title="Review Report",
        tenant_id="org_123",
        priority=TaskPriority.HIGH,
    ))

Author: ATLAS Team
Date: Jan 10, 2026
"""

from .types import (
    # Enums
    TaskStatus,
    TaskPriority,
    DependencyType,
    # Events
    TaskEvent,
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
)
from .exceptions import (
    TaskError,
    TaskNotFoundError,
    TaskTransitionError,
    TaskDependencyError,
    TaskConcurrencyError,
    TaskValidationError,
    TaskTimeoutError,
    TaskCircularDependencyError,
)
from .permissions import TaskPermissionChecker
from .service import TaskService


__all__ = [
    # Service
    "TaskService",
    
    # Permission Checker
    "TaskPermissionChecker",
    
    # Enums
    "TaskStatus",
    "TaskPriority",
    "DependencyType",
    
    # Events
    "TaskEvent",
    "TaskCreated",
    "TaskUpdated",
    "TaskDeleted",
    "TaskStatusChanged",
    "TaskAssigned",
    "TaskCompleted",
    "TaskCancelled",
    "TaskAgentAssigned",
    "SubtaskCreated",
    
    # DTOs
    "TaskCreate",
    "TaskUpdate",
    "TaskFilters",
    "TaskResult",
    "DependencyCreate",
    "TaskResponse",
    "TaskListResponse",
    
    # Exceptions
    "TaskError",
    "TaskNotFoundError",
    "TaskTransitionError",
    "TaskDependencyError",
    "TaskConcurrencyError",
    "TaskValidationError",
    "TaskTimeoutError",
    "TaskCircularDependencyError",
]
