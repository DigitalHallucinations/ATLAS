"""
Task service exceptions.

Author: ATLAS Team
Date: Jan 10, 2026
"""

from __future__ import annotations


class TaskError(Exception):
    """Base class for task service errors."""
    
    def __init__(self, message: str, error_code: str | None = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code


class TaskNotFoundError(TaskError):
    """Raised when a task is not found."""
    
    def __init__(self, task_id: str, tenant_id: str | None = None):
        message = f"Task not found: {task_id}"
        if tenant_id:
            message += f" in tenant {tenant_id}"
        super().__init__(message, "TASK_NOT_FOUND")
        self.task_id = task_id
        self.tenant_id = tenant_id


class TaskTransitionError(TaskError):
    """Raised when an invalid lifecycle transition is requested."""
    
    def __init__(self, message: str, from_status: str | None = None, to_status: str | None = None):
        super().__init__(message, "INVALID_TRANSITION")
        self.from_status = from_status
        self.to_status = to_status


class TaskDependencyError(TaskError):
    """Raised when dependencies prevent the requested transition."""
    
    def __init__(self, message: str, incomplete_dependencies: list[str] | None = None):
        super().__init__(message, "DEPENDENCY_ERROR")
        self.incomplete_dependencies = incomplete_dependencies or []


class TaskConcurrencyError(TaskError):
    """Raised when a concurrent modification conflict occurs."""
    
    def __init__(self, task_id: str, message: str | None = None):
        super().__init__(message or f"Concurrent modification conflict for task: {task_id}", "CONCURRENCY_ERROR")
        self.task_id = task_id


class TaskValidationError(TaskError):
    """Raised when task data validation fails."""
    
    def __init__(self, message: str, field: str | None = None):
        super().__init__(message, "VALIDATION_ERROR")
        self.field = field


class TaskTimeoutError(TaskError):
    """Raised when a task execution times out."""
    
    def __init__(self, task_id: str, timeout_seconds: int):
        super().__init__(f"Task {task_id} timed out after {timeout_seconds} seconds", "TIMEOUT_ERROR")
        self.task_id = task_id
        self.timeout_seconds = timeout_seconds


class TaskCircularDependencyError(TaskError):
    """Raised when a circular dependency would be created."""
    
    def __init__(self, task_id: str, cycle_path: list[str] | None = None):
        super().__init__(f"Circular dependency detected for task: {task_id}", "CIRCULAR_DEPENDENCY")
        self.task_id = task_id
        self.cycle_path = cycle_path or []


__all__ = [
    "TaskError",
    "TaskNotFoundError",
    "TaskTransitionError",
    "TaskDependencyError",
    "TaskConcurrencyError",
    "TaskValidationError",
    "TaskTimeoutError",
    "TaskCircularDependencyError",
]
