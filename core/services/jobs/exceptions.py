"""
Job service exceptions.

Author: ATLAS Team
Date: Jan 10, 2026
"""

from __future__ import annotations


class JobError(Exception):
    """Base class for job service errors."""
    
    def __init__(self, message: str, error_code: str | None = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code


class JobNotFoundError(JobError):
    """Raised when a job is not found."""
    
    def __init__(self, job_id: str, tenant_id: str | None = None):
        message = f"Job not found: {job_id}"
        if tenant_id:
            message += f" in tenant {tenant_id}"
        super().__init__(message, "JOB_NOT_FOUND")
        self.job_id = job_id
        self.tenant_id = tenant_id


class JobTransitionError(JobError):
    """Raised when an invalid lifecycle transition is requested."""
    
    def __init__(self, message: str, from_status: str | None = None, to_status: str | None = None):
        super().__init__(message, "INVALID_TRANSITION")
        self.from_status = from_status
        self.to_status = to_status


class JobDependencyError(JobError):
    """Raised when linked tasks prevent the requested transition."""
    
    def __init__(self, message: str, incomplete_tasks: list[str] | None = None):
        super().__init__(message, "DEPENDENCY_ERROR")
        self.incomplete_tasks = incomplete_tasks or []


class JobConcurrencyError(JobError):
    """Raised when a concurrent modification conflict occurs."""
    
    def __init__(self, job_id: str, message: str | None = None):
        super().__init__(message or f"Concurrent modification conflict for job: {job_id}", "CONCURRENCY_ERROR")
        self.job_id = job_id


class JobValidationError(JobError):
    """Raised when job data validation fails."""
    
    def __init__(self, message: str, field: str | None = None):
        super().__init__(message, "VALIDATION_ERROR")
        self.field = field


class JobTimeoutError(JobError):
    """Raised when a job execution times out."""
    
    def __init__(self, job_id: str, timeout_seconds: int):
        super().__init__(f"Job {job_id} timed out after {timeout_seconds} seconds", "TIMEOUT_ERROR")
        self.job_id = job_id
        self.timeout_seconds = timeout_seconds


class JobBudgetExceededError(JobError):
    """Raised when job would exceed budget limits."""
    
    def __init__(self, job_id: str, estimated_cost: str, available_budget: str):
        super().__init__(
            f"Job {job_id} estimated cost {estimated_cost} exceeds available budget {available_budget}",
            "BUDGET_EXCEEDED"
        )
        self.job_id = job_id
        self.estimated_cost = estimated_cost
        self.available_budget = available_budget


__all__ = [
    "JobError",
    "JobNotFoundError",
    "JobTransitionError",
    "JobDependencyError",
    "JobConcurrencyError",
    "JobValidationError",
    "JobTimeoutError",
    "JobBudgetExceededError",
]
