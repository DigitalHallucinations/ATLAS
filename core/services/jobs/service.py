"""
Job service for ATLAS.

Provides job lifecycle management with permission checks,
MessageBus events, and OperationResult returns.

Author: ATLAS Team
Date: Jan 10, 2026
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Mapping, Optional, Sequence
from uuid import UUID

from core.services.common import Actor, OperationResult
from core.services.common.exceptions import PermissionDeniedError

from .exceptions import (
    JobError,
    JobNotFoundError,
    JobTransitionError,
    JobDependencyError,
    JobConcurrencyError,
    JobValidationError,
)
from .permissions import JobPermissionChecker
from .types import (
    JobStatus,
    JobRunStatus,
    CancellationMode,
    DependencyType,
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
    ScheduleCreate,
    JobResult,
    TaskLinkCreate,
    JobCheckpoint,
    JobResponse,
    JobListResponse,
)

if TYPE_CHECKING:
    from modules.job_store.repository import JobStoreRepository
    from modules.orchestration.job_scheduler import JobScheduler
    from core.messaging import MessageBus


logger = logging.getLogger(__name__)


# Allowed status transitions
_ALLOWED_TRANSITIONS: Dict[JobStatus, set[JobStatus]] = {
    JobStatus.DRAFT: {JobStatus.SCHEDULED, JobStatus.RUNNING, JobStatus.CANCELLED},
    JobStatus.SCHEDULED: {JobStatus.RUNNING, JobStatus.CANCELLED},
    JobStatus.RUNNING: {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED},
    JobStatus.SUCCEEDED: set(),
    JobStatus.FAILED: set(),
    JobStatus.CANCELLED: set(),
}


def _coerce_status(value: Any) -> JobStatus:
    """Coerce a value to JobStatus enum."""
    if isinstance(value, JobStatus):
        return value
    if isinstance(value, str):
        return JobStatus(value.lower())
    raise ValueError(f"Cannot coerce {value} to JobStatus")


def _now_utc() -> datetime:
    """Get current UTC timestamp."""
    return datetime.now(timezone.utc)


class JobService:
    """
    Application service that coordinates job lifecycle operations.
    
    Provides:
    - CRUD operations with permission checks
    - Status transitions with validation
    - MessageBus event emission
    - SOTA features: execution context, checkpointing, agent assignment
    
    All operations return OperationResult for consistent error handling.
    
    Example:
        service = JobService(repository, message_bus, permission_checker)
        
        result = await service.create_job(actor, JobCreate(
            name="Data Analysis",
            tenant_id="org_123",
            assigned_agent="analyst-persona",
            estimated_cost=Decimal("0.50"),
        ))
        
        if result.success:
            job = result.data
            print(f"Created job: {job.id}")
    """
    
    def __init__(
        self,
        repository: "JobStoreRepository",
        message_bus: Optional["MessageBus"] = None,
        permission_checker: Optional[JobPermissionChecker] = None,
    ) -> None:
        """Initialize the job service.
        
        Args:
            repository: Job store repository for persistence
            message_bus: Optional message bus for event emission
            permission_checker: Optional permission checker (creates default if None)
        """
        self._repository = repository
        self._message_bus = message_bus
        self._permissions = permission_checker or JobPermissionChecker()
    
    def _emit(self, event: Any) -> None:
        """Emit an event to the message bus."""
        if self._message_bus is not None:
            try:
                self._message_bus.publish(event.event_type, event.to_dict())
            except Exception as e:
                logger.warning(f"Failed to emit event {event.event_type}: {e}")
    
    def _extract_sota_metadata(self, job_data: JobCreate | JobUpdate) -> Dict[str, Any]:
        """Extract SOTA fields into metadata dict."""
        sota_fields = {}
        
        if hasattr(job_data, 'assigned_agent') and job_data.assigned_agent is not None:
            sota_fields['assigned_agent'] = job_data.assigned_agent
        if hasattr(job_data, 'estimated_cost') and job_data.estimated_cost is not None:
            sota_fields['estimated_cost'] = str(job_data.estimated_cost)
        if hasattr(job_data, 'actual_cost') and job_data.actual_cost is not None:
            sota_fields['actual_cost'] = str(job_data.actual_cost)
        if hasattr(job_data, 'timeout_seconds') and job_data.timeout_seconds is not None:
            sota_fields['timeout_seconds'] = job_data.timeout_seconds
        if hasattr(job_data, 'execution_context') and job_data.execution_context is not None:
            sota_fields['execution_context'] = job_data.execution_context
        if hasattr(job_data, 'checkpoint_data') and job_data.checkpoint_data is not None:
            sota_fields['checkpoint_data'] = job_data.checkpoint_data
        if hasattr(job_data, 'plan_id') and job_data.plan_id is not None:
            sota_fields['plan_id'] = job_data.plan_id
        if hasattr(job_data, 'plan_step_index') and job_data.plan_step_index is not None:
            sota_fields['plan_step_index'] = job_data.plan_step_index
            
        return sota_fields
    
    # =========================================================================
    # CRUD Operations
    # =========================================================================
    
    def create_job(
        self,
        actor: Actor,
        job_data: JobCreate,
    ) -> OperationResult[JobResponse]:
        """Create a new job.
        
        Args:
            actor: The actor performing the operation
            job_data: Job creation data
            
        Returns:
            OperationResult containing the created job or error
        """
        try:
            # Check permissions
            self._permissions.require_write(actor, job_data.tenant_id)
            
            # Build metadata with SOTA fields
            metadata = dict(job_data.metadata or {})
            metadata.update(self._extract_sota_metadata(job_data))
            
            # Create via repository
            record = self._repository.create_job(
                job_data.name,
                tenant_id=job_data.tenant_id,
                description=job_data.description,
                status=None,  # Default to DRAFT
                owner_id=job_data.owner_id or actor.id,
                conversation_id=job_data.conversation_id,
                metadata=metadata,
            )
            
            # Emit event
            event = JobCreated(
                job_id=str(record["id"]),
                job_name=record["name"],
                tenant_id=record["tenant_id"],
                actor_id=actor.id,
                actor_type=actor.type,
                status=record.get("status", "draft"),
                owner_id=str(record["owner_id"]) if record.get("owner_id") else None,
                assigned_agent=job_data.assigned_agent,
                estimated_cost=job_data.estimated_cost,
            )
            self._emit(event)
            
            return OperationResult.success(JobResponse.from_dict(record))
            
        except PermissionDeniedError as e:
            logger.error(f"Permission denied creating job: {e}")
            return OperationResult.failure(str(e), e.error_code or "PERMISSION_DENIED")
        except JobValidationError as e:
            return OperationResult.failure(e.message, e.error_code)
        except Exception as e:
            logger.exception(f"Failed to create job: {e}")
            return OperationResult.failure(f"Failed to create job: {e}", "INTERNAL_ERROR")
    
    def get_job(
        self,
        actor: Actor,
        job_id: str,
        tenant_id: str,
        *,
        with_schedule: bool = False,
        with_runs: bool = False,
        with_tasks: bool = False,
    ) -> OperationResult[JobResponse]:
        """Get a job by ID.
        
        Args:
            actor: The actor performing the operation
            job_id: The job ID
            tenant_id: The tenant ID
            with_schedule: Include schedule data
            with_runs: Include run history
            with_tasks: Include linked tasks
            
        Returns:
            OperationResult containing the job or error
        """
        try:
            # Get job first to check permissions
            record = self._repository.get_job(
                job_id,
                tenant_id=tenant_id,
                with_schedule=with_schedule,
                with_runs=with_runs,
                with_events=False,
            )
            
            # Check permissions against the specific job
            self._permissions.require_job_read(actor, record)
            
            # Optionally load tasks
            if with_tasks:
                tasks = self._repository.list_linked_tasks(job_id, tenant_id=tenant_id)
                record["tasks"] = tasks
            
            return OperationResult.success(JobResponse.from_dict(record))
            
        except PermissionDeniedError as e:
            logger.error(f"Permission denied getting job: {e}")
            return OperationResult.failure(str(e), e.error_code or "PERMISSION_DENIED")
        except Exception as e:
            if "not found" in str(e).lower():
                return OperationResult.failure(f"Job not found: {job_id}", "JOB_NOT_FOUND")
            logger.exception(f"Failed to get job: {e}")
            return OperationResult.failure(f"Failed to get job: {e}", "INTERNAL_ERROR")
    
    def update_job(
        self,
        actor: Actor,
        job_id: str,
        tenant_id: str,
        updates: JobUpdate,
        *,
        expected_updated_at: datetime | None = None,
    ) -> OperationResult[JobResponse]:
        """Update an existing job.
        
        Args:
            actor: The actor performing the operation
            job_id: The job ID
            tenant_id: The tenant ID
            updates: Job update data
            expected_updated_at: Optimistic concurrency check
            
        Returns:
            OperationResult containing the updated job or error
        """
        try:
            # Get current job
            snapshot = self._repository.get_job(job_id, tenant_id=tenant_id)
            
            # Check permissions
            self._permissions.require_job_write(actor, snapshot)
            
            # Build changes dict
            changes: Dict[str, Any] = {}
            if updates.name is not None:
                changes["name"] = updates.name
            if updates.description is not None:
                changes["description"] = updates.description
            if updates.owner_id is not None:
                changes["owner_id"] = updates.owner_id
            
            # Merge SOTA fields into metadata
            current_metadata = dict(snapshot.get("metadata", {}))
            sota_updates = self._extract_sota_metadata(updates)
            if sota_updates:
                current_metadata.update(sota_updates)
                changes["metadata"] = current_metadata
            elif updates.metadata is not None:
                current_metadata.update(updates.metadata)
                changes["metadata"] = current_metadata
            
            if not changes:
                return OperationResult.success(JobResponse.from_dict(snapshot))
            
            # Update via repository
            record = self._repository.update_job(
                job_id,
                tenant_id=tenant_id,
                changes=changes,
                expected_updated_at=expected_updated_at,
            )
            
            # Emit event
            event = JobUpdated(
                job_id=str(record["id"]),
                job_name=record["name"],
                changed_fields=updates.changed_fields(),
                tenant_id=record["tenant_id"],
                actor_id=actor.id,
                actor_type=actor.type,
            )
            self._emit(event)
            
            return OperationResult.success(JobResponse.from_dict(record))
            
        except PermissionDeniedError as e:
            logger.error(f"Permission denied updating job: {e}")
            return OperationResult.failure(str(e), e.error_code or "PERMISSION_DENIED")
        except Exception as e:
            if "concurrency" in str(e).lower():
                return OperationResult.failure("Job was modified by another transaction", "CONCURRENCY_ERROR")
            logger.exception(f"Failed to update job: {e}")
            return OperationResult.failure(f"Failed to update job: {e}", "INTERNAL_ERROR")
    
    def delete_job(
        self,
        actor: Actor,
        job_id: str,
        tenant_id: str,
    ) -> OperationResult[None]:
        """Delete a job.
        
        Args:
            actor: The actor performing the operation
            job_id: The job ID
            tenant_id: The tenant ID
            
        Returns:
            OperationResult indicating success or error
        """
        try:
            # Get job to check permissions
            snapshot = self._repository.get_job(job_id, tenant_id=tenant_id)
            
            # Check permissions (delete requires owner or admin)
            self._permissions.require_job_delete(actor, snapshot)
            
            # Delete via repository (TODO: add delete_job to repository)
            self._repository.delete_job(job_id, tenant_id=tenant_id)  # type: ignore[attr-defined]
            
            # Emit event
            event = JobDeleted(
                job_id=str(snapshot["id"]),
                job_name=snapshot["name"],
                tenant_id=snapshot["tenant_id"],
                actor_id=actor.id,
                actor_type=actor.type,
            )
            self._emit(event)
            
            return OperationResult.success(None)
            
        except Exception as e:
            if "not found" in str(e).lower():
                return OperationResult.failure(f"Job not found: {job_id}", "JOB_NOT_FOUND")
            logger.exception(f"Failed to delete job: {e}")
            return OperationResult.failure(f"Failed to delete job: {e}", "INTERNAL_ERROR")
    
    def list_jobs(
        self,
        actor: Actor,
        tenant_id: str,
        filters: JobFilters | None = None,
    ) -> OperationResult[JobListResponse]:
        """List jobs with optional filtering.
        
        Args:
            actor: The actor performing the operation
            tenant_id: The tenant ID
            filters: Optional filters
            
        Returns:
            OperationResult containing the job list or error
        """
        try:
            # Check permissions
            self._permissions.require_read(actor, tenant_id)
            
            filters = filters or JobFilters()
            
            # Query via repository
            result = self._repository.list_jobs(
                tenant_id=tenant_id,
                status=filters.status,
                owner_id=filters.owner_id,
                cursor=filters.cursor,
                limit=filters.limit,
            )
            
            # Handle both dict and list returns from repository
            if isinstance(result, dict):
                jobs_list = result.get("jobs", [])
                next_cursor = result.get("next_cursor")
            else:
                jobs_list = result if isinstance(result, list) else []
                next_cursor = None
            
            jobs = [JobResponse.from_dict(j) for j in jobs_list if isinstance(j, dict)]
            
            return OperationResult.success(JobListResponse(
                jobs=jobs,
                next_cursor=next_cursor,
            ))
            
        except Exception as e:
            logger.exception(f"Failed to list jobs: {e}")
            return OperationResult.failure(f"Failed to list jobs: {e}", "INTERNAL_ERROR")
    
    # =========================================================================
    # Lifecycle Operations
    # =========================================================================
    
    def transition_job(
        self,
        actor: Actor,
        job_id: str,
        tenant_id: str,
        target_status: JobStatus | str,
        *,
        reason: str | None = None,
        expected_updated_at: datetime | None = None,
    ) -> OperationResult[JobResponse]:
        """Transition a job to a new status.
        
        Args:
            actor: The actor performing the operation
            job_id: The job ID
            tenant_id: The tenant ID
            target_status: The target status
            reason: Optional reason for the transition
            expected_updated_at: Optimistic concurrency check
            
        Returns:
            OperationResult containing the updated job or error
        """
        try:
            # Get current job
            snapshot = self._repository.get_job(
                job_id,
                tenant_id=tenant_id,
                with_schedule=True,
            )
            
            # Check execute permission for status changes
            self._permissions.require_job_execute(actor, snapshot)
            
            current_status = _coerce_status(snapshot["status"])
            desired_status = _coerce_status(target_status) if isinstance(target_status, str) else target_status
            
            # Same status is a no-op
            if desired_status == current_status:
                return OperationResult.success(JobResponse.from_dict(snapshot))
            
            # Validate transition
            allowed = _ALLOWED_TRANSITIONS.get(current_status, set())
            if desired_status not in allowed:
                return OperationResult.failure(
                    f"Cannot transition job from {current_status.value} to {desired_status.value}",
                    "INVALID_TRANSITION"
                )
            
            # Update via repository
            reference_timestamp = expected_updated_at or snapshot.get("updated_at")
            record = self._repository.update_job(
                job_id,
                tenant_id=tenant_id,
                changes={"status": desired_status.value},
                expected_updated_at=reference_timestamp,
            )
            
            # Emit event
            event = JobStatusChanged(
                job_id=str(record["id"]),
                job_name=record["name"],
                from_status=current_status.value,
                to_status=desired_status.value,
                reason=reason,
                tenant_id=record["tenant_id"],
                actor_id=actor.id,
                actor_type=actor.type,
            )
            self._emit(event)
            
            return OperationResult.success(JobResponse.from_dict(record))
            
        except Exception as e:
            if "concurrency" in str(e).lower():
                return OperationResult.failure("Job was modified by another transaction", "CONCURRENCY_ERROR")
            logger.exception(f"Failed to transition job: {e}")
            return OperationResult.failure(f"Failed to transition job: {e}", "INTERNAL_ERROR")
    
    def schedule_job(
        self,
        actor: Actor,
        job_id: str,
        tenant_id: str,
        schedule: ScheduleCreate,
    ) -> OperationResult[JobResponse]:
        """Schedule a job for execution.
        
        Args:
            actor: The actor performing the operation
            job_id: The job ID
            tenant_id: The tenant ID
            schedule: Schedule configuration
            
        Returns:
            OperationResult containing the updated job or error
        """
        try:
            # Get current job
            snapshot = self._repository.get_job(job_id, tenant_id=tenant_id)
            
            # Check permissions
            self._permissions.require_job_execute(actor, snapshot)
            
            # Create or update schedule (using upsert_schedule)
            self._repository.upsert_schedule(
                job_id,
                tenant_id=tenant_id,
                schedule_type=schedule.schedule_type,
                expression=schedule.expression,
                timezone_name=schedule.timezone,
                metadata=schedule.metadata,
            )
            
            # Transition to SCHEDULED if still in DRAFT
            current_status = _coerce_status(snapshot["status"])
            if current_status == JobStatus.DRAFT:
                self._repository.update_job(
                    job_id,
                    tenant_id=tenant_id,
                    changes={"status": JobStatus.SCHEDULED.value},
                )
            
            # Get updated job
            record = self._repository.get_job(job_id, tenant_id=tenant_id, with_schedule=True)
            
            # Emit event
            schedule_data = record.get("schedule", {})
            event = JobScheduled(
                job_id=str(record["id"]),
                job_name=record["name"],
                schedule_type=schedule.schedule_type,
                expression=schedule.expression,
                next_run_at=schedule_data.get("next_run_at"),
                tenant_id=record["tenant_id"],
                actor_id=actor.id,
                actor_type=actor.type,
            )
            self._emit(event)
            
            return OperationResult.success(JobResponse.from_dict(record))
            
        except Exception as e:
            logger.exception(f"Failed to schedule job: {e}")
            return OperationResult.failure(f"Failed to schedule job: {e}", "INTERNAL_ERROR")
    
    def start_job(
        self,
        actor: Actor,
        job_id: str,
        tenant_id: str,
        *,
        assigned_agent: str | None = None,
    ) -> OperationResult[JobResponse]:
        """Start job execution.
        
        Args:
            actor: The actor performing the operation
            job_id: The job ID
            tenant_id: The tenant ID
            assigned_agent: Optional agent to assign
            
        Returns:
            OperationResult containing the updated job or error
        """
        try:
            # Get current job
            snapshot = self._repository.get_job(job_id, tenant_id=tenant_id, with_runs=True)
            
            # Check permissions
            self._permissions.require_job_execute(actor, snapshot)
            
            current_status = _coerce_status(snapshot["status"])
            if current_status == JobStatus.RUNNING:
                return OperationResult.failure("Job is already running", "ALREADY_RUNNING")
            if current_status == JobStatus.DRAFT:
                return OperationResult.failure("Cannot start job directly from draft status. Use submit_job first.", "INVALID_TRANSITION")
            if current_status in (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED):
                return OperationResult.failure(f"Cannot start job in {current_status.value} status", "INVALID_TRANSITION")
            
            # Create a new run
            runs = snapshot.get("runs", [])
            run_number = len(runs) + 1
            run_record = self._repository.create_run(
                job_id,
                tenant_id=tenant_id,
                status=JobRunStatus.RUNNING,
                metadata={"started_by": actor.id, "assigned_agent": assigned_agent},
            )
            
            # Update job status and agent assignment
            changes: Dict[str, Any] = {"status": JobStatus.RUNNING.value}
            if assigned_agent:
                metadata = dict(snapshot.get("metadata", {}))
                previous_agent = metadata.get("assigned_agent")
                metadata["assigned_agent"] = assigned_agent
                changes["metadata"] = metadata
            
            record = self._repository.update_job(job_id, tenant_id=tenant_id, changes=changes)
            
            # Emit events
            event = JobStarted(
                job_id=str(record["id"]),
                job_name=record["name"],
                run_number=run_number,
                assigned_agent=assigned_agent,
                tenant_id=record["tenant_id"],
                actor_id=actor.id,
                actor_type=actor.type,
            )
            self._emit(event)
            
            if assigned_agent:
                agent_event = JobAgentAssigned(
                    job_id=str(record["id"]),
                    job_name=record["name"],
                    assigned_agent=assigned_agent,
                    previous_agent=snapshot.get("metadata", {}).get("assigned_agent"),
                    tenant_id=record["tenant_id"],
                    actor_id=actor.id,
                    actor_type=actor.type,
                )
                self._emit(agent_event)
            
            return OperationResult.success(JobResponse.from_dict(record))
            
        except Exception as e:
            logger.exception(f"Failed to start job: {e}")
            return OperationResult.failure(f"Failed to start job: {e}", "INTERNAL_ERROR")
    
    def complete_job(
        self,
        actor: Actor,
        job_id: str,
        tenant_id: str,
        result: JobResult,
    ) -> OperationResult[JobResponse]:
        """Mark a job as successfully completed.
        
        Args:
            actor: The actor performing the operation
            job_id: The job ID
            tenant_id: The tenant ID
            result: Job completion result
            
        Returns:
            OperationResult containing the updated job or error
        """
        try:
            # Get current job
            snapshot = self._repository.get_job(job_id, tenant_id=tenant_id, with_runs=True)
            
            # Check permissions
            self._permissions.require_job_execute(actor, snapshot)
            
            current_status = _coerce_status(snapshot["status"])
            if current_status != JobStatus.RUNNING:
                return OperationResult.failure(
                    f"Cannot complete job in {current_status.value} status",
                    "INVALID_TRANSITION"
                )
            
            # Update metadata with result
            metadata = dict(snapshot.get("metadata", {}))
            if result.actual_cost is not None:
                metadata["actual_cost"] = str(result.actual_cost)
            if result.result_data is not None:
                metadata["result_data"] = result.result_data
            if result.execution_time_ms is not None:
                metadata["execution_time_ms"] = result.execution_time_ms
            if result.tokens_used is not None:
                metadata["tokens_used"] = result.tokens_used
            if result.tool_calls is not None:
                metadata["tool_calls"] = result.tool_calls
            
            # Update job
            record = self._repository.update_job(
                job_id,
                tenant_id=tenant_id,
                changes={"status": JobStatus.SUCCEEDED.value, "metadata": metadata},
            )
            
            # Update current run
            runs = snapshot.get("runs", [])
            if runs:
                current_run = runs[-1]
                self._repository.update_run(
                    current_run["id"],
                    tenant_id=tenant_id,
                    changes={
                        "status": JobRunStatus.SUCCEEDED,
                        "finished_at": _now_utc(),
                        "metadata": {"result_summary": result.result_summary},
                    },
                )
            
            # Get run number
            run_number = len(runs) if runs else 1
            
            # Emit event
            event = JobSucceeded(
                job_id=str(record["id"]),
                job_name=record["name"],
                run_number=run_number,
                actual_cost=result.actual_cost,
                result_summary=result.result_summary,
                tenant_id=record["tenant_id"],
                actor_id=actor.id,
                actor_type=actor.type,
            )
            self._emit(event)
            
            return OperationResult.success(JobResponse.from_dict(record))
            
        except Exception as e:
            logger.exception(f"Failed to complete job: {e}")
            return OperationResult.failure(f"Failed to complete job: {e}", "INTERNAL_ERROR")
    
    def fail_job(
        self,
        actor: Actor,
        job_id: str,
        tenant_id: str,
        error: str,
        *,
        error_code: str | None = None,
        retryable: bool = True,
    ) -> OperationResult[JobResponse]:
        """Mark a job as failed.
        
        Args:
            actor: The actor performing the operation
            job_id: The job ID
            tenant_id: The tenant ID
            error: Error message
            error_code: Optional error code
            retryable: Whether the job can be retried
            
        Returns:
            OperationResult containing the updated job or error
        """
        try:
            # Get current job
            snapshot = self._repository.get_job(job_id, tenant_id=tenant_id, with_runs=True)
            
            # Check permissions
            self._permissions.require_job_execute(actor, snapshot)
            
            current_status = _coerce_status(snapshot["status"])
            if current_status != JobStatus.RUNNING:
                return OperationResult.failure(
                    f"Cannot fail job in {current_status.value} status",
                    "INVALID_TRANSITION"
                )
            
            # Update metadata with error
            metadata = dict(snapshot.get("metadata", {}))
            metadata["last_error"] = error
            metadata["last_error_code"] = error_code
            metadata["retryable"] = retryable
            
            # Update job
            record = self._repository.update_job(
                job_id,
                tenant_id=tenant_id,
                changes={"status": JobStatus.FAILED.value, "metadata": metadata},
            )
            
            # Update current run
            runs = snapshot.get("runs", [])
            if runs:
                current_run = runs[-1]
                self._repository.update_run(
                    current_run["id"],
                    tenant_id=tenant_id,
                    changes={
                        "status": JobRunStatus.FAILED,
                        "finished_at": _now_utc(),
                        "metadata": {"error": error, "error_code": error_code},
                    },
                )
            
            run_number = len(runs) if runs else 1
            
            # Emit event
            event = JobFailed(
                job_id=str(record["id"]),
                job_name=record["name"],
                run_number=run_number,
                error=error,
                error_code=error_code,
                retryable=retryable,
                tenant_id=record["tenant_id"],
                actor_id=actor.id,
                actor_type=actor.type,
            )
            self._emit(event)
            
            return OperationResult.success(JobResponse.from_dict(record))
            
        except Exception as e:
            logger.exception(f"Failed to fail job: {e}")
            return OperationResult.failure(f"Failed to fail job: {e}", "INTERNAL_ERROR")
    
    def cancel_job(
        self,
        actor: Actor,
        job_id: str,
        tenant_id: str,
        reason: str,
        *,
        mode: CancellationMode = CancellationMode.GRACEFUL,
    ) -> OperationResult[JobResponse]:
        """Cancel a job.
        
        Args:
            actor: The actor performing the operation
            job_id: The job ID
            tenant_id: The tenant ID
            reason: Cancellation reason
            mode: Cancellation mode (graceful or hard)
            
        Returns:
            OperationResult containing the updated job or error
        """
        try:
            # Get current job
            snapshot = self._repository.get_job(job_id, tenant_id=tenant_id, with_runs=True)
            
            # Check permissions
            self._permissions.require_job_execute(actor, snapshot)
            
            current_status = _coerce_status(snapshot["status"])
            if current_status in (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED):
                return OperationResult.failure(
                    f"Cannot cancel job in {current_status.value} status",
                    "INVALID_TRANSITION"
                )
            
            # Update metadata with cancellation info
            metadata = dict(snapshot.get("metadata", {}))
            metadata["cancellation_reason"] = reason
            metadata["cancellation_mode"] = mode.name
            metadata["cancelled_by"] = actor.id
            metadata["cancelled_at"] = _now_utc().isoformat()
            
            # Update job
            record = self._repository.update_job(
                job_id,
                tenant_id=tenant_id,
                changes={"status": JobStatus.CANCELLED.value, "metadata": metadata},
            )
            
            # Update current run if running
            runs = snapshot.get("runs", [])
            if runs and current_status == JobStatus.RUNNING:
                current_run = runs[-1]
                self._repository.update_run(
                    current_run["id"],
                    tenant_id=tenant_id,
                    changes={
                        "status": JobRunStatus.CANCELLED,
                        "finished_at": _now_utc(),
                        "metadata": {"reason": reason, "mode": mode.name},
                    },
                )
            
            # Emit event
            event = JobCancelled(
                job_id=str(record["id"]),
                job_name=record["name"],
                reason=reason,
                cancellation_mode=mode.name,
                tenant_id=record["tenant_id"],
                actor_id=actor.id,
                actor_type=actor.type,
            )
            self._emit(event)
            
            return OperationResult.success(JobResponse.from_dict(record))
            
        except Exception as e:
            logger.exception(f"Failed to cancel job: {e}")
            return OperationResult.failure(f"Failed to cancel job: {e}", "INTERNAL_ERROR")
    
    # =========================================================================
    # SOTA Features
    # =========================================================================
    
    def save_checkpoint(
        self,
        actor: Actor,
        job_id: str,
        tenant_id: str,
        checkpoint: JobCheckpoint,
    ) -> OperationResult[JobResponse]:
        """Save a checkpoint for a running job (SOTA).
        
        Args:
            actor: The actor performing the operation
            job_id: The job ID
            tenant_id: The tenant ID
            checkpoint: Checkpoint data
            
        Returns:
            OperationResult containing the updated job or error
        """
        try:
            # Get current job
            snapshot = self._repository.get_job(job_id, tenant_id=tenant_id)
            
            # Check permissions
            self._permissions.require_job_execute(actor, snapshot)
            
            current_status = _coerce_status(snapshot["status"])
            if current_status != JobStatus.RUNNING:
                return OperationResult.failure(
                    f"Cannot checkpoint job in {current_status.value} status",
                    "INVALID_STATE"
                )
            
            # Update metadata with checkpoint
            metadata = dict(snapshot.get("metadata", {}))
            metadata["checkpoint_data"] = {
                "step_index": checkpoint.step_index,
                "step_name": checkpoint.step_name,
                "state": checkpoint.state,
                "execution_context": checkpoint.execution_context,
                "created_at": checkpoint.created_at.isoformat(),
            }
            metadata["last_checkpoint_at"] = _now_utc().isoformat()
            
            # Update job
            record = self._repository.update_job(
                job_id,
                tenant_id=tenant_id,
                changes={"metadata": metadata},
            )
            
            # Emit event
            event = JobCheckpointed(
                job_id=str(record["id"]),
                job_name=record["name"],
                checkpoint_step=checkpoint.step_index,
                tenant_id=record["tenant_id"],
                actor_id=actor.id,
            )
            self._emit(event)
            
            return OperationResult.success(JobResponse.from_dict(record))
            
        except Exception as e:
            logger.exception(f"Failed to save checkpoint: {e}")
            return OperationResult.failure(f"Failed to save checkpoint: {e}", "INTERNAL_ERROR")
    
    def assign_agent(
        self,
        actor: Actor,
        job_id: str,
        tenant_id: str,
        agent_name: str,
    ) -> OperationResult[JobResponse]:
        """Assign an agent (persona) to a job (SOTA).
        
        Args:
            actor: The actor performing the operation
            job_id: The job ID
            tenant_id: The tenant ID
            agent_name: Name of the agent/persona to assign
            
        Returns:
            OperationResult containing the updated job or error
        """
        try:
            # Get current job
            snapshot = self._repository.get_job(job_id, tenant_id=tenant_id)
            
            # Check permissions
            self._permissions.require_job_write(actor, snapshot)
            
            # Update metadata with agent assignment
            metadata = dict(snapshot.get("metadata", {}))
            previous_agent = metadata.get("assigned_agent")
            metadata["assigned_agent"] = agent_name
            metadata["agent_assigned_at"] = _now_utc().isoformat()
            metadata["agent_assigned_by"] = actor.id
            
            # Track delegation chain for handoffs
            delegation_chain = metadata.get("delegation_chain", [])
            if previous_agent:
                delegation_chain.append({
                    "agent": previous_agent,
                    "handed_off_at": _now_utc().isoformat(),
                })
            metadata["delegation_chain"] = delegation_chain
            
            # Update job
            record = self._repository.update_job(
                job_id,
                tenant_id=tenant_id,
                changes={"metadata": metadata},
            )
            
            # Emit event
            event = JobAgentAssigned(
                job_id=str(record["id"]),
                job_name=record["name"],
                assigned_agent=agent_name,
                previous_agent=previous_agent,
                tenant_id=record["tenant_id"],
                actor_id=actor.id,
                actor_type=actor.type,
            )
            self._emit(event)
            
            return OperationResult.success(JobResponse.from_dict(record))
            
        except Exception as e:
            logger.exception(f"Failed to assign agent: {e}")
            return OperationResult.failure(f"Failed to assign agent: {e}", "INTERNAL_ERROR")
    
    def update_execution_context(
        self,
        actor: Actor,
        job_id: str,
        tenant_id: str,
        context_updates: Dict[str, Any],
        *,
        merge: bool = True,
    ) -> OperationResult[JobResponse]:
        """Update the execution context (scratchpad) for a job (SOTA).
        
        Args:
            actor: The actor performing the operation
            job_id: The job ID
            tenant_id: The tenant ID
            context_updates: Context data to add/update
            merge: If True, merge with existing; if False, replace
            
        Returns:
            OperationResult containing the updated job or error
        """
        try:
            # Get current job
            snapshot = self._repository.get_job(job_id, tenant_id=tenant_id)
            
            # Check permissions
            self._permissions.require_job_execute(actor, snapshot)
            
            # Update metadata with execution context
            metadata = dict(snapshot.get("metadata", {}))
            if merge:
                existing_context = metadata.get("execution_context", {})
                existing_context.update(context_updates)
                metadata["execution_context"] = existing_context
            else:
                metadata["execution_context"] = context_updates
            
            # Update job
            record = self._repository.update_job(
                job_id,
                tenant_id=tenant_id,
                changes={"metadata": metadata},
            )
            
            return OperationResult.success(JobResponse.from_dict(record))
            
        except Exception as e:
            logger.exception(f"Failed to update execution context: {e}")
            return OperationResult.failure(f"Failed to update execution context: {e}", "INTERNAL_ERROR")
    
    # =========================================================================
    # Task Linking
    # =========================================================================
    
    def link_task(
        self,
        actor: Actor,
        job_id: str,
        tenant_id: str,
        link: TaskLinkCreate,
    ) -> OperationResult[Dict[str, Any]]:
        """Link a task to a job.
        
        Args:
            actor: The actor performing the operation
            job_id: The job ID
            tenant_id: The tenant ID
            link: Task link configuration
            
        Returns:
            OperationResult containing the link record or error
        """
        try:
            # Get job to check permissions
            snapshot = self._repository.get_job(job_id, tenant_id=tenant_id)
            self._permissions.require_job_write(actor, snapshot)
            
            # Create link
            metadata = dict(link.metadata or {})
            metadata["dependency_type"] = link.dependency_type.name
            
            record = self._repository.attach_task(
                job_id,
                link.task_id,
                tenant_id=tenant_id,
                relationship_type=link.relationship_type,
                metadata=metadata,
            )
            
            return OperationResult.success(record)
            
        except Exception as e:
            logger.exception(f"Failed to link task: {e}")
            return OperationResult.failure(f"Failed to link task: {e}", "INTERNAL_ERROR")
    
    def unlink_task(
        self,
        actor: Actor,
        job_id: str,
        tenant_id: str,
        task_id: str,
    ) -> OperationResult[None]:
        """Unlink a task from a job.
        
        Args:
            actor: The actor performing the operation
            job_id: The job ID
            tenant_id: The tenant ID
            task_id: The task ID to unlink
            
        Returns:
            OperationResult indicating success or error
        """
        try:
            # Get job to check permissions
            snapshot = self._repository.get_job(job_id, tenant_id=tenant_id)
            self._permissions.require_job_write(actor, snapshot)
            
            # Remove link
            self._repository.detach_task(
                job_id,
                tenant_id=tenant_id,
                task_id=task_id,
            )
            
            return OperationResult.success(None)
            
        except Exception as e:
            logger.exception(f"Failed to unlink task: {e}")
            return OperationResult.failure(f"Failed to unlink task: {e}", "INTERNAL_ERROR")
    
    def get_linked_tasks(
        self,
        actor: Actor,
        job_id: str,
        tenant_id: str,
    ) -> OperationResult[List[Dict[str, Any]]]:
        """Get all tasks linked to a job.
        
        Args:
            actor: The actor performing the operation
            job_id: The job ID
            tenant_id: The tenant ID
            
        Returns:
            OperationResult containing the list of linked tasks or error
        """
        try:
            # Get job to check permissions
            snapshot = self._repository.get_job(job_id, tenant_id=tenant_id)
            self._permissions.require_job_read(actor, snapshot)
            
            # Get linked tasks
            tasks = self._repository.list_linked_tasks(job_id, tenant_id=tenant_id)
            
            return OperationResult.success(tasks)
            
        except Exception as e:
            logger.exception(f"Failed to get linked tasks: {e}")
            return OperationResult.failure(f"Failed to get linked tasks: {e}", "INTERNAL_ERROR")
    
    # =========================================================================
    # Query Operations
    # =========================================================================
    
    def get_pending_jobs(
        self,
        actor: Actor,
        tenant_id: str,
    ) -> OperationResult[List[JobResponse]]:
        """Get all jobs pending execution.
        
        Args:
            actor: The actor performing the operation
            tenant_id: The tenant ID
            
        Returns:
            OperationResult containing pending jobs or error
        """
        result = self.list_jobs(
            actor,
            tenant_id,
            JobFilters(status=[JobStatus.DRAFT.value, JobStatus.SCHEDULED.value]),
        )
        if result.success and result.data:
            return OperationResult.success(result.data.jobs)
        return OperationResult.failure(result.error or "Unknown error", result.error_code or "UNKNOWN_ERROR")
    
    def get_running_jobs(
        self,
        actor: Actor,
        tenant_id: str,
    ) -> OperationResult[List[JobResponse]]:
        """Get all currently running jobs.
        
        Args:
            actor: The actor performing the operation
            tenant_id: The tenant ID
            
        Returns:
            OperationResult containing running jobs or error
        """
        result = self.list_jobs(actor, tenant_id, JobFilters(status=JobStatus.RUNNING.value))
        if result.success and result.data:
            return OperationResult.success(result.data.jobs)
        return OperationResult.failure(result.error or "Unknown error", result.error_code or "UNKNOWN_ERROR")
    
    def dependencies_complete(
        self,
        actor: Actor,
        job_id: str,
        tenant_id: str,
    ) -> OperationResult[bool]:
        """Check if all linked tasks are complete.
        
        Args:
            actor: The actor performing the operation
            job_id: The job ID
            tenant_id: The tenant ID
            
        Returns:
            OperationResult containing True if all complete, False otherwise
        """
        try:
            tasks_result = self.get_linked_tasks(actor, job_id, tenant_id)
            if not tasks_result.success or tasks_result.data is None:
                return OperationResult.failure(tasks_result.error or "Unknown error", tasks_result.error_code or "UNKNOWN_ERROR")
            
            from modules.task_store.models import TaskStatus
            complete_statuses = {TaskStatus.DONE, TaskStatus.CANCELLED}
            
            for link in tasks_result.data:
                task = link.get("task", {})
                status = task.get("status")
                if status and status not in [s.value for s in complete_statuses]:
                    return OperationResult.success(False)
            
            return OperationResult.success(True)
            
        except Exception as e:
            logger.exception(f"Failed to check dependencies: {e}")
            return OperationResult.failure(f"Failed to check dependencies: {e}", "INTERNAL_ERROR")


__all__ = [
    "JobService",
]
