"""
Task service for ATLAS.

Provides task lifecycle management with permission checks,
MessageBus events, and OperationResult returns.

Author: ATLAS Team
Date: Jan 10, 2026
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Mapping, Optional, Sequence
from uuid import UUID

from core.services.common import Actor, OperationResult
from core.services.common.exceptions import PermissionDeniedError

from .exceptions import (
    TaskError,
    TaskNotFoundError,
    TaskTransitionError,
    TaskDependencyError,
    TaskConcurrencyError,
    TaskValidationError,
    TaskCircularDependencyError,
)
from .permissions import TaskPermissionChecker
from .types import (
    TaskStatus,
    TaskPriority,
    DependencyType,
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
)

if TYPE_CHECKING:
    from modules.task_store.repository import TaskStoreRepository
    from core.messaging import MessageBus


logger = logging.getLogger(__name__)


# Allowed status transitions
_ALLOWED_TRANSITIONS: Dict[TaskStatus, set[TaskStatus]] = {
    TaskStatus.DRAFT: {TaskStatus.READY, TaskStatus.CANCELLED},
    TaskStatus.READY: {TaskStatus.IN_PROGRESS, TaskStatus.CANCELLED},
    TaskStatus.IN_PROGRESS: {TaskStatus.REVIEW, TaskStatus.CANCELLED},
    TaskStatus.REVIEW: {TaskStatus.DONE, TaskStatus.CANCELLED},
    TaskStatus.DONE: set(),
    TaskStatus.CANCELLED: set(),
}

# Complete statuses
_COMPLETE_STATUSES = {TaskStatus.DONE, TaskStatus.CANCELLED}


def _coerce_status(value: Any) -> TaskStatus:
    """Coerce a value to TaskStatus enum."""
    if isinstance(value, TaskStatus):
        return value
    if isinstance(value, str):
        return TaskStatus(value.lower())
    raise ValueError(f"Cannot coerce {value} to TaskStatus")


def _now_utc() -> datetime:
    """Get current UTC timestamp."""
    return datetime.now(timezone.utc)


class TaskService:
    """
    Application service that coordinates task lifecycle operations.
    
    Provides:
    - CRUD operations with permission checks
    - Status transitions with dependency validation
    - Subtask hierarchy
    - MessageBus event emission
    - SOTA features: execution context, agent assignment, priority levels
    
    All operations return OperationResult for consistent error handling.
    
    Example:
        service = TaskService(repository, message_bus, permission_checker)
        
        result = await service.create_task(actor, TaskCreate(
            title="Analyze Data",
            tenant_id="org_123",
            conversation_id="conv_456",
            priority=TaskPriority.HIGH,
            assigned_agent="analyst-persona",
        ))
        
        if result.success:
            task = result.data
            print(f"Created task: {task.id}")
    """
    
    def __init__(
        self,
        repository: "TaskStoreRepository",
        message_bus: Optional["MessageBus"] = None,
        permission_checker: Optional[TaskPermissionChecker] = None,
    ) -> None:
        """Initialize the task service.
        
        Args:
            repository: Task store repository for persistence
            message_bus: Optional message bus for event emission
            permission_checker: Optional permission checker (creates default if None)
        """
        self._repository = repository
        self._message_bus = message_bus
        self._permissions = permission_checker or TaskPermissionChecker()
    
    def _emit(self, event: Any) -> None:
        """Emit an event to the message bus."""
        if self._message_bus is not None:
            try:
                self._message_bus.publish(event.event_type, event.to_dict())
            except Exception as e:
                logger.warning(f"Failed to emit event {event.event_type}: {e}")
    
    def _extract_sota_metadata(self, task_data: TaskCreate | TaskUpdate) -> Dict[str, Any]:
        """Extract SOTA fields into metadata dict."""
        sota_fields = {}
        
        if hasattr(task_data, 'assigned_agent') and task_data.assigned_agent is not None:
            sota_fields['assigned_agent'] = task_data.assigned_agent
        if hasattr(task_data, 'estimated_cost') and task_data.estimated_cost is not None:
            sota_fields['estimated_cost'] = str(task_data.estimated_cost)
        if hasattr(task_data, 'actual_cost') and task_data.actual_cost is not None:
            sota_fields['actual_cost'] = str(task_data.actual_cost)
        if hasattr(task_data, 'timeout_seconds') and task_data.timeout_seconds is not None:
            sota_fields['timeout_seconds'] = task_data.timeout_seconds
        if hasattr(task_data, 'execution_context') and task_data.execution_context is not None:
            sota_fields['execution_context'] = task_data.execution_context
        if hasattr(task_data, 'plan_id') and task_data.plan_id is not None:
            sota_fields['plan_id'] = task_data.plan_id
        if hasattr(task_data, 'plan_step_index') and task_data.plan_step_index is not None:
            sota_fields['plan_step_index'] = task_data.plan_step_index
        if hasattr(task_data, 'parent_task_id') and task_data.parent_task_id is not None:
            sota_fields['parent_task_id'] = task_data.parent_task_id
            
        return sota_fields
    
    # =========================================================================
    # CRUD Operations
    # =========================================================================
    
    def create_task(
        self,
        actor: Actor,
        task_data: TaskCreate,
    ) -> OperationResult[TaskResponse]:
        """Create a new task.
        
        Args:
            actor: The actor performing the operation
            task_data: Task creation data
            
        Returns:
            OperationResult containing the created task or error
        """
        try:
            # Check permissions
            self._permissions.require_write(actor, task_data.tenant_id)
            
            # Build metadata with SOTA fields
            metadata = dict(task_data.metadata or {})
            metadata.update(self._extract_sota_metadata(task_data))
            
            # Create via repository
            record = self._repository.create_task(
                task_data.title,
                tenant_id=task_data.tenant_id,
                description=task_data.description,
                status=None,  # Default to DRAFT
                priority=task_data.priority,
                owner_id=task_data.owner_id or actor.id,
                session_id=task_data.session_id,
                conversation_id=task_data.conversation_id,
                due_at=task_data.due_at,
                metadata=metadata,
            )
            
            # Emit event
            event = TaskCreated(
                task_id=str(record["id"]),
                title=record["title"],
                tenant_id=record.get("tenant_id", task_data.tenant_id),
                actor_id=actor.id,
                actor_type=actor.type,
                status=record.get("status", "draft"),
                priority=task_data.priority,
                owner_id=str(record["owner_id"]) if record.get("owner_id") else None,
                assigned_agent=task_data.assigned_agent,
                conversation_id=task_data.conversation_id,
                parent_task_id=task_data.parent_task_id,
            )
            self._emit(event)
            
            # Handle subtask creation
            if task_data.parent_task_id:
                subtask_event = SubtaskCreated(
                    task_id=str(record["id"]),
                    parent_task_id=task_data.parent_task_id,
                    title=record["title"],
                    tenant_id=record.get("tenant_id", task_data.tenant_id),
                    actor_id=actor.id,
                    actor_type=actor.type,
                )
                self._emit(subtask_event)
            
            return OperationResult.success(TaskResponse.from_dict(record))
            
        except PermissionDeniedError as e:
            return OperationResult.failure(str(e), "PERMISSION_DENIED")
        except TaskValidationError as e:
            return OperationResult.failure(e.message, e.error_code)
        except Exception as e:
            logger.exception(f"Failed to create task: {e}")
            return OperationResult.failure(f"Failed to create task: {e}", "INTERNAL_ERROR")
    
    def get_task(
        self,
        actor: Actor,
        task_id: str,
        tenant_id: str,
        *,
        with_dependencies: bool = False,
        with_subtasks: bool = False,
    ) -> OperationResult[TaskResponse]:
        """Get a task by ID.
        
        Args:
            actor: The actor performing the operation
            task_id: The task ID
            tenant_id: The tenant ID
            with_dependencies: Include dependencies
            with_subtasks: Include subtasks
            
        Returns:
            OperationResult containing the task or error
        """
        try:
            # Get task first to check permissions
            record = self._repository.get_task(
                task_id,
                tenant_id=tenant_id,
                with_events=False,
            )
            
            # Check permissions against the specific task
            self._permissions.require_task_read(actor, record)
            
            response = TaskResponse.from_dict(record)
            
            # Optionally load subtasks
            if with_subtasks:
                subtasks = self._get_subtasks_internal(task_id, tenant_id)
                response.subtasks = [TaskResponse.from_dict(s) for s in subtasks]
            
            return OperationResult.success(response)
            
        except Exception as e:
            if "not found" in str(e).lower():
                return OperationResult.failure(f"Task not found: {task_id}", "TASK_NOT_FOUND")
            logger.exception(f"Failed to get task: {e}")
            return OperationResult.failure(f"Failed to get task: {e}", "INTERNAL_ERROR")
    
    def _get_subtasks_internal(self, parent_task_id: str, tenant_id: str) -> List[Dict[str, Any]]:
        """Get subtasks for a parent task (internal helper)."""
        # Query tasks with parent_task_id in metadata
        result = self._repository.list_tasks(tenant_id=tenant_id, limit=1000)
        all_tasks = result.get("tasks", []) if isinstance(result, dict) else result
        subtasks = []
        for task in all_tasks:
            metadata = task.get("metadata", {})
            if metadata.get("parent_task_id") == parent_task_id:
                subtasks.append(task)
        return subtasks
    
    def update_task(
        self,
        actor: Actor,
        task_id: str,
        tenant_id: str,
        updates: TaskUpdate,
        *,
        expected_updated_at: datetime | None = None,
    ) -> OperationResult[TaskResponse]:
        """Update an existing task.
        
        Args:
            actor: The actor performing the operation
            task_id: The task ID
            tenant_id: The tenant ID
            updates: Task update data
            expected_updated_at: Optimistic concurrency check
            
        Returns:
            OperationResult containing the updated task or error
        """
        try:
            # Get current task
            snapshot = self._repository.get_task(task_id, tenant_id=tenant_id)
            
            # Check permissions
            self._permissions.require_task_write(actor, snapshot)
            
            # Build changes dict
            changes: Dict[str, Any] = {}
            if updates.title is not None:
                changes["title"] = updates.title
            if updates.description is not None:
                changes["description"] = updates.description
            if updates.priority is not None:
                changes["priority"] = updates.priority
            if updates.owner_id is not None:
                changes["owner_id"] = updates.owner_id
            if updates.due_at is not None:
                changes["due_at"] = updates.due_at
            
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
                return OperationResult.success(TaskResponse.from_dict(snapshot))
            
            # Update via repository
            record = self._repository.update_task(
                task_id,
                tenant_id=tenant_id,
                changes=changes,
                expected_updated_at=expected_updated_at,
            )
            
            # Emit event
            event = TaskUpdated(
                task_id=str(record["id"]),
                title=record["title"],
                changed_fields=updates.changed_fields(),
                tenant_id=record.get("tenant_id", tenant_id),
                actor_id=actor.id,
                actor_type=actor.type,
            )
            self._emit(event)
            
            return OperationResult.success(TaskResponse.from_dict(record))
            
        except Exception as e:
            if "concurrency" in str(e).lower():
                return OperationResult.failure("Task was modified by another transaction", "CONCURRENCY_ERROR")
            logger.exception(f"Failed to update task: {e}")
            return OperationResult.failure(f"Failed to update task: {e}", "INTERNAL_ERROR")
    
    def delete_task(
        self,
        actor: Actor,
        task_id: str,
        tenant_id: str,
    ) -> OperationResult[None]:
        """Delete a task.
        
        Args:
            actor: The actor performing the operation
            task_id: The task ID
            tenant_id: The tenant ID
            
        Returns:
            OperationResult indicating success or error
        """
        try:
            # Get task to check permissions
            snapshot = self._repository.get_task(task_id, tenant_id=tenant_id)
            
            # Check permissions (delete requires owner or admin)
            self._permissions.require_task_delete(actor, snapshot)
            
            # Delete via repository (TODO: add delete_task to repository)
            self._repository.delete_task(task_id, tenant_id=tenant_id)  # type: ignore[attr-defined]
            
            # Emit event
            event = TaskDeleted(
                task_id=str(snapshot["id"]),
                title=snapshot["title"],
                tenant_id=snapshot.get("tenant_id", tenant_id),
                actor_id=actor.id,
                actor_type=actor.type,
            )
            self._emit(event)
            
            return OperationResult.success(None)
            
        except Exception as e:
            if "not found" in str(e).lower():
                return OperationResult.failure(f"Task not found: {task_id}", "TASK_NOT_FOUND")
            logger.exception(f"Failed to delete task: {e}")
            return OperationResult.failure(f"Failed to delete task: {e}", "INTERNAL_ERROR")
    
    def list_tasks(
        self,
        actor: Actor,
        tenant_id: str,
        filters: TaskFilters | None = None,
    ) -> OperationResult[TaskListResponse]:
        """List tasks with optional filtering.
        
        Args:
            actor: The actor performing the operation
            tenant_id: The tenant ID
            filters: Optional filters
            
        Returns:
            OperationResult containing the task list or error
        """
        try:
            # Check permissions
            self._permissions.require_read(actor, tenant_id)
            
            filters = filters or TaskFilters()
            
            # Query via repository
            result = self._repository.list_tasks(
                tenant_id=tenant_id,
                status=filters.status,
                owner_id=filters.owner_id,
                conversation_id=filters.conversation_id,
                job_id=filters.job_id,
                limit=filters.limit,
                cursor=filters.cursor,
            )
            
            # Handle both dict with "tasks" key and raw list
            task_list = result.get("tasks", []) if isinstance(result, dict) else result
            next_cursor = result.get("next_cursor") if isinstance(result, dict) else None
            
            tasks = [TaskResponse.from_dict(t) for t in task_list]
            
            # Filter by priority if specified
            if filters.priority_min is not None:
                tasks = [t for t in tasks if t.priority >= filters.priority_min]
            if filters.priority_max is not None:
                tasks = [t for t in tasks if t.priority <= filters.priority_max]
            
            # Filter by due date if specified
            if filters.due_before is not None:
                tasks = [t for t in tasks if t.due_at and t.due_at <= filters.due_before]
            if filters.due_after is not None:
                tasks = [t for t in tasks if t.due_at and t.due_at >= filters.due_after]
            
            return OperationResult.success(TaskListResponse(
                tasks=tasks,
                next_cursor=next_cursor,
            ))
            
        except Exception as e:
            logger.exception(f"Failed to list tasks: {e}")
            return OperationResult.failure(f"Failed to list tasks: {e}", "INTERNAL_ERROR")
    
    # =========================================================================
    # Lifecycle Operations
    # =========================================================================
    
    def transition_task(
        self,
        actor: Actor,
        task_id: str,
        tenant_id: str,
        target_status: TaskStatus | str,
        *,
        reason: str | None = None,
        expected_updated_at: datetime | None = None,
    ) -> OperationResult[TaskResponse]:
        """Transition a task to a new status.
        
        Args:
            actor: The actor performing the operation
            task_id: The task ID
            tenant_id: The tenant ID
            target_status: The target status
            reason: Optional reason for the transition
            expected_updated_at: Optimistic concurrency check
            
        Returns:
            OperationResult containing the updated task or error
        """
        try:
            # Get current task
            snapshot = self._repository.get_task(task_id, tenant_id=tenant_id)
            
            # Check permissions
            self._permissions.require_task_write(actor, snapshot)
            
            current_status = _coerce_status(snapshot["status"])
            desired_status = _coerce_status(target_status) if isinstance(target_status, str) else target_status
            
            # Same status is a no-op
            if desired_status == current_status:
                return OperationResult.success(TaskResponse.from_dict(snapshot))
            
            # Validate transition
            allowed = _ALLOWED_TRANSITIONS.get(current_status, set())
            if desired_status not in allowed:
                return OperationResult.failure(
                    f"Cannot transition task from {current_status.value} to {desired_status.value}",
                    "INVALID_TRANSITION"
                )
            
            # Check dependencies for non-cancel transitions
            if desired_status not in {TaskStatus.CANCELLED, TaskStatus.DRAFT}:
                deps_complete = self._check_dependencies_complete(task_id, tenant_id)
                if not deps_complete:
                    return OperationResult.failure(
                        "Cannot advance task because dependencies are incomplete",
                        "DEPENDENCY_ERROR"
                    )
            
            # Update via repository
            reference_timestamp = expected_updated_at or snapshot.get("updated_at")
            record = self._repository.update_task(
                task_id,
                tenant_id=tenant_id,
                changes={"status": desired_status.value},
                expected_updated_at=reference_timestamp,
            )
            
            # Emit event
            event = TaskStatusChanged(
                task_id=str(record["id"]),
                title=record["title"],
                from_status=current_status.value,
                to_status=desired_status.value,
                reason=reason,
                tenant_id=record.get("tenant_id", tenant_id),
                actor_id=actor.id,
                actor_type=actor.type,
            )
            self._emit(event)
            
            # Emit completion event if applicable
            if desired_status == TaskStatus.DONE:
                complete_event = TaskCompleted(
                    task_id=str(record["id"]),
                    title=record["title"],
                    tenant_id=record.get("tenant_id", tenant_id),
                    actor_id=actor.id,
                    actor_type=actor.type,
                )
                self._emit(complete_event)
            
            return OperationResult.success(TaskResponse.from_dict(record))
            
        except Exception as e:
            if "concurrency" in str(e).lower():
                return OperationResult.failure("Task was modified by another transaction", "CONCURRENCY_ERROR")
            logger.exception(f"Failed to transition task: {e}")
            return OperationResult.failure(f"Failed to transition task: {e}", "INTERNAL_ERROR")

    def start_task(
        self,
        actor: Actor,
        task_id: str,
        tenant_id: str,
    ) -> OperationResult[TaskResponse]:
        """Start a task (transition from READY to IN_PROGRESS).
        
        Args:
            actor: The actor performing the operation
            task_id: The task ID
            tenant_id: The tenant ID
            
        Returns:
            OperationResult containing the updated task or error
        """
        return self.transition_task(
            actor, task_id, tenant_id,
            target_status=TaskStatus.IN_PROGRESS,
            reason="Task started",
        )
    
    def _check_dependencies_complete(self, task_id: str, tenant_id: str) -> bool:
        """Check if all dependencies are complete."""
        try:
            statuses = self._repository.dependency_statuses(task_id, tenant_id=tenant_id)
            return all(_coerce_status(s) in _COMPLETE_STATUSES for s in statuses)
        except Exception:
            return True  # If we can't check, assume OK
    
    def complete_task(
        self,
        actor: Actor,
        task_id: str,
        tenant_id: str,
        result: TaskResult | None = None,
        *,
        completion_notes: str | None = None,
    ) -> OperationResult[TaskResponse]:
        """Mark a task as complete.
        
        Args:
            actor: The actor performing the operation
            task_id: The task ID
            tenant_id: The tenant ID
            result: Optional completion result
            completion_notes: Optional completion notes
            
        Returns:
            OperationResult containing the updated task or error
        """
        try:
            # Get current task
            snapshot = self._repository.get_task(task_id, tenant_id=tenant_id)
            
            # Check permissions
            self._permissions.require_task_write(actor, snapshot)
            
            current_status = _coerce_status(snapshot["status"])
            
            # Must be in REVIEW to complete
            if current_status != TaskStatus.REVIEW:
                return OperationResult.failure(
                    f"Cannot complete task in {current_status.value} status (must be in review)",
                    "INVALID_TRANSITION"
                )
            
            # Update metadata with result
            metadata = dict(snapshot.get("metadata", {}))
            if completion_notes:
                metadata["completion_notes"] = completion_notes
            if result:
                if result.actual_cost is not None:
                    metadata["actual_cost"] = str(result.actual_cost)
                if result.result_data is not None:
                    metadata["result_data"] = result.result_data
                if result.result_summary is not None:
                    metadata["result_summary"] = result.result_summary
                if result.execution_time_ms is not None:
                    metadata["execution_time_ms"] = result.execution_time_ms
                if result.tokens_used is not None:
                    metadata["tokens_used"] = result.tokens_used
                if result.tool_calls is not None:
                    metadata["tool_calls"] = result.tool_calls
            
            # Update task
            record = self._repository.update_task(
                task_id,
                tenant_id=tenant_id,
                changes={"status": TaskStatus.DONE.value, "metadata": metadata},
            )
            
            # Emit events
            status_event = TaskStatusChanged(
                task_id=str(record["id"]),
                title=record["title"],
                from_status=current_status.value,
                to_status=TaskStatus.DONE.value,
                tenant_id=record.get("tenant_id", tenant_id),
                actor_id=actor.id,
                actor_type=actor.type,
            )
            self._emit(status_event)
            
            complete_event = TaskCompleted(
                task_id=str(record["id"]),
                title=record["title"],
                actual_cost=result.actual_cost if result else None,
                result_summary=result.result_summary if result else None,
                tenant_id=record.get("tenant_id", tenant_id),
                actor_id=actor.id,
                actor_type=actor.type,
            )
            self._emit(complete_event)
            
            return OperationResult.success(TaskResponse.from_dict(record))
            
        except Exception as e:
            logger.exception(f"Failed to complete task: {e}")
            return OperationResult.failure(f"Failed to complete task: {e}", "INTERNAL_ERROR")
    
    def cancel_task(
        self,
        actor: Actor,
        task_id: str,
        tenant_id: str,
        reason: str,
    ) -> OperationResult[TaskResponse]:
        """Cancel a task.
        
        Args:
            actor: The actor performing the operation
            task_id: The task ID
            tenant_id: The tenant ID
            reason: Cancellation reason
            
        Returns:
            OperationResult containing the updated task or error
        """
        try:
            # Get current task
            snapshot = self._repository.get_task(task_id, tenant_id=tenant_id)
            
            # Check permissions
            self._permissions.require_task_write(actor, snapshot)
            
            current_status = _coerce_status(snapshot["status"])
            
            # Cannot cancel already terminal states
            if current_status in _COMPLETE_STATUSES:
                return OperationResult.failure(
                    f"Cannot cancel task in {current_status.value} status",
                    "INVALID_TRANSITION"
                )
            
            # Update metadata with cancellation info
            metadata = dict(snapshot.get("metadata", {}))
            metadata["cancellation_reason"] = reason
            metadata["cancelled_by"] = actor.id
            metadata["cancelled_at"] = _now_utc().isoformat()
            
            # Update task
            record = self._repository.update_task(
                task_id,
                tenant_id=tenant_id,
                changes={"status": TaskStatus.CANCELLED.value, "metadata": metadata},
            )
            
            # Emit events
            status_event = TaskStatusChanged(
                task_id=str(record["id"]),
                title=record["title"],
                from_status=current_status.value,
                to_status=TaskStatus.CANCELLED.value,
                reason=reason,
                tenant_id=record.get("tenant_id", tenant_id),
                actor_id=actor.id,
                actor_type=actor.type,
            )
            self._emit(status_event)
            
            cancel_event = TaskCancelled(
                task_id=str(record["id"]),
                title=record["title"],
                reason=reason,
                tenant_id=record.get("tenant_id", tenant_id),
                actor_id=actor.id,
                actor_type=actor.type,
            )
            self._emit(cancel_event)
            
            return OperationResult.success(TaskResponse.from_dict(record))
            
        except Exception as e:
            logger.exception(f"Failed to cancel task: {e}")
            return OperationResult.failure(f"Failed to cancel task: {e}", "INTERNAL_ERROR")
    
    def reopen_task(
        self,
        actor: Actor,
        task_id: str,
        tenant_id: str,
    ) -> OperationResult[TaskResponse]:
        """Reopen a completed or cancelled task.
        
        Args:
            actor: The actor performing the operation
            task_id: The task ID
            tenant_id: The tenant ID
            
        Returns:
            OperationResult containing the updated task or error
        """
        try:
            # Get current task
            snapshot = self._repository.get_task(task_id, tenant_id=tenant_id)
            
            # Check permissions
            self._permissions.require_task_write(actor, snapshot)
            
            current_status = _coerce_status(snapshot["status"])
            
            # Can only reopen from terminal states
            if current_status not in _COMPLETE_STATUSES:
                return OperationResult.failure(
                    f"Cannot reopen task in {current_status.value} status",
                    "INVALID_TRANSITION"
                )
            
            # Update metadata
            metadata = dict(snapshot.get("metadata", {}))
            metadata["reopened_at"] = _now_utc().isoformat()
            metadata["reopened_by"] = actor.id
            
            # Update task to DRAFT
            record = self._repository.update_task(
                task_id,
                tenant_id=tenant_id,
                changes={"status": TaskStatus.DRAFT.value, "metadata": metadata},
            )
            
            # Emit event
            event = TaskStatusChanged(
                task_id=str(record["id"]),
                title=record["title"],
                from_status=current_status.value,
                to_status=TaskStatus.DRAFT.value,
                reason="Task reopened",
                tenant_id=record.get("tenant_id", tenant_id),
                actor_id=actor.id,
                actor_type=actor.type,
            )
            self._emit(event)
            
            return OperationResult.success(TaskResponse.from_dict(record))
            
        except Exception as e:
            logger.exception(f"Failed to reopen task: {e}")
            return OperationResult.failure(f"Failed to reopen task: {e}", "INTERNAL_ERROR")
    
    # =========================================================================
    # Assignment Operations
    # =========================================================================
    
    def assign_task(
        self,
        actor: Actor,
        task_id: str,
        tenant_id: str,
        assignee_id: str,
        *,
        assignee_type: Literal["user", "agent"] = "user",
        role: str = "owner",
    ) -> OperationResult[TaskResponse]:
        """Assign a task to a user or agent.
        
        Args:
            actor: The actor performing the operation
            task_id: The task ID
            tenant_id: The tenant ID
            assignee_id: ID of the assignee
            assignee_type: Type of assignee ("user" or "agent")
            role: Assignment role
            
        Returns:
            OperationResult containing the updated task or error
        """
        try:
            # Get current task
            snapshot = self._repository.get_task(task_id, tenant_id=tenant_id)
            
            # Check permissions
            self._permissions.require_task_write(actor, snapshot)
            
            previous_owner = snapshot.get("owner_id")
            
            # Update owner
            changes: Dict[str, Any] = {"owner_id": assignee_id}
            
            # Track agent in metadata if assignee is an agent
            if assignee_type == "agent":
                metadata = dict(snapshot.get("metadata", {}))
                previous_agent = metadata.get("assigned_agent")
                metadata["assigned_agent"] = assignee_id
                changes["metadata"] = metadata
            
            record = self._repository.update_task(
                task_id,
                tenant_id=tenant_id,
                changes=changes,
            )
            
            # Emit event
            event = TaskAssigned(
                task_id=str(record["id"]),
                title=record["title"],
                assignee_id=assignee_id,
                assignee_type=assignee_type,
                previous_assignee_id=str(previous_owner) if previous_owner else None,
                role=role,
                tenant_id=record.get("tenant_id", tenant_id),
                actor_id=actor.id,
                actor_type=actor.type,
            )
            self._emit(event)
            
            # Emit agent assignment event if applicable
            if assignee_type == "agent":
                agent_event = TaskAgentAssigned(
                    task_id=str(record["id"]),
                    title=record["title"],
                    assigned_agent=assignee_id,
                    previous_agent=snapshot.get("metadata", {}).get("assigned_agent"),
                    tenant_id=record.get("tenant_id", tenant_id),
                    actor_id=actor.id,
                    actor_type=actor.type,
                )
                self._emit(agent_event)
            
            return OperationResult.success(TaskResponse.from_dict(record))
            
        except Exception as e:
            logger.exception(f"Failed to assign task: {e}")
            return OperationResult.failure(f"Failed to assign task: {e}", "INTERNAL_ERROR")
    
    # =========================================================================
    # Subtask Operations
    # =========================================================================
    
    def create_subtask(
        self,
        actor: Actor,
        parent_task_id: str,
        tenant_id: str,
        task_data: TaskCreate,
    ) -> OperationResult[TaskResponse]:
        """Create a subtask under a parent task.
        
        Args:
            actor: The actor performing the operation
            parent_task_id: The parent task ID
            tenant_id: The tenant ID
            task_data: Subtask creation data
            
        Returns:
            OperationResult containing the created subtask or error
        """
        # Set parent reference
        task_data.parent_task_id = parent_task_id
        
        # Inherit conversation_id from parent if not set
        if not task_data.conversation_id:
            parent_result = self.get_task(actor, parent_task_id, tenant_id)
            if parent_result.success and parent_result.data and parent_result.data.conversation_id:
                task_data.conversation_id = parent_result.data.conversation_id
        
        return self.create_task(actor, task_data)
    
    def get_subtasks(
        self,
        actor: Actor,
        parent_task_id: str,
        tenant_id: str,
    ) -> OperationResult[List[TaskResponse]]:
        """Get all subtasks of a parent task.
        
        Args:
            actor: The actor performing the operation
            parent_task_id: The parent task ID
            tenant_id: The tenant ID
            
        Returns:
            OperationResult containing the list of subtasks or error
        """
        try:
            # Check permissions
            self._permissions.require_read(actor, tenant_id)
            
            # Use repository method if available, otherwise fall back to internal
            if hasattr(self._repository, 'get_subtasks'):
                subtasks = self._repository.get_subtasks(parent_task_id)
            else:
                subtasks = self._get_subtasks_internal(parent_task_id, tenant_id)
            return OperationResult.success([TaskResponse.from_dict(s) for s in subtasks])
            
        except Exception as e:
            logger.exception(f"Failed to get subtasks: {e}")
            return OperationResult.failure(f"Failed to get subtasks: {e}", "INTERNAL_ERROR")
    
    # =========================================================================
    # Dependency Operations
    # =========================================================================
    
    def add_dependency(
        self,
        actor: Actor,
        task_id: str,
        tenant_id: str,
        dependency: DependencyCreate,
    ) -> OperationResult[Dict[str, Any]]:
        """Add a dependency to a task.
        
        Args:
            actor: The actor performing the operation
            task_id: The task ID
            tenant_id: The tenant ID
            dependency: Dependency configuration
            
        Returns:
            OperationResult containing the dependency record or error
        """
        try:
            # Get task to check permissions
            snapshot = self._repository.get_task(task_id, tenant_id=tenant_id)
            self._permissions.require_task_write(actor, snapshot)
            
            # Check for circular dependency
            if self._would_create_cycle(task_id, dependency.depends_on_id, tenant_id):
                return OperationResult.failure(
                    f"Adding dependency would create a circular dependency",
                    "CIRCULAR_DEPENDENCY"
                )
            
            # Create dependency (TODO: add add_dependency to repository)
            record = self._repository.add_dependency(  # type: ignore[attr-defined]
                task_id,
                depends_on_id=dependency.depends_on_id,
                tenant_id=tenant_id,
                relationship_type=dependency.relationship_type,
            )
            
            return OperationResult.success(record)
            
        except Exception as e:
            logger.exception(f"Failed to add dependency: {e}")
            return OperationResult.failure(f"Failed to add dependency: {e}", "INTERNAL_ERROR")
    
    def _would_create_cycle(self, task_id: str, depends_on_id: str, tenant_id: str) -> bool:
        """Check if adding a dependency would create a cycle."""
        # Simple cycle detection: check if depends_on_id has task_id in its dependency chain
        visited = set()
        queue = [depends_on_id]
        
        while queue:
            current = queue.pop(0)
            if current == task_id:
                return True
            if current in visited:
                continue
            visited.add(current)
            
            try:
                deps = self._repository.get_dependencies(current, tenant_id=tenant_id)  # type: ignore[attr-defined]
                for dep in deps:
                    dep_id = dep.get("depends_on_id")
                    if dep_id and dep_id not in visited:
                        queue.append(str(dep_id))
            except Exception:
                pass
        
        return False
    
    def remove_dependency(
        self,
        actor: Actor,
        task_id: str,
        tenant_id: str,
        depends_on_id: str,
    ) -> OperationResult[None]:
        """Remove a dependency from a task.
        
        Args:
            actor: The actor performing the operation
            task_id: The task ID
            tenant_id: The tenant ID
            depends_on_id: ID of the task to remove dependency on
            
        Returns:
            OperationResult indicating success or error
        """
        try:
            # Get task to check permissions
            snapshot = self._repository.get_task(task_id, tenant_id=tenant_id)
            self._permissions.require_task_write(actor, snapshot)
            
            # Remove dependency (TODO: add remove_dependency to repository)
            self._repository.remove_dependency(  # type: ignore[attr-defined]
                task_id,
                depends_on_id=depends_on_id,
                tenant_id=tenant_id,
            )
            
            return OperationResult.success(None)
            
        except Exception as e:
            logger.exception(f"Failed to remove dependency: {e}")
            return OperationResult.failure(f"Failed to remove dependency: {e}", "INTERNAL_ERROR")
    
    def dependencies_complete(
        self,
        actor: Actor,
        task_id: str,
        tenant_id: str,
    ) -> OperationResult[bool]:
        """Check if all dependencies are complete.
        
        Args:
            actor: The actor performing the operation
            task_id: The task ID
            tenant_id: The tenant ID
            
        Returns:
            OperationResult containing True if all complete, False otherwise
        """
        try:
            self._permissions.require_read(actor, tenant_id)
            result = self._check_dependencies_complete(task_id, tenant_id)
            return OperationResult.success(result)
        except Exception as e:
            logger.exception(f"Failed to check dependencies: {e}")
            return OperationResult.failure(f"Failed to check dependencies: {e}", "INTERNAL_ERROR")
    
    # =========================================================================
    # SOTA Features
    # =========================================================================
    
    def assign_agent(
        self,
        actor: Actor,
        task_id: str,
        tenant_id: str,
        agent_name: str,
    ) -> OperationResult[TaskResponse]:
        """Assign an agent (persona) to a task (SOTA).
        
        Args:
            actor: The actor performing the operation
            task_id: The task ID
            tenant_id: The tenant ID
            agent_name: Name of the agent/persona to assign
            
        Returns:
            OperationResult containing the updated task or error
        """
        return self.assign_task(actor, task_id, tenant_id, agent_name, assignee_type="agent")
    
    def update_execution_context(
        self,
        actor: Actor,
        task_id: str,
        tenant_id: str,
        context_updates: Dict[str, Any],
        *,
        merge: bool = True,
    ) -> OperationResult[TaskResponse]:
        """Update the execution context (scratchpad) for a task (SOTA).
        
        Args:
            actor: The actor performing the operation
            task_id: The task ID
            tenant_id: The tenant ID
            context_updates: Context data to add/update
            merge: If True, merge with existing; if False, replace
            
        Returns:
            OperationResult containing the updated task or error
        """
        try:
            # Get current task
            snapshot = self._repository.get_task(task_id, tenant_id=tenant_id)
            
            # Check permissions
            self._permissions.require_task_write(actor, snapshot)
            
            # Update metadata with execution context
            metadata = dict(snapshot.get("metadata", {}))
            if merge:
                existing_context = metadata.get("execution_context", {})
                existing_context.update(context_updates)
                metadata["execution_context"] = existing_context
            else:
                metadata["execution_context"] = context_updates
            
            # Update task
            record = self._repository.update_task(
                task_id,
                tenant_id=tenant_id,
                changes={"metadata": metadata},
            )
            
            return OperationResult.success(TaskResponse.from_dict(record))
            
        except Exception as e:
            logger.exception(f"Failed to update execution context: {e}")
            return OperationResult.failure(f"Failed to update execution context: {e}", "INTERNAL_ERROR")

    def get_dependencies(
        self,
        actor: Actor,
        task_id: str,
        tenant_id: str,
    ) -> OperationResult[List[Dict[str, Any]]]:
        """Get all dependencies for a task.
        
        Args:
            actor: The actor performing the operation
            task_id: The task ID
            tenant_id: The tenant ID
            
        Returns:
            OperationResult containing list of dependencies or error
        """
        try:
            # Get task to verify it exists and check permissions
            snapshot = self._repository.get_task(task_id, tenant_id=tenant_id)
            if not snapshot:
                return OperationResult.failure(f"Task {task_id} not found", "NOT_FOUND")
            
            # Check permissions
            self._permissions.require_task_read(actor, snapshot)
            
            # Get dependencies from repository
            dependencies = self._repository.get_dependencies(task_id)
            
            return OperationResult.success(dependencies)
            
        except PermissionDeniedError as e:
            return OperationResult.failure(str(e), "PERMISSION_DENIED")
        except Exception as e:
            logger.exception(f"Failed to get dependencies: {e}")
            return OperationResult.failure(f"Failed to get dependencies: {e}", "INTERNAL_ERROR")

    def set_priority(
        self,
        actor: Actor,
        task_id: str,
        tenant_id: str,
        priority: int,
    ) -> OperationResult[TaskResponse]:
        """Set the priority of a task.
        
        Args:
            actor: The actor performing the operation
            task_id: The task ID
            tenant_id: The tenant ID
            priority: New priority value (1-100)
            
        Returns:
            OperationResult containing the updated task or error
        """
        try:
            # Validate priority range
            if priority < 1 or priority > 100:
                return OperationResult.failure(
                    f"Priority must be between 1 and 100, got {priority}",
                    "VALIDATION_ERROR"
                )
            
            # Get task to verify it exists
            snapshot = self._repository.get_task(task_id, tenant_id=tenant_id)
            if not snapshot:
                return OperationResult.failure(f"Task {task_id} not found", "NOT_FOUND")
            
            # Check permissions
            self._permissions.require_task_write(actor, snapshot)
            
            # Update task
            record = self._repository.update_task(
                task_id,
                tenant_id=tenant_id,
                changes={"priority": priority},
            )
            
            return OperationResult.success(TaskResponse.from_dict(record))
            
        except PermissionDeniedError as e:
            return OperationResult.failure(str(e), "PERMISSION_DENIED")
        except Exception as e:
            logger.exception(f"Failed to set priority: {e}")
            return OperationResult.failure(f"Failed to set priority: {e}", "INTERNAL_ERROR")

    def get_tasks_by_job(
        self,
        actor: Actor,
        job_id: str,
        tenant_id: str,
    ) -> OperationResult[List[TaskResponse]]:
        """Get all tasks associated with a job.
        
        Args:
            actor: The actor performing the operation
            job_id: The job ID to filter by
            tenant_id: The tenant ID
            
        Returns:
            OperationResult containing list of tasks or error
        """
        try:
            # Check permissions - actor needs read permission for the tenant
            if not self._permissions.can_read(actor, tenant_id):
                return OperationResult.failure("Permission denied", "PERMISSION_DENIED")
            
            # Query tasks filtered by job_id
            result = self._repository.list_tasks(
                tenant_id=tenant_id,
                job_id=job_id,
            )
            
            tasks = [
                TaskResponse.from_dict(task) 
                for task in result.get("tasks", [])
            ]
            
            return OperationResult.success(tasks)
            
        except PermissionDeniedError as e:
            return OperationResult.failure(str(e), "PERMISSION_DENIED")
        except Exception as e:
            logger.exception(f"Failed to get tasks by job: {e}")
            return OperationResult.failure(f"Failed to get tasks by job: {e}", "INTERNAL_ERROR")


__all__ = [
    "TaskService",
]
