"""
Tests for TaskService.

Comprehensive test coverage for task CRUD operations,
lifecycle transitions, subtasks, dependencies, and SOTA features.

Author: ATLAS Team
Date: Jan 10, 2026
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

# Import directly from submodules to avoid circular import issues
from core.services.common.types import Actor, OperationResult
from core.services.tasks.service import TaskService
from core.services.tasks.permissions import TaskPermissionChecker
from core.services.tasks.types import (
    TaskStatus,
    TaskPriority,
    TaskCreate,
    TaskUpdate,
    TaskFilters,
    SubtaskCreate,
    DependencyCreate,
    TaskResponse,
    TaskListResponse,
    TaskCreated,
    TaskUpdated,
    TaskDeleted,
    TaskStatusChanged,
    TaskCompleted,
    TaskCancelled,
    TaskAssigned,
)
from core.services.tasks.exceptions import (
    TaskError,
    TaskNotFoundError,
    TaskCircularDependencyError,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def system_actor() -> Actor:
    """System actor with full permissions."""
    return Actor(
        type="system",
        id="system",
        tenant_id="system",
        permissions={"*"},
    )


@pytest.fixture
def admin_actor() -> Actor:
    """Admin user with full task permissions."""
    return Actor(
        type="user",
        id="admin_user",
        tenant_id="tenant_1",
        permissions={"task:admin", "task:write", "task:read"},
    )


@pytest.fixture
def writer_actor() -> Actor:
    """User with write permissions."""
    return Actor(
        type="user",
        id="writer_1",
        tenant_id="tenant_1",
        permissions={"task:write", "task:read"},
    )


@pytest.fixture
def reader_actor() -> Actor:
    """User with only read permissions."""
    return Actor(
        type="user",
        id="reader_1",
        tenant_id="tenant_1",
        permissions={"task:read"},
    )


@pytest.fixture
def other_tenant_actor() -> Actor:
    """User in a different tenant."""
    return Actor(
        type="user",
        id="user_2",
        tenant_id="tenant_2",
        permissions={"task:write", "task:read"},
    )


def _make_task_record(
    task_id: str = "task_1",
    title: str = "Test Task",
    tenant_id: str = "tenant_1",
    owner_id: str = "writer_1",
    status: str = "ready",
    priority: int = 50,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Create a task record dict."""
    now = datetime.now(timezone.utc)
    return {
        "id": task_id,
        "title": title,
        "description": kwargs.get("description"),
        "tenant_id": tenant_id,
        "owner_id": owner_id,
        "status": status,
        "priority": priority,
        "parent_id": kwargs.get("parent_id"),
        "job_id": kwargs.get("job_id"),
        "conversation_id": kwargs.get("conversation_id"),
        "due_date": kwargs.get("due_date"),
        "metadata": kwargs.get("metadata", {}),
        "created_at": now,
        "updated_at": now,
    }


@pytest.fixture
def mock_repository() -> MagicMock:
    """Mock task repository."""
    repo = MagicMock()
    
    # Default behaviors
    repo.get_task = MagicMock(return_value=_make_task_record())
    repo.create_task = MagicMock(return_value=_make_task_record())
    repo.update_task = MagicMock(return_value=_make_task_record())
    repo.list_tasks = MagicMock(return_value={"tasks": [], "next_cursor": None})
    repo.delete_task = MagicMock(return_value=True)
    repo.get_subtasks = MagicMock(return_value=[])
    repo.add_dependency = MagicMock(return_value=True)
    repo.get_dependencies = MagicMock(return_value=[])
    repo.remove_dependency = MagicMock(return_value=True)
    
    return repo


@pytest.fixture
def mock_message_bus() -> MagicMock:
    """Mock message bus."""
    bus = MagicMock()
    bus.publish = MagicMock()
    return bus


@pytest.fixture
def service(mock_repository: MagicMock, mock_message_bus: MagicMock) -> TaskService:
    """Task service with mocks."""
    return TaskService(
        repository=mock_repository,
        message_bus=mock_message_bus,
        permission_checker=TaskPermissionChecker(),
    )


# =============================================================================
# Task CRUD Tests
# =============================================================================


class TestCreateTask:
    """Tests for create_task operation."""
    
    def test_create_task_success(
        self,
        service: TaskService,
        writer_actor: Actor,
        mock_repository: MagicMock,
    ) -> None:
        """Successfully create a task."""
        task_data = TaskCreate(
            title="New Task",
            tenant_id="tenant_1",
            description="Test description",
            priority=75,
        )
        
        mock_repository.create_task.return_value = _make_task_record(
            title="New Task",
            description="Test description",
            priority=75,
        )
        
        result = service.create_task(writer_actor, task_data)
        
        assert result.success
        assert result.data is not None
        assert result.data.title == "New Task"
        mock_repository.create_task.assert_called_once()
    
    def test_create_task_publishes_event(
        self,
        service: TaskService,
        writer_actor: Actor,
        mock_message_bus: MagicMock,
        mock_repository: MagicMock,
    ) -> None:
        """Creating a task publishes TaskCreated event."""
        task_data = TaskCreate(
            title="New Task",
            tenant_id="tenant_1",
        )
        
        mock_repository.create_task.return_value = _make_task_record(title="New Task")
        
        service.create_task(writer_actor, task_data)
        
        mock_message_bus.publish.assert_called()
        call_args = mock_message_bus.publish.call_args
        assert call_args[0][0] == "task.created"
    
    def test_create_task_with_job_link(
        self,
        service: TaskService,
        writer_actor: Actor,
        mock_repository: MagicMock,
    ) -> None:
        """Create task linked to a job."""
        task_data = TaskCreate(
            title="Job Task",
            tenant_id="tenant_1",
            job_id="job_123",
        )
        
        mock_repository.create_task.return_value = _make_task_record(
            title="Job Task",
            job_id="job_123",
        )
        
        result = service.create_task(writer_actor, task_data)
        
        assert result.success
        create_call = mock_repository.create_task.call_args
        assert create_call is not None
    
    def test_create_task_with_sota_fields(
        self,
        service: TaskService,
        writer_actor: Actor,
        mock_repository: MagicMock,
    ) -> None:
        """Create task with SOTA enhancement fields."""
        task_data = TaskCreate(
            title="SOTA Task",
            tenant_id="tenant_1",
            assigned_agent="researcher-persona",
            estimated_cost=Decimal("5.00"),
            timeout_seconds=1800,
            execution_context={"mode": "deep_analysis"},
        )
        
        mock_repository.create_task.return_value = _make_task_record(
            title="SOTA Task",
            metadata={
                "assigned_agent": "researcher-persona",
                "estimated_cost": "5.00",
                "timeout_seconds": 1800,
                "execution_context": {"mode": "deep_analysis"},
            }
        )
        
        result = service.create_task(writer_actor, task_data)
        
        assert result.success
    
    def test_create_task_denied_without_write_permission(
        self,
        service: TaskService,
        reader_actor: Actor,
    ) -> None:
        """Users without write permission cannot create tasks."""
        task_data = TaskCreate(
            title="New Task",
            tenant_id="tenant_1",
        )
        
        result = service.create_task(reader_actor, task_data)
        
        assert not result.success
        assert result.error_code == "PERMISSION_DENIED"
    
    def test_create_task_denied_cross_tenant(
        self,
        service: TaskService,
        other_tenant_actor: Actor,
    ) -> None:
        """Users cannot create tasks in other tenants."""
        task_data = TaskCreate(
            title="New Task",
            tenant_id="tenant_1",
        )
        
        result = service.create_task(other_tenant_actor, task_data)
        
        assert not result.success


class TestGetTask:
    """Tests for get_task operation."""
    
    def test_get_task_success(
        self,
        service: TaskService,
        reader_actor: Actor,
        mock_repository: MagicMock,
    ) -> None:
        """Successfully get a task."""
        mock_repository.get_task.return_value = _make_task_record()
        
        result = service.get_task(reader_actor, "task_1", "tenant_1")
        
        assert result.success
        assert result.data is not None
        assert result.data.id == "task_1"
    
    def test_get_task_not_found(
        self,
        service: TaskService,
        reader_actor: Actor,
        mock_repository: MagicMock,
    ) -> None:
        """Get non-existent task returns not found error."""
        mock_repository.get_task.side_effect = Exception("Task not found")
        
        result = service.get_task(reader_actor, "nonexistent", "tenant_1")
        
        assert not result.success
        assert result.error_code == "TASK_NOT_FOUND"
    
    def test_get_task_denied_other_tenant(
        self,
        service: TaskService,
        other_tenant_actor: Actor,
        mock_repository: MagicMock,
    ) -> None:
        """Cannot get task from another tenant."""
        mock_repository.get_task.return_value = _make_task_record(
            tenant_id="tenant_1"
        )
        
        result = service.get_task(other_tenant_actor, "task_1", "tenant_1")
        
        assert not result.success


class TestUpdateTask:
    """Tests for update_task operation."""
    
    def test_update_task_success(
        self,
        service: TaskService,
        writer_actor: Actor,
        mock_repository: MagicMock,
    ) -> None:
        """Successfully update a task."""
        mock_repository.get_task.return_value = _make_task_record()
        mock_repository.update_task.return_value = _make_task_record(
            title="Updated Title",
            priority=90,
        )
        
        updates = TaskUpdate(
            title="Updated Title",
            priority=90,
        )
        
        result = service.update_task(writer_actor, "task_1", "tenant_1", updates)
        
        assert result.success
        mock_repository.update_task.assert_called_once()
    
    def test_update_task_publishes_event(
        self,
        service: TaskService,
        writer_actor: Actor,
        mock_repository: MagicMock,
        mock_message_bus: MagicMock,
    ) -> None:
        """Updating a task publishes TaskUpdated event."""
        mock_repository.get_task.return_value = _make_task_record()
        mock_repository.update_task.return_value = _make_task_record(title="Updated")
        
        updates = TaskUpdate(title="Updated")
        service.update_task(writer_actor, "task_1", "tenant_1", updates)
        
        mock_message_bus.publish.assert_called()
        call_args = mock_message_bus.publish.call_args
        assert call_args[0][0] == "task.updated"
    
    def test_update_task_denied_reader(
        self,
        service: TaskService,
        reader_actor: Actor,
        mock_repository: MagicMock,
    ) -> None:
        """Readers cannot update tasks."""
        mock_repository.get_task.return_value = _make_task_record()
        
        updates = TaskUpdate(title="Updated")
        result = service.update_task(reader_actor, "task_1", "tenant_1", updates)
        
        assert not result.success


class TestDeleteTask:
    """Tests for delete_task operation."""
    
    def test_delete_task_success(
        self,
        service: TaskService,
        writer_actor: Actor,
        mock_repository: MagicMock,
    ) -> None:
        """Successfully delete a task."""
        mock_repository.get_task.return_value = _make_task_record()
        mock_repository.delete_task.return_value = True
        
        result = service.delete_task(writer_actor, "task_1", "tenant_1")
        
        assert result.success
    
    def test_delete_task_publishes_event(
        self,
        service: TaskService,
        writer_actor: Actor,
        mock_repository: MagicMock,
        mock_message_bus: MagicMock,
    ) -> None:
        """Deleting a task publishes TaskDeleted event."""
        mock_repository.get_task.return_value = _make_task_record()
        mock_repository.delete_task.return_value = True
        
        service.delete_task(writer_actor, "task_1", "tenant_1")
        
        mock_message_bus.publish.assert_called()
        call_args = mock_message_bus.publish.call_args
        assert call_args[0][0] == "task.deleted"


class TestListTasks:
    """Tests for list_tasks operation."""
    
    def test_list_tasks_success(
        self,
        service: TaskService,
        reader_actor: Actor,
        mock_repository: MagicMock,
    ) -> None:
        """Successfully list tasks."""
        mock_repository.list_tasks.return_value = {
            "tasks": [_make_task_record(), _make_task_record(task_id="task_2")],
            "next_cursor": None,
        }
        
        result = service.list_tasks(reader_actor, "tenant_1")
        
        assert result.success
        assert result.data is not None
        assert len(result.data.tasks) == 2
    
    def test_list_tasks_with_filters(
        self,
        service: TaskService,
        reader_actor: Actor,
        mock_repository: MagicMock,
    ) -> None:
        """List tasks with status filter."""
        mock_repository.list_tasks.return_value = {"tasks": [], "next_cursor": None}
        
        filters = TaskFilters(status="in_progress")
        service.list_tasks(reader_actor, "tenant_1", filters)
        
        mock_repository.list_tasks.assert_called_once()
        call_kwargs = mock_repository.list_tasks.call_args[1]
        assert call_kwargs["status"] == "in_progress"
    
    def test_list_tasks_with_job_filter(
        self,
        service: TaskService,
        reader_actor: Actor,
        mock_repository: MagicMock,
    ) -> None:
        """List tasks filtered by job."""
        mock_repository.list_tasks.return_value = {"tasks": [], "next_cursor": None}
        
        filters = TaskFilters(job_id="job_123")
        service.list_tasks(reader_actor, "tenant_1", filters)
        
        call_kwargs = mock_repository.list_tasks.call_args[1]
        assert call_kwargs["job_id"] == "job_123"


# =============================================================================
# Lifecycle Transition Tests
# =============================================================================


class TestTaskLifecycle:
    """Tests for task lifecycle transitions."""
    
    def test_start_task_success(
        self,
        service: TaskService,
        writer_actor: Actor,
        mock_repository: MagicMock,
    ) -> None:
        """Successfully start a ready task."""
        mock_repository.get_task.return_value = _make_task_record(status="ready")
        mock_repository.update_task.return_value = _make_task_record(status="in_progress")
        
        result = service.start_task(writer_actor, "task_1", "tenant_1")
        
        assert result.success
    
    def test_complete_task_success(
        self,
        service: TaskService,
        writer_actor: Actor,
        mock_repository: MagicMock,
        mock_message_bus: MagicMock,
    ) -> None:
        """Successfully complete a task in review."""
        mock_repository.get_task.return_value = _make_task_record(status="review")
        mock_repository.update_task.return_value = _make_task_record(status="done")
        
        result = service.complete_task(
            writer_actor, "task_1", "tenant_1",
            completion_notes="Done successfully",
        )
        
        assert result.success
        mock_message_bus.publish.assert_called()
    
    def test_cancel_task_success(
        self,
        service: TaskService,
        writer_actor: Actor,
        mock_repository: MagicMock,
    ) -> None:
        """Successfully cancel a task."""
        mock_repository.get_task.return_value = _make_task_record(status="in_progress")
        mock_repository.update_task.return_value = _make_task_record(status="cancelled")
        
        result = service.cancel_task(
            writer_actor, "task_1", "tenant_1",
            reason="No longer needed",
        )
        
        assert result.success
    
    def test_cannot_complete_cancelled_task(
        self,
        service: TaskService,
        writer_actor: Actor,
        mock_repository: MagicMock,
    ) -> None:
        """Cannot complete an already cancelled task."""
        mock_repository.get_task.return_value = _make_task_record(status="cancelled")
        
        result = service.complete_task(
            writer_actor, "task_1", "tenant_1",
        )
        
        assert not result.success
        assert result.error_code == "INVALID_TRANSITION"


# =============================================================================
# Subtask Tests
# =============================================================================


class TestSubtasks:
    """Tests for subtask operations."""
    
    def test_create_subtask_success(
        self,
        service: TaskService,
        writer_actor: Actor,
        mock_repository: MagicMock,
    ) -> None:
        """Successfully create a subtask."""
        mock_repository.get_task.return_value = _make_task_record()
        mock_repository.create_task.return_value = _make_task_record(
            task_id="subtask_1",
            title="Subtask",
            parent_id="task_1",
        )
        
        subtask_data = SubtaskCreate(
            title="Subtask",
            tenant_id="tenant_1",
        )
        
        result = service.create_subtask(
            writer_actor, "task_1", "tenant_1", subtask_data
        )
        
        assert result.success
        assert result.data is not None
    
    def test_get_subtasks_success(
        self,
        service: TaskService,
        reader_actor: Actor,
        mock_repository: MagicMock,
    ) -> None:
        """Successfully get subtasks of a task."""
        mock_repository.get_task.return_value = _make_task_record()
        mock_repository.get_subtasks.return_value = [
            _make_task_record(task_id="sub_1", parent_id="task_1"),
            _make_task_record(task_id="sub_2", parent_id="task_1"),
        ]
        
        result = service.get_subtasks(reader_actor, "task_1", "tenant_1")
        
        assert result.success
        assert result.data is not None
        assert len(result.data) == 2


# =============================================================================
# Dependency Tests
# =============================================================================


class TestDependencies:
    """Tests for task dependency operations."""
    
    def test_add_dependency_success(
        self,
        service: TaskService,
        writer_actor: Actor,
        mock_repository: MagicMock,
    ) -> None:
        """Successfully add a dependency."""
        mock_repository.get_task.return_value = _make_task_record()
        mock_repository.get_dependencies.return_value = []
        mock_repository.add_dependency.return_value = True
        
        dependency = DependencyCreate(
            depends_on_id="task_2",
            dependency_type="blocks",
        )
        
        result = service.add_dependency(
            writer_actor, "task_1", "tenant_1", dependency
        )
        
        assert result.success
    
    def test_get_dependencies_success(
        self,
        service: TaskService,
        reader_actor: Actor,
        mock_repository: MagicMock,
    ) -> None:
        """Successfully get task dependencies."""
        mock_repository.get_task.return_value = _make_task_record()
        mock_repository.get_dependencies.return_value = [
            {"depends_on_id": "task_2", "dependency_type": "blocks"},
        ]
        
        result = service.get_dependencies(reader_actor, "task_1", "tenant_1")
        
        assert result.success
        assert result.data is not None
    
    def test_remove_dependency_success(
        self,
        service: TaskService,
        writer_actor: Actor,
        mock_repository: MagicMock,
    ) -> None:
        """Successfully remove a dependency."""
        mock_repository.get_task.return_value = _make_task_record()
        mock_repository.remove_dependency.return_value = True
        
        result = service.remove_dependency(
            writer_actor, "task_1", "tenant_1", "task_2"
        )
        
        assert result.success


# =============================================================================
# SOTA Feature Tests
# =============================================================================


class TestSOTAFeatures:
    """Tests for SOTA enhancement features."""
    
    def test_assign_agent(
        self,
        service: TaskService,
        writer_actor: Actor,
        mock_repository: MagicMock,
        mock_message_bus: MagicMock,
    ) -> None:
        """Assign an agent to a task."""
        mock_repository.get_task.return_value = _make_task_record()
        mock_repository.update_task.return_value = _make_task_record(
            metadata={"assigned_agent": "researcher-persona"},
        )
        
        result = service.assign_task(
            writer_actor, "task_1", "tenant_1",
            assignee_id="researcher-persona",
            assignee_type="agent",
        )
        
        assert result.success
        mock_message_bus.publish.assert_called()
    
    def test_assign_user(
        self,
        service: TaskService,
        writer_actor: Actor,
        mock_repository: MagicMock,
        mock_message_bus: MagicMock,
    ) -> None:
        """Assign a user to a task."""
        mock_repository.get_task.return_value = _make_task_record()
        mock_repository.update_task.return_value = _make_task_record(
            metadata={"assignee_id": "user_123"},
        )
        
        result = service.assign_task(
            writer_actor, "task_1", "tenant_1",
            assignee_id="user_123",
            assignee_type="user",
        )
        
        assert result.success
    
    def test_update_execution_context(
        self,
        service: TaskService,
        writer_actor: Actor,
        mock_repository: MagicMock,
    ) -> None:
        """Update execution context (scratchpad) for a task."""
        mock_repository.get_task.return_value = _make_task_record(
            status="in_progress",
            metadata={"execution_context": {"step": 1}},
        )
        mock_repository.update_task.return_value = _make_task_record(
            status="in_progress",
            metadata={"execution_context": {"step": 1, "notes": "processing"}},
        )
        
        result = service.update_execution_context(
            writer_actor, "task_1", "tenant_1",
            context_updates={"notes": "processing"},
            merge=True,
        )
        
        assert result.success
    
    def test_set_priority(
        self,
        service: TaskService,
        writer_actor: Actor,
        mock_repository: MagicMock,
    ) -> None:
        """Set task priority."""
        mock_repository.get_task.return_value = _make_task_record(priority=50)
        mock_repository.update_task.return_value = _make_task_record(priority=100)
        
        result = service.set_priority(
            writer_actor, "task_1", "tenant_1",
            priority=100,
        )
        
        assert result.success
    
    def test_priority_validation(
        self,
        service: TaskService,
        writer_actor: Actor,
        mock_repository: MagicMock,
    ) -> None:
        """Priority must be within valid range (1-100)."""
        mock_repository.get_task.return_value = _make_task_record()
        
        result = service.set_priority(
            writer_actor, "task_1", "tenant_1",
            priority=150,  # Invalid - over 100
        )
        
        assert not result.success
        assert result.error_code == "VALIDATION_ERROR"


# =============================================================================
# Tasks by Job Tests
# =============================================================================


class TestTasksByJob:
    """Tests for job-related task operations."""
    
    def test_get_tasks_by_job(
        self,
        service: TaskService,
        reader_actor: Actor,
        mock_repository: MagicMock,
    ) -> None:
        """Get all tasks associated with a job."""
        mock_repository.list_tasks.return_value = {
            "tasks": [
                _make_task_record(task_id="t1", job_id="job_1"),
                _make_task_record(task_id="t2", job_id="job_1"),
            ],
            "next_cursor": None,
        }
        
        result = service.get_tasks_by_job(reader_actor, "job_1", "tenant_1")
        
        assert result.success
        assert result.data is not None
        assert len(result.data) == 2
