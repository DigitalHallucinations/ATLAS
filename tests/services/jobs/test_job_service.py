"""
Tests for JobService.

Comprehensive test coverage for job CRUD operations,
lifecycle transitions, permission checks, and SOTA features.

Author: ATLAS Team
Date: Jan 10, 2026
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

# Import directly from submodules to avoid circular import issues
from core.services.common.types import Actor, OperationResult
from core.services.jobs.service import JobService
from core.services.jobs.permissions import JobPermissionChecker
from core.services.jobs.types import (
    JobStatus,
    JobResult,
    JobCreate,
    JobUpdate,
    JobFilters,
    JobCheckpoint,
    JobResponse,
    JobListResponse,
    ScheduleCreate,
    JobCreated,
    JobUpdated,
    JobDeleted,
    JobStatusChanged,
    JobStarted,
    JobSucceeded,
    JobFailed,
    JobCancelled,
)
from core.services.jobs.exceptions import JobError, JobNotFoundError


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
    """Admin user with job admin permissions."""
    return Actor(
        type="user",
        id="admin_user",
        tenant_id="tenant_1",
        permissions={"job:admin", "job:write", "job:read", "job:execute"},
    )


@pytest.fixture
def executor_actor() -> Actor:
    """User with execute permissions."""
    return Actor(
        type="user",
        id="executor_1",
        tenant_id="tenant_1",
        permissions={"job:execute", "job:read"},
    )


@pytest.fixture
def writer_actor() -> Actor:
    """User with write permissions."""
    return Actor(
        type="user",
        id="writer_1",
        tenant_id="tenant_1",
        permissions={"job:write", "job:read"},
    )


@pytest.fixture
def reader_actor() -> Actor:
    """User with only read permissions."""
    return Actor(
        type="user",
        id="reader_1",
        tenant_id="tenant_1",
        permissions={"job:read"},
    )


@pytest.fixture
def other_tenant_actor() -> Actor:
    """User in a different tenant."""
    return Actor(
        type="user",
        id="user_2",
        tenant_id="tenant_2",
        permissions={"job:write", "job:read", "job:execute"},
    )


def _make_job_record(
    job_id: str = "job_1",
    name: str = "Test Job",
    tenant_id: str = "tenant_1",
    owner_id: str = "writer_1",
    status: str = "draft",
    **kwargs: Any,
) -> Dict[str, Any]:
    """Create a job record dict."""
    now = datetime.now(timezone.utc)
    return {
        "id": job_id,
        "name": name,
        "description": kwargs.get("description"),
        "tenant_id": tenant_id,
        "owner_id": owner_id,
        "status": status,
        "conversation_id": kwargs.get("conversation_id"),
        "metadata": kwargs.get("metadata", {}),
        "created_at": now,
        "updated_at": now,
    }


@pytest.fixture
def mock_repository() -> MagicMock:
    """Mock job repository."""
    repo = MagicMock()
    
    # Default behaviors
    repo.get_job = MagicMock(return_value=_make_job_record())
    repo.create_job = MagicMock(return_value=_make_job_record())
    repo.update_job = MagicMock(return_value=_make_job_record())
    repo.list_jobs = MagicMock(return_value={"jobs": [], "next_cursor": None})
    repo.upsert_schedule = MagicMock(return_value={})
    repo.delete_job = MagicMock(return_value=True)
    
    return repo


@pytest.fixture
def mock_message_bus() -> MagicMock:
    """Mock message bus."""
    bus = MagicMock()
    bus.publish = MagicMock()
    return bus


@pytest.fixture
def service(mock_repository: MagicMock, mock_message_bus: MagicMock) -> JobService:
    """Job service with mocks."""
    return JobService(
        repository=mock_repository,
        message_bus=mock_message_bus,
        permission_checker=JobPermissionChecker(),
    )


# =============================================================================
# Job CRUD Tests
# =============================================================================


class TestCreateJob:
    """Tests for create_job operation."""
    
    def test_create_job_success(
        self,
        service: JobService,
        writer_actor: Actor,
        mock_repository: MagicMock,
    ) -> None:
        """Successfully create a job."""
        job_data = JobCreate(
            name="New Job",
            tenant_id="tenant_1",
            description="Test description",
        )
        
        mock_repository.create_job.return_value = _make_job_record(
            name="New Job",
            description="Test description",
        )
        
        result = service.create_job(writer_actor, job_data)
        
        assert result.success
        assert result.data is not None
        assert result.data.name == "New Job"
        mock_repository.create_job.assert_called_once()
    
    def test_create_job_publishes_event(
        self,
        service: JobService,
        writer_actor: Actor,
        mock_message_bus: MagicMock,
        mock_repository: MagicMock,
    ) -> None:
        """Creating a job publishes JobCreated event."""
        job_data = JobCreate(
            name="New Job",
            tenant_id="tenant_1",
        )
        
        mock_repository.create_job.return_value = _make_job_record(name="New Job")
        
        service.create_job(writer_actor, job_data)
        
        mock_message_bus.publish.assert_called()
        call_args = mock_message_bus.publish.call_args
        assert call_args[0][0] == "job.created"
    
    def test_create_job_with_sota_fields(
        self,
        service: JobService,
        writer_actor: Actor,
        mock_repository: MagicMock,
    ) -> None:
        """Create job with SOTA enhancement fields."""
        job_data = JobCreate(
            name="SOTA Job",
            tenant_id="tenant_1",
            assigned_agent="researcher-persona",
            estimated_cost=Decimal("10.50"),
            timeout_seconds=3600,
            plan_id="plan_123",
            plan_step_index=2,
        )
        
        mock_repository.create_job.return_value = _make_job_record(
            name="SOTA Job",
            metadata={
                "assigned_agent": "researcher-persona",
                "estimated_cost": "10.50",
                "timeout_seconds": 3600,
                "plan_id": "plan_123",
                "plan_step_index": 2,
            }
        )
        
        result = service.create_job(writer_actor, job_data)
        
        assert result.success
        # Verify SOTA fields passed to repository
        create_call = mock_repository.create_job.call_args
        assert create_call is not None
    
    def test_create_job_denied_without_write_permission(
        self,
        service: JobService,
        reader_actor: Actor,
    ) -> None:
        """Users without write permission cannot create jobs."""
        job_data = JobCreate(
            name="New Job",
            tenant_id="tenant_1",
        )
        
        result = service.create_job(reader_actor, job_data)
        
        assert not result.success
        assert "DENIED" in (result.error_code or "")
    
    def test_create_job_denied_cross_tenant(
        self,
        service: JobService,
        other_tenant_actor: Actor,
    ) -> None:
        """Users cannot create jobs in other tenants."""
        job_data = JobCreate(
            name="New Job",
            tenant_id="tenant_1",  # Different from actor's tenant_2
        )
        
        result = service.create_job(other_tenant_actor, job_data)
        
        assert not result.success
        assert "DENIED" in (result.error_code or "")


class TestGetJob:
    """Tests for get_job operation."""
    
    def test_get_job_success(
        self,
        service: JobService,
        reader_actor: Actor,
        mock_repository: MagicMock,
    ) -> None:
        """Successfully get a job."""
        mock_repository.get_job.return_value = _make_job_record()
        
        result = service.get_job(reader_actor, "job_1", "tenant_1")
        
        assert result.success
        assert result.data is not None
        assert result.data.id == "job_1"
    
    def test_get_job_not_found(
        self,
        service: JobService,
        reader_actor: Actor,
        mock_repository: MagicMock,
    ) -> None:
        """Get non-existent job returns not found error."""
        mock_repository.get_job.side_effect = Exception("Job not found")
        
        result = service.get_job(reader_actor, "nonexistent", "tenant_1")
        
        assert not result.success
        assert result.error_code == "JOB_NOT_FOUND"
    
    def test_get_job_denied_other_tenant(
        self,
        service: JobService,
        other_tenant_actor: Actor,
        mock_repository: MagicMock,
    ) -> None:
        """Cannot get job from another tenant."""
        mock_repository.get_job.return_value = _make_job_record(
            tenant_id="tenant_1"  # Different from actor's tenant_2
        )
        
        result = service.get_job(other_tenant_actor, "job_1", "tenant_1")
        
        assert not result.success
        assert "DENIED" in (result.error_code or "")


class TestUpdateJob:
    """Tests for update_job operation."""
    
    def test_update_job_success(
        self,
        service: JobService,
        writer_actor: Actor,
        mock_repository: MagicMock,
    ) -> None:
        """Successfully update a job."""
        mock_repository.get_job.return_value = _make_job_record()
        mock_repository.update_job.return_value = _make_job_record(
            name="Updated Name",
            description="Updated description",
        )
        
        updates = JobUpdate(
            name="Updated Name",
            description="Updated description",
        )
        
        result = service.update_job(writer_actor, "job_1", "tenant_1", updates)
        
        assert result.success
        mock_repository.update_job.assert_called_once()
    
    def test_update_job_publishes_event(
        self,
        service: JobService,
        writer_actor: Actor,
        mock_repository: MagicMock,
        mock_message_bus: MagicMock,
    ) -> None:
        """Updating a job publishes JobUpdated event."""
        mock_repository.get_job.return_value = _make_job_record()
        mock_repository.update_job.return_value = _make_job_record(name="Updated")
        
        updates = JobUpdate(name="Updated")
        service.update_job(writer_actor, "job_1", "tenant_1", updates)
        
        mock_message_bus.publish.assert_called()
        call_args = mock_message_bus.publish.call_args
        assert call_args[0][0] == "job.updated"
    
    def test_update_job_denied_reader(
        self,
        service: JobService,
        reader_actor: Actor,
        mock_repository: MagicMock,
    ) -> None:
        """Readers cannot update jobs."""
        mock_repository.get_job.return_value = _make_job_record()
        
        updates = JobUpdate(name="Updated")
        result = service.update_job(reader_actor, "job_1", "tenant_1", updates)
        
        assert not result.success
        assert "DENIED" in (result.error_code or "")


class TestListJobs:
    """Tests for list_jobs operation."""
    
    def test_list_jobs_success(
        self,
        service: JobService,
        reader_actor: Actor,
        mock_repository: MagicMock,
    ) -> None:
        """Successfully list jobs."""
        mock_repository.list_jobs.return_value = {
            "jobs": [_make_job_record(), _make_job_record(job_id="job_2")],
            "next_cursor": None,
        }
        
        result = service.list_jobs(reader_actor, "tenant_1")
        
        assert result.success
        assert result.data is not None
        assert len(result.data.jobs) == 2
    
    def test_list_jobs_with_filters(
        self,
        service: JobService,
        reader_actor: Actor,
        mock_repository: MagicMock,
    ) -> None:
        """List jobs with status filter."""
        mock_repository.list_jobs.return_value = {"jobs": [], "next_cursor": None}
        
        filters = JobFilters(status="running")
        service.list_jobs(reader_actor, "tenant_1", filters)
        
        mock_repository.list_jobs.assert_called_once()
        call_kwargs = mock_repository.list_jobs.call_args[1]
        assert call_kwargs["status"] == "running"


# =============================================================================
# Lifecycle Transition Tests
# =============================================================================


class TestJobLifecycle:
    """Tests for job lifecycle transitions."""
    
    def test_start_job_success(
        self,
        service: JobService,
        executor_actor: Actor,
        mock_repository: MagicMock,
    ) -> None:
        """Successfully start a scheduled job."""
        mock_repository.get_job.return_value = _make_job_record(status="scheduled")
        mock_repository.update_job.return_value = _make_job_record(status="running")
        
        result = service.start_job(executor_actor, "job_1", "tenant_1")
        
        assert result.success
        assert result.data is not None
    
    def test_start_job_from_draft_fails(
        self,
        service: JobService,
        executor_actor: Actor,
        mock_repository: MagicMock,
    ) -> None:
        """Cannot start a job directly from draft status."""
        mock_repository.get_job.return_value = _make_job_record(status="draft")
        
        result = service.start_job(executor_actor, "job_1", "tenant_1")
        
        assert not result.success
        assert result.error_code == "INVALID_TRANSITION"
    
    def test_complete_job_success(
        self,
        service: JobService,
        executor_actor: Actor,
        mock_repository: MagicMock,
        mock_message_bus: MagicMock,
    ) -> None:
        """Successfully complete a running job."""
        mock_repository.get_job.return_value = _make_job_record(status="running")
        mock_repository.update_job.return_value = _make_job_record(status="succeeded")
        
        job_result = JobResult(
            success=True,
            result_summary="Completed successfully",
            actual_cost=Decimal("5.25"),
        )
        
        result = service.complete_job(executor_actor, "job_1", "tenant_1", job_result)
        
        assert result.success
        mock_message_bus.publish.assert_called()
    
    def test_fail_job_success(
        self,
        service: JobService,
        executor_actor: Actor,
        mock_repository: MagicMock,
    ) -> None:
        """Successfully fail a running job."""
        mock_repository.get_job.return_value = _make_job_record(status="running")
        mock_repository.update_job.return_value = _make_job_record(status="failed")
        
        result = service.fail_job(
            executor_actor, "job_1", "tenant_1",
            error="Something went wrong",
        )
        
        assert result.success
    
    def test_cancel_job_success(
        self,
        service: JobService,
        executor_actor: Actor,
        mock_repository: MagicMock,
    ) -> None:
        """Successfully cancel a job."""
        mock_repository.get_job.return_value = _make_job_record(status="running")
        mock_repository.update_job.return_value = _make_job_record(status="cancelled")
        
        result = service.cancel_job(
            executor_actor, "job_1", "tenant_1",
            reason="No longer needed",
        )
        
        assert result.success
    
    def test_cannot_cancel_completed_job(
        self,
        service: JobService,
        executor_actor: Actor,
        mock_repository: MagicMock,
    ) -> None:
        """Cannot cancel an already completed job."""
        mock_repository.get_job.return_value = _make_job_record(status="succeeded")
        
        result = service.cancel_job(
            executor_actor, "job_1", "tenant_1",
            reason="Changed mind",
        )
        
        assert not result.success
        assert result.error_code == "INVALID_TRANSITION"


# =============================================================================
# SOTA Feature Tests
# =============================================================================


class TestSOTAFeatures:
    """Tests for SOTA enhancement features."""
    
    def test_save_checkpoint(
        self,
        service: JobService,
        executor_actor: Actor,
        mock_repository: MagicMock,
        mock_message_bus: MagicMock,
    ) -> None:
        """Save a checkpoint for a running job."""
        mock_repository.get_job.return_value = _make_job_record(status="running")
        mock_repository.update_job.return_value = _make_job_record(
            status="running",
            metadata={"checkpoint_data": {"step": 3}},
        )
        
        checkpoint = JobCheckpoint(
            step_index=3,
            step_name="Data Processing",
            state={"processed_items": 150},
            execution_context={"current_batch": 2},
        )
        
        result = service.save_checkpoint(executor_actor, "job_1", "tenant_1", checkpoint)
        
        assert result.success
        mock_message_bus.publish.assert_called()
    
    def test_assign_agent(
        self,
        service: JobService,
        writer_actor: Actor,
        mock_repository: MagicMock,
        mock_message_bus: MagicMock,
    ) -> None:
        """Assign an agent to a job."""
        mock_repository.get_job.return_value = _make_job_record()
        mock_repository.update_job.return_value = _make_job_record(
            metadata={"assigned_agent": "analyst-persona"},
        )
        
        result = service.assign_agent(
            writer_actor, "job_1", "tenant_1",
            agent_name="analyst-persona",
        )
        
        assert result.success
        mock_message_bus.publish.assert_called()
    
    def test_update_execution_context(
        self,
        service: JobService,
        executor_actor: Actor,
        mock_repository: MagicMock,
    ) -> None:
        """Update execution context (scratchpad) for a job."""
        mock_repository.get_job.return_value = _make_job_record(
            status="running",
            metadata={"execution_context": {"key1": "value1"}},
        )
        mock_repository.update_job.return_value = _make_job_record(
            status="running",
            metadata={"execution_context": {"key1": "value1", "key2": "value2"}},
        )
        
        result = service.update_execution_context(
            executor_actor, "job_1", "tenant_1",
            context_updates={"key2": "value2"},
            merge=True,
        )
        
        assert result.success


# =============================================================================
# Schedule Tests
# =============================================================================


class TestJobScheduling:
    """Tests for job scheduling functionality."""
    
    def test_schedule_job_success(
        self,
        service: JobService,
        executor_actor: Actor,
        mock_repository: MagicMock,
    ) -> None:
        """Successfully schedule a job."""
        mock_repository.get_job.return_value = _make_job_record(status="draft")
        mock_repository.upsert_schedule.return_value = {}
        mock_repository.update_job.return_value = _make_job_record(status="scheduled")
        
        schedule = ScheduleCreate(
            schedule_type="cron",
            expression="0 9 * * *",
            timezone="UTC",
        )
        
        result = service.schedule_job(executor_actor, "job_1", "tenant_1", schedule)
        
        assert result.success
        mock_repository.upsert_schedule.assert_called_once()
