"""
Tests for JobPermissionChecker.

Comprehensive test coverage for job permission checking,
tenant isolation, and role-based access control.

Author: ATLAS Team
Date: Jan 10, 2026
"""

from __future__ import annotations

import pytest

from core.services.common.types import Actor
from core.services.common.exceptions import PermissionDeniedError
from core.services.jobs.permissions import JobPermissionChecker


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def checker() -> JobPermissionChecker:
    """Job permission checker instance."""
    return JobPermissionChecker()


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


@pytest.fixture
def sample_job() -> dict:
    """Sample job dict for tests."""
    return {
        "id": "job_1",
        "name": "Test Job",
        "tenant_id": "tenant_1",
        "owner_id": "writer_1",
        "status": "draft",
    }


@pytest.fixture
def other_tenant_job() -> dict:
    """Job in a different tenant."""
    return {
        "id": "job_2",
        "name": "Other Tenant Job",
        "tenant_id": "tenant_2",
        "owner_id": "user_2",
        "status": "draft",
    }


# =============================================================================
# Permission Hierarchy Tests
# =============================================================================


class TestPermissionHierarchy:
    """Tests for permission inheritance and hierarchy."""
    
    def test_wildcard_grants_all(
        self,
        checker: JobPermissionChecker,
        system_actor: Actor,
    ) -> None:
        """Wildcard permission grants all access."""
        checker.require_read(system_actor, "tenant_1")
        checker.require_write(system_actor, "tenant_1")
        checker.require_execute(system_actor, "tenant_1")
        checker.require_admin(system_actor)
        # No exceptions raised
    
    def test_admin_implies_all(
        self,
        checker: JobPermissionChecker,
        admin_actor: Actor,
    ) -> None:
        """Admin permission implies read, write, and execute."""
        # Remove explicit permissions, keeping only admin
        admin_only = Actor(
            type="user",
            id="admin_only",
            tenant_id="tenant_1",
            permissions={"job:admin"},
        )
        checker.require_read(admin_only, "tenant_1")
        checker.require_write(admin_only, "tenant_1")
        checker.require_execute(admin_only, "tenant_1")
        checker.require_admin(admin_only)
    
    def test_write_implies_read(
        self,
        checker: JobPermissionChecker,
    ) -> None:
        """Write permission implies read."""
        write_only = Actor(
            type="user",
            id="write_only",
            tenant_id="tenant_1",
            permissions={"job:write"},
        )
        checker.require_read(write_only, "tenant_1")
        # No exception
    
    def test_execute_implies_read(
        self,
        checker: JobPermissionChecker,
    ) -> None:
        """Execute permission implies read."""
        execute_only = Actor(
            type="user",
            id="execute_only",
            tenant_id="tenant_1",
            permissions={"job:execute"},
        )
        checker.require_read(execute_only, "tenant_1")
        # No exception


# =============================================================================
# Tenant Isolation Tests
# =============================================================================


class TestTenantIsolation:
    """Tests for tenant isolation enforcement."""
    
    def test_user_can_access_own_tenant(
        self,
        checker: JobPermissionChecker,
        writer_actor: Actor,
    ) -> None:
        """Users can access resources in their own tenant."""
        checker.require_read(writer_actor, "tenant_1")
        checker.require_write(writer_actor, "tenant_1")
    
    def test_user_cannot_access_other_tenant(
        self,
        checker: JobPermissionChecker,
        writer_actor: Actor,
    ) -> None:
        """Users cannot access resources in other tenants."""
        with pytest.raises(PermissionDeniedError) as exc_info:
            checker.require_read(writer_actor, "tenant_2")
        assert "JOB_TENANT_DENIED" in str(exc_info.value)
    
    def test_system_actor_bypasses_tenant(
        self,
        checker: JobPermissionChecker,
        system_actor: Actor,
    ) -> None:
        """System actors can access any tenant."""
        checker.require_read(system_actor, "tenant_1")
        checker.require_read(system_actor, "tenant_2")
        checker.require_read(system_actor, "any_tenant")
    
    def test_admin_can_access_any_tenant(
        self,
        checker: JobPermissionChecker,
        admin_actor: Actor,
    ) -> None:
        """Admin users can access any tenant."""
        checker.require_read(admin_actor, "tenant_1")
        checker.require_read(admin_actor, "tenant_2")


# =============================================================================
# Job-Specific Permission Tests
# =============================================================================


class TestJobAccess:
    """Tests for job-specific access control."""
    
    def test_owner_can_access_job(
        self,
        checker: JobPermissionChecker,
        writer_actor: Actor,
        sample_job: dict,
    ) -> None:
        """Job owner can access their job."""
        checker.require_job_read(writer_actor, sample_job)
    
    def test_tenant_user_can_read_tenant_job(
        self,
        checker: JobPermissionChecker,
        reader_actor: Actor,
        sample_job: dict,
    ) -> None:
        """Users with read permission can read jobs in their tenant."""
        checker.require_job_read(reader_actor, sample_job)
    
    def test_user_cannot_access_other_tenant_job(
        self,
        checker: JobPermissionChecker,
        writer_actor: Actor,
        other_tenant_job: dict,
    ) -> None:
        """Users cannot access jobs in other tenants."""
        with pytest.raises(PermissionDeniedError):
            checker.require_job_read(writer_actor, other_tenant_job)
    
    def test_reader_cannot_write_job(
        self,
        checker: JobPermissionChecker,
        reader_actor: Actor,
        sample_job: dict,
    ) -> None:
        """Users with only read permission cannot write jobs."""
        with pytest.raises(PermissionDeniedError) as exc_info:
            checker.require_job_write(reader_actor, sample_job)
        assert "JOB_WRITE_DENIED" in str(exc_info.value)
    
    def test_writer_can_modify_tenant_job(
        self,
        checker: JobPermissionChecker,
        writer_actor: Actor,
        sample_job: dict,
    ) -> None:
        """Users with write permission can modify jobs in their tenant."""
        checker.require_job_write(writer_actor, sample_job)
    
    def test_executor_can_execute_job(
        self,
        checker: JobPermissionChecker,
        executor_actor: Actor,
        sample_job: dict,
    ) -> None:
        """Users with execute permission can execute jobs."""
        checker.require_job_execute(executor_actor, sample_job)
    
    def test_writer_cannot_execute_job(
        self,
        checker: JobPermissionChecker,
        writer_actor: Actor,
        sample_job: dict,
    ) -> None:
        """Users with only write permission cannot execute jobs."""
        with pytest.raises(PermissionDeniedError) as exc_info:
            checker.require_job_execute(writer_actor, sample_job)
        assert "JOB_EXECUTE_DENIED" in str(exc_info.value)


# =============================================================================
# Delete Permission Tests
# =============================================================================


class TestJobDelete:
    """Tests for job deletion permissions."""
    
    def test_owner_can_delete_job(
        self,
        checker: JobPermissionChecker,
        writer_actor: Actor,
        sample_job: dict,
    ) -> None:
        """Job owner can delete their job."""
        checker.require_job_delete(writer_actor, sample_job)
    
    def test_admin_can_delete_any_job(
        self,
        checker: JobPermissionChecker,
        admin_actor: Actor,
        sample_job: dict,
    ) -> None:
        """Admin can delete any job in accessible tenants."""
        checker.require_job_delete(admin_actor, sample_job)
    
    def test_non_owner_cannot_delete(
        self,
        checker: JobPermissionChecker,
        sample_job: dict,
    ) -> None:
        """Non-owner with write permission cannot delete others' jobs."""
        other_writer = Actor(
            type="user",
            id="other_writer",
            tenant_id="tenant_1",
            permissions={"job:write", "job:read"},
        )
        with pytest.raises(PermissionDeniedError) as exc_info:
            checker.require_job_delete(other_writer, sample_job)
        assert "JOB_DELETE_DENIED" in str(exc_info.value)
    
    def test_system_actor_can_delete_any_job(
        self,
        checker: JobPermissionChecker,
        system_actor: Actor,
        sample_job: dict,
    ) -> None:
        """System actors can delete any job."""
        checker.require_job_delete(system_actor, sample_job)


# =============================================================================
# Boolean Check Methods
# =============================================================================


class TestBooleanChecks:
    """Tests for boolean check methods (non-raising)."""
    
    def test_can_read_returns_true(
        self,
        checker: JobPermissionChecker,
        reader_actor: Actor,
    ) -> None:
        """can_read returns True when permitted."""
        assert checker.can_read(reader_actor, "tenant_1") is True
    
    def test_can_read_returns_false(
        self,
        checker: JobPermissionChecker,
    ) -> None:
        """can_read returns False when not permitted."""
        no_perms = Actor(
            type="user",
            id="no_perms",
            tenant_id="tenant_1",
            permissions=set(),
        )
        assert checker.can_read(no_perms, "tenant_1") is False
    
    def test_can_write_returns_true(
        self,
        checker: JobPermissionChecker,
        writer_actor: Actor,
    ) -> None:
        """can_write returns True when permitted."""
        assert checker.can_write(writer_actor, "tenant_1") is True
    
    def test_can_write_returns_false_for_reader(
        self,
        checker: JobPermissionChecker,
        reader_actor: Actor,
    ) -> None:
        """can_write returns False for read-only users."""
        assert checker.can_write(reader_actor, "tenant_1") is False
    
    def test_can_execute_returns_true(
        self,
        checker: JobPermissionChecker,
        executor_actor: Actor,
    ) -> None:
        """can_execute returns True when permitted."""
        assert checker.can_execute(executor_actor, "tenant_1") is True
    
    def test_can_access_job_respects_tenant(
        self,
        checker: JobPermissionChecker,
        writer_actor: Actor,
        sample_job: dict,
        other_tenant_job: dict,
    ) -> None:
        """can_access_job respects tenant isolation."""
        assert checker.can_access_job(writer_actor, sample_job) is True
        assert checker.can_access_job(writer_actor, other_tenant_job) is False
