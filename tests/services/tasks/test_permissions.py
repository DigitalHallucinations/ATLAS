"""
Tests for TaskPermissionChecker.

Validates permission hierarchy, tenant isolation, and task access controls.

Author: ATLAS Team
Date: Jan 10, 2026
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict

import pytest

from core.services.common.types import Actor
from core.services.tasks.permissions import TaskPermissionChecker


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def checker() -> TaskPermissionChecker:
    """Task permission checker instance."""
    return TaskPermissionChecker()


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
    """Admin with full task permissions."""
    return Actor(
        type="user",
        id="admin_1",
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
    """User with read-only permissions."""
    return Actor(
        type="user",
        id="reader_1",
        tenant_id="tenant_1",
        permissions={"task:read"},
    )


@pytest.fixture
def no_perms_actor() -> Actor:
    """User with no task permissions."""
    return Actor(
        type="user",
        id="noperms_1",
        tenant_id="tenant_1",
        permissions=set(),
    )


@pytest.fixture
def other_tenant_actor() -> Actor:
    """User in different tenant."""
    return Actor(
        type="user",
        id="user_2",
        tenant_id="tenant_2",
        permissions={"task:admin", "task:write", "task:read"},
    )


def _make_task_record(
    task_id: str = "task_1",
    tenant_id: str = "tenant_1",
    owner_id: str = "writer_1",
    **kwargs: Any,
) -> Dict[str, Any]:
    """Create a mock task record."""
    now = datetime.now(timezone.utc)
    return {
        "id": task_id,
        "title": kwargs.get("title", "Test Task"),
        "tenant_id": tenant_id,
        "owner_id": owner_id,
        "status": kwargs.get("status", "pending"),
        "priority": kwargs.get("priority", 50),
        "created_at": now,
        "updated_at": now,
        "metadata": kwargs.get("metadata", {}),
    }


# =============================================================================
# Permission Hierarchy Tests
# =============================================================================


class TestPermissionHierarchy:
    """Test permission inheritance and hierarchy."""
    
    def test_wildcard_grants_all(
        self,
        checker: TaskPermissionChecker,
        system_actor: Actor,
    ) -> None:
        """Wildcard permission grants all capabilities."""
        assert checker.can_read(system_actor, "tenant_1")
        assert checker.can_write(system_actor, "tenant_1")
    
    def test_admin_implies_write_and_read(
        self,
        checker: TaskPermissionChecker,
    ) -> None:
        """Admin permission implies write and read."""
        admin_only = Actor(
            type="user",
            id="admin_only",
            tenant_id="tenant_1",
            permissions={"task:admin"},
        )
        
        assert checker.can_read(admin_only, "tenant_1")
        assert checker.can_write(admin_only, "tenant_1")
    
    def test_write_implies_read(
        self,
        checker: TaskPermissionChecker,
    ) -> None:
        """Write permission implies read."""
        write_only = Actor(
            type="user",
            id="write_only",
            tenant_id="tenant_1",
            permissions={"task:write"},
        )
        
        assert checker.can_read(write_only, "tenant_1")
        assert checker.can_write(write_only, "tenant_1")
    
    def test_read_is_minimal(
        self,
        checker: TaskPermissionChecker,
        reader_actor: Actor,
    ) -> None:
        """Read permission is the minimal permission level."""
        assert checker.can_read(reader_actor, "tenant_1")
        assert not checker.can_write(reader_actor, "tenant_1")


# =============================================================================
# Tenant Isolation Tests
# =============================================================================


class TestTenantIsolation:
    """Test multi-tenant access controls."""
    
    def test_own_tenant_access(
        self,
        checker: TaskPermissionChecker,
        writer_actor: Actor,
    ) -> None:
        """Users can access their own tenant."""
        assert checker.can_read(writer_actor, "tenant_1")
    
    def test_cross_tenant_denied(
        self,
        checker: TaskPermissionChecker,
        writer_actor: Actor,
    ) -> None:
        """Users cannot access other tenants."""
        assert not checker.can_read(writer_actor, "other_tenant")
    
    def test_system_bypasses_tenant(
        self,
        checker: TaskPermissionChecker,
        system_actor: Actor,
    ) -> None:
        """System actor can access any tenant."""
        assert checker.can_read(system_actor, "any_tenant")
    
    def test_admin_in_other_tenant_denied(
        self,
        checker: TaskPermissionChecker,
        other_tenant_actor: Actor,
    ) -> None:
        """Admin in different tenant still denied."""
        assert not checker.can_write(other_tenant_actor, "tenant_1")


# =============================================================================
# Task Access Tests
# =============================================================================


class TestTaskAccess:
    """Test task-level access controls."""
    
    def test_owner_can_read_own_task(
        self,
        checker: TaskPermissionChecker,
        writer_actor: Actor,
    ) -> None:
        """Task owner can read their task."""
        task = _make_task_record(owner_id="writer_1")
        assert checker.can_access_task(writer_actor, task)
    
    def test_owner_can_write_own_task(
        self,
        checker: TaskPermissionChecker,
        writer_actor: Actor,
    ) -> None:
        """Task owner can modify their task."""
        task = _make_task_record(owner_id="writer_1")
        # Should not raise
        checker.require_task_write(writer_actor, task)
    
    def test_tenant_user_can_read_others_task(
        self,
        checker: TaskPermissionChecker,
        reader_actor: Actor,
    ) -> None:
        """Tenant users can read other users' tasks."""
        task = _make_task_record(owner_id="other_user")
        assert checker.can_access_task(reader_actor, task)
    
    def test_non_owner_write_denied_without_write_perm(
        self,
        checker: TaskPermissionChecker,
        reader_actor: Actor,
    ) -> None:
        """Non-owners cannot write without write permission."""
        from core.services.common import PermissionDeniedError
        task = _make_task_record(owner_id="other_user")
        with pytest.raises(PermissionDeniedError):
            checker.require_task_write(reader_actor, task)
    
    def test_cross_tenant_task_access_denied(
        self,
        checker: TaskPermissionChecker,
        other_tenant_actor: Actor,
    ) -> None:
        """Cannot access tasks from other tenant."""
        task = _make_task_record(tenant_id="tenant_1")
        assert not checker.can_access_task(other_tenant_actor, task)


# =============================================================================
# Task Delete Tests
# =============================================================================


class TestTaskDelete:
    """Test task deletion access controls."""
    
    def test_owner_can_delete(
        self,
        checker: TaskPermissionChecker,
        writer_actor: Actor,
    ) -> None:
        """Task owner can delete their task."""
        task = _make_task_record(owner_id="writer_1")
        # Should not raise
        checker.require_task_delete(writer_actor, task)
    
    def test_admin_can_delete_any(
        self,
        checker: TaskPermissionChecker,
        admin_actor: Actor,
    ) -> None:
        """Admin can delete any task in tenant."""
        task = _make_task_record(owner_id="other_user")
        # Should not raise
        checker.require_task_delete(admin_actor, task)
    
    def test_non_owner_cannot_delete(
        self,
        checker: TaskPermissionChecker,
        writer_actor: Actor,
    ) -> None:
        """Non-owners without admin cannot delete."""
        from core.services.common import PermissionDeniedError
        task = _make_task_record(owner_id="other_user")
        with pytest.raises(PermissionDeniedError):
            checker.require_task_delete(writer_actor, task)
    
    def test_system_can_delete_any(
        self,
        checker: TaskPermissionChecker,
        system_actor: Actor,
    ) -> None:
        """System actor can delete any task."""
        task = _make_task_record(tenant_id="any_tenant")
        # Should not raise
        checker.require_task_delete(system_actor, task)


# =============================================================================
# Boolean Check Tests
# =============================================================================


class TestBooleanChecks:
    """Test boolean permission check methods."""
    
    def test_can_read_true(
        self,
        checker: TaskPermissionChecker,
        reader_actor: Actor,
    ) -> None:
        """can_read returns True with read permission."""
        assert checker.can_read(reader_actor, "tenant_1")
    
    def test_can_read_false_no_perms(
        self,
        checker: TaskPermissionChecker,
        no_perms_actor: Actor,
    ) -> None:
        """can_read returns False without permissions."""
        assert not checker.can_read(no_perms_actor, "tenant_1")
    
    def test_can_write_true(
        self,
        checker: TaskPermissionChecker,
        writer_actor: Actor,
    ) -> None:
        """can_write returns True with write permission."""
        assert checker.can_write(writer_actor, "tenant_1")
    
    def test_can_write_false_reader(
        self,
        checker: TaskPermissionChecker,
        reader_actor: Actor,
    ) -> None:
        """can_write returns False with only read permission."""
        assert not checker.can_write(reader_actor, "tenant_1")
    
    def test_can_access_task_true(
        self,
        checker: TaskPermissionChecker,
        writer_actor: Actor,
    ) -> None:
        """can_access_task returns True for owner."""
        task = _make_task_record(owner_id="writer_1")
        assert checker.can_access_task(writer_actor, task)
    
    def test_can_access_task_false_cross_tenant(
        self,
        checker: TaskPermissionChecker,
        other_tenant_actor: Actor,
    ) -> None:
        """can_access_task returns False for cross-tenant."""
        task = _make_task_record(tenant_id="tenant_1")
        assert not checker.can_access_task(other_tenant_actor, task)


# =============================================================================
# Subtask Permission Tests
# =============================================================================


class TestSubtaskPermissions:
    """Test permission checks for subtask operations."""
    
    def test_create_subtask_requires_write(
        self,
        checker: TaskPermissionChecker,
        writer_actor: Actor,
    ) -> None:
        """Creating subtasks requires write permission on parent."""
        parent_task = _make_task_record(owner_id="writer_1")
        # Should not raise
        checker.require_task_write(writer_actor, parent_task)
    
    def test_read_subtask_with_parent_read(
        self,
        checker: TaskPermissionChecker,
        reader_actor: Actor,
    ) -> None:
        """Reading subtasks works with read on parent."""
        parent_task = _make_task_record(owner_id="other_user")
        assert checker.can_access_task(reader_actor, parent_task)
