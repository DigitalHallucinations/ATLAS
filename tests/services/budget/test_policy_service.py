"""
Tests for BudgetPolicyService.

Comprehensive test coverage for budget policy CRUD operations,
permission checks, and budget pre-flight validation.

Author: ATLAS Team
Date: Jan 8, 2026
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.services.common import Actor, OperationResult, PermissionDeniedError
from core.services.budget import (
    BudgetPolicyService,
    BudgetPermissionChecker,
    BudgetPolicyCreate,
    BudgetPolicyUpdate,
    BudgetCheckRequest,
    BudgetCheckResponse,
    BudgetPolicyCreated,
    BudgetPolicyUpdated,
    BudgetPolicyDeleted,
    BudgetError,
    BudgetPolicyNotFoundError,
    BudgetExceededError,
    BudgetValidationError,
)

from modules.budget.models import (
    BudgetPolicy,
    BudgetScope,
    BudgetPeriod,
    LimitAction,
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
    """Admin user with budget admin permissions."""
    return Actor(
        type="user",
        id="admin_user",
        tenant_id="tenant_1",
        permissions={"budget:admin", "budget:write", "budget:read"},
    )


@pytest.fixture
def tenant_user() -> Actor:
    """Regular user with write permissions in their tenant."""
    return Actor(
        type="user",
        id="user_1",
        tenant_id="tenant_1",
        permissions={"budget:write", "budget:read"},
    )


@pytest.fixture
def readonly_user() -> Actor:
    """Regular user with only read permissions."""
    return Actor(
        type="user",
        id="reader_1",
        tenant_id="tenant_1",
        permissions={"budget:read"},
    )


@pytest.fixture
def other_tenant_user() -> Actor:
    """User in a different tenant."""
    return Actor(
        type="user",
        id="user_2",
        tenant_id="tenant_2",
        permissions={"budget:write", "budget:read"},
    )


@pytest.fixture
def sample_policy() -> BudgetPolicy:
    """Sample budget policy for tests."""
    return BudgetPolicy(
        id="policy_1",
        name="Monthly Team Budget",
        scope=BudgetScope.TEAM,
        scope_id="tenant_1",
        limit_amount=Decimal("500.00"),
        period=BudgetPeriod.MONTHLY,
        soft_limit_percent=0.80,
        hard_limit_action=LimitAction.WARN,
    )


@pytest.fixture
def global_policy() -> BudgetPolicy:
    """Global budget policy for tests."""
    return BudgetPolicy(
        id="policy_global",
        name="Global Budget",
        scope=BudgetScope.GLOBAL,
        scope_id=None,
        limit_amount=Decimal("10000.00"),
        period=BudgetPeriod.MONTHLY,
    )


@pytest.fixture
def mock_repository() -> AsyncMock:
    """Mock budget repository."""
    repo = AsyncMock()
    repo.get_policy = AsyncMock(return_value=None)
    repo.save_policy = AsyncMock(side_effect=lambda p: p)
    repo.delete_policy = AsyncMock(return_value=True)
    repo.list_policies = AsyncMock(return_value=[])
    repo.get_current_spend = AsyncMock(return_value=Decimal("0"))
    return repo


@pytest.fixture
def mock_publisher() -> AsyncMock:
    """Mock event publisher."""
    publisher = AsyncMock()
    publisher.publish = AsyncMock()
    return publisher


@pytest.fixture
def service(mock_repository: AsyncMock, mock_publisher: AsyncMock) -> BudgetPolicyService:
    """Budget policy service with mocks."""
    return BudgetPolicyService(
        repository=mock_repository,
        permission_checker=BudgetPermissionChecker(),
        event_publisher=mock_publisher,
    )


# =============================================================================
# Policy CRUD Tests
# =============================================================================


class TestCreatePolicy:
    """Tests for create_policy operation."""
    
    @pytest.mark.asyncio
    async def test_create_policy_success(
        self,
        service: BudgetPolicyService,
        tenant_user: Actor,
        mock_repository: AsyncMock,
    ) -> None:
        """Successfully create a tenant-scoped policy."""
        policy_data = BudgetPolicyCreate(
            name="Test Budget",
            scope=BudgetScope.TEAM,
            scope_id="tenant_1",
            limit_amount=Decimal("100.00"),
        )
        
        result = await service.create_policy(tenant_user, policy_data)
        
        assert result.success
        assert result.data is not None
        assert result.data.name == "Test Budget"
        assert result.data.limit_amount == Decimal("100.00")
        mock_repository.save_policy.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_policy_publishes_event(
        self,
        service: BudgetPolicyService,
        tenant_user: Actor,
        mock_publisher: AsyncMock,
    ) -> None:
        """Creating a policy publishes BudgetPolicyCreated event."""
        policy_data = BudgetPolicyCreate(
            name="Test Budget",
            scope=BudgetScope.TEAM,
            scope_id="tenant_1",
            limit_amount=Decimal("100.00"),
        )
        
        await service.create_policy(tenant_user, policy_data)
        
        mock_publisher.publish.assert_called_once()
        event = mock_publisher.publish.call_args[0][0]
        assert isinstance(event, BudgetPolicyCreated)
        assert event.policy_name == "Test Budget"
    
    @pytest.mark.asyncio
    async def test_create_policy_denied_without_write_permission(
        self,
        service: BudgetPolicyService,
        readonly_user: Actor,
    ) -> None:
        """Users without write permission cannot create policies."""
        policy_data = BudgetPolicyCreate(
            name="Test Budget",
            scope=BudgetScope.TEAM,
            scope_id="tenant_1",
            limit_amount=Decimal("100.00"),
        )
        
        result = await service.create_policy(readonly_user, policy_data)
        
        assert not result.success
        assert result.error_code == "BUDGET_WRITE_DENIED"
    
    @pytest.mark.asyncio
    async def test_create_policy_denied_cross_tenant(
        self,
        service: BudgetPolicyService,
        other_tenant_user: Actor,
    ) -> None:
        """Users cannot create policies in other tenants."""
        policy_data = BudgetPolicyCreate(
            name="Test Budget",
            scope=BudgetScope.TEAM,
            scope_id="tenant_1",  # Different from other_tenant_user's tenant_2
            limit_amount=Decimal("100.00"),
        )
        
        result = await service.create_policy(other_tenant_user, policy_data)
        
        assert not result.success
        assert result.error_code is not None
        assert "DENIED" in result.error_code
    
    @pytest.mark.asyncio
    async def test_create_global_policy_requires_admin(
        self,
        service: BudgetPolicyService,
        tenant_user: Actor,
        admin_actor: Actor,
    ) -> None:
        """Creating global policies requires admin permission."""
        policy_data = BudgetPolicyCreate(
            name="Global Budget",
            scope=BudgetScope.GLOBAL,
            scope_id=None,
            limit_amount=Decimal("10000.00"),
        )
        
        # Regular user fails
        result = await service.create_policy(tenant_user, policy_data)
        assert not result.success
        
        # Admin succeeds
        result = await service.create_policy(admin_actor, policy_data)
        assert result.success
    
    @pytest.mark.asyncio
    async def test_create_policy_validation_empty_name(
        self,
        service: BudgetPolicyService,
        tenant_user: Actor,
    ) -> None:
        """Policy name cannot be empty."""
        policy_data = BudgetPolicyCreate(
            name="",
            scope=BudgetScope.TEAM,
            scope_id="tenant_1",
            limit_amount=Decimal("100.00"),
        )
        
        result = await service.create_policy(tenant_user, policy_data)
        
        assert not result.success
        assert result.error_code == "VALIDATION_ERROR"
        assert result.error is not None
        assert "name" in result.error.lower()
    
    @pytest.mark.asyncio
    async def test_create_policy_validation_negative_limit(
        self,
        service: BudgetPolicyService,
        tenant_user: Actor,
    ) -> None:
        """Limit amount must be positive."""
        policy_data = BudgetPolicyCreate(
            name="Test",
            scope=BudgetScope.TEAM,
            scope_id="tenant_1",
            limit_amount=Decimal("-50.00"),
        )
        
        result = await service.create_policy(tenant_user, policy_data)
        
        assert not result.success
        assert result.error_code == "VALIDATION_ERROR"
    
    @pytest.mark.asyncio
    async def test_create_policy_detects_conflict(
        self,
        service: BudgetPolicyService,
        tenant_user: Actor,
        sample_policy: BudgetPolicy,
        mock_repository: AsyncMock,
    ) -> None:
        """Detect conflicting policies with same scope/period."""
        # Existing policy with same scope
        mock_repository.list_policies.return_value = [sample_policy]
        
        policy_data = BudgetPolicyCreate(
            name="Duplicate Budget",
            scope=BudgetScope.TEAM,
            scope_id="tenant_1",
            limit_amount=Decimal("200.00"),
            period=BudgetPeriod.MONTHLY,  # Same as sample_policy
        )
        
        result = await service.create_policy(tenant_user, policy_data)
        
        assert not result.success
        assert result.error_code == "POLICY_CONFLICT"

    @pytest.mark.asyncio
    async def test_create_policy_exceeds_global_ceiling(
        self,
        service: BudgetPolicyService,
        tenant_user: Actor,
        global_policy: BudgetPolicy,
        mock_repository: AsyncMock,
    ) -> None:
        """Policy limit cannot exceed global budget ceiling."""
        # Set up global policy with $500 limit
        mock_repository.list_policies.side_effect = lambda **kwargs: (
            [global_policy] if kwargs.get("scope") == BudgetScope.GLOBAL else []
        )
        
        # Try to create team policy with $600 (exceeds global $10000)
        # Note: global_policy has limit_amount=10000
        policy_data = BudgetPolicyCreate(
            name="Exceeds Global",
            scope=BudgetScope.TEAM,
            scope_id="tenant_1",
            limit_amount=Decimal("15000.00"),  # Exceeds global ceiling
            period=BudgetPeriod.MONTHLY,
        )
        
        result = await service.create_policy(tenant_user, policy_data)
        
        assert not result.success
        assert result.error_code == "GLOBAL_CEILING_EXCEEDED"
        assert "exceeds global" in result.error.lower()

    @pytest.mark.asyncio
    async def test_create_policy_combined_exceeds_global_ceiling(
        self,
        service: BudgetPolicyService,
        tenant_user: Actor,
        mock_repository: AsyncMock,
    ) -> None:
        """Combined scoped policies cannot exceed global ceiling."""
        global_policy = BudgetPolicy(
            id="global_1",
            name="Global Budget",
            scope=BudgetScope.GLOBAL,
            scope_id=None,
            limit_amount=Decimal("1000.00"),
            period=BudgetPeriod.MONTHLY,
        )
        
        existing_team_policy = BudgetPolicy(
            id="team_existing",
            name="Existing Team Budget",
            scope=BudgetScope.TEAM,
            scope_id="tenant_2",
            limit_amount=Decimal("600.00"),
            period=BudgetPeriod.MONTHLY,
        )
        
        def mock_list(**kwargs):
            if kwargs.get("scope") == BudgetScope.GLOBAL:
                return [global_policy]
            elif kwargs.get("scope") == BudgetScope.TEAM:
                if kwargs.get("scope_id") == "tenant_1":
                    return []  # No conflict
                return [existing_team_policy]
            return []
        
        mock_repository.list_policies.side_effect = mock_list
        
        # Try to create another team policy with $500 (600 + 500 = 1100 > 1000)
        policy_data = BudgetPolicyCreate(
            name="New Team Budget",
            scope=BudgetScope.TEAM,
            scope_id="tenant_1",
            limit_amount=Decimal("500.00"),
            period=BudgetPeriod.MONTHLY,
        )
        
        result = await service.create_policy(tenant_user, policy_data)
        
        assert not result.success
        assert result.error_code == "GLOBAL_CEILING_EXCEEDED"
        assert "combined" in result.error.lower()


class TestGetPolicy:
    """Tests for get_policy operation."""
    
    @pytest.mark.asyncio
    async def test_get_policy_success(
        self,
        service: BudgetPolicyService,
        tenant_user: Actor,
        sample_policy: BudgetPolicy,
        mock_repository: AsyncMock,
    ) -> None:
        """Successfully retrieve a policy."""
        mock_repository.get_policy.return_value = sample_policy
        
        result = await service.get_policy(tenant_user, "policy_1")
        
        assert result.success
        assert result.data is not None
        assert result.data.id == "policy_1"
        assert result.data.name == "Monthly Team Budget"
    
    @pytest.mark.asyncio
    async def test_get_policy_not_found(
        self,
        service: BudgetPolicyService,
        tenant_user: Actor,
        mock_repository: AsyncMock,
    ) -> None:
        """Return error for non-existent policy."""
        mock_repository.get_policy.return_value = None
        
        result = await service.get_policy(tenant_user, "nonexistent")
        
        assert not result.success
        assert result.error_code == "NOT_FOUND"
    
    @pytest.mark.asyncio
    async def test_get_policy_denied_cross_tenant(
        self,
        service: BudgetPolicyService,
        other_tenant_user: Actor,
        sample_policy: BudgetPolicy,
        mock_repository: AsyncMock,
    ) -> None:
        """Users cannot read policies from other tenants."""
        mock_repository.get_policy.return_value = sample_policy  # tenant_1 policy
        
        result = await service.get_policy(other_tenant_user, "policy_1")  # tenant_2 user
        
        assert not result.success
        assert result.error_code is not None
        assert "DENIED" in result.error_code
    
    @pytest.mark.asyncio
    async def test_get_global_policy_allowed(
        self,
        service: BudgetPolicyService,
        tenant_user: Actor,
        global_policy: BudgetPolicy,
        mock_repository: AsyncMock,
    ) -> None:
        """Any user can read global policies."""
        mock_repository.get_policy.return_value = global_policy
        
        result = await service.get_policy(tenant_user, "policy_global")
        
        assert result.success
        assert result.data is not None
        assert result.data.scope == BudgetScope.GLOBAL


class TestUpdatePolicy:
    """Tests for update_policy operation."""
    
    @pytest.mark.asyncio
    async def test_update_policy_success(
        self,
        service: BudgetPolicyService,
        tenant_user: Actor,
        sample_policy: BudgetPolicy,
        mock_repository: AsyncMock,
    ) -> None:
        """Successfully update a policy."""
        mock_repository.get_policy.return_value = sample_policy
        
        update_data = BudgetPolicyUpdate(
            limit_amount=Decimal("750.00"),
        )
        
        result = await service.update_policy(tenant_user, "policy_1", update_data)
        
        assert result.success
        assert result.data is not None
        assert result.data.limit_amount == Decimal("750.00")
    
    @pytest.mark.asyncio
    async def test_update_policy_publishes_event(
        self,
        service: BudgetPolicyService,
        tenant_user: Actor,
        sample_policy: BudgetPolicy,
        mock_repository: AsyncMock,
        mock_publisher: AsyncMock,
    ) -> None:
        """Updating a policy publishes BudgetPolicyUpdated event."""
        mock_repository.get_policy.return_value = sample_policy
        
        update_data = BudgetPolicyUpdate(name="Updated Name")
        await service.update_policy(tenant_user, "policy_1", update_data)
        
        mock_publisher.publish.assert_called_once()
        event = mock_publisher.publish.call_args[0][0]
        assert isinstance(event, BudgetPolicyUpdated)
        assert "name" in event.changed_fields
    
    @pytest.mark.asyncio
    async def test_update_policy_denied_readonly(
        self,
        service: BudgetPolicyService,
        readonly_user: Actor,
        sample_policy: BudgetPolicy,
        mock_repository: AsyncMock,
    ) -> None:
        """Users with only read permission cannot update."""
        mock_repository.get_policy.return_value = sample_policy
        
        update_data = BudgetPolicyUpdate(limit_amount=Decimal("999.00"))
        result = await service.update_policy(readonly_user, "policy_1", update_data)
        
        assert not result.success
        assert result.error_code is not None
        assert "DENIED" in result.error_code
    
    @pytest.mark.asyncio
    async def test_update_global_policy_requires_admin(
        self,
        service: BudgetPolicyService,
        tenant_user: Actor,
        admin_actor: Actor,
        global_policy: BudgetPolicy,
        mock_repository: AsyncMock,
    ) -> None:
        """Updating global policies requires admin permission."""
        mock_repository.get_policy.return_value = global_policy
        update_data = BudgetPolicyUpdate(limit_amount=Decimal("15000.00"))
        
        # Regular user fails
        result = await service.update_policy(tenant_user, "policy_global", update_data)
        assert not result.success
        
        # Admin succeeds
        result = await service.update_policy(admin_actor, "policy_global", update_data)
        assert result.success


class TestDeletePolicy:
    """Tests for delete_policy operation."""
    
    @pytest.mark.asyncio
    async def test_delete_policy_success(
        self,
        service: BudgetPolicyService,
        tenant_user: Actor,
        sample_policy: BudgetPolicy,
        mock_repository: AsyncMock,
    ) -> None:
        """Successfully delete a policy."""
        mock_repository.get_policy.return_value = sample_policy
        
        result = await service.delete_policy(tenant_user, "policy_1")
        
        assert result.success
        assert result.data is True
        mock_repository.delete_policy.assert_called_once_with("policy_1")
    
    @pytest.mark.asyncio
    async def test_delete_policy_publishes_event(
        self,
        service: BudgetPolicyService,
        tenant_user: Actor,
        sample_policy: BudgetPolicy,
        mock_repository: AsyncMock,
        mock_publisher: AsyncMock,
    ) -> None:
        """Deleting a policy publishes BudgetPolicyDeleted event."""
        mock_repository.get_policy.return_value = sample_policy
        
        await service.delete_policy(tenant_user, "policy_1")
        
        mock_publisher.publish.assert_called_once()
        event = mock_publisher.publish.call_args[0][0]
        assert isinstance(event, BudgetPolicyDeleted)
    
    @pytest.mark.asyncio
    async def test_delete_policy_not_found(
        self,
        service: BudgetPolicyService,
        tenant_user: Actor,
        mock_repository: AsyncMock,
    ) -> None:
        """Cannot delete non-existent policy."""
        mock_repository.get_policy.return_value = None
        
        result = await service.delete_policy(tenant_user, "nonexistent")
        
        assert not result.success
        assert result.error_code == "NOT_FOUND"


class TestListPolicies:
    """Tests for list_policies operation."""
    
    @pytest.mark.asyncio
    async def test_list_policies_success(
        self,
        service: BudgetPolicyService,
        tenant_user: Actor,
        sample_policy: BudgetPolicy,
        global_policy: BudgetPolicy,
        mock_repository: AsyncMock,
    ) -> None:
        """Successfully list policies."""
        mock_repository.list_policies.return_value = [sample_policy, global_policy]
        
        result = await service.list_policies(tenant_user)
        
        assert result.success
        assert result.data is not None
        assert len(result.data) == 2
    
    @pytest.mark.asyncio
    async def test_list_policies_filters_by_tenant(
        self,
        service: BudgetPolicyService,
        tenant_user: Actor,
        sample_policy: BudgetPolicy,
        mock_repository: AsyncMock,
    ) -> None:
        """List only policies accessible to the actor."""
        other_tenant_policy = BudgetPolicy(
            id="policy_other",
            name="Other Tenant Budget",
            scope=BudgetScope.TEAM,
            scope_id="tenant_2",  # Different tenant
            limit_amount=Decimal("500.00"),
        )
        mock_repository.list_policies.return_value = [sample_policy, other_tenant_policy]
        
        result = await service.list_policies(tenant_user)
        
        assert result.success
        # Should only see own tenant's policy
        assert result.data is not None
        assert len(result.data) == 1
        assert result.data[0].scope_id == "tenant_1"


# =============================================================================
# Budget Check Tests
# =============================================================================


class TestCheckBudget:
    """Tests for check_budget operation."""
    
    @pytest.mark.asyncio
    async def test_check_budget_allowed_under_limit(
        self,
        service: BudgetPolicyService,
        tenant_user: Actor,
        sample_policy: BudgetPolicy,
        mock_repository: AsyncMock,
    ) -> None:
        """Budget check passes when under limit."""
        mock_repository.list_policies.return_value = [sample_policy]
        mock_repository.get_current_spend.return_value = Decimal("100.00")
        
        request = BudgetCheckRequest(
            provider="openai",
            model="gpt-4o",
            estimated_cost=Decimal("5.00"),
            team_id="tenant_1",
        )
        
        result = await service.check_budget(tenant_user, request)
        
        assert result.success
        assert result.data is not None
        assert result.data.allowed is True
    
    @pytest.mark.asyncio
    async def test_check_budget_blocked_over_limit(
        self,
        service: BudgetPolicyService,
        tenant_user: Actor,
        mock_repository: AsyncMock,
    ) -> None:
        """Budget check blocked when over limit with BLOCK action."""
        blocking_policy = BudgetPolicy(
            id="policy_strict",
            name="Strict Budget",
            scope=BudgetScope.TEAM,
            scope_id="tenant_1",
            limit_amount=Decimal("100.00"),
            hard_limit_action=LimitAction.BLOCK,
        )
        mock_repository.list_policies.return_value = [blocking_policy]
        mock_repository.get_current_spend.return_value = Decimal("99.00")
        
        request = BudgetCheckRequest(
            provider="openai",
            model="gpt-4o",
            estimated_cost=Decimal("10.00"),  # Would exceed limit
            team_id="tenant_1",
        )
        
        result = await service.check_budget(tenant_user, request)
        
        assert result.success
        assert result.data is not None
        assert result.data.allowed is False
        assert result.data.action == LimitAction.BLOCK
    
    @pytest.mark.asyncio
    async def test_check_budget_warns_near_limit(
        self,
        service: BudgetPolicyService,
        tenant_user: Actor,
        sample_policy: BudgetPolicy,
        mock_repository: AsyncMock,
    ) -> None:
        """Budget check includes warnings when near soft limit."""
        mock_repository.list_policies.return_value = [sample_policy]
        mock_repository.get_current_spend.return_value = Decimal("420.00")  # 84% of 500
        
        request = BudgetCheckRequest(
            provider="openai",
            model="gpt-4o",
            estimated_cost=Decimal("5.00"),
            team_id="tenant_1",
        )
        
        result = await service.check_budget(tenant_user, request)
        
        assert result.success
        assert result.data is not None
        assert result.data.allowed is True
        assert len(result.data.warnings) > 0
        assert "84" in result.data.warnings[0]  # Should mention percentage
    
    @pytest.mark.asyncio
    async def test_check_budget_disabled_always_allows(
        self,
        mock_repository: AsyncMock,
        mock_publisher: AsyncMock,
        tenant_user: Actor,
    ) -> None:
        """When service is disabled, all checks are allowed."""
        service = BudgetPolicyService(
            repository=mock_repository,
            event_publisher=mock_publisher,
        )
        service._enabled = False
        
        request = BudgetCheckRequest(
            provider="openai",
            model="gpt-4o",
            estimated_cost=Decimal("9999.00"),
        )
        
        result = await service.check_budget(tenant_user, request)
        
        assert result.success
        assert result.data is not None
        assert result.data.allowed is True


# =============================================================================
# Permission Checker Tests
# =============================================================================


class TestBudgetPermissionChecker:
    """Tests for BudgetPermissionChecker."""
    
    @pytest.mark.asyncio
    async def test_system_actor_has_all_access(
        self,
        system_actor: Actor,
        sample_policy: BudgetPolicy,
    ) -> None:
        """System actors can access any policy."""
        checker = BudgetPermissionChecker()
        
        # Should not raise
        await checker.require_policy_read(system_actor, sample_policy)
        await checker.require_policy_write(system_actor, sample_policy)
    
    @pytest.mark.asyncio
    async def test_admin_has_global_access(
        self,
        admin_actor: Actor,
        global_policy: BudgetPolicy,
    ) -> None:
        """Admin can modify global policies."""
        checker = BudgetPermissionChecker()
        
        await checker.require_policy_write(admin_actor, global_policy)
    
    @pytest.mark.asyncio
    async def test_user_cannot_modify_global(
        self,
        tenant_user: Actor,
        global_policy: BudgetPolicy,
    ) -> None:
        """Regular users cannot modify global policies."""
        checker = BudgetPermissionChecker()
        
        with pytest.raises(PermissionDeniedError):
            await checker.require_policy_write(tenant_user, global_policy)
    
    @pytest.mark.asyncio
    async def test_user_can_read_global(
        self,
        tenant_user: Actor,
        global_policy: BudgetPolicy,
    ) -> None:
        """Regular users can read global policies."""
        checker = BudgetPermissionChecker()
        
        # Should not raise
        await checker.require_policy_read(tenant_user, global_policy)
    
    def test_filter_policies_by_tenant(
        self,
        tenant_user: Actor,
        sample_policy: BudgetPolicy,
        global_policy: BudgetPolicy,
    ) -> None:
        """Filter policies to only those accessible by actor."""
        other_tenant_policy = BudgetPolicy(
            id="other",
            name="Other",
            scope=BudgetScope.TEAM,
            scope_id="tenant_2",
            limit_amount=Decimal("100.00"),
        )
        
        checker = BudgetPermissionChecker()
        policies = [sample_policy, global_policy, other_tenant_policy]
        
        filtered = checker.filter_policies_for_actor(tenant_user, policies)
        
        assert len(filtered) == 2
        assert sample_policy in filtered
        assert global_policy in filtered
        assert other_tenant_policy not in filtered


# =============================================================================
# Exception Tests
# =============================================================================


class TestBudgetExceptions:
    """Tests for budget-specific exceptions."""
    
    def test_budget_policy_not_found_error(self) -> None:
        """BudgetPolicyNotFoundError includes policy ID."""
        exc = BudgetPolicyNotFoundError("policy_123")
        
        assert "policy_123" in str(exc)
        assert exc.policy_id == "policy_123"
        assert exc.error_code == "BUDGET_POLICY_NOT_FOUND"
    
    def test_budget_exceeded_error(self) -> None:
        """BudgetExceededError includes spending details."""
        exc = BudgetExceededError(
            policy_id="policy_1",
            policy_name="Monthly Budget",
            current_spend=Decimal("95.00"),
            limit_amount=Decimal("100.00"),
            estimated_cost=Decimal("10.00"),
        )
        
        assert exc.policy_id == "policy_1"
        assert exc.current_spend == Decimal("95.00")
        assert exc.error_code == "BUDGET_EXCEEDED"
        assert "Monthly Budget" in str(exc)
    
    def test_budget_validation_error(self) -> None:
        """BudgetValidationError includes field info."""
        exc = BudgetValidationError(
            "Invalid value",
            field="limit_amount",
        )
        
        assert exc.field == "limit_amount"
        assert exc.error_code == "BUDGET_VALIDATION_ERROR"


# =============================================================================
# Service Lifecycle Tests
# =============================================================================


class TestServiceLifecycle:
    """Tests for service initialization and cleanup."""
    
    @pytest.mark.asyncio
    async def test_initialize(self, service: BudgetPolicyService) -> None:
        """Service initializes successfully."""
        await service.initialize()
        # Should complete without error
    
    @pytest.mark.asyncio
    async def test_health_check(self, service: BudgetPolicyService) -> None:
        """Health check returns status."""
        result = await service.health_check()
        
        assert result.success
        assert result.data is not None
        assert result.data["status"] == "healthy"
    
    @pytest.mark.asyncio
    async def test_cleanup(self, service: BudgetPolicyService) -> None:
        """Service cleanup completes successfully."""
        await service.cleanup()
        # Should complete without error
