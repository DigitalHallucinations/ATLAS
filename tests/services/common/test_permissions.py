"""
Tests for permission checking system.

Tests the PermissionChecker class and related permission validation
logic to ensure proper access control enforcement.

Author: ATLAS Team
Date: Jan 7, 2026
"""

import pytest
from unittest.mock import AsyncMock

from core.services.common.exceptions import PermissionDeniedError
from core.services.common.permissions import (
    InMemoryPermissionProvider,
    PermissionChecker,
    PermissionProvider,
)
from core.services.common.types import Actor


class TestInMemoryPermissionProvider:
    """Test the in-memory permission provider."""
    
    @pytest.fixture
    def provider(self):
        """Permission provider instance."""
        return InMemoryPermissionProvider()
    
    @pytest.fixture
    def user_actor(self):
        """Standard user actor."""
        return Actor(
            type="user",
            id="user_123",
            tenant_id="org_456",
            permissions={"conversations:read", "conversations:write"}
        )
    
    @pytest.fixture
    def admin_actor(self):
        """Admin actor with admin permission."""
        return Actor(
            type="user",
            id="admin_123",
            tenant_id="org_456",
            permissions={"conversations:admin"}
        )
    
    @pytest.fixture  
    def system_actor(self):
        """System actor with wildcard permissions."""
        return Actor(
            type="system",
            id="atlas_system",
            tenant_id="system",
            permissions={"*"}
        )
    
    async def test_get_permissions(self, provider, user_actor):
        """Test getting permissions for an actor."""
        permissions = await provider.get_permissions(user_actor)
        assert permissions == {"conversations:read", "conversations:write"}
    
    async def test_has_permission_direct(self, provider, user_actor):
        """Test direct permission checking."""
        assert await provider.has_permission(user_actor, "conversations:read")
        assert await provider.has_permission(user_actor, "conversations:write") 
        assert not await provider.has_permission(user_actor, "admin:delete")
    
    async def test_has_permission_wildcard(self, provider, system_actor):
        """Test wildcard permission checking."""
        assert await provider.has_permission(system_actor, "anything")
        assert await provider.has_permission(system_actor, "admin:delete")
        assert await provider.has_permission(system_actor, "conversations:read")
    
    async def test_has_permission_system_actor(self, provider):
        """Test that system actors always have permission."""
        system_actor = Actor(
            type="system",
            id="system",
            tenant_id="test",
            permissions={"limited:permission"}  # Limited permissions
        )
        
        # System actors should have all permissions regardless
        assert await provider.has_permission(system_actor, "anything")
    
    async def test_permission_hierarchy_admin(self, provider, admin_actor):
        """Test that admin permission implies other permissions."""
        # Admin should imply write and read
        assert await provider.has_permission(admin_actor, "conversations:admin")
        assert await provider.has_permission(admin_actor, "conversations:write")
        assert await provider.has_permission(admin_actor, "conversations:read")
    
    async def test_permission_hierarchy_write(self, provider):
        """Test that write permission implies read permission."""
        write_actor = Actor(
            type="user",
            id="user_write",
            tenant_id="org",
            permissions={"conversations:write"}
        )
        
        assert await provider.has_permission(write_actor, "conversations:write")
        assert await provider.has_permission(write_actor, "conversations:read")
        assert not await provider.has_permission(write_actor, "conversations:admin")
    
    async def test_permission_hierarchy_different_domain(self, provider, admin_actor):
        """Test that admin in one domain doesn't grant access to other domains."""
        # Has conversations:admin but not tasks:admin
        assert await provider.has_permission(admin_actor, "conversations:admin")
        assert not await provider.has_permission(admin_actor, "tasks:admin")
        assert not await provider.has_permission(admin_actor, "tasks:write")


class TestPermissionChecker:
    """Test the PermissionChecker class."""
    
    @pytest.fixture
    def mock_provider(self):
        """Mock permission provider."""
        return AsyncMock(spec=PermissionProvider)
    
    @pytest.fixture
    def checker(self, mock_provider):
        """Permission checker with mock provider.""" 
        return PermissionChecker(mock_provider)
    
    @pytest.fixture
    def user_actor(self):
        """Standard user actor."""
        return Actor(
            type="user",
            id="user_123",
            tenant_id="org_456",
            permissions={"conversations:read"}
        )
    
    async def test_require_success(self, checker, mock_provider, user_actor):
        """Test successful permission requirement."""
        mock_provider.has_permission.return_value = True
        
        # Should not raise an exception
        await checker.require(user_actor, "conversations:read")
        
        mock_provider.has_permission.assert_called_once_with(
            user_actor, 
            "conversations:read"
        )
    
    async def test_require_failure(self, checker, mock_provider, user_actor):
        """Test permission requirement failure."""
        mock_provider.has_permission.return_value = False
        
        with pytest.raises(PermissionDeniedError) as exc_info:
            await checker.require(user_actor, "admin:delete")
        
        error = exc_info.value
        assert error.error_code == "INSUFFICIENT_PERMISSIONS"
        assert "admin:delete" in error.message
        assert error.details["required_permission"] == "admin:delete"
        assert error.details["actor_type"] == "user"
    
    async def test_has_permission(self, checker, mock_provider, user_actor):
        """Test has_permission method."""
        mock_provider.has_permission.return_value = True
        
        result = await checker.has_permission(user_actor, "conversations:read")
        
        assert result is True
        mock_provider.has_permission.assert_called_once_with(
            user_actor,
            "conversations:read"
        )
    
    async def test_has_permission_error_handling(self, checker, mock_provider, user_actor):
        """Test error handling in has_permission."""
        mock_provider.has_permission.side_effect = Exception("Database error")
        
        # Should fail closed (return False) on errors
        result = await checker.has_permission(user_actor, "conversations:read")
        assert result is False
    
    async def test_require_any_success(self, checker, mock_provider, user_actor):
        """Test require_any with successful permission."""
        # First permission check fails, second succeeds
        mock_provider.has_permission.side_effect = [False, True]
        
        await checker.require_any(user_actor, ["admin:delete", "conversations:read"])
        
        assert mock_provider.has_permission.call_count == 2
    
    async def test_require_any_failure(self, checker, mock_provider, user_actor):
        """Test require_any when all permissions fail."""
        mock_provider.has_permission.return_value = False
        
        permissions = ["admin:delete", "conversations:admin"]
        
        with pytest.raises(PermissionDeniedError) as exc_info:
            await checker.require_any(user_actor, permissions)
        
        error = exc_info.value
        assert error.error_code == "INSUFFICIENT_PERMISSIONS"
        assert error.details["required_permissions"] == permissions
    
    async def test_require_all_success(self, checker, mock_provider, user_actor):
        """Test require_all with all permissions present."""
        mock_provider.has_permission.return_value = True
        
        permissions = ["conversations:read", "conversations:write"]
        await checker.require_all(user_actor, permissions)
        
        assert mock_provider.has_permission.call_count == len(permissions)
    
    async def test_require_all_partial_failure(self, checker, mock_provider, user_actor):
        """Test require_all when some permissions are missing."""
        # First permission succeeds, second fails
        mock_provider.has_permission.side_effect = [True, False]
        
        permissions = ["conversations:read", "conversations:admin"]
        
        with pytest.raises(PermissionDeniedError) as exc_info:
            await checker.require_all(user_actor, permissions)
        
        error = exc_info.value
        assert error.error_code == "INSUFFICIENT_PERMISSIONS"
        assert error.details["required_permissions"] == permissions
        assert error.details["missing_permissions"] == ["conversations:admin"]
    
    async def test_get_permissions(self, checker, mock_provider, user_actor):
        """Test get_permissions method."""
        expected_permissions = {"conversations:read", "conversations:write"}
        mock_provider.get_permissions.return_value = expected_permissions
        
        permissions = await checker.get_permissions(user_actor)
        
        assert permissions == expected_permissions
        mock_provider.get_permissions.assert_called_once_with(user_actor)
    
    async def test_get_permissions_error_handling(self, checker, mock_provider, user_actor):
        """Test error handling in get_permissions."""
        mock_provider.get_permissions.side_effect = Exception("Database error")
        
        # Should return empty set on errors
        permissions = await checker.get_permissions(user_actor)
        assert permissions == set()
    
    async def test_default_provider(self):
        """Test that PermissionChecker uses InMemoryPermissionProvider by default."""
        checker = PermissionChecker()
        assert isinstance(checker._provider, InMemoryPermissionProvider)


class TestPermissionCheckerIntegration:
    """Integration tests using real InMemoryPermissionProvider."""
    
    @pytest.fixture
    def checker(self):
        """Permission checker with real provider."""
        return PermissionChecker(InMemoryPermissionProvider())
    
    @pytest.fixture
    def user_actor(self):
        """User actor with basic permissions."""
        return Actor(
            type="user",
            id="user_123",
            tenant_id="org_456",
            permissions={"conversations:read", "conversations:write"}
        )
    
    @pytest.fixture
    def admin_actor(self):
        """Admin actor."""
        return Actor(
            type="user",
            id="admin_123", 
            tenant_id="org_456",
            permissions={"conversations:admin"}
        )
    
    async def test_integration_basic_permissions(self, checker, user_actor):
        """Test basic permission checking end-to-end."""
        # Should succeed
        await checker.require(user_actor, "conversations:read")
        await checker.require(user_actor, "conversations:write")
        
        # Should fail
        with pytest.raises(PermissionDeniedError):
            await checker.require(user_actor, "admin:delete")
    
    async def test_integration_permission_hierarchy(self, checker, admin_actor):
        """Test permission hierarchy integration."""
        # Admin should have all conversation permissions
        await checker.require(admin_actor, "conversations:admin")
        await checker.require(admin_actor, "conversations:write") 
        await checker.require(admin_actor, "conversations:read")
    
    async def test_integration_require_any(self, checker, user_actor):
        """Test require_any integration."""
        # Should succeed (has conversations:read)
        await checker.require_any(user_actor, ["admin:delete", "conversations:read"])
        
        # Should fail (has neither)
        with pytest.raises(PermissionDeniedError):
            await checker.require_any(user_actor, ["admin:delete", "tasks:admin"])
    
    async def test_integration_require_all(self, checker, user_actor):
        """Test require_all integration."""
        # Should succeed (has both)
        await checker.require_all(user_actor, ["conversations:read", "conversations:write"])
        
        # Should fail (missing admin)
        with pytest.raises(PermissionDeniedError):
            await checker.require_all(user_actor, ["conversations:read", "conversations:admin"])