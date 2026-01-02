"""Tests for ExecutionContext and context propagation."""

import asyncio
import pytest

from ATLAS.context import (
    ExecutionContext,
    get_current_context,
    get_context_or_default,
    set_current_context,
    clear_current_context,
    execution_context,
    require_context,
    ContextNotSetError,
    context_to_legacy_dict,
    context_from_legacy_dict,
)


class TestExecutionContext:
    """Test ExecutionContext dataclass."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        ctx = ExecutionContext()
        assert ctx.tenant_id == "default"
        assert ctx.user_id is None
        assert ctx.conversation_id is None
        assert ctx.trace_id is not None  # Auto-generated
        assert ctx.metadata is None

    def test_with_all_values(self):
        """Test context with all values specified."""
        ctx = ExecutionContext(
            tenant_id="acme",
            user_id="user123",
            conversation_id="conv456",
            trace_id="trace789",
            metadata={"key": "value"},
        )
        assert ctx.tenant_id == "acme"
        assert ctx.user_id == "user123"
        assert ctx.conversation_id == "conv456"
        assert ctx.trace_id == "trace789"
        assert ctx.metadata == {"key": "value"}

    def test_immutability(self):
        """Test that context is immutable (frozen)."""
        ctx = ExecutionContext(tenant_id="test")
        with pytest.raises(AttributeError):
            ctx.tenant_id = "changed"

    def test_with_conversation(self):
        """Test creating new context with conversation ID."""
        ctx = ExecutionContext(tenant_id="acme", user_id="user1")
        new_ctx = ctx.with_conversation("conv123")
        
        assert new_ctx.conversation_id == "conv123"
        assert new_ctx.tenant_id == "acme"
        assert new_ctx.user_id == "user1"
        assert ctx.conversation_id is None  # Original unchanged

    def test_with_user(self):
        """Test creating new context with user ID."""
        ctx = ExecutionContext(tenant_id="acme")
        new_ctx = ctx.with_user("user456")
        
        assert new_ctx.user_id == "user456"
        assert new_ctx.tenant_id == "acme"
        assert ctx.user_id is None  # Original unchanged

    def test_with_metadata(self):
        """Test creating new context with additional metadata."""
        ctx = ExecutionContext(metadata={"existing": "value"})
        new_ctx = ctx.with_metadata(new_key="new_value")
        
        assert new_ctx.metadata == {"existing": "value", "new_key": "new_value"}
        assert ctx.metadata == {"existing": "value"}  # Original unchanged

    def test_with_roles(self):
        """Test creating new context with roles."""
        ctx = ExecutionContext(tenant_id="acme")
        new_ctx = ctx.with_roles(("admin", "user"))
        
        assert new_ctx.roles == ("admin", "user")
        assert new_ctx.tenant_id == "acme"
        assert ctx.roles == ()  # Original unchanged

    def test_with_roles_from_list(self):
        """Test creating new context with roles from a list."""
        ctx = ExecutionContext()
        new_ctx = ctx.with_roles(["system"])
        
        assert new_ctx.roles == ("system",)

    def test_to_dict(self):
        """Test serialization to dictionary."""
        ctx = ExecutionContext(
            tenant_id="acme",
            user_id="user1",
            conversation_id="conv1",
            trace_id="trace1",
        )
        data = ctx.to_dict()
        
        assert data["tenant_id"] == "acme"
        assert data["user_id"] == "user1"
        assert data["conversation_id"] == "conv1"
        assert data["trace_id"] == "trace1"

    def test_to_dict_with_roles(self):
        """Test serialization includes roles."""
        ctx = ExecutionContext(
            tenant_id="acme",
            roles=("admin", "user"),
        )
        data = ctx.to_dict()
        
        assert data["roles"] == ("admin", "user")

    def test_to_dict_omits_empty_roles(self):
        """Test serialization omits empty roles tuple."""
        ctx = ExecutionContext(tenant_id="acme")
        data = ctx.to_dict()
        
        assert "roles" not in data

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "tenant_id": "acme",
            "user_id": "user1",
            "conversation_id": "conv1",
            "trace_id": "trace1",
        }
        ctx = ExecutionContext.from_dict(data)
        
        assert ctx.tenant_id == "acme"
        assert ctx.user_id == "user1"
        assert ctx.conversation_id == "conv1"
        assert ctx.trace_id == "trace1"

    def test_from_dict_with_roles(self):
        """Test deserialization includes roles."""
        data = {
            "tenant_id": "acme",
            "roles": ["admin", "user"],
        }
        ctx = ExecutionContext.from_dict(data)
        
        assert ctx.roles == ("admin", "user")

    def test_from_dict_with_tuple_roles(self):
        """Test deserialization handles tuple roles."""
        data = {
            "tenant_id": "acme",
            "roles": ("system",),
        }
        ctx = ExecutionContext.from_dict(data)
        
        assert ctx.roles == ("system",)


class TestContextAccess:
    """Test context access functions."""

    def test_get_current_context_none(self):
        """Test that get_current_context returns None when not set."""
        clear_current_context()
        assert get_current_context() is None

    def test_get_context_or_default(self):
        """Test that get_context_or_default returns default when not set."""
        clear_current_context()
        ctx = get_context_or_default()
        assert ctx.tenant_id == "default"

    def test_set_and_get_context(self):
        """Test setting and getting context."""
        ctx = ExecutionContext(tenant_id="test")
        token = set_current_context(ctx)
        
        try:
            retrieved = get_current_context()
            assert retrieved is ctx
            assert retrieved.tenant_id == "test"
        finally:
            clear_current_context()

    def test_clear_current_context(self):
        """Test clearing context."""
        set_current_context(ExecutionContext(tenant_id="test"))
        clear_current_context()
        assert get_current_context() is None


class TestExecutionContextManager:
    """Test execution_context context manager."""

    def test_basic_context_manager(self):
        """Test basic context manager usage."""
        clear_current_context()
        
        with execution_context(tenant_id="acme", user_id="user1") as ctx:
            assert ctx.tenant_id == "acme"
            assert ctx.user_id == "user1"
            
            retrieved = get_current_context()
            assert retrieved is ctx
        
        # Context should be cleared after exiting
        assert get_current_context() is None

    def test_nested_context_managers(self):
        """Test nested context managers with inheritance."""
        with execution_context(tenant_id="outer", user_id="outer_user") as outer:
            assert outer.tenant_id == "outer"
            
            with execution_context(conversation_id="inner_conv") as inner:
                # Should inherit from parent
                assert inner.tenant_id == "outer"
                assert inner.user_id == "outer_user"
                assert inner.conversation_id == "inner_conv"
            
            # Should restore outer context
            current = get_current_context()
            assert current is outer
            assert current.conversation_id is None

    def test_context_manager_no_inherit(self):
        """Test context manager without inheritance."""
        with execution_context(tenant_id="parent"):
            with execution_context(tenant_id="child", inherit=False) as ctx:
                assert ctx.tenant_id == "child"
                assert ctx.user_id is None  # Not inherited

    @pytest.mark.asyncio
    async def test_context_propagates_across_await(self):
        """Test that context propagates across async boundaries."""
        async def inner_coro():
            ctx = get_current_context()
            return ctx.tenant_id if ctx else None
        
        with execution_context(tenant_id="async_test"):
            result = await inner_coro()
            assert result == "async_test"


class TestRequireContextDecorator:
    """Test require_context decorator."""

    def test_require_context_raises_when_missing(self):
        """Test that require_context raises when context not set."""
        clear_current_context()
        
        @require_context
        def needs_context():
            return get_current_context()
        
        with pytest.raises(ContextNotSetError):
            needs_context()

    def test_require_context_passes_when_set(self):
        """Test that require_context passes when context is set."""
        @require_context
        def needs_context():
            return get_current_context()
        
        with execution_context(tenant_id="test"):
            ctx = needs_context()
            assert ctx.tenant_id == "test"

    @pytest.mark.asyncio
    async def test_require_context_async(self):
        """Test require_context with async functions."""
        @require_context
        async def async_needs_context():
            return get_current_context()
        
        # Should raise when not set
        clear_current_context()
        with pytest.raises(ContextNotSetError):
            await async_needs_context()
        
        # Should pass when set
        with execution_context(tenant_id="async_test"):
            ctx = await async_needs_context()
            assert ctx.tenant_id == "async_test"


class TestLegacyCompatibility:
    """Test legacy compatibility helpers."""

    def test_context_to_legacy_dict_no_context(self):
        """Test legacy dict conversion when no context set."""
        clear_current_context()
        result = context_to_legacy_dict()
        assert result == {"tenant_id": "default"}

    def test_context_to_legacy_dict_with_context(self):
        """Test legacy dict conversion with context."""
        with execution_context(tenant_id="acme"):
            result = context_to_legacy_dict()
            assert result == {"tenant_id": "acme"}

    def test_context_from_legacy_dict(self):
        """Test creating context from legacy dict."""
        legacy = {
            "tenant_id": "acme",
            "user": "user123",
            "conversation_id": "conv456",
        }
        ctx = context_from_legacy_dict(legacy)
        
        assert ctx.tenant_id == "acme"
        assert ctx.user_id == "user123"
        assert ctx.conversation_id == "conv456"

    def test_context_from_legacy_dict_with_roles(self):
        """Test creating context from legacy dict with roles."""
        legacy = {
            "tenant_id": "acme",
            "user_id": "user123",
            "roles": ["admin", "user"],
            "metadata": {"source": "api"},
        }
        ctx = context_from_legacy_dict(legacy)
        
        assert ctx.tenant_id == "acme"
        assert ctx.user_id == "user123"
        assert ctx.roles == ("admin", "user")
        assert ctx.metadata == {"source": "api"}
