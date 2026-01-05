"""Execution context for request-scoped state propagation.

This module provides a lightweight execution context that carries essential
request metadata (tenant_id, user_id, conversation_id, trace_id) across
async boundaries without explicit parameter passing.

Usage:
    # Establish context at request boundary
    with execution_context(tenant_id="acme", user_id="user123"):
        await handle_request()
    
    # Access context anywhere in the call stack
    ctx = get_current_context()
    print(ctx.tenant_id)  # "acme"
    
    # Require context (raises if not set)
    @require_context
    async def must_have_context():
        ctx = get_current_context()
        ...
"""

from __future__ import annotations

import uuid
from contextvars import ContextVar, Token
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from functools import wraps
from typing import Any, Callable, Iterator, Optional, Sequence, TypeVar, TYPE_CHECKING

if TYPE_CHECKING:
    from modules.Server.conversation_routes import RequestContext as ServerRequestContext

# Context variable for async-safe propagation
_current_context: ContextVar[Optional["ExecutionContext"]] = ContextVar(
    "atlas_execution_context", default=None
)


def _generate_trace_id() -> str:
    """Generate a unique trace ID for request tracking."""
    return str(uuid.uuid4())


@dataclass(frozen=True, slots=True)
class ExecutionContext:
    """Immutable request-scoped execution context.
    
    Carries essential metadata for multi-tenant isolation, user tracking,
    conversation association, and distributed tracing.
    
    Attributes:
        tenant_id: Tenant identifier for multi-tenant isolation. Defaults to "default".
        user_id: Optional user identifier for the active user.
        conversation_id: Optional conversation identifier for chat context.
        trace_id: Unique trace ID for request tracking and logging.
        roles: Tuple of role identifiers for authorization.
        metadata: Optional additional context metadata.
    """
    
    tenant_id: str = "default"
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None
    trace_id: str = field(default_factory=_generate_trace_id)
    roles: tuple[str, ...] = field(default_factory=tuple)
    metadata: Optional[dict[str, Any]] = None
    
    def with_conversation(self, conversation_id: str) -> ExecutionContext:
        """Return a new context with the specified conversation ID."""
        return replace(self, conversation_id=conversation_id)
    
    def with_user(self, user_id: str) -> ExecutionContext:
        """Return a new context with the specified user ID."""
        return replace(self, user_id=user_id)
    
    def with_roles(self, roles: tuple[str, ...] | list[str]) -> ExecutionContext:
        """Return a new context with the specified roles.
        
        Args:
            roles: A tuple or list of role strings.
            
        Returns:
            A new ExecutionContext with the specified roles.
        """
        return replace(self, roles=tuple(roles))
    
    def with_metadata(self, **kwargs: Any) -> ExecutionContext:
        """Return a new context with additional metadata merged in."""
        existing = dict(self.metadata) if self.metadata else {}
        existing.update(kwargs)
        return replace(self, metadata=existing)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert context to a dictionary for serialization or logging."""
        result: dict[str, Any] = {
            "tenant_id": self.tenant_id,
            "trace_id": self.trace_id,
        }
        if self.user_id is not None:
            result["user_id"] = self.user_id
        if self.conversation_id is not None:
            result["conversation_id"] = self.conversation_id
        if self.roles:
            result["roles"] = self.roles
        if self.metadata:
            result["metadata"] = self.metadata
        return result
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExecutionContext:
        """Create an ExecutionContext from a dictionary."""
        roles_data = data.get("roles", ())
        if isinstance(roles_data, (list, tuple)):
            roles = tuple(str(r) for r in roles_data)
        else:
            roles = ()
        
        return cls(
            tenant_id=data.get("tenant_id", "default"),
            user_id=data.get("user_id"),
            conversation_id=data.get("conversation_id"),
            trace_id=data.get("trace_id") or _generate_trace_id(),
            roles=roles,
            metadata=data.get("metadata"),
        )
    
    def to_server_context(self) -> "ServerRequestContext":
        """Convert to a server RequestContext for route calls.
        
        Returns:
            A RequestContext instance for use with AtlasServer routes.
        """
        from modules.Server.conversation_routes import RequestContext as ServerRequestContext
        
        return ServerRequestContext(
            tenant_id=self.tenant_id,
            user_id=self.user_id,
            session_id=self.trace_id,
            roles=self.roles,
            metadata=self.metadata,
        )
    
    @classmethod
    def from_server_context(
        cls,
        ctx: "ServerRequestContext",
        *,
        conversation_id: Optional[str] = None,
    ) -> ExecutionContext:
        """Create an ExecutionContext from a server RequestContext.
        
        Args:
            ctx: The server RequestContext to convert.
            conversation_id: Optional conversation ID to include.
            
        Returns:
            An ExecutionContext instance.
        """
        # Extract roles from ctx if available
        roles: tuple[str, ...] = ()
        if hasattr(ctx, "roles") and ctx.roles:
            if isinstance(ctx.roles, (list, tuple)):
                roles = tuple(str(r) for r in ctx.roles)
        
        return cls(
            tenant_id=ctx.tenant_id,
            user_id=ctx.user_id,
            conversation_id=conversation_id,
            trace_id=ctx.session_id or _generate_trace_id(),
            roles=roles,
            metadata=dict(ctx.metadata) if ctx.metadata else None,
        )


# ---------------------------------------------------------------------------
# Context access functions
# ---------------------------------------------------------------------------

def get_current_context() -> Optional[ExecutionContext]:
    """Get the current execution context, or None if not set.
    
    Returns:
        The current ExecutionContext, or None if no context is active.
    """
    return _current_context.get()


def get_context_or_default() -> ExecutionContext:
    """Get the current execution context, or a default context if not set.
    
    Returns:
        The current ExecutionContext, or a new default context.
    """
    ctx = _current_context.get()
    if ctx is None:
        ctx = ExecutionContext()
    return ctx


def set_current_context(ctx: ExecutionContext) -> Token[Optional[ExecutionContext]]:
    """Set the current execution context.
    
    Args:
        ctx: The ExecutionContext to set as current.
        
    Returns:
        A token that can be used to reset the context.
    """
    return _current_context.set(ctx)


def clear_current_context() -> None:
    """Clear the current execution context."""
    _current_context.set(None)


# ---------------------------------------------------------------------------
# Context managers and decorators
# ---------------------------------------------------------------------------

@contextmanager
def execution_context(
    tenant_id: str = "default",
    user_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
    *,
    inherit: bool = True,
) -> Iterator[ExecutionContext]:
    """Context manager for establishing an execution context.
    
    Automatically propagates across async boundaries within the context block.
    
    Args:
        tenant_id: Tenant identifier for multi-tenant isolation.
        user_id: Optional user identifier.
        conversation_id: Optional conversation identifier.
        trace_id: Optional trace ID (generated if not provided).
        metadata: Optional additional metadata.
        inherit: If True and a parent context exists, inherit unspecified values.
        
    Yields:
        The established ExecutionContext.
        
    Example:
        with execution_context(tenant_id="acme", user_id="user123") as ctx:
            # ctx is now available throughout this block and all async calls
            await process_request()
    """
    parent = _current_context.get() if inherit else None
    
    # Build context, inheriting from parent if requested
    if parent is not None and inherit:
        ctx = ExecutionContext(
            tenant_id=tenant_id if tenant_id != "default" else parent.tenant_id,
            user_id=user_id if user_id is not None else parent.user_id,
            conversation_id=conversation_id if conversation_id is not None else parent.conversation_id,
            trace_id=trace_id if trace_id is not None else parent.trace_id,
            metadata={**(parent.metadata or {}), **(metadata or {})} if (parent.metadata or metadata) else None,
        )
    else:
        ctx = ExecutionContext(
            tenant_id=tenant_id,
            user_id=user_id,
            conversation_id=conversation_id,
            trace_id=trace_id or _generate_trace_id(),
            metadata=metadata,
        )
    
    token = _current_context.set(ctx)
    try:
        yield ctx
    finally:
        _current_context.reset(token)


class ContextNotSetError(RuntimeError):
    """Raised when execution context is required but not set."""
    
    def __init__(self, message: str = "Execution context is not set"):
        super().__init__(message)


F = TypeVar("F", bound=Callable[..., Any])


def require_context(func: F) -> F:
    """Decorator that requires an execution context to be set.
    
    Raises ContextNotSetError if the context is not established.
    
    Example:
        @require_context
        async def tenant_aware_operation():
            ctx = get_current_context()
            # ctx is guaranteed to be set here
    """
    @wraps(func)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        if _current_context.get() is None:
            raise ContextNotSetError(
                f"Execution context required for {func.__qualname__}"
            )
        return func(*args, **kwargs)
    
    @wraps(func)
    async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
        if _current_context.get() is None:
            raise ContextNotSetError(
                f"Execution context required for {func.__qualname__}"
            )
        return await func(*args, **kwargs)
    
    # Return appropriate wrapper based on function type
    import asyncio
    if asyncio.iscoroutinefunction(func):
        return async_wrapper  # type: ignore[return-value]
    return sync_wrapper  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Legacy compatibility helpers
# ---------------------------------------------------------------------------

def context_to_legacy_dict() -> dict[str, Any]:
    """Convert current context to legacy {"tenant_id": ...} format.
    
    Provides backwards compatibility for code expecting the old dict format.
    
    Returns:
        A dictionary with tenant_id key, or {"tenant_id": "default"} if no context.
    """
    ctx = _current_context.get()
    if ctx is None:
        return {"tenant_id": "default"}
    return {"tenant_id": ctx.tenant_id}


def context_from_legacy_dict(data: dict[str, Any]) -> ExecutionContext:
    """Create ExecutionContext from legacy context dictionary.
    
    Args:
        data: Dictionary potentially containing tenant_id, user, conversation_id, roles.
        
    Returns:
        An ExecutionContext populated from the dictionary.
    """
    roles_data = data.get("roles", ())
    if isinstance(roles_data, (list, tuple)):
        roles = tuple(str(r) for r in roles_data)
    else:
        roles = ()
    
    return ExecutionContext(
        tenant_id=data.get("tenant_id", "default"),
        user_id=data.get("user") or data.get("user_id"),
        conversation_id=data.get("conversation_id"),
        trace_id=data.get("trace_id") or _generate_trace_id(),
        roles=roles,
        metadata=data.get("metadata"),
    )
