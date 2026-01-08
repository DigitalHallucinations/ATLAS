"""
Common service types for ATLAS.

Provides standard result types, domain events, and actor definitions
that all ATLAS services should use for consistency and type safety.

Author: ATLAS Team
Date: Jan 7, 2026
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Generic, Literal, TypeVar
from uuid import UUID

# Type variable for generic result types
T = TypeVar('T')


@dataclass
class OperationResult(Generic[T]):
    """
    Standard result wrapper for service operations.
    
    Provides consistent success/failure handling across all services,
    with optional error details for debugging and user feedback.
    
    Attributes:
        success: Whether the operation completed successfully
        data: The result data if successful, None if failed
        error: Human-readable error message if failed
        error_code: Machine-readable error code for specific error types
        
    Example:
        # Success case
        result = OperationResult.create_success({"user_id": "123"})
        
        # Failure case
        result = OperationResult.create_failure("User not found", "USER_NOT_FOUND")
        
        # Usage
        if result.success:
            process_data(result.data)
        else:
            logger.error(f"Operation failed: {result.error}")
    """
    success: bool
    data: T | None = None
    error: str | None = None
    error_code: str | None = None

    @classmethod
    def success(cls, data: T) -> "OperationResult[T]":
        """Create a successful result with data."""
        return cls(success=True, data=data)
    
    @classmethod
    def failure(cls, error: str, error_code: str | None = None) -> "OperationResult[T]":
        """Create a failed result with error details."""
        return cls(success=False, error=error, error_code=error_code)
    
    @property
    def value(self) -> T | None:
        """Alias for data property for compatibility."""
        return self.data
    
    @property
    def is_success(self) -> bool:
        """Check if the operation was successful."""
        return self.success
    
    @property
    def is_failure(self) -> bool:
        """Check if the operation failed."""
        return not self.success
    
    def unwrap(self) -> T:
        """
        Get the data from a successful result.
        
        Raises:
            RuntimeError: If the result represents a failure
            
        Returns:
            The wrapped data
        """
        if not self.success:
            raise RuntimeError(f"Cannot unwrap failed result: {self.error}")
        assert self.data is not None, "Data should not be None for successful results"
        return self.data
    
    def unwrap_or(self, default: T) -> T:
        """
        Get the data from the result, or return default if failed.
        
        Args:
            default: Value to return if the operation failed
            
        Returns:
            The wrapped data or the default value
        """
        return self.data if self.success and self.data is not None else default

    @property
    def value(self) -> T | None:
        """Alias for the wrapped data to mirror legacy call sites."""
        return self.data


@dataclass
class Actor:
    """
    Represents an entity performing an action in the system.
    
    Used for permission checking, audit logging, and context tracking.
    All service operations should include an Actor to maintain security
    and traceability.
    
    Attributes:
        type: The kind of actor (user, system, agent, etc.)
        id: Unique identifier for this actor
        tenant_id: Tenant/organization this actor belongs to
        permissions: Set of permissions this actor has been granted
        
    Example:
        # User actor
        user = Actor(
            type="user",
            id="user_123",
            tenant_id="org_456", 
            permissions={"conversations:read", "conversations:write"}
        )
        
        # System actor (for automated operations)
        system = Actor(
            type="system",
            id="atlas_system",
            tenant_id="system",
            permissions={"*"}  # Full permissions
        )
    """
    type: Literal["user", "system", "agent", "sync", "job", "task"]
    id: str
    tenant_id: str
    permissions: set[str] | list[str]
    
    def __post_init__(self) -> None:
        """Normalize permissions collection after initialization."""
        perms = self.permissions
        if not isinstance(perms, set):
            perms = set(perms)
        # Normalize dotted permission notation to the colon style used internally
        self.permissions = {perm.replace(".", ":") for perm in perms}
    
    def has_permission(self, permission: str) -> bool:
        """Check if this actor has a specific permission."""
        return permission in self.permissions or "*" in self.permissions
    
    def is_system(self) -> bool:
        """Check if this is a system actor."""
        return self.type == "system"
    
    def is_user(self) -> bool:
        """Check if this is a user actor."""
        return self.type == "user"

    @property
    def user_id(self) -> str:
        """Provide an alias used by calendar code paths."""
        return self.id


@dataclass(frozen=True)
class DomainEvent:
    """
    Base class for domain events in the ATLAS event system.
    
    Domain events represent something important that happened in a
    specific business domain. They are published to the message bus
    to enable loose coupling between services.
    
    All events are immutable (frozen) to ensure they cannot be
    modified after creation, maintaining event integrity.
    
    Attributes:
        event_type: Dot-separated event type (e.g., "conversation.created")
        entity_id: ID of the primary entity this event concerns
        tenant_id: Tenant/organization context for this event
        timestamp: When this event occurred
        actor: Who or what caused this event to happen
        metadata: Additional event-specific data
        
    Example:
        # Conversation created event
        event = DomainEvent(
            event_type="conversation.created",
            entity_id=UUID("12345678-1234-5678-9012-123456789abc"),
            tenant_id="org_456",
            timestamp=datetime.utcnow(),
            actor="user",
            metadata={"title": "New Chat", "model": "gpt-4"}
        )
    """
    event_type: str
    entity_id: UUID
    tenant_id: str
    timestamp: datetime
    actor: Literal["user", "system", "agent", "sync", "job", "task"]
    metadata: dict[str, Any] | None = None
    
    @classmethod
    def create(
        cls,
        event_type: str,
        entity_id: UUID | str,
        tenant_id: str,
        actor: Literal["user", "system", "agent", "sync", "job", "task"],
        metadata: dict[str, Any] | None = None,
        timestamp: datetime | None = None,
    ) -> "DomainEvent":
        """
        Create a new domain event with current timestamp.
        
        Args:
            event_type: Dot-separated event type identifier
            entity_id: ID of the entity (will be converted to UUID if string)
            tenant_id: Tenant context
            actor: Who/what caused this event
            metadata: Optional additional event data
            timestamp: Event timestamp (defaults to now)
            
        Returns:
            New DomainEvent instance
        """
        if isinstance(entity_id, str):
            entity_id = UUID(entity_id)
        
        return cls(
            event_type=event_type,
            entity_id=entity_id,
            tenant_id=tenant_id,
            timestamp=timestamp or datetime.utcnow(),
            actor=actor,
            metadata=metadata or {}
        )
    
    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            "event_type": self.event_type,
            "entity_id": str(self.entity_id),
            "tenant_id": self.tenant_id,
            "timestamp": self.timestamp.isoformat(),
            "actor": self.actor,
            "metadata": self.metadata,
        }
    
    @property
    def event_id(self) -> str:
        """Expose the entity identifier using calendar terminology."""
        return str(self.entity_id)
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DomainEvent":
        """Create event from dictionary (for deserialization)."""
        return cls(
            event_type=data["event_type"],
            entity_id=UUID(data["entity_id"]),
            tenant_id=data["tenant_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            actor=data["actor"],
            metadata=data.get("metadata"),
        )