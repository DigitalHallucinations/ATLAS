"""
Protocol definitions for ATLAS service patterns.

Defines standard interfaces that services and repositories should implement
to ensure consistency and enable dependency injection for testing.

Author: ATLAS Team
Date: Jan 7, 2026
"""

from typing import Any, Generic, List, Optional, Protocol, TypeVar, runtime_checkable
from uuid import UUID

from .types import Actor, DomainEvent, OperationResult


# Type variables for generic protocols
T = TypeVar('T')  # Entity type
K = TypeVar('K')  # Key type (usually UUID or str)


@runtime_checkable
class Repository(Protocol, Generic[T, K]):
    """
    Standard repository interface for data access.
    
    All repositories should implement this protocol to ensure
    consistent data access patterns across services.
    
    Type parameters:
        T: Entity type (e.g., Conversation, User)
        K: Key type (e.g., UUID, str)
        
    Example:
        class ConversationRepository(Repository[Conversation, UUID]):
            async def get(self, key: UUID) -> Conversation | None:
                return await self._db.get_conversation(key)
    """
    
    async def get(self, key: K) -> T | None:
        """
        Get entity by key.
        
        Args:
            key: Entity identifier
            
        Returns:
            Entity if found, None if not found
        """
        ...
    
    async def save(self, entity: T) -> T:
        """
        Save entity (create or update).
        
        Args:
            entity: Entity to save
            
        Returns:
            Saved entity (may include generated fields like ID, timestamps)
        """
        ...
    
    async def delete(self, key: K) -> bool:
        """
        Delete entity by key.
        
        Args:
            key: Entity identifier
            
        Returns:
            True if deleted, False if not found
        """
        ...
    
    async def list(
        self,
        *,
        tenant_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[T]:
        """
        List entities with pagination.
        
        Args:
            tenant_id: Optional tenant filter
            limit: Maximum number of entities to return
            offset: Number of entities to skip
            
        Returns:
            List of entities
        """
        ...
    
    async def count(self, *, tenant_id: str | None = None) -> int:
        """
        Count entities.
        
        Args:
            tenant_id: Optional tenant filter
            
        Returns:
            Number of entities
        """
        ...


@runtime_checkable
class SearchableRepository(Repository[T, K], Protocol):
    """
    Repository that supports search operations.
    
    Extends basic Repository with search capabilities.
    """
    
    async def search(
        self,
        query: str,
        *,
        tenant_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[T]:
        """
        Search entities by text query.
        
        Args:
            query: Search query string
            tenant_id: Optional tenant filter
            limit: Maximum results to return
            offset: Number of results to skip
            
        Returns:
            List of matching entities
        """
        ...
    
    async def find_by_criteria(
        self,
        criteria: dict[str, Any],
        *,
        tenant_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[T]:
        """
        Find entities matching criteria.
        
        Args:
            criteria: Search criteria as key-value pairs
            tenant_id: Optional tenant filter
            limit: Maximum results to return
            offset: Number of results to skip
            
        Returns:
            List of matching entities
        """
        ...


@runtime_checkable
class Service(Protocol):
    """
    Base service interface.
    
    All services should implement this protocol to ensure
    consistent service lifecycle and behavior.
    """
    
    async def initialize(self) -> None:
        """
        Initialize the service.
        
        Called during service startup to set up connections,
        validate configuration, etc.
        """
        ...
    
    async def health_check(self) -> OperationResult[dict[str, Any]]:
        """
        Check service health.
        
        Returns:
            Result containing health status information
        """
        ...
    
    async def cleanup(self) -> None:
        """
        Clean up service resources.
        
        Called during service shutdown to close connections,
        flush buffers, etc.
        """
        ...


@runtime_checkable  
class EventPublisher(Protocol):
    """
    Protocol for publishing domain events.
    
    Services should use this to publish events to the message bus
    without coupling to specific messaging implementations.
    """
    
    async def publish(self, event: DomainEvent) -> None:
        """
        Publish a domain event.
        
        Args:
            event: Event to publish
        """
        ...
    
    async def publish_many(self, events: List[DomainEvent]) -> None:
        """
        Publish multiple domain events.
        
        Args:
            events: List of events to publish
        """
        ...


@runtime_checkable
class EventSubscriber(Protocol):
    """
    Protocol for subscribing to domain events.
    
    Services that need to react to events should implement this.
    """
    
    async def handle_event(self, event: DomainEvent) -> None:
        """
        Handle a domain event.
        
        Args:
            event: Event to handle
        """
        ...
    
    def get_subscribed_event_types(self) -> List[str]:
        """
        Get list of event types this subscriber handles.
        
        Returns:
            List of event type strings
        """
        ...


@runtime_checkable
class AuditLogger(Protocol):
    """
    Protocol for audit logging service operations.
    
    Used to track who did what when for security and compliance.
    """
    
    async def log_action(
        self,
        actor: Actor,
        action: str,
        resource_type: str,
        resource_id: str | UUID | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """
        Log an action performed by an actor.
        
        Args:
            actor: Who performed the action
            action: What action was performed
            resource_type: Type of resource affected
            resource_id: ID of specific resource (optional)
            details: Additional context about the action
        """
        ...


@runtime_checkable
class TenantService(Protocol):
    """
    Protocol for services that support multi-tenancy.
    
    Services implementing this protocol can be safely used
    in multi-tenant environments.
    """
    
    async def validate_tenant_access(
        self,
        actor: Actor,
        tenant_id: str,
    ) -> bool:
        """
        Validate that actor has access to tenant.
        
        Args:
            actor: Actor attempting access
            tenant_id: Tenant being accessed
            
        Returns:
            True if access is allowed
        """
        ...
    
    async def list_accessible_tenants(self, actor: Actor) -> List[str]:
        """
        List tenants accessible to actor.
        
        Args:
            actor: Actor to check
            
        Returns:
            List of tenant IDs actor can access
        """
        ...


@runtime_checkable
class CacheableService(Protocol):
    """
    Protocol for services that support caching.
    """
    
    async def invalidate_cache(self, key: str) -> None:
        """
        Invalidate cached data for key.
        
        Args:
            key: Cache key to invalidate
        """
        ...
    
    async def warm_cache(self) -> None:
        """
        Pre-populate cache with commonly accessed data.
        """
        ...


@runtime_checkable
class BackgroundTaskService(Protocol):
    """
    Protocol for services that support background task scheduling.
    """
    
    async def schedule_task(
        self,
        task_name: str,
        payload: dict[str, Any],
        *,
        delay: float = 0,
        retry_count: int = 3,
    ) -> str:
        """
        Schedule a background task.
        
        Args:
            task_name: Name/type of task to execute
            payload: Task parameters
            delay: Delay before execution (seconds)
            retry_count: Number of retries if task fails
            
        Returns:
            Task ID for tracking
        """
        ...
    
    async def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a scheduled task.
        
        Args:
            task_id: ID of task to cancel
            
        Returns:
            True if cancelled, False if not found or already executed
        """
        ...


@runtime_checkable
class MetricsService(Protocol):
    """
    Protocol for services that emit metrics.
    """
    
    async def increment_counter(
        self,
        metric_name: str,
        value: int = 1,
        tags: dict[str, str] | None = None,
    ) -> None:
        """
        Increment a counter metric.
        
        Args:
            metric_name: Name of the metric
            value: Amount to increment by
            tags: Optional tags for the metric
        """
        ...
    
    async def record_histogram(
        self,
        metric_name: str,
        value: float,
        tags: dict[str, str] | None = None,
    ) -> None:
        """
        Record a value in a histogram.
        
        Args:
            metric_name: Name of the metric
            value: Value to record
            tags: Optional tags for the metric
        """
        ...