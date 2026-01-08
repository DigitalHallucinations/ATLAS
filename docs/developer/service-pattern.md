# ATLAS Service Pattern

> **Author**: ATLAS Team  
> **Date**: Jan 7, 2026  
> **Status**: Active

This document describes the standard pattern for implementing services in ATLAS. All new services should follow this pattern for consistency, testability, and maintainability.

## Overview

ATLAS services are async-first, permission-aware components that:

1. Use `OperationResult[T]` for consistent return values
2. Require `Actor` for all operations (authorization + audit)
3. Publish `DomainEvent`s for loose coupling
4. Follow standard exception hierarchy
5. Implement common protocols for testability

## Basic Service Structure

```python
# core/services/example/service.py

from core.services.common import (
    Actor,
    DomainEvent,
    OperationResult,
    PermissionChecker,
    Service,
    ServiceError,
    ValidationError,
)
from core.messaging.agent_bus import AgentBus

class ExampleService(Service):
    """
    Example service following ATLAS patterns.
    
    All services should:
    - Accept dependencies through __init__
    - Implement Service protocol
    - Use async methods
    - Return OperationResult[T]
    - Check permissions
    - Publish domain events
    """
    
    def __init__(
        self,
        *,
        repository: ExampleRepository,
        permission_checker: PermissionChecker,
        event_publisher: EventPublisher,
        logger: Logger,
        tenant_id: str,
    ) -> None:
        self._repository = repository
        self._permissions = permission_checker
        self._events = event_publisher
        self._logger = logger
        self._tenant_id = tenant_id
    
    async def initialize(self) -> None:
        """Initialize service (part of Service protocol)."""
        await self._repository.initialize()
        self._logger.info("ExampleService initialized")
    
    async def create_entity(
        self, 
        actor: Actor, 
        data: CreateEntityRequest
    ) -> OperationResult[Entity]:
        """
        Create a new entity.
        
        Standard pattern:
        1. Validate permissions
        2. Validate input
        3. Perform operation
        4. Publish domain event
        5. Return result
        """
        try:
            # 1. Permission check
            await self._permissions.require(actor, "entities:write")
            
            # 2. Input validation
            if not data.name or len(data.name) < 3:
                return OperationResult.failure(
                    "Entity name must be at least 3 characters",
                    "INVALID_NAME"
                )
            
            # 3. Business logic
            entity = Entity(
                id=uuid.uuid4(),
                name=data.name,
                tenant_id=actor.tenant_id,
                created_by=actor.id,
                created_at=datetime.utcnow(),
            )
            
            saved_entity = await self._repository.save(entity)
            
            # 4. Publish domain event
            event = DomainEvent.create(
                event_type="entity.created",
                entity_id=saved_entity.id,
                tenant_id=actor.tenant_id,
                actor=actor.type,
                metadata={"name": saved_entity.name}
            )
            await self._events.publish(event)
            
            # 5. Return result
            self._logger.info(
                f"Entity created: {saved_entity.id}",
                extra={"entity_id": str(saved_entity.id), "actor": actor.id}
            )
            
            return OperationResult.success(saved_entity)
            
        except PermissionDeniedError:
            # Re-raise permission errors as-is
            raise
        except Exception as e:
            # Convert other exceptions to ServiceError
            self._logger.error(f"Failed to create entity: {e}", exc_info=True)
            return OperationResult.failure(
                "Failed to create entity",
                "CREATION_FAILED"
            )
    
    async def get_entity(
        self, 
        actor: Actor, 
        entity_id: UUID
    ) -> OperationResult[Entity]:
        """Get entity by ID."""
        try:
            await self._permissions.require(actor, "entities:read")
            
            entity = await self._repository.get(entity_id)
            if not entity:
                return OperationResult.failure(
                    f"Entity {entity_id} not found",
                    "ENTITY_NOT_FOUND"
                )
            
            # Tenant isolation check
            if entity.tenant_id != actor.tenant_id:
                return OperationResult.failure(
                    "Entity not found",  # Don't leak existence
                    "ENTITY_NOT_FOUND"
                )
            
            return OperationResult.success(entity)
            
        except PermissionDeniedError:
            raise
        except Exception as e:
            self._logger.error(f"Failed to get entity {entity_id}: {e}")
            return OperationResult.failure(
                "Failed to retrieve entity",
                "RETRIEVAL_FAILED"
            )
    
    async def health_check(self) -> OperationResult[dict]:
        """Service health check (part of Service protocol)."""
        try:
            # Check repository health
            await self._repository.count()
            
            return OperationResult.success({
                "status": "healthy",
                "service": "ExampleService",
                "tenant_id": self._tenant_id,
                "timestamp": datetime.utcnow().isoformat(),
            })
        except Exception as e:
            return OperationResult.failure(
                f"Health check failed: {e}",
                "HEALTH_CHECK_FAILED"
            )
    
    async def cleanup(self) -> None:
        """Cleanup resources (part of Service protocol)."""
        await self._repository.cleanup()
        self._logger.info("ExampleService cleaned up")
```

## Repository Pattern

```python
# core/services/example/repository.py

from core.services.common import Repository, SearchableRepository

class ExampleRepository(SearchableRepository[Entity, UUID]):
    """
    Repository implementing standard patterns.
    
    All repositories should:
    - Implement Repository or SearchableRepository protocol
    - Handle tenant isolation
    - Use proper error handling
    - Support pagination
    """
    
    def __init__(self, db_connection: DatabaseConnection) -> None:
        self._db = db_connection
    
    async def get(self, entity_id: UUID) -> Entity | None:
        """Get entity by ID with tenant isolation."""
        query = """
            SELECT * FROM entities 
            WHERE id = $1 AND tenant_id = $2
        """
        row = await self._db.fetchrow(query, entity_id, self._tenant_id)
        return Entity.from_row(row) if row else None
    
    async def save(self, entity: Entity) -> Entity:
        """Save entity (create or update)."""
        if hasattr(entity, 'id') and entity.id:
            return await self._update(entity)
        else:
            return await self._create(entity)
    
    async def delete(self, entity_id: UUID) -> bool:
        """Delete entity by ID."""
        query = """
            DELETE FROM entities 
            WHERE id = $1 AND tenant_id = $2
        """
        result = await self._db.execute(query, entity_id, self._tenant_id)
        return result == "DELETE 1"
    
    async def list(
        self, 
        *, 
        tenant_id: str | None = None,
        limit: int = 100, 
        offset: int = 0
    ) -> List[Entity]:
        """List entities with pagination."""
        query = """
            SELECT * FROM entities 
            WHERE tenant_id = $1
            ORDER BY created_at DESC
            LIMIT $2 OFFSET $3
        """
        rows = await self._db.fetch(query, tenant_id, limit, offset)
        return [Entity.from_row(row) for row in rows]
    
    async def search(
        self,
        query: str,
        *,
        tenant_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Entity]:
        """Search entities by text."""
        sql = """
            SELECT * FROM entities 
            WHERE tenant_id = $1 
            AND (name ILIKE $2 OR description ILIKE $2)
            ORDER BY created_at DESC
            LIMIT $3 OFFSET $4
        """
        rows = await self._db.fetch(sql, tenant_id, f"%{query}%", limit, offset)
        return [Entity.from_row(row) for row in rows]
```

## Domain Types

```python
# core/services/example/types.py

from dataclasses import dataclass
from uuid import UUID
from datetime import datetime

from core.services.common import DomainEvent

@dataclass
class Entity:
    """Domain entity following standard patterns."""
    id: UUID
    name: str
    description: str | None
    tenant_id: str
    created_by: str
    created_at: datetime
    updated_at: datetime | None = None
    
    @classmethod
    def from_row(cls, row: dict) -> "Entity":
        """Create from database row."""
        return cls(**row)

@dataclass
class CreateEntityRequest:
    """Request type for creating entities."""
    name: str
    description: str | None = None

@dataclass(frozen=True)
class EntityCreatedEvent(DomainEvent):
    """Typed domain event for entity creation."""
    pass

@dataclass(frozen=True)
class EntityUpdatedEvent(DomainEvent):
    """Typed domain event for entity updates."""
    pass
```

## Permissions

```python
# core/services/example/permissions.py

from core.services.common import PermissionChecker

class ExamplePermissionChecker(PermissionChecker):
    """
    Domain-specific permission checker.
    
    Extends base PermissionChecker with domain-specific logic.
    """
    
    PERMISSIONS = {
        "entities:read": "View entities",
        "entities:write": "Create/update entities", 
        "entities:delete": "Delete entities",
        "entities:admin": "Full administrative access",
    }
    
    async def can_access_entity(
        self, 
        actor: Actor, 
        entity: Entity
    ) -> bool:
        """Check if actor can access specific entity."""
        # Basic permission check
        if not await self.has_permission(actor, "entities:read"):
            return False
        
        # Tenant isolation
        if entity.tenant_id != actor.tenant_id:
            return False
        
        # Owner can always access
        if entity.created_by == actor.id:
            return True
        
        # Admin can access all in tenant
        if await self.has_permission(actor, "entities:admin"):
            return True
        
        return False
```

## Service Configuration and Wiring

```python
# core/services/example/__init__.py

from .service import ExampleService
from .repository import ExampleRepository
from .permissions import ExamplePermissionChecker
from .types import Entity, CreateEntityRequest

__all__ = [
    "ExampleService",
    "ExampleRepository", 
    "ExamplePermissionChecker",
    "Entity",
    "CreateEntityRequest",
]

# Factory function for service setup
async def create_example_service(
    db_connection: DatabaseConnection,
    agent_bus: AgentBus,
    tenant_id: str,
) -> ExampleService:
    """Create and initialize ExampleService with dependencies."""
    repository = ExampleRepository(db_connection)
    permission_checker = ExamplePermissionChecker()
    event_publisher = create_domain_event_publisher(agent_bus)
    logger = logging.getLogger("atlas.services.example")
    
    service = ExampleService(
        repository=repository,
        permission_checker=permission_checker,
        event_publisher=event_publisher,
        logger=logger,
        tenant_id=tenant_id,
    )
    
    await service.initialize()
    return service
```

## Testing Patterns

```python
# tests/services/example/test_service.py

import pytest
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

from core.services.example import ExampleService, ExampleRepository
from core.services.common import (
    Actor,
    OperationResult,
    PermissionChecker,
    PermissionDeniedError,
)

class TestExampleService:
    
    @pytest.fixture
    def actor(self):
        """Test actor with standard permissions."""
        return Actor(
            type="user",
            id="test_user",
            tenant_id="test_tenant",
            permissions={"entities:read", "entities:write"}
        )
    
    @pytest.fixture
    def mock_repository(self):
        """Mock repository for testing."""
        return AsyncMock(spec=ExampleRepository)
    
    @pytest.fixture
    def mock_permission_checker(self):
        """Mock permission checker."""
        checker = AsyncMock(spec=PermissionChecker)
        checker.require.return_value = None  # Allow by default
        return checker
    
    @pytest.fixture
    def mock_event_publisher(self):
        """Mock event publisher."""
        return AsyncMock()
    
    @pytest.fixture
    def service(
        self, 
        mock_repository, 
        mock_permission_checker,
        mock_event_publisher
    ):
        """Service instance with mocked dependencies."""
        return ExampleService(
            repository=mock_repository,
            permission_checker=mock_permission_checker,
            event_publisher=mock_event_publisher,
            logger=Mock(),
            tenant_id="test_tenant",
        )
    
    async def test_create_entity_success(
        self, 
        service, 
        actor, 
        mock_repository,
        mock_event_publisher
    ):
        """Test successful entity creation."""
        # Arrange
        request = CreateEntityRequest(name="Test Entity")
        expected_entity = Entity(
            id=uuid4(),
            name="Test Entity", 
            tenant_id="test_tenant",
            created_by="test_user",
            created_at=datetime.utcnow()
        )
        mock_repository.save.return_value = expected_entity
        
        # Act
        result = await service.create_entity(actor, request)
        
        # Assert
        assert result.is_success()
        assert result.data == expected_entity
        mock_repository.save.assert_called_once()
        mock_event_publisher.publish.assert_called_once()
    
    async def test_create_entity_permission_denied(
        self,
        service,
        actor,
        mock_permission_checker
    ):
        """Test entity creation with insufficient permissions."""
        # Arrange
        mock_permission_checker.require.side_effect = PermissionDeniedError(
            "Insufficient permissions",
            "INSUFFICIENT_PERMISSIONS"
        )
        request = CreateEntityRequest(name="Test Entity")
        
        # Act & Assert
        with pytest.raises(PermissionDeniedError):
            await service.create_entity(actor, request)
    
    async def test_create_entity_validation_error(self, service, actor):
        """Test entity creation with invalid input."""
        # Arrange  
        request = CreateEntityRequest(name="")  # Invalid name
        
        # Act
        result = await service.create_entity(actor, request)
        
        # Assert
        assert result.is_failure()
        assert result.error_code == "INVALID_NAME"
```

## Error Handling Guidelines

### 1. Use Appropriate Exception Types

```python
# Permission errors - let them bubble up
await self._permissions.require(actor, "entities:write")

# Validation errors - return OperationResult
if not data.name:
    return OperationResult.failure("Name is required", "MISSING_NAME")

# Infrastructure errors - convert to ServiceError
try:
    await self._external_api.call()
except HTTPError as e:
    raise ExternalServiceError(f"API call failed: {e}", "API_ERROR")
```

### 2. Fail Gracefully

```python
# Don't leak information across tenant boundaries
if entity.tenant_id != actor.tenant_id:
    return OperationResult.failure(
        "Entity not found",  # Don't reveal it exists in another tenant
        "ENTITY_NOT_FOUND"
    )
```

### 3. Log Appropriately

```python
# Log errors with context
self._logger.error(
    f"Failed to create entity: {e}",
    extra={
        "actor_id": actor.id,
        "tenant_id": actor.tenant_id,
        "error": str(e),
    },
    exc_info=True
)

# Log successful operations
self._logger.info(
    f"Entity created: {entity.id}",
    extra={"entity_id": str(entity.id), "actor_id": actor.id}
)
```

## Event Publishing Guidelines

### 1. Standard Event Types

Use consistent event naming:
- `{domain}.{action}` - e.g., `conversation.created`, `task.updated`
- Include relevant metadata in event
- Publish after successful database operations

```python
event = DomainEvent.create(
    event_type="entity.created",
    entity_id=entity.id,
    tenant_id=actor.tenant_id,
    actor=actor.type,
    metadata={
        "name": entity.name,
        "created_by": actor.id,
    }
)
await self._events.publish(event)
```

### 2. Event Subscription

```python
class EntityEventHandler(DomainEventSubscriber):
    
    def get_subscribed_event_types(self) -> List[str]:
        return ["entity.created", "entity.updated", "entity.deleted"]
    
    async def handle_event(self, event: DomainEvent) -> None:
        if event.event_type == "entity.created":
            await self._handle_entity_created(event)
        elif event.event_type == "entity.updated":
            await self._handle_entity_updated(event)
```

## Summary

Following this pattern ensures:

1. **Consistency** - All services use the same interfaces and patterns
2. **Testability** - Dependencies are injected and mockable
3. **Security** - Permission checks are mandatory and consistent
4. **Observability** - Events and logs provide insight into operations
5. **Maintainability** - Standard patterns make code easier to understand

For any questions about service patterns, refer to this document or reach out to the ATLAS team.