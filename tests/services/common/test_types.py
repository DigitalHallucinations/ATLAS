"""
Tests for core service types.

Tests the OperationResult, Actor, and DomainEvent classes that
form the foundation of the ATLAS service pattern.

Author: ATLAS Team
Date: Jan 7, 2026
"""

import pytest
from datetime import datetime
from uuid import UUID, uuid4

from core.services.common.types import Actor, DomainEvent, OperationResult


class TestOperationResult:
    """Test OperationResult generic type."""
    
    def test_success_creation(self):
        """Test creating successful results."""
        data = {"user_id": "123", "name": "John"}
        result = OperationResult.success(data)
        
        assert result.success is True
        assert result.data == data
        assert result.error is None
        assert result.error_code is None
        assert result.is_success()
        assert not result.is_failure()
    
    def test_failure_creation(self):
        """Test creating failed results."""
        result = OperationResult.failure("Something went wrong", "ERROR_CODE")
        
        assert result.success is False
        assert result.data is None
        assert result.error == "Something went wrong"
        assert result.error_code == "ERROR_CODE"
        assert not result.is_success()
        assert result.is_failure()
    
    def test_failure_without_error_code(self):
        """Test failure creation without explicit error code."""
        result = OperationResult.failure("Error message")
        
        assert result.success is False
        assert result.error == "Error message"
        assert result.error_code is None
    
    def test_unwrap_success(self):
        """Test unwrapping successful results."""
        data = {"value": 42}
        result = OperationResult.success(data)
        
        unwrapped = result.unwrap()
        assert unwrapped == data
    
    def test_unwrap_failure_raises(self):
        """Test that unwrapping failed results raises exception."""
        result = OperationResult.failure("Failed")
        
        with pytest.raises(RuntimeError, match="Cannot unwrap failed result"):
            result.unwrap()
    
    def test_unwrap_or_success(self):
        """Test unwrap_or with successful result."""
        data = {"value": 42}
        result = OperationResult.success(data)
        
        unwrapped = result.unwrap_or({"default": True})
        assert unwrapped == data
    
    def test_unwrap_or_failure(self):
        """Test unwrap_or with failed result returns default."""
        result = OperationResult.failure("Failed")
        default = {"default": True}
        
        unwrapped = result.unwrap_or(default)
        assert unwrapped == default


class TestActor:
    """Test Actor class."""
    
    @pytest.fixture
    def user_actor(self):
        """Standard user actor for testing."""
        return Actor(
            type="user",
            id="user_123",
            tenant_id="org_456",
            permissions={"conversations:read", "conversations:write"}
        )
    
    @pytest.fixture
    def system_actor(self):
        """System actor with full permissions."""
        return Actor(
            type="system",
            id="atlas_system", 
            tenant_id="system",
            permissions={"*"}
        )
    
    def test_actor_creation(self, user_actor):
        """Test actor creation with basic attributes."""
        assert user_actor.type == "user"
        assert user_actor.id == "user_123"
        assert user_actor.tenant_id == "org_456"
        assert "conversations:read" in user_actor.permissions
        assert "conversations:write" in user_actor.permissions
    
    def test_has_permission(self, user_actor):
        """Test permission checking."""
        assert user_actor.has_permission("conversations:read")
        assert user_actor.has_permission("conversations:write")
        assert not user_actor.has_permission("admin:delete")
    
    def test_wildcard_permission(self, system_actor):
        """Test wildcard permissions."""
        assert system_actor.has_permission("anything")
        assert system_actor.has_permission("admin:delete")
        assert system_actor.has_permission("conversations:read")
    
    def test_is_system(self, user_actor, system_actor):
        """Test system actor detection."""
        assert not user_actor.is_system()
        assert system_actor.is_system()
    
    def test_is_user(self, user_actor, system_actor):
        """Test user actor detection."""
        assert user_actor.is_user()
        assert not system_actor.is_user()


class TestDomainEvent:
    """Test DomainEvent class."""
    
    def test_domain_event_creation(self):
        """Test creating domain events."""
        entity_id = uuid4()
        timestamp = datetime.utcnow()
        
        event = DomainEvent(
            event_type="conversation.created",
            entity_id=entity_id,
            tenant_id="org_123",
            timestamp=timestamp,
            actor="user",
            metadata={"title": "New Chat"}
        )
        
        assert event.event_type == "conversation.created"
        assert event.entity_id == entity_id
        assert event.tenant_id == "org_123"
        assert event.timestamp == timestamp
        assert event.actor == "user"
        assert event.metadata == {"title": "New Chat"}
    
    def test_domain_event_create_factory(self):
        """Test domain event creation using factory method."""
        entity_id = uuid4()
        
        event = DomainEvent.create(
            event_type="task.created",
            entity_id=entity_id,
            tenant_id="org_456",
            actor="system",
            metadata={"priority": "high"}
        )
        
        assert event.event_type == "task.created"
        assert event.entity_id == entity_id
        assert event.tenant_id == "org_456"
        assert event.actor == "system"
        assert event.metadata == {"priority": "high"}
        assert isinstance(event.timestamp, datetime)
    
    def test_domain_event_create_with_string_id(self):
        """Test domain event creation with string entity ID."""
        entity_id_str = str(uuid4())
        
        event = DomainEvent.create(
            event_type="user.login",
            entity_id=entity_id_str,
            tenant_id="org_789",
            actor="user"
        )
        
        assert event.entity_id == UUID(entity_id_str)
    
    def test_domain_event_to_dict(self):
        """Test serializing domain event to dictionary."""
        entity_id = uuid4()
        timestamp = datetime(2026, 1, 7, 12, 0, 0)
        
        event = DomainEvent(
            event_type="test.event",
            entity_id=entity_id,
            tenant_id="test_tenant",
            timestamp=timestamp,
            actor="user",
            metadata={"key": "value"}
        )
        
        data = event.to_dict()
        
        expected = {
            "event_type": "test.event",
            "entity_id": str(entity_id),
            "tenant_id": "test_tenant",
            "timestamp": "2026-01-07T12:00:00",
            "actor": "user",
            "metadata": {"key": "value"}
        }
        
        assert data == expected
    
    def test_domain_event_from_dict(self):
        """Test deserializing domain event from dictionary."""
        entity_id = uuid4()
        
        data = {
            "event_type": "test.event",
            "entity_id": str(entity_id),
            "tenant_id": "test_tenant", 
            "timestamp": "2026-01-07T12:00:00",
            "actor": "user",
            "metadata": {"key": "value"}
        }
        
        event = DomainEvent.from_dict(data)
        
        assert event.event_type == "test.event"
        assert event.entity_id == entity_id
        assert event.tenant_id == "test_tenant"
        assert event.timestamp == datetime(2026, 1, 7, 12, 0, 0)
        assert event.actor == "user"
        assert event.metadata == {"key": "value"}
    
    def test_domain_event_immutable(self):
        """Test that domain events are immutable."""
        event = DomainEvent.create(
            event_type="test.event",
            entity_id=uuid4(),
            tenant_id="test",
            actor="user"
        )
        
        # This should raise an error since event is frozen
        with pytest.raises(Exception):  # FrozenInstanceError in Python 3.7+
            event.event_type = "modified.event"
    
    def test_domain_event_roundtrip_serialization(self):
        """Test that events survive roundtrip serialization."""
        original = DomainEvent.create(
            event_type="roundtrip.test",
            entity_id=uuid4(),
            tenant_id="test_tenant",
            actor="system",
            metadata={"complex": {"nested": {"data": [1, 2, 3]}}}
        )
        
        # Serialize and deserialize
        data = original.to_dict()
        restored = DomainEvent.from_dict(data)
        
        # Should be equal
        assert restored.event_type == original.event_type
        assert restored.entity_id == original.entity_id
        assert restored.tenant_id == original.tenant_id
        assert restored.actor == original.actor
        assert restored.metadata == original.metadata