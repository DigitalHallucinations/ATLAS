"""
Tests for PersonaMemoryService.

Tests working memory, episodic memory, and semantic knowledge management.
"""

import pytest
from datetime import datetime
from typing import Any, Dict, List
from unittest.mock import MagicMock

from core.services.common import Actor
from core.services.personas.memory import PersonaMemoryService
from core.services.personas.types import (
    PersonaMemoryContext,
    EpisodicMemory,
    LearnedFact,
    PersonaKnowledge,
)


# =============================================================================
# Fixtures
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
def user_actor() -> Actor:
    """Regular user actor."""
    return Actor(
        type="user",
        id="test-user-1",
        tenant_id="test-tenant",
        permissions={"personas:read", "personas:write", "personas:admin"},
    )


@pytest.fixture
def memory_service() -> PersonaMemoryService:
    """Create a memory service instance."""
    return PersonaMemoryService()


@pytest.fixture
def memory_service_with_bus() -> PersonaMemoryService:
    """Create a memory service with a message bus."""
    mock_bus = MagicMock()
    mock_bus.publish = MagicMock()
    return PersonaMemoryService(message_bus=mock_bus)


# =============================================================================
# Working Memory Tests
# =============================================================================


class TestWorkingMemory:
    """Tests for working memory operations."""

    @pytest.mark.asyncio
    async def test_get_empty_working_memory(
        self,
        memory_service: PersonaMemoryService,
        system_actor: Actor,
    ):
        """Test getting working memory when none exists."""
        result = await memory_service.get_working_memory(
            system_actor, "TestPersona"
        )
        
        assert result.success
        assert result.data is not None
        assert result.data.persona_id == "TestPersona"
        assert result.data.scratchpad == {}

    @pytest.mark.asyncio
    async def test_set_working_memory(
        self,
        memory_service: PersonaMemoryService,
        user_actor: Actor,
    ):
        """Test setting a value in working memory."""
        result = await memory_service.set_working_memory(
            actor=user_actor,
            persona_id="TestPersona",
            key="test_key",
            value="test_value",
        )
        
        assert result.success
        assert result.data is not None
        assert result.data.scratchpad.get("test_key") == "test_value"

    @pytest.mark.asyncio
    async def test_set_multiple_values(
        self,
        memory_service: PersonaMemoryService,
        user_actor: Actor,
    ):
        """Test setting multiple values in working memory."""
        # Set first value
        await memory_service.set_working_memory(
            actor=user_actor,
            persona_id="TestPersona",
            key="key1",
            value="value1",
        )
        
        # Set second value
        result = await memory_service.set_working_memory(
            actor=user_actor,
            persona_id="TestPersona",
            key="key2",
            value="value2",
        )
        
        assert result.success
        assert result.data.scratchpad.get("key1") == "value1"
        assert result.data.scratchpad.get("key2") == "value2"

    @pytest.mark.asyncio
    async def test_set_active_goals(
        self,
        memory_service: PersonaMemoryService,
        user_actor: Actor,
    ):
        """Test setting active goals."""
        goals = ["goal1", "goal2", "goal3"]
        
        result = await memory_service.set_active_goals(
            actor=user_actor,
            persona_id="TestPersona",
            goals=goals,
        )
        
        assert result.success
        assert result.data is not None
        assert result.data.active_goals == goals

    @pytest.mark.asyncio
    async def test_set_task_context(
        self,
        memory_service: PersonaMemoryService,
        user_actor: Actor,
    ):
        """Test setting task context."""
        result = await memory_service.set_task_context(
            actor=user_actor,
            persona_id="TestPersona",
            task_context="Working on task 123",
        )
        
        assert result.success
        assert result.data is not None
        assert result.data.task_context == "Working on task 123"

    @pytest.mark.asyncio
    async def test_clear_working_memory(
        self,
        memory_service: PersonaMemoryService,
        user_actor: Actor,
    ):
        """Test clearing working memory."""
        # Set a value first
        await memory_service.set_working_memory(
            actor=user_actor,
            persona_id="TestPersona",
            key="test_key",
            value="test_value",
        )
        
        # Clear it
        result = await memory_service.clear_working_memory(
            user_actor, "TestPersona"
        )
        
        assert result.success


# =============================================================================
# Episodic Memory Tests
# =============================================================================


class TestEpisodicMemory:
    """Tests for episodic memory operations."""

    @pytest.mark.asyncio
    async def test_record_episode(
        self,
        memory_service: PersonaMemoryService,
        user_actor: Actor,
    ):
        """Test recording an episode."""
        result = await memory_service.record_episode(
            actor=user_actor,
            persona_id="TestPersona",
            summary="User asked about weather",
            outcome="success",
            key_topics=["weather", "forecast"],
        )
        
        assert result.success
        assert result.data is not None
        assert result.data.summary == "User asked about weather"
        assert result.data.outcome == "success"

    @pytest.mark.asyncio
    async def test_get_recent_episodes(
        self,
        memory_service: PersonaMemoryService,
        user_actor: Actor,
    ):
        """Test getting recent episodes."""
        # Record some episodes
        for i in range(3):
            await memory_service.record_episode(
                actor=user_actor,
                persona_id="TestPersona",
                summary=f"Episode {i}",
            )
        
        # Get recent
        result = await memory_service.get_recent_episodes(
            actor=user_actor,
            persona_id="TestPersona",
            limit=10,
        )
        
        assert result.success
        assert result.data is not None
        assert len(result.data) == 3

    @pytest.mark.asyncio
    async def test_search_episodes(
        self,
        memory_service: PersonaMemoryService,
        user_actor: Actor,
    ):
        """Test searching episodes."""
        # Record episodes with different topics
        await memory_service.record_episode(
            actor=user_actor,
            persona_id="TestPersona",
            summary="Discussion about Python programming",
            key_topics=["python", "programming"],
        )
        await memory_service.record_episode(
            actor=user_actor,
            persona_id="TestPersona",
            summary="Discussion about weather",
            key_topics=["weather"],
        )
        
        # Search for Python-related episodes
        result = await memory_service.search_episodes(
            actor=user_actor,
            persona_id="TestPersona",
            query="python",
        )
        
        assert result.success
        assert result.data is not None


# =============================================================================
# Semantic Knowledge Tests
# =============================================================================


class TestSemanticKnowledge:
    """Tests for semantic knowledge operations."""

    @pytest.mark.asyncio
    async def test_learn_fact(
        self,
        memory_service: PersonaMemoryService,
        user_actor: Actor,
    ):
        """Test learning a fact."""
        result = await memory_service.learn_fact(
            actor=user_actor,
            persona_id="TestPersona",
            subject="user",
            predicate="prefers",
            obj="dark mode",
        )
        
        assert result.success
        assert result.data is not None
        assert result.data.subject == "user"
        assert result.data.predicate == "prefers"

    @pytest.mark.asyncio
    async def test_query_facts(
        self,
        memory_service: PersonaMemoryService,
        user_actor: Actor,
    ):
        """Test querying facts."""
        # Learn some facts
        await memory_service.learn_fact(
            actor=user_actor,
            persona_id="FactsPersona",
            subject="user",
            predicate="likes",
            obj="blue",
        )
        await memory_service.learn_fact(
            actor=user_actor,
            persona_id="FactsPersona",
            subject="user",
            predicate="prefers",
            obj="pizza",
        )
        
        # Query facts
        result = await memory_service.query_facts(
            actor=user_actor,
            persona_id="FactsPersona",
        )
        
        assert result.success
        assert result.data is not None
        assert len(result.data) >= 2

    @pytest.mark.asyncio
    async def test_set_user_preference(
        self,
        memory_service: PersonaMemoryService,
        user_actor: Actor,
    ):
        """Test setting a user preference."""
        result = await memory_service.set_user_preference(
            actor=user_actor,
            persona_id="TestPersona",
            preference_key="theme",
            preference_value="dark",
        )
        
        assert result.success

    @pytest.mark.asyncio
    async def test_get_user_preferences(
        self,
        memory_service: PersonaMemoryService,
        user_actor: Actor,
    ):
        """Test getting user preferences."""
        # Set some preferences
        await memory_service.set_user_preference(
            actor=user_actor,
            persona_id="PrefsPersona",
            preference_key="theme",
            preference_value="dark",
        )
        await memory_service.set_user_preference(
            actor=user_actor,
            persona_id="PrefsPersona",
            preference_key="language",
            preference_value="en",
        )
        
        # Get preferences
        result = await memory_service.get_user_preferences(
            actor=user_actor,
            persona_id="PrefsPersona",
        )
        
        assert result.success
        assert result.data is not None

    @pytest.mark.asyncio
    async def test_get_persona_knowledge(
        self,
        memory_service: PersonaMemoryService,
        user_actor: Actor,
    ):
        """Test getting combined persona knowledge."""
        result = await memory_service.get_persona_knowledge(
            actor=user_actor,
            persona_id="TestPersona",
        )
        
        assert result.success
        # Should return some structure even if empty
        assert result.data is not None


# =============================================================================
# Context Injection Tests
# =============================================================================


class TestContextInjection:
    """Tests for context injection."""

    @pytest.mark.asyncio
    async def test_get_context_injection(
        self,
        memory_service: PersonaMemoryService,
        user_actor: Actor,
    ):
        """Test getting context for injection into prompts."""
        # Set up some context
        await memory_service.set_working_memory(
            actor=user_actor,
            persona_id="TestPersona",
            key="current_topic",
            value="weather",
        )
        await memory_service.set_active_goals(
            actor=user_actor,
            persona_id="TestPersona",
            goals=["help user with query"],
        )
        
        # Get context injection
        result = await memory_service.get_context_injection(
            actor=user_actor,
            persona_id="TestPersona",
        )
        
        assert result.success
        assert result.data is not None
        # Result is a dict with context data
        assert "active_goals" in result.data or "scratchpad" in result.data


# =============================================================================
# Service Instantiation Tests
# =============================================================================


class TestServiceInstantiation:
    """Tests for service instantiation."""

    @pytest.mark.asyncio
    async def test_create_service_with_message_bus(
        self,
        memory_service_with_bus: PersonaMemoryService,
    ):
        """Test creating service with message bus."""
        assert memory_service_with_bus._message_bus is not None

    @pytest.mark.asyncio
    async def test_create_service_without_message_bus(
        self,
        memory_service: PersonaMemoryService,
    ):
        """Test creating service without message bus."""
        assert memory_service._message_bus is None
