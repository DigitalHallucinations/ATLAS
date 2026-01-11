"""
Tests for PersonaSwitchingService.

Tests persona handoff, switching rules, and fallback chains.
"""

import pytest
from datetime import datetime
from typing import Any, Dict, List
from unittest.mock import MagicMock

from core.services.common import Actor
from core.services.personas.switching import PersonaSwitchingService
from core.services.personas.types import (
    PersonaHandoff,
    HandoffContext,
    HandoffStatus,
    SwitchingRule,
    FallbackChain,
    SwitchTriggerType,
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
def switching_service() -> PersonaSwitchingService:
    """Create a switching service instance."""
    return PersonaSwitchingService()


@pytest.fixture
def switching_service_with_bus() -> PersonaSwitchingService:
    """Create a switching service with a message bus."""
    mock_bus = MagicMock()
    mock_bus.publish = MagicMock()
    return PersonaSwitchingService(message_bus=mock_bus)


@pytest.fixture
def sample_handoff_context() -> HandoffContext:
    """Sample handoff context."""
    return HandoffContext(
        conversation_summary="User was asking about weather",
        active_goals=["provide_weather_info"],
        working_memory_snapshot={"last_query": "weather in NYC"},
        user_preferences={"units": "metric"},
    )


@pytest.fixture
def sample_switching_rule() -> SwitchingRule:
    """Sample switching rule."""
    return SwitchingRule(
        name="Code Question Rule",
        description="Switch to code expert for programming questions",
        trigger_pattern=r"(code|programming|python|javascript)",
        from_persona=None,  # From any persona
        to_persona="CodeExpert",
        priority=50,
        require_confirmation=True,
    )


@pytest.fixture
def sample_fallback_chain() -> FallbackChain:
    """Sample fallback chain."""
    return FallbackChain(
        name="General Assistant Fallback",
        primary_persona="GeneralAssistant",
        fallback_personas=["BasicAssistant", "FallbackBot"],
        trigger_on_error=True,
        trigger_on_timeout=True,
        max_fallbacks=2,
    )


# =============================================================================
# Handoff Protocol Tests
# =============================================================================


class TestHandoffProtocol:
    """Tests for handoff protocol operations."""

    @pytest.mark.asyncio
    async def test_initiate_handoff(
        self,
        switching_service: PersonaSwitchingService,
        user_actor: Actor,
        sample_handoff_context: HandoffContext,
    ):
        """Test initiating a persona handoff."""
        result = await switching_service.initiate_handoff(
            actor=user_actor,
            from_persona="GeneralAssistant",
            to_persona="TechExpert",
            reason="User asked technical question",
            context=sample_handoff_context,
        )
        
        assert result.success
        assert result.data is not None
        assert result.data.from_persona == "GeneralAssistant"
        assert result.data.to_persona == "TechExpert"
        assert result.data.status == HandoffStatus.PENDING

    @pytest.mark.asyncio
    async def test_accept_handoff(
        self,
        switching_service: PersonaSwitchingService,
        user_actor: Actor,
    ):
        """Test accepting a handoff."""
        # First initiate a handoff
        initiate_result = await switching_service.initiate_handoff(
            actor=user_actor,
            from_persona="PersonaA",
            to_persona="PersonaB",
            reason="Testing acceptance",
        )
        
        assert initiate_result.success
        handoff_id = initiate_result.data.handoff_id
        
        # Accept it
        result = await switching_service.accept_handoff(
            handoff_id=handoff_id,
        )
        
        assert result.success
        assert result.data.status == HandoffStatus.ACCEPTED

    @pytest.mark.asyncio
    async def test_complete_handoff(
        self,
        switching_service: PersonaSwitchingService,
        user_actor: Actor,
    ):
        """Test completing a handoff."""
        # Initiate and accept
        initiate_result = await switching_service.initiate_handoff(
            actor=user_actor,
            from_persona="PersonaA",
            to_persona="PersonaB",
            reason="Testing completion",
        )
        handoff_id = initiate_result.data.handoff_id
        
        await switching_service.accept_handoff(handoff_id)
        
        # Complete it
        result = await switching_service.complete_handoff(
            handoff_id=handoff_id,
        )
        
        assert result.success
        assert result.data.status == HandoffStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_cancel_handoff(
        self,
        switching_service: PersonaSwitchingService,
        user_actor: Actor,
    ):
        """Test canceling a handoff."""
        # Initiate
        initiate_result = await switching_service.initiate_handoff(
            actor=user_actor,
            from_persona="PersonaA",
            to_persona="PersonaB",
            reason="Testing cancellation",
        )
        handoff_id = initiate_result.data.handoff_id
        
        # Cancel it
        result = await switching_service.cancel_handoff(
            actor=user_actor,
            handoff_id=handoff_id,
            reason="User changed mind",
        )
        
        assert result.success
        assert result.data.status == HandoffStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_get_handoff(
        self,
        switching_service: PersonaSwitchingService,
        user_actor: Actor,
    ):
        """Test getting handoff details."""
        # Initiate
        initiate_result = await switching_service.initiate_handoff(
            actor=user_actor,
            from_persona="PersonaA",
            to_persona="PersonaB",
            reason="Testing status check",
        )
        handoff_id = initiate_result.data.handoff_id
        
        # Get details
        result = await switching_service.get_handoff(
            handoff_id=handoff_id,
        )
        
        assert result.success
        assert result.data.handoff_id == handoff_id

    @pytest.mark.asyncio
    async def test_get_pending_handoffs(
        self,
        switching_service: PersonaSwitchingService,
        user_actor: Actor,
    ):
        """Test getting pending handoffs for a user."""
        # Create some handoffs
        for i in range(3):
            await switching_service.initiate_handoff(
                actor=user_actor,
                from_persona=f"PersonaA{i}",
                to_persona=f"PersonaB{i}",
                reason=f"Test handoff {i}",
            )
        
        # Get pending
        result = await switching_service.get_pending_handoffs(
            user_id=user_actor.id,
        )
        
        assert result.success
        assert result.data is not None
        assert len(result.data) >= 3


# =============================================================================
# Switching Rules Tests
# =============================================================================


class TestSwitchingRules:
    """Tests for switching rules management."""

    @pytest.mark.asyncio
    async def test_add_switching_rule(
        self,
        switching_service: PersonaSwitchingService,
        user_actor: Actor,
        sample_switching_rule: SwitchingRule,
    ):
        """Test adding a switching rule."""
        result = await switching_service.add_switching_rule(
            actor=user_actor,
            rule=sample_switching_rule,
        )
        
        assert result.success
        assert result.data is not None

    @pytest.mark.asyncio
    async def test_get_switching_rules(
        self,
        switching_service: PersonaSwitchingService,
        user_actor: Actor,
        sample_switching_rule: SwitchingRule,
    ):
        """Test getting switching rules."""
        # Add a rule (set from_persona on the rule itself)
        sample_switching_rule.from_persona = "RulesPersona"
        await switching_service.add_switching_rule(
            actor=user_actor,
            rule=sample_switching_rule,
        )
        
        # Get rules
        result = await switching_service.get_switching_rules(
            actor=user_actor,
            persona_id="RulesPersona",
        )
        
        assert result.success
        assert result.data is not None
        assert len(result.data) >= 1

    @pytest.mark.asyncio
    async def test_remove_switching_rule(
        self,
        switching_service: PersonaSwitchingService,
        user_actor: Actor,
        sample_switching_rule: SwitchingRule,
    ):
        """Test removing a switching rule."""
        # Add a rule
        sample_switching_rule.from_persona = "RemoveRulePersona"
        add_result = await switching_service.add_switching_rule(
            actor=user_actor,
            rule=sample_switching_rule,
        )
        rule_id = add_result.data.rule_id
        
        # Remove it
        result = await switching_service.remove_switching_rule(
            actor=user_actor,
            rule_id=rule_id,
        )
        
        assert result.success

    @pytest.mark.asyncio
    async def test_evaluate_switching_rules(
        self,
        switching_service: PersonaSwitchingService,
        user_actor: Actor,
        sample_switching_rule: SwitchingRule,
    ):
        """Test evaluating switching rules against input."""
        # Add a rule (set from_persona on the rule)
        sample_switching_rule.from_persona = "EvalPersona"
        await switching_service.add_switching_rule(
            actor=user_actor,
            rule=sample_switching_rule,
        )
        
        # Evaluate with matching input - note: method signature differs
        result = await switching_service.evaluate_switching_rules(
            current_persona="EvalPersona",
            user_input="I have a python programming question",
        )
        
        assert result.success

    @pytest.mark.asyncio
    async def test_suggest_persona_switch(
        self,
        switching_service: PersonaSwitchingService,
    ):
        """Test suggesting a persona switch based on context."""
        result = await switching_service.suggest_persona_switch(
            current_persona="GeneralAssistant",
            user_input="User is asking about advanced machine learning algorithms",
        )
        
        assert result.success
        # May or may not have a suggestion


# =============================================================================
# Fallback Chain Tests
# =============================================================================


class TestFallbackChains:
    """Tests for fallback chain management."""

    @pytest.mark.asyncio
    async def test_set_fallback_chain(
        self,
        switching_service: PersonaSwitchingService,
        user_actor: Actor,
        sample_fallback_chain: FallbackChain,
    ):
        """Test setting a fallback chain."""
        result = await switching_service.set_fallback_chain(
            actor=user_actor,
            chain=sample_fallback_chain,
        )
        
        assert result.success
        assert result.data is not None

    @pytest.mark.asyncio
    async def test_get_fallback_chain(
        self,
        switching_service: PersonaSwitchingService,
        user_actor: Actor,
        sample_fallback_chain: FallbackChain,
    ):
        """Test getting a fallback chain."""
        # Set a chain
        await switching_service.set_fallback_chain(
            actor=user_actor,
            chain=sample_fallback_chain,
        )
        
        # Get it
        result = await switching_service.get_fallback_chain(
            primary_persona=sample_fallback_chain.primary_persona,
        )
        
        assert result.success
        assert result.data is not None
        assert result.data.primary_persona == sample_fallback_chain.primary_persona

    @pytest.mark.asyncio
    async def test_get_next_fallback(
        self,
        switching_service: PersonaSwitchingService,
        user_actor: Actor,
        sample_fallback_chain: FallbackChain,
    ):
        """Test getting the next fallback persona."""
        # Set a chain
        await switching_service.set_fallback_chain(
            actor=user_actor,
            chain=sample_fallback_chain,
        )
        
        # Get next fallback (first in chain after primary)
        result = await switching_service.get_next_fallback(
            primary_persona=sample_fallback_chain.primary_persona,
            failed_personas=[sample_fallback_chain.primary_persona],
        )
        
        assert result.success
        # Should return first fallback
        if result.data:
            assert result.data == sample_fallback_chain.fallback_personas[0]


# =============================================================================
# Context Transfer Tests
# =============================================================================


class TestContextTransfer:
    """Tests for context transfer between personas."""

    @pytest.mark.asyncio
    async def test_transfer_context(
        self,
        switching_service: PersonaSwitchingService,
        user_actor: Actor,
        sample_handoff_context: HandoffContext,
    ):
        """Test transferring context between personas."""
        result = await switching_service.transfer_context(
            actor=user_actor,
            from_persona="PersonaA",
            to_persona="PersonaB",
            context=sample_handoff_context,
        )
        
        assert result.success


# =============================================================================
# Active Persona Management Tests
# =============================================================================


class TestActivePersonaManagement:
    """Tests for active persona management."""

    @pytest.mark.asyncio
    async def test_get_active_persona(
        self,
        switching_service: PersonaSwitchingService,
        user_actor: Actor,
    ):
        """Test getting the active persona for a user."""
        result = await switching_service.get_active_persona(
            user_id=user_actor.id,
        )
        
        # Should succeed even if no active persona
        assert result.success

    @pytest.mark.asyncio
    async def test_set_active_persona(
        self,
        switching_service: PersonaSwitchingService,
        user_actor: Actor,
    ):
        """Test setting the active persona for a user."""
        result = await switching_service.set_active_persona(
            user_id=user_actor.id,
            persona_id="TestPersona",
        )
        
        assert result.success
        
        # Verify it was set
        get_result = await switching_service.get_active_persona(
            user_id=user_actor.id,
        )
        assert get_result.success
        assert get_result.data == "TestPersona"


# =============================================================================
# Service Instantiation Tests
# =============================================================================


class TestServiceInstantiation:
    """Tests for service instantiation."""

    @pytest.mark.asyncio
    async def test_create_service_with_message_bus(
        self,
        switching_service_with_bus: PersonaSwitchingService,
    ):
        """Test creating service with message bus."""
        assert switching_service_with_bus._message_bus is not None

    @pytest.mark.asyncio
    async def test_create_service_without_message_bus(
        self,
        switching_service: PersonaSwitchingService,
    ):
        """Test creating service without message bus."""
        assert switching_service._message_bus is None
