"""
Tests for PersonaAnalyticsService.

Tests interaction recording, metrics, A/B testing, and self-improvement.
"""

import pytest
from datetime import datetime
from typing import Any, Dict, List
from unittest.mock import MagicMock

from core.services.common import Actor
from core.services.personas.analytics import PersonaAnalyticsService
from core.services.personas.types import (
    PersonaPerformanceMetrics,
    PersonaVariant,
    PromptRefinement,
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
def analytics_service() -> PersonaAnalyticsService:
    """Create an analytics service instance."""
    return PersonaAnalyticsService()


@pytest.fixture
def analytics_service_with_bus() -> PersonaAnalyticsService:
    """Create an analytics service with a message bus."""
    mock_bus = MagicMock()
    mock_bus.publish = MagicMock()
    return PersonaAnalyticsService(message_bus=mock_bus)


# =============================================================================
# Interaction Recording Tests
# =============================================================================


class TestInteractionRecording:
    """Tests for interaction recording operations."""

    @pytest.mark.asyncio
    async def test_record_interaction(
        self,
        analytics_service: PersonaAnalyticsService,
        user_actor: Actor,
    ):
        """Test recording a successful interaction."""
        result = await analytics_service.record_interaction(
            persona_id="TestPersona",
            user_id=user_actor.id,
            response_time_ms=150,
            success=True,
            metadata={"topic": "weather"},
        )
        
        assert result.success
        assert result.data is not None
        assert result.data["persona_id"] == "TestPersona"
        assert result.data["success"] is True

    @pytest.mark.asyncio
    async def test_record_failed_interaction(
        self,
        analytics_service: PersonaAnalyticsService,
        user_actor: Actor,
    ):
        """Test recording a failed interaction."""
        result = await analytics_service.record_interaction(
            persona_id="TestPersona",
            user_id=user_actor.id,
            response_time_ms=500,
            success=False,
            error="LLM timeout",
        )
        
        assert result.success
        assert result.data is not None
        assert result.data["success"] is False

    @pytest.mark.asyncio
    async def test_record_escalation(
        self,
        analytics_service: PersonaAnalyticsService,
        user_actor: Actor,
    ):
        """Test recording an escalation."""
        result = await analytics_service.record_escalation(
            persona_id="TestPersona",
            user_id=user_actor.id,
            to_persona="ExpertAssistant",
            reason="complex_query",
        )
        
        assert result.success

    @pytest.mark.asyncio
    async def test_record_capability_gap(
        self,
        analytics_service: PersonaAnalyticsService,
        user_actor: Actor,
    ):
        """Test recording a capability gap."""
        result = await analytics_service.record_capability_gap(
            persona_id="TestPersona",
            user_id=user_actor.id,
            requested_capability="code_execution",
            request_context="user_asked_to_run_code",
        )
        
        assert result.success


# =============================================================================
# Metrics Retrieval Tests
# =============================================================================


class TestMetricsRetrieval:
    """Tests for metrics retrieval."""

    @pytest.mark.asyncio
    async def test_get_metrics_empty(
        self,
        analytics_service: PersonaAnalyticsService,
        user_actor: Actor,
    ):
        """Test getting metrics when no interactions recorded."""
        result = await analytics_service.get_metrics(
            actor=user_actor,
            persona_id="EmptyPersona",
        )
        
        assert result.success
        assert result.data is not None
        # Should return empty/zero metrics
        assert result.data.total_interactions == 0

    @pytest.mark.asyncio
    async def test_get_metrics_after_interactions(
        self,
        analytics_service: PersonaAnalyticsService,
        user_actor: Actor,
    ):
        """Test getting metrics after recording interactions."""
        # Record some interactions
        for i in range(5):
            await analytics_service.record_interaction(
                persona_id="MetricsPersona",
                user_id=user_actor.id,
                response_time_ms=100 + i * 10,
                success=True,
            )
        
        # Record one failure
        await analytics_service.record_interaction(
            persona_id="MetricsPersona",
            user_id=user_actor.id,
            response_time_ms=500,
            success=False,
        )
        
        # Get metrics
        result = await analytics_service.get_metrics(
            actor=user_actor,
            persona_id="MetricsPersona",
        )
        
        assert result.success
        assert result.data is not None
        assert result.data.total_interactions == 6
        assert result.data.task_success_rate < 1.0

    @pytest.mark.asyncio
    async def test_compare_personas(
        self,
        analytics_service: PersonaAnalyticsService,
        user_actor: Actor,
    ):
        """Test comparing metrics between personas."""
        # Record interactions for two personas
        for persona in ["PersonaA", "PersonaB"]:
            await analytics_service.record_interaction(
                persona_id=persona,
                user_id=user_actor.id,
                response_time_ms=100,
                success=True,
            )
        
        # Compare
        result = await analytics_service.compare_personas(
            actor=user_actor,
            persona_ids=["PersonaA", "PersonaB"],
        )
        
        assert result.success
        assert result.data is not None


# =============================================================================
# Improvement Analysis Tests
# =============================================================================


class TestImprovementAnalysis:
    """Tests for improvement analysis."""

    @pytest.mark.asyncio
    async def test_identify_improvement_areas(
        self,
        analytics_service: PersonaAnalyticsService,
        user_actor: Actor,
    ):
        """Test identifying areas for improvement."""
        # Record some interactions with varying success
        for i in range(10):
            await analytics_service.record_interaction(
                persona_id="ImprovementPersona",
                user_id=user_actor.id,
                response_time_ms=100 + i * 50,
                success=i % 3 != 0,  # 70% success rate
            )
        
        # Identify improvements
        result = await analytics_service.identify_improvement_areas(
            actor=user_actor,
            persona_id="ImprovementPersona",
        )
        
        assert result.success
        assert result.data is not None


# =============================================================================
# A/B Testing Tests
# =============================================================================


class TestABTesting:
    """Tests for A/B testing functionality."""

    @pytest.mark.asyncio
    async def test_create_variant(
        self,
        analytics_service: PersonaAnalyticsService,
        user_actor: Actor,
    ):
        """Test creating a persona variant."""
        result = await analytics_service.create_variant(
            actor=user_actor,
            base_persona_id="BasePersona",
            name="VariantA",
            description="Test variant with lower temperature",
            modifications={"temperature": 0.7},
            traffic_percentage=20.0,
        )
        
        assert result.success
        assert result.data is not None
        assert result.data.name == "VariantA"

    @pytest.mark.asyncio
    async def test_get_variants(
        self,
        analytics_service: PersonaAnalyticsService,
        user_actor: Actor,
    ):
        """Test getting variants for a persona."""
        # Create some variants
        await analytics_service.create_variant(
            actor=user_actor,
            base_persona_id="VariantTestPersona",
            name="V1",
            description="Variant 1",
            modifications={"temperature": 0.5},
            traffic_percentage=30.0,
        )
        await analytics_service.create_variant(
            actor=user_actor,
            base_persona_id="VariantTestPersona",
            name="V2",
            description="Variant 2",
            modifications={"temperature": 0.9},
            traffic_percentage=30.0,
        )
        
        # Get variants
        result = await analytics_service.get_variants(
            actor=user_actor,
            base_persona_id="VariantTestPersona",
        )
        
        assert result.success
        assert result.data is not None
        assert len(result.data) >= 2

    # Note: select_variant_for_user would be implemented in future iterations

    @pytest.mark.asyncio
    async def test_promote_variant(
        self,
        analytics_service: PersonaAnalyticsService,
        user_actor: Actor,
    ):
        """Test promoting a variant to be the new default."""
        # Create a variant
        create_result = await analytics_service.create_variant(
            actor=user_actor,
            base_persona_id="PromotePersona",
            name="SuccessfulVariant",
            description="Successful variant",
            modifications={"temperature": 0.6},
            traffic_percentage=25.0,
        )
        
        assert create_result.success
        variant_id = create_result.data.variant_id
        
        # Promote it
        result = await analytics_service.promote_variant(
            actor=user_actor,
            variant_id=variant_id,
        )
        
        assert result.success


# =============================================================================
# Prompt Refinement Tests
# =============================================================================


class TestPromptRefinement:
    """Tests for prompt refinement suggestions."""

    @pytest.mark.asyncio
    async def test_suggest_refinement(
        self,
        analytics_service: PersonaAnalyticsService,
        user_actor: Actor,
    ):
        """Test suggesting a prompt refinement."""
        result = await analytics_service.suggest_refinement(
            actor=user_actor,
            persona_id="RefinePersona",
            field_path="content.greeting",
            suggested_value="Hello! How can I help you today?",
            reason="Users prefer friendlier greetings",
        )
        
        assert result.success
        assert result.data is not None

    @pytest.mark.asyncio
    async def test_get_pending_refinements(
        self,
        analytics_service: PersonaAnalyticsService,
        user_actor: Actor,
    ):
        """Test getting pending refinements."""
        result = await analytics_service.get_pending_refinements(
            actor=user_actor,
            persona_id="TestPersona",
        )
        
        assert result.success
        assert result.data is not None

    @pytest.mark.asyncio
    async def test_apply_refinement(
        self,
        analytics_service: PersonaAnalyticsService,
        user_actor: Actor,
    ):
        """Test applying a refinement."""
        # First suggest a refinement
        suggest_result = await analytics_service.suggest_refinement(
            actor=user_actor,
            persona_id="ApplyPersona",
            field_path="content.greeting",
            suggested_value="Hello! How can I help you?",
            reason="Improve greeting",
        )
        
        assert suggest_result.success
        refinement_id = suggest_result.data.refinement_id
        
        # Apply it
        result = await analytics_service.apply_refinement(
            actor=user_actor,
            refinement_id=refinement_id,
        )
        
        assert result.success


# =============================================================================
# Service Instantiation Tests
# =============================================================================


class TestServiceInstantiation:
    """Tests for service instantiation."""

    @pytest.mark.asyncio
    async def test_create_service_with_message_bus(
        self,
        analytics_service_with_bus: PersonaAnalyticsService,
    ):
        """Test creating service with message bus."""
        assert analytics_service_with_bus._message_bus is not None

    @pytest.mark.asyncio
    async def test_create_service_without_message_bus(
        self,
        analytics_service: PersonaAnalyticsService,
    ):
        """Test creating service without message bus."""
        assert analytics_service._message_bus is None
