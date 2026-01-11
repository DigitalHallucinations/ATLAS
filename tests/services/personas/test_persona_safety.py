"""
Tests for PersonaSafetyService.

Tests guardrails, content filtering, and HITL approval workflows.
"""

import pytest
from unittest.mock import MagicMock

from core.services.common import Actor
from core.services.personas.safety import PersonaSafetyService
from core.services.personas.types import (
    PersonaSafetyPolicy,
    ContentFilterRule,
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
def safety_service() -> PersonaSafetyService:
    """Create a safety service instance."""
    return PersonaSafetyService()


@pytest.fixture
def safety_service_with_bus() -> PersonaSafetyService:
    """Create a safety service with a message bus."""
    mock_bus = MagicMock()
    mock_bus.publish = MagicMock()
    return PersonaSafetyService(message_bus=mock_bus)


@pytest.fixture
def sample_policy() -> PersonaSafetyPolicy:
    """Sample safety policy."""
    return PersonaSafetyPolicy(
        persona_id="TestPersona",
        max_actions_per_minute=10,
        max_tokens_per_conversation=1000,
        max_tool_calls_per_turn=3,
        risk_threshold=0.7,
        auto_escalation_threshold=0.9,
    )


@pytest.fixture
def policy_with_filters() -> PersonaSafetyPolicy:
    """Policy with content filters."""
    return PersonaSafetyPolicy(
        persona_id="FilteredPersona",
        max_actions_per_minute=10,
        output_filters=[
            ContentFilterRule(
                rule_id="profanity-filter",
                name="Profanity Filter",
                pattern=r"badword|offensive",
                action="mask",
                applies_to="output",
                message="Profanity detected",
                priority=1,
            ),
            ContentFilterRule(
                rule_id="block-filter",
                name="Block Filter",
                pattern=r"blocked_content",
                action="block",
                applies_to="output",
                message="Content blocked",
                priority=0,
            ),
        ],
    )


# =============================================================================
# Policy Management Tests
# =============================================================================


class TestPolicyManagement:
    """Tests for policy management operations."""

    @pytest.mark.asyncio
    async def test_get_default_policy_when_none_set(
        self,
        safety_service: PersonaSafetyService,
        system_actor: Actor,
    ):
        """Test getting a default policy when none is set."""
        result = await safety_service.get_safety_policy(system_actor, "TestPersona")
        
        assert result.success
        assert result.data is not None
        assert result.data.persona_id == "TestPersona"
        # Should have default limits
        assert result.data.max_actions_per_minute > 0
        assert result.data.max_tokens_per_conversation > 0

    @pytest.mark.asyncio
    async def test_update_safety_policy(
        self,
        safety_service: PersonaSafetyService,
        user_actor: Actor,
        sample_policy: PersonaSafetyPolicy,
    ):
        """Test updating a safety policy."""
        result = await safety_service.update_safety_policy(
            user_actor,
            sample_policy.persona_id,
            sample_policy,
        )
        
        assert result.success
        
        # Verify the policy was stored
        get_result = await safety_service.get_safety_policy(
            user_actor, sample_policy.persona_id
        )
        assert get_result.success
        assert get_result.data is not None
        assert get_result.data.max_actions_per_minute == 10

    @pytest.mark.asyncio
    async def test_delete_safety_policy(
        self,
        safety_service: PersonaSafetyService,
        user_actor: Actor,
        sample_policy: PersonaSafetyPolicy,
    ):
        """Test deleting a safety policy."""
        # First set a policy
        await safety_service.update_safety_policy(
            user_actor,
            sample_policy.persona_id,
            sample_policy,
        )
        
        # Delete it
        result = await safety_service.delete_safety_policy(
            user_actor, sample_policy.persona_id
        )
        assert result.success


# =============================================================================
# Action Checking Tests
# =============================================================================


class TestActionChecking:
    """Tests for action checking."""

    @pytest.mark.asyncio
    async def test_check_allowed_action(
        self,
        safety_service: PersonaSafetyService,
        system_actor: Actor,
    ):
        """Test checking an allowed action."""
        result = await safety_service.check_action(
            actor=system_actor,
            persona_id="TestPersona",
            action="send_message",
        )
        
        assert result.success
        assert result.data is not None
        # ActionCheckResult has 'allowed' attribute
        assert result.data.allowed is True

    @pytest.mark.asyncio
    async def test_check_blocked_action(
        self,
        safety_service: PersonaSafetyService,
        user_actor: Actor,
    ):
        """Test checking an action that should be blocked."""
        # Create a policy that blocks certain actions
        policy = PersonaSafetyPolicy(
            persona_id="BlockedPersona",
            max_actions_per_minute=10,
            blocked_actions=["dangerous_action"],
        )
        await safety_service.update_safety_policy(
            user_actor, "BlockedPersona", policy
        )
        
        result = await safety_service.check_action(
            actor=user_actor,
            persona_id="BlockedPersona",
            action="dangerous_action",
        )
        
        assert result.success
        assert result.data is not None
        assert result.data.allowed is False


# =============================================================================
# Content Filtering Tests
# =============================================================================


class TestContentFiltering:
    """Tests for content filtering."""

    @pytest.mark.asyncio
    async def test_filter_clean_content(
        self,
        safety_service: PersonaSafetyService,
        user_actor: Actor,
        policy_with_filters: PersonaSafetyPolicy,
    ):
        """Test filtering clean content passes through."""
        await safety_service.update_safety_policy(
            user_actor,
            policy_with_filters.persona_id,
            policy_with_filters,
        )
        
        result = await safety_service.filter_content(
            persona_id=policy_with_filters.persona_id,
            content="This is clean content",
            direction="output",
        )
        
        assert result.success
        assert result.data is not None
        assert result.data["filtered_content"] == "This is clean content"
        assert result.data["blocked"] is False

    @pytest.mark.asyncio
    async def test_filter_masked_content(
        self,
        safety_service: PersonaSafetyService,
        user_actor: Actor,
        policy_with_filters: PersonaSafetyPolicy,
    ):
        """Test filtering content that should be masked."""
        await safety_service.update_safety_policy(
            user_actor,
            policy_with_filters.persona_id,
            policy_with_filters,
        )
        
        result = await safety_service.filter_content(
            persona_id=policy_with_filters.persona_id,
            content="This contains a badword in it",
            direction="output",
        )
        
        assert result.success
        assert result.data is not None
        # Should have masked the bad word
        assert "badword" not in result.data["filtered_content"]
        assert len(result.data["violations"]) > 0

    @pytest.mark.asyncio
    async def test_filter_blocked_content(
        self,
        safety_service: PersonaSafetyService,
        user_actor: Actor,
        policy_with_filters: PersonaSafetyPolicy,
    ):
        """Test filtering content that should be blocked."""
        await safety_service.update_safety_policy(
            user_actor,
            policy_with_filters.persona_id,
            policy_with_filters,
        )
        
        result = await safety_service.filter_content(
            persona_id=policy_with_filters.persona_id,
            content="This has blocked_content in it",
            direction="output",
        )
        
        assert result.success
        assert result.data is not None
        assert result.data["blocked"] is True


# =============================================================================
# Approval Workflow Tests
# =============================================================================


class TestApprovalWorkflow:
    """Tests for HITL approval workflow."""

    @pytest.mark.asyncio
    async def test_get_pending_approvals_empty(
        self,
        safety_service: PersonaSafetyService,
        user_actor: Actor,
    ):
        """Test getting pending approvals when none exist."""
        result = await safety_service.get_pending_approvals(
            user_actor, "TestPersona"
        )
        
        assert result.success
        assert result.data is not None
        assert len(result.data) == 0


# =============================================================================
# Service Instantiation Tests
# =============================================================================


class TestServiceInstantiation:
    """Tests for service instantiation."""

    @pytest.mark.asyncio
    async def test_create_service_with_message_bus(
        self,
        safety_service_with_bus: PersonaSafetyService,
    ):
        """Test creating service with message bus."""
        assert safety_service_with_bus._message_bus is not None

    @pytest.mark.asyncio
    async def test_create_service_without_message_bus(
        self,
        safety_service: PersonaSafetyService,
    ):
        """Test creating service without message bus."""
        assert safety_service._message_bus is None
