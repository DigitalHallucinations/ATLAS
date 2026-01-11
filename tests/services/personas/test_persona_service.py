"""
Tests for PersonaService.

Comprehensive test coverage for persona CRUD operations,
validation, active persona management, and permission checks.

Author: ATLAS Team
Date: Jan 10, 2026
"""

from __future__ import annotations

import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Import directly from submodules to avoid circular import issues
from core.services.common.types import Actor, OperationResult
from core.services.personas.service import PersonaService
from core.services.personas.permissions import PersonaPermissionChecker
from core.services.personas.validation import PersonaValidator
from core.services.personas.types import (
    PersonaCreate,
    PersonaUpdate,
    PersonaFilters,
    PersonaResponse,
    PersonaListResponse,
    PersonaSummary,
    PersonaCapabilities,
    ValidationResult,
    PersonaCreated,
    PersonaUpdated,
    PersonaDeleted,
    PersonaActivated,
    PersonaDeactivated,
)
from core.services.personas.exceptions import (
    PersonaError,
    PersonaNotFoundError,
    PersonaValidationError,
    PersonaAlreadyExistsError,
)


# =============================================================================
# Test Fixtures
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
def admin_actor() -> Actor:
    """Admin user with persona admin permissions."""
    return Actor(
        type="user",
        id="admin_user",
        tenant_id="tenant_1",
        permissions={"personas:admin", "personas:write", "personas:read", "personas:delete"},
    )


@pytest.fixture
def user_actor() -> Actor:
    """Regular user with read permissions."""
    return Actor(
        type="user",
        id="regular_user",
        tenant_id="tenant_1",
        permissions={"personas:read", "personas:activate"},
    )


@pytest.fixture
def limited_actor() -> Actor:
    """User with no persona permissions."""
    return Actor(
        type="user",
        id="limited_user",
        tenant_id="tenant_1",
        permissions=set(),
    )


@pytest.fixture
def mock_message_bus() -> AsyncMock:
    """Mock message bus for event testing."""
    bus = AsyncMock()
    bus.publish = AsyncMock()
    return bus


@pytest.fixture
def mock_validator() -> MagicMock:
    """Mock validator that always passes."""
    validator = MagicMock(spec=PersonaValidator)
    validator.validate.return_value = ValidationResult(
        is_valid=True,
        errors=[],
        warnings=[],
        persona_name="TestPersona",
    )
    validator.validate_name.return_value = ValidationResult(
        is_valid=True,
        errors=[],
        warnings=[],
        persona_name="TestPersona",
    )
    validator.validate_content.return_value = ValidationResult(
        is_valid=True,
        errors=[],
        warnings=[],
    )
    return validator


@pytest.fixture
def sample_persona_dict() -> Dict[str, Any]:
    """Sample persona dictionary."""
    return {
        "name": "TestPersona",
        "meaning": "A test persona",
        "content": {
            "start_locked": "You are a test assistant.",
            "editable_content": "Additional instructions here.",
            "end_locked": "Be helpful and accurate.",
        },
        "provider": "openai",
        "model": "gpt-4",
        "Speech_provider": "elevenlabs",
        "voice": "test_voice",
        "sys_info_enabled": True,
        "user_profile_enabled": False,
        "allowed_tools": ["calculator", "web_search"],
        "allowed_skills": ["code_analysis"],
        "type": {
            "personal_assistant": {
                "enabled": True,
                "access_to_calendar": True,
                "calendar_write_enabled": False,
            }
        },
    }


@pytest.fixture
def persona_service(mock_message_bus: AsyncMock, mock_validator: MagicMock) -> PersonaService:
    """Create PersonaService with mocked dependencies."""
    return PersonaService(
        config_manager=None,
        message_bus=mock_message_bus,
        validator=mock_validator,
    )


# =============================================================================
# Permission Tests
# =============================================================================


class TestPersonaPermissions:
    """Tests for PersonaPermissionChecker."""

    @pytest.mark.asyncio
    async def test_system_actor_has_all_permissions(self, system_actor: Actor):
        """System actor should bypass all permission checks."""
        checker = PersonaPermissionChecker()

        # All checks should pass without raising
        await checker.check_read(system_actor)
        await checker.check_write(system_actor, "TestPersona")
        await checker.check_delete(system_actor, "TestPersona")
        await checker.check_activate(system_actor, "TestPersona")
        await checker.check_validate(system_actor)

    @pytest.mark.asyncio
    async def test_admin_has_write_permissions(self, admin_actor: Actor):
        """Admin actor should have write permissions."""
        checker = PersonaPermissionChecker()

        await checker.check_read(admin_actor)
        await checker.check_write(admin_actor, "TestPersona")
        await checker.check_delete(admin_actor, "TestPersona")

    @pytest.mark.asyncio
    async def test_user_has_read_permissions(self, user_actor: Actor):
        """Regular user should have read permissions."""
        checker = PersonaPermissionChecker()

        await checker.check_read(user_actor)
        await checker.check_activate(user_actor, "TestPersona")

    @pytest.mark.asyncio
    async def test_limited_user_denied_read(self, limited_actor: Actor):
        """Limited user should be denied read access."""
        from core.services.common.exceptions import PermissionDeniedError

        checker = PersonaPermissionChecker()

        with pytest.raises(PermissionDeniedError):
            await checker.check_read(limited_actor)

    @pytest.mark.asyncio
    async def test_user_denied_write(self, user_actor: Actor):
        """Regular user should be denied write access."""
        from core.services.common.exceptions import PermissionDeniedError

        checker = PersonaPermissionChecker()

        with pytest.raises(PermissionDeniedError):
            await checker.check_write(user_actor, "TestPersona")

    @pytest.mark.asyncio
    async def test_user_denied_delete(self, user_actor: Actor):
        """Regular user should be denied delete access."""
        from core.services.common.exceptions import PermissionDeniedError

        checker = PersonaPermissionChecker()

        with pytest.raises(PermissionDeniedError):
            await checker.check_delete(user_actor, "TestPersona")


# =============================================================================
# Validation Tests
# =============================================================================


class TestPersonaValidation:
    """Tests for PersonaValidator."""

    def test_validate_name_valid(self):
        """Valid names should pass validation."""
        validator = PersonaValidator()

        result = validator.validate_name("ATLAS")
        assert result.is_valid
        assert len(result.errors) == 0

    def test_validate_name_empty(self):
        """Empty names should fail validation."""
        validator = PersonaValidator()

        result = validator.validate_name("")
        assert not result.is_valid
        assert len(result.errors) > 0

    def test_validate_name_with_path_separator(self):
        """Names with path separators should fail validation."""
        validator = PersonaValidator()

        result = validator.validate_name("test/persona")
        assert not result.is_valid
        assert any("path" in e.lower() for e in result.errors)

    def test_validate_name_with_traversal(self):
        """Names with traversal sequences should fail validation."""
        validator = PersonaValidator()

        result = validator.validate_name("..persona")
        assert not result.is_valid

    def test_validate_name_relative_dir(self):
        """Names that are relative directories should fail."""
        validator = PersonaValidator()

        result = validator.validate_name("..")
        assert not result.is_valid

    def test_validate_content_valid(self):
        """Valid content should pass validation."""
        validator = PersonaValidator()

        result = validator.validate_content({
            "start_locked": "Start",
            "editable_content": "Middle",
            "end_locked": "End",
        })
        assert result.is_valid

    def test_validate_content_missing_field(self):
        """Content missing required fields should fail."""
        validator = PersonaValidator()

        result = validator.validate_content({
            "start_locked": "Start",
            "editable_content": "Middle",
            # Missing end_locked
        })
        assert not result.is_valid
        assert any("end_locked" in e for e in result.errors)


# =============================================================================
# Service CRUD Tests
# =============================================================================


class TestPersonaServiceCRUD:
    """Tests for PersonaService CRUD operations."""

    @pytest.mark.asyncio
    async def test_list_personas_success(
        self,
        persona_service: PersonaService,
        admin_actor: Actor,
    ):
        """Should list available personas."""
        with patch.object(persona_service, "_list_persona_directories") as mock_list:
            mock_list.return_value = ["ATLAS", "TestPersona"]
            with patch.object(persona_service, "_load_persona_raw") as mock_load:
                mock_load.side_effect = [
                    {"name": "ATLAS", "meaning": "Main assistant", "allowed_tools": ["a", "b"]},
                    {"name": "TestPersona", "meaning": "Test", "allowed_skills": ["x"]},
                ]

                result = await persona_service.list_personas(admin_actor)

        assert result.success
        assert result.data is not None
        assert len(result.data.personas) == 2
        assert result.data.total == 2

    @pytest.mark.asyncio
    async def test_list_personas_with_filter(
        self,
        persona_service: PersonaService,
        admin_actor: Actor,
    ):
        """Should filter personas by name pattern."""
        with patch.object(persona_service, "_list_persona_directories") as mock_list:
            mock_list.return_value = ["ATLAS", "TestPersona", "OtherPersona"]
            with patch.object(persona_service, "_load_persona_raw") as mock_load:
                mock_load.return_value = {"name": "TestPersona", "meaning": "Test"}

                filters = PersonaFilters(name_pattern="test")
                result = await persona_service.list_personas(admin_actor, filters)

        assert result.success
        assert result.data is not None
        # Only "TestPersona" matches "test" pattern
        assert len(result.data.personas) <= 2

    @pytest.mark.asyncio
    async def test_list_personas_permission_denied(
        self,
        persona_service: PersonaService,
        limited_actor: Actor,
    ):
        """Should deny list for unauthorized users."""
        result = await persona_service.list_personas(limited_actor)

        assert not result.success
        assert result.error is not None
        assert "denied" in result.error.lower()

    @pytest.mark.asyncio
    async def test_get_persona_success(
        self,
        persona_service: PersonaService,
        admin_actor: Actor,
        sample_persona_dict: Dict[str, Any],
    ):
        """Should get a specific persona."""
        with patch.object(persona_service, "_load_persona_raw") as mock_load:
            mock_load.return_value = sample_persona_dict

            result = await persona_service.get_persona(admin_actor, "TestPersona")

        assert result.success
        assert result.data is not None
        assert result.data.name == "TestPersona"
        assert result.data.provider == "openai"

    @pytest.mark.asyncio
    async def test_get_persona_not_found(
        self,
        persona_service: PersonaService,
        admin_actor: Actor,
    ):
        """Should return error for missing persona."""
        with patch.object(persona_service, "_load_persona_raw") as mock_load:
            mock_load.return_value = None

            result = await persona_service.get_persona(admin_actor, "NonExistent")

        assert not result.success
        assert result.error is not None
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_create_persona_success(
        self,
        persona_service: PersonaService,
        admin_actor: Actor,
        mock_message_bus: AsyncMock,
    ):
        """Should create a new persona."""
        persona_create = PersonaCreate(
            name="NewPersona",
            content={
                "start_locked": "Start",
                "editable_content": "Middle",
                "end_locked": "End",
            },
            meaning="A new test persona",
            provider="openai",
        )

        with patch.object(persona_service, "_load_persona_raw") as mock_load:
            # First call: check if exists (None = doesn't exist)
            # Second call: reload after create
            mock_load.side_effect = [None, {
                "name": "NewPersona",
                "content": persona_create.content,
                "meaning": "A new test persona",
                "provider": "openai",
            }]

            with patch("core.services.personas.service._get_persona_deps") as mock_deps:
                mock_deps.return_value = {
                    "persist_persona_definition": MagicMock(),
                    "load_persona_definition": MagicMock(),
                    "normalize_allowed_tools": lambda x, **_: x or [],
                    "normalize_allowed_skills": lambda x, **_: x or [],
                    "load_tool_metadata": MagicMock(return_value=([], {})),
                    "load_skill_catalog": MagicMock(return_value=([], {})),
                    "resolve_app_root": MagicMock(return_value=Path("/tmp")),
                }

                result = await persona_service.create_persona(admin_actor, persona_create)

        assert result.success
        assert result.data is not None
        assert result.data.name == "NewPersona"
        # Event should be published
        assert mock_message_bus.publish.called

    @pytest.mark.asyncio
    async def test_create_persona_already_exists(
        self,
        persona_service: PersonaService,
        admin_actor: Actor,
    ):
        """Should fail when persona already exists."""
        persona_create = PersonaCreate(
            name="ExistingPersona",
            content={
                "start_locked": "Start",
                "editable_content": "Middle",
                "end_locked": "End",
            },
        )

        with patch.object(persona_service, "_load_persona_raw") as mock_load:
            mock_load.return_value = {"name": "ExistingPersona"}  # Already exists

            result = await persona_service.create_persona(admin_actor, persona_create)

        assert not result.success
        assert result.error is not None
        assert "already exists" in result.error.lower()

    @pytest.mark.asyncio
    async def test_create_persona_validation_error(
        self,
        persona_service: PersonaService,
        admin_actor: Actor,
        mock_validator: MagicMock,
    ):
        """Should fail with invalid name."""
        mock_validator.validate_name.return_value = ValidationResult(
            is_valid=False,
            errors=["Invalid name"],
            warnings=[],
        )

        persona_create = PersonaCreate(
            name="Invalid/Name",
            content={
                "start_locked": "Start",
                "editable_content": "Middle",
                "end_locked": "End",
            },
        )

        result = await persona_service.create_persona(admin_actor, persona_create)

        assert not result.success
        assert result.error is not None
        assert "Invalid name" in result.error

    @pytest.mark.asyncio
    async def test_update_persona_success(
        self,
        persona_service: PersonaService,
        admin_actor: Actor,
        sample_persona_dict: Dict[str, Any],
        mock_message_bus: AsyncMock,
    ):
        """Should update an existing persona."""
        update = PersonaUpdate(meaning="Updated meaning")

        with patch.object(persona_service, "_load_persona_raw") as mock_load:
            mock_load.return_value = sample_persona_dict

            with patch("core.services.personas.service._get_persona_deps") as mock_deps:
                mock_deps.return_value = {
                    "persist_persona_definition": MagicMock(),
                    "load_persona_definition": MagicMock(),
                    "normalize_allowed_tools": lambda x, **_: x or [],
                    "normalize_allowed_skills": lambda x, **_: x or [],
                    "load_tool_metadata": MagicMock(return_value=([], {})),
                    "load_skill_catalog": MagicMock(return_value=([], {})),
                    "resolve_app_root": MagicMock(return_value=Path("/tmp")),
                }

                result = await persona_service.update_persona(
                    admin_actor, "TestPersona", update
                )

        assert result.success
        assert mock_message_bus.publish.called

    @pytest.mark.asyncio
    async def test_update_persona_not_found(
        self,
        persona_service: PersonaService,
        admin_actor: Actor,
    ):
        """Should fail to update missing persona."""
        update = PersonaUpdate(meaning="Updated")

        with patch.object(persona_service, "_load_persona_raw") as mock_load:
            mock_load.return_value = None

            result = await persona_service.update_persona(
                admin_actor, "NonExistent", update
            )

        assert not result.success
        assert result.error is not None
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_delete_persona_success(
        self,
        persona_service: PersonaService,
        admin_actor: Actor,
        sample_persona_dict: Dict[str, Any],
        mock_message_bus: AsyncMock,
    ):
        """Should delete a persona."""
        with patch.object(persona_service, "_load_persona_raw") as mock_load:
            mock_load.return_value = sample_persona_dict

            with patch.object(persona_service, "_get_personas_root") as mock_root:
                mock_root.return_value = Path(tempfile.mkdtemp())
                # Create the persona dir
                persona_dir = mock_root.return_value / "TestPersona"
                persona_dir.mkdir(parents=True)

                result = await persona_service.delete_persona(admin_actor, "TestPersona")

        assert result.success
        assert result.data is True
        assert mock_message_bus.publish.called

    @pytest.mark.asyncio
    async def test_delete_protected_persona(
        self,
        persona_service: PersonaService,
        admin_actor: Actor,
    ):
        """Should not delete protected personas."""
        result = await persona_service.delete_persona(admin_actor, "ATLAS")

        assert not result.success
        assert result.error is not None
        assert "protected" in result.error.lower()

    @pytest.mark.asyncio
    async def test_delete_active_persona(
        self,
        persona_service: PersonaService,
        admin_actor: Actor,
        sample_persona_dict: Dict[str, Any],
    ):
        """Should not delete persona while active."""
        # Set persona as active
        persona_service._active_personas["some_user"] = "TestPersona"

        with patch.object(persona_service, "_load_persona_raw") as mock_load:
            mock_load.return_value = sample_persona_dict

            result = await persona_service.delete_persona(admin_actor, "TestPersona")

        assert not result.success
        assert result.error is not None
        assert "active" in result.error.lower()


# =============================================================================
# Active Persona Tests
# =============================================================================


class TestActivePersonaManagement:
    """Tests for active persona management."""

    @pytest.mark.asyncio
    async def test_get_active_persona_none(
        self,
        persona_service: PersonaService,
        user_actor: Actor,
    ):
        """Should return None when no persona is active."""
        result = await persona_service.get_active_persona(user_actor)

        assert result.success
        assert result.data is None

    @pytest.mark.asyncio
    async def test_set_active_persona_success(
        self,
        persona_service: PersonaService,
        user_actor: Actor,
        sample_persona_dict: Dict[str, Any],
        mock_message_bus: AsyncMock,
    ):
        """Should set active persona."""
        with patch.object(persona_service, "_load_persona_raw") as mock_load:
            mock_load.return_value = sample_persona_dict

            result = await persona_service.set_active_persona(
                user_actor, "TestPersona"
            )

        assert result.success
        assert result.data is not None
        assert result.data.name == "TestPersona"
        assert mock_message_bus.publish.called

    @pytest.mark.asyncio
    async def test_set_active_persona_publishes_deactivate(
        self,
        persona_service: PersonaService,
        user_actor: Actor,
        sample_persona_dict: Dict[str, Any],
        mock_message_bus: AsyncMock,
    ):
        """Should publish deactivate event when switching personas."""
        # Set initial active persona
        persona_service._active_personas[user_actor.id] = "OldPersona"

        with patch.object(persona_service, "_load_persona_raw") as mock_load:
            mock_load.return_value = sample_persona_dict

            await persona_service.set_active_persona(user_actor, "TestPersona")

        # Should have published both deactivate and activate events
        assert mock_message_bus.publish.call_count >= 2

    @pytest.mark.asyncio
    async def test_set_active_persona_not_found(
        self,
        persona_service: PersonaService,
        user_actor: Actor,
    ):
        """Should fail to activate missing persona."""
        with patch.object(persona_service, "_load_persona_raw") as mock_load:
            mock_load.return_value = None

            result = await persona_service.set_active_persona(
                user_actor, "NonExistent"
            )

        assert not result.success
        assert result.error is not None
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_get_active_persona_after_set(
        self,
        persona_service: PersonaService,
        user_actor: Actor,
        sample_persona_dict: Dict[str, Any],
    ):
        """Should get the active persona after setting it."""
        with patch.object(persona_service, "_load_persona_raw") as mock_load:
            mock_load.return_value = sample_persona_dict

            await persona_service.set_active_persona(user_actor, "TestPersona")
            result = await persona_service.get_active_persona(user_actor)

        assert result.success
        assert result.data is not None
        assert result.data.name == "TestPersona"


# =============================================================================
# Validation Service Tests
# =============================================================================


class TestValidationService:
    """Tests for persona validation through the service."""

    @pytest.mark.asyncio
    async def test_validate_persona_success(
        self,
        persona_service: PersonaService,
        admin_actor: Actor,
        sample_persona_dict: Dict[str, Any],
        mock_message_bus: AsyncMock,
    ):
        """Should validate an existing persona."""
        with patch.object(persona_service, "_load_persona_raw") as mock_load:
            mock_load.return_value = sample_persona_dict

            result = await persona_service.validate_persona(
                admin_actor, persona_name="TestPersona"
            )

        assert result.success
        assert result.data is not None
        assert result.data.is_valid

    @pytest.mark.asyncio
    async def test_validate_persona_data(
        self,
        persona_service: PersonaService,
        admin_actor: Actor,
        sample_persona_dict: Dict[str, Any],
    ):
        """Should validate provided persona data."""
        result = await persona_service.validate_persona(
            admin_actor, persona_data=sample_persona_dict
        )

        assert result.success
        assert result.data is not None

    @pytest.mark.asyncio
    async def test_validate_persona_not_found(
        self,
        persona_service: PersonaService,
        admin_actor: Actor,
    ):
        """Should fail to validate missing persona."""
        with patch.object(persona_service, "_load_persona_raw") as mock_load:
            mock_load.return_value = None

            result = await persona_service.validate_persona(
                admin_actor, persona_name="NonExistent"
            )

        assert not result.success
        assert result.error is not None
        assert "not found" in result.error.lower()


# =============================================================================
# Capabilities Tests
# =============================================================================


class TestPersonaCapabilities:
    """Tests for persona capabilities."""

    @pytest.mark.asyncio
    async def test_get_capabilities_success(
        self,
        persona_service: PersonaService,
        admin_actor: Actor,
        sample_persona_dict: Dict[str, Any],
    ):
        """Should get persona capabilities."""
        with patch.object(persona_service, "_load_persona_raw") as mock_load:
            mock_load.return_value = sample_persona_dict

            result = await persona_service.get_persona_capabilities(
                admin_actor, "TestPersona"
            )

        assert result.success
        assert result.data is not None
        assert "calculator" in result.data.tools
        assert result.data.has_calendar_access

    @pytest.mark.asyncio
    async def test_get_capabilities_not_found(
        self,
        persona_service: PersonaService,
        admin_actor: Actor,
    ):
        """Should fail for missing persona."""
        with patch.object(persona_service, "_load_persona_raw") as mock_load:
            mock_load.return_value = None

            result = await persona_service.get_persona_capabilities(
                admin_actor, "NonExistent"
            )

        assert not result.success


# =============================================================================
# Event Tests
# =============================================================================


class TestPersonaEvents:
    """Tests for persona domain events."""

    def test_persona_created_event(self):
        """PersonaCreated event should serialize correctly."""
        event = PersonaCreated(
            persona_name="TestPersona",
            tenant_id="tenant_1",
            actor_id="user_1",
            provider="openai",
        )

        data = event.to_dict()
        assert data["event_type"] == "persona.created"
        assert data["persona_name"] == "TestPersona"
        assert data["provider"] == "openai"

    def test_persona_updated_event(self):
        """PersonaUpdated event should serialize correctly."""
        event = PersonaUpdated(
            persona_name="TestPersona",
            changed_fields=("meaning", "provider"),
            tenant_id="tenant_1",
            actor_id="user_1",
        )

        data = event.to_dict()
        assert data["event_type"] == "persona.updated"
        assert "meaning" in data["changed_fields"]
        assert "provider" in data["changed_fields"]

    def test_persona_activated_event(self):
        """PersonaActivated event should serialize correctly."""
        event = PersonaActivated(
            persona_name="TestPersona",
            tenant_id="tenant_1",
            actor_id="user_1",
            previous_persona="OldPersona",
        )

        data = event.to_dict()
        assert data["event_type"] == "persona.activated"
        assert data["previous_persona"] == "OldPersona"


# =============================================================================
# Response Type Tests
# =============================================================================


class TestPersonaResponseTypes:
    """Tests for PersonaResponse conversion."""

    def test_from_persona_dict_basic(self, sample_persona_dict: Dict[str, Any]):
        """Should convert basic persona dict to response."""
        response = PersonaResponse.from_persona_dict(sample_persona_dict)

        assert response.name == "TestPersona"
        assert response.meaning == "A test persona"
        assert response.provider == "openai"
        assert response.model == "gpt-4"
        assert "calculator" in response.allowed_tools

    def test_from_persona_dict_capabilities(self, sample_persona_dict: Dict[str, Any]):
        """Should extract capabilities correctly."""
        response = PersonaResponse.from_persona_dict(sample_persona_dict)

        assert response.capabilities.has_calendar_access
        assert not response.capabilities.has_calendar_write
        assert "personal_assistant" in response.capabilities.persona_types

    def test_from_persona_dict_minimal(self):
        """Should handle minimal persona dict."""
        minimal = {
            "name": "Minimal",
            "content": {
                "start_locked": "",
                "editable_content": "Hello",
                "end_locked": "",
            },
        }

        response = PersonaResponse.from_persona_dict(minimal)

        assert response.name == "Minimal"
        assert response.provider is None
        assert len(response.allowed_tools) == 0
