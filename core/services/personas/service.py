"""
Persona service for ATLAS.

Provides persona management with permission checks,
MessageBus events, and OperationResult returns.

Author: ATLAS Team
Date: Jan 10, 2026
"""

from __future__ import annotations

import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence

from core.services.common import Actor, OperationResult
from core.services.common.exceptions import PermissionDeniedError

from .exceptions import (
    PersonaError,
    PersonaNotFoundError,
    PersonaValidationError,
    PersonaAlreadyExistsError,
    PersonaDeleteError,
    PersonaActiveError,
    PersonaIOError,
)
from .permissions import PersonaPermissionChecker
from .types import (
    PersonaCreate,
    PersonaUpdate,
    PersonaFilters,
    PersonaResponse,
    PersonaListResponse,
    PersonaSummary,
    PersonaCapabilities,
    ValidationResult,
    # Events
    PersonaCreated,
    PersonaUpdated,
    PersonaDeleted,
    PersonaActivated,
    PersonaDeactivated,
    PersonaValidated,
)
from .validation import PersonaValidator

if TYPE_CHECKING:
    from core.config import ConfigManager
    from core.messaging import MessageBus


logger = logging.getLogger(__name__)


def _now_utc() -> datetime:
    """Get current UTC timestamp."""
    return datetime.now(timezone.utc)


def _get_persona_deps():
    """Lazy import persona dependencies to avoid circular imports."""
    from modules.Personas import (
        load_persona_definition,
        persist_persona_definition,
        normalize_allowed_tools,
        normalize_allowed_skills,
        load_tool_metadata,
        load_skill_catalog,
    )
    from modules.store_common.manifest_utils import resolve_app_root

    return {
        "load_persona_definition": load_persona_definition,
        "persist_persona_definition": persist_persona_definition,
        "normalize_allowed_tools": normalize_allowed_tools,
        "normalize_allowed_skills": normalize_allowed_skills,
        "load_tool_metadata": load_tool_metadata,
        "load_skill_catalog": load_skill_catalog,
        "resolve_app_root": resolve_app_root,
    }


class PersonaService:
    """
    Application service that coordinates persona operations.

    Provides:
    - CRUD operations with permission checks
    - Schema validation
    - MessageBus event publishing
    - Active persona tracking per user/session
    """

    # System personas that cannot be deleted
    PROTECTED_PERSONAS = frozenset({"ATLAS"})

    def __init__(
        self,
        config_manager: Optional["ConfigManager"] = None,
        message_bus: Optional["MessageBus"] = None,
        permission_checker: Optional[PersonaPermissionChecker] = None,
        validator: Optional[PersonaValidator] = None,
    ) -> None:
        """
        Initialize the PersonaService.

        Args:
            config_manager: Configuration manager for path resolution
            message_bus: Message bus for publishing events
            permission_checker: Custom permission checker (uses default if None)
            validator: Custom validator (uses default if None)
        """
        self._config_manager = config_manager
        self._message_bus = message_bus
        self._permission_checker = permission_checker or PersonaPermissionChecker()
        self._validator = validator or PersonaValidator(config_manager)

        # Active persona tracking (user_id -> persona_name)
        self._active_personas: Dict[str, str] = {}

        # Metadata caches
        self._tool_metadata_cache: Optional[tuple] = None
        self._skill_metadata_cache: Optional[tuple] = None

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    def _get_personas_root(self) -> Path:
        """Get the root directory for persona definitions."""
        deps = _get_persona_deps()
        app_root = deps["resolve_app_root"](self._config_manager, logger=logger)
        return app_root / "modules" / "Personas"

    def _get_tool_metadata(self) -> tuple:
        """Get cached tool metadata."""
        if self._tool_metadata_cache is None:
            deps = _get_persona_deps()
            order, lookup = deps["load_tool_metadata"](config_manager=self._config_manager)
            self._tool_metadata_cache = (order, lookup)
        return self._tool_metadata_cache

    def _get_skill_metadata(self) -> tuple:
        """Get cached skill metadata."""
        if self._skill_metadata_cache is None:
            deps = _get_persona_deps()
            order, lookup = deps["load_skill_catalog"](config_manager=self._config_manager)
            self._skill_metadata_cache = (order, lookup)
        return self._skill_metadata_cache

    def _list_persona_directories(self) -> List[str]:
        """List all persona directory names."""
        personas_root = self._get_personas_root()
        persona_names: List[str] = []

        try:
            for entry in os.scandir(personas_root):
                if not entry.is_dir():
                    continue
                # Must have a Persona subdirectory
                persona_marker = os.path.join(entry.path, "Persona")
                if not os.path.isdir(persona_marker):
                    continue
                persona_names.append(entry.name)
        except OSError as e:
            logger.error("Failed to list persona directories: %s", e)

        return sorted(persona_names)

    def _load_persona_raw(self, persona_name: str) -> Optional[Dict[str, Any]]:
        """Load raw persona definition from disk."""
        deps = _get_persona_deps()
        order, lookup = self._get_tool_metadata()
        skill_order, skill_lookup = self._get_skill_metadata()

        try:
            return deps["load_persona_definition"](
                persona_name,
                config_manager=self._config_manager,
                metadata_order=order,
                metadata_lookup=lookup,
                skill_metadata_order=skill_order,
                skill_metadata_lookup=skill_lookup,
            )
        except Exception as e:
            logger.error("Failed to load persona '%s': %s", persona_name, e)
            return None

    async def _publish_event(self, event: Any) -> None:
        """Publish event to message bus if available."""
        if self._message_bus is None:
            return
        try:
            if hasattr(event, "to_dict"):
                await self._message_bus.publish(event.event_type, event.to_dict())
            else:
                await self._message_bus.publish(str(type(event).__name__), event)
        except Exception as e:
            logger.warning("Failed to publish event: %s", e)

    # =========================================================================
    # Public API
    # =========================================================================

    async def list_personas(
        self,
        actor: Actor,
        filters: Optional[PersonaFilters] = None,
    ) -> OperationResult[PersonaListResponse]:
        """
        List available personas.

        Args:
            actor: The actor performing the operation
            filters: Optional filters to apply

        Returns:
            OperationResult containing PersonaListResponse
        """
        try:
            await self._permission_checker.check_read(actor)
        except PermissionDeniedError as e:
            return OperationResult.failure(str(e), e.error_code)

        filters = filters or PersonaFilters()
        persona_names = self._list_persona_directories()

        # Apply name pattern filter
        if filters.name_pattern:
            pattern = filters.name_pattern.lower()
            persona_names = [n for n in persona_names if pattern in n.lower()]

        total = len(persona_names)

        # Apply pagination
        start = filters.offset
        end = start + filters.limit
        paginated_names = persona_names[start:end]

        # Build summaries
        summaries: List[PersonaSummary] = []
        for name in paginated_names:
            persona_data = self._load_persona_raw(name)
            if persona_data is None:
                continue

            # Apply additional filters
            if filters.provider:
                if persona_data.get("provider") != filters.provider:
                    continue
            if filters.has_tool:
                tools = persona_data.get("allowed_tools", [])
                if filters.has_tool not in tools:
                    continue
            if filters.has_skill:
                skills = persona_data.get("allowed_skills", [])
                if filters.has_skill not in skills:
                    continue
            if filters.persona_type:
                type_config = persona_data.get("type", {})
                type_entry = type_config.get(filters.persona_type, {})
                if not type_entry.get("enabled", False):
                    continue

            summaries.append(PersonaSummary(
                name=persona_data.get("name", name),
                meaning=persona_data.get("meaning"),
                provider=persona_data.get("provider"),
                model=persona_data.get("model"),
                tool_count=len(persona_data.get("allowed_tools", [])),
                skill_count=len(persona_data.get("allowed_skills", [])),
            ))

        response = PersonaListResponse(
            personas=summaries,
            total=total,
            limit=filters.limit,
            offset=filters.offset,
            has_more=end < total,
        )

        return OperationResult.success(response)

    async def get_persona(
        self,
        actor: Actor,
        persona_name: str,
    ) -> OperationResult[PersonaResponse]:
        """
        Get a specific persona by name.

        Args:
            actor: The actor performing the operation
            persona_name: Name of the persona to retrieve

        Returns:
            OperationResult containing PersonaResponse
        """
        try:
            await self._permission_checker.check_read(actor, persona_name)
        except PermissionDeniedError as e:
            return OperationResult.failure(str(e), e.error_code)

        persona_data = self._load_persona_raw(persona_name)
        if persona_data is None:
            return OperationResult.failure(
                f"Persona '{persona_name}' not found",
                "PERSONA_NOT_FOUND",
            )

        response = PersonaResponse.from_persona_dict(persona_data)
        return OperationResult.success(response)

    async def create_persona(
        self,
        actor: Actor,
        persona: PersonaCreate,
    ) -> OperationResult[PersonaResponse]:
        """
        Create a new persona.

        Args:
            actor: The actor performing the operation
            persona: The persona creation data

        Returns:
            OperationResult containing the created PersonaResponse
        """
        try:
            await self._permission_checker.check_write(actor, persona.name, is_create=True)
        except PermissionDeniedError as e:
            return OperationResult.failure(str(e), e.error_code)

        # Validate name
        name_result = self._validator.validate_name(persona.name)
        if not name_result.is_valid:
            return OperationResult.failure(
                "; ".join(name_result.errors),
                "PERSONA_VALIDATION_ERROR",
            )

        # Check if persona already exists
        existing = self._load_persona_raw(persona.name)
        if existing is not None:
            return OperationResult.failure(
                f"Persona '{persona.name}' already exists",
                "PERSONA_ALREADY_EXISTS",
            )

        # Build persona dict
        persona_dict = self._build_persona_dict(persona)

        # Validate content
        validation_result = self._validator.validate(persona_dict, persona_name=persona.name)
        if not validation_result.is_valid:
            return OperationResult.failure(
                "; ".join(validation_result.errors),
                "PERSONA_VALIDATION_ERROR",
            )

        # Persist
        try:
            deps = _get_persona_deps()
            deps["persist_persona_definition"](
                persona.name,
                persona_dict,
                config_manager=self._config_manager,
                rationale=f"Created by {actor.id}",
            )
        except Exception as e:
            logger.error("Failed to persist persona '%s': %s", persona.name, e)
            return OperationResult.failure(
                f"Failed to create persona: {e}",
                "PERSONA_IO_ERROR",
            )

        # Clear caches
        self._tool_metadata_cache = None
        self._skill_metadata_cache = None

        # Reload to get normalized data
        created_data = self._load_persona_raw(persona.name)
        if created_data is None:
            return OperationResult.failure(
                "Persona created but failed to reload",
                "PERSONA_IO_ERROR",
            )

        # Publish event
        await self._publish_event(PersonaCreated(
            persona_name=persona.name,
            tenant_id=actor.tenant_id,
            actor_id=actor.id,
            provider=persona.provider,
            model=persona.model,
            actor_type=actor.type,
        ))

        return OperationResult.success(PersonaResponse.from_persona_dict(created_data))

    async def update_persona(
        self,
        actor: Actor,
        persona_name: str,
        update: PersonaUpdate,
    ) -> OperationResult[PersonaResponse]:
        """
        Update an existing persona.

        Args:
            actor: The actor performing the operation
            persona_name: Name of the persona to update
            update: The update data

        Returns:
            OperationResult containing the updated PersonaResponse
        """
        try:
            await self._permission_checker.check_write(actor, persona_name)
        except PermissionDeniedError as e:
            return OperationResult.failure(str(e), e.error_code)

        # Load existing persona
        existing = self._load_persona_raw(persona_name)
        if existing is None:
            return OperationResult.failure(
                f"Persona '{persona_name}' not found",
                "PERSONA_NOT_FOUND",
            )

        # Apply updates
        updated_dict, changed_fields = self._apply_update(existing, update)

        if not changed_fields:
            # No changes
            return OperationResult.success(PersonaResponse.from_persona_dict(existing))

        # Validate
        validation_result = self._validator.validate(updated_dict, persona_name=persona_name)
        if not validation_result.is_valid:
            return OperationResult.failure(
                "; ".join(validation_result.errors),
                "PERSONA_VALIDATION_ERROR",
            )

        # Handle rename
        target_name = persona_name
        if update.name and update.name != persona_name:
            # Validate new name
            name_result = self._validator.validate_name(update.name)
            if not name_result.is_valid:
                return OperationResult.failure(
                    "; ".join(name_result.errors),
                    "PERSONA_VALIDATION_ERROR",
                )
            # Check new name doesn't exist
            if self._load_persona_raw(update.name) is not None:
                return OperationResult.failure(
                    f"Cannot rename: persona '{update.name}' already exists",
                    "PERSONA_ALREADY_EXISTS",
                )
            target_name = update.name
            updated_dict["name"] = update.name

        # Persist
        try:
            deps = _get_persona_deps()
            deps["persist_persona_definition"](
                target_name,
                updated_dict,
                config_manager=self._config_manager,
                rationale=f"Updated by {actor.id}",
            )

            # Handle directory rename if needed
            if target_name != persona_name:
                self._rename_persona_directory(persona_name, target_name)
        except Exception as e:
            logger.error("Failed to update persona '%s': %s", persona_name, e)
            return OperationResult.failure(
                f"Failed to update persona: {e}",
                "PERSONA_IO_ERROR",
            )

        # Publish event
        await self._publish_event(PersonaUpdated(
            persona_name=target_name,
            changed_fields=tuple(changed_fields),
            tenant_id=actor.tenant_id,
            actor_id=actor.id,
            actor_type=actor.type,
        ))

        # Reload
        updated_data = self._load_persona_raw(target_name)
        if updated_data is None:
            return OperationResult.failure(
                "Persona updated but failed to reload",
                "PERSONA_IO_ERROR",
            )

        return OperationResult.success(PersonaResponse.from_persona_dict(updated_data))

    async def delete_persona(
        self,
        actor: Actor,
        persona_name: str,
    ) -> OperationResult[bool]:
        """
        Delete a persona.

        Args:
            actor: The actor performing the operation
            persona_name: Name of the persona to delete

        Returns:
            OperationResult indicating success
        """
        try:
            await self._permission_checker.check_delete(actor, persona_name)
        except PermissionDeniedError as e:
            return OperationResult.failure(str(e), e.error_code)

        # Check if protected
        if persona_name in self.PROTECTED_PERSONAS:
            return OperationResult.failure(
                f"Cannot delete protected persona '{persona_name}'",
                "PERSONA_DELETE_ERROR",
            )

        # Check if active
        for user_id, active_name in self._active_personas.items():
            if active_name == persona_name:
                return OperationResult.failure(
                    f"Cannot delete persona '{persona_name}' while it is active for user '{user_id}'",
                    "PERSONA_ACTIVE_ERROR",
                )

        # Check exists
        existing = self._load_persona_raw(persona_name)
        if existing is None:
            return OperationResult.failure(
                f"Persona '{persona_name}' not found",
                "PERSONA_NOT_FOUND",
            )

        # Delete directory
        try:
            personas_root = self._get_personas_root()
            persona_dir = personas_root / persona_name
            if persona_dir.exists():
                shutil.rmtree(persona_dir)
        except Exception as e:
            logger.error("Failed to delete persona '%s': %s", persona_name, e)
            return OperationResult.failure(
                f"Failed to delete persona: {e}",
                "PERSONA_IO_ERROR",
            )

        # Publish event
        await self._publish_event(PersonaDeleted(
            persona_name=persona_name,
            tenant_id=actor.tenant_id,
            actor_id=actor.id,
            actor_type=actor.type,
        ))

        return OperationResult.success(True)

    async def validate_persona(
        self,
        actor: Actor,
        persona_name: Optional[str] = None,
        persona_data: Optional[Mapping[str, Any]] = None,
    ) -> OperationResult[ValidationResult]:
        """
        Validate a persona definition.

        Args:
            actor: The actor performing the operation
            persona_name: Name of existing persona to validate
            persona_data: Raw persona data to validate (if not loading from disk)

        Returns:
            OperationResult containing ValidationResult
        """
        try:
            await self._permission_checker.check_validate(actor, persona_name)
        except PermissionDeniedError as e:
            return OperationResult.failure(str(e), e.error_code)

        if persona_data is None and persona_name:
            persona_data = self._load_persona_raw(persona_name)
            if persona_data is None:
                return OperationResult.failure(
                    f"Persona '{persona_name}' not found",
                    "PERSONA_NOT_FOUND",
                )

        if persona_data is None:
            return OperationResult.failure(
                "Either persona_name or persona_data must be provided",
                "VALIDATION_ERROR",
            )

        result = self._validator.validate(persona_data, persona_name=persona_name)

        # Publish event
        await self._publish_event(PersonaValidated(
            persona_name=persona_name or "Unknown",
            tenant_id=actor.tenant_id,
            actor_id=actor.id,
            is_valid=result.is_valid,
            error_count=len(result.errors),
            warning_count=len(result.warnings),
            actor_type=actor.type,
        ))

        return OperationResult.success(result)

    async def get_active_persona(
        self,
        actor: Actor,
        user_id: Optional[str] = None,
    ) -> OperationResult[Optional[PersonaResponse]]:
        """
        Get the currently active persona for a user.

        Args:
            actor: The actor performing the operation
            user_id: The user to get active persona for (defaults to actor.id)

        Returns:
            OperationResult containing PersonaResponse or None
        """
        try:
            await self._permission_checker.check_read(actor)
        except PermissionDeniedError as e:
            return OperationResult.failure(str(e), e.error_code)

        target_user = user_id or actor.id
        active_name = self._active_personas.get(target_user)

        if active_name is None:
            return OperationResult.success(None)

        result = await self.get_persona(actor, active_name)
        if not result.success:
            return OperationResult.failure(result.error or "Unknown error", result.error_code)
        return OperationResult.success(result.data)

    async def set_active_persona(
        self,
        actor: Actor,
        persona_name: str,
        user_id: Optional[str] = None,
    ) -> OperationResult[PersonaResponse]:
        """
        Set the active persona for a user.

        Args:
            actor: The actor performing the operation
            persona_name: Name of the persona to activate
            user_id: The user to set active persona for (defaults to actor.id)

        Returns:
            OperationResult containing the activated PersonaResponse
        """
        try:
            await self._permission_checker.check_activate(actor, persona_name)
        except PermissionDeniedError as e:
            return OperationResult.failure(str(e), e.error_code)

        # Verify persona exists
        persona_data = self._load_persona_raw(persona_name)
        if persona_data is None:
            return OperationResult.failure(
                f"Persona '{persona_name}' not found",
                "PERSONA_NOT_FOUND",
            )

        target_user = user_id or actor.id
        previous_persona = self._active_personas.get(target_user)

        # Deactivate previous if different
        if previous_persona and previous_persona != persona_name:
            await self._publish_event(PersonaDeactivated(
                persona_name=previous_persona,
                tenant_id=actor.tenant_id,
                actor_id=actor.id,
                actor_type=actor.type,
            ))

        # Activate new
        self._active_personas[target_user] = persona_name

        await self._publish_event(PersonaActivated(
            persona_name=persona_name,
            tenant_id=actor.tenant_id,
            actor_id=actor.id,
            previous_persona=previous_persona,
            actor_type=actor.type,
        ))

        return OperationResult.success(PersonaResponse.from_persona_dict(persona_data))

    async def get_persona_capabilities(
        self,
        actor: Actor,
        persona_name: str,
    ) -> OperationResult[PersonaCapabilities]:
        """
        Get the capabilities of a persona.

        Args:
            actor: The actor performing the operation
            persona_name: Name of the persona

        Returns:
            OperationResult containing PersonaCapabilities
        """
        result = await self.get_persona(actor, persona_name)
        if not result.success or result.data is None:
            return OperationResult.failure(result.error or "Unknown error", result.error_code)

        return OperationResult.success(result.data.capabilities)

    # =========================================================================
    # Private Helpers
    # =========================================================================

    def _build_persona_dict(self, persona: PersonaCreate) -> Dict[str, Any]:
        """Build a persona dictionary from PersonaCreate DTO."""
        data: Dict[str, Any] = {
            "name": persona.name,
            "content": persona.content,
        }

        if persona.meaning:
            data["meaning"] = persona.meaning
        if persona.provider:
            data["provider"] = persona.provider
        if persona.model:
            data["model"] = persona.model
        if persona.speech_provider:
            data["Speech_provider"] = persona.speech_provider
        if persona.voice:
            data["voice"] = persona.voice

        data["sys_info_enabled"] = persona.sys_info_enabled
        data["user_profile_enabled"] = persona.user_profile_enabled

        if persona.allowed_tools:
            data["allowed_tools"] = persona.allowed_tools
        if persona.allowed_skills:
            data["allowed_skills"] = persona.allowed_skills
        if persona.persona_type:
            data["type"] = persona.persona_type
        if persona.image_generation:
            data["image_generation"] = persona.image_generation
        if persona.ui_state:
            data["ui_state"] = persona.ui_state

        return data

    def _apply_update(
        self,
        existing: Dict[str, Any],
        update: PersonaUpdate,
    ) -> tuple[Dict[str, Any], List[str]]:
        """Apply update to existing persona dict and track changed fields."""
        result = dict(existing)
        changed: List[str] = []

        if update.meaning is not None and update.meaning != existing.get("meaning"):
            result["meaning"] = update.meaning
            changed.append("meaning")

        if update.content is not None:
            if update.content != existing.get("content"):
                result["content"] = update.content
                changed.append("content")

        if update.provider is not None and update.provider != existing.get("provider"):
            result["provider"] = update.provider
            changed.append("provider")

        if update.model is not None and update.model != existing.get("model"):
            result["model"] = update.model
            changed.append("model")

        if update.speech_provider is not None and update.speech_provider != existing.get("Speech_provider"):
            result["Speech_provider"] = update.speech_provider
            changed.append("speech_provider")

        if update.voice is not None and update.voice != existing.get("voice"):
            result["voice"] = update.voice
            changed.append("voice")

        if update.sys_info_enabled is not None and update.sys_info_enabled != existing.get("sys_info_enabled"):
            result["sys_info_enabled"] = update.sys_info_enabled
            changed.append("sys_info_enabled")

        if update.user_profile_enabled is not None and update.user_profile_enabled != existing.get("user_profile_enabled"):
            result["user_profile_enabled"] = update.user_profile_enabled
            changed.append("user_profile_enabled")

        if update.allowed_tools is not None:
            if update.allowed_tools != existing.get("allowed_tools"):
                result["allowed_tools"] = update.allowed_tools
                changed.append("allowed_tools")

        if update.allowed_skills is not None:
            if update.allowed_skills != existing.get("allowed_skills"):
                result["allowed_skills"] = update.allowed_skills
                changed.append("allowed_skills")

        if update.persona_type is not None:
            if update.persona_type != existing.get("type"):
                result["type"] = update.persona_type
                changed.append("type")

        if update.image_generation is not None:
            if update.image_generation != existing.get("image_generation"):
                result["image_generation"] = update.image_generation
                changed.append("image_generation")

        if update.ui_state is not None:
            if update.ui_state != existing.get("ui_state"):
                result["ui_state"] = update.ui_state
                changed.append("ui_state")

        if update.name is not None and update.name != existing.get("name"):
            changed.append("name")

        return result, changed

    def _rename_persona_directory(self, old_name: str, new_name: str) -> None:
        """Rename a persona directory."""
        personas_root = self._get_personas_root()
        old_dir = personas_root / old_name
        new_dir = personas_root / new_name

        if old_dir.exists() and not new_dir.exists():
            shutil.move(str(old_dir), str(new_dir))
            # Also rename the JSON file inside
            old_json = new_dir / "Persona" / f"{old_name}.json"
            new_json = new_dir / "Persona" / f"{new_name}.json"
            if old_json.exists():
                shutil.move(str(old_json), str(new_json))
