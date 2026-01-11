"""
Persona validation utilities.

Wraps the lower-level modules/Personas validation logic and provides
a clean service-layer interface for persona validation.

Author: ATLAS Team
Date: Jan 10, 2026
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence

from .exceptions import PersonaValidationError, PersonaSchemaError
from .types import ValidationResult

if TYPE_CHECKING:
    from core.config import ConfigManager


logger = logging.getLogger(__name__)


def _get_validation_deps():
    """Lazy import validation dependencies to avoid circular imports."""
    from modules.Personas import (
        PersonaValidationError as ModuleValidationError,
        _validate_persona_payload,
        load_tool_metadata,
        load_skill_catalog,
    )
    return {
        "ModuleValidationError": ModuleValidationError,
        "_validate_persona_payload": _validate_persona_payload,
        "load_tool_metadata": load_tool_metadata,
        "load_skill_catalog": load_skill_catalog,
    }


class PersonaValidator:
    """
    Validates persona definitions against the schema.

    Wraps the lower-level validation logic from modules/Personas and
    provides a consistent service-layer interface.
    """

    def __init__(
        self,
        config_manager: Optional["ConfigManager"] = None,
        *,
        tool_ids: Optional[Sequence[str]] = None,
        skill_ids: Optional[Sequence[str]] = None,
    ) -> None:
        """
        Initialize the validator.

        Args:
            config_manager: Optional config manager for path resolution
            tool_ids: Known tool IDs for validation. If None, loaded from metadata.
            skill_ids: Known skill IDs for validation. If None, loaded from catalog.
        """
        self._config_manager = config_manager
        self._tool_ids = list(tool_ids) if tool_ids else None
        self._skill_ids = list(skill_ids) if skill_ids else None
        self._skill_catalog: Optional[Dict[str, Any]] = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Load tool and skill metadata if not already loaded."""
        if self._initialized:
            return

        deps = _get_validation_deps()

        if self._tool_ids is None:
            try:
                order, _ = deps["load_tool_metadata"](config_manager=self._config_manager)
                self._tool_ids = list(order)
            except Exception as e:
                logger.warning("Failed to load tool metadata: %s", e)
                self._tool_ids = []

        if self._skill_ids is None:
            try:
                order, catalog = deps["load_skill_catalog"](config_manager=self._config_manager)
                self._skill_ids = list(order)
                self._skill_catalog = dict(catalog)
            except Exception as e:
                logger.warning("Failed to load skill catalog: %s", e)
                self._skill_ids = []
                self._skill_catalog = {}

        self._initialized = True

    def validate(
        self,
        persona_data: Mapping[str, Any],
        *,
        persona_name: Optional[str] = None,
        raise_on_error: bool = False,
    ) -> ValidationResult:
        """
        Validate a persona definition.

        Args:
            persona_data: The persona definition to validate. Can be either:
                - A full wrapper {"persona": [...]}
                - A single persona entry dict
            persona_name: Name of the persona (for error messages)
            raise_on_error: If True, raise PersonaValidationError on failure

        Returns:
            ValidationResult with validation outcome

        Raises:
            PersonaValidationError: If raise_on_error=True and validation fails
        """
        self._ensure_initialized()
        deps = _get_validation_deps()

        # Normalize to wrapper format if needed
        if "persona" not in persona_data:
            # Single persona entry - wrap it
            persona_payload = {"persona": [persona_data]}
        else:
            persona_payload = persona_data

        # Extract name from data if not provided
        if persona_name is None:
            personas = persona_payload.get("persona", [])
            if personas and isinstance(personas, list) and len(personas) > 0:
                first_persona = personas[0]
                if isinstance(first_persona, Mapping):
                    persona_name = first_persona.get("name", "Unknown")

        errors: List[str] = []
        warnings: List[str] = []

        try:
            deps["_validate_persona_payload"](
                persona_payload,
                persona_name=persona_name or "Unknown",
                tool_ids=self._tool_ids or [],
                skill_ids=self._skill_ids,
                skill_catalog=self._skill_catalog,
                config_manager=self._config_manager,
            )
        except deps["ModuleValidationError"] as e:
            # Parse the error message to extract individual errors
            error_msg = str(e)
            if "failed schema validation:" in error_msg:
                # Extract individual error lines
                lines = error_msg.split("\n")
                for line in lines[1:]:  # Skip the header line
                    line = line.strip()
                    if line.startswith("- "):
                        line = line[2:]
                    if line:
                        errors.append(line)
            else:
                errors.append(error_msg)

        is_valid = len(errors) == 0
        result = ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            persona_name=persona_name,
        )

        if raise_on_error and not is_valid:
            raise PersonaValidationError(
                persona_name=persona_name,
                errors=errors,
                warnings=warnings,
            )

        return result

    def validate_name(self, name: str) -> ValidationResult:
        """
        Validate a persona name for filesystem safety.

        Args:
            name: The persona name to validate

        Returns:
            ValidationResult with validation outcome
        """
        errors: List[str] = []
        warnings: List[str] = []

        if not name or not name.strip():
            errors.append("Persona name must be a non-empty string")
        else:
            normalized = name.strip()
            if normalized in {".", ".."}:
                errors.append("Persona name cannot reference relative directories")
            if "/" in normalized or "\\" in normalized:
                errors.append("Persona name cannot contain path separators")
            if ".." in normalized:
                errors.append("Persona name cannot include traversal sequences")

            unsafe_chars = set('<>:"|?*')
            found_unsafe = [c for c in normalized if c in unsafe_chars]
            if found_unsafe:
                errors.append(f"Persona name contains unsupported characters: {found_unsafe}")

            if any(ord(c) < 32 for c in normalized):
                errors.append("Persona name contains control characters")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            persona_name=name,
        )

    def validate_content(self, content: Mapping[str, Any]) -> ValidationResult:
        """
        Validate persona content structure.

        Args:
            content: The content dict to validate

        Returns:
            ValidationResult with validation outcome
        """
        errors: List[str] = []
        warnings: List[str] = []

        required_fields = ["start_locked", "editable_content", "end_locked"]
        for field in required_fields:
            if field not in content:
                errors.append(f"Missing required content field: {field}")
            elif not isinstance(content[field], str):
                errors.append(f"Content field '{field}' must be a string")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )


def create_validator(
    config_manager: Optional["ConfigManager"] = None,
    *,
    tool_ids: Optional[Sequence[str]] = None,
    skill_ids: Optional[Sequence[str]] = None,
) -> PersonaValidator:
    """
    Factory function to create a PersonaValidator.

    Args:
        config_manager: Optional config manager for path resolution
        tool_ids: Known tool IDs for validation
        skill_ids: Known skill IDs for validation

    Returns:
        Configured PersonaValidator instance
    """
    return PersonaValidator(
        config_manager=config_manager,
        tool_ids=tool_ids,
        skill_ids=skill_ids,
    )
