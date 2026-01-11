"""
Persona service exceptions.

Provides specific exception types for persona operations,
built on the common service exception hierarchy.

Author: ATLAS Team
Date: Jan 10, 2026
"""

from typing import Any, Dict, List, Optional

from core.services.common.exceptions import (
    ServiceError,
    NotFoundError,
    ValidationError,
)


class PersonaError(ServiceError):
    """
    Base exception for all persona-related errors.

    Provides a clean interface that doesn't leak implementation details
    about file system operations or JSON handling.
    """

    pass


class PersonaNotFoundError(PersonaError, NotFoundError):
    """
    Raised when a requested persona cannot be found.

    Example:
        raise PersonaNotFoundError("MEDIC")
    """

    def __init__(
        self,
        persona_name: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        message = f"Persona '{persona_name}' not found"
        super().__init__(
            message=message,
            error_code=error_code or "PERSONA_NOT_FOUND",
            details={"persona_name": persona_name, **(details or {})},
        )
        self.persona_name = persona_name


class PersonaValidationError(PersonaError, ValidationError):
    """
    Raised when persona validation fails.

    Contains detailed information about what validation rules were violated.

    Example:
        raise PersonaValidationError(
            persona_name="TestPersona",
            errors=["Missing required field: content.editable_content"],
            warnings=["Field 'voice' is deprecated"],
        )
    """

    def __init__(
        self,
        persona_name: Optional[str] = None,
        errors: Optional[List[str]] = None,
        warnings: Optional[List[str]] = None,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        errors = errors or []
        warnings = warnings or []
        
        error_summary = "; ".join(errors) if errors else "Unknown validation error"
        if persona_name:
            message = f"Persona '{persona_name}' validation failed: {error_summary}"
        else:
            message = f"Persona validation failed: {error_summary}"
        
        super().__init__(
            message=message,
            error_code=error_code or "PERSONA_VALIDATION_ERROR",
            details={
                "persona_name": persona_name,
                "errors": errors,
                "warnings": warnings,
                **(details or {}),
            },
        )
        self.persona_name = persona_name
        self.errors = errors
        self.warnings = warnings


class PersonaAlreadyExistsError(PersonaError):
    """
    Raised when attempting to create a persona that already exists.

    Example:
        raise PersonaAlreadyExistsError("ATLAS")
    """

    def __init__(
        self,
        persona_name: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        message = f"Persona '{persona_name}' already exists"
        super().__init__(
            message=message,
            error_code=error_code or "PERSONA_ALREADY_EXISTS",
            details={"persona_name": persona_name, **(details or {})},
        )
        self.persona_name = persona_name


class PersonaDeleteError(PersonaError):
    """
    Raised when persona deletion fails.

    Example:
        raise PersonaDeleteError("ATLAS", "Cannot delete system persona")
    """

    def __init__(
        self,
        persona_name: str,
        reason: Optional[str] = None,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        message = f"Cannot delete persona '{persona_name}'"
        if reason:
            message += f": {reason}"
        super().__init__(
            message=message,
            error_code=error_code or "PERSONA_DELETE_ERROR",
            details={"persona_name": persona_name, "reason": reason, **(details or {})},
        )
        self.persona_name = persona_name
        self.reason = reason


class PersonaActiveError(PersonaError):
    """
    Raised when an operation is invalid due to persona active state.

    Example:
        raise PersonaActiveError("Cannot delete active persona 'ATLAS'")
    """

    def __init__(
        self,
        message: str,
        persona_name: Optional[str] = None,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code=error_code or "PERSONA_ACTIVE_ERROR",
            details={"persona_name": persona_name, **(details or {})},
        )
        self.persona_name = persona_name


class PersonaIOError(PersonaError):
    """
    Raised when file I/O operations fail for personas.

    Example:
        raise PersonaIOError(
            "Failed to write persona file",
            persona_name="MEDIC",
            operation="write",
        )
    """

    def __init__(
        self,
        message: str,
        persona_name: Optional[str] = None,
        operation: Optional[str] = None,  # read, write, delete
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code=error_code or "PERSONA_IO_ERROR",
            details={
                "persona_name": persona_name,
                "operation": operation,
                **(details or {}),
            },
        )
        self.persona_name = persona_name
        self.operation = operation


class PersonaSchemaError(PersonaError):
    """
    Raised when the persona schema itself is invalid or missing.

    This is typically a configuration issue rather than a user error.

    Example:
        raise PersonaSchemaError("Failed to load persona schema from disk")
    """

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code=error_code or "PERSONA_SCHEMA_ERROR",
            details=details,
        )


# =============================================================================
# SOTA Safety Exceptions
# =============================================================================


class PersonaSafetyError(PersonaError):
    """
    Base exception for persona safety-related errors.

    Example:
        raise PersonaSafetyError("Safety policy violation detected")
    """

    def __init__(
        self,
        message: str,
        persona_name: Optional[str] = None,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code=error_code or "PERSONA_SAFETY_ERROR",
            details={"persona_name": persona_name, **(details or {})},
        )
        self.persona_name = persona_name


class ActionBlockedError(PersonaSafetyError):
    """
    Raised when an action is blocked by safety policy.

    Example:
        raise ActionBlockedError(
            action_type="delete_all",
            persona_name="ATLAS",
            reason="Action is on blocked list",
        )
    """

    def __init__(
        self,
        action_type: str,
        persona_name: Optional[str] = None,
        reason: Optional[str] = None,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        message = f"Action '{action_type}' is blocked"
        if reason:
            message += f": {reason}"
        super().__init__(
            message=message,
            persona_name=persona_name,
            error_code=error_code or "ACTION_BLOCKED",
            details={
                "action_type": action_type,
                "reason": reason,
                **(details or {}),
            },
        )
        self.action_type = action_type
        self.reason = reason


class ApprovalRequiredError(PersonaSafetyError):
    """
    Raised when an action requires approval before proceeding.

    Example:
        raise ApprovalRequiredError(
            action_type="financial_transfer",
            request_id="req_123",
            approver_role="supervisor",
        )
    """

    def __init__(
        self,
        action_type: str,
        request_id: str,
        approver_role: Optional[str] = None,
        persona_name: Optional[str] = None,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        message = f"Action '{action_type}' requires approval (request: {request_id})"
        super().__init__(
            message=message,
            persona_name=persona_name,
            error_code=error_code or "APPROVAL_REQUIRED",
            details={
                "action_type": action_type,
                "request_id": request_id,
                "approver_role": approver_role,
                **(details or {}),
            },
        )
        self.action_type = action_type
        self.request_id = request_id
        self.approver_role = approver_role


# =============================================================================
# SOTA Memory Exceptions
# =============================================================================


class PersonaMemoryError(PersonaError):
    """
    Base exception for persona memory-related errors.

    Example:
        raise PersonaMemoryError("Failed to retrieve working memory")
    """

    def __init__(
        self,
        message: str,
        persona_name: Optional[str] = None,
        user_id: Optional[str] = None,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code=error_code or "PERSONA_MEMORY_ERROR",
            details={
                "persona_name": persona_name,
                "user_id": user_id,
                **(details or {}),
            },
        )
        self.persona_name = persona_name
        self.user_id = user_id


# =============================================================================
# SOTA Analytics Exceptions
# =============================================================================


class PersonaAnalyticsError(PersonaError):
    """
    Base exception for persona analytics-related errors.

    Example:
        raise PersonaAnalyticsError("Failed to record interaction metrics")
    """

    def __init__(
        self,
        message: str,
        persona_name: Optional[str] = None,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code=error_code or "PERSONA_ANALYTICS_ERROR",
            details={"persona_name": persona_name, **(details or {})},
        )
        self.persona_name = persona_name


# =============================================================================
# SOTA Switching Exceptions
# =============================================================================


class PersonaSwitchingError(PersonaError):
    """
    Base exception for persona switching-related errors.

    Example:
        raise PersonaSwitchingError("Invalid handoff state transition")
    """

    def __init__(
        self,
        message: str,
        from_persona: Optional[str] = None,
        to_persona: Optional[str] = None,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code=error_code or "PERSONA_SWITCHING_ERROR",
            details={
                "from_persona": from_persona,
                "to_persona": to_persona,
                **(details or {}),
            },
        )
        self.from_persona = from_persona
        self.to_persona = to_persona


class HandoffFailedError(PersonaSwitchingError):
    """
    Raised when a persona handoff fails.

    Example:
        raise HandoffFailedError(
            handoff_id="hoff_123",
            from_persona="PersonaA",
            to_persona="PersonaB",
            reason="Target persona unavailable",
        )
    """

    def __init__(
        self,
        handoff_id: str,
        from_persona: Optional[str] = None,
        to_persona: Optional[str] = None,
        reason: Optional[str] = None,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        message = f"Handoff '{handoff_id}' failed"
        if reason:
            message += f": {reason}"
        super().__init__(
            message=message,
            from_persona=from_persona,
            to_persona=to_persona,
            error_code=error_code or "HANDOFF_FAILED",
            details={
                "handoff_id": handoff_id,
                "reason": reason,
                **(details or {}),
            },
        )
        self.handoff_id = handoff_id
        self.reason = reason