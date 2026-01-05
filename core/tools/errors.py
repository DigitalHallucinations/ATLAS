"""Shared tool-related exception definitions."""
from __future__ import annotations

from typing import Any, Dict, Optional


class ToolExecutionError(RuntimeError):
    """Exception raised when a tool call fails to execute successfully."""

    def __init__(
        self,
        message: str,
        *,
        tool_call_id: Optional[str] = None,
        function_name: Optional[str] = None,
        error_type: Optional[str] = None,
        entry: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.tool_call_id = tool_call_id
        self.function_name = function_name
        self.error_type = error_type
        self.entry = entry


class ToolManifestValidationError(RuntimeError):
    """Raised when a tool manifest fails schema validation."""

    def __init__(
        self,
        message: str,
        *,
        persona: Optional[str] = None,
        errors: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.persona = persona
        self.errors = errors or {}
