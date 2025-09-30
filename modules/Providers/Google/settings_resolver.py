"""Shared helpers for normalising Google Gemini configuration payloads."""

from __future__ import annotations

import json
from typing import Any, Dict, Mapping, Optional


class GoogleSettingsResolver:
    """Utility helpers for normalising Google Gemini configuration values."""

    def __init__(
        self,
        *,
        stored: Optional[Mapping[str, Any]] = None,
        defaults: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self._stored: Dict[str, Any] = dict(stored or {})
        self._defaults: Dict[str, Any] = dict(defaults or {})

    def resolve_float(
        self,
        key: str,
        provided: Optional[Any],
        *,
        field: str,
        minimum: float,
        maximum: float,
        allow_invalid_stored: bool = False,
    ) -> float:
        """Resolve a floating-point configuration value with bounds checking."""

        if provided is not None:
            return self._coerce_float(provided, field=field, minimum=minimum, maximum=maximum)

        candidate = self._stored.get(key, self._defaults.get(key))
        default_value = self._defaults.get(key)

        if candidate is None:
            candidate = default_value

        try:
            return self._coerce_float(candidate, field=field, minimum=minimum, maximum=maximum)
        except ValueError:
            if allow_invalid_stored and default_value is not None:
                return self._coerce_float(
                    default_value, field=field, minimum=minimum, maximum=maximum
                )
            raise

    def resolve_optional_int(
        self,
        key: str,
        provided: Optional[Any],
        *,
        field: str,
        minimum: int,
        allow_invalid_stored: bool = False,
    ) -> Optional[int]:
        """Resolve an optional integer ensuring it is above ``minimum`` when set."""

        if provided is not None:
            return self._coerce_optional_int(provided, field=field, minimum=minimum)

        candidate = self._stored.get(key, self._defaults.get(key))
        default_value = self._defaults.get(key)

        if candidate in {None, ""}:
            return None

        try:
            return self._coerce_optional_int(candidate, field=field, minimum=minimum)
        except ValueError:
            if allow_invalid_stored:
                if default_value in {None, ""}:
                    return None
                return self._coerce_optional_int(default_value, field=field, minimum=minimum)
            raise

    def resolve_int(
        self,
        key: str,
        provided: Optional[Any],
        *,
        field: str,
        minimum: int,
        allow_invalid_stored: bool = False,
    ) -> int:
        """Resolve a required integer ensuring it meets the ``minimum`` bound."""

        if provided is not None:
            return self._coerce_int(provided, field=field, minimum=minimum)

        default_value = self._defaults.get(key, minimum)
        candidate = self._stored.get(key, default_value)

        try:
            return self._coerce_int(candidate, field=field, minimum=minimum)
        except ValueError:
            if allow_invalid_stored:
                return self._coerce_int(default_value, field=field, minimum=minimum)
            raise

    def resolve_max_output_tokens(
        self,
        provided: Optional[Any],
        *,
        allow_invalid_stored: bool = False,
    ) -> Optional[int]:
        """Resolve the ``max_output_tokens`` field accepting blank values as ``None``."""

        if provided is not None:
            return self._coerce_max_output_tokens(provided)

        key = "max_output_tokens"
        default_value = self._defaults.get(key)
        if key in self._stored:
            try:
                return self._coerce_max_output_tokens(self._stored.get(key))
            except ValueError:
                if allow_invalid_stored:
                    return self._coerce_max_output_tokens(default_value)
                raise

        return self._coerce_max_output_tokens(default_value)

    def resolve_bool(
        self,
        key: str,
        provided: Optional[Any],
        *,
        default: Optional[bool] = None,
    ) -> bool:
        """Resolve a boolean flag falling back to stored state or defaults."""

        baseline = self._defaults.get(key) if default is None else default
        baseline = bool(baseline) if baseline is not None else False

        if provided is not None:
            return self._coerce_optional_bool(provided, default=baseline)

        if key in self._stored:
            return self._coerce_optional_bool(self._stored.get(key), default=baseline)

        return baseline

    def resolve_response_schema(
        self,
        provided: Optional[Any],
        *,
        key: str = "response_schema",
        allow_invalid_stored: bool = False,
    ) -> Dict[str, Any]:
        """Resolve a response schema mapping shared between config and runtime."""

        if provided is None:
            if key in self._stored:
                try:
                    return self._coerce_response_schema(self._stored.get(key))
                except ValueError:
                    if allow_invalid_stored:
                        return {}
                    raise
            return self._coerce_response_schema(self._defaults.get(key, {}))

        if provided in ({}, ""):
            return {}

        return self._coerce_response_schema(provided)

    @staticmethod
    def _coerce_float(
        value: Any,
        *,
        field: str,
        minimum: float,
        maximum: float,
    ) -> float:
        if value is None:
            raise ValueError(f"{field} must be a number.")

        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} must be a number.") from exc

        if parsed < minimum or parsed > maximum:
            raise ValueError(f"{field} must be between {minimum} and {maximum}.")

        return parsed

    @staticmethod
    def _coerce_optional_int(
        value: Any,
        *,
        field: str,
        minimum: int,
    ) -> Optional[int]:
        if value in {None, ""}:
            return None

        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} must be an integer or left blank.") from exc

        if parsed < minimum:
            raise ValueError(f"{field} must be >= {minimum}.")

        return parsed

    @staticmethod
    def _coerce_int(
        value: Any,
        *,
        field: str,
        minimum: int,
    ) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} must be an integer.") from exc

        if parsed < minimum:
            raise ValueError(f"{field} must be >= {minimum}.")

        return parsed

    @staticmethod
    def _coerce_max_output_tokens(value: Any) -> Optional[int]:
        if value is None or value == "":
            return None

        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("Max output tokens must be an integer or left blank.") from exc

        if parsed <= 0:
            return None

        return parsed

    @staticmethod
    def _coerce_optional_bool(value: Any, *, default: bool) -> bool:
        if isinstance(value, bool):
            return value

        if value is None:
            return default

        if isinstance(value, str):
            cleaned = value.strip().lower()
            if cleaned in {"", "none"}:
                return default
            if cleaned in {"true", "1", "yes", "on"}:
                return True
            if cleaned in {"false", "0", "no", "off"}:
                return False

        try:
            numeric = int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return bool(value)

        return bool(numeric)

    @staticmethod
    def _coerce_response_schema(value: Any) -> Dict[str, Any]:
        if value is None or value == "" or value == {}:
            return {}

        if isinstance(value, str):
            text = value.strip()
            if not text:
                return {}
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError("Response schema must be valid JSON.") from exc
            if not isinstance(parsed, Mapping):
                raise ValueError("Response schema JSON must describe an object.")
            return dict(parsed)

        if isinstance(value, Mapping):
            try:
                serialised = json.dumps(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Response schema must contain JSON-serialisable values."
                ) from exc
            parsed = json.loads(serialised)
            if not isinstance(parsed, Mapping):
                raise ValueError("Response schema must be a JSON object.")
            return dict(parsed)

        raise ValueError(
            "Response schema must be provided as a mapping or JSON object string."
        )
