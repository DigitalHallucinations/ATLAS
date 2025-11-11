"""Shared helpers for manifest loader modules."""

from __future__ import annotations

import copy
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple


__all__ = [
    "Draft7Validator",
    "FallbackDraft7Validator",
    "ValidationError",
    "ensure_yaml_stub",
    "get_manifest_logger",
    "coerce_string",
    "coerce_string_tuple",
    "iter_persona_manifest_paths",
    "merge_with_base",
    "resolve_app_root",
]


try:  # Prefer the real jsonschema implementation when available
    from jsonschema import Draft7Validator as _Draft7Validator
    from jsonschema import ValidationError as _ValidationError
except (ModuleNotFoundError, ImportError):  # pragma: no cover - optional dependency
    _Draft7Validator = None  # type: ignore[assignment]
    _ValidationError = None  # type: ignore[assignment]


class ValidationError(Exception):
    """Lightweight substitute mirroring :class:`jsonschema.ValidationError`."""

    def __init__(self, message: str, path: Optional[List[Any]] = None):
        super().__init__(message)
        self.message = message
        self.path = tuple(path or [])


class FallbackDraft7Validator:
    """Very small subset of :class:`jsonschema.Draft7Validator` used in tests."""

    SUPPORTED_TYPES = {"object", "array", "string", "boolean", "integer", "number"}

    def __init__(self, schema: dict[str, Any]):
        self.schema = schema

    def iter_errors(self, instance: Any) -> Iterable[ValidationError]:
        yield from _validate_with_schema(instance, self.schema, [])


def _validate_with_schema(instance: Any, schema: dict[str, Any], path: List[Any]):
    schema_type = schema.get("type")

    if isinstance(schema_type, list):
        collected: List[ValidationError] = []
        for subtype in schema_type:
            nested_schema = dict(schema)
            nested_schema["type"] = subtype
            errors = list(_validate_with_schema(instance, nested_schema, path))
            if not errors:
                return
            collected.extend(errors)
        for error in collected:
            yield error
        return

    if schema_type == "object":
        if not isinstance(instance, dict):
            yield ValidationError("Expected object", path)
            return

        required = schema.get("required", [])
        for key in required:
            if key not in instance:
                yield ValidationError(f"'{key}' is a required property", path + [key])

        properties = schema.get("properties", {})
        allow_additional = schema.get("additionalProperties", True)
        for key, value in instance.items():
            subschema = properties.get(key)
            if subschema is None:
                if allow_additional is False:
                    yield ValidationError(
                        f"Additional property '{key}' is not allowed", path + [key]
                    )
                continue
            yield from _validate_with_schema(value, subschema, path + [key])
        return

    if schema_type == "array":
        if not isinstance(instance, list):
            yield ValidationError("Expected array", path)
            return

        min_items = schema.get("minItems")
        if isinstance(min_items, int) and len(instance) < min_items:
            yield ValidationError("Array has too few items", path)

        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(instance):
                yield from _validate_with_schema(item, item_schema, path + [index])
        return

    if schema_type == "string":
        if not isinstance(instance, str):
            yield ValidationError("Expected string", path)
            return

        min_length = schema.get("minLength")
        if isinstance(min_length, int) and len(instance) < min_length:
            yield ValidationError("String is too short", path)

        enum = schema.get("enum")
        if enum is not None and instance not in enum:
            yield ValidationError("Value is not one of the allowed options", path)
        return

    if schema_type == "boolean":
        if not isinstance(instance, bool):
            yield ValidationError("Expected boolean", path)
        return

    if schema_type == "integer":
        if not (isinstance(instance, int) and not isinstance(instance, bool)):
            yield ValidationError("Expected integer", path)
            return

        minimum = schema.get("minimum")
        if isinstance(minimum, int) and instance < minimum:
            yield ValidationError("Value is less than the minimum", path)

        maximum = schema.get("maximum")
        if isinstance(maximum, int) and instance > maximum:
            yield ValidationError("Value exceeds the maximum", path)
        return

    if schema_type == "number":
        if not isinstance(instance, (int, float)) or isinstance(instance, bool):
            yield ValidationError("Expected number", path)
            return

        minimum = schema.get("minimum")
        if isinstance(minimum, (int, float)) and instance < minimum:
            yield ValidationError("Value is less than the minimum", path)

        maximum = schema.get("maximum")
        if isinstance(maximum, (int, float)) and instance > maximum:
            yield ValidationError("Value exceeds the maximum", path)
        return

    # Unknown or unsupported schema keywords fall back to no-op validation.
    return


if _Draft7Validator is not None and _ValidationError is not None:  # pragma: no branch
    Draft7Validator = _Draft7Validator
    ValidationError = _ValidationError  # type: ignore[assignment]
else:  # pragma: no cover - exercised when jsonschema is unavailable
    Draft7Validator = FallbackDraft7Validator


def ensure_yaml_stub() -> None:
    """Provide a minimal stub for :mod:`yaml` when the dependency is absent."""

    if "yaml" not in sys.modules:
        sys.modules["yaml"] = SimpleNamespace(
            safe_load=lambda *_args, **_kwargs: {},
            dump=lambda *_args, **_kwargs: None,
        )


ensure_yaml_stub()


from modules.logging.logger import setup_logger

try:  # ConfigManager may not be available in certain test scenarios
    from ATLAS.config import ConfigManager as _ConfigManager
except Exception:  # pragma: no cover - defensive import guard
    _ConfigManager = None  # type: ignore


def get_manifest_logger(name: str) -> logging.Logger:
    """Return a logger configured for manifest loader modules."""

    ensure_yaml_stub()
    return setup_logger(name)


def resolve_app_root(config_manager=None, *, logger: Optional[logging.Logger] = None) -> Path:
    """Resolve the application root using :class:`ConfigManager` fallbacks."""

    ensure_yaml_stub()
    active_logger = logger or get_manifest_logger(__name__)

    if config_manager is not None:
        getter = getattr(config_manager, "get_app_root", None)
        if callable(getter):
            try:
                root = getter()
                candidate = _validate_app_root(root)
                if candidate is not None:
                    return candidate
            except Exception:  # pragma: no cover - defensive guard
                active_logger.warning(
                    "Failed to resolve app root from supplied config manager", exc_info=True
                )

    if _ConfigManager is not None:
        try:
            manager = config_manager or _ConfigManager()
            getter = getattr(manager, "get_app_root", None)
            if callable(getter):
                candidate = _validate_app_root(getter())
                if candidate is not None:
                    return candidate
        except Exception:  # pragma: no cover - defensive guard
            active_logger.warning("Unable to resolve app root via ConfigManager", exc_info=True)

    fallback = Path(__file__).resolve().parents[2]
    active_logger.debug("Falling back to computed app root at %s", fallback)
    return fallback


def coerce_string(value: Any) -> str:
    """Return a trimmed string representation for manifest fields."""

    if value is None:
        return ""
    return str(value).strip()


def coerce_string_tuple(value: Any) -> Tuple[str, ...]:
    """Normalize manifest values that should be sequences of strings."""

    if not value:
        return tuple()
    if isinstance(value, str):
        value = [value]

    result: List[str] = []
    for item in value:
        text = coerce_string(item)
        if text:
            result.append(text)
    return tuple(result)


def merge_with_base(
    entry: Mapping[str, Any], known_entries: Mapping[str, Mapping[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Merge manifest entries with a referenced base entry when present."""

    name_value = entry.get("name")
    extends_value = entry.get("extends")

    base_key: Optional[str] = None
    if isinstance(extends_value, str) and extends_value.strip():
        base_key = extends_value.strip()
    elif isinstance(name_value, str) and name_value.strip() in known_entries:
        base_key = name_value.strip()

    merged: Dict[str, Any]
    if base_key:
        base_entry = known_entries.get(base_key)
        if base_entry is None:
            return None
        merged = copy.deepcopy(dict(base_entry))
    else:
        merged = {}

    merged.update({k: v for k, v in entry.items() if k != "extends"})

    if "name" not in merged and isinstance(name_value, str):
        merged["name"] = name_value

    merged.pop("extends", None)
    return merged


def iter_persona_manifest_paths(
    app_root: Path, *relative_parts: str
) -> Iterator[Tuple[str, Path]]:
    """Yield persona names with the corresponding manifest path."""

    personas_root = app_root / "modules" / "Personas"
    if not personas_root.is_dir():
        return

    for persona_dir in sorted(personas_root.iterdir()):
        if not persona_dir.is_dir():
            continue
        manifest_path = persona_dir
        if relative_parts:
            manifest_path = manifest_path.joinpath(*relative_parts)
        yield persona_dir.name, manifest_path


def _validate_app_root(root: Optional[str]) -> Optional[Path]:
    if not root:
        return None

    candidate = Path(root).expanduser().resolve()
    modules_dir = candidate / "modules"
    if modules_dir.exists():
        return candidate
    return None
