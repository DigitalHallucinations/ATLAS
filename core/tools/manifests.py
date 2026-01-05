"""Manifest loading and validation utilities for tool execution."""
from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import os
import sys
import threading
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:  # Prefer the canonical jsonschema implementation when available
    from jsonschema import Draft7Validator, ValidationError
except (ModuleNotFoundError, ImportError):  # pragma: no cover - fallback implementation
    class ValidationError(Exception):
        """Minimal substitute mirroring jsonschema.ValidationError."""

        def __init__(self, message: str, path: Optional[Iterable[Any]] = None) -> None:
            super().__init__(message)
            self.message = message
            self.path = tuple(path or [])

        @property
        def absolute_path(self):  # pragma: no cover - compatibility shim
            return self.path

    class Draft7Validator:
        """Very small subset of jsonschema.Draft7Validator used by ToolManager."""

        def __init__(self, schema: Dict[str, Any]) -> None:
            self.schema = schema

        def iter_errors(self, instance: Any):
            yield from _validate_with_schema(instance, self.schema, [])

        def validate(self, instance: Any) -> None:
            for error in self.iter_errors(instance):
                raise error

    def _validate_with_schema(
        instance: Any, schema: Dict[str, Any], path: list[Any]
    ):
        schema_type = schema.get("type")

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
                    if not allow_additional:
                        yield ValidationError(
                            f"Additional property '{key}' is not allowed", path + [key]
                        )
                    continue
                yield from _validate_with_schema(value, subschema, path + [key])

        elif schema_type == "array":
            if not isinstance(instance, list):
                yield ValidationError("Expected array", path)
                return

            item_schema = schema.get("items")
            if item_schema is not None:
                for index, item in enumerate(instance):
                    yield from _validate_with_schema(item, item_schema, path + [index])

        elif schema_type == "string":
            if not isinstance(instance, str):
                yield ValidationError("Expected string", path)
                return
            min_length = schema.get("minLength")
            if min_length and len(instance) < min_length:
                yield ValidationError("String is too short", path)

        else:
            # Types we don't recognise are treated as pass-through.
            return

from modules.logging.logger import setup_logger
from modules.orchestration.capability_registry import get_capability_registry

from core.config import ConfigManager

from .errors import ToolManifestValidationError
from .cache import get_config_manager

logger = setup_logger(__name__)

_DEFAULT_FUNCTIONS_CACHE_KEY = "__default__"

_function_map_cache: Dict[str, Tuple[float, Optional[Tuple[str, ...]], Dict[str, Any]]] = {}
_function_payload_cache: Dict[str, Tuple[int, Optional[Tuple[str, ...]], Any]] = {}
_function_payload_cache_lock = threading.Lock()
_default_function_map_cache: Optional[Tuple[float, Dict[str, Any]]] = None
_default_function_map_lock = threading.Lock()
_tool_manifest_validator: Optional[Draft7Validator] = None
_tool_manifest_validator_lock = threading.Lock()


def _ensure_namespace_package(package_name: str, package_path: str) -> None:
    """Ensure a namespace package exists for persona toolboxes."""

    if not package_path:
        return

    module = sys.modules.get(package_name)
    if module is None:
        spec = importlib.machinery.ModuleSpec(
            package_name,
            loader=None,
            is_package=True,
        )
        spec.submodule_search_locations = [package_path]
        module = importlib.util.module_from_spec(spec)
        module.__path__ = [package_path]
        sys.modules[package_name] = module
        return

    existing_path = getattr(module, "__path__", None)
    if existing_path is None:
        module.__path__ = [package_path]
    elif package_path not in existing_path:
        module.__path__ = list(existing_path) + [package_path]


def _load_default_functions_payload(*, refresh: bool = False, config_manager=None):
    """Load the shared functions.json payload for default tools."""

    cache_key = _DEFAULT_FUNCTIONS_CACHE_KEY
    if refresh:
        with _function_payload_cache_lock:
            _function_payload_cache.pop(cache_key, None)

    registry = get_capability_registry(config_manager=config_manager)
    if refresh:
        registry.refresh(force=True)
    else:
        registry.refresh_if_stale()

    revision = registry.revision

    with _function_payload_cache_lock:
        cache_entry = _function_payload_cache.get(cache_key)
        if cache_entry and not refresh:
            cached_revision, _cached_signature, cached_payload = cache_entry
            if cached_revision == revision:
                return cached_payload

    payload = registry.get_tool_manifest_payload(persona=None)
    if payload is None:
        with _function_payload_cache_lock:
            _function_payload_cache.pop(cache_key, None)
        return None

    with _function_payload_cache_lock:
        _function_payload_cache[cache_key] = (revision, None, payload)
    return payload


def _get_tool_manifest_validator(config_manager=None):
    """Return a cached JSON schema validator for tool manifests."""

    global _tool_manifest_validator

    with _tool_manifest_validator_lock:
        if _tool_manifest_validator is not None:
            return _tool_manifest_validator

        try:
            app_root = get_config_manager(config_manager).get_app_root()
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning(
                "Unable to determine application root when loading tool manifest schema: %s",
                exc,
            )
            return None

        schema_path = os.path.join(
            app_root,
            "modules",
            "Tools",
            "tool_maps",
            "schema.json",
        )

        try:
            with open(schema_path, "r", encoding="utf-8") as schema_file:
                schema = json.load(schema_file)
        except FileNotFoundError:
            logger.warning("Tool manifest schema not found at path: %s", schema_path)
            return None
        except json.JSONDecodeError as exc:
            logger.error(
                "Invalid JSON in tool manifest schema at %s: %s",
                schema_path,
                exc,
                exc_info=True,
            )
            return None
        except Exception as exc:  # pragma: no cover - unexpected I/O errors
            logger.error(
                "Unexpected error loading tool manifest schema from %s: %s",
                schema_path,
                exc,
                exc_info=True,
            )
            return None

        try:
            _tool_manifest_validator = Draft7Validator(schema)
        except Exception as exc:  # pragma: no cover - schema compilation errors
            logger.error(
                "Failed to build tool manifest validator from schema %s: %s",
                schema_path,
                exc,
                exc_info=True,
            )
            _tool_manifest_validator = None

        return _tool_manifest_validator


def _build_metadata_lookup(
    manifest_payload: Optional[Iterable[Mapping[str, Any]]],
    *,
    persona: Optional[str],
    config_manager=None,
) -> Dict[str, Any]:
    """Construct lookup dictionary keyed by tool name for metadata access."""

    metadata_lookup: Dict[str, Any] = {}
    if not manifest_payload:
        return metadata_lookup

    for entry in manifest_payload:
        if not isinstance(entry, Mapping):
            continue
        name = entry.get("name")
        if not name:
            continue
        metadata = dict(entry)
        metadata.setdefault("persona", persona)
        metadata.setdefault("source", "manifest")
        metadata_lookup[name] = metadata
    return metadata_lookup


def _annotate_function_map(
    function_map: Mapping[str, Any],
    *,
    metadata_lookup: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Return a copy of ``function_map`` with manifest metadata merged in."""

    if not metadata_lookup:
        return dict(function_map)

    annotated: Dict[str, Any] = {}
    for name, entry in function_map.items():
        if isinstance(entry, Mapping):
            annotated_entry = dict(entry)
        else:
            annotated_entry = {"callable": entry}
        metadata = metadata_lookup.get(name)
        if metadata:
            annotated_entry.setdefault("metadata", {}).update(metadata)
        annotated[name] = annotated_entry
    return annotated


def load_default_function_map(*, refresh=False, config_manager=None):
    """Return the default function map, optionally refreshing cached values."""

    global _default_function_map_cache

    if refresh:
        with _default_function_map_lock:
            _default_function_map_cache = None

    try:
        app_root = get_config_manager(config_manager).get_app_root()
    except Exception as exc:
        logger.error(
            "Unable to determine application root when loading default function map: %s",
            exc,
        )
        return None

    function_map_path = Path(app_root) / "modules" / "Tools" / "tool_maps" / "maps.py"

    try:
        mtime = function_map_path.stat().st_mtime
    except FileNotFoundError:
        logger.error("Function map file not found: %s", function_map_path)
        return None

    with _default_function_map_lock:
        cache_entry = _default_function_map_cache
        if cache_entry and not refresh:
            cached_mtime, cached_map = cache_entry
            if cached_mtime == mtime:
                logger.debug("Returning cached default function map")
                metadata_lookup = _build_metadata_lookup(
                    _load_default_functions_payload(refresh=refresh, config_manager=config_manager),
                    persona=None,
                    config_manager=config_manager,
                )
                return _annotate_function_map(cached_map, metadata_lookup=metadata_lookup)

    spec = importlib.util.spec_from_file_location("modules.Tools.tool_maps.maps", function_map_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load function map from {function_map_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    function_map = getattr(module, "function_map", None)
    if not isinstance(function_map, Mapping):
        logger.error("Function map missing or invalid in %s", function_map_path)
        return None

    with _default_function_map_lock:
        _default_function_map_cache = (mtime, dict(function_map))

    metadata_lookup = _build_metadata_lookup(
        _load_default_functions_payload(refresh=refresh, config_manager=config_manager),
        persona=None,
        config_manager=config_manager,
    )
    return _annotate_function_map(function_map, metadata_lookup=metadata_lookup)


def load_functions_from_json(
    current_persona,
    *,
    refresh=False,
    config_manager=None,
):
    logger.debug("Loading functions from JSON for persona manifest")

    persona_name = None
    allowed_names = None

    if isinstance(current_persona, Mapping):
        persona_name = current_persona.get("name")
        allowed = current_persona.get("allowed_tools") or current_persona.get("allowed_functions")
        if isinstance(allowed, Iterable):
            allowed_names = tuple(str(item) for item in allowed if item)

    if persona_name is None:
        payload = _load_default_functions_payload(refresh=refresh, config_manager=config_manager)
        return payload or []

    registry = get_capability_registry(config_manager=config_manager)
    if refresh:
        registry.refresh(force=True)
    else:
        registry.refresh_if_stale()

    revision = registry.revision
    allowed_signature = allowed_names

    with _function_payload_cache_lock:
        if not refresh:
            cache_entry = _function_payload_cache.get(persona_name)
            if cache_entry:
                cached_revision, cached_signature, cached_functions = cache_entry
                if cached_revision == revision and cached_signature == allowed_signature:
                    logger.debug(
                        "Returning cached functions for persona '%s' (revision %s)",
                        persona_name,
                        cached_revision,
                    )
                    return cached_functions

    try:
        if registry.persona_has_tool_manifest(persona_name):
            validator = _get_tool_manifest_validator(config_manager=config_manager)
            if validator is not None and hasattr(validator, "validate"):
                try:
                    manifest_payload = registry.get_tool_manifest_payload(
                        persona=persona_name,
                        allowed_names=None,
                    ) or []
                    validator.validate(manifest_payload)
                except ValidationError as exc:
                    error_details = {
                        "persona": persona_name,
                        "path": list(getattr(exc, "absolute_path", []) or []),
                        "message": getattr(exc, "message", str(exc)),
                    }
                    logger.error(
                        "Tool manifest validation error for persona '%s': %s",
                        persona_name,
                        error_details,
                    )
                    with _function_payload_cache_lock:
                        _function_payload_cache.pop(persona_name, None)
                    raise ToolManifestValidationError(
                        f"Invalid tool manifest for persona '{persona_name}': {error_details['message']}",
                        persona=persona_name,
                        errors=error_details,
                    ) from exc

        selected = registry.get_tool_manifest_payload(
            persona=persona_name,
            allowed_names=allowed_names,
        ) or []

        with _function_payload_cache_lock:
            _function_payload_cache[persona_name] = (
                revision,
                allowed_signature,
                selected,
            )

        return selected
    except ToolManifestValidationError:
        raise
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error(
            "Unexpected error loading functions for persona '%s': %s",
            persona_name,
            exc,
            exc_info=True,
        )
        with _function_payload_cache_lock:
            _function_payload_cache.pop(persona_name, None)
    return None


def load_function_map_from_current_persona(
    current_persona,
    *,
    refresh=False,
    config_manager=None,
):
    logger.debug("Attempting to load function map from current persona")
    if not current_persona or "name" not in current_persona:
        logger.error("Current persona is None or does not have a 'name' key")
        return load_default_function_map(refresh=refresh, config_manager=config_manager)

    persona_name = current_persona["name"]
    allowed_names = current_persona.get("allowed_tools")
    if isinstance(allowed_names, Iterable):
        allowed_signature = tuple(str(item) for item in allowed_names if item)
        allowed_names = list(allowed_signature)
    else:
        allowed_signature = None
    try:
        app_root = get_config_manager(config_manager).get_app_root()
    except Exception as exc:
        logger.error(
            "Unable to determine application root when loading persona '%s': %s",
            persona_name,
            exc,
        )
        return load_default_function_map(refresh=refresh, config_manager=config_manager)

    toolbox_root = os.path.join(app_root, "modules", "Personas", persona_name, "Toolbox")
    maps_path = os.path.join(toolbox_root, "maps.py")
    toolbox_package = f"modules.Personas.{persona_name}.Toolbox"
    module_name = f"{toolbox_package}.maps"
    try:
        if refresh:
            logger.debug(
                "Refresh requested for persona '%s'; clearing cached module and function map",
                persona_name,
            )
            sys.modules.pop(module_name, None)
            _function_map_cache.pop(persona_name, None)
            _function_payload_cache.pop(persona_name, None)

        file_mtime = os.path.getmtime(maps_path)

        cache_entry = _function_map_cache.get(persona_name)

        if not refresh and cache_entry:
            cached_mtime, cached_signature, cached_map = cache_entry
            if cached_mtime == file_mtime and cached_signature == allowed_signature:
                logger.debug(
                    "Returning cached function map for persona '%s' without reloading module",
                    persona_name,
                )
                metadata_lookup = _build_metadata_lookup(
                    load_functions_from_json(
                        current_persona,
                        refresh=refresh,
                        config_manager=config_manager,
                    ),
                    persona=persona_name,
                    config_manager=config_manager,
                )
                return _annotate_function_map(
                    cached_map,
                    metadata_lookup=metadata_lookup,
                )

            logger.debug(
                "Detected updated maps.py for persona '%s' (cached mtime %s, current mtime %s); reloading",
                persona_name,
                cached_mtime,
                file_mtime,
            )
            sys.modules.pop(module_name, None)
            _function_map_cache.pop(persona_name, None)

        module = sys.modules.get(module_name)

        if module is None:
            logger.debug(
                "Module '%s' not found in sys.modules; loading from '%s'",
                module_name,
                maps_path,
            )
            for package_name, package_path in [
                ("modules.Personas", os.path.join(app_root, "modules", "Personas")),
                (
                    f"modules.Personas.{persona_name}",
                    os.path.join(app_root, "modules", "Personas", persona_name),
                ),
                (toolbox_package, toolbox_root),
            ]:
                _ensure_namespace_package(package_name, package_path)

            spec = importlib.util.spec_from_file_location(module_name, maps_path)
            if spec is None or spec.loader is None:
                raise ImportError(
                    f"Could not load specification for persona '{persona_name}' from {maps_path}"
                )
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
        else:
            logger.debug(
                "Reusing already loaded module '%s' from sys.modules for persona '%s'",
                module_name,
                persona_name,
            )

        if hasattr(module, "function_map"):
            logger.debug(
                "Function map successfully loaded for persona '%s': %s",
                persona_name,
                module.function_map,
            )
            filtered_map = _filter_function_map_by_allowlist(
                module.function_map, allowed_names
            )
            _function_map_cache[persona_name] = (
                file_mtime,
                allowed_signature,
                filtered_map,
            )
        else:
            logger.error("function_map attribute missing in module '%s'", module_name)
            return load_default_function_map(refresh=refresh, config_manager=config_manager)

        metadata_lookup = _build_metadata_lookup(
            load_functions_from_json(
                current_persona,
                refresh=refresh,
                config_manager=config_manager,
            ),
            persona=persona_name,
            config_manager=config_manager,
        )
        return _annotate_function_map(
            module.function_map,
            metadata_lookup=metadata_lookup,
        )
    except FileNotFoundError:
        logger.error(
            "maps.py not found for persona '%s' at path '%s'",
            persona_name,
            maps_path,
        )
    except Exception as exc:
        logger.error(
            "Error loading function map for persona '%s': %s",
            persona_name,
            exc,
            exc_info=True,
        )
        _function_map_cache.pop(persona_name, None)
    return load_default_function_map(refresh=refresh, config_manager=config_manager)


def _filter_function_map_by_allowlist(function_map, allowed_names):
    if not allowed_names:
        return function_map
    filtered_map = {}
    for name, definition in function_map.items():
        if name in allowed_names:
            filtered_map[name] = definition
    return filtered_map


__all__ = [
    "Draft7Validator",
    "ValidationError",
    "load_default_function_map",
    "load_function_map_from_current_persona",
    "load_functions_from_json",
    "ToolManifestValidationError",
]
