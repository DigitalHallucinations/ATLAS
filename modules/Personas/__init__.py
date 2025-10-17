"""Utilities for loading persona configuration and tool metadata.

This module centralizes persona configuration helpers so other layers (such as
the GTK UI or persona manager) can operate on a consistent schema. The helpers
focus on normalizing the ``allowed_tools`` list introduced for persona
tooling and on merging persona selections with the global tool metadata stored
under :mod:`modules.Tools`.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import os
from pathlib import Path
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple

from jsonschema import Draft202012Validator, exceptions as jsonschema_exceptions

from modules.logging.audit import PersonaAuditLogger, get_persona_audit_logger
from modules.logging.logger import setup_logger

try:  # ConfigManager is heavy but required for accurate path resolution.
    from ATLAS.config import ConfigManager
except Exception:  # pragma: no cover - defensive import guard for test stubs
    ConfigManager = None  # type: ignore


logger = setup_logger(__name__)

PersonaPayload = MutableMapping[str, Any]
ToolMetadata = Mapping[str, Any]


class PersonaValidationError(ValueError):
    """Raised when a persona definition fails schema validation."""


@dataclass(frozen=True)
class ToolStateEntry:
    """Normalized tool entry for editor displays."""

    name: str
    enabled: bool
    order: int
    metadata: Mapping[str, Any]


def _resolve_app_root(config_manager=None) -> Path:
    """Return the repository/application root for persona files."""

    if config_manager is not None:
        getter = getattr(config_manager, "get_app_root", None)
        if callable(getter):
            try:
                root = getter()
            except Exception:  # pragma: no cover - defensive logging only
                logger.warning("Failed to resolve app root from provided config manager", exc_info=True)
            else:
                if root:
                    return Path(root).expanduser().resolve()

    if ConfigManager is not None:
        try:
            manager = config_manager or ConfigManager()
            root = manager.get_app_root()
            if root:
                return Path(root).expanduser().resolve()
        except Exception:  # pragma: no cover - fallback to repository discovery
            logger.warning("ConfigManager failed to resolve app root", exc_info=True)

    return Path(__file__).resolve().parents[2]


def _persona_file_path(persona_name: str, *, config_manager=None) -> Path:
    base = _resolve_app_root(config_manager)
    canonical_dir = base / "modules" / "Personas" / persona_name
    canonical_file = canonical_dir / "Persona" / f"{persona_name}.json"
    if canonical_file.exists():
        return canonical_file

    personas_root = base / "modules" / "Personas"
    try:
        candidates = list(personas_root.iterdir())
    except FileNotFoundError:
        return canonical_file

    lowered = persona_name.lower()
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        if candidate.name.lower() != lowered:
            continue
        persona_dir = candidate / "Persona"
        primary = persona_dir / f"{candidate.name}.json"
        if primary.exists():
            return primary
        for persona_file in persona_dir.glob("*.json"):
            if persona_file.stem.lower() == lowered:
                return persona_file

    return canonical_file


def _persona_schema_path(*, config_manager=None) -> Path:
    base = _resolve_app_root(config_manager)
    return base / "modules" / "Personas" / "schema.json"


@lru_cache(maxsize=8)
def _cached_persona_schema(path: str) -> Mapping[str, Any]:
    schema_path = Path(path)
    try:
        raw = schema_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:  # pragma: no cover - configuration error
        raise PersonaValidationError(f"Persona schema file is missing at {schema_path}") from exc
    except OSError as exc:  # pragma: no cover - unexpected I/O failure
        raise PersonaValidationError(f"Failed to read persona schema at {schema_path}") from exc

    try:
        payload = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError as exc:
        raise PersonaValidationError(f"Persona schema at {schema_path} is not valid JSON") from exc

    if not isinstance(payload, Mapping):
        raise PersonaValidationError(f"Persona schema at {schema_path} must be a JSON object")

    return payload


def _load_persona_schema(*, config_manager=None) -> Dict[str, Any]:
    schema_path = _persona_schema_path(config_manager=config_manager)
    return deepcopy(dict(_cached_persona_schema(str(schema_path))))


def _build_persona_validator(
    *,
    tool_ids: Iterable[str],
    config_manager=None,
) -> Draft202012Validator:
    schema = _load_persona_schema(config_manager=config_manager)

    defs = schema.setdefault("$defs", {})
    allowed_tool_def = defs.setdefault("allowedTool", {"type": "string"})
    normalized_ids = sorted({str(name).strip() for name in tool_ids if str(name).strip()})
    if normalized_ids:
        allowed_tool_def["enum"] = normalized_ids
    else:
        allowed_tool_def.pop("enum", None)

    try:
        return Draft202012Validator(schema)
    except jsonschema_exceptions.SchemaError as exc:  # pragma: no cover - developer error
        raise PersonaValidationError("Persona schema is invalid and cannot be used for validation") from exc


def _validate_persona_payload(
    payload: Mapping[str, Any],
    *,
    persona_name: str,
    tool_ids: Iterable[str],
    config_manager=None,
) -> None:
    validator = _build_persona_validator(tool_ids=tool_ids, config_manager=config_manager)
    errors = sorted(validator.iter_errors(payload), key=lambda error: error.json_path)
    if not errors:
        return

    details = []
    for error in errors:
        location = error.json_path or "$"
        details.append(f"{location}: {error.message}")

    message = "Persona '{name}' failed schema validation:\n{details}".format(
        name=persona_name,
        details="\n".join(details),
    )
    raise PersonaValidationError(message)


def load_tool_metadata(*, config_manager=None) -> Tuple[List[str], Dict[str, ToolMetadata]]:
    """Load the shared tool metadata map keyed by tool name."""

    app_root = _resolve_app_root(config_manager)
    manifest = app_root / "modules" / "Tools" / "tool_maps" / "functions.json"

    order: List[str] = []
    lookup: Dict[str, ToolMetadata] = {}

    try:
        raw = manifest.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.warning("Shared tool metadata file not found at %s", manifest)
        return order, lookup
    except OSError:  # pragma: no cover - unexpected I/O failure
        logger.exception("Failed to read shared tool metadata from %s", manifest)
        return order, lookup

    try:
        payload = json.loads(raw) if raw.strip() else []
    except json.JSONDecodeError:  # pragma: no cover - invalid JSON should not crash UI
        logger.exception("Invalid JSON in shared tool metadata: %s", manifest)
        return order, lookup

    entries: Iterable[Mapping[str, Any]]
    if isinstance(payload, list):
        entries = (entry for entry in payload if isinstance(entry, Mapping))
    elif isinstance(payload, Mapping):  # pragma: no cover - defensive guard
        entries = (entry for entry in payload.values() if isinstance(entry, Mapping))
    else:
        entries = []

    for entry in entries:
        name_value = entry.get("name")
        if name_value is None:
            continue
        name = str(name_value).strip()
        if not name:
            continue
        if name not in lookup:
            order.append(name)
        lookup[name] = entry

    return order, lookup


def _load_persona_tool_overrides(
    persona_name: str,
    *,
    config_manager=None,
) -> Dict[str, Mapping[str, Any]]:
    """Load persona-local tool metadata overrides when available."""

    app_root = _resolve_app_root(config_manager)
    manifest = app_root / "modules" / "Personas" / persona_name / "Toolbox" / "functions.json"

    try:
        raw = manifest.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError:  # pragma: no cover - unexpected I/O failure
        logger.exception("Failed to read persona tool overrides from %s", manifest)
        return {}

    try:
        payload = json.loads(raw) if raw.strip() else []
    except json.JSONDecodeError:
        logger.exception("Invalid JSON in persona tool overrides: %s", manifest)
        return {}

    overrides: Dict[str, Mapping[str, Any]] = {}
    if isinstance(payload, list):
        entries = payload
    elif isinstance(payload, Mapping):  # pragma: no cover - legacy defensive branch
        entries = list(payload.values())
    else:
        entries = []

    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        raw_name = entry.get("name")
        if raw_name is None:
            continue
        name = str(raw_name).strip()
        if not name:
            continue
        overrides[name] = entry

    return overrides


def normalize_allowed_tools(
    allowed_tools: Any,
    *,
    metadata_order: Optional[Iterable[str]] = None,
) -> List[str]:
    """Normalize ``allowed_tools`` into a deduplicated list of strings."""

    names: List[str] = []

    if isinstance(allowed_tools, str):
        candidate = allowed_tools.strip()
        if candidate:
            names.append(candidate)
    elif isinstance(allowed_tools, Iterable):
        for item in allowed_tools:
            if isinstance(item, str):
                candidate = item.strip()
            elif isinstance(item, Mapping):
                candidate_value = item.get("name")
                candidate = str(candidate_value).strip() if candidate_value is not None else ""
            else:
                candidate = ""

            if candidate and candidate not in names:
                names.append(candidate)

    if allowed_tools is None and metadata_order is not None:
        names.extend(name for name in metadata_order if name not in names)

    return names


def load_persona_definition(
    persona_name: str,
    *,
    config_manager=None,
    metadata_order: Optional[Iterable[str]] = None,
    metadata_lookup: Optional[Mapping[str, ToolMetadata]] = None,
) -> Optional[PersonaPayload]:
    """Load and normalize persona configuration for ``persona_name``."""

    path = _persona_file_path(persona_name, config_manager=config_manager)

    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.error("Persona file not found for '%s' at %s", persona_name, path)
        return None
    except OSError:
        logger.exception("Failed to read persona file for '%s'", persona_name)
        return None

    try:
        payload = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        logger.exception("Invalid JSON in persona file for '%s'", persona_name)
        return None

    entries = payload.get("persona") if isinstance(payload, Mapping) else None
    if not isinstance(entries, list) or not entries:
        logger.error("Persona file for '%s' does not contain a persona list", persona_name)
        return None

    persona_entry = entries[0]
    if not isinstance(persona_entry, MutableMapping):
        logger.error("Persona entry for '%s' is not a mapping", persona_name)
        return None

    order_list: Optional[List[str]] = list(metadata_order) if metadata_order is not None else None
    lookup_map: Optional[Dict[str, ToolMetadata]] = dict(metadata_lookup) if metadata_lookup is not None else None

    if order_list is None or lookup_map is None:
        shared_order, shared_lookup = load_tool_metadata(config_manager=config_manager)
        if order_list is None:
            order_list = shared_order
        if lookup_map is None:
            lookup_map = shared_lookup

    tool_ids: set[str] = set()
    if lookup_map:
        tool_ids.update(str(name) for name in lookup_map.keys())
    if order_list:
        tool_ids.update(str(name) for name in order_list)

    overrides = _load_persona_tool_overrides(persona_name, config_manager=config_manager)
    if overrides:
        tool_ids.update(str(name) for name in overrides.keys())

    try:
        _validate_persona_payload(
            payload,
            persona_name=persona_name,
            tool_ids=tool_ids,
            config_manager=config_manager,
        )
    except PersonaValidationError:
        logger.error("Persona '%s' failed validation", persona_name, exc_info=True)
        raise

    persona_entry = dict(persona_entry)
    persona_entry["allowed_tools"] = normalize_allowed_tools(
        persona_entry.get("allowed_tools"), metadata_order=order_list
    )

    return persona_entry


def _extract_allowed_tools(candidate: Optional[Mapping[str, Any]]) -> List[str]:
    if not isinstance(candidate, Mapping):
        return []
    raw_tools = candidate.get("allowed_tools")
    return normalize_allowed_tools(raw_tools or [])


def persist_persona_definition(
    persona_name: str,
    persona: Mapping[str, Any],
    *,
    config_manager=None,
    rationale: str = "Persona update",
    audit_logger: Optional[PersonaAuditLogger] = None,
) -> None:
    """Write ``persona`` back to disk preserving the schema wrapper."""

    path = _persona_file_path(persona_name, config_manager=config_manager)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {"persona": [persona]}

    previous_tools: List[str] = []
    audit = audit_logger or get_persona_audit_logger()
    try:
        existing = load_persona_definition(persona_name, config_manager=config_manager)
    except PersonaValidationError:
        existing = None
    except Exception:  # pragma: no cover - defensive logging for audit lookup
        existing = None
        logger.warning("Failed to load existing persona for audit logging", exc_info=True)
    else:
        previous_tools = _extract_allowed_tools(existing)

    new_tools = _extract_allowed_tools(persona)
    try:
        path.write_text(json.dumps(payload, indent=4), encoding="utf-8")
    except OSError:  # pragma: no cover - filesystem issues are logged for visibility
        logger.exception("Failed to persist persona '%s' to %s", persona_name, path)
    else:
        if audit is not None:
            try:
                audit.record_change(
                    persona_name,
                    previous_tools,
                    new_tools,
                    rationale=rationale,
                )
            except Exception:  # pragma: no cover - audit logging should not block persistence
                logger.warning("Failed to record persona audit event", exc_info=True)


def build_tool_state(
    persona: Mapping[str, Any],
    *,
    config_manager=None,
    metadata_order: Optional[Iterable[str]] = None,
    metadata_lookup: Optional[Mapping[str, ToolMetadata]] = None,
) -> Dict[str, Any]:
    """Return merged tool metadata with persona selections."""

    if metadata_lookup is None or metadata_order is None:
        order, lookup = load_tool_metadata(config_manager=config_manager)
    else:
        order = list(metadata_order)
        lookup = dict(metadata_lookup)

    allowed = normalize_allowed_tools(persona.get("allowed_tools"), metadata_order=order)

    persona_name = str(persona.get("name") or "").strip()
    overrides: Dict[str, Mapping[str, Any]] = {}
    if persona_name:
        overrides = _load_persona_tool_overrides(persona_name, config_manager=config_manager)

    combined_order: List[str] = []
    seen: set[str] = set()

    for name in allowed:
        if name not in seen:
            combined_order.append(name)
            seen.add(name)

    for name in order:
        if name not in seen:
            combined_order.append(name)
            seen.add(name)

    for name in overrides:
        if name not in seen:
            combined_order.append(name)
            seen.add(name)

    entries: List[ToolStateEntry] = []
    for index, name in enumerate(combined_order):
        base_metadata = lookup.get(name, {})
        override_metadata = overrides.get(name)
        merged: Dict[str, Any] = {}
        if isinstance(base_metadata, Mapping):
            merged.update(base_metadata)
        if isinstance(override_metadata, Mapping):
            merged.update(override_metadata)
        merged.setdefault("name", name)
        entries.append(
            ToolStateEntry(
                name=name,
                enabled=name in allowed,
                order=index,
                metadata=merged,
            )
        )

    serializable = [
        {
            "name": entry.name,
            "enabled": entry.enabled,
            "order": entry.order,
            "metadata": dict(entry.metadata),
        }
        for entry in entries
    ]

    return {
        "allowed": list(allowed),
        "available": serializable,
    }


__all__ = [
    "PersonaValidationError",
    "ToolStateEntry",
    "build_tool_state",
    "load_persona_definition",
    "persist_persona_definition",
    "load_tool_metadata",
    "normalize_allowed_tools",
]
