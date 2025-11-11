"""Utilities for loading and validating task manifest metadata."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple

try:  # Prefer the real jsonschema implementation when available
    from jsonschema import Draft7Validator, ValidationError
except (ModuleNotFoundError, ImportError):  # pragma: no cover - lightweight fallback
    class ValidationError(Exception):
        """Minimal substitute mirroring :class:`jsonschema.ValidationError`."""

        def __init__(self, message: str, path: Optional[List[Any]] = None):
            super().__init__(message)
            self.message = message
            self.path = tuple(path or [])

    class Draft7Validator:
        """Very small subset of :class:`jsonschema.Draft7Validator` used in tests."""

        def __init__(self, schema: dict[str, Any]):
            self.schema = schema

        def iter_errors(self, instance: Any):
            yield from _validate_with_schema(instance, self.schema, [])

    def _validate_with_schema(instance: Any, schema: dict[str, Any], path: List[Any]):
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
            return

from modules.store_common.manifest_utils import get_manifest_logger, resolve_app_root


logger = get_manifest_logger(__name__)


def _resolve_app_root(config_manager, logger=logger):
    return resolve_app_root(config_manager, logger=logger)

_SCHEMA_PATH = Path(__file__).resolve().with_name("schema.json")
try:
    _SCHEMA = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
except OSError as exc:  # pragma: no cover - schema should exist
    logger.error("Unable to read task schema at %s: %s", _SCHEMA_PATH, exc)
    _SCHEMA = {"definitions": {"task": {"type": "object"}}}

_ENTRY_VALIDATOR = Draft7Validator(_SCHEMA["definitions"]["task"])


@dataclass(frozen=True)
class TaskMetadata:
    """Represents a normalized task manifest entry."""

    name: str
    summary: str
    description: str
    required_skills: Tuple[str, ...]
    required_tools: Tuple[str, ...]
    acceptance_criteria: Tuple[str, ...]
    escalation_policy: Mapping[str, Any]
    tags: Tuple[str, ...]
    priority: str
    persona: Optional[str]
    source: str


def load_task_metadata(*, config_manager=None) -> List[TaskMetadata]:
    """Return normalized metadata for shared and persona-specific tasks."""

    app_root = _resolve_app_root(config_manager)
    tasks: List[TaskMetadata] = []

    shared_manifest = app_root / "modules" / "Tasks" / "tasks.json"
    shared_entries, shared_lookup = _load_task_file(
        shared_manifest, persona=None, app_root=app_root, base_entries=None
    )
    tasks.extend(shared_entries)

    personas_root = app_root / "modules" / "Personas"
    if personas_root.is_dir():
        for persona_dir in sorted(p for p in personas_root.iterdir() if p.is_dir()):
            manifest_path = persona_dir / "Tasks" / "tasks.json"
            persona_entries, _ = _load_task_file(
                manifest_path,
                persona=persona_dir.name,
                app_root=app_root,
                base_entries=shared_lookup,
            )
            tasks.extend(persona_entries)

    tasks.sort(key=lambda entry: ((entry.persona or ""), entry.name.lower()))
    return tasks

def _load_task_file(
    path: Path,
    *,
    persona: Optional[str],
    app_root: Path,
    base_entries: Optional[Mapping[str, Mapping[str, Any]]],
) -> Tuple[List[TaskMetadata], Dict[str, Mapping[str, Any]]]:
    if not path.exists():
        return ([], {})

    try:
        raw = path.read_text(encoding="utf-8")
        payload = json.loads(raw) if raw.strip() else []
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse task manifest at %s: %s", path, exc)
        return ([], {})
    except OSError as exc:  # pragma: no cover - unexpected I/O errors
        logger.error("Error reading task manifest at %s: %s", path, exc)
        return ([], {})

    if not isinstance(payload, list):
        logger.error("Task manifest at %s must be a JSON array", path)
        return ([], {})

    known_entries: Dict[str, Mapping[str, Any]] = {}
    if base_entries:
        known_entries.update({name: copy.deepcopy(value) for name, value in base_entries.items()})

    normalized_entries: List[TaskMetadata] = []
    persona_entries: Dict[str, Mapping[str, Any]] = {}

    for index, raw_entry in enumerate(_iter_task_entries(payload)):
        merged_entry = _merge_with_base(raw_entry, known_entries)
        if merged_entry is None:
            logger.error(
                "Task manifest entry %s in %s references unknown base", index, path
            )
            continue

        missing_required = [
            key
            for key in ("summary", "required_skills", "required_tools", "acceptance_criteria", "escalation_policy")
            if key not in merged_entry
        ]
        if missing_required:
            logger.error(
                "Task manifest entry %s in %s missing required fields: %s",
                index,
                path,
                ", ".join(missing_required),
            )
            continue

        errors = list(_ENTRY_VALIDATOR.iter_errors(merged_entry))
        if errors:
            for error in errors:
                _log_validation_error(path, index, error)
            continue

        normalized = _normalize_entry(
            merged_entry,
            persona=persona,
            source=path,
            app_root=app_root,
        )
        normalized_entries.append(normalized)
        persona_entries[normalized.name] = merged_entry
        known_entries[normalized.name] = copy.deepcopy(merged_entry)

    return normalized_entries, persona_entries


def _iter_task_entries(payload: List[Any]) -> Iterator[Mapping[str, Any]]:
    for item in payload:
        if isinstance(item, Mapping):
            yield item
        else:
            logger.error("Task manifest entries must be objects; skipping %r", item)


def _merge_with_base(
    entry: Mapping[str, Any], known_entries: Mapping[str, Mapping[str, Any]]
) -> Optional[Dict[str, Any]]:
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


def _normalize_entry(
    entry: Mapping[str, Any], *, persona: Optional[str], source: Path, app_root: Path
) -> TaskMetadata:
    name = _coerce_string(entry.get("name"))
    summary = _coerce_string(entry.get("summary"))
    description = _coerce_string(entry.get("description"))
    priority = _coerce_string(entry.get("priority"))

    required_skills = _coerce_string_tuple(entry.get("required_skills"))
    required_tools = _coerce_string_tuple(entry.get("required_tools"))
    acceptance_criteria = _coerce_string_tuple(entry.get("acceptance_criteria"))
    tags = _coerce_string_tuple(entry.get("tags"))

    escalation_policy = _normalize_escalation_policy(entry.get("escalation_policy"))

    resolved_persona = persona or (_coerce_string(entry.get("persona")) or None)

    return TaskMetadata(
        name=name,
        summary=summary,
        description=description,
        required_skills=required_skills,
        required_tools=required_tools,
        acceptance_criteria=acceptance_criteria,
        escalation_policy=MappingProxyType(escalation_policy),
        tags=tags,
        priority=priority,
        persona=resolved_persona,
        source=_relative_source(source, app_root),
    )


def _coerce_string(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _coerce_string_tuple(value: Any) -> Tuple[str, ...]:
    if not value:
        return tuple()
    if isinstance(value, str):
        value = [value]
    result: List[str] = []
    for item in value:
        text = _coerce_string(item)
        if text:
            result.append(text)
    return tuple(result)


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on", "enabled"}:
            return True
        if lowered in {"false", "0", "no", "off", "disabled"}:
            return False
    return bool(value)


def _normalize_escalation_policy(value: Any) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        return {"level": "", "contact": ""}

    policy: Dict[str, Any] = {
        "level": _coerce_string(value.get("level")),
        "contact": _coerce_string(value.get("contact")),
    }

    timeframe = _coerce_string(value.get("timeframe"))
    if timeframe:
        policy["timeframe"] = timeframe

    triggers = list(_coerce_string_tuple(value.get("triggers")))
    if triggers:
        policy["triggers"] = triggers

    actions = list(_coerce_string_tuple(value.get("actions")))
    if actions:
        policy["actions"] = actions

    notes = _coerce_string(value.get("notes"))
    if notes:
        policy["notes"] = notes

    if "auto_escalate" in value:
        policy["auto_escalate"] = _coerce_bool(value.get("auto_escalate"))

    return policy


def _relative_source(path: Path, app_root: Path) -> str:
    try:
        return str(path.relative_to(app_root))
    except ValueError:
        return str(path)


def _log_validation_error(path: Path, index: int, error: ValidationError) -> None:
    location = "->".join(str(part) for part in getattr(error, "path", []))
    suffix = f".{location}" if location else ""
    logger.error(
        "Task manifest validation error at %s[%s]%s: %s",
        path,
        index,
        suffix,
        getattr(error, "message", str(error)),
    )


__all__ = ["TaskMetadata", "load_task_metadata"]
