"""Utilities for loading and validating job manifest metadata."""

from __future__ import annotations

import copy
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
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

        elif schema_type == "boolean":
            if not isinstance(instance, bool):
                yield ValidationError("Expected boolean", path)

        elif schema_type == "integer":
            if not isinstance(instance, int):
                yield ValidationError("Expected integer", path)

        else:
            return

if "yaml" not in sys.modules:
    sys.modules["yaml"] = SimpleNamespace(
        safe_load=lambda *_args, **_kwargs: {},
        dump=lambda *_args, **_kwargs: None,
    )

from modules.logging.logger import setup_logger

try:  # ConfigManager may not be available in certain test scenarios
    from ATLAS.config import ConfigManager
except Exception:  # pragma: no cover - defensive import guard
    ConfigManager = None  # type: ignore


logger = setup_logger(__name__)

_SCHEMA_PATH = Path(__file__).resolve().with_name("schema.json")
try:
    _SCHEMA = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
except OSError as exc:  # pragma: no cover - schema should exist
    logger.error("Unable to read job schema at %s: %s", _SCHEMA_PATH, exc)
    _SCHEMA = {"definitions": {"job": {"type": "object"}}}

_ENTRY_VALIDATOR = Draft7Validator(_SCHEMA["definitions"]["job"])


@dataclass(frozen=True)
class JobMetadata:
    """Represents a normalized job manifest entry."""

    name: str
    summary: str
    description: str
    personas: Tuple[str, ...]
    required_skills: Tuple[str, ...]
    required_tools: Tuple[str, ...]
    task_graph: Tuple[Mapping[str, Any], ...]
    recurrence: Mapping[str, Any]
    acceptance_criteria: Tuple[str, ...]
    escalation_policy: Mapping[str, Any]
    persona: Optional[str]
    source: str


class JobManifestError(ValueError):
    """Raised when a job manifest cannot be parsed or validated."""


def load_job_metadata(*, config_manager=None) -> List[JobMetadata]:
    """Return normalized metadata for shared and persona-specific jobs."""

    app_root = _resolve_app_root(config_manager)
    jobs: List[JobMetadata] = []

    shared_manifest = app_root / "modules" / "Jobs" / "jobs.json"
    shared_entries, shared_lookup = _load_job_file(
        shared_manifest, persona=None, app_root=app_root, base_entries=None
    )
    jobs.extend(shared_entries)

    personas_root = app_root / "modules" / "Personas"
    if personas_root.is_dir():
        for persona_dir in sorted(p for p in personas_root.iterdir() if p.is_dir()):
            manifest_path = persona_dir / "Jobs" / "jobs.json"
            persona_entries, _ = _load_job_file(
                manifest_path,
                persona=persona_dir.name,
                app_root=app_root,
                base_entries=shared_lookup,
            )
            jobs.extend(persona_entries)

    jobs.sort(key=lambda entry: ((entry.persona or ""), entry.name.lower()))
    return jobs


def _resolve_app_root(config_manager) -> Path:
    if config_manager is not None:
        getter = getattr(config_manager, "get_app_root", None)
        if callable(getter):
            try:
                return Path(getter()).expanduser().resolve()
            except Exception:  # pragma: no cover - defensive guard
                logger.warning(
                    "Failed to resolve app root from supplied config manager", exc_info=True
                )

    if ConfigManager is not None:
        try:
            manager = config_manager or ConfigManager()
            root = getattr(manager, "get_app_root", lambda: None)()
            if root:
                return Path(root).expanduser().resolve()
        except Exception:  # pragma: no cover - defensive guard
            logger.warning("Unable to resolve app root via ConfigManager", exc_info=True)

    fallback = Path(__file__).resolve().parents[2]
    logger.debug("Falling back to computed app root at %s", fallback)
    return fallback


def _load_job_file(
    path: Path,
    *,
    persona: Optional[str],
    app_root: Path,
    base_entries: Optional[Mapping[str, Mapping[str, Any]]],
) -> Tuple[List[JobMetadata], Dict[str, Mapping[str, Any]]]:
    if not path.exists():
        return ([], {})

    try:
        raw = path.read_text(encoding="utf-8")
        payload = json.loads(raw) if raw.strip() else []
    except json.JSONDecodeError as exc:
        raise JobManifestError(f"Failed to parse job manifest at {path}: {exc}") from exc
    except OSError as exc:  # pragma: no cover - unexpected I/O errors
        raise JobManifestError(f"Error reading job manifest at {path}: {exc}") from exc

    if not isinstance(payload, list):
        raise JobManifestError(f"Job manifest at {path} must be a JSON array")

    known_entries: Dict[str, Mapping[str, Any]] = {}
    if base_entries:
        known_entries.update({name: copy.deepcopy(value) for name, value in base_entries.items()})

    normalized_entries: List[JobMetadata] = []
    persona_entries: Dict[str, Mapping[str, Any]] = {}

    for index, raw_entry in enumerate(_iter_job_entries(payload, path)):
        merged_entry = _merge_with_base(raw_entry, known_entries)
        if merged_entry is None:
            raise JobManifestError(
                f"Job manifest entry {index} in {path} references unknown base"
            )

        errors = list(_ENTRY_VALIDATOR.iter_errors(merged_entry))
        if errors:
            messages = [_format_validation_error(path, index, error) for error in errors]
            raise JobManifestError("; ".join(messages))

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


def _iter_job_entries(payload: List[Any], path: Path) -> Iterator[Mapping[str, Any]]:
    for index, item in enumerate(payload):
        if isinstance(item, Mapping):
            yield item
        else:
            raise JobManifestError(
                f"Job manifest entries must be objects; found {type(item).__name__} at {path}[{index}]"
            )


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
) -> JobMetadata:
    name = _coerce_string(entry.get("name"))
    summary = _coerce_string(entry.get("summary"))
    description = _coerce_string(entry.get("description"))

    personas = _coerce_string_tuple(entry.get("personas"))
    if persona and not personas:
        personas = (persona,)

    required_skills = _coerce_string_tuple(entry.get("required_skills"))
    required_tools = _coerce_string_tuple(entry.get("required_tools"))
    acceptance_criteria = _coerce_string_tuple(entry.get("acceptance_criteria"))

    task_graph = _normalize_task_graph(entry.get("task_graph"))
    recurrence = MappingProxyType(_normalize_recurrence(entry.get("recurrence")))
    escalation_policy = MappingProxyType(_normalize_escalation_policy(entry.get("escalation_policy")))

    resolved_persona = persona or None

    return JobMetadata(
        name=name,
        summary=summary,
        description=description,
        personas=personas,
        required_skills=required_skills,
        required_tools=required_tools,
        task_graph=task_graph,
        recurrence=recurrence,
        acceptance_criteria=acceptance_criteria,
        escalation_policy=escalation_policy,
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


def _normalize_task_graph(value: Any) -> Tuple[Mapping[str, Any], ...]:
    if not isinstance(value, Iterable):
        return tuple()

    result: List[Mapping[str, Any]] = []
    for node in value:
        if not isinstance(node, Mapping):
            continue
        normalized_node: Dict[str, Any] = {
            "task": _coerce_string(node.get("task")),
        }

        depends_on = _coerce_string_tuple(node.get("depends_on"))
        if depends_on:
            normalized_node["depends_on"] = depends_on

        description = _coerce_string(node.get("description"))
        if description:
            normalized_node["description"] = description

        metadata = node.get("metadata")
        if isinstance(metadata, Mapping):
            normalized_node["metadata"] = {str(key): value for key, value in metadata.items()}

        result.append(MappingProxyType(normalized_node))
    return tuple(result)


def _normalize_recurrence(value: Any) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}

    recognized_keys = {
        "frequency",
        "interval",
        "cron",
        "timezone",
        "start_date",
        "end_date",
    }
    recurrence: Dict[str, Any] = {}
    for key, val in value.items():
        if key in recognized_keys:
            if key == "interval" and isinstance(val, (int, float)):
                recurrence[key] = val
            else:
                recurrence[key] = _coerce_string(val)
        else:
            recurrence[str(key)] = val
    return recurrence


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
        policy["auto_escalate"] = value.get("auto_escalate")

    return policy


def _relative_source(path: Path, app_root: Path) -> str:
    try:
        return str(path.relative_to(app_root))
    except ValueError:
        return str(path)


def _format_validation_error(path: Path, index: int, error: ValidationError) -> str:
    location = "->".join(str(part) for part in getattr(error, "path", []))
    suffix = f".{location}" if location else ""
    message = getattr(error, "message", str(error))
    return f"Job manifest validation error at {path}[{index}]{suffix}: {message}"


__all__ = ["JobMetadata", "JobManifestError", "load_job_metadata"]
