"""Utilities for loading and validating job manifest metadata."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple

from modules.store_common.manifest_utils import (
    Draft7Validator,
    ValidationError,
    coerce_string,
    coerce_string_tuple,
    get_manifest_logger,
    iter_persona_manifest_paths,
    merge_with_base,
    resolve_app_root,
)


logger = get_manifest_logger(__name__)


def _resolve_app_root(config_manager, logger=logger):
    return resolve_app_root(config_manager, logger=logger)

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

    for persona_name, manifest_path in iter_persona_manifest_paths(
        app_root, "Jobs", "jobs.json"
    ):
        persona_entries, _ = _load_job_file(
            manifest_path,
            persona=persona_name,
            app_root=app_root,
            base_entries=shared_lookup,
        )
        jobs.extend(persona_entries)

    jobs.sort(key=lambda entry: ((entry.persona or ""), entry.name.lower()))
    return jobs

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
        merged_entry = merge_with_base(raw_entry, known_entries)
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


def _normalize_entry(
    entry: Mapping[str, Any], *, persona: Optional[str], source: Path, app_root: Path
) -> JobMetadata:
    name = coerce_string(entry.get("name"))
    summary = coerce_string(entry.get("summary"))
    description = coerce_string(entry.get("description"))

    personas = coerce_string_tuple(entry.get("personas"))
    if persona and not personas:
        personas = (persona,)

    required_skills = coerce_string_tuple(entry.get("required_skills"))
    required_tools = coerce_string_tuple(entry.get("required_tools"))
    acceptance_criteria = coerce_string_tuple(entry.get("acceptance_criteria"))

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
def _normalize_task_graph(value: Any) -> Tuple[Mapping[str, Any], ...]:
    if not isinstance(value, Iterable):
        return tuple()

    result: List[Mapping[str, Any]] = []
    for node in value:
        if not isinstance(node, Mapping):
            continue
        normalized_node: Dict[str, Any] = {
            "task": coerce_string(node.get("task")),
        }

        depends_on = coerce_string_tuple(node.get("depends_on"))
        if depends_on:
            normalized_node["depends_on"] = depends_on

        description = coerce_string(node.get("description"))
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
                recurrence[key] = coerce_string(val)
        else:
            recurrence[str(key)] = val
    return recurrence


def _normalize_escalation_policy(value: Any) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        return {"level": "", "contact": ""}

    policy: Dict[str, Any] = {
        "level": coerce_string(value.get("level")),
        "contact": coerce_string(value.get("contact")),
    }

    timeframe = coerce_string(value.get("timeframe"))
    if timeframe:
        policy["timeframe"] = timeframe

    triggers = list(coerce_string_tuple(value.get("triggers")))
    if triggers:
        policy["triggers"] = triggers

    actions = list(coerce_string_tuple(value.get("actions")))
    if actions:
        policy["actions"] = actions

    notes = coerce_string(value.get("notes"))
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
