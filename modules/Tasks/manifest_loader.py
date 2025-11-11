"""Utilities for loading and validating task manifest metadata."""

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

    for persona_name, manifest_path in iter_persona_manifest_paths(
        app_root, "Tasks", "tasks.json"
    ):
        persona_entries, _ = _load_task_file(
            manifest_path,
            persona=persona_name,
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
        merged_entry = merge_with_base(raw_entry, known_entries)
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


def _normalize_entry(
    entry: Mapping[str, Any], *, persona: Optional[str], source: Path, app_root: Path
) -> TaskMetadata:
    name = coerce_string(entry.get("name"))
    summary = coerce_string(entry.get("summary"))
    description = coerce_string(entry.get("description"))
    priority = coerce_string(entry.get("priority"))

    required_skills = coerce_string_tuple(entry.get("required_skills"))
    required_tools = coerce_string_tuple(entry.get("required_tools"))
    acceptance_criteria = coerce_string_tuple(entry.get("acceptance_criteria"))
    tags = coerce_string_tuple(entry.get("tags"))

    escalation_policy = _normalize_escalation_policy(entry.get("escalation_policy"))

    resolved_persona = persona or (coerce_string(entry.get("persona")) or None)

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
