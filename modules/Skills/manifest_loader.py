"""Utilities for loading and validating skill manifest metadata."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional

from modules.store_common.manifest_utils import (
    Draft7Validator,
    ValidationError,
    coerce_string,
    get_manifest_logger,
    iter_persona_manifest_paths,
    resolve_app_root,
)


logger = get_manifest_logger(__name__)


def _resolve_app_root(config_manager, logger=logger):
    return resolve_app_root(config_manager, logger=logger)

_SCHEMA_PATH = Path(__file__).resolve().with_name("schema.json")
try:
    _SCHEMA = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
except OSError as exc:  # pragma: no cover - schema should exist
    logger.error("Unable to read skill schema at %s: %s", _SCHEMA_PATH, exc)
    _SCHEMA = {"definitions": {"skill": {"type": "object"}}}

_ENTRY_VALIDATOR = Draft7Validator(_SCHEMA["definitions"]["skill"])
_REQUIRED_STRING_FIELDS = {"name", "version", "instruction_prompt", "safety_notes"}
_REQUIRED_FIELDS = (
    "name",
    "version",
    "instruction_prompt",
    "required_tools",
    "required_capabilities",
    "safety_notes",
)


@dataclass(frozen=True)
class SkillMetadata:
    """Represents a normalized skill manifest entry."""

    name: str
    version: str
    instruction_prompt: str
    required_tools: List[str]
    required_capabilities: List[str]
    safety_notes: str
    summary: str
    category: str
    capability_tags: List[str]
    persona: Optional[str]
    source: str
    collaboration: Optional[Mapping[str, Any]]


def load_skill_metadata(*, config_manager=None) -> List[SkillMetadata]:
    """Return normalized metadata for shared and persona-specific skills."""

    app_root = _resolve_app_root(config_manager)
    skills: List[SkillMetadata] = []

    shared_manifest = app_root / "modules" / "Skills" / "skills.json"
    skills.extend(_load_skill_file(shared_manifest, persona=None, app_root=app_root))

    for persona_name, manifest_path in iter_persona_manifest_paths(
        app_root, "Skills", "skills.json"
    ):
        skills.extend(
            _load_skill_file(manifest_path, persona=persona_name, app_root=app_root)
        )

    skills.sort(key=lambda entry: ((entry.persona or ""), entry.name.lower()))
    return skills

def _load_skill_file(path: Path, *, persona: Optional[str], app_root: Path) -> Iterable[SkillMetadata]:
    if not path.exists():
        return []

    try:
        raw = path.read_text(encoding="utf-8")
        payload = json.loads(raw) if raw.strip() else []
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse skill manifest at %s: %s", path, exc)
        return []
    except OSError as exc:  # pragma: no cover - unexpected I/O errors
        logger.error("Error reading skill manifest at %s: %s", path, exc)
        return []

    if not isinstance(payload, list):
        logger.error("Skill manifest at %s must be a JSON array", path)
        return []

    entries: List[SkillMetadata] = []
    for index, entry in enumerate(_iter_skill_entries(payload)):
        missing_fields = _missing_required_fields(entry)
        if missing_fields:
            _log_manual_validation_error(path, index, missing_fields)
            continue

        errors = list(_ENTRY_VALIDATOR.iter_errors(entry))
        if errors:
            for error in errors:
                _log_validation_error(path, index, error)
            continue

        normalized = _normalize_entry(entry, persona=persona, source=path, app_root=app_root)
        entries.append(normalized)

    return entries


def _iter_skill_entries(payload: List[Any]) -> Iterator[dict[str, Any]]:
    for item in payload:
        if isinstance(item, dict):
            yield item
        else:
            logger.error("Skill manifest entries must be objects; skipping %r", item)


def _missing_required_fields(entry: Mapping[str, Any]) -> List[str]:
    missing: List[str] = []
    for field in _REQUIRED_FIELDS:
        if field not in entry:
            missing.append(field)
            continue
        if field in _REQUIRED_STRING_FIELDS and not str(entry.get(field) or "").strip():
            missing.append(field)
    return missing


def _normalize_entry(
    entry: dict[str, Any], *, persona: Optional[str], source: Path, app_root: Path
) -> SkillMetadata:
    name = coerce_string(entry.get("name"))
    version = coerce_string(entry.get("version"))
    instruction_prompt = coerce_string(entry.get("instruction_prompt"))
    required_tools = _coerce_string_list(entry.get("required_tools"))
    required_capabilities = _coerce_string_list(entry.get("required_capabilities"))
    safety_notes = coerce_string(entry.get("safety_notes"))
    summary = coerce_string(entry.get("summary"))
    category = coerce_string(entry.get("category"))
    capability_tags = _coerce_string_list(entry.get("capability_tags"))
    collaboration = _normalize_collaboration(entry.get("collaboration"))

    return SkillMetadata(
        name=name,
        version=version,
        instruction_prompt=instruction_prompt,
        required_tools=required_tools,
        required_capabilities=required_capabilities,
        safety_notes=safety_notes,
        summary=summary,
        category=category,
        capability_tags=capability_tags,
        persona=persona,
        source=_relative_source(source, app_root),
        collaboration=collaboration,
    )
def _coerce_string_list(value: Any) -> List[str]:
    if not value:
        return []
    if isinstance(value, str):
        value = [value]
    result = []
    for item in value:
        text = coerce_string(item)
        if text:
            result.append(text)
    return result


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on", "enabled"}:
            return True
        if lowered in {"false", "0", "no", "off", "disabled"}:
            return False
    return bool(value)


def _coerce_float(value: Any, default: float) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _normalize_collaboration(value: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(value, Mapping):
        return None

    enabled = _coerce_bool(value.get("enabled"))
    protocol = str(value.get("protocol") or "vote").strip().lower() or "vote"
    quorum = max(0.0, min(1.0, _coerce_float(value.get("quorum"), 0.5)))
    timeout = max(0.0, _coerce_float(value.get("timeout"), 10.0))

    config: Dict[str, Any] = {
        "enabled": enabled,
        "protocol": protocol,
        "quorum": quorum,
        "timeout": timeout,
    }

    participants: List[Dict[str, Any]] = []
    raw_participants = value.get("participants")
    if isinstance(raw_participants, Iterable) and not isinstance(raw_participants, (str, bytes, bytearray)):
        for index, entry in enumerate(raw_participants):
            data: Dict[str, Any]
            if isinstance(entry, Mapping):
                data = dict(entry)
            else:
                data = {}

            identifier = str(data.get("id") or data.get("name") or "").strip()
            if not identifier:
                identifier = f"agent_{index + 1}"

            participant: Dict[str, Any] = {"id": identifier}
            provider = data.get("provider")
            if provider is not None:
                participant["provider"] = str(provider).strip()
            model = data.get("model")
            if model is not None:
                participant["model"] = str(model).strip()
            system_prompt = data.get("system_prompt")
            if system_prompt is not None:
                participant["system_prompt"] = str(system_prompt)
            weight = data.get("weight")
            if weight is not None:
                participant["weight"] = _coerce_float(weight, 1.0)
            metadata = data.get("metadata")
            if isinstance(metadata, Mapping):
                participant["metadata"] = dict(metadata)
            participants.append(participant)

    if participants:
        config["participants"] = participants

    return config


def _relative_source(path: Path, app_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(app_root))
    except ValueError:
        return str(path)


def _log_validation_error(path: Path, index: int, error: ValidationError) -> None:
    location = "->".join(str(part) for part in error.path)
    logger.error(
        "Validation error in %s at entry %s%s: %s",
        path,
        index,
        f" ({location})" if location else "",
        error.message,
    )


def _log_manual_validation_error(path: Path, index: int, missing_fields: List[str]) -> None:
    formatted = ", ".join(sorted(missing_fields))
    logger.error(
        "Validation error in %s at entry %s: missing required fields: %s",
        path,
        index,
        formatted,
    )
