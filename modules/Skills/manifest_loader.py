"""Utilities for loading and validating skill manifest metadata."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional

try:  # Prefer the real jsonschema implementation when available
    from jsonschema import Draft7Validator, ValidationError
except (ModuleNotFoundError, ImportError):  # pragma: no cover - lightweight fallback for limited envs
    class ValidationError(Exception):
        """Minimal substitute mirroring jsonschema.ValidationError."""

        def __init__(self, message: str, path: Optional[List[Any]] = None):
            super().__init__(message)
            self.message = message
            self.path = tuple(path or [])


    class Draft7Validator:
        """Very small subset of jsonschema.Draft7Validator used in tests."""

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
            # Types we don't recognize are treated as pass-through.
            return


if "yaml" not in sys.modules:
    from types import SimpleNamespace

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
    logger.error("Unable to read skill schema at %s: %s", _SCHEMA_PATH, exc)
    _SCHEMA = {"definitions": {"skill": {"type": "object"}}}

_ENTRY_VALIDATOR = Draft7Validator(_SCHEMA["definitions"]["skill"])


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

    personas_root = app_root / "modules" / "Personas"
    if personas_root.is_dir():
        for persona_dir in sorted(p for p in personas_root.iterdir() if p.is_dir()):
            manifest_path = persona_dir / "Skills" / "skills.json"
            skills.extend(
                _load_skill_file(manifest_path, persona=persona_dir.name, app_root=app_root)
            )

    skills.sort(key=lambda entry: ((entry.persona or ""), entry.name.lower()))
    return skills


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


def _normalize_entry(
    entry: dict[str, Any], *, persona: Optional[str], source: Path, app_root: Path
) -> SkillMetadata:
    name = _coerce_string(entry.get("name"))
    version = _coerce_string(entry.get("version"))
    instruction_prompt = _coerce_string(entry.get("instruction_prompt"))
    required_tools = _coerce_string_list(entry.get("required_tools"))
    required_capabilities = _coerce_string_list(entry.get("required_capabilities"))
    safety_notes = _coerce_string(entry.get("safety_notes"))
    summary = _coerce_string(entry.get("summary"))
    category = _coerce_string(entry.get("category"))
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


def _coerce_string(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _coerce_string_list(value: Any) -> List[str]:
    if not value:
        return []
    if isinstance(value, str):
        value = [value]
    result = []
    for item in value:
        text = _coerce_string(item)
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
