"""Utilities for loading and normalizing tool manifest metadata."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional

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


@dataclass(frozen=True)
class ToolManifestEntry:
    """Represents a normalized tool manifest entry."""

    name: str
    persona: Optional[str]
    description: str
    version: Optional[str]
    capabilities: List[str]
    auth: Dict[str, Any]
    safety_level: Optional[str]
    requires_consent: Optional[bool]
    allow_parallel: Optional[bool]
    idempotency_key: Any
    default_timeout: Optional[int]
    side_effects: Optional[str]
    cost_per_call: Optional[float]
    cost_unit: Optional[str]
    persona_allowlist: List[str]
    requires_flags: Dict[str, List[str]]
    providers: List[Mapping[str, Any]]
    source: str

    @property
    def auth_required(self) -> bool:
        required = self.auth.get("required") if isinstance(self.auth, dict) else None
        return bool(required)


def load_manifest_entries(*, config_manager=None) -> List[ToolManifestEntry]:
    """Return normalized manifest entries for shared and persona tools."""

    app_root = _resolve_app_root(config_manager)
    manifests: List[ToolManifestEntry] = []

    shared_manifest = app_root / "modules" / "Tools" / "tool_maps" / "functions.json"
    manifests.extend(_load_manifest_file(shared_manifest, persona=None, app_root=app_root))

    personas_root = app_root / "modules" / "Personas"
    if personas_root.is_dir():
        for persona_dir in sorted(p for p in personas_root.iterdir() if p.is_dir()):
            manifest_path = persona_dir / "Toolbox" / "functions.json"
            manifests.extend(
                _load_manifest_file(manifest_path, persona=persona_dir.name, app_root=app_root)
            )

    manifests.sort(key=lambda entry: ((entry.persona or ""), entry.name.lower()))
    return manifests


def _resolve_app_root(config_manager) -> Path:
    if config_manager is not None:
        getter = getattr(config_manager, "get_app_root", None)
        if callable(getter):
            try:
                return Path(getter()).expanduser().resolve()
            except Exception:  # pragma: no cover - defensive guard
                logger.warning("Failed to resolve app root from supplied config manager", exc_info=True)

    if ConfigManager is not None:
        try:
            manager = config_manager or ConfigManager()
            root = getattr(manager, "get_app_root", lambda: None)()
            if root:
                return Path(root).expanduser().resolve()
        except Exception:  # pragma: no cover - defensive guard
            logger.warning("Unable to resolve app root via ConfigManager", exc_info=True)

    # Fallback: assume repository root is two levels up from this module
    fallback = Path(__file__).resolve().parents[2]
    logger.debug("Falling back to computed app root at %s", fallback)
    return fallback


def _load_manifest_file(
    path: Path, *, persona: Optional[str], app_root: Path
) -> Iterable[ToolManifestEntry]:
    if not path.exists():
        return []

    try:
        raw = path.read_text(encoding="utf-8")
        payload = json.loads(raw) if raw.strip() else []
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse manifest at %s: %s", path, exc)
        return []
    except OSError as exc:  # pragma: no cover - unexpected I/O errors
        logger.error("Error reading manifest at %s: %s", path, exc)
        return []

    entries: List[ToolManifestEntry] = []
    for entry in _iter_manifest_entries(payload):
        normalized = _normalize_entry(entry, persona=persona, source=path, app_root=app_root)
        if normalized is not None:
            entries.append(normalized)
    return entries


def _iter_manifest_entries(payload: Any) -> Iterator[Dict[str, Any]]:
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item
    elif isinstance(payload, dict):
        for item in payload.values():
            if isinstance(item, dict):
                yield item


def _normalize_entry(
    entry: Dict[str, Any], *, persona: Optional[str], source: Path, app_root: Path
) -> Optional[ToolManifestEntry]:
    name_value = entry.get("name")
    if name_value is None:
        return None
    name = str(name_value).strip()
    if not name:
        return None

    description = _coerce_string(entry.get("description"))
    version = _coerce_optional_string(entry.get("version"))
    capabilities = _coerce_string_list(entry.get("capabilities"))
    auth = _normalize_auth(entry.get("auth"))

    metadata = {
        "safety_level": _coerce_optional_string(entry.get("safety_level")),
        "requires_consent": _coerce_optional_bool(entry.get("requires_consent")),
        "allow_parallel": _coerce_optional_bool(entry.get("allow_parallel")),
        "idempotency_key": entry.get("idempotency_key"),
        "default_timeout": _coerce_optional_int(entry.get("default_timeout")),
        "side_effects": _coerce_optional_string(entry.get("side_effects")),
        "cost_per_call": _coerce_optional_float(entry.get("cost_per_call")),
        "cost_unit": _coerce_optional_string(entry.get("cost_unit")),
        "persona_allowlist": _coerce_string_list(entry.get("persona_allowlist")),
        "requires_flags": _normalize_requires_flags(entry.get("requires_flags")),
        "providers": _coerce_provider_list(entry.get("providers")),
    }

    return ToolManifestEntry(
        name=name,
        persona=persona,
        description=description,
        version=version,
        capabilities=capabilities,
        auth=auth,
        safety_level=metadata["safety_level"],
        requires_consent=metadata["requires_consent"],
        allow_parallel=metadata["allow_parallel"],
        idempotency_key=metadata["idempotency_key"],
        default_timeout=metadata["default_timeout"],
        side_effects=metadata["side_effects"],
        cost_per_call=metadata["cost_per_call"],
        cost_unit=metadata["cost_unit"],
        persona_allowlist=metadata["persona_allowlist"],
        requires_flags=metadata["requires_flags"],
        providers=metadata["providers"],
        source=_relative_source(source, app_root),
    )


def _coerce_string(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _coerce_optional_string(value: Any) -> Optional[str]:
    text = _coerce_string(value)
    return text or None


def _coerce_optional_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return None


def _coerce_optional_int(value: Any) -> Optional[int]:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            try:
                return int(stripped)
            except ValueError:
                return None
    return None


def _coerce_optional_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        try:
            return float(stripped)
        except ValueError:
            return None
    return None


def _coerce_string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        items = []
        for item in value:
            text = _coerce_string(item)
            if text:
                items.append(text)
        return items
    return []


def _coerce_provider_list(value: Any) -> List[Mapping[str, Any]]:
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes, bytearray)):
        return []

    providers: List[Mapping[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue

        name = _coerce_string(item.get("name"))
        if not name:
            continue

        provider: Dict[str, Any] = {"name": name}

        priority = item.get("priority")
        if isinstance(priority, (int, float)):
            provider["priority"] = int(priority)
        elif isinstance(priority, str) and priority.strip():
            try:
                provider["priority"] = int(float(priority.strip()))
            except ValueError:
                pass

        interval = item.get("health_check_interval")
        if isinstance(interval, (int, float)) and interval >= 0:
            provider["health_check_interval"] = float(interval)
        elif isinstance(interval, str) and interval.strip():
            try:
                parsed = float(interval.strip())
            except ValueError:
                parsed = None
            if parsed is not None and parsed >= 0:
                provider["health_check_interval"] = parsed

        config = item.get("config")
        if isinstance(config, Mapping):
            provider["config"] = dict(config)

        # Preserve any additional metadata for consumers that may rely on it.
        for key, raw_value in item.items():
            if key in provider:
                continue
            if key in {"name", "priority", "config", "health_check_interval"}:
                continue
            provider[key] = raw_value

        providers.append(provider)

    return providers


def _normalize_auth(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, bool):
        return {"required": value}
    if value is None:
        return {"required": False}
    return {"required": bool(value)}


def _normalize_requires_flags(value: Any) -> Dict[str, List[str]]:
    if not isinstance(value, Mapping):
        return {}
    normalized: Dict[str, List[str]] = {}
    for key, entries in value.items():
        flag_name = _coerce_string(key)
        if not flag_name:
            continue
        normalized[flag_name] = _coerce_string_list(entries)
    return normalized


def _relative_source(path: Path, app_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(app_root))
    except Exception:  # pragma: no cover - fallback to absolute
        return str(path.resolve())
