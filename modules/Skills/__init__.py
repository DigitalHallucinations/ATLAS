"""Skill metadata loading and bundle utilities."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple

from modules.store_common.bundle_utils import (
    BUNDLE_ALGORITHM,
    sign_payload,
    utcnow_isoformat,
    verify_signature,
)
from modules.store_common.manifest_utils import resolve_app_root

from .manifest_loader import SkillMetadata, load_skill_metadata

__all__ = [
    "SkillMetadata",
    "load_skill_metadata",
    "SkillBundleError",
    "export_skill_bundle_bytes",
    "import_skill_bundle_bytes",
]


class SkillBundleError(ValueError):
    """Raised when skill bundle export or import fails."""


_BUNDLE_VERSION = 1


def _resolve_app_root(config_manager=None) -> Path:
    return resolve_app_root(config_manager)


def _normalize_skill_entry(entry: SkillMetadata) -> Dict[str, Any]:
    payload = asdict(entry)
    payload.pop("source", None)
    return payload


def _resolve_manifest_path(persona: Optional[str], *, config_manager=None) -> Path:
    app_root = _resolve_app_root(config_manager)
    if persona:
        return (
            app_root / "modules" / "Personas" / persona / "Skills" / "skills.json"
        )
    return app_root / "modules" / "Skills" / "skills.json"


def _load_manifest_for_update(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []

    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SkillBundleError(f"Failed to read skill manifest at {path}.") from exc

    if not raw.strip():
        return []

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SkillBundleError(f"Skill manifest at {path} is not valid JSON.") from exc

    entries: Iterable[Mapping[str, Any]]
    if isinstance(payload, list):
        entries = (item for item in payload if isinstance(item, Mapping))
    elif isinstance(payload, Mapping):
        entries = (item for item in payload.values() if isinstance(item, Mapping))
    else:
        raise SkillBundleError("Skill manifest must be a JSON array or object.")

    return [dict(item) for item in entries]


def _persist_manifest(path: Path, entries: List[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(
            json.dumps(entries, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise SkillBundleError(f"Failed to write skill manifest to {path}.") from exc


def export_skill_bundle_bytes(
    skill_name: str,
    *,
    signing_key: str,
    persona: Optional[str] = None,
    config_manager=None,
) -> Tuple[bytes, Dict[str, Any]]:
    """Return a signed bundle for ``skill_name`` as bytes."""

    normalized_name = str(skill_name or "").strip()
    if not normalized_name:
        raise SkillBundleError("Skill name is required for export.")

    entries = load_skill_metadata(config_manager=config_manager)
    candidates = [entry for entry in entries if entry.name == normalized_name]

    if persona is not None:
        persona_normalized = persona.strip()
        candidates = [entry for entry in candidates if (entry.persona or "") == persona_normalized]
    else:
        persona_normalized = None

    if not candidates:
        raise SkillBundleError(f"Skill '{normalized_name}' was not found for export.")

    if len(candidates) > 1:
        available_personas = sorted({entry.persona or "shared" for entry in candidates})
        raise SkillBundleError(
            "Multiple skill entries found; specify persona explicitly. Options: "
            + ", ".join(available_personas)
        )

    entry = candidates[0]
    persona_owner = entry.persona
    manifest_path = _resolve_manifest_path(persona_owner, config_manager=config_manager)

    metadata: Dict[str, Any] = {
        "version": _BUNDLE_VERSION,
        "exported_at": utcnow_isoformat(),
        "asset_type": "skill",
        "name": entry.name,
        "persona": persona_owner,
        "manifest": str(manifest_path.relative_to(_resolve_app_root(config_manager))),
    }

    bundle_payload: Dict[str, Any] = {
        "metadata": metadata,
        "skill": _normalize_skill_entry(entry),
    }

    signature = sign_payload(
        bundle_payload,
        signing_key=signing_key,
        error_cls=SkillBundleError,
    )

    signed_bundle = {
        **bundle_payload,
        "signature": {
            "algorithm": BUNDLE_ALGORITHM,
            "value": signature,
        },
    }

    return json.dumps(signed_bundle, indent=2).encode("utf-8"), bundle_payload["skill"]


def import_skill_bundle_bytes(
    bundle_bytes: bytes,
    *,
    signing_key: str,
    config_manager=None,
    rationale: str = "Imported skill bundle",
) -> Dict[str, Any]:
    """Import ``bundle_bytes`` and persist the skill manifest entry."""

    try:
        payload = json.loads(bundle_bytes.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise SkillBundleError("Skill bundle is not valid UTF-8 data.") from exc
    except json.JSONDecodeError as exc:
        raise SkillBundleError("Skill bundle payload is not valid JSON.") from exc

    if not isinstance(payload, MutableMapping):
        raise SkillBundleError("Skill bundle payload must be a JSON object.")

    metadata = payload.get("metadata")
    skill_entry = payload.get("skill")
    signature_info = payload.get("signature")

    if not isinstance(metadata, Mapping):
        raise SkillBundleError("Skill bundle metadata is missing or invalid.")
    if not isinstance(skill_entry, Mapping):
        raise SkillBundleError("Skill bundle does not include a skill definition.")
    if not isinstance(signature_info, Mapping):
        raise SkillBundleError("Skill bundle signature block is missing or invalid.")

    algorithm = signature_info.get("algorithm")
    signature_value = signature_info.get("value")
    if algorithm != BUNDLE_ALGORITHM:
        raise SkillBundleError(f"Unsupported skill bundle algorithm: {algorithm!r}")
    if not isinstance(signature_value, str) or not signature_value.strip():
        raise SkillBundleError("Skill bundle signature is missing.")

    payload_for_signature: Dict[str, Any] = {
        "metadata": dict(metadata),
        "skill": dict(skill_entry),
    }

    verify_signature(
        payload_for_signature,
        signature=signature_value,
        signing_key=signing_key,
        error_cls=SkillBundleError,
    )

    persona_owner = metadata.get("persona")
    if persona_owner is not None:
        persona_owner = str(persona_owner).strip() or None

    skill_name = str(skill_entry.get("name") or "").strip()
    if not skill_name:
        raise SkillBundleError("Skill bundle is missing the skill name.")

    manifest_path = _resolve_manifest_path(persona_owner, config_manager=config_manager)

    entries = _load_manifest_for_update(manifest_path)

    updated_entry = {str(key): value for key, value in skill_entry.items()}
    updated_entry.pop("source", None)
    if persona_owner:
        updated_entry["persona"] = persona_owner
    else:
        updated_entry.pop("persona", None)

    normalized_name = skill_name.strip()
    replaced = False
    new_entries: List[Dict[str, Any]] = []
    for existing in entries:
        existing_name = str(existing.get("name") or "").strip()
        if existing_name == normalized_name:
            if not replaced:
                new_entries.append(updated_entry)
                replaced = True
            continue
        new_entries.append(dict(existing))

    if not replaced:
        new_entries.append(updated_entry)

    _persist_manifest(manifest_path, new_entries)

    normalized_entries = load_skill_metadata(config_manager=config_manager)
    for entry in normalized_entries:
        if entry.name == normalized_name and (entry.persona or None) == persona_owner:
            normalized_skill = _normalize_skill_entry(entry)
            break
    else:
        normalized_skill = dict(updated_entry)

    return {
        "success": True,
        "skill": normalized_skill,
        "metadata": dict(metadata),
        "rationale": rationale,
    }

