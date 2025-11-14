"""Task metadata utilities and bundle import/export helpers."""

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

from .manifest_loader import TaskMetadata, load_task_metadata

__all__ = [
    "TaskMetadata",
    "load_task_metadata",
    "TaskBundleError",
    "export_task_bundle_bytes",
    "import_task_bundle_bytes",
]


class TaskBundleError(ValueError):
    """Raised when task bundle export or import fails."""


_BUNDLE_VERSION = 1


def _resolve_app_root(config_manager=None) -> Path:
    return resolve_app_root(config_manager)


def _normalize_task_entry(entry: TaskMetadata) -> Dict[str, Any]:
    payload = asdict(entry)
    payload.pop("source", None)
    return payload


def _resolve_manifest_path(persona: Optional[str], *, config_manager=None) -> Path:
    app_root = _resolve_app_root(config_manager)
    if persona:
        return app_root / "modules" / "Personas" / persona / "Tasks" / "tasks.json"
    return app_root / "modules" / "Tasks" / "tasks.json"


def _load_manifest_for_update(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []

    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise TaskBundleError(f"Failed to read task manifest at {path}.") from exc

    if not raw.strip():
        return []

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise TaskBundleError(f"Task manifest at {path} is not valid JSON.") from exc

    entries: Iterable[Mapping[str, Any]]
    if isinstance(payload, list):
        entries = (item for item in payload if isinstance(item, Mapping))
    elif isinstance(payload, Mapping):
        entries = (item for item in payload.values() if isinstance(item, Mapping))
    else:
        raise TaskBundleError("Task manifest must be a JSON array or object.")

    return [dict(item) for item in entries]


def _persist_manifest(path: Path, entries: List[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(
            json.dumps(entries, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise TaskBundleError(f"Failed to write task manifest to {path}.") from exc


def export_task_bundle_bytes(
    task_name: str,
    *,
    signing_key: str,
    persona: Optional[str] = None,
    config_manager=None,
) -> Tuple[bytes, Dict[str, Any]]:
    """Return a signed bundle for ``task_name`` as bytes."""

    normalized_name = str(task_name or "").strip()
    if not normalized_name:
        raise TaskBundleError("Task name is required for export.")

    entries = load_task_metadata(config_manager=config_manager)
    candidates = [entry for entry in entries if entry.name == normalized_name]

    if persona is not None:
        persona_normalized = persona.strip()
        candidates = [
            entry
            for entry in candidates
            if (entry.persona or "") == persona_normalized
        ]
    else:
        persona_normalized = None

    if not candidates:
        raise TaskBundleError(f"Task '{normalized_name}' was not found for export.")

    if len(candidates) > 1:
        available_personas = sorted({entry.persona or "shared" for entry in candidates})
        raise TaskBundleError(
            "Multiple task entries found; specify persona explicitly. Options: "
            + ", ".join(available_personas)
        )

    entry = candidates[0]
    persona_owner = entry.persona
    manifest_path = _resolve_manifest_path(persona_owner, config_manager=config_manager)

    metadata: Dict[str, Any] = {
        "version": _BUNDLE_VERSION,
        "exported_at": utcnow_isoformat(),
        "asset_type": "task",
        "name": entry.name,
        "persona": persona_owner,
        "manifest": str(manifest_path.relative_to(_resolve_app_root(config_manager))),
    }

    bundle_payload: Dict[str, Any] = {
        "metadata": metadata,
        "task": _normalize_task_entry(entry),
    }

    signature = sign_payload(
        bundle_payload,
        signing_key=signing_key,
        error_cls=TaskBundleError,
    )

    signed_bundle = {
        **bundle_payload,
        "signature": {
            "algorithm": BUNDLE_ALGORITHM,
            "value": signature,
        },
    }

    return json.dumps(signed_bundle, indent=2).encode("utf-8"), bundle_payload["task"]


def import_task_bundle_bytes(
    bundle_bytes: bytes,
    *,
    signing_key: str,
    config_manager=None,
    rationale: str = "Imported task bundle",
) -> Dict[str, Any]:
    """Import ``bundle_bytes`` and persist the task manifest entry."""

    try:
        payload = json.loads(bundle_bytes.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise TaskBundleError("Task bundle is not valid UTF-8 data.") from exc
    except json.JSONDecodeError as exc:
        raise TaskBundleError("Task bundle payload is not valid JSON.") from exc

    if not isinstance(payload, MutableMapping):
        raise TaskBundleError("Task bundle payload must be a JSON object.")

    metadata = payload.get("metadata")
    task_entry = payload.get("task")
    signature_info = payload.get("signature")

    if not isinstance(metadata, Mapping):
        raise TaskBundleError("Task bundle metadata is missing or invalid.")
    if not isinstance(task_entry, Mapping):
        raise TaskBundleError("Task bundle does not include a task definition.")
    if not isinstance(signature_info, Mapping):
        raise TaskBundleError("Task bundle signature block is missing or invalid.")

    algorithm = signature_info.get("algorithm")
    signature_value = signature_info.get("value")
    if algorithm != BUNDLE_ALGORITHM:
        raise TaskBundleError(f"Unsupported task bundle algorithm: {algorithm!r}")
    if not isinstance(signature_value, str) or not signature_value.strip():
        raise TaskBundleError("Task bundle signature is missing.")

    payload_for_signature: Dict[str, Any] = {
        "metadata": dict(metadata),
        "task": dict(task_entry),
    }

    verify_signature(
        payload_for_signature,
        signature=signature_value,
        signing_key=signing_key,
        error_cls=TaskBundleError,
    )

    persona_owner = metadata.get("persona")
    if persona_owner is not None:
        persona_owner = str(persona_owner).strip() or None

    task_name = str(task_entry.get("name") or "").strip()
    if not task_name:
        raise TaskBundleError("Task bundle is missing the task name.")

    manifest_path = _resolve_manifest_path(persona_owner, config_manager=config_manager)

    entries = _load_manifest_for_update(manifest_path)

    updated_entry = {str(key): value for key, value in task_entry.items()}
    updated_entry.pop("source", None)
    if persona_owner:
        updated_entry["persona"] = persona_owner
    else:
        updated_entry.pop("persona", None)

    normalized_name = task_name
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

    normalized_entries = load_task_metadata(config_manager=config_manager)
    for entry in normalized_entries:
        if entry.name == normalized_name and (entry.persona or None) == persona_owner:
            normalized_task = _normalize_task_entry(entry)
            break
    else:
        normalized_task = dict(updated_entry)

    return {
        "success": True,
        "task": normalized_task,
        "metadata": dict(metadata),
        "rationale": rationale,
    }

