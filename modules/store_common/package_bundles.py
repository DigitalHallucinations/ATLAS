"""Utilities for packaging multiple capability bundles into a single archive."""

from __future__ import annotations

import base64
import binascii
import json
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from modules.Jobs import export_job_bundle_bytes, import_job_bundle_bytes
from modules.Personas import export_persona_bundle_bytes, import_persona_bundle_bytes
from modules.Skills import export_skill_bundle_bytes, import_skill_bundle_bytes
from modules.Tasks import export_task_bundle_bytes, import_task_bundle_bytes
from modules.Tools import export_tool_bundle_bytes, import_tool_bundle_bytes

from .bundle_utils import BUNDLE_ALGORITHM, sign_payload, utcnow_isoformat, verify_signature

__all__ = [
    "AssetPackageError",
    "export_asset_package_bytes",
    "import_asset_package_bytes",
]


class AssetPackageError(ValueError):
    """Raised when multi-asset package export or import fails."""


_PACKAGE_VERSION = 1


def _parse_scoped_identifier(value: str) -> Tuple[Optional[str], str]:
    token = str(value or "").strip()
    if not token:
        return (None, "")
    if ":" in token:
        persona, name = token.split(":", 1)
        persona = persona.strip() or None
        name = name.strip()
        return (persona, name)
    return (None, token)


def _encode_bundle_bytes(bundle_bytes: bytes) -> str:
    return base64.b64encode(bundle_bytes).decode("ascii")


def _decode_bundle_bytes(data: Any) -> bytes:
    if not isinstance(data, str) or not data.strip():
        raise AssetPackageError("Asset package entry is missing bundle data.")
    try:
        return base64.b64decode(data.encode("ascii"))
    except (ValueError, binascii.Error) as exc:
        raise AssetPackageError("Asset package entry bundle is not valid base64.") from exc


def _append_asset(
    assets: List[Dict[str, Any]],
    *,
    asset_type: str,
    name: str,
    persona: Optional[str],
    bundle_bytes: bytes,
) -> None:
    try:
        payload = json.loads(bundle_bytes.decode("utf-8"))
    except Exception:
        payload = {}

    asset_metadata = {
        "asset_type": asset_type,
        "name": name,
        "persona": persona,
        "bundle": _encode_bundle_bytes(bundle_bytes),
    }

    if isinstance(payload, Mapping):
        for key in (asset_type, "metadata"):
            value = payload.get(key)
            if isinstance(value, Mapping):
                asset_metadata[key] = dict(value)
    assets.append(asset_metadata)


def _normalize_sequence(sequence: Optional[Sequence[str]]) -> List[str]:
    if not sequence:
        return []
    return [str(item) for item in sequence if str(item or "").strip()]


def export_asset_package_bytes(
    *,
    personas: Optional[Sequence[str]] = None,
    tools: Optional[Sequence[str]] = None,
    skills: Optional[Sequence[str]] = None,
    tasks: Optional[Sequence[str]] = None,
    jobs: Optional[Sequence[str]] = None,
    signing_key: str,
    config_manager=None,
) -> Tuple[bytes, Dict[str, Any]]:
    """Export a multi-asset package containing the requested bundles."""

    requested_assets: List[Tuple[str, Optional[str], str]] = []

    for persona_name in _normalize_sequence(personas):
        _, name = _parse_scoped_identifier(persona_name)
        requested_assets.append(("persona", None, name))

    for scoped in _normalize_sequence(tools):
        persona, name = _parse_scoped_identifier(scoped)
        requested_assets.append(("tool", persona, name))

    for scoped in _normalize_sequence(skills):
        persona, name = _parse_scoped_identifier(scoped)
        requested_assets.append(("skill", persona, name))

    for scoped in _normalize_sequence(tasks):
        persona, name = _parse_scoped_identifier(scoped)
        requested_assets.append(("task", persona, name))

    for scoped in _normalize_sequence(jobs):
        persona, name = _parse_scoped_identifier(scoped)
        requested_assets.append(("job", persona, name))

    if not requested_assets:
        raise AssetPackageError("Specify at least one persona, tool, skill, task, or job to export.")

    assets_payload: List[Dict[str, Any]] = []

    for asset_type, persona, name in requested_assets:
        if asset_type == "persona":
            bundle_bytes, persona_payload = export_persona_bundle_bytes(
                name,
                signing_key=signing_key,
                config_manager=config_manager,
            )
            _append_asset(
                assets_payload,
                asset_type="persona",
                name=persona_payload.get("name", name),
                persona=persona_payload.get("name", name),
                bundle_bytes=bundle_bytes,
            )
            continue

        if asset_type == "tool":
            bundle_bytes, _ = export_tool_bundle_bytes(
                name,
                signing_key=signing_key,
                persona=persona,
                config_manager=config_manager,
            )
            _append_asset(
                assets_payload,
                asset_type="tool",
                name=name,
                persona=persona,
                bundle_bytes=bundle_bytes,
            )
            continue

        if asset_type == "skill":
            bundle_bytes, _ = export_skill_bundle_bytes(
                name,
                signing_key=signing_key,
                persona=persona,
                config_manager=config_manager,
            )
            _append_asset(
                assets_payload,
                asset_type="skill",
                name=name,
                persona=persona,
                bundle_bytes=bundle_bytes,
            )
            continue

        if asset_type == "task":
            bundle_bytes, _ = export_task_bundle_bytes(
                name,
                signing_key=signing_key,
                persona=persona,
                config_manager=config_manager,
            )
            _append_asset(
                assets_payload,
                asset_type="task",
                name=name,
                persona=persona,
                bundle_bytes=bundle_bytes,
            )
            continue

        if asset_type == "job":
            bundle_bytes, _ = export_job_bundle_bytes(
                name,
                signing_key=signing_key,
                persona=persona,
                config_manager=config_manager,
            )
            _append_asset(
                assets_payload,
                asset_type="job",
                name=name,
                persona=persona,
                bundle_bytes=bundle_bytes,
            )
            continue

        raise AssetPackageError(f"Unsupported asset type requested: {asset_type!r}")

    metadata: Dict[str, Any] = {
        "version": _PACKAGE_VERSION,
        "exported_at": utcnow_isoformat(),
        "counts": {
            "personas": sum(1 for asset in assets_payload if asset["asset_type"] == "persona"),
            "tools": sum(1 for asset in assets_payload if asset["asset_type"] == "tool"),
            "skills": sum(1 for asset in assets_payload if asset["asset_type"] == "skill"),
            "tasks": sum(1 for asset in assets_payload if asset["asset_type"] == "task"),
            "jobs": sum(1 for asset in assets_payload if asset["asset_type"] == "job"),
        },
    }

    package_payload = {
        "metadata": metadata,
        "assets": assets_payload,
    }

    signature = sign_payload(
        package_payload,
        signing_key=signing_key,
        error_cls=AssetPackageError,
    )

    signed_package = {
        **package_payload,
        "signature": {
            "algorithm": BUNDLE_ALGORITHM,
            "value": signature,
        },
    }

    return json.dumps(signed_package, indent=2).encode("utf-8"), metadata


def import_asset_package_bytes(
    bundle_bytes: bytes,
    *,
    signing_key: str,
    config_manager=None,
    rationale: str = "Imported asset package",
) -> Dict[str, Any]:
    """Import a package of persona/tool/skill/task/job bundles."""

    try:
        payload = json.loads(bundle_bytes.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise AssetPackageError("Asset package is not valid UTF-8 data.") from exc
    except json.JSONDecodeError as exc:
        raise AssetPackageError("Asset package payload is not valid JSON.") from exc

    if not isinstance(payload, MutableMapping):
        raise AssetPackageError("Asset package payload must be a JSON object.")

    metadata = payload.get("metadata")
    assets = payload.get("assets")
    signature_info = payload.get("signature")

    if not isinstance(metadata, Mapping):
        raise AssetPackageError("Asset package metadata is missing or invalid.")
    if not isinstance(assets, list):
        raise AssetPackageError("Asset package assets list is missing or invalid.")
    if not isinstance(signature_info, Mapping):
        raise AssetPackageError("Asset package signature block is missing or invalid.")

    algorithm = signature_info.get("algorithm")
    signature_value = signature_info.get("value")
    if algorithm != BUNDLE_ALGORITHM:
        raise AssetPackageError(f"Unsupported asset package algorithm: {algorithm!r}")
    if not isinstance(signature_value, str) or not signature_value.strip():
        raise AssetPackageError("Asset package signature is missing.")

    payload_for_signature: Dict[str, Any] = {
        "metadata": dict(metadata),
        "assets": list(assets),
    }

    verify_signature(
        payload_for_signature,
        signature=signature_value,
        signing_key=signing_key,
        error_cls=AssetPackageError,
    )

    imported_assets: List[Dict[str, Any]] = []

    for asset in assets:
        if not isinstance(asset, Mapping):
            raise AssetPackageError("Asset package entry is not an object.")

        asset_type = str(asset.get("asset_type") or "").strip().lower()
        name = str(asset.get("name") or "").strip()
        persona = asset.get("persona")
        if persona is not None:
            persona = str(persona).strip() or None

        bundle_data = _decode_bundle_bytes(asset.get("bundle"))

        if asset_type == "persona":
            result = import_persona_bundle_bytes(
                bundle_data,
                signing_key=signing_key,
                config_manager=config_manager,
                rationale=f"{rationale} (persona)",
            )
        elif asset_type == "tool":
            result = import_tool_bundle_bytes(
                bundle_data,
                signing_key=signing_key,
                config_manager=config_manager,
                rationale=f"{rationale} (tool)",
            )
        elif asset_type == "skill":
            result = import_skill_bundle_bytes(
                bundle_data,
                signing_key=signing_key,
                config_manager=config_manager,
                rationale=f"{rationale} (skill)",
            )
        elif asset_type == "task":
            result = import_task_bundle_bytes(
                bundle_data,
                signing_key=signing_key,
                config_manager=config_manager,
                rationale=f"{rationale} (task)",
            )
        elif asset_type == "job":
            result = import_job_bundle_bytes(
                bundle_data,
                signing_key=signing_key,
                config_manager=config_manager,
                rationale=f"{rationale} (job)",
            )
        else:
            raise AssetPackageError(f"Asset package contains unsupported asset type: {asset_type!r}")

        imported_assets.append(
            {
                "asset_type": asset_type,
                "name": name or result.get("metadata", {}).get("name"),
                "persona": persona,
                "result": result,
            }
        )

    return {
        "success": True,
        "metadata": dict(metadata),
        "assets": imported_assets,
        "rationale": rationale,
    }

