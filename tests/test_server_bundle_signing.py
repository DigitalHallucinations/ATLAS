"""Regression tests covering server-managed bundle signing secrets."""

from __future__ import annotations

import base64
import json
from typing import Any, Dict, Optional

import pytest

from modules.Personas import PersonaBundleError
from modules.Server.routes import AtlasServer
from modules.store_common.bundle_utils import (
    BUNDLE_ALGORITHM,
    sign_payload,
    verify_signature,
)


class _ConfigStub:
    def __init__(self, secrets: Dict[str, str]) -> None:
        self._secrets = {key.lower(): value for key, value in secrets.items()}

    def get_bundle_signing_key(self, asset_type: str, *, require: bool = True) -> Optional[str]:
        key = self._secrets.get(str(asset_type).lower())
        if key:
            return key
        if require:
            raise RuntimeError(f"Missing signing key for {asset_type}")
        return None

    def get_persona_bundle_signing_key(self, *, require: bool = True) -> Optional[str]:
        return self.get_bundle_signing_key("persona", require=require)


def _stub_import_bundle(
    bundle_bytes: bytes,
    *,
    signing_key: str,
    config_manager: object | None,
    rationale: str,
) -> Dict[str, Any]:
    payload = json.loads(bundle_bytes.decode("utf-8"))
    signature_info = payload.get("signature")
    if not isinstance(signature_info, dict):
        raise PersonaBundleError("Persona bundle signature block is missing or invalid.")

    algorithm = signature_info.get("algorithm")
    if algorithm != BUNDLE_ALGORITHM:
        raise PersonaBundleError(f"Unsupported persona bundle algorithm: {algorithm!r}")

    signature_value = signature_info.get("value")
    if not isinstance(signature_value, str) or not signature_value:
        raise PersonaBundleError("Persona bundle signature is missing.")

    payload_for_signature = {
        "metadata": payload.get("metadata", {}),
        "persona": payload.get("persona", {}),
    }

    verify_signature(
        payload_for_signature,
        signature=signature_value,
        signing_key=signing_key,
        error_cls=PersonaBundleError,
    )

    return {"success": True, "persona": payload.get("persona", {})}


def test_persona_export_uses_configured_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _ConfigStub({"persona": "server-secret"})
    server = AtlasServer(config_manager=config)

    captured: Dict[str, str] = {}

    def _fake_export(name: str, *, signing_key: str, config_manager: object | None):
        captured["signing_key"] = signing_key
        return b"{}", {"name": name}

    monkeypatch.setattr(
        "modules.Server.routes.export_persona_bundle_bytes",
        _fake_export,
    )

    response = server.export_persona_bundle("Helper", signing_key="client-secret")

    assert response["success"] is True
    assert captured["signing_key"] == "server-secret"
    serialized = json.dumps(response)
    assert "client-secret" not in serialized


def test_export_without_configured_secret() -> None:
    server = AtlasServer(config_manager=_ConfigStub({}))

    result = server.export_persona_bundle("Helper")

    assert result["success"] is False
    assert "signing key" in result["error"].lower()


def test_import_rejects_unsigned_bundle(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _ConfigStub({"persona": "server-secret"})
    server = AtlasServer(config_manager=config)

    monkeypatch.setattr(
        "modules.Server.routes.import_persona_bundle_bytes",
        _stub_import_bundle,
    )

    payload = {
        "metadata": {"persona_name": "Helper"},
        "persona": {"name": "Helper"},
    }
    encoded = base64.b64encode(json.dumps(payload).encode("utf-8")).decode("ascii")

    result = server.import_persona_bundle(bundle_base64=encoded, signing_key="unused")

    assert result["success"] is False
    assert "signature" in result["error"].lower()


def test_import_rejects_bundle_with_wrong_signature(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _ConfigStub({"persona": "server-secret"})
    server = AtlasServer(config_manager=config)

    monkeypatch.setattr(
        "modules.Server.routes.import_persona_bundle_bytes",
        _stub_import_bundle,
    )

    payload = {
        "metadata": {"persona_name": "Helper"},
        "persona": {"name": "Helper"},
    }
    signed_payload = {
        **payload,
        "signature": {
            "algorithm": BUNDLE_ALGORITHM,
            "value": sign_payload(
                payload,
                signing_key="other-secret",
                error_cls=PersonaBundleError,
            ),
        },
    }
    encoded = base64.b64encode(json.dumps(signed_payload).encode("utf-8")).decode("ascii")

    result = server.import_persona_bundle(bundle_base64=encoded, signing_key="unused")

    assert result["success"] is False
    assert "verification failed" in result["error"].lower()
