"""Conversation summarisation configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, MutableMapping
from collections.abc import Mapping


_DEFAULT_SETTINGS = {
    "enabled": False,
    "cadence_seconds": 300.0,
    "window_seconds": 300.0,
    "batch_size": 10,
    "tool": "context_tracker",
    "retention": {"default_days": None, "tenants": {}},
    "tenants": {},
}


@dataclass
class ConversationSummaryConfigSection:
    """Normalise configuration for automatic conversation snapshots."""

    config: MutableMapping[str, Any]
    yaml_config: MutableMapping[str, Any]
    logger: Any
    write_yaml_callback: Callable[[], None]

    def apply(self) -> None:
        block = self.config.get("conversation_summary")
        if not isinstance(block, Mapping):
            block = {}
        else:
            block = dict(block)
        normalized = self._normalise_settings(block)
        self.config["conversation_summary"] = normalized

    def get_settings(self) -> dict[str, Any]:
        configured = self.config.get("conversation_summary")
        if not isinstance(configured, Mapping):
            return dict(_DEFAULT_SETTINGS)
        return self._normalise_settings(configured)

    def set_settings(self, **updates: Any) -> dict[str, Any]:
        existing = self.get_settings()
        merged = dict(existing)
        merged.update(updates)
        normalized = self._normalise_settings(merged)
        self.yaml_config["conversation_summary"] = dict(normalized)
        self.config["conversation_summary"] = dict(normalized)
        self.write_yaml_callback()
        return normalized

    # ------------------------------------------------------------------
    # Internal helpers

    def _normalise_settings(self, block: Mapping[str, Any]) -> dict[str, Any]:
        settings = dict(_DEFAULT_SETTINGS)
        enabled = block.get("enabled")
        settings["enabled"] = self._as_bool(enabled)
        settings["cadence_seconds"] = self._as_float(block.get("cadence_seconds"), fallback=300.0)
        settings["window_seconds"] = self._as_float(block.get("window_seconds"), fallback=300.0)
        settings["batch_size"] = self._as_int(block.get("batch_size"), fallback=10)
        tool = str(block.get("tool") or "context_tracker").strip()
        settings["tool"] = tool or "context_tracker"
        retention = self._normalise_retention(block.get("retention"))
        settings["retention"] = retention
        tenant_overrides = self._normalise_tenants(block.get("tenants"))
        settings["tenants"] = tenant_overrides
        persona = block.get("persona")
        if isinstance(persona, str) and persona.strip():
            settings["persona"] = persona.strip()
        else:
            settings.pop("persona", None)
        return settings

    def _normalise_retention(self, block: Any) -> dict[str, Any]:
        if not isinstance(block, Mapping):
            block = {}
        result: dict[str, Any] = {"default_days": None, "tenants": {}}
        default_days = self._as_int(block.get("default_days"))
        if default_days is not None:
            result["default_days"] = default_days
        tenants: dict[str, Any] = {}
        tenant_block = block.get("tenants")
        if isinstance(tenant_block, Mapping):
            for tenant, payload in tenant_block.items():
                if not isinstance(payload, Mapping):
                    continue
                ttl = self._as_int(payload.get("days"))
                if ttl is None:
                    continue
                tenants[str(tenant)] = {"days": ttl}
        result["tenants"] = tenants
        return result

    def _normalise_tenants(self, block: Any) -> dict[str, Any]:
        if not isinstance(block, Mapping):
            return {}
        overrides: dict[str, Any] = {}
        for tenant, payload in block.items():
            if not isinstance(payload, Mapping):
                continue
            tenant_key = str(tenant)
            entry: dict[str, Any] = {}
            cadence = self._as_float(payload.get("cadence_seconds"))
            if cadence is not None:
                entry["cadence_seconds"] = cadence
            window = self._as_float(payload.get("window_seconds"))
            if window is not None:
                entry["window_seconds"] = window
            batch = self._as_int(payload.get("batch_size"))
            if batch is not None:
                entry["batch_size"] = batch
            ttl = self._as_int(payload.get("retention_days"))
            if ttl is not None:
                entry["retention_days"] = ttl
            tool = payload.get("tool")
            if isinstance(tool, str) and tool.strip():
                entry["tool"] = tool.strip()
            persona = payload.get("persona")
            if isinstance(persona, str) and persona.strip():
                entry["persona"] = persona.strip()
            if entry:
                overrides[tenant_key] = entry
        return overrides

    @staticmethod
    def _as_bool(value: Any) -> bool:
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"false", "0", "no", "off", "disabled"}:
                return False
            if lowered in {"true", "1", "yes", "on", "enabled"}:
                return True
        return bool(value)

    @staticmethod
    def _as_float(value: Any, fallback: float | None = None) -> float | None:
        if value is None:
            return fallback
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return fallback
        return numeric if numeric > 0 else fallback

    @staticmethod
    def _as_int(value: Any, fallback: int | None = None) -> int | None:
        if value is None:
            return fallback
        try:
            numeric = int(value)
        except (TypeError, ValueError):
            return fallback
        return numeric if numeric > 0 else fallback
