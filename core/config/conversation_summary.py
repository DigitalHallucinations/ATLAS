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
    "followups": {"defaults": [], "personas": {}},
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
        followups = self._normalise_followups(block.get("followups"))
        settings["followups"] = followups
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
            followups = self._normalise_followups(payload.get("followups"))
            if followups["defaults"] or followups["personas"]:
                entry["followups"] = followups
            if entry:
                overrides[tenant_key] = entry
        return overrides

    def _normalise_followups(self, value: Any) -> dict[str, Any]:
        if isinstance(value, Mapping):
            defaults = self._coerce_template_list(value.get("defaults") or value.get("default"))
            personas_raw = value.get("personas")
        elif isinstance(value, (list, tuple, set)):
            defaults = self._coerce_template_list(value)
            personas_raw = None
        else:
            return {"defaults": [], "personas": {}}

        personas: dict[str, list[dict[str, Any]]] = {}
        if isinstance(personas_raw, Mapping):
            for persona, templates in personas_raw.items():
                if not persona:
                    continue
                persona_key = str(persona).strip().lower()
                template_list = self._coerce_template_list(templates)
                if template_list:
                    personas[persona_key] = template_list

        return {"defaults": defaults, "personas": personas}

    def _coerce_template_list(self, value: Any) -> list[dict[str, Any]]:
        if isinstance(value, Mapping):
            candidates = [value]
        elif isinstance(value, (list, tuple, set)):
            candidates = list(value)
        else:
            return []

        templates: list[dict[str, Any]] = []
        for item in candidates:
            template = self._normalise_followup_template(item)
            if template is not None:
                templates.append(template)
        return templates

    def _normalise_followup_template(self, value: Any) -> dict[str, Any] | None:
        if not isinstance(value, Mapping):
            return None

        template_id = str(value.get("id") or value.get("name") or "").strip()
        if not template_id:
            return None

        enabled = value.get("enabled")
        if enabled is not None and not self._as_bool(enabled):
            return None

        title = str(value.get("title") or template_id).strip()
        kind = str(value.get("kind") or "action_item").strip() or "action_item"
        description = str(value.get("description") or "").strip()

        matching = value.get("matching") or value.get("matcher")
        normalized_matching = self._normalise_matching_block(matching)

        task = value.get("task")
        normalized_task = self._normalise_task_block(task)

        escalation = value.get("escalation")
        normalized_escalation = self._normalise_escalation_block(escalation)

        return {
            "id": template_id,
            "title": title,
            "kind": kind,
            "description": description,
            "matching": normalized_matching,
            "task": normalized_task,
            "escalation": normalized_escalation,
        }

    def _normalise_matching_block(self, block: Any) -> dict[str, Any]:
        if not isinstance(block, Mapping):
            return {}

        result: dict[str, Any] = {}

        if self._as_bool(block.get("unanswered_question")):
            result["unanswered_question"] = True

        keywords = block.get("keywords")
        if isinstance(keywords, (list, tuple, set)):
            result["keywords"] = [str(item).strip().lower() for item in keywords if str(item).strip()]

        include_roles = block.get("include_roles") or block.get("roles")
        if isinstance(include_roles, (list, tuple, set)):
            result["include_roles"] = [
                str(role).strip().lower() for role in include_roles if str(role).strip()
            ]

        exclude_roles = block.get("exclude_roles")
        if isinstance(exclude_roles, (list, tuple, set)):
            result["exclude_roles"] = [
                str(role).strip().lower() for role in exclude_roles if str(role).strip()
            ]

        response_roles = block.get("response_roles")
        if isinstance(response_roles, (list, tuple, set)):
            result["response_roles"] = [
                str(role).strip().lower() for role in response_roles if str(role).strip()
            ]

        scope = block.get("scope") or block.get("scopes")
        if isinstance(scope, (list, tuple, set)):
            normalized_scope = []
            for item in scope:
                text = str(item).strip().lower()
                if text in {"history", "summary", "highlights"}:
                    normalized_scope.append(text)
            if normalized_scope:
                result["scope"] = normalized_scope

        pattern = block.get("pattern") or block.get("regex")
        if isinstance(pattern, str) and pattern.strip():
            result["pattern"] = pattern.strip()

        window = block.get("history_window") or block.get("window")
        window_value = self._as_int(window)
        if window_value is not None:
            result["history_window"] = window_value

        return result

    def _normalise_task_block(self, block: Any) -> dict[str, Any] | None:
        if not isinstance(block, Mapping):
            return None

        manifest = str(block.get("manifest") or block.get("name") or "").strip()
        if not manifest:
            return None

        task: dict[str, Any] = {"manifest": manifest}

        persona = block.get("persona")
        if isinstance(persona, str) and persona.strip():
            task["persona"] = persona.strip()

        priority = block.get("priority")
        if isinstance(priority, str) and priority.strip():
            task["priority"] = priority.strip()

        inputs = block.get("inputs")
        if isinstance(inputs, Mapping):
            task["inputs"] = dict(inputs)

        metadata = block.get("metadata")
        if isinstance(metadata, Mapping):
            task["metadata"] = dict(metadata)

        return task

    def _normalise_escalation_block(self, block: Any) -> dict[str, Any] | None:
        if not isinstance(block, Mapping):
            return None

        job = str(block.get("job") or block.get("name") or "").strip()
        if not job:
            return None

        escalation: dict[str, Any] = {"job": job}

        persona = block.get("persona")
        if isinstance(persona, str) and persona.strip():
            escalation["persona"] = persona.strip()

        delay = block.get("delay_minutes") or block.get("delay")
        delay_minutes = self._as_int(delay)
        if delay_minutes is not None:
            escalation["delay_minutes"] = delay_minutes

        metadata = block.get("metadata")
        if isinstance(metadata, Mapping):
            escalation["metadata"] = dict(metadata)

        return escalation

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
