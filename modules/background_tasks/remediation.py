"""Background worker that resolves persona remediation playbooks."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence

from modules.logging.logger import setup_logger
from core.messaging import (
    AgentBus,
    AgentMessage,
    Subscription,
    get_agent_bus,
)
from modules.orchestration.remediation import (
    RemediationActionContext,
    RemediationOrchestrator,
)


DEFAULT_THRESHOLDS = {"high": 4.0, "medium": 2.5, "low": 1.5}
DEFAULT_ACTIONS = {
    "high": ("health_check", "throttle", "notify"),
    "medium": ("health_check", "throttle"),
    "low": ("health_check",),
    "info": (),
}


def _normalize_persona(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    return text or None


def _normalize_metric(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    return text or None


@dataclass(frozen=True)
class _Policy:
    persona: Optional[str]
    metric: Optional[str]
    thresholds: Mapping[str, float]
    actions: Mapping[str, Sequence[str]]

    def matches(self, persona: Optional[str], metric: Optional[str]) -> bool:
        persona_key = _normalize_persona(persona)
        metric_key = _normalize_metric(metric)

        if self.persona not in {None, "*"} and self.persona != persona_key:
            return False

        pattern = self.metric
        if pattern in {None, "*"}:
            return True

        candidate = metric_key or ""
        if pattern.endswith(".*"):
            prefix = pattern[:-2]
            return candidate.startswith(prefix)
        return candidate == pattern


class PersonaMetricRemediationWorker:
    """Subscribe to analytics alerts and run remediation playbooks."""

    def __init__(
        self,
        *,
        orchestrator: RemediationOrchestrator | None = None,
        agent_bus: AgentBus | None = None,
        config_getter: Callable[[], Mapping[str, Any]] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._bus = agent_bus or get_agent_bus()
        self._orchestrator = orchestrator or RemediationOrchestrator(agent_bus=self._bus)
        self._config_getter = config_getter or (lambda: {})
        self._logger = logger or setup_logger(__name__)
        self._subscription: Subscription | None = None

    # ------------------------------------------------------------------
    # Lifecycle management
    # ------------------------------------------------------------------
    @property
    def is_running(self) -> bool:
        subscription = self._subscription
        return subscription is not None

    async def start(self) -> None:
        if self._subscription is not None:
            return
        self._subscription = await self._bus.subscribe(
            "persona_metrics.alert",
            self._handle_message,
        )

    async def stop(self) -> None:
        subscription = self._subscription
        if subscription is not None:
            await subscription.cancel()
            self._subscription = None

    async def process_alert(self, alert: Mapping[str, Any]) -> None:
        await self._evaluate_alert(alert)

    # ------------------------------------------------------------------
    # Message handling
    # ------------------------------------------------------------------
    async def _handle_message(self, message: AgentMessage) -> None:
        payload = message.payload
        if not isinstance(payload, Mapping):
            self._logger.debug("Ignoring persona_metrics.alert without mapping payload: %s", payload)
            return
        await self._evaluate_alert(payload)

    async def _evaluate_alert(self, alert: Mapping[str, Any]) -> None:
        persona = _normalize_persona(alert.get("persona"))
        metric = _normalize_metric(alert.get("metric"))
        tenant_id = _normalize_persona(alert.get("tenant_id"))
        baseline = alert.get("baseline")
        if not isinstance(baseline, Mapping):
            baseline = {}

        severity, actions, thresholds = self._resolve_actions(
            tenant_id=tenant_id,
            persona=persona,
            metric=metric,
            baseline=baseline,
        )

        if not actions:
            self._logger.debug(
                "No remediation actions mapped for persona=%s metric=%s severity=%s (thresholds=%s)",
                persona,
                metric,
                severity,
                thresholds,
            )
            return

        observed = alert.get("observed")
        try:
            observed_value: Optional[float]
            observed_value = float(observed) if observed is not None else None
        except (TypeError, ValueError):
            observed_value = None

        context = RemediationActionContext(
            tenant_id=tenant_id,
            persona=persona,
            metric=metric,
            severity=severity,
            observed=observed_value,
            baseline=baseline,
            alert=dict(alert),
        )

        for action in actions:
            try:
                if action == "health_check":
                    await self._orchestrator.trigger_health_check(context)
                elif action == "throttle":
                    await self._orchestrator.apply_tool_throttle(context)
                elif action == "notify":
                    await self._orchestrator.notify_operators(context)
                else:
                    self._logger.debug("Unknown remediation action '%s'", action)
            except Exception:  # noqa: BLE001 - surface remediation failures for observability
                self._logger.exception(
                    "Remediation action '%s' failed for persona=%s metric=%s severity=%s",
                    action,
                    persona,
                    metric,
                    severity,
                )

    # ------------------------------------------------------------------
    # Policy resolution helpers
    # ------------------------------------------------------------------
    def _resolve_actions(
        self,
        *,
        tenant_id: Optional[str],
        persona: Optional[str],
        metric: Optional[str],
        baseline: Mapping[str, Any],
    ) -> tuple[str, Sequence[str], Mapping[str, float]]:
        config = self._config_getter()
        defaults, tenant_overrides, policies = self._normalize_config(config, tenant_id)

        thresholds = dict(DEFAULT_THRESHOLDS)
        thresholds.update(defaults["thresholds"])
        actions_map: Dict[str, Sequence[str]] = {
            key: tuple(value) for key, value in DEFAULT_ACTIONS.items()
        }
        actions_map.update({key: tuple(value) for key, value in defaults["actions"].items()})

        thresholds.update(tenant_overrides["thresholds"])
        actions_map.update({key: tuple(value) for key, value in tenant_overrides["actions"].items()})

        matched_policy = self._match_policy(policies, persona, metric)
        if matched_policy is not None:
            thresholds.update(matched_policy.thresholds)
            actions_map.update(
                {key: tuple(value) for key, value in matched_policy.actions.items()}
            )

        severity = self._determine_severity(baseline, thresholds)
        actions = actions_map.get(severity)
        if actions is None or not isinstance(actions, Iterable):
            actions = ()
        return severity, tuple(actions), thresholds

    def _normalize_config(
        self,
        config: Mapping[str, Any] | None,
        tenant_id: Optional[str],
    ) -> tuple[Dict[str, Dict[str, Sequence[str] | Mapping[str, float]]], Dict[str, Any], list[_Policy]]:
        if not isinstance(config, Mapping):
            config = {}

        defaults = self._normalize_policy_block(config.get("defaults"))

        tenants: Mapping[str, Any]
        tenants = config.get("tenants") if isinstance(config.get("tenants"), Mapping) else {}

        tenant_key = tenant_id or "default"
        tenant_block = tenants.get(tenant_key) if isinstance(tenants, Mapping) else None
        if not isinstance(tenant_block, Mapping) and tenant_id:
            tenant_block = tenants.get("default") if isinstance(tenants, Mapping) else None

        if isinstance(tenant_block, Mapping):
            tenant_defaults = self._normalize_policy_block(tenant_block.get("defaults"))
            raw_policies = tenant_block.get("policies")
        else:
            tenant_defaults = {"thresholds": {}, "actions": {}}
            raw_policies = None

        policies: list[_Policy] = []
        if isinstance(raw_policies, Sequence):
            for entry in raw_policies:
                policy = self._normalize_policy_entry(entry)
                if policy is not None:
                    policies.append(policy)

        return defaults, tenant_defaults, policies

    def _normalize_policy_block(self, block: Any) -> Dict[str, Dict[str, Any]]:
        thresholds: Dict[str, float] = {}
        actions: Dict[str, Sequence[str]] = {}
        if isinstance(block, Mapping):
            raw_thresholds = block.get("thresholds")
            if isinstance(raw_thresholds, Mapping):
                for key, value in raw_thresholds.items():
                    try:
                        thresholds[str(key).strip().lower()] = float(value)
                    except (TypeError, ValueError):
                        continue
            raw_actions = block.get("actions")
            if isinstance(raw_actions, Mapping):
                for key, value in raw_actions.items():
                    if isinstance(value, (Sequence, set, tuple, list)):
                        actions[str(key).strip().lower()] = [
                            str(item).strip() for item in value if str(item).strip()
                        ]
        return {"thresholds": thresholds, "actions": actions}

    def _normalize_policy_entry(self, entry: Any) -> _Policy | None:
        if not isinstance(entry, Mapping):
            return None
        persona = _normalize_persona(entry.get("persona"))
        metric = _normalize_metric(entry.get("metric"))
        block = self._normalize_policy_block(entry)
        return _Policy(persona=persona, metric=metric, thresholds=block["thresholds"], actions=block["actions"])

    def _match_policy(
        self,
        policies: Sequence[_Policy],
        persona: Optional[str],
        metric: Optional[str],
    ) -> _Policy | None:
        for policy in policies:
            if policy.matches(persona, metric):
                return policy
        return None

    def _determine_severity(
        self,
        baseline: Mapping[str, Any],
        thresholds: Mapping[str, float],
    ) -> str:
        z_score = baseline.get("z_score")
        try:
            score = float(z_score)
        except (TypeError, ValueError):
            score = None

        if score is None:
            return "info"

        high = thresholds.get("high", DEFAULT_THRESHOLDS["high"])
        medium = thresholds.get("medium", DEFAULT_THRESHOLDS["medium"])
        low = thresholds.get("low", DEFAULT_THRESHOLDS["low"])

        if score >= high:
            return "high"
        if score >= medium:
            return "medium"
        if score >= low:
            return "low"
        return "info"


__all__ = ["PersonaMetricRemediationWorker"]

