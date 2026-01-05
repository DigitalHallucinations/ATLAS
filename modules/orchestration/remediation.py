"""Utilities for executing persona remediation playbooks.

This module bridges remediation actions triggered from analytics alerts into
the orchestration subsystem.  Actions are intentionally coarse grained so that
additional handlers can subscribe to the emitted bus topics without requiring
direct coupling to the worker implementation.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Dict, Mapping, MutableMapping, Optional, Tuple

from modules.orchestration import budget_tracker
from core.messaging import (
    AgentBus,
    AgentMessage,
    MessagePriority,
    get_agent_bus,
)


@dataclass(frozen=True)
class RemediationActionContext:
    """Context payload describing a remediation action request."""

    tenant_id: Optional[str]
    persona: Optional[str]
    metric: Optional[str]
    severity: str
    observed: Optional[float]
    baseline: Mapping[str, Any]
    alert: Mapping[str, Any]


class RemediationOrchestrator:
    """Dispatch remediation actions to orchestration collaborators."""

    HEALTH_CHECK_TOPIC = "orchestration.health_checks.request"
    TOOL_THROTTLE_TOPIC = "orchestration.tooling.throttle"
    OPERATOR_NOTIFY_TOPIC = "operations.notifications"

    def __init__(
        self,
        *,
        agent_bus: AgentBus | None = None,
    ) -> None:
        self._bus = agent_bus or get_agent_bus()
        self._throttle_state: MutableMapping[
            Tuple[Optional[str], Optional[str], Optional[str]],
            Mapping[str, Any],
        ] = {}
        self._lock = threading.Lock()

    async def trigger_health_check(self, context: RemediationActionContext) -> None:
        """Publish a health-check request for the impacted persona metric."""

        message = AgentMessage(
            channel=self.HEALTH_CHECK_TOPIC,
            payload={
                "tenant_id": context.tenant_id,
                "persona": context.persona,
                "metric": context.metric,
                "severity": context.severity,
                "observed": context.observed,
                "baseline": dict(context.baseline),
                "alert": dict(context.alert),
            },
            priority=MessagePriority.HIGH,
            headers={"component": "remediation"},
        )
        await self._bus.publish(message)

    async def apply_tool_throttle(self, context: RemediationActionContext) -> Mapping[str, Any]:
        """Record and publish a throttle directive for the affected persona."""

        key = (context.tenant_id, context.persona, context.metric)
        payload = {
            "tenant_id": context.tenant_id,
            "persona": context.persona,
            "metric": context.metric,
            "severity": context.severity,
            "observed": context.observed,
            "baseline": dict(context.baseline),
            "alert": dict(context.alert),
        }

        with self._lock:
            snapshot = {
                "count": int(self._throttle_state.get(key, {}).get("count", 0)) + 1,
                "severity": context.severity,
                "last_observed": context.observed,
                "baseline": dict(context.baseline),
            }
            self._throttle_state[key] = snapshot

        message = AgentMessage(
            channel=self.TOOL_THROTTLE_TOPIC,
            payload=payload,
            priority=MessagePriority.HIGH,
            headers={"component": "remediation"},
        )
        await self._bus.publish(message)

        return snapshot

    async def notify_operators(self, context: RemediationActionContext) -> None:
        """Emit an operator notification request for downstream handlers."""

        message = AgentMessage(
            channel=self.OPERATOR_NOTIFY_TOPIC,
            payload={
                "tenant_id": context.tenant_id,
                "persona": context.persona,
                "metric": context.metric,
                "severity": context.severity,
                "observed": context.observed,
                "baseline": dict(context.baseline),
                "alert": dict(context.alert),
            },
            priority=MessagePriority.HIGH,
            headers={"component": "remediation"},
        )
        await self._bus.publish(message)

    def get_throttle_snapshot(
        self,
        tenant_id: Optional[str] = None,
        persona: Optional[str] = None,
        metric: Optional[str] = None,
    ) -> Dict[Tuple[Optional[str], Optional[str], Optional[str]], Mapping[str, Any]]:
        """Return recorded throttle directives filtered by the provided keys."""

        with self._lock:
            if tenant_id is None and persona is None and metric is None:
                return dict(self._throttle_state)

            result: Dict[Tuple[Optional[str], Optional[str], Optional[str]], Mapping[str, Any]] = {}
            for key, value in self._throttle_state.items():
                key_tenant, key_persona, key_metric = key
                if tenant_id is not None and tenant_id != key_tenant:
                    continue
                if persona is not None and persona != key_persona:
                    continue
                if metric is not None and metric != key_metric:
                    continue
                result[key] = value
            return result

    async def reset_runtime_budget(self, conversation_id: Optional[str] = None) -> None:
        """Proxy for resetting the tracked runtime budgets via budget tracker."""

        await budget_tracker.reset_runtime(conversation_id)


__all__ = [
    "RemediationActionContext",
    "RemediationOrchestrator",
]

