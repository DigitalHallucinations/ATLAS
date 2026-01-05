from __future__ import annotations

import asyncio
from typing import Any, Dict

from core.messaging import AgentBus, AgentMessage
from modules.background_tasks.remediation import PersonaMetricRemediationWorker
from modules.orchestration.remediation import RemediationOrchestrator


def _build_alert(
    *,
    persona: str,
    metric: str,
    z_score: float,
    tenant_id: str | None = None,
    observed: float = 0.0,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "persona": persona,
        "metric": metric,
        "observed": observed,
        "baseline": {"z_score": z_score},
    }
    if tenant_id is not None:
        payload["tenant_id"] = tenant_id
    return payload


def test_remediation_worker_executes_default_actions() -> None:
    asyncio.run(_run_default_action_scenario())


async def _run_default_action_scenario() -> None:
    bus = AgentBus()
    await bus.start()
    orchestrator = RemediationOrchestrator(agent_bus=bus)

    config = {
        "defaults": {
            "thresholds": {"high": 4.0, "medium": 2.0, "low": 1.0},
            "actions": {
                "high": ["health_check", "throttle", "notify"],
                "medium": ["health_check", "throttle"],
                "low": ["health_check"],
            },
        }
    }

    worker = PersonaMetricRemediationWorker(
        orchestrator=orchestrator,
        agent_bus=bus,
        config_getter=lambda: config,
    )

    results: list[tuple[str, Dict[str, Any]]] = []
    done = asyncio.Event()

    async def _collector(topic: str, message: AgentMessage) -> None:
        results.append((topic, message.payload))
        if len(results) >= 3:
            done.set()

    subscriptions = [
        await bus.subscribe(
            RemediationOrchestrator.HEALTH_CHECK_TOPIC,
            lambda message: _collector(RemediationOrchestrator.HEALTH_CHECK_TOPIC, message),
        ),
        await bus.subscribe(
            RemediationOrchestrator.TOOL_THROTTLE_TOPIC,
            lambda message: _collector(RemediationOrchestrator.TOOL_THROTTLE_TOPIC, message),
        ),
        await bus.subscribe(
            RemediationOrchestrator.OPERATOR_NOTIFY_TOPIC,
            lambda message: _collector(RemediationOrchestrator.OPERATOR_NOTIFY_TOPIC, message),
        ),
    ]

    await worker.start()

    await bus.publish(AgentMessage(
        channel="persona_metrics.alert",
        payload=_build_alert(persona="Atlas", metric="tool.failure_rate", z_score=4.5, observed=0.42),
    ))

    await asyncio.wait_for(done.wait(), timeout=2.0)

    topics = {topic for topic, _ in results}
    assert topics == {
        RemediationOrchestrator.HEALTH_CHECK_TOPIC,
        RemediationOrchestrator.TOOL_THROTTLE_TOPIC,
        RemediationOrchestrator.OPERATOR_NOTIFY_TOPIC,
    }

    for topic, payload in results:
        assert payload["persona"] == "atlas"
        assert payload["metric"] == "tool.failure_rate"
        assert payload["severity"] == "high"
        assert payload["baseline"]["z_score"] == 4.5

    await worker.stop()
    for subscription in subscriptions:
        await subscription.cancel()
    await bus.stop()


def test_remediation_worker_respects_tenant_policies() -> None:
    asyncio.run(_run_tenant_policy_scenario())


async def _run_tenant_policy_scenario() -> None:
    bus = AgentBus()
    await bus.start()
    orchestrator = RemediationOrchestrator(agent_bus=bus)

    config = {
        "defaults": {
            "thresholds": {"high": 4.0, "medium": 2.5, "low": 1.0},
            "actions": {"high": ["notify"], "medium": ["throttle"], "low": ["health_check"]},
        },
        "tenants": {
            "tenant-b": {
                "defaults": {
                    "thresholds": {"medium": 1.0},
                    "actions": {"medium": ["health_check"]},
                },
                "policies": [
                    {
                        "persona": "Atlas",
                        "metric": "tool.failure_rate",
                        "thresholds": {"medium": 1.2},
                        "actions": {"medium": ["health_check"]},
                    }
                ],
            }
        },
    }

    worker = PersonaMetricRemediationWorker(
        orchestrator=orchestrator,
        agent_bus=bus,
        config_getter=lambda: config,
    )

    actions: list[tuple[str, Dict[str, Any]]] = []
    done = asyncio.Event()

    async def _collect(topic: str, message: AgentMessage) -> None:
        actions.append((topic, message.payload))
        done.set()

    subscriptions = [
        await bus.subscribe(
            RemediationOrchestrator.HEALTH_CHECK_TOPIC,
            lambda message: _collect(RemediationOrchestrator.HEALTH_CHECK_TOPIC, message),
        ),
        await bus.subscribe(
            RemediationOrchestrator.TOOL_THROTTLE_TOPIC,
            lambda message: _collect(RemediationOrchestrator.TOOL_THROTTLE_TOPIC, message),
        ),
        await bus.subscribe(
            RemediationOrchestrator.OPERATOR_NOTIFY_TOPIC,
            lambda message: _collect(RemediationOrchestrator.OPERATOR_NOTIFY_TOPIC, message),
        ),
    ]

    await worker.start()

    await bus.publish(AgentMessage(
        channel="persona_metrics.alert",
        payload=_build_alert(
            persona="Atlas",
            metric="tool.failure_rate",
            tenant_id="tenant-b",
            z_score=1.3,
            observed=0.11,
        ),
    ))

    await asyncio.wait_for(done.wait(), timeout=2.0)
    await asyncio.sleep(0.05)

    assert len(actions) == 1
    topic, payload = actions[0]
    assert topic == RemediationOrchestrator.HEALTH_CHECK_TOPIC
    assert payload["tenant_id"] == "tenant-b"
    assert payload["severity"] == "medium"
    assert payload["metric"] == "tool.failure_rate"

    await worker.stop()
    for subscription in subscriptions:
        await subscription.cancel()
    await bus.stop()
