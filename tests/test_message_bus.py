import asyncio

import pytest

from modules.orchestration.message_bus import InMemoryQueueBackend, MessageBus, MessagePriority
from modules.orchestration.policy import MessagePolicy, PolicyResolver


@pytest.mark.asyncio
async def test_message_bus_priority_ordering():
    loop = asyncio.get_running_loop()
    backend = InMemoryQueueBackend()
    bus = MessageBus(backend=backend, loop=loop)
    received = []
    done = asyncio.Event()

    async def handler(message):
        received.append(message.payload["value"])
        if len(received) == 3:
            done.set()

    subscription = bus.subscribe("priority", handler)

    await bus.publish("priority", {"value": "low"}, priority=MessagePriority.LOW)
    await bus.publish("priority", {"value": "high"}, priority=MessagePriority.HIGH)
    await bus.publish("priority", {"value": "normal"}, priority=MessagePriority.NORMAL)

    await asyncio.wait_for(done.wait(), timeout=1.0)
    subscription.cancel()
    await bus.close()

    assert received == ["high", "normal", "low"]


@pytest.mark.asyncio
async def test_message_bus_concurrency():
    loop = asyncio.get_running_loop()
    bus = MessageBus(backend=InMemoryQueueBackend(), loop=loop)
    first_started = asyncio.Event()
    second_started = asyncio.Event()
    processed = []
    done = asyncio.Event()

    async def handler(message):
        value = message.payload["value"]
        if value == "one":
            first_started.set()
            await second_started.wait()
        else:
            second_started.set()
            await first_started.wait()
        processed.append(value)
        if len(processed) == 2:
            done.set()

    subscription = bus.subscribe("concurrency", handler, concurrency=2)

    await bus.publish("concurrency", {"value": "one"})
    await bus.publish("concurrency", {"value": "two"})

    await asyncio.wait_for(done.wait(), timeout=1.0)
    subscription.cancel()
    await bus.close()

    assert set(processed) == {"one", "two"}


@pytest.mark.asyncio
async def test_message_bus_retry_on_failure():
    loop = asyncio.get_running_loop()
    bus = MessageBus(backend=InMemoryQueueBackend(), loop=loop)
    attempts = 0
    done = asyncio.Event()

    async def handler(message):
        nonlocal attempts
        attempts += 1
        if attempts < 2:
            raise RuntimeError("boom")
        done.set()

    subscription = bus.subscribe(
        "retry",
        handler,
        retry_attempts=2,
        retry_delay=0.01,
    )

    await bus.publish("retry", {"value": "payload"})

    await asyncio.wait_for(done.wait(), timeout=1.0)
    subscription.cancel()
    await bus.close()

    assert attempts == 2


def test_policy_resolver_prefers_longest_prefix():
    resolver = PolicyResolver(
        {
            "alerts": MessagePolicy(tier="standard"),
            "alerts.critical": MessagePolicy(tier="critical"),
        }
    )

    assert resolver.resolve("alerts.critical.failure").tier == "critical"
    assert resolver.resolve("alerts.minor").tier == "standard"


@pytest.mark.asyncio
async def test_message_bus_uses_policy_retry_defaults_when_unset():
    loop = asyncio.get_running_loop()
    resolver = PolicyResolver({"policy": MessagePolicy(retry_attempts=1, retry_delay=0)})
    bus = MessageBus(backend=InMemoryQueueBackend(), loop=loop, policy_resolver=resolver)
    attempts = 0
    done = asyncio.Event()

    async def handler(message):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("fail first")
        done.set()

    subscription = bus.subscribe("policy.topic", handler)

    await bus.publish("policy.topic", {"value": "payload"})
    await asyncio.wait_for(done.wait(), timeout=1.0)
    subscription.cancel()
    await bus.close()

    assert attempts == 2


@pytest.mark.asyncio
async def test_message_bus_publishes_dlq_on_exhaustion():
    loop = asyncio.get_running_loop()
    resolver = PolicyResolver({"critical": MessagePolicy(retry_attempts=0)})
    bus = MessageBus(backend=InMemoryQueueBackend(), loop=loop, policy_resolver=resolver)
    dlq_event = asyncio.Event()
    dlq_messages = []

    async def dlq_handler(message):
        dlq_messages.append(message.payload)
        dlq_event.set()

    dlq_subscription = bus.subscribe("dlq.critical.topic", dlq_handler)

    async def failing_handler(message):
        raise RuntimeError("boom")

    failing_subscription = bus.subscribe("critical.topic", failing_handler)

    await bus.publish("critical.topic", {"value": "payload"})
    await asyncio.wait_for(dlq_event.wait(), timeout=1.0)
    failing_subscription.cancel()
    dlq_subscription.cancel()
    await bus.close()

    assert dlq_messages and dlq_messages[0]["original_topic"] == "critical.topic"


@pytest.mark.asyncio
async def test_message_bus_skips_duplicates_with_idempotency_policy():
    loop = asyncio.get_running_loop()
    resolver = PolicyResolver(
        {"idempotent": MessagePolicy(idempotency_key_field="id", idempotency_ttl_seconds=10)}
    )
    bus = MessageBus(backend=InMemoryQueueBackend(), loop=loop, policy_resolver=resolver)
    received = []
    done = asyncio.Event()

    async def handler(message):
        received.append(message.payload["id"])
        if len(received) == 2:
            done.set()

    subscription = bus.subscribe("idempotent.topic", handler)

    await bus.publish("idempotent.topic", {"id": "one"})
    await bus.publish("idempotent.topic", {"id": "one"})
    await bus.publish("idempotent.topic", {"id": "two"})

    await asyncio.wait_for(done.wait(), timeout=1.0)
    subscription.cancel()
    await bus.close()

    assert received == ["one", "two"]


@pytest.mark.asyncio
async def test_message_bus_honors_idempotency_ttl():
    loop = asyncio.get_running_loop()
    resolver = PolicyResolver(
        {"expiring": MessagePolicy(idempotency_key_field="id", idempotency_ttl_seconds=0.05)}
    )
    bus = MessageBus(backend=InMemoryQueueBackend(), loop=loop, policy_resolver=resolver)
    received = []
    first = asyncio.Event()
    done = asyncio.Event()

    async def handler(message):
        received.append(message.payload["id"])
        if len(received) == 1:
            first.set()
        if len(received) == 2:
            done.set()

    subscription = bus.subscribe("expiring.topic", handler)

    await bus.publish("expiring.topic", {"id": "repeat"})
    await asyncio.wait_for(first.wait(), timeout=1.0)
    await asyncio.sleep(0.06)
    await bus.publish("expiring.topic", {"id": "repeat"})

    await asyncio.wait_for(done.wait(), timeout=1.0)
    subscription.cancel()
    await bus.close()

    assert received == ["repeat", "repeat"]
