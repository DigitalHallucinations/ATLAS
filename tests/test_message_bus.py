import asyncio

import pytest

from modules.orchestration.message_bus import (
    InMemoryQueueBackend,
    MessageBus,
    MessagePriority,
)


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
