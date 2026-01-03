"""Tests for AgentBus messaging system (NCB-backed).

These tests verify the AgentBus API which provides typed messaging
on top of the Neural Cognitive Bus (NCB).
"""

import asyncio

import pytest

from ATLAS.messaging import AgentBus, AgentMessage, MessagePriority


@pytest.fixture
def agent_bus():
    """Create an AgentBus for testing (not started - tests must manage lifecycle)."""
    return AgentBus()


@pytest.mark.asyncio
async def test_agent_bus_priority_ordering(agent_bus):
    await agent_bus.start()
    try:
        received = []
        done = asyncio.Event()

        async def handler(message: AgentMessage):
            received.append(message.payload["value"])
            if len(received) == 3:
                done.set()

        await agent_bus.subscribe("test.priority", handler)

        await agent_bus.publish(AgentMessage(
            channel="test.priority",
            payload={"value": "low"},
            priority=MessagePriority.LOW
        ))
        await agent_bus.publish(AgentMessage(
            channel="test.priority",
            payload={"value": "high"},
            priority=MessagePriority.HIGH
        ))
        await agent_bus.publish(AgentMessage(
            channel="test.priority",
            payload={"value": "normal"},
            priority=MessagePriority.NORMAL
        ))

        await asyncio.wait_for(done.wait(), timeout=2.0)

        assert received == ["high", "normal", "low"]
    finally:
        await agent_bus.stop()


@pytest.mark.asyncio
async def test_agent_bus_concurrency(agent_bus):
    await agent_bus.start()
    try:
        first_started = asyncio.Event()
        second_started = asyncio.Event()
        processed = []
        done = asyncio.Event()

        async def handler(message: AgentMessage):
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

        await agent_bus.subscribe("test.concurrency", handler, concurrency=2)

        await agent_bus.publish(AgentMessage(
            channel="test.concurrency",
            payload={"value": "one"}
        ))
        await agent_bus.publish(AgentMessage(
            channel="test.concurrency",
            payload={"value": "two"}
        ))

        await asyncio.wait_for(done.wait(), timeout=2.0)

        assert set(processed) == {"one", "two"}
    finally:
        await agent_bus.stop()


@pytest.mark.asyncio
async def test_agent_bus_retry_on_failure(agent_bus):
    await agent_bus.start()
    try:
        attempts = 0
        done = asyncio.Event()

        async def handler(message: AgentMessage):
            nonlocal attempts
            attempts += 1
            if attempts < 2:
                raise RuntimeError("boom")
            done.set()

        await agent_bus.subscribe(
            "test.retry",
            handler,
            retry_attempts=2,
            retry_delay=0.01,
        )

        await agent_bus.publish(AgentMessage(
            channel="test.retry",
            payload={"value": "payload"}
        ))

        await asyncio.wait_for(done.wait(), timeout=2.0)

        assert attempts == 2
    finally:
        await agent_bus.stop()


@pytest.mark.asyncio
async def test_agent_bus_filter_function(agent_bus):
    await agent_bus.start()
    try:
        received = []
        done = asyncio.Event()

        async def handler(message: AgentMessage):
            received.append(message.payload["value"])
            if len(received) == 2:
                done.set()

        def only_even(msg: AgentMessage) -> bool:
            return msg.payload.get("value", 0) % 2 == 0

        await agent_bus.subscribe("test.filter", handler, filter_fn=only_even)

        for i in range(5):
            await agent_bus.publish(AgentMessage(
                channel="test.filter",
                payload={"value": i}
            ))

        await asyncio.wait_for(done.wait(), timeout=2.0)

        assert received == [0, 2]
    finally:
        await agent_bus.stop()


@pytest.mark.asyncio
async def test_agent_bus_subscription_cancel(agent_bus):
    await agent_bus.start()
    try:
        received = []

        async def handler(message: AgentMessage):
            received.append(message.payload["value"])

        subscription = await agent_bus.subscribe("test.cancel", handler)

        await agent_bus.publish(AgentMessage(
            channel="test.cancel",
            payload={"value": "first"}
        ))
        await asyncio.sleep(0.1)

        await subscription.cancel()

        await agent_bus.publish(AgentMessage(
            channel="test.cancel",
            payload={"value": "second"}
        ))
        await asyncio.sleep(0.1)

        # Only first message should be received
        assert received == ["first"]
    finally:
        await agent_bus.stop()


@pytest.mark.asyncio
async def test_agent_bus_typed_message_fields():
    """Test that AgentMessage correctly carries metadata fields."""
    bus = AgentBus()
    await bus.start()

    captured_message = None
    done = asyncio.Event()

    async def handler(message: AgentMessage):
        nonlocal captured_message
        captured_message = message
        done.set()

    await bus.subscribe("test.metadata", handler)

    await bus.publish(AgentMessage(
        channel="test.metadata",
        payload={"data": "test"},
        agent_id="test-agent",
        conversation_id="conv-123",
        request_id="req-456",
        user_id="user-789",
        headers={"custom": "value"},
    ))

    await asyncio.wait_for(done.wait(), timeout=2.0)
    await bus.stop()

    assert captured_message is not None
    assert captured_message.payload == {"data": "test"}
    assert captured_message.agent_id == "test-agent"
    assert captured_message.conversation_id == "conv-123"
    assert captured_message.request_id == "req-456"
    assert captured_message.user_id == "user-789"
    assert captured_message.headers.get("custom") == "value"


@pytest.mark.asyncio
async def test_agent_bus_channel_configuration(agent_bus):
    """Test configuring channel with idempotency settings."""
    await agent_bus.start()
    try:
        await agent_bus.configure_channel(
            "test.configured",
            idempotency_key_field="id",
            idempotency_ttl_seconds=60.0,
        )

        received = []
        done = asyncio.Event()

        async def handler(message: AgentMessage):
            received.append(message.payload["id"])
            done.set()

        await agent_bus.subscribe("test.configured", handler)

        await agent_bus.publish(AgentMessage(
            channel="test.configured",
            payload={"id": "unique-1"}
        ))

        await asyncio.wait_for(done.wait(), timeout=2.0)

        assert received == ["unique-1"]
    finally:
        await agent_bus.stop()
