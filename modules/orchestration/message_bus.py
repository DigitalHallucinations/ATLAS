"""Asynchronous message bus utilities.

This module provides a thin abstraction around multiple durable pub/sub
backends.  The default backend is an in-process asyncio priority queue which
requires no external services.  Optionally a Redis Stream backend can be used
for production durability.

Messages published on the bus support topic based routing, priority ordering,
correlation identifiers and structured tracing metadata.  Subscribers run in
background tasks and automatically retry message delivery when handlers raise
exceptions.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Optional

_LOGGER = logging.getLogger(__name__)


class MessagePriority:
    """Common priority levels used by the message bus."""

    HIGH = 0
    NORMAL = 5
    LOW = 10


@dataclass(order=True)
class BusMessage:
    """Envelope representing a single message queued on the bus."""

    sort_key: tuple[int, float] = field(init=False, repr=False)
    topic: str = field(compare=False)
    priority: int = field(default=MessagePriority.NORMAL, compare=False)
    payload: Dict[str, Any] = field(default_factory=dict, compare=False)
    correlation_id: str = field(default_factory=lambda: uuid.uuid4().hex, compare=False)
    tracing: Dict[str, Any] | None = field(default=None, compare=False)
    delivery_attempts: int = field(default=0, compare=False)
    metadata: Dict[str, Any] = field(default_factory=dict, compare=False)
    enqueued_time: float = field(default_factory=time.time, compare=False)

    def __post_init__(self) -> None:  # noqa: D401 - dataclass hook
        # Lower priority numbers should be processed first.  When priorities are
        # identical we fall back to FIFO ordering using the enqueue timestamp.
        self.sort_key = (self.priority, self.enqueued_time)


class MessageBackend(ABC):
    """Abstract interface implemented by message backends."""

    @abstractmethod
    async def publish(self, message: BusMessage) -> None:
        """Persist *message* for consumption by subscribers."""

    @abstractmethod
    async def get(self, topic: str) -> BusMessage:
        """Fetch the next available message for *topic*."""

    @abstractmethod
    async def requeue(self, message: BusMessage) -> None:
        """Requeue *message* for later delivery."""

    @abstractmethod
    def acknowledge(self, topic: str) -> None:
        """Mark the current message for *topic* as processed."""

    @abstractmethod
    async def close(self) -> None:
        """Release any backend resources."""


class InMemoryQueueBackend(MessageBackend):
    """Priority queue backed by :mod:`asyncio` primitives."""

    def __init__(self) -> None:
        self._queues: Dict[str, asyncio.PriorityQueue[tuple[tuple[int, float], BusMessage]]] = defaultdict(
            asyncio.PriorityQueue
        )

    async def publish(self, message: BusMessage) -> None:
        queue = self._queues[message.topic]
        await queue.put((message.sort_key, message))

    async def get(self, topic: str) -> BusMessage:
        queue = self._queues[topic]
        _, message = await queue.get()
        return message

    async def requeue(self, message: BusMessage) -> None:
        queue = self._queues[message.topic]
        await queue.put((message.sort_key, message))

    def acknowledge(self, topic: str) -> None:
        queue = self._queues.get(topic)
        if queue is not None:
            queue.task_done()

    async def close(self) -> None:
        # Nothing to release for the in-memory implementation.
        return None


class RedisStreamBackend(MessageBackend):
    """Redis Streams based backend.

    The implementation is intentionally lazy to avoid importing redis unless it
    is required at runtime.  Messages are stored as JSON friendly dictionaries
    in Redis streams using the pattern ``{stream_prefix}:{topic}``.
    """

    def __init__(self, dsn: str, stream_prefix: str = "atlas_bus", blocking_timeout: int = 1000) -> None:
        try:
            import redis.asyncio as redis_async  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Redis backend requested but redis-py is not installed."
            ) from exc

        self._redis = redis_async.from_url(dsn, decode_responses=True)
        self._stream_prefix = stream_prefix
        self._blocking_timeout = blocking_timeout
        self._pending: Dict[str, str] = {}

    async def publish(self, message: BusMessage) -> None:
        stream_name = self._stream_name(message.topic)
        await self._redis.xadd(
            stream_name,
            {
                "priority": message.priority,
                "payload": self._serialize_payload(message.payload),
                "correlation_id": message.correlation_id,
                "tracing": self._serialize_payload(message.tracing or {}),
                "metadata": self._serialize_payload(message.metadata),
                "enqueued_time": message.enqueued_time,
                "delivery_attempts": message.delivery_attempts,
            },
        )

    async def get(self, topic: str) -> BusMessage:
        stream_name = self._stream_name(topic)
        while True:
            response = await self._redis.xread({stream_name: self._pending.get(topic, "$")}, count=1, block=self._blocking_timeout)
            if not response:
                continue
            _, entries = response[0]
            entry_id, payload = entries[0]
            self._pending[topic] = entry_id
            message = BusMessage(
                topic=topic,
                priority=int(payload.get("priority", MessagePriority.NORMAL)),
                payload=self._deserialize_payload(payload.get("payload")),
                correlation_id=payload.get("correlation_id") or uuid.uuid4().hex,
                tracing=self._deserialize_payload(payload.get("tracing")),
                metadata=self._deserialize_payload(payload.get("metadata")),
            )
            message.delivery_attempts = int(payload.get("delivery_attempts", 0))
            message.enqueued_time = float(payload.get("enqueued_time", time.time()))
            message.__post_init__()  # recompute sort key
            return message

    async def requeue(self, message: BusMessage) -> None:
        await self.publish(message)

    def acknowledge(self, topic: str) -> None:
        pending_id = self._pending.pop(topic, None)
        if pending_id:
            stream_name = self._stream_name(topic)
            asyncio.create_task(self._redis.xdel(stream_name, pending_id))

    async def close(self) -> None:
        await self._redis.close()

    def _stream_name(self, topic: str) -> str:
        return f"{self._stream_prefix}:{topic}"

    @staticmethod
    def _serialize_payload(payload: Any) -> str:
        import json

        return json.dumps(payload, default=str)

    @staticmethod
    def _deserialize_payload(value: Optional[str]) -> Dict[str, Any]:
        import json

        if not value:
            return {}
        try:
            data = json.loads(value)
        except json.JSONDecodeError:
            return {"value": value}
        if isinstance(data, dict):
            return data
        return {"value": data}


class Subscription:
    """Handle to a bus subscription."""

    def __init__(self, cancel_callback: Callable[[], None]) -> None:
        self._cancel = cancel_callback

    def cancel(self) -> None:
        self._cancel()


class MessageBus:
    """Coordinate publication and consumption of bus messages."""

    def __init__(
        self,
        backend: Optional[MessageBackend] = None,
        *,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> None:
        self._backend = backend or InMemoryQueueBackend()
        self._loop = loop
        self._loop_owner = False
        self._sequence = 0
        self._sequence_lock = threading.Lock()
        self._subscriptions: Dict[str, set[Any]] = defaultdict(set)
        self._thread: Optional[threading.Thread] = None
        if self._loop is None:
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                self._loop = asyncio.new_event_loop()
                self._loop_owner = True
                self._thread = threading.Thread(target=self._run_loop, args=(self._loop,), daemon=True)
                self._thread.start()

    @staticmethod
    def _run_loop(loop: asyncio.AbstractEventLoop) -> None:
        asyncio.set_event_loop(loop)
        loop.run_forever()

    async def publish(
        self,
        topic: str,
        payload: Dict[str, Any],
        *,
        priority: int = MessagePriority.NORMAL,
        correlation_id: Optional[str] = None,
        tracing: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Publish a message to *topic* and return the correlation identifier."""

        message = self._create_message(
            topic,
            payload,
            priority=priority,
            correlation_id=correlation_id,
            tracing=tracing,
            metadata=metadata,
        )
        await self._backend.publish(message)
        return message.correlation_id

    def publish_from_sync(
        self,
        topic: str,
        payload: Dict[str, Any],
        *,
        priority: int = MessagePriority.NORMAL,
        correlation_id: Optional[str] = None,
        tracing: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Publish *payload* from synchronous code."""

        message = self._create_message(
            topic,
            payload,
            priority=priority,
            correlation_id=correlation_id,
            tracing=tracing,
            metadata=metadata,
        )

        async def _publish() -> None:
            await self._backend.publish(message)

        if self._loop_owner:
            future = asyncio.run_coroutine_threadsafe(_publish(), self._loop)
            future.result()
        elif self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(lambda: asyncio.create_task(_publish()))
        else:
            asyncio.run(_publish())
        return message.correlation_id

    def subscribe(
        self,
        topic: str,
        handler: Callable[[BusMessage], Awaitable[None]],
        *,
        retry_attempts: int = 3,
        retry_delay: float = 0.1,
        concurrency: int = 1,
    ) -> Subscription:
        """Register *handler* for *topic* messages."""

        tasks: set[Any] = set()

        async def worker() -> None:
            while True:
                message = await self._backend.get(topic)
                try:
                    await handler(message)
                except Exception:  # pylint: disable=broad-except
                    message.delivery_attempts += 1
                    if message.delivery_attempts <= retry_attempts:
                        _LOGGER.exception(
                            "Handler failure for topic '%s'. Retrying (%d/%d).",
                            topic,
                            message.delivery_attempts,
                            retry_attempts,
                        )
                        await asyncio.sleep(retry_delay)
                        message.enqueued_time = time.time() + self._next_sequence_offset()
                        message.__post_init__()
                        await self._backend.requeue(message)
                    else:
                        _LOGGER.exception(
                            "Handler failure for topic '%s'. Max retries exceeded.", topic
                        )
                finally:
                    self._backend.acknowledge(topic)

        for _ in range(max(concurrency, 1)):
            if self._loop_owner:
                task = asyncio.run_coroutine_threadsafe(worker(), self._loop)
            else:
                task = self._loop.create_task(worker())
            tasks.add(task)

        self._subscriptions[topic] |= tasks

        def cancel() -> None:
            for task in list(tasks):
                task.cancel()
            self._subscriptions[topic] -= tasks
            if not self._subscriptions[topic]:
                self._subscriptions.pop(topic, None)

        return Subscription(cancel)

    async def close(self) -> None:
        for topic, tasks in list(self._subscriptions.items()):
            for task in tasks:
                task.cancel()
            self._subscriptions.pop(topic, None)
        await self._backend.close()
        if self._loop_owner and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._thread is not None:
                self._thread.join(timeout=1)

    def _next_sequence_offset(self) -> float:
        with self._sequence_lock:
            self._sequence += 1
            return self._sequence / 1_000_000

    def _create_message(
        self,
        topic: str,
        payload: Dict[str, Any],
        *,
        priority: int,
        correlation_id: Optional[str],
        tracing: Optional[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]],
    ) -> BusMessage:
        message = BusMessage(
            topic=topic,
            payload=payload,
            priority=priority,
            correlation_id=correlation_id or uuid.uuid4().hex,
            tracing=tracing,
            metadata=metadata or {},
        )
        message.enqueued_time = time.time() + self._next_sequence_offset()
        message.__post_init__()
        return message


_global_bus: Optional[MessageBus] = None


def configure_message_bus(backend: Optional[MessageBackend] = None) -> MessageBus:
    """Configure the global message bus instance."""

    global _global_bus
    _global_bus = MessageBus(backend=backend)
    return _global_bus


def get_message_bus() -> MessageBus:
    """Return the global message bus, creating it if necessary."""

    global _global_bus
    if _global_bus is None:
        _global_bus = MessageBus()
    return _global_bus


async def shutdown_message_bus() -> None:
    """Gracefully close the global message bus."""

    global _global_bus
    if _global_bus is not None:
        await _global_bus.close()
        _global_bus = None
