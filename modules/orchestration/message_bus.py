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
from typing import Any, Awaitable, Callable, Dict, Mapping, Optional

from ATLAS.messaging.NCB import NeuralCognitiveBus

from .policy import MessagePolicy, PolicyResolver

_LOGGER = logging.getLogger(__name__)


class IdempotencyStore:
    """Store for tracking idempotency keys to prevent duplicate message processing."""

    def __init__(self, redis_client: Any | None = None) -> None:
        self._redis = redis_client
        self._in_memory_store: Dict[str, float] = {}

    async def check_and_set(self, key: str, ttl_seconds: float) -> bool:
        """Check if key exists, set it if not. Return True if set (new), False if exists (duplicate)."""
        if self._redis is not None:
            # Use Redis SET with NX (not exists) and EX (expire)
            success = await self._redis.set(key, "1", ex=int(ttl_seconds), nx=True)
            return success is not None
        else:
            # In-memory implementation
            now = time.time()
            expiry = now + ttl_seconds
            if key in self._in_memory_store and self._in_memory_store[key] > now:
                return False  # Exists and not expired
            self._in_memory_store[key] = expiry
            # Clean up expired entries occasionally
            if len(self._in_memory_store) > 1000:
                expired = [k for k, v in self._in_memory_store.items() if v <= now]
                for k in expired:
                    del self._in_memory_store[k]
            return True


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
    backend_id: Optional[str] = field(default=None, compare=False, repr=False)

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
    def acknowledge(self, message: BusMessage) -> None:
        """Mark *message* as processed."""

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

    def acknowledge(self, message: BusMessage) -> None:
        queue = self._queues.get(message.topic)
        if queue is not None:
            queue.task_done()

    async def close(self) -> None:
        # Nothing to release for the in-memory implementation.
        return None


class RedisStreamGroupBackend(MessageBackend):
    """Redis Streams backend backed by consumer groups."""

    def __init__(
        self,
        dsn: str,
        stream_prefix: str = "atlas_bus",
        *,
        redis_client: Any | None = None,
        replay_start: str | None = None,
        blocking_timeout: int = 1000,
        batch_size: int = 1,
        auto_claim_idle_ms: int = 60_000,
        auto_claim_count: int = 10,
        delete_acknowledged: bool = True,
        trim_max_length: int | None = None,
        consumer_name: str | None = None,
    ) -> None:
        if redis_client is None:
            try:
                import redis.asyncio as redis_async  # type: ignore
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError(
                    "Redis backend requested but redis-py is not installed."
                ) from exc

            self._redis = redis_async.from_url(dsn, decode_responses=True)
        else:
            self._redis = redis_client

        self._stream_prefix = stream_prefix
        self._blocking_timeout = int(blocking_timeout)
        self._batch_size = max(int(batch_size), 1)
        self._replay_start = self._normalize_start(replay_start)
        self._auto_claim_idle_ms = max(int(auto_claim_idle_ms), 0)
        self._auto_claim_count = max(int(auto_claim_count), 1)
        self._delete_acknowledged = bool(delete_acknowledged)
        self._trim_max_length = trim_max_length
        self._consumer_name = consumer_name or uuid.uuid4().hex
        self._known_groups: set[str] = set()
        self._auto_claim_cursors: Dict[str, str] = defaultdict(lambda: "0-0")

    async def publish(self, message: BusMessage) -> None:
        stream_name = self._stream_name(message.topic)
        entry_id = await self._redis.xadd(
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
        message.backend_id = entry_id

    async def get(self, topic: str) -> BusMessage:
        stream_name = self._stream_name(topic)
        group_name = self._group_name(topic)
        await self._ensure_group(stream_name, group_name)

        while True:
            entries = await self._read_group(stream_name, group_name)
            if entries:
                entry_id, payload = entries[0]
                return self._to_message(topic, entry_id, payload)

            reclaimed = await self._recover_pending(stream_name, group_name)
            if reclaimed:
                entry_id, payload = reclaimed[0]
                return self._to_message(topic, entry_id, payload)

    async def requeue(self, message: BusMessage) -> None:
        replacement = BusMessage(
            topic=message.topic,
            payload=message.payload,
            priority=message.priority,
            correlation_id=message.correlation_id,
            tracing=message.tracing,
            metadata=dict(message.metadata),
        )
        replacement.delivery_attempts = message.delivery_attempts
        replacement.enqueued_time = message.enqueued_time
        replacement.__post_init__()
        await self.publish(replacement)

    def acknowledge(self, message: BusMessage) -> None:
        entry_id = getattr(message, "backend_id", None)
        topic = message.topic
        if not entry_id:
            return

        async def _acknowledge() -> None:
            stream_name = self._stream_name(topic)
            group_name = self._group_name(topic)
            await self._redis.xack(stream_name, group_name, entry_id)
            if self._delete_acknowledged:
                await self._redis.xdel(stream_name, entry_id)
            elif self._trim_max_length is not None:
                await self._redis.xtrim(stream_name, maxlen=self._trim_max_length, approximate=True)

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(_acknowledge())
        else:
            loop.create_task(_acknowledge())

    async def close(self) -> None:
        await self._redis.close()

    def _stream_name(self, topic: str) -> str:
        return f"{self._stream_prefix}:{topic}"

    def _group_name(self, topic: str) -> str:
        return f"{self._stream_prefix}:{topic}:group"

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

    @staticmethod
    def _parse_stream_id(entry_id: str) -> tuple[int, int]:
        timestamp_str, sequence_str = entry_id.split("-", maxsplit=1)
        return int(timestamp_str), int(sequence_str)

    @classmethod
    def _normalize_start(cls, entry_id: str | None) -> str:
        candidate = (entry_id or "$").strip()
        if not candidate:
            return "$"
        if candidate == "$":
            return "$"
        try:
            cls._parse_stream_id(candidate)
        except ValueError:
            return "$"
        return candidate

    async def _ensure_group(self, stream_name: str, group_name: str) -> None:
        if group_name in self._known_groups:
            return
        try:
            await self._redis.xgroup_create(
                stream_name,
                group_name,
                id=self._replay_start,
                mkstream=True,
            )
        except Exception as exc:  # pragma: no cover - defensive BUSYGROUP handling
            if "BUSYGROUP" not in str(exc):
                raise
        self._known_groups.add(group_name)

    async def _read_group(self, stream_name: str, group_name: str) -> list[tuple[str, dict]]:
        response = await self._redis.xreadgroup(
            group_name,
            self._consumer_name,
            {stream_name: ">"},
            count=self._batch_size,
            block=self._blocking_timeout,
        )
        if not response:
            return []
        _, entries = response[0]
        return entries

    async def _recover_pending(self, stream_name: str, group_name: str) -> list[tuple[str, dict]]:
        if self._auto_claim_idle_ms <= 0:
            return []
        cursor_key = f"{stream_name}:{group_name}"
        start_id = self._auto_claim_cursors[cursor_key]
        try:
            new_start, entries = await self._redis.xautoclaim(
                stream_name,
                group_name,
                self._consumer_name,
                min_idle_time=self._auto_claim_idle_ms,
                start_id=start_id,
                count=self._auto_claim_count,
            )
        except AttributeError:  # pragma: no cover - older redis-py shims
            return []

        self._auto_claim_cursors[cursor_key] = new_start or start_id
        return list(entries or [])

    def _to_message(self, topic: str, entry_id: str, payload: dict[str, Any]) -> BusMessage:
        message = BusMessage(
            topic=topic,
            priority=int(payload.get("priority", MessagePriority.NORMAL)),
            payload=self._deserialize_payload(payload.get("payload")),
            correlation_id=payload.get("correlation_id") or uuid.uuid4().hex,
            tracing=self._deserialize_payload(payload.get("tracing")),
            metadata=self._deserialize_payload(payload.get("metadata")),
        )
        message.backend_id = entry_id
        message.delivery_attempts = int(payload.get("delivery_attempts", 0))
        message.enqueued_time = float(payload.get("enqueued_time", time.time()))
        message.__post_init__()
        return message


RedisStreamBackend = RedisStreamGroupBackend


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
        policy_resolver: Optional[PolicyResolver] = None,
    ) -> None:
        self._backend = backend or InMemoryQueueBackend()
        self._loop = loop
        self._loop_owner = False
        self._sequence = 0
        self._sequence_lock = threading.Lock()
        self._subscriptions: Dict[str, set[Any]] = defaultdict(set)
        self._thread: Optional[threading.Thread] = None
        self._policy_resolver = policy_resolver
        backend_redis = getattr(self._backend, "_redis", None)
        self._idempotency_store = IdempotencyStore(redis_client=backend_redis)
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
        retry_attempts: Optional[int] = None,
        retry_delay: Optional[float] = None,
        concurrency: int = 1,
        policy_resolver: Optional[PolicyResolver] = None,
    ) -> Subscription:
        """Register *handler* for *topic* messages."""

        tasks: set[Any] = set()
        resolver = policy_resolver or self._policy_resolver
        default_policy = MessagePolicy()

        def _resolve_policy() -> MessagePolicy:
            if resolver is None:
                return default_policy
            resolved = resolver.resolve(topic)
            return resolved or default_policy

        async def worker() -> None:
            policy = _resolve_policy()
            effective_retry_attempts = retry_attempts if retry_attempts is not None else policy.retry_attempts
            effective_retry_delay = retry_delay if retry_delay is not None else policy.retry_delay
            idempotency_field = (policy.idempotency_key_field or "").strip()
            idempotency_ttl = self._normalize_idempotency_ttl(policy.idempotency_ttl_seconds)
            while True:
                message = await self._backend.get(topic)
                try:
                    duplicate = False
                    idempotency_key = None
                    if idempotency_field and idempotency_ttl is not None:
                        idempotency_key = self._extract_idempotency_key(message, idempotency_field)
                        if idempotency_key is not None:
                            duplicate = not await self._idempotency_store.check_and_set(
                                idempotency_key, idempotency_ttl
                            )
                            if duplicate:
                                _LOGGER.debug(
                                    "Skipping duplicate message for topic '%s' with idempotency key '%s'.",
                                    topic,
                                    idempotency_key,
                                )

                    if duplicate:
                        continue

                    await handler(message)
                except Exception:  # pylint: disable=broad-except
                    message.delivery_attempts += 1
                    if message.delivery_attempts <= effective_retry_attempts:
                        _LOGGER.exception(
                            "Handler failure for topic '%s'. Retrying (%d/%d).",
                            topic,
                            message.delivery_attempts,
                            effective_retry_attempts,
                        )
                        await asyncio.sleep(effective_retry_delay)
                        message.enqueued_time = time.time() + self._next_sequence_offset()
                        message.__post_init__()
                        await self._backend.requeue(message)
                    else:
                        _LOGGER.exception(
                            "Handler failure for topic '%s'. Max retries exceeded.", topic
                        )
                        self._schedule_dlq_publish(message, policy)
                finally:
                    self._backend.acknowledge(message)

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

    def _schedule_dlq_publish(self, message: BusMessage, policy: MessagePolicy) -> None:
        template = policy.dlq_topic_template
        if not template:
            return

        dlq_topic = template.format(topic=message.topic)
        dlq_payload = {
            "original_topic": message.topic,
            "payload": message.payload,
            "metadata": message.metadata,
            "correlation_id": message.correlation_id,
            "tracing": message.tracing,
            "delivery_attempts": message.delivery_attempts,
            "enqueued_time": message.enqueued_time,
        }
        dlq_metadata = dict(message.metadata)
        dlq_metadata.setdefault("source_topic", message.topic)
        dlq_message = self._create_message(
            dlq_topic,
            dlq_payload,
            priority=message.priority,
            correlation_id=message.correlation_id,
            tracing=message.tracing,
            metadata=dlq_metadata,
        )

        async def _publish_dlq() -> None:
            try:
                await self._backend.publish(dlq_message)
            except Exception:  # pragma: no cover - defensive logging path
                _LOGGER.exception("Failed to publish DLQ message for topic '%s'.", message.topic)

        self._dispatch_background_task(_publish_dlq())

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

    def _dispatch_background_task(self, coro: Awaitable[None]) -> None:
        """Run *coro* without blocking the current handler."""

        if self._loop_owner and self._loop is not None:
            asyncio.run_coroutine_threadsafe(coro, self._loop)
            return

        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(lambda: asyncio.create_task(coro))
            return

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(coro)
        else:
            loop.create_task(coro)

    @staticmethod
    def _get_nested_field(mapping: Mapping[str, Any] | None, path: str) -> Any:
        if not isinstance(mapping, Mapping):
            return None
        current: Any = mapping
        for part in path.split("."):
            if not isinstance(current, Mapping) or part not in current:
                return None
            current = current[part]
        return current

    def _extract_idempotency_key(self, message: BusMessage, field_path: str) -> str | None:
        key = self._get_nested_field(message.payload, field_path)
        if key is None:
            key = self._get_nested_field(message.metadata, field_path)
        if key is None:
            return None
        normalized = str(key).strip()
        return normalized or None

    @staticmethod
    def _normalize_idempotency_ttl(value: float | int | None) -> float | None:
        if value is None:
            return None
        try:
            ttl = float(value)
        except (TypeError, ValueError):
            return None
        if ttl <= 0:
            return None
        return ttl


_global_bus: Optional[MessageBus] = None


class NCBMessageBus(MessageBus):
    """MessageBus implementation using NeuralCognitiveBus."""

    def __init__(self, ncb: NeuralCognitiveBus, *, loop: Optional[asyncio.AbstractEventLoop] = None, policy_resolver: Optional[PolicyResolver] = None):
        # Don't call super().__init__ to avoid backend setup
        self._ncb = ncb
        self._loop = loop
        self._policy_resolver = policy_resolver
        self._idempotency_store = IdempotencyStore(redis_client=None)  # NCB handles persistence
        self._subscriptions: Dict[str, set[Any]] = defaultdict(set)

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
        meta = dict(metadata or {})
        if correlation_id:
            meta["correlation_id"] = correlation_id
        if tracing:
            meta.update(tracing)
        return await self._ncb.publish(topic, payload, priority=priority, meta=meta)

    def subscribe(
        self,
        topic: str,
        handler: Callable[[BusMessage], Awaitable[None]],
        *,
        retry_attempts: Optional[int] = None,
        retry_delay: Optional[float] = None,
        concurrency: int = 1,
        policy_resolver: Optional[PolicyResolver] = None,
    ) -> Subscription:
        # Use NCB's subscription with wrapped handler
        resolver = policy_resolver or self._policy_resolver
        default_policy = MessagePolicy()

        def _resolve_policy() -> MessagePolicy:
            if resolver is None:
                return default_policy
            resolved = resolver.resolve(topic)
            return resolved or default_policy

        async def wrapped_handler(msg: Any):
            # Convert NCB Message to BusMessage
            bus_msg = BusMessage(
                topic=msg.channel,
                payload=msg.payload,
                correlation_id=msg.meta.get("correlation_id", ""),
                tracing=msg.meta,
                metadata=msg.meta,
                enqueued_time=msg.ts,
                backend_id=msg.id,
            )
            # Handle retries, idempotency as in MessageBus
            policy = _resolve_policy()
            effective_retry_attempts = retry_attempts if retry_attempts is not None else policy.retry_attempts
            effective_retry_delay = retry_delay if retry_delay is not None else policy.retry_delay
            idempotency_field = (policy.idempotency_key_field or "").strip()
            idempotency_ttl = self._normalize_idempotency_ttl(policy.idempotency_ttl_seconds)

            duplicate = False
            idempotency_key = None
            if idempotency_field and idempotency_ttl is not None:
                idempotency_key = self._extract_idempotency_key(bus_msg, idempotency_field)
                if idempotency_key is not None:
                    duplicate = not await self._idempotency_store.check_and_set(idempotency_key, idempotency_ttl)
                    if duplicate:
                        _LOGGER.debug(
                            "Skipping duplicate message for topic '%s' with idempotency key '%s'.",
                            topic,
                            idempotency_key,
                        )

            if duplicate:
                return

            try:
                await handler(bus_msg)
            except Exception:
                bus_msg.delivery_attempts += 1
                if bus_msg.delivery_attempts <= effective_retry_attempts:
                    _LOGGER.exception(
                        "Handler failure for topic '%s'. Retrying (%d/%d).",
                        topic,
                        bus_msg.delivery_attempts,
                        effective_retry_attempts,
                    )
                    await asyncio.sleep(effective_retry_delay)
                    # Re-publish or handle retry
                    # For simplicity, just log; NCB handles dead letters

        # Register with NCB
        module_name = f"atlas_{id(handler)}"
        asyncio.create_task(self._ncb.register_subscriber(topic, module_name, wrapped_handler))
        self._subscriptions[topic].add((module_name, handler))
        
        def cancel():
            asyncio.create_task(self._ncb.unregister_subscriber(topic, module_name))
            self._subscriptions[topic].discard((module_name, handler))
        
        return Subscription(cancel)

    async def close(self) -> None:
        await self._ncb.stop()

    # Helper methods from MessageBus
    def _normalize_idempotency_ttl(self, ttl_seconds: Optional[int]) -> Optional[int]:
        if ttl_seconds is None or ttl_seconds <= 0:
            return None
        return ttl_seconds

    def _extract_idempotency_key(self, message: BusMessage, field: str) -> Optional[str]:
        try:
            value = message.payload
            for key in field.split("."):
                if isinstance(value, Mapping):
                    value = value.get(key)
                else:
                    return None
            return str(value) if value is not None else None
        except Exception:
            return None


def configure_message_bus(backend: Optional[MessageBackend] = None, *, policy_resolver: Optional[PolicyResolver] = None) -> MessageBus:
    """Configure the global message bus instance."""

    global _global_bus
    _global_bus = MessageBus(backend=backend, policy_resolver=policy_resolver)
    return _global_bus


def configure_ncb_message_bus(ncb: NeuralCognitiveBus, *, loop: Optional[asyncio.AbstractEventLoop] = None, policy_resolver: Optional[PolicyResolver] = None) -> NCBMessageBus:
    """Configure the global message bus instance with NCB."""

    global _global_bus
    _global_bus = NCBMessageBus(ncb, loop=loop, policy_resolver=policy_resolver)  # type: ignore
    return _global_bus  # type: ignore


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
