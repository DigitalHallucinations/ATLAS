import asyncio
import time
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from modules.orchestration.message_bus import BusMessage, MessageBus, MessagePriority, RedisStreamGroupBackend
from modules.orchestration.policy import MessagePolicy, PolicyResolver


class _FakeRedisStream:
    """Minimal Redis Streams stand-in to exercise backend sequencing."""

    def __init__(self) -> None:
        self._streams: Dict[str, List[Tuple[str, dict]]] = defaultdict(list)
        self._latest: Dict[str, str] = {}
        self._events: Dict[str, asyncio.Event] = defaultdict(asyncio.Event)
        self._groups: Dict[Tuple[str, str], dict] = {}
        self._pending: Dict[Tuple[str, str], Dict[str, dict]] = defaultdict(dict)

    async def xadd(self, stream_name: str, payload: dict) -> str:
        entry_id = f"{int(time.time() * 1000)}-{len(self._streams[stream_name]) + 1}"
        self._streams[stream_name].append((entry_id, dict(payload)))
        self._latest[stream_name] = entry_id
        self._events[stream_name].set()
        return entry_id

    async def xgroup_create(self, stream_name: str, group_name: str, *, id: str, mkstream: bool = False):
        key = (stream_name, group_name)
        if key in self._groups:
            raise RuntimeError("BUSYGROUP")
        if mkstream and stream_name not in self._streams:
            self._streams[stream_name] = []
        if id == "$":
            last_id = self._latest.get(stream_name)
            if last_id is None and self._streams.get(stream_name):
                last_id = self._streams[stream_name][-1][0]
            id = last_id or "0-0"
        self._groups[key] = {"last_id": id or "0-0"}
        self._pending[key] = {}

    async def xreadgroup(self, group_name: str, consumer_name: str, streams: dict, count: int = 1, block: int = 0):
        stream_name, start_id = next(iter(streams.items()))
        group_key = (stream_name, group_name)
        group = self._groups[group_key]
        start_after = group["last_id"] if start_id == ">" else start_id
        deadline = time.time() + (block / 1000.0 if block else 0.0)

        while True:
            entries = [
                (entry_id, payload)
                for entry_id, payload in self._streams.get(stream_name, [])
                if self._greater(entry_id, start_after) and entry_id not in self._pending[group_key]
            ]
            if entries:
                selected = entries[:count]
                for entry_id, payload in selected:
                    group["last_id"] = entry_id
                    self._pending[group_key][entry_id] = {
                        "payload": payload,
                        "consumer": consumer_name,
                        "timestamp": time.time(),
                    }
                return [(stream_name, selected)]

            if block <= 0:
                return []

            timeout = deadline - time.time()
            if timeout <= 0:
                return []

            try:
                await asyncio.wait_for(self._events[stream_name].wait(), timeout=timeout)
            except asyncio.TimeoutError:
                return []
            finally:
                self._events[stream_name].clear()

    async def xack(self, stream_name: str, group_name: str, *entry_ids: str):
        group_key = (stream_name, group_name)
        pending = self._pending.get(group_key, {})
        for entry_id in entry_ids:
            pending.pop(entry_id, None)
        self._pending[group_key] = pending

    async def xdel(self, stream_name: str, entry_id: str):
        items = self._streams.get(stream_name, [])
        self._streams[stream_name] = [(eid, payload) for eid, payload in items if eid != entry_id]
        for key in list(self._pending.keys()):
            self._pending[key].pop(entry_id, None)

    async def xtrim(self, stream_name: str, *, maxlen: int, approximate: bool = True):
        items = self._streams.get(stream_name, [])
        if maxlen is None or len(items) <= maxlen:
            return
        self._streams[stream_name] = items[-maxlen:]

    async def xautoclaim(
        self,
        stream_name: str,
        group_name: str,
        consumer_name: str,
        *,
        min_idle_time: int,
        start_id: str,
        count: int = 1,
    ):
        group_key = (stream_name, group_name)
        now = time.time()
        pending = self._pending.get(group_key, {})
        entries: list[Tuple[str, dict]] = []
        for entry_id, payload in pending.items():
            if not self._greater(entry_id, start_id):
                continue
            idle_ms = (now - payload["timestamp"]) * 1000
            if idle_ms < min_idle_time:
                continue
            payload["timestamp"] = now
            payload["consumer"] = consumer_name
            entries.append((entry_id, payload["payload"]))
            if len(entries) >= count:
                break
        new_start = entries[-1][0] if entries else start_id
        return new_start, entries

    def age_pending(self, stream_name: str, group_name: str, entry_id: str, *, seconds: float) -> None:
        group_key = (stream_name, group_name)
        if entry_id in self._pending.get(group_key, {}):
            self._pending[group_key][entry_id]["timestamp"] -= seconds

    async def close(self):
        return None

    @staticmethod
    def _greater(candidate: str, floor: str | None) -> bool:
        if floor is None or floor == "$":
            return True
        try:
            cand_ts, cand_seq = candidate.split("-", maxsplit=1)
            floor_ts, floor_seq = floor.split("-", maxsplit=1)
            return (int(cand_ts), int(cand_seq)) > (int(floor_ts), int(floor_seq))
        except ValueError:
            return False


class _TrackingRedisStream(_FakeRedisStream):
    """Redis Stream helper that records concurrent reads."""

    def __init__(self) -> None:
        super().__init__()
        self.max_concurrent_reads = 0
        self._inflight_reads = 0

    async def xreadgroup(self, group_name: str, consumer_name: str, streams: dict, count: int = 1, block: int = 0):
        self._inflight_reads += 1
        self.max_concurrent_reads = max(self.max_concurrent_reads, self._inflight_reads)
        try:
            return await super().xreadgroup(group_name, consumer_name, streams, count=count, block=block)
        finally:
            self._inflight_reads -= 1


async def _roundtrip_requeue() -> tuple[BusMessage, BusMessage, BusMessage]:
    fake_redis = _FakeRedisStream()
    backend = RedisStreamGroupBackend(
        "redis://example",
        stream_prefix="test_bus",
        blocking_timeout=50,
        redis_client=fake_redis,
    )

    original = BusMessage(
        topic="alerts",
        payload={"status": "open"},
        priority=MessagePriority.NORMAL,
        tracing={"source": "test"},
    )

    initial_task = asyncio.create_task(backend.get("alerts"))
    await asyncio.sleep(0)
    await backend.publish(original)
    initial = await asyncio.wait_for(initial_task, timeout=0.5)
    await backend.requeue(initial)
    backend.acknowledge(initial)

    replayed = await asyncio.wait_for(backend.get("alerts"), timeout=0.5)
    await backend.close()
    return original, initial, replayed


def test_redis_backend_requeues_without_skipping_entries():
    original, initial, replayed = asyncio.run(_roundtrip_requeue())

    assert replayed.payload == original.payload
    assert replayed.backend_id != initial.backend_id


async def _consume_existing_entries() -> list[int]:
    fake_redis = _FakeRedisStream()
    backend = RedisStreamGroupBackend(
        "redis://example",
        stream_prefix="existing_entries",
        blocking_timeout=50,
        redis_client=fake_redis,
        replay_start="0-0",
    )

    first = BusMessage(topic="events", payload={"seq": 1})
    second = BusMessage(topic="events", payload={"seq": 2})
    await backend.publish(first)
    await backend.publish(second)

    received_first = await asyncio.wait_for(backend.get("events"), timeout=0.5)
    backend.acknowledge(received_first)
    await asyncio.sleep(0)

    received_second = await asyncio.wait_for(backend.get("events"), timeout=0.5)
    backend.acknowledge(received_second)
    await asyncio.sleep(0)

    await backend.close()
    return [int(received_first.payload["seq"]), int(received_second.payload["seq"])]


def test_redis_backend_consumes_existing_stream_entries():
    consumed = asyncio.run(_consume_existing_entries())

    assert consumed == [1, 2]


async def _tail_only_consumes_new_entries() -> int:
    fake_redis = _FakeRedisStream()
    backend = RedisStreamGroupBackend(
        "redis://example",
        stream_prefix="tail_only",
        blocking_timeout=200,
        redis_client=fake_redis,
        replay_start="$",
    )

    await backend.publish(BusMessage(topic="events", payload={"seq": 1}))
    async def _publish_new_message() -> None:
        await asyncio.sleep(0.05)
        await backend.publish(BusMessage(topic="events", payload={"seq": 2}))

    publish_task = asyncio.create_task(_publish_new_message())
    next_message = await asyncio.wait_for(backend.get("events"), timeout=1.0)
    await publish_task
    backend.acknowledge(next_message)
    await backend.close()
    return int(next_message.payload["seq"])


def test_redis_backend_tail_offset_skips_existing_entries():
    consumed = asyncio.run(_tail_only_consumes_new_entries())

    assert consumed == 2


async def _capture_parallel_consumption() -> tuple[int, list[int]]:
    redis_client = _TrackingRedisStream()
    backend = RedisStreamGroupBackend(
        "redis://example",
        stream_prefix="concurrent_bus",
        blocking_timeout=50,
        redis_client=redis_client,
    )
    bus = MessageBus(backend=backend)
    processed: list[int] = []
    completed = asyncio.Event()

    async def handler(message: BusMessage) -> None:
        processed.append(int(message.payload["seq"]))
        await asyncio.sleep(0.05)
        if len(processed) >= 2:
            completed.set()

    bus.subscribe("events", handler, concurrency=2)
    await asyncio.sleep(0)

    await asyncio.gather(
        bus.publish("events", {"seq": 1}),
        bus.publish("events", {"seq": 2}),
    )

    await asyncio.wait_for(completed.wait(), timeout=1.0)
    await bus.close()
    return redis_client.max_concurrent_reads, processed


def test_message_bus_redis_subscription_allows_parallel_stream_reads():
    max_concurrent_reads, processed = asyncio.run(_capture_parallel_consumption())

    assert max_concurrent_reads >= 2
    assert sorted(processed) == [1, 2]


async def _concurrent_workers_receive_unique_messages() -> list[int]:
    redis_client = _FakeRedisStream()
    backend = RedisStreamGroupBackend(
        "redis://example",
        stream_prefix="dedupe_bus",
        blocking_timeout=50,
        redis_client=redis_client,
    )
    bus = MessageBus(backend=backend)
    processed: list[int] = []
    seen_backend_ids: set[str] = set()
    lock = asyncio.Lock()
    completed = asyncio.Event()
    total = 5

    async def handler(message: BusMessage) -> None:
        async with lock:
            assert message.backend_id not in seen_backend_ids
            seen_backend_ids.add(str(message.backend_id))
            processed.append(int(message.payload["seq"]))
            if len(processed) >= total:
                completed.set()

    bus.subscribe("events", handler, concurrency=3)
    await asyncio.sleep(0)

    await asyncio.gather(*(bus.publish("events", {"seq": i}) for i in range(1, total + 1)))

    await asyncio.wait_for(completed.wait(), timeout=1.0)
    await bus.close()
    return processed


def test_redis_backend_deduplicates_parallel_consumers():
    processed = asyncio.run(_concurrent_workers_receive_unique_messages())

    assert sorted(processed) == [1, 2, 3, 4, 5]


async def _reclaims_idle_pending_message() -> tuple[str, str]:
    stream_prefix = "recover_bus"
    redis_client = _FakeRedisStream()
    backend = RedisStreamGroupBackend(
        "redis://example",
        stream_prefix=stream_prefix,
        blocking_timeout=10,
        auto_claim_idle_ms=1,
        redis_client=redis_client,
        replay_start="0-0",
    )
    await backend.publish(BusMessage(topic="events", payload={"seq": 1}))

    first_delivery = await asyncio.wait_for(backend.get("events"), timeout=0.5)
    stream_name = f"{stream_prefix}:events"
    group_name = f"{stream_prefix}:events:group"
    redis_client.age_pending(stream_name, group_name, str(first_delivery.backend_id), seconds=2)

    reclaimed = await asyncio.wait_for(backend.get("events"), timeout=0.5)
    backend.acknowledge(reclaimed)
    await backend.close()
    return str(first_delivery.backend_id), str(reclaimed.backend_id)


def test_redis_backend_recovers_idle_pending_entries():
    first_id, reclaimed_id = asyncio.run(_reclaims_idle_pending_message())

    assert first_id == reclaimed_id


async def _redis_dlq_delivery() -> dict:
    redis_client = _FakeRedisStream()
    backend = RedisStreamGroupBackend(
        "redis://example",
        stream_prefix="dlq_bus",
        blocking_timeout=50,
        redis_client=redis_client,
    )
    resolver = PolicyResolver({"alerts": MessagePolicy(retry_attempts=0)})
    bus = MessageBus(backend=backend, policy_resolver=resolver)
    dlq_received = asyncio.Event()
    dlq_payload: dict[str, Any] = {}

    async def dlq_handler(message: BusMessage) -> None:
        nonlocal dlq_payload
        dlq_payload = dict(message.payload)
        dlq_received.set()

    bus.subscribe("dlq.alerts", dlq_handler)

    async def failing_handler(message: BusMessage) -> None:
        raise RuntimeError("boom")

    bus.subscribe("alerts", failing_handler)
    await asyncio.sleep(0)
    await bus.publish("alerts", {"value": "payload"})
    await asyncio.wait_for(dlq_received.wait(), timeout=1.0)
    await bus.close()
    return dlq_payload


def test_redis_backend_publishes_dlq_on_failure():
    payload = asyncio.run(_redis_dlq_delivery())

    assert payload.get("original_topic") == "alerts"
