# Message bus deep dive

This note documents the ATLAS message bus in enough detail to reason about design changes, performance tuning, and operational behavior without additional context.

## High-level architecture
- **Role:** Asynchronous pub/sub layer for internal events (topic-based routing, priority ordering, correlation IDs, tracing metadata).
- **Backends:** Pluggable implementations behind the `MessageBackend` interface.
  - Default: `InMemoryQueueBackend` (per-topic `asyncio.PriorityQueue`, no external deps).
  - Optional: `RedisStreamBackend` (Redis Streams, durable, multi-process).
- **Orchestrator:** `MessageBus` wraps a backend, manages subscription workers, and owns an event loop if none is running.
- **Global accessors:** `configure_message_bus` sets a process-wide instance; `get_message_bus` lazily creates one; `shutdown_message_bus` closes and clears it.

## Message envelope (`BusMessage`)
- Fields: `topic`, `priority` (lower is higher priority), `payload` (dict), `correlation_id`, `tracing` (dict), `metadata` (dict), `delivery_attempts`, `enqueued_time`, `backend_id` (backend-specific identifier).
- Ordering: `sort_key = (priority, enqueued_time)`; ties by enqueue timestamp preserve FIFO within a priority.
- Correlation: defaults to a new UUID hex string unless provided.
- Enqueue time: bumped with a micro-offset to maintain stable ordering across rapid publications.
- Backend ID: populated by backends that persist entries (e.g., Redis stream ID) and used during acknowledgement.

## InMemoryQueueBackend (asyncio priority queues)
- Per-topic `PriorityQueue` stores `(sort_key, message)`.
- `publish` enqueues; `get` blocks and pops the next entry; `requeue` re-enqueues preserving priority/sort key; `acknowledge` calls `task_done` to release queue backpressure.
- No external resources; `close` is a no-op.
- Best for single-process or ephemeral workloads where durability is not required.

## RedisStreamBackend (Redis Streams)
- **Storage model:** Messages serialized as JSON-friendly fields into streams named `{stream_prefix}:{topic}` using `XADD`.
- **Reading:** `XREAD` with `count=1` and blocking timeout (default 1000 ms). Per-topic async locks currently serialize reads (restores consistency but limits parallelism; see known gaps).
- **Start offsets:** `_start_id` picks the max of pending IDs (unacked deliveries) and the last acknowledged ID; if none, defaults to `$` (tail-only, skips historical entries).
- **Pending tracking:** `_pending[topic]` holds stream IDs handed to consumers; aids resume points and ack updates.
- **Acknowledgement:** Removes the ID from pending, promotes `_last_ids[topic]` to the highest acknowledged/pending ID, and issues `XDEL` asynchronously to prune the stream. If no backend ID is present, it still advances the last ID based on pending.
- **Requeue:** Republishes the message, preserving metadata/tracing; delivery_attempts can be incremented by the worker before requeue.
- **Durability:** Depends on Redis persistence/replication; stream entry is deleted only after ack.

## MessageBus orchestration
- **Loop management:** If no running loop exists, starts a dedicated event loop thread and marks itself as the owner; otherwise reuses the current loop.
- **Publish flows:**
  - `publish` (async) → create `BusMessage` with monotonic enqueue time → backend.publish.
  - `publish_from_sync` (sync-safe) runs the publish coroutine on the owned loop (or current loop) or falls back to `asyncio.run`.
- **Subscriptions:**
  - `subscribe(topic, handler, retry_attempts=3, retry_delay=0.1, concurrency=1)`.
  - Spawns N worker coroutines; each worker `get`s, runs handler, retries on exceptions up to `retry_attempts` with `retry_delay`, requeues failed messages (bumping enqueue time), and always calls `acknowledge` in a `finally` block.
  - Returns a `Subscription` handle with `cancel()` to stop workers and unregister the topic.
- **Shutdown:** Cancels subscription tasks, closes the backend, and stops the owned loop thread if it created one.

## Configuration touchpoints
- `ATLAS/config/messaging.py` normalizes the `messaging` block:
  - Defaults: `backend: in_memory`.
  - For `redis`: populates `redis_url` (default from `REDIS_URL` env or `redis://localhost:6379/0`) and `stream_prefix` (`atlas_bus` default).
- `setup_message_bus(settings, logger)` builds the backend (`RedisStreamBackend` with fall back to in-memory on errors) and calls `configure_message_bus`.
- `ConfigManager.configure_message_bus()` caches the backend/bus using stored messaging settings.
- Setup flows (CLI/GTK) currently expose backend, Redis URL, and stream prefix; initial offset is fixed to tail (`$`) in the backend and not yet user-configurable.

## Reliability and delivery semantics
- At-least-once delivery: messages may be re-delivered on handler failure; duplication is possible on retries.
- Ordering:
  - In-memory: priority-then-FIFO ordering per topic.
  - Redis: stream order per topic; current locking serializes consumption per topic, preventing interleaving across workers.
- Retry policy: handled at the worker level (per subscription) with capped attempts; failed messages beyond the cap are acknowledged (i.e., dropped from the queue/stream).
- Resume points (Redis): computed from pending/acknowledged IDs to avoid skipping entries after requeue; stream entries are deleted only after ack.

## Known gaps / follow-ups
- **Parallel Redis consumption:** A single per-topic lock around `XREAD` removes concurrency benefits; needs per-worker cursors or a narrower critical section to honor `concurrency > 1` without skipping entries.
- **Cold-start replay:** `_start_id` defaults to `$`, so consumers tail only new entries and skip historical messages; no configuration knob yet for replaying from `0-0` or a stored checkpoint.
- **Stream retention:** No TTL/length trimming is configured; operators must manage Redis stream growth externally if acks are delayed or disabled.
- **Dead-letter strategy:** No built-in DLQ after retry exhaustion; handlers must surface failures elsewhere if loss is unacceptable.

## Extension points and usage notes
- Add new backends by implementing `MessageBackend` methods (`publish`, `get`, `requeue`, `acknowledge`, `close`).
- Handlers should be idempotent to tolerate retries and potential duplicates.
- For synchronous publishers, prefer `publish_from_sync` to avoid event-loop handling boilerplate.
- When using Redis, ensure appropriate persistence/replication and consider configuring stream length limits if message volume is high.
