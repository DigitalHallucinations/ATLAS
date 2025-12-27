---
audience: Operators and backend developers
status: in_review
last_verified: 2026-07-02
last_updated_hint: Documented Redis initial stream offsets alongside Redis vs in-memory defaults.
source_of_truth: modules/orchestration/message_bus.py
---

# Messaging Bus Deployment

The ATLAS runtime now routes tool, skill, and analytics events through the
central asynchronous message bus located in `modules/orchestration/message_bus.py`.
The bus supports multiple backends so that local development remains lightweight
while production deployments can switch to a durable Redis Streams cluster.

## Configuration

Message bus settings are controlled by the `messaging` block in
`config.yaml` or via environment variables loaded by `ConfigManager`:

```yaml
messaging:
  backend: in_memory  # or "redis"
  redis_url: redis://localhost:6379/0
  stream_prefix: atlas_bus
  initial_stream_id: "$"  # use "0-0" to replay existing entries
```

* `backend` — defaults to `in_memory`. Use `redis` to enable Redis Streams.
* `redis_url` — connection string used when the Redis backend is active. If not
  provided, `ConfigManager` falls back to the `REDIS_URL` environment variable or
  `redis://localhost:6379/0`.
* `stream_prefix` — namespace prefix applied to all stream keys created by the
  bus when using Redis.
* `initial_stream_id` — starting offset for new consumers on a Redis stream. The
  default value `$` only delivers entries published after the consumer starts;
  set to `0-0` or a saved checkpoint to replay existing messages from that ID.

Changes take effect on the next application start. The bus is configured during
`ATLAS` initialization via `ConfigManager.configure_message_bus()`.

## Default backends

- **Local development** defaults to `in_memory` to avoid external dependencies. Events remain in-process and clear on restart.
- **Wizard “Enterprise” preset** enables `redis` automatically so background workers and schedulers can persist queues across service restarts.
- **Missing Redis dependency** triggers a warning and forces the in-memory fallback even if `backend: redis` is set. Install `redis` Python bindings and confirm the server is reachable to return to Redis Streams.

Use Redis when you need durability, multiple worker processes, or cross-host messaging. Prefer the in-memory backend for single-user laptops and short-lived demos where state loss is acceptable.

## Redis Backend Deployment

1. Provision a Redis 6.x (or newer) instance. Redis Streams are required.
2. Secure the deployment with authentication and, if possible, TLS.
3. Expose the connection string as `REDIS_URL` or update the `messaging` block in
   `config.yaml`.
4. Ensure the application host can reach the Redis instance and restart ATLAS.

When Redis is unavailable or the dependency is not installed, ATLAS logs a
warning and automatically falls back to the in-memory backend. In-memory queues
retain events only for the lifetime of the process; prefer Redis for durable or
multi-process scenarios.

## Observability and Tracing

Each message published on the bus includes correlation IDs and tracing metadata
that capture conversation, persona, tool, and skill identifiers. Use these
fields to connect application logs with downstream analytics or monitoring
pipelines when deploying a Redis-backed bus.
