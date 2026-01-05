---
audience: Operators and backend developers
status: current
last_verified: 2026-01-01
last_updated_hint: Migrated from legacy MessageBus to NCB/AgentBus architecture.
source_of_truth: ATLAS/messaging/agent_bus.py; ATLAS/messaging/NCB.py
---

# Messaging Bus Deployment

The ATLAS runtime routes tool, skill, orchestration, and analytics events through
the **AgentBus**, a high-level messaging API backed by the **Neural Cognitive Bus (NCB)**.
The NCB provides domain-specific channels, priority queues, idempotency, dead-letter
handling, and optional external transport bridging (Redis, Kafka).

## Architecture

The messaging system consists of:

- **AgentBus** (`ATLAS/messaging/agent_bus.py`) — High-level typed API with `publish()`,
  `subscribe()`, channel configuration, and convenience methods for common message patterns.
- **NCB** (`ATLAS/messaging/NCB.py`) — Core async engine with priority queues, SQLite
  persistence, retry support, and optional Redis/Kafka bridging.
- **Channels** (`ATLAS/messaging/channels.py`) — 36+ fine-grained semantic channels
  (e.g., `user.input`, `llm.request`, `tool.invoke`, `task.created`, `job.complete`).
- **AgentMessage** (`ATLAS/messaging/messages.py`) — Base message type with ATLAS context
  fields (agent_id, conversation_id, request_id, user_id, trace_id, payload).

## Configuration

Message bus settings are controlled by the `messaging` block in `config.yaml`:

```yaml
messaging:
  backend: ncb  # default; "redis" or "kafka" for external bridging
  redis_url: redis://localhost:6379/0
  kafka:
    enabled: false
    bootstrap_servers: kafka:9092
```

* `backend` — defaults to `ncb` (in-process async queues). External transports
  bridge events to Redis or Kafka when configured.
* `redis_url` — connection string for Redis bridging when enabled.
* `kafka.enabled` — enables Kafka producer bridging for external consumers.
* `kafka.bootstrap_servers` — Kafka cluster connection string.

Changes take effect on the next application start. The bus is configured during
`ATLAS` initialization via `configure_agent_bus()`.

## Channel Architecture

The NCB uses domain-specific channels instead of generic topics:

| Channel Prefix | Purpose |
| --- | --- |
| `user.*` | User input/output events |
| `llm.*` | LLM request/response/streaming |
| `tool.*` | Tool invocation and results |
| `agent.*` | Agent lifecycle and routing |
| `task.*` | Task lifecycle events |
| `job.*` | Job scheduling and completion |
| `conversation.*` | Conversation lifecycle |
| `system.*` | System health and metrics |
| `blackboard.*` | Shared collaboration state |
| `skill.*` | Skill execution events |

## Usage Patterns

### Publishing messages

```python
from core.messaging import get_agent_bus, AgentMessage

bus = get_agent_bus()
await bus.publish(AgentMessage(
    channel="tool.invoke",
    payload={"tool": "calculator", "args": {"x": 1}},
    agent_id="main",
    conversation_id="conv-123",
))
```

### Subscribing to channels

```python
async def handle_tool_result(message: AgentMessage):
    print(f"Tool result: {message.payload}")

await bus.subscribe("tool.result", handle_tool_result)
```

### Priority and retry

Messages support priority levels (LOW, NORMAL, HIGH, CRITICAL) and automatic
retry with exponential backoff for transient failures.

## Deployment Options

### Local Development (default)

Uses in-process async queues. Events remain in-memory and clear on restart.
No external dependencies required.

### Production with Redis

Enable Redis bridging for durable queues and cross-process messaging:

```yaml
messaging:
  backend: ncb
  redis_url: redis://localhost:6379/0
```

### Production with Kafka

Enable Kafka bridging for external analytics consumers:

```yaml
messaging:
  kafka:
    enabled: true
    bootstrap_servers: kafka:9092
```

## Observability

Each message includes correlation IDs and tracing metadata:
- `trace_id` — Distributed trace identifier
- `request_id` — Request correlation
- `agent_id` — Originating agent
- `conversation_id` — Conversation context
- `user_id` — User context

Use these fields to connect application logs with downstream analytics or
monitoring pipelines.
