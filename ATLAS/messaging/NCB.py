"""
Neural Cognitive Bus (NCB)
====================================

Multi-channel, async communication bus for multi-agent systems.

Author: Jeremy Shows – Digital Hallucinations
Date: Dec 30, 2025
"""

from __future__ import annotations

import asyncio
import contextvars
import dataclasses
import heapq
import json
import logging
import os
import sqlite3
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    Union,
    cast,
)

# ---------------- Optional deps (degrade gracefully) ----------------

try:
    import yaml  
except Exception:
    yaml = None  

try:
    import msgpack as _msgpack  
except Exception:
    _msgpack = None  

try:
    import zlib as _zlib
except Exception:
    _zlib = None  

try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server  
except Exception:
    Counter = Gauge = Histogram = start_http_server = None  

try:
    import jwt    # PyJWT
except Exception:
    jwt = None  

try:
    import redis.asyncio as aioredis  
except Exception:
    aioredis = None  

try:
    from aiokafka import AIOKafkaProducer, AIOKafkaConsumer  
except Exception:
    AIOKafkaProducer = AIOKafkaConsumer = None  

try:
    import jsonschema  
except Exception:
    jsonschema = None  

# Optional torch nn.Module compatibility (NCB does not require torch)
try:
    import torch.nn as nn  
except Exception:
    nn = None  


###############################################################################
# Structured Logging (JSON) + trace_id context
###############################################################################

_trace_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("ncb_trace_id", default="")


def get_trace_id() -> str:
    return _trace_id_var.get() or ""


class TraceIdFilter(logging.Filter):
    """Injects trace_id into log records (from record.extra or contextvar)."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "trace_id") or not getattr(record, "trace_id"):
            setattr(record, "trace_id", get_trace_id())
        return True


class JSONFormatter(logging.Formatter):
    """Lightweight JSON formatter for structured logs."""

    def format(self, record: logging.LogRecord) -> str:
        base: Dict[str, Any] = {
            "ts": time.time(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }

        # Standard extras if present
        for k in ("event", "channel", "subscriber", "msg_id", "trace_id", "error_type"):
            if hasattr(record, k):
                v = getattr(record, k)
                if v is not None and v != "":
                    base[k] = v

        # Exception info
        if record.exc_info:
            base["exc"] = self.formatException(record.exc_info)

        # Add selected fields from record.__dict__ that are not built-ins
        # (kept conservative to avoid dumping huge objects)
        extras: Dict[str, Any] = {}
        skip = {
            "name",
            "msg",
            "args",
            "levelname",
            "levelno",
            "pathname",
            "filename",
            "module",
            "exc_info",
            "exc_text",
            "stack_info",
            "lineno",
            "funcName",
            "created",
            "msecs",
            "relativeCreated",
            "thread",
            "threadName",
            "processName",
            "process",
            "message",
        }
        for k, v in record.__dict__.items():
            if k in skip or k in base:
                continue
            # don't explode logs with massive payloads
            if isinstance(v, (str, int, float, bool)) or v is None:
                extras[k] = v
            elif isinstance(v, (list, dict)) and len(str(v)) < 2000:
                extras[k] = v
        if extras:
            base["extra"] = extras

        return json.dumps(base, ensure_ascii=False, separators=(",", ":"))


def setup_json_logging(level: int = logging.INFO, logger_name: str = "NCB") -> logging.Logger:
    """
    Create a JSON-logging logger with a TraceIdFilter.
    Safe to call multiple times (won't double-add handlers if already configured).
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    # Avoid duplicate StreamHandlers
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        h = logging.StreamHandler()
        h.setLevel(level)
        h.setFormatter(JSONFormatter())
        h.addFilter(TraceIdFilter())
        logger.addHandler(h)
        logger.propagate = False

    # Ensure all handlers have trace filter
    for h in logger.handlers:
        if not any(isinstance(f, TraceIdFilter) for f in getattr(h, "filters", [])):
            h.addFilter(TraceIdFilter())

    return logger


def setup_logging(level: int = logging.INFO, logger_name: str = "NCB") -> logging.Logger:
    """
    Convenience helper: create a logger named `logger_name` with standard formatting.
    (Non-JSON; mostly useful for local dev.)
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        handler.addFilter(TraceIdFilter())
        logger.addHandler(handler)
        logger.propagate = False

    return logger


###############################################################################
# Utility: Token Bucket (rate limiting)
###############################################################################

class TokenBucket:
    """
    Simple async-safe token bucket limiter.
    rate_per_sec: refill rate
    capacity: max tokens stored
    """

    __slots__ = ("rate", "capacity", "_tokens", "_last", "_lock")

    def __init__(self, rate_per_sec: float, capacity: float):
        if rate_per_sec <= 0 or capacity <= 0:
            raise ValueError("rate_per_sec and capacity must be > 0")
        self.rate = float(rate_per_sec)
        self.capacity = float(capacity)
        self._tokens = float(capacity)
        self._last = time.perf_counter()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: float = 1.0) -> None:
        tokens = float(tokens)
        if tokens <= 0:
            return
        async with self._lock:
            while True:
                now = time.perf_counter()
                elapsed = now - self._last
                if elapsed > 0:
                    self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
                    self._last = now
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                needed = tokens - self._tokens
                sleep_for = max(0.001, needed / self.rate)
                await asyncio.sleep(min(sleep_for, 0.25))


###############################################################################
# Message Envelope
###############################################################################

@dataclass(frozen=True)
class Message:
    id: str
    channel: str
    ts: float
    priority: int
    payload: Any
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def trace_id(self) -> str:
        return str(self.meta.get("trace_id", ""))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "channel": self.channel,
            "ts": self.ts,
            "priority": self.priority,
            "payload": self.payload,
            "meta": self.meta,
        }


###############################################################################
# Serialization / Compression
###############################################################################

class CodecError(RuntimeError):
    pass


def _msgpack_packb(obj: Any) -> bytes:
    if _msgpack is None:
        raise CodecError("msgpack not installed")
    mp = cast(Any, _msgpack)
    return cast(bytes, mp.packb(obj, use_bin_type=True))


def _msgpack_unpackb(data: bytes) -> Any:
    if _msgpack is None:
        raise CodecError("msgpack not installed")
    mp = cast(Any, _msgpack)
    return mp.unpackb(data, raw=False)


def _zlib_compress(data: bytes, level: int) -> bytes:
    if _zlib is None:
        raise CodecError("zlib not available")
    zl = cast(Any, _zlib)
    return cast(bytes, zl.compress(data, level=level))


def _zlib_decompress(data: bytes) -> bytes:
    if _zlib is None:
        raise CodecError("zlib not available")
    zl = cast(Any, _zlib)
    return cast(bytes, zl.decompress(data))


class Codec:
    """
    Supports:
      - serialization: msgpack (preferred) or json
      - compression: zlib (optional)
    """

    def __init__(self, serializer: str = "msgpack", compress: bool = False, compression_level: int = 3):
        self.serializer = serializer.lower()
        self.compress = bool(compress)
        self.compression_level = int(compression_level)

        if self.serializer == "msgpack" and _msgpack is None:
            self.serializer = "json"

        if self.compress and _zlib is None:
            self.compress = False

    def dumps(self, obj: Any) -> bytes:
        try:
            if self.serializer == "msgpack":
                raw = _msgpack_packb(obj)
            elif self.serializer == "json":
                raw = json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
            else:
                raise CodecError(f"Unknown serializer: {self.serializer}")
        except Exception as e:
            raise CodecError(f"Serialize failed: {e}") from e

        if self.compress:
            try:
                raw = _zlib_compress(raw, level=self.compression_level)
            except Exception as e:
                raise CodecError(f"Compress failed: {e}") from e

        return raw

    def loads(self, data: bytes) -> Any:
        raw = data
        if self.compress:
            try:
                raw = _zlib_decompress(raw)
            except Exception as e:
                raise CodecError(f"Decompress failed: {e}") from e

        try:
            if self.serializer == "msgpack":
                return _msgpack_unpackb(raw)
            if self.serializer == "json":
                return json.loads(raw.decode("utf-8"))
            raise CodecError(f"Unknown serializer: {self.serializer}")
        except Exception as e:
            raise CodecError(f"Deserialize failed: {e}") from e


###############################################################################
# Persistence (SQLite)
###############################################################################

class PersistenceError(RuntimeError):
    pass


class SQLitePersistence:
    """
    Lightweight SQLite persistence for critical channels.
    Stores raw bytes (serialized+compressed message dict).
    """

    def __init__(self, db_path: Union[str, Path], logger: logging.Logger):
        self.db_path = str(db_path)
        self.logger = logger
        self._init_lock = asyncio.Lock()
        self._initialized = False

    async def init(self) -> None:
        async with self._init_lock:
            if self._initialized:
                return

            def _init() -> None:
                os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
                con = sqlite3.connect(self.db_path)
                try:
                    cur = con.cursor()
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS ncb_messages (
                            channel TEXT NOT NULL,
                            seq INTEGER PRIMARY KEY AUTOINCREMENT,
                            msg_id TEXT NOT NULL,
                            ts REAL NOT NULL,
                            priority INTEGER NOT NULL,
                            data BLOB NOT NULL
                        );
                        """
                    )
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_ncb_channel_seq ON ncb_messages(channel, seq);")
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_ncb_channel_ts ON ncb_messages(channel, ts);")
                    con.commit()
                finally:
                    con.close()

            await asyncio.to_thread(_init)
            self._initialized = True
            self.logger.info("NCB persistence initialized", extra={"event": "persistence_init"})

    async def append(self, channel: str, msg: Message, data_blob: bytes) -> int:
        await self.init()

        def _append() -> int:
            con = sqlite3.connect(self.db_path)
            try:
                cur = con.cursor()
                cur.execute(
                    "INSERT INTO ncb_messages(channel, msg_id, ts, priority, data) VALUES(?,?,?,?,?)",
                    (channel, msg.id, msg.ts, msg.priority, data_blob),
                )
                con.commit()
                return int(cur.lastrowid or 0)
            finally:
                con.close()

        try:
            return await asyncio.to_thread(_append)
        except Exception as e:
            raise PersistenceError(f"SQLite append failed: {e}") from e

    async def replay(self, channel: str, after_seq: int, limit: int = 1000) -> List[Tuple[int, bytes]]:
        await self.init()

        def _replay() -> List[Tuple[int, bytes]]:
            con = sqlite3.connect(self.db_path)
            try:
                cur = con.cursor()
                cur.execute(
                    "SELECT seq, data FROM ncb_messages WHERE channel=? AND seq>? ORDER BY seq ASC LIMIT ?",
                    (channel, after_seq, limit),
                )
                rows = cur.fetchall()
                return [(int(seq), cast(bytes, blob)) for (seq, blob) in rows]
            finally:
                con.close()

        try:
            return await asyncio.to_thread(_replay)
        except Exception as e:
            raise PersistenceError(f"SQLite replay failed: {e}") from e

    async def prune_old_messages(self, channel: str, max_age_sec: float) -> int:
        """
        Delete messages older than now - max_age_sec for a given channel.
        Returns number of rows deleted.
        """
        await self.init()
        cutoff = time.time() - float(max_age_sec)

        def _prune() -> int:
            con = sqlite3.connect(self.db_path)
            try:
                cur = con.cursor()
                cur.execute("DELETE FROM ncb_messages WHERE channel=? AND ts < ?", (channel, cutoff))
                deleted = cur.rowcount if cur.rowcount is not None else 0
                con.commit()
                return int(deleted)
            finally:
                con.close()

        try:
            deleted = await asyncio.to_thread(_prune)
            self.logger.info(
                "Pruned old persisted messages",
                extra={"event": "persistence_prune", "channel": channel},
            )
            return deleted
        except Exception as e:
            raise PersistenceError(f"SQLite prune failed: {e}") from e


###############################################################################
# Security (API keys + JWT) + RBAC
###############################################################################

@dataclass
class SecurityConfig:
    enabled: bool = False
    api_keys: Set[str] = field(default_factory=set)  # simple shared-secret keys
    jwt_secret: Optional[str] = None  # HS256 secret (optional)
    jwt_audience: Optional[str] = None
    jwt_issuer: Optional[str] = None
    jwt_algorithms: List[str] = field(default_factory=lambda: ["HS256"])
    # channel -> roles allowed
    channel_roles: Dict[str, Set[str]] = field(default_factory=dict)

    def verify(self, auth: Optional[Dict[str, Any]], channel: str) -> None:
        if not self.enabled:
            return

        if not auth:
            raise PermissionError("Missing auth")

        # 1) API key (fast path)
        api_key = auth.get("api_key")
        if api_key:
            if api_key not in self.api_keys:
                raise PermissionError("Invalid API key")
            roles = set(auth.get("roles") or [])
            self._enforce_roles(channel, roles)
            return

        # 2) JWT (optional)
        token = auth.get("jwt")
        if token:
            if jwt is None:
                raise PermissionError("JWT support not installed (PyJWT missing)")
            if not self.jwt_secret:
                raise PermissionError("JWT secret not configured")
            try:
                claims = jwt.decode(
                    token,
                    self.jwt_secret,
                    algorithms=self.jwt_algorithms,
                    audience=self.jwt_audience,
                    issuer=self.jwt_issuer,
                    options={"require": ["exp"]},
                )
            except Exception as e:
                raise PermissionError(f"Invalid JWT: {e}") from e
            roles = set((claims.get("roles") or []))
            self._enforce_roles(channel, roles)
            return

        raise PermissionError("No supported auth method provided")

    def _enforce_roles(self, channel: str, roles: Set[str]) -> None:
        allowed = self.channel_roles.get(channel)
        if not allowed:
            return  # no RBAC for this channel
        if roles.isdisjoint(allowed):
            raise PermissionError(f"Insufficient role for channel '{channel}'")


###############################################################################
# Hooks / Plugins
###############################################################################

HookFn = Callable[[Message], Awaitable[Message]]
SinkFn = Callable[[Message], Awaitable[None]]
ValidatorFn = Callable[[Message], Awaitable[None]]


@dataclass
class Plugin:
    name: str
    # Transformations
    pre_publish: Optional[HookFn] = None
    post_publish: Optional[SinkFn] = None
    pre_dispatch: Optional[HookFn] = None
    post_dispatch: Optional[SinkFn] = None
    # Validation
    validate: Optional[ValidatorFn] = None


###############################################################################
# Channel / Subscriber Config
###############################################################################

@dataclass
class ChannelConfig:
    name: str
    max_queue_size: int = 1000
    drop_policy: str = "drop_new"  # drop_new | drop_old
    default_priority: int = 100

    # rate limiting
    rate_limit_per_sec: Optional[float] = None
    rate_limit_burst: Optional[float] = None

    # persistence
    persistent: bool = False

    # serialization
    serializer: str = "msgpack"  # msgpack | json
    compress: bool = False
    compression_level: int = 3

    # distributed bridging
    redis_in: Optional[str] = None  # redis pubsub channel for inbound
    redis_out: Optional[str] = None  # redis pubsub channel for outbound
    kafka_in: Optional[str] = None  # kafka topic for inbound
    kafka_out: Optional[str] = None  # kafka topic for outbound

    # dead-letter channel (optional)
    dead_letter_channel: Optional[str] = None

    # bounded fan-out
    fanout_max_concurrency: int = 500  # cap concurrent subscriber callbacks per channel


@dataclass
class SubscriberConfig:
    module_name: str
    callback_fn: Callable[[Message], Awaitable[None]]
    filter_fn: Optional[Callable[[Message], bool]] = None

    # per-subscriber throttling
    rate_limit_per_sec: Optional[float] = None
    rate_limit_burst: Optional[float] = None

    # replay
    replay_on_register: bool = False
    replay_after_seq: int = 0  # last seen persistence seq (if persistence enabled)


###############################################################################
# Metrics
###############################################################################

@dataclass
class LocalMetrics:
    published: int = 0
    dropped: int = 0
    dispatched: int = 0
    errors: int = 0
    publish_latency_ms_sum: float = 0.0
    dispatch_latency_ms_sum: float = 0.0


class MetricsRegistry:
    """
    Built-in counters always work.
    If prometheus_client is present and enabled, also exports standard metrics.
    """

    def __init__(
        self,
        enable_prometheus: bool = False,
        prometheus_port: int = 8000,
        logger: Optional[logging.Logger] = None,
    ):
        self.logger = logger or logging.getLogger("NCB.metrics")
        self.local: Dict[str, LocalMetrics] = {}
        self.enable_prometheus = bool(
            enable_prometheus
            and Counter is not None
            and Gauge is not None
            and Histogram is not None
            and start_http_server is not None
        )

        self._p_published: Any = None
        self._p_dropped: Any = None
        self._p_dispatched: Any = None
        self._p_errors: Any = None
        self._p_queue: Any = None
        self._p_pub_lat: Any = None
        self._p_disp_lat: Any = None

        if self.enable_prometheus and Counter and Gauge and Histogram and start_http_server:
            self._p_published = Counter("ncb_published_total", "Messages published", ["channel"])
            self._p_dropped = Counter("ncb_dropped_total", "Messages dropped", ["channel"])
            self._p_dispatched = Counter("ncb_dispatched_total", "Messages dispatched", ["channel"])
            self._p_errors = Counter("ncb_errors_total", "Errors", ["channel"])
            self._p_queue = Gauge("ncb_queue_size", "Channel queue size", ["channel"])
            self._p_pub_lat = Histogram("ncb_publish_latency_ms", "Publish latency (ms)", ["channel"])
            self._p_disp_lat = Histogram("ncb_dispatch_latency_ms", "Dispatch latency (ms)", ["channel"])
            try:
                start_http_server(prometheus_port)  
                self.logger.info("Prometheus exporter started", extra={"event": "prometheus_start"})
            except Exception as e:
                self.logger.warning(
                    "Failed to start Prometheus exporter: %s",
                    e,
                    extra={"event": "prometheus_fail"},
                )

    def _lm(self, channel: str) -> LocalMetrics:
        if channel not in self.local:
            self.local[channel] = LocalMetrics()
        return self.local[channel]

    def inc_published(self, channel: str, n: int = 1) -> None:
        self._lm(channel).published += n
        if self._p_published:
            self._p_published.labels(channel=channel).inc(n)

    def inc_dropped(self, channel: str, n: int = 1) -> None:
        self._lm(channel).dropped += n
        if self._p_dropped:
            self._p_dropped.labels(channel=channel).inc(n)

    def inc_dispatched(self, channel: str, n: int = 1) -> None:
        self._lm(channel).dispatched += n
        if self._p_dispatched:
            self._p_dispatched.labels(channel=channel).inc(n)

    def inc_errors(self, channel: str, n: int = 1) -> None:
        self._lm(channel).errors += n
        if self._p_errors:
            self._p_errors.labels(channel=channel).inc(n)

    def observe_publish_latency_ms(self, channel: str, ms: float) -> None:
        self._lm(channel).publish_latency_ms_sum += ms
        if self._p_pub_lat:
            self._p_pub_lat.labels(channel=channel).observe(ms)

    def observe_dispatch_latency_ms(self, channel: str, ms: float) -> None:
        self._lm(channel).dispatch_latency_ms_sum += ms
        if self._p_disp_lat:
            self._p_disp_lat.labels(channel=channel).observe(ms)

    def set_queue_size(self, channel: str, size: int) -> None:
        if self._p_queue:
            self._p_queue.labels(channel=channel).set(size)


###############################################################################
# Config Schema Validation (optional)
###############################################################################

NCB_CONFIG_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "channels": {
            "type": "object",
            "additionalProperties": {
                "type": "object",
                "properties": {
                    "max_queue_size": {"type": "integer", "minimum": 1},
                    "drop_policy": {"type": "string", "enum": ["drop_new", "drop_old"]},
                    "default_priority": {"type": "integer"},
                    "rate_limit_per_sec": {"type": ["number", "null"], "exclusiveMinimum": 0},
                    "rate_limit_burst": {"type": ["number", "null"], "exclusiveMinimum": 0},
                    "persistent": {"type": "boolean"},
                    "serializer": {"type": "string", "enum": ["msgpack", "json"]},
                    "compress": {"type": "boolean"},
                    "compression_level": {"type": "integer", "minimum": 1, "maximum": 9},
                    "redis_in": {"type": ["string", "null"]},
                    "redis_out": {"type": ["string", "null"]},
                    "kafka_in": {"type": ["string", "null"]},
                    "kafka_out": {"type": ["string", "null"]},
                    "dead_letter_channel": {"type": ["string", "null"]},
                    "fanout_max_concurrency": {"type": "integer", "minimum": 1},
                },
                "additionalProperties": True,
            },
        },
        "security": {"type": "object"},
    },
    "additionalProperties": True,
}


###############################################################################
# Priority Queue (async) with drop policies + queue size hook
###############################################################################

class AsyncPriorityQueue:
    """
    Heap-based priority queue with FIFO tie-break via sequence number.

    Items are tuples: (priority, seq, message)
      lower priority value == higher urgency (like UNIX nice)

    Supports a queue-size hook `on_qsize(size:int)` for real-time metrics.
    """

    def __init__(
        self,
        maxsize: int,
        drop_policy: str,
        logger: logging.Logger,
        *,
        on_qsize: Optional[Callable[[int], None]] = None,
    ):
        self.maxsize = int(maxsize)
        self.drop_policy = drop_policy
        self.logger = logger
        self._heap: List[Tuple[int, int, Message]] = []
        self._seq = 0
        self._cv = asyncio.Condition()
        self._on_qsize = on_qsize

    def qsize(self) -> int:
        return len(self._heap)

    async def put(self, priority: int, msg: Message) -> bool:
        async with self._cv:
            if self.maxsize > 0 and len(self._heap) >= self.maxsize:
                if self.drop_policy == "drop_old":
                    # drop the worst (highest priority number; FIFO tie-break)
                    worst_idx = max(range(len(self._heap)), key=lambda i: (self._heap[i][0], self._heap[i][1]))
                    dropped = self._heap[worst_idx][2]
                    self._heap[worst_idx] = self._heap[-1]
                    self._heap.pop()
                    if worst_idx < len(self._heap):
                        import heapq

                        heapq.heapify(self._heap)
                    self.logger.warning(
                        "Queue full; drop_old dropped message",
                        extra={
                            "event": "queue_drop_old",
                            "channel": msg.channel,
                            "msg_id": dropped.id,
                            "trace_id": str(dropped.meta.get("trace_id", "")),
                        },
                    )
                else:
                    return False

            self._seq += 1
            item = (int(priority), self._seq, msg)
            import heapq

            heapq.heappush(self._heap, item)
            if self._on_qsize:
                self._on_qsize(len(self._heap))
            self._cv.notify(1)
            return True

    async def get(self) -> Message:
        async with self._cv:
            while not self._heap:
                await self._cv.wait()
            import heapq

            _, _, msg = heapq.heappop(self._heap)
            if self._on_qsize:
                self._on_qsize(len(self._heap))
            return msg

    async def drain(self) -> None:
        async with self._cv:
            self._heap.clear()
            if self._on_qsize:
                self._on_qsize(0)
            self._cv.notify_all()


###############################################################################
# Neural Cognitive Bus
###############################################################################

# Dynamic base class: use nn.Module when torch is available for neural integration
_BaseClass: type = nn.Module if nn is not None else object


class NeuralCognitiveBus(_BaseClass):  # type: ignore[misc]
    """
    Neural Cognitive Bus (NCB)
    -------------------------
    Multi-channel async messaging bus with:
      - priority queue per channel
      - optional persistence (SQLite)
      - optional distributed bridging (Redis, Kafka)
      - security + RBAC
      - plugin hooks
      - metrics
      - rate limiting
      - dynamic config + hot reload
      - structured logging (JSON + trace IDs per message)
      - dead-letter channel for subscriber failures
      - bounded fan-out to cap concurrent subscriber callbacks
    """

    def __init__(
        self,
        *,
        default_max_queue_size: int = 1000,
        persistence_path: Optional[Union[str, Path]] = None,
        enable_prometheus: bool = False,
        prometheus_port: int = 8000,
        logger: Optional[logging.Logger] = None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        channel_validator: Optional[Callable[[str], bool]] = None,
        security: Optional[SecurityConfig] = None,
        hot_reload_config_path: Optional[Union[str, Path]] = None,
        hot_reload_interval_sec: float = 2.0,
        # Subscriber GC
        subscriber_gc_interval_sec: float = 60.0,
        subscriber_ttl_sec: float = 15 * 60.0,
    ):
        super().__init__()

        self.logger = logger or setup_json_logging(logger_name="NCB")
        self._loop = loop
        self._channel_validator = channel_validator
        self._kafka_topic_prefix = "ncb"  # default
        self.default_max_queue_size = int(default_max_queue_size)

        self.running: bool = False
        self._tasks: Set[asyncio.Task] = set()
        self._redis_tasks: Set[asyncio.Task] = set()
        self._kafka_tasks: Set[asyncio.Task] = set()

        self._channel_cfg: Dict[str, ChannelConfig] = {}
        self._queues: Dict[str, AsyncPriorityQueue] = {}
        self._subs: Dict[str, List[SubscriberConfig]] = {}

        self._channel_limiters: Dict[str, TokenBucket] = {}
        self._subscriber_limiters: Dict[Tuple[str, str], TokenBucket] = {}
        self._fanout_semaphores: Dict[str, asyncio.Semaphore] = {}

        self._codec: Dict[str, Codec] = {}

        self._metrics = MetricsRegistry(enable_prometheus=enable_prometheus, prometheus_port=prometheus_port, logger=self.logger)
        self._security = security or SecurityConfig(enabled=False)

        self._plugins: List[Plugin] = []

        self._persistence: Optional[SQLitePersistence] = (
            SQLitePersistence(persistence_path, self.logger) if persistence_path else None
        )

        # Replay/seq bookkeeping (bounded by GC)
        self._last_seq_seen: Dict[Tuple[str, str], int] = {}
        self._subscriber_last_seen: Dict[Tuple[str, str], float] = {}
        self._subscriber_gc_interval = float(subscriber_gc_interval_sec)
        self._subscriber_ttl = float(subscriber_ttl_sec)

        # Redis client
        self._redis_client: Any = None

        # Kafka clients
        self._kafka_producer: Any = None
        self._kafka_consumer: Any = None

        # Hot reload
        self._hot_cfg_path = Path(hot_reload_config_path) if hot_reload_config_path else None
        self._hot_interval = float(hot_reload_interval_sec)
        self._hot_mtime: Optional[float] = None

        # Lifecycle lock
        self._life_lock = asyncio.Lock()

    # ----------------------- Plugins -----------------------

    def add_plugin(self, plugin: Plugin) -> None:
        self._plugins.append(plugin)
        self.logger.info("Plugin registered", extra={"event": "plugin_add", "subscriber": plugin.name})

    # ----------------------- Config -----------------------

    def create_channel(self, channel_name: str, cfg: Optional[ChannelConfig] = None) -> None:
        if channel_name in self._queues:
            self.logger.warning("Channel already exists", extra={"event": "channel_exists", "channel": channel_name})
            return

        cfg = cfg or ChannelConfig(name=channel_name, max_queue_size=self.default_max_queue_size)
        cfg.name = channel_name

        self._channel_cfg[channel_name] = cfg
        self._codec[channel_name] = Codec(serializer=cfg.serializer, compress=cfg.compress, compression_level=cfg.compression_level)

        self._queues[channel_name] = AsyncPriorityQueue(
            maxsize=cfg.max_queue_size,
            drop_policy=cfg.drop_policy,
            logger=self.logger,
            on_qsize=lambda sz, ch=channel_name: self._metrics.set_queue_size(ch, sz),
        )
        self._subs[channel_name] = []

        if cfg.rate_limit_per_sec and cfg.rate_limit_burst:
            self._channel_limiters[channel_name] = TokenBucket(float(cfg.rate_limit_per_sec), float(cfg.rate_limit_burst))

        # Bounded fan-out semaphore per channel
        self._fanout_semaphores[channel_name] = asyncio.Semaphore(max(1, int(cfg.fanout_max_concurrency)))

        # DLQ channel creation (if requested)
        if cfg.dead_letter_channel:
            if cfg.dead_letter_channel == channel_name:
                self.logger.warning(
                    "dead_letter_channel equals channel; ignoring DLQ for channel",
                    extra={"event": "dlq_invalid", "channel": channel_name},
                )
            elif cfg.dead_letter_channel not in self._queues:
                self.create_channel(
                    cfg.dead_letter_channel,
                    ChannelConfig(name=cfg.dead_letter_channel, max_queue_size=self.default_max_queue_size),
                )

        self.logger.info("Created channel", extra={"event": "channel_create", "channel": channel_name})

    def remove_channel(self, channel_name: str) -> None:
        self._channel_cfg.pop(channel_name, None)
        self._queues.pop(channel_name, None)
        self._subs.pop(channel_name, None)
        self._codec.pop(channel_name, None)
        self._channel_limiters.pop(channel_name, None)
        self._fanout_semaphores.pop(channel_name, None)
        self.logger.info("Removed channel", extra={"event": "channel_remove", "channel": channel_name})

    def load_config_file(self, path: Union[str, Path]) -> Dict[str, Any]:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(str(p))

        text = p.read_text(encoding="utf-8")
        if p.suffix.lower() in (".yaml", ".yml"):
            if yaml is None:
                raise RuntimeError("PyYAML not installed; cannot load YAML config")
            return cast(Dict[str, Any], yaml.safe_load(text) or {})
        return cast(Dict[str, Any], json.loads(text))

    def apply_config(self, cfg: Mapping[str, Any]) -> None:
        """
        Expected structure:
        {
          "channels": {
             "agent_commands": {...ChannelConfig fields...},
             ...
          },
          "security": {...SecurityConfig...}
        }
        """
        if jsonschema is not None:
            jsonschema.validate(instance=cfg, schema=NCB_CONFIG_SCHEMA)

        channels = cfg.get("channels") or {}
        if not isinstance(channels, Mapping):
            raise ValueError("config.channels must be a mapping")

        # Update security (safe merge)
        sec = cfg.get("security")
        if isinstance(sec, Mapping):
            self._security.enabled = bool(sec.get("enabled", self._security.enabled))
            api_keys = sec.get("api_keys")
            if isinstance(api_keys, (list, set, tuple)):
                self._security.api_keys = set(map(str, api_keys))
            self._security.jwt_secret = cast(Optional[str], sec.get("jwt_secret", self._security.jwt_secret))
            self._security.jwt_audience = cast(Optional[str], sec.get("jwt_audience", self._security.jwt_audience))
            self._security.jwt_issuer = cast(Optional[str], sec.get("jwt_issuer", self._security.jwt_issuer))
            algos = sec.get("jwt_algorithms")
            if isinstance(algos, (list, tuple)):
                self._security.jwt_algorithms = [str(a) for a in algos]
            roles = sec.get("channel_roles")
            if isinstance(roles, Mapping):
                self._security.channel_roles = {str(k): set(map(str, v or [])) for k, v in roles.items()}

        # Create / update channels
        for name, raw in channels.items():
            cname = str(name)
            m = cast(Mapping[str, Any], raw or {})
            if cname not in self._queues:
                self.create_channel(cname, self._channelcfg_from_mapping(cname, m))
            else:
                self._hot_update_channel(cname, m)

    def _channelcfg_from_mapping(self, name: str, m: Mapping[str, Any]) -> ChannelConfig:
        return ChannelConfig(
            name=name,
            max_queue_size=int(m.get("max_queue_size", self.default_max_queue_size)),
            drop_policy=str(m.get("drop_policy", "drop_new")),
            default_priority=int(m.get("default_priority", 100)),
            rate_limit_per_sec=cast(Optional[float], m.get("rate_limit_per_sec")),
            rate_limit_burst=cast(Optional[float], m.get("rate_limit_burst")),
            persistent=bool(m.get("persistent", False)),
            serializer=str(m.get("serializer", "msgpack")),
            compress=bool(m.get("compress", False)),
            compression_level=int(m.get("compression_level", 3)),
            redis_in=cast(Optional[str], m.get("redis_in")),
            redis_out=cast(Optional[str], m.get("redis_out")),
            kafka_in=cast(Optional[str], m.get("kafka_in")),
            kafka_out=cast(Optional[str], m.get("kafka_out")),
            dead_letter_channel=cast(Optional[str], m.get("dead_letter_channel")),
            fanout_max_concurrency=int(m.get("fanout_max_concurrency", 500)),
        )

    def _hot_update_channel(self, name: str, raw: Mapping[str, Any]) -> None:
        cfg = self._channel_cfg[name]

        # Update limiter settings
        rl = raw.get("rate_limit_per_sec", cfg.rate_limit_per_sec)
        burst = raw.get("rate_limit_burst", cfg.rate_limit_burst)
        cfg.rate_limit_per_sec = cast(Optional[float], rl)
        cfg.rate_limit_burst = cast(Optional[float], burst)
        if cfg.rate_limit_per_sec and cfg.rate_limit_burst:
            self._channel_limiters[name] = TokenBucket(float(cfg.rate_limit_per_sec), float(cfg.rate_limit_burst))
        else:
            self._channel_limiters.pop(name, None)

        # Persistence toggle
        cfg.persistent = bool(raw.get("persistent", cfg.persistent))

        # Codec update
        cfg.serializer = str(raw.get("serializer", cfg.serializer))
        cfg.compress = bool(raw.get("compress", cfg.compress))
        cfg.compression_level = int(raw.get("compression_level", cfg.compression_level))
        self._codec[name] = Codec(cfg.serializer, cfg.compress, cfg.compression_level)

        # Drop policy update
        cfg.drop_policy = str(raw.get("drop_policy", cfg.drop_policy))
        self._queues[name].drop_policy = cfg.drop_policy

        # DLQ update
        cfg.dead_letter_channel = cast(Optional[str], raw.get("dead_letter_channel", cfg.dead_letter_channel))
        if cfg.dead_letter_channel and cfg.dead_letter_channel not in self._queues and cfg.dead_letter_channel != name:
            self.create_channel(cfg.dead_letter_channel, ChannelConfig(name=cfg.dead_letter_channel, max_queue_size=self.default_max_queue_size))

        # Fan-out semaphore update
        cfg.fanout_max_concurrency = int(raw.get("fanout_max_concurrency", cfg.fanout_max_concurrency))
        self._fanout_semaphores[name] = asyncio.Semaphore(max(1, cfg.fanout_max_concurrency))

        self.logger.info("Hot-updated channel", extra={"event": "channel_hot_update", "channel": name})

    # ----------------------- Subscribers -----------------------

    async def register_subscriber(
        self,
        channel_name: str,
        module_name: str,
        callback_fn: Callable[[Message], Awaitable[None]],
        filter_fn: Optional[Callable[[Message], bool]] = None,
        *,
        rate_limit_per_sec: Optional[float] = None,
        rate_limit_burst: Optional[float] = None,
        replay_on_register: bool = False,
        replay_after_seq: int = 0,
        auth: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self._channel_validator and not self._channel_validator(channel_name):
            raise ValueError(f"Invalid channel name: {channel_name}")

        if channel_name not in self._queues:
            raise ValueError(f"Channel '{channel_name}' does not exist.")

        self._security.verify(auth, channel_name)

        sub = SubscriberConfig(
            module_name=module_name,
            callback_fn=callback_fn,
            filter_fn=filter_fn,
            rate_limit_per_sec=rate_limit_per_sec,
            rate_limit_burst=rate_limit_burst,
            replay_on_register=replay_on_register,
            replay_after_seq=replay_after_seq,
        )
        self._subs[channel_name].append(sub)

        if rate_limit_per_sec and rate_limit_burst:
            self._subscriber_limiters[(channel_name, module_name)] = TokenBucket(float(rate_limit_per_sec), float(rate_limit_burst))

        self._subscriber_last_seen[(channel_name, module_name)] = time.time()

        # Replay support
        if replay_on_register and self._persistence and self._channel_cfg[channel_name].persistent:
            last = int(replay_after_seq)
            self._last_seq_seen[(channel_name, module_name)] = last
            await self._replay_to_subscriber(channel_name, sub, after_seq=last)

        self.logger.info(
            "Registered subscriber",
            extra={"event": "subscriber_register", "channel": channel_name, "subscriber": module_name},
        )

    async def unregister_subscriber(self, channel_name: str, module_name: str) -> None:
        if channel_name not in self._subs:
            return
        before = len(self._subs[channel_name])
        self._subs[channel_name] = [s for s in self._subs[channel_name] if s.module_name != module_name]
        self._subscriber_limiters.pop((channel_name, module_name), None)
        self._last_seq_seen.pop((channel_name, module_name), None)
        self._subscriber_last_seen.pop((channel_name, module_name), None)
        after = len(self._subs[channel_name])
        if after != before:
            self.logger.info(
                "Unregistered subscriber",
                extra={"event": "subscriber_unregister", "channel": channel_name, "subscriber": module_name},
            )

    # ----------------------- Lifecycle -----------------------

    async def start(
        self,
        *,
        redis_url: Optional[str] = None,
        kafka_bootstrap_servers: Optional[str] = None,
        kafka_topic_prefix: str = "ncb",
        kafka_client_id: str = "ncb-bridge",
        kafka_acks: str = "all",
        kafka_max_in_flight: int = 5,
        **kwargs: Any,
    ) -> None:
        async with self._life_lock:
            if self.running:
                self.logger.warning("NCB already running", extra={"event": "start_noop"})
                return

            self.running = True

            # Per-channel processing tasks
            for channel_name in list(self._queues.keys()):
                task = asyncio.create_task(self._process_channel(channel_name), name=f"NCB.process.{channel_name}")
                self._tasks.add(task)
                task.add_done_callback(self._tasks.discard)

            # Subscriber GC
            gc_task = asyncio.create_task(self._subscriber_gc_loop(), name="NCB.subscriber_gc")
            self._tasks.add(gc_task)
            gc_task.add_done_callback(self._tasks.discard)

            # Hot reload
            if self._hot_cfg_path:
                t = asyncio.create_task(self._hot_reload_loop(), name="NCB.hot_reload")
                self._tasks.add(t)
                t.add_done_callback(self._tasks.discard)

            # Redis bridge tasks
            if redis_url and aioredis is not None:
                await self._start_redis(redis_url)
            elif redis_url and aioredis is None:
                self.logger.warning("Redis requested but redis.asyncio not installed", extra={"event": "redis_missing"})

            # Kafka bridge tasks
            if kafka_bootstrap_servers and AIOKafkaProducer is not None:
                await self._start_kafka(kafka_bootstrap_servers, kafka_topic_prefix, kafka_client_id, kafka_acks, kafka_max_in_flight)
            elif kafka_bootstrap_servers and AIOKafkaProducer is None:
                self.logger.warning("Kafka requested but aiokafka not installed", extra={"event": "kafka_missing"})

            self.logger.info("NCB started", extra={"event": "start"})

    async def stop(self) -> None:
        async with self._life_lock:
            if not self.running:
                self.logger.warning("NCB not running", extra={"event": "stop_noop"})
                return
            self.running = False

            # Cancel redis tasks
            for t in list(self._redis_tasks):
                t.cancel()
            if self._redis_tasks:
                await asyncio.gather(*self._redis_tasks, return_exceptions=True)
            self._redis_tasks.clear()
            if self._redis_client is not None:
                try:
                    await self._redis_client.close()
                except Exception:
                    pass
                self._redis_client = None

            # Cancel kafka tasks
            for t in list(self._kafka_tasks):
                t.cancel()
            if self._kafka_tasks:
                await asyncio.gather(*self._kafka_tasks, return_exceptions=True)
            self._kafka_tasks.clear()
            if self._kafka_producer is not None:
                try:
                    await self._kafka_producer.stop()
                except Exception:
                    pass
                self._kafka_producer = None
            if self._kafka_consumer is not None:
                try:
                    await self._kafka_consumer.stop()
                except Exception:
                    pass
                self._kafka_consumer = None

            # Cancel main tasks
            for task in list(self._tasks):
                task.cancel()
            if self._tasks:
                await asyncio.gather(*self._tasks, return_exceptions=True)
            self._tasks.clear()

            # Drain queues
            for q in self._queues.values():
                try:
                    await q.drain()
                except Exception:
                    pass

            self.logger.info("NCB stopped", extra={"event": "stop"})

    # ----------------------- Publish -----------------------

    async def publish(
        self,
        channel_name: str,
        payload: Any,
        *,
        priority: Optional[int] = None,
        meta: Optional[Dict[str, Any]] = None,
        auth: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Publish a message into a channel.
        Returns message id if enqueued, else None if dropped.
        """
        if self._channel_validator and not self._channel_validator(channel_name):
            raise ValueError(f"Invalid channel name: {channel_name}")

        if channel_name not in self._queues:
            self.logger.error("Channel does not exist", extra={"event": "publish_no_channel", "channel": channel_name})
            return None

        self._security.verify(auth, channel_name)

        cfg = self._channel_cfg[channel_name]
        pr = int(priority if priority is not None else cfg.default_priority)

        m = dict(meta or {})
        m.setdefault("trace_id", str(uuid.uuid4()))
        m.setdefault("published_by", (auth or {}).get("sub") or (auth or {}).get("api_key") or "unknown")

        msg = Message(
            id=str(uuid.uuid4()),
            channel=channel_name,
            ts=time.time(),
            priority=pr,
            payload=payload,
            meta=m,
        )

        # Ensure logs inside this publish path carry this message trace id
        token = _trace_id_var.set(msg.trace_id)
        try:
            # Plugins: validate + pre_publish transforms
            try:
                msg = await self._run_validate_and_transform(msg, phase="pre_publish")
            except Exception as e:
                self._metrics.inc_errors(channel_name)
                self.logger.error(
                    "pre_publish failed",
                    extra={
                        "event": "pre_publish_fail",
                        "channel": channel_name,
                        "msg_id": msg.id,
                        "trace_id": msg.trace_id,
                        "error_type": type(e).__name__,
                    },
                    exc_info=True,
                )
                return None

            limiter = self._channel_limiters.get(channel_name)
            if limiter:
                await limiter.acquire(1.0)

            t0 = time.perf_counter()
            ok = await self._queues[channel_name].put(pr, msg)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            self._metrics.observe_publish_latency_ms(channel_name, dt_ms)

            if not ok:
                self._metrics.inc_dropped(channel_name)
                self.logger.warning(
                    "Queue full; dropped publish",
                    extra={"event": "publish_drop", "channel": channel_name, "msg_id": msg.id, "trace_id": msg.trace_id},
                )
                return None

            self._metrics.inc_published(channel_name)

            self.logger.debug("Published", extra={"event": "publish", "channel": channel_name, "msg_id": msg.id, "trace_id": msg.trace_id})

            # Persistence (fire-and-forget append)
            if self._persistence and cfg.persistent:
                asyncio.create_task(self._persist_message(msg), name=f"NCB.persist.{channel_name}")

            # Plugins: post_publish sinks
            asyncio.create_task(self._run_sinks(msg, phase="post_publish"), name=f"NCB.post_publish.{channel_name}")

            # Redis outbound bridge
            if self._redis_client and cfg.redis_out:
                asyncio.create_task(self._redis_publish(cfg.redis_out, msg), name=f"NCB.redis_out.{channel_name}")

            # Kafka outbound bridge
            if self._kafka_producer and cfg.kafka_out:
                topic = f"{self._kafka_topic_prefix}.{cfg.kafka_out}"
                asyncio.create_task(self._kafka_publish(topic, msg), name=f"NCB.kafka_out.{channel_name}")

            return msg.id
        finally:
            _trace_id_var.reset(token)

    # ----------------------- Internal loops -----------------------

    async def _process_channel(self, channel_name: str) -> None:
        q = self._queues[channel_name]
        while self.running:
            try:
                msg = await q.get()

                token = _trace_id_var.set(msg.trace_id)
                try:
                    # Plugins: pre_dispatch transforms
                    try:
                        msg = await self._run_validate_and_transform(msg, phase="pre_dispatch")
                    except Exception as e:
                        self._metrics.inc_errors(channel_name)
                        self.logger.error(
                            "pre_dispatch failed",
                            extra={
                                "event": "pre_dispatch_fail",
                                "channel": channel_name,
                                "msg_id": msg.id,
                                "trace_id": msg.trace_id,
                                "error_type": type(e).__name__,
                            },
                            exc_info=True,
                        )
                        continue

                    subs = list(self._subs.get(channel_name, []))
                    if not subs:
                        continue

                    t0 = time.perf_counter()
                    await self._dispatch_to_subscribers(channel_name, msg, subs)
                    dt_ms = (time.perf_counter() - t0) * 1000.0
                    self._metrics.observe_dispatch_latency_ms(channel_name, dt_ms)

                    asyncio.create_task(self._run_sinks(msg, phase="post_dispatch"), name=f"NCB.post_dispatch.{channel_name}")
                finally:
                    _trace_id_var.reset(token)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._metrics.inc_errors(channel_name)
                self.logger.error(
                    "Channel processing error",
                    extra={"event": "channel_loop_error", "channel": channel_name, "error_type": type(e).__name__},
                    exc_info=True,
                )

    async def _dispatch_to_subscribers(self, channel_name: str, msg: Message, subs: List[SubscriberConfig]) -> None:
        """
        Bounded fan-out: uses per-channel semaphore to cap concurrent subscriber callbacks.
        Failures publish to DLQ if configured.
        """
        sem = self._fanout_semaphores.get(channel_name) or asyncio.Semaphore(500)
        cfg = self._channel_cfg[channel_name]
        dlq = cfg.dead_letter_channel

        async def _call_sub(sub: SubscriberConfig) -> None:
            async with sem:
                key = (channel_name, sub.module_name)
                limiter = self._subscriber_limiters.get(key)
                if limiter:
                    await limiter.acquire(1.0)

                if sub.filter_fn is not None:
                    try:
                        if not sub.filter_fn(msg):
                            return
                    except Exception as e:
                        self._metrics.inc_errors(channel_name)
                        self.logger.error(
                            "Filter error",
                            extra={
                                "event": "filter_error",
                                "channel": channel_name,
                                "subscriber": sub.module_name,
                                "msg_id": msg.id,
                                "trace_id": msg.trace_id,
                                "error_type": type(e).__name__,
                            },
                            exc_info=True,
                        )
                        await self._publish_to_dlq(dlq, channel_name, msg, sub.module_name, e)
                        return

                try:
                    await sub.callback_fn(msg)
                    self._metrics.inc_dispatched(channel_name)
                    self._subscriber_last_seen[(channel_name, sub.module_name)] = time.time()
                except Exception as e:
                    self._metrics.inc_errors(channel_name)
                    self.logger.error(
                        "Subscriber error",
                        extra={
                            "event": "subscriber_error",
                            "channel": channel_name,
                            "subscriber": sub.module_name,
                            "msg_id": msg.id,
                            "trace_id": msg.trace_id,
                            "error_type": type(e).__name__,
                        },
                        exc_info=True,
                    )
                    await self._publish_to_dlq(dlq, channel_name, msg, sub.module_name, e)

        await asyncio.gather(*(_call_sub(s) for s in subs), return_exceptions=True)

    async def _publish_to_dlq(
        self,
        dlq_channel: Optional[str],
        origin_channel: str,
        msg: Message,
        subscriber: str,
        err: Exception,
    ) -> None:
        """
        Publish subscriber failures to a DLQ, if configured.
        DLQ payload contains original message + error info.

        Safety:
          - if the message is already a DLQ message, do not DLQ it again (avoids loops).
        """
        if not dlq_channel:
            return
        if msg.meta.get("dlq"):
            return
        if dlq_channel == origin_channel:
            return

        if dlq_channel not in self._queues:
            self.create_channel(dlq_channel, ChannelConfig(name=dlq_channel, max_queue_size=self.default_max_queue_size))

        payload = {
            "origin_channel": origin_channel,
            "subscriber": subscriber,
            "error_type": type(err).__name__,
            "error": str(err),
            "traceback": traceback.format_exc(),
            "message": msg.to_dict(),
        }

        dlq_meta = {"trace_id": msg.trace_id, "dlq": True, "origin_msg_id": msg.id}
        await self.publish(dlq_channel, payload, priority=0, meta=dlq_meta, auth=None)

    # ----------------------- Persistence helpers -----------------------

    async def _persist_message(self, msg: Message) -> None:
        if not self._persistence:
            return
        cfg = self._channel_cfg[msg.channel]
        if not cfg.persistent:
            return
        try:
            codec = self._codec[msg.channel]
            blob = codec.dumps(msg.to_dict())
            _ = await self._persistence.append(msg.channel, msg, blob)
        except Exception as e:
            self._metrics.inc_errors(msg.channel)
            self.logger.error(
                "Persistence append failed",
                extra={
                    "event": "persistence_append_fail",
                    "channel": msg.channel,
                    "msg_id": msg.id,
                    "trace_id": msg.trace_id,
                    "error_type": type(e).__name__,
                },
                exc_info=True,
            )

    async def _replay_to_subscriber(self, channel: str, sub: SubscriberConfig, after_seq: int) -> None:
        if not self._persistence:
            return
        if not self._channel_cfg[channel].persistent:
            return

        try:
            rows = await self._persistence.replay(channel, after_seq=after_seq, limit=5000)
            codec = self._codec[channel]
            for seq, blob in rows:
                d = cast(Dict[str, Any], codec.loads(blob))
                meta = dict(d.get("meta") or {})
                meta["replay"] = True
                meta["persist_seq"] = seq
                meta.setdefault("trace_id", meta.get("trace_id") or str(uuid.uuid4()))

                replay_msg = Message(
                    id=str(d.get("id") or ""),
                    channel=str(d.get("channel") or ""),
                    ts=float(d.get("ts") or 0.0),
                    priority=int(d.get("priority") or 0),
                    payload=d.get("payload"),
                    meta=meta,
                )

                await self._dispatch_to_subscribers(channel, replay_msg, [sub])
                self._last_seq_seen[(channel, sub.module_name)] = seq
                self._subscriber_last_seen[(channel, sub.module_name)] = time.time()

            self.logger.info("Replay complete", extra={"event": "replay", "channel": channel, "subscriber": sub.module_name})
        except Exception as e:
            self._metrics.inc_errors(channel)
            self.logger.error(
                "Replay failed",
                extra={"event": "replay_fail", "channel": channel, "subscriber": sub.module_name, "error_type": type(e).__name__},
                exc_info=True,
            )

    # ----------------------- Redis bridge -----------------------

    async def _start_redis(self, redis_url: str) -> None:
        if aioredis is None:
            return

        try:
            self._redis_client = aioredis.from_url(redis_url, decode_responses=False)

            for ch, cfg in self._channel_cfg.items():
                if cfg.redis_in:
                    t = asyncio.create_task(self._redis_in_loop(cfg.redis_in, ch), name=f"NCB.redis_in.{ch}")
                    self._redis_tasks.add(t)
                    t.add_done_callback(self._redis_tasks.discard)

            self.logger.info("Redis bridge started", extra={"event": "redis_start"})
        except Exception as e:
            self.logger.warning("Failed to start Redis bridge: %s", e, extra={"event": "redis_start_fail"})
            self._redis_client = None

    async def _redis_publish(self, redis_channel: str, msg: Message) -> None:
        if not self._redis_client:
            return
        try:
            codec = self._codec[msg.channel]
            blob = codec.dumps(msg.to_dict())
            await self._redis_client.publish(redis_channel, blob)
        except Exception as e:
            self._metrics.inc_errors(msg.channel)
            self.logger.error(
                "Redis publish failed",
                extra={
                    "event": "redis_publish_fail",
                    "channel": msg.channel,
                    "msg_id": msg.id,
                    "trace_id": msg.trace_id,
                    "error_type": type(e).__name__,
                },
                exc_info=True,
            )

    async def _redis_in_loop(self, redis_channel: str, local_channel: str) -> None:
        """
        Redis inbound with retry + exponential backoff.
        """
        if not self._redis_client:
            return

        backoff = 0.5
        backoff_max = 15.0

        while self.running:
            pubsub = None
            try:
                pubsub = self._redis_client.pubsub()
                await pubsub.subscribe(redis_channel)
                self.logger.info("Redis inbound subscribed", extra={"event": "redis_subscribe", "channel": local_channel})
                backoff = 0.5

                async for item in pubsub.listen():
                    if not self.running:
                        break
                    if item is None or item.get("type") != "message":
                        continue
                    data = item.get("data")
                    if not isinstance(data, (bytes, bytearray)):
                        continue

                    try:
                        d = cast(Dict[str, Any], self._codec[local_channel].loads(bytes(data)))
                        meta = dict(d.get("meta") or {})
                        meta.setdefault("trace_id", meta.get("trace_id") or str(uuid.uuid4()))
                        meta["redis_in"] = redis_channel

                        ncb_msg = Message(
                            id=str(d.get("id") or uuid.uuid4()),
                            channel=local_channel,
                            ts=float(d.get("ts") or time.time()),
                            priority=int(d.get("priority") or self._channel_cfg[local_channel].default_priority),
                            payload=d.get("payload"),
                            meta=meta,
                        )

                        await self._queues[local_channel].put(ncb_msg.priority, ncb_msg)
                        self._metrics.inc_published(local_channel)
                    except Exception as e:
                        self._metrics.inc_errors(local_channel)
                        self.logger.error(
                            "Redis inbound decode failed",
                            extra={"event": "redis_decode_fail", "channel": local_channel, "error_type": type(e).__name__},
                            exc_info=True,
                        )

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._metrics.inc_errors(local_channel)
                self.logger.warning(
                    "Redis inbound loop error; retrying",
                    extra={"event": "redis_in_retry", "channel": local_channel, "error_type": type(e).__name__},
                    exc_info=True,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff_max, backoff * 2)
            finally:
                if pubsub is not None:
                    try:
                        await pubsub.unsubscribe(redis_channel)
                        await pubsub.close()
                    except Exception:
                        pass

    # ----------------------- Kafka bridge -----------------------

    async def _start_kafka(
        self,
        bootstrap_servers: str,
        topic_prefix: str,
        client_id: str,
        acks: str,
        max_in_flight: int,
    ) -> None:
        if AIOKafkaProducer is None or AIOKafkaConsumer is None:
            return

        self._kafka_topic_prefix = topic_prefix

        try:
            self._kafka_producer = AIOKafkaProducer(
                bootstrap_servers=bootstrap_servers,
                client_id=client_id,
                acks=acks,
                max_batch_size=max(1, int(max_in_flight)) * 16384,  # Scale batch size based on concurrency
            )
            await self._kafka_producer.start()

            self._kafka_consumer = AIOKafkaConsumer(  
                bootstrap_servers=bootstrap_servers,
                client_id=f"{client_id}-consumer",
                group_id=f"{client_id}-group",
                auto_offset_reset="latest",
                enable_auto_commit=True,
            )
            await self._kafka_consumer.start()

            for ch, cfg in self._channel_cfg.items():
                if cfg.kafka_in:
                    topic = f"{topic_prefix}.{cfg.kafka_in}"
                    t = asyncio.create_task(self._kafka_in_loop(topic, ch), name=f"NCB.kafka_in.{ch}")
                    self._kafka_tasks.add(t)
                    t.add_done_callback(self._kafka_tasks.discard)

            self.logger.info("Kafka bridge started", extra={"event": "kafka_start"})
        except Exception as e:
            self.logger.warning("Failed to start Kafka bridge: %s", e, extra={"event": "kafka_start_fail"})
            self._kafka_producer = None
            self._kafka_consumer = None

    async def _kafka_publish(self, kafka_topic: str, msg: Message) -> None:
        if not self._kafka_producer:
            return
        try:
            codec = self._codec[msg.channel]
            blob = codec.dumps(msg.to_dict())
            await self._kafka_producer.send_and_wait(kafka_topic, blob)
        except Exception as e:
            self._metrics.inc_errors(msg.channel)
            self.logger.error(
                "Kafka publish failed",
                extra={"event": "kafka_publish_fail", "channel": msg.channel, "msg_id": msg.id, "trace_id": msg.trace_id, "error_type": type(e).__name__},
                exc_info=True,
            )

    async def _kafka_in_loop(self, kafka_topic: str, local_channel: str) -> None:
        """
        Kafka inbound with retry + exponential backoff.
        """
        if not self._kafka_consumer:
            return

        backoff = 0.5
        backoff_max = 15.0

        while self.running:
            try:
                # aiokafka subscribe is sync
                self._kafka_consumer.subscribe([kafka_topic])
                self.logger.info("Kafka inbound subscribed", extra={"event": "kafka_subscribe", "channel": local_channel})

                async for km in self._kafka_consumer:
                    if not self.running:
                        break
                    try:
                        if km.value is None:
                            continue
                        d = cast(Dict[str, Any], self._codec[local_channel].loads(cast(bytes, km.value)))
                        meta = dict(d.get("meta") or {})
                        meta.setdefault("trace_id", meta.get("trace_id") or str(uuid.uuid4()))
                        meta["kafka_in"] = kafka_topic

                        ncb_msg = Message(
                            id=str(d.get("id") or uuid.uuid4()),
                            channel=local_channel,
                            ts=float(d.get("ts") or time.time()),
                            priority=int(d.get("priority") or self._channel_cfg[local_channel].default_priority),
                            payload=d.get("payload"),
                            meta=meta,
                        )

                        await self._queues[local_channel].put(ncb_msg.priority, ncb_msg)
                        self._metrics.inc_published(local_channel)
                    except Exception as e:
                        self._metrics.inc_errors(local_channel)
                        self.logger.error(
                            "Kafka inbound decode failed",
                            extra={"event": "kafka_decode_fail", "channel": local_channel, "error_type": type(e).__name__},
                            exc_info=True,
                        )

                backoff = 0.5  # if loop exits cleanly, reset
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._metrics.inc_errors(local_channel)
                self.logger.warning(
                    "Kafka inbound loop error; retrying",
                    extra={"event": "kafka_in_retry", "channel": local_channel, "error_type": type(e).__name__},
                    exc_info=True,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff_max, backoff * 2)
            finally:
                try:
                    self._kafka_consumer.unsubscribe()
                except Exception:
                    pass

    # ----------------------- Hot reload -----------------------

    async def _hot_reload_loop(self) -> None:
        assert self._hot_cfg_path is not None
        p = self._hot_cfg_path

        self.logger.info("Hot reload enabled", extra={"event": "hot_reload_start"})

        while self.running:
            try:
                if p.exists():
                    mtime = p.stat().st_mtime
                    if self._hot_mtime is None or mtime > self._hot_mtime:
                        cfg = self.load_config_file(p)
                        self.apply_config(cfg)
                        self._hot_mtime = mtime
                        self.logger.info("Hot reloaded config", extra={"event": "hot_reload_apply"})
                await asyncio.sleep(self._hot_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.warning(
                    "Hot reload error",
                    extra={"event": "hot_reload_error", "error_type": type(e).__name__},
                    exc_info=True,
                )
                await asyncio.sleep(self._hot_interval)

    # ----------------------- Subscriber GC -----------------------

    async def _subscriber_gc_loop(self) -> None:
        while self.running:
            try:
                now = time.time()
                stale = [k for k, ts in self._subscriber_last_seen.items() if (now - ts) > self._subscriber_ttl]
                for key in stale:
                    self._subscriber_last_seen.pop(key, None)
                    self._last_seq_seen.pop(key, None)
                await asyncio.sleep(self._subscriber_gc_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.warning(
                    "Subscriber GC error",
                    extra={"event": "subscriber_gc_error", "error_type": type(e).__name__},
                    exc_info=True,
                )
                await asyncio.sleep(self._subscriber_gc_interval)

    # ----------------------- Plugin execution -----------------------

    async def _run_validate_and_transform(self, msg: Message, *, phase: str) -> Message:
        for pl in self._plugins:
            if pl.validate:
                await pl.validate(msg)

        hook_attr = "pre_publish" if phase == "pre_publish" else "pre_dispatch"
        for pl in self._plugins:
            hook = getattr(pl, hook_attr, None)
            if hook:
                msg = await hook(msg)
        return msg

    async def _run_sinks(self, msg: Message, *, phase: str) -> None:
        sink_attr = "post_publish" if phase == "post_publish" else "post_dispatch"
        for pl in self._plugins:
            sink = getattr(pl, sink_attr, None)
            if sink:
                try:
                    await sink(msg)
                except Exception as e:
                    self._metrics.inc_errors(msg.channel)
                    self.logger.error(
                        "Plugin sink error",
                        extra={
                            "event": "plugin_sink_error",
                            "channel": msg.channel,
                            "msg_id": msg.id,
                            "trace_id": msg.trace_id,
                            "error_type": type(e).__name__,
                        },
                        exc_info=True,
                    )

    # ----------------------- Observability -----------------------

    def snapshot_metrics(self) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for ch, lm in self._metrics.local.items():
            out[ch] = dataclasses.asdict(lm)
            out[ch]["queue_size"] = self._queues[ch].qsize() if ch in self._queues else 0
        return out

    # ----------------------- Benchmarking -----------------------

    async def benchmark(
        self,
        channel: str,
        *,
        messages: int = 100_000,
        concurrency: int = 50,
        payload_factory: Optional[Callable[[int], Any]] = None,
        priority: int = 100,
    ) -> Dict[str, Any]:
        """
        Stress benchmark with race-condition-safe counter.
        NOTE: Subscriber callbacks dominate results. For raw bus throughput,
              benchmark with a no-op subscriber.
        """
        if channel not in self._queues:
            raise ValueError(f"Channel '{channel}' does not exist")

        payload_factory = payload_factory or (lambda i: {"i": i, "t": time.time()})

        done = 0
        done_lock = asyncio.Lock()
        done_evt = asyncio.Event()

        async def noop_sub(_msg: Message) -> None:
            nonlocal done
            async with done_lock:
                done += 1
                if done >= messages:
                    done_evt.set()

        await self.register_subscriber(channel, "__bench_noop__", noop_sub)

        t0 = time.perf_counter()

        async def producer(start_i: int, count: int) -> None:
            for j in range(count):
                i = start_i + j
                await self.publish(channel, payload_factory(i), priority=priority)

        per = messages // concurrency
        extra = messages % concurrency
        tasks = []
        idx = 0
        for k in range(concurrency):
            cnt = per + (1 if k < extra else 0)
            tasks.append(asyncio.create_task(producer(idx, cnt)))
            idx += cnt

        await asyncio.gather(*tasks)
        await asyncio.wait_for(done_evt.wait(), timeout=max(5.0, messages / 10_000))

        t1 = time.perf_counter()
        dt = t1 - t0
        mps = messages / dt if dt > 0 else float("inf")

        return {
            "channel": channel,
            "messages": messages,
            "concurrency": concurrency,
            "seconds": dt,
            "msgs_per_sec": mps,
            "metrics": self.snapshot_metrics().get(channel, {}),
        }


###############################################################################
# Example Usage
###############################################################################

if __name__ == "__main__":
    async def example_callback(msg: Message) -> None:
        print(f"[{msg.channel}] pr={msg.priority} id={msg.id} trace={msg.trace_id} payload={msg.payload}")

    async def main() -> None:
        # Structured logging (JSON)
        logger = setup_json_logging(level=logging.INFO, logger_name="NCB")

        # Security: require API key for "agent_commands"
        security = SecurityConfig(
            enabled=True,
            api_keys={"dev-key-123"},
            channel_roles={"agent_commands": {"operator", "admin"}},
        )

        ncb = NeuralCognitiveBus(
            persistence_path="./ncb_messages.sqlite",
            enable_prometheus=False,
            security=security,
            hot_reload_config_path=None,
            logger=logger,
        )

        # Channels (DLQ + bounded fan-out)
        ncb.create_channel(
            "agent_commands",
            ChannelConfig(
                name="agent_commands",
                max_queue_size=5000,
                drop_policy="drop_old",
                persistent=True,
                serializer="msgpack",
                compress=True,
                rate_limit_per_sec=50_000,
                rate_limit_burst=10_000,
                dead_letter_channel="agent_commands_dlq",
                fanout_max_concurrency=250,
            ),
        )

        ncb.create_channel(
            "sensor_data",
            ChannelConfig(
                name="sensor_data",
                max_queue_size=20_000,
                drop_policy="drop_new",
                persistent=False,
                serializer="msgpack",
                compress=False,
                rate_limit_per_sec=100_000,
                rate_limit_burst=50_000,
                dead_letter_channel="sensor_data_dlq",
                fanout_max_concurrency=500,
            ),
        )

        await ncb.start(redis_url=None)

        auth = {"api_key": "dev-key-123", "roles": ["operator"], "sub": "local_dev"}

        # Subscribers
        await ncb.register_subscriber("agent_commands", "Agent1", example_callback, auth=auth, replay_on_register=True, replay_after_seq=0)

        # A subscriber that will throw (to demonstrate DLQ)
        async def bad_sub(msg: Message) -> None:
            if isinstance(msg.payload, dict) and msg.payload.get("command") == "move":
                raise RuntimeError("Simulated subscriber failure")

        await ncb.register_subscriber("agent_commands", "BadAgent", bad_sub, auth=auth)

        await ncb.register_subscriber(
            "sensor_data",
            "Agent2",
            example_callback,
            filter_fn=lambda m: isinstance(m.payload, dict) and m.payload.get("type") == "vision",
        )

        # Publish
        await ncb.publish("agent_commands", {"command": "move", "direction": "north"}, priority=10, auth=auth)
        await ncb.publish("sensor_data", {"type": "vision", "data": "image_bytes"}, priority=200)
        await ncb.publish("sensor_data", {"type": "audio", "data": "audio_bytes"}, priority=200)

        await asyncio.sleep(0.5)

        # Optional: prune persisted messages older than 1 hour
        if ncb._persistence:
            deleted = await ncb._persistence.prune_old_messages("agent_commands", max_age_sec=3600)
            print("Pruned rows:", deleted)

        # Benchmark (optional)
        stats = await ncb.benchmark("sensor_data", messages=20_000, concurrency=50)
        print("Benchmark:", stats)

        await ncb.stop()

    asyncio.run(main())
