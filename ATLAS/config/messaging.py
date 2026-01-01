"""Messaging configuration helpers for :mod:`ATLAS.config`."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, MutableMapping, Sequence, Tuple
from collections.abc import Mapping

from modules.orchestration.policy import MessagePolicy, PolicyResolver

from ATLAS.messaging import AgentBus, configure_agent_bus


@dataclass
class MessagingConfigSection:
    """Normalize messaging backend configuration."""

    config: MutableMapping[str, Any]
    yaml_config: MutableMapping[str, Any]
    env_config: Mapping[str, Any]
    logger: Any
    write_yaml_callback: Callable[[], None]

    def apply(self) -> None:
        messaging_block = _normalize_messaging_settings(self.config.get("messaging"), env_config=self.env_config)
        self.config["messaging"] = messaging_block

    def get_settings(self) -> dict[str, Any]:
        return _normalize_messaging_settings(self.config.get("messaging"), env_config=self.env_config)

    def set_settings(
        self,
        *,
        backend: str,
        redis_url: str | None = None,
        stream_prefix: str | None = None,
        initial_offset: str | None = None,
        initial_stream_id: str | None = None,
        replay_start: str | None = None,
        batch_size: int | None = None,
        blocking_timeout_ms: int | None = None,
        auto_claim_idle_ms: int | None = None,
        auto_claim_count: int | None = None,
        delete_acknowledged: bool | None = None,
        trim_max_length: int | None = None,
        delete_on_ack: bool | None = None,
        trim_maxlen: int | None = None,
        min_idle_ms: int | None = None,
        policy: Mapping[str, Any] | None = None,
        kafka: Mapping[str, Any] | None = None,
        # NCB settings
        persistence_path: str | None = None,
        enable_prometheus: bool | None = None,
        prometheus_port: int | None = None,
        kafka_bootstrap_servers: str | None = None,
        kafka_topic_prefix: str | None = None,
        kafka_client_id: str | None = None,
        kafka_acks: str | None = None,
        kafka_max_in_flight: int | None = None,
    ) -> dict[str, Any]:
        sanitized_backend = str(backend or "in_memory").strip().lower()
        if sanitized_backend not in {"in_memory", "redis", "ncb"}:
            sanitized_backend = "in_memory"

        block: dict[str, Any] = {"backend": sanitized_backend}
        if policy is not None:
            block["policy"] = dict(policy)
        if sanitized_backend == "redis":
            if redis_url:
                block["redis_url"] = str(redis_url).strip()
            if stream_prefix:
                block["stream_prefix"] = str(stream_prefix).strip()
            normalized_replay = _normalize_replay_start(
                replay_start if replay_start is not None else initial_offset or initial_stream_id
            )
            block["replay_start"] = normalized_replay
            block["initial_offset"] = normalized_replay
            block["initial_stream_id"] = normalized_replay
            if batch_size is not None:
                block["batch_size"] = _coerce_positive_int(batch_size, default=1)
            if blocking_timeout_ms is not None:
                block["blocking_timeout_ms"] = _coerce_positive_int(blocking_timeout_ms, default=1000)
            min_idle = min_idle_ms if min_idle_ms is not None else auto_claim_idle_ms
            if min_idle is not None:
                block["min_idle_ms"] = _coerce_positive_int(min_idle, default=60_000, allow_zero=True)
                block["auto_claim_idle_ms"] = block["min_idle_ms"]
            if auto_claim_count is not None:
                block["auto_claim_count"] = _coerce_positive_int(auto_claim_count, default=10)
            delete_flag = delete_on_ack if delete_on_ack is not None else delete_acknowledged
            if delete_flag is not None:
                block["delete_on_ack"] = bool(delete_flag)
            trim_value = trim_maxlen if trim_maxlen is not None else trim_max_length
            if trim_value is not None:
                block["trim_maxlen"] = _coerce_positive_int(trim_value, default=None)
        elif sanitized_backend == "ncb":
            if persistence_path is not None:
                block["persistence_path"] = str(persistence_path).strip()
            if enable_prometheus is not None:
                block["enable_prometheus"] = bool(enable_prometheus)
            if prometheus_port is not None:
                block["prometheus_port"] = _coerce_positive_int(prometheus_port, default=8000)
            if redis_url is not None:
                block["redis_url"] = str(redis_url).strip()
            if kafka_bootstrap_servers is not None:
                block["kafka_bootstrap_servers"] = str(kafka_bootstrap_servers).strip()
            if kafka_topic_prefix is not None:
                block["kafka_topic_prefix"] = str(kafka_topic_prefix).strip()
            if kafka_client_id is not None:
                block["kafka_client_id"] = str(kafka_client_id).strip()
            if kafka_acks is not None:
                block["kafka_acks"] = str(kafka_acks).strip()
            if kafka_max_in_flight is not None:
                block["kafka_max_in_flight"] = _coerce_positive_int(kafka_max_in_flight, default=5)
        if kafka is not None:
            block["kafka"] = _normalize_kafka_block(kafka, env_config=self.env_config)

        normalized = _normalize_messaging_settings(block, env_config=self.env_config)
        self.yaml_config["messaging"] = dict(normalized)
        self.config["messaging"] = dict(normalized)
        self.write_yaml_callback()
        return dict(normalized)


def setup_message_bus(settings: Mapping[str, Any], *, logger: Any) -> Tuple[Any, AgentBus]:
    """Instantiate an AgentBus according to the provided settings.

    The AgentBus wraps the Neural Cognitive Bus (NCB) and supports configuration
    for persistence, Prometheus metrics, Redis bridging, and Kafka integration.
    
    Legacy Redis/in-memory backends are no longer supported - all configurations
    now use the NCB-backed AgentBus.
    """

    backend_name = str(settings.get("backend", "in_memory") or "in_memory").lower()
    
    # Extract NCB settings
    persistence_path = settings.get("persistence_path")
    enable_prometheus = settings.get("enable_prometheus", False)
    prometheus_port = settings.get("prometheus_port", 9090)
    
    try:
        bus = configure_agent_bus(
            persistence_path=persistence_path,
            enable_prometheus=enable_prometheus,
            prometheus_port=prometheus_port,
        )
        # Return bus as both backend and bus for API compatibility
        return bus, bus
    except Exception as exc:
        logger.warning(
            "Failed to configure AgentBus: %s. Creating default instance.",
            exc,
        )
        bus = AgentBus()
        return bus, bus


_STREAM_ID_PATTERN = re.compile(r"^\d+-\d+$")


def _normalize_replay_start(value: Any | None) -> str:
    candidate = (str(value).strip() if value is not None else "") or "$"
    if candidate == "$":
        return "$"
    if _STREAM_ID_PATTERN.match(candidate):
        return candidate
    return "$"


def _normalize_policy_settings(value: Any | None) -> dict[str, Any]:
    block = dict(value) if isinstance(value, Mapping) else {}
    default_policy = _normalize_policy_entry(block.get("default"))
    rules = _normalize_policy_rules(block.get("prefixes") or block.get("rules"))
    return {"default": default_policy, "prefixes": rules}


def _normalize_policy_entry(value: Any | None) -> dict[str, Any]:
    defaults = MessagePolicy()
    policy = dict(value) if isinstance(value, Mapping) else {}
    retry_attempts = _coerce_positive_int(policy.get("retry_attempts"), default=defaults.retry_attempts, allow_zero=True)
    retry_delay = _coerce_positive_float(policy.get("retry_delay"), default=defaults.retry_delay, allow_zero=True)
    dlq_template = policy.get("dlq_topic_template")
    if dlq_template is None:
        dlq_template = policy.get("dlq_topic")
    if dlq_template is None:
        dlq_template = defaults.dlq_topic_template
    dlq_template = str(dlq_template).strip() if dlq_template else None
    replay_start = _normalize_replay_start(policy.get("replay_start") or defaults.replay_start)

    return {
        "tier": str(policy.get("tier") or defaults.tier),
        "retry_attempts": defaults.retry_attempts if retry_attempts is None else retry_attempts,
        "retry_delay": defaults.retry_delay if retry_delay is None else retry_delay,
        "dlq_topic_template": dlq_template if dlq_template else None,
        "replay_start": replay_start,
        "retention_seconds": _coerce_positive_int(
            policy.get("retention_seconds"), default=defaults.retention_seconds, allow_zero=True
        ),
        "idempotency_key_field": str(policy.get("idempotency_key_field") or defaults.idempotency_key_field or "") or None,
        "idempotency_ttl_seconds": _coerce_positive_int(
            policy.get("idempotency_ttl_seconds"),
            default=defaults.idempotency_ttl_seconds,
            allow_zero=True,
        ),
    }


def _normalize_policy_rules(value: Any | None) -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = []
    if isinstance(value, Mapping):
        candidates = []
        for key, val in value.items():
            candidate: dict[str, Any] = {"prefix": key}
            if isinstance(val, Mapping):
                candidate.update(dict(val))
            candidates.append(candidate)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        candidates = list(value)
    else:
        candidates = []

    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        prefix = str(candidate.get("prefix") or "").strip()
        if not prefix:
            continue
        rule_policy = dict(candidate.get("policy") or candidate)
        rule_policy.pop("prefix", None)
        rules.append({"prefix": prefix, "policy": _normalize_policy_entry(rule_policy)})
    return rules


def _build_policy_resolver(value: Any | None) -> PolicyResolver | None:
    settings = _normalize_policy_settings(value)
    default_policy = _policy_from_mapping(settings.get("default"))
    prefixes: dict[str, MessagePolicy] = {}
    for rule in settings.get("prefixes") or []:
        if not isinstance(rule, Mapping):
            continue
        prefix = str(rule.get("prefix") or "").strip()
        policy = _policy_from_mapping(rule.get("policy")) or default_policy
        if prefix:
            prefixes[prefix] = policy
    if default_policy is None and not prefixes:
        return None
    return PolicyResolver(prefixes, default_policy=default_policy or MessagePolicy())


def _policy_from_mapping(value: Any | None) -> MessagePolicy | None:
    if not isinstance(value, Mapping):
        return None
    return MessagePolicy(
        tier=str(value.get("tier") or "standard"),
        retry_attempts=_coerce_positive_int(value.get("retry_attempts"), default=3, allow_zero=True) or 0,
        retry_delay=_coerce_positive_float(value.get("retry_delay"), default=0.1, allow_zero=True) or 0.0,
        dlq_topic_template=value.get("dlq_topic_template"),
        replay_start=value.get("replay_start"),
        retention_seconds=_coerce_positive_int(value.get("retention_seconds"), default=None, allow_zero=True),
        idempotency_key_field=value.get("idempotency_key_field"),
        idempotency_ttl_seconds=_coerce_positive_int(
            value.get("idempotency_ttl_seconds"), default=None, allow_zero=True
        ),
    )


def _normalize_messaging_settings(
    messaging_block: Any | None, *, env_config: Mapping[str, Any]
) -> dict[str, Any]:
    block = dict(messaging_block) if isinstance(messaging_block, Mapping) else {}
    backend_name = str(block.get("backend") or "in_memory").lower()
    block["backend"] = backend_name

    block["policy"] = _normalize_policy_settings(block.get("policy"))

    if backend_name == "redis":
        redis_settings = _normalize_redis_settings(block, env_config=env_config)
        block.update(redis_settings)
    elif backend_name == "ncb":
        ncb_settings = _normalize_ncb_settings(block, env_config=env_config)
        block.update(ncb_settings)
    else:
        replay = _normalize_replay_start(block.get("replay_start") or block.get("initial_offset"))
        block["replay_start"] = replay
        block["initial_offset"] = replay
        block["initial_stream_id"] = replay
    block["kafka"] = _normalize_kafka_block(block.get("kafka"), env_config=env_config)
    return block


def _normalize_redis_settings(value: Mapping[str, Any] | None, *, env_config: Mapping[str, Any]) -> dict[str, Any]:
    redis_settings = dict(value) if isinstance(value, Mapping) else {}
    default_url = env_config.get("REDIS_URL", "redis://localhost:6379/0")
    redis_settings.setdefault("redis_url", default_url)
    redis_settings.setdefault("stream_prefix", "atlas_bus")

    raw_replay = redis_settings.get("replay_start")
    if raw_replay is None:
        raw_replay = redis_settings.get("initial_offset")
    if raw_replay is None:
        raw_replay = redis_settings.get("initial_stream_id")
    normalized_replay = _normalize_replay_start(raw_replay)
    redis_settings["replay_start"] = normalized_replay
    redis_settings["initial_offset"] = normalized_replay
    redis_settings["initial_stream_id"] = normalized_replay

    redis_settings["batch_size"] = _coerce_positive_int(redis_settings.get("batch_size"), default=1)
    redis_settings["blocking_timeout_ms"] = _coerce_positive_int(
        redis_settings.get("blocking_timeout_ms"), default=1000
    )
    min_idle_raw = redis_settings.get("min_idle_ms")
    if min_idle_raw is None:
        min_idle_raw = redis_settings.get("auto_claim_idle_ms")
    min_idle_ms = _coerce_positive_int(min_idle_raw, default=60_000, allow_zero=True)
    redis_settings["min_idle_ms"] = min_idle_ms
    redis_settings["auto_claim_idle_ms"] = min_idle_ms
    redis_settings["auto_claim_count"] = _coerce_positive_int(redis_settings.get("auto_claim_count"), default=10)

    delete_flag = redis_settings.get("delete_on_ack")
    if delete_flag is None:
        delete_flag = redis_settings.get("delete_acknowledged")
    redis_settings["delete_on_ack"] = True if delete_flag is None else bool(delete_flag)
    redis_settings["delete_acknowledged"] = redis_settings["delete_on_ack"]

    trim_raw = redis_settings.get("trim_maxlen")
    if trim_raw is None:
        trim_raw = redis_settings.get("trim_max_length")
    trim_value = _coerce_positive_int(trim_raw, default=None)
    if trim_value is not None:
        redis_settings["trim_maxlen"] = trim_value
        redis_settings["trim_max_length"] = trim_value
    else:
        redis_settings.pop("trim_maxlen", None)
        if "trim_max_length" in redis_settings and redis_settings["trim_max_length"] is None:
            redis_settings.pop("trim_max_length", None)
    return redis_settings


def _normalize_ncb_settings(value: Mapping[str, Any] | None, *, env_config: Mapping[str, Any]) -> dict[str, Any]:
    ncb_settings = dict(value) if isinstance(value, Mapping) else {}
    ncb_settings.setdefault("persistence_path", None)
    ncb_settings.setdefault("enable_prometheus", False)
    ncb_settings.setdefault("prometheus_port", 8000)
    ncb_settings.setdefault("redis_url", env_config.get("REDIS_URL"))
    ncb_settings.setdefault("kafka_bootstrap_servers", env_config.get("KAFKA_BOOTSTRAP_SERVERS"))
    ncb_settings.setdefault("kafka_topic_prefix", "atlas.bus")
    ncb_settings.setdefault("kafka_client_id", "atlas-message-bridge")
    ncb_settings.setdefault("kafka_enable_idempotence", True)
    ncb_settings.setdefault("kafka_acks", "all")
    ncb_settings.setdefault("kafka_max_in_flight", 5)
    ncb_settings.setdefault("kafka_delivery_timeout", 10.0)
    ncb_settings.setdefault("kafka_bridge_enabled", False)
    return ncb_settings


def _normalize_kafka_block(value: Any | None, *, env_config: Mapping[str, Any]) -> dict[str, Any]:
    block = dict(value) if isinstance(value, Mapping) else {}
    enabled = _coerce_bool(block.get("enabled"), default=False)
    block["enabled"] = enabled

    bootstrap_servers = block.get("bootstrap_servers") or env_config.get("KAFKA_BOOTSTRAP_SERVERS")
    block["bootstrap_servers"] = str(bootstrap_servers).strip() if bootstrap_servers else None

    topic_prefix = block.get("topic_prefix")
    block["topic_prefix"] = str(topic_prefix).strip() if topic_prefix else "atlas.bus"

    client_id = block.get("client_id")
    block["client_id"] = str(client_id).strip() if client_id else "atlas-message-bridge"

    driver = block.get("driver") or block.get("preferred_driver")
    block["driver"] = str(driver).strip().lower() if driver else None

    extra_config = block.get("producer_config")
    block["producer_config"] = dict(extra_config) if isinstance(extra_config, Mapping) else {}

    block["enable_idempotence"] = _coerce_bool(
        block.get("enable_idempotence") if block.get("enable_idempotence") is not None else block.get("idempotence"),
        default=True,
    )
    block["acks"] = str(block.get("acks") or "all").strip() or "all"
    block["max_in_flight"] = _coerce_positive_int(block.get("max_in_flight"), default=5, allow_zero=True)
    block["delivery_timeout"] = _coerce_positive_float(block.get("delivery_timeout"), default=10.0)
    block["bridge"] = _normalize_kafka_bridge_block(block.get("bridge"))
    return block


def _normalize_kafka_bridge_block(value: Any | None) -> dict[str, Any]:
    bridge = dict(value) if isinstance(value, Mapping) else {}
    bridge["enabled"] = _coerce_bool(bridge.get("enabled"), default=False)
    source_prefix = bridge.get("source_prefix")
    bridge["source_prefix"] = str(source_prefix).strip() if source_prefix else "redis_kafka"

    topics_raw = bridge.get("topics")
    topics_value: Sequence[Any] = (
        topics_raw if isinstance(topics_raw, Sequence) and not isinstance(topics_raw, (str, bytes, bytearray)) else []
    )
    topics: list[str] = []
    for topic in topics_value or []:
        cleaned = str(topic or "").strip()
        if cleaned:
            topics.append(cleaned)
    bridge["topics"] = topics

    topic_map = bridge.get("topic_map")
    if isinstance(topic_map, Mapping):
        bridge["topic_map"] = {str(key).strip(): str(value).strip() for key, value in topic_map.items()}
    else:
        bridge["topic_map"] = {}

    dlq_topic = bridge.get("dlq_topic")
    bridge["dlq_topic"] = str(dlq_topic).strip() if dlq_topic else "atlas.bridge.dlq"

    bridge["batch_size"] = _coerce_positive_int(bridge.get("batch_size"), default=1)
    bridge["max_attempts"] = _coerce_positive_int(bridge.get("max_attempts"), default=3)
    bridge["backoff_seconds"] = _coerce_positive_float(bridge.get("backoff_seconds"), default=1.0, allow_zero=True)
    return bridge


def _coerce_positive_int(value: Any | None, *, default: int | None, allow_zero: bool = False) -> int | None:
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    if parsed > 0 or (allow_zero and parsed == 0):
        return parsed
    return default


def _coerce_positive_float(value: Any | None, *, default: float | None, allow_zero: bool = False) -> float | None:
    if value is None:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if parsed > 0 or (allow_zero and parsed == 0):
        return parsed
    return default


def _coerce_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    try:
        return bool(int(value))
    except Exception:
        return bool(value)
