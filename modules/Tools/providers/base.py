"""Base classes and data structures for tool provider management."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Optional

from modules.logging.logger import setup_logger


_DEFAULT_HEALTH_CHECK_INTERVAL = 300.0
_DEFAULT_BACKOFF_BASE_SECONDS = 1.0
_MAX_BACKOFF_SECONDS = 60.0


@dataclass(frozen=True)
class ToolProviderSpec:
    """Configuration describing a provider option for a tool."""

    name: str
    priority: int = 100
    config: Mapping[str, Any] = field(default_factory=dict)
    health_check_interval: float = _DEFAULT_HEALTH_CHECK_INTERVAL

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ToolProviderSpec":
        name = str(raw.get("name", "")).strip()
        if not name:
            raise ValueError("Provider specification is missing a name")

        priority_value = raw.get("priority", 100)
        try:
            priority = int(priority_value)
        except (TypeError, ValueError):
            priority = 100

        interval_value = raw.get("health_check_interval", _DEFAULT_HEALTH_CHECK_INTERVAL)
        try:
            interval = float(interval_value)
        except (TypeError, ValueError):
            interval = _DEFAULT_HEALTH_CHECK_INTERVAL
        else:
            if interval < 0:
                interval = _DEFAULT_HEALTH_CHECK_INTERVAL

        config_value = raw.get("config")
        if isinstance(config_value, Mapping):
            config: Mapping[str, Any] = MappingProxyType(dict(config_value))
        else:
            config = MappingProxyType({})

        return cls(name=name, priority=priority, config=config, health_check_interval=interval)


@dataclass
class ProviderHealth:
    """Tracks health statistics for a tool provider."""

    successes: int = 0
    failures: int = 0
    consecutive_failures: int = 0
    last_success: Optional[float] = None
    last_failure: Optional[float] = None
    last_check: Optional[float] = None
    backoff_until: Optional[float] = None

    def record_success(self, timestamp: Optional[float] = None) -> None:
        timestamp = timestamp or time.time()
        self.successes += 1
        self.consecutive_failures = 0
        self.last_success = timestamp
        self.backoff_until = None

    def record_failure(self, timestamp: Optional[float] = None) -> float:
        """Record a provider failure and return the applied backoff duration."""

        timestamp = timestamp or time.time()
        self.failures += 1
        self.consecutive_failures += 1
        self.last_failure = timestamp

        exponent = max(self.consecutive_failures - 1, 0)
        delay = min(
            _MAX_BACKOFF_SECONDS,
            _DEFAULT_BACKOFF_BASE_SECONDS * (2 ** exponent),
        )
        self.backoff_until = timestamp + delay
        return delay

    def mark_health_check(self, success: bool, timestamp: Optional[float] = None) -> None:
        timestamp = timestamp or time.time()
        self.last_check = timestamp
        if success:
            self.record_success(timestamp)
        else:
            self.record_failure(timestamp)

    @property
    def total(self) -> int:
        return self.successes + self.failures

    @property
    def failure_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.failures / self.total

    def score(self) -> float:
        """Return a score used to compare providers (lower is better)."""

        penalty = 0.2 * min(self.consecutive_failures, 5)
        freshness = 0.0
        if self.last_check is not None:
            freshness = max(0.0, (time.time() - self.last_check) / 3600.0)
        return self.failure_rate + penalty + (freshness * 0.01)

    def is_available(self, timestamp: Optional[float] = None) -> bool:
        timestamp = timestamp or time.time()
        return self.backoff_until is None or timestamp >= self.backoff_until

    def should_run_health_check(self, interval: float, timestamp: Optional[float] = None) -> bool:
        if interval <= 0:
            return False
        timestamp = timestamp or time.time()
        if self.last_check is None:
            return True
        return (timestamp - self.last_check) >= interval

    def snapshot(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "successes": self.successes,
                "failures": self.failures,
                "consecutive_failures": self.consecutive_failures,
                "failure_rate": round(self.failure_rate, 4),
                "last_success": self.last_success,
                "last_failure": self.last_failure,
                "last_check": self.last_check,
                "backoff_until": self.backoff_until,
            }
        )


class ToolProvider:
    """Base class for tool provider implementations."""

    def __init__(
        self,
        spec: ToolProviderSpec,
        *,
        tool_name: str,
        fallback_callable: Optional[Any] = None,
    ) -> None:
        self.spec = spec
        self.tool_name = tool_name
        self.fallback_callable = fallback_callable
        self.config: Mapping[str, Any]
        if isinstance(spec.config, Mapping):
            self.config = MappingProxyType(dict(spec.config))
        else:
            self.config = MappingProxyType({})
        self.logger = setup_logger(f"{__name__}.{self.__class__.__name__}")
        self._health = ProviderHealth()

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def priority(self) -> int:
        return self.spec.priority

    @property
    def health(self) -> ProviderHealth:
        return self._health

    @property
    def health_check_interval(self) -> float:
        return max(self.spec.health_check_interval, 0.0)

    async def call(self, **kwargs: Any) -> Any:  # pragma: no cover - interface contract
        raise NotImplementedError

    async def health_check(self) -> bool:
        """Return ``True`` when the provider is considered healthy."""

        return True

    def health_snapshot(self) -> Mapping[str, Any]:
        return self._health.snapshot()


__all__ = [
    "ToolProviderSpec",
    "ProviderHealth",
    "ToolProvider",
]
