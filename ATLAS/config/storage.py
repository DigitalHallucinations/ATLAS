from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Dict, Mapping


class PerformanceMode(str, Enum):
    """Enumerate performance tuning profiles for storage components."""

    ECO = "eco"
    BALANCED = "balanced"
    PERFORMANCE = "performance"

    @classmethod
    def coerce(cls, value: Any, default: "PerformanceMode") -> "PerformanceMode":
        """Return the enum value matching ``value`` or ``default`` when invalid."""

        if isinstance(value, PerformanceMode):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            for candidate in cls:
                if candidate.value == lowered:
                    return candidate
        return default


@dataclass
class StorageArchitecture:
    """Describe storage component preferences and their performance posture."""

    performance_mode: PerformanceMode = PerformanceMode.BALANCED
    conversation_backend: str = "postgresql"
    kv_reuse_conversation_store: bool = True
    vector_store_adapter: str = "in_memory"

    def to_dict(self) -> Dict[str, Any]:
        """Return a serialisable representation suitable for YAML storage."""

        return {
            "performance_mode": self.performance_mode.value,
            "conversation_backend": self.conversation_backend,
            "kv_reuse_conversation_store": bool(self.kv_reuse_conversation_store),
            "vector_store_adapter": self.vector_store_adapter,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "StorageArchitecture":
        """Instantiate ``StorageArchitecture`` from a configuration mapping."""

        base_mode = PerformanceMode.BALANCED
        if isinstance(data, Mapping):
            base_mode = PerformanceMode.coerce(data.get("performance_mode"), base_mode)
        preset = PERFORMANCE_PRESETS.get(base_mode, PERFORMANCE_PRESETS[PerformanceMode.BALANCED])
        architecture = replace(preset)

        if not isinstance(data, Mapping):
            return architecture

        backend = data.get("conversation_backend")
        if isinstance(backend, str) and backend.strip():
            architecture.conversation_backend = backend.strip().lower()

        reuse_kv = data.get("kv_reuse_conversation_store")
        if reuse_kv is not None:
            architecture.kv_reuse_conversation_store = bool(reuse_kv)

        vector_adapter = data.get("vector_store_adapter")
        if isinstance(vector_adapter, str) and vector_adapter.strip():
            architecture.vector_store_adapter = vector_adapter.strip().lower()

        performance_mode = data.get("performance_mode")
        architecture.performance_mode = PerformanceMode.coerce(
            performance_mode, architecture.performance_mode
        )

        return architecture


PERFORMANCE_PRESETS: Dict[PerformanceMode, StorageArchitecture] = {
    PerformanceMode.ECO: StorageArchitecture(
        performance_mode=PerformanceMode.ECO,
        conversation_backend="sqlite",
        kv_reuse_conversation_store=True,
        vector_store_adapter="in_memory",
    ),
    PerformanceMode.BALANCED: StorageArchitecture(
        performance_mode=PerformanceMode.BALANCED,
        conversation_backend="postgresql",
        kv_reuse_conversation_store=True,
        vector_store_adapter="in_memory",
    ),
    PerformanceMode.PERFORMANCE: StorageArchitecture(
        performance_mode=PerformanceMode.PERFORMANCE,
        conversation_backend="postgresql",
        kv_reuse_conversation_store=False,
        vector_store_adapter="mongodb",
    ),
}
