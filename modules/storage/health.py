"""Health check system for storage subsystems.

Provides structured health status reporting for all storage backends
including SQL databases, MongoDB, KV stores, and vector stores.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from modules.logging.logger import setup_logger

logger = setup_logger(__name__)


class HealthStatus(str, Enum):
    """Health status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class StoreHealth:
    """Health status for a single storage component.

    Attributes:
        name: Name of the storage component.
        status: Current health status.
        latency_ms: Response time in milliseconds.
        message: Optional status message.
        details: Additional diagnostic information.
        checked_at: When the check was performed.
    """

    name: str
    status: HealthStatus
    latency_ms: Optional[float] = None
    message: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    checked_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def is_healthy(self) -> bool:
        """Check if the component is healthy."""
        return self.status == HealthStatus.HEALTHY

    @property
    def is_available(self) -> bool:
        """Check if the component is available (healthy or degraded)."""
        return self.status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "status": self.status.value,
            "latency_ms": self.latency_ms,
            "message": self.message,
            "details": self.details,
            "checked_at": self.checked_at.isoformat(),
        }


@dataclass(slots=True)
class StorageHealthStatus:
    """Aggregate health status for all storage subsystems.

    Attributes:
        overall: Overall system health status.
        sql: SQL database health.
        mongo: MongoDB health (if configured).
        kv_store: Key-value store health.
        vector_store: Vector store health.
        conversation_store: Conversation store health.
        task_store: Task store health.
        job_store: Job store health.
        checked_at: When the aggregate check was performed.
    """

    overall: HealthStatus
    sql: Optional[StoreHealth] = None
    mongo: Optional[StoreHealth] = None
    kv_store: Optional[StoreHealth] = None
    vector_store: Optional[StoreHealth] = None
    conversation_store: Optional[StoreHealth] = None
    task_store: Optional[StoreHealth] = None
    job_store: Optional[StoreHealth] = None
    checked_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def is_healthy(self) -> bool:
        """Check if all storage systems are healthy."""
        return self.overall == HealthStatus.HEALTHY

    @property
    def is_available(self) -> bool:
        """Check if storage is available (may be degraded)."""
        return self.overall in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)

    @property
    def components(self) -> List[StoreHealth]:
        """Get list of all component health statuses."""
        components = []
        for attr in ("sql", "mongo", "kv_store", "vector_store", "conversation_store", "task_store", "job_store"):
            health = getattr(self, attr)
            if health is not None:
                components.append(health)
        return components

    @property
    def unhealthy_components(self) -> List[StoreHealth]:
        """Get list of unhealthy components."""
        return [c for c in self.components if not c.is_healthy]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "overall": self.overall.value,
            "checked_at": self.checked_at.isoformat(),
            "components": {},
        }
        for component in self.components:
            result["components"][component.name] = component.to_dict()
        return result


async def check_sql_health(
    pool: Any,
    *,
    timeout: float = 5.0,
    name: str = "sql",
) -> StoreHealth:
    """Check SQL database health.

    Args:
        pool: SQLPool instance.
        timeout: Health check timeout in seconds.
        name: Component name for reporting.

    Returns:
        Health status for the SQL database.
    """
    start = datetime.utcnow()

    try:
        if not pool.is_initialized:
            return StoreHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message="Pool not initialized",
            )

        healthy = await pool.health_check(timeout=timeout)
        latency_ms = (datetime.utcnow() - start).total_seconds() * 1000

        if healthy:
            return StoreHealth(
                name=name,
                status=HealthStatus.HEALTHY,
                latency_ms=latency_ms,
                message="Connected",
            )
        else:
            return StoreHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                latency_ms=latency_ms,
                message="Health check failed",
            )

    except asyncio.TimeoutError:
        return StoreHealth(
            name=name,
            status=HealthStatus.UNHEALTHY,
            message=f"Health check timed out after {timeout}s",
        )
    except Exception as exc:
        return StoreHealth(
            name=name,
            status=HealthStatus.UNHEALTHY,
            message=f"Error: {exc}",
        )


async def check_mongo_health(
    pool: Any,
    *,
    timeout: float = 5.0,
    name: str = "mongo",
) -> StoreHealth:
    """Check MongoDB health.

    Args:
        pool: MongoPool instance.
        timeout: Health check timeout in seconds.
        name: Component name for reporting.

    Returns:
        Health status for MongoDB.
    """
    start = datetime.utcnow()

    try:
        if not pool.is_initialized:
            return StoreHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message="Pool not initialized",
            )

        healthy = await pool.health_check(timeout=timeout)
        latency_ms = (datetime.utcnow() - start).total_seconds() * 1000

        if healthy:
            return StoreHealth(
                name=name,
                status=HealthStatus.HEALTHY,
                latency_ms=latency_ms,
                message="Connected",
            )
        else:
            return StoreHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                latency_ms=latency_ms,
                message="Health check failed",
            )

    except asyncio.TimeoutError:
        return StoreHealth(
            name=name,
            status=HealthStatus.UNHEALTHY,
            message=f"Health check timed out after {timeout}s",
        )
    except Exception as exc:
        return StoreHealth(
            name=name,
            status=HealthStatus.UNHEALTHY,
            message=f"Error: {exc}",
        )


async def check_vector_health(
    provider: Any,
    *,
    timeout: float = 5.0,
    name: str = "vector_store",
) -> StoreHealth:
    """Check vector store health.

    Args:
        provider: VectorProvider instance.
        timeout: Health check timeout in seconds.
        name: Component name for reporting.

    Returns:
        Health status for the vector store.
    """
    start = datetime.utcnow()

    try:
        if not provider.is_initialized:
            return StoreHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message="Provider not initialized",
            )

        healthy = await provider.health_check(timeout=timeout)
        latency_ms = (datetime.utcnow() - start).total_seconds() * 1000

        if healthy:
            return StoreHealth(
                name=name,
                status=HealthStatus.HEALTHY,
                latency_ms=latency_ms,
                message=f"Connected ({provider.name})",
                details={"provider": provider.name},
            )
        else:
            return StoreHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                latency_ms=latency_ms,
                message="Health check failed",
                details={"provider": provider.name},
            )

    except asyncio.TimeoutError:
        return StoreHealth(
            name=name,
            status=HealthStatus.UNHEALTHY,
            message=f"Health check timed out after {timeout}s",
        )
    except Exception as exc:
        return StoreHealth(
            name=name,
            status=HealthStatus.UNHEALTHY,
            message=f"Error: {exc}",
        )


def compute_overall_status(components: List[StoreHealth]) -> HealthStatus:
    """Compute overall health status from component statuses.

    Rules:
    - If any critical component is unhealthy -> UNHEALTHY
    - If any component is degraded -> DEGRADED
    - If all components are healthy -> HEALTHY
    - If no components -> UNKNOWN

    Args:
        components: List of component health statuses.

    Returns:
        Overall health status.
    """
    if not components:
        return HealthStatus.UNKNOWN

    statuses = [c.status for c in components]

    if HealthStatus.UNHEALTHY in statuses:
        return HealthStatus.UNHEALTHY
    if HealthStatus.DEGRADED in statuses:
        return HealthStatus.DEGRADED
    if HealthStatus.UNKNOWN in statuses:
        return HealthStatus.DEGRADED
    return HealthStatus.HEALTHY


__all__ = [
    "HealthStatus",
    "StoreHealth",
    "StorageHealthStatus",
    "check_sql_health",
    "check_mongo_health",
    "check_vector_health",
    "compute_overall_status",
]
