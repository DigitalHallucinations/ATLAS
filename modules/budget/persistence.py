"""Persistence layer for budget data.

Provides storage backends for budget policies, usage records,
and alerts following the store_common patterns.
"""

from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Type, TypeVar
from uuid import uuid4

from modules.logging.logger import setup_logger

from .models import (
    AlertSeverity,
    AlertTriggerType,
    BudgetAlert,
    BudgetPeriod,
    BudgetPolicy,
    BudgetScope,
    LimitAction,
    OperationType,
    UsageRecord,
)

logger = setup_logger(__name__)

T = TypeVar("T")


class DecimalEncoder(json.JSONEncoder):
    """JSON encoder that handles Decimal values."""

    def default(self, o: Any) -> Any:
        if isinstance(o, Decimal):
            return str(o)
        if isinstance(o, datetime):
            return o.isoformat()
        if isinstance(o, Enum):
            return o.value
        return super().default(o)


def _serialize_for_storage(obj: Any) -> Any:
    """Serialize an object for storage.

    Args:
        obj: Object to serialize.

    Returns:
        JSON-serializable representation.
    """
    if isinstance(obj, dict):
        return {k: _serialize_for_storage(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_serialize_for_storage(v) for v in obj]
    elif isinstance(obj, Decimal):
        return str(obj)
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, Enum):
        return obj.value
    elif hasattr(obj, "as_dict"):
        return _serialize_for_storage(obj.as_dict())
    else:
        return obj


def _deserialize_decimal(value: Any) -> Decimal:
    """Deserialize a value to Decimal.

    Args:
        value: Value to deserialize.

    Returns:
        Decimal value.
    """
    if isinstance(value, Decimal):
        return value
    elif isinstance(value, (int, float)):
        return Decimal(str(value))
    elif isinstance(value, str):
        return Decimal(value)
    else:
        return Decimal("0")


def _deserialize_datetime(value: Any) -> datetime:
    """Deserialize a value to datetime.

    Args:
        value: Value to deserialize.

    Returns:
        datetime value.
    """
    if isinstance(value, datetime):
        return value
    elif isinstance(value, str):
        return datetime.fromisoformat(value)
    else:
        return datetime.now(timezone.utc)


class BudgetStoreBackend(ABC):
    """Abstract base class for budget storage backends.

    Implementations should handle:
    - Budget policies
    - Usage records
    - Alerts
    """

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the storage backend."""
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the storage backend."""
        pass

    # =========================================================================
    # Policy Operations
    # =========================================================================

    @abstractmethod
    async def save_policy(self, policy: BudgetPolicy) -> str:
        """Save a budget policy.

        Args:
            policy: Policy to save.

        Returns:
            Policy ID.
        """
        pass

    @abstractmethod
    async def get_policy(self, policy_id: str) -> Optional[BudgetPolicy]:
        """Get a policy by ID.

        Args:
            policy_id: Policy identifier.

        Returns:
            Policy if found.
        """
        pass

    @abstractmethod
    async def get_policies(
        self,
        scope: Optional[BudgetScope] = None,
        scope_id: Optional[str] = None,
        enabled_only: bool = False,
    ) -> List[BudgetPolicy]:
        """Get policies matching criteria.

        Args:
            scope: Filter by scope.
            scope_id: Filter by scope ID.
            enabled_only: Only return enabled policies.

        Returns:
            List of matching policies.
        """
        pass

    @abstractmethod
    async def delete_policy(self, policy_id: str) -> bool:
        """Delete a policy.

        Args:
            policy_id: Policy to delete.

        Returns:
            True if deleted.
        """
        pass

    # =========================================================================
    # Usage Record Operations
    # =========================================================================

    @abstractmethod
    async def save_usage_records(
        self,
        records: List[UsageRecord],
    ) -> int:
        """Save usage records in batch.

        Args:
            records: Records to save.

        Returns:
            Number of records saved.
        """
        pass

    @abstractmethod
    async def get_usage_records(
        self,
        start_date: datetime,
        end_date: datetime,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        limit: int = 1000,
    ) -> List[UsageRecord]:
        """Get usage records matching criteria.

        Args:
            start_date: Start of period.
            end_date: End of period.
            user_id: Filter by user.
            tenant_id: Filter by tenant.
            provider: Filter by provider.
            model: Filter by model.
            limit: Maximum records to return.

        Returns:
            List of matching records.
        """
        pass

    @abstractmethod
    async def get_aggregate_spend(
        self,
        start_date: datetime,
        end_date: datetime,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Decimal:
        """Get aggregate spend for a period.

        Args:
            start_date: Start of period.
            end_date: End of period.
            user_id: Filter by user.
            tenant_id: Filter by tenant.
            provider: Filter by provider.
            model: Filter by model.

        Returns:
            Total spend in USD.
        """
        pass

    @abstractmethod
    async def prune_old_records(
        self,
        older_than: datetime,
    ) -> int:
        """Delete records older than a date.

        Args:
            older_than: Delete records before this date.

        Returns:
            Number of records deleted.
        """
        pass

    # =========================================================================
    # Alert Operations
    # =========================================================================

    @abstractmethod
    async def save_alert(self, alert: BudgetAlert) -> str:
        """Save a budget alert.

        Args:
            alert: Alert to save.

        Returns:
            Alert ID.
        """
        pass

    @abstractmethod
    async def get_alerts(
        self,
        active_only: bool = False,
        severity: Optional[AlertSeverity] = None,
        limit: int = 100,
    ) -> List[BudgetAlert]:
        """Get alerts matching criteria.

        Args:
            active_only: Only return non-acknowledged alerts.
            severity: Filter by severity.
            limit: Maximum alerts to return.

        Returns:
            List of matching alerts.
        """
        pass

    @abstractmethod
    async def acknowledge_alert(
        self,
        alert_id: str,
        acknowledged_by: str,
    ) -> bool:
        """Acknowledge an alert.

        Args:
            alert_id: Alert to acknowledge.
            acknowledged_by: User who acknowledged.

        Returns:
            True if acknowledged.
        """
        pass


class InMemoryBudgetStore(BudgetStoreBackend):
    """In-memory budget store for testing and development.

    Warning:
        Data is not persisted across restarts.
    """

    def __init__(self):
        self._policies: Dict[str, BudgetPolicy] = {}
        self._usage_records: List[UsageRecord] = []
        self._alerts: Dict[str, BudgetAlert] = {}
        self._lock = asyncio.Lock()
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the in-memory store."""
        self._initialized = True
        logger.info("In-memory budget store initialized")

    async def shutdown(self) -> None:
        """Shutdown the in-memory store."""
        self._policies.clear()
        self._usage_records.clear()
        self._alerts.clear()
        self._initialized = False

    # Policy Operations

    async def save_policy(self, policy: BudgetPolicy) -> str:
        """Save a policy to memory."""
        async with self._lock:
            self._policies[policy.id] = policy
            return policy.id

    async def get_policy(self, policy_id: str) -> Optional[BudgetPolicy]:
        """Get a policy from memory."""
        return self._policies.get(policy_id)

    async def get_policies(
        self,
        scope: Optional[BudgetScope] = None,
        scope_id: Optional[str] = None,
        enabled_only: bool = False,
    ) -> List[BudgetPolicy]:
        """Get policies matching criteria."""
        policies = list(self._policies.values())

        if scope:
            policies = [p for p in policies if p.scope == scope]
        if scope_id:
            policies = [p for p in policies if p.scope_id == scope_id]
        if enabled_only:
            policies = [p for p in policies if p.enabled]

        return sorted(policies, key=lambda p: p.priority, reverse=True)

    async def delete_policy(self, policy_id: str) -> bool:
        """Delete a policy from memory."""
        async with self._lock:
            if policy_id in self._policies:
                del self._policies[policy_id]
                return True
            return False

    # Usage Record Operations

    async def save_usage_records(
        self,
        records: List[UsageRecord],
    ) -> int:
        """Save usage records to memory."""
        async with self._lock:
            self._usage_records.extend(records)
            return len(records)

    async def get_usage_records(
        self,
        start_date: datetime,
        end_date: datetime,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        limit: int = 1000,
    ) -> List[UsageRecord]:
        """Get usage records from memory."""
        records = [
            r for r in self._usage_records
            if start_date <= r.timestamp <= end_date
        ]

        if user_id:
            records = [r for r in records if r.user_id == user_id]
        if tenant_id:
            records = [r for r in records if r.tenant_id == tenant_id]
        if provider:
            records = [r for r in records if r.provider == provider]
        if model:
            records = [r for r in records if r.model == model]

        return sorted(
            records,
            key=lambda r: r.timestamp,
            reverse=True,
        )[:limit]

    async def get_aggregate_spend(
        self,
        start_date: datetime,
        end_date: datetime,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Decimal:
        """Get aggregate spend from memory."""
        records = await self.get_usage_records(
            start_date=start_date,
            end_date=end_date,
            user_id=user_id,
            tenant_id=tenant_id,
            provider=provider,
            model=model,
            limit=100000,
        )
        return sum((r.cost_usd for r in records), Decimal("0"))

    async def prune_old_records(
        self,
        older_than: datetime,
    ) -> int:
        """Delete old records from memory."""
        async with self._lock:
            original_count = len(self._usage_records)
            self._usage_records = [
                r for r in self._usage_records
                if r.timestamp >= older_than
            ]
            return original_count - len(self._usage_records)

    # Alert Operations

    async def save_alert(self, alert: BudgetAlert) -> str:
        """Save an alert to memory."""
        async with self._lock:
            self._alerts[alert.id] = alert
            return alert.id

    async def get_alerts(
        self,
        active_only: bool = False,
        severity: Optional[AlertSeverity] = None,
        limit: int = 100,
    ) -> List[BudgetAlert]:
        """Get alerts from memory."""
        alerts = list(self._alerts.values())

        if active_only:
            alerts = [a for a in alerts if a.acknowledged_at is None]
        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        return sorted(
            alerts,
            key=lambda a: a.triggered_at,
            reverse=True,
        )[:limit]

    async def acknowledge_alert(
        self,
        alert_id: str,
        acknowledged_by: str,
    ) -> bool:
        """Acknowledge an alert in memory."""
        async with self._lock:
            if alert_id in self._alerts:
                alert = self._alerts[alert_id]
                # Create updated alert
                self._alerts[alert_id] = BudgetAlert(
                    id=alert.id,
                    policy_id=alert.policy_id,
                    trigger_type=alert.trigger_type,
                    severity=alert.severity,
                    message=alert.message,
                    current_spend=alert.current_spend,
                    limit_amount=alert.limit_amount,
                    threshold_percent=alert.threshold_percent,
                    triggered_at=alert.triggered_at,
                    acknowledged=True,
                    acknowledged_at=datetime.now(timezone.utc),
                    acknowledged_by=acknowledged_by,
                    metadata=alert.metadata,
                )
                return True
            return False


class SQLBudgetStore(BudgetStoreBackend):
    """SQL-based budget store using StorageManager patterns.

    Supports PostgreSQL and SQLite backends via StorageManager's
    session factory and engine.
    """

    def __init__(
        self,
        session_factory: Optional[Any] = None,
        engine: Optional[Any] = None,
        table_prefix: str = "budget_",
    ):
        """Initialize the SQL store.

        Args:
            session_factory: SQLAlchemy session factory from StorageManager.
            engine: SQLAlchemy engine from StorageManager.
            table_prefix: Prefix for table names.
        """
        self._engine = engine
        self._session_factory = session_factory
        self.table_prefix = table_prefix
        self._initialized = False
        self._memory_fallback: Optional[InMemoryBudgetStore] = None
        self._owns_engine = False  # Track if we created the engine ourselves

    async def initialize(self) -> None:
        """Initialize the SQL store and create tables."""
        # If we have a session factory from StorageManager, use it
        if self._session_factory is not None and self._engine is not None:
            try:
                from .sql_models import create_budget_schema

                # Create schema if needed
                create_budget_schema(self._engine, logger=logger)

                self._initialized = True
                logger.info("SQL budget store initialized via StorageManager")
                return
            except Exception as exc:
                logger.error("Failed to create budget schema: %s", exc)
                # Fall through to in-memory fallback

        # Fall back to in-memory store if no StorageManager provided
        logger.info("Using in-memory budget store (StorageManager not available)")
        self._memory_fallback = InMemoryBudgetStore()
        await self._memory_fallback.initialize()
        self._initialized = True

    async def shutdown(self) -> None:
        """Shutdown the SQL store."""
        if self._memory_fallback:
            await self._memory_fallback.shutdown()
            self._memory_fallback = None
        # Note: We don't dispose the engine here because StorageManager owns it
        self._session_factory = None
        self._engine = None
        self._initialized = False

    def _use_memory_fallback(self) -> bool:
        """Check if we should use memory fallback."""
        return self._memory_fallback is not None

    # =========================================================================
    # Policy Operations
    # =========================================================================

    async def save_policy(self, policy: BudgetPolicy) -> str:
        """Save a policy to the database."""
        if self._use_memory_fallback():
            return await self._memory_fallback.save_policy(policy)

        from .sql_models import BudgetPolicyTable

        def _save():
            with self._session_factory() as session:
                # Check if exists
                existing = session.get(BudgetPolicyTable, policy.id)
                if existing:
                    # Update
                    existing.name = policy.name
                    existing.description = policy.description
                    existing.scope = policy.scope.value
                    existing.scope_id = policy.scope_id
                    existing.period = policy.period.value
                    existing.limit_amount = policy.limit_amount
                    existing.soft_limit_percent = policy.soft_limit_percent
                    existing.action = policy.action.value
                    existing.enabled = policy.enabled
                    existing.priority = policy.priority
                    existing.rollover_enabled = policy.rollover_enabled
                    existing.rollover_max_percent = policy.rollover_max_percent
                    existing.updated_at = policy.updated_at
                    existing.metadata_ = policy.metadata
                else:
                    # Insert
                    row = BudgetPolicyTable.from_domain(policy)
                    session.add(row)
                session.commit()
                return policy.id

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _save)

    async def get_policy(self, policy_id: str) -> Optional[BudgetPolicy]:
        """Get a policy from the database."""
        if self._use_memory_fallback():
            return await self._memory_fallback.get_policy(policy_id)

        from .sql_models import BudgetPolicyTable

        def _get():
            with self._session_factory() as session:
                row = session.get(BudgetPolicyTable, policy_id)
                if row:
                    return row.to_domain()
                return None

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _get)

    async def get_policies(
        self,
        scope: Optional[BudgetScope] = None,
        scope_id: Optional[str] = None,
        enabled_only: bool = False,
    ) -> List[BudgetPolicy]:
        """Get policies matching criteria from the database."""
        if self._use_memory_fallback():
            return await self._memory_fallback.get_policies(scope, scope_id, enabled_only)

        from sqlalchemy import select
        from .sql_models import BudgetPolicyTable

        def _query():
            with self._session_factory() as session:
                stmt = select(BudgetPolicyTable)
                if scope:
                    stmt = stmt.where(BudgetPolicyTable.scope == scope.value)
                if scope_id:
                    stmt = stmt.where(BudgetPolicyTable.scope_id == scope_id)
                if enabled_only:
                    stmt = stmt.where(BudgetPolicyTable.enabled == True)
                stmt = stmt.order_by(BudgetPolicyTable.priority.desc())

                result = session.execute(stmt)
                return [row.to_domain() for row in result.scalars()]

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _query)

    async def delete_policy(self, policy_id: str) -> bool:
        """Delete a policy from the database."""
        if self._use_memory_fallback():
            return await self._memory_fallback.delete_policy(policy_id)

        from .sql_models import BudgetPolicyTable

        def _delete():
            with self._session_factory() as session:
                row = session.get(BudgetPolicyTable, policy_id)
                if row:
                    session.delete(row)
                    session.commit()
                    return True
                return False

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _delete)

    # =========================================================================
    # Usage Record Operations
    # =========================================================================

    async def save_usage_records(
        self,
        records: List[UsageRecord],
    ) -> int:
        """Save usage records to the database."""
        if self._use_memory_fallback():
            return await self._memory_fallback.save_usage_records(records)

        if not records:
            return 0

        from .sql_models import UsageRecordTable

        def _save():
            with self._session_factory() as session:
                for record in records:
                    row = UsageRecordTable.from_domain(record)
                    session.add(row)
                session.commit()
                return len(records)

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _save)

    async def get_usage_records(
        self,
        start_date: datetime,
        end_date: datetime,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        limit: int = 1000,
    ) -> List[UsageRecord]:
        """Get usage records from the database."""
        if self._use_memory_fallback():
            return await self._memory_fallback.get_usage_records(
                start_date, end_date, user_id, tenant_id, provider, model, limit
            )

        from sqlalchemy import select
        from .sql_models import UsageRecordTable

        def _query():
            with self._session_factory() as session:
                stmt = select(UsageRecordTable).where(
                    UsageRecordTable.timestamp >= start_date,
                    UsageRecordTable.timestamp <= end_date,
                )
                if user_id:
                    stmt = stmt.where(UsageRecordTable.user_id == user_id)
                if tenant_id:
                    stmt = stmt.where(UsageRecordTable.tenant_id == tenant_id)
                if provider:
                    stmt = stmt.where(UsageRecordTable.provider == provider)
                if model:
                    stmt = stmt.where(UsageRecordTable.model == model)
                stmt = stmt.order_by(UsageRecordTable.timestamp.desc()).limit(limit)

                result = session.execute(stmt)
                return [row.to_domain() for row in result.scalars()]

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _query)

    async def get_aggregate_spend(
        self,
        start_date: datetime,
        end_date: datetime,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Decimal:
        """Get aggregate spend from the database."""
        if self._use_memory_fallback():
            return await self._memory_fallback.get_aggregate_spend(
                start_date, end_date, user_id, tenant_id, provider, model
            )

        from sqlalchemy import func, select
        from .sql_models import UsageRecordTable

        def _aggregate():
            with self._session_factory() as session:
                stmt = select(func.coalesce(func.sum(UsageRecordTable.cost_usd), 0)).where(
                    UsageRecordTable.timestamp >= start_date,
                    UsageRecordTable.timestamp <= end_date,
                )
                if user_id:
                    stmt = stmt.where(UsageRecordTable.user_id == user_id)
                if tenant_id:
                    stmt = stmt.where(UsageRecordTable.tenant_id == tenant_id)
                if provider:
                    stmt = stmt.where(UsageRecordTable.provider == provider)
                if model:
                    stmt = stmt.where(UsageRecordTable.model == model)

                result = session.execute(stmt).scalar()
                return Decimal(str(result)) if result else Decimal("0")

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _aggregate)

    async def prune_old_records(
        self,
        older_than: datetime,
    ) -> int:
        """Delete old records from the database."""
        if self._use_memory_fallback():
            return await self._memory_fallback.prune_old_records(older_than)

        from sqlalchemy import delete
        from .sql_models import UsageRecordTable

        def _prune():
            with self._session_factory() as session:
                stmt = delete(UsageRecordTable).where(
                    UsageRecordTable.timestamp < older_than
                )
                result = session.execute(stmt)
                session.commit()
                return result.rowcount

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _prune)

    # =========================================================================
    # Alert Operations
    # =========================================================================

    async def save_alert(self, alert: BudgetAlert) -> str:
        """Save an alert to the database."""
        if self._use_memory_fallback():
            return await self._memory_fallback.save_alert(alert)

        from .sql_models import BudgetAlertTable

        def _save():
            with self._session_factory() as session:
                row = BudgetAlertTable.from_domain(alert)
                session.merge(row)
                session.commit()
                return alert.id

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _save)

    async def get_alerts(
        self,
        active_only: bool = False,
        severity: Optional[AlertSeverity] = None,
        limit: int = 100,
    ) -> List[BudgetAlert]:
        """Get alerts from the database."""
        if self._use_memory_fallback():
            return await self._memory_fallback.get_alerts(active_only, severity, limit)

        from sqlalchemy import select
        from .sql_models import BudgetAlertTable

        def _query():
            with self._session_factory() as session:
                stmt = select(BudgetAlertTable)
                if active_only:
                    stmt = stmt.where(BudgetAlertTable.acknowledged == False)
                if severity:
                    stmt = stmt.where(BudgetAlertTable.severity == severity.value)
                stmt = stmt.order_by(BudgetAlertTable.triggered_at.desc()).limit(limit)

                result = session.execute(stmt)
                return [row.to_domain() for row in result.scalars()]

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _query)

    async def acknowledge_alert(
        self,
        alert_id: str,
        acknowledged_by: str,
    ) -> bool:
        """Acknowledge an alert in the database."""
        if self._use_memory_fallback():
            return await self._memory_fallback.acknowledge_alert(alert_id, acknowledged_by)

        from .sql_models import BudgetAlertTable

        def _acknowledge():
            with self._session_factory() as session:
                row = session.get(BudgetAlertTable, alert_id)
                if row:
                    row.acknowledged = True
                    row.acknowledged_at = datetime.now(timezone.utc)
                    row.acknowledged_by = acknowledged_by
                    session.commit()
                    return True
                return False

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _acknowledge)


class BudgetStore:
    """Budget store facade with backend abstraction.

    Provides a unified interface for budget persistence,
    with support for multiple storage backends.

    Usage::

        store = BudgetStore()
        await store.initialize()

        # Save a policy
        policy = BudgetPolicy(...)
        policy_id = await store.save_policy(policy)

        # Get usage records
        records = await store.get_usage_records(
            start_date=datetime.now() - timedelta(days=30),
            end_date=datetime.now(),
        )
    """

    _instance: Optional["BudgetStore"] = None
    _lock = asyncio.Lock()

    def __init__(
        self,
        backend: Optional[BudgetStoreBackend] = None,
    ):
        """Initialize the store.

        Args:
            backend: Storage backend to use.
        """
        self._backend = backend or InMemoryBudgetStore()
        self._initialized = False

    @classmethod
    async def create(
        cls,
        backend: Optional[BudgetStoreBackend] = None,
    ) -> "BudgetStore":
        """Create or get the singleton instance.

        Args:
            backend: Storage backend to use.

        Returns:
            BudgetStore instance.
        """
        async with cls._lock:
            if cls._instance is None:
                cls._instance = cls(backend)
                await cls._instance.initialize()
            return cls._instance

    @classmethod
    def get_instance(cls) -> Optional["BudgetStore"]:
        """Get the current instance if it exists.

        Returns:
            Current instance or None.
        """
        return cls._instance

    async def initialize(self) -> None:
        """Initialize the store."""
        if not self._initialized:
            await self._backend.initialize()
            self._initialized = True
            logger.info("Budget store initialized")

    async def shutdown(self) -> None:
        """Shutdown the store."""
        if self._initialized:
            await self._backend.shutdown()
            self._initialized = False
            BudgetStore._instance = None
            logger.info("Budget store shutdown")

    # =========================================================================
    # Policy Operations
    # =========================================================================

    async def save_policy(self, policy: BudgetPolicy) -> str:
        """Save a budget policy."""
        return await self._backend.save_policy(policy)

    async def get_policy(self, policy_id: str) -> Optional[BudgetPolicy]:
        """Get a policy by ID."""
        return await self._backend.get_policy(policy_id)

    async def get_policies(
        self,
        scope: Optional[BudgetScope] = None,
        scope_id: Optional[str] = None,
        enabled_only: bool = False,
    ) -> List[BudgetPolicy]:
        """Get policies matching criteria."""
        return await self._backend.get_policies(scope, scope_id, enabled_only)

    async def delete_policy(self, policy_id: str) -> bool:
        """Delete a policy."""
        return await self._backend.delete_policy(policy_id)

    # =========================================================================
    # Usage Record Operations
    # =========================================================================

    async def save_usage_records(
        self,
        records: List[UsageRecord],
    ) -> int:
        """Save usage records in batch."""
        return await self._backend.save_usage_records(records)

    async def get_usage_records(
        self,
        start_date: datetime,
        end_date: datetime,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        limit: int = 1000,
    ) -> List[UsageRecord]:
        """Get usage records matching criteria."""
        return await self._backend.get_usage_records(
            start_date, end_date, user_id, tenant_id, provider, model, limit
        )

    async def get_aggregate_spend(
        self,
        start_date: datetime,
        end_date: datetime,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Decimal:
        """Get aggregate spend for a period."""
        return await self._backend.get_aggregate_spend(
            start_date, end_date, user_id, tenant_id, provider, model
        )

    async def prune_old_records(
        self,
        older_than: datetime,
    ) -> int:
        """Delete records older than a date."""
        return await self._backend.prune_old_records(older_than)

    # =========================================================================
    # Alert Operations
    # =========================================================================

    async def save_alert(self, alert: BudgetAlert) -> str:
        """Save a budget alert."""
        return await self._backend.save_alert(alert)

    async def get_alerts(
        self,
        active_only: bool = False,
        severity: Optional[AlertSeverity] = None,
        limit: int = 100,
    ) -> List[BudgetAlert]:
        """Get alerts matching criteria."""
        return await self._backend.get_alerts(active_only, severity, limit)

    async def acknowledge_alert(
        self,
        alert_id: str,
        acknowledged_by: str,
    ) -> bool:
        """Acknowledge an alert."""
        return await self._backend.acknowledge_alert(alert_id, acknowledged_by)


# Convenience function to get the store
async def get_budget_store() -> BudgetStore:
    """Get the budget store singleton.

    Returns:
        BudgetStore instance.
    """
    return await BudgetStore.create()
