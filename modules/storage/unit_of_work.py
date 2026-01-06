"""Unit of Work pattern for cross-store atomic operations.

Provides transactional coordination across multiple storage backends
when operations need to be atomic across stores.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from modules.logging.logger import setup_logger

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

logger = setup_logger(__name__)


class UnitOfWorkState(str, Enum):
    """State of the unit of work."""

    PENDING = "pending"
    ACTIVE = "active"
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"


class UnitOfWorkError(Exception):
    """Error during unit of work execution."""

    pass


class RollbackError(UnitOfWorkError):
    """Error during rollback (original error preserved)."""

    def __init__(self, message: str, original: Exception, rollback_errors: List[Exception]) -> None:
        super().__init__(message)
        self.original = original
        self.rollback_errors = rollback_errors


@dataclass
class UnitOfWork:
    """Coordinates atomic operations across multiple stores.

    Implements a simplified unit of work pattern that tracks changes
    and ensures all-or-nothing semantics across SQL and document stores.

    Usage::

        async with storage.unit_of_work() as uow:
            # Register operations
            uow.add_sql_operation(lambda session: session.add(entity))
            uow.add_compensating_action(lambda: cleanup())

            # Commit happens automatically on context exit
            # Rollback happens on exception

    Note: True distributed transactions across SQL and NoSQL stores
    are not possible without a transaction coordinator. This implementation
    provides best-effort consistency with compensating actions for rollback.
    """

    sql_session: Optional["Session"] = None
    _state: UnitOfWorkState = field(default=UnitOfWorkState.PENDING, init=False)
    _sql_operations: List[Callable[["Session"], Any]] = field(default_factory=list, init=False)
    _compensating_actions: List[Callable[[], Any]] = field(default_factory=list, init=False)
    _committed_operations: List[str] = field(default_factory=list, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    @property
    def state(self) -> UnitOfWorkState:
        """Get the current state of the unit of work."""
        return self._state

    @property
    def is_active(self) -> bool:
        """Check if the unit of work is active."""
        return self._state == UnitOfWorkState.ACTIVE

    def add_sql_operation(self, operation: Callable[["Session"], Any]) -> None:
        """Register a SQL operation to be executed.

        Args:
            operation: Callable that takes a Session and performs the operation.
        """
        if self._state != UnitOfWorkState.ACTIVE:
            raise UnitOfWorkError("Cannot add operations to inactive unit of work")
        self._sql_operations.append(operation)

    def add_compensating_action(self, action: Callable[[], Any]) -> None:
        """Register a compensating action for rollback.

        Compensating actions are executed in reverse order during rollback
        to undo effects of non-transactional operations (e.g., external calls).

        Args:
            action: Callable to execute during rollback.
        """
        if self._state != UnitOfWorkState.ACTIVE:
            raise UnitOfWorkError("Cannot add actions to inactive unit of work")
        self._compensating_actions.append(action)

    def mark_committed(self, operation_id: str) -> None:
        """Mark an operation as committed (for tracking)."""
        self._committed_operations.append(operation_id)

    async def begin(self) -> None:
        """Begin the unit of work."""
        async with self._lock:
            if self._state != UnitOfWorkState.PENDING:
                raise UnitOfWorkError(f"Cannot begin unit of work in state {self._state}")
            self._state = UnitOfWorkState.ACTIVE
            logger.debug("Unit of work started")

    async def commit(self) -> None:
        """Commit all pending operations.

        For SQL operations, this commits the session transaction.
        """
        async with self._lock:
            if self._state != UnitOfWorkState.ACTIVE:
                raise UnitOfWorkError(f"Cannot commit unit of work in state {self._state}")

            try:
                # Execute SQL operations
                if self.sql_session is not None and self._sql_operations:
                    session = self.sql_session
                    await asyncio.to_thread(self._execute_sql_operations, session)

                # Commit SQL session
                if self.sql_session is not None:
                    await asyncio.to_thread(self.sql_session.commit)

                self._state = UnitOfWorkState.COMMITTED
                logger.debug(
                    "Unit of work committed",
                    extra={"operations": len(self._sql_operations)},
                )

            except Exception as exc:
                # Attempt rollback on commit failure
                await self._rollback_internal(exc)
                raise

    def _execute_sql_operations(self, session: "Session") -> None:
        """Execute all registered SQL operations."""
        for operation in self._sql_operations:
            operation(session)

    async def rollback(self) -> None:
        """Rollback all pending operations."""
        async with self._lock:
            await self._rollback_internal(None)

    async def _rollback_internal(self, original_error: Optional[Exception]) -> None:
        """Internal rollback implementation."""
        if self._state not in (UnitOfWorkState.ACTIVE, UnitOfWorkState.PENDING):
            return

        rollback_errors: List[Exception] = []

        # Rollback SQL session
        if self.sql_session is not None:
            try:
                await asyncio.to_thread(self.sql_session.rollback)
            except Exception as exc:
                rollback_errors.append(exc)
                logger.error(f"SQL rollback failed: {exc}")

        # Execute compensating actions in reverse order
        for action in reversed(self._compensating_actions):
            try:
                result = action()
                if asyncio.iscoroutine(result):
                    await result
            except Exception as exc:
                rollback_errors.append(exc)
                logger.error(f"Compensating action failed: {exc}")

        self._state = UnitOfWorkState.ROLLED_BACK
        logger.debug("Unit of work rolled back")

        if original_error and rollback_errors:
            raise RollbackError(
                "Rollback completed with errors",
                original_error,
                rollback_errors,
            )


@dataclass
class UnitOfWorkManager:
    """Factory for creating and managing units of work.

    Provides the context manager interface used by StorageManager.
    """

    session_factory: Optional[Callable[[], "Session"]] = None

    @contextlib.asynccontextmanager
    async def create(self):
        """Create a new unit of work context.

        Yields:
            UnitOfWork instance for registering operations.
        """
        session = None
        if self.session_factory is not None:
            session = await asyncio.to_thread(self.session_factory)

        uow = UnitOfWork(sql_session=session)
        await uow.begin()

        try:
            yield uow
            if uow.state == UnitOfWorkState.ACTIVE:
                await uow.commit()
        except Exception:
            if uow.state == UnitOfWorkState.ACTIVE:
                await uow.rollback()
            raise
        finally:
            if session is not None:
                await asyncio.to_thread(session.close)


__all__ = [
    "UnitOfWorkState",
    "UnitOfWorkError",
    "RollbackError",
    "UnitOfWork",
    "UnitOfWorkManager",
]
