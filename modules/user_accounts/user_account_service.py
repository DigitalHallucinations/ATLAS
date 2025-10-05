"""High-level helpers for working with user accounts.

This module provides a lightweight façade around :class:`UserAccountDatabase`
so callers do not need to interact with the SQLite layer directly.  The
service is responsible for persisting the currently active user through the
configuration layer and exposes a small set of convenience helpers that the
UI can safely call from asynchronous code via thread executors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from ATLAS.config import ConfigManager
from modules.logging.logger import setup_logger

from .user_account_db import UserAccountDatabase


@dataclass(frozen=True)
class UserAccount:
    """Serializable representation of a user account."""

    id: int
    username: str
    email: str
    name: Optional[str]
    dob: Optional[str]


class UserAccountService:
    """Provide high-level helpers for managing user accounts."""

    def __init__(
        self,
        *,
        config_manager: Optional[ConfigManager] = None,
        database: Optional[UserAccountDatabase] = None,
    ) -> None:
        self.logger = setup_logger(__name__)
        self.config_manager = config_manager or ConfigManager()
        self._database = database or UserAccountDatabase()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _normalise_username(username: Optional[str]) -> Optional[str]:
        if username is None:
            return None

        if not isinstance(username, str):
            raise TypeError("Username must be a string or None")

        cleaned = username.strip()
        return cleaned or None

    def _require_existing_user(self, username: str) -> None:
        if not self._database.get_user(username):
            raise ValueError(f"Unknown user: {username}")

    @staticmethod
    def _row_to_account(row: Iterable[object]) -> UserAccount:
        data = list(row)
        return UserAccount(
            id=int(data[0]),
            username=str(data[1]),
            email=str(data[3]),
            name=str(data[4]) if len(data) > 4 and data[4] is not None else None,
            dob=str(data[5]) if len(data) > 5 and data[5] is not None else None,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def register_user(
        self,
        username: str,
        password: str,
        email: str,
        name: Optional[str] = None,
        dob: Optional[str] = None,
    ) -> UserAccount:
        """Create a new user account in the backing store."""

        normalised_username = self._normalise_username(username)
        if not normalised_username:
            raise ValueError("Username must not be empty")

        if not password:
            raise ValueError("Password must not be empty")

        if self._database.get_user(normalised_username):
            raise ValueError(f"User '{normalised_username}' already exists")

        self._database.add_user(
            normalised_username,
            password,
            email,
            name,
            dob,
        )

        record = self._database.get_user(normalised_username)
        if not record:  # pragma: no cover - defensive safeguard
            raise RuntimeError("Failed to retrieve user after creation")

        account = self._row_to_account(record)
        self.logger.info("Registered new user '%s'", account.username)
        return account

    def authenticate_user(self, username: str, password: str) -> bool:
        """Return ``True`` when supplied credentials are valid."""

        normalised_username = self._normalise_username(username)
        if not normalised_username:
            return False

        if password is None:
            return False

        return bool(self._database.verify_user_password(normalised_username, password))

    def list_users(self) -> List[Dict[str, object]]:
        """Return a list of stored user accounts as dictionaries."""

        rows = self._database.get_all_users() or []
        accounts = [self._row_to_account(row) for row in rows]
        # Stable ordering ensures deterministic UI updates/tests.
        accounts.sort(key=lambda account: account.username.lower())
        return [
            {
                "id": account.id,
                "username": account.username,
                "email": account.email,
                "name": account.name,
                "dob": account.dob,
            }
            for account in accounts
        ]

    def get_active_user(self) -> Optional[str]:
        """Return the username persisted as active in configuration."""

        value = self.config_manager.get_active_user()
        return self._normalise_username(value)

    def set_active_user(self, username: Optional[str]) -> Optional[str]:
        """Persist the active user in configuration."""

        normalised_username = self._normalise_username(username)

        if normalised_username is not None:
            self._require_existing_user(normalised_username)
            self.logger.info("Setting active user to '%s'", normalised_username)
        else:
            self.logger.info("Clearing active user")

        return self.config_manager.set_active_user(normalised_username)

    def close(self) -> None:
        """Release resources associated with the service."""

        try:
            self._database.close_connection()
        except Exception:  # pragma: no cover - defensive cleanup
            pass

