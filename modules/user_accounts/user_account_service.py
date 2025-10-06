"""High-level helpers for working with user accounts.

This module provides a lightweight façade around :class:`UserAccountDatabase`
so callers do not need to interact with the SQLite layer directly.  The
service is responsible for persisting the currently active user through the
configuration layer and exposes a small set of convenience helpers that the
UI can safely call from asynchronous code via thread executors.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from ATLAS.config import ConfigManager
from modules.logging.logger import setup_logger

from .user_account_db import DuplicateUserError, UserAccountDatabase

__all__ = ["UserAccount", "UserAccountService", "DuplicateUserError"]


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

    _EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

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

    @staticmethod
    def _account_to_mapping(account: UserAccount) -> Dict[str, object]:
        return {
            "id": account.id,
            "username": account.username,
            "email": account.email,
            "name": account.name,
            "dob": account.dob,
        }

    def _rows_to_mappings(self, rows: Iterable[Iterable[object]]) -> List[Dict[str, object]]:
        accounts = [self._row_to_account(row) for row in rows]
        accounts.sort(key=lambda account: account.username.lower())
        return [self._account_to_mapping(account) for account in accounts]

    def _validate_email(self, email: str) -> str:
        if not isinstance(email, str):
            self.logger.error("Email must be provided as a string.")
            raise ValueError("Email must be a valid email address.")

        candidate = email.strip()
        if not candidate or not self._EMAIL_PATTERN.fullmatch(candidate):
            self.logger.error("Invalid email address provided: %r", email)
            raise ValueError("Email must be a valid email address.")

        return candidate

    def _validate_password(self, password: str) -> str:
        if not isinstance(password, str):
            self.logger.error("Password must be provided as a string.")
            raise ValueError(
                "Password must be at least 8 characters long and include letters and numbers."
            )

        if len(password) < 8:
            self.logger.error("Password failed minimum length requirement.")
            raise ValueError(
                "Password must be at least 8 characters long and include letters and numbers."
            )

        has_letter = any(char.isalpha() for char in password)
        has_digit = any(char.isdigit() for char in password)

        if not (has_letter and has_digit):
            self.logger.error("Password missing required character diversity.")
            raise ValueError(
                "Password must be at least 8 characters long and include letters and numbers."
            )

        return password

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

        validated_email = self._validate_email(email)
        validated_password = self._validate_password(password)

        self._database.add_user(
            normalised_username,
            validated_password,
            validated_email,
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
        return self._rows_to_mappings(rows)

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

    def delete_user(self, username: str) -> bool:
        """Remove a user account and clear it from configuration if active."""

        normalised_username = self._normalise_username(username)
        if not normalised_username:
            raise ValueError("Username must not be empty")

        active_user = self.get_active_user()
        deleted = self._database.delete_user(normalised_username)

        if not deleted:
            self.logger.info(
                "No user found with username '%s' to delete", normalised_username
            )
            return False

        if active_user == normalised_username:
            self.config_manager.set_active_user(None)
            self.logger.info(
                "Deleted active user '%s' and cleared active user selection",
                normalised_username,
            )
        else:
            self.logger.info("Deleted user '%s'", normalised_username)

        return True

    def update_user(
        self,
        username: str,
        *,
        password: Optional[str] = None,
        email: Optional[str] = None,
        name: Optional[str] = None,
        dob: Optional[str] = None,
    ) -> UserAccount:
        """Validate updates and persist them via the backing database."""

        normalised_username = self._normalise_username(username)
        if not normalised_username:
            raise ValueError("Username must not be empty")

        self._require_existing_user(normalised_username)

        validated_password = self._validate_password(password) if password is not None else None
        validated_email = self._validate_email(email) if email is not None else None

        self._database.update_user(
            normalised_username,
            password=validated_password,
            email=validated_email,
            name=name,
            dob=dob,
        )

        record = self._database.get_user(normalised_username)
        if not record:  # pragma: no cover - defensive safeguard
            raise RuntimeError("Failed to retrieve user after update")

        account = self._row_to_account(record)
        self.logger.info("Updated user '%s'", normalised_username)
        return account

    def close(self) -> None:
        """Release resources associated with the service."""

        try:
            self._database.close_connection()
        except Exception:  # pragma: no cover - defensive cleanup
            pass

    # ------------------------------------------------------------------
    # Extended queries
    # ------------------------------------------------------------------
    def search_users(self, query_text: Optional[str]) -> List[Dict[str, object]]:
        """Search for accounts by username, e-mail or display name."""

        rows = self._database.search_users(query_text)
        return self._rows_to_mappings(rows)

    def get_user_details(self, username: str) -> Optional[Dict[str, object]]:
        """Return a mapping of account details for the given username."""

        normalised_username = self._normalise_username(username)
        if not normalised_username:
            return None

        details = self._database.get_user_details(normalised_username)
        if not details:
            return None

        account = self._row_to_account(
            [
                details.get("id"),
                details.get("username"),
                None,
                details.get("email"),
                details.get("name"),
                details.get("dob"),
            ]
        )
        return self._account_to_mapping(account)

