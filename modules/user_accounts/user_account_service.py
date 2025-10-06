"""High-level helpers for working with user accounts.

This module provides a lightweight façade around :class:`UserAccountDatabase`
so callers do not need to interact with the SQLite layer directly.  The
service is responsible for persisting the currently active user through the
configuration layer and exposes a small set of convenience helpers that the
UI can safely call from asynchronous code via thread executors.
"""

from __future__ import annotations

import re
import threading
import datetime as _dt
from dataclasses import dataclass, replace
from typing import Callable, Dict, Iterable, List, Optional

from ATLAS.config import ConfigManager
from modules.logging.logger import setup_logger

from .user_account_db import (
    DuplicateUserError,
    InvalidCurrentPasswordError,
    UserAccountDatabase,
)

__all__ = [
    "UserAccount",
    "PasswordRequirements",
    "UserAccountService",
    "DuplicateUserError",
    "InvalidCurrentPasswordError",
    "AccountLockedError",
]


@dataclass(frozen=True)
class UserAccount:
    """Serializable representation of a user account."""

    id: int
    username: str
    email: str
    name: Optional[str]
    dob: Optional[str]
    last_login: Optional[str]



class AccountLockedError(RuntimeError):
    """Raised when authentication is temporarily blocked for a username."""

    def __init__(
        self,
        username: str,
        *,
        retry_at: Optional[_dt.datetime] = None,
        retry_after: Optional[int] = None,
    ) -> None:
        self.username = username
        self.retry_at = retry_at
        self.retry_after = retry_after

        message = "Too many failed login attempts. Please try again later."
        if retry_after is not None and retry_after > 0:
            plural = "s" if retry_after != 1 else ""
            message = (
                f"Too many failed login attempts. Try again in {retry_after} second{plural}."
            )

        super().__init__(message)


class UserAccountService:
    """Provide high-level helpers for managing user accounts."""

    _EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    _USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")
    _DOB_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    _MAX_DISPLAY_NAME_LENGTH = 80
    _DEFAULT_PASSWORD_REQUIREMENTS: "PasswordRequirements"

    def __init__(
        self,
        *,
        config_manager: Optional[ConfigManager] = None,
        database: Optional[UserAccountDatabase] = None,
        clock: Optional[Callable[[], _dt.datetime]] = None,
    ) -> None:
        self.logger = setup_logger(__name__)
        self.config_manager = config_manager or ConfigManager()
        self._database = database or UserAccountDatabase()
        self._clock: Callable[[], _dt.datetime] = clock or self._default_clock
        self._password_requirements = self._resolve_password_requirements()

        self._lockout_threshold = int(
            self.config_manager.get_config("ACCOUNT_LOCKOUT_MAX_FAILURES", 5)
            or 0
        )
        self._lockout_window_seconds = int(
            self.config_manager.get_config("ACCOUNT_LOCKOUT_WINDOW_SECONDS", 300)
            or 0
        )
        self._lockout_duration_seconds = int(
            self.config_manager.get_config("ACCOUNT_LOCKOUT_DURATION_SECONDS", 300)
            or 0
        )

        self._failed_attempts: Dict[str, List[_dt.datetime]] = {}
        self._active_lockouts: Dict[str, _dt.datetime] = {}
        self._lockout_lock = threading.RLock()

    def _resolve_password_requirements(self) -> PasswordRequirements:
        """Return the password policy taking configuration overrides into account."""

        base = self._DEFAULT_PASSWORD_REQUIREMENTS
        overrides: Dict[str, object] = {}

        min_length = self.config_manager.get_config(
            "ACCOUNT_PASSWORD_MIN_LENGTH", ConfigManager.UNSET
        )
        if min_length not in (None, ConfigManager.UNSET):
            try:
                parsed = int(min_length)
            except (TypeError, ValueError):
                self.logger.warning(
                    "Ignoring ACCOUNT_PASSWORD_MIN_LENGTH override: %r is not an integer",
                    min_length,
                )
            else:
                if parsed > 0:
                    overrides["min_length"] = parsed
                else:
                    self.logger.warning(
                        "Ignoring ACCOUNT_PASSWORD_MIN_LENGTH override: %r is not positive",
                        min_length,
                    )

        def _resolve_bool(key: str) -> Optional[bool]:
            raw_value = self.config_manager.get_config(key, ConfigManager.UNSET)
            if raw_value in (None, ConfigManager.UNSET):
                return None
            if isinstance(raw_value, bool):
                return raw_value
            if isinstance(raw_value, (int, float)):
                return bool(raw_value)
            if isinstance(raw_value, str):
                normalised = raw_value.strip().lower()
                if normalised in {"1", "true", "yes", "on"}:
                    return True
                if normalised in {"0", "false", "no", "off"}:
                    return False
            self.logger.warning(
                "Ignoring %s override: %r is not a recognised boolean value",
                key,
                raw_value,
            )
            return None

        bool_overrides = {
            "ACCOUNT_PASSWORD_REQUIRE_UPPERCASE": "require_uppercase",
            "ACCOUNT_PASSWORD_REQUIRE_LOWERCASE": "require_lowercase",
            "ACCOUNT_PASSWORD_REQUIRE_DIGIT": "require_digit",
            "ACCOUNT_PASSWORD_REQUIRE_SYMBOL": "require_symbol",
            "ACCOUNT_PASSWORD_FORBID_WHITESPACE": "forbid_whitespace",
        }

        for key, field_name in bool_overrides.items():
            resolved = _resolve_bool(key)
            if resolved is not None:
                overrides[field_name] = resolved

        if not overrides:
            return base

        return replace(base, **overrides)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _default_clock() -> _dt.datetime:
        return _dt.datetime.now(_dt.timezone.utc)

    def _current_time(self) -> _dt.datetime:
        return self._clock()

    def _clear_failures(self, username: str) -> None:
        with self._lockout_lock:
            self._failed_attempts.pop(username, None)
            self._active_lockouts.pop(username, None)

    def _prune_failures(self, username: str, now: _dt.datetime) -> None:
        if self._lockout_window_seconds <= 0:
            return

        with self._lockout_lock:
            attempts = self._failed_attempts.get(username)
            if not attempts:
                return

            threshold_time = now - _dt.timedelta(seconds=self._lockout_window_seconds)
            self._failed_attempts[username] = [
                attempt for attempt in attempts if attempt >= threshold_time
            ]

    def _record_failure(self, username: str, now: _dt.datetime) -> None:
        if self._lockout_threshold <= 0 or self._lockout_duration_seconds <= 0:
            return

        with self._lockout_lock:
            attempts = self._failed_attempts.setdefault(username, [])
            attempts.append(now)
            self._prune_failures(username, now)

            attempts = self._failed_attempts.get(username, [])
            if len(attempts) < self._lockout_threshold:
                return

            lockout_until = now + _dt.timedelta(seconds=self._lockout_duration_seconds)
            self._active_lockouts[username] = lockout_until
            self.logger.warning(
                "Temporarily locking user '%s' due to too many failed login attempts.",
                username,
            )

    def _remaining_lockout(self, username: str, now: _dt.datetime) -> Optional[int]:
        with self._lockout_lock:
            lockout_until = self._active_lockouts.get(username)
        if lockout_until is None:
            return None

        if now >= lockout_until:
            self._clear_failures(username)
            return None

        remaining = int((lockout_until - now).total_seconds())
        return max(remaining, 0)

    def _enforce_lockout(self, username: str, now: _dt.datetime) -> None:
        remaining = self._remaining_lockout(username, now)
        if remaining is None:
            return

        with self._lockout_lock:
            retry_at = self._active_lockouts.get(username)
        raise AccountLockedError(
            username,
            retry_at=retry_at,
            retry_after=remaining,
        )

    @staticmethod
    def _normalise_username(username: Optional[str]) -> Optional[str]:
        if username is None:
            return None

        if not isinstance(username, str):
            raise TypeError("Username must be a string or None")

        cleaned = username.strip()
        return cleaned or None

    @classmethod
    def _validate_username(cls, username: str) -> str:
        normalised_username = cls._normalise_username(username)
        if not normalised_username:
            raise ValueError("Username must not be empty")

        if not cls._USERNAME_PATTERN.fullmatch(normalised_username):
            raise ValueError(
                "Username must be 3-32 characters using letters, numbers, dots, hyphens or underscores."
            )

        return normalised_username

    @staticmethod
    def _normalise_optional_text(value: Optional[str], *, allow_empty: bool = False) -> Optional[str]:
        if value is None:
            return None

        if not isinstance(value, str):
            raise TypeError("Optional values must be strings when provided.")

        cleaned = value.strip()
        if cleaned:
            return cleaned

        if allow_empty:
            return ""

        return None

    @classmethod
    def _validate_display_name(
        cls, value: Optional[str], *, allow_empty: bool = False
    ) -> Optional[str]:
        cleaned = cls._normalise_optional_text(value, allow_empty=allow_empty)

        if cleaned in (None, ""):
            return cleaned

        if len(cleaned) > cls._MAX_DISPLAY_NAME_LENGTH:
            raise ValueError(
                f"Display name must be {cls._MAX_DISPLAY_NAME_LENGTH} characters or fewer."
            )

        return cleaned

    @classmethod
    def _validate_dob(
        cls, value: Optional[str], *, allow_empty: bool = False
    ) -> Optional[str]:
        cleaned = cls._normalise_optional_text(value, allow_empty=allow_empty)

        if cleaned in (None, ""):
            return cleaned

        if not cls._DOB_PATTERN.fullmatch(cleaned):
            raise ValueError("Date of birth must be in YYYY-MM-DD format.")

        try:
            parsed = _dt.date.fromisoformat(cleaned)
        except ValueError:
            raise ValueError("Date of birth must be in YYYY-MM-DD format.") from None

        if parsed > _dt.date.today():
            raise ValueError("Date of birth cannot be in the future.")

        return cleaned

    def _require_existing_user(self, username: str) -> None:
        if not self._database.get_user(username):
            raise ValueError(f"Unknown user: {username}")

    @staticmethod
    def _row_to_account(row: Iterable[object]) -> UserAccount:
        data = list(row)

        def _normalise_optional(value: object) -> Optional[str]:
            if value in (None, ""):
                return None
            return str(value)

        return UserAccount(
            id=int(data[0]),
            username=str(data[1]),
            email=str(data[3]),
            name=_normalise_optional(data[4]) if len(data) > 4 else None,
            dob=_normalise_optional(data[5]) if len(data) > 5 else None,
            last_login=_normalise_optional(data[6]) if len(data) > 6 else None,
        )

    @staticmethod
    def _account_to_mapping(account: UserAccount) -> Dict[str, object]:
        return {
            "id": account.id,
            "username": account.username,
            "email": account.email,
            "name": account.name,
            "dob": account.dob,
            "display_name": account.name or account.username,
            "last_login": account.last_login,
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

        return candidate.lower()

    def _validate_password(self, password: str) -> str:
        requirements = self._password_requirements
        error_message = requirements.describe()

        if not isinstance(password, str):
            self.logger.error("Password must be provided as a string.")
            raise ValueError(error_message)

        if requirements.forbid_whitespace and any(ch.isspace() for ch in password):
            self.logger.error("Password contains forbidden whitespace.")
            raise ValueError(error_message)

        if len(password) < requirements.min_length:
            self.logger.error("Password failed minimum length requirement.")
            raise ValueError(error_message)

        if requirements.require_lowercase and not any(ch.islower() for ch in password):
            self.logger.error("Password missing lowercase character.")
            raise ValueError(error_message)

        if requirements.require_uppercase and not any(ch.isupper() for ch in password):
            self.logger.error("Password missing uppercase character.")
            raise ValueError(error_message)

        if requirements.require_digit and not any(ch.isdigit() for ch in password):
            self.logger.error("Password missing numeric character.")
            raise ValueError(error_message)

        if requirements.require_symbol and not any(
            (not ch.isalnum()) and (not ch.isspace()) for ch in password
        ):
            self.logger.error("Password missing symbol character.")
            raise ValueError(error_message)

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

        normalised_username = self._validate_username(username)

        validated_email = self._validate_email(email)
        validated_password = self._validate_password(password)
        validated_name = self._validate_display_name(name)
        validated_dob = self._validate_dob(dob)

        self._database.add_user(
            normalised_username,
            validated_password,
            validated_email,
            validated_name,
            validated_dob,
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

        timestamp: Optional[str] = None
        with self._lockout_lock:
            now = self._current_time()
            self._prune_failures(normalised_username, now)
            self._enforce_lockout(normalised_username, now)

            valid = bool(
                self._database.verify_user_password(normalised_username, password)
            )

            if valid:
                self._clear_failures(normalised_username)
                timestamp = self._current_timestamp()
            else:
                self._record_failure(normalised_username, now)

        if valid and timestamp is not None:
            try:
                self._database.update_last_login(normalised_username, timestamp)
            except Exception as exc:  # pragma: no cover - defensive logging
                self.logger.error(
                    "Failed to update last-login timestamp for '%s': %s",
                    normalised_username,
                    exc,
                )
        return valid

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
        current_password: Optional[str] = None,
        email: Optional[str] = None,
        name: Optional[str] = None,
        dob: Optional[str] = None,
    ) -> UserAccount:
        """Validate updates and persist them via the backing database."""

        normalised_username = self._normalise_username(username)
        if not normalised_username:
            raise ValueError("Username must not be empty")

        self._require_existing_user(normalised_username)

        if password is not None:
            if not isinstance(current_password, str) or not current_password:
                raise ValueError("Current password is required to change the password.")

            if not self._database.verify_user_password(normalised_username, current_password):
                raise InvalidCurrentPasswordError("Current password is incorrect.")

        validated_password = self._validate_password(password) if password is not None else None
        validated_email = self._validate_email(email) if email is not None else None
        validated_name = self._validate_display_name(name, allow_empty=True)
        validated_dob = self._validate_dob(dob, allow_empty=True)

        self._database.update_user(
            normalised_username,
            password=validated_password,
            current_password=current_password if password is not None else None,
            email=validated_email,
            name=validated_name,
            dob=validated_dob,
        )

        record = self._database.get_user(normalised_username)
        if not record:  # pragma: no cover - defensive safeguard
            raise RuntimeError("Failed to retrieve user after update")

        account = self._row_to_account(record)
        self.logger.info("Updated user '%s'", normalised_username)
        return account

    def get_password_requirements(self) -> PasswordRequirements:
        """Return the password policy enforced by the service."""

        return self._password_requirements

    def describe_password_requirements(self) -> str:
        """Return a human readable summary of the password policy."""

        return self._password_requirements.describe()

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
                details.get("last_login"),
            ]
        )
        return self._account_to_mapping(account)

    @staticmethod
    def _current_timestamp() -> str:
        """Return the current UTC timestamp for last-login tracking."""

        now = _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0)
        iso = now.isoformat()
        return iso.replace("+00:00", "Z")


@dataclass(frozen=True)
class PasswordRequirements:
    """Describe the password policy enforced by :class:`UserAccountService`."""

    min_length: int
    require_uppercase: bool
    require_lowercase: bool
    require_digit: bool
    require_symbol: bool
    forbid_whitespace: bool

    def bullet_points(self) -> list[str]:
        """Return human readable bullet points describing the policy."""

        lines: list[str] = [f"Be at least {self.min_length} characters long"]
        if self.require_uppercase:
            lines.append("Contain an uppercase letter")
        if self.require_lowercase:
            lines.append("Contain a lowercase letter")
        if self.require_digit:
            lines.append("Include a number")
        if self.require_symbol:
            lines.append("Include a symbol (e.g. !, @, #)")
        if self.forbid_whitespace:
            lines.append("Avoid spaces")
        return lines

    def describe(self) -> str:
        """Return a single-sentence human readable description."""

        fragments = self.bullet_points()
        if not fragments:
            return "Passwords must be provided."

        if len(fragments) == 1:
            return f"Passwords must {fragments[0].lower()}."

        if len(fragments) == 2:
            return f"Passwords must {fragments[0].lower()} and {fragments[1].lower()}."

        requirements = ", ".join(fragment.lower() for fragment in fragments[:-1])
        return f"Passwords must {requirements}, and {fragments[-1].lower()}."


UserAccountService._DEFAULT_PASSWORD_REQUIREMENTS = PasswordRequirements(
    min_length=10,
    require_uppercase=True,
    require_lowercase=True,
    require_digit=True,
    require_symbol=True,
    forbid_whitespace=True,
)
