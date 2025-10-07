"""High-level helpers for working with user accounts.

This module provides a lightweight façade around :class:`UserAccountDatabase`
so callers do not need to interact with the SQLite layer directly.  The
service is responsible for persisting the currently active user through the
configuration layer and exposes a small set of convenience helpers that the
UI can safely call from asynchronous code via thread executors.
"""

from __future__ import annotations

import re
import secrets
import threading
import datetime as _dt
from dataclasses import dataclass, replace
from typing import Callable, Dict, Iterable, List, Optional, Tuple

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
    "PasswordResetChallenge",
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


@dataclass(frozen=True)
class PasswordResetChallenge:
    """Describe a password reset token issued to a user."""

    username: str
    token: str
    expires_at: _dt.datetime

    def expires_at_iso(self) -> str:
        timestamp = self.expires_at
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=_dt.timezone.utc)
        timestamp = timestamp.astimezone(_dt.timezone.utc).replace(microsecond=0)
        return timestamp.isoformat().replace("+00:00", "Z")


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

        self._lockout_threshold = self._resolve_lockout_setting(
            "ACCOUNT_LOCKOUT_MAX_FAILURES", 5
        )
        self._lockout_window_seconds = self._resolve_lockout_setting(
            "ACCOUNT_LOCKOUT_WINDOW_SECONDS", 300
        )
        self._lockout_duration_seconds = self._resolve_lockout_setting(
            "ACCOUNT_LOCKOUT_DURATION_SECONDS", 300
        )
        self._password_reset_validity_seconds = self._resolve_lockout_setting(
            "ACCOUNT_PASSWORD_RESET_TOKEN_LIFETIME_SECONDS",
            900,
        )

        self._stale_account_threshold_days = self._resolve_lockout_setting(
            "ACCOUNT_STALE_THRESHOLD_DAYS",
            90,
        )

        self._login_history_limit = 10
        self._failed_attempts: Dict[str, List[_dt.datetime]] = {}
        self._active_lockouts: Dict[str, _dt.datetime] = {}
        self._lockout_lock = threading.RLock()
        self._load_lockout_state()

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

    @staticmethod
    def _format_timestamp(moment: _dt.datetime) -> str:
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=_dt.timezone.utc)
        moment = moment.astimezone(_dt.timezone.utc).replace(microsecond=0)
        return moment.isoformat().replace("+00:00", "Z")

    @staticmethod
    def _parse_timestamp(value: Optional[str]) -> Optional[_dt.datetime]:
        if not value or not isinstance(value, str):
            return None

        text = value.strip()
        if not text:
            return None

        if text.endswith("Z"):
            text = text[:-1] + "+00:00"

        try:
            parsed = _dt.datetime.fromisoformat(text)
        except ValueError:
            return None

        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=_dt.timezone.utc)
        return parsed.astimezone(_dt.timezone.utc)

    def _load_lockout_state(self) -> None:
        """Initialise in-memory lockout tracking from persisted records."""

        try:
            entries = self._database.get_all_lockout_entries()
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Failed to load persisted lockout data: %s", exc)
            return

        now = self._current_time()
        threshold = None
        if self._lockout_window_seconds > 0:
            threshold = now - _dt.timedelta(seconds=self._lockout_window_seconds)

        updates: List[
            Tuple[str, List[_dt.datetime], Optional[_dt.datetime], List[str], Optional[str]]
        ] = []
        deletions: List[str] = []

        for username, attempt_isos, lockout_iso in entries:
            parsed_attempts: List[_dt.datetime] = []
            for attempt_iso in attempt_isos:
                parsed = self._parse_timestamp(attempt_iso)
                if parsed is None:
                    continue
                parsed_attempts.append(parsed)

            if threshold is not None:
                parsed_attempts = [
                    attempt for attempt in parsed_attempts if attempt >= threshold
                ]

            parsed_attempts.sort()

            lockout_until = self._parse_timestamp(lockout_iso)
            if lockout_until is not None and lockout_until <= now:
                lockout_until = None

            if not parsed_attempts and lockout_until is None:
                deletions.append(username)
                continue

            updates.append(
                (username, parsed_attempts, lockout_until, attempt_isos, lockout_iso)
            )

        if not updates and not deletions:
            return

        with self._lockout_lock:
            for username, attempts, lockout_until, _, _ in updates:
                if attempts:
                    self._failed_attempts[username] = attempts
                else:
                    self._failed_attempts.pop(username, None)

                if lockout_until is not None:
                    self._active_lockouts[username] = lockout_until
                else:
                    self._active_lockouts.pop(username, None)

            for username in deletions:
                self._failed_attempts.pop(username, None)
                self._active_lockouts.pop(username, None)

        for username in deletions:
            try:
                self._database.delete_lockout_entry(username)
            except Exception as exc:  # pragma: no cover - defensive logging
                self.logger.error(
                    "Failed to remove expired lockout record for '%s': %s",
                    username,
                    exc,
                )

        for username, attempts, lockout_until, original_attempts, original_lockout in updates:
            attempts_iso = [self._format_timestamp(moment) for moment in attempts]
            lockout_iso = (
                self._format_timestamp(lockout_until)
                if lockout_until is not None
                else None
            )

            if attempts_iso == list(original_attempts) and lockout_iso == original_lockout:
                continue

            try:
                self._database.set_lockout_entry(username, attempts_iso, lockout_iso)
            except Exception as exc:  # pragma: no cover - defensive logging
                self.logger.error(
                    "Failed to update persisted lockout data for '%s': %s",
                    username,
                    exc,
                )

    def _persist_lockout_state_locked(self, username: str) -> None:
        """Persist the current lockout tracking state for ``username``."""

        attempts = [
            self._format_timestamp(moment)
            for moment in self._failed_attempts.get(username, [])
        ]
        lockout_until = self._active_lockouts.get(username)
        lockout_iso = (
            self._format_timestamp(lockout_until)
            if lockout_until is not None
            else None
        )

        try:
            if attempts or lockout_iso is not None:
                self._database.set_lockout_entry(username, attempts, lockout_iso)
            else:
                self._database.delete_lockout_entry(username)
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error(
                "Failed to persist lockout data for '%s': %s", username, exc
            )

    def _clear_failures(self, username: str) -> None:
        with self._lockout_lock:
            removed_attempts = self._failed_attempts.pop(username, None)
            removed_lockout = self._active_lockouts.pop(username, None)

            if not removed_attempts and removed_lockout is None:
                return

            self._persist_lockout_state_locked(username)

    def _resolve_lockout_setting(self, key: str, default: int) -> int:
        """Return a validated integer lockout configuration value."""

        raw_value = self.config_manager.get_config(key, default)
        unset_marker = getattr(self.config_manager, "UNSET", object())

        if raw_value in (None, unset_marker):
            return default

        try:
            parsed = int(raw_value)
        except (TypeError, ValueError):
            self.logger.warning(
                "Ignoring %s override: %r is not an integer; using default %s",
                key,
                raw_value,
                default,
            )
            return default

        if parsed < 0:
            self.logger.warning(
                "Ignoring %s override: %r must be zero or positive; using default %s",
                key,
                raw_value,
                default,
            )
            return default

        return parsed

    def _prune_failures(self, username: str, now: _dt.datetime) -> None:
        if self._lockout_window_seconds <= 0:
            return

        with self._lockout_lock:
            attempts = self._failed_attempts.get(username)
            if not attempts:
                return

            threshold_time = now - _dt.timedelta(seconds=self._lockout_window_seconds)
            pruned_attempts = [
                attempt for attempt in attempts if attempt >= threshold_time
            ]

            if pruned_attempts == attempts:
                return

            self._failed_attempts[username] = pruned_attempts
            self._persist_lockout_state_locked(username)

    def _record_failure(self, username: str, now: _dt.datetime) -> None:
        if self._lockout_threshold <= 0 or self._lockout_duration_seconds <= 0:
            return

        with self._lockout_lock:
            attempts = self._failed_attempts.setdefault(username, [])
            attempts.append(now)
            self._prune_failures(username, now)

            attempts = self._failed_attempts.get(username, [])
            if len(attempts) < self._lockout_threshold:
                self._persist_lockout_state_locked(username)
                return

            lockout_until = now + _dt.timedelta(seconds=self._lockout_duration_seconds)
            self._active_lockouts[username] = lockout_until
            self.logger.warning(
                "Temporarily locking user '%s' due to too many failed login attempts.",
                username,
            )
            self._persist_lockout_state_locked(username)

    def _record_login_attempt(
        self,
        username: Optional[str],
        timestamp: str,
        successful: bool,
        reason: Optional[str],
    ) -> None:
        """Persist a login attempt and prune older records for the account."""

        try:
            self._database.add_login_attempt(username, timestamp, successful, reason)
            if username:
                self._database.prune_login_attempts(
                    username,
                    self._login_history_limit,
                )
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error(
                "Failed to record login attempt for '%s': %s",
                username,
                exc,
                exc_info=True,
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

    def _resolve_username_from_identifier(self, identifier: str) -> Optional[str]:
        """Attempt to resolve a username from a login identifier."""

        candidate = self._normalise_username(identifier)
        if candidate:
            if self._database.get_user(candidate):
                return candidate

        if self._EMAIL_PATTERN.fullmatch(identifier):
            username = self._database.get_username_for_email(identifier)
            if username:
                return username

        return None

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

        now = self._current_time()
        active_username = self.get_active_user()
        locked_usernames = self._locked_usernames(now)

        mappings: List[Dict[str, object]] = []
        for account in accounts:
            mapping = self._account_to_mapping(account)
            metadata = self._augment_account_metadata(
                mapping,
                account,
                active_username,
                locked_usernames,
                now,
            )
            mappings.append(metadata)

        return mappings

    def _augment_account_metadata(
        self,
        mapping: Dict[str, object],
        account: "UserAccount",
        active_username: Optional[str],
        locked_usernames: Iterable[str],
        now: Optional[_dt.datetime] = None,
    ) -> Dict[str, object]:
        enriched = dict(mapping)

        if now is None:
            now = self._current_time()

        enriched["is_active"] = bool(
            active_username and account.username.lower() == active_username.lower()
        )

        locked_set = {username.lower() for username in locked_usernames}
        enriched["is_locked"] = account.username.lower() in locked_set

        parsed_last_login = self._parse_timestamp(account.last_login)
        if parsed_last_login is not None and parsed_last_login > now:
            parsed_last_login = now

        last_login_age_days: Optional[float] = None
        if parsed_last_login is not None:
            last_login_age_days = max(
                (now - parsed_last_login).total_seconds() / 86400.0,
                0.0,
            )
        enriched["last_login_age_days"] = last_login_age_days

        enriched["status_badge"] = self._derive_status_badge(
            enriched,
            parsed_last_login,
        )

        return enriched

    def _derive_status_badge(
        self,
        mapping: Dict[str, object],
        last_login: Optional[_dt.datetime],
    ) -> str:
        if mapping.get("is_locked"):
            return "Locked"

        if mapping.get("is_active"):
            return "Active"

        if last_login is None:
            return "Never signed in"

        threshold_days = max(int(self._stale_account_threshold_days or 0), 0)
        if threshold_days:
            threshold = last_login + _dt.timedelta(days=threshold_days)
            now = self._current_time()
            if now >= threshold:
                return f"Inactive {threshold_days}+ days"

        return ""

    def _locked_usernames(self, now: Optional[_dt.datetime] = None) -> List[str]:
        if now is None:
            now = self._current_time()

        active: List[str] = []
        expired: List[str] = []

        with self._lockout_lock:
            for username, lockout_until in list(self._active_lockouts.items()):
                if lockout_until is None:
                    continue
                if now >= lockout_until:
                    expired.append(username)
                else:
                    active.append(username)

            for username in expired:
                self._failed_attempts.pop(username, None)
                self._active_lockouts.pop(username, None)
                self._persist_lockout_state_locked(username)

        return active

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

        timestamp = self._current_timestamp()
        identifier = self._normalise_username(username)
        if not identifier:
            self._record_login_attempt(None, timestamp, False, "invalid-identifier")
            return False

        lookup_username = identifier
        if self._EMAIL_PATTERN.fullmatch(identifier):
            resolved_username = self._database.get_username_for_email(identifier)
            if not resolved_username:
                self._record_login_attempt(identifier, timestamp, False, "unknown-identifier")
                return False
            lookup_username = resolved_username

        normalised_username = self._normalise_username(lookup_username)
        if not normalised_username:
            self._record_login_attempt(lookup_username, timestamp, False, "invalid-identifier")
            return False

        if password is None:
            self._record_login_attempt(normalised_username, timestamp, False, "missing-password")
            return False

        valid = False
        try:
            with self._lockout_lock:
                now = self._current_time()
                self._prune_failures(normalised_username, now)
                self._enforce_lockout(normalised_username, now)

                valid = bool(
                    self._database.verify_user_password(normalised_username, password)
                )

                if valid:
                    self._clear_failures(normalised_username)
                else:
                    self._record_failure(normalised_username, now)
        except AccountLockedError:
            self._record_login_attempt(normalised_username, timestamp, False, "account-locked")
            raise
        except Exception:
            self._record_login_attempt(normalised_username, timestamp, False, "error")
            raise

        if valid:
            timestamp = self._current_timestamp()
            self._record_login_attempt(normalised_username, timestamp, True, None)
            try:
                self._database.update_last_login(normalised_username, timestamp)
            except Exception as exc:  # pragma: no cover - defensive logging
                self.logger.error(
                    "Failed to update last-login timestamp for '%s': %s",
                    normalised_username,
                    exc,
                )
            return True

        self._record_login_attempt(normalised_username, timestamp, False, "invalid-credentials")
        return False

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

    def initiate_password_reset(
        self,
        identifier: str,
        *,
        expires_in_seconds: Optional[int] = None,
    ) -> Optional[PasswordResetChallenge]:
        """Generate a password reset token for a username or e-mail."""

        if not isinstance(identifier, str):
            raise TypeError("Username or email must be provided as a string")

        cleaned_identifier = identifier.strip()
        if not cleaned_identifier:
            raise ValueError("Username or email is required to reset a password.")

        username = self._resolve_username_from_identifier(cleaned_identifier)
        if not username:
            self.logger.info(
                "Password reset requested for unknown identifier %r", cleaned_identifier
            )
            return None

        if expires_in_seconds is None:
            lifetime = self._password_reset_validity_seconds
        else:
            try:
                lifetime = int(expires_in_seconds)
            except (TypeError, ValueError) as exc:
                raise ValueError("expires_in_seconds must be an integer value") from exc

        if lifetime < 0:
            lifetime = 0

        now = self._current_time()
        expires_at = now + _dt.timedelta(seconds=lifetime) if lifetime > 0 else now
        token = secrets.token_urlsafe(24)
        expires_at_iso = self._format_timestamp(expires_at)

        self._database.create_password_reset_token(username, token, expires_at_iso)
        self.logger.info("Issued password reset token for '%s'", username)
        return PasswordResetChallenge(username=username, token=token, expires_at=expires_at)

    def verify_password_reset_token(self, username: str, token: str) -> bool:
        """Validate whether a password reset token is valid for the user."""

        normalised_username = self._normalise_username(username)
        if not normalised_username:
            raise ValueError("Username must not be empty")

        if not isinstance(token, str) or not token.strip():
            raise ValueError("Reset token must not be empty")

        token_value = token.strip()

        self._require_existing_user(normalised_username)

        matches, expires_at_raw = self._database.verify_password_reset_token(
            normalised_username, token_value
        )
        if not matches:
            return False

        expires_at = self._parse_timestamp(expires_at_raw)
        if not expires_at:
            self._database.delete_password_reset_token(normalised_username)
            return False

        if self._current_time() > expires_at:
            self._database.delete_password_reset_token(normalised_username)
            return False

        return True

    def complete_password_reset(
        self, username: str, token: str, new_password: str
    ) -> bool:
        """Update the password using a valid reset token."""

        normalised_username = self._normalise_username(username)
        if not normalised_username:
            raise ValueError("Username must not be empty")

        if not isinstance(token, str) or not token.strip():
            raise ValueError("Reset token must not be empty")

        if not isinstance(new_password, str) or not new_password:
            raise ValueError("New password must not be empty")

        if not self.verify_password_reset_token(normalised_username, token.strip()):
            return False

        validated_password = self._validate_password(new_password)
        updated = self._database.set_user_password(normalised_username, validated_password)
        if not updated:
            return False

        self._database.delete_password_reset_token(normalised_username)
        self._clear_failures(normalised_username)
        self.logger.info("Password reset completed for '%s'", normalised_username)
        return True

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
        mapping = self._account_to_mapping(account)
        attempts = self._database.get_login_attempts(
            account.username,
            self._login_history_limit,
        )
        mapping["login_attempts"] = attempts
        return mapping

    def get_user_overview(self) -> Dict[str, object]:
        """Return aggregated statistics describing stored user accounts."""

        rows = self._database.get_all_users() or []
        total_accounts = len(rows)

        now = self._current_time()
        locked_usernames = set(self._locked_usernames(now))
        stale_threshold_days = max(int(self._stale_account_threshold_days or 0), 0)
        stale_delta = _dt.timedelta(days=stale_threshold_days) if stale_threshold_days else None

        active_username = self.get_active_user()
        active_display_name = ""

        never_signed_in = 0
        stale_accounts = 0
        most_recent_login: Optional[_dt.datetime] = None
        most_recent_username: Optional[str] = None

        for row in rows:
            account = self._row_to_account(row)
            mapping = self._account_to_mapping(account)

            if active_username and account.username.lower() == active_username.lower():
                active_display_name = str(mapping.get("display_name") or account.username)

            parsed_last_login = self._parse_timestamp(account.last_login)
            if parsed_last_login is None:
                never_signed_in += 1
            else:
                if stale_delta is not None and now - parsed_last_login >= stale_delta:
                    stale_accounts += 1

                if most_recent_login is None or parsed_last_login > most_recent_login:
                    most_recent_login = parsed_last_login
                    most_recent_username = account.username

        overview: Dict[str, object] = {
            "total_accounts": total_accounts,
            "active_username": active_username or "",
            "active_display_name": active_display_name or active_username or "",
            "locked_accounts": len(locked_usernames),
            "stale_accounts": stale_accounts,
            "never_signed_in": never_signed_in,
        }

        if most_recent_login is not None:
            overview["latest_sign_in_username"] = most_recent_username
            overview["latest_sign_in_at"] = self._format_timestamp(most_recent_login)
        else:
            overview["latest_sign_in_username"] = None
            overview["latest_sign_in_at"] = None

        return overview

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
