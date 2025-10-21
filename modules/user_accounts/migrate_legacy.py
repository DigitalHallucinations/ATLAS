"""Utilities for migrating legacy SQLite user accounts into PostgreSQL."""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, Optional

from sqlalchemy.exc import IntegrityError

from modules.conversation_store import ConversationStoreRepository

from .user_account_service import ConversationCredentialStore


def _is_valid_password_hash(value: Optional[str]) -> bool:
    if not value or not isinstance(value, str):
        return False
    try:
        algorithm, iterations, salt_hex, hash_hex = value.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        int(iterations)
        bytes.fromhex(salt_hex)
        bytes.fromhex(hash_hex)
    except (ValueError, TypeError):
        return False
    return True


def _normalise_attempt_payload(payload: Optional[str]) -> Iterable[str]:
    if not payload:
        return []
    try:
        loaded = json.loads(payload)
    except (TypeError, json.JSONDecodeError):
        return []
    if not isinstance(loaded, list):
        return []
    return [str(entry) for entry in loaded if entry not in (None, "")]


def migrate_sqlite_accounts(
    sqlite_path: str | Path,
    repository: ConversationStoreRepository,
    *,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, int]:
    """Replay legacy SQLite user accounts into PostgreSQL.

    Returns a mapping describing how many entities were migrated.
    """

    path = Path(sqlite_path)
    if not path.exists():
        raise FileNotFoundError(f"Legacy database not found: {path}")

    log = logger or logging.getLogger(__name__)
    store = ConversationCredentialStore(repository)

    counters = {
        "users_migrated": 0,
        "users_skipped": 0,
        "lockouts_migrated": 0,
        "reset_tokens_migrated": 0,
        "login_attempts_migrated": 0,
    }

    user_ids: Dict[str, str] = {}

    with sqlite3.connect(str(path)) as conn:
        conn.row_factory = sqlite3.Row

        # migrate user rows
        for row in conn.execute(
            "SELECT username, password, email, name, DOB, last_login FROM user_accounts"
        ):
            username = str(row["username"]).strip()
            password_hash = row["password"]
            email = row["email"]
            name = row["name"] if row["name"] not in (None, "") else None
            dob = row["DOB"] if row["DOB"] not in (None, "") else None
            last_login = row["last_login"]

            if not username or not _is_valid_password_hash(password_hash):
                log.warning("Skipping user '%s' due to invalid password hash", username)
                counters["users_skipped"] += 1
                continue

            canonical_email = ConversationCredentialStore._canonicalize_email(email)
            if not canonical_email:
                log.warning("Skipping user '%s' due to invalid email", username)
                counters["users_skipped"] += 1
                continue

            user_uuid: Optional[str] = None
            try:
                repository.create_user_account(
                    username,
                    password_hash,
                    canonical_email,
                    name=name,
                    dob=dob,
                )
            except IntegrityError:
                existing = repository.get_user_account(username)
                if not existing:
                    log.warning("Unable to migrate user '%s' due to conflicts", username)
                    counters["users_skipped"] += 1
                    continue
                repository.update_user_account(
                    username,
                    name=name,
                    dob=dob,
                )
            else:
                counters["users_migrated"] += 1

            metadata = {"email": canonical_email}
            if name:
                metadata["name"] = name
            if dob:
                metadata["dob"] = dob

            try:
                user_uuid = str(
                    repository.ensure_user(
                        username,
                        display_name=name or username,
                        metadata=metadata,
                    )
                )
            except Exception as exc:  # pragma: no cover - defensive logging
                log.warning("Failed to ensure user '%s': %s", username, exc)
                user_uuid = None

            if user_uuid is not None:
                user_ids[username] = user_uuid
                try:
                    repository.update_user_account(username, user_id=user_uuid)
                except Exception as exc:  # pragma: no cover - defensive logging
                    log.warning(
                        "Failed to backfill user UUID for '%s': %s", username, exc
                    )

            if last_login not in (None, ""):
                try:
                    repository.update_last_login(
                        username,
                        str(last_login),
                        user_id=user_uuid,
                    )
                except Exception as exc:  # pragma: no cover - defensive logging
                    log.warning(
                        "Failed to migrate last login for '%s': %s", username, exc
                    )

        # migrate lockout states
        for row in conn.execute(
            "SELECT username, failed_attempts, lockout_until FROM account_lockouts"
        ):
            username = str(row["username"]).strip()
            if not username:
                continue
            attempts = list(_normalise_attempt_payload(row["failed_attempts"]))
            lockout_until = row["lockout_until"]
            user_uuid = user_ids.get(username)
            try:
                repository.set_lockout_state(
                    username,
                    attempts,
                    lockout_until,
                    user_id=user_uuid,
                )
                counters["lockouts_migrated"] += 1
            except Exception as exc:  # pragma: no cover - defensive logging
                log.warning(
                    "Failed to migrate lockout state for '%s': %s", username, exc
                )

        # migrate reset tokens
        for row in conn.execute(
            "SELECT username, token_hash, expires_at FROM password_reset_tokens"
        ):
            username = str(row["username"]).strip()
            token_hash = row["token_hash"]
            expires_at = row["expires_at"]
            if not username or not token_hash:
                continue
            try:
                repository.upsert_password_reset_token(
                    username,
                    str(token_hash),
                    str(expires_at) if expires_at not in (None, "") else None,
                    None,
                )
                counters["reset_tokens_migrated"] += 1
            except Exception as exc:  # pragma: no cover - defensive logging
                log.warning(
                    "Failed to migrate password reset token for '%s': %s", username, exc
                )

        # migrate login attempts
        for row in conn.execute(
            "SELECT username, attempted_at, successful, reason FROM user_login_attempts"
        ):
            username = row["username"]
            attempted_at = row["attempted_at"]
            successful = bool(row["successful"])
            reason = row["reason"]
            try:
                repository.record_login_attempt(
                    username if username not in (None, "") else None,
                    attempted_at,
                    successful,
                    reason,
                )
                counters["login_attempts_migrated"] += 1
            except Exception as exc:  # pragma: no cover - defensive logging
                log.warning(
                    "Failed to migrate login attempt for '%s': %s", username, exc
                )

    return counters
