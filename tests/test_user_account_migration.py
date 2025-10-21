import sqlite3
from pathlib import Path

import pytest

sqlalchemy = pytest.importorskip("sqlalchemy")
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from modules.conversation_store import Base, ConversationStoreRepository
from modules.user_accounts.migrate_legacy import migrate_sqlite_accounts
from modules.user_accounts.user_account_service import ConversationCredentialStore


@pytest.fixture
def repository(postgresql):
    engine = create_engine(postgresql.dsn(), future=True)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, future=True)
    repo = ConversationStoreRepository(factory)
    repo.create_schema()
    try:
        yield repo
    finally:
        engine.dispose()


def _prepare_sqlite(db_path: Path) -> None:
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE user_accounts (
            id INTEGER PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            email TEXT NOT NULL,
            name TEXT,
            DOB TEXT,
            last_login TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE account_lockouts (
            id INTEGER PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            failed_attempts TEXT,
            lockout_until TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE password_reset_tokens (
            id INTEGER PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            token_hash TEXT NOT NULL,
            expires_at TEXT,
            created_at TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE user_login_attempts (
            id INTEGER PRIMARY KEY,
            username TEXT,
            attempted_at TEXT,
            successful INTEGER,
            reason TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def test_migrate_sqlite_accounts(tmp_path, repository):
    db_path = tmp_path / "legacy.db"
    _prepare_sqlite(db_path)

    hashed = ConversationCredentialStore._hash_password("Password123!")
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            "INSERT INTO user_accounts (username, password, email, name, DOB, last_login) VALUES (?, ?, ?, ?, ?, ?)",
            (
                "legacy",
                hashed,
                "legacy@example.com",
                "Legacy User",
                "1990-01-01",
                "2024-01-01T12:00:00Z",
            ),
        )
        conn.execute(
            "INSERT INTO account_lockouts (username, failed_attempts, lockout_until) VALUES (?, ?, ?)",
            ("legacy", '["2024-01-01T12:00:00Z"]', "2024-01-01T12:05:00Z"),
        )
        conn.execute(
            "INSERT INTO password_reset_tokens (username, token_hash, expires_at, created_at) VALUES (?, ?, ?, ?)",
            ("legacy", "a" * 64, "2024-01-02T00:00:00Z", "2023-12-31T00:00:00Z"),
        )
        conn.execute(
            "INSERT INTO user_login_attempts (username, attempted_at, successful, reason) VALUES (?, ?, ?, ?)",
            ("legacy", "2024-01-01T10:00:00Z", 0, "invalid"),
        )
        conn.commit()

    counters = migrate_sqlite_accounts(db_path, repository)

    assert counters["users_migrated"] == 1
    assert counters["lockouts_migrated"] == 1
    assert counters["reset_tokens_migrated"] == 1
    assert counters["login_attempts_migrated"] == 1

    record = repository.get_user_account("legacy")
    assert record is not None
    assert record["email"] == "legacy@example.com"
    assert record["name"] == "Legacy User"
    assert record["dob"] == "1990-01-01"
    assert record["last_login"] == "2024-01-01T12:00:00Z"

    lockout = repository.get_lockout_state("legacy")
    assert lockout is not None
    assert lockout["failed_attempts"] == ["2024-01-01T12:00:00Z"]
    assert lockout["lockout_until"] == "2024-01-01T12:05:00Z"

    attempts = repository.get_login_attempts("legacy")
    assert len(attempts) == 1
    assert attempts[0]["successful"] is False

