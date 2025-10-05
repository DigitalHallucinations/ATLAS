import hashlib
import hmac
import os
import shutil
import sqlite3
import threading
from pathlib import Path
from typing import Dict, Optional

try:
    from platformdirs import user_data_dir as _user_data_dir
except ImportError:  # pragma: no cover - optional dependency
    try:
        from appdirs import user_data_dir as _user_data_dir  # type: ignore
    except ImportError:  # pragma: no cover - optional dependency
        _user_data_dir = None

from ATLAS.config import ConfigManager
from modules.logging.logger import setup_logger


LEGACY_USER_PROFILES_DIR = Path(__file__).resolve().parent / 'user_profiles'


class DuplicateUserError(RuntimeError):
    """Raised when attempting to create a user with duplicate credentials."""


class UserAccountDatabase:
    def __init__(self, db_name: str = "User.db", base_dir: Optional[str] = None):
        self.logger = setup_logger(__name__)
        self.config_manager = ConfigManager()

        base_directory = self._determine_base_directory(base_dir)
        self.user_profiles_dir = base_directory / 'user_profiles'
        self.user_profiles_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = self.user_profiles_dir / db_name
        self._migrate_legacy_database(db_name)

        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._lock = threading.RLock()

        self.create_table()

    def _determine_base_directory(self, override_dir: Optional[str]) -> Path:
        if override_dir is not None:
            self.logger.debug("Using provided base directory for user accounts: %s", override_dir)
            return Path(override_dir).expanduser().resolve()

        app_root = self._get_app_root()
        if app_root:
            resolved = (Path(app_root).expanduser().resolve() / 'modules' / 'user_accounts')
            self.logger.debug("Using app root directory for user accounts: %s", resolved)
            return resolved

        if _user_data_dir:
            fallback_base = Path(_user_data_dir("ATLAS", "ATLAS"))
            self.logger.debug("Using OS user data directory for user accounts: %s", fallback_base)
        else:  # pragma: no cover - only used when no helper available
            fallback_base = Path.home() / '.atlas'
            self.logger.debug("Falling back to home directory for user accounts: %s", fallback_base)

        return fallback_base

    def _get_app_root(self) -> Optional[str]:
        get_app_root = getattr(self.config_manager, 'get_app_root', None)
        if not callable(get_app_root):
            return None

        try:
            return get_app_root()
        except Exception as exc:  # pragma: no cover - defensive programming
            self.logger.warning("Failed to retrieve app root from ConfigManager: %s", exc)
            return None

    def _migrate_legacy_database(self, db_name: str) -> None:
        legacy_db_path = LEGACY_USER_PROFILES_DIR / db_name

        try:
            if not legacy_db_path.exists():
                return

            new_db_path = self.user_profiles_dir / db_name
            if legacy_db_path.resolve() == new_db_path.resolve():
                return

            if new_db_path.exists():
                self.logger.info(
                    "Legacy database found at %s but new database already exists at %s; skipping migration.",
                    legacy_db_path,
                    new_db_path,
                )
                return

            new_db_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(legacy_db_path), str(new_db_path))
            self.logger.info(
                "Migrated user account database from legacy location %s to %s.",
                legacy_db_path,
                new_db_path,
            )
        except OSError as exc:
            self.logger.error(
                "Failed to migrate legacy user account database from %s: %s",
                legacy_db_path,
                exc,
            )
            raise

    def create_table(self):
        query = """
        CREATE TABLE IF NOT EXISTS user_accounts (
            id INTEGER PRIMARY KEY,
            username TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            name TEXT,
            DOB TEXT
        );
        """
        with self._lock:
            cursor = self.conn.cursor()
            try:
                cursor.execute(query)
                self.conn.commit()
                self._ensure_unique_constraints()
            finally:
                cursor.close()

    def _ensure_unique_constraints(self) -> None:
        """Ensure unique constraints exist for usernames and e-mail addresses."""

        required_indices: Dict[str, str] = {
            "idx_user_accounts_username": (
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_user_accounts_username"
                " ON user_accounts (username)"
            ),
            "idx_user_accounts_email": (
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_user_accounts_email"
                " ON user_accounts (email)"
            ),
        }

        with self._lock:
            cursor = self.conn.cursor()
            try:
                for index_name, create_statement in required_indices.items():
                    try:
                        cursor.execute(create_statement)
                    except sqlite3.IntegrityError as exc:
                        self.conn.rollback()
                        duplicates = self._find_duplicates(index_name)
                        details = ", ".join(
                            f"{field}: {values}" for field, values in duplicates.items() if values
                        )
                        message = (
                            "Existing user account records violate uniqueness constraints. "
                            "Please resolve duplicate entries before restarting the application."
                        )
                        if details:
                            message = f"{message} ({details})"
                        self.logger.error(message)
                        raise RuntimeError(message) from exc

                self.conn.commit()
            finally:
                cursor.close()

    def _find_duplicates(self, index_name: str) -> Dict[str, str]:
        if index_name == "idx_user_accounts_username":
            column = "username"
        elif index_name == "idx_user_accounts_email":
            column = "email"
        else:  # pragma: no cover - defensive guard for future indices
            return {}

        query = f"""
        SELECT {column}
        FROM user_accounts
        GROUP BY {column}
        HAVING COUNT(*) > 1
        """
        with self._lock:
            cursor = self.conn.cursor()
            try:
                rows = cursor.execute(query).fetchall()
            finally:
                cursor.close()
        values = ", ".join(sorted(str(row[0]) for row in rows))
        return {column: values} if values else {}

    @staticmethod
    def _hash_password(password, *, iterations=100_000):
        if password is None:
            raise ValueError("Password must not be None when hashing.")

        salt = os.urandom(16)
        password_bytes = password.encode('utf-8')
        dk = hashlib.pbkdf2_hmac('sha256', password_bytes, salt, iterations)
        return f"pbkdf2_sha256${iterations}${salt.hex()}${dk.hex()}"

    @staticmethod
    def _verify_password(stored_hash, candidate_password):
        if not stored_hash or candidate_password is None:
            return False

        try:
            algorithm, iterations, salt_hex, hash_hex = stored_hash.split('$', 3)
            iterations = int(iterations)
            salt = bytes.fromhex(salt_hex)
            expected_hash = bytes.fromhex(hash_hex)
        except (ValueError, TypeError):
            return False

        if algorithm != 'pbkdf2_sha256':
            return False

        candidate_bytes = candidate_password.encode('utf-8')
        candidate_hash = hashlib.pbkdf2_hmac('sha256', candidate_bytes, salt, iterations)
        return hmac.compare_digest(candidate_hash, expected_hash)

    def add_user(self, username, password, email, name, dob):
        hashed_password = self._hash_password(password)
        query = """
        INSERT INTO user_accounts (username, password, email, name, DOB)
        VALUES (?, ?, ?, ?, ?)
        """
        with self._lock:
            cursor = self.conn.cursor()
            try:
                cursor.execute(query, (username, hashed_password, email, name, dob))
                self.conn.commit()
            except sqlite3.IntegrityError as exc:
                self.conn.rollback()
                message = (
                    "A user with the same username or email already exists."
                )
                raise DuplicateUserError(message) from exc
            finally:
                cursor.close()

    def get_user(self, username):
        query = "SELECT * FROM user_accounts WHERE username = ?"
        with self._lock:
            cursor = self.conn.cursor()
            try:
                cursor.execute(query, (username,))
                return cursor.fetchone()
            finally:
                cursor.close()

    def get_all_users(self):
        query = "SELECT * FROM user_accounts"
        with self._lock:
            cursor = self.conn.cursor()
            try:
                cursor.execute(query)
                return cursor.fetchall()
            finally:
                cursor.close()

    def get_user_profile(self, username):
        query = "SELECT * FROM user_accounts WHERE username = ?"
        with self._lock:
            cursor = self.conn.cursor()
            try:
                cursor.execute(query, (username,))
                result = cursor.fetchone()
            finally:
                cursor.close()

        return result

    def close_connection(self):
        """Close the connection to the SQLite database."""
        self.logger.info("Closing UA database connection.")
        with self._lock:
            try:
                self.conn.close()
            except sqlite3.Error as e:
                self.logger.info(f"Error closing connection: {e}")
                raise

    def update_user(self, username, password=None, email=None, name=None, dob=None):
        with self._lock:
            cursor = self.conn.cursor()
            try:
                if password:
                    hashed_password = self._hash_password(password)
                    query = "UPDATE user_accounts SET password = ? WHERE username = ?"
                    cursor.execute(query, (hashed_password, username))
                if email:
                    query = "UPDATE user_accounts SET email = ? WHERE username = ?"
                    cursor.execute(query, (email, username))
                if name:
                    query = "UPDATE user_accounts SET name = ? WHERE username = ?"
                    cursor.execute(query, (name, username))
                if dob:
                    query = "UPDATE user_accounts SET DOB = ? WHERE username = ?"
                    cursor.execute(query, (dob, username))
                self.conn.commit()
            finally:
                cursor.close()

    def verify_user_password(self, username, candidate_password):
        user_record = self.get_user(username)
        if not user_record:
            return False

        stored_hash = user_record[2]
        return self._verify_password(stored_hash, candidate_password)

    def delete_user(self, username: str) -> bool:
        """Remove a user account and any associated profile data."""

        profile_path = self.user_profiles_dir / f"{username}.json"
        emr_path = self.user_profiles_dir / f"{username}_emr.txt"

        with self._lock:
            cursor = self.conn.cursor()
            try:
                cursor.execute("DELETE FROM user_accounts WHERE username = ?", (username,))
                deleted = cursor.rowcount > 0
                self.conn.commit()
            except sqlite3.DatabaseError:
                self.conn.rollback()
                raise
            finally:
                cursor.close()

            if deleted:
                for path, description in ((profile_path, "profile"), (emr_path, "EMR")):
                    try:
                        path.unlink()
                    except FileNotFoundError:
                        continue
                    except OSError as exc:  # pragma: no cover - filesystem issues shouldn't abort deletion
                        self.logger.warning(
                            "Failed to delete %s file for user '%s': %s", description, username, exc
                        )

        return deleted
