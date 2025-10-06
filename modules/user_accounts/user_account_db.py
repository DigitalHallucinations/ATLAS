import hashlib
import hmac
import json
import os
import shutil
import sqlite3
import tempfile
import threading
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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


class InvalidCurrentPasswordError(RuntimeError):
    """Raised when the supplied current password does not match the stored hash."""


_DUPLICATE_USER_MESSAGE = "A user with the same username or email already exists."


class UserAccountDatabase:
    def __init__(self, db_name: str = "User.db", base_dir: Optional[str] = None):
        self.logger = setup_logger(__name__)
        self.config_manager = ConfigManager()

        base_directory = self._determine_base_directory(base_dir)
        self.user_profiles_dir = base_directory / 'user_profiles'
        self.user_profiles_dir.mkdir(parents=True, exist_ok=True)

        self._profile_template_path = Path(__file__).resolve().parent / 'user_template'

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
            DOB TEXT,
            last_login TEXT
        );
        """
        with self._lock:
            cursor = self.conn.cursor()
            try:
                cursor.execute(query)
                self.conn.commit()
                self._ensure_unique_constraints()
                self._ensure_last_login_column()
            finally:
                cursor.close()

    def _ensure_last_login_column(self) -> None:
        """Add the ``last_login`` column for databases created before it existed."""

        with self._lock:
            cursor = self.conn.cursor()
            try:
                cursor.execute("PRAGMA table_info(user_accounts)")
                columns = {row[1] for row in cursor.fetchall()}
                if "last_login" in columns:
                    return

                cursor.execute("ALTER TABLE user_accounts ADD COLUMN last_login TEXT")
                self.conn.commit()
            except sqlite3.DatabaseError:
                self.conn.rollback()
                raise
            finally:
                cursor.close()

    def _ensure_unique_constraints(self) -> None:
        """Ensure unique constraints exist for usernames and e-mail addresses."""

        with self._lock:
            cursor = self.conn.cursor()
            try:
                self._normalise_existing_emails(cursor)
                self._ensure_username_index(cursor)
                self._ensure_email_index(cursor)
                self.conn.commit()
            except sqlite3.DatabaseError:
                self.conn.rollback()
                raise
            finally:
                cursor.close()

    def _normalise_existing_emails(self, cursor: sqlite3.Cursor) -> None:
        """Normalise stored e-mail addresses and detect duplicates ignoring case."""

        cursor.execute(
            """
            SELECT LOWER(TRIM(email)) AS normalised_email, GROUP_CONCAT(username)
            FROM user_accounts
            WHERE email IS NOT NULL
            GROUP BY normalised_email
            HAVING COUNT(*) > 1
            """
        )
        duplicates = cursor.fetchall()
        if duplicates:
            details = ", ".join(
                f"{value}: {usernames}" for value, usernames in duplicates if value
            )
            message = (
                "Existing user account records share the same email address when case is ignored. "
                "Please resolve duplicate entries before restarting the application."
            )
            if details:
                message = f"{message} ({details})"
            self.logger.error(message)
            raise RuntimeError(message)

        cursor.execute(
            """
            UPDATE user_accounts
            SET email = LOWER(TRIM(email))
            WHERE email IS NOT NULL AND email <> LOWER(TRIM(email))
            """
        )

    def _ensure_username_index(self, cursor: sqlite3.Cursor) -> None:
        create_statement = (
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_user_accounts_username"
            " ON user_accounts (username)"
        )
        self._create_unique_index(cursor, "idx_user_accounts_username", create_statement)

    def _ensure_email_index(self, cursor: sqlite3.Cursor) -> None:
        index_name = "idx_user_accounts_email"
        index_exists, needs_rebuild = self._email_index_status(cursor, index_name)
        if needs_rebuild:
            cursor.execute(f"DROP INDEX IF EXISTS {index_name}")
            index_exists = False

        if index_exists:
            return

        create_statement = (
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_user_accounts_email"
            " ON user_accounts (email COLLATE NOCASE)"
        )
        self._create_unique_index(cursor, index_name, create_statement)

    def _email_index_status(
        self, cursor: sqlite3.Cursor, index_name: str
    ) -> Tuple[bool, bool]:
        cursor.execute("PRAGMA index_list('user_accounts')")
        for _, existing_name, *_rest in cursor.fetchall():
            if existing_name != index_name:
                continue
            cursor.execute(f"PRAGMA index_xinfo('{index_name}')")
            info_rows = cursor.fetchall()
            collations = [
                (row[4] or "").upper() for row in info_rows if row[1] != -1
            ]
            if not collations:
                return True, True
            needs_rebuild = any(collation != "NOCASE" for collation in collations)
            return True, needs_rebuild
        return False, False

    def _create_unique_index(
        self, cursor: sqlite3.Cursor, index_name: str, create_statement: str
    ) -> None:
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

    @staticmethod
    def _canonicalize_email(email: Optional[str]) -> Optional[str]:
        if email is None:
            return None

        if not isinstance(email, str):
            email = str(email)

        cleaned = email.strip()
        return cleaned.lower()

    def add_user(self, username, password, email, name, dob):
        hashed_password = self._hash_password(password)
        canonical_email = self._canonicalize_email(email)
        query = """
        INSERT INTO user_accounts (username, password, email, name, DOB, last_login)
        VALUES (?, ?, ?, ?, ?, ?)
        """
        profile_data = None
        with self._lock:
            cursor = self.conn.cursor()
            try:
                cursor.execute(
                    query,
                    (
                        username,
                        hashed_password,
                        canonical_email,
                        name if name not in ("", None) else None,
                        dob if dob not in ("", None) else None,
                        None,
                    ),
                )
                self.conn.commit()
                profile_data = {
                    'username': username,
                    'email': canonical_email,
                    'name': name if name not in ("", None) else None,
                    'dob': dob if dob not in ("", None) else None,
                }
            except sqlite3.IntegrityError as exc:
                self.conn.rollback()
                raise DuplicateUserError(_DUPLICATE_USER_MESSAGE) from exc
            finally:
                cursor.close()

        if profile_data is not None:
            self._write_user_profile_files(**profile_data, ensure_emr=True)

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

    def search_users(self, query_text: Optional[str] = None) -> List[Sequence[Any]]:
        """Return user rows filtered by a case-insensitive search term."""

        if query_text is None:
            query_text = ""

        search_term = str(query_text).strip().lower()

        if not search_term:
            rows = self.get_all_users() or []
            return list(rows)

        like_term = f"%{search_term}%"
        query = (
            "SELECT * FROM user_accounts "
            "WHERE LOWER(username) LIKE ? OR LOWER(email) LIKE ? OR LOWER(IFNULL(name, '')) LIKE ?"
        )

        with self._lock:
            cursor = self.conn.cursor()
            try:
                cursor.execute(query, (like_term, like_term, like_term))
                rows = cursor.fetchall()
            finally:
                cursor.close()

        return list(rows)

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

    def get_user_details(self, username: str) -> Optional[Dict[str, Any]]:
        """Return a mapping of stored data for the given username."""

        row = self.get_user(username)
        if not row:
            return None

        return {
            "id": row[0],
            "username": row[1],
            "email": row[3],
            "name": row[4],
            "dob": row[5],
            "last_login": row[6] if len(row) > 6 else None,
        }

    def close_connection(self):
        """Close the connection to the SQLite database."""
        self.logger.info("Closing UA database connection.")
        with self._lock:
            try:
                self.conn.close()
            except sqlite3.Error as e:
                self.logger.info(f"Error closing connection: {e}")
                raise

    def update_user(
        self,
        username,
        password=None,
        *,
        current_password=None,
        email=None,
        name=None,
        dob=None,
    ):
        profile_data = None
        profile_fields_modified = False
        with self._lock:
            cursor = self.conn.cursor()
            try:
                def _execute_update(query: str, parameters):
                    try:
                        cursor.execute(query, parameters)
                    except sqlite3.IntegrityError as exc:
                        self.conn.rollback()
                        raise DuplicateUserError(_DUPLICATE_USER_MESSAGE) from exc

                if password is not None:
                    if current_password is None:
                        self.conn.rollback()
                        raise InvalidCurrentPasswordError("Current password is required.")

                    if not self.verify_user_password(username, current_password):
                        self.conn.rollback()
                        raise InvalidCurrentPasswordError("Current password is incorrect.")

                    hashed_password = self._hash_password(password)
                    query = "UPDATE user_accounts SET password = ? WHERE username = ?"
                    _execute_update(query, (hashed_password, username))
                if email is not None:
                    canonical_email = self._canonicalize_email(email)
                    query = "UPDATE user_accounts SET email = ? WHERE username = ?"
                    _execute_update(query, (canonical_email, username))
                    profile_fields_modified = True
                if name is not None:
                    query = "UPDATE user_accounts SET name = ? WHERE username = ?"
                    _execute_update(query, (None if name == "" else name, username))
                    profile_fields_modified = True
                if dob is not None:
                    query = "UPDATE user_accounts SET DOB = ? WHERE username = ?"
                    _execute_update(query, (None if dob == "" else dob, username))
                    profile_fields_modified = True
                self.conn.commit()

                if profile_fields_modified:
                    cursor.execute(
                        "SELECT username, email, name, DOB FROM user_accounts WHERE username = ?",
                        (username,),
                    )
                    row = cursor.fetchone()
                    if row:
                        profile_data = {
                            'username': row[0],
                            'email': row[1],
                            'name': row[2] if row[2] not in ("", None) else None,
                            'dob': row[3] if row[3] not in ("", None) else None,
                        }
            finally:
                cursor.close()

        if profile_data is not None:
            self._write_user_profile_files(**profile_data, ensure_emr=False)

    def verify_user_password(self, username, candidate_password):
        user_record = self.get_user(username)
        if not user_record:
            return False

        stored_hash = user_record[2]
        return self._verify_password(stored_hash, candidate_password)

    def update_last_login(self, username: str, timestamp: str) -> bool:
        """Persist the timestamp of the user's most recent successful login."""

        with self._lock:
            cursor = self.conn.cursor()
            try:
                cursor.execute(
                    "UPDATE user_accounts SET last_login = ? WHERE username = ?",
                    (timestamp, username),
                )
                updated = cursor.rowcount > 0
                self.conn.commit()
                return updated
            except sqlite3.DatabaseError:
                self.conn.rollback()
                raise
            finally:
                cursor.close()

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

    def _load_user_profile_template(self) -> Dict:
        try:
            with self._profile_template_path.open('r', encoding='utf-8') as template_file:
                return json.load(template_file)
        except (OSError, json.JSONDecodeError) as exc:
            self.logger.error(
                "Failed to load user profile template from %s: %s",
                self._profile_template_path,
                exc,
            )
            return {}

    def _write_user_profile_files(
        self,
        *,
        username: str,
        email: Optional[str],
        name: Optional[str],
        dob: Optional[str],
        ensure_emr: bool,
    ) -> None:
        profile_path = self.user_profiles_dir / f"{username}.json"

        try:
            with profile_path.open('r', encoding='utf-8') as profile_file:
                loaded_contents = json.load(profile_file)
            if isinstance(loaded_contents, dict):
                profile_contents: Dict[str, Any] = loaded_contents
            else:  # pragma: no cover - defensive guard for unexpected structures
                raise ValueError("Profile JSON must contain an object")
        except (OSError, json.JSONDecodeError, ValueError):
            profile_contents = self._load_user_profile_template()

        profile_contents['Username'] = username
        profile_contents['Full Name'] = name or ''
        profile_contents['DOB'] = dob or ''
        profile_contents['Email'] = email or ''

        age: Any = ''
        if dob:
            try:
                parsed_dob = date.fromisoformat(dob)
                today = date.today()
                computed_age = today.year - parsed_dob.year - (
                    (today.month, today.day) < (parsed_dob.month, parsed_dob.day)
                )
                age = max(computed_age, 0)
            except ValueError:
                age = ''

        profile_contents['Age'] = age

        tmp_path: Optional[Path] = None

        try:
            with tempfile.NamedTemporaryFile(
                'w', encoding='utf-8', dir=str(profile_path.parent), delete=False
            ) as tmp_file:
                json.dump(profile_contents, tmp_file, indent=4)
                tmp_file.write('\n')
                tmp_file.flush()
                os.fsync(tmp_file.fileno())
                tmp_path = Path(tmp_file.name)
            tmp_path.replace(profile_path)
        except OSError as exc:
            self.logger.error("Failed to write profile JSON for user '%s': %s", username, exc)
            if tmp_path is not None:
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
        else:
            if ensure_emr:
                emr_path = self.user_profiles_dir / f"{username}_emr.txt"
                try:
                    if not emr_path.exists():
                        emr_path.touch()
                except OSError as exc:
                    self.logger.error("Failed to create EMR file for user '%s': %s", username, exc)
