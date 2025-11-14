"""Helpers for bootstrapping SQL and document conversation store backends."""

from __future__ import annotations
import importlib
import os
import platform
import secrets
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Iterable
from urllib.parse import urlparse

from sqlalchemy.engine.url import URL, make_url

from .mongo_repository import MongoConversationStoreRepository


class BootstrapError(RuntimeError):
    """Raised when the PostgreSQL bootstrap process fails."""


connect = None
OperationalError = None
sql = None


_MISSING = object()


def _is_psycopg_import_error(exc: Exception) -> bool:
    """Return ``True`` when *exc* stems from psycopg import failures."""

    if isinstance(exc, ImportError):
        return True
    return exc.__class__.__name__ == "ProgrammingError" and "psycopg" in exc.__class__.__module__


def _compose_psycopg_guidance(system: str, apt_available: bool) -> str:
    """Build remediation guidance for missing libpq installations."""

    instructions: list[str] = []
    system_lower = system.lower()

    if system_lower == "linux":
        if apt_available:
            instructions.append(
                "Linux: ensure the libpq development headers are installed (e.g. `apt-get install -y libpq-dev`)."
            )
        else:
            instructions.append(
                "Linux: install the libpq development headers using your distribution's package manager (for Debian/Ubuntu: `apt-get install -y libpq-dev`)."
            )
    elif system_lower == "darwin":
        instructions.append(
            "macOS: install libpq via Homebrew: `brew install libpq` and ensure it is available on your PATH."
        )
    elif system_lower.startswith("win"):
        instructions.append(
            "Windows: install the PostgreSQL client libraries from https://www.postgresql.org/download/windows/ and ensure `libpq.dll` is on your PATH."
        )
    else:
        instructions.append(
            f"{system}: install the PostgreSQL client libraries for your platform and retry."
        )

    if system_lower != "darwin":
        instructions.append(
            "macOS users can install libpq via Homebrew: `brew install libpq`."
        )
    if not system_lower.startswith("win"):
        instructions.append(
            "Windows users should install the PostgreSQL client libraries from https://www.postgresql.org/download/windows/ and ensure `libpq.dll` is on the PATH."
        )

    return " ".join(instructions)


def _raise_psycopg_bootstrap_error(
    *,
    exc: Exception,
    commands: list[list[str]],
    system: str,
    apt_available: bool,
) -> None:
    """Raise a :class:`BootstrapError` with actionable remediation guidance."""

    commands_text = ", ".join(" ".join(command) for command in commands) if commands else "none"
    details = [
        "Unable to import psycopg because libpq could not be loaded even after attempting automatic installation.",
        f"Commands tried: {commands_text}.",
        f"Last error: {exc}.",
    ]

    if system.lower() == "linux" and not apt_available:
        details.append("Automatic apt-based installation was skipped because 'apt-get' was not found.")

    details.append(_compose_psycopg_guidance(system, apt_available))

    raise BootstrapError(" ".join(details)) from exc


def _recover_psycopg_import(
    *,
    exc: Exception,
    attempt_import: Callable[[], tuple[object, object]],
    run: Callable[..., subprocess.CompletedProcess],
    which: Callable[[str], str | None],
    system: str,
    executable: str,
) -> tuple[object, object]:
    """Attempt to install libpq dependencies before retrying the import."""

    if "libpq" not in str(exc).lower():
        raise BootstrapError(f"Unable to import psycopg: {exc}") from exc

    commands: list[list[str]] = []
    pip_command = [executable, "-m", "pip", "install", "psycopg[binary]"]
    commands.append(pip_command)

    pip_failed = True
    try:
        result = run(pip_command, check=False)
    except Exception:
        pass
    else:
        pip_failed = getattr(result, "returncode", 1) != 0

    apt_available = which("apt-get") is not None
    if system == "linux" and pip_failed and apt_available:
        apt_command = ["apt-get", "install", "-y", "libpq-dev"]
        commands.append(apt_command)
        try:
            run(apt_command, check=False)
        except Exception:
            pass

    try:
        return attempt_import()
    except Exception as final_exc:
        if _is_psycopg_import_error(final_exc):
            _raise_psycopg_bootstrap_error(
                exc=final_exc,
                commands=commands,
                system=system,
                apt_available=apt_available,
            )
        raise


def _ensure_psycopg_loaded(
    *,
    run: Callable[..., subprocess.CompletedProcess],
    which: Callable[[str], str | None],
    platform_system: Callable[[], str],
    executable: str | None = None,
) -> None:
    """Lazily import psycopg, attempting automatic remediation when required."""

    global OperationalError, connect, sql

    if connect is not None and OperationalError is not None and sql is not None:
        return

    executable = executable or sys.executable
    system_name = platform_system()
    system = system_name.lower()

    def attempt_import() -> tuple[object, object]:
        psycopg_module = importlib.import_module("psycopg")
        sql_module = importlib.import_module("psycopg.sql")
        return psycopg_module, sql_module

    try:
        psycopg_module, sql_module = attempt_import()
    except Exception as exc:  # pragma: no cover - exercised via explicit tests
        if not _is_psycopg_import_error(exc):
            raise
        psycopg_module, sql_module = _recover_psycopg_import(
            exc=exc,
            attempt_import=attempt_import,
            run=run,
            which=which,
            system=system,
            executable=executable,
        )

    connect = getattr(psycopg_module, "connect")
    OperationalError = getattr(psycopg_module, "OperationalError")
    sql = sql_module


def _render_psycopg_conninfo(url: URL) -> str:
    """Render a psycopg-compatible connection string for the given URL."""

    backend_name = url.get_backend_name()
    sanitized_url = url.set(drivername=backend_name)
    return sanitized_url.render_as_string(hide_password=False)


def _error_indicates_missing_role(exc: Exception) -> bool:
    message = str(exc).lower()
    return "role" in message and "does not exist" in message


def _error_indicates_bad_password(exc: Exception) -> bool:
    message = str(exc).lower()
    return "password" in message and "authentication failed" in message


def _attempt_candidate_connection(
    *,
    connector: Callable[[str], object],
    base_url: URL,
    candidate_databases: Iterable[str],
    username=_MISSING,
    password=_MISSING,
):
    last_error: Exception | None = None
    missing_role_error: Exception | None = None
    bad_password_error: Exception | None = None

    for candidate in candidate_databases:
        candidate_url = base_url.set(database=candidate)
        if username is not _MISSING:
            candidate_url = candidate_url.set(username=username)
        if password is not _MISSING:
            candidate_url = candidate_url.set(password=password)
        try:
            connection = connector(_render_psycopg_conninfo(candidate_url))
        except OperationalError as exc:  # type: ignore[misc]
            last_error = exc
            if _error_indicates_missing_role(exc):
                missing_role_error = exc
            if _error_indicates_bad_password(exc):
                bad_password_error = exc
            continue
        else:
            return connection, None, None, None

    return None, last_error, missing_role_error, bad_password_error


def _install_postgres_client(
    *,
    run: Callable[..., subprocess.CompletedProcess],
    which: Callable[[str], str | None],
    platform_system: Callable[[], str],
    request_privileged_password: Callable[[], str | None] | None = None,
    geteuid: Callable[[], int] | None = None,
) -> None:
    """Ensure the PostgreSQL client utilities are available."""

    system = platform_system().lower()
    commands: Iterable[list[str]]
    start_commands: Iterable[list[str]] = []

    if which("psql") is None:
        if system == "linux":
            commands = [
                ["apt-get", "update"],
                ["apt-get", "install", "-y", "postgresql", "postgresql-client"],
            ]
        elif system == "darwin":
            # Homebrew installs typically expose versioned packages; 14 is a stable default.
            brew_package = "postgresql@14"
            commands = [
                ["brew", "update"],
                ["brew", "install", brew_package],
            ]
        elif system.startswith("win"):
            raise BootstrapError(
                "PostgreSQL client utilities are not installed. "
                "Please install the PostgreSQL server manually and ensure it is running. "
                "Visit https://www.postgresql.org/download/windows/ for installation instructions."
            )
        else:
            raise BootstrapError(
                "PostgreSQL client utilities are not installed and cannot be "
                f"automatically provisioned on platform '{system}'."
            )

        sudo_password: str | None = None
        if system == "linux":
            geteuid_func = geteuid or getattr(os, "geteuid", None)
            is_non_root = False
            if callable(geteuid_func):
                try:
                    is_non_root = bool(geteuid_func() != 0)
                except Exception:  # pragma: no cover - defensive guard
                    is_non_root = False
            else:
                is_non_root = False

            if is_non_root:
                if request_privileged_password is None:
                    raise BootstrapError(
                        "PostgreSQL client utilities are not installed and elevated privileges are required. "
                        "Retry with administrator credentials."
                    )
                sudo_password = request_privileged_password()
                if sudo_password is None:
                    raise BootstrapError(
                        "PostgreSQL client installation cancelled before acquiring administrator credentials."
                    )

        for command in commands:
            if system == "linux" and sudo_password is not None:
                sudo_command = ["sudo", "-S", *command]
                run(
                    sudo_command,
                    check=True,
                    input=f"{sudo_password}\n",
                    text=True,
                )
            else:
                run(command, check=True)

        if which("psql") is None:
            raise BootstrapError(
                "Attempted to install PostgreSQL client utilities but 'psql' is still unavailable."
            )
    if system == "linux":
        start_commands = [
            ["systemctl", "start", "postgresql"],
            ["service", "postgresql", "start"],
        ]
    elif system == "darwin":
        brew_package = "postgresql@14"
        start_commands = [["brew", "services", "start", brew_package]]
    elif system.startswith("win"):
        start_commands = []

    readiness_command = ["pg_isready", "-q"]

    def _is_ready() -> bool:
        try:
            result = run(readiness_command, check=False)
        except FileNotFoundError as exc:  # pragma: no cover - defensive guard
            raise BootstrapError(
                "Unable to locate 'pg_isready' after installing PostgreSQL."
            ) from exc
        return result.returncode == 0

    if _is_ready():
        return

    started = False
    for command in start_commands:
        try:
            run(command, check=True)
        except Exception:  # pragma: no cover - start command best effort
            continue
        else:
            started = True
            break

    if not start_commands:
        raise BootstrapError(
            "PostgreSQL server is not running. "
            "Please start the PostgreSQL service manually by following the platform-specific documentation."
        )

    if not started:
        raise BootstrapError(
            "Unable to start the PostgreSQL server automatically. "
            "Please start it manually and retry."
        )

    for _ in range(10):
        if _is_ready():
            return
        time.sleep(1)

    raise BootstrapError(
        "PostgreSQL server did not become ready after installation/start attempts."
    )


def _ensure_role(
    cursor,
    *,
    username: str | None,
    password: str | None,
) -> tuple[str | None, str | None]:
    """Ensure the target role exists, creating it if required."""

    if not username:
        return None, password

    cursor.execute(
        "SELECT 1 FROM pg_roles WHERE rolname = %s",
        (username,),
    )
    if cursor.fetchone():
        if password:
            try:
                cursor.execute(
                    sql.SQL("ALTER ROLE {} WITH PASSWORD %s").format(
                        sql.Identifier(username)
                    ),
                    (password,),
                )
            except OperationalError as exc:  # type: ignore[misc]
                message = str(exc).lower()
                if "permission" in message or "must be" in message:
                    raise BootstrapError(
                        "Unable to update the password for existing PostgreSQL role "
                        f"'{username}'. Provide privileged credentials capable of "
                        "modifying role passwords and retry."
                    ) from exc
                raise
        return username, password

    final_password = password or secrets.token_urlsafe(18)
    cursor.execute(
        sql.SQL("CREATE ROLE {} WITH LOGIN PASSWORD %s").format(
            sql.Identifier(username)
        ),
        (final_password,),
    )
    return username, final_password


def _ensure_database(cursor, *, database: str | None, owner: str | None) -> None:
    """Create the target database when it is missing."""

    if not database:
        return

    cursor.execute(
        "SELECT 1 FROM pg_database WHERE datname = %s",
        (database,),
    )
    if cursor.fetchone():
        return

    if owner:
        cursor.execute(
            sql.SQL("CREATE DATABASE {} OWNER {}").format(
                sql.Identifier(database), sql.Identifier(owner)
            )
        )
    else:
        cursor.execute(
            sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database))
        )


def bootstrap_conversation_store(
    dsn: str,
    *,
    which: Callable[[str], str | None] = shutil.which,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    platform_system: Callable[[], str] = platform.system,
    connector: Callable[[str], object] | None = None,
    request_privileged_password: Callable[[], str | None] | None = None,
    privileged_credentials: tuple[str | None, str | None] | None = None,
    geteuid: Callable[[], int] | None = None,
) -> str:
    """Bootstrap the configured conversation store."""

    normalized_dsn = (dsn or "").strip()
    if normalized_dsn.startswith(("mongodb://", "mongodb+srv://")):
        return _bootstrap_mongo_store(normalized_dsn)

    url = make_url(dsn)
    backend = url.get_backend_name()

    if backend == "sqlite":
        return _bootstrap_sqlite_store(url)

    if backend != "postgresql":
        return url.render_as_string(hide_password=False)

    _ensure_psycopg_loaded(
        run=run,
        which=which,
        platform_system=platform_system,
    )

    active_connector = connector or connect

    _install_postgres_client(
        run=run,
        which=which,
        platform_system=platform_system,
        request_privileged_password=request_privileged_password,
        geteuid=geteuid,
    )

    database = url.database
    username = url.username
    password = url.password

    candidate_databases: list[str] = []
    if database:
        candidate_databases.append(database)
    candidate_databases.extend(["postgres", "template1"])

    ensured_username = username
    ensured_password = password

    (
        maintenance_conn,
        last_error,
        missing_role_error,
        bad_password_error,
    ) = _attempt_candidate_connection(
        connector=active_connector,
        base_url=url,
        candidate_databases=candidate_databases,
    )

    def _provision(connection) -> None:
        nonlocal ensured_username, ensured_password
        try:
            connection.autocommit = True  # type: ignore[attr-defined]
            with connection.cursor() as cursor:  # type: ignore[attr-defined]
                ensured_username, ensured_password = _ensure_role(
                    cursor,
                    username=username,
                    password=password,
                )
                _ensure_database(
                    cursor,
                    database=database,
                    owner=ensured_username,
                )
        finally:
            connection.close()  # type: ignore[attr-defined]

    if maintenance_conn is not None:
        _provision(maintenance_conn)
    else:
        if missing_role_error is not None or bad_password_error is not None:
            if not privileged_credentials or not (privileged_credentials[0] or "").strip():
                if missing_role_error is not None:
                    raise BootstrapError(
                        "Unable to connect using the configured PostgreSQL user because the role does not exist. "
                        "Provide privileged credentials so the setup wizard can create the role and database automatically.",
                    ) from missing_role_error
                raise BootstrapError(
                    "Unable to connect using the configured PostgreSQL user because the password was rejected. "
                    "Provide privileged credentials so the setup wizard can reset the role password automatically.",
                ) from bad_password_error

            privileged_username, privileged_password = privileged_credentials
            (
                privileged_conn,
                privileged_error,
                _,
                _,
            ) = _attempt_candidate_connection(
                connector=active_connector,
                base_url=url,
                candidate_databases=candidate_databases,
                username=privileged_username,
                password=privileged_password,
            )
            if privileged_conn is None:
                raise BootstrapError(
                    "Unable to connect using the provided privileged PostgreSQL credentials: "
                    f"{privileged_error}",
                ) from privileged_error

            _provision(privileged_conn)
        else:
            host = url.host or "localhost"
            port = url.port or 5432
            raise BootstrapError(
                f"Unable to connect to PostgreSQL server at {host}:{port}: {last_error}"
            )

    if ensured_username and ensured_password and ensured_password != password:
        url = url.set(password=ensured_password)

    connection_dsn = url.render_as_string(hide_password=False)

    try:
        verification_conn = active_connector(_render_psycopg_conninfo(url))
    except OperationalError as exc:
        raise BootstrapError(
            f"Unable to connect to the conversation store database '{database}': {exc}"
        ) from exc
    else:
        verification_conn.close()  # type: ignore[attr-defined]

    return connection_dsn


def _bootstrap_sqlite_store(url: URL) -> str:
    """Ensure SQLite conversation stores have an initialized database file."""

    database = url.database or ""

    if database and database != ":memory:":
        db_path = Path(database)
        if not db_path.is_absolute():
            db_path = Path(os.getcwd()) / db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        if not db_path.exists():
            db_path.touch()
        url = url.set(database=str(db_path))

    return url.render_as_string(hide_password=False)


def _bootstrap_mongo_store(dsn: str) -> str:
    """Ensure MongoDB-backed stores provision the required indexes."""

    try:  # pragma: no cover - optional dependency
        from pymongo import MongoClient
    except Exception:
        return dsn

    parsed = urlparse(dsn)
    database_name = parsed.path.lstrip("/").split("?", 1)[0] or "atlas"

    client: MongoClient | None = None
    try:
        client = MongoClient(dsn)
    except Exception:
        return dsn

    try:
        database = client.get_database(database_name)
        repository = MongoConversationStoreRepository.from_database(database, client=client)
        try:
            repository.ensure_indexes()
        except RuntimeError:
            raise
        except Exception:
            pass
    finally:
        try:
            client.close()
        except Exception:
            pass

    return dsn


__all__ = ["BootstrapError", "bootstrap_conversation_store"]
