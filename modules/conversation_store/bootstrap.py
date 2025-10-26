"""Helpers for bootstrapping the PostgreSQL conversation store."""

from __future__ import annotations

import platform
import secrets
import shutil
import subprocess
import time
from typing import Callable, Iterable

from sqlalchemy.engine.url import make_url

try:  # pragma: no cover - import guards for optional dependency environments
    from psycopg import OperationalError, connect, sql
except Exception as exc:  # pragma: no cover - fail fast when psycopg is unavailable
    raise RuntimeError(
        "psycopg is required to bootstrap the conversation store"
    ) from exc


class BootstrapError(RuntimeError):
    """Raised when the PostgreSQL bootstrap process fails."""


def _install_postgres_client(
    *,
    run: Callable[..., subprocess.CompletedProcess],
    which: Callable[[str], str | None],
    platform_system: Callable[[], str],
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

        for command in commands:
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
    connector: Callable[[str], object] = connect,
) -> str:
    """Bootstrap the configured PostgreSQL conversation store."""

    url = make_url(dsn)
    if url.get_dialect().name != "postgresql":
        raise BootstrapError(
            "Conversation store DSN must use the 'postgresql' dialect."
        )

    _install_postgres_client(
        run=run,
        which=which,
        platform_system=platform_system,
    )

    database = url.database
    username = url.username
    password = url.password

    maintenance_conn = None
    last_error: Exception | None = None

    candidate_databases: list[str] = []
    if database:
        candidate_databases.append(database)
    candidate_databases.extend(["postgres", "template1"])

    for candidate in candidate_databases:
        try:
            candidate_url = url.set(database=candidate)
            maintenance_conn = connector(
                candidate_url.render_as_string(hide_password=False)
            )
        except OperationalError as exc:  # pragma: no cover - error handling path
            last_error = exc
            continue
        else:
            break

    if maintenance_conn is None:
        host = url.host or "localhost"
        port = url.port or 5432
        raise BootstrapError(
            f"Unable to connect to PostgreSQL server at {host}:{port}: {last_error}"
        )

    ensured_username = username
    ensured_password = password

    try:
        maintenance_conn.autocommit = True  # type: ignore[attr-defined]
        with maintenance_conn.cursor() as cursor:  # type: ignore[attr-defined]
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
        maintenance_conn.close()  # type: ignore[attr-defined]

    if ensured_username and ensured_password and ensured_password != password:
        url = url.set(password=ensured_password)

    connection_dsn = url.render_as_string(hide_password=False)

    try:
        verification_conn = connector(connection_dsn)
    except OperationalError as exc:
        raise BootstrapError(
            f"Unable to connect to the conversation store database '{database}': {exc}"
        ) from exc
    else:
        verification_conn.close()  # type: ignore[attr-defined]

    return connection_dsn


__all__ = ["BootstrapError", "bootstrap_conversation_store"]
