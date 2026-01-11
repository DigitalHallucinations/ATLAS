from __future__ import annotations

import importlib.util
from pathlib import Path
import types
from urllib.parse import urlparse

import pytest
from sqlalchemy.engine.url import make_url as sa_make_url

_BOOTSTRAP_PATH = Path(__file__).resolve().parents[2] / "modules" / "conversation_store" / "bootstrap.py"
_BOOTSTRAP_SPEC = importlib.util.spec_from_file_location(
    "conversation_store_bootstrap", _BOOTSTRAP_PATH
)
if _BOOTSTRAP_SPEC is None or _BOOTSTRAP_SPEC.loader is None:  # pragma: no cover - defensive guard
    raise RuntimeError("Unable to load conversation store bootstrap helper")

bootstrap_module = importlib.util.module_from_spec(_BOOTSTRAP_SPEC)
_BOOTSTRAP_SPEC.loader.exec_module(bootstrap_module)

bootstrap_conversation_store = bootstrap_module.bootstrap_conversation_store
BootstrapError = bootstrap_module.BootstrapError


class _FormattedSQL(str):
    pass


class _Identifier:
    def __init__(self, name: str):
        self._name = name

    def __str__(self) -> str:
        return f'"{self._name}"'


def _compose_url(*, drivername, username, password, host, port, database):
    auth = ""
    if username:
        auth = username
        if password is not None:
            auth += f":{password}"
        auth += "@"
    host_part = host or ""
    port_part = f":{port}" if port is not None else ""
    path_part = f"/{database}" if database else ""
    return f"{drivername}://{auth}{host_part}{port_part}{path_part}"


class _SQL:
    def __init__(self, template: str):
        self._template = template

    def format(self, *identifiers):
        formatted = self._template
        for identifier in identifiers:
            formatted = formatted.replace("{}", str(identifier), 1)
        return _FormattedSQL(formatted)


class _OperationalError(Exception):
    pass


class _URL:
    def __init__(self, dsn: str):
        self._dsn = dsn
        parsed = urlparse(dsn)
        self.drivername = parsed.scheme
        self.username = parsed.username
        self.password = parsed.password
        self.host = parsed.hostname
        self.port = parsed.port
        self.database = parsed.path.lstrip("/") if parsed.path else None

    def get_dialect(self):
        return types.SimpleNamespace(name=self.drivername.split("+", 1)[0])

    def get_backend_name(self):
        return self.drivername.split("+", 1)[0]

    def set(self, **updates):
        params = {
            "drivername": updates.get("drivername", self.drivername),
            "username": updates.get("username", self.username),
            "password": updates.get("password", self.password),
            "host": updates.get("host", self.host),
            "port": updates.get("port", self.port),
            "database": updates.get("database", self.database),
        }
        return _URL(_compose_url(**params))

    def render_as_string(self, hide_password: bool = True):
        password = None if hide_password else self.password
        return _compose_url(
            drivername=self.drivername,
            username=self.username,
            password=password,
            host=self.host,
            port=self.port,
            database=self.database,
        )


def _ensure_psycopg_stub(**_kwargs):
    bootstrap_module.OperationalError = _OperationalError
    bootstrap_module.sql = types.SimpleNamespace(SQL=_SQL, Identifier=_Identifier)
    if getattr(bootstrap_module, "connect", None) is None:
        bootstrap_module.connect = lambda *_args, **_kwargs: None


bootstrap_module._ensure_psycopg_loaded = _ensure_psycopg_stub  # type: ignore[attr-defined]
bootstrap_module.OperationalError = _OperationalError
bootstrap_module.sql = types.SimpleNamespace(SQL=_SQL, Identifier=_Identifier)
bootstrap_module.make_url = lambda dsn: _URL(dsn)
sa_make_url = bootstrap_module.make_url  # type: ignore[assignment]


class RecordingCursor:
    def __init__(self, executed, responses):
        self._executed = executed
        self._responses = responses
        self._last_key = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql_query, params=None):
        text = str(sql_query)
        key = (text, tuple(params or ()))
        self._executed.append(key)
        self._last_key = key
        response = self._responses.get(key)
        if isinstance(response, Exception):
            raise response

    def fetchone(self):
        if self._last_key is None:
            return None
        return self._responses.get(self._last_key)


class RecordingConnection:
    def __init__(self, responses=None):
        self._responses = responses or {}
        self.executed = []
        self.autocommit = False
        self.closed = False

    def cursor(self):
        return RecordingCursor(self.executed, self._responses)

    def close(self):
        self.closed = True


def make_connector(*connections):
    iterator = iter(connections)

    def _connector(dsn):
        try:
            connection = next(iterator)
        except StopIteration:  # pragma: no cover - defensive guard
            raise AssertionError("Unexpected connection attempt")
        return connection

    return _connector


@pytest.fixture
def base_dsn():
    return "postgresql+psycopg://atlas:secret@localhost:5432/atlas"


def test_bootstrap_installs_postgres_client_when_psql_missing(base_dsn, monkeypatch):
    which_calls = []

    def fake_which(command):
        which_calls.append(command)
        return None if len(which_calls) == 1 else "/usr/bin/psql"

    commands = []
    readiness_results = iter([1, 0])

    def fake_run(command, check):
        commands.append((command, check))
        if command[0] == "pg_isready":
            return types.SimpleNamespace(returncode=next(readiness_results))
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(bootstrap_module.time, "sleep", lambda _: None)

    maintenance = RecordingConnection(
        responses={
            ("SELECT 1 FROM pg_roles WHERE rolname = %s", ("atlas",)): (1,),
            ("SELECT 1 FROM pg_database WHERE datname = %s", ("atlas",)): (1,),
        }
    )
    verification = RecordingConnection()
    connector = make_connector(maintenance, verification)

    result = bootstrap_conversation_store(
        base_dsn,
        which=fake_which,
        run=fake_run,
        platform_system=lambda: "Linux",
        connector=connector,
    )

    result_url = sa_make_url(result)
    expected_url = sa_make_url(base_dsn)
    assert result_url.username == expected_url.username
    assert result_url.password == expected_url.password
    assert result_url.database == expected_url.database
    assert result_url.host == expected_url.host
    assert result_url.port == expected_url.port
    assert commands == [
        (["apt-get", "update"], True),
        (["apt-get", "install", "-y", "postgresql", "postgresql-client"], True),
        (["pg_isready", "-q"], False),
        (["systemctl", "start", "postgresql"], True),
        (["pg_isready", "-q"], False),
    ]
    assert which_calls == ["psql", "psql"]
    assert maintenance.autocommit is True
    assert maintenance.closed is True
    assert verification.closed is True


def test_bootstrap_creates_missing_database(base_dsn):
    maintenance = RecordingConnection(
        responses={
            ("SELECT 1 FROM pg_roles WHERE rolname = %s", ("atlas",)): (1,),
            ("SELECT 1 FROM pg_database WHERE datname = %s", ("atlas",)): None,
        }
    )
    verification = RecordingConnection()
    connector = make_connector(maintenance, verification)

    result = bootstrap_conversation_store(
        base_dsn,
        which=lambda *_: "/usr/bin/psql",
        run=lambda *args, **kwargs: types.SimpleNamespace(returncode=0),
        platform_system=lambda: "Linux",
        connector=connector,
    )

    result_url = sa_make_url(result)
    expected_url = sa_make_url(base_dsn)
    assert result_url.username == expected_url.username
    assert result_url.password == expected_url.password
    assert result_url.database == expected_url.database
    assert result_url.host == expected_url.host
    assert result_url.port == expected_url.port
    assert any("CREATE DATABASE" in stmt for stmt, _ in maintenance.executed)
    assert maintenance.closed is True


def test_bootstrap_noop_when_already_provisioned(base_dsn):
    maintenance = RecordingConnection(
        responses={
            ("SELECT 1 FROM pg_roles WHERE rolname = %s", ("atlas",)): (1,),
            ("SELECT 1 FROM pg_database WHERE datname = %s", ("atlas",)): (1,),
        }
    )
    verification = RecordingConnection()
    connector = make_connector(maintenance, verification)
    commands = []

    def recording_run(command, check):
        if command[0] != "pg_isready":
            commands.append(command)
        return types.SimpleNamespace(returncode=0)

    result = bootstrap_conversation_store(
        base_dsn,
        which=lambda *_: "/usr/bin/psql",
        run=recording_run,
        platform_system=lambda: "Linux",
        connector=connector,
    )

    result_url = sa_make_url(result)
    expected_url = sa_make_url(base_dsn)
    assert result_url.username == expected_url.username
    assert result_url.password == expected_url.password
    assert result_url.database == expected_url.database
    assert result_url.host == expected_url.host
    assert result_url.port == expected_url.port
    assert not any("CREATE" in stmt for stmt, _ in maintenance.executed)
    assert commands == []


def test_bootstrap_raises_when_postgres_service_cannot_start(base_dsn, monkeypatch):
    which_calls = []

    def fake_which(command):
        which_calls.append(command)
        return None if len(which_calls) == 1 else "/usr/bin/psql"

    commands = []

    def fake_run(command, check):
        commands.append((command, check))
        if command[0] == "pg_isready":
            return types.SimpleNamespace(returncode=1)
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(bootstrap_module.time, "sleep", lambda _: None)

    maintenance = RecordingConnection(
        responses={
            ("SELECT 1 FROM pg_roles WHERE rolname = %s", ("atlas",)): (1,),
            ("SELECT 1 FROM pg_database WHERE datname = %s", ("atlas",)): (1,),
        }
    )
    connector = make_connector(maintenance)

    with pytest.raises(BootstrapError):
        bootstrap_conversation_store(
            base_dsn,
            which=fake_which,
            run=fake_run,
            platform_system=lambda: "Linux",
            connector=connector,
        )

    assert commands[:2] == [
        (["apt-get", "update"], True),
        (["apt-get", "install", "-y", "postgresql", "postgresql-client"], True),
    ]
    assert any(cmd[0][0] == "systemctl" for cmd in commands)
    assert which_calls == ["psql", "psql"]


@pytest.mark.parametrize(
    "dsn",
    [
        "postgresql+psycopg://atlas:secret@localhost:5432/atlas",
        "postgresql+asyncpg://atlas:secret@localhost:5432/atlas",
    ],
)
def test_bootstrap_sanitises_driver_specific_urls(dsn):
    maintenance = RecordingConnection(
        responses={
            ("SELECT 1 FROM pg_roles WHERE rolname = %s", ("atlas",)): (1,),
            ("SELECT 1 FROM pg_database WHERE datname = %s", ("atlas",)): (1,),
        }
    )
    verification = RecordingConnection()
    connections = iter([maintenance, verification])
    connector_calls = []

    def recording_connector(conninfo):
        connector_calls.append(conninfo)
        try:
            return next(connections)
        except StopIteration:  # pragma: no cover - defensive guard
            raise AssertionError("Unexpected connection attempt")

    result = bootstrap_conversation_store(
        dsn,
        which=lambda *_: "/usr/bin/psql",
        run=lambda *args, **kwargs: types.SimpleNamespace(returncode=0),
        platform_system=lambda: "Linux",
        connector=recording_connector,
    )

    parsed_url = sa_make_url(dsn)
    expected_conninfo = (
        parsed_url
        .set(drivername=parsed_url.get_backend_name())
        .render_as_string(hide_password=False)
    )
    assert result == dsn
    assert connector_calls == [expected_conninfo, expected_conninfo]
    assert maintenance.closed is True
    assert verification.closed is True


def test_bootstrap_resets_password_with_privileged_credentials_on_auth_failure(base_dsn):
    op_error_cls = getattr(bootstrap_module, "OperationalError", None)
    if op_error_cls is None:  # pragma: no cover - defensive guard
        op_error_cls = type("OperationalError", (Exception,), {})
        bootstrap_module.OperationalError = op_error_cls

    maintenance = RecordingConnection(
        responses={
            ("SELECT 1 FROM pg_roles WHERE rolname = %s", ("atlas",)): (1,),
            ("SELECT 1 FROM pg_database WHERE datname = %s", ("atlas",)): (1,),
        }
    )
    verification = RecordingConnection()

    connection_attempts = []

    def connector(conninfo):
        connection_attempts.append(conninfo)
        attempt = len(connection_attempts)
        if attempt <= 3:
            raise op_error_cls('password authentication failed for user "atlas"')
        if attempt == 4:
            return maintenance
        if attempt == 5:
            return verification
        raise AssertionError("Unexpected connection attempt")

    privileged_credentials = ("postgres", "supersecret")

    result = bootstrap_conversation_store(
        base_dsn,
        which=lambda *_: "/usr/bin/psql",
        run=lambda *args, **kwargs: types.SimpleNamespace(returncode=0),
        platform_system=lambda: "Linux",
        connector=connector,
        privileged_credentials=privileged_credentials,
    )

    assert result == base_dsn
    assert any(
        statement == 'ALTER ROLE "atlas" WITH PASSWORD %s' and params == ("secret",)
        for statement, params in maintenance.executed
    )
    assert maintenance.autocommit is True
    assert maintenance.closed is True
    assert verification.closed is True
    assert len(connection_attempts) == 5


def test_bootstrap_updates_existing_role_password_with_privileged_credentials(base_dsn):
    updated_password = "refreshed-secret"
    updated_dsn = base_dsn.replace(":secret@", f":{updated_password}@")

    op_error_cls = getattr(bootstrap_module, "OperationalError", None)
    if op_error_cls is None:  # pragma: no cover - defensive guard
        op_error_cls = type("OperationalError", (Exception,), {})
        bootstrap_module.OperationalError = op_error_cls

    maintenance = RecordingConnection(
        responses={
            ("SELECT 1 FROM pg_roles WHERE rolname = %s", ("atlas",)): (1,),
            ("SELECT 1 FROM pg_database WHERE datname = %s", ("atlas",)): (1,),
        }
    )
    verification = RecordingConnection()

    connection_attempts = []

    def connector(conninfo):
        connection_attempts.append(conninfo)
        if len(connection_attempts) == 1:
            raise op_error_cls('role "atlas" does not exist')
        if len(connection_attempts) == 2:
            return maintenance
        if len(connection_attempts) == 3:
            return verification
        raise AssertionError("Unexpected connection attempt")

    privileged_credentials = ("postgres", "admin")

    result = bootstrap_conversation_store(
        updated_dsn,
        which=lambda *_: "/usr/bin/psql",
        run=lambda *args, **kwargs: types.SimpleNamespace(returncode=0),
        platform_system=lambda: "Linux",
        connector=connector,
        privileged_credentials=privileged_credentials,
    )

    assert f":{updated_password}@" in result
    assert any(
        statement == 'ALTER ROLE "atlas" WITH PASSWORD %s' and params == (updated_password,)
        for statement, params in maintenance.executed
    )
    assert verification.closed is True
    assert len(connection_attempts) == 3


def test_bootstrap_raises_when_password_update_lacks_privilege(base_dsn):
    desired_password = "new-password"
    desired_dsn = base_dsn.replace(":secret@", f":{desired_password}@")

    op_error_cls = getattr(bootstrap_module, "OperationalError", None)
    if op_error_cls is None:  # pragma: no cover - defensive guard
        op_error_cls = type("OperationalError", (Exception,), {})
        bootstrap_module.OperationalError = op_error_cls

    maintenance = RecordingConnection(
        responses={
            ("SELECT 1 FROM pg_roles WHERE rolname = %s", ("atlas",)): (1,),
            ("SELECT 1 FROM pg_database WHERE datname = %s", ("atlas",)): (1,),
            ('ALTER ROLE "atlas" WITH PASSWORD %s', (desired_password,)): op_error_cls(
                "permission denied to alter role"
            ),
        }
    )
    verification = RecordingConnection()
    connector = make_connector(maintenance, verification)

    with pytest.raises(BootstrapError) as excinfo:
        bootstrap_conversation_store(
            desired_dsn,
            which=lambda *_: "/usr/bin/psql",
            run=lambda *args, **kwargs: types.SimpleNamespace(returncode=0),
            platform_system=lambda: "Linux",
            connector=connector,
        )

    assert "privileged credentials" in str(excinfo.value)
    assert maintenance.closed is True
