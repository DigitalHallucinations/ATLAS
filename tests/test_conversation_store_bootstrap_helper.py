from __future__ import annotations

import importlib.util
from pathlib import Path
import types

import pytest
from sqlalchemy.engine.url import make_url as sa_make_url

_BOOTSTRAP_PATH = Path(__file__).resolve().parents[1] / "modules" / "conversation_store" / "bootstrap.py"
_BOOTSTRAP_SPEC = importlib.util.spec_from_file_location(
    "conversation_store_bootstrap", _BOOTSTRAP_PATH
)
if _BOOTSTRAP_SPEC is None or _BOOTSTRAP_SPEC.loader is None:  # pragma: no cover - defensive guard
    raise RuntimeError("Unable to load conversation store bootstrap helper")

bootstrap_module = importlib.util.module_from_spec(_BOOTSTRAP_SPEC)
_BOOTSTRAP_SPEC.loader.exec_module(bootstrap_module)

bootstrap_conversation_store = bootstrap_module.bootstrap_conversation_store
BootstrapError = bootstrap_module.BootstrapError


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

    result = bootstrap_conversation_store(
        base_dsn,
        which=lambda *_: "/usr/bin/psql",
        run=lambda command, check: commands.append(command),
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
