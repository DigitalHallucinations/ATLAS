from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

_BOOTSTRAP_PATH = Path(__file__).resolve().parents[3] / "modules" / "conversation_store" / "bootstrap.py"
_BOOTSTRAP_SPEC = importlib.util.spec_from_file_location(
    "modules.conversation_store.bootstrap",
    _BOOTSTRAP_PATH,
)
if _BOOTSTRAP_SPEC is None or _BOOTSTRAP_SPEC.loader is None:  # pragma: no cover - defensive guard
    raise RuntimeError("Unable to load conversation store bootstrap module")

bootstrap_module = importlib.util.module_from_spec(_BOOTSTRAP_SPEC)
_BOOTSTRAP_SPEC.loader.exec_module(bootstrap_module)

BootstrapError = bootstrap_module.BootstrapError
sa_make_url = bootstrap_module.make_url


class _Result(types.SimpleNamespace):
    def __init__(self, returncode: int = 0) -> None:
        super().__init__(returncode=returncode)


def test_bootstrap_import_requires_sqlalchemy(tmp_path, monkeypatch):
    bootstrap_path = Path(bootstrap_module.__file__).resolve()
    fake_root = tmp_path / "fake_packages"
    fake_sqlalchemy = fake_root / "sqlalchemy"
    fake_sqlalchemy.mkdir(parents=True)
    (fake_sqlalchemy / "__init__.py").write_text("raise ImportError('sqlalchemy unavailable')")

    monkeypatch.syspath_prepend(str(fake_root))

    cached_sqlalchemy = {
        name: sys.modules.pop(name)
        for name in list(sys.modules)
        if name.startswith("sqlalchemy")
    }
    try:
        spec = importlib.util.spec_from_file_location(
            "modules.conversation_store.bootstrap_missing_sqlalchemy",
            bootstrap_path,
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        with pytest.raises(ImportError):
            spec.loader.exec_module(module)
    finally:
        for name in list(sys.modules):
            if name.startswith("sqlalchemy") and name not in cached_sqlalchemy:
                sys.modules.pop(name, None)
        sys.modules.pop("modules.conversation_store.bootstrap_missing_sqlalchemy", None)
        sys.modules.update(cached_sqlalchemy)


def test_install_postgres_client_prompts_for_password_and_uses_sudo():
    commands: list[tuple[list[str], bool, str | None, bool | None]] = []
    psql_available = False
    prompt_calls: list[str | None] = []

    def fake_run(cmd, check=True, input=None, text=None):
        nonlocal psql_available
        commands.append((cmd, check, input, text))
        if "apt-get" in cmd:
            psql_available = True
        if cmd and cmd[0] == "pg_isready":
            return _Result(returncode=0)
        return _Result(returncode=0)

    def fake_which(name: str) -> str | None:
        if name == "psql":
            return "/usr/bin/psql" if psql_available else None
        return f"/usr/bin/{name}"

    def prompt() -> str:
        prompt_calls.append("secret")
        return "secret"

    bootstrap_module._install_postgres_client(
        run=fake_run,
        which=fake_which,
        platform_system=lambda: "Linux",
        request_privileged_password=prompt,
        geteuid=lambda: 1000,
    )

    assert prompt_calls == ["secret"]
    assert commands[0][0][:3] == ["sudo", "-S", "apt-get"]
    assert commands[0][0][3:] == ["update"]
    assert commands[0][2] == "secret\n"
    assert commands[0][3] is True
    assert commands[1][0][:3] == ["sudo", "-S", "apt-get"]
    assert commands[1][0][3:] == ["install", "-y", "postgresql", "postgresql-client"]


def test_install_postgres_client_cancelled_prompt_raises_bootstrap_error():
    def fake_run(*_args, **_kwargs):  # pragma: no cover - should not be called
        raise AssertionError("run should not be invoked when prompt is cancelled")

    def fake_which(name: str) -> str | None:
        if name == "psql":
            return None
        return f"/usr/bin/{name}"

    def prompt() -> None:
        return None

    with pytest.raises(BootstrapError) as excinfo:
        bootstrap_module._install_postgres_client(
            run=fake_run,
            which=fake_which,
            platform_system=lambda: "Linux",
            request_privileged_password=prompt,
            geteuid=lambda: 1000,
        )

    assert "cancelled" in str(excinfo.value).lower()


def test_bootstrap_creates_role_using_privileged_credentials(monkeypatch):
    class DummyOperationalError(Exception):
        pass

    class _Identifier:
        def __init__(self, value: str) -> None:
            self.value = value

    class _SQLWrapper:
        def __init__(self, template: str) -> None:
            self.template = template

        def format(self, *args: _Identifier) -> str:
            replacements = [getattr(arg, "value", str(arg)) for arg in args]
            return self.template.format(*replacements)

    monkeypatch.setattr(bootstrap_module, "OperationalError", DummyOperationalError)
    monkeypatch.setattr(bootstrap_module, "_ensure_psycopg_loaded", lambda **_kwargs: None)
    monkeypatch.setattr(bootstrap_module, "_install_postgres_client", lambda **_kwargs: None)
    monkeypatch.setattr(
        bootstrap_module,
        "sql",
        types.SimpleNamespace(
            SQL=lambda template: _SQLWrapper(template),
            Identifier=lambda value: _Identifier(value),
        ),
    )

    queries: list[tuple[str, tuple | None]] = []
    provisioning_active = {"value": False}

    class DummyCursor:
        def __init__(self) -> None:
            self._last_query: str | None = None

        def __enter__(self) -> "DummyCursor":
            return self

        def __exit__(self, *_exc) -> None:
            return None

        def execute(self, query, params=None) -> None:
            rendered = str(query)
            queries.append((rendered, params))
            self._last_query = rendered

        def fetchone(self):
            if not self._last_query:
                return None
            if "pg_roles" in self._last_query:
                return None
            if "pg_database" in self._last_query:
                return None
            return None

    class DummyConnection:
        def __init__(self) -> None:
            self.autocommit = False

        def cursor(self):
            return DummyCursor()

        def close(self) -> None:
            return None

    def fake_connector(conninfo: str):
        if conninfo.startswith("postgresql://postgres"):
            provisioning_active["value"] = True
            return DummyConnection()
        if conninfo.startswith("postgresql://atlas"):
            if not provisioning_active["value"]:
                raise DummyOperationalError('role "atlas" does not exist')
            return DummyConnection()
        raise AssertionError(f"Unexpected connection string: {conninfo}")

    result = bootstrap_module.bootstrap_conversation_store(
        "postgresql://atlas@localhost:5432/atlas",
        connector=fake_connector,
        privileged_credentials=("postgres", "secret"),
    )

    assert provisioning_active["value"] is True
    assert any("CREATE ROLE" in query for query, _ in queries)
    assert any("CREATE DATABASE" in query for query, _ in queries)
    assert result.startswith("postgresql://atlas")


def test_bootstrap_passes_through_sqlite(tmp_path):
    db_path = tmp_path / "bootstrap.sqlite"
    result = bootstrap_module.bootstrap_conversation_store(f"sqlite:///{db_path}")
    parsed = sa_make_url(result)
    assert Path(parsed.database).exists()
    assert parsed.get_backend_name() == "sqlite"
