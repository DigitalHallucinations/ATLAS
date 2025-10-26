from __future__ import annotations

import types

import pytest

from modules.conversation_store import bootstrap as bootstrap_module
from modules.conversation_store.bootstrap import BootstrapError


class _Result(types.SimpleNamespace):
    def __init__(self, returncode: int = 0) -> None:
        super().__init__(returncode=returncode)


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
