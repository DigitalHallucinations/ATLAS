from __future__ import annotations

import asyncio
from unittest import mock
import importlib
import sys
import types
import datetime
from pathlib import Path

import pytest

from modules.Tools.tool_event_system import event_system


@pytest.fixture(autouse=True)
def stub_tool_manager_dependencies(monkeypatch):
    """Provide minimal dependencies required to import sandboxed tools."""

    for key in [
        "OPENAI_API_KEY",
        "MISTRAL_API_KEY",
        "GOOGLE_API_KEY",
        "HUGGINGFACE_API_KEY",
        "ANTHROPIC_API_KEY",
        "GROK_API_KEY",
        "XI_API_KEY",
    ]:
        monkeypatch.delenv(key, raising=False)

    if "yaml" not in sys.modules:
        yaml_stub = types.ModuleType("yaml")
        yaml_stub.safe_load = lambda stream: {}
        monkeypatch.setitem(sys.modules, "yaml", yaml_stub)

    if "jsonschema" not in sys.modules:
        jsonschema_stub = types.ModuleType("jsonschema")

        class _DummyValidationError(Exception):
            pass

        class _DummyValidator:
            def __init__(self, *_args, **_kwargs):
                return

            def is_valid(self, *_args, **_kwargs):  # pragma: no cover - defensive stub
                return True

        jsonschema_stub.ValidationError = _DummyValidationError
        jsonschema_stub.Draft7Validator = _DummyValidator
        monkeypatch.setitem(sys.modules, "jsonschema", jsonschema_stub)


    if "pytz" not in sys.modules:
        pytz_stub = types.ModuleType("pytz")

        class _StubTimezone(datetime.tzinfo):
            def utcoffset(self, _dt):  # pragma: no cover - deterministic offset
                return datetime.timedelta(0)

            def dst(self, _dt):  # pragma: no cover - deterministic offset
                return datetime.timedelta(0)

            def tzname(self, _dt):  # pragma: no cover - deterministic name
                return "UTC"

        def _timezone(_name):  # pragma: no cover - deterministic tzinfo
            return _StubTimezone()

        pytz_stub.timezone = _timezone
        monkeypatch.setitem(sys.modules, "pytz", pytz_stub)
    if "dotenv" not in sys.modules:
        dotenv_stub = types.ModuleType("dotenv")

        def _load_dotenv(*_args, **_kwargs):  # pragma: no cover - stub helper
            return True

        def _set_key(*_args, **_kwargs):  # pragma: no cover - stub helper
            return None

        def _find_dotenv(*_args, **_kwargs):  # pragma: no cover - stub helper
            return ""

        dotenv_stub.load_dotenv = _load_dotenv
        dotenv_stub.set_key = _set_key
        dotenv_stub.find_dotenv = _find_dotenv
        monkeypatch.setitem(sys.modules, "dotenv", dotenv_stub)


    if "aiohttp" not in sys.modules:
        aiohttp_stub = types.ModuleType("aiohttp")

        class _DummyResponse:
            status = 200

            async def __aenter__(self):  # pragma: no cover - stub
                return self

            async def __aexit__(self, *_args):  # pragma: no cover - stub
                return False

            async def json(self):  # pragma: no cover - stub
                return []

        class ClientSession:
            def __init__(self, *_args, **_kwargs):
                return

            async def __aenter__(self):  # pragma: no cover - stub
                return self

            async def __aexit__(self, *_args):  # pragma: no cover - stub
                return False

            def get(self, *_args, **_kwargs):  # pragma: no cover - stub
                return _DummyResponse()

        class ClientTimeout:
            def __init__(self, *_args, **_kwargs):  # pragma: no cover - stub
                return

        aiohttp_stub.ClientSession = ClientSession
        aiohttp_stub.ClientTimeout = ClientTimeout
        monkeypatch.setitem(sys.modules, "aiohttp", aiohttp_stub)

@pytest.fixture()
def terminal_command_module():
    return importlib.import_module("modules.Tools.Base_Tools.terminal_command")


@pytest.fixture()
def capture_terminal_events():
    events: list[dict] = []

    def _listener(payload):
        events.append(payload)

    event_system.subscribe("terminal_command_executed", _listener)
    try:
        yield events
    finally:
        event_system.unsubscribe("terminal_command_executed", _listener)


def test_terminal_command_success(terminal_command_module, capture_terminal_events):
    result = asyncio.run(
        terminal_command_module.TerminalCommand(
            command="echo",
            arguments=["hello world"],
            working_directory=str(terminal_command_module.DEFAULT_TERMINAL_JAIL),
        )
    )

    assert result["exit_code"] == 0
    assert result["stdout"].strip() == "hello world"

    assert capture_terminal_events, "An execution event should be emitted."
    event = capture_terminal_events[-1]
    assert event["status"] == "success"
    assert event["command"][0] == "echo"
    assert event["stdout_truncated"] is False


def test_terminal_command_timeout(terminal_command_module, capture_terminal_events):
    with pytest.raises(terminal_command_module.TerminalCommandTimeoutError):
        asyncio.run(terminal_command_module.TerminalCommand(command="sleep", arguments=["1"], timeout=0.1))

    assert capture_terminal_events[-1]["status"] == "timeout"


def test_terminal_command_jail_violation(terminal_command_module):
    outside_root = Path(terminal_command_module.DEFAULT_TERMINAL_JAIL).parent
    with pytest.raises(terminal_command_module.WorkingDirectoryViolationError):
        asyncio.run(
            terminal_command_module.TerminalCommand(
                command="echo",
                arguments=["escape"],
                working_directory=str(outside_root),
            )
        )


def test_terminal_command_event_redaction(terminal_command_module, capture_terminal_events):
    result = asyncio.run(
        terminal_command_module.TerminalCommand(
            command="echo",
            arguments=["--password=super-secret", "token=abc123"],
        )
    )

    assert "super-secret" in result["stdout"]
    assert "abc123" in result["stdout"]

    event = capture_terminal_events[-1]
    assert event["command"][1] == "--password=[REDACTED]"
    assert event["command"][2] == "token=[REDACTED]"
    assert event["stdout"] == "[REDACTED]"


def test_terminal_command_logs_are_redacted(terminal_command_module):
    with mock.patch.object(terminal_command_module.logger, "info") as mock_info:
        result = asyncio.run(
            terminal_command_module.TerminalCommand(
                command="echo",
                arguments=["--password=super-secret", "token=abc123"],
            )
        )

    assert result["exit_code"] == 0
    log_output = "\n".join(
        call.args[0] % call.args[1:] if call.args else str(call.kwargs)
        for call in mock_info.call_args_list
    )
    assert "super-secret" not in log_output
    assert "abc123" not in log_output
    assert "[REDACTED]" in log_output


def test_terminal_command_sequence_arguments(terminal_command_module):
    result = asyncio.run(
        terminal_command_module.TerminalCommand(
            command=["echo", "repeat", "repeat"],
            arguments=["repeat"],
        )
    )

    # echo collapses whitespace but preserves duplicated tokens
    assert result["stdout"].strip() == "repeat repeat repeat"


def test_terminal_command_sequence_allowlist(terminal_command_module):
    with pytest.raises(terminal_command_module.CommandNotAllowedError):
        asyncio.run(
            terminal_command_module.TerminalCommand(
                command=["python", "-V"],
            )
        )


def test_terminal_command_sequence_empty(terminal_command_module):
    with pytest.raises(terminal_command_module.CommandNotAllowedError):
        asyncio.run(
            terminal_command_module.TerminalCommand(
                command=[],
            )
        )
