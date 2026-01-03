from __future__ import annotations

import asyncio
import datetime
import importlib
import sys
import types

import pytest


@pytest.fixture(autouse=True)
def stub_tool_manager_dependencies(monkeypatch):
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
def filesystem_module(monkeypatch, tmp_path):
    sandbox_root = tmp_path / "sandbox"
    sandbox_root.mkdir()
    monkeypatch.setenv("ATLAS_FILESYSTEM_SANDBOX", str(sandbox_root))
    monkeypatch.delenv("ATLAS_FILESYSTEM_MAX_TOTAL_BYTES", raising=False)
    monkeypatch.delenv("ATLAS_FILESYSTEM_MAX_WRITE_BYTES", raising=False)
    monkeypatch.delenv("ATLAS_FILESYSTEM_MAX_READ_BYTES", raising=False)
    monkeypatch.delenv("ATLAS_FILESYSTEM_MAX_LIST_ENTRIES", raising=False)
    monkeypatch.delenv("ATLAS_FILESYSTEM_OPERATION_TIMEOUT", raising=False)
    module = importlib.import_module("modules.Tools.Base_Tools.filesystem_io")
    importlib.reload(module)
    return module


def test_filesystem_read_write_list(filesystem_module):
    root = filesystem_module._current_config().root
    notes_dir = root / "notes"
    notes_dir.mkdir(parents=True, exist_ok=True)

    write_result = asyncio.run(
        filesystem_module.write_file(
            path="notes/hello.txt",
            content="hello sandbox",
        )
    )

    assert write_result["path"] == "notes/hello.txt"
    assert write_result["size"] == len("hello sandbox")

    read_result = asyncio.run(filesystem_module.read_file("notes/hello.txt"))
    assert read_result["content_encoding"] == "text"
    assert read_result["content"].strip() == "hello sandbox"
    assert read_result["truncated"] is False

    list_result = asyncio.run(filesystem_module.list_dir("notes"))
    assert list_result["path"] == "notes"
    assert list_result["entries"]
    entry = list_result["entries"][0]
    assert entry["name"] == "hello.txt"
    assert entry["is_dir"] is False


def test_filesystem_write_quota(filesystem_module, monkeypatch):
    monkeypatch.setenv("ATLAS_FILESYSTEM_MAX_WRITE_BYTES", "4")
    with pytest.raises(filesystem_module.QuotaExceededError):
        asyncio.run(filesystem_module.write_file(path="small.txt", content="12345"))


def test_filesystem_sandbox_escape(filesystem_module):
    with pytest.raises(filesystem_module.SandboxViolationError):
        asyncio.run(filesystem_module.read_file("../escape.txt"))

