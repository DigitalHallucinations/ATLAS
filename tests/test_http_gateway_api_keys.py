import importlib
import logging
import sys
import types
from pathlib import Path
from typing import Generator

import pytest


def _install_fastapi_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    fastapi_stub = types.ModuleType("fastapi")

    class HTTPException(Exception):
        def __init__(self, status_code: int | None = None, detail: str | None = None):
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    class FastAPI:  # pragma: no cover - only needed for import
        def __init__(self, *_: object, **__: object) -> None:
            self.state = types.SimpleNamespace()

        @staticmethod
        def _decorator(*_: object, **__: object):
            def wrapper(func):
                return func

            return wrapper

        middleware = _decorator
        get = _decorator
        post = _decorator
        delete = _decorator
        patch = _decorator

    class Request:  # pragma: no cover - only needed for import
        def __init__(self, app: FastAPI | None = None) -> None:
            self.app = app or FastAPI()
            self.headers: dict[str, str] = {}
            self.url = types.SimpleNamespace(path="")

    fastapi_stub.HTTPException = HTTPException
    fastapi_stub.FastAPI = FastAPI
    fastapi_stub.Request = Request

    concurrency_stub = types.ModuleType("fastapi.concurrency")

    def run_in_threadpool(func, *args, **kwargs):  # pragma: no cover - only needed for import
        return func(*args, **kwargs)

    concurrency_stub.run_in_threadpool = run_in_threadpool

    responses_stub = types.ModuleType("fastapi.responses")

    class JSONResponse:  # pragma: no cover - only needed for import
        def __init__(self, content=None, status_code=None, media_type=None):
            self.content = content
            self.status_code = status_code
            self.media_type = media_type

    class StreamingResponse:  # pragma: no cover - only needed for import
        def __init__(self, *_: object, **__: object) -> None:
            ...

    responses_stub.JSONResponse = JSONResponse
    responses_stub.StreamingResponse = StreamingResponse

    monkeypatch.setitem(sys.modules, "fastapi", fastapi_stub)
    monkeypatch.setitem(sys.modules, "fastapi.concurrency", concurrency_stub)
    monkeypatch.setitem(sys.modules, "fastapi.responses", responses_stub)


@pytest.fixture(autouse=True)
def gateway_module(monkeypatch: pytest.MonkeyPatch) -> Generator[types.ModuleType, None, None]:
    monkeypatch.delenv("ATLAS_HTTP_API_KEYS", raising=False)
    monkeypatch.delenv("ATLAS_HTTP_API_KEY_FILE", raising=False)
    monkeypatch.delenv("ATLAS_HTTP_API_KEY_PUBLIC_PATHS", raising=False)
    _install_fastapi_stubs(monkeypatch)

    module = importlib.import_module("server.http_gateway")
    importlib.reload(module)
    try:
        yield module
    finally:
        monkeypatch.undo()


def test_read_api_key_file_collects_unique_tokens(gateway_module: types.ModuleType, tmp_path: Path):
    key_file = tmp_path / "keys.txt"
    key_file.write_text("env-token\n\n file-token \nENV-token\n", encoding="utf-8")

    tokens = gateway_module._read_api_key_file(str(key_file))

    assert tokens == {"env-token", "file-token", "ENV-token"}


def test_read_api_key_file_missing_is_logged(
    gateway_module: types.ModuleType, tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    missing_file = tmp_path / "missing.txt"

    with caplog.at_level(logging.WARNING, logger="atlas.http_gateway"):
        tokens = gateway_module._read_api_key_file(str(missing_file))

    assert tokens == set()


def test_load_api_key_config_combines_env_and_file(
    gateway_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    key_file = tmp_path / "keys.txt"
    key_file.write_text("file-token\nshared-token\n", encoding="utf-8")

    monkeypatch.setenv(gateway_module._ENV_API_KEYS, "env-token, shared-token ,  ")
    monkeypatch.setenv(gateway_module._ENV_API_KEY_FILE, str(key_file))
    monkeypatch.setenv(gateway_module._ENV_API_KEY_PUBLIC_PATHS, "/healthz, /status ")

    config = gateway_module._load_api_key_config()

    assert config.enabled is True
    assert config.valid_tokens == {"env-token", "shared-token", "file-token"}
    assert config.public_paths == {"/healthz", "/status"}


def test_load_api_key_config_defaults_with_no_tokens(gateway_module: types.ModuleType):
    config = gateway_module._load_api_key_config()

    assert config.enabled is False
    assert config.valid_tokens == frozenset()
    assert config.public_paths == {"/healthz"}
