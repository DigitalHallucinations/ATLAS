from typing import Any

import pytest

from core.config import core
from core.config.core import ConfigCore


class DummyLogger:
    def __init__(self) -> None:
        self.records: list[str] = []

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        if args:
            try:
                message = message % args
            except Exception:
                message = str(message)
        self.records.append(str(message))

    def debug(self, *_args: Any, **_kwargs: Any) -> None:  # pragma: no cover - no-op
        return None


@pytest.fixture(autouse=True)
def _stub_dotenv(monkeypatch):
    monkeypatch.setattr(core, "load_dotenv", lambda *args, **kwargs: None)


def test_config_core_normalizes_model_cache(tmp_path, monkeypatch):
    dummy_logger = DummyLogger()
    monkeypatch.setattr(core, "setup_logger", lambda name: dummy_logger)
    monkeypatch.setattr(
        ConfigCore,
        "_compute_yaml_path",
        lambda self: str(tmp_path / "atlas_config.yaml"),
    )
    monkeypatch.setattr(
        ConfigCore,
        "_load_env_config",
        lambda self: {"APP_ROOT": tmp_path.as_posix()},
    )
    monkeypatch.setattr(
        ConfigCore,
        "_load_yaml_config",
        lambda self: {"MODEL_CACHE": {"OpenAI": ["gpt-4o", "gpt-4o", "  gpt-4o-mini "]}},
    )

    core_instance = ConfigCore()

    assert core_instance.config["MODEL_CACHE"] == {"OpenAI": ["gpt-4o", "gpt-4o-mini"]}
    assert core_instance.yaml_config["MODEL_CACHE"] == {"OpenAI": ["gpt-4o", "gpt-4o-mini"]}


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, None),
        ("example.com", ["example.com"]),
        (["example.com", " example.org ", ""], ["example.com", "example.org"]),
        ([], None),
    ],
)
def test_normalize_network_allowlist(value, expected, monkeypatch):
    monkeypatch.setattr(core, "setup_logger", lambda name: DummyLogger())
    monkeypatch.setattr(ConfigCore, "_compute_yaml_path", lambda self: "dummy.yaml")
    monkeypatch.setattr(ConfigCore, "_load_env_config", lambda self: {"APP_ROOT": "."})
    monkeypatch.setattr(ConfigCore, "_load_yaml_config", lambda self: {})

    instance = ConfigCore()
    assert instance._normalize_network_allowlist(value) == expected
