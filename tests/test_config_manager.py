import os
import sys
import types

import pytest

if "yaml" not in sys.modules:
    yaml_module = types.ModuleType("yaml")
    yaml_module.safe_load = lambda *args, **kwargs: {}
    yaml_module.dump = lambda *args, **kwargs: None
    sys.modules["yaml"] = yaml_module

if "dotenv" not in sys.modules:
    dotenv_module = types.ModuleType("dotenv")
    dotenv_module.load_dotenv = lambda *args, **kwargs: None
    dotenv_module.set_key = lambda *args, **kwargs: None
    dotenv_module.find_dotenv = lambda *args, **kwargs: ""
    sys.modules["dotenv"] = dotenv_module

import ATLAS.config as config_module
from ATLAS.config import ConfigManager


@pytest.fixture
def config_manager(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("")

    monkeypatch.setenv("OPENAI_API_KEY", "initial-key")
    monkeypatch.setenv("DEFAULT_PROVIDER", "OpenAI")
    monkeypatch.setenv("DEFAULT_MODEL", "gpt-4o")

    recorded = {}

    def fake_set_key(path, key, value):
        recorded[(path, key)] = value

    monkeypatch.setattr(config_module, "set_key", fake_set_key)
    monkeypatch.setattr(config_module, "find_dotenv", lambda: str(env_file))
    monkeypatch.setattr(config_module, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        ConfigManager,
        "_load_env_config",
        lambda self: {
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
            "DEFAULT_PROVIDER": os.getenv("DEFAULT_PROVIDER", "OpenAI"),
            "DEFAULT_MODEL": os.getenv("DEFAULT_MODEL", "gpt-4o"),
            "MISTRAL_API_KEY": None,
            "HUGGINGFACE_API_KEY": None,
            "GOOGLE_API_KEY": None,
            "ANTHROPIC_API_KEY": None,
            "GROK_API_KEY": None,
            "APP_ROOT": tmp_path.as_posix(),
            "OPENAI_BASE_URL": os.getenv("OPENAI_BASE_URL"),
            "OPENAI_ORGANIZATION": os.getenv("OPENAI_ORGANIZATION"),
        },
    )
    monkeypatch.setattr(ConfigManager, "_load_yaml_config", lambda self: {})

    manager = ConfigManager()
    manager._recorded_set_key = recorded
    manager._env_path = str(env_file)
    return manager


def test_set_google_credentials_updates_state(config_manager):
    config_manager.set_google_credentials("/tmp/creds.json")

    assert config_manager.config["GOOGLE_APPLICATION_CREDENTIALS"] == "/tmp/creds.json"
    assert os.environ["GOOGLE_APPLICATION_CREDENTIALS"] == "/tmp/creds.json"
    assert (
        config_manager._recorded_set_key[(config_manager._env_path, "GOOGLE_APPLICATION_CREDENTIALS")]
        == "/tmp/creds.json"
    )


def test_set_google_credentials_failure_rolls_back(config_manager, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("write error")

    monkeypatch.setattr(config_module, "set_key", fail)
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)

    with pytest.raises(RuntimeError):
        config_manager.set_google_credentials("/tmp/new.json")

    assert "GOOGLE_APPLICATION_CREDENTIALS" not in config_manager.config
    assert "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ


def test_set_openai_speech_config_updates_values(config_manager):
    config_manager.set_openai_speech_config(
        api_key="fresh-key",
        stt_provider="GPT-4o STT",
        tts_provider="GPT-4o Mini TTS",
        language="en",
        task="transcribe",
        initial_prompt="hello",
    )

    assert config_manager.config["OPENAI_API_KEY"] == "fresh-key"
    assert config_manager.config["OPENAI_STT_PROVIDER"] == "GPT-4o STT"
    assert config_manager.config["OPENAI_TTS_PROVIDER"] == "GPT-4o Mini TTS"
    assert config_manager.config["OPENAI_LANGUAGE"] == "en"
    assert config_manager.config["OPENAI_TASK"] == "transcribe"
    assert config_manager.config["OPENAI_INITIAL_PROMPT"] == "hello"
    assert os.environ["OPENAI_API_KEY"] == "fresh-key"
    assert (
        config_manager._recorded_set_key[(config_manager._env_path, "OPENAI_API_KEY")]
        == "fresh-key"
    )


def test_set_openai_speech_config_rejects_empty_key(config_manager):
    with pytest.raises(ValueError):
        config_manager.set_openai_speech_config(api_key="")


def test_set_hf_token_updates_state(config_manager):
    config_manager.set_hf_token("hf-token")

    assert config_manager.config["HUGGINGFACE_API_KEY"] == "hf-token"
    assert os.environ["HUGGINGFACE_API_KEY"] == "hf-token"
    assert (
        config_manager._recorded_set_key[(config_manager._env_path, "HUGGINGFACE_API_KEY")]
        == "hf-token"
    )


def test_set_openai_llm_settings_updates_state(config_manager):
    result = config_manager.set_openai_llm_settings(
        model="gpt-4o-mini",
        temperature=0.75,
        max_tokens=2048,
        stream=False,
        api_key="sk-test",
        organization="org-42",
    )

    assert result["model"] == "gpt-4o-mini"
    stored = config_manager.config["OPENAI_LLM"]
    assert stored["temperature"] == 0.75
    assert stored["max_tokens"] == 2048
    assert stored["stream"] is False
    assert stored["organization"] == "org-42"

    assert config_manager.config["DEFAULT_MODEL"] == "gpt-4o-mini"
    assert os.environ["DEFAULT_MODEL"] == "gpt-4o-mini"
    assert config_manager.config["OPENAI_API_KEY"] == "sk-test"
    assert os.environ["OPENAI_API_KEY"] == "sk-test"
    assert os.environ["OPENAI_ORGANIZATION"] == "org-42"
    assert (
        config_manager._recorded_set_key[(config_manager._env_path, "DEFAULT_MODEL")]
        == "gpt-4o-mini"
    )


def test_set_openai_llm_settings_clears_optional_fields(config_manager):
    config_manager.set_openai_llm_settings(
        model="gpt-4o",
        organization="",
    )

    stored = config_manager.config["OPENAI_LLM"]
    assert stored["organization"] is None
    assert "OPENAI_ORGANIZATION" not in os.environ
