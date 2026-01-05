import json
import math
import os
import sys
import types
from typing import Any, Dict, Optional

import pytest

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - fallback for constrained environments
    import json as _json

    yaml = types.ModuleType("yaml")

    def _safe_load_fallback(stream):
        if hasattr(stream, "read"):
            content = stream.read()
        else:
            content = stream or ""

        content = content or ""
        if not content.strip():
            return {}

        try:
            return _json.loads(content)
        except _json.JSONDecodeError:
            return {}

    def _dump_fallback(data, stream=None, **_kwargs):
        text = _json.dumps(data or {})
        if stream is None:
            return text
        stream.write(text)
        return text

    yaml.safe_load = _safe_load_fallback
    yaml.dump = _dump_fallback
    sys.modules["yaml"] = yaml

if "dotenv" not in sys.modules:
    dotenv_module = types.ModuleType("dotenv")
    dotenv_module.load_dotenv = lambda *args, **kwargs: None
    dotenv_module.set_key = lambda *args, **kwargs: None
    dotenv_module.find_dotenv = lambda *args, **kwargs: ""
    dotenv_module.dotenv_values = lambda *args, **kwargs: {}
    sys.modules["dotenv"] = dotenv_module

import core.config.config_manager as config_impl
from core.config import ConfigManager


class DummyLogger:
    def __init__(self):
        self.warnings = []

    def warning(self, message, *args, **kwargs):
        if args:
            message = message % args
        self.warnings.append(message)

    def info(self, *args, **kwargs):
        return None

    def error(self, *args, **kwargs):
        return None

    def debug(self, *args, **kwargs):
        return None


@pytest.fixture
def config_manager(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("")

    monkeypatch.setenv("OPENAI_API_KEY", "initial-key")
    monkeypatch.setenv("DEFAULT_PROVIDER", "OpenAI")
    monkeypatch.setenv("DEFAULT_MODEL", "gpt-4o")

    recorded: Dict[tuple[str, str], str] = {}

    def fake_set_key(path, key, value):
        recorded[(path, key)] = value

    monkeypatch.setattr(config_impl, "set_key", fake_set_key)
    monkeypatch.setattr(config_impl, "find_dotenv", lambda: str(env_file))
    monkeypatch.setattr(config_impl, "load_dotenv", lambda *args, **kwargs: None)
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
    manager._recorded_set_key = recorded  # type: ignore[attr-defined]
    manager._env_path = str(env_file)  # type: ignore[attr-defined]
    return manager


def test_init_missing_default_provider_records_warning(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("")

    dummy_logger = DummyLogger()

    monkeypatch.setattr(config_impl, "setup_logger", lambda name: dummy_logger)
    monkeypatch.setattr(config_impl, "set_key", lambda *args, **kwargs: None)
    monkeypatch.setattr(config_impl, "find_dotenv", lambda: str(env_file))
    monkeypatch.setattr(config_impl, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("DEFAULT_PROVIDER", "OpenAI")
    monkeypatch.setattr(
        ConfigManager,
        "_load_env_config",
        lambda self: {
            "OPENAI_API_KEY": None,
            "DEFAULT_PROVIDER": "OpenAI",
            "DEFAULT_MODEL": "gpt-4o",
            "APP_ROOT": tmp_path.as_posix(),
        },
    )
    monkeypatch.setattr(ConfigManager, "_load_yaml_config", lambda self: {})

    manager = ConfigManager()

    assert not manager.providers.is_default_provider_ready()
    warnings = manager.providers.get_pending_provider_warnings()
    assert warnings == {
        "OpenAI": "API key for provider 'OpenAI' is not configured. Protected features will remain unavailable until a key is provided.",
    }
    assert dummy_logger.warnings


def test_provider_warning_cleared_after_api_key_update(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("")
    recorded: Dict[tuple[str, str], str] = {}

    def fake_set_key(path, key, value):
        recorded[(path, key)] = value

    monkeypatch.setattr(config_impl, "setup_logger", lambda name: DummyLogger())
    monkeypatch.setattr(config_impl, "set_key", fake_set_key)
    monkeypatch.setattr(config_impl, "find_dotenv", lambda: str(env_file))
    monkeypatch.setattr(config_impl, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(
        ConfigManager,
        "_load_env_config",
        lambda self: {
            "OPENAI_API_KEY": None,
            "DEFAULT_PROVIDER": "OpenAI",
            "DEFAULT_MODEL": "gpt-4o",
            "APP_ROOT": tmp_path.as_posix(),
        },
    )
    monkeypatch.setattr(ConfigManager, "_load_yaml_config", lambda self: {})

    manager = ConfigManager()
    assert not manager.providers.is_default_provider_ready()

    manager.providers.update_api_key("OpenAI", "new-key")

    assert manager.providers.is_default_provider_ready()
    assert manager.providers.get_pending_provider_warnings() == {}
    assert recorded[(str(env_file), "OPENAI_API_KEY")] == "new-key"


def test_init_with_default_provider_credentials_has_no_warnings(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("")

    dummy_logger = DummyLogger()

    monkeypatch.setattr(config_impl, "setup_logger", lambda name: dummy_logger)
    monkeypatch.setattr(config_impl, "set_key", lambda *args, **kwargs: None)
    monkeypatch.setattr(config_impl, "find_dotenv", lambda: str(env_file))
    monkeypatch.setattr(config_impl, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setenv("OPENAI_API_KEY", "configured-key")
    monkeypatch.setenv("DEFAULT_PROVIDER", "OpenAI")
    monkeypatch.setattr(
        ConfigManager,
        "_load_env_config",
        lambda self: {
            "OPENAI_API_KEY": "configured-key",
            "DEFAULT_PROVIDER": "OpenAI",
            "DEFAULT_MODEL": "gpt-4o",
            "APP_ROOT": tmp_path.as_posix(),
        },
    )
    monkeypatch.setattr(ConfigManager, "_load_yaml_config", lambda self: {})

    manager = ConfigManager()

    assert manager.providers.is_default_provider_ready()
    assert manager.providers.get_pending_provider_warnings() == {}


def test_update_api_key_creates_env_when_missing(tmp_path, monkeypatch):
    expected_env_path = tmp_path / ".env"
    assert not expected_env_path.exists()

    load_calls: list[tuple[str, bool]] = []

    def fake_load_dotenv(path=None, override=False):
        load_calls.append((path, override))
        return None

    def fake_set_key(path, key, value):
        existing: Dict[str, str] = {}
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh.read().splitlines():
                    if "=" in line:
                        k, v = line.split("=", 1)
                        existing[k] = v
        existing[key] = value
        with open(path, "w", encoding="utf-8") as fh:
            for item_key, item_value in existing.items():
                fh.write(f"{item_key}={item_value}\n")
        return key, value

    logger = types.SimpleNamespace(
        info=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
        debug=lambda *args, **kwargs: None,
    )

    monkeypatch.setattr(config_impl, "setup_logger", lambda name: logger)
    monkeypatch.setattr(config_impl, "set_key", fake_set_key)
    monkeypatch.setattr(config_impl, "find_dotenv", lambda: "")
    monkeypatch.setattr(config_impl, "load_dotenv", fake_load_dotenv)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("DEFAULT_PROVIDER", "OpenAI")
    monkeypatch.setenv("DEFAULT_MODEL", "gpt-4o")
    monkeypatch.setattr(
        ConfigManager,
        "_load_env_config",
        lambda self: {
            "OPENAI_API_KEY": None,
            "DEFAULT_PROVIDER": "OpenAI",
            "DEFAULT_MODEL": "gpt-4o",
            "APP_ROOT": tmp_path.as_posix(),
        },
    )
    monkeypatch.setattr(ConfigManager, "_load_yaml_config", lambda self: {})

    manager = ConfigManager()
    manager.providers.update_api_key("OpenAI", "created-key")

    assert expected_env_path.exists()
    with open(expected_env_path, "r", encoding="utf-8") as fh:
        contents = fh.read()
    assert "OPENAI_API_KEY=created-key" in contents
    assert os.environ["OPENAI_API_KEY"] == "created-key"
    assert manager.env_config["OPENAI_API_KEY"] == "created-key"
    assert manager.config["OPENAI_API_KEY"] == "created-key"
    assert load_calls[-1] == (expected_env_path.as_posix(), True)
    assert manager.providers.get_pending_provider_warnings() == {}


def test_set_google_credentials_updates_state(config_manager):
    config_manager.providers.google.set_credentials("/tmp/creds.json")

    assert (
        config_manager.config["GOOGLE_APPLICATION_CREDENTIALS"] == "/tmp/creds.json"
    )
    assert os.environ["GOOGLE_APPLICATION_CREDENTIALS"] == "/tmp/creds.json"
    assert (
        config_manager._recorded_set_key[
            (config_manager._env_path, "GOOGLE_APPLICATION_CREDENTIALS")
        ]
        == "/tmp/creds.json"
    )


def test_set_google_credentials_failure_rolls_back(config_manager, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("write error")

    monkeypatch.setattr(config_impl, "set_key", fail)
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)

    with pytest.raises(RuntimeError):
        config_manager.providers.google.set_credentials("/tmp/new.json")

    assert "GOOGLE_APPLICATION_CREDENTIALS" not in config_manager.config
    assert "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ


def test_set_openai_speech_config_updates_values(config_manager):
    config_manager.providers.openai.set_speech_config(
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
    assert config_manager.yaml_config["OPENAI_SPEECH"] == {
        "stt_provider": "GPT-4o STT",
        "tts_provider": "GPT-4o Mini TTS",
        "language": "en",
        "task": "transcribe",
        "initial_prompt": "hello",
    }
    assert config_manager.config["OPENAI_SPEECH"] == {
        "stt_provider": "GPT-4o STT",
        "tts_provider": "GPT-4o Mini TTS",
        "language": "en",
        "task": "transcribe",
        "initial_prompt": "hello",
    }
    assert os.environ["OPENAI_API_KEY"] == "fresh-key"
    assert (
        config_manager._recorded_set_key[(config_manager._env_path, "OPENAI_API_KEY")]
        == "fresh-key"
    )


def test_set_openai_speech_config_rejects_empty_key(config_manager):
    with pytest.raises(ValueError):
        config_manager.providers.openai.set_speech_config(api_key="")


def test_openai_speech_settings_persist_to_yaml(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("")

    dummy_logger = DummyLogger()

    monkeypatch.setattr(config_impl, "setup_logger", lambda name: dummy_logger)
    recorded: Dict[tuple[str, str], str] = {}

    def fake_set_key(path, key, value):
        recorded[(path, key)] = value

    monkeypatch.setattr(config_impl, "set_key", fake_set_key)
    monkeypatch.setattr(config_impl, "find_dotenv", lambda: str(env_file))
    monkeypatch.setattr(config_impl, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setenv("OPENAI_API_KEY", "initial-key")
    monkeypatch.setenv("DEFAULT_PROVIDER", "OpenAI")
    monkeypatch.setenv("DEFAULT_MODEL", "gpt-4o")

    monkeypatch.setattr(
        ConfigManager,
        "_load_env_config",
        lambda self: {
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
            "DEFAULT_PROVIDER": os.getenv("DEFAULT_PROVIDER", "OpenAI"),
            "DEFAULT_MODEL": os.getenv("DEFAULT_MODEL", "gpt-4o"),
            "APP_ROOT": tmp_path.as_posix(),
        },
    )

    manager = ConfigManager()
    manager.providers.openai.set_speech_config(
        stt_provider="GPT-4o STT",
        tts_provider="GPT-4o Mini TTS",
        language="en",
        task="transcribe",
        initial_prompt="hello",
    )

    with open(manager._yaml_path, "r", encoding="utf-8") as fh:  # type: ignore[attr-defined]
        stored = yaml.safe_load(fh) or {}

    assert stored["OPENAI_SPEECH"] == {
        "stt_provider": "GPT-4o STT",
        "tts_provider": "GPT-4o Mini TTS",
        "language": "en",
        "task": "transcribe",
        "initial_prompt": "hello",
    }

    reloaded = ConfigManager()
    assert reloaded.config["OPENAI_SPEECH"] == stored["OPENAI_SPEECH"]
    assert reloaded.config["OPENAI_STT_PROVIDER"] == "GPT-4o STT"
    assert reloaded.config["OPENAI_TTS_PROVIDER"] == "GPT-4o Mini TTS"
    assert reloaded.config["OPENAI_LANGUAGE"] == "en"
    assert reloaded.config["OPENAI_TASK"] == "transcribe"
    assert reloaded.config["OPENAI_INITIAL_PROMPT"] == "hello"


def test_set_google_speech_settings_updates_state(config_manager):
    result = config_manager.providers.google.set_speech_settings(
        tts_voice="Wave",
        stt_language="en-US",
        auto_punctuation=False,
    )

    assert result == {
        "tts_voice": "Wave",
        "stt_language": "en-US",
        "auto_punctuation": False,
    }
    assert config_manager.yaml_config["GOOGLE_SPEECH"]["tts_voice"] == "Wave"
    assert config_manager.yaml_config["GOOGLE_SPEECH"]["auto_punctuation"] is False

    cleared = config_manager.providers.google.set_speech_settings(
        tts_voice=None,
        stt_language=None,
        auto_punctuation=None,
    )

    assert cleared == {
        "tts_voice": None,
        "stt_language": None,
        "auto_punctuation": None,
    }
    assert "GOOGLE_SPEECH" not in config_manager.yaml_config
    assert "GOOGLE_SPEECH" not in config_manager.config


def test_set_hf_token_updates_state(config_manager):
    config_manager.providers.update_api_key("HuggingFace", "hf-token")

    assert config_manager.config["HUGGINGFACE_API_KEY"] == "hf-token"
    assert os.environ["HUGGINGFACE_API_KEY"] == "hf-token"
    assert (
        config_manager._recorded_set_key[(config_manager._env_path, "HUGGINGFACE_API_KEY")]
        == "hf-token"
    )


def test_set_google_llm_settings_updates_state(config_manager):
    result = config_manager.providers.google.set_llm_settings(
        model="gemini-1.5-flash",
        temperature=0.55,
        top_p=0.9,
        top_k=32,
        candidate_count=2,
        max_output_tokens=16000,
        stop_sequences=["STOP"],
        safety_settings=[
            {"category": "HARM_CATEGORY_DEROGATORY", "threshold": "BLOCK_NONE"}
        ],
        response_mime_type="application/json",
        system_instruction="Respond in JSON.",
        function_call_mode="require",
        allowed_function_names=["tool_a", "tool_b"],
        seed=12345,
        response_logprobs=True,
    )

    assert result["model"] == "gemini-1.5-flash"
    stored = config_manager.providers.google.get_llm_settings()
    assert stored["temperature"] == 0.55
    assert stored["top_p"] == 0.9
    assert stored["top_k"] == 32
    assert stored["candidate_count"] == 2
    assert stored["max_output_tokens"] == 16000
    assert stored["stop_sequences"] == ["STOP"]
    assert stored["safety_settings"] == [
        {"category": "HARM_CATEGORY_DEROGATORY", "threshold": "BLOCK_NONE"}
    ]
    assert stored["response_mime_type"] == "application/json"
    assert stored["system_instruction"] == "Respond in JSON."
    assert stored["function_call_mode"] == "require"
    assert stored["allowed_function_names"] == ["tool_a", "tool_b"]
    assert stored["cached_allowed_function_names"] == ["tool_a", "tool_b"]
    assert stored["seed"] == 12345
    assert stored["response_logprobs"] is True
    assert config_manager.config["GOOGLE_LLM"]["top_k"] == 32


def test_get_available_providers_masks_secrets(config_manager):
    providers = config_manager.providers.get_available_providers()

    assert "OpenAI" in providers
    openai_payload = providers["OpenAI"]

    assert openai_payload["available"] is True
    assert openai_payload["length"] == len("initial-key")
    assert openai_payload["hint"] == "\u2022" * min(len("initial-key"), 8)
    assert "initial-key" not in json.dumps(openai_payload)

    for provider, payload in providers.items():
        if provider == "OpenAI":
            continue

        assert payload["available"] is False
        assert payload["length"] == 0
        assert payload["hint"] == ""


def test_set_google_llm_settings_normalizes_string_payloads(config_manager):
    snapshot = config_manager.providers.google.set_llm_settings(
        model="gemini-1.5-flash",
        temperature="0.4",
        top_p="0.8",
        top_k="27",
        candidate_count="2",
        max_output_tokens="4100",
        stop_sequences=["DONE"],
        stream="false",
        function_calling="0",
        response_schema='{"type": "object"}',
        response_mime_type="",
        seed="73",
        response_logprobs="true",
    )

    assert snapshot["temperature"] == 0.4
    assert snapshot["top_p"] == 0.8
    assert snapshot["top_k"] == 27
    assert snapshot["candidate_count"] == 2
    assert snapshot["max_output_tokens"] == 4100
    assert snapshot["stream"] is False
    assert snapshot["function_calling"] is False
    assert snapshot["response_mime_type"] == "application/json"
    assert snapshot["response_schema"] == {"type": "object"}
    assert snapshot["seed"] == 73
    assert snapshot["response_logprobs"] is True

    persisted = config_manager.providers.google.get_llm_settings()
    assert persisted["stream"] is False
    assert persisted["function_calling"] is False
    assert persisted["temperature"] == 0.4
    assert persisted["seed"] == 73


def test_set_google_llm_settings_autofills_json_mime_for_schema(config_manager):
    snapshot = config_manager.providers.google.set_llm_settings(
        model="gemini-1.5-flash",
        response_schema={"type": "object"},
        response_mime_type="",
    )

    assert snapshot["response_mime_type"] == "application/json"
    stored = config_manager.providers.google.get_llm_settings()
    assert stored["response_mime_type"] == "application/json"


def test_set_google_llm_settings_rejects_non_json_mime_with_schema(config_manager):
    with pytest.raises(ValueError):
        config_manager.providers.google.set_llm_settings(
            model="gemini-1.5-flash",
            response_schema={"type": "object"},
            response_mime_type="text/plain",
        )


def test_set_google_llm_settings_rejects_invalid_function_mode(config_manager):
    with pytest.raises(ValueError):
        config_manager.providers.google.set_llm_settings(
            model="gemini-1.5-flash",
            function_call_mode="unsupported",
        )


def test_get_google_llm_settings_returns_copy(config_manager):
    config_manager.providers.google.set_llm_settings(
        model="gemini-1.5-pro",
        stop_sequences=["DONE"],
        safety_settings=[
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_LOW_AND_ABOVE"}
        ],
        allowed_function_names=["alpha", "beta"],
    )

    retrieved = config_manager.providers.google.get_llm_settings()
    retrieved["stop_sequences"].append("NEW")
    retrieved["allowed_function_names"].append("gamma")
    retrieved["cached_allowed_function_names"].append("delta")
    assert config_manager.config["GOOGLE_LLM"]["stop_sequences"] == ["DONE"]
    assert config_manager.config["GOOGLE_LLM"]["allowed_function_names"] == [
        "alpha",
        "beta",
    ]
    assert config_manager.config["GOOGLE_LLM"]["cached_allowed_function_names"] == [
        "alpha",
        "beta",
    ]


def test_set_google_llm_settings_allows_clearing_max_output_tokens(config_manager):
    config_manager.providers.google.set_llm_settings(
        model="gemini-1.5-flash",
        max_output_tokens=4096,
        seed=321,
    )

    config_manager.providers.google.set_llm_settings(
        model="gemini-1.5-flash",
        max_output_tokens="",
        seed="",
    )

    snapshot = config_manager.providers.google.get_llm_settings()
    assert snapshot["max_output_tokens"] is None
    assert snapshot["seed"] is None


def test_set_google_llm_settings_rejects_negative_seed(config_manager):
    with pytest.raises(ValueError):
        config_manager.providers.google.set_llm_settings(
            model="gemini-1.5-flash",
            seed=-10,
        )


def test_set_google_llm_settings_caches_allowlist_when_disabled(config_manager):
    config_manager.providers.google.set_llm_settings(
        model="gemini-1.5-pro",
        allowed_function_names=["alpha", "beta"],
    )

    disabled = config_manager.providers.google.set_llm_settings(
        model="gemini-1.5-pro",
        function_calling=False,
        allowed_function_names=[],
        cached_allowed_function_names=["alpha", "beta"],
    )

    assert disabled["allowed_function_names"] == []
    assert disabled["cached_allowed_function_names"] == ["alpha", "beta"]

    re_enabled = config_manager.providers.google.set_llm_settings(
        model="gemini-1.5-pro",
        function_calling=True,
        function_call_mode="auto",
    )

    assert re_enabled["allowed_function_names"] == ["alpha", "beta"]
    assert re_enabled["cached_allowed_function_names"] == ["alpha", "beta"]


def test_set_openai_llm_settings_updates_state(config_manager):
    result = config_manager.providers.openai.set_llm_settings(
        model="gpt-4o-mini",
        temperature=0.75,
        top_p=0.95,
        frequency_penalty=0.25,
        presence_penalty=-0.5,
        max_tokens=2048,
        max_output_tokens=512,
        stream=False,
        function_calling=False,
        base_url=" https://example/v1 ",
        organization="org-42",
        reasoning_effort="high",
        audio_enabled=True,
        audio_voice="aria",
        audio_format="MP3",
    )

    assert result["model"] == "gpt-4o-mini"
    stored = config_manager.config["OPENAI_LLM"]
    assert stored["temperature"] == 0.75
    assert math.isclose(stored["top_p"], 0.95)
    assert math.isclose(stored["frequency_penalty"], 0.25)
    assert math.isclose(stored["presence_penalty"], -0.5)
    assert stored["max_tokens"] == 2048
    assert stored["max_output_tokens"] == 512
    assert stored["stream"] is False
    assert stored["function_calling"] is False
    assert stored["reasoning_effort"] == "high"
    assert stored["base_url"] == "https://example/v1"
    assert stored["organization"] == "org-42"
    assert stored["json_mode"] is False
    assert stored["json_schema"] is None
    assert stored["audio_enabled"] is True
    assert stored["audio_voice"] == "aria"
    assert stored["audio_format"] == "mp3"

    assert config_manager.config["DEFAULT_MODEL"] == "gpt-4o-mini"
    assert os.environ["DEFAULT_MODEL"] == "gpt-4o-mini"
    assert os.environ["OPENAI_BASE_URL"] == "https://example/v1"
    assert os.environ["OPENAI_ORGANIZATION"] == "org-42"
    assert (
        config_manager._recorded_set_key[(config_manager._env_path, "DEFAULT_MODEL")]
        == "gpt-4o-mini"
    )


def test_set_openai_llm_settings_clears_optional_fields(config_manager):
    config_manager.providers.openai.set_llm_settings(
        model="gpt-4o",
        base_url="https://custom/v1",
        organization="org-keep",
    )

    assert os.environ["OPENAI_BASE_URL"] == "https://custom/v1"
    assert os.environ["OPENAI_ORGANIZATION"] == "org-keep"

    config_manager.providers.openai.set_llm_settings(
        model="gpt-4o",
        base_url="  ",
        organization="",
    )

    stored = config_manager.config["OPENAI_LLM"]
    assert stored["base_url"] is None
    assert stored["organization"] is None
    assert "OPENAI_BASE_URL" not in os.environ
    assert "OPENAI_ORGANIZATION" not in os.environ


def test_set_openai_llm_settings_rejects_unknown_effort(config_manager):
    with pytest.raises(ValueError):
        config_manager.providers.openai.set_llm_settings(
            model="gpt-4o",
            reasoning_effort="extreme",
        )


def test_get_openai_llm_settings_includes_sampling_defaults(config_manager):
    snapshot = config_manager.providers.openai.get_llm_settings()

    assert snapshot["temperature"] == 0.0
    assert snapshot["top_p"] == 1.0
    assert snapshot["frequency_penalty"] == 0.0
    assert snapshot["presence_penalty"] == 0.0
    assert snapshot["max_tokens"] == 4000
    assert snapshot["function_calling"] is True
    assert snapshot["parallel_tool_calls"] is True
    assert snapshot["max_output_tokens"] is None
    assert snapshot["reasoning_effort"] == "medium"
    assert snapshot["json_mode"] is False
    assert snapshot["json_schema"] is None
    assert snapshot["tool_choice"] is None
    assert snapshot["enable_code_interpreter"] is False
    assert snapshot["enable_file_search"] is False
    assert snapshot["audio_enabled"] is False
    assert snapshot["audio_voice"] == "alloy"
    assert snapshot["audio_format"] == "wav"


def test_set_openai_llm_settings_persists_json_mode(config_manager):
    config_manager.providers.openai.set_llm_settings(model="gpt-4o", json_mode=True)

    stored = config_manager.config["OPENAI_LLM"]
    assert stored["json_mode"] is True

    config_manager.providers.openai.set_llm_settings(model="gpt-4o", json_mode=False)
    stored = config_manager.config["OPENAI_LLM"]
    assert stored["json_mode"] is False


def test_set_openai_llm_settings_handles_json_schema(config_manager):
    schema_payload = {
        "name": "atlas_response",
        "schema": {
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
            "additionalProperties": False,
        },
        "strict": True,
    }

    config_manager.providers.openai.set_llm_settings(
        model="gpt-4o",
        json_schema=json.dumps(schema_payload),
    )

    stored = config_manager.config["OPENAI_LLM"]
    assert stored["json_schema"]["name"] == "atlas_response"
    assert stored["json_schema"]["schema"]["required"] == ["ok"]
    assert stored["json_mode"] is False

    config_manager.providers.openai.set_llm_settings(model="gpt-4o", json_schema=" ")
    stored = config_manager.config["OPENAI_LLM"]
    assert stored["json_schema"] is None


def test_set_openai_llm_settings_rejects_invalid_schema(config_manager):
    with pytest.raises(ValueError):
        config_manager.providers.openai.set_llm_settings(
            model="gpt-4o",
            json_schema="not json",
        )

    with pytest.raises(ValueError):
        config_manager.providers.openai.set_llm_settings(
            model="gpt-4o",
            json_schema={"name": "bad"},
        )


def test_get_mistral_llm_settings_includes_json_defaults(config_manager):
    snapshot = config_manager.providers.mistral.get_llm_settings()

    assert snapshot["json_mode"] is False
    assert snapshot["json_schema"] is None
    assert snapshot["max_retries"] == 3
    assert snapshot["retry_min_seconds"] == 4
    assert snapshot["retry_max_seconds"] == 10
    assert snapshot["base_url"] is None
    assert snapshot["prompt_mode"] is None


def test_set_mistral_llm_settings_handles_json_options(config_manager):
    schema_payload = {
        "name": "atlas_response",
        "schema": {
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
        },
    }

    config_manager.providers.mistral.set_llm_settings(
        model="mistral-large-latest",
        json_mode=True,
        json_schema=json.dumps(schema_payload),
    )

    stored = config_manager.config["MISTRAL_LLM"]
    assert stored["json_mode"] is True
    assert stored["json_schema"]["schema"]["required"] == ["ok"]

    config_manager.providers.mistral.set_llm_settings(
        model="mistral-large-latest",
        json_schema="  ",
    )

    stored = config_manager.config["MISTRAL_LLM"]
    assert stored["json_schema"] is None


def test_set_mistral_llm_settings_tracks_prompt_mode(config_manager):
    saved = config_manager.providers.mistral.set_llm_settings(
        model="mistral-large-latest",
        prompt_mode="reasoning",
    )

    assert saved["prompt_mode"] == "reasoning"
    snapshot = config_manager.providers.mistral.get_llm_settings()
    assert snapshot["prompt_mode"] == "reasoning"
    config_manager.providers.mistral.set_llm_settings(
        model="mistral-large-latest",
        prompt_mode="  ",
    )
    assert config_manager.providers.mistral.get_llm_settings()["prompt_mode"] is None


def test_set_mistral_llm_settings_rejects_invalid_prompt_mode(config_manager):
    with pytest.raises(ValueError):
        config_manager.providers.mistral.set_llm_settings(
            model="mistral-large-latest",
            prompt_mode="unsupported",
        )


def test_set_mistral_llm_settings_updates_retry_policy(config_manager):
    config_manager.providers.mistral.set_llm_settings(
        model="mistral-large-latest",
        max_retries=6,
        retry_min_seconds=2,
        retry_max_seconds=9,
    )

    stored = config_manager.config["MISTRAL_LLM"]
    assert stored["max_retries"] == 6
    assert stored["retry_min_seconds"] == 2
    assert stored["retry_max_seconds"] == 9


def test_set_mistral_llm_settings_tracks_tool_preferences(config_manager):
    config_manager.providers.mistral.set_llm_settings(
        model="mistral-large-latest",
        parallel_tool_calls=False,
        tool_choice="required",
    )

    stored = config_manager.config["MISTRAL_LLM"]
    assert stored["parallel_tool_calls"] is False
    assert stored["tool_choice"] == "required"


def test_set_mistral_llm_settings_handles_base_url_round_trip(config_manager):
    saved = config_manager.providers.mistral.set_llm_settings(
        model="mistral-large-latest",
        base_url="https://alt.mistral.ai/v1",
    )

    assert saved["base_url"] == "https://alt.mistral.ai/v1"
    assert (
        config_manager.providers.mistral.get_llm_settings()["base_url"]
        == "https://alt.mistral.ai/v1"
    )

    updated = config_manager.providers.mistral.set_llm_settings(
        model="mistral-large-latest",
        base_url="  ",
    )

    assert updated["base_url"] is None
    assert config_manager.providers.mistral.get_llm_settings()["base_url"] is None


def test_set_mistral_llm_settings_rejects_invalid_base_url(config_manager):
    with pytest.raises(ValueError):
        config_manager.providers.mistral.set_llm_settings(
            model="mistral-large-latest",
            base_url="ftp://example.com",
        )

    config_manager.providers.mistral.set_llm_settings(
        model="mistral-large-latest",
        tool_choice="none",
    )

    stored = config_manager.config["MISTRAL_LLM"]
    assert stored["tool_choice"] == "none"

    config_manager.providers.mistral.set_llm_settings(
        model="mistral-large-latest",
        tool_choice={"type": "function", "name": "math"},
    )

    stored = config_manager.config["MISTRAL_LLM"]
    assert stored["tool_choice"] == {"type": "function", "name": "math"}

    config_manager.providers.mistral.set_llm_settings(
        model="mistral-large-latest",
        tool_choice="  ",
    )

    stored = config_manager.config["MISTRAL_LLM"]
    assert stored["tool_choice"] is None


def test_set_openai_llm_settings_tracks_tool_preferences(config_manager):
    config_manager.providers.openai.set_llm_settings(
        model="gpt-4o",
        parallel_tool_calls=False,
        tool_choice="required",
    )

    stored = config_manager.config["OPENAI_LLM"]
    assert stored["parallel_tool_calls"] is False
    assert stored["tool_choice"] == "required"

    config_manager.providers.openai.set_llm_settings(
        model="gpt-4o",
        tool_choice="none",
    )
    stored = config_manager.config["OPENAI_LLM"]
    assert stored["parallel_tool_calls"] is False
    assert stored["tool_choice"] == "none"

    config_manager.providers.openai.set_llm_settings(
        model="gpt-4o",
        tool_choice=" ",
    )
    stored = config_manager.config["OPENAI_LLM"]
    assert stored["tool_choice"] is None

    config_manager.providers.openai.set_llm_settings(
        model="gpt-4o",
        enable_code_interpreter=True,
        enable_file_search=True,
    )
    stored = config_manager.config["OPENAI_LLM"]
    assert stored["enable_code_interpreter"] is True
    assert stored["enable_file_search"] is True

    config_manager.providers.openai.set_llm_settings(
        model="gpt-4o",
        function_calling=False,
    )
    stored = config_manager.config["OPENAI_LLM"]
    assert stored["enable_code_interpreter"] is False
    assert stored["enable_file_search"] is False
