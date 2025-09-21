"""Unit tests for the SpeechManager TTS summary helper."""

from types import MethodType
import logging
import os
import sys
import types

import pytest


# Provide a lightweight stub for google.cloud.speech to satisfy imports during testing.
if "google" not in sys.modules:
    google_module = types.ModuleType("google")
    google_module.__path__ = []  # Mark as package

    cloud_module = types.ModuleType("google.cloud")
    cloud_module.__path__ = []

    speech_module = types.ModuleType("google.cloud.speech_v1p1beta1")

    class _DummyVoiceSelectionParams:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class _DummySpeechClient:
        def __init__(self, *args, **kwargs):
            pass

        def synthesize_speech(self, *args, **kwargs):
            class _Response:
                audio_content = b""

            return _Response()

        def list_voices(self):
            class _Response:
                voices = []

            return _Response()

    class _DummySynthesisInput:
        def __init__(self, *args, **kwargs):
            pass

    class _DummyAudioConfig:
        def __init__(self, *args, **kwargs):
            pass

    class _DummyAudioEncoding:
        MP3 = "MP3"

    class _DummySsmlVoiceGender:
        NEUTRAL = 0

        def __call__(self, value):
            class _Result:
                name = "NEUTRAL"

            return _Result()

    speech_module.VoiceSelectionParams = _DummyVoiceSelectionParams
    speech_module.SpeechClient = _DummySpeechClient
    speech_module.SynthesisInput = _DummySynthesisInput
    speech_module.AudioConfig = _DummyAudioConfig
    speech_module.AudioEncoding = _DummyAudioEncoding
    speech_module.SsmlVoiceGender = _DummySsmlVoiceGender()

    cloud_module.speech_v1p1beta1 = speech_module
    google_module.cloud = cloud_module

    sys.modules["google"] = google_module
    sys.modules["google.cloud"] = cloud_module
    sys.modules["google.cloud.speech_v1p1beta1"] = speech_module


if "pygame" not in sys.modules:
    music = types.SimpleNamespace(
        load=lambda *args, **kwargs: None,
        play=lambda *args, **kwargs: None,
        get_busy=lambda: False,
    )
    mixer = types.SimpleNamespace(init=lambda *args, **kwargs: None, music=music)

    class _DummyClock:
        def tick(self, *args, **kwargs):
            pass

    time_module = types.SimpleNamespace(Clock=lambda: _DummyClock())

    pygame_module = types.ModuleType("pygame")
    pygame_module.mixer = mixer
    pygame_module.time = time_module

    sys.modules["pygame"] = pygame_module


if "requests" not in sys.modules:
    class _DummyResponse:
        ok = True
        text = ""

        def json(self):
            return {}

        def iter_content(self, chunk_size=1024):
            return iter(())

    def _dummy_request(*args, **kwargs):
        return _DummyResponse()

    requests_module = types.ModuleType("requests")
    requests_module.get = _dummy_request
    requests_module.post = _dummy_request

    sys.modules["requests"] = requests_module


if "dotenv" not in sys.modules:
    dotenv_module = types.ModuleType("dotenv")

    def _noop(*args, **kwargs):
        return None

    dotenv_module.load_dotenv = _noop
    dotenv_module.set_key = _noop
    dotenv_module.find_dotenv = lambda *args, **kwargs: ""

    sys.modules["dotenv"] = dotenv_module


if "yaml" not in sys.modules:
    yaml_module = types.ModuleType("yaml")
    yaml_module.safe_load = lambda *args, **kwargs: {}

    sys.modules["yaml"] = yaml_module


if "openai" not in sys.modules:
    class _DummyAudio:
        @staticmethod
        def create(*args, **kwargs):
            return {"data": b""}

        @staticmethod
        def transcribe(*args, **kwargs):
            return {"text": ""}

    openai_module = types.ModuleType("openai")
    openai_module.Audio = _DummyAudio
    openai_module.api_key = ""

    sys.modules["openai"] = openai_module


if "sounddevice" not in sys.modules:
    class _DummyInputStream:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

        def stop(self):
            pass

        def close(self):
            pass

    sounddevice_module = types.ModuleType("sounddevice")
    sounddevice_module.InputStream = _DummyInputStream

    sys.modules["sounddevice"] = sounddevice_module


if "numpy" not in sys.modules:
    numpy_module = types.ModuleType("numpy")
    numpy_module.concatenate = lambda frames: frames

    sys.modules["numpy"] = numpy_module


if "soundfile" not in sys.modules:
    soundfile_module = types.ModuleType("soundfile")
    soundfile_module.write = lambda *args, **kwargs: None

    sys.modules["soundfile"] = soundfile_module


from modules.Speech_Services.speech_manager import (
    SpeechManager,
    get_openai_language_options,
    get_openai_stt_provider_options,
    get_openai_task_options,
    get_openai_tts_provider_options,
    prepare_openai_settings,
)


class _DummyConfig:
    def __init__(self):
        self.config = {"OPENAI_API_KEY": "stored-key"}
        self.yaml_config = {}
        self._tts_enabled = True
        self.raise_openai_error = False
        self.raise_google_error = False
        self.openai_calls = []
        self.google_credentials = None

    def get_tts_enabled(self):
        return self._tts_enabled

    def set_tts_enabled(self, value):
        self._tts_enabled = value
        self.config['TTS_ENABLED'] = value

    def get_config(self, key, default=None):
        return self.config.get(key, default)

    def set_openai_speech_config(
        self,
        *,
        api_key=None,
        stt_provider=None,
        tts_provider=None,
        language=None,
        task=None,
        initial_prompt=None,
    ):
        if self.raise_openai_error:
            raise RuntimeError("persist failed")
        self.openai_calls.append(
            {
                "api_key": api_key,
                "stt_provider": stt_provider,
                "tts_provider": tts_provider,
                "language": language,
                "task": task,
                "initial_prompt": initial_prompt,
            }
        )
        if api_key is not None:
            self.config['OPENAI_API_KEY'] = api_key
        if stt_provider is not None:
            self.config['OPENAI_STT_PROVIDER'] = stt_provider
        if tts_provider is not None:
            self.config['OPENAI_TTS_PROVIDER'] = tts_provider
        if language is not None:
            self.config['OPENAI_LANGUAGE'] = language
        if task is not None:
            self.config['OPENAI_TASK'] = task
        if initial_prompt is not None:
            self.config['OPENAI_INITIAL_PROMPT'] = initial_prompt

    def set_google_credentials(self, credentials_path: str):
        if self.raise_google_error:
            raise RuntimeError("persist failed")
        self.google_credentials = credentials_path
        self.config['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path


class _SpeechManagerForTest(SpeechManager):
    def initialize_services(self):
        # Avoid heavy provider initialization during tests.
        self.tts_services = {}
        self.stt_services = {}
        self.active_tts = None
        self._active_stt_instance = None
        self._active_stt_key = None


@pytest.fixture
def speech_manager():
    return _SpeechManagerForTest(_DummyConfig())


def test_get_tts_provider_names_returns_ordered_copy(speech_manager):
    speech_manager.tts_services["alpha"] = object()
    speech_manager.tts_services["beta"] = object()

    names = speech_manager.get_tts_provider_names()

    assert names == ("alpha", "beta")

    speech_manager.tts_services["gamma"] = object()

    assert names == ("alpha", "beta")


def test_resolve_tts_provider_prefers_registered_choice(speech_manager):
    speech_manager.tts_services["eleven_labs"] = object()
    speech_manager.tts_services["custom"] = object()

    resolved = speech_manager.resolve_tts_provider("custom")

    assert resolved == "custom"


def test_resolve_tts_provider_falls_back_to_eleven_labs(speech_manager):
    speech_manager.tts_services["eleven_labs"] = object()
    speech_manager.tts_services["google"] = object()

    resolved = speech_manager.resolve_tts_provider("missing")

    assert resolved == "eleven_labs"


def test_resolve_tts_provider_returns_first_available_when_no_fallback(speech_manager):
    speech_manager.tts_services["google"] = object()
    speech_manager.tts_services["second"] = object()

    resolved = speech_manager.resolve_tts_provider(None)

    assert resolved == "google"


def test_resolve_tts_provider_handles_no_services(speech_manager):
    assert speech_manager.resolve_tts_provider("whatever") is None


def test_get_stt_provider_names_returns_ordered_copy(speech_manager):
    speech_manager.stt_services["delta"] = object()
    speech_manager.stt_services["epsilon"] = object()

    names = speech_manager.get_stt_provider_names()

    assert names == ("delta", "epsilon")

    speech_manager.stt_services["zeta"] = object()

    assert names == ("delta", "epsilon")


def test_get_default_tts_provider_index_tracks_provider(speech_manager):
    speech_manager.tts_services["one"] = object()
    speech_manager.tts_services["two"] = object()
    speech_manager.set_default_tts_provider("two")

    assert speech_manager.get_default_tts_provider_index() == 1

    speech_manager.remove_tts_provider("two")

    assert speech_manager.get_default_tts_provider_index() is None


def test_get_default_stt_provider_index_tracks_provider(speech_manager):
    speech_manager.stt_services["uno"] = object()
    speech_manager.stt_services["dos"] = object()
    speech_manager.set_default_stt_provider("dos")

    assert speech_manager.get_default_stt_provider_index() == 1

    speech_manager.remove_stt_provider("dos")

    assert speech_manager.get_default_stt_provider_index() is None


def test_summary_prefers_get_current_voice_dict(speech_manager):
    class Provider:
        def get_current_voice(self):
            return {"name": "Alice", "voice_id": "v1"}

    provider = Provider()
    speech_manager.tts_services["dict_provider"] = provider
    speech_manager.active_tts = provider

    provider_name, voice_label = speech_manager.get_active_tts_summary()

    assert provider_name == "dict_provider"
    assert voice_label == "Alice"


def test_summary_uses_voice_ids_metadata(speech_manager):
    class Provider:
        def __init__(self):
            self.voice_ids = [{"name": "Beta", "voice_id": "v2"}]

    provider = Provider()
    speech_manager.tts_services["voice_ids_provider"] = provider
    speech_manager.active_tts = provider

    provider_name, voice_label = speech_manager.get_active_tts_summary()

    assert provider_name == "voice_ids_provider"
    assert voice_label == "Beta"


def test_summary_reads_voice_attribute_objects(speech_manager):
    class Voice:
        name = "Gamma"

    class Provider:
        def __init__(self):
            self.voice = Voice()

    provider = Provider()
    speech_manager.tts_services["voice_attr_provider"] = provider
    speech_manager.active_tts = provider

    provider_name, voice_label = speech_manager.get_active_tts_summary()

    assert provider_name == "voice_attr_provider"
    assert voice_label == "Gamma"


def test_summary_supports_unregistered_active_service(speech_manager):
    class Voice:
        name = "Fallback"

    class Provider:
        def __init__(self):
            self.voice = Voice()

    speech_manager.active_tts = Provider()

    provider_name, voice_label = speech_manager.get_active_tts_summary()

    assert provider_name == "None"
    assert voice_label == "Fallback"


def test_summary_logs_missing_provider(speech_manager):
    logger = logging.getLogger("speech_manager.py")
    captured_messages = []

    class _CaptureHandler(logging.Handler):
        def emit(self, record):
            captured_messages.append(record.getMessage())

    handler = _CaptureHandler(level=logging.ERROR)
    logger.addHandler(handler)
    try:
        speech_manager.get_default_tts_provider = MethodType(lambda self: "ghost", speech_manager)

        provider_name, voice_label = speech_manager.get_active_tts_summary()

        assert provider_name == "ghost"
        assert voice_label == "Not Set"
        assert any("Active TTS provider 'ghost' is not registered." in message for message in captured_messages)
    finally:
        logger.removeHandler(handler)


def test_set_default_speech_providers_updates_state(speech_manager):
    speech_manager.tts_services["alpha"] = object()
    speech_manager.stt_services["beta"] = object()

    speech_manager.set_default_speech_providers(tts_provider="alpha", stt_provider="beta")

    assert speech_manager.get_default_tts_provider() == "alpha"
    assert speech_manager.get_default_stt_provider() == "beta"
    assert speech_manager.config_manager.config["DEFAULT_TTS_PROVIDER"] == "alpha"
    assert speech_manager.config_manager.config["DEFAULT_STT_PROVIDER"] == "beta"


def test_disable_stt_clears_active_provider(speech_manager):
    stt_provider = object()
    speech_manager.stt_services["gamma"] = stt_provider
    speech_manager.set_default_speech_providers(stt_provider="gamma")

    speech_manager.disable_stt()

    assert speech_manager.get_default_stt_provider() is None
    assert speech_manager.active_stt is None
    assert speech_manager.config_manager.config["DEFAULT_STT_PROVIDER"] is None


def test_configure_defaults_updates_state_and_persists(speech_manager):
    class DummyTTS:
        def __init__(self):
            self.enabled = None

        def set_tts(self, value):
            self.enabled = value

        def get_tts(self):
            return self.enabled

    dummy_tts = DummyTTS()
    speech_manager.tts_services["alpha"] = dummy_tts
    speech_manager.stt_services["beta"] = object()

    speech_manager.configure_defaults(
        tts_enabled=True,
        tts_provider="alpha",
        stt_enabled=True,
        stt_provider="beta",
    )

    assert speech_manager.config_manager.get_tts_enabled() is True
    assert speech_manager.config_manager.config["TTS_ENABLED"] is True
    assert dummy_tts.enabled is True
    assert speech_manager.get_default_tts_provider() == "alpha"
    assert speech_manager.get_default_stt_provider() == "beta"
    assert speech_manager.config_manager.config["DEFAULT_TTS_PROVIDER"] == "alpha"
    assert speech_manager.config_manager.config["DEFAULT_STT_PROVIDER"] == "beta"


def test_configure_defaults_disables_stt_when_requested(speech_manager):
    class DummyTTS:
        def __init__(self):
            self.enabled = None

        def set_tts(self, value):
            self.enabled = value

        def get_tts(self):
            return self.enabled

    dummy_tts = DummyTTS()
    speech_manager.tts_services["alpha"] = dummy_tts
    speech_manager.stt_services["beta"] = object()
    speech_manager.set_default_speech_providers(stt_provider="beta")

    speech_manager.configure_defaults(
        tts_enabled=False,
        tts_provider="alpha",
        stt_enabled=False,
        stt_provider="beta",
    )

    assert speech_manager.config_manager.get_tts_enabled() is False
    assert speech_manager.config_manager.config["TTS_ENABLED"] is False
    assert dummy_tts.enabled is False
    assert speech_manager.get_default_tts_provider() == "alpha"
    assert speech_manager.get_default_stt_provider() is None
    assert speech_manager.config_manager.config["DEFAULT_TTS_PROVIDER"] == "alpha"
    assert speech_manager.config_manager.config["DEFAULT_STT_PROVIDER"] is None


def test_set_openai_speech_config_registers_providers(speech_manager, monkeypatch):
    stt_instance = object()
    tts_instance = object()
    speech_manager._openai_stt_factories = {"GPT-4o STT": lambda: stt_instance}
    speech_manager._openai_tts_factories = {"GPT-4o Mini TTS": lambda: tts_instance}
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    speech_manager.set_openai_speech_config(
        api_key="unit-test-key",
        stt_provider="GPT-4o STT",
        language="en",
        task="transcribe",
        initial_prompt="Hello",
        tts_provider="GPT-4o Mini TTS",
    )

    assert speech_manager.stt_services["openai_stt"] is stt_instance
    assert speech_manager.tts_services["openai_tts"] is tts_instance
    assert speech_manager.get_default_stt_provider() == "openai_stt"
    assert speech_manager.get_default_tts_provider() == "openai_tts"
    assert speech_manager.config_manager.config["OPENAI_TASK"] == "transcribe"
    assert speech_manager.config_manager.openai_calls[-1]["initial_prompt"] == "Hello"
    assert os.environ["OPENAI_API_KEY"] == "unit-test-key"


def test_set_openai_speech_config_persistence_failure_rolls_back(speech_manager, monkeypatch):
    existing_stt = object()
    existing_tts = object()
    speech_manager.stt_services["openai_stt"] = existing_stt
    speech_manager.tts_services["openai_tts"] = existing_tts
    speech_manager.set_default_speech_providers(tts_provider="openai_tts", stt_provider="openai_stt")
    speech_manager._openai_stt_factories = {"GPT-4o STT": lambda: object()}
    speech_manager.config_manager.raise_openai_error = True
    monkeypatch.setenv("OPENAI_API_KEY", "previous")

    with pytest.raises(RuntimeError):
        speech_manager.set_openai_speech_config(
            api_key="new-key",
            stt_provider="GPT-4o STT",
            language="en",
            task="translate",
            initial_prompt=None,
            tts_provider=None,
        )

    assert speech_manager.stt_services["openai_stt"] is existing_stt
    assert speech_manager.tts_services["openai_tts"] is existing_tts
    assert speech_manager.get_default_stt_provider() == "openai_stt"
    assert os.environ["OPENAI_API_KEY"] == "previous"


def test_set_openai_speech_config_factory_failure_rolls_back(speech_manager, monkeypatch):
    def fail():
        raise RuntimeError("factory error")

    speech_manager._openai_stt_factories = {"GPT-4o STT": fail}
    monkeypatch.setenv("OPENAI_API_KEY", "stored")

    with pytest.raises(RuntimeError):
        speech_manager.set_openai_speech_config(
            api_key=None,
            stt_provider="GPT-4o STT",
            language=None,
            task=None,
            initial_prompt=None,
            tts_provider=None,
        )

    assert speech_manager.config_manager.openai_calls == []


def test_prepare_openai_settings_normalizes_and_validates():
    payload = {
        "api_key": "  secret-key  ",
        "stt_provider": "gPt-4O stt",
        "language": "English (en)",
        "task": "Transcribe",
        "initial_prompt": "  Hello Atlas  ",
        "tts_provider": "gpt-4o mini tts",
    }

    prepared = prepare_openai_settings(payload)

    assert prepared["api_key"] == "secret-key"
    assert prepared["stt_provider"] == "GPT-4o STT"
    assert prepared["language"] == "en"
    assert prepared["task"] == "transcribe"
    assert prepared["initial_prompt"] == "Hello Atlas"
    assert prepared["tts_provider"] == "GPT-4o Mini TTS"


def test_prepare_openai_settings_raises_on_invalid_choice():
    with pytest.raises(ValueError) as excinfo:
        prepare_openai_settings({"stt_provider": "Invalid Provider"})

    assert "Invalid Provider" in str(excinfo.value)


def test_openai_option_exports_include_expected_entries():
    stt_labels = [label for label, _ in get_openai_stt_provider_options()]
    tts_labels = [label for label, _ in get_openai_tts_provider_options()]
    languages = [label for label, _ in get_openai_language_options()]
    tasks = [label for label, _ in get_openai_task_options()]

    assert "Whisper Online" in stt_labels
    assert "GPT-4o Mini TTS" in tts_labels
    assert "Auto" in languages
    assert any(label.lower() == "transcribe" for label in tasks)


def test_set_google_credentials_reconfigures_providers(speech_manager, monkeypatch):
    new_tts = object()
    new_stt = object()
    speech_manager._google_tts_factory = lambda: new_tts
    speech_manager._google_stt_factory = lambda: new_stt
    speech_manager.tts_services['google'] = object()
    speech_manager.stt_services['google'] = object()
    speech_manager.set_default_tts_provider('google')
    speech_manager.set_default_stt_provider('google')
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)

    speech_manager.set_google_credentials("/tmp/google.json")

    assert speech_manager.tts_services['google'] is new_tts
    assert speech_manager.stt_services['google'] is new_stt
    assert speech_manager.get_default_tts_provider() == 'google'
    assert speech_manager.get_default_stt_provider() == 'google'
    assert speech_manager.config_manager.google_credentials == "/tmp/google.json"
    assert os.environ["GOOGLE_APPLICATION_CREDENTIALS"] == "/tmp/google.json"


def test_set_google_credentials_persist_failure_rolls_back(speech_manager, monkeypatch):
    existing_tts = object()
    existing_stt = object()
    speech_manager.tts_services['google'] = existing_tts
    speech_manager.stt_services['google'] = existing_stt
    speech_manager.set_default_tts_provider('google')
    speech_manager.set_default_stt_provider('google')
    speech_manager._google_tts_factory = lambda: object()
    speech_manager._google_stt_factory = lambda: object()
    speech_manager.config_manager.raise_google_error = True
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "old.json")

    with pytest.raises(RuntimeError):
        speech_manager.set_google_credentials("/tmp/new.json")

    assert speech_manager.tts_services['google'] is existing_tts
    assert speech_manager.stt_services['google'] is existing_stt
    assert os.environ["GOOGLE_APPLICATION_CREDENTIALS"] == "old.json"


def test_set_google_credentials_factory_failure_rolls_back(speech_manager, monkeypatch):
    def fail():
        raise RuntimeError("boom")

    speech_manager._google_tts_factory = fail
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "old.json")

    with pytest.raises(RuntimeError):
        speech_manager.set_google_credentials("/tmp/new.json")

    assert os.environ["GOOGLE_APPLICATION_CREDENTIALS"] == "old.json"
