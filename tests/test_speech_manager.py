"""Unit tests for the SpeechManager TTS summary helper."""

import asyncio
from concurrent.futures import Future
from types import MethodType
import logging
import os
import sys
import types
from unittest.mock import mock_open

import pytest


# Provide a lightweight stub for google.cloud.texttospeech to satisfy imports during testing.
if "google.cloud.texttospeech" not in sys.modules:
    google_module = sys.modules.get("google")
    if google_module is None:
        google_module = types.ModuleType("google")
        google_module.__path__ = []  # Mark as package
        sys.modules["google"] = google_module

    cloud_module = getattr(google_module, "cloud", None)
    if cloud_module is None:
        cloud_module = types.ModuleType("google.cloud")
        cloud_module.__path__ = []
        google_module.cloud = cloud_module
        sys.modules["google.cloud"] = cloud_module

    texttospeech_module = types.ModuleType("google.cloud.texttospeech")

    class _DummyVoiceSelectionParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _DummyTextToSpeechClient:
        synthesize_calls = []
        list_voices_calls = []
        voices_response = []

        def __init__(self, *args, **kwargs):
            pass

        def synthesize_speech(self, *args, **kwargs):
            self.__class__.synthesize_calls.append((args, kwargs))
            return types.SimpleNamespace(audio_content=b"")

        def list_voices(self):
            self.__class__.list_voices_calls.append(())
            return types.SimpleNamespace(voices=list(self.__class__.voices_response))

    class _DummySynthesisInput:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class _DummyAudioConfig:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class _DummyAudioEncoding:
        MP3 = "MP3"

    class _DummySsmlVoiceGender:
        SSML_VOICE_GENDER_UNSPECIFIED = 0
        MALE = 1
        FEMALE = 2
        NEUTRAL = 3

        def __call__(self, value):
            mapping = {
                self.SSML_VOICE_GENDER_UNSPECIFIED: "SSML_VOICE_GENDER_UNSPECIFIED",
                self.MALE: "MALE",
                self.FEMALE: "FEMALE",
                self.NEUTRAL: "NEUTRAL",
            }
            return types.SimpleNamespace(name=mapping.get(value, "SSML_VOICE_GENDER_UNSPECIFIED"))

    def _reset_stub_state():
        _DummyTextToSpeechClient.synthesize_calls.clear()
        _DummyTextToSpeechClient.list_voices_calls.clear()
        _DummyTextToSpeechClient.voices_response = []

    def _set_voices(voices):
        _DummyTextToSpeechClient.voices_response = voices

    texttospeech_module.VoiceSelectionParams = _DummyVoiceSelectionParams
    texttospeech_module.TextToSpeechClient = _DummyTextToSpeechClient
    texttospeech_module.SynthesisInput = _DummySynthesisInput
    texttospeech_module.AudioConfig = _DummyAudioConfig
    texttospeech_module.AudioEncoding = _DummyAudioEncoding
    texttospeech_module.SsmlVoiceGender = _DummySsmlVoiceGender()
    texttospeech_module._reset_stub_state = _reset_stub_state
    texttospeech_module._set_voices = _set_voices

    cloud_module.texttospeech = texttospeech_module
    sys.modules["google.cloud.texttospeech"] = texttospeech_module


speech_module = types.ModuleType("google.cloud.speech_v1p1beta1")


class _DummyRecognitionConfig:
    class AudioEncoding:
        LINEAR16 = "LINEAR16"

    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class _DummyRecognitionAudio:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _DummySpeechClient:
    def __init__(self, *args, **kwargs):
        pass

    def recognize(self, *args, **kwargs):
        return types.SimpleNamespace(results=[])


speech_module.RecognitionConfig = _DummyRecognitionConfig
speech_module.RecognitionAudio = _DummyRecognitionAudio
speech_module.SpeechClient = _DummySpeechClient

google_module = sys.modules.setdefault("google", types.ModuleType("google"))
google_module.__path__ = getattr(google_module, "__path__", [])
cloud_module = getattr(google_module, "cloud", None)
if cloud_module is None:
    cloud_module = types.ModuleType("google.cloud")
    cloud_module.__path__ = []
    google_module.cloud = cloud_module
    sys.modules["google.cloud"] = cloud_module

cloud_module.speech_v1p1beta1 = speech_module
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
    requests_module.__path__ = []

    exceptions_module = types.ModuleType("requests.exceptions")

    class _RequestException(Exception):
        """Stub base class mirroring requests.exceptions.RequestException."""

    class _HTTPError(_RequestException):
        """Stub HTTP error matching requests.exceptions.HTTPError."""

    exceptions_module.RequestException = _RequestException
    exceptions_module.HTTPError = _HTTPError

    requests_module.exceptions = exceptions_module
    class _DummyResponse:
        status_code = 200
        text = ""
        content = b""

        def json(self):
            return {}

    requests_module.HTTPError = exceptions_module.HTTPError
    requests_module.RequestException = exceptions_module.RequestException
    requests_module.Response = _DummyResponse

    adapters_module = types.ModuleType("requests.adapters")

    class _HTTPAdapter:
        """Stub adapter mirroring requests.adapters.HTTPAdapter."""

        def __init__(self, *args, **kwargs):
            pass

        def send(self, request, **kwargs):  # pragma: no cover - stub helper
            return request

    adapters_module.HTTPAdapter = _HTTPAdapter
    requests_module.adapters = adapters_module

    sys.modules["requests"] = requests_module
    sys.modules["requests.exceptions"] = exceptions_module
    sys.modules["requests.adapters"] = adapters_module


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
            self.args = args
            self.kwargs = kwargs

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

from modules.Speech_Services.elevenlabs_tts import ElevenLabsTTS


class _DummyConfig:
    def __init__(self):
        self.config = {"OPENAI_API_KEY": "stored-key"}
        self.yaml_config = {}
        self._tts_enabled = True
        self.raise_openai_error = False
        self.raise_google_error = False
        self.openai_calls = []
        self.google_credentials = None
        self.google_speech_settings = {}

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

    def get_google_speech_settings(self):
        return dict(self.google_speech_settings)

    def set_google_speech_settings(
        self,
        *,
        tts_voice=None,
        stt_language=None,
        auto_punctuation=None,
    ):
        if self.raise_google_error:
            raise RuntimeError("persist failed")
        self.google_speech_settings = {
            "tts_voice": tts_voice,
            "stt_language": stt_language,
            "auto_punctuation": auto_punctuation,
        }
        self.config['GOOGLE_SPEECH'] = dict(self.google_speech_settings)
        return self.google_speech_settings


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


@pytest.fixture(autouse=True)
def reset_texttospeech_stub():
    from google.cloud import texttospeech

    if hasattr(texttospeech, "_reset_stub_state"):
        texttospeech._reset_stub_state()


def test_initialize_schedules_default_tts_bootstrap(speech_manager, monkeypatch):
    sentinel_future = Future()
    calls = []

    def fake_start(self, provider_key):
        calls.append(provider_key)
        self._tts_initialization_futures[provider_key] = sentinel_future
        return sentinel_future

    speech_manager._tts_factories["unit_default"] = lambda: object()
    speech_manager.get_default_tts_provider = MethodType(
        lambda self: "unit_default",
        speech_manager,
    )
    monkeypatch.setattr(
        speech_manager,
        "_start_tts_initialization",
        MethodType(fake_start, speech_manager),
    )

    asyncio.run(speech_manager.initialize())

    assert calls == ["unit_default"]
    assert speech_manager._tts_initialization_futures.get("unit_default") is sentinel_future


def test_initialize_handles_missing_default_provider_gracefully(speech_manager):
    speech_manager.get_default_tts_provider = MethodType(
        lambda self: "ghost",
        speech_manager,
    )

    asyncio.run(speech_manager.initialize())

    assert speech_manager._tts_initialization_futures == {}


def test_initialize_handles_initialization_failure_gracefully(speech_manager, monkeypatch):
    speech_manager._tts_factories["unit_default"] = lambda: object()
    speech_manager.get_default_tts_provider = MethodType(
        lambda self: "unit_default",
        speech_manager,
    )

    def raising_start(self, provider_key):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        speech_manager,
        "_start_tts_initialization",
        MethodType(raising_start, speech_manager),
    )

    asyncio.run(speech_manager.initialize())

    assert speech_manager._tts_initialization_futures == {}


def test_get_tts_provider_names_returns_ordered_copy(speech_manager):
    speech_manager.tts_services["alpha"] = object()
    speech_manager.tts_services["beta"] = object()

    names = speech_manager.get_tts_provider_names()

    assert isinstance(names, tuple)
    assert names[:2] == ("alpha", "beta")
    assert "gamma" not in names
    assert "eleven_labs" in names

    speech_manager.tts_services["gamma"] = object()

    assert names[:2] == ("alpha", "beta")
    assert "gamma" not in names


def test_resolve_tts_provider_prefers_registered_choice(speech_manager):
    speech_manager.tts_services["eleven_labs"] = object()
    speech_manager.tts_services["custom"] = object()

    resolved = speech_manager.resolve_tts_provider("custom")

    assert resolved == "custom"


def test_google_tts_uses_unique_temp_files_and_cleans_up(monkeypatch, tmp_path):
    from modules.Speech_Services import Google_tts

    google_tts = Google_tts.GoogleTTS()
    google_tts.set_tts(True)

    created_paths = []
    original_named_tempfile = Google_tts.tempfile.NamedTemporaryFile

    def tracking_named_tempfile(*args, **kwargs):
        kwargs.setdefault("dir", tmp_path)
        tmp_file = original_named_tempfile(*args, **kwargs)
        created_paths.append(tmp_file.name)
        return tmp_file

    monkeypatch.setattr(Google_tts.tempfile, "NamedTemporaryFile", tracking_named_tempfile)

    loaded_files = []

    def tracking_load(filename):
        loaded_files.append(filename)

    pygame = sys.modules["pygame"]
    monkeypatch.setattr(pygame.mixer.music, "load", tracking_load)

    busy_states = [True, False]

    def tracking_get_busy():
        if busy_states:
            return busy_states.pop(0)
        return False

    monkeypatch.setattr(pygame.mixer.music, "get_busy", tracking_get_busy)

    class ImmediateThread:
        def __init__(self, *, target=None, args=(), kwargs=None):
            self._target = target
            self._args = args
            self._kwargs = kwargs or {}

        def start(self):
            if self._target is not None:
                self._target(*self._args, **self._kwargs)

    monkeypatch.setattr(Google_tts.threading, "Thread", lambda target=None, args=(), kwargs=None: ImmediateThread(target=target, args=args, kwargs=kwargs))

    async def run_requests():
        await google_tts.text_to_speech("hello")
        await google_tts.text_to_speech("world")

    asyncio.run(run_requests())

    assert len(created_paths) == 2
    assert len(set(created_paths)) == 2, "Each synthesis should create a unique temp file"
    assert loaded_files == created_paths, "Playback should load the specific temp files"
    for path in created_paths:
        assert not os.path.exists(path), "Temporary audio files should be removed after playback"


def test_resolve_tts_provider_falls_back_to_eleven_labs(speech_manager):
    speech_manager.tts_services["eleven_labs"] = object()
    speech_manager.tts_services["google"] = object()

    resolved = speech_manager.resolve_tts_provider("missing")

    assert resolved == "eleven_labs"


def test_resolve_tts_provider_returns_first_available_when_no_fallback(speech_manager):
    speech_manager.tts_services["google"] = object()
    speech_manager.tts_services["second"] = object()

    resolved = speech_manager.resolve_tts_provider(None)

    assert resolved == "eleven_labs"


def test_resolve_tts_provider_handles_no_services(speech_manager):
    assert speech_manager.resolve_tts_provider("whatever") == "eleven_labs"


def test_get_stt_provider_names_returns_ordered_copy(speech_manager):
    speech_manager.stt_services["delta"] = object()
    speech_manager.stt_services["epsilon"] = object()

    names = speech_manager.get_stt_provider_names()

    assert isinstance(names, tuple)
    assert names[:2] == ("delta", "epsilon")
    assert "zeta" not in names
    assert "google" in names

    speech_manager.stt_services["zeta"] = object()

    assert names[:2] == ("delta", "epsilon")
    assert "zeta" not in names


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


def test_google_tts_text_to_speech_uses_stubbed_client():
    from google.cloud import texttospeech
    from modules.Speech_Services.Google_tts import GoogleTTS
    import asyncio

    tts = GoogleTTS()
    tts.set_tts(True)

    asyncio.run(tts.text_to_speech("Hello world"))

    assert len(texttospeech.TextToSpeechClient.synthesize_calls) == 1


def test_google_tts_set_voice_accepts_dict_payload():
    from modules.Speech_Services.Google_tts import GoogleTTS

    tts = GoogleTTS()
    payload = {"name": "en-GB-Wavenet-B", "language_codes": ["en-GB"]}

    tts.set_voice(payload)

    assert tts.voice.kwargs["name"] == "en-GB-Wavenet-B"
    assert tts.voice.kwargs["language_code"] == "en-GB"


def test_google_tts_get_voices_returns_expected_structure():
    from google.cloud import texttospeech
    from modules.Speech_Services.Google_tts import GoogleTTS

    voice = types.SimpleNamespace(
        name="sample",
        language_codes=["en-US"],
        ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL,
        natural_sample_rate_hertz=24000,
    )
    texttospeech._set_voices([voice])

    tts = GoogleTTS()

    voices = tts.get_voices()

    assert voices == [
        {
            "name": "sample",
            "language_codes": ["en-US"],
            "ssml_gender": "NEUTRAL",
            "natural_sample_rate_hertz": 24000,
        }
    ]


def test_google_stt_factory_uses_configured_sample_rate(speech_manager):
    speech_manager.config_manager.config["GOOGLE_STT_SAMPLE_RATE"] = "22050"

    google_stt = speech_manager._google_stt_factory()

    assert google_stt.config.sample_rate_hertz == 22050
    assert google_stt.fs == 22050


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


def test_get_provider_credential_status_detects_saved_key(speech_manager):
    speech_manager.config_manager.config["XI_API_KEY"] = "secret-token"

    status = speech_manager.get_provider_credential_status("ElevenLabs")

    assert status["has_key"] is True
    assert status["metadata"]["length"] == len("secret-token")


def test_get_provider_credential_status_handles_unknown_provider(speech_manager):
    status = speech_manager.get_provider_credential_status("Unknown")

    assert status == {"has_key": False, "metadata": {}}


def test_describe_general_settings_includes_defaults(speech_manager):
    speech_manager.tts_services["eleven_labs"] = object()
    speech_manager.stt_services["google"] = object()
    speech_manager.set_default_tts_provider("eleven_labs")
    speech_manager.set_default_stt_provider("google")

    summary = speech_manager.describe_general_settings()

    assert summary["tts_enabled"] is True
    assert summary["default_tts_provider"] == "eleven_labs"
    assert "eleven_labs" in summary["tts_providers"]
    assert summary["stt_enabled"] is True
    assert summary["default_stt_provider"] == "google"
    assert "google" in summary["stt_providers"]


def test_list_tts_voice_options_normalizes_items(speech_manager, monkeypatch):
    monkeypatch.setattr(
        speech_manager,
        "get_tts_voices",
        lambda provider=None: [
            {"name": "Alpha", "voice_id": "alpha"},
            "Beta",
        ],
    )

    voices = speech_manager.list_tts_voice_options("eleven_labs")

    assert voices == [
        {"name": "Alpha", "voice_id": "alpha"},
        {"name": "Beta"},
    ]


def test_get_openai_display_config_returns_stored_values(speech_manager):
    speech_manager.config_manager.config.update(
        {
            "OPENAI_STT_PROVIDER": "Whisper Online",
            "OPENAI_TTS_PROVIDER": "GPT-4o Mini TTS",
            "OPENAI_LANGUAGE": "en",
            "OPENAI_TASK": "transcribe",
            "OPENAI_INITIAL_PROMPT": "Hello",
        }
    )

    config = speech_manager.get_openai_display_config()

    assert config["stt_provider"] == "Whisper Online"
    assert config["tts_provider"] == "GPT-4o Mini TTS"
    assert config["language"] == "en"
    assert config["task"] == "transcribe"
    assert config["initial_prompt"] == "Hello"


def test_get_openai_option_sets_returns_expected_keys(speech_manager):
    options = speech_manager.get_openai_option_sets()

    assert {"stt", "tts", "language", "task"}.issubset(options.keys())


def test_normalize_openai_display_settings_matches_prepare_function(speech_manager):
    payload = {
        "api_key": "abc123",
        "stt_provider": "Whisper Online",
        "language": "English (en)",
        "task": "Transcribe",
        "initial_prompt": "Prompt",
        "tts_provider": "GPT-4o Mini TTS",
    }

    prepared = prepare_openai_settings(payload)
    manager_prepared = speech_manager.normalize_openai_display_settings(payload)

    assert manager_prepared == prepared


def test_set_google_credentials_reconfigures_providers(speech_manager, monkeypatch):
    class _TTSStub:
        def __init__(self):
            self.voice = None

        def set_voice(self, voice):
            self.voice = voice

        def get_voices(self):
            return []

    class _STTStub:
        def __init__(self):
            self.config = types.SimpleNamespace(
                language_code="en-US",
                enable_automatic_punctuation=True,
            )

    new_tts = _TTSStub()
    new_stt = _STTStub()
    speech_manager._google_tts_factory = lambda: new_tts
    speech_manager._google_stt_factory = lambda: new_stt
    speech_manager.tts_services['google'] = object()
    speech_manager.stt_services['google'] = object()
    speech_manager.set_default_tts_provider('google')
    speech_manager.set_default_stt_provider('google')
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)

    speech_manager.set_google_credentials(
        "/tmp/google.json",
        voice_name={"name": "NewVoice"},
        stt_language="es-ES",
        auto_punctuation=False,
    )

    assert speech_manager.tts_services['google'] is new_tts
    assert speech_manager.stt_services['google'] is new_stt
    assert speech_manager.get_default_tts_provider() == 'google'
    assert speech_manager.get_default_stt_provider() == 'google'
    assert speech_manager.config_manager.google_credentials == "/tmp/google.json"
    assert os.environ["GOOGLE_APPLICATION_CREDENTIALS"] == "/tmp/google.json"
    assert new_tts.voice == {"name": "NewVoice", "language_code": "es-ES"}
    assert new_stt.config.language_code == "es-ES"
    assert new_stt.config.enable_automatic_punctuation is False
    assert speech_manager.config_manager.google_speech_settings == {
        "tts_voice": "NewVoice",
        "stt_language": "es-ES",
        "auto_punctuation": False,
    }


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
    assert speech_manager.config_manager.google_speech_settings == {}


def test_set_google_credentials_factory_failure_rolls_back(speech_manager, monkeypatch):
    def fail():
        raise RuntimeError("boom")

    speech_manager._google_tts_factory = fail
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "old.json")

    with pytest.raises(RuntimeError):
        speech_manager.set_google_credentials("/tmp/new.json")

    assert os.environ["GOOGLE_APPLICATION_CREDENTIALS"] == "old.json"
    assert speech_manager.config_manager.google_speech_settings == {}


def test_get_google_credentials_path_reads_config(speech_manager):
    speech_manager.config_manager.config["GOOGLE_APPLICATION_CREDENTIALS"] = "/tmp/creds.json"

    assert speech_manager.get_google_credentials_path() == "/tmp/creds.json"


def test_set_google_credentials_uses_stored_preferences(speech_manager, monkeypatch):
    class _TTSStub:
        def __init__(self):
            self.voice = None

        def set_voice(self, voice):
            self.voice = voice

        def get_voices(self):
            return []

    class _STTStub:
        def __init__(self):
            self.config = types.SimpleNamespace(
                language_code="en-US",
                enable_automatic_punctuation=True,
            )

    stored = {
        "tts_voice": "StoredVoice",
        "stt_language": "fr-FR",
        "auto_punctuation": False,
    }
    speech_manager.config_manager.google_speech_settings = dict(stored)
    speech_manager.config_manager.config['GOOGLE_SPEECH'] = dict(stored)

    new_tts = _TTSStub()
    new_stt = _STTStub()
    speech_manager._google_tts_factory = lambda: new_tts
    speech_manager._google_stt_factory = lambda: new_stt
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)

    speech_manager.set_google_credentials("/tmp/google.json")

    assert new_tts.voice == {"name": "StoredVoice", "language_code": "fr-FR"}
    assert new_stt.config.language_code == "fr-FR"
    assert new_stt.config.enable_automatic_punctuation is False
    assert speech_manager.config_manager.google_speech_settings == stored


def test_elevenlabs_text_to_speech_uses_async_executor(monkeypatch, tmp_path):
    monkeypatch.setenv("XI_API_KEY", "key")

    tts = ElevenLabsTTS()
    tts._use_tts = True
    tts.configured = True
    tts.voice_ids = [{"voice_id": "voice-1", "name": "Voice"}]

    class _FixedDatetime:
        @staticmethod
        def now():
            from datetime import datetime as _real_datetime

            return _real_datetime(2024, 1, 2, 3, 4, 5)

    monkeypatch.setattr("modules.Speech_Services.elevenlabs_tts.datetime", _FixedDatetime)

    class _DummyResponse:
        status_code = 200
        ok = True
        text = ""

        def iter_content(self, chunk_size=1):
            yield b"chunk-1"
            yield b""
            yield b"chunk-2"

    monkeypatch.setattr(
        "modules.Speech_Services.elevenlabs_tts.requests.post",
        lambda *args, **kwargs: _DummyResponse(),
    )

    original_join = os.path.join

    def _join(base, *paths):
        if base == "assets/SCOUT/tts_mp3/":
            return str(tmp_path.joinpath(*paths))
        return original_join(base, *paths)

    monkeypatch.setattr("modules.Speech_Services.elevenlabs_tts.os.path.join", _join)
    monkeypatch.setattr("modules.Speech_Services.elevenlabs_tts.os.makedirs", lambda *a, **k: None)

    playback_paths = []

    def _play_audio(path):
        playback_paths.append(path)

    monkeypatch.setattr(tts, "play_audio", _play_audio)

    write_handle = mock_open()
    monkeypatch.setattr("builtins.open", write_handle)

    calls = []

    async def _to_thread(func, *args, **kwargs):
        calls.append((func, args, kwargs))
        return func(*args, **kwargs)

    monkeypatch.setattr("modules.Speech_Services.elevenlabs_tts.asyncio.to_thread", _to_thread)

    asyncio.run(tts.text_to_speech("hello world"))

    assert len(calls) == 2
    download_func, download_args, download_kwargs = calls[0]
    playback_func, playback_args, playback_kwargs = calls[1]

    assert download_func.__name__ == "download_and_save_audio"
    assert download_args == () and download_kwargs == {}

    handle = write_handle()
    handle.write.assert_any_call(b"chunk-1")
    handle.write.assert_any_call(b"chunk-2")

    expected_path = str(tmp_path / "output_20240102030405.mp3")
    assert playback_func == tts.play_audio
    assert playback_args == (expected_path,)
    assert playback_kwargs == {}
    assert playback_paths == [expected_path]
