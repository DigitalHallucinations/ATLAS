"""Unit tests for the SpeechManager TTS summary helper."""

from types import MethodType
import logging
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


from modules.Speech_Services.speech_manager import SpeechManager


class _DummyConfig:
    def get_tts_enabled(self):
        return True


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
