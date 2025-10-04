import sys
import types


if "pygame" not in sys.modules:
    music = types.SimpleNamespace(
        load=lambda *_args, **_kwargs: None,
        play=lambda *_args, **_kwargs: None,
        get_busy=lambda: False,
    )
    mixer = types.SimpleNamespace(init=lambda *_args, **_kwargs: None, music=music)

    class _Clock:
        def tick(self, *_args, **_kwargs):
            return None

    time_module = types.SimpleNamespace(Clock=lambda: _Clock())

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

    def _dummy_request(*_args, **_kwargs):
        return _DummyResponse()

    requests_module = types.ModuleType("requests")
    requests_module.get = _dummy_request
    requests_module.post = _dummy_request

    sys.modules["requests"] = requests_module


if "google" not in sys.modules:
    google_module = types.ModuleType("google")
    cloud_module = types.ModuleType("google.cloud")
    speech_module = types.ModuleType("google.cloud.speech_v1p1beta1")

    class _DummyVoiceSelectionParams:
        def __init__(self, *args, **kwargs):
            pass

    class _DummySpeechClient:
        def synthesize_speech(self, *args, **kwargs):
            return types.SimpleNamespace(audio_content=b"")

        def list_voices(self):
            return types.SimpleNamespace(voices=[])

    class _DummySynthesisInput:
        def __init__(self, *args, **kwargs):
            pass

    class _DummyAudioConfig:
        def __init__(self, *args, **kwargs):
            pass

    class _DummySsmlVoiceGender:
        MALE = 1
        FEMALE = 2
        NEUTRAL = 3

        def __call__(self, *_args, **_kwargs):
            return types.SimpleNamespace(name="NEUTRAL")

    speech_module.VoiceSelectionParams = _DummyVoiceSelectionParams
    speech_module.SpeechClient = _DummySpeechClient
    speech_module.SynthesisInput = _DummySynthesisInput
    speech_module.AudioConfig = _DummyAudioConfig
    speech_module.AudioEncoding = types.SimpleNamespace(MP3=1)
    speech_module.SsmlVoiceGender = _DummySsmlVoiceGender()

    texttospeech_module = types.ModuleType("google.cloud.texttospeech")
    texttospeech_module.VoiceSelectionParams = _DummyVoiceSelectionParams
    texttospeech_module.TextToSpeechClient = _DummySpeechClient
    texttospeech_module.SynthesisInput = _DummySynthesisInput
    texttospeech_module.AudioConfig = _DummyAudioConfig
    texttospeech_module.AudioEncoding = types.SimpleNamespace(MP3=1)
    texttospeech_module.SsmlVoiceGender = _DummySsmlVoiceGender()

    cloud_module.speech_v1p1beta1 = speech_module
    cloud_module.texttospeech = texttospeech_module
    google_module.cloud = cloud_module

    sys.modules["google"] = google_module
    sys.modules["google.cloud"] = cloud_module
    sys.modules["google.cloud.speech_v1p1beta1"] = speech_module
    sys.modules["google.cloud.texttospeech"] = texttospeech_module


if "sounddevice" not in sys.modules:
    class _DummyInputStream:
        def __init__(self, callback=None, *args, **kwargs):
            self.callback = callback

        def start(self):
            return None

        def stop(self):
            return None

        def close(self):
            return None

    sounddevice_module = types.ModuleType("sounddevice")
    sounddevice_module.InputStream = _DummyInputStream

    sys.modules["sounddevice"] = sounddevice_module


if "soundfile" not in sys.modules:
    soundfile_module = types.ModuleType("soundfile")
    soundfile_module.write = lambda *_args, **_kwargs: None

    sys.modules["soundfile"] = soundfile_module


if "numpy" not in sys.modules:
    def _concatenate(items):
        result = []
        for entry in items:
            if hasattr(entry, "tolist"):
                result.extend(entry.tolist())
            elif isinstance(entry, (list, tuple)):
                result.extend(entry)
            else:
                result.append(entry)
        return result

    numpy_module = types.ModuleType("numpy")
    numpy_module.concatenate = _concatenate
    numpy_module.array = lambda value: value

    sys.modules["numpy"] = numpy_module
