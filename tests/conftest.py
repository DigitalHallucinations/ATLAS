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

if "huggingface_hub" not in sys.modules:
    hf_module = types.ModuleType("huggingface_hub")

    class _StubHfApi:
        """Lightweight stub for Hugging Face client used in tests."""

        def __init__(self, *args, **kwargs):
            pass

        def whoami(self, *args, **kwargs):  # pragma: no cover - stub helper
            return {}

    hf_module.HfApi = _StubHfApi
    hf_module.hf_hub_download = lambda *args, **kwargs: ""
    sys.modules["huggingface_hub"] = hf_module

if "modules.Providers.HuggingFace.HF_gen_response" not in sys.modules:
    hf_provider_module = types.ModuleType("modules.Providers.HuggingFace.HF_gen_response")

    class _StubModelManager:
        def __init__(self):
            self.installed = []

        def remove_installed_model(self, model_name):
            self.installed = [name for name in self.installed if name != model_name]

        def get_installed_models(self):
            return list(self.installed)

    class HuggingFaceGenerator:
        def __init__(self, *_args, **_kwargs):
            self.model_manager = _StubModelManager()

        async def load_model(self, model_name, *_args, **_kwargs):
            self.model_manager.installed.append(model_name)

        async def generate_response(self, messages, model, stream=True):  # pragma: no cover - stub helper
            return ""

        async def process_streaming_response(self, response):  # pragma: no cover - stub helper
            collected = []
            async for chunk in response:
                collected.append(chunk)
            return "".join(collected)

        def unload_model(self):  # pragma: no cover - stub helper
            return None

        def get_installed_models(self):
            return self.model_manager.get_installed_models()

    async def search_models(*_args, **_kwargs):  # pragma: no cover - stub helper
        return []

    async def download_model(generator, model_id, force=False):  # pragma: no cover - stub helper
        generator.model_manager.installed.append(model_id)

    def update_model_settings(*_args, **_kwargs):  # pragma: no cover - stub helper
        return {}

    def clear_cache(*_args, **_kwargs):  # pragma: no cover - stub helper
        return None

    hf_provider_module.HuggingFaceGenerator = HuggingFaceGenerator
    hf_provider_module.search_models = search_models
    hf_provider_module.download_model = download_model
    hf_provider_module.update_model_settings = update_model_settings
    hf_provider_module.clear_cache = clear_cache

    sys.modules["modules.Providers.HuggingFace.HF_gen_response"] = hf_provider_module

if "modules.Providers.Grok.grok_generate_response" not in sys.modules:
    grok_module = types.ModuleType("modules.Providers.Grok.grok_generate_response")

    class GrokGenerator:
        def __init__(self, *_args, **_kwargs):
            pass

        async def generate_response(self, *_args, **_kwargs):  # pragma: no cover - stub helper
            return ""

        async def process_streaming_response(self, response):  # pragma: no cover - stub helper
            collected = []
            async for chunk in response:
                collected.append(chunk)
            return "".join(collected)

        async def unload_model(self):  # pragma: no cover - stub helper
            return None

    grok_module.GrokGenerator = GrokGenerator
    sys.modules["modules.Providers.Grok.grok_generate_response"] = grok_module

if "modules.Providers.OpenAI.OA_gen_response" not in sys.modules:
    openai_module = types.ModuleType("modules.Providers.OpenAI.OA_gen_response")

    from weakref import WeakKeyDictionary

    class _StubModelManager:
        def __init__(self):
            self.current_model = "gpt-3.5-turbo"

        def set_model(self, model, provider):
            self.current_model = model

        def get_current_model(self):
            return self.current_model

    class OpenAIGenerator:
        def __init__(self, config_manager):
            self.config_manager = config_manager
            self.model_manager = _StubModelManager()

        async def generate_response(self, **_kwargs):  # pragma: no cover - stub helper
            return ""

        async def process_streaming_response(self, response):  # pragma: no cover - stub helper
            chunks = []
            async for entry in response:
                chunks.append(entry)
            return "".join(chunks)

        async def aclose(self):  # pragma: no cover - stub helper
            return None

        async def close(self):  # pragma: no cover - stub helper
            return None

    _GENERATOR_CACHE = WeakKeyDictionary()

    def get_generator(config_manager, _module=openai_module, _cache=_GENERATOR_CACHE):
        generator = _cache.get(config_manager)
        if generator is None:
            generator_cls = _module.OpenAIGenerator
            generator = generator_cls(config_manager)
            _cache[config_manager] = generator
        return generator

    openai_module.OpenAIGenerator = OpenAIGenerator
    openai_module._GENERATOR_CACHE = _GENERATOR_CACHE
    openai_module.get_generator = get_generator
    sys.modules["modules.Providers.OpenAI.OA_gen_response"] = openai_module

if "modules.Providers.Mistral.Mistral_gen_response" not in sys.modules:
    mistral_module = types.ModuleType("modules.Providers.Mistral.Mistral_gen_response")

    from weakref import WeakKeyDictionary

    class MistralGenerator:
        def __init__(self, config_manager):
            self.config_manager = config_manager

        async def generate_response(self, **_kwargs):  # pragma: no cover - stub helper
            return ""

        async def process_streaming_response(self, response):  # pragma: no cover - stub helper
            chunks = []
            async for entry in response:
                chunks.append(entry)
            return "".join(chunks)

    _GENERATOR_CACHE = WeakKeyDictionary()

    def get_generator(config_manager, _module=mistral_module, _cache=_GENERATOR_CACHE):
        generator = _cache.get(config_manager)
        if generator is None:
            generator_cls = _module.MistralGenerator
            generator = generator_cls(config_manager)
            _cache[config_manager] = generator
        return generator

    mistral_module.MistralGenerator = MistralGenerator
    mistral_module._GENERATOR_CACHE = _GENERATOR_CACHE
    mistral_module.get_generator = get_generator
    sys.modules["modules.Providers.Mistral.Mistral_gen_response"] = mistral_module

if "modules.Providers.Google.GG_gen_response" not in sys.modules:
    google_module = types.ModuleType("modules.Providers.Google.GG_gen_response")

    from weakref import WeakKeyDictionary

    class GoogleGeminiGenerator:
        def __init__(self, config_manager):
            self.config_manager = config_manager

        async def generate_response(self, **_kwargs):  # pragma: no cover - stub helper
            return ""

        async def process_streaming_response(self, response):  # pragma: no cover - stub helper
            collected = []
            async for chunk in response:
                collected.append(chunk)
            return "".join(collected)

    _GENERATOR_CACHE = WeakKeyDictionary()

    def get_generator(config_manager, _module=google_module, _cache=_GENERATOR_CACHE):
        generator = _cache.get(config_manager)
        if generator is None:
            generator_cls = _module.GoogleGeminiGenerator
            generator = generator_cls(config_manager)
            _cache[config_manager] = generator
        return generator

    google_module.GoogleGeminiGenerator = GoogleGeminiGenerator
    google_module._GENERATOR_CACHE = _GENERATOR_CACHE
    google_module.get_generator = get_generator
    sys.modules["modules.Providers.Google.GG_gen_response"] = google_module

if "modules.Providers.Anthropic.Anthropic_gen_response" not in sys.modules:
    anthropic_provider_module = types.ModuleType("modules.Providers.Anthropic.Anthropic_gen_response")

    class AnthropicGenerator:
        def __init__(self, config_manager):
            self.config_manager = config_manager

        async def generate_response(self, **_kwargs):  # pragma: no cover - stub helper
            return ""

        async def process_streaming_response(self, response):  # pragma: no cover - stub helper
            chunks = []
            async for entry in response:
                chunks.append(entry)
            return "".join(chunks)

    anthropic_provider_module.AnthropicGenerator = AnthropicGenerator
    sys.modules["modules.Providers.Anthropic.Anthropic_gen_response"] = anthropic_provider_module

if "anthropic" not in sys.modules:
    anthropic_module = types.ModuleType("anthropic")

    class _StubAsyncAnthropic:
        def __init__(self, *args, **kwargs):
            pass

        class _Messages:
            async def create(self, *args, **kwargs):  # pragma: no cover - stub helper
                return types.SimpleNamespace(content=[])

            def stream(self, *args, **kwargs):  # pragma: no cover - stub helper
                raise RuntimeError("stream not supported in stub")

        @property
        def messages(self):  # pragma: no cover - stub helper
            return _StubAsyncAnthropic._Messages()

    anthropic_module.AsyncAnthropic = _StubAsyncAnthropic
    anthropic_module.APIError = Exception
    anthropic_module.RateLimitError = Exception
    sys.modules["anthropic"] = anthropic_module


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
    numpy_module.isscalar = lambda value: isinstance(value, (int, float, complex, bool, str))
    numpy_module.bool_ = bool

    sys.modules["numpy"] = numpy_module
