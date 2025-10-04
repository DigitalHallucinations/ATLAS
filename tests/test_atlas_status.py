import os
import sys
import types

import pytest

if "yaml" not in sys.modules:
    yaml_stub = types.ModuleType("yaml")
    yaml_stub.safe_load = lambda *_args, **_kwargs: {}
    yaml_stub.dump = lambda *_args, **_kwargs: ""
    sys.modules["yaml"] = yaml_stub

if "dotenv" not in sys.modules:
    dotenv_stub = types.ModuleType("dotenv")

    def _noop(*_args, **_kwargs):  # pragma: no cover - placeholder
        return None

    dotenv_stub.load_dotenv = _noop
    dotenv_stub.set_key = _noop
    dotenv_stub.find_dotenv = lambda *_args, **_kwargs: ""
    sys.modules["dotenv"] = dotenv_stub

if "huggingface_hub" not in sys.modules:
    hf_stub = types.ModuleType("huggingface_hub")

    class _PlaceholderHfApi:  # pragma: no cover - placeholder
        def __init__(self, *_args, **_kwargs):
            pass

        def whoami(self, *_args, **_kwargs):
            return {}

    hf_stub.HfApi = _PlaceholderHfApi
    hf_stub.hf_hub_download = lambda *_args, **_kwargs: ""
    hf_stub.InferenceClient = type(
        "InferenceClient",
        (),
        {"__init__": lambda self, *args, **kwargs: None, "text_generation": lambda self, *args, **kwargs: ""},
    )
    sys.modules["huggingface_hub"] = hf_stub

if "tenacity" not in sys.modules:
    tenacity_stub = types.ModuleType("tenacity")

    def _identity_decorator(*_args, **_kwargs):  # pragma: no cover - placeholder
        def _decorator(fn):
            return fn

        return _decorator

    tenacity_stub.retry = _identity_decorator
    tenacity_stub.stop_after_attempt = lambda *_args, **_kwargs: None
    tenacity_stub.wait_exponential = lambda *_args, **_kwargs: None
    tenacity_stub.retry_if_exception_type = lambda *_args, **_kwargs: None

    class _AsyncRetrying:  # pragma: no cover - placeholder
        def __init__(self, *_args, **_kwargs):
            pass

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    class _Attempt:  # pragma: no cover - placeholder
        def __enter__(self):
            return None

        def __exit__(self, *_args):
            return False

    tenacity_stub.AsyncRetrying = lambda *_args, **_kwargs: _AsyncRetrying()
    tenacity_stub.Attempt = _Attempt
    sys.modules["tenacity"] = tenacity_stub

if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")

    class _CudaModule:
        @staticmethod
        def is_available():
            return False

        @staticmethod
        def device_count():
            return 0

        @staticmethod
        def get_device_properties(_index):
            return types.SimpleNamespace(total_memory=0)

        @staticmethod
        def memory_allocated(_index):
            return 0

        @staticmethod
        def empty_cache():
            return None

    torch_stub.cuda = _CudaModule()
    torch_stub.float16 = "float16"
    torch_stub.bfloat16 = "bfloat16"
    torch_stub.compile = lambda model: model
    torch_stub.device = lambda *_args, **_kwargs: None
    torch_stub.nn = types.SimpleNamespace(Linear=type("Linear", (), {}))
    sys.modules["torch"] = torch_stub

if "transformers" not in sys.modules:
    transformers_stub = types.ModuleType("transformers")

    class _Placeholder:  # pragma: no cover - placeholder
        def __init__(self, *_args, **_kwargs):
            pass

        def __call__(self, *_args, **_kwargs):
            return None

    transformers_stub.AutoTokenizer = _Placeholder
    transformers_stub.AutoModelForCausalLM = _Placeholder
    transformers_stub.AutoConfig = _Placeholder
    transformers_stub.pipeline = lambda *_args, **_kwargs: None
    transformers_stub.BitsAndBytesConfig = _Placeholder
    transformers_stub.Trainer = _Placeholder
    transformers_stub.TrainingArguments = _Placeholder
    transformers_stub.DataCollatorForLanguageModeling = _Placeholder
    sys.modules["transformers"] = transformers_stub

    integrations_module = types.ModuleType("transformers.integrations")
    deepspeed_module = types.ModuleType("transformers.integrations.deepspeed")

    class _HfDeepSpeedConfig:  # pragma: no cover - placeholder
        def __init__(self, *_args, **_kwargs):
            pass

    deepspeed_module.HfDeepSpeedConfig = _HfDeepSpeedConfig
    integrations_module.deepspeed = deepspeed_module
    transformers_stub.integrations = integrations_module
    sys.modules["transformers.integrations"] = integrations_module
    sys.modules["transformers.integrations.deepspeed"] = deepspeed_module

if "accelerate" not in sys.modules:
    accelerate_stub = types.ModuleType("accelerate")
    accelerate_stub.infer_auto_device_map = lambda *_args, **_kwargs: {}
    sys.modules["accelerate"] = accelerate_stub

if "datasets" not in sys.modules:
    datasets_stub = types.ModuleType("datasets")

    class _Dataset:  # pragma: no cover - placeholder
        pass

    datasets_stub.Dataset = _Dataset
    sys.modules["datasets"] = datasets_stub

if "psutil" not in sys.modules:
    psutil_stub = types.ModuleType("psutil")
    psutil_stub.virtual_memory = lambda: types.SimpleNamespace(total=0, available=0)
    psutil_stub.cpu_count = lambda *_args, **_kwargs: 1
    sys.modules["psutil"] = psutil_stub

if "xai_sdk" not in sys.modules:
    xai_stub = types.ModuleType("xai_sdk")

    class _Client:  # pragma: no cover - placeholder
        def __init__(self, *_args, **_kwargs):
            pass

        async def chat(self, *_args, **_kwargs):
            return types.SimpleNamespace(output=types.SimpleNamespace(text=""))

    xai_stub.Client = _Client
    sys.modules["xai_sdk"] = xai_stub

if "openai" not in sys.modules:
    openai_stub = types.ModuleType("openai")

    class _AsyncOpenAI:  # pragma: no cover - placeholder
        class Responses:
            async def create(self, *_args, **_kwargs):
                return types.SimpleNamespace(output=[types.SimpleNamespace(content=[])])

        class ChatCompletions:
            async def create(self, *_args, **_kwargs):
                return types.SimpleNamespace(choices=[])

        def __init__(self, *_args, **_kwargs):
            self.responses = self.Responses()
            self.chat = types.SimpleNamespace(completions=self.ChatCompletions())

    openai_stub.AsyncOpenAI = _AsyncOpenAI
    sys.modules["openai"] = openai_stub

if "ATLAS.ToolManager" not in sys.modules:
    tool_manager_stub = types.ModuleType("ATLAS.ToolManager")
    tool_manager_stub.load_function_map_from_current_persona = lambda *_args, **_kwargs: {}
    tool_manager_stub.load_functions_from_json = lambda *_args, **_kwargs: []
    tool_manager_stub.use_tool = lambda *_args, **_kwargs: {}
    sys.modules["ATLAS.ToolManager"] = tool_manager_stub

os.environ.setdefault("OPENAI_API_KEY", "test-key")

if "mistralai" not in sys.modules:
    mistral_stub = types.ModuleType("mistralai")

    class _Mistral:  # pragma: no cover - placeholder
        class APIError(Exception):
            pass

        def __init__(self, *_args, **_kwargs):
            self.chat = types.SimpleNamespace(
                complete=lambda *_args, **_kwargs: types.SimpleNamespace(
                    choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=""))]
                )
            )

    mistral_stub.Mistral = _Mistral
    sys.modules["mistralai"] = mistral_stub

sys.modules.pop("google", None)
google_stub = types.ModuleType("google")
google_stub.__path__ = []  # type: ignore[attr-defined]
genai_module = types.ModuleType("google.generativeai")
cloud_module = types.ModuleType("google.cloud")
genai_types_module = types.ModuleType("google.generativeai.types")
texttospeech_module = types.ModuleType("google.cloud.texttospeech")

class _GenerativeClient:  # pragma: no cover - placeholder
    def __init__(self, *_args, **_kwargs):
        pass

    class GenerativeModel:
        def __init__(self, *_args, **_kwargs):
            pass

        async def generate_content_async(self, *_args, **_kwargs):
            return types.SimpleNamespace(text="")

genai_module.configure = lambda *_args, **_kwargs: None
genai_module.GenerativeModel = _GenerativeClient.GenerativeModel
genai_module.types = genai_types_module
google_stub.generativeai = genai_module

genai_types_module.GenerationConfig = lambda *_args, **_kwargs: {}
genai_types_module.ContentDict = dict
genai_types_module.PartDict = dict


class _Tool:  # pragma: no cover - placeholder
    def __init__(self, *args, **kwargs):
        pass


class _FunctionDeclaration:  # pragma: no cover - placeholder
    def __init__(self, *args, **kwargs):
        pass


genai_types_module.Tool = _Tool
genai_types_module.FunctionDeclaration = _FunctionDeclaration

class _SpeechClient:  # pragma: no cover - placeholder
    def synthesize_speech(self, *_args, **_kwargs):
        return types.SimpleNamespace(audio_content=b"")

    def list_voices(self, *_args, **_kwargs):
        return types.SimpleNamespace(voices=[])

cloud_module.speech_v1p1beta1 = types.SimpleNamespace(
    SpeechClient=_SpeechClient,
    VoiceSelectionParams=lambda *_args, **_kwargs: None,
    SynthesisInput=lambda *_args, **_kwargs: None,
    AudioConfig=lambda *_args, **_kwargs: None,
    AudioEncoding=types.SimpleNamespace(MP3="MP3"),
    SsmlVoiceGender=lambda value: types.SimpleNamespace(name=str(value)),
)


class _VoiceSelectionParams:  # pragma: no cover - placeholder
    def __init__(self, *_args, **_kwargs):
        pass


class _TextToSpeechClient:  # pragma: no cover - placeholder
    def synthesize_speech(self, *_args, **_kwargs):
        return types.SimpleNamespace(audio_content=b"")

    def list_voices(self, *_args, **_kwargs):
        return types.SimpleNamespace(voices=[])


texttospeech_module.VoiceSelectionParams = _VoiceSelectionParams
texttospeech_module.TextToSpeechClient = _TextToSpeechClient
texttospeech_module.SynthesisInput = lambda *_args, **_kwargs: None
texttospeech_module.AudioConfig = lambda *_args, **_kwargs: None
texttospeech_module.AudioEncoding = types.SimpleNamespace(MP3="MP3")
texttospeech_module.SsmlVoiceGender = lambda value: types.SimpleNamespace(name=str(value))
cloud_module.texttospeech = texttospeech_module
google_stub.cloud = cloud_module
sys.modules["google"] = google_stub
sys.modules["google.generativeai"] = genai_module
sys.modules["google.generativeai.types"] = genai_types_module
sys.modules["google.cloud"] = cloud_module
sys.modules["google.cloud.texttospeech"] = texttospeech_module

if "anthropic" not in sys.modules:
    anthropic_stub = types.ModuleType("anthropic")

    class _AsyncAnthropic:  # pragma: no cover - placeholder
        class _Stream:
            def __init__(self):
                self._final_response = types.SimpleNamespace(
                    content=[types.SimpleNamespace(type="text", text="")]
                )

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_exc):
                return False

            def __aiter__(self):
                return self

            async def __anext__(self):
                raise StopAsyncIteration

            async def get_final_response(self):
                return self._final_response

        class Messages:
            async def create(self, *_args, **_kwargs):
                return types.SimpleNamespace(
                    content=[types.SimpleNamespace(type="text", text="")]
                )

            def stream(self, *_args, **_kwargs):
                return _AsyncAnthropic._Stream()

        def __init__(self, *_args, **_kwargs):
            self.messages = self.Messages()

    anthropic_stub.AsyncAnthropic = _AsyncAnthropic
    anthropic_stub.APIError = Exception
    anthropic_stub.RateLimitError = Exception
    anthropic_stub.HUMAN_PROMPT = ""
    anthropic_stub.AI_PROMPT = ""
    sys.modules["anthropic"] = anthropic_stub

if "pygame" not in sys.modules:
    pygame_stub = types.ModuleType("pygame")
    pygame_stub.mixer = types.SimpleNamespace(
        init=lambda: None,
        music=types.SimpleNamespace(
            load=lambda *_args, **_kwargs: None,
            play=lambda *_args, **_kwargs: None,
            get_busy=lambda: False,
        ),
    )
    pygame_stub.time = types.SimpleNamespace(Clock=lambda: types.SimpleNamespace(tick=lambda *_args, **_kwargs: None))
    sys.modules["pygame"] = pygame_stub

if "requests" not in sys.modules:
    requests_stub = types.ModuleType("requests")
    requests_stub.post = lambda *_args, **_kwargs: types.SimpleNamespace(json=lambda: {})
    requests_stub.get = lambda *_args, **_kwargs: types.SimpleNamespace(json=lambda: {})
    sys.modules["requests"] = requests_stub

if "sounddevice" not in sys.modules:
    sd_stub = types.ModuleType("sounddevice")
    sd_stub.rec = lambda *_args, **_kwargs: []
    sd_stub.wait = lambda: None
    sd_stub.play = lambda *_args, **_kwargs: None
    sd_stub.stop = lambda: None
    sys.modules["sounddevice"] = sd_stub

if "numpy" not in sys.modules:
    numpy_stub = types.ModuleType("numpy")
    numpy_stub.array = lambda data, dtype=None: data
    numpy_stub.float32 = "float32"
    numpy_stub.zeros = lambda shape, dtype=None: [0] * (shape[0] if isinstance(shape, tuple) else shape)
    numpy_stub.int16 = "int16"
    sys.modules["numpy"] = numpy_stub

if "soundfile" not in sys.modules:
    soundfile_stub = types.ModuleType("soundfile")
    soundfile_stub.write = lambda *_args, **_kwargs: None
    soundfile_stub.read = lambda *_args, **_kwargs: ([], 16000)
    sys.modules["soundfile"] = soundfile_stub

from ATLAS.ATLAS import ATLAS


class _SilentLogger:
    def error(self, *_args, **_kwargs):  # pragma: no cover - logging helper
        return None


@pytest.fixture()
def atlas_stub():
    instance = object.__new__(ATLAS)
    instance.logger = _SilentLogger()
    return instance


@pytest.mark.parametrize(
    "summary, expected",
    [
        (
            {
                "llm_provider": "OpenAI",
                "llm_model": "gpt-4o",
                "tts_provider": "ElevenLabs",
                "tts_voice": "Rachel",
            },
            "LLM: OpenAI • Model: gpt-4o • TTS: ElevenLabs (Voice: Rachel)",
        ),
        (
            {
                "llm_provider": "Anthropic",
                "llm_model": "claude-3",
                "tts_provider": "",
                "tts_voice": "Alloy",
            },
            "LLM: Anthropic • Model: claude-3 • TTS: None (Voice: Alloy)",
        ),
        (
            {
                "llm_provider": "",
                "llm_model": None,
                "tts_provider": None,
                "tts_voice": "",
            },
            "LLM: Unknown • Model: No model selected • TTS: None (Voice: Not Set)",
        ),
        (
            {
                "llm_provider": "OpenAI",
                "llm_model": "gpt-4o",
                "tts_provider": "ElevenLabs",
                "tts_voice": "Rachel",
                "llm_warning": "OpenAI key missing",
            },
            "LLM: OpenAI • Model: gpt-4o • TTS: ElevenLabs (Voice: Rachel) • Warning: OpenAI key missing",
        ),
    ],
)
def test_format_chat_status_with_summary(atlas_stub, summary, expected):
    assert atlas_stub.format_chat_status(summary) == expected


def test_format_chat_status_fetches_summary_when_missing(atlas_stub):
    captured = {
        "llm_provider": "Google",
        "llm_model": "gemini-pro",
        "tts_provider": "Coqui",
        "tts_voice": "Spoken",
    }

    atlas_stub.get_chat_status_summary = types.MethodType(
        lambda _self: captured, atlas_stub
    )

    result = atlas_stub.format_chat_status()

    assert result == "LLM: Google • Model: gemini-pro • TTS: Coqui (Voice: Spoken)"


def test_chat_status_summary_includes_pending_warning(atlas_stub):
    atlas_stub.provider_manager = types.SimpleNamespace(
        get_current_provider=lambda: None,
        get_current_model=lambda: None,
    )
    atlas_stub.speech_manager = types.SimpleNamespace(
        get_active_tts_summary=lambda: (None, None)
    )

    warning_message = "API key for provider 'OpenAI' is not configured."

    atlas_stub.config_manager = types.SimpleNamespace(
        get_pending_provider_warnings=lambda: {"OpenAI": warning_message},
        get_default_provider=lambda: "OpenAI",
    )

    summary = atlas_stub.get_chat_status_summary()

    assert summary["llm_provider"] == "OpenAI (Not Configured)"
    assert summary["llm_model"] == "Unavailable"
    assert summary["llm_warning"] == warning_message
