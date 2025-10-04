import sys
import types
from typing import Any, Dict, Optional

import pytest

if "yaml" not in sys.modules:
    yaml_stub = types.ModuleType("yaml")
    yaml_stub.safe_load = lambda *_args, **_kwargs: {}
    yaml_stub.dump = lambda *_args, **_kwargs: ""
    sys.modules["yaml"] = yaml_stub

if "dotenv" not in sys.modules:
    dotenv_stub = types.ModuleType("dotenv")
    dotenv_stub.load_dotenv = lambda *_args, **_kwargs: None
    dotenv_stub.set_key = lambda *_args, **_kwargs: None
    dotenv_stub.find_dotenv = lambda *_args, **_kwargs: ""
    sys.modules["dotenv"] = dotenv_stub

if "anthropic" not in sys.modules:
    anthropic_stub = types.ModuleType("anthropic")

    class _StubAsyncAnthropic:
        def __init__(self, *_args, **_kwargs):
            pass

    anthropic_stub.AsyncAnthropic = _StubAsyncAnthropic
    anthropic_stub.APIError = Exception
    anthropic_stub.RateLimitError = Exception
    sys.modules["anthropic"] = anthropic_stub

if "huggingface_hub" not in sys.modules:
    hf_stub = types.ModuleType("huggingface_hub")

    class _StubHfApi:  # pragma: no cover - simple stub
        def __init__(self, *_args, **_kwargs):
            pass

    hf_stub.HfApi = _StubHfApi
    sys.modules["huggingface_hub"] = hf_stub

if "pygame" not in sys.modules:
    class _Clock:  # pragma: no cover - simple stub
        def tick(self, *_args, **_kwargs):
            return None

    pygame_stub = types.ModuleType("pygame")
    pygame_stub.mixer = types.SimpleNamespace(
        init=lambda *_args, **_kwargs: None,
        music=types.SimpleNamespace(
            load=lambda *_args, **_kwargs: None,
            play=lambda *_args, **_kwargs: None,
            get_busy=lambda *_args, **_kwargs: False,
        ),
        Sound=lambda *_args, **_kwargs: types.SimpleNamespace(play=lambda *_args, **_kwargs: None),
    )
    pygame_stub.time = types.SimpleNamespace(Clock=lambda: _Clock())
    sys.modules["pygame"] = pygame_stub

if "requests" not in sys.modules:
    class _DummyResponse:  # pragma: no cover - simple stub
        ok = True
        text = ""

        def json(self):
            return {}

        def iter_content(self, chunk_size=1024):
            return iter(())

    def _dummy_request(*_args, **_kwargs):
        return _DummyResponse()

    requests_stub = types.ModuleType("requests")
    requests_stub.get = _dummy_request
    requests_stub.post = _dummy_request
    requests_stub.__path__ = []
    exceptions_module = types.ModuleType("requests.exceptions")

    class _RequestException(Exception):
        """Stub base class mirroring requests.exceptions.RequestException."""

    class _HTTPError(_RequestException):
        """Stub HTTP error matching requests.exceptions.HTTPError."""

    exceptions_module.RequestException = _RequestException
    exceptions_module.HTTPError = _HTTPError
    requests_stub.exceptions = exceptions_module
    class _DummyResponse:
        status_code = 200
        text = ""
        content = b""

        def json(self):
            return {}

    requests_stub.HTTPError = exceptions_module.HTTPError
    requests_stub.RequestException = exceptions_module.RequestException
    requests_stub.Response = _DummyResponse
    adapters_module = types.ModuleType("requests.adapters")

    class _HTTPAdapter:
        """Stub adapter mirroring requests.adapters.HTTPAdapter."""

        def __init__(self, *args, **kwargs):
            pass

        def send(self, request, **kwargs):  # pragma: no cover - stub helper
            return request

    adapters_module.HTTPAdapter = _HTTPAdapter
    requests_stub.adapters = adapters_module
    sys.modules["requests"] = requests_stub
    sys.modules["requests.exceptions"] = exceptions_module
    sys.modules["requests.adapters"] = adapters_module

if "google" not in sys.modules:
    google_module = types.ModuleType("google")
    google_module.__path__ = []

    cloud_module = types.ModuleType("google.cloud")
    cloud_module.__path__ = []

    speech_module = types.ModuleType("google.cloud.speech_v1p1beta1")

    class _DummySpeechClient:  # pragma: no cover - simple stub
        def __init__(self, *_args, **_kwargs):
            pass

        def synthesize_speech(self, *_args, **_kwargs):
            class _Response:
                audio_content = b""

            return _Response()

        def list_voices(self, *_args, **_kwargs):
            class _Response:
                voices = []

            return _Response()

    class _DummySsmlVoiceGender:
        NEUTRAL = 0

        def __call__(self, *_args, **_kwargs):
            class _Result:
                name = "NEUTRAL"

            return _Result()

    speech_module.SpeechClient = _DummySpeechClient
    speech_module.SynthesisInput = lambda *_args, **_kwargs: None
    speech_module.AudioConfig = lambda *_args, **_kwargs: None
    speech_module.AudioEncoding = types.SimpleNamespace(MP3="MP3")
    speech_module.SsmlVoiceGender = _DummySsmlVoiceGender()
    speech_module.VoiceSelectionParams = lambda *_args, **_kwargs: None

    cloud_module.speech_v1p1beta1 = speech_module
    google_module.cloud = cloud_module
    sys.modules["google"] = google_module
    sys.modules["google.cloud"] = cloud_module
    sys.modules["google.cloud.speech_v1p1beta1"] = speech_module

if "modules.Providers.HuggingFace.HF_gen_response" not in sys.modules:
    hf_module = types.ModuleType("modules.Providers.HuggingFace.HF_gen_response")

    class _StubHuggingFaceGenerator:  # pragma: no cover - simple stub
        def __init__(self, *_args, **_kwargs):
            self.model_manager = types.SimpleNamespace(installed=[])  # minimal attribute

        async def load_model(self, *_args, **_kwargs):  # pragma: no cover - async stub
            return None

    hf_module.HuggingFaceGenerator = _StubHuggingFaceGenerator
    hf_module.search_models = lambda *_args, **_kwargs: []
    hf_module.download_model = lambda *_args, **_kwargs: {}
    hf_module.update_model_settings = lambda *_args, **_kwargs: {}
    hf_module.clear_cache = lambda *_args, **_kwargs: None
    sys.modules["modules.Providers.HuggingFace.HF_gen_response"] = hf_module

if "modules.Providers.Grok.grok_generate_response" not in sys.modules:
    grok_module = types.ModuleType("modules.Providers.Grok.grok_generate_response")

    class _StubGrokGenerator:  # pragma: no cover - simple stub
        async def generate_response(self, *_args, **_kwargs):
            return ""

        async def process_streaming_response(self, *_args, **_kwargs):
            return ""

        async def unload_model(self):
            return None

    grok_module.GrokGenerator = _StubGrokGenerator
    sys.modules["modules.Providers.Grok.grok_generate_response"] = grok_module

for module_name in [
    "modules.Providers.OpenAI.OA_gen_response",
    "modules.Providers.Mistral.Mistral_gen_response",
    "modules.Providers.Google.GG_gen_response",
]:
    if module_name not in sys.modules:
        module = types.ModuleType(module_name)

        async def _async_response(*_args, **_kwargs):  # pragma: no cover - async stub
            return ""

        module.generate_response = _async_response

        if module_name.endswith("Anthropic_gen_response"):
            class _StubAnthropicGenerator:
                def __init__(self, *_args, **_kwargs):
                    self.default_model = "claude-3-opus-20240229"

                async def generate_response(self, *_args, **_kwargs):  # pragma: no cover
                    return ""

                async def process_streaming_response(self, response):  # pragma: no cover
                    items = []
                    async for entry in response:
                        items.append(entry)
                    return "".join(items)

                def set_default_model(self, model):
                    self.default_model = model

                def set_streaming(self, _value):
                    return None

                def set_function_calling(self, _value):
                    return None

                def set_timeout(self, _value):
                    return None

                def set_max_retries(self, _value):
                    return None

                def set_retry_delay(self, _value):
                    return None

            module.AnthropicGenerator = _StubAnthropicGenerator
            module.setup_anthropic_generator = lambda _cfg=None: _StubAnthropicGenerator()

            class _StubAsyncAnthropic:
                def __init__(self, *_args, **_kwargs):
                    pass

            module.AsyncAnthropic = _StubAsyncAnthropic
            module.APIError = RuntimeError
            module.RateLimitError = RuntimeError

        sys.modules[module_name] = module

if "tenacity" not in sys.modules:
    tenacity_stub = types.ModuleType("tenacity")

    def _identity_decorator(*_args, **_kwargs):
        def _wrap(func):
            return func

        if _args and callable(_args[0]) and len(_args) == 1 and not _kwargs:
            return _args[0]
        return _wrap

    tenacity_stub.retry = _identity_decorator
    tenacity_stub.stop_after_attempt = lambda *_args, **_kwargs: None
    tenacity_stub.wait_exponential = lambda *_args, **_kwargs: None
    tenacity_stub.retry_if_exception_type = lambda *_args, **_kwargs: None

    sys.modules["tenacity"] = tenacity_stub

if "modules.Speech_Services.Google_stt" not in sys.modules:
    google_stt_module = types.ModuleType("modules.Speech_Services.Google_stt")

    class _StubGoogleSTT:  # pragma: no cover - simple stub
        def __init__(self, *_args, **_kwargs):
            pass

        def listen(self, *_args, **_kwargs):
            return None

        def stop_listening(self, *_args, **_kwargs):
            return None

        def transcribe(self, *_args, **_kwargs):
            return ""

    google_stt_module.GoogleSTT = _StubGoogleSTT
    sys.modules["modules.Speech_Services.Google_stt"] = google_stt_module

from ATLAS.ATLAS import ATLAS


class _DummyLogger:
    def debug(self, *_args, **_kwargs):
        pass

    def info(self, *_args, **_kwargs):
        pass

    def warning(self, *_args, **_kwargs):
        pass

    def error(self, *_args, **_kwargs):
        pass


class _StubPersonaManager:
    def __init__(self):
        self.persona_names = ["ATLAS", "Helper"]
        self.default_persona_name = "ATLAS"
        self.current_persona: Dict[str, Any] = {"name": "Helper"}
        self.current_system_prompt = "system-prompt"
        self.editor_requests: list[str] = []
        self.compute_calls: list[tuple[Optional[str], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]] = []
        self.flag_calls: list[tuple[str, str, Any, Optional[Dict[str, Any]]]] = []
        self.messages: list[tuple[str, str]] = []
        self.update_payload: Optional[tuple] = None

    def get_current_persona_prompt(self) -> str:
        return self.current_system_prompt

    def get_editor_state(self, persona_name: str) -> Dict[str, Any]:
        self.editor_requests.append(persona_name)
        return {"general": {"name": persona_name}}

    def compute_locked_content(
        self,
        persona_name: Optional[str] = None,
        *,
        general: Optional[Dict[str, Any]] = None,
        flags: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        self.compute_calls.append((persona_name, general, flags))
        return {"start_locked": "start", "end_locked": "end"}

    def set_flag(
        self,
        persona_name: str,
        flag: str,
        enabled: Any,
        extras: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self.flag_calls.append((persona_name, flag, enabled, extras))
        return {"success": True, "persona": {"name": persona_name}}

    def update_persona_from_form(
        self,
        persona_name: str,
        general: Optional[Dict[str, Any]] = None,
        persona_type: Optional[Dict[str, Any]] = None,
        provider: Optional[Dict[str, Any]] = None,
        speech: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self.update_payload = (persona_name, general, persona_type, provider, speech)
        return {"success": True, "persona": {"name": persona_name}}

    def show_message(self, role: str, message: str) -> None:
        self.messages.append((role, message))


@pytest.fixture
def atlas_with_stub_personas():
    atlas = ATLAS.__new__(ATLAS)
    atlas.logger = _DummyLogger()
    atlas.persona_manager = _StubPersonaManager()
    atlas.current_persona = atlas.persona_manager.current_persona
    return atlas


def test_get_active_persona_properties_use_persona_manager(atlas_with_stub_personas):
    atlas = atlas_with_stub_personas

    assert atlas.get_persona_names() == ["ATLAS", "Helper"]
    assert atlas.get_active_persona_name() == "Helper"
    assert atlas.get_current_persona_prompt() == "system-prompt"

    atlas.persona_manager.current_persona = {}
    assert atlas.get_active_persona_name() == "ATLAS"

    atlas.persona_manager.default_persona_name = ""
    assert atlas.get_active_persona_name() == "Assistant"


def test_persona_facade_delegates_calls(atlas_with_stub_personas):
    atlas = atlas_with_stub_personas

    state = atlas.get_persona_editor_state("Helper")
    assert state["general"]["name"] == "Helper"
    assert atlas.persona_manager.editor_requests == ["Helper"]

    preview = atlas.compute_persona_locked_content(
        "Helper",
        general={"name": "Helper"},
        flags={"user_profile_enabled": True},
    )
    assert preview == {"start_locked": "start", "end_locked": "end"}
    assert atlas.persona_manager.compute_calls == [
        ("Helper", {"name": "Helper"}, {"user_profile_enabled": True})
    ]

    response = atlas.set_persona_flag("Helper", "sys_info_enabled", True, {"extra": "value"})
    assert response["success"] is True
    assert atlas.persona_manager.flag_calls == [
        ("Helper", "sys_info_enabled", True, {"extra": "value"})
    ]

    update_response = atlas.update_persona_from_editor(
        "Helper",
        general={"name": "Helper"},
        persona_type={"sys_info_enabled": True},
        provider={"provider": "openai", "model": "gpt"},
        speech={"Speech_provider": "11labs", "voice": "jack"},
    )
    assert update_response["success"] is True
    assert atlas.persona_manager.update_payload == (
        "Helper",
        {"name": "Helper"},
        {"sys_info_enabled": True},
        {"provider": "openai", "model": "gpt"},
        {"Speech_provider": "11labs", "voice": "jack"},
    )

    atlas.show_persona_message("system", "saved")
    assert atlas.persona_manager.messages == [("system", "saved")]
