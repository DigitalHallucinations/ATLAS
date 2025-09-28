import asyncio
import asyncio
import importlib
import importlib
import inspect
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest


if "tenacity" not in sys.modules:
    tenacity_stub = types.ModuleType("tenacity")

    def _retry(*_args, **_kwargs):  # pragma: no cover - stub helper
        def decorator(func):
            return func

        return decorator

    tenacity_stub.retry = _retry
    tenacity_stub.stop_after_attempt = lambda *_args, **_kwargs: None  # pragma: no cover - stub helper
    tenacity_stub.wait_exponential = lambda *_args, **_kwargs: None  # pragma: no cover - stub helper
    tenacity_stub.retry_if_exception_type = lambda *_args, **_kwargs: None  # pragma: no cover - stub helper
    sys.modules["tenacity"] = tenacity_stub

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
        def __init__(self, *_, **__):
            pass

        class _Messages:
            async def create(self, *_, **__):  # pragma: no cover - stub helper
                return types.SimpleNamespace(content=[])

            def stream(self, *_, **__):  # pragma: no cover - stub helper
                raise RuntimeError("stream not supported in stub")

        @property
        def messages(self):  # pragma: no cover - stub helper
            return _StubAsyncAnthropic._Messages()

    anthropic_stub.AsyncAnthropic = _StubAsyncAnthropic
    anthropic_stub.APIError = Exception
    anthropic_stub.RateLimitError = Exception
    sys.modules["anthropic"] = anthropic_stub


@pytest.fixture
def atlas_class(monkeypatch):
    """Provide the ATLAS class with provider manager dependencies stubbed."""

    def ensure_module(name: str, module: types.ModuleType) -> None:
        monkeypatch.setitem(sys.modules, name, module)

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))

    # Reuse existing test stubs for extensive dependency mocking.
    import tests.test_atlas_status  # noqa: F401
    import tests.test_speech_manager  # noqa: F401

    yaml_stub = types.ModuleType("yaml")
    yaml_stub.safe_load = lambda *_args, **_kwargs: {}
    yaml_stub.dump = lambda *_args, **_kwargs: ""
    ensure_module("yaml", yaml_stub)

    dotenv_stub = types.ModuleType("dotenv")
    dotenv_stub.load_dotenv = lambda *_args, **_kwargs: None
    dotenv_stub.set_key = lambda *_args, **_kwargs: None
    dotenv_stub.find_dotenv = lambda *_args, **_kwargs: ""
    ensure_module("dotenv", dotenv_stub)

    hf_api_module = types.ModuleType("huggingface_hub")

    class _StubHfApi:
        def whoami(self, *_args, **_kwargs):
            return {}

    hf_api_module.HfApi = _StubHfApi
    ensure_module("huggingface_hub", hf_api_module)

    hf_module = types.ModuleType("modules.Providers.HuggingFace.HF_gen_response")

    class _FakeModelManager:
        def __init__(self):
            self.installed = []

        def remove_installed_model(self, model_name):
            self.installed = [m for m in self.installed if m != model_name]

    class _HuggingFaceGenerator:
        def __init__(self, *_args, **_kwargs):
            self.model_manager = _FakeModelManager()

        async def load_model(self, model_name, *_args, **_kwargs):
            self.model_manager.installed.append(model_name)

        def unload_model(self):
            return None

        def get_installed_models(self):
            return list(self.model_manager.installed)

    async def _hf_search(generator, *_args, **_kwargs):  # pragma: no cover - stub helper
        return []

    async def _hf_download(generator, model_id, force=False):  # pragma: no cover - stub helper
        generator.model_manager.installed.append(model_id)

    def _hf_update(generator, settings):  # pragma: no cover - stub helper
        return settings

    hf_module.HuggingFaceGenerator = _HuggingFaceGenerator
    hf_module.search_models = _hf_search
    hf_module.download_model = _hf_download
    hf_module.update_model_settings = _hf_update
    hf_module.clear_cache = lambda *_args, **_kwargs: None
    ensure_module("modules.Providers.HuggingFace.HF_gen_response", hf_module)

    grok_module = types.ModuleType("modules.Providers.Grok.grok_generate_response")

    class _GrokGenerator:
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

    grok_module.GrokGenerator = _GrokGenerator
    ensure_module("modules.Providers.Grok.grok_generate_response", grok_module)

    def _make_async_provider_module(name: str) -> types.ModuleType:
        module = types.ModuleType(name)

        async def _generate_response(*_args, **_kwargs):  # pragma: no cover - stub helper
            return ""

        module.generate_response = _generate_response
        return module

    for provider_module in [
        "modules.Providers.OpenAI.OA_gen_response",
        "modules.Providers.Mistral.Mistral_gen_response",
        "modules.Providers.Google.GG_gen_response",
        "modules.Providers.Anthropic.Anthropic_gen_response",
    ]:
        ensure_module(provider_module, _make_async_provider_module(provider_module))

    google_module = types.ModuleType("google")
    google_module.__path__ = []
    cloud_module = types.ModuleType("google.cloud")
    cloud_module.__path__ = []
    speech_module = types.ModuleType("google.cloud.speech_v1p1beta1")

    class _DummySpeechClient:
        def __init__(self, *_args, **_kwargs):
            pass

        def synthesize_speech(self, *_args, **_kwargs):
            return types.SimpleNamespace(audio_content=b"")

        def list_voices(self, *_args, **_kwargs):
            return types.SimpleNamespace(voices=[])

    class _DummySsmlVoiceGender:
        NEUTRAL = 0

        def __call__(self, *_args, **_kwargs):
            return types.SimpleNamespace(name="NEUTRAL")

    speech_module.SpeechClient = _DummySpeechClient
    speech_module.SynthesisInput = lambda *_args, **_kwargs: None
    speech_module.AudioConfig = lambda *_args, **_kwargs: None
    speech_module.AudioEncoding = types.SimpleNamespace(MP3="MP3")
    speech_module.SsmlVoiceGender = _DummySsmlVoiceGender()
    speech_module.VoiceSelectionParams = lambda *_args, **_kwargs: None

    cloud_module.speech_v1p1beta1 = speech_module
    google_module.cloud = cloud_module
    ensure_module("google", google_module)
    ensure_module("google.cloud", cloud_module)
    ensure_module("google.cloud.speech_v1p1beta1", speech_module)

    if "pygame" not in sys.modules:
        music = types.SimpleNamespace(
            load=lambda *_args, **_kwargs: None,
            play=lambda *_args, **_kwargs: None,
            get_busy=lambda: False,
        )
        mixer = types.SimpleNamespace(init=lambda *_args, **_kwargs: None, music=music)

        class _DummyClock:
            def tick(self, *_args, **_kwargs):
                return None

        pygame_stub = types.ModuleType("pygame")
        pygame_stub.mixer = mixer
        pygame_stub.time = types.SimpleNamespace(Clock=lambda: _DummyClock())
        ensure_module("pygame", pygame_stub)

    if "requests" not in sys.modules:
        class _DummyResponse:
            ok = True
            text = ""

            def json(self):
                return {}

            def iter_content(self, chunk_size=1024):  # pragma: no cover - stub helper
                return iter(())

        def _dummy_request(*_args, **_kwargs):  # pragma: no cover - stub helper
            return _DummyResponse()

        requests_stub = types.ModuleType("requests")
        requests_stub.get = _dummy_request
        requests_stub.post = _dummy_request
        ensure_module("requests", requests_stub)

    if "numpy" not in sys.modules:
        numpy_stub = types.ModuleType("numpy")
        numpy_stub.array = lambda *_args, **_kwargs: []
        numpy_stub.zeros = lambda *_args, **_kwargs: []
        ensure_module("numpy", numpy_stub)

    if "soundfile" not in sys.modules:
        soundfile_stub = types.ModuleType("soundfile")
        soundfile_stub.read = lambda *_args, **_kwargs: ([], 0)
        soundfile_stub.write = lambda *_args, **_kwargs: None
        ensure_module("soundfile", soundfile_stub)

    if "sounddevice" not in sys.modules:
        class _DummyInputStream:
            def __init__(self, *_args, **_kwargs):
                pass

            def start(self):
                return None

            def stop(self):
                return None

        sounddevice_stub = types.ModuleType("sounddevice")
        sounddevice_stub.InputStream = _DummyInputStream
        ensure_module("sounddevice", sounddevice_stub)

    for module_name in ["ATLAS.ATLAS", "ATLAS.provider_manager"]:
        sys.modules.pop(module_name, None)

    importlib.import_module("ATLAS")
    atlas_module = importlib.import_module("ATLAS.ATLAS")
    return atlas_module.ATLAS


def _stub_anthropic_generator_factory(collected_instances, streaming_enabled=True):
    class _StubAnthropicGenerator:
        def __init__(self, *_args, **_kwargs):
            self.streaming_enabled = streaming_enabled
            collected_instances.append(self)

        async def generate_response(self, *_args, stream=None, **_kwargs):
            self.last_stream = stream

            async def _iter():
                yield "hello"
                yield " world"

            return _iter()

        async def process_response(self, response):
            chunks = []
            async for part in response:
                chunks.append(part)
            return "".join(chunks)

    return _StubAnthropicGenerator


def test_anthropic_generate_response_sync_returns_text(monkeypatch):
    from modules.Providers.Anthropic import Anthropic_gen_response as anthropic

    instances = []
    monkeypatch.setattr(
        anthropic,
        "setup_anthropic_generator",
        lambda _cfg: _stub_anthropic_generator_factory(instances)(),
    )

    result = anthropic.generate_response_sync(Mock(), messages=[{"role": "user", "content": "hi"}])

    assert isinstance(result, str)
    assert result == "hello world"
    assert instances and instances[0].last_stream is True


def test_anthropic_generate_response_sync_running_loop_error(monkeypatch):
    from modules.Providers.Anthropic import Anthropic_gen_response as anthropic

    instances = []
    monkeypatch.setattr(
        anthropic,
        "setup_anthropic_generator",
        lambda _cfg: _stub_anthropic_generator_factory(instances)(),
    )

    loop = asyncio.new_event_loop()
    try:
        async def _invoke():
            with pytest.raises(RuntimeError) as excinfo:
                anthropic.generate_response_sync(Mock(), messages=[])
            return excinfo.value

        error = loop.run_until_complete(_invoke())
    finally:
        loop.close()

    assert "use the async Anthropic API instead" in str(error)


def test_anthropic_generator_uses_default_config(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
    monkeypatch.setenv("DEFAULT_PROVIDER", "OpenAI")

    from modules.Providers.Anthropic import Anthropic_gen_response as anthropic_module

    class _StubClient:
        def __init__(self, *_args, **_kwargs):
            pass

    monkeypatch.setattr(anthropic_module, "AsyncAnthropic", _StubClient)

    generator = anthropic_module.AnthropicGenerator()

    assert generator.api_key == "anthropic-key"


def _build_atlas(atlas_class):
    atlas = atlas_class.__new__(atlas_class)
    atlas.provider_manager = None
    return atlas


def test_run_in_background_delegates_to_task_runner(atlas_class, monkeypatch):
    atlas = _build_atlas(atlas_class)
    atlas.logger = Mock()

    import ATLAS.ATLAS as atlas_module

    sentinel = object()
    captured = {}

    def fake_runner(
        factory,
        *,
        on_success=None,
        on_error=None,
        logger=None,
        thread_name=None,
    ):
        captured.update(
            {
                "factory": factory,
                "on_success": on_success,
                "on_error": on_error,
                "logger": logger,
                "thread_name": thread_name,
            }
        )
        return sentinel

    monkeypatch.setattr(atlas_module, "run_async_in_thread", fake_runner)

    async def sample():  # pragma: no cover - executed via background helper
        return "ok"

    future = atlas.run_in_background(sample, thread_name="provider-thread")

    assert future is sentinel
    assert captured["factory"] is sample
    assert captured["logger"] is atlas.logger
    assert captured["thread_name"] == "provider-thread"
    assert captured["on_success"] is None
    assert captured["on_error"] is None


def test_set_current_provider_notifies_listeners(atlas_class):
    atlas = _build_atlas(atlas_class)
    atlas.logger = Mock()

    set_provider_mock = AsyncMock()
    get_model_mock = Mock(return_value="gpt-4o")
    atlas.provider_manager = SimpleNamespace(
        set_current_provider=set_provider_mock,
        get_current_model=get_model_mock,
    )

    set_provider_call = Mock()
    set_model_call = Mock()
    atlas.chat_session = SimpleNamespace(
        set_provider=set_provider_call,
        set_model=set_model_call,
    )

    atlas._provider_change_listeners = []
    summary = {"provider": "OpenAI"}
    atlas.get_chat_status_summary = Mock(return_value=summary)

    listener = Mock()
    atlas.add_provider_change_listener(listener)

    asyncio.run(atlas.set_current_provider("OpenAI"))

    set_provider_mock.assert_awaited_once_with("OpenAI")
    set_provider_call.assert_called_once_with("OpenAI")
    set_model_call.assert_called_once_with("gpt-4o")
    listener.assert_called_once_with(summary)


def test_set_current_provider_background_wrapper_uses_helper(atlas_class):
    atlas = _build_atlas(atlas_class)
    atlas.logger = Mock()

    async_call = AsyncMock()
    atlas.set_current_provider = async_call

    sentinel = object()
    success = Mock()
    error = Mock()
    atlas.run_in_background = Mock(return_value=sentinel)

    future = atlas.set_current_provider_in_background(
        "OpenAI", on_success=success, on_error=error
    )

    (factory,), kwargs = atlas.run_in_background.call_args
    coro = factory()
    assert inspect.isawaitable(coro)
    asyncio.run(coro)

    async_call.assert_awaited_once_with("OpenAI")
    assert kwargs["on_success"] is success
    assert kwargs["on_error"] is error
    assert kwargs["thread_name"] == "set-provider-OpenAI"
    assert future is sentinel


def test_update_api_key_background_wrapper_uses_helper(atlas_class):
    atlas = _build_atlas(atlas_class)
    atlas.logger = Mock()

    async_call = AsyncMock()
    atlas.update_provider_api_key = async_call

    sentinel = object()
    success = Mock()
    error = Mock()
    atlas.run_in_background = Mock(return_value=sentinel)

    future = atlas.update_provider_api_key_in_background(
        "OpenAI", "token", on_success=success, on_error=error
    )

    (factory,), kwargs = atlas.run_in_background.call_args
    coro = factory()
    assert inspect.isawaitable(coro)
    asyncio.run(coro)

    async_call.assert_awaited_once_with("OpenAI", "token")
    assert kwargs["on_success"] is success
    assert kwargs["on_error"] is error
    assert kwargs["thread_name"] == "update-api-key-OpenAI"
    assert future is sentinel


@pytest.mark.parametrize(
    "method_name, args, kwargs",
    [
        ("list_hf_models", (), {}),
        ("ensure_huggingface_ready", (), {}),
        ("get_provider_api_key_status", ("OpenAI",), {}),
        ("get_openai_llm_settings", (), {}),
    ],
)
def test_sync_provider_wrappers_require_manager(atlas_class, method_name, args, kwargs):
    atlas = _build_atlas(atlas_class)

    with pytest.raises(RuntimeError):
        getattr(atlas, method_name)(*args, **kwargs)


@pytest.mark.parametrize(
    "method_name, args, kwargs",
    [
        ("load_hf_model", ("alpha",), {}),
        ("update_provider_api_key", ("OpenAI", "token"), {}),
        ("refresh_current_provider", ("OpenAI",), {}),
        ("list_openai_models", (), {}),
    ],
)
def test_async_provider_wrappers_require_manager(atlas_class, method_name, args, kwargs):
    atlas = _build_atlas(atlas_class)

    with pytest.raises(RuntimeError):
        asyncio.run(getattr(atlas, method_name)(*args, **kwargs))


@pytest.mark.parametrize(
    "wrapper_name, provider_attr, args, kwargs",
    [
        ("load_hf_model", "load_hf_model", ("alpha",), {"force_download": True}),
        ("unload_hf_model", "unload_hf_model", tuple(), {}),
        ("remove_hf_model", "remove_hf_model", ("beta",), {}),
        ("download_hf_model", "download_huggingface_model", ("gamma",), {"force": True}),
        (
            "search_hf_models",
            "search_huggingface_models",
            ("llama", {"pipeline_tag": "text-generation"}),
            {"limit": 5},
        ),
        ("update_provider_api_key", "update_provider_api_key", ("OpenAI", "token"), {}),
    ],
)
def test_async_hf_wrappers_delegate(atlas_class, wrapper_name, provider_attr, args, kwargs):
    atlas = _build_atlas(atlas_class)
    expected = {"success": True, "message": "ok"}
    async_mock = AsyncMock(return_value=expected)
    atlas.provider_manager = SimpleNamespace(**{provider_attr: async_mock})

    result = asyncio.run(getattr(atlas, wrapper_name)(*args, **kwargs))

    assert result is expected
    async_mock.assert_awaited_once_with(*args, **kwargs)


def test_get_openai_llm_settings_delegates(atlas_class):
    atlas = _build_atlas(atlas_class)
    expected = {"model": "gpt-4o"}
    atlas.provider_manager = SimpleNamespace(
        get_openai_llm_settings=Mock(return_value=expected)
    )

    result = atlas.get_openai_llm_settings()

    assert result == expected
    atlas.provider_manager.get_openai_llm_settings.assert_called_once_with()


def test_list_openai_models_delegates(atlas_class):
    atlas = _build_atlas(atlas_class)
    expected = {"models": ["gpt-4o"], "error": None}
    async_mock = AsyncMock(return_value=expected)
    atlas.provider_manager = SimpleNamespace(list_openai_models=async_mock)

    result = asyncio.run(
        atlas.list_openai_models(base_url="https://api.example/v1", organization="org-1")
    )

    assert result is expected
    async_mock.assert_awaited_once_with(
        base_url="https://api.example/v1", organization="org-1"
    )


def test_async_hf_wrappers_propagate_exceptions(atlas_class):
    atlas = _build_atlas(atlas_class)
    failure = RuntimeError("boom")
    atlas.provider_manager = SimpleNamespace(load_hf_model=AsyncMock(side_effect=failure))

    with pytest.raises(RuntimeError) as excinfo:
        asyncio.run(atlas.load_hf_model("alpha"))

    assert excinfo.value is failure


@pytest.mark.parametrize(
    "wrapper_name, provider_attr, args, kwargs",
    [
        ("list_hf_models", "list_hf_models", tuple(), {}),
        ("update_hf_settings", "update_huggingface_settings", ({"temperature": 0.5},), {}),
        ("clear_hf_cache", "clear_huggingface_cache", tuple(), {}),
        ("save_hf_token", "save_huggingface_token", ("token",), {}),
        ("get_provider_api_key_status", "get_provider_api_key_status", ("HuggingFace",), {}),
        ("ensure_huggingface_ready", "ensure_huggingface_ready", tuple(), {}),
    ],
)
def test_sync_hf_wrappers_delegate(atlas_class, wrapper_name, provider_attr, args, kwargs):
    atlas = _build_atlas(atlas_class)
    expected = {"success": True, "message": "ok"}
    mock_method = Mock(return_value=expected)
    atlas.provider_manager = SimpleNamespace(**{provider_attr: mock_method})

    result = getattr(atlas, wrapper_name)(*args, **kwargs)

    assert result is expected
    mock_method.assert_called_once_with(*args, **kwargs)


def test_sync_hf_wrappers_propagate_exceptions(atlas_class):
    atlas = _build_atlas(atlas_class)
    failure = ValueError("bad cache")
    atlas.provider_manager = SimpleNamespace(clear_huggingface_cache=Mock(side_effect=failure))

    with pytest.raises(ValueError) as excinfo:
        atlas.clear_hf_cache()

    assert excinfo.value is failure


def test_refresh_current_provider_delegates_when_active(atlas_class):
    atlas = _build_atlas(atlas_class)
    set_mock = AsyncMock(return_value=None)
    atlas.provider_manager = SimpleNamespace(
        get_current_provider=Mock(return_value="OpenAI"),
        set_current_provider=set_mock,
    )

    result = asyncio.run(atlas.refresh_current_provider("OpenAI"))

    assert result == {
        "success": True,
        "message": "Provider OpenAI refreshed.",
        "provider": "OpenAI",
    }
    set_mock.assert_awaited_once_with("OpenAI")


def test_refresh_current_provider_skips_when_not_active(atlas_class):
    atlas = _build_atlas(atlas_class)
    set_mock = AsyncMock(return_value=None)
    atlas.provider_manager = SimpleNamespace(
        get_current_provider=Mock(return_value="OpenAI"),
        set_current_provider=set_mock,
    )

    result = asyncio.run(atlas.refresh_current_provider("Mistral"))

    assert result == {
        "success": False,
        "error": "Provider 'Mistral' is not the active provider.",
        "active_provider": "OpenAI",
    }
    set_mock.assert_not_called()


def test_refresh_current_provider_without_active_provider(atlas_class):
    atlas = _build_atlas(atlas_class)
    set_mock = AsyncMock(return_value=None)
    atlas.provider_manager = SimpleNamespace(
        get_current_provider=Mock(return_value=""),
        set_current_provider=set_mock,
    )

    result = asyncio.run(atlas.refresh_current_provider())

    assert result == {
        "success": False,
        "error": "No active provider is configured.",
    }
    set_mock.assert_not_called()


def test_refresh_current_provider_propagates_error(atlas_class):
    atlas = _build_atlas(atlas_class)
    failure = RuntimeError("boom")
    atlas.provider_manager = SimpleNamespace(
        get_current_provider=Mock(return_value="OpenAI"),
        set_current_provider=AsyncMock(side_effect=failure),
    )

    with pytest.raises(RuntimeError) as excinfo:
        asyncio.run(atlas.refresh_current_provider("OpenAI"))

    assert excinfo.value is failure


def test_get_speech_defaults_delegates_to_manager(atlas_class):
    atlas = _build_atlas(atlas_class)
    expected = {"tts_enabled": True}
    atlas.speech_manager = SimpleNamespace(describe_general_settings=Mock(return_value=expected))

    result = atlas.get_speech_defaults()

    assert result is expected


def test_get_speech_provider_status_delegates(atlas_class):
    atlas = _build_atlas(atlas_class)
    expected = {"has_key": True}
    atlas.speech_manager = SimpleNamespace(
        get_provider_credential_status=Mock(return_value=expected)
    )

    result = atlas.get_speech_provider_status("ElevenLabs")

    assert result is expected


def test_update_speech_defaults_invokes_manager(atlas_class):
    atlas = _build_atlas(atlas_class)
    configure_mock = Mock()
    atlas.speech_manager = SimpleNamespace(
        configure_defaults=configure_mock,
        describe_general_settings=Mock(return_value={}),
    )

    atlas.update_speech_defaults(
        tts_enabled=True,
        tts_provider="eleven_labs",
        stt_enabled=False,
        stt_provider="google",
    )

    configure_mock.assert_called_once_with(
        tts_enabled=True,
        tts_provider="eleven_labs",
        stt_enabled=False,
        stt_provider="google",
    )


def test_update_elevenlabs_settings_updates_voice(atlas_class):
    atlas = _build_atlas(atlas_class)
    voice = {"name": "Alpha", "voice_id": "alpha"}
    atlas.speech_manager = SimpleNamespace(
        set_elevenlabs_api_key=Mock(),
        resolve_tts_provider=Mock(return_value="eleven_labs"),
        get_default_tts_provider=Mock(return_value="eleven_labs"),
        get_tts_voices=Mock(return_value=[voice]),
        set_tts_voice=Mock(),
    )

    result = atlas.update_elevenlabs_settings(api_key="key", voice_id="alpha")

    atlas.speech_manager.set_elevenlabs_api_key.assert_called_once_with("key")
    atlas.speech_manager.set_tts_voice.assert_called_once_with(voice, "eleven_labs")
    assert result["updated_api_key"] is True
    assert result["updated_voice"] is True


def test_get_openai_speech_options_delegates(atlas_class):
    atlas = _build_atlas(atlas_class)
    expected = {"stt": []}
    atlas.speech_manager = SimpleNamespace(get_openai_option_sets=Mock(return_value=expected))

    assert atlas.get_openai_speech_options() is expected


def test_update_openai_speech_settings_persists_prepared_values(atlas_class):
    atlas = _build_atlas(atlas_class)
    prepared = {"api_key": "k", "stt_provider": "Whisper"}
    atlas.speech_manager = SimpleNamespace(
        normalize_openai_display_settings=Mock(return_value=prepared),
        set_openai_speech_config=Mock(),
    )

    result = atlas.update_openai_speech_settings({"api_key": "k"})

    atlas.speech_manager.normalize_openai_display_settings.assert_called_once_with({"api_key": "k"})
    atlas.speech_manager.set_openai_speech_config.assert_called_once_with(
        api_key="k",
        stt_provider="Whisper",
        language=prepared.get("language"),
        task=prepared.get("task"),
        initial_prompt=prepared.get("initial_prompt"),
        tts_provider=prepared.get("tts_provider"),
    )
    assert result is prepared


def test_get_transcription_history_delegates(atlas_class):
    atlas = _build_atlas(atlas_class)
    expected = [{"transcript": "hi"}]
    atlas.speech_manager = SimpleNamespace(
        describe_general_settings=Mock(return_value={}),
        get_provider_credential_status=Mock(return_value={}),
        list_tts_voice_options=Mock(return_value=[]),
        get_active_tts_summary=Mock(return_value=(None, None)),
        configure_defaults=Mock(),
        resolve_tts_provider=Mock(return_value=None),
        get_default_tts_provider=Mock(return_value=None),
        get_tts_voices=Mock(return_value=[]),
        set_tts_voice=Mock(),
        get_openai_option_sets=Mock(return_value={}),
        get_openai_display_config=Mock(return_value={}),
        normalize_openai_display_settings=Mock(return_value={}),
        set_openai_speech_config=Mock(),
        get_transcription_history=Mock(return_value=expected),
    )

    assert atlas.get_transcription_history(formatted=True) is expected
