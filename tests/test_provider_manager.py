import asyncio
import sys
import types

if "gi" not in sys.modules:
    gi_stub = types.ModuleType("gi")

    def _require_version(_name, _version):  # pragma: no cover - simple stub
        return None

    gi_stub.require_version = _require_version

    gtk_module = types.ModuleType("Gtk")

    class _Widget:
        def __init__(self, *args, **kwargs):
            self.children = []

        def set_tooltip_text(self, text):
            self.tooltip_text = text

        def set_hexpand(self, value):
            self.hexpand = value

        def set_vexpand(self, value):
            self.vexpand = value

        def set_halign(self, value):
            self.halign = value

        def set_valign(self, value):
            self.valign = value

        def set_margin_top(self, value):
            self.margin_top = value

        def set_margin_bottom(self, value):
            self.margin_bottom = value

        def set_margin_start(self, value):
            self.margin_start = value

        def set_margin_end(self, value):
            self.margin_end = value

    class Window(_Widget):
        def __init__(self, title=""):
            super().__init__()
            self.title = title
            self.child = None
            self.modal = False
            self.presented = False
            self.closed = False

        def set_transient_for(self, parent):
            self.transient_for = parent

        def set_modal(self, modal):
            self.modal = modal

        def set_default_size(self, width, height):
            self.default_size = (width, height)

        def set_child(self, child):
            self.child = child

        def present(self):
            self.presented = True

        def close(self):
            self.closed = True

    class Box(_Widget):
        def __init__(self, orientation=None, spacing=0):
            super().__init__()
            self.orientation = orientation
            self.spacing = spacing

        def append(self, child):
            self.children.append(child)

    class Grid(_Widget):
        def __init__(self, column_spacing=0, row_spacing=0):
            super().__init__()
            self.column_spacing = column_spacing
            self.row_spacing = row_spacing
            self.attachments = []

        def attach(self, child, column, row, width, height):
            self.attachments.append((child, column, row, width, height))
            self.children.append(child)

    class Label(_Widget):
        def __init__(self, label=""):
            super().__init__()
            self.label = label

        def set_xalign(self, value):
            self.xalign = value

        def set_yalign(self, value):
            self.yalign = value

    class ComboBoxText(_Widget):
        def __init__(self):
            super().__init__()
            self._items = []
            self._active = -1

        def append_text(self, text):
            self._items.append(text)

        def remove_all(self):
            self._items = []
            self._active = -1

        def set_active(self, index):
            if 0 <= index < len(self._items):
                self._active = index

        def get_active_text(self):
            if 0 <= self._active < len(self._items):
                return self._items[self._active]
            return None

    class Adjustment:
        def __init__(self, value=0.0, lower=0.0, upper=1.0, step_increment=0.1, page_increment=0.1):
            self.value = value
            self.lower = lower
            self.upper = upper
            self.step_increment = step_increment
            self.page_increment = page_increment

    class SpinButton(_Widget):
        def __init__(self, adjustment=None, digits=0):
            super().__init__()
            self.adjustment = adjustment or Adjustment()
            self.digits = digits
            self.value = self.adjustment.value

        def set_increments(self, step, page):
            self.step_increment = step
            self.page_increment = page

        def set_value(self, value):
            self.value = value

        def get_value(self):
            return self.value

        def get_value_as_int(self):
            return int(round(self.value))

    class CheckButton(_Widget):
        def __init__(self, label=""):
            super().__init__()
            self.label = label
            self.active = False
            self._handlers = {}

        def set_active(self, value):
            self.active = bool(value)

        def get_active(self):
            return self.active

        def connect(self, signal, callback):
            self._handlers.setdefault(signal, []).append(callback)

    class Entry(_Widget):
        def __init__(self):
            super().__init__()
            self.text = ""
            self.placeholder = ""
            self.visible = True

        def set_text(self, text):
            self.text = text

        def get_text(self):
            return self.text

        def set_placeholder_text(self, text):
            self.placeholder = text

        def set_visibility(self, value):
            self.visible = bool(value)

    class Button(_Widget):
        def __init__(self, label=""):
            super().__init__()
            self.label = label
            self._handlers = {}

        def connect(self, signal, callback):
            self._handlers.setdefault(signal, []).append(callback)

    class MessageType:
        ERROR = "error"
        INFO = "info"

    class ButtonsType:
        OK = "ok"

    class MessageDialog(Window):
        def __init__(self, transient_for=None, modal=False, message_type=None, buttons=None, text=""):
            super().__init__(title=text)
            self.transient_for = transient_for
            self.modal = modal
            self.message_type = message_type
            self.buttons = buttons
            self.secondary_text = ""

        def set_secondary_text(self, text):
            self.secondary_text = text

        def connect(self, signal, callback):
            if signal == "response":
                self._response_handler = callback

        def present(self):
            self.presented = True

    class Orientation:
        VERTICAL = "vertical"
        HORIZONTAL = "horizontal"

    class Align:
        START = "start"
        END = "end"

    class AccessibleRole:
        BUTTON = "button"

    class PolicyType:
        AUTOMATIC = "automatic"
        NEVER = "never"

    gtk_module.Window = Window
    gtk_module.Box = Box
    gtk_module.Grid = Grid
    gtk_module.Label = Label
    gtk_module.ComboBoxText = ComboBoxText
    gtk_module.Adjustment = Adjustment
    gtk_module.SpinButton = SpinButton
    gtk_module.CheckButton = CheckButton
    gtk_module.Entry = Entry
    gtk_module.Button = Button
    gtk_module.MessageDialog = MessageDialog
    gtk_module.MessageType = MessageType
    gtk_module.ButtonsType = ButtonsType
    gtk_module.Orientation = Orientation
    gtk_module.Align = Align
    gtk_module.AccessibleRole = AccessibleRole
    gtk_module.PolicyType = PolicyType

    repository_module = types.ModuleType("gi.repository")
    repository_module.Gtk = gtk_module

    glib_module = types.ModuleType("GLib")

    def idle_add(func, *args, **kwargs):
        return func(*args, **kwargs)

    glib_module.idle_add = idle_add
    repository_module.GLib = glib_module

    gdk_module = types.ModuleType("Gdk")
    repository_module.Gdk = gdk_module

    sys.modules["gi"] = gi_stub
    sys.modules["gi"].repository = repository_module
    sys.modules["gi.repository"] = repository_module
    sys.modules["gi.repository.Gtk"] = gtk_module
    sys.modules["gi.repository.GLib"] = glib_module
    sys.modules["gi.repository.Gdk"] = gdk_module

if "tenacity" not in sys.modules:
    tenacity_stub = types.ModuleType("tenacity")

    def _identity_decorator(*_args, **_kwargs):
        def _decorator(fn):
            return fn

        return _decorator

    tenacity_stub.retry = _identity_decorator
    tenacity_stub.stop_after_attempt = lambda *args, **kwargs: None
    tenacity_stub.wait_exponential = lambda *args, **kwargs: None
    sys.modules["tenacity"] = tenacity_stub

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
    hf_hub_stub = types.ModuleType("huggingface_hub")

    class _PlaceholderHfApi:
        def __init__(self, *_args, **_kwargs):  # pragma: no cover - placeholder
            self.token = None

        def whoami(self, token=None):  # pragma: no cover - placeholder
            self.token = token
            return {}

    hf_hub_stub.HfApi = _PlaceholderHfApi
    sys.modules["huggingface_hub"] = hf_hub_stub

hf_module_name = "modules.Providers.HuggingFace.HF_gen_response"
if hf_module_name not in sys.modules:
    hf_stub = types.ModuleType(hf_module_name)

    class _PlaceholderGenerator:
        async def load_model(self, *_args, **_kwargs):
            return None

        def unload_model(self):
            return None

        def get_installed_models(self):
            return []

        async def generate_response(self, *_args, **_kwargs):  # pragma: no cover - placeholder
            return ""

        async def process_streaming_response(self, response):  # pragma: no cover - placeholder
            collected = []
            async for chunk in response:
                collected.append(chunk)
            return "".join(collected)

    hf_stub.HuggingFaceGenerator = _PlaceholderGenerator
    async def _async_list(*_args, **_kwargs):  # pragma: no cover - placeholder
        return []

    async def _async_dict(*_args, **_kwargs):  # pragma: no cover - placeholder
        return {}

    def _sync_noop(*_args, **_kwargs):  # pragma: no cover - placeholder
        return {}

    hf_stub.search_models = _async_list
    hf_stub.download_model = _async_dict
    hf_stub.update_model_settings = _sync_noop
    hf_stub.clear_cache = lambda *_args, **_kwargs: None
    sys.modules[hf_module_name] = hf_stub

grok_module_name = "modules.Providers.Grok.grok_generate_response"
if grok_module_name not in sys.modules:
    grok_stub = types.ModuleType(grok_module_name)

    class _PlaceholderGrokGenerator:
        def __init__(self, *_args, **_kwargs):
            pass

        async def generate_response(self, *_args, **_kwargs):  # pragma: no cover - placeholder
            return ""

        async def process_streaming_response(self, response):  # pragma: no cover - placeholder
            collected = []
            async for chunk in response:
                collected.append(chunk)
            return "".join(collected)

        async def unload_model(self):  # pragma: no cover - placeholder
            return None

    grok_stub.GrokGenerator = _PlaceholderGrokGenerator
    sys.modules[grok_module_name] = grok_stub


def _ensure_async_function_module(module_name: str, return_value: str):
    if module_name in sys.modules:
        return

    stub = types.ModuleType(module_name)

    async def _generate_response(*_args, **_kwargs):  # pragma: no cover - placeholder
        return return_value

    stub.generate_response = _generate_response
    sys.modules[module_name] = stub


_ensure_async_function_module("modules.Providers.OpenAI.OA_gen_response", "openai")
_ensure_async_function_module("modules.Providers.Mistral.Mistral_gen_response", "mistral")
_ensure_async_function_module("modules.Providers.Google.GG_gen_response", "google")
_ensure_async_function_module("modules.Providers.Anthropic.Anthropic_gen_response", "anthropic")

import pytest

import ATLAS.provider_manager as provider_manager_module
from ATLAS.provider_manager import ProviderManager
from GTKUI.Provider_manager.Settings.OA_settings import OpenAISettingsWindow


class DummyConfig:
    def __init__(self, root_path):
        self._root_path = root_path
        self._hf_token = ""
        self._api_keys = {}
        self._default_model = "gpt-4o"
        self._openai_settings = {
            "model": "gpt-4o",
            "temperature": 0.0,
            "max_tokens": 4000,
            "stream": True,
            "organization": None,
        }
        self._openai_api_key = ""

    def get_default_provider(self):
        return "OpenAI"

    def get_openai_llm_settings(self):
        return dict(self._openai_settings)

    def set_openai_llm_settings(
        self,
        *,
        model,
        temperature=None,
        max_tokens=None,
        stream=None,
        api_key=None,
        organization=None,
    ):
        if model:
            self._openai_settings["model"] = model
            self._default_model = model
        if temperature is not None:
            self._openai_settings["temperature"] = float(temperature)
        if max_tokens is not None:
            self._openai_settings["max_tokens"] = int(max_tokens)
        if stream is not None:
            self._openai_settings["stream"] = bool(stream)
        self._openai_settings["organization"] = organization
        if api_key is not None:
            self._openai_api_key = api_key
        return dict(self._openai_settings)

    def get_app_root(self):
        return self._root_path

    def get_model_cache_dir(self):
        return self._root_path

    def get_llm_config(self, *_args, **_kwargs):
        return {
            "provider": "OpenAI",
            "model": "gpt-4o",
            "max_tokens": 4000,
            "temperature": 0.0,
            "stream": True,
        }

    def get_config(self, *_args, **_kwargs):
        return None

    def get_huggingface_api_key(self):
        return self._hf_token

    def get_openai_api_key(self):
        return self._openai_api_key or self._api_keys.get("OpenAI", "")

    def set_huggingface_api_key(self, token):
        self.set_hf_token(token)

    def set_hf_token(self, token):
        if not token:
            raise ValueError("Hugging Face token cannot be empty.")
        self._hf_token = token

    def update_api_key(self, provider_name: str, new_api_key: str):
        if not provider_name:
            raise ValueError("Provider name must be provided.")
        self._api_keys[provider_name] = new_api_key

    def has_provider_api_key(self, provider_name: str) -> bool:
        return bool(self._api_keys.get(provider_name))

    def get_available_providers(self):
        return {
            "OpenAI": self._api_keys.get("OpenAI"),
            "Mistral": self._api_keys.get("Mistral"),
            "Google": self._api_keys.get("Google"),
            "HuggingFace": self._hf_token,
            "Anthropic": self._api_keys.get("Anthropic"),
            "Grok": self._api_keys.get("Grok"),
            "ElevenLabs": self._api_keys.get("ElevenLabs"),
        }


class FakeHFModelManager:
    def __init__(self):
        self.installed_models = ["alpha", "beta"]
        self.current_model = None

    def get_installed_models(self):
        return list(self.installed_models)

    def remove_installed_model(self, model_name: str):
        if model_name in self.installed_models:
            self.installed_models.remove(model_name)
        if self.current_model == model_name:
            self.current_model = None


class FakeHFGenerator:
    def __init__(self, _config_manager):
        self.model_manager = FakeHFModelManager()
        self.loaded_models = []
        self.unload_calls = 0
        self.updated_settings = None
        self.cache_cleared = 0

    async def load_model(self, model_name: str, force_download: bool = False):
        await asyncio.sleep(0)
        self.loaded_models.append((model_name, force_download))
        self.model_manager.current_model = model_name

    def unload_model(self):
        self.unload_calls += 1
        self.model_manager.current_model = None

    def get_installed_models(self):
        return self.model_manager.get_installed_models()

    def get_current_model(self):
        return self.model_manager.current_model

    def update_model_settings(self, settings):  # pragma: no cover - updated via helper
        self.updated_settings = settings

    def clear_model_cache(self):  # pragma: no cover - updated via helper
        self.cache_cleared += 1

    async def generate_response(self, *_args, **_kwargs):  # pragma: no cover - not exercised
        return "stubbed"

    async def process_streaming_response(self, response):  # pragma: no cover - not exercised
        collected = []
        async for chunk in response:
            collected.append(chunk)
        return "".join(collected)


@pytest.fixture
def provider_manager(tmp_path, monkeypatch):
    ProviderManager._instance = None

    def fake_load_models(self):
        self.models = {
            "OpenAI": ["gpt-4o"],
            "HuggingFace": ["alpha", "beta"],
            "Mistral": [],
            "Google": [],
            "Anthropic": [],
            "Grok": [],
        }
        self.current_model = None
        self.current_provider = None

    monkeypatch.setattr(provider_manager_module.ModelManager, "load_models", fake_load_models, raising=False)
    monkeypatch.setattr(provider_manager_module, "HuggingFaceGenerator", FakeHFGenerator)

    config = DummyConfig(tmp_path.as_posix())
    manager = asyncio.run(ProviderManager.create(config))
    yield manager

    ProviderManager._instance = None


def test_huggingface_facade_handles_model_lifecycle(provider_manager):
    ensure_first = provider_manager.ensure_huggingface_ready()
    assert ensure_first["success"] is True
    generator = provider_manager.huggingface_generator
    assert isinstance(generator, FakeHFGenerator)

    ensure_second = provider_manager.ensure_huggingface_ready()
    assert ensure_second["success"] is True
    assert provider_manager.huggingface_generator is generator

    models_result = provider_manager.list_hf_models()
    assert models_result["success"] is True
    assert models_result["data"] == ["alpha", "beta"]

    async def exercise():
        await provider_manager.switch_llm_provider("HuggingFace")
        assert generator.loaded_models[0] == ("alpha", False)

        load_result = await provider_manager.load_hf_model("beta", force_download=True)
        assert load_result["success"] is True
        assert ("beta", True) in generator.loaded_models
        assert provider_manager.current_model == "beta"

        unload_result = await provider_manager.unload_hf_model()
        assert unload_result["success"] is True
        assert generator.unload_calls == 1
        assert provider_manager.current_model is None

        remove_result = await provider_manager.remove_hf_model("beta")
        assert remove_result["success"] is True

    asyncio.run(exercise())

    refreshed = provider_manager.list_hf_models()
    assert refreshed["success"] is True
    assert refreshed["data"] == ["alpha"]


def test_huggingface_backend_helpers(provider_manager, monkeypatch):
    search_calls = {}

    async def fake_search(generator, query, filters=None, limit=10):
        search_calls["args"] = (generator, query, filters, limit)
        return [{"id": "zeta", "tags": ["tag"], "downloads": 42, "likes": 5}]

    download_calls = []

    async def fake_download(generator, model_id, force=False):
        download_calls.append((generator, model_id, force))
        return {"model_id": model_id}

    updated_settings = {}

    def fake_update(generator, settings):
        updated_settings["settings"] = settings
        return settings

    clear_calls = []

    def fake_clear(generator):
        clear_calls.append(generator)

    monkeypatch.setattr(provider_manager_module, "hf_search_models", fake_search)
    monkeypatch.setattr(provider_manager_module, "hf_download_model", fake_download)
    monkeypatch.setattr(provider_manager_module, "hf_update_model_settings", fake_update)
    monkeypatch.setattr(provider_manager_module, "hf_clear_cache", fake_clear)

    async def exercise():
        result = await provider_manager.search_huggingface_models("llama", {"pipeline_tag": "text-generation"}, limit=5)
        assert result["success"] is True
        assert result["data"][0]["id"] == "zeta"
        assert search_calls["args"][1] == "llama"
        assert search_calls["args"][2]["pipeline_tag"] == "text-generation"
        assert search_calls["args"][0] is provider_manager.huggingface_generator

        download_result = await provider_manager.download_huggingface_model("omega", force=True)
        assert download_result["success"] is True
        assert download_calls[0] == (provider_manager.huggingface_generator, "omega", True)

    asyncio.run(exercise())

    update_result = provider_manager.update_huggingface_settings({"temperature": 0.5})
    assert update_result["success"] is True
    assert updated_settings["settings"]["temperature"] == 0.5

    clear_result = provider_manager.clear_huggingface_cache()
    assert clear_result["success"] is True
    assert clear_calls[0] is provider_manager.huggingface_generator


def test_save_huggingface_token_refreshes_generator(provider_manager):
    initial = provider_manager.ensure_huggingface_ready()
    assert initial["success"] is True
    original_generator = provider_manager.huggingface_generator

    result = provider_manager.save_huggingface_token("  new-token  ")
    assert result["success"] is True
    assert provider_manager.config_manager.get_huggingface_api_key() == "new-token"
    assert provider_manager.huggingface_generator is not original_generator


def test_save_huggingface_token_rejects_empty(provider_manager):
    result = provider_manager.save_huggingface_token("   ")
    assert result["success"] is False
    assert "cannot be empty" in result["error"].lower()


def test_test_huggingface_token_success(provider_manager, monkeypatch):
    provider_manager.config_manager.set_hf_token("stored-token")

    class StubHfApi:
        def __init__(self):
            self.called_with = None

        def whoami(self, token=None):
            self.called_with = token
            return {"name": "tester"}

    stub_instance = StubHfApi()
    monkeypatch.setattr(provider_manager_module, "HfApi", lambda *args, **kwargs: stub_instance)

    async def exercise():
        return await provider_manager.test_huggingface_token(None)

    result = asyncio.run(exercise())
    assert result["success"] is True
    assert result["data"]["name"] == "tester"
    assert stub_instance.called_with == "stored-token"
    assert "Signed in as" in result["message"]


def test_test_huggingface_token_failure(provider_manager, monkeypatch):
    class StubHfApi:
        def __init__(self):
            self.called_with = None

        def whoami(self, token=None):
            self.called_with = token
            raise RuntimeError("invalid token")

    stub_instance = StubHfApi()
    monkeypatch.setattr(provider_manager_module, "HfApi", lambda *args, **kwargs: stub_instance)

    async def exercise():
        return await provider_manager.test_huggingface_token("explicit-token")

    result = asyncio.run(exercise())
    assert result["success"] is False
    assert "invalid token" in result["error"]
    assert stub_instance.called_with == "explicit-token"


def test_update_provider_api_key_refreshes_active_provider(provider_manager, monkeypatch):
    observed = {}

    async def fake_set_current_provider(provider):
        observed["provider"] = provider

    monkeypatch.setattr(provider_manager, "set_current_provider", fake_set_current_provider)

    result = asyncio.run(provider_manager.update_provider_api_key("OpenAI", "sk-test"))

    assert result["success"] is True
    assert "refreshed" in result["message"].lower()
    assert observed.get("provider") == "OpenAI"
    assert provider_manager.config_manager.get_openai_api_key() == "sk-test"


def test_update_provider_api_key_handles_refresh_failure(provider_manager, monkeypatch):
    async def fake_set_current_provider(_provider):
        raise RuntimeError("refresh failed")

    monkeypatch.setattr(provider_manager, "set_current_provider", fake_set_current_provider)

    result = asyncio.run(provider_manager.update_provider_api_key("OpenAI", "sk-failed"))

    assert result["success"] is False
    assert "refresh" in result["error"].lower()
    assert provider_manager.config_manager.get_openai_api_key() == "sk-failed"


def test_update_provider_api_key_rejects_empty_input(provider_manager):
    result = asyncio.run(provider_manager.update_provider_api_key("OpenAI", "   "))

    assert result["success"] is False
    assert "empty" in result["error"].lower()


def test_get_provider_api_key_status_without_saved_key(provider_manager):
    status = provider_manager.get_provider_api_key_status("OpenAI")

    assert status["has_key"] is False
    assert status["metadata"] == {}


def test_get_provider_api_key_status_with_saved_key(provider_manager):
    asyncio.run(provider_manager.update_provider_api_key("OpenAI", "sk-test"))
    status = provider_manager.get_provider_api_key_status("OpenAI")

    assert status["has_key"] is True
    metadata = status["metadata"]
    assert metadata["length"] == len("sk-test")
    assert metadata["hint"] == "\u2022" * len("sk-test")
    assert metadata["source"] == "environment"


def test_set_openai_llm_settings_updates_provider_state(provider_manager):
    result = provider_manager.set_openai_llm_settings(
        model="gpt-4o-mini",
        temperature=0.6,
        max_tokens=1024,
        stream=False,
        api_key="sk-live",
        organization="org-99",
    )

    assert result["success"] is True
    settings = provider_manager.get_openai_llm_settings()
    assert settings["model"] == "gpt-4o-mini"
    assert settings["stream"] is False
    assert provider_manager.model_manager.models["OpenAI"][0] == "gpt-4o-mini"
    assert provider_manager.current_model == "gpt-4o-mini"
    assert provider_manager.config_manager.get_openai_api_key() == "sk-live"


def test_openai_settings_window_populates_defaults_and_saves(provider_manager):
    atlas_stub = types.SimpleNamespace()

    saved_payload = {}

    def fake_set_openai_llm_settings(**kwargs):
        saved_payload.update(kwargs)
        return {"success": True, "message": "saved"}

    atlas_stub.provider_manager = provider_manager
    atlas_stub.set_openai_llm_settings = fake_set_openai_llm_settings
    atlas_stub.get_openai_llm_settings = lambda: {
        "model": "gpt-4o-mini",
        "temperature": 0.65,
        "max_tokens": 2048,
        "stream": False,
        "organization": "org-42",
    }

    provider_manager.config_manager._openai_api_key = "sk-stored"

    window = OpenAISettingsWindow(atlas_stub, provider_manager.config_manager, None)

    assert window.model_combo.get_active_text() == "gpt-4o-mini"
    assert window.temperature_spin.get_value() == 0.65
    assert window.max_tokens_spin.get_value_as_int() == 2048
    assert window.stream_toggle.get_active() is False
    assert window.api_key_entry.get_text() == "sk-stored"
    assert window.organization_entry.get_text() == "org-42"

    window.model_combo.set_active(1)
    window.temperature_spin.set_value(0.5)
    window.max_tokens_spin.set_value(4096)
    window.stream_toggle.set_active(True)
    window.api_key_entry.set_text("sk-new")
    window.organization_entry.set_text("org-new")

    window.on_save_clicked(window.model_combo)

    assert saved_payload["model"] == "gpt-4o"
    assert saved_payload["temperature"] == 0.5
    assert saved_payload["max_tokens"] == 4096
    assert saved_payload["stream"] is True
    assert saved_payload["api_key"] == "sk-new"
    assert saved_payload["organization"] == "org-new"
    assert window._last_message[0] == "Success"
    assert window.closed is True
