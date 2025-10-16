import asyncio
import importlib.util
import json
import math
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import URLError
from weakref import WeakKeyDictionary

needs_stub = True
existing_gi = sys.modules.get("gi")
if existing_gi is not None and getattr(existing_gi, "_provider_manager_stub", False):
    needs_stub = False

if needs_stub:
    gi_stub = types.ModuleType("gi")
    gi_stub._provider_manager_stub = True

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

        def set_sensitive(self, value):
            self.sensitive = value

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

        def get_style_context(self):
            return types.SimpleNamespace(add_class=lambda *_args, **_kwargs: None)

    class Box(_Widget):
        def __init__(self, orientation=None, spacing=0):
            super().__init__()
            self.orientation = orientation
            self.spacing = spacing

        def append(self, child):
            self.children.append(child)

        def set_child(self, child):
            self.child = child

    class Grid(_Widget):
        def __init__(self, column_spacing=0, row_spacing=0):
            super().__init__()
            self.column_spacing = column_spacing
            self.row_spacing = row_spacing
            self.attachments = []

        def attach(self, child, column, row, width, height):
            self.attachments.append((child, column, row, width, height))
            self.children.append(child)

    class Frame(_Widget):
        def __init__(self, label=""):
            super().__init__()
            self.label = label
            self.child = None

        def set_child(self, child):
            self.child = child

    class Label(_Widget):
        def __init__(self, label=""):
            super().__init__()
            self.label = label

        def set_xalign(self, value):
            self.xalign = value

        def set_yalign(self, value):
            self.yalign = value

        def set_label(self, value):
            self.label = value

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

        def connect(self, signal, callback, *args):  # pragma: no cover - minimal signal support
            self._handlers.setdefault(signal, []).append((callback, args))

    class Entry(_Widget):
        def __init__(self):
            super().__init__()
            self.text = ""
            self.placeholder = ""

        def set_text(self, text):
            self.text = text

        def get_text(self):
            return self.text

        def set_placeholder_text(self, text):
            self.placeholder = text

    class TextBuffer:
        def __init__(self):
            self.text = ""

        def set_text(self, text):
            self.text = text

        def get_text(self, _start=None, _end=None, _include_hidden_chars=True):
            return self.text

        def get_start_iter(self):
            return 0

        def get_end_iter(self):
            return len(self.text)

    class TextView(_Widget):
        def __init__(self):
            super().__init__()
            self.buffer = TextBuffer()

        def get_buffer(self):
            return self.buffer

        def set_buffer(self, buffer):
            self.buffer = buffer

        def set_wrap_mode(self, mode):
            self.wrap_mode = mode

    class ScrolledWindow(_Widget):
        def __init__(self):
            super().__init__()
            self.child = None
            self.policy = (None, None)

        def set_policy(self, horizontal, vertical):
            self.policy = (horizontal, vertical)

        def set_child(self, child):
            self.child = child

    class Button(_Widget):
        def __init__(self, label=""):
            super().__init__()
            self.label = label
            self._handlers = {}

        def connect(self, signal, callback):
            self._handlers.setdefault(signal, []).append(callback)

    class Notebook(_Widget):
        def __init__(self):
            super().__init__()
            self.pages = []

        def append_page(self, child, tab_label):
            self.pages.append((child, tab_label))

        def append(self, child):
            self.children.append(child)

        def set_child(self, child):
            self.child = child

        def set_action_widget(self, child, pack_type):
            self.action_widget = (child, pack_type)

        def get_style_context(self):
            return types.SimpleNamespace(add_class=lambda *_args, **_kwargs: None)

    class MessageType:
        ERROR = "error"
        INFO = "info"
        WARNING = "warning"

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

    class PackType:
        START = "start"
        END = "end"

    class AccessibleRole:
        BUTTON = "button"

    class PolicyType:
        AUTOMATIC = "automatic"
        NEVER = "never"

    class WrapMode:
        WORD_CHAR = "word_char"
        WORD = "word"

    class CssProvider:
        def __init__(self):
            self.path = None

        def load_from_path(self, path):
            self.path = path

    class _StyleContext:
        @staticmethod
        def add_provider_for_display(display, provider, priority):
            return None

    gtk_module.CssProvider = CssProvider
    gtk_module.StyleContext = _StyleContext
    gtk_module.STYLE_PROVIDER_PRIORITY_APPLICATION = 600

    gtk_module.Window = Window
    gtk_module.Box = Box
    gtk_module.Grid = Grid
    gtk_module.Frame = Frame
    gtk_module.Label = Label
    gtk_module.ComboBoxText = ComboBoxText
    gtk_module.Adjustment = Adjustment
    gtk_module.SpinButton = SpinButton
    gtk_module.CheckButton = CheckButton
    gtk_module.Entry = Entry
    gtk_module.TextView = TextView
    gtk_module.TextBuffer = TextBuffer
    gtk_module.ScrolledWindow = ScrolledWindow
    gtk_module.Button = Button
    gtk_module.Notebook = Notebook
    gtk_module.PackType = PackType
    gtk_module.Widget = _Widget
    gtk_module.MessageDialog = MessageDialog
    gtk_module.MessageType = MessageType
    gtk_module.ButtonsType = ButtonsType
    gtk_module.Orientation = Orientation
    gtk_module.Align = Align
    gtk_module.AccessibleRole = AccessibleRole
    gtk_module.PolicyType = PolicyType
    gtk_module.WrapMode = WrapMode

    repository_module = types.ModuleType("gi.repository")
    repository_module.Gtk = gtk_module

    glib_module = types.ModuleType("GLib")

    def idle_add(func, *args, **kwargs):
        return func(*args, **kwargs)

    glib_module.idle_add = idle_add
    repository_module.GLib = glib_module

    gdk_module = types.ModuleType("Gdk")
    class _Display:
        @staticmethod
        def get_default():
            return types.SimpleNamespace()

    gdk_module.Display = _Display

    repository_module.Gdk = gdk_module

    sys.modules["gi"] = gi_stub
    sys.modules["gi"].repository = repository_module
    sys.modules["gi.repository"] = repository_module
    sys.modules["gi.repository.Gtk"] = gtk_module
    sys.modules["gi.repository.GLib"] = glib_module
    sys.modules["gi.repository.Gdk"] = gdk_module

from gi.repository import Gtk

if "anthropic" not in sys.modules:
    anthropic_module = types.ModuleType("anthropic")

    class _StubAsyncAnthropic:
        def __init__(self, *_args, **_kwargs):
            pass

    anthropic_module.AsyncAnthropic = _StubAsyncAnthropic
    anthropic_module.APIError = Exception
    anthropic_module.RateLimitError = Exception
    sys.modules["anthropic"] = anthropic_module

if "tenacity" not in sys.modules:
    tenacity_module = types.ModuleType("tenacity")

    def _identity_decorator(*_args, **_kwargs):
        def _wrap(func):
            return func

        if _args and callable(_args[0]) and len(_args) == 1 and not _kwargs:
            return _args[0]
        return _wrap

    tenacity_module.retry = _identity_decorator
    tenacity_module.stop_after_attempt = lambda *_args, **_kwargs: None
    tenacity_module.wait_exponential = lambda *_args, **_kwargs: None
    tenacity_module.retry_if_exception_type = lambda *_args, **_kwargs: None
    sys.modules["tenacity"] = tenacity_module

openai_stub = sys.modules.get("openai")
if openai_stub is None:
    openai_stub = types.ModuleType("openai")
    sys.modules["openai"] = openai_stub


class _ModelList:
    def list(self):
        return types.SimpleNamespace(data=[])


class _OpenAI:
    def __init__(self, *_args, **_kwargs):
        self.models = _ModelList()


class _AsyncOpenAI:
    class _ChatCompletions:
        async def create(self, *_args, **_kwargs):  # pragma: no cover - placeholder
            return types.SimpleNamespace(choices=[])

    def __init__(self, *_args, **_kwargs):
        self.chat = types.SimpleNamespace(completions=self._ChatCompletions())


openai_stub.OpenAI = getattr(openai_stub, "OpenAI", _OpenAI)
openai_stub.AsyncOpenAI = getattr(openai_stub, "AsyncOpenAI", _AsyncOpenAI)

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
    stub._GENERATOR_CACHE = WeakKeyDictionary()

    if module_name.endswith("OA_gen_response"):
        generator_attr = "OpenAIGenerator"
    elif module_name.endswith("Mistral_gen_response"):
        generator_attr = "MistralGenerator"
    elif module_name.endswith("GG_gen_response"):
        generator_attr = "GoogleGeminiGenerator"
    else:
        generator_attr = None

    class _StubGenerator:
        async def generate_response(self, *_args, **_kwargs):  # pragma: no cover - placeholder
            return return_value

        async def process_streaming_response(self, *_args, **_kwargs):  # pragma: no cover - placeholder
            return return_value

        async def process_response(self, response):  # pragma: no cover - placeholder
            if isinstance(response, str):
                return response
            collected = []
            async for chunk in response:
                collected.append(chunk)
            return "".join(collected)

    def _get_generator(config_manager=None):  # pragma: no cover - placeholder
        cache = stub._GENERATOR_CACHE
        generator_type = getattr(stub, generator_attr, None) if generator_attr else None
        if generator_type is None:
            generator_type = _StubGenerator

        def _create(config):
            if generator_type is _StubGenerator:
                return generator_type()
            try:
                return generator_type(config)
            except TypeError:  # pragma: no cover - fallback when signature differs
                return generator_type()

        if config_manager is None:
            return _create(config_manager)

        generator = cache.get(config_manager)
        if generator is None:
            generator = _create(config_manager)
            cache[config_manager] = generator
        return generator

    stub.get_generator = _get_generator

    if module_name.endswith("Anthropic_gen_response"):
        class _StubAnthropicGenerator:
            def __init__(self, *_args, **_kwargs):
                self.default_model = "claude-3-opus-20240229"

            async def generate_response(self, *args, **kwargs):  # pragma: no cover - placeholder
                return return_value

            async def process_streaming_response(self, response):  # pragma: no cover - placeholder
                chunks = []
                async for item in response:
                    chunks.append(item)
                return "".join(chunks)

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

            def set_stop_sequences(self, _value):
                return None

            def set_tool_choice(self, *_args, **_kwargs):
                return None

            def set_metadata(self, *_args, **_kwargs):
                return None

            def set_thinking(self, *_args, **_kwargs):
                return None

        stub.AnthropicGenerator = _StubAnthropicGenerator
        stub.setup_anthropic_generator = lambda _cfg=None: _StubAnthropicGenerator()

        class _StubAsyncAnthropic:
            def __init__(self, *_args, **_kwargs):
                pass

        stub.AsyncAnthropic = _StubAsyncAnthropic
        stub.APIError = RuntimeError
        stub.RateLimitError = RuntimeError

    sys.modules[module_name] = stub


_ensure_async_function_module("modules.Providers.OpenAI.OA_gen_response", "openai")
_ensure_async_function_module("modules.Providers.Mistral.Mistral_gen_response", "mistral")
_ensure_async_function_module("modules.Providers.Google.GG_gen_response", "google")

import pytest

import ATLAS.provider_manager as provider_manager_module
from ATLAS.provider_manager import ProviderManager
from GTKUI.Provider_manager.Settings.OA_settings import OpenAISettingsWindow
from GTKUI.Provider_manager.Settings.Anthropic_settings import AnthropicSettingsWindow
from GTKUI.Provider_manager.Settings import Anthropic_settings
from GTKUI.Provider_manager.Settings import Mistral_settings
from GTKUI.Provider_manager.Settings.Google_settings import GoogleSettingsWindow
from GTKUI.Provider_manager.Settings.Mistral_settings import MistralSettingsWindow


def reset_provider_manager_singleton():
    ProviderManager.reset_singleton()


class DummyConfig:
    def __init__(self, root_path):
        self._root_path = root_path
        self._hf_token = ""
        self._api_keys = {}
        self._default_model = "gpt-4o"
        self._openai_settings = {
            "model": "gpt-4o",
            "temperature": 0.0,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "max_tokens": 4000,
            "max_output_tokens": None,
            "stream": True,
            "function_calling": True,
            "parallel_tool_calls": True,
            "tool_choice": None,
            "base_url": None,
            "organization": None,
            "reasoning_effort": "medium",
            "json_mode": False,
            "json_schema": None,
            "audio_enabled": False,
            "audio_voice": "alloy",
            "audio_format": "wav",
        }
        self._anthropic_settings = {
            "model": "claude-3-opus-20240229",
            "stream": True,
            "function_calling": False,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": None,
            "max_output_tokens": None,
            "timeout": 60,
            "max_retries": 3,
            "retry_delay": 5,
            "stop_sequences": [],
            "tool_choice": "auto",
            "tool_choice_name": None,
            "metadata": {},
            "thinking": False,
            "thinking_budget": None,
        }
        self._mistral_settings = {
            "model": "mistral-large-latest",
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": None,
            "safe_prompt": False,
            "stream": True,
            "random_seed": None,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "tool_choice": "auto",
            "parallel_tool_calls": True,
            "json_mode": False,
            "json_schema": None,
            "base_url": None,
            "prompt_mode": None,
        }
        self._google_settings = {
            "model": "gemini-1.5-pro-latest",
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": None,
            "candidate_count": 1,
            "stop_sequences": [],
            "safety_settings": [],
            "response_mime_type": None,
            "system_instruction": None,
            "max_output_tokens": 32000,
            "stream": True,
            "function_calling": True,
            "function_call_mode": "auto",
            "allowed_function_names": [],
            "cached_allowed_function_names": [],
            "response_schema": {},
        }
        self._fallback_config: Dict[str, Any] = {}

    def get_default_provider(self):
        return "OpenAI"

    def get_default_model(self):
        return self._default_model

    def set_default_model(self, model):
        self._default_model = model

    def get_openai_llm_settings(self):
        return dict(self._openai_settings)

    def set_openai_llm_settings(
        self,
        *,
        model,
        temperature=None,
        top_p=None,
        frequency_penalty=None,
        presence_penalty=None,
        max_tokens=None,
        max_output_tokens=None,
        stream=None,
        function_calling=None,
        parallel_tool_calls=None,
        tool_choice=None,
        base_url=None,
        organization=None,
        reasoning_effort=None,
        json_mode=None,
        json_schema=None,
        audio_enabled=None,
        audio_voice=None,
        audio_format=None,
    ):
        if model:
            self._openai_settings["model"] = model
            self._default_model = model
        if temperature is not None:
            self._openai_settings["temperature"] = float(temperature)
        if top_p is not None:
            self._openai_settings["top_p"] = float(top_p)
        if frequency_penalty is not None:
            self._openai_settings["frequency_penalty"] = float(frequency_penalty)
        if presence_penalty is not None:
            self._openai_settings["presence_penalty"] = float(presence_penalty)
        if max_tokens is not None:
            self._openai_settings["max_tokens"] = int(max_tokens)
        if max_output_tokens is not None:
            self._openai_settings["max_output_tokens"] = int(max_output_tokens)
        if stream is not None:
            self._openai_settings["stream"] = bool(stream)
        if function_calling is not None:
            self._openai_settings["function_calling"] = bool(function_calling)
        if parallel_tool_calls is not None:
            self._openai_settings["parallel_tool_calls"] = bool(parallel_tool_calls)
        if tool_choice is not None:
            self._openai_settings["tool_choice"] = tool_choice
        self._openai_settings["base_url"] = base_url
        self._openai_settings["organization"] = organization
        if reasoning_effort is not None:
            self._openai_settings["reasoning_effort"] = reasoning_effort
        if json_mode is not None:
            self._openai_settings["json_mode"] = bool(json_mode)
        if json_schema is not None:
            self._openai_settings["json_schema"] = json_schema
        if audio_enabled is not None:
            self._openai_settings["audio_enabled"] = bool(audio_enabled)
        if audio_voice is not None:
            self._openai_settings["audio_voice"] = audio_voice
        if audio_format is not None:
            self._openai_settings["audio_format"] = audio_format
        return dict(self._openai_settings)

    def get_anthropic_settings(self):
        return dict(self._anthropic_settings)

    def set_anthropic_settings(
        self,
        *,
        model=None,
        stream=None,
        function_calling=None,
        temperature=None,
        top_p=None,
        top_k=None,
        max_output_tokens=None,
        timeout=None,
        max_retries=None,
        retry_delay=None,
        stop_sequences=None,
        tool_choice=None,
        tool_choice_name=None,
        metadata=None,
        thinking=None,
        thinking_budget=None,
    ):
        if model is not None:
            self._anthropic_settings["model"] = model
        if stream is not None:
            self._anthropic_settings["stream"] = bool(stream)
        if function_calling is not None:
            self._anthropic_settings["function_calling"] = bool(function_calling)
        if temperature is not None:
            self._anthropic_settings["temperature"] = float(temperature)
        if top_p is not None:
            self._anthropic_settings["top_p"] = float(top_p)
        if top_k is not None:
            self._anthropic_settings["top_k"] = (
                int(top_k) if top_k not in {"", None} else None
            )
        if max_output_tokens is not None:
            self._anthropic_settings["max_output_tokens"] = (
                int(max_output_tokens)
                if max_output_tokens is not None
                else None
            )
        if timeout is not None:
            self._anthropic_settings["timeout"] = int(timeout)
        if max_retries is not None:
            self._anthropic_settings["max_retries"] = int(max_retries)
        if retry_delay is not None:
            self._anthropic_settings["retry_delay"] = int(retry_delay)
        if stop_sequences is not None:
            if isinstance(stop_sequences, str):
                items = [part.strip() for part in stop_sequences.split(",") if part.strip()]
            else:
                items = [
                    str(item).strip()
                    for item in (stop_sequences or [])
                    if str(item).strip()
                ]
            self._anthropic_settings["stop_sequences"] = items
        if tool_choice is not None:
            self._anthropic_settings["tool_choice"] = str(tool_choice)
        if tool_choice_name is not None:
            self._anthropic_settings["tool_choice_name"] = (
                str(tool_choice_name).strip() if str(tool_choice_name).strip() else None
            )
        if metadata is not None:
            if isinstance(metadata, dict):
                self._anthropic_settings["metadata"] = dict(metadata)
            else:
                self._anthropic_settings["metadata"] = {}
        if thinking is not None:
            self._anthropic_settings["thinking"] = bool(thinking)
        if thinking_budget is not None:
            self._anthropic_settings["thinking_budget"] = (
                int(thinking_budget) if str(thinking_budget).strip() else None
            )
        return dict(self._anthropic_settings)

    def get_mistral_llm_settings(self):
        return dict(self._mistral_settings)

    def set_mistral_llm_settings(
        self,
        *,
        model,
        temperature=None,
        top_p=None,
        max_tokens=None,
        safe_prompt=None,
        random_seed=None,
        frequency_penalty=None,
        presence_penalty=None,
        tool_choice=None,
        parallel_tool_calls=None,
        stream=None,
        stop_sequences=None,
        json_mode=None,
        json_schema=None,
        max_retries=None,
        retry_min_seconds=None,
        retry_max_seconds=None,
        base_url=None,
        prompt_mode=None,
    ):
        if model:
            self._mistral_settings["model"] = model
            self._default_model = model
        if temperature is not None:
            self._mistral_settings["temperature"] = float(temperature)
        if top_p is not None:
            self._mistral_settings["top_p"] = float(top_p)
        if max_tokens in ("", None):
            self._mistral_settings["max_tokens"] = None
        elif max_tokens is not None:
            numeric = int(max_tokens)
            if numeric <= 0:
                self._mistral_settings["max_tokens"] = None
            else:
                self._mistral_settings["max_tokens"] = numeric
        if safe_prompt is not None:
            self._mistral_settings["safe_prompt"] = bool(safe_prompt)
        if stream is not None:
            self._mistral_settings["stream"] = bool(stream)
        if random_seed in ("", None):
            self._mistral_settings["random_seed"] = None
        elif random_seed is not None:
            self._mistral_settings["random_seed"] = int(random_seed)
        if frequency_penalty is not None:
            self._mistral_settings["frequency_penalty"] = float(frequency_penalty)
        if presence_penalty is not None:
            self._mistral_settings["presence_penalty"] = float(presence_penalty)
        if tool_choice is not None:
            self._mistral_settings["tool_choice"] = tool_choice
        if parallel_tool_calls is not None:
            self._mistral_settings["parallel_tool_calls"] = bool(parallel_tool_calls)
        if stop_sequences is not None:
            if isinstance(stop_sequences, str):
                items = [part.strip() for part in stop_sequences.split(",") if part.strip()]
            else:
                items = [
                    str(item).strip()
                    for item in (stop_sequences or [])
                    if str(item).strip()
                ]
            self._mistral_settings["stop_sequences"] = items
        if json_mode is not None:
            self._mistral_settings["json_mode"] = bool(json_mode)
        if json_schema is not None:
            if json_schema == "":
                self._mistral_settings["json_schema"] = None
            else:
                if isinstance(json_schema, str):
                    try:
                        self._mistral_settings["json_schema"] = json.loads(json_schema)
                    except json.JSONDecodeError:
                        self._mistral_settings["json_schema"] = json_schema
                else:
                    self._mistral_settings["json_schema"] = json_schema
        if max_retries is not None:
            self._mistral_settings["max_retries"] = int(max_retries)
        if retry_min_seconds is not None:
            self._mistral_settings["retry_min_seconds"] = int(retry_min_seconds)
        if retry_max_seconds is not None:
            self._mistral_settings["retry_max_seconds"] = int(retry_max_seconds)
        if base_url is not None:
            cleaned = str(base_url).strip()
            self._mistral_settings["base_url"] = cleaned or None
        if prompt_mode is None:
            self._mistral_settings["prompt_mode"] = None
        elif isinstance(prompt_mode, str):
            cleaned_mode = prompt_mode.strip()
            self._mistral_settings["prompt_mode"] = cleaned_mode or None
        else:
            self._mistral_settings["prompt_mode"] = prompt_mode
        return dict(self._mistral_settings)

    def get_google_llm_settings(self):
        return dict(self._google_settings)

    def set_google_llm_settings(
        self,
        *,
        model,
        temperature=None,
        top_p=None,
        top_k=None,
        candidate_count=None,
        stop_sequences=None,
        safety_settings=None,
        response_mime_type=None,
        system_instruction=None,
        max_output_tokens=None,
        stream=None,
        function_calling=None,
        function_call_mode=None,
        allowed_function_names=None,
        response_schema=None,
        cached_allowed_function_names=None,
        seed=None,
        response_logprobs=None,
    ):
        if model:
            self._google_settings["model"] = model
        if temperature is not None:
            self._google_settings["temperature"] = float(temperature)
        if top_p is not None:
            self._google_settings["top_p"] = float(top_p)
        if top_k in ("", None):
            self._google_settings["top_k"] = None
        elif top_k is not None:
            self._google_settings["top_k"] = int(top_k)
        if candidate_count is not None:
            self._google_settings["candidate_count"] = int(candidate_count)
        if stop_sequences is not None:
            if isinstance(stop_sequences, str):
                items = [part.strip() for part in stop_sequences.split(",") if part.strip()]
            else:
                items = [
                    str(item).strip()
                    for item in (stop_sequences or [])
                    if str(item).strip()
                ]
            self._google_settings["stop_sequences"] = items
        if safety_settings is not None:
            self._google_settings["safety_settings"] = list(safety_settings)
        if response_mime_type is not None:
            self._google_settings["response_mime_type"] = response_mime_type
        if system_instruction is not None:
            self._google_settings["system_instruction"] = system_instruction
        if max_output_tokens in ("", None):
            if max_output_tokens == "":
                self._google_settings["max_output_tokens"] = None
        elif max_output_tokens is not None:
            self._google_settings["max_output_tokens"] = int(max_output_tokens)
        if seed in ("", None):
            self._google_settings["seed"] = None
        elif seed is not None:
            self._google_settings["seed"] = int(seed)
        if stream is not None:
            self._google_settings["stream"] = bool(stream)
        if function_calling is not None:
            self._google_settings["function_calling"] = bool(function_calling)
        if function_call_mode is not None:
            self._google_settings["function_call_mode"] = str(function_call_mode)
        if allowed_function_names is not None:
            if isinstance(allowed_function_names, str):
                names = [
                    part.strip()
                    for part in allowed_function_names.split(",")
                    if part.strip()
                ]
            else:
                names = [
                    str(item).strip()
                    for item in (allowed_function_names or [])
                    if str(item).strip()
                ]
            self._google_settings["allowed_function_names"] = names
            if cached_allowed_function_names is None:
                cached_allowed_function_names = names
        if cached_allowed_function_names is not None:
            if isinstance(cached_allowed_function_names, str):
                cached_names = [
                    part.strip()
                    for part in cached_allowed_function_names.split(",")
                    if part.strip()
                ]
            else:
                cached_names = [
                    str(item).strip()
                    for item in (cached_allowed_function_names or [])
                    if str(item).strip()
                ]
            self._google_settings["cached_allowed_function_names"] = cached_names
        if response_schema is not None:
            if response_schema in ("", {}, None):
                self._google_settings["response_schema"] = {}
            elif isinstance(response_schema, str):
                text = response_schema.strip()
                if not text:
                    self._google_settings["response_schema"] = {}
                else:
                    try:
                        parsed = json.loads(text)
                    except json.JSONDecodeError as exc:
                        raise ValueError("Response schema must be valid JSON.") from exc
                    if not isinstance(parsed, dict):
                        raise ValueError("Response schema must be a JSON object.")
                    self._google_settings["response_schema"] = parsed
            elif isinstance(response_schema, dict):
                self._google_settings["response_schema"] = dict(response_schema)
            else:
                raise ValueError(
                    "Response schema must be provided as a mapping or JSON string."
                )
        if response_logprobs is not None:
            self._google_settings["response_logprobs"] = bool(response_logprobs)

        return dict(self._google_settings)

    def get_app_root(self):
        return self._root_path

    def get_model_cache_dir(self):
        return self._root_path

    def _get_provider_defaults(self, provider: str) -> Dict[str, Any]:
        mapping = {
            "OpenAI": self._openai_settings,
            "Mistral": self._mistral_settings,
            "Google": self._google_settings,
            "Anthropic": self._anthropic_settings,
        }
        defaults = mapping.get(provider, {})
        return dict(defaults)

    def get_llm_config(self, *_args, **_kwargs):
        return {
            "provider": "OpenAI",
            "model": "gpt-4o",
            "max_tokens": 4000,
            "temperature": 0.0,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "stream": True,
        }

    def set_llm_fallback_config(self, **kwargs):
        self._fallback_config.update(kwargs)
        return dict(self._fallback_config)

    def get_llm_fallback_config(self):
        provider = self._fallback_config.get("provider") or self.get_default_provider()
        merged = self._get_provider_defaults(provider)
        merged.update(self._fallback_config)
        merged["provider"] = provider
        if not merged.get("model"):
            merged["model"] = self._get_provider_defaults(provider).get("model")
        return merged

    def get_config(self, *_args, **_kwargs):
        return None

    def get_huggingface_api_key(self):
        return self._hf_token

    def get_openai_api_key(self):
        return self._api_keys.get("OpenAI", "")

    def get_mistral_api_key(self):
        return self._api_keys.get("Mistral", "")

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
        def build_payload(secret: Optional[str]):
            token = secret or ""
            available = bool(token)
            length = len(token) if available else 0
            hint = "\u2022" * min(length, 8) if available else ""
            return {"available": available, "length": length, "hint": hint}

        return {
            "OpenAI": build_payload(self._api_keys.get("OpenAI")),
            "Mistral": build_payload(self._api_keys.get("Mistral")),
            "Google": build_payload(self._api_keys.get("Google")),
            "HuggingFace": build_payload(self._hf_token),
            "Anthropic": build_payload(self._api_keys.get("Anthropic")),
            "Grok": build_payload(self._api_keys.get("Grok")),
            "ElevenLabs": build_payload(self._api_keys.get("ElevenLabs")),
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
    reset_provider_manager_singleton()

    def fake_load_models(self):
        self.models = {
            "OpenAI": [
                "gpt-4o",
                "gpt-4o-mini",
                "gpt-4o-mini-tts",
                "gpt-4o-transcribe",
                "gpt-4o-mini-transcribe",
                "gpt-4.1",
                "gpt-4.1-mini",
                "o1",
                "o1-mini",
            ],
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

    reset_provider_manager_singleton()


def test_openai_default_model_uses_cached_list(provider_manager):
    default = provider_manager.get_default_model_for_provider("OpenAI")

    assert default == "gpt-4o"
    assert "gpt-4.1" in provider_manager.model_manager.models["OpenAI"]
    assert "o1" in provider_manager.model_manager.models["OpenAI"]


def test_get_default_model_uses_provider_settings_when_cache_empty(provider_manager):
    provider_manager.model_manager.models["Mistral"] = []

    expected = provider_manager.config_manager.get_mistral_llm_settings()["model"]
    default = provider_manager.get_default_model_for_provider("Mistral")

    assert default == expected


def test_get_default_model_uses_global_default_when_no_provider_settings(provider_manager):
    provider_manager.model_manager.models["Grok"] = []
    provider_manager.config_manager.set_default_model("fallback-default-model")

    default = provider_manager.get_default_model_for_provider("Grok")

    assert default == "fallback-default-model"


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
        assert generator.loaded_models == []
        assert provider_manager.current_model is None
        assert provider_manager.is_model_loaded("HuggingFace") is False
        assert provider_manager.get_pending_model_for_provider("HuggingFace") == "alpha"

        load_result = await provider_manager.load_hf_model("beta", force_download=True)
        assert load_result["success"] is True
        assert ("beta", True) in generator.loaded_models
        assert provider_manager.is_model_loaded("HuggingFace") is True
        assert provider_manager.get_pending_model_for_provider("HuggingFace") is None
        assert provider_manager.current_model == "beta"

        unload_result = await provider_manager.unload_hf_model()
        assert unload_result["success"] is True
        assert generator.unload_calls == 1
        assert provider_manager.current_model is None
        assert provider_manager.is_model_loaded("HuggingFace") is False

        remove_result = await provider_manager.remove_hf_model("beta")
        assert remove_result["success"] is True

    asyncio.run(exercise())

    refreshed = provider_manager.list_hf_models()
    assert refreshed["success"] is True
    assert refreshed["data"] == ["alpha"]


def test_switch_llm_provider_releases_generators(tmp_path, monkeypatch):
    reset_provider_manager_singleton()

    class CleanupConfig(DummyConfig):
        def __init__(self, root_path):
            super().__init__(root_path)
            self._api_keys.update(
                {
                    "OpenAI": "openai-key",
                    "Mistral": "mistral-key",
                    "Anthropic": "anthropic-key",
                    "Grok": "grok-key",
                    "HuggingFace": "hf-key",
                }
            )
            self._hf_token = "hf-token"

        def get_grok_api_key(self):
            return self._api_keys["Grok"]

    class TrackingGenerator:
        def __init__(self, name: str, *, default_model: Optional[str] = None):
            self.name = name
            self.default_model = default_model
            self.close_calls = 0
            self.aclose_calls = 0

        async def close(self):
            self.close_calls += 1

        async def aclose(self):
            self.aclose_calls += 1

        async def generate_response(self, *_args, **_kwargs):
            return f"{self.name}-response"

        async def process_streaming_response(self, _response):
            return ""

    created_generators: Dict[str, List[Any]] = {}

    def register_created(name: str, generator: Any) -> Any:
        created_generators.setdefault(name, []).append(generator)
        return generator

    def make_factory(name: str, *, default_model: Optional[str] = None):
        def factory(*_args, **_kwargs):
            return register_created(name, TrackingGenerator(name, default_model=default_model))

        return factory

    class TrackingAnthropicGenerator(TrackingGenerator):
        def __init__(self, _config_manager):
            super().__init__("Anthropic", default_model="anthropic-default")
            register_created("Anthropic", self)

    class TrackingGrokGenerator:
        def __init__(self, _config_manager):
            self.unload_calls = 0
            register_created("Grok", self)

        async def unload_model(self):
            self.unload_calls += 1

        async def generate_response(self, *_args, **_kwargs):  # pragma: no cover - simple stub
            return "grok-response"

        async def process_streaming_response(self, _response):  # pragma: no cover - simple stub
            return ""

    def fake_load_models(self):
        self.models = {
            "OpenAI": ["openai-default"],
            "HuggingFace": ["hf-default"],
            "Mistral": ["mistral-default"],
            "Google": ["google-default"],
            "Anthropic": ["anthropic-default"],
            "Grok": ["grok-default"],
        }
        self.current_model = None
        self.current_provider = None

    async def fake_list_openai_models(self, **_kwargs):
        return {"models": []}

    monkeypatch.setattr(provider_manager_module.ModelManager, "load_models", fake_load_models, raising=False)
    monkeypatch.setattr(provider_manager_module.ProviderManager, "list_openai_models", fake_list_openai_models, raising=False)
    monkeypatch.setattr(provider_manager_module, "HuggingFaceGenerator", FakeHFGenerator)
    monkeypatch.setattr(provider_manager_module, "get_openai_generator", make_factory("OpenAI"), raising=False)
    monkeypatch.setattr(provider_manager_module, "get_mistral_generator", make_factory("Mistral"), raising=False)
    monkeypatch.setattr(provider_manager_module, "get_google_generator", make_factory("Google"), raising=False)
    monkeypatch.setattr(provider_manager_module, "AnthropicGenerator", TrackingAnthropicGenerator, raising=False)
    monkeypatch.setattr(provider_manager_module, "GrokGenerator", TrackingGrokGenerator, raising=False)

    config = CleanupConfig(tmp_path.as_posix())

    async def exercise():
        manager = await ProviderManager.create(config)

        openai_gen = created_generators["OpenAI"][0]
        assert openai_gen.close_calls == 0
        assert manager._openai_generator is openai_gen

        await manager.switch_llm_provider("Grok")
        assert openai_gen.close_calls == 1
        assert manager._openai_generator is None

        grok_gen = created_generators["Grok"][0]
        await manager.switch_llm_provider("HuggingFace")
        assert grok_gen.unload_calls == 1
        hf_generator = manager.huggingface_generator
        assert isinstance(hf_generator, FakeHFGenerator)

        await manager.switch_llm_provider("Mistral")
        assert hf_generator.unload_calls == 1
        assert manager.huggingface_generator is None
        mistral_gen = created_generators["Mistral"][0]
        assert manager._mistral_generator is mistral_gen

        await manager.switch_llm_provider("Anthropic")
        assert mistral_gen.close_calls == 1
        assert manager._mistral_generator is None
        anthropic_gen = created_generators["Anthropic"][0]

        await manager.switch_llm_provider("OpenAI")
        assert anthropic_gen.close_calls == 1
        assert manager.anthropic_generator is None
        assert len(created_generators["OpenAI"]) == 2
        new_openai_gen = created_generators["OpenAI"][1]
        assert manager._openai_generator is new_openai_gen
        assert new_openai_gen.close_calls == 0
        assert manager.grok_generator is None
        assert manager.huggingface_generator is None
        assert manager._provider_model_ready.get("HuggingFace") is False

    try:
        asyncio.run(exercise())
    finally:
        reset_provider_manager_singleton()


def test_huggingface_default_provider_startup_defers_loading(tmp_path, monkeypatch):
    reset_provider_manager_singleton()

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

    class HFDefaultConfig(DummyConfig):
        def __init__(self, root_path):
            super().__init__(root_path)
            self._hf_token = "token-123"
            self._api_keys["HuggingFace"] = self._hf_token

        def get_default_provider(self):
            return "HuggingFace"

    config = HFDefaultConfig(tmp_path.as_posix())

    async def exercise():
        manager = await ProviderManager.create(config)
        generator = manager.huggingface_generator
        before = {
            "loaded_models": list(generator.loaded_models),
            "current_model": manager.current_model,
            "is_loaded": manager.is_model_loaded("HuggingFace"),
            "pending": manager.get_pending_model_for_provider("HuggingFace"),
        }
        load_result = await manager.load_hf_model("alpha")
        after = {
            "loaded_models": list(generator.loaded_models),
            "current_model": manager.current_model,
            "is_loaded": manager.is_model_loaded("HuggingFace"),
            "pending": manager.get_pending_model_for_provider("HuggingFace"),
        }
        return manager, generator, before, after, load_result

    manager, generator, before, after, load_result = asyncio.run(exercise())

    try:
        assert isinstance(generator, FakeHFGenerator)
        assert before["loaded_models"] == []
        assert before["current_model"] is None
        assert before["is_loaded"] is False
        assert before["pending"] == "alpha"

        assert load_result["success"] is True
        assert after["loaded_models"][0] == ("alpha", False)
        assert after["current_model"] == "alpha"
        assert after["is_loaded"] is True
        assert after["pending"] is None
    finally:
        reset_provider_manager_singleton()


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


def test_list_openai_models_requires_api_key(provider_manager):
    result = asyncio.run(provider_manager.list_openai_models())

    assert result["models"] == []
    assert "API key" in (result["error"] or "")


def test_list_openai_models_fetches_and_prioritizes(provider_manager, monkeypatch):
    provider_manager.config_manager.update_api_key("OpenAI", "sk-test")

    payload = {
        "data": [
            {"id": "gpt-4o"},
            {"id": "chat-awesome"},
            {"id": "text-embedding-3-small"},
        ]
    }

    async def fake_to_thread(callable_, *args, **kwargs):  # pragma: no cover - test helper
        return payload

    monkeypatch.setattr(provider_manager_module.asyncio, "to_thread", fake_to_thread)

    result = asyncio.run(provider_manager.list_openai_models())

    assert result["error"] is None
    assert result["models"] == [
        "chat-awesome",
        "gpt-4o",
        "text-embedding-3-small",
    ]
    assert result["base_url"].endswith("/v1")
    cached = provider_manager.model_manager.models["OpenAI"]
    assert cached[0] == "gpt-4o"
    assert cached[1] == "chat-awesome"
    assert "text-embedding-3-small" in cached


def test_list_openai_models_handles_network_error(provider_manager, monkeypatch):
    provider_manager.config_manager.update_api_key("OpenAI", "sk-test")

    async def fake_to_thread(callable_, *args, **kwargs):  # pragma: no cover - test helper
        raise URLError("network down")

    monkeypatch.setattr(provider_manager_module.asyncio, "to_thread", fake_to_thread)

    result = asyncio.run(
        provider_manager.list_openai_models(
            base_url="https://alt.example/v1", organization="org-1234"
        )
    )

    assert result["models"] == []
    assert "network down" in (result["error"] or "")
    assert result["base_url"] == "https://alt.example/v1"
    assert result["organization"] == "org-1234"


def test_fetch_mistral_models_requires_api_key(provider_manager, monkeypatch):
    class _NoopClient:
        def __init__(self, *_args, **_kwargs):
            self.models = types.SimpleNamespace(list=lambda: [])

    monkeypatch.setattr(provider_manager_module, "Mistral", _NoopClient, raising=False)

    result = asyncio.run(provider_manager.fetch_mistral_models())

    assert result["success"] is False
    assert "API key" in (result.get("error") or "")


def test_fetch_mistral_models_updates_cache_and_file(provider_manager, monkeypatch, tmp_path):
    provider_manager.config_manager.update_api_key("Mistral", "mst-test")
    provider_manager.config_manager.set_mistral_llm_settings(
        model="mistral-large-latest",
        base_url="https://api.alt-mistral/v1",
    )

    models = [
        types.SimpleNamespace(id="mistral-large-latest"),
        {"id": "open-mixtral-8x7b", "name": "ignored"},
        types.SimpleNamespace(slug="mistral-small-latest"),
    ]

    class _StubModelPage:
        def __init__(self, data):
            self.data = data

    class _StubModelClient:
        def __init__(self, *args, **kwargs):
            captured_kwargs["value"] = dict(kwargs)
            self.models = types.SimpleNamespace(list=lambda: _StubModelPage(models))

        def close(self):
            self.closed = True

    async def fake_to_thread(callable_, *args, **kwargs):  # pragma: no cover - test helper
        return callable_()

    captured_kwargs: Dict[str, Any] = {}
    monkeypatch.setattr(provider_manager_module, "Mistral", _StubModelClient)
    monkeypatch.setattr(provider_manager_module.asyncio, "to_thread", fake_to_thread)

    result = asyncio.run(provider_manager.fetch_mistral_models())

    assert result["success"] is True
    data = result.get("data", {})
    discovered = data.get("models")
    assert discovered[:2] == ["mistral-large-latest", "open-mixtral-8x7b"]
    assert "mistral-small-latest" in discovered

    cached = provider_manager.model_manager.models.get("Mistral", [])
    assert cached[0] == "mistral-large-latest"
    assert "open-mixtral-8x7b" in cached

    forwarded = captured_kwargs.get("value", {})
    assert forwarded.get("server_url") == "https://api.alt-mistral/v1"

    persisted = data.get("persisted_to")
    assert persisted
    path = Path(persisted)
    assert path.exists()
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["models"][0] == "mistral-large-latest"
    assert data.get("base_url") == "https://api.alt-mistral/v1"


def test_fetch_mistral_models_handles_exception(provider_manager, monkeypatch):
    provider_manager.config_manager.update_api_key("Mistral", "mst-test")

    class _FailingClient:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("boom")

    monkeypatch.setattr(provider_manager_module, "Mistral", _FailingClient)

    result = asyncio.run(provider_manager.fetch_mistral_models())

    assert result["success"] is False
    assert "boom" in (result.get("error") or "")


def test_fetch_mistral_models_accepts_override(provider_manager, monkeypatch):
    provider_manager.config_manager.update_api_key("Mistral", "mst-test")
    provider_manager.config_manager.set_mistral_llm_settings(
        model="mistral-large-latest",
        base_url="https://default.mistral/v1",
    )

    captured: Dict[str, Any] = {}

    class _StubClient:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)
            self.models = types.SimpleNamespace(list=lambda: [])

        def close(self):
            return None

    async def fake_to_thread(callable_, *args, **kwargs):
        return callable_()

    monkeypatch.setattr(provider_manager_module, "Mistral", _StubClient)
    monkeypatch.setattr(provider_manager_module.asyncio, "to_thread", fake_to_thread)

    result = asyncio.run(
        provider_manager.fetch_mistral_models(base_url="https://override.mistral/v3")
    )

    assert captured.get("server_url") == "https://override.mistral/v3"
    data = result.get("data", {})
    assert data.get("base_url") == "https://override.mistral/v3"


def test_provider_manager_primes_openai_models_on_startup(tmp_path, monkeypatch):
    reset_provider_manager_singleton()

    def fake_load_models(self):
        self.models = {
            "OpenAI": ["gpt-4o"],
            "HuggingFace": [],
            "Mistral": ["mistral-large-latest"],
            "Google": ["gemini-1.5-pro-latest"],
            "Anthropic": ["claude-3-opus-20240229"],
            "Grok": [],
        }
        self.current_model = None
        self.current_provider = None

    monkeypatch.setattr(
        provider_manager_module.ModelManager, "load_models", fake_load_models, raising=False
    )

    payload = {"data": [{"id": "gpt-4.1"}, {"id": "o1"}, {"id": "gpt-4o"}]}

    async def fake_to_thread(callable_, *args, **kwargs):  # pragma: no cover - helper
        return payload

    monkeypatch.setattr(provider_manager_module.asyncio, "to_thread", fake_to_thread)

    config = DummyConfig(tmp_path.as_posix())
    config.update_api_key("OpenAI", "sk-start")

    manager = asyncio.run(ProviderManager.create(config))
    try:
        cached = manager.model_manager.models["OpenAI"]
        assert cached[0] == "gpt-4o"
        assert "gpt-4.1" in cached
        assert "o1" in cached
    finally:
        reset_provider_manager_singleton()


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


def test_google_settings_round_trips_custom_fields(tmp_path):
    class StubAtlas:
        def __init__(self):
            self.saved_payload = None
            self.refresh_calls = 0
            self.last_provider = None

        def get_models_for_provider(self, provider_name):
            assert provider_name == "Google"
            return ["gemini-1.5-pro-latest"]

        def get_google_llm_settings(self):
            return {
                "model": "gemini-1.5-pro-latest",
                "response_mime_type": "text/plain",
                "system_instruction": "Keep it short.",
                "stream": False,
                "function_calling": False,
                "response_schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                },
            }

        def get_provider_api_key_status(self, provider_name):
            return {"has_key": False}

        def set_google_llm_settings(self, **payload):
            self.saved_payload = payload
            return {"success": True, "message": "saved"}

        def run_in_background(self, func, on_error=None, thread_name=None):
            try:
                func()
            except Exception as exc:  # pragma: no cover - defensive
                if on_error is not None:
                    on_error(exc)

        def refresh_current_provider(self, provider_name):
            self.refresh_calls += 1
            self.last_provider = provider_name

    atlas = StubAtlas()
    config = DummyConfig(tmp_path.as_posix())

    window = GoogleSettingsWindow(atlas, config, None)

    assert isinstance(window.child, Gtk.Notebook)
    assert getattr(window.child, "action_widget", None) is not None

    assert window.response_mime_entry.get_text() == "text/plain"
    buffer = window.system_instruction_view.get_buffer()
    assert buffer.get_text(None, None, True) == "Keep it short."
    assert window.stream_toggle.get_active() is False
    assert window.function_call_toggle.get_active() is False
    schema_buffer = window.response_schema_view.get_buffer()
    assert json.loads(schema_buffer.get_text(None, None, True)) == {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
    }

    window.response_mime_entry.set_text("  application/json  ")
    buffer.set_text("  Follow policy.  ")
    window.stream_toggle.set_active(True)
    window.function_call_toggle.set_active(True)
    window.function_call_mode_combo.set_active(
        window._function_call_mode_index_map["require"]
    )
    window.allowed_functions_entry.set_text(" persona_tool , helper ")
    schema_buffer.set_text(
        "{\n  \"type\": \"object\",\n  \"properties\": {\n    \"value\": {\n      \"type\": \"integer\"\n    }\n  }\n}"
    )

    window.on_save_clicked()

    assert atlas.saved_payload is not None
    assert atlas.saved_payload["response_mime_type"] == "application/json"
    assert atlas.saved_payload["system_instruction"] == "Follow policy."
    assert atlas.saved_payload["stream"] is True
    assert atlas.saved_payload["function_calling"] is True
    assert atlas.saved_payload["function_call_mode"] == "require"
    assert atlas.saved_payload["allowed_function_names"] == [
        "persona_tool",
        "helper",
    ]
    assert atlas.saved_payload["cached_allowed_function_names"] == [
        "persona_tool",
        "helper",
    ]
    assert atlas.saved_payload["response_schema"] == {
        "type": "object",
        "properties": {"value": {"type": "integer"}},
    }
    assert atlas.last_provider == "Google"
    assert atlas.refresh_calls == 1


def test_google_settings_restores_allowlist_cache_when_reenabled(tmp_path):
    class StubAtlas:
        def __init__(self):
            self.saved_payloads = []
            self.refresh_calls = 0
            self.last_provider = None
            self._settings = {
                "model": "gemini-1.5-pro-latest",
                "function_calling": True,
                "function_call_mode": "auto",
                "allowed_function_names": ["alpha_tool", "beta_tool"],
                "cached_allowed_function_names": ["alpha_tool", "beta_tool"],
            }

        def get_models_for_provider(self, provider_name):
            assert provider_name == "Google"
            return ["gemini-1.5-pro-latest"]

        def get_google_llm_settings(self):
            return dict(self._settings)

        def get_provider_api_key_status(self, provider_name):
            return {"has_key": False}

        def set_google_llm_settings(self, **payload):
            self.saved_payloads.append(payload)
            for key in (
                "function_calling",
                "function_call_mode",
                "allowed_function_names",
                "cached_allowed_function_names",
            ):
                if key in payload:
                    self._settings[key] = payload[key]
            return {"success": True, "message": "saved"}

        def run_in_background(self, func, on_error=None, thread_name=None):
            try:
                func()
            except Exception as exc:  # pragma: no cover - defensive
                if on_error is not None:
                    on_error(exc)

        def refresh_current_provider(self, provider_name):
            self.refresh_calls += 1
            self.last_provider = provider_name

    atlas = StubAtlas()
    config = DummyConfig(tmp_path.as_posix())

    window = GoogleSettingsWindow(atlas, config, None)
    window.allowed_functions_entry.set_text("alpha_tool, beta_tool")
    window.function_call_toggle.set_active(False)
    window._on_function_call_toggled(window.function_call_toggle)
    window.on_save_clicked()

    assert atlas.saved_payloads[0]["allowed_function_names"] == []
    assert atlas.saved_payloads[0]["cached_allowed_function_names"] == [
        "alpha_tool",
        "beta_tool",
    ]

    window = GoogleSettingsWindow(atlas, config, None)
    assert window.function_call_toggle.get_active() is False
    window.allowed_functions_entry.set_text("")
    window.function_call_toggle.set_active(True)
    window._on_function_call_toggled(window.function_call_toggle)
    assert window.allowed_functions_entry.get_text() == "alpha_tool, beta_tool"
    window.on_save_clicked()

    assert atlas.saved_payloads[1]["allowed_function_names"] == [
        "alpha_tool",
        "beta_tool",
    ]
    assert atlas.saved_payloads[1]["cached_allowed_function_names"] == [
        "alpha_tool",
        "beta_tool",
    ]


def test_google_settings_supports_extended_safety_categories(tmp_path):
    class StubAtlas:
        def __init__(self):
            self.saved_payload = None
            self.refresh_calls = 0
            self.last_provider = None

        def get_models_for_provider(self, provider_name):
            assert provider_name == "Google"
            return ["gemini-1.5-pro-latest"]

        def get_google_llm_settings(self):
            return {
                "model": "gemini-1.5-pro-latest",
                "safety_settings": [
                    {"category": "HARM_CATEGORY_DEROGATORY", "threshold": "BLOCK_ONLY_HIGH"},
                    {"category": "HARM_CATEGORY_HARASSMENT_ABUSE", "threshold": "BLOCK_LOW_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_SELF_HARM", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                ],
            }

        def get_provider_api_key_status(self, provider_name):
            return {"has_key": False}

        def set_google_llm_settings(self, **payload):
            self.saved_payload = payload
            return {"success": True, "message": "saved"}

        def run_in_background(self, func, on_error=None, thread_name=None):
            try:
                func()
            except Exception as exc:  # pragma: no cover - defensive
                if on_error is not None:
                    on_error(exc)

        def refresh_current_provider(self, provider_name):
            self.refresh_calls += 1
            self.last_provider = provider_name

    atlas = StubAtlas()
    config = DummyConfig(tmp_path.as_posix())

    window = GoogleSettingsWindow(atlas, config, None)

    controls = window._safety_controls
    assert "HARM_CATEGORY_DEROGATORY" in controls
    assert "HARM_CATEGORY_CIVIC_INTEGRITY" in controls
    assert "HARM_CATEGORY_DANGEROUS" in controls

    der_toggle, der_combo = controls["HARM_CATEGORY_DEROGATORY"]
    harassment_toggle, harassment_combo = controls["HARM_CATEGORY_HARASSMENT"]
    dangerous_toggle, dangerous_combo = controls["HARM_CATEGORY_DANGEROUS"]

    assert der_toggle.get_active() is True
    assert (
        window._SAFETY_THRESHOLDS[window._get_combo_active_index(der_combo)][1]
        == "BLOCK_ONLY_HIGH"
    )
    assert harassment_toggle.get_active() is True
    assert (
        window._SAFETY_THRESHOLDS[window._get_combo_active_index(harassment_combo)][1]
        == "BLOCK_LOW_AND_ABOVE"
    )
    assert dangerous_toggle.get_active() is True
    assert (
        window._SAFETY_THRESHOLDS[window._get_combo_active_index(dangerous_combo)][1]
        == "BLOCK_MEDIUM_AND_ABOVE"
    )

    der_combo.set_active(0)
    harassment_combo.set_active(2)
    dangerous_combo.set_active(3)

    civic_toggle, civic_combo = controls["HARM_CATEGORY_CIVIC_INTEGRITY"]
    civic_toggle.set_active(True)
    civic_combo.set_active(1)

    captured = {}

    def fake_show_message(title, message, message_type):
        captured.update({"title": title, "message": message, "type": message_type})

    window._show_message = fake_show_message  # type: ignore[method-assign]

    window.on_save_clicked()

    assert atlas.saved_payload is not None
    assert atlas.saved_payload["safety_settings"] == [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_CIVIC_INTEGRITY", "threshold": "BLOCK_ONLY_HIGH"},
        {"category": "HARM_CATEGORY_DEROGATORY", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS", "threshold": "BLOCK_LOW_AND_ABOVE"},
    ]
    assert captured.get("title") == "Success"
    assert atlas.last_provider == "Google"
    assert atlas.refresh_calls == 1

def test_google_settings_rejects_invalid_schema(tmp_path):
    class StubAtlas:
        def __init__(self):
            self.saved_payload = None

        def get_models_for_provider(self, _provider_name):
            return ["gemini-1.5-pro-latest"]

        def get_google_llm_settings(self):
            return {"model": "gemini-1.5-pro-latest"}

        def get_provider_api_key_status(self, _provider_name):
            return {"has_key": False}

        def set_google_llm_settings(self, **payload):
            self.saved_payload = payload
            return {"success": True, "message": "saved"}

    atlas = StubAtlas()
    config = DummyConfig(tmp_path.as_posix())

    window = GoogleSettingsWindow(atlas, config, None)
    captured = {}

    def fake_show_message(title, message, message_type):
        captured.update({"title": title, "message": message, "type": message_type})

    window._show_message = fake_show_message  # type: ignore[method-assign]
    schema_buffer = window.response_schema_view.get_buffer()
    schema_buffer.set_text("not json")

    window.on_save_clicked()

    assert atlas.saved_payload is None
    assert "Response schema must be valid JSON" in captured.get("message", "")


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
    assert "sk-test" not in json.dumps(metadata)


def test_set_openai_llm_settings_updates_provider_state(provider_manager):
    result = provider_manager.set_openai_llm_settings(
        model="gpt-4o-mini",
        temperature=0.6,
        top_p=0.9,
        frequency_penalty=0.1,
        presence_penalty=-0.2,
        max_tokens=1024,
        max_output_tokens=256,
        stream=False,
        function_calling=False,
        parallel_tool_calls=False,
        tool_choice="required",
        base_url="https://example/v1",
        organization="org-99",
        reasoning_effort="low",
        json_mode=True,
        audio_enabled=True,
        audio_voice="verse",
        audio_format="wav",
    )

    assert result["success"] is True
    settings = provider_manager.get_openai_llm_settings()
    assert settings["model"] == "gpt-4o-mini"
    assert math.isclose(settings["top_p"], 0.9)
    assert math.isclose(settings["frequency_penalty"], 0.1)
    assert math.isclose(settings["presence_penalty"], -0.2)
    assert settings["stream"] is False
    assert settings["function_calling"] is False
    assert settings["parallel_tool_calls"] is False
    assert settings["max_output_tokens"] == 256
    assert settings["reasoning_effort"] == "low"
    assert settings["json_mode"] is True
    assert settings["json_schema"] is None
    assert settings["tool_choice"] == "required"
    assert settings["audio_enabled"] is True
    assert settings["audio_voice"] == "verse"
    assert settings["audio_format"] == "wav"
    assert provider_manager.model_manager.models["OpenAI"][0] == "gpt-4o-mini"
    assert provider_manager.current_model == "gpt-4o-mini"


def test_set_google_llm_settings_updates_provider_state(provider_manager):
    provider_manager.model_manager.models["Google"] = ["gemini-1.5-pro-latest", "legacy"]
    provider_manager.current_llm_provider = "Google"

    result = provider_manager.set_google_llm_settings(
        model="gemini-1.5-flash-latest",
        temperature=0.55,
        top_p=0.8,
        top_k=64,
        candidate_count=2,
        max_output_tokens=4096,
        stop_sequences=["STOP", "END"],
        safety_settings=[
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_LOW_AND_ABOVE"}
        ],
        stream=False,
        function_calling=False,
        function_call_mode="none",
        allowed_function_names=["tool_action"],
        response_schema={
            "type": "object",
            "properties": {"result": {"type": "string"}},
        },
        seed=2024,
        response_logprobs=True,
    )

    assert result["success"] is True
    assert result["data"]["max_output_tokens"] == 4096
    assert result["data"]["stream"] is False
    assert result["data"]["response_schema"] == {
        "type": "object",
        "properties": {"result": {"type": "string"}},
    }
    assert result["data"]["seed"] == 2024
    assert result["data"]["response_logprobs"] is True
    settings = provider_manager.config_manager.get_google_llm_settings()
    assert settings["model"] == "gemini-1.5-flash-latest"
    assert math.isclose(settings["temperature"], 0.55)
    assert math.isclose(settings["top_p"], 0.8)
    assert settings["top_k"] == 64
    assert settings["candidate_count"] == 2
    assert settings["max_output_tokens"] == 4096
    assert settings["stop_sequences"] == ["STOP", "END"]
    assert settings["safety_settings"][0]["category"] == "HARM_CATEGORY_DANGEROUS_CONTENT"
    assert settings["safety_settings"][0]["threshold"] == "BLOCK_LOW_AND_ABOVE"
    assert settings["stream"] is False
    assert settings["function_calling"] is False
    assert settings["function_call_mode"] == "none"
    assert settings["allowed_function_names"] == ["tool_action"]
    assert settings["response_schema"] == {
        "type": "object",
        "properties": {"result": {"type": "string"}},
    }
    assert settings["seed"] == 2024
    assert settings["response_logprobs"] is True
    assert provider_manager.model_manager.models["Google"][0] == "gemini-1.5-flash-latest"
    assert provider_manager.current_model == "gemini-1.5-flash-latest"


def test_set_google_llm_settings_rejects_invalid_schema(provider_manager):
    result = provider_manager.set_google_llm_settings(
        model="gemini-1.5-pro-latest",
        response_schema="{invalid",
    )

    assert result["success"] is False
    assert "Response schema" in result["error"]


def test_generate_response_uses_google_defaults(provider_manager):
    safety_settings = [{"category": "HARM_CATEGORY_DEROGATORY", "threshold": "BLOCK_LOW"}]

    provider_manager.config_manager.set_google_llm_settings(
        model="gemini-1.5-flash-latest",
        temperature=0.25,
        top_p=0.9,
        top_k=32,
        candidate_count=2,
        stop_sequences=["STOP"],
        safety_settings=safety_settings,
        response_mime_type="text/plain",
        system_instruction="Follow the instructions.",
        max_output_tokens=12345,
        stream=False,
        response_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
        },
        function_call_mode="require",
        allowed_function_names=["persona_tool"],
    )

    captured = {}

    async def fake_generate_response(_config_manager, **kwargs):
        captured.update(kwargs)
        settings = _config_manager.get_google_llm_settings()
        mode = str(settings.get("function_call_mode", "auto")).strip().lower()
        allowed = settings.get("allowed_function_names", []) or []
        if kwargs.get("enable_functions", True):
            payload_mode = mode.upper()
            payload_allowed = list(allowed)
        else:
            payload_mode = "NONE"
            payload_allowed = []
        payload = {"function_calling_config": {"mode": payload_mode}}
        if payload_allowed:
            payload["function_calling_config"]["allowed_function_names"] = payload_allowed
        captured["tool_config"] = payload
        return "ok"

    provider_manager.generate_response_func = fake_generate_response
    provider_manager.providers["Google"] = fake_generate_response
    provider_manager.current_llm_provider = "Google"
    provider_manager.current_model = "gemini-1.5-flash-latest"

    async def exercise():
        return await provider_manager.generate_response(
            messages=[{"role": "user", "content": "hi"}],
            provider="Google",
        )

    result = asyncio.run(exercise())
    assert result == "ok"
    assert captured["temperature"] == 0.25
    assert captured["top_p"] == 0.9
    assert captured["top_k"] == 32
    assert captured["candidate_count"] == 2
    assert captured["stop_sequences"] == ["STOP"]
    assert captured["safety_settings"] == safety_settings
    assert captured["response_mime_type"] == "text/plain"
    assert captured["system_instruction"] == "Follow the instructions."
    assert captured["max_tokens"] == 12345
    assert captured["stream"] is False
    assert captured["enable_functions"] is True
    assert captured["response_schema"] == {
        "type": "object",
        "properties": {"text": {"type": "string"}},
    }
    assert captured["tool_config"]["function_calling_config"]["mode"] == "REQUIRE"
    assert captured["tool_config"]["function_calling_config"]["allowed_function_names"] == [
        "persona_tool"
    ]


def test_generate_response_google_omits_hardcoded_max_tokens(provider_manager):
    provider_manager.config_manager.set_google_llm_settings(
        model="gemini-1.5-pro-latest",
        temperature=0.0,
        top_p=0.8,
    )
    provider_manager.config_manager.set_google_llm_settings(
        model="gemini-1.5-pro-latest",
        max_output_tokens="",
    )

    captured = {}

    async def fake_generate_response(_config_manager, **kwargs):
        captured.update(kwargs)
        settings = _config_manager.get_google_llm_settings()
        mode = str(settings.get("function_call_mode", "auto")).strip().lower()
        allowed = settings.get("allowed_function_names", []) or []
        if kwargs.get("enable_functions", True):
            payload_mode = mode.upper()
            payload_allowed = list(allowed)
        else:
            payload_mode = "NONE"
            payload_allowed = []
        payload = {"function_calling_config": {"mode": payload_mode}}
        if payload_allowed:
            payload["function_calling_config"]["allowed_function_names"] = payload_allowed
        captured["tool_config"] = payload
        return "ok"

    provider_manager.generate_response_func = fake_generate_response
    provider_manager.providers["Google"] = fake_generate_response
    provider_manager.current_llm_provider = "Google"
    provider_manager.current_model = "gemini-1.5-pro-latest"

    async def exercise():
        return await provider_manager.generate_response(
            messages=[{"role": "user", "content": "hello"}],
            provider="Google",
        )

    result = asyncio.run(exercise())
    assert result == "ok"
    assert "max_tokens" in captured
    assert captured["max_tokens"] is None
    assert captured["enable_functions"] is True


def test_generate_response_google_skips_functions_when_disabled(provider_manager):
    provider_manager.config_manager.set_google_llm_settings(
        model="gemini-1.5-pro-latest",
        function_calling=False,
    )

    captured = {}

    async def fake_generate_response(_config_manager, **kwargs):
        captured.update(kwargs)
        settings = _config_manager.get_google_llm_settings()
        mode = str(settings.get("function_call_mode", "auto")).strip().lower()
        allowed = settings.get("allowed_function_names", []) or []
        if kwargs.get("enable_functions", True):
            payload_mode = mode.upper()
            payload_allowed = list(allowed)
        else:
            payload_mode = "NONE"
            payload_allowed = []
        payload = {"function_calling_config": {"mode": payload_mode}}
        if payload_allowed:
            payload["function_calling_config"]["allowed_function_names"] = payload_allowed
        captured["tool_config"] = payload
        return "ok"

    provider_manager.generate_response_func = fake_generate_response
    provider_manager.providers["Google"] = fake_generate_response
    provider_manager.current_llm_provider = "Google"
    provider_manager.current_model = "gemini-1.5-pro-latest"
    provider_manager.current_functions = [{"name": "persona_tool"}]

    async def exercise():
        return await provider_manager.generate_response(
            messages=[{"role": "user", "content": "hello"}],
            provider="Google",
        )

    result = asyncio.run(exercise())
    assert result == "ok"
    assert captured["functions"] is None
    assert captured["enable_functions"] is False
    assert captured["tool_config"]["function_calling_config"]["mode"] == "NONE"


def test_generate_response_respects_function_calling_enabled(provider_manager):
    captured = {}

    async def fake_generate_response(_config_manager, **kwargs):
        captured["function_calling"] = kwargs.get("function_calling")
        captured["parallel_tool_calls"] = kwargs.get("parallel_tool_calls")
        captured["tool_choice"] = kwargs.get("tool_choice")
        captured["audio_enabled"] = kwargs.get("audio_enabled")
        captured["audio_voice"] = kwargs.get("audio_voice")
        captured["audio_format"] = kwargs.get("audio_format")
        return "ok"

    provider_manager.generate_response_func = fake_generate_response
    provider_manager.providers["OpenAI"] = fake_generate_response
    provider_manager.current_llm_provider = "OpenAI"
    provider_manager.current_model = "gpt-4o"

    async def exercise():
        return await provider_manager.generate_response(
            messages=[{"role": "user", "content": "hi"}],
            functions=[{"name": "tool"}],
        )

    result = asyncio.run(exercise())
    assert result == "ok"
    assert captured["function_calling"] is True
    assert captured["parallel_tool_calls"] is True
    assert captured["tool_choice"] is None
    assert captured["audio_enabled"] is False
    assert captured["audio_voice"] == provider_manager.get_openai_llm_settings()["audio_voice"]
    assert captured["audio_format"] == provider_manager.get_openai_llm_settings()["audio_format"]


def test_generate_response_respects_function_calling_disabled(provider_manager):
    provider_manager.config_manager.set_openai_llm_settings(model="gpt-4o", function_calling=False)

    captured = {}

    async def fake_generate_response(_config_manager, **kwargs):
        captured["function_calling"] = kwargs.get("function_calling")
        captured["parallel_tool_calls"] = kwargs.get("parallel_tool_calls")
        captured["tool_choice"] = kwargs.get("tool_choice")
        captured["audio_enabled"] = kwargs.get("audio_enabled")
        return "ok"

    provider_manager.generate_response_func = fake_generate_response
    provider_manager.providers["OpenAI"] = fake_generate_response
    provider_manager.current_llm_provider = "OpenAI"
    provider_manager.current_model = "gpt-4o"

    async def exercise():
        return await provider_manager.generate_response(
            messages=[{"role": "user", "content": "hello"}],
            functions=[{"name": "tool"}],
        )

    result = asyncio.run(exercise())
    assert result == "ok"
    assert captured["function_calling"] is False
    assert captured["parallel_tool_calls"] is True
    assert captured["tool_choice"] == "none"
    assert captured["audio_enabled"] is False


def test_generate_response_huggingface_uses_adapter(provider_manager):
    captured = {}

    async def fake_generate_response(self, messages, model, stream=True):
        captured["messages"] = messages
        captured["model"] = model
        captured["stream"] = stream
        return "hf-ok"

    async def exercise():
        await provider_manager.switch_llm_provider("HuggingFace")
        generator = provider_manager.huggingface_generator
        generator.generate_response = types.MethodType(fake_generate_response, generator)
        provider_manager.generate_response_func = generator.generate_response
        provider_manager.providers["HuggingFace"] = generator.generate_response
        return await provider_manager.generate_response(
            messages=[{"role": "user", "content": "ping"}],
            provider="HuggingFace",
            model="alpha",
            stream=False,
            temperature=0.75,
            max_tokens=256,
            functions=[{"name": "tool"}],
        )

    result = asyncio.run(exercise())
    assert result == "hf-ok"
    assert captured["messages"] == [{"role": "user", "content": "ping"}]
    assert captured["model"] == "alpha"
    assert captured["stream"] is False


def test_generate_response_switches_provider_uses_default_models(provider_manager, monkeypatch):
    calls = []

    async def fake_set_model(model):
        calls.append((provider_manager.current_llm_provider, model))
        provider_manager.current_model = model

    async def fake_openai_generate(_config_manager, **kwargs):
        return {"provider": "OpenAI", "model": kwargs.get("model")}

    async def fake_hf_generate(messages, model, stream=True):
        return {"provider": "HuggingFace", "model": model, "stream": stream}

    async def fake_switch(provider):
        provider_manager.current_llm_provider = provider
        provider_manager.current_model = None
        if provider == "HuggingFace":
            provider_manager.generate_response_func = fake_hf_generate
        elif provider == "OpenAI":
            provider_manager.generate_response_func = fake_openai_generate
        else:  # pragma: no cover - defensive guard for test
            raise AssertionError(f"Unexpected provider {provider}")
        provider_manager.providers[provider] = provider_manager.generate_response_func

    monkeypatch.setattr(provider_manager, "set_model", fake_set_model, raising=False)
    monkeypatch.setattr(provider_manager, "switch_llm_provider", fake_switch, raising=False)

    provider_manager.huggingface_generator = None
    provider_manager.current_llm_provider = "OpenAI"
    provider_manager.current_model = "gpt-4o"
    provider_manager.generate_response_func = fake_openai_generate
    provider_manager.providers["OpenAI"] = fake_openai_generate
    provider_manager.providers["HuggingFace"] = fake_hf_generate

    messages = [{"role": "user", "content": "hello"}]

    async def exercise():
        await provider_manager.generate_response(messages, provider="HuggingFace")
        await provider_manager.generate_response(messages, provider="OpenAI")

    asyncio.run(exercise())

    assert calls == [("HuggingFace", "alpha"), ("OpenAI", "gpt-4o")]


def test_generate_response_grok_uses_adapter(provider_manager):
    captured: Dict[str, Any] = {}
    persona = {"name": "Helper", "description": "Assistant persona"}
    functions_payload = [{"name": "tool", "description": "Does work"}]
    conversation_manager = object()

    async def fake_generate_response(
        self,
        messages,
        model="grok-2",
        max_tokens=1000,
        stream=False,
        **kwargs,
    ):
        captured["messages"] = messages
        captured["model"] = model
        captured["max_tokens"] = max_tokens
        captured["stream"] = stream
        captured["kwargs"] = kwargs
        return "grok-ok"

    async def exercise():
        provider_manager.model_manager.models["Grok"] = ["grok-2"]
        await provider_manager.switch_llm_provider("Grok")
        generator = provider_manager.grok_generator
        generator.generate_response = types.MethodType(fake_generate_response, generator)
        provider_manager.generate_response_func = generator.generate_response
        provider_manager.providers["Grok"] = generator.generate_response
        provider_manager.current_functions = [{"name": "unused"}]
        result = await provider_manager.generate_response(
            messages=[{"role": "user", "content": "pong"}],
            provider="Grok",
            model="grok-special",
            max_tokens=42,
            temperature=0.5,
            top_p=0.25,
            frequency_penalty=0.1,
            presence_penalty=0.05,
            functions=functions_payload,
            current_persona=persona,
            conversation_manager=conversation_manager,
            conversation_id="conv-123",
            user="alice",
            stream=True,
        )

        async def fake_stream():
            for chunk in ("gro", "k-stream"):
                yield chunk

        streaming_result = await provider_manager.process_streaming_response(fake_stream())
        return result, streaming_result

    result, streaming_output = asyncio.run(exercise())
    assert result == "grok-ok"
    assert captured["messages"] == [{"role": "user", "content": "pong"}]
    assert captured["model"] == "grok-special"
    assert captured["max_tokens"] == 42
    assert captured["stream"] is True
    forwarded_kwargs = captured["kwargs"]
    assert forwarded_kwargs["temperature"] == 0.5
    assert forwarded_kwargs["top_p"] == 0.25
    assert forwarded_kwargs["frequency_penalty"] == 0.1
    assert forwarded_kwargs["presence_penalty"] == 0.05
    assert forwarded_kwargs["current_persona"] is persona
    assert forwarded_kwargs["functions"] == functions_payload
    assert forwarded_kwargs["functions"] is functions_payload
    assert forwarded_kwargs["conversation_manager"] is conversation_manager
    assert forwarded_kwargs["conversation_id"] == "conv-123"
    assert forwarded_kwargs["user"] == "alice"
    assert streaming_output == "grok-stream"


def _load_real_grok_generator(module_label: str = "test"):
    module_name = f"_atlas_actual_grok_{module_label}"
    module_path = Path(__file__).resolve().parents[1] / "modules/Providers/Grok/grok_generate_response.py"

    if "xai_sdk" not in sys.modules:
        xai_stub = types.ModuleType("xai_sdk")

        class _StubClient:
            def __init__(self, *_args, **_kwargs):
                empty_response = types.SimpleNamespace(content="", tool_calls=[])

                def _create(**_kwargs):
                    return types.SimpleNamespace(
                        sample=lambda: empty_response,
                        stream=lambda: iter(()),
                    )

                self.chat = types.SimpleNamespace(create=_create)

        xai_stub.Client = _StubClient
        sys.modules["xai_sdk"] = xai_stub

        chat_module = types.ModuleType("xai_sdk.chat")

        def _wrap(role):
            return lambda content: {"role": role, "content": content}

        def _tool(name, description, parameters):
            return {
                "name": name,
                "description": description,
                "parameters": parameters,
            }

        chat_module.assistant = _wrap("assistant")
        chat_module.system = _wrap("system")
        chat_module.user = _wrap("user")
        chat_module.tool_result = _wrap("tool")
        chat_module.tool = _tool

        sys.modules["xai_sdk.chat"] = chat_module

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        assert spec.loader is not None
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)

    return module.GrokGenerator


def test_grok_generator_passes_model_to_sampler_non_streaming():
    GrokGenerator = _load_real_grok_generator("non_stream")

    class _Config:
        @staticmethod
        def get_grok_api_key():
            return "token"

    create_calls = []

    class _StubResponse:
        def __init__(self):
            self.content = "hello world"
            self.tool_calls = []

    class _StubChat:
        def __init__(self):
            self.sample_count = 0

        def sample(self):
            self.sample_count += 1
            return _StubResponse()

    async def exercise():
        generator = GrokGenerator(_Config())

        def create(**kwargs):
            create_calls.append(kwargs)
            return _StubChat()

        generator.client = types.SimpleNamespace(
            chat=types.SimpleNamespace(create=create)
        )

        response = await generator.generate_response(
            messages=[{"role": "user", "content": "Hi"}],
            model="grok-special",
            max_tokens=77,
            stream=False,
        )

        return response

    response = asyncio.run(exercise())

    assert response == "hello world"
    assert len(create_calls) == 1
    call = create_calls[0]
    assert call["model"] == "grok-special"
    assert call["max_tokens"] == 77
    assert len(call["messages"]) == 1


def test_grok_generator_passes_model_to_sampler_streaming():
    GrokGenerator = _load_real_grok_generator("stream")

    class _Config:
        @staticmethod
        def get_grok_api_key():
            return "token"

    create_calls = []

    class _StubResponse:
        def __init__(self):
            self.content = ""
            self.tool_calls = []

    class _StubChunk:
        def __init__(self, content: str):
            self.content = content

    class _StubChat:
        def stream(self):
            response = _StubResponse()
            for part in ("hi", " there"):
                response.content += part
                yield response, _StubChunk(part)

    async def exercise():
        generator = GrokGenerator(_Config())

        def create(**kwargs):
            create_calls.append(kwargs)
            return _StubChat()

        generator.client = types.SimpleNamespace(
            chat=types.SimpleNamespace(create=create)
        )

        stream = await generator.generate_response(
            messages=[{"role": "user", "content": "Hi"}],
            model="grok-stream",
            max_tokens=55,
            stream=True,
        )

        chunks = []
        async for chunk in stream:
            chunks.append(chunk)

        return chunks

    chunks = asyncio.run(exercise())

    assert chunks == ["hi", " there"]
    assert len(create_calls) == 1
    call = create_calls[0]
    assert call["model"] == "grok-stream"
    assert call["max_tokens"] == 55
    assert len(call["messages"]) == 1


def test_grok_generator_merges_generation_settings_metadata():
    GrokGenerator = _load_real_grok_generator("merge")

    class _Config:
        @staticmethod
        def get_grok_api_key():
            return "token"

    generator = GrokGenerator(_Config())

    merged = generator._merge_generation_settings(  # type: ignore[attr-defined]
        {"existing": "value"},
        function_calling=True,
        parallel_tool_calls=False,
        tool_choice="auto",
        tool_choice_name="special",
        allowed_function_names=["tool_a"],
        function_call_mode="require",
        tool_prompt_data={"notes": "metadata"},
    )

    assert merged["existing"] == "value"
    assert merged["function_calling"] is True
    assert merged["parallel_tool_calls"] is False
    assert merged["tool_choice"] == "auto"
    assert merged["tool_choice_name"] == "special"
    assert merged["allowed_function_names"] == ["tool_a"]
    assert merged["function_call_mode"] == "require"
    assert merged["tool_prompt_data"] == {"notes": "metadata"}


def test_generate_response_uses_configured_fallback(provider_manager):
    fallback_result = "fallback-response"

    class StubMistralGenerator:
        def __init__(self):
            self.calls = []

        async def generate_response(self, **kwargs):
            self.calls.append(kwargs)
            return fallback_result

    stub_generator = StubMistralGenerator()
    provider_manager._mistral_generator = stub_generator
    provider_manager.providers.pop("Mistral", None)

    provider_manager.config_manager.update_api_key("Mistral", "token")
    provider_manager.config_manager.set_llm_fallback_config(
        provider="Mistral",
        model="mistral-large-latest",
        temperature=0.1,
        max_tokens=2048,
    )

    async def failing_generate_response(*_args, **_kwargs):
        raise RuntimeError("primary failed")

    provider_manager.generate_response_func = failing_generate_response
    provider_manager.providers["OpenAI"] = failing_generate_response
    provider_manager.current_llm_provider = "OpenAI"
    provider_manager.current_model = "gpt-4o"

    messages = [{"role": "user", "content": "hello"}]

    async def exercise():
        return await provider_manager.generate_response(
            messages,
            provider="OpenAI",
            llm_call_type="chat",
            max_tokens=123,
            temperature=0.5,
        )

    result = asyncio.run(exercise())

    assert result == fallback_result
    assert stub_generator.calls, "Fallback provider was not invoked."
    call_kwargs = stub_generator.calls[0]
    assert call_kwargs["messages"] == messages
    assert call_kwargs["model"] == "mistral-large-latest"
    assert call_kwargs["max_tokens"] == 2048
    assert call_kwargs["temperature"] == 0.1


def test_generate_response_openai_fallback_filters_manager_kwargs(provider_manager):
    fallback_result = "openai-fallback"

    class StrictOpenAIGenerator:
        def __init__(self):
            self.calls = []

        async def generate_response(
            self,
            messages,
            model=None,
            max_tokens=None,
            temperature=None,
            stream=None,
            current_persona=None,
            functions=None,
            user=None,
            conversation_id=None,
            conversation_manager=None,
            top_p=None,
            frequency_penalty=None,
            presence_penalty=None,
        ):
            payload = {
                "messages": messages,
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": stream,
                "current_persona": current_persona,
                "functions": functions,
                "user": user,
                "conversation_id": conversation_id,
                "conversation_manager": conversation_manager,
                "top_p": top_p,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
            }
            self.calls.append(payload)
            return fallback_result

    stub_generator = StrictOpenAIGenerator()
    provider_manager._openai_generator = stub_generator
    provider_manager.providers.pop("OpenAI", None)

    provider_manager.config_manager.update_api_key("OpenAI", "token")
    provider_manager.config_manager.set_llm_fallback_config(
        provider="OpenAI",
        model="gpt-4o-mini",
        temperature=0.25,
        max_tokens=512,
        top_p=0.8,
        frequency_penalty=0.1,
        presence_penalty=0.2,
    )

    async def failing_generate_response(*_args, **_kwargs):
        raise RuntimeError("primary failed")

    provider_manager.generate_response_func = failing_generate_response
    provider_manager.providers["Mistral"] = failing_generate_response
    provider_manager.current_llm_provider = "Mistral"
    provider_manager.current_model = "mistral-large-latest"

    messages = [{"role": "user", "content": "hello"}]

    async def exercise():
        return await provider_manager.generate_response(
            messages,
            provider="Mistral",
            llm_call_type="chat",
            max_tokens=321,
            temperature=0.9,
        )

    result = asyncio.run(exercise())

    assert result == fallback_result
    assert stub_generator.calls, "Fallback provider was not invoked."
    call_kwargs = stub_generator.calls[0]
    assert call_kwargs["messages"] == messages
    assert call_kwargs["model"] == "gpt-4o-mini"
    assert call_kwargs["max_tokens"] == 512
    assert call_kwargs["temperature"] == 0.25
    assert call_kwargs["top_p"] == 0.8
    assert call_kwargs["frequency_penalty"] == 0.1
    assert call_kwargs["presence_penalty"] == 0.2
    assert "llm_call_type" not in call_kwargs


def test_close_unloads_grok_generator(provider_manager):
    unload_calls = {"count": 0}

    async def exercise():
        provider_manager.model_manager.models["Grok"] = ["grok-2"]
        await provider_manager.switch_llm_provider("Grok")
        generator = provider_manager.grok_generator

        async def tracked_unload(self):
            unload_calls["count"] += 1

        generator.unload_model = types.MethodType(tracked_unload, generator)

        await provider_manager.close()
        return provider_manager.grok_generator

    generator_after_close = asyncio.run(exercise())

    assert unload_calls["count"] == 1
    assert generator_after_close is None
    assert provider_manager.grok_generator is None


def test_close_disposes_cached_generators(provider_manager):
    class TrackingGenerator:
        def __init__(self):
            self.calls = []

        async def aclose(self):
            self.calls.append("aclose")

        async def close(self):
            self.calls.append("close")
            await self.aclose()

    async def exercise():
        openai = TrackingGenerator()
        mistral = TrackingGenerator()
        anthropic = TrackingGenerator()

        provider_manager._openai_generator = openai
        provider_manager._mistral_generator = mistral
        provider_manager.anthropic_generator = anthropic

        await provider_manager.close()

        return openai, mistral, anthropic

    openai, mistral, anthropic = asyncio.run(exercise())

    assert openai.calls == ["close", "aclose"]
    assert mistral.calls == ["close", "aclose"]
    assert anthropic.calls == ["close", "aclose"]
    assert provider_manager._openai_generator is None
    assert provider_manager._mistral_generator is None
    assert provider_manager.anthropic_generator is None


def test_openai_settings_window_populates_defaults_and_saves(provider_manager):
    atlas_stub = types.SimpleNamespace()

    saved_payload = {}
    model_calls: Dict[str, Tuple[Optional[str], Optional[str]]] = {}

    def fake_set_openai_llm_settings(**kwargs):
        saved_payload.update(kwargs)
        return {"success": True, "message": "saved", "data": dict(kwargs)}

    async def fake_list_openai_models(*, base_url=None, organization=None):
        model_calls["values"] = (base_url, organization)
        return {"models": ["gpt-4o-mini", "gpt-4o"], "error": None}

    def fake_run_in_background(factory, *, on_success=None, on_error=None, **_kwargs):
        try:
            result = asyncio.run(factory())
        except Exception as exc:  # pragma: no cover - defensive logging
            if on_error is not None:
                on_error(exc)
        else:
            if on_success is not None:
                on_success(result)
        return types.SimpleNamespace()

    atlas_stub.provider_manager = provider_manager
    atlas_stub.set_openai_llm_settings = fake_set_openai_llm_settings
    atlas_stub.get_openai_llm_settings = lambda: {
        "model": "gpt-4o-mini",
        "temperature": 0.65,
        "top_p": 0.85,
        "frequency_penalty": 0.05,
        "presence_penalty": -0.1,
        "max_tokens": 2048,
        "max_output_tokens": 512,
        "stream": False,
        "function_calling": False,
        "parallel_tool_calls": False,
        "tool_choice": "required",
        "base_url": "https://example/v1",
        "organization": "org-42",
        "reasoning_effort": "high",
        "json_schema": None,
        "enable_code_interpreter": True,
        "enable_file_search": True,
        "audio_enabled": True,
        "audio_voice": "lumen",
        "audio_format": "ogg",
    }
    atlas_stub.update_provider_api_key = provider_manager.update_provider_api_key
    atlas_stub.list_openai_models = fake_list_openai_models
    atlas_stub.run_in_background = fake_run_in_background
    atlas_stub.get_provider_api_key_status = lambda name: {
        "has_key": True,
        "metadata": {"hint": "••••"},
    }

    window = OpenAISettingsWindow(atlas_stub, provider_manager.config_manager, None)

    assert window.model_combo.get_active_text() == "gpt-4o-mini"
    assert window.temperature_spin.get_value() == 0.65
    assert window.top_p_spin.get_value() == 0.85
    assert math.isclose(window.frequency_penalty_spin.get_value(), 0.05)
    assert math.isclose(window.presence_penalty_spin.get_value(), -0.1)
    assert window.max_tokens_spin.get_value_as_int() == 2048
    assert window.max_output_tokens_spin.get_value_as_int() == 512
    assert window.stream_toggle.get_active() is False
    assert window.function_call_toggle.get_active() is False
    assert window.parallel_tool_calls_toggle.get_active() is False
    assert window.require_tool_toggle.get_active() is False
    assert window.code_interpreter_toggle.get_active() is False
    assert window.file_search_toggle.get_active() is False
    assert window.audio_reply_toggle.get_active() is True
    assert window.audio_voice_combo.get_active_text() == "lumen"
    assert window.audio_format_combo.get_active_text() == "ogg"
    assert window.organization_entry.get_text() == "org-42"
    assert window.reasoning_effort_combo.get_active_text() == "high"
    status_text = getattr(window.api_key_status_label, "label", None)
    if status_text is None and hasattr(window.api_key_status_label, "get_text"):
        status_text = window.api_key_status_label.get_text()
    assert status_text == "An API key is saved for OpenAI. (••••)"
    assert window._stored_base_url == "https://example/v1"
    assert model_calls["values"] == ("https://example/v1", "org-42")

    window.model_combo.set_active(1)
    window.temperature_spin.set_value(0.5)
    window.top_p_spin.set_value(0.75)
    window.frequency_penalty_spin.set_value(0.2)
    window.presence_penalty_spin.set_value(-0.15)
    window.max_tokens_spin.set_value(4096)
    window.max_output_tokens_spin.set_value(1024)
    window.stream_toggle.set_active(True)
    window.function_call_toggle.set_active(True)
    window.parallel_tool_calls_toggle.set_active(True)
    window.require_tool_toggle.set_active(True)
    window.code_interpreter_toggle.set_active(True)
    window.file_search_toggle.set_active(False)
    window.audio_reply_toggle.set_active(True)
    window.audio_voice_combo.set_active(0)
    window.audio_format_combo.set_active(0)
    window.organization_entry.set_text("org-new")
    window.base_url_entry.set_text("https://alt.example/v2")
    window.reasoning_effort_combo.set_active(1)

    window.on_save_clicked(window.model_combo)

    assert saved_payload["model"] == "gpt-4o"
    assert saved_payload["temperature"] == 0.5
    assert saved_payload["top_p"] == 0.75
    assert saved_payload["frequency_penalty"] == 0.2
    assert saved_payload["presence_penalty"] == -0.15
    assert saved_payload["max_tokens"] == 4096
    assert saved_payload["max_output_tokens"] == 1024
    assert saved_payload["stream"] is True
    assert saved_payload["function_calling"] is True
    assert saved_payload["parallel_tool_calls"] is True
    assert saved_payload["base_url"] == "https://alt.example/v2"
    assert saved_payload["organization"] == "org-new"
    assert saved_payload["reasoning_effort"] == "medium"
    assert saved_payload["tool_choice"] == "required"
    assert saved_payload["json_schema"] == ""
    assert saved_payload["enable_code_interpreter"] is True
    assert saved_payload["enable_file_search"] is False
    assert saved_payload["audio_enabled"] is True
    assert saved_payload["audio_voice"] == "alloy"
    assert saved_payload["audio_format"] == "wav"
    assert window._stored_base_url == "https://alt.example/v2"
    assert window._last_message[0] == "Success"
    assert window.closed is True


def test_openai_settings_window_saves_api_key(provider_manager):
    atlas_stub = types.SimpleNamespace()

    atlas_stub.provider_manager = provider_manager
    atlas_stub.get_openai_llm_settings = lambda: {
        "model": "gpt-4o",
        "temperature": 0.0,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "max_tokens": 4000,
        "max_output_tokens": None,
        "stream": True,
        "base_url": None,
        "organization": None,
        "reasoning_effort": "medium",
        "json_schema": None,
        "audio_enabled": False,
        "audio_voice": "alloy",
        "audio_format": "wav",
    }
    atlas_stub.set_openai_llm_settings = lambda **_: {"success": True, "message": "saved"}
    atlas_stub.update_provider_api_key = provider_manager.update_provider_api_key

    async def fake_list_models(*, base_url=None, organization=None):
        return {"models": ["gpt-4o"], "error": None}

    def fake_run_in_background(factory, *, on_success=None, on_error=None, **_kwargs):
        try:
            result = asyncio.run(factory())
        except Exception as exc:  # pragma: no cover - defensive logging
            if on_error is not None:
                on_error(exc)
        else:
            if on_success is not None:
                on_success(result)
        return types.SimpleNamespace()

    def fake_status(_name: str):
        has_key = provider_manager.config_manager.has_provider_api_key("OpenAI")
        metadata = {"hint": "••••"} if has_key else {}
        return {"has_key": has_key, "metadata": metadata}

    atlas_stub.list_openai_models = fake_list_models
    atlas_stub.run_in_background = fake_run_in_background
    atlas_stub.get_provider_api_key_status = fake_status

    window = OpenAISettingsWindow(atlas_stub, provider_manager.config_manager, None)
    window.api_key_entry.set_text("sk-test")

    assert window.top_p_spin.get_value() == 1.0
    assert window.frequency_penalty_spin.get_value() == 0.0
    assert window.presence_penalty_spin.get_value() == 0.0

    window.on_save_api_key_clicked(None)

    assert provider_manager.config_manager.get_openai_api_key() == "sk-test"
    assert window._last_message[0] == "Success"


def test_anthropic_settings_window_dispatches_updates(provider_manager):
    atlas_stub = types.SimpleNamespace()

    saved_payload: Dict[str, object] = {}
    atlas_stub.provider_manager = provider_manager
    atlas_stub.get_anthropic_settings = lambda: {
        "model": "claude-3-opus-20240229",
        "stream": True,
        "function_calling": False,
        "temperature": 0.1,
        "top_p": 0.95,
        "top_k": 12,
        "max_output_tokens": None,
        "timeout": 75,
        "max_retries": 2,
        "retry_delay": 6,
        "stop_sequences": ["END"],
        "tool_choice": "auto",
        "tool_choice_name": None,
        "metadata": {"team": "atlas"},
        "thinking": False,
        "thinking_budget": None,
    }
    atlas_stub.get_models_for_provider = lambda name: [
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
    ]

    def fake_set_anthropic_settings(**kwargs):
        saved_payload.update(kwargs)
        return {"success": True, "message": "saved", "data": dict(kwargs)}

    atlas_stub.set_anthropic_settings = fake_set_anthropic_settings

    window = AnthropicSettingsWindow(atlas_stub, provider_manager.config_manager, None)

    assert window.model_combo.get_active_text() == "claude-3-opus-20240229"
    assert window.streaming_toggle.get_active() is True
    assert window.function_call_toggle.get_active() is False
    assert round(window.temperature_spin.get_value(), 2) == 0.1
    assert round(window.top_p_spin.get_value(), 2) == 0.95
    assert window.top_k_spin.get_value_as_int() == 12
    assert window.max_output_tokens_spin.get_value_as_int() == 0
    assert window.stop_sequences_entry.get_text() == "END"
    assert window.timeout_spin.get_value_as_int() == 75
    assert window.max_retries_spin.get_value_as_int() == 2
    assert window.retry_delay_spin.get_value_as_int() == 6
    assert window.metadata_entry.get_text() == "team=atlas"
    assert window._get_tool_choice_value() == "auto"
    assert window.tool_name_entry.get_text() == ""
    assert window.thinking_toggle.get_active() is False

    window.model_combo.set_active(1)
    window.streaming_toggle.set_active(False)
    window.function_call_toggle.set_active(True)
    window.tool_choice_combo.set_active(3)
    window.tool_name_entry.set_text("calendar")
    window.temperature_spin.set_value(0.55)
    window.top_p_spin.set_value(0.85)
    window.top_k_spin.set_value(25)
    window.max_output_tokens_spin.set_value(4096)
    window.stop_sequences_entry.set_text("END, FINISH")
    window.timeout_spin.set_value(180)
    window.max_retries_spin.set_value(6)
    window.retry_delay_spin.set_value(15)
    window.metadata_entry.set_text("team=qa, priority=high")
    window.thinking_toggle.set_active(True)
    window.thinking_budget_spin.set_value(3072)

    window.on_save_clicked(None)

    assert saved_payload["model"] == "claude-3-sonnet-20240229"
    assert saved_payload["stream"] is False
    assert saved_payload["function_calling"] is True
    assert math.isclose(saved_payload["temperature"], 0.55)
    assert math.isclose(saved_payload["top_p"], 0.85)
    assert saved_payload["top_k"] == 25
    assert saved_payload["max_output_tokens"] == 4096
    assert saved_payload["stop_sequences"] == ["END", "FINISH"]
    assert saved_payload["timeout"] == 180
    assert saved_payload["max_retries"] == 6
    assert saved_payload["retry_delay"] == 15
    assert saved_payload["tool_choice"] == "tool"
    assert saved_payload["tool_choice_name"] == "calendar"
    assert saved_payload["metadata"] == {"team": "qa", "priority": "high"}
    assert saved_payload["thinking"] is True
    assert saved_payload["thinking_budget"] == 3072
    assert window._last_message[0] == "Success"
    assert window.closed is True


def test_mistral_settings_window_round_trips_defaults(provider_manager, monkeypatch):
    monkeypatch.setattr("GTKUI.Utils.utils.apply_css", lambda *_, **__: None)
    monkeypatch.setattr("GTKUI.Utils.styled_window.apply_css", lambda *_, **__: None)
    monkeypatch.setattr(
        Gtk.Window,
        "get_style_context",
        lambda self: types.SimpleNamespace(add_class=lambda *_args, **_kwargs: None),
        raising=False,
    )
    config = provider_manager.config_manager
    schema_payload = {
        "name": "atlas_response",
        "schema": {"type": "object", "properties": {"ok": {"type": "boolean"}}},
    }
    config.set_mistral_llm_settings(
        model="mistral-medium-latest",
        temperature=0.4,
        top_p=0.8,
        max_tokens=2048,
        safe_prompt=True,
        stream=False,
        random_seed=1234,
        frequency_penalty=0.15,
        presence_penalty=-0.3,
        parallel_tool_calls=False,
        tool_choice={"type": "function", "name": "math"},
        stop_sequences=["HALT"],
        json_mode=True,
        json_schema=schema_payload,
        max_retries=4,
        retry_min_seconds=5,
        retry_max_seconds=18,
        base_url="https://example.mistral/v1",
        prompt_mode="reasoning",
    )

    call_state = {"count": 0}

    original_has_provider_api_key = config.has_provider_api_key

    def tracked_has_provider_api_key(provider_name: str) -> bool:
        call_state["count"] += 1
        return original_has_provider_api_key(provider_name)

    monkeypatch.setattr(config, "has_provider_api_key", tracked_has_provider_api_key)

    atlas_stub = types.SimpleNamespace(
        provider_manager=provider_manager,
        config_manager=config,
        get_provider_api_key_status=lambda _name: {"has_key": False, "metadata": {}},
    )

    window = MistralSettingsWindow(atlas_stub, config, None)

    assert call_state["count"] >= 1
    status_text = getattr(window.api_key_status_label, "label", None)
    if status_text is None and hasattr(window.api_key_status_label, "get_text"):
        status_text = window.api_key_status_label.get_text()
    assert status_text == "No API key saved for Mistral."

    assert window.model_combo.get_active_text() == "mistral-medium-latest"
    assert window._custom_entry_visible is False
    assert window.base_url_entry.get_text() == "https://example.mistral/v1"
    assert window._stored_base_url == "https://example.mistral/v1"
    assert math.isclose(window.temperature_spin.get_value(), 0.4)
    assert math.isclose(window.top_p_spin.get_value(), 0.8)
    assert math.isclose(window.frequency_penalty_spin.get_value(), 0.15)
    assert math.isclose(window.presence_penalty_spin.get_value(), -0.3)
    assert window.max_tokens_spin.get_value_as_int() == 2048
    assert window.safe_prompt_toggle.get_active() is True
    assert window.stream_toggle.get_active() is False
    assert window._get_selected_prompt_mode() == "reasoning"
    assert window.max_retries_spin.get_value_as_int() == 4
    assert window.retry_min_spin.get_value_as_int() == 5
    assert window.retry_max_spin.get_value_as_int() == 18
    assert window.tool_call_toggle.get_active() is True
    assert window.parallel_tool_calls_toggle.get_active() is False
    assert window.require_tool_toggle.get_active() is False
    assert window.random_seed_entry.get_text() == "1234"
    assert json.loads(window.tool_choice_entry.get_text()) == {"name": "math", "type": "function"}
    assert window.stop_sequences_entry.get_text() == "HALT"
    assert window.json_mode_toggle.get_active() is True
    assert "\"type\": \"object\"" in window._json_schema_text_cache

    config.set_mistral_llm_settings(
        model="mistral-large-latest",
        temperature=0.9,
        top_p=0.6,
        max_tokens=None,
        safe_prompt=False,
        stream=True,
        random_seed=None,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        parallel_tool_calls=True,
        tool_choice="auto",
        stop_sequences=[],
        json_mode=False,
        json_schema="",
        max_retries=3,
        retry_min_seconds=4,
        retry_max_seconds=12,
        base_url="",
        prompt_mode=None,
    )

    window.refresh_settings()

    assert call_state["count"] >= 2

    assert window.model_combo.get_active_text() == "mistral-large-latest"
    assert window._custom_entry_visible is False
    assert window.custom_model_entry.get_text() == ""
    assert math.isclose(window.temperature_spin.get_value(), 0.9)
    assert math.isclose(window.top_p_spin.get_value(), 0.6)
    assert window.max_tokens_spin.get_value_as_int() == 0
    assert window.safe_prompt_toggle.get_active() is False
    assert window.stream_toggle.get_active() is True
    assert window._get_selected_prompt_mode() is None
    assert window.max_retries_spin.get_value_as_int() == 3
    assert window.retry_min_spin.get_value_as_int() == 4
    assert window.retry_max_spin.get_value_as_int() == 12
    assert window.tool_call_toggle.get_active() is True
    assert window.parallel_tool_calls_toggle.get_active() is True
    assert window.require_tool_toggle.get_active() is False
    assert window.random_seed_entry.get_text() == ""
    assert window.tool_choice_entry.get_text() == ""
    assert window.stop_sequences_entry.get_text() == ""
    assert window.json_mode_toggle.get_active() is False
    assert window._json_schema_text_cache == ""
    assert window.base_url_entry.get_text() == ""
    assert window._stored_base_url is None

    small_index = window._available_models.index("mistral-small-latest")
    window.model_combo.set_active(small_index)
    window._on_model_combo_changed(window.model_combo)
    window.temperature_spin.set_value(0.25)
    window.top_p_spin.set_value(0.55)
    window.frequency_penalty_spin.set_value(-0.2)
    window.presence_penalty_spin.set_value(0.35)
    window.max_tokens_spin.set_value(1024)
    window.safe_prompt_toggle.set_active(True)
    window.stream_toggle.set_active(False)
    window.max_retries_spin.set_value(7)
    window.retry_min_spin.set_value(3)
    window.retry_max_spin.set_value(9)
    window.parallel_tool_calls_toggle.set_active(True)
    window.tool_call_toggle.set_active(False)
    window.random_seed_entry.set_text("0")
    window.stop_sequences_entry.set_text("END, FINISH")
    window.json_mode_toggle.set_active(True)
    window._write_json_schema_text(
        json.dumps(
            {
                "name": "atlas_custom",
                "schema": {
                    "type": "object",
                    "properties": {"result": {"type": "string"}},
                },
                "strict": True,
            },
            indent=2,
        )
    )
    if hasattr(window.prompt_mode_combo, "set_active_id"):
        window.prompt_mode_combo.set_active_id("reasoning")
    elif hasattr(window.prompt_mode_combo, "set_active"):
        window.prompt_mode_combo.set_active(1)
    window.base_url_entry.set_text("https://alt.mistral/v2")

    window.on_save_clicked(None)

    stored = config.get_mistral_llm_settings()

    assert stored["model"] == "mistral-small-latest"
    assert math.isclose(stored["temperature"], 0.25)
    assert math.isclose(stored["top_p"], 0.55)
    assert math.isclose(stored["frequency_penalty"], -0.2)
    assert math.isclose(stored["presence_penalty"], 0.35)
    assert stored["max_tokens"] == 1024
    assert stored["safe_prompt"] is True
    assert stored["stream"] is False
    assert stored["parallel_tool_calls"] is False
    assert stored["max_retries"] == 7
    assert stored["retry_min_seconds"] == 3
    assert stored["retry_max_seconds"] == 9
    assert stored["random_seed"] == 0
    assert stored["tool_choice"] == "none"
    assert stored["stop_sequences"] == ["END", "FINISH"]
    assert stored["json_mode"] is True
    assert stored["json_schema"]["schema"]["properties"]["result"]["type"] == "string"
    assert stored["base_url"] == "https://alt.mistral/v2"
    assert stored["prompt_mode"] == "reasoning"
    assert window._last_message[0] == "Success"
    assert window.model_combo.get_active_text() == "mistral-small-latest"
    assert window.base_url_entry.get_text() == "https://alt.mistral/v2"
    assert window._stored_base_url == "https://alt.mistral/v2"

    custom_index = len(window._available_models)
    window.model_combo.set_active(custom_index)
    window._on_model_combo_changed(window.model_combo)
    assert window._custom_entry_visible is True
    window.custom_model_entry.set_text("mistral-experimental")

    window.on_save_clicked(None)

    stored = config.get_mistral_llm_settings()
    assert stored["model"] == "mistral-experimental"
    assert window.model_combo.get_active_text() == window._custom_option_text
    assert window._custom_entry_visible is True
    assert window.custom_model_entry.get_text() == "mistral-experimental"

    window.refresh_settings()
    assert window.model_combo.get_active_text() == window._custom_option_text
    assert window._custom_entry_visible is True
    assert window.custom_model_entry.get_text() == "mistral-experimental"
    assert window.tool_choice_entry.get_text() == ""
    assert window.tool_call_toggle.get_active() is False
    assert window.parallel_tool_calls_toggle.get_active() is False
    assert window.require_tool_toggle.get_active() is False
    assert window.base_url_entry.get_text() == "https://alt.mistral/v2"
    assert window._get_selected_prompt_mode() == "reasoning"

    window.max_tokens_spin.set_value(0)

    window.on_save_clicked(None)

    stored = config.get_mistral_llm_settings()
    assert stored["max_tokens"] is None
    assert stored["model"] == "mistral-experimental"
    assert window.max_tokens_spin.get_value_as_int() == 0
    assert window._custom_entry_visible is True
    assert stored["base_url"] == "https://alt.mistral/v2"
    assert stored["prompt_mode"] == "reasoning"


def test_mistral_settings_window_saves_api_key(provider_manager, monkeypatch):
    monkeypatch.setattr("GTKUI.Utils.utils.apply_css", lambda *_, **__: None)
    monkeypatch.setattr("GTKUI.Utils.styled_window.apply_css", lambda *_, **__: None)
    monkeypatch.setattr(
        Gtk.Window,
        "get_style_context",
        lambda self: types.SimpleNamespace(add_class=lambda *_args, **_kwargs: None),
        raising=False,
    )

    config = provider_manager.config_manager
    saved_call: Dict[str, object] = {}

    def fake_status(_name: str) -> Dict[str, object]:
        has_key = config.has_provider_api_key("Mistral")
        metadata: Dict[str, str] = {}
        if has_key:
            key = config._api_keys.get("Mistral", "")
            hint = f"••••{key[-4:]}" if len(key) >= 4 else "••••"
            metadata["hint"] = hint
        return {"has_key": has_key, "metadata": metadata}

    def fake_update_in_background(
        provider: str,
        api_key: str,
        *,
        on_success=None,
        on_error=None,
    ) -> None:
        try:
            config.update_api_key(provider, api_key)
        except Exception as exc:  # pragma: no cover - defensive fallback
            if on_error is not None:
                on_error(exc)
            return
        saved_call["provider"] = provider
        saved_call["api_key"] = api_key
        if on_success is not None:
            on_success({"success": True, "message": "saved"})

    atlas_stub = types.SimpleNamespace(
        provider_manager=provider_manager,
        config_manager=config,
        update_provider_api_key_in_background=fake_update_in_background,
        update_provider_api_key=provider_manager.update_provider_api_key,
        get_provider_api_key_status=fake_status,
    )

    window = MistralSettingsWindow(atlas_stub, config, None)

    window.api_key_entry.set_text("mst-test-key")
    window.on_save_api_key_clicked(None)

    assert saved_call == {"provider": "Mistral", "api_key": "mst-test-key"}
    assert window.api_key_entry.get_text() == ""
    assert window._last_message[0] == "Success"
    status_text = getattr(window.api_key_status_label, "label", None)
    if status_text is None and hasattr(window.api_key_status_label, "get_text"):
        status_text = window.api_key_status_label.get_text()
    assert status_text.startswith("An API key is saved for Mistral.")
    placeholder = getattr(window.api_key_entry, "placeholder", None)
    if placeholder is None and hasattr(window.api_key_entry, "get_placeholder_text"):
        placeholder = window.api_key_entry.get_placeholder_text()
    assert placeholder and placeholder.startswith("Saved key:")


def test_mistral_settings_window_saves_api_key_with_fallback(
    provider_manager, monkeypatch
):
    monkeypatch.setattr("GTKUI.Utils.utils.apply_css", lambda *_, **__: None)
    monkeypatch.setattr("GTKUI.Utils.styled_window.apply_css", lambda *_, **__: None)
    monkeypatch.setattr(
        Gtk.Window,
        "get_style_context",
        lambda self: types.SimpleNamespace(add_class=lambda *_args, **_kwargs: None),
        raising=False,
    )

    config = provider_manager.config_manager
    status_payload = {"has_key": False, "metadata": {}}

    def fake_status(_name: str) -> Dict[str, object]:
        return dict(status_payload)

    state = {"scheduled": False, "saved": False}

    def fake_update_provider_api_key(provider: str, api_key: str):
        state["saved"] = True
        config.update_api_key(provider, api_key)
        return {"success": True, "message": "saved"}

    class _FakeFuture:
        def result(self):
            return None

    def fake_run_async_in_thread(factory, *, on_success=None, on_error=None, **_kwargs):
        state["scheduled"] = True
        try:
            result = factory()
        except Exception as exc:  # pragma: no cover - defensive logging
            if on_error is not None:
                on_error(exc)
        else:
            if on_success is not None:
                on_success(result)
        return _FakeFuture()

    monkeypatch.setattr(Mistral_settings, "run_async_in_thread", fake_run_async_in_thread)

    atlas_stub = types.SimpleNamespace(
        provider_manager=provider_manager,
        config_manager=config,
        update_provider_api_key=fake_update_provider_api_key,
        get_provider_api_key_status=fake_status,
    )

    window = MistralSettingsWindow(atlas_stub, config, None)
    window.api_key_entry.set_text("fallback-key")
    window.on_save_api_key_clicked(None)

    assert state["scheduled"] is True
    assert state["saved"] is True
    assert window._last_message[0] == "Success"
    assert window.api_key_entry.get_text() == ""


def test_mistral_settings_window_refresh_requires_api_key(
    provider_manager, monkeypatch
):
    monkeypatch.setattr("GTKUI.Utils.utils.apply_css", lambda *_, **__: None)
    monkeypatch.setattr("GTKUI.Utils.styled_window.apply_css", lambda *_, **__: None)
    monkeypatch.setattr(
        Gtk.Window,
        "get_style_context",
        lambda self: types.SimpleNamespace(add_class=lambda *_args, **_kwargs: None),
        raising=False,
    )

    config = provider_manager.config_manager

    state = {"scheduled": False}

    def fake_run_async_in_thread(*_args, **_kwargs):
        state["scheduled"] = True
        return types.SimpleNamespace(result=lambda: None)

    monkeypatch.setattr(Mistral_settings, "run_async_in_thread", fake_run_async_in_thread)

    atlas_stub = types.SimpleNamespace(
        provider_manager=provider_manager,
        config_manager=config,
        get_provider_api_key_status=lambda _name: {"has_key": False, "metadata": {}},
    )

    window = MistralSettingsWindow(atlas_stub, config, None)

    window._on_refresh_models_clicked(None)

    assert state["scheduled"] is False
    assert window._last_message[0] == "Error"
    assert "API key" in window._last_message[1]


def test_mistral_settings_window_refresh_updates_models(provider_manager, monkeypatch):
    monkeypatch.setattr("GTKUI.Utils.utils.apply_css", lambda *_, **__: None)
    monkeypatch.setattr("GTKUI.Utils.styled_window.apply_css", lambda *_, **__: None)
    monkeypatch.setattr(
        Gtk.Window,
        "get_style_context",
        lambda self: types.SimpleNamespace(add_class=lambda *_args, **_kwargs: None),
        raising=False,
    )

    config = provider_manager.config_manager
    config.update_api_key("Mistral", "mst-refresh")

    new_models = ["mistral-small-latest", "mistral-large-latest"]

    async def fake_fetch(self, *, base_url=None):
        state["base_url"] = base_url
        return {
            "success": True,
            "message": "models fetched",
            "data": {"models": list(new_models)},
        }

    provider_manager.fetch_mistral_models = fake_fetch.__get__(
        provider_manager, provider_manager.__class__
    )

    class _FakeFuture:
        def __init__(self, result=None):
            self._result = result

        def result(self):
            return self._result

    state = {"scheduled": False, "result": None, "base_url": None}

    def fake_run_async_in_thread(factory, *, on_success=None, on_error=None, **_kwargs):
        state["scheduled"] = True
        try:
            result = asyncio.run(factory())
        except Exception as exc:  # pragma: no cover - defensive logging
            if on_error is not None:
                on_error(exc)
            return _FakeFuture()
        else:
            state["result"] = result
            if on_success is not None:
                on_success(result)
            return _FakeFuture(result)

    monkeypatch.setattr(Mistral_settings, "run_async_in_thread", fake_run_async_in_thread)

    status_payload = {"has_key": True, "metadata": {"hint": "••••resh"}}

    atlas_stub = types.SimpleNamespace(
        provider_manager=provider_manager,
        config_manager=config,
        get_provider_api_key_status=lambda _name: dict(status_payload),
    )

    window = MistralSettingsWindow(atlas_stub, config, None)
    window.model_combo.set_active(0)

    window._on_refresh_models_clicked(None)

    assert state["scheduled"] is True
    assert window._last_message[0] == "Success"
    assert window._available_models == new_models
    assert window.model_refresh_button.sensitive is True


def test_mistral_settings_window_refresh_handles_error(provider_manager, monkeypatch):
    monkeypatch.setattr("GTKUI.Utils.utils.apply_css", lambda *_, **__: None)
    monkeypatch.setattr("GTKUI.Utils.styled_window.apply_css", lambda *_, **__: None)
    monkeypatch.setattr(
        Gtk.Window,
        "get_style_context",
        lambda self: types.SimpleNamespace(add_class=lambda *_args, **_kwargs: None),
        raising=False,
    )

    config = provider_manager.config_manager
    config.update_api_key("Mistral", "mst-error")

    async def fake_fetch(self, *, base_url=None):
        state["base_url"] = base_url
        return {"success": False, "error": "network issue"}

    provider_manager.fetch_mistral_models = fake_fetch.__get__(
        provider_manager, provider_manager.__class__
    )

    class _FakeFuture:
        def result(self):
            return None

    state = {"scheduled": False, "base_url": None}

    def fake_run_async_in_thread(factory, *, on_success=None, on_error=None, **_kwargs):
        state["scheduled"] = True
        try:
            result = asyncio.run(factory())
        except Exception as exc:  # pragma: no cover - defensive logging
            if on_error is not None:
                on_error(exc)
        else:
            if on_success is not None:
                on_success(result)
        return _FakeFuture()

    monkeypatch.setattr(Mistral_settings, "run_async_in_thread", fake_run_async_in_thread)

    atlas_stub = types.SimpleNamespace(
        provider_manager=provider_manager,
        config_manager=config,
        get_provider_api_key_status=lambda _name: {"has_key": True, "metadata": {}},
    )

    window = MistralSettingsWindow(atlas_stub, config, None)

    window._on_refresh_models_clicked(None)

    assert state["scheduled"] is True
    assert window._last_message[0] == "Error"
    assert "network issue" in window._last_message[1]
    assert window.model_refresh_button.sensitive is True


def test_anthropic_settings_window_saves_api_key(provider_manager):
    atlas_stub = types.SimpleNamespace()

    status_payload: Dict[str, object] = {"has_key": False, "metadata": {}}
    saved_call: Dict[str, object] = {}

    atlas_stub.get_anthropic_settings = lambda: {
        "model": "claude-3-opus-20240229",
        "stream": True,
        "function_calling": False,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": None,
        "max_output_tokens": None,
        "timeout": 60,
        "max_retries": 3,
        "retry_delay": 5,
        "stop_sequences": [],
    }
    atlas_stub.get_models_for_provider = lambda name: ["claude-3-opus-20240229"]

    def fake_update_in_background(provider, api_key, *, on_success=None, on_error=None):
        saved_call["provider"] = provider
        saved_call["api_key"] = api_key
        status_payload.update({"has_key": True, "metadata": {"hint": "••••5678"}})
        if on_success is not None:
            on_success({"success": True, "message": "API key saved."})

    atlas_stub.update_provider_api_key_in_background = fake_update_in_background
    atlas_stub.get_provider_api_key_status = lambda name: dict(status_payload)

    window = AnthropicSettingsWindow(atlas_stub, provider_manager.config_manager, None)

    assert window.api_key_status_label.label == "No API key saved for Anthropic."

    window.api_key_entry.set_text("test-key")
    window.on_save_api_key_clicked(None)

    assert saved_call == {"provider": "Anthropic", "api_key": "test-key"}
    assert window._last_message[0] == "Success"
    assert window.api_key_entry.get_text() == ""
    assert window.api_key_status_label.label.startswith("An API key is saved for Anthropic.")
    assert "••••5678" in window.api_key_status_label.label
    assert window.api_key_entry.placeholder == "Saved key: ••••5678"


def test_anthropic_settings_window_fallback_is_non_blocking(provider_manager, monkeypatch):
    atlas_stub = types.SimpleNamespace()

    status_payload: Dict[str, object] = {"has_key": False, "metadata": {}}

    atlas_stub.get_anthropic_settings = lambda: {
        "model": "claude-3-opus-20240229",
        "stream": True,
        "function_calling": False,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": None,
        "max_output_tokens": None,
        "timeout": 60,
        "max_retries": 3,
        "retry_delay": 5,
        "stop_sequences": [],
    }
    atlas_stub.get_models_for_provider = lambda name: ["claude-3-opus-20240229"]

    def fake_update_provider_api_key(provider, api_key):
        status_payload.update({"has_key": True, "metadata": {"hint": "••••4321"}})
        return {"success": True, "message": "saved"}

    atlas_stub.update_provider_api_key = fake_update_provider_api_key
    atlas_stub.get_provider_api_key_status = lambda name: dict(status_payload)

    window = AnthropicSettingsWindow(atlas_stub, provider_manager.config_manager, None)

    state = {"scheduled": False, "button_disabled_before_success": False}

    class _FakeFuture:
        def result(self):  # pragma: no cover - ensuring the old blocking call is gone
            raise AssertionError("future.result() should not be invoked")

    def fake_run_async_in_thread(factory, *, on_success=None, on_error=None, **_kwargs):
        state["scheduled"] = True
        state["button_disabled_before_success"] = not getattr(window.save_key_button, "sensitive", True)
        try:
            result = factory()
        except Exception as exc:  # pragma: no cover - defensive fallback
            if on_error is not None:
                on_error(exc)
        else:
            if on_success is not None:
                on_success(result)
        return _FakeFuture()

    monkeypatch.setattr(
        Anthropic_settings,
        "run_async_in_thread",
        fake_run_async_in_thread,
    )

    window.api_key_entry.set_text("fallback-key")
    window.on_save_api_key_clicked(None)

    assert state["scheduled"] is True
    assert state["button_disabled_before_success"] is True
    assert getattr(window.save_key_button, "sensitive", True) is True
    assert window._last_message[0] == "Success"
    assert window.api_key_entry.get_text() == ""


def test_openai_settings_window_falls_back_to_cached_models(provider_manager):
    atlas_stub = types.SimpleNamespace()

    atlas_stub.provider_manager = provider_manager
    atlas_stub.get_openai_llm_settings = lambda: {
        "model": "gpt-4o",
        "temperature": 0.0,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "max_tokens": 4000,
        "max_output_tokens": None,
        "stream": True,
        "base_url": None,
        "organization": None,
        "reasoning_effort": "medium",
        "json_schema": None,
    }
    atlas_stub.set_openai_llm_settings = lambda **_: {"success": True, "message": "saved"}

    async def failing_list_models(*, base_url=None, organization=None):
        raise RuntimeError("network down")

    def fake_run_in_background(factory, *, on_success=None, on_error=None, **_kwargs):
        try:
            asyncio.run(factory())
        except Exception as exc:  # pragma: no cover - defensive logging
            if on_error is not None:
                on_error(exc)
        else:  # pragma: no cover - defensive fallback
            if on_success is not None:
                on_success(None)
        return types.SimpleNamespace()

    atlas_stub.list_openai_models = failing_list_models
    atlas_stub.run_in_background = fake_run_in_background
    atlas_stub.get_provider_api_key_status = lambda _name: {"has_key": False, "metadata": {}}

    window = OpenAISettingsWindow(atlas_stub, provider_manager.config_manager, None)

    assert window.model_combo.get_active_text() == "gpt-4o"
    assert "gpt-4.1" in window._available_models
    assert "o1" in window._available_models
    assert window._last_message[0] == "Model Load Failed"


def test_provider_manager_set_anthropic_settings_updates_generator(provider_manager, monkeypatch):
    captured: Dict[str, object] = {}

    class _StubGenerator:
        def __init__(self):
            self.default_model = "claude-3-opus-20240229"

        async def generate_response(self, **_kwargs):  # pragma: no cover - not exercised
            return "ok"

        async def process_streaming_response(self, response):  # pragma: no cover - defensive
            chunks = []
            async for part in response:
                chunks.append(part)
            return "".join(chunks)

        def set_default_model(self, value):
            captured["model"] = value
            self.default_model = value

        def set_streaming(self, value):
            captured["stream"] = value

        def set_function_calling(self, value):
            captured["function_calling"] = value

        def set_temperature(self, value):
            captured["temperature"] = value

        def set_top_p(self, value):
            captured["top_p"] = value

        def set_top_k(self, value):
            captured["top_k"] = value

        def set_max_output_tokens(self, value):
            captured["max_output_tokens"] = value

        def set_timeout(self, value):
            captured["timeout"] = value

        def set_max_retries(self, value):
            captured["max_retries"] = value

        def set_retry_delay(self, value):
            captured["retry_delay"] = value

        def set_stop_sequences(self, value):
            captured["stop_sequences"] = value

        def set_tool_choice(self, choice, name=None):
            captured["tool_choice"] = (choice, name)

        def set_metadata(self, value):
            captured["metadata"] = value

        def set_thinking(self, enabled, budget=None):
            captured["thinking"] = (enabled, budget)

    stub = _StubGenerator()
    monkeypatch.setattr(provider_manager, "_ensure_anthropic_generator", lambda: stub)

    provider_manager.current_llm_provider = "Anthropic"

    result = provider_manager.set_anthropic_settings(
        model="claude-3-haiku-20240229",
        stream=False,
        function_calling=True,
        temperature=0.4,
        top_p=0.88,
        top_k=55,
        max_output_tokens=2048,
        timeout=90,
        max_retries=4,
        retry_delay=12,
        stop_sequences=["END", "STOP"],
        tool_choice="tool",
        tool_choice_name="calendar_lookup",
        metadata={"team": "qa"},
        thinking=True,
        thinking_budget=4096,
    )

    assert result["success"] is True
    data = result.get("data", {})
    assert data["model"] == "claude-3-haiku-20240229"
    assert math.isclose(data["temperature"], 0.4)
    assert math.isclose(data["top_p"], 0.88)
    assert data["top_k"] == 55
    assert data["max_output_tokens"] == 2048
    assert data["stop_sequences"] == ["END", "STOP"]
    assert captured["model"] == "claude-3-haiku-20240229"
    assert captured["stream"] is False
    assert captured["function_calling"] is True
    assert math.isclose(captured["temperature"], 0.4)
    assert math.isclose(captured["top_p"], 0.88)
    assert captured["top_k"] == 55
    assert captured["max_output_tokens"] == 2048
    assert captured["timeout"] == 90
    assert captured["max_retries"] == 4
    assert captured["retry_delay"] == 12
    assert captured["stop_sequences"] == ["END", "STOP"]
    assert captured["tool_choice"] == ("tool", "calendar_lookup")
    assert captured["metadata"] == {"team": "qa"}
    assert captured["thinking"] == (True, 4096)

    assert data["tool_choice"] == "tool"
    assert data["tool_choice_name"] == "calendar_lookup"
    assert data["metadata"] == {"team": "qa"}
    assert data["thinking"] is True
    assert data["thinking_budget"] == 4096
    assert provider_manager.current_model == "claude-3-haiku-20240229"
    assert provider_manager.model_manager.models["Anthropic"][0] == "claude-3-haiku-20240229"


def test_create_resets_singleton_after_initialization_failure(monkeypatch, tmp_path):
    reset_provider_manager_singleton()
    config = DummyConfig(tmp_path.as_posix())

    attempts: Dict[str, int] = {"count": 0}
    closed: List[Tuple[int, Optional[object]]] = []
    first_instance_id: Dict[str, int] = {}
    sentinel_by_instance: Dict[int, object] = {}

    async def failing_initialize(self):
        attempts["count"] += 1
        sentinel = object()
        instance_id = id(self)
        sentinel_by_instance[instance_id] = sentinel
        self._openai_generator = sentinel  # type: ignore[attr-defined]
        if attempts["count"] == 1:
            first_instance_id["value"] = instance_id
            raise RuntimeError("forced initialization failure")
        self.initialized_marker = "ok"  # type: ignore[attr-defined]

    async def recording_close(self):
        closed.append((id(self), getattr(self, "_openai_generator", None)))
        self._openai_generator = None  # type: ignore[attr-defined]

    monkeypatch.setattr(
        provider_manager_module.ProviderManager,
        "initialize_all_providers",
        failing_initialize,
        raising=False,
    )
    monkeypatch.setattr(
        provider_manager_module.ProviderManager,
        "close",
        recording_close,
        raising=False,
    )

    with pytest.raises(RuntimeError):
        asyncio.run(ProviderManager.create(config))

    assert attempts["count"] == 1
    assert provider_manager_module.ProviderManager._instance is None
    assert len(closed) == 1
    assert closed[0][0] == first_instance_id["value"]
    assert closed[0][1] is sentinel_by_instance[first_instance_id["value"]]

    manager = asyncio.run(ProviderManager.create(config))
    try:
        assert attempts["count"] == 2
        assert getattr(manager, "initialized_marker", None) == "ok"
        assert id(manager) != first_instance_id["value"]
    finally:
        reset_provider_manager_singleton()


def test_switch_openai_reuses_cached_generator(monkeypatch, tmp_path):
    from modules.Providers.OpenAI import OA_gen_response as openai_module

    instantiate_count = 0

    class StubGenerator:
        def __init__(self, config_manager):
            nonlocal instantiate_count
            instantiate_count += 1
            self.config_manager = config_manager

        async def generate_response(self, messages=None, **_kwargs):
            return {"messages": messages or []}

        async def process_streaming_response(self, *_args, **_kwargs):
            return "stream"

    openai_module._GENERATOR_CACHE = WeakKeyDictionary()
    monkeypatch.setattr(openai_module, "OpenAIGenerator", StubGenerator, raising=False)

    async def fake_list_models(self, *_args, **_kwargs):
        return {"models": []}

    monkeypatch.setattr(
        provider_manager_module.ProviderManager,
        "list_openai_models",
        fake_list_models,
        raising=False,
    )

    def fake_load_models(self):
        self.models = {
            "OpenAI": ["gpt-4o"],
            "HuggingFace": [],
            "Mistral": ["mistral-large-latest"],
            "Google": ["gemini-1.5-pro-latest"],
            "Anthropic": ["claude-3-opus-20240229"],
            "Grok": [],
        }
        self.current_model = None
        self.current_provider = None

    monkeypatch.setattr(
        provider_manager_module.ModelManager,
        "load_models",
        fake_load_models,
        raising=False,
    )

    reset_provider_manager_singleton()
    config = DummyConfig(tmp_path.as_posix())
    config.update_api_key("OpenAI", "sk-test")

    async def exercise():
        manager = await ProviderManager.create(config)
        try:
            messages = [{"role": "user", "content": "hello"}]
            await manager.generate_response(messages, provider="OpenAI")
            await manager.generate_response(messages, provider="OpenAI")
        finally:
            reset_provider_manager_singleton()

    asyncio.run(exercise())
    assert instantiate_count == 1


def test_switch_mistral_reuses_cached_generator(monkeypatch, tmp_path):
    from modules.Providers.Mistral import Mistral_gen_response as mistral_module

    instantiate_count = 0

    class StubGenerator:
        def __init__(self, config_manager):
            nonlocal instantiate_count
            instantiate_count += 1
            self.config_manager = config_manager

        async def generate_response(self, messages=None, **_kwargs):
            return {"messages": messages or []}

        async def process_response(self, response):
            if isinstance(response, str):
                return response
            collected = []
            async for chunk in response:
                collected.append(chunk)
            return "".join(collected)

    mistral_module._GENERATOR_CACHE = WeakKeyDictionary()
    monkeypatch.setattr(mistral_module, "MistralGenerator", StubGenerator, raising=False)

    async def fake_list_models(self, *_args, **_kwargs):
        return {"models": []}

    monkeypatch.setattr(
        provider_manager_module.ProviderManager,
        "list_openai_models",
        fake_list_models,
        raising=False,
    )

    def fake_load_models(self):
        self.models = {
            "OpenAI": ["gpt-4o"],
            "HuggingFace": [],
            "Mistral": ["mistral-large-latest"],
            "Google": ["gemini-1.5-pro-latest"],
            "Anthropic": ["claude-3-opus-20240229"],
            "Grok": [],
        }
        self.current_model = None
        self.current_provider = None

    monkeypatch.setattr(
        provider_manager_module.ModelManager,
        "load_models",
        fake_load_models,
        raising=False,
    )

    reset_provider_manager_singleton()
    config = DummyConfig(tmp_path.as_posix())
    config.update_api_key("OpenAI", "sk-test")

    async def exercise():
        manager = await ProviderManager.create(config)
        try:
            await manager.switch_llm_provider("Mistral")
            messages = [{"role": "user", "content": "hello"}]
            await manager.generate_response(messages, provider="Mistral")
            await manager.generate_response(messages, provider="Mistral")
        finally:
            reset_provider_manager_singleton()

    asyncio.run(exercise())
    assert instantiate_count == 1


def test_switch_google_reuses_cached_generator(monkeypatch, tmp_path):
    from modules.Providers.Google import GG_gen_response as google_module

    instantiate_count = 0

    class StubGenerator:
        def __init__(self, config_manager):
            nonlocal instantiate_count
            instantiate_count += 1
            self.config_manager = config_manager

        async def generate_response(self, messages=None, **_kwargs):
            return {"messages": messages or []}

        async def process_response(self, response):
            if isinstance(response, str):
                return response
            collected = []
            async for chunk in response:
                collected.append(chunk)
            return "".join(collected)

    google_module._GENERATOR_CACHE = WeakKeyDictionary()
    monkeypatch.setattr(google_module, "GoogleGeminiGenerator", StubGenerator, raising=False)

    async def fake_list_models(self, *_args, **_kwargs):
        return {"models": []}

    monkeypatch.setattr(
        provider_manager_module.ProviderManager,
        "list_openai_models",
        fake_list_models,
        raising=False,
    )

    def fake_load_models(self):
        self.models = {
            "OpenAI": ["gpt-4o"],
            "HuggingFace": [],
            "Mistral": ["mistral-large-latest"],
            "Google": ["gemini-1.5-pro-latest"],
            "Anthropic": ["claude-3-opus-20240229"],
            "Grok": [],
        }
        self.current_model = None
        self.current_provider = None

    monkeypatch.setattr(
        provider_manager_module.ModelManager,
        "load_models",
        fake_load_models,
        raising=False,
    )

    reset_provider_manager_singleton()
    config = DummyConfig(tmp_path.as_posix())
    config.update_api_key("OpenAI", "sk-test")

    async def exercise():
        manager = await ProviderManager.create(config)
        try:
            await manager.switch_llm_provider("Google")
            messages = [{"role": "user", "content": "hello"}]
            await manager.generate_response(messages, provider="Google")
            await manager.generate_response(messages, provider="Google")
        finally:
            reset_provider_manager_singleton()

    asyncio.run(exercise())
    assert instantiate_count == 1
