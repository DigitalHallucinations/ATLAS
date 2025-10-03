import asyncio
import json
import math
import sys
import types
from pathlib import Path
from typing import Dict, Optional, Tuple
from urllib.error import URLError

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

    def get_default_provider(self):
        return "OpenAI"

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

    ProviderManager._instance = None


def test_openai_default_model_uses_cached_list(provider_manager):
    default = provider_manager.get_default_model_for_provider("OpenAI")

    assert default == "gpt-4o"
    assert "gpt-4.1" in provider_manager.model_manager.models["OpenAI"]
    assert "o1" in provider_manager.model_manager.models["OpenAI"]


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
    assert result["models"] == ["chat-awesome", "gpt-4o"]
    assert result["base_url"].endswith("/v1")
    cached = provider_manager.model_manager.models["OpenAI"]
    assert cached[0] == "gpt-4o"
    assert "chat-awesome" in cached


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
            self.models = types.SimpleNamespace(list=lambda: _StubModelPage(models))

        def close(self):
            self.closed = True

    async def fake_to_thread(callable_, *args, **kwargs):  # pragma: no cover - test helper
        return callable_()

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

    persisted = data.get("persisted_to")
    assert persisted
    path = Path(persisted)
    assert path.exists()
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["models"][0] == "mistral-large-latest"


def test_fetch_mistral_models_handles_exception(provider_manager, monkeypatch):
    provider_manager.config_manager.update_api_key("Mistral", "mst-test")

    class _FailingClient:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("boom")

    monkeypatch.setattr(provider_manager_module, "Mistral", _FailingClient)

    result = asyncio.run(provider_manager.fetch_mistral_models())

    assert result["success"] is False
    assert "boom" in (result.get("error") or "")


def test_provider_manager_primes_openai_models_on_startup(tmp_path, monkeypatch):
    ProviderManager._instance = None

    def fake_load_models(self):
        self.models = {
            "OpenAI": ["gpt-4o"],
            "HuggingFace": [],
            "Mistral": [],
            "Google": [],
            "Anthropic": [],
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
        ProviderManager._instance = None


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
    monkeypatch.setattr("GTKUI.Utils.utils.apply_css", lambda: None)
    monkeypatch.setattr(
        "GTKUI.Provider_manager.Settings.Mistral_settings.apply_css", lambda: None
    )
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
    assert math.isclose(window.temperature_spin.get_value(), 0.4)
    assert math.isclose(window.top_p_spin.get_value(), 0.8)
    assert math.isclose(window.frequency_penalty_spin.get_value(), 0.15)
    assert math.isclose(window.presence_penalty_spin.get_value(), -0.3)
    assert window.max_tokens_spin.get_value_as_int() == 2048
    assert window.safe_prompt_toggle.get_active() is True
    assert window.stream_toggle.get_active() is False
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
    assert window.tool_call_toggle.get_active() is True
    assert window.parallel_tool_calls_toggle.get_active() is True
    assert window.require_tool_toggle.get_active() is False
    assert window.random_seed_entry.get_text() == ""
    assert window.tool_choice_entry.get_text() == ""
    assert window.stop_sequences_entry.get_text() == ""
    assert window.json_mode_toggle.get_active() is False
    assert window._json_schema_text_cache == ""

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
    assert stored["random_seed"] == 0
    assert stored["tool_choice"] == "none"
    assert stored["stop_sequences"] == ["END", "FINISH"]
    assert stored["json_mode"] is True
    assert stored["json_schema"]["schema"]["properties"]["result"]["type"] == "string"
    assert window._last_message[0] == "Success"
    assert window.model_combo.get_active_text() == "mistral-small-latest"

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

    window.max_tokens_spin.set_value(0)

    window.on_save_clicked(None)

    stored = config.get_mistral_llm_settings()
    assert stored["max_tokens"] is None
    assert stored["model"] == "mistral-experimental"
    assert window.max_tokens_spin.get_value_as_int() == 0
    assert window._custom_entry_visible is True


def test_mistral_settings_window_saves_api_key(provider_manager, monkeypatch):
    monkeypatch.setattr("GTKUI.Utils.utils.apply_css", lambda: None)
    monkeypatch.setattr(
        "GTKUI.Provider_manager.Settings.Mistral_settings.apply_css", lambda: None
    )
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
    monkeypatch.setattr("GTKUI.Utils.utils.apply_css", lambda: None)
    monkeypatch.setattr(
        "GTKUI.Provider_manager.Settings.Mistral_settings.apply_css", lambda: None
    )
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
    monkeypatch.setattr("GTKUI.Utils.utils.apply_css", lambda: None)
    monkeypatch.setattr(
        "GTKUI.Provider_manager.Settings.Mistral_settings.apply_css", lambda: None
    )
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
    monkeypatch.setattr("GTKUI.Utils.utils.apply_css", lambda: None)
    monkeypatch.setattr(
        "GTKUI.Provider_manager.Settings.Mistral_settings.apply_css", lambda: None
    )
    monkeypatch.setattr(
        Gtk.Window,
        "get_style_context",
        lambda self: types.SimpleNamespace(add_class=lambda *_args, **_kwargs: None),
        raising=False,
    )

    config = provider_manager.config_manager
    config.update_api_key("Mistral", "mst-refresh")

    new_models = ["mistral-small-latest", "mistral-large-latest"]

    async def fake_fetch(self):
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

    state = {"scheduled": False, "result": None}

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
    monkeypatch.setattr("GTKUI.Utils.utils.apply_css", lambda: None)
    monkeypatch.setattr(
        "GTKUI.Provider_manager.Settings.Mistral_settings.apply_css", lambda: None
    )
    monkeypatch.setattr(
        Gtk.Window,
        "get_style_context",
        lambda self: types.SimpleNamespace(add_class=lambda *_args, **_kwargs: None),
        raising=False,
    )

    config = provider_manager.config_manager
    config.update_api_key("Mistral", "mst-error")

    async def fake_fetch(self):
        return {"success": False, "error": "network issue"}

    provider_manager.fetch_mistral_models = fake_fetch.__get__(
        provider_manager, provider_manager.__class__
    )

    class _FakeFuture:
        def result(self):
            return None

    state = {"scheduled": False}

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
