import sys
import types
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


def _ensure_gtk_stubs():
    if "gi" in sys.modules:
        return

    gi_module = types.ModuleType("gi")

    def require_version(*_args, **_kwargs):
        return None

    gi_module.require_version = require_version

    repository_module = types.ModuleType("gi.repository")
    gtk_module = types.ModuleType("Gtk")
    gdk_module = types.ModuleType("Gdk")
    glib_module = types.ModuleType("GLib")

    class _DummyWindow:
        def __init__(self, *args, **kwargs):
            pass

    gtk_module.Window = _DummyWindow
    gtk_module.Box = _DummyWindow
    gtk_module.Notebook = _DummyWindow
    gtk_module.Switch = _DummyWindow
    gtk_module.Label = _DummyWindow
    gtk_module.ComboBoxText = _DummyWindow
    gtk_module.Entry = _DummyWindow
    gtk_module.Button = _DummyWindow
    gtk_module.ScrolledWindow = _DummyWindow
    gtk_module.Frame = _DummyWindow
    gtk_module.ToggleButton = _DummyWindow
    gtk_module.Picture = _DummyWindow
    gtk_module.Image = _DummyWindow
    gtk_module.MessageDialog = _DummyWindow
    gtk_module.FileChooserNative = _DummyWindow
    gtk_module.Widget = _DummyWindow

    gtk_module.MessageType = types.SimpleNamespace(INFO=0, ERROR=1, WARNING=2)
    gtk_module.ButtonsType = types.SimpleNamespace(OK=0, NONE=1)
    gtk_module.ResponseType = types.SimpleNamespace(OK=0, CANCEL=1, ACCEPT=2)
    gtk_module.FileChooserAction = types.SimpleNamespace(OPEN=0)
    gtk_module.PolicyType = types.SimpleNamespace(NEVER=0, AUTOMATIC=1)
    gtk_module.Orientation = types.SimpleNamespace(VERTICAL=0, HORIZONTAL=1)
    gtk_module.ContentFit = types.SimpleNamespace(CONTAIN=0)
    gtk_module.AccessibleRole = types.SimpleNamespace(BUTTON=0)

    repository_module.Gtk = gtk_module
    repository_module.Gdk = gdk_module
    repository_module.GLib = glib_module

    sys.modules["gi"] = gi_module
    sys.modules["gi.repository"] = repository_module
    sys.modules["gi.repository.Gtk"] = gtk_module
    sys.modules["gi.repository.Gdk"] = gdk_module
    sys.modules["gi.repository.GLib"] = glib_module


_ensure_gtk_stubs()

from GTKUI.Settings.Speech.speech_settings import SpeechSettings


class _FakeSwitch:
    def __init__(self, active):
        self._active = active

    def get_active(self):
        return self._active


class _FakeCombo:
    def __init__(self, value):
        self._value = value
        self.appended = []
        self.active_index = 0

    def get_active_text(self):
        return self._value

    def remove_all(self):
        self.appended.clear()

    def append_text(self, value):
        self.appended.append(value)

    def set_active(self, index):
        self.active_index = index


class _FakeEntry:
    def __init__(self, text=""):
        self._text = text

    def get_text(self):
        return self._text

    def set_text(self, value):  # pragma: no cover - helper for completeness
        self._text = value


@pytest.fixture
def speech_settings():
    settings = SpeechSettings.__new__(SpeechSettings)
    settings.tab_dirty = {}
    settings._apply_provider_status_to_entry = Mock()
    settings._show_message = Mock()
    return settings


def test_save_general_tab_uses_atlas_facade(speech_settings):
    atlas = SimpleNamespace(update_speech_defaults=Mock())
    speech_settings.ATLAS = atlas
    speech_settings.general_tts_switch = _FakeSwitch(True)
    speech_settings.general_stt_switch = _FakeSwitch(False)
    speech_settings.default_tts_combo = _FakeCombo("eleven_labs")
    speech_settings.default_stt_combo = _FakeCombo("google")
    speech_settings.tab_dirty[0] = True

    speech_settings.save_general_tab()

    atlas.update_speech_defaults.assert_called_once_with(
        tts_enabled=True,
        tts_provider="eleven_labs",
        stt_enabled=False,
        stt_provider="google",
    )


def test_save_eleven_labs_tab_uses_facade(speech_settings):
    voice = {"name": "Alpha", "voice_id": "alpha"}
    atlas = SimpleNamespace(
        update_elevenlabs_settings=Mock(return_value={
            "updated_api_key": True,
            "updated_voice": True,
            "provider": "eleven_labs",
        }),
        get_speech_voice_options=Mock(return_value=[voice]),
    )

    speech_settings.ATLAS = atlas
    speech_settings.eleven_api_entry = _FakeEntry("key")
    speech_settings.voice_combo = _FakeCombo("Alpha")
    speech_settings._voice_lookup = {"Alpha": voice}
    speech_settings._get_provider_key_status = Mock(return_value={"has_key": False})
    speech_settings.tab_dirty[1] = True

    speech_settings.save_eleven_labs_tab()

    atlas.update_elevenlabs_settings.assert_called_once_with(api_key="key", voice_id="alpha")


def test_save_openai_tab_uses_facade(speech_settings):
    atlas = SimpleNamespace(
        update_openai_speech_settings=Mock(
            return_value={"stt_provider": "Whisper Online", "tts_provider": "GPT"}
        ),
        get_speech_provider_status=Mock(return_value={"has_key": True, "metadata": {}}),
    )

    speech_settings.ATLAS = atlas
    speech_settings.openai_api_entry = _FakeEntry("key")
    speech_settings.openai_stt_combo = _FakeCombo("Whisper Online")
    speech_settings.openai_language_combo = _FakeCombo("English (en)")
    speech_settings.openai_task_combo = _FakeCombo("Transcribe")
    speech_settings.openai_prompt_entry = _FakeEntry("Prompt")
    speech_settings.openai_tts_combo = _FakeCombo("GPT-4o Mini TTS")
    speech_settings.tab_dirty[3] = True

    speech_settings.save_openai_tab()

    atlas.update_openai_speech_settings.assert_called_once()


def test_save_google_tab_uses_facade(speech_settings):
    atlas = SimpleNamespace(update_google_speech_settings=Mock())

    speech_settings.ATLAS = atlas
    speech_settings.google_credentials_entry = _FakeEntry(" /tmp/google.json ")
    speech_settings._google_voice_lookup = {"Preferred": {"name": "Preferred"}}
    speech_settings.google_voice_combo = _FakeCombo("Preferred")
    speech_settings._google_language_lookup = {"Auto detect": None, "en-US": "en-US"}
    speech_settings.google_language_combo = _FakeCombo("en-US")
    speech_settings.google_autopunct_switch = _FakeSwitch(True)
    speech_settings.tab_dirty[2] = True

    speech_settings.save_google_tab()

    atlas.update_google_speech_settings.assert_called_once_with(
        "/tmp/google.json",
        tts_voice={"name": "Preferred"},
        stt_language="en-US",
        auto_punctuation=True,
    )
