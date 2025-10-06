import inspect
import sys
import types
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import tests.test_speech_settings_facade  # noqa: F401 - ensure baseline GTK stubs
import tests.test_provider_manager  # noqa: F401 - ensure extended GTK stubs


gi_repository = sys.modules.setdefault("gi.repository", types.ModuleType("gi.repository"))


class _DummyWidget:
    def __init__(self, *args, **kwargs):  # pragma: no cover - helper baseline
        self.children = []
        self._tooltip = None
        self._hexpand = False
        self._vexpand = False
        self._halign = None
        self._valign = None
        self._sensitive = True
        self.visible = True
        self._css_classes: set[str] = set()

    def __call__(self, *args, **kwargs):  # pragma: no cover - convenience hook
        return self

    def __getattr__(self, _name):  # pragma: no cover - benign callable fallback
        return lambda *_a, **_kw: None

    def connect(self, *_args, **_kwargs):  # pragma: no cover - event helper
        return None

    def add_css_class(self, name, *_args, **_kwargs):  # pragma: no cover - styling helper
        self._css_classes.add(name)
        return None

    def remove_css_class(self, name, *_args, **_kwargs):  # pragma: no cover - styling helper
        self._css_classes.discard(name)
        return None

    def has_css_class(self, name):  # pragma: no cover - styling helper
        return name in self._css_classes

    def get_style_context(self):  # pragma: no cover - styling helper
        return self

    def set_tooltip_text(self, *_args, **_kwargs):  # pragma: no cover - tooltip helper
        self._tooltip = _args[0] if _args else None

    def set_child(self, *_args, **_kwargs):  # pragma: no cover - composite helper
        if _args:
            self.children = [_args[0]]

    def append(self, *_args, **_kwargs):  # pragma: no cover - container helper
        if _args:
            self.children.append(_args[0])

    def set_sensitive(self, value, *_args, **_kwargs):  # pragma: no cover - widget helper
        self._sensitive = bool(value)
        return None

    def set_visible(self, value=True, *_args, **_kwargs):  # pragma: no cover - widget helper
        self.visible = bool(value)
        return None

    def start(self, *_args, **_kwargs):  # pragma: no cover - spinner helper
        return None

    def stop(self, *_args, **_kwargs):  # pragma: no cover - spinner helper
        return None

    def grab_focus(self, *_args, **_kwargs):  # pragma: no cover - widget helper
        return None

    def set_hexpand(self, value):  # pragma: no cover - widget helper
        self._hexpand = value

    def set_vexpand(self, value):  # pragma: no cover - widget helper
        self._vexpand = value

    def set_halign(self, value):  # pragma: no cover - widget helper
        self._halign = value

    def set_valign(self, value):  # pragma: no cover - widget helper
        self._valign = value


Gtk = sys.modules.get("gi.repository.Gtk")
if Gtk is None:
    Gtk = types.ModuleType("Gtk")
    sys.modules["gi.repository.Gtk"] = Gtk
    gi_repository.Gtk = Gtk

Gdk = sys.modules.get("gi.repository.Gdk")
if Gdk is None:
    Gdk = types.ModuleType("Gdk")
    sys.modules["gi.repository.Gdk"] = Gdk
    gi_repository.Gdk = Gdk

GLib = sys.modules.get("gi.repository.GLib")
if GLib is None:
    GLib = types.ModuleType("GLib")
    sys.modules["gi.repository.GLib"] = GLib
    gi_repository.GLib = GLib

Gio = sys.modules.get("gi.repository.Gio")
if Gio is None:
    Gio = types.ModuleType("Gio")
    sys.modules["gi.repository.Gio"] = Gio
    gi_repository.Gio = Gio


def _ensure_widget(module: types.ModuleType, name: str) -> None:
    if not hasattr(module, name):
        setattr(module, name, type(name, (_DummyWidget,), {}))


for widget_name in [
    "Window",
    "HeaderBar",
    "Label",
    "Button",
    "Box",
    "ScrolledWindow",
    "Grid",
    "ListBox",
    "TextBuffer",
    "TextView",
    "EventControllerKey",
    "Spinner",
    "Picture",
    "Image",
    "GestureClick",
    "PopoverMenu",
    "FileChooserNative",
    "SimpleAction",
    "SimpleActionGroup",
    "ComboBoxText",
    "Adjustment",
    "SpinButton",
    "CheckButton",
]:
    _ensure_widget(Gtk, widget_name)


class _ComboBoxText(_DummyWidget):
    def __init__(self):
        super().__init__()
        self._items: list[str] = []
        self._active = -1

    def append_text(self, text: str):
        self._items.append(text)

    def remove_all(self):
        self._items.clear()
        self._active = -1

    def set_active(self, index: int):
        if 0 <= index < len(self._items):
            self._active = index

    def get_active_text(self):
        if 0 <= self._active < len(self._items):
            return self._items[self._active]
        return None


class _Adjustment:
    def __init__(self, value=0.0, lower=0.0, upper=1.0, step_increment=0.1, page_increment=0.1):
        self.value = value
        self.lower = lower
        self.upper = upper
        self.step_increment = step_increment
        self.page_increment = page_increment


class _SpinButton(_DummyWidget):
    def __init__(self, adjustment=None, digits: int = 0):
        super().__init__()
        self.adjustment = adjustment or _Adjustment()
        self.digits = digits
        self.value = self.adjustment.value

    def set_increments(self, step: float, page: float):
        self.adjustment.step_increment = step
        self.adjustment.page_increment = page

    def set_value(self, value: float):
        self.value = value

    def get_value(self):
        return self.value

    def get_value_as_int(self) -> int:
        return int(self.value)


class _CheckButton(_DummyWidget):
    def __init__(self, label: str = ""):
        super().__init__()
        self.label = label
        self.active = False

    def set_active(self, value: bool):
        self.active = bool(value)

    def get_active(self) -> bool:
        return self.active


class _Entry(_DummyWidget):
    def __init__(self):
        super().__init__()
        self._text = ""
        self.placeholder = ""
        self.visible = True
        self.invisible_char = "*"

    def set_text(self, text: str):
        self._text = text

    def get_text(self) -> str:
        return self._text

    def set_placeholder_text(self, text: str):
        self.placeholder = text

    def set_visibility(self, visible: bool):
        self.visible = bool(visible)

    def set_invisible_char(self, char: str):
        self.invisible_char = char


class _Label(_DummyWidget):
    def __init__(self, label: str = "", **_kwargs):
        super().__init__()
        self.label = label
        self.xalign = 0.0

    def set_xalign(self, value: float):
        self.xalign = value

    def set_label(self, value: str):
        self.label = value

    def set_text(self, value: str):
        self.label = value

    def get_text(self) -> str:
        return getattr(self, "label", "")


class _Button(_DummyWidget):
    def __init__(self, label: str = ""):
        super().__init__()
        self.label = label
        self._callbacks: list[tuple[str, callable]] = []

    def set_label(self, label: str):
        self.label = label

    def connect(self, signal: str, callback, *args):
        self._callbacks.append((signal, callback))


class _Box(_DummyWidget):
    def __init__(self, orientation=None, spacing: int = 0):
        super().__init__()
        self.orientation = orientation
        self.spacing = spacing


class _Grid(_DummyWidget):
    def __init__(self, column_spacing: int = 0, row_spacing: int = 0):
        super().__init__()
        self.column_spacing = column_spacing
        self.row_spacing = row_spacing
        self.attachments: list[tuple[object, int, int, int, int]] = []

    def attach(self, child, column: int, row: int, width: int, height: int):
        self.attachments.append((child, column, row, width, height))
        self.children.append(child)


class _ScrolledWindow(_DummyWidget):
    def __init__(self):
        super().__init__()
        self.policy = (None, None)

    def set_policy(self, horizontal, vertical):
        self.policy = (horizontal, vertical)


class _ListBox(_DummyWidget):
    def __init__(self):
        super().__init__()
        self.children = []

    def append(self, child):
        self.children.append(child)

    def remove(self, child):
        if child in self.children:
            self.children.remove(child)

    def remove_all(self):
        self.children.clear()

    def get_children(self):
        return list(self.children)


class _AlertDialog(_DummyWidget):
    _next_future = None

    def __init__(self, title: str = "", body: str = ""):
        super().__init__()
        self.title = title
        self.body = body
        self.buttons: list[str] = []
        self.response = None

    def set_buttons(self, buttons):
        self.buttons = list(buttons)

    def choose(self, _parent):
        future = getattr(self, "future_response", None)
        if future is not None:
            return future
        future = getattr(type(self), "_next_future", None)
        if future is not None:
            type(self)._next_future = None
            return future
        if self.response is not None:
            return self.response
        if self.buttons:
            return self.buttons[-1]
        return None


class _AlertDialogFuture:
    def __init__(self, result=None):
        self._result = result
        self.wait_calls = 0
        self.wait_result_calls = 0
        self._wait_hook = None
        self._wait_result_hook = None

    def set_result(self, result):
        self._result = result

    def set_wait_hook(self, hook):
        self._wait_hook = hook

    def set_wait_result_hook(self, hook):
        self._wait_result_hook = hook

    def wait(self):
        self.wait_calls += 1
        if callable(self._wait_hook):
            return self._wait_hook()
        return self._result

    def wait_result(self):
        self.wait_result_calls += 1
        if callable(self._wait_result_hook):
            return self._wait_result_hook()
        return self._result


Gtk.ComboBoxText = _ComboBoxText
Gtk.Adjustment = _Adjustment
Gtk.SpinButton = _SpinButton
Gtk.CheckButton = _CheckButton
Gtk.Entry = _Entry
Gtk.Label = _Label
Gtk.Button = _Button
Gtk.Box = _Box
Gtk.Grid = _Grid
Gtk.ScrolledWindow = _ScrolledWindow
Gtk.ListBox = _ListBox
Gtk.AlertDialog = _AlertDialog

if not hasattr(Gio, "Future"):
    Gio.Future = _AlertDialogFuture


def make_alert_dialog_future(result=None):
    future = _AlertDialogFuture(result)
    return future

if hasattr(Gtk, "Window"):
    for method_name in ["set_modal", "set_transient_for", "set_default_size"]:
        if not hasattr(Gtk.Window, method_name):
            setattr(Gtk.Window, method_name, lambda self, *args, **kwargs: None)
    if not hasattr(Gtk.Window, "close"):
        def _close(self):
            self.closed = True

        Gtk.Window.close = _close
    if "__init__" in Gtk.Window.__dict__:
        original_init = Gtk.Window.__init__

        def _init(self, *args, **kwargs):  # pragma: no cover - ensure closed flag exists
            original_init(self, *args, **kwargs)
            self.closed = False

        Gtk.Window.__init__ = _init
    else:
        def _default_init(self, *args, **kwargs):  # pragma: no cover - ensure closed flag exists
            _DummyWidget.__init__(self, *args, **kwargs)
            self.closed = False

        Gtk.Window.__init__ = _default_init


def _ensure_getattr(cls):
    if cls is None:
        return
    if "__getattr__" not in cls.__dict__:
        cls.__getattr__ = lambda self, _name: (lambda *_a, **_kw: None)


for name in [
    "Window",
    "ScrolledWindow",
    "Box",
    "Grid",
    "Entry",
    "Label",
    "Button",
    "ComboBoxText",
    "Adjustment",
    "SpinButton",
    "CheckButton",
]:
    _ensure_getattr(getattr(Gtk, name, None))

for enum_name, default in [
    ("Align", types.SimpleNamespace(START=0, END=1, CENTER=2)),
    ("Orientation", types.SimpleNamespace(VERTICAL=0, HORIZONTAL=1)),
    ("PolicyType", types.SimpleNamespace(NEVER=0, AUTOMATIC=1)),
    ("WrapMode", types.SimpleNamespace(WORD_CHAR=0)),
    ("ContentFit", types.SimpleNamespace(CONTAIN=0)),
    ("FileChooserAction", types.SimpleNamespace(SAVE=0)),
    ("ResponseType", types.SimpleNamespace(ACCEPT=0)),
]:
    if not hasattr(Gtk, enum_name):
        setattr(Gtk, enum_name, default)

if not hasattr(Gdk, "Rectangle"):
    _ensure_widget(Gdk, "Rectangle")

GLib.idle_add = getattr(GLib, "idle_add", lambda func, *args, **kwargs: func(*args, **kwargs))
GLib.timeout_add = getattr(GLib, "timeout_add", lambda *_args, **_kwargs: 0)
GLib.timeout_add_seconds = getattr(GLib, "timeout_add_seconds", lambda *_args, **_kwargs: 0)

for gio_name in ["Menu", "SimpleAction", "SimpleActionGroup"]:
    _ensure_widget(Gio, gio_name)


from ATLAS.ATLAS import ATLAS
from GTKUI.Chat.chat_page import ChatPage, GLib


class _AtlasForChatPage:
    def __init__(self):
        self.user_display = "Guest"
        self.provider_listener = None
        self.active_listener = None

    def add_provider_change_listener(self, listener):
        self.provider_listener = listener

    def remove_provider_change_listener(self, listener):
        if self.provider_listener == listener:
            self.provider_listener = None

    def add_active_user_change_listener(self, listener):
        self.active_listener = listener
        listener("guest", "Guest")

    def remove_active_user_change_listener(self, listener):
        if self.active_listener == listener:
            self.active_listener = None

    def get_active_persona_name(self):
        return "Helper"

    def get_user_display_name(self):
        return self.user_display

    def get_chat_status_summary(self):
        return {}

    def format_chat_status(self, _summary):
        return "status"


class _DummyChatSession:
    def __init__(self):
        self.last_factory = None
        self.last_success = None
        self.last_error = None
        self.last_thread_name = None
        self.messages = []
        self._conversation_id = "dummy-session-id"

    async def send_message(self, message):  # pragma: no cover - exercised via factory
        self.messages.append(message)
        return f"response:{message}"

    def run_in_background(
        self,
        coroutine_factory,
        *,
        on_success=None,
        on_error=None,
        thread_name=None,
    ):
        self.last_factory = coroutine_factory
        self.last_success = on_success
        self.last_error = on_error
        self.last_thread_name = thread_name
        return Future()

    @property
    def conversation_id(self):
        return self._conversation_id


class _FakeBuffer:
    def __init__(self, text: str):
        self._text = text
        self.cleared = False

    def get_start_iter(self):  # pragma: no cover - sentinel for interface compatibility
        return object()

    def get_end_iter(self):  # pragma: no cover - sentinel for interface compatibility
        return object()

    def get_text(self, *_args, **_kwargs):
        return self._text

    def set_text(self, value: str):
        self._text = value
        self.cleared = value == ""


class _FakeTextView:
    def __init__(self):
        self.focused = False

    def grab_focus(self):
        self.focused = True


class _AtlasStub:
    def __init__(self, user="Tester"):
        self._user_display_name = user
        self.calls = []
        self.last_success = None
        self.last_error = None

    def get_user_display_name(self):
        return self._user_display_name

    def send_chat_message_async(
        self,
        message,
        *,
        on_success=None,
        on_error=None,
        thread_name=None,
    ):
        self.calls.append(
            {
                "message": message,
                "on_success": on_success,
                "on_error": on_error,
                "thread_name": thread_name,
            }
        )
        self.last_success = on_success
        self.last_error = on_error
        return SimpleNamespace()


def _make_atlas_with_session():
    atlas = ATLAS.__new__(ATLAS)
    atlas.logger = SimpleNamespace(error=Mock())
    atlas.chat_session = _DummyChatSession()
    atlas.get_active_persona_name = Mock(return_value="Initial Persona")
    return atlas


def test_send_chat_message_async_invokes_callbacks_with_persona():
    atlas = _make_atlas_with_session()
    on_success = Mock()

    future = atlas.send_chat_message_async("Hello", on_success=on_success)

    assert isinstance(future, Future)
    assert atlas.chat_session.last_thread_name == "ChatResponseWorker"
    coroutine = atlas.chat_session.last_factory()
    assert inspect.isawaitable(coroutine)
    coroutine.close()

    atlas.get_active_persona_name.return_value = "Responder"
    atlas.chat_session.last_success("Thanks")

    on_success.assert_called_once_with("Responder", "Thanks")


def test_send_chat_message_async_error_callback_receives_persona():
    atlas = _make_atlas_with_session()
    on_error = Mock()

    atlas.send_chat_message_async("Hi", on_error=on_error)

    failure = RuntimeError("boom")
    atlas.get_active_persona_name.return_value = "Responder"
    atlas.chat_session.last_error(failure)

    on_error.assert_called_once_with("Responder", failure)
    atlas.logger.error.assert_not_called()


def test_send_chat_message_async_logs_when_no_error_callback():
    atlas = _make_atlas_with_session()

    atlas.send_chat_message_async("Hi there")

    failure = RuntimeError("uh oh")
    atlas.chat_session.last_error(failure)

    atlas.logger.error.assert_called()


def test_chat_page_on_send_message_dispatches_via_atlas(monkeypatch):
    page = ChatPage.__new__(ChatPage)
    page.awaiting_response = False
    page.input_buffer = _FakeBuffer("Hello world")
    page.input_textview = _FakeTextView()
    page._set_busy_state = Mock()
    page.add_message_bubble = Mock()
    page._on_response_complete = Mock(return_value=False)
    page.ATLAS = _AtlasStub()

    def immediate_idle_add(callback, *args, **kwargs):
        callback(*args, **kwargs)
        return 1

    monkeypatch.setattr(GLib, "idle_add", immediate_idle_add)

    page.on_send_message(None)

    assert page.ATLAS.calls == [
        {
            "message": "Hello world",
            "on_success": page.ATLAS.last_success,
            "on_error": page.ATLAS.last_error,
            "thread_name": "ChatResponseWorker",
        }
    ]
    assert page.input_buffer.cleared
    assert page.input_textview.focused
    page._set_busy_state.assert_called_once_with(True)

    page.ATLAS.last_success("Persona", "Reply text")

    page.add_message_bubble.assert_any_call("Tester", "Hello world", is_user=True, audio=None)
    page.add_message_bubble.assert_any_call("Persona", "Reply text", audio=None)
    page._on_response_complete.assert_called()


def test_chat_page_on_send_message_handles_errors(monkeypatch):
    page = ChatPage.__new__(ChatPage)
    page.awaiting_response = False
    page.input_buffer = _FakeBuffer("Hi")
    page.input_textview = _FakeTextView()
    page._set_busy_state = Mock()
    page.add_message_bubble = Mock()
    page._on_response_complete = Mock(return_value=False)
    page.ATLAS = _AtlasStub()

    def immediate_idle_add(callback, *args, **kwargs):
        callback(*args, **kwargs)
        return 1

    monkeypatch.setattr(GLib, "idle_add", immediate_idle_add)

    page.on_send_message(None)

    err = RuntimeError("failure")
    page.ATLAS.last_error("Helper", err)

    page.add_message_bubble.assert_any_call("Helper", "Error: failure", audio=None)
    page._on_response_complete.assert_called()


def test_chat_page_updates_active_user(monkeypatch):
    atlas = _AtlasForChatPage()
    page = ChatPage.__new__(ChatPage)
    page.ATLAS = atlas
    page.user_title_label = _Label()
    page.persona_title_label = _Label()
    page._current_user_display_name = atlas.get_user_display_name()
    page.update_persona_label = Mock()
    page.update_status_bar = Mock()
    page._active_user_listener = None
    page._provider_change_handler = object()

    page._register_active_user_listener()
    assert atlas.active_listener is not None

    atlas.active_listener("alice", "Alice")

    assert page._current_user_display_name == "Alice"
    assert page.user_title_label.get_text() == "Active user: Alice"
    page.update_persona_label.assert_called()
    page.update_status_bar.assert_called()

    page._on_close_request()
    assert atlas.active_listener is None
