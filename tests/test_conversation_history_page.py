import logging
import pytest
import sys
import types
import uuid
from datetime import datetime, timedelta, timezone


def _install_gtk_stubs() -> None:
    if "gi" in sys.modules:
        return

    gi_module = types.ModuleType("gi")
    repository_module = types.ModuleType("gi.repository")

    def _require_version(*_args, **_kwargs):  # pragma: no cover - stub hook
        return None

    gi_module.require_version = _require_version
    gi_module.repository = repository_module
    sys.modules["gi"] = gi_module
    sys.modules["gi.repository"] = repository_module

    class _DummyWidget:
        def __init__(self, *args, **kwargs):
            self.children: list = []
            self._parent = None
            self._label = ""
            self._sensitive = True

        def _set_parent(self, parent):
            self._parent = parent

        def append(self, child):
            if hasattr(child, "_set_parent"):
                child._set_parent(self)
            self.children.append(child)

        def set_child(self, child):
            if hasattr(child, "_set_parent"):
                child._set_parent(self)
            self.children = [child]

        def remove(self, child):
            if child in self.children:
                self.children.remove(child)

        def get_children(self):
            return list(self.children)

        def get_first_child(self):
            return self.children[0] if self.children else None

        def get_next_sibling(self):
            if self._parent is None:
                return None
            siblings = getattr(self._parent, "children", [])
            try:
                index = siblings.index(self)
            except ValueError:
                return None
            next_index = index + 1
            if next_index < len(siblings):
                return siblings[next_index]
            return None

        def add_css_class(self, *_args, **_kwargs):
            return None

        def set_margin_top(self, *_args, **_kwargs):
            return None

        def set_margin_bottom(self, *_args, **_kwargs):
            return None

        def set_margin_start(self, *_args, **_kwargs):
            return None

        def set_margin_end(self, *_args, **_kwargs):
            return None

        def set_hexpand(self, *_args, **_kwargs):
            return None

        def set_vexpand(self, *_args, **_kwargs):
            return None

        def set_xalign(self, *_args, **_kwargs):
            return None

        def set_wrap(self, *_args, **_kwargs):
            return None

        def set_wrap_mode(self, *_args, **_kwargs):
            return None

        def set_max_width_chars(self, *_args, **_kwargs):
            return None

        def set_label(self, value):
            self._label = value

        def set_sensitive(self, value):
            self._sensitive = bool(value)

        def connect(self, *_args, **_kwargs):
            return None

        def add_controller(self, *_args, **_kwargs):
            return None

        def set_accessible_role(self, *_args, **_kwargs):
            return None

        def set_activatable(self, *_args, **_kwargs):
            return None

        def set_selection_mode(self, *_args, **_kwargs):
            return None

        def select_row(self, *_args, **_kwargs):
            return None

    class _ListBox(_DummyWidget):
        pass

    class _ListBoxRow(_DummyWidget):
        pass

    class _GestureClick(_DummyWidget):
        pass

    class _Button(_DummyWidget):
        pass

    class _Label(_DummyWidget):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._label = kwargs.get("label", "")

    class _Box(_DummyWidget):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.orientation = kwargs.get("orientation")
            self.spacing = kwargs.get("spacing")

    class _ScrolledWindow(_DummyWidget):
        def set_min_content_width(self, *_args, **_kwargs):
            return None

    Gtk = types.ModuleType("Gtk")
    Gtk.Box = _Box
    Gtk.Label = _Label
    Gtk.Button = _Button
    Gtk.ScrolledWindow = _ScrolledWindow
    Gtk.ListBox = _ListBox
    Gtk.ListBoxRow = _ListBoxRow
    Gtk.GestureClick = _GestureClick
    Gtk.AccessibleRole = types.SimpleNamespace(LIST=0, LIST_ITEM=1)
    Gtk.SelectionMode = types.SimpleNamespace(SINGLE=1)
    Gtk.PolicyType = types.SimpleNamespace(NEVER=0, AUTOMATIC=1)
    Gtk.Orientation = types.SimpleNamespace(VERTICAL=0, HORIZONTAL=1)

    repository_module.Gtk = Gtk
    sys.modules["gi.repository.Gtk"] = Gtk

    GLib = types.ModuleType("GLib")
    GLib.idle_add = lambda func, *args, **kwargs: func(*args, **kwargs)
    repository_module.GLib = GLib
    sys.modules["gi.repository.GLib"] = GLib

    Pango = types.ModuleType("Pango")
    Pango.WrapMode = types.SimpleNamespace(WORD_CHAR=0)
    repository_module.Pango = Pango
    sys.modules["gi.repository.Pango"] = Pango


_install_gtk_stubs()

pytest.importorskip("sqlalchemy")

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ATLAS.ATLAS import ATLAS
from GTKUI.Chat.conversation_history_page import ConversationHistoryPage
from modules.conversation_store import ConversationStoreRepository

from gi.repository import Gtk


class _HistoryAtlasStub:
    """Minimal ATLAS facade exposing conversation history primitives for tests."""

    get_conversation_messages = ATLAS.get_conversation_messages

    def __init__(self, repository: ConversationStoreRepository) -> None:
        self.conversation_repository = repository
        self.logger = logging.getLogger("atlas.tests.history")
        self._conversation_history_listeners = []

    def _conversation_tenant(self) -> str:
        return "tenant"

    def list_all_conversations(self, *, order: str = "desc"):
        return self.conversation_repository.list_conversations_for_tenant(
            self._conversation_tenant(),
            order=order,
        )


def test_history_page_renders_entire_conversation():
    engine = create_engine("sqlite:///:memory:", future=True)
    try:
        factory = sessionmaker(bind=engine, future=True)
        repository = ConversationStoreRepository(factory)
        repository.create_schema()

        conversation_id = uuid.uuid4()
        repository.ensure_conversation(
            conversation_id,
            tenant_id="tenant",
            title="Integration conversation",
        )

        base_time = datetime.now(timezone.utc)
        total_messages = 550
        for idx in range(total_messages):
            repository.add_message(
                conversation_id,
                tenant_id="tenant",
                role="assistant" if idx % 2 else "user",
                content={"text": f"message-{idx}"},
                metadata={"idx": idx},
                timestamp=(base_time + timedelta(seconds=idx)).isoformat(),
            )

        atlas = _HistoryAtlasStub(repository)

        all_messages = atlas.get_conversation_messages(conversation_id)
        assert len(all_messages) == total_messages
        assert all_messages[-1]["metadata"]["idx"] == total_messages - 1

        limited_messages = atlas.get_conversation_messages(conversation_id, limit=500)
        assert len(limited_messages) == 500

        page = ConversationHistoryPage(atlas)
        page._load_messages(str(conversation_id))

        rendered_children = getattr(page.message_box, "children", [])
        assert len(rendered_children) == total_messages
    finally:
        engine.dispose()
