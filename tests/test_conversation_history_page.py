import logging
import logging
from collections import deque
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

        def set_width_chars(self, *_args, **_kwargs):
            return None

        def set_placeholder_text(self, *_args, **_kwargs):
            return None

        def set_visible(self, *_args, **_kwargs):
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

    class _Entry(_DummyWidget):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._text = kwargs.get("text", "")

        def set_text(self, value):
            self._text = str(value)

        def get_text(self):
            return self._text

    class _CheckButton(_Button):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._active = bool(kwargs.get("active", False))

        def set_active(self, value):
            self._active = bool(value)

        def get_active(self):
            return self._active

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
    Gtk.Entry = _Entry
    Gtk.SearchEntry = _Entry
    Gtk.CheckButton = _CheckButton
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
pytest.importorskip(
    "pytest_postgresql",
    reason="PostgreSQL fixture is required for conversation history tests",
)

try:
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
except ImportError as exc:  # pragma: no cover - skip when SQLAlchemy helpers missing
    pytest.skip(
        f"SQLAlchemy runtime helpers unavailable: {exc}", allow_module_level=True
    )

if getattr(create_engine, "__module__", "").startswith("tests.conftest"):
    pytest.skip(
        "SQLAlchemy runtime is unavailable for conversation history tests",
        allow_module_level=True,
    )

from GTKUI.Chat.conversation_history_page import ConversationHistoryPage
from modules.conversation_store import Base, ConversationStoreRepository

from gi.repository import Gtk


@pytest.fixture
def repository(postgresql):
    engine = create_engine(postgresql.dsn(), future=True)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, future=True)
    repo = ConversationStoreRepository(factory)
    repo.create_schema()
    try:
        yield repo
    finally:
        engine.dispose()


class _HistoryAtlasStub:
    """Minimal ATLAS facade exposing conversation history primitives for tests."""

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

    def get_conversation_messages(
        self,
        conversation_id,
        *,
        limit=None,
        page_size=None,
        include_deleted=True,
        batch_size=200,
        cursor=None,
        direction=None,
        metadata=None,
        message_types=None,
        statuses=None,
    ):
        batch = max(int(batch_size or 200), 1)
        stream = self.conversation_repository.stream_conversation_messages(
            conversation_id=conversation_id,
            tenant_id=self._conversation_tenant(),
            include_deleted=include_deleted,
            batch_size=batch,
        )
        messages = list(stream)
        window = page_size or limit
        if window:
            window = max(int(window), 1)
            messages = messages[:window]
        return {
            "items": messages,
            "page": {
                "size": window or len(messages),
                "direction": str(direction or "forward"),
                "next_cursor": None,
                "previous_cursor": None,
            },
        }


class _PagingAtlasStub:
    def __init__(self, responses):
        self.logger = logging.getLogger("atlas.tests.history.pagination")
        self._responses = deque(responses)
        self.calls = []
        self._conversation_id = str(uuid.uuid4())

    def list_all_conversations(self, *, order: str = "desc"):
        return [
            {
                "id": self._conversation_id,
                "title": "Paged Conversation",
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        ]

    def get_conversation_messages(self, conversation_id, **kwargs):
        record = dict(kwargs)
        record["conversation_id"] = conversation_id
        self.calls.append(record)
        if self._responses:
            return self._responses.popleft()
        return {
            "items": [],
            "page": {
                "size": kwargs.get("page_size") or 0,
                "direction": kwargs.get("direction") or "forward",
                "next_cursor": None,
                "previous_cursor": None,
            },
        }


def test_history_page_renders_entire_conversation(repository):
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
    page._load_messages(str(conversation_id), reset_cursor=True)

    rendered_children = getattr(page.message_box, "children", [])
    assert len(rendered_children) == total_messages


def test_history_page_supports_pagination_and_filters():
    base_time = datetime.now(timezone.utc)
    responses = deque(
        [
            {
                "items": [
                    {
                        "id": "m3",
                        "created_at": (base_time + timedelta(seconds=3)).isoformat(),
                        "content": {"text": "latest"},
                        "metadata": {"category": "support"},
                    }
                ],
                "page": {
                    "size": 1,
                    "direction": "backward",
                    "previous_cursor": "cursor-older",
                    "next_cursor": None,
                },
            },
            {
                "items": [
                    {
                        "id": "m2",
                        "created_at": (base_time + timedelta(seconds=2)).isoformat(),
                        "content": {"text": "older"},
                    }
                ],
                "page": {
                    "size": 1,
                    "direction": "backward",
                    "previous_cursor": None,
                    "next_cursor": "cursor-newer",
                },
            },
            {
                "items": [],
                "page": {
                    "size": 0,
                    "direction": "backward",
                    "previous_cursor": None,
                    "next_cursor": None,
                },
            },
            {
                "items": [
                    {
                        "id": "m1",
                        "created_at": (base_time + timedelta(seconds=1)).isoformat(),
                        "content": {"text": "filtered"},
                        "message_type": "note",
                        "status": "complete",
                    }
                ],
                "page": {
                    "size": 1,
                    "direction": "backward",
                    "previous_cursor": None,
                    "next_cursor": None,
                },
            },
        ]
    )

    atlas = _PagingAtlasStub(responses)
    page = ConversationHistoryPage(atlas)
    conversation_id = atlas.list_all_conversations()[0]["id"]
    page._selected_id = conversation_id

    page._load_messages(conversation_id, reset_cursor=True)
    first_call = atlas.calls[0]
    assert first_call["cursor"] is None
    assert first_call["direction"] == "backward"
    assert first_call["include_deleted"] is False
    assert first_call["page_size"] == 50
    assert page.next_page_button._sensitive is True

    page._on_next_page_clicked(page.next_page_button)
    second_call = atlas.calls[1]
    assert second_call["cursor"] == "cursor-older"
    assert page.previous_page_button._sensitive is True

    page.message_metadata_entry.set_text('{"category": "support"}')
    page._on_message_metadata_activate(page.message_metadata_entry)
    third_call = atlas.calls[2]
    assert third_call["metadata"] == {"category": "support"}
    assert third_call["cursor"] is None
    assert page.previous_page_button._sensitive is False
    assert len(page.message_box.children) == 1  # placeholder for empty page

    page.message_type_entry.set_text("note, reply")
    page.message_status_entry.set_text("complete")
    page.include_deleted_toggle.set_active(True)
    fourth_call = atlas.calls[3]
    assert fourth_call["message_types"] == ["note", "reply"]
    assert fourth_call["statuses"] == ["complete"]
    assert fourth_call["include_deleted"] is True
    assert len(page.message_box.children) == 1
