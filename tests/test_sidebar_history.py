"""Tests for the navigation sidebar conversation history fallback."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

gi = pytest.importorskip("gi")
from gi.repository import Gtk

from GTKUI.sidebar import _NavigationSidebar


@pytest.fixture(scope="module")
def gtk_display():  # pragma: no cover - infrastructure helper
    try:
        Gtk.init()
    except TypeError:
        Gtk.init([])
    yield


def _listbox_children(listbox: Gtk.ListBox) -> list[Gtk.ListBoxRow]:
    getter = getattr(listbox, "get_children", None)
    if callable(getter):
        return list(getter())

    rows: list[Gtk.ListBoxRow] = []
    child = listbox.get_first_child()
    while child is not None:
        rows.append(child)
        child = child.get_next_sibling()
    return rows


class _SessionStub:
    def __init__(self) -> None:
        self._conversation_id = "session-identifier"
        self.conversation_history = [
            {"timestamp": "2024-01-01T12:00:00", "role": "user", "content": "hello"}
        ]

    def get_conversation_id(self) -> str:
        return self._conversation_id


class _AtlasStub:
    def __init__(self, session: _SessionStub) -> None:
        self.chat_session = session
        self.logger = logging.getLogger("atlas.tests.sidebar")

    def get_recent_conversations(self, *, limit: int = 20):  # pragma: no cover - stub signature
        return []


class _MainWindowStub:
    def __init__(self, atlas: _AtlasStub) -> None:
        self.ATLAS = atlas
        dummy_widget = Gtk.Box()
        management = SimpleNamespace(get_embeddable_widget=lambda: dummy_widget)
        self.persona_management = management
        self.provider_management = management
        self.tool_management = management
        self.skill_management = management
        self.task_management = management
        self.job_management = management

    # Navigation callbacks -------------------------------------------------
    def show_provider_menu(self) -> None:  # pragma: no cover - GUI stub
        pass

    def show_chat_page(self) -> None:  # pragma: no cover - GUI stub
        pass

    def show_tools_menu(self, tool_name: str | None = None) -> None:  # pragma: no cover
        pass

    def show_task_workspace(self, task_id: str | None = None) -> None:  # pragma: no cover
        pass

    def show_job_workspace(self, job_id: str | None = None) -> None:  # pragma: no cover
        pass

    def show_skills_menu(self, skill_name: str | None = None) -> None:  # pragma: no cover
        pass

    def show_speech_settings(self) -> None:  # pragma: no cover - GUI stub
        pass

    def show_persona_menu(self) -> None:  # pragma: no cover - GUI stub
        pass

    def show_conversation_history_page(self, conversation_id: str | None = None) -> None:
        pass  # pragma: no cover - GUI stub

    def show_accounts_page(self) -> None:  # pragma: no cover - GUI stub
        pass

    def show_settings_page(self) -> None:  # pragma: no cover - GUI stub
        pass


@pytest.mark.usefixtures("gtk_display")
def test_sidebar_history_renders_active_session_when_repository_missing() -> None:
    session = _SessionStub()
    atlas = _AtlasStub(session)
    main_window = _MainWindowStub(atlas)

    sidebar = _NavigationSidebar(main_window)
    try:
        history_rows = _listbox_children(sidebar._history_listbox)
        rendered_rows = [
            row for row in history_rows if row is not sidebar._history_header_row
        ]

        assert rendered_rows, "Expected the active conversation to appear in the sidebar"
        if sidebar._history_placeholder_row is not None:
            assert (
                sidebar._history_placeholder_row not in rendered_rows
            ), "Placeholder should not be shown when a conversation is available"

        first_row = rendered_rows[0]
        container = first_row.get_child()
        assert container is not None
        title_widget = container.get_first_child()
        assert isinstance(title_widget, Gtk.Label)
        assert "Current conversation" in title_widget.get_text()
    finally:
        sidebar.unparent()
