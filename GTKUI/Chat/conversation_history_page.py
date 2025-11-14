"""Conversation history management page for the main notebook."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from collections.abc import Iterable
from typing import Any, Dict, List, Mapping, Optional, Tuple

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Pango", "1.0")
from gi.repository import GLib, Gtk, Pango


class ConversationHistoryPage(Gtk.Box):
    """Display stored conversations and allow basic database operations."""

    def __init__(self, atlas) -> None:
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        self.ATLAS = atlas
        self._conversations: List[Dict[str, Any]] = []
        self._all_conversations: List[Dict[str, Any]] = []
        self._row_lookup: Dict[Gtk.ListBoxRow, str] = {}
        self._id_to_row: Dict[str, Gtk.ListBoxRow] = {}
        self._selected_id: Optional[str] = None
        self._pending_focus: Optional[str] = None
        self._conversation_listener = None
        self._active_user_listener = None
        self._search_timeout_id: int = 0
        self._search_query: str = ""
        self._search_active: bool = False
        self._search_results: List[Dict[str, Any]] = []
        self._retention_backend_disabled = False
        self._retention_failure_reason: Optional[str] = None
        self._message_cursor: Optional[str] = None
        self._cursor_history: List[Optional[str]] = []
        self._message_page_state: Dict[str, Any] = {}
        self._default_page_size = 50
        self._timeline_data: Dict[str, Any] = {
            "bins": [],
            "switches": [],
            "retention": [],
            "summaries": [],
            "total": 0,
        }
        self._message_widgets: List[Gtk.Widget] = []
        self._playback_sequence: List[int] = []
        self._playback_index: Optional[int] = None
        self._active_message_widget: Optional[Gtk.Widget] = None
        self._pending_highlight_index: Optional[int] = None

        self.set_margin_top(12)
        self.set_margin_bottom(12)
        self.set_margin_start(12)
        self.set_margin_end(12)

        header = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        header.set_hexpand(True)

        title = Gtk.Label(label="Conversation History")
        title.set_xalign(0.0)
        title.add_css_class("title-3")
        title.set_hexpand(True)
        header.append(title)

        self.refresh_button = Gtk.Button(label="Refresh")
        self.refresh_button.connect("clicked", lambda _btn: self._reload_conversations(preserve_selection=True))
        header.append(self.refresh_button)

        self.reset_button = Gtk.Button(label="Reset Conversation")
        self.reset_button.connect("clicked", self._on_reset_clicked)
        self.reset_button.set_sensitive(False)
        header.append(self.reset_button)

        self.delete_button = Gtk.Button(label="Delete Conversation")
        self.delete_button.connect("clicked", self._on_delete_clicked)
        self.delete_button.set_sensitive(False)
        header.append(self.delete_button)

        self.retention_button = Gtk.Button(label="Run retention")
        self.retention_button.connect("clicked", self._on_retention_clicked)
        self.retention_button.set_sensitive(False)
        header.append(self.retention_button)

        self.append(header)

        search_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        search_box.set_hexpand(True)

        self.search_entry = Gtk.SearchEntry()
        self.search_entry.set_placeholder_text("Search conversations…")
        self.search_entry.set_hexpand(True)
        self.search_entry.connect("search-changed", self._on_search_changed)
        self.search_entry.connect("stop-search", self._on_search_cleared)
        self.search_entry.connect("activate", self._on_search_activate)
        search_box.append(self.search_entry)

        self.metadata_toggle = Gtk.CheckButton(label="Metadata filter")
        self.metadata_toggle.connect("toggled", self._on_metadata_toggled)
        search_box.append(self.metadata_toggle)

        self.metadata_entry = Gtk.Entry()
        self.metadata_entry.set_placeholder_text("Filter metadata (JSON object)")
        self.metadata_entry.set_hexpand(True)
        self.metadata_entry.set_visible(False)
        self.metadata_entry.connect("changed", self._on_metadata_changed)
        self.metadata_entry.connect("activate", self._on_metadata_activate)
        search_box.append(self.metadata_entry)

        self.append(search_box)

        self.status_label = Gtk.Label(label="")
        self.status_label.set_xalign(0.0)
        self.status_label.set_wrap(True)
        self.status_label.add_css_class("history-nav-subtitle")
        self.append(self.status_label)

        content = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        content.set_hexpand(True)
        content.set_vexpand(True)

        left_scroller = Gtk.ScrolledWindow()
        left_scroller.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        left_scroller.set_hexpand(False)
        left_scroller.set_vexpand(True)
        left_scroller.set_min_content_width(260)

        self.conversation_list = Gtk.ListBox()
        self.conversation_list.set_selection_mode(Gtk.SelectionMode.SINGLE)
        self.conversation_list.connect("row-selected", self._on_conversation_selected)
        left_scroller.set_child(self.conversation_list)
        content.append(left_scroller)

        right_panel = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        right_panel.set_hexpand(True)
        right_panel.set_vexpand(True)

        controls_panel = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        right_panel.append(controls_panel)

        pagination_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        controls_panel.append(pagination_row)

        page_size_label = Gtk.Label(label="Page size")
        page_size_label.set_xalign(0.0)
        pagination_row.append(page_size_label)

        self.page_size_entry = Gtk.Entry()
        self.page_size_entry.set_width_chars(4)
        self.page_size_entry.set_text(str(self._default_page_size))
        self.page_size_entry.connect("activate", self._on_page_size_activate)
        pagination_row.append(self.page_size_entry)

        self.direction_toggle = Gtk.CheckButton(label="Newest first")
        self.direction_toggle.set_active(True)
        self.direction_toggle.connect("toggled", self._on_direction_toggled)
        pagination_row.append(self.direction_toggle)

        self.include_deleted_toggle = Gtk.CheckButton(label="Include deleted")
        self.include_deleted_toggle.set_active(False)
        self.include_deleted_toggle.connect("toggled", self._on_include_deleted_toggled)
        pagination_row.append(self.include_deleted_toggle)

        self.previous_page_button = Gtk.Button(label="Previous page")
        self.previous_page_button.set_sensitive(False)
        self.previous_page_button.connect("clicked", self._on_previous_page_clicked)
        pagination_row.append(self.previous_page_button)

        self.next_page_button = Gtk.Button(label="Next page")
        self.next_page_button.set_sensitive(False)
        self.next_page_button.connect("clicked", self._on_next_page_clicked)
        pagination_row.append(self.next_page_button)

        filter_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        controls_panel.append(filter_row)

        self.message_metadata_entry = Gtk.Entry()
        self.message_metadata_entry.set_placeholder_text("Metadata filter (JSON)")
        self.message_metadata_entry.set_hexpand(True)
        self.message_metadata_entry.connect("activate", self._on_message_metadata_activate)
        filter_row.append(self.message_metadata_entry)

        self.message_type_entry = Gtk.Entry()
        self.message_type_entry.set_placeholder_text("Message types")
        self.message_type_entry.set_hexpand(True)
        self.message_type_entry.connect("activate", self._on_message_type_activate)
        filter_row.append(self.message_type_entry)

        self.message_status_entry = Gtk.Entry()
        self.message_status_entry.set_placeholder_text("Statuses")
        self.message_status_entry.set_hexpand(True)
        self.message_status_entry.connect("activate", self._on_message_status_activate)
        filter_row.append(self.message_status_entry)

        timeline_panel = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        timeline_panel.set_hexpand(True)
        timeline_panel.set_margin_top(6)

        timeline_header = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        timeline_label = Gtk.Label(label="Conversation timeline")
        timeline_label.set_xalign(0.0)
        timeline_label.add_css_class("history-nav-label")
        timeline_header.append(timeline_label)

        overlay_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        overlay_box.add_css_class("linked")

        self.retention_overlay_toggle = Gtk.CheckButton(label="Retention overlay")
        self.retention_overlay_toggle.connect("toggled", self._on_timeline_overlay_toggled)
        overlay_box.append(self.retention_overlay_toggle)

        self.summary_overlay_toggle = Gtk.CheckButton(label="Episodic summaries")
        self.summary_overlay_toggle.connect("toggled", self._on_timeline_overlay_toggled)
        overlay_box.append(self.summary_overlay_toggle)

        timeline_header.append(overlay_box)
        timeline_panel.append(timeline_header)

        self.timeline_area = Gtk.DrawingArea()
        self.timeline_area.set_content_height(140)
        self.timeline_area.set_hexpand(True)
        self.timeline_area.set_vexpand(False)
        self.timeline_area.set_draw_func(self._draw_timeline)
        timeline_panel.append(self.timeline_area)

        self.timeline_status_label = Gtk.Label(label="Select a conversation to view its timeline.")
        self.timeline_status_label.set_xalign(0.0)
        self.timeline_status_label.add_css_class("history-nav-subtitle")
        timeline_panel.append(self.timeline_status_label)

        playback_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)

        self.playback_restart_button = Gtk.Button(label="Restart")
        self.playback_restart_button.connect("clicked", self._on_playback_restart)
        playback_row.append(self.playback_restart_button)

        self.playback_previous_button = Gtk.Button(label="Previous")
        self.playback_previous_button.connect("clicked", self._on_playback_previous)
        playback_row.append(self.playback_previous_button)

        self.playback_next_button = Gtk.Button(label="Next")
        self.playback_next_button.connect("clicked", self._on_playback_next)
        playback_row.append(self.playback_next_button)

        self.playback_status_label = Gtk.Label(label="No playback data loaded.")
        self.playback_status_label.set_xalign(0.0)
        self.playback_status_label.add_css_class("history-nav-subtitle")
        playback_row.append(self.playback_status_label)

        timeline_panel.append(playback_row)
        right_panel.append(timeline_panel)

        right_scroller = Gtk.ScrolledWindow()
        right_scroller.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        right_scroller.set_hexpand(True)
        right_scroller.set_vexpand(True)

        self.message_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        self.message_box.set_margin_top(6)
        self.message_box.set_margin_bottom(6)
        self.message_box.set_margin_start(6)
        self.message_box.set_margin_end(6)
        right_scroller.set_child(self.message_box)
        right_panel.append(right_scroller)

        content.append(right_panel)

        self.append(content)

        self._message_placeholder = Gtk.Label(
            label="Select a conversation to view its messages."
        )
        self._message_placeholder.set_xalign(0.0)
        self._message_placeholder.set_wrap(True)
        self._message_placeholder.add_css_class("history-placeholder")
        self.message_box.append(self._message_placeholder)

        listener = getattr(self.ATLAS, "add_conversation_history_listener", None)
        if callable(listener):
            callback = lambda event: GLib.idle_add(self._handle_conversation_event, event)
            listener(callback)
            self._conversation_listener = callback

        self._register_active_user_listener()
        self._update_retention_button_state()
        self.connect("unrealize", self._on_unrealize)

        self._reset_message_pagination_state()
        self._reload_conversations()

    # ------------------------------------------------------------------
    # Public API used by the main window
    # ------------------------------------------------------------------
    def focus_conversation(self, conversation_id: str) -> None:
        if not conversation_id:
            return
        self._pending_focus = str(conversation_id)
        self._reload_conversations(preserve_selection=True)

    # ------------------------------------------------------------------
    # UI callbacks
    # ------------------------------------------------------------------
    def _on_conversation_selected(self, _listbox: Gtk.ListBox, row: Optional[Gtk.ListBoxRow]) -> None:
        if row is None:
            self._selected_id = None
            self._show_message_placeholder()
            self._update_action_buttons()
            return

        identifier = self._row_lookup.get(row)
        if identifier is None:
            self._selected_id = None
            self._show_message_placeholder()
            self._update_action_buttons()
            self._reset_message_pagination_state()
            return

        self._selected_id = identifier
        self._update_action_buttons()
        self._load_messages(identifier, reset_cursor=True)

    def _on_reset_clicked(self, _button: Gtk.Button) -> None:
        if not self._selected_id:
            return
        result = self.ATLAS.reset_conversation_messages(self._selected_id)
        if result.get("success"):
            self._set_status("Conversation messages reset.")
            self._load_messages(self._selected_id)
        else:
            error = result.get("error") or "Unable to reset conversation."
            self._set_status(error)

    def _on_delete_clicked(self, _button: Gtk.Button) -> None:
        if not self._selected_id:
            return
        result = self.ATLAS.delete_conversation(self._selected_id)
        if result.get("success"):
            self._set_status("Conversation deleted.")
            self._selected_id = None
            self._reload_conversations()
            self._show_message_placeholder()
        else:
            error = result.get("error") or "Unable to delete conversation."
            self._set_status(error)

    # ------------------------------------------------------------------
    # Data loading helpers
    # ------------------------------------------------------------------
    def _handle_conversation_event(self, _event: Mapping[str, Any]) -> bool:
        if self._search_active and self._search_query:
            self._execute_search(force=True)
        else:
            self._reload_conversations(preserve_selection=True)
        return False

    def _reload_conversations(self, *, preserve_selection: bool = False) -> None:
        previous = self._selected_id if preserve_selection else None
        try:
            records = self.ATLAS.list_all_conversations(order="desc")
        except Exception as exc:  # pragma: no cover - defensive logging upstream
            self._set_status(f"Failed to load conversations: {exc}")
            records = []

        self._populate_conversation_list(records, store_source=True)
        if not self._search_active:
            self._set_status("")

        target = self._pending_focus or previous
        self._pending_focus = None
        if target:
            row = self._id_to_row.get(target)
            if row is not None:
                self.conversation_list.select_row(row)

    def _populate_conversation_list(
        self, records: List[Dict[str, Any]], *, store_source: bool = False
    ) -> None:
        if store_source:
            self._all_conversations = list(records)
        self._conversations = list(records)
        self._row_lookup.clear()
        self._id_to_row.clear()
        self._selected_id = None

        children_getter = getattr(self.conversation_list, "get_children", None)
        if callable(children_getter):
            for child in list(children_getter()):
                self.conversation_list.remove(child)
        else:
            child = self.conversation_list.get_first_child()
            while child is not None:
                next_child = child.get_next_sibling()
                self.conversation_list.remove(child)
                child = next_child

        if not records:
            placeholder_row = Gtk.ListBoxRow()
            placeholder_row.set_activatable(False)
            box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
            box.set_margin_top(6)
            box.set_margin_bottom(6)
            box.set_margin_start(12)
            box.set_margin_end(12)
            placeholder_text = (
                "No conversations matched the current query."
                if self._search_active and self._search_query
                else "No stored conversations yet."
            )
            label = Gtk.Label(label=placeholder_text)
            label.set_xalign(0.0)
            label.add_css_class("history-placeholder")
            box.append(label)
            placeholder_row.set_child(box)
            self.conversation_list.append(placeholder_row)
            self._selected_id = None
            self._update_action_buttons()
            return

        for record in records:
            identifier = str(record.get("id"))
            row = Gtk.ListBoxRow()
            row.set_accessible_role(Gtk.AccessibleRole.LIST_ITEM)
            row.add_css_class("sidebar-nav-item")
            row.set_activatable(True)

            box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
            box.set_margin_top(6)
            box.set_margin_bottom(6)
            box.set_margin_start(12)
            box.set_margin_end(12)

            title_label = Gtk.Label(label=self._summarise_conversation(record))
            title_label.set_xalign(0.0)
            title_label.add_css_class("history-nav-label")
            box.append(title_label)

            timestamp_label = Gtk.Label(label=self._format_timestamp(record.get("created_at")))
            timestamp_label.set_xalign(0.0)
            timestamp_label.add_css_class("history-nav-subtitle")
            box.append(timestamp_label)

            search_hit = record.get("__search_hit__") if isinstance(record, Mapping) else None
            if isinstance(search_hit, Mapping):
                snippet = search_hit.get("snippet")
                if isinstance(snippet, str) and snippet.strip():
                    snippet_label = Gtk.Label(label=snippet.strip())
                    snippet_label.set_xalign(0.0)
                    snippet_label.set_wrap(True)
                    snippet_label.set_wrap_mode(Pango.WrapMode.WORD_CHAR)
                    snippet_label.add_css_class("history-nav-subtitle")
                    box.append(snippet_label)

            row.set_child(box)

            gesture = Gtk.GestureClick()
            gesture.connect(
                "released",
                lambda _gesture, _n_press, _x, _y, conv_id=identifier: self.focus_conversation(conv_id),
            )
            row.add_controller(gesture)

            self.conversation_list.append(row)
            self._row_lookup[row] = identifier
            self._id_to_row[identifier] = row

        self._update_action_buttons()

    def _load_messages(self, conversation_id: str, *, reset_cursor: bool = False) -> None:
        if reset_cursor:
            self._reset_message_pagination_state()

        request = self._build_message_request()
        if request is None:
            return

        try:
            response = self.ATLAS.get_conversation_messages(conversation_id, **request)
        except Exception as exc:  # pragma: no cover - defensive logging upstream
            self._set_status(f"Failed to load messages: {exc}")
            self._render_messages([])
            return

        items, page_state = self._coerce_message_response(response, request)
        self._message_page_state = page_state
        self._set_status("")
        self._render_messages(items)
        self._update_page_buttons()

    def _build_message_request(self) -> Optional[Dict[str, Any]]:
        page_size = self._resolve_page_size()
        metadata_filter, error = self._parse_metadata_filter()
        if error:
            self._set_status(error)
            return None

        message_types = self._parse_filter_values(self.message_type_entry)
        statuses = self._parse_filter_values(self.message_status_entry)
        include_deleted = bool(self.include_deleted_toggle.get_active())
        direction = "backward" if self.direction_toggle.get_active() else "forward"

        request: Dict[str, Any] = {
            "page_size": page_size,
            "include_deleted": include_deleted,
            "direction": direction,
            "cursor": self._message_cursor,
        }

        if metadata_filter:
            request["metadata"] = metadata_filter
        if message_types:
            request["message_types"] = message_types
        if statuses:
            request["statuses"] = statuses

        return request

    def _coerce_message_response(
        self,
        response: Any,
        request: Mapping[str, Any],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if isinstance(response, Mapping):
            items = list(response.get("items") or [])
            page_info = response.get("page")
            if not isinstance(page_info, Mapping):
                page_info = {}
            direction = str(page_info.get("direction") or request.get("direction") or "forward")
            return (
                items,
                {
                    "size": page_info.get("size") or request.get("page_size") or len(items),
                    "direction": direction,
                    "next_cursor": page_info.get("next_cursor"),
                    "previous_cursor": page_info.get("previous_cursor"),
                },
            )

        if isinstance(response, Iterable) and not isinstance(response, (str, bytes)):
            items = list(response)
        else:
            items = []

        return (
            items,
            {
                "size": request.get("page_size") or len(items),
                "direction": str(request.get("direction") or "forward"),
                "next_cursor": None,
                "previous_cursor": None,
            },
        )

    def _render_messages(self, messages: List[Dict[str, Any]]) -> None:
        self._clear_message_box()
        self._update_timeline(messages)

        if not messages:
            self._show_message_placeholder("No messages stored for this conversation.")
            return

        iterator = iter(messages)

        def process_batch() -> bool:
            processed = 0
            for message in iterator:
                container = self._build_message_widget(message)
                self.message_box.append(container)
                self._message_widgets.append(container)
                if self._pending_highlight_index is not None:
                    self._highlight_widget_by_index(self._pending_highlight_index)
                processed += 1
                if processed >= 100:
                    GLib.idle_add(process_batch)
                    return False
            self._update_playback_display()
            return False

        process_batch()
        self._update_playback_display()

    def _build_message_widget(self, message: Mapping[str, Any]) -> Gtk.Widget:
        container = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        container.add_css_class("conversation-history-entry")
        container.set_margin_bottom(8)
        container.set_focusable(True)

        header = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        header.set_hexpand(True)

        role = str(message.get("role") or "message").capitalize()
        role_label = Gtk.Label(label=role)
        role_label.set_xalign(0.0)
        role_label.add_css_class("history-nav-label")
        header.append(role_label)

        status = message.get("status")
        if status:
            status_label = Gtk.Label(label=f"[{status}]")
            status_label.set_xalign(0.0)
            status_label.add_css_class("history-nav-subtitle")
            header.append(status_label)

        timestamp = self._format_timestamp(message.get("created_at") or message.get("timestamp"))
        if timestamp:
            time_label = Gtk.Label(label=timestamp)
            time_label.set_xalign(1.0)
            time_label.set_hexpand(True)
            time_label.add_css_class("history-nav-subtitle")
            header.append(time_label)

        container.append(header)

        content_label = Gtk.Label(label=self._format_content(message.get("content")))
        content_label.set_xalign(0.0)
        content_label.set_wrap(True)
        content_label.set_wrap_mode(Pango.WrapMode.WORD_CHAR)
        content_label.set_max_width_chars(80)
        container.append(content_label)

        metadata = message.get("metadata")
        if isinstance(metadata, Mapping) and metadata:
            meta_label = Gtk.Label(label=self._format_metadata(metadata))
            meta_label.set_xalign(0.0)
            meta_label.set_wrap(True)
            meta_label.set_wrap_mode(Pango.WrapMode.WORD_CHAR)
            meta_label.add_css_class("history-nav-subtitle")
            container.append(meta_label)

        return container

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------
    def _summarise_conversation(self, record: Mapping[str, Any]) -> str:
        title = record.get("title")
        if isinstance(title, str) and title.strip():
            return title.strip()

        metadata = record.get("metadata")
        if isinstance(metadata, Mapping):
            candidate = metadata.get("title") or metadata.get("name")
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()

        identifier = record.get("id")
        text = str(identifier)
        return f"Conversation {text[:8]}" if text else "Conversation"

    def _format_timestamp(self, value: Any) -> str:
        if not value:
            return ""
        text = str(value)
        if text.endswith("Z"):
            text = text.replace("Z", "+00:00")
        try:
            moment = datetime.fromisoformat(text)
        except ValueError:
            return str(value)
        if moment.tzinfo is not None:
            moment = moment.astimezone()
        return moment.strftime("%Y-%m-%d %H:%M:%S")

    def _format_content(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, Mapping):
            text = content.get("text")
            if isinstance(text, str):
                return text
        try:
            return json.dumps(content, ensure_ascii=False, indent=2)
        except (TypeError, ValueError):
            return str(content)

    def _format_metadata(self, metadata: Mapping[str, Any]) -> str:
        try:
            return json.dumps(metadata, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(metadata)

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------
    def _clear_message_box(self) -> None:
        children_getter = getattr(self.message_box, "get_children", None)
        if callable(children_getter):
            for child in list(children_getter()):
                self.message_box.remove(child)
        else:
            child = self.message_box.get_first_child()
            while child is not None:
                next_child = child.get_next_sibling()
                self.message_box.remove(child)
                child = next_child
        if hasattr(self.message_box, "children"):
            self.message_box.children = []
        self._message_widgets = []
        self._active_message_widget = None
        self._pending_highlight_index = None

    def _show_message_placeholder(self, text: str = "Select a conversation to view its messages.") -> None:
        self._clear_message_box()
        placeholder = Gtk.Label(label=text)
        placeholder.set_xalign(0.0)
        placeholder.set_wrap(True)
        placeholder.add_css_class("history-placeholder")
        self.message_box.append(placeholder)
        self._update_page_buttons()
        self._update_timeline([])

    def _update_action_buttons(self) -> None:
        has_selection = bool(self._selected_id)
        self.reset_button.set_sensitive(has_selection)
        self.delete_button.set_sensitive(has_selection)

    def _update_page_buttons(self) -> None:
        has_selection = bool(self._selected_id)
        next_cursor = self._resolve_next_cursor() if has_selection else None
        previous_available = bool(self._cursor_history)
        self.next_page_button.set_sensitive(bool(next_cursor))
        self.previous_page_button.set_sensitive(previous_available)

    def _resolve_next_cursor(self) -> Optional[str]:
        direction = str(self._message_page_state.get("direction") or "forward")
        if direction == "forward":
            return self._message_page_state.get("next_cursor")
        return self._message_page_state.get("previous_cursor")

    def _register_active_user_listener(self) -> None:
        if self._active_user_listener is not None:
            return

        adder = getattr(self.ATLAS, "add_active_user_change_listener", None)
        if not callable(adder):
            return

        def _listener(_username: str, _display_name: str) -> None:
            GLib.idle_add(self._update_retention_button_state)

        adder(_listener)
        self._active_user_listener = _listener

    def _update_retention_button_state(self) -> bool:
        available = False
        tooltip: Optional[str] = None

        status_getter = getattr(self.ATLAS, "conversation_retention_status", None)
        if callable(status_getter):
            status = status_getter()
            available = bool(status.get("available"))
            reason = status.get("reason")
            if not available:
                tooltip = str(reason) if reason else "Administrator role required to run retention."
        else:
            tooltip = "Retention endpoint unavailable."

        if self._retention_backend_disabled:
            available = False
            if self._retention_failure_reason:
                tooltip = self._retention_failure_reason
            elif tooltip is None:
                tooltip = "Retention temporarily disabled after a failed attempt."

        if available:
            tooltip = "Run conversation retention policies now."

        self.retention_button.set_sensitive(available)
        self.retention_button.set_tooltip_text(tooltip)
        return False

    def _on_timeline_overlay_toggled(self, _button: Gtk.CheckButton) -> None:
        self.timeline_area.queue_draw()
        self._update_timeline_status_label()

    def _draw_timeline(self, _area: Gtk.DrawingArea, ctx, width: int, height: int) -> None:
        data = self._timeline_data
        bins: List[int] = data.get("bins") or []
        total = int(data.get("total") or 0)

        ctx.save()
        ctx.set_source_rgba(0.0, 0.0, 0.0, 0.08)
        ctx.rectangle(0, 0, width, height)
        ctx.fill()
        ctx.restore()

        if not bins or total <= 0:
            return

        max_count = max(bins) if bins else 0
        if max_count <= 0:
            return

        padding = 8
        available_width = max(width - 2 * padding, 1)
        available_height = max(height - 2 * padding, 1)
        baseline = padding + available_height
        bar_width = available_width / float(len(bins))

        ctx.save()
        ctx.translate(padding, padding)

        for index, count in enumerate(bins):
            ratio = float(count) / float(max_count)
            bar_height = ratio * available_height
            x = index * bar_width
            ctx.set_source_rgba(0.27, 0.51, 0.85, 0.6)
            ctx.rectangle(x, available_height - bar_height, max(bar_width - 1.0, 1.0), bar_height)
            ctx.fill()

        ctx.set_source_rgba(0.18, 0.18, 0.18, 0.4)
        ctx.move_to(0, available_height)
        ctx.line_to(available_width, available_height)
        ctx.stroke()

        if total > 1:
            switches = data.get("switches") or []
            ctx.set_source_rgba(0.9, 0.36, 0.0, 0.7)
            for fraction in switches:
                x = fraction * available_width
                ctx.move_to(x, 0)
                ctx.line_to(x, available_height)
                ctx.stroke()

        if self.retention_overlay_toggle.get_active():
            retention = data.get("retention") or []
            ctx.set_source_rgba(0.56, 0.0, 0.56, 0.85)
            radius = max(min(bar_width * 0.25, 6), 3)
            for fraction in retention:
                x = fraction * available_width
                ctx.arc(x, available_height - radius, radius, 0, 2 * math.pi)
                ctx.fill()

        if self.summary_overlay_toggle.get_active():
            summaries = data.get("summaries") or []
            ctx.set_source_rgba(0.0, 0.55, 0.37, 0.9)
            size = max(min(bar_width * 0.3, 7), 3)
            for fraction in summaries:
                x = fraction * available_width
                y = size
                ctx.move_to(x, y)
                ctx.line_to(x + size, y + size)
                ctx.line_to(x, y + 2 * size)
                ctx.line_to(x - size, y + size)
                ctx.close_path()
                ctx.fill()

        ctx.restore()

    def _on_playback_restart(self, _button: Gtk.Button) -> None:
        if not self._playback_sequence:
            return
        self._playback_index = None
        self._step_playback(forward=True)

    def _on_playback_previous(self, _button: Gtk.Button) -> None:
        self._step_playback(forward=False)

    def _on_playback_next(self, _button: Gtk.Button) -> None:
        self._step_playback(forward=True)

    def _step_playback(self, *, forward: bool) -> None:
        if not self._playback_sequence:
            return

        if self._playback_index is None:
            index = 0 if forward else len(self._playback_sequence) - 1
        else:
            delta = 1 if forward else -1
            index = self._playback_index + delta
            index = max(0, min(index, len(self._playback_sequence) - 1))

        self._set_playback_index(index)

    def _set_playback_index(self, index: int) -> None:
        if not (0 <= index < len(self._playback_sequence)):
            return
        self._playback_index = index
        target_idx = self._playback_sequence[index]
        self._highlight_widget_by_index(target_idx)
        self._update_playback_display()

    def _highlight_widget_by_index(self, widget_index: int) -> None:
        if not (0 <= widget_index < len(self._message_widgets)):
            self._pending_highlight_index = widget_index
            return

        widget = self._message_widgets[widget_index]
        if self._active_message_widget is not None and self._active_message_widget is not widget:
            self._active_message_widget.remove_css_class("conversation-history-active")
        widget.add_css_class("conversation-history-active")
        widget.grab_focus()
        self._active_message_widget = widget
        self._pending_highlight_index = None

    def _update_playback_display(self) -> None:
        total = len(self._playback_sequence)
        if total == 0:
            self.playback_status_label.set_label("No playback data loaded.")
        else:
            position = (self._playback_index or 0) + 1 if self._playback_index is not None else 0
            if self._playback_index is None:
                self.playback_status_label.set_label(f"Ready: {total} messages")
            else:
                self.playback_status_label.set_label(f"Message {position} of {total}")

        has_previous = bool(self._playback_sequence) and (self._playback_index or 0) > 0
        has_next = bool(self._playback_sequence) and (
            self._playback_index is None or self._playback_index < len(self._playback_sequence) - 1
        )
        self.playback_restart_button.set_sensitive(bool(self._playback_sequence))
        self.playback_previous_button.set_sensitive(has_previous)
        self.playback_next_button.set_sensitive(has_next)

    def _update_timeline(self, messages: List[Mapping[str, Any]]) -> None:
        ordered: List[Dict[str, Any]] = []
        for index, message in enumerate(messages):
            timestamp = self._coerce_datetime(
                message.get("created_at") or message.get("timestamp") or message.get("occurred_at")
            )
            key = self._timeline_sort_key(timestamp, index)
            ordered.append({"key": key, "index": index, "message": message, "timestamp": timestamp})

        ordered.sort(key=lambda entry: entry["key"])

        total = len(ordered)
        bins: List[int] = []
        switches: List[float] = []
        retention_positions: List[float] = []
        summary_positions: List[float] = []

        if total:
            bin_count = min(total, 30)
            bins = [0 for _ in range(bin_count)]
            for position, _entry in enumerate(ordered):
                target_bin = min(int(position * bin_count / max(total, 1)), bin_count - 1)
                bins[target_bin] += 1

            previous_role: Optional[str] = None
            for position, entry in enumerate(ordered):
                message = entry["message"]
                role = str(message.get("role") or "").lower()
                if previous_role is not None and role and role != previous_role:
                    switches.append(self._normalise_fraction(position, total))
                if role:
                    previous_role = role

                metadata = message.get("metadata")
                fraction = self._normalise_fraction(position, total)
                if self._contains_retention_marker(metadata):
                    retention_positions.append(fraction)
                if self._contains_summary_marker(metadata):
                    summary_positions.append(fraction)

        self._timeline_data = {
            "bins": bins,
            "switches": switches,
            "retention": retention_positions,
            "summaries": summary_positions,
            "total": total,
        }

        self._playback_sequence = [entry["index"] for entry in ordered]
        self._playback_index = None
        self._pending_highlight_index = None
        self.timeline_area.queue_draw()
        self._update_timeline_status_label()
        self._update_playback_display()

    def _timeline_sort_key(self, timestamp: Optional[datetime], index: int) -> Tuple[int, float, int]:
        if timestamp is None:
            return (1, float(index), index)
        return (0, float(timestamp.timestamp()), index)

    def _normalise_fraction(self, position: int, total: int) -> float:
        if total <= 1:
            return 0.0
        return position / float(total - 1)

    def _coerce_datetime(self, value: Any) -> Optional[datetime]:
        if isinstance(value, datetime):
            moment = value
        elif isinstance(value, (int, float)):
            moment = datetime.fromtimestamp(float(value), tz=timezone.utc)
        elif isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            try:
                moment = datetime.fromisoformat(text)
            except ValueError:
                return None
        else:
            return None

        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        return moment

    def _contains_retention_marker(self, metadata: Any) -> bool:
        if not isinstance(metadata, Mapping):
            return False
        keys = {key.lower() for key in metadata.keys() if isinstance(key, str)}
        return bool(keys & {"retention", "retention_event", "retention_marker", "history_retention"})

    def _contains_summary_marker(self, metadata: Any) -> bool:
        if not isinstance(metadata, Mapping):
            return False
        for key in ("episodic_summary", "summary", "episode", "memory_summary"):
            value = metadata.get(key)
            if isinstance(value, (str, Mapping)):
                return True
        return False

    def _update_timeline_status_label(self) -> None:
        total = int(self._timeline_data.get("total") or 0)
        if not total:
            if self._selected_id:
                message = "Timeline data unavailable for this selection."
            else:
                message = "Select a conversation to view its timeline."
            self.timeline_status_label.set_label(message)
            self.retention_overlay_toggle.set_active(False)
            self.retention_overlay_toggle.set_sensitive(False)
            self.retention_overlay_toggle.set_tooltip_text("No retention metadata available for this conversation.")
            self.summary_overlay_toggle.set_active(False)
            self.summary_overlay_toggle.set_sensitive(False)
            self.summary_overlay_toggle.set_tooltip_text("No episodic summary metadata available for this conversation.")
        else:
            message = f"Timeline loaded for {total} message{'s' if total != 1 else ''}."
            if not (self._timeline_data.get("retention") or []):
                self.retention_overlay_toggle.set_active(False)
                self.retention_overlay_toggle.set_sensitive(False)
                self.retention_overlay_toggle.set_tooltip_text("No retention metadata available for this conversation.")
            else:
                self.retention_overlay_toggle.set_sensitive(True)
                self.retention_overlay_toggle.set_tooltip_text("Highlight retention events on the timeline.")
            if not (self._timeline_data.get("summaries") or []):
                self.summary_overlay_toggle.set_active(False)
                self.summary_overlay_toggle.set_sensitive(False)
                self.summary_overlay_toggle.set_tooltip_text("No episodic summary metadata available for this conversation.")
            else:
                self.summary_overlay_toggle.set_sensitive(True)
                self.summary_overlay_toggle.set_tooltip_text("Highlight episodic summaries on the timeline.")
            self.timeline_status_label.set_label(message)

    def _on_retention_clicked(self, _button: Gtk.Button) -> None:
        self._set_status("Running conversation retention…")
        self.retention_button.set_sensitive(False)

        try:
            result = self.ATLAS.run_conversation_retention()
        except Exception as exc:  # pragma: no cover - defensive logging upstream
            message = f"Retention failed: {exc}"
            self._set_status(message)
            self._retention_backend_disabled = True
            self._retention_failure_reason = message
            self._update_retention_button_state()
            return

        if result.get("success"):
            stats = result.get("stats") or result.get("result")
            summary = self._format_retention_summary(stats)
            self._set_status(summary)
            self._retention_backend_disabled = False
            self._retention_failure_reason = None
            self._reload_conversations(preserve_selection=True)
        else:
            error = result.get("error") or "Unable to run retention."
            self._set_status(error)
            self._retention_backend_disabled = True
            self._retention_failure_reason = str(error)

        self._update_retention_button_state()

    def _format_retention_summary(self, stats: Any) -> str:
        if not isinstance(stats, Mapping):
            return "Conversation retention completed."

        fragments: List[str] = []
        for section in ("messages", "conversations"):
            counts = stats.get(section)
            if not isinstance(counts, Mapping):
                continue
            components = []
            for key, value in sorted(counts.items()):
                if isinstance(value, (int, float)):
                    components.append(f"{key}={value}")
            if components:
                fragments.append(f"{section.capitalize()}: {', '.join(components)}")

        if not fragments:
            return "Conversation retention completed."

        return f"Retention complete. {'; '.join(fragments)}"

    def _set_status(self, message: str) -> None:
        self.status_label.set_label(message or "")

    def _reset_message_pagination_state(self) -> None:
        self._message_cursor = None
        self._cursor_history = []
        self._message_page_state = {
            "size": self._default_page_size,
            "direction": "backward" if self.direction_toggle.get_active() else "forward",
            "next_cursor": None,
            "previous_cursor": None,
        }
        self._update_page_buttons()

    def _refresh_message_list(self, *, reset_cursor: bool = False) -> None:
        if not self._selected_id:
            return
        self._load_messages(self._selected_id, reset_cursor=reset_cursor)

    def _resolve_page_size(self) -> int:
        text = self.page_size_entry.get_text().strip()
        if not text:
            value = self._default_page_size
        else:
            try:
                value = max(int(text), 1)
            except (TypeError, ValueError):
                value = self._default_page_size
                self._set_status("Invalid page size; using default value.")
        self.page_size_entry.set_text(str(value))
        return value

    def _parse_metadata_filter(self) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        raw_text = self.message_metadata_entry.get_text().strip()
        if not raw_text:
            return None, None
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            return None, f"Metadata filter is not valid JSON: {exc}"
        if not isinstance(parsed, Mapping):
            return None, "Metadata filter must be a JSON object."
        return dict(parsed), None

    def _parse_filter_values(self, entry: Gtk.Entry) -> List[str]:
        raw_text = entry.get_text().strip()
        if not raw_text:
            return []
        values: List[str] = []
        for chunk in raw_text.replace(";", ",").split(","):
            text = chunk.strip()
            if text and text not in values:
                values.append(text)
        return values

    def _on_page_size_activate(self, _entry: Gtk.Entry) -> None:
        self._refresh_message_list(reset_cursor=True)

    def _on_direction_toggled(self, _button: Gtk.CheckButton) -> None:
        self._refresh_message_list(reset_cursor=True)

    def _on_include_deleted_toggled(self, _button: Gtk.CheckButton) -> None:
        self._refresh_message_list(reset_cursor=True)

    def _on_next_page_clicked(self, _button: Gtk.Button) -> None:
        target = self._resolve_next_cursor()
        if not target or not self._selected_id:
            return
        self._cursor_history.append(self._message_cursor)
        self._message_cursor = target
        self._load_messages(self._selected_id)

    def _on_previous_page_clicked(self, _button: Gtk.Button) -> None:
        if not self._cursor_history or not self._selected_id:
            return
        target = self._cursor_history.pop()
        self._message_cursor = target
        self._load_messages(self._selected_id)

    def _on_message_metadata_activate(self, _entry: Gtk.Entry) -> None:
        self._refresh_message_list(reset_cursor=True)

    def _on_message_type_activate(self, _entry: Gtk.Entry) -> None:
        self._refresh_message_list(reset_cursor=True)

    def _on_message_status_activate(self, _entry: Gtk.Entry) -> None:
        self._refresh_message_list(reset_cursor=True)

    def _on_unrealize(self, *_args) -> None:
        remover = getattr(self.ATLAS, "remove_conversation_history_listener", None)
        if callable(remover) and self._conversation_listener is not None:
            remover(self._conversation_listener)
            self._conversation_listener = None
        user_remover = getattr(self.ATLAS, "remove_active_user_change_listener", None)
        if callable(user_remover) and self._active_user_listener is not None:
            user_remover(self._active_user_listener)
            self._active_user_listener = None
        if self._search_timeout_id:
            GLib.source_remove(self._search_timeout_id)
            self._search_timeout_id = 0

    # ------------------------------------------------------------------
    # Search helpers
    # ------------------------------------------------------------------
    def _on_search_changed(self, entry: Gtk.Entry) -> None:
        self._search_query = entry.get_text().strip()
        if not self._search_query:
            self._clear_search()
            return
        self._schedule_search()

    def _on_search_activate(self, entry: Gtk.Entry) -> None:
        self._search_query = entry.get_text().strip()
        if not self._search_query:
            self._clear_search()
            return
        self._execute_search(force=True)

    def _on_search_cleared(self, entry: Gtk.Entry) -> None:
        entry.set_text("")
        self._clear_search()

    def _on_metadata_toggled(self, button: Gtk.CheckButton) -> None:
        active = bool(button.get_active())
        self.metadata_entry.set_visible(active)
        if not active:
            self.metadata_entry.set_text("")
        if self._search_query:
            if active:
                self._schedule_search()
            else:
                self._execute_search(force=True)

    def _on_metadata_changed(self, _entry: Gtk.Entry) -> None:
        if self._search_query:
            self._schedule_search()

    def _on_metadata_activate(self, _entry: Gtk.Entry) -> None:
        if self._search_query:
            self._execute_search(force=True)

    def _schedule_search(self) -> None:
        if self._search_timeout_id:
            GLib.source_remove(self._search_timeout_id)
        self._search_timeout_id = GLib.timeout_add(350, self._execute_search)

    def _clear_search(self) -> None:
        if self._search_timeout_id:
            GLib.source_remove(self._search_timeout_id)
            self._search_timeout_id = 0
        self._search_query = ""
        self._search_active = False
        self._search_results = []
        self._set_status("")
        self._reload_conversations(preserve_selection=True)

    def _execute_search(self, *_args, force: bool = False) -> bool:
        if self._search_timeout_id:
            GLib.source_remove(self._search_timeout_id)
            self._search_timeout_id = 0

        query = self._search_query.strip()
        if not query:
            if force:
                self._clear_search()
            return False

        metadata_filter = None
        if self.metadata_toggle.get_active():
            raw_metadata = self.metadata_entry.get_text().strip()
            if raw_metadata:
                try:
                    parsed = json.loads(raw_metadata)
                except json.JSONDecodeError as exc:
                    self._set_status(f"Metadata filter must be valid JSON: {exc}")
                    return False
                if isinstance(parsed, Mapping):
                    metadata_filter = parsed
                else:
                    self._set_status("Metadata filter must be a JSON object.")
                    return False

        payload: Dict[str, Any] = {"text": query, "limit": 50}
        if metadata_filter is not None:
            payload["metadata"] = metadata_filter

        try:
            response = self.ATLAS.search_conversations(payload)
        except Exception as exc:  # pragma: no cover - defensive logging upstream
            self._set_status(f"Search failed: {exc}")
            return False

        if not isinstance(response, Mapping):
            self._set_status("Unexpected search response.")
            return False

        items = response.get("items")
        if not isinstance(items, list):
            items = []

        self._search_active = True
        prepared = self._prepare_search_records(items)
        self._search_results = prepared
        self._populate_conversation_list(prepared)

        count = len(prepared)
        if count:
            self._set_status(f"Found {count} conversation{'s' if count != 1 else ''} for '{query}'.")
        else:
            self._set_status(f"No conversations matched '{query}'.")

        return False

    def _prepare_search_records(self, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        lookup = {
            str(record.get("id")): record
            for record in self._all_conversations
            if isinstance(record, Mapping) and record.get("id")
        }
        aggregated: Dict[str, Dict[str, Any]] = {}
        order: List[str] = []
        for entry in hits:
            if not isinstance(entry, Mapping):
                continue
            conversation_id = str(entry.get("conversation_id") or "")
            if not conversation_id:
                continue
            message = entry.get("message")
            if not isinstance(message, Mapping):
                continue
            base = lookup.get(conversation_id, {})
            metadata = dict(base.get("metadata") or {}) if isinstance(base, Mapping) else {}
            created_at = (
                message.get("created_at")
                or message.get("timestamp")
                or base.get("created_at")
                if isinstance(base, Mapping)
                else None
            )
            score = float(entry.get("score") or 0.0)
            snippet = self._format_content(message.get("content"))
            record = {
                "id": conversation_id,
                "metadata": metadata,
                "created_at": created_at,
                "__search_hit__": {
                    "message": dict(message),
                    "score": score,
                    "snippet": snippet,
                },
            }
            title = base.get("title") if isinstance(base, Mapping) else None
            if isinstance(title, str):
                record["title"] = title
            if conversation_id not in aggregated or score > aggregated[conversation_id]["__search_hit__"]["score"]:
                aggregated[conversation_id] = record
            if conversation_id not in order:
                order.append(conversation_id)
        return [aggregated[identifier] for identifier in order]

