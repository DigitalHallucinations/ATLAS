"""GTK task management workspace used by the sidebar."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import Gdk, GLib, Gtk

from ATLAS.utils import normalize_sequence

from .widgets import clear_container, create_badge, create_badge_container, sync_badge_section

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _TaskEntry:
    """Normalized task record returned by the ATLAS server."""

    task_id: str
    title: str
    description: str
    status: str
    priority: int
    owner_id: Optional[str]
    conversation_id: Optional[str]
    persona: Optional[str]
    metadata: Dict[str, Any]
    required_skills: Tuple[str, ...]
    required_tools: Tuple[str, ...]
    acceptance_criteria: Tuple[str, ...]
    dependencies: Tuple[Any, ...]
    created_at: Optional[str]
    updated_at: Optional[str]
    due_at: Optional[str]


class TaskManagement:
    """Controller responsible for rendering the task management workspace."""

    _PERSONA_UNASSIGNED = "__unassigned__"
    _STATUS_SEQUENCE: Tuple[str, ...] = (
        "draft",
        "ready",
        "in_progress",
        "review",
        "done",
        "cancelled",
    )
    _FETCH_PAGE_SIZE = 50
    _ADVANCE_ACTIONS: Mapping[str, Tuple[Tuple[str, str], ...]] = {
        "draft": (("Mark ready", "ready"), ("Cancel task", "cancelled")),
        "ready": (("Start work", "in_progress"), ("Cancel task", "cancelled")),
        "in_progress": (("Request review", "review"), ("Cancel task", "cancelled")),
        "review": (("Mark complete", "done"), ("Cancel task", "cancelled")),
        "done": tuple(),
        "cancelled": tuple(),
    }

    def __init__(self, atlas: Any, parent_window: Any) -> None:
        self.ATLAS = atlas
        self.parent_window = parent_window

        self._widget: Optional[Gtk.Widget] = None
        self._task_list: Optional[Gtk.ListBox] = None
        self._view_stack: Optional[Gtk.Stack] = None
        self._persona_filter_combo: Optional[Gtk.ComboBoxText] = None
        self._status_filter_combo: Optional[Gtk.ComboBoxText] = None
        self._owner_filter_combo: Optional[Gtk.ComboBoxText] = None
        self._conversation_filter_combo: Optional[Gtk.ComboBoxText] = None
        self._owner_option_lookup: List[Optional[str]] = []
        self._conversation_option_lookup: List[Optional[str]] = []
        self._search_entry: Optional[Gtk.Entry] = None
        self._search_apply_button: Optional[Gtk.Button] = None
        self._search_clear_button: Optional[Gtk.Button] = None
        self._metadata_filter_container: Optional[Gtk.Box] = None
        self._metadata_filter_rows: List[Dict[str, Gtk.Widget]] = []
        self._title_label: Optional[Gtk.Label] = None
        self._status_label: Optional[Gtk.Label] = None
        self._description_label: Optional[Gtk.Label] = None
        self._persona_label: Optional[Gtk.Label] = None
        self._owner_label: Optional[Gtk.Label] = None
        self._conversation_label: Optional[Gtk.Label] = None
        self._priority_label: Optional[Gtk.Label] = None
        self._due_label: Optional[Gtk.Label] = None
        self._updated_label: Optional[Gtk.Label] = None
        self._required_skills_box: Optional[Gtk.Widget] = None
        self._required_tools_box: Optional[Gtk.Widget] = None
        self._acceptance_box: Optional[Gtk.Widget] = None
        self._dependencies_box: Optional[Gtk.Widget] = None
        self._action_box: Optional[Gtk.Box] = None
        self._primary_action_button: Optional[Gtk.Button] = None
        self._secondary_action_button: Optional[Gtk.Button] = None

        self._entries: List[_TaskEntry] = []
        self._entry_lookup: Dict[str, _TaskEntry] = {}
        self._display_entries: List[_TaskEntry] = []
        self._row_lookup: Dict[str, Gtk.ListBoxRow] = {}
        self._board_container: Optional[Gtk.Box] = None
        self._board_columns: Dict[str, Gtk.Box] = {}
        self._board_column_wrappers: Dict[str, Gtk.Widget] = {}
        self._board_status_labels: Dict[str, Gtk.Label] = {}
        self._board_card_lookup: Dict[str, Gtk.Widget] = {}
        self._active_board_card: Optional[Gtk.Widget] = None
        self._drag_in_progress_card: Optional[Gtk.Widget] = None
        self._persona_filter: Optional[str] = None
        self._status_filter: Optional[str] = None
        self._persona_option_lookup: List[Optional[str]] = []
        self._status_option_lookup: List[Optional[str]] = []
        self._active_task: Optional[str] = None
        self._pending_focus_task: Optional[str] = None
        self._refresh_pending = False
        self._bus_subscriptions: List[Any] = []
        self._catalog_entries: List[Mapping[str, Any]] = []
        self._catalog_container: Optional[Gtk.Widget] = None
        self._catalog_hint_label: Optional[Gtk.Label] = None
        self._search_text: str = ""
        self._metadata_filter: Dict[str, Any] = {}
        self._owner_filter: Optional[str] = None
        self._conversation_filter: Optional[str] = None
        self._suppress_owner_filter_event = False
        self._suppress_conversation_filter_event = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_embeddable_widget(self) -> Gtk.Widget:
        """Return the workspace widget, creating it on first use."""

        if self._widget is None:
            self._widget = self._build_workspace()
            self._subscribe_to_bus()
            self._refresh_state()
        else:
            self._refresh_state()
        return self._widget

    def focus_task(self, task_id: str) -> bool:
        """Ensure ``task_id`` is visible and selected."""

        identifier = str(task_id or "").strip()
        if not identifier:
            return False

        if identifier not in self._entry_lookup:
            self._pending_focus_task = identifier
            self._refresh_state()
        if identifier not in self._entry_lookup:
            return False

        self._pending_focus_task = identifier
        self._select_task(identifier)
        return True

    def _on_close_request(self) -> None:
        """Clean up resources when the workspace tab is closed."""

        for subscription in list(self._bus_subscriptions):
            try:
                subscription.cancel()
            except Exception:  # pragma: no cover - defensive cleanup
                logger.debug("Failed to cancel task bus subscription", exc_info=True)
            finally:
                self._bus_subscriptions.remove(subscription)

    # ------------------------------------------------------------------
    # Workspace construction
    # ------------------------------------------------------------------
    def _build_workspace(self) -> Gtk.Widget:
        root = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        for setter_name in ("set_margin_top", "set_margin_bottom", "set_margin_start", "set_margin_end"):
            setter = getattr(root, setter_name, None)
            if callable(setter):
                try:
                    setter(12)
                except Exception:  # pragma: no cover - GTK fallback
                    continue

        sidebar = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        sidebar.set_hexpand(False)
        sidebar.set_vexpand(True)
        root.append(sidebar)

        filter_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        sidebar.append(filter_box)

        persona_combo = Gtk.ComboBoxText()
        persona_combo.connect("changed", self._on_persona_filter_changed)
        filter_box.append(persona_combo)
        self._persona_filter_combo = persona_combo

        status_combo = Gtk.ComboBoxText()
        status_combo.connect("changed", self._on_status_filter_changed)
        filter_box.append(status_combo)
        self._status_filter_combo = status_combo
        self._populate_status_filter_options()

        owner_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        owner_label = Gtk.Label(label="Owner")
        owner_label.set_xalign(0.0)
        owner_box.append(owner_label)
        owner_combo = Gtk.ComboBoxText()
        owner_combo.set_hexpand(True)
        owner_combo.connect("changed", self._on_owner_filter_changed)
        owner_box.append(owner_combo)
        filter_box.append(owner_box)
        self._owner_filter_combo = owner_combo

        conversation_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        conversation_label = Gtk.Label(label="Conversation")
        conversation_label.set_xalign(0.0)
        conversation_box.append(conversation_label)
        conversation_combo = Gtk.ComboBoxText()
        conversation_combo.set_hexpand(True)
        conversation_combo.connect("changed", self._on_conversation_filter_changed)
        conversation_box.append(conversation_combo)
        filter_box.append(conversation_box)
        self._conversation_filter_combo = conversation_combo

        search_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        filter_box.append(search_row)

        search_entry = Gtk.Entry()
        search_entry.set_placeholder_text("Search tasks")
        search_entry.set_hexpand(True)
        search_entry.connect("activate", self._on_search_activate)
        search_row.append(search_entry)
        self._search_entry = search_entry

        apply_button = Gtk.Button(label="Apply")
        apply_button.connect("clicked", self._on_search_apply)
        search_row.append(apply_button)
        self._search_apply_button = apply_button

        clear_button = Gtk.Button(label="Clear")
        clear_button.connect("clicked", self._on_search_clear)
        search_row.append(clear_button)
        self._search_clear_button = clear_button

        metadata_section = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        filter_box.append(metadata_section)

        metadata_label = Gtk.Label(label="Metadata filters")
        metadata_label.set_xalign(0.0)
        metadata_section.append(metadata_label)

        metadata_container = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        metadata_container.set_hexpand(True)
        metadata_section.append(metadata_container)
        self._metadata_filter_container = metadata_container

        add_metadata_button = Gtk.Button(label="Add filter")
        add_metadata_button.connect("clicked", self._on_add_metadata_filter)
        metadata_section.append(add_metadata_button)

        view_stack = Gtk.Stack()
        view_stack.set_hexpand(False)
        view_stack.set_vexpand(True)
        if hasattr(view_stack, "set_transition_type"):
            view_stack.set_transition_type(Gtk.StackTransitionType.CROSSFADE)
        sidebar.append(view_stack)
        self._view_stack = view_stack

        view_switcher = Gtk.StackSwitcher()
        view_switcher.set_stack(view_stack)
        view_switcher.set_hexpand(True)
        if hasattr(view_switcher, "set_halign"):
            view_switcher.set_halign(Gtk.Align.FILL)
        filter_box.append(view_switcher)

        listbox = Gtk.ListBox()
        listbox.set_selection_mode(Gtk.SelectionMode.SINGLE)
        listbox.connect("row-selected", self._on_row_selected)
        list_scroller = Gtk.ScrolledWindow()
        list_scroller.set_hexpand(False)
        list_scroller.set_vexpand(True)
        list_scroller.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        list_scroller.set_child(listbox)
        view_stack.add_titled(list_scroller, "list", "List")
        self._task_list = listbox

        board_view = self._build_board_view()
        view_stack.add_titled(board_view, "board", "Board")
        view_stack.set_visible_child_name("list")

        detail_panel = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        detail_panel.set_hexpand(True)
        detail_panel.set_vexpand(True)
        root.append(detail_panel)

        title_label = Gtk.Label()
        title_label.set_xalign(0.0)
        title_label.set_wrap(True)
        title_label.add_css_class("heading") if hasattr(title_label, "add_css_class") else None
        detail_panel.append(title_label)
        self._title_label = title_label

        status_label = Gtk.Label()
        status_label.set_xalign(0.0)
        detail_panel.append(status_label)
        self._status_label = status_label

        info_grid = Gtk.Grid(column_spacing=12, row_spacing=6)
        detail_panel.append(info_grid)

        self._persona_label = self._add_info_row(info_grid, 0, "Persona")
        self._owner_label = self._add_info_row(info_grid, 1, "Owner")
        self._conversation_label = self._add_info_row(info_grid, 2, "Conversation")
        self._priority_label = self._add_info_row(info_grid, 3, "Priority")
        self._due_label = self._add_info_row(info_grid, 4, "Due date")
        self._updated_label = self._add_info_row(info_grid, 5, "Last updated")

        action_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        detail_panel.append(action_box)
        self._action_box = action_box

        primary_button = Gtk.Button()
        primary_button.connect("clicked", self._on_action_button_clicked)
        action_box.append(primary_button)
        self._primary_action_button = primary_button

        secondary_button = Gtk.Button()
        secondary_button.connect("clicked", self._on_action_button_clicked)
        action_box.append(secondary_button)
        self._secondary_action_button = secondary_button

        description_header = Gtk.Label(label="Description")
        description_header.set_xalign(0.0)
        detail_panel.append(description_header)

        description_label = Gtk.Label()
        description_label.set_wrap(True)
        description_label.set_xalign(0.0)
        detail_panel.append(description_label)
        self._description_label = description_label

        self._required_skills_box = self._add_badge_section(detail_panel, "Required skills")
        self._required_tools_box = self._add_badge_section(detail_panel, "Required tools")
        self._acceptance_box = self._add_badge_section(detail_panel, "Acceptance criteria")
        self._dependencies_box = self._add_badge_section(detail_panel, "Dependencies")

        detail_panel.append(Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL))

        expander_cls = getattr(Gtk, "Expander", None)
        if expander_cls is not None:
            catalog_wrapper = expander_cls(label="Reusable task catalog")
            catalog_wrapper.set_hexpand(True)
            catalog_wrapper.set_expanded(False)
            detail_panel.append(catalog_wrapper)
            catalog_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
            for setter_name in (
                "set_margin_top",
                "set_margin_bottom",
                "set_margin_start",
                "set_margin_end",
            ):
                setter = getattr(catalog_box, setter_name, None)
                if callable(setter):
                    try:
                        setter(6)
                    except Exception:  # pragma: no cover - GTK fallback
                        continue
            setter = getattr(catalog_wrapper, "set_child", None)
            if callable(setter):
                setter(catalog_box)
            else:  # pragma: no cover - GTK fallback
                getattr(catalog_wrapper, "add", lambda *_args: None)(catalog_box)
        else:
            catalog_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
            detail_panel.append(catalog_box)
            header = Gtk.Label(label="Reusable task catalog")
            header.set_xalign(0.0)
            header.add_css_class("heading") if hasattr(header, "add_css_class") else None
            catalog_box.append(header)

        hint_label = Gtk.Label()
        hint_label.set_xalign(0.0)
        hint_label.set_wrap(True)
        hint_label.add_css_class("dim-label") if hasattr(hint_label, "add_css_class") else None
        catalog_box.append(hint_label)
        self._catalog_hint_label = hint_label

        catalog_scroller = Gtk.ScrolledWindow()
        catalog_scroller.set_hexpand(True)
        catalog_scroller.set_vexpand(False)
        catalog_scroller.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        catalog_scroller.set_min_content_height(160)
        catalog_box.append(catalog_scroller)

        catalog_container = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        catalog_container.set_hexpand(True)
        catalog_scroller.set_child(catalog_container)
        self._catalog_container = catalog_container

        detail_panel.connect("destroy", lambda *_args: self._on_close_request())
        self._update_search_widgets()
        return root

    def _add_info_row(self, grid: Gtk.Grid, row: int, title: str) -> Gtk.Label:
        label_widget = Gtk.Label(label=title)
        label_widget.set_xalign(0.0)
        label_widget.set_yalign(0.5)
        grid.attach(label_widget, 0, row, 1, 1)

        value_label = Gtk.Label()
        value_label.set_xalign(0.0)
        value_label.set_yalign(0.5)
        value_label.set_wrap(True)
        grid.attach(value_label, 1, row, 1, 1)
        return value_label

    def _add_badge_section(self, container: Gtk.Box, title: str) -> Gtk.Widget:
        header = Gtk.Label(label=title)
        header.set_xalign(0.0)
        container.append(header)

        badge_container = create_badge_container()
        badge_container.set_visible(False)
        container.append(badge_container)
        return badge_container

    # ------------------------------------------------------------------
    # Board construction and interaction
    # ------------------------------------------------------------------
    def _build_board_view(self) -> Gtk.Widget:
        scroller = Gtk.ScrolledWindow()
        scroller.set_hexpand(False)
        scroller.set_vexpand(True)
        scroller.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        if hasattr(scroller, "add_css_class"):
            scroller.add_css_class("kanban-board")

        container = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        container.set_margin_top(6)
        container.set_margin_bottom(6)
        container.set_margin_start(6)
        container.set_margin_end(6)
        container.set_hexpand(True)
        container.set_vexpand(False)
        scroller.set_child(container)

        self._board_container = container
        self._board_columns.clear()
        self._board_column_wrappers.clear()
        self._board_status_labels.clear()
        for status in self._STATUS_SEQUENCE:
            self._ensure_board_column(status)
        return scroller

    def _ensure_board_column(self, status: str) -> Gtk.Box:
        container = self._board_container
        if container is None:
            raise RuntimeError("Board view has not been initialised")

        if status in self._board_columns:
            return self._board_columns[status]

        wrapper = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        wrapper.set_hexpand(False)
        wrapper.set_vexpand(True)
        wrapper.set_size_request(240, -1)
        if hasattr(wrapper, "add_css_class"):
            wrapper.add_css_class("kanban-column")

        header = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        header.set_hexpand(True)
        if hasattr(header, "add_css_class"):
            header.add_css_class("kanban-column-header")
        title = Gtk.Label(label=self._format_status(status))
        title.set_xalign(0.0)
        title.set_hexpand(True)
        header.append(title)

        count_label = Gtk.Label(label="0")
        count_label.set_xalign(1.0)
        if hasattr(count_label, "add_css_class"):
            count_label.add_css_class("kanban-column-count")
        header.append(count_label)

        column_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        column_box.set_hexpand(True)
        column_box.set_vexpand(False)
        if hasattr(column_box, "add_css_class"):
            column_box.add_css_class("kanban-column-body")

        wrapper.append(header)
        wrapper.append(column_box)
        container.append(wrapper)

        drop_target = Gtk.DropTarget.new(str, Gdk.DragAction.MOVE)
        drop_target.connect("enter", self._on_board_drag_enter, status)
        drop_target.connect("leave", self._on_board_drag_leave, status)
        drop_target.connect("drop", self._on_board_drop, status)
        column_box.add_controller(drop_target)

        self._board_columns[status] = column_box
        self._board_column_wrappers[status] = wrapper
        self._board_status_labels[status] = count_label
        return column_box

    def _rebuild_task_board(self, entries: Sequence[_TaskEntry]) -> None:
        if not self._board_container:
            return

        for column in self._board_columns.values():
            clear_container(column)
        counts: Dict[str, int] = {status: 0 for status in self._board_columns}
        self._board_card_lookup.clear()

        for entry in entries:
            column = self._ensure_board_column(entry.status)
            card = self._create_board_card(entry)
            column.append(card)
            counts[entry.status] = counts.get(entry.status, 0) + 1
            self._board_card_lookup[entry.task_id] = card

        for status, label in self._board_status_labels.items():
            label.set_text(str(counts.get(status, 0)))

        self._update_board_selection(self._active_task)

    def _create_board_card(self, entry: _TaskEntry) -> Gtk.Widget:
        card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        card.set_margin_top(4)
        card.set_margin_bottom(4)
        card.set_margin_start(4)
        card.set_margin_end(4)
        card.set_hexpand(True)
        card._task_id = entry.task_id  # type: ignore[attr-defined]
        if hasattr(card, "add_css_class"):
            card.add_css_class("kanban-card")

        title = Gtk.Label(label=entry.title or "Untitled task")
        title.set_xalign(0.0)
        title.set_wrap(True)
        if hasattr(title, "add_css_class"):
            title.add_css_class("kanban-card-title")
        card.append(title)

        persona_label = Gtk.Label(label=entry.persona or "Unassigned")
        persona_label.set_xalign(0.0)
        if entry.persona is None and hasattr(persona_label, "add_css_class"):
            persona_label.add_css_class("dim-label")
        if hasattr(persona_label, "add_css_class"):
            persona_label.add_css_class("kanban-card-meta")
        card.append(persona_label)

        if entry.owner_id:
            owner_label = Gtk.Label(label=f"Owner: {entry.owner_id}")
            owner_label.set_xalign(0.0)
            if hasattr(owner_label, "add_css_class"):
                owner_label.add_css_class("kanban-card-meta")
            card.append(owner_label)

        summary = Gtk.Label(label=entry.description)
        summary.set_xalign(0.0)
        summary.set_wrap(True)
        if hasattr(summary, "add_css_class"):
            summary.add_css_class("kanban-card-summary")
        card.append(summary)

        gesture = Gtk.GestureClick()
        gesture.connect("pressed", self._on_board_card_pressed, entry.task_id)
        card.add_controller(gesture)

        drag_source = Gtk.DragSource()
        drag_source.set_actions(Gdk.DragAction.MOVE)
        drag_source.connect("prepare", self._on_board_drag_prepare, card)
        drag_source.connect("drag-begin", self._on_board_drag_begin, card)
        drag_source.connect("drag-end", self._on_board_drag_end, card)
        card.add_controller(drag_source)
        return card

    def _on_board_card_pressed(
        self, gesture: Gtk.GestureClick, _n_press: int, _x: float, _y: float, task_id: str
    ) -> None:
        try:
            gesture.set_state(Gtk.EventSequenceState.CLAIMED)
        except Exception:  # pragma: no cover - GTK fallback
            pass
        if task_id:
            self._select_task(task_id)

    def _on_board_drag_prepare(
        self, _source: Gtk.DragSource, _x: float, _y: float, card: Gtk.Widget
    ) -> Optional[Gdk.ContentProvider]:
        task_id = getattr(card, "_task_id", None)
        if isinstance(task_id, str):
            return Gdk.ContentProvider.new_for_value(task_id)
        return None

    def _on_board_drag_begin(
        self, _source: Gtk.DragSource, _drag: Gdk.Drag, card: Gtk.Widget
    ) -> None:
        self._drag_in_progress_card = card
        if hasattr(card, "add_css_class"):
            card.add_css_class("kanban-card-dragging")

    def _on_board_drag_end(self, _source: Gtk.DragSource, _drag: Gdk.Drag, card: Gtk.Widget) -> None:
        if self._drag_in_progress_card is card:
            self._drag_in_progress_card = None
        if hasattr(card, "remove_css_class"):
            card.remove_css_class("kanban-card-dragging")

    def _on_board_drag_enter(
        self, _target: Gtk.DropTarget, _x: float, _y: float, status: str
    ) -> Gdk.DragAction:
        self._set_column_drop_state(status, True)
        return Gdk.DragAction.MOVE

    def _on_board_drag_leave(self, _target: Gtk.DropTarget, status: str) -> None:
        self._set_column_drop_state(status, False)

    def _on_board_drop(
        self, _target: Gtk.DropTarget, value: Any, _x: float, _y: float, status: str
    ) -> bool:
        self._set_column_drop_state(status, False)
        task_id = value if isinstance(value, str) else None
        if task_id is None:
            get_string = getattr(value, "get_string", None)
            if callable(get_string):
                try:
                    task_id = str(get_string())
                except Exception:  # pragma: no cover - GTK fallback
                    task_id = None
        if not task_id:
            return False
        entry = self._entry_lookup.get(task_id)
        if entry is None:
            return False
        if entry.status == status:
            return True
        self._transition_task(entry, status)
        return True

    def _set_column_drop_state(self, status: str, active: bool) -> None:
        wrapper = self._board_column_wrappers.get(status)
        if wrapper is None:
            return
        if hasattr(wrapper, "add_css_class") and hasattr(wrapper, "remove_css_class"):
            if active:
                wrapper.add_css_class("kanban-drop-target")
            else:
                wrapper.remove_css_class("kanban-drop-target")

    def _update_board_selection(self, task_id: Optional[str]) -> None:
        desired = self._board_card_lookup.get(task_id) if task_id else None
        current = self._active_board_card
        if current is desired:
            return
        if current is not None and hasattr(current, "remove_css_class"):
            current.remove_css_class("kanban-card-active")
        self._active_board_card = desired
        if desired is not None and hasattr(desired, "add_css_class"):
            desired.add_css_class("kanban-card-active")

    # ------------------------------------------------------------------
    # Data loading and synchronization
    # ------------------------------------------------------------------
    def _refresh_state(self) -> None:
        entries = self._load_task_entries()
        self._entries = entries
        self._entry_lookup = {entry.task_id: entry for entry in entries}
        self._populate_persona_filter_options()
        self._populate_owner_filter_options()
        self._populate_conversation_filter_options()
        self._update_search_widgets()
        self._rebuild_task_list()
        self._refresh_task_catalog()

    def _schedule_refresh(self) -> None:
        if self._refresh_pending:
            return
        self._refresh_pending = True

        def _callback() -> bool:
            self._refresh_pending = False
            if self._widget is None:
                return False
            self._refresh_state()
            return False

        try:
            GLib.idle_add(_callback)
        except Exception:  # pragma: no cover - fallback when GLib idle is unavailable
            _callback()

    def _load_task_entries(self) -> List[_TaskEntry]:
        atlas = getattr(self, "ATLAS", None)
        search_helper = getattr(atlas, "search_tasks", None)
        server = getattr(atlas, "server", None)
        list_tasks = getattr(server, "list_tasks", None) if server is not None else None

        use_helper = callable(search_helper)
        if not use_helper and not callable(list_tasks):
            logger.warning("ATLAS task search helpers are unavailable; returning empty workspace")
            return []

        base_filters: Dict[str, Any] = {}
        if self._status_filter:
            base_filters["status"] = self._status_filter
        if self._owner_filter:
            base_filters["owner_id"] = self._owner_filter
        if self._conversation_filter:
            base_filters["conversation_id"] = self._conversation_filter

        text_query = str(self._search_text or "").strip()
        metadata_filter = {
            str(key): value
            for key, value in (self._metadata_filter or {}).items()
            if str(key).strip()
        }
        if not metadata_filter:
            metadata_filter = {}

        use_search_filters = bool(text_query or metadata_filter)

        aggregated: List[Mapping[str, Any]] = []
        cursor: Optional[str] = None
        offset = 0
        iterations = 0
        max_iterations = 50
        page_size = self._FETCH_PAGE_SIZE
        context = {"tenant_id": getattr(atlas, "tenant_id", "default")}

        while True:
            iterations += 1
            if iterations > max_iterations:
                break

            request_params: Dict[str, Any] = dict(base_filters)
            if use_search_filters:
                if text_query:
                    request_params["text"] = text_query
                if metadata_filter:
                    request_params["metadata"] = metadata_filter
                request_params["limit"] = page_size
                request_params["offset"] = offset
            else:
                request_params["limit"] = page_size
                if cursor:
                    request_params["cursor"] = cursor

            try:
                if use_helper:
                    payload = {key: value for key, value in request_params.items() if value not in (None, "")}
                    response = search_helper(**payload)
                else:
                    list_params = {key: value for key, value in request_params.items() if key not in {"limit"}}
                    list_params["page_size"] = request_params.get("limit", page_size)
                    response = list_tasks(list_params, context=context)
            except Exception as exc:
                logger.error("Failed to load task list: %s", exc, exc_info=True)
                self._handle_backend_error("Unable to load tasks from ATLAS.")
                return []

            mapping_response = response if isinstance(response, Mapping) else None
            items_payload: Any
            if isinstance(mapping_response, Mapping):
                items_payload = mapping_response.get("items")
            else:
                items_payload = response

            if isinstance(items_payload, Mapping):
                iterator: Iterable[Any] = items_payload.values()
            else:
                iterator = items_payload if isinstance(items_payload, Iterable) else []

            batch: List[Mapping[str, Any]] = []
            if isinstance(iterator, Iterable) and not isinstance(iterator, (str, bytes)):
                for entry in iterator:
                    if isinstance(entry, Mapping):
                        batch.append(dict(entry))

            aggregated.extend(batch)

            if use_search_filters:
                offset += len(batch)
                total_count: Optional[int] = None
                if isinstance(mapping_response, Mapping):
                    count_value = mapping_response.get("count")
                    if isinstance(count_value, int):
                        total_count = count_value
                if not batch:
                    break
                if total_count is not None and offset >= total_count:
                    break
                if len(batch) < page_size:
                    break
            else:
                next_cursor: Optional[str] = None
                if isinstance(mapping_response, Mapping):
                    page_info = mapping_response.get("page")
                    if isinstance(page_info, Mapping):
                        raw_cursor = page_info.get("next_cursor")
                        if raw_cursor:
                            next_cursor = str(raw_cursor)
                if not batch or not next_cursor:
                    break
                cursor = next_cursor

        raw_entries: List[Mapping[str, Any]] = aggregated
        if use_search_filters and not use_helper:
            text_lower = text_query.lower()
            metadata_copy = dict(metadata_filter)

            filtered: List[Mapping[str, Any]] = []
            for entry in raw_entries:
                if text_lower:
                    haystack = " ".join(
                        [
                            str(entry.get("title") or ""),
                            str(entry.get("description") or ""),
                        ]
                    ).lower()
                    if text_lower not in haystack:
                        continue
                if metadata_copy:
                    metadata_block = entry.get("metadata") if isinstance(entry.get("metadata"), Mapping) else {}
                    if not all(metadata_block.get(key) == value for key, value in metadata_copy.items()):
                        continue
                filtered.append(entry)
            raw_entries = filtered

        entries: List[_TaskEntry] = []
        for raw_entry in raw_entries:
            normalized = self._normalize_entry(raw_entry)
            if normalized is not None:
                entries.append(normalized)
        return entries

    def _load_task_catalog(self) -> List[Mapping[str, Any]]:
        server = getattr(self.ATLAS, "server", None)
        get_catalog = getattr(server, "get_task_catalog", None)
        if not callable(get_catalog):
            get_catalog = getattr(self.ATLAS, "get_task_catalog", None)
        if not callable(get_catalog):
            return []

        persona_filter = self._persona_filter
        persona_param: Optional[str]
        if persona_filter == self._PERSONA_UNASSIGNED:
            persona_param = "shared"
        else:
            persona_param = persona_filter

        try:
            response = get_catalog(persona=persona_param)
        except Exception as exc:
            logger.error("Failed to load task catalog: %s", exc, exc_info=True)
            return []

        items: List[Mapping[str, Any]] = []
        if isinstance(response, Mapping):
            payload = response.get("tasks") or response.get("items")
        else:
            payload = response

        if isinstance(payload, Mapping):
            iterator = payload.values()
        else:
            iterator = payload

        if isinstance(iterator, Iterable) and not isinstance(iterator, (str, bytes)):
            for entry in iterator:
                if isinstance(entry, Mapping):
                    items.append(dict(entry))
        return items

    def _normalize_entry(self, payload: Any) -> Optional[_TaskEntry]:
        if not isinstance(payload, Mapping):
            return None

        raw_id = payload.get("id")
        identifier = str(raw_id).strip() if raw_id else ""
        if not identifier:
            return None

        metadata = payload.get("metadata")
        if isinstance(metadata, Mapping):
            metadata_dict = dict(metadata)
        else:
            metadata_dict = {}

        token_normalizer = lambda value: normalize_sequence(
            value,
            as_tuple=True,
            transform=lambda item: str(item).strip(),
            filter_falsy=True,
            accept_scalar=False,
        )

        def _normalize_dependencies(value: Any) -> Tuple[Any, ...]:
            dependencies: List[Any] = []
            candidate: Iterable[Any]
            if isinstance(value, Mapping):
                candidate = value.values()
            else:
                candidate = value
            if isinstance(candidate, Iterable) and not isinstance(candidate, (str, bytes)):
                for entry in candidate:
                    if isinstance(entry, Mapping):
                        record: Dict[str, Any] = {}
                        raw_dep_id = entry.get("id") or entry.get("task_id")
                        if raw_dep_id:
                            dep_id = str(raw_dep_id).strip()
                            if dep_id:
                                record["id"] = dep_id
                        title = entry.get("title") or entry.get("name")
                        if title:
                            record["title"] = str(title).strip()
                        status = entry.get("status")
                        if status:
                            record["status"] = str(status).strip().lower()
                        if record:
                            dependencies.append(record)
                    else:
                        text = str(entry).strip()
                        if text:
                            dependencies.append(text)
            return tuple(dependencies)

        description_source = payload.get("description") or metadata_dict.get("description")
        description = str(description_source).strip() if description_source else "No description provided."

        persona_value = metadata_dict.get("persona")
        persona = str(persona_value).strip() if persona_value else None

        return _TaskEntry(
            task_id=identifier,
            title=str(payload.get("title") or metadata_dict.get("title") or "Untitled task").strip(),
            description=description,
            status=str(payload.get("status") or "draft").strip().lower(),
            priority=int(payload.get("priority") or 0),
            owner_id=str(payload.get("owner_id")).strip() if payload.get("owner_id") else None,
            conversation_id=str(payload.get("conversation_id")).strip() if payload.get("conversation_id") else None,
            persona=persona,
            metadata=metadata_dict,
            required_skills=token_normalizer(metadata_dict.get("required_skills")),
            required_tools=token_normalizer(metadata_dict.get("required_tools")),
            acceptance_criteria=token_normalizer(metadata_dict.get("acceptance_criteria")),
            dependencies=_normalize_dependencies(metadata_dict.get("dependencies")),
            created_at=self._coerce_timestamp(payload.get("created_at")),
            updated_at=self._coerce_timestamp(payload.get("updated_at")),
            due_at=self._coerce_timestamp(payload.get("due_at")),
        )

    @staticmethod
    def _coerce_timestamp(value: Any) -> Optional[str]:
        if not value:
            return None
        text = str(value).strip()
        if not text:
            return None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return text
        return parsed.isoformat()

    # ------------------------------------------------------------------
    # Filtering helpers
    # ------------------------------------------------------------------
    def _populate_persona_filter_options(self) -> None:
        combo = self._persona_filter_combo
        if combo is None:
            return
        if hasattr(combo, "remove_all"):
            combo.remove_all()

        personas = sorted({entry.persona for entry in self._entries if entry.persona})
        has_unassigned = any(entry.persona is None for entry in self._entries)

        options: List[Tuple[Optional[str], str]] = [(None, "All personas")]
        options.extend((name, name) for name in personas)
        if has_unassigned:
            options.append((self._PERSONA_UNASSIGNED, "Unassigned"))

        self._persona_option_lookup = []
        for index, (value, label) in enumerate(options):
            combo.append_text(label)
            self._persona_option_lookup.append(value)
            if value == self._persona_filter:
                combo.set_active(index)

        if combo.get_active_text() is None:
            combo.set_active(0)
            self._persona_filter = None

    def _populate_status_filter_options(self) -> None:
        combo = self._status_filter_combo
        if combo is None:
            return
        if hasattr(combo, "remove_all"):
            combo.remove_all()

        options: List[Tuple[Optional[str], str]] = [(None, "All statuses")]
        options.extend((status, self._format_status(status)) for status in self._STATUS_SEQUENCE)

        self._status_option_lookup = []
        for index, (value, label) in enumerate(options):
            combo.append_text(label)
            self._status_option_lookup.append(value)
            if value == self._status_filter:
                combo.set_active(index)

        if combo.get_active_text() is None:
            combo.set_active(0)
            self._status_filter = None

    def _populate_owner_filter_options(self) -> None:
        combo = self._owner_filter_combo
        if combo is None:
            return
        if hasattr(combo, "remove_all"):
            combo.remove_all()

        owners = sorted({entry.owner_id for entry in self._entries if entry.owner_id})
        options: List[Tuple[Optional[str], str]] = [(None, "All owners")]
        options.extend((owner, owner) for owner in owners)

        self._owner_option_lookup = []
        self._suppress_owner_filter_event = True
        for index, (value, label) in enumerate(options):
            combo.append_text(label)
            self._owner_option_lookup.append(value)
            if value == self._owner_filter:
                combo.set_active(index)

        if combo.get_active_text() is None:
            combo.set_active(0)
            self._owner_filter = None
        self._suppress_owner_filter_event = False

    def _populate_conversation_filter_options(self) -> None:
        combo = self._conversation_filter_combo
        if combo is None:
            return
        if hasattr(combo, "remove_all"):
            combo.remove_all()

        conversations = sorted({entry.conversation_id for entry in self._entries if entry.conversation_id})
        options: List[Tuple[Optional[str], str]] = [(None, "All conversations")]
        options.extend((conversation, conversation) for conversation in conversations)

        self._conversation_option_lookup = []
        self._suppress_conversation_filter_event = True
        for index, (value, label) in enumerate(options):
            combo.append_text(label)
            self._conversation_option_lookup.append(value)
            if value == self._conversation_filter:
                combo.set_active(index)

        if combo.get_active_text() is None:
            combo.set_active(0)
            self._conversation_filter = None
        self._suppress_conversation_filter_event = False

    def _on_persona_filter_changed(self, combo: Gtk.ComboBoxText) -> None:
        index = self._combo_active_index(combo)
        if 0 <= index < len(self._persona_option_lookup):
            self._persona_filter = self._persona_option_lookup[index]
        else:
            self._persona_filter = None
        self._rebuild_task_list()
        self._refresh_task_catalog()

    def _on_status_filter_changed(self, combo: Gtk.ComboBoxText) -> None:
        index = self._combo_active_index(combo)
        if 0 <= index < len(self._status_option_lookup):
            self._status_filter = self._status_option_lookup[index]
        else:
            self._status_filter = None
        self._refresh_state()

    def _on_owner_filter_changed(self, combo: Gtk.ComboBoxText) -> None:
        if self._suppress_owner_filter_event:
            return
        index = self._combo_active_index(combo)
        if 0 <= index < len(self._owner_option_lookup):
            self._owner_filter = self._owner_option_lookup[index]
        else:
            self._owner_filter = None
        self._refresh_state()

    def _on_conversation_filter_changed(self, combo: Gtk.ComboBoxText) -> None:
        if self._suppress_conversation_filter_event:
            return
        index = self._combo_active_index(combo)
        if 0 <= index < len(self._conversation_option_lookup):
            self._conversation_filter = self._conversation_option_lookup[index]
        else:
            self._conversation_filter = None
        self._refresh_state()

    def _combo_active_index(self, combo: Gtk.ComboBoxText) -> int:
        getter = getattr(combo, "get_active", None)
        if callable(getter):
            try:
                value = getter()
            except Exception:  # pragma: no cover - GTK fallback
                value = None
            if isinstance(value, int):
                return value

        active_text = getattr(combo, "get_active_text", lambda: None)()
        items = list(getattr(combo, "_items", []))
        if active_text in items:
            return items.index(active_text)
        return -1

    def _update_search_widgets(self) -> None:
        entry = self._search_entry
        if entry is not None:
            desired = self._search_text or ""
            current = getattr(entry, "get_text", lambda: "")()
            if current != desired:
                entry.set_text(desired)

        container = self._metadata_filter_container
        if container is None:
            return

        clear_container(container)
        self._metadata_filter_rows = []

        if self._metadata_filter:
            for key in sorted(self._metadata_filter):
                value = self._metadata_filter[key]
                self._add_metadata_filter_row(str(key), str(value))
        else:
            self._add_metadata_filter_row()

    def _collect_metadata_filters(self) -> Dict[str, Any]:
        filters: Dict[str, Any] = {}
        for record in self._metadata_filter_rows:
            key_entry = record.get("key")
            value_entry = record.get("value")
            if key_entry is None or value_entry is None:
                continue
            key_text = getattr(key_entry, "get_text", lambda: "")().strip()
            if not key_text:
                continue
            value_text = getattr(value_entry, "get_text", lambda: "")().strip()
            filters[key_text] = value_text
        return filters

    def _add_metadata_filter_row(self, key: str = "", value: str = "") -> None:
        container = self._metadata_filter_container
        if container is None:
            return

        row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        container.append(row)

        key_entry = Gtk.Entry()
        key_entry.set_placeholder_text("Key")
        key_entry.set_hexpand(True)
        if key:
            key_entry.set_text(key)
        key_entry.connect("activate", self._on_metadata_entry_activate)
        row.append(key_entry)

        value_entry = Gtk.Entry()
        value_entry.set_placeholder_text("Value")
        value_entry.set_hexpand(True)
        if value:
            value_entry.set_text(value)
        value_entry.connect("activate", self._on_metadata_entry_activate)
        row.append(value_entry)

        remove_button = Gtk.Button(label="Remove")
        remove_button.connect("clicked", self._on_remove_metadata_filter, row)
        row.append(remove_button)

        self._metadata_filter_rows.append({"row": row, "key": key_entry, "value": value_entry})

    def _on_add_metadata_filter(self, _button: Gtk.Button) -> None:
        self._add_metadata_filter_row()

    def _on_remove_metadata_filter(self, _button: Gtk.Button, row: Gtk.Widget) -> None:
        container = self._metadata_filter_container
        if container is None:
            return

        self._metadata_filter_rows = [
            record for record in self._metadata_filter_rows if record.get("row") is not row
        ]

        remover = getattr(container, "remove", None)
        if callable(remover):
            remover(row)
        else:  # pragma: no cover - GTK fallback
            getattr(row, "set_parent", lambda *_args: None)(None)

        if not self._metadata_filter_rows:
            self._add_metadata_filter_row()

    def _on_metadata_entry_activate(self, _entry: Gtk.Entry) -> None:
        self._apply_search_filters()

    def _on_search_activate(self, _entry: Gtk.Entry) -> None:
        self._apply_search_filters()

    def _on_search_apply(self, _button: Gtk.Button) -> None:
        self._apply_search_filters()

    def _on_search_clear(self, _button: Gtk.Button) -> None:
        self._search_text = ""
        self._metadata_filter = {}
        entry = self._search_entry
        if entry is not None:
            entry.set_text("")
        self._update_search_widgets()
        self._refresh_state()

    def _apply_search_filters(self) -> None:
        entry = self._search_entry
        text = getattr(entry, "get_text", lambda: "")().strip() if entry is not None else ""
        self._search_text = text
        self._metadata_filter = self._collect_metadata_filters()
        self._refresh_state()

    # ------------------------------------------------------------------
    # List management
    # ------------------------------------------------------------------
    def _rebuild_task_list(self) -> None:
        listbox = self._task_list
        if listbox is None:
            return
        clear_container(listbox)
        setattr(listbox, "_selected_row", None)
        self._row_lookup.clear()

        filtered: List[_TaskEntry] = []
        for entry in self._entries:
            if not self._passes_filters(entry):
                continue
            filtered.append(entry)

        self._display_entries = filtered
        for entry in filtered:
            row = self._create_list_row(entry)
            listbox.append(row)
            self._row_lookup[entry.task_id] = row

        self._rebuild_task_board(filtered)

        target = self._pending_focus_task or self._active_task
        self._pending_focus_task = None
        if target and target in {entry.task_id for entry in filtered}:
            self._select_task(target)
        elif filtered:
            self._select_task(filtered[0].task_id)
        else:
            self._select_task(None)

    def _passes_filters(self, entry: _TaskEntry) -> bool:
        persona_filter = self._persona_filter
        if persona_filter == self._PERSONA_UNASSIGNED:
            if entry.persona is not None:
                return False
        elif persona_filter and entry.persona != persona_filter:
            return False

        status_filter = self._status_filter
        if status_filter and entry.status != status_filter:
            return False
        return True

    def _refresh_task_catalog(self) -> None:
        entries = self._load_task_catalog()
        self._catalog_entries = entries
        self._update_catalog_hint()
        self._rebuild_task_catalog()

    def _rebuild_task_catalog(self) -> None:
        container = self._catalog_container
        if container is None:
            return
        clear_container(container)

        if not self._catalog_entries:
            placeholder = Gtk.Label(
                label="No reusable tasks were found in the manifest files."
            )
            placeholder.set_xalign(0.0)
            placeholder.set_wrap(True)
            container.append(placeholder)
            return

        for index, entry in enumerate(self._catalog_entries):
            container.append(self._create_catalog_card(entry))
            if index < len(self._catalog_entries) - 1:
                container.append(Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL))

    def _create_catalog_card(self, entry: Mapping[str, Any]) -> Gtk.Widget:
        card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        for setter_name in (
            "set_margin_top",
            "set_margin_bottom",
            "set_margin_start",
            "set_margin_end",
        ):
            setter = getattr(card, setter_name, None)
            if callable(setter):
                try:
                    setter(4)
                except Exception:  # pragma: no cover - GTK fallback
                    continue

        name = str(entry.get("name") or "Unnamed task").strip() or "Unnamed task"
        title = Gtk.Label(label=name)
        title.set_xalign(0.0)
        title.set_wrap(True)
        title.add_css_class("heading") if hasattr(title, "add_css_class") else None
        card.append(title)

        persona_text = str(entry.get("persona") or "Shared").strip() or "Shared"
        priority = str(entry.get("priority") or "").strip()
        metadata_tokens = [f"Persona: {persona_text}"]
        if priority:
            metadata_tokens.append(f"Priority: {priority}")
        tags = entry.get("tags")
        tag_tokens = [str(token).strip() for token in tags or [] if str(token).strip()]
        if tag_tokens:
            metadata_tokens.append("Tags: " + ", ".join(tag_tokens))
        metadata_label = Gtk.Label(label=" • ".join(metadata_tokens))
        metadata_label.set_xalign(0.0)
        metadata_label.set_wrap(True)
        metadata_label.add_css_class("dim-label") if hasattr(metadata_label, "add_css_class") else None
        card.append(metadata_label)

        summary = str(entry.get("summary") or entry.get("description") or "").strip()
        if summary:
            summary_label = Gtk.Label(label=summary)
            summary_label.set_xalign(0.0)
            summary_label.set_wrap(True)
            card.append(summary_label)

        skills = [str(token).strip() for token in entry.get("required_skills", []) if str(token).strip()]
        if skills:
            skills_label = Gtk.Label(label=f"Required skills: {', '.join(skills)}")
            skills_label.set_xalign(0.0)
            skills_label.set_wrap(True)
            card.append(skills_label)

        tools = [str(token).strip() for token in entry.get("required_tools", []) if str(token).strip()]
        if tools:
            tools_label = Gtk.Label(label=f"Required tools: {', '.join(tools)}")
            tools_label.set_xalign(0.0)
            tools_label.set_wrap(True)
            card.append(tools_label)

        acceptance = [
            str(item).strip()
            for item in entry.get("acceptance_criteria", [])
            if str(item).strip()
        ]
        if acceptance:
            bullets = "\n".join(f"• {item}" for item in acceptance)
            acceptance_label = Gtk.Label(label=f"Acceptance criteria:\n{bullets}")
            acceptance_label.set_xalign(0.0)
            acceptance_label.set_wrap(True)
            card.append(acceptance_label)

        escalation = entry.get("escalation_policy")
        if isinstance(escalation, Mapping):
            level = str(escalation.get("level") or "").strip()
            contact = str(escalation.get("contact") or "").strip()
            timeframe = str(escalation.get("timeframe") or "").strip()
            summary_parts = []
            if level:
                summary_parts.append(level)
            if contact:
                summary_parts.append(contact)
            if timeframe:
                summary_parts.append(timeframe)
            if summary_parts:
                escalation_label = Gtk.Label(
                    label="Escalation: " + " • ".join(summary_parts)
                )
                escalation_label.set_xalign(0.0)
                escalation_label.set_wrap(True)
                escalation_label.add_css_class("dim-label") if hasattr(escalation_label, "add_css_class") else None
                card.append(escalation_label)

        source = str(entry.get("source") or "").strip()
        if source:
            source_label = Gtk.Label(label=f"Source: {source}")
            source_label.set_xalign(0.0)
            source_label.set_wrap(True)
            source_label.add_css_class("dim-label") if hasattr(source_label, "add_css_class") else None
            card.append(source_label)

        return card

    def _update_catalog_hint(self) -> None:
        hint_label = self._catalog_hint_label
        if hint_label is None:
            return

        persona_filter = self._persona_filter
        if persona_filter == self._PERSONA_UNASSIGNED:
            scope = "shared tasks"
        elif persona_filter:
            scope = f"tasks for {persona_filter}"
        else:
            scope = "all personas"
        hint_label.set_text(f"Manifest-backed templates available for {scope}.")

    def _create_list_row(self, entry: _TaskEntry) -> Gtk.ListBoxRow:
        row = Gtk.ListBoxRow()
        row._task_id = entry.task_id  # type: ignore[attr-defined]

        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        box.set_margin_top(6)
        box.set_margin_bottom(6)
        box.set_margin_start(6)
        box.set_margin_end(6)

        title = Gtk.Label(label=entry.title or "Untitled task")
        title.set_xalign(0.0)
        title.set_wrap(True)
        box.append(title)

        info = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        info.set_hexpand(True)
        box.append(info)

        status_badge = create_badge(self._format_status(entry.status), self._status_css(entry.status))
        info.append(status_badge)

        if entry.persona:
            persona_label = Gtk.Label(label=entry.persona)
            persona_label.set_xalign(0.0)
            info.append(persona_label)
        elif entry.persona is None:
            persona_label = Gtk.Label(label="Unassigned")
            persona_label.set_xalign(0.0)
            persona_label.add_css_class("dim-label") if hasattr(persona_label, "add_css_class") else None
            info.append(persona_label)

        summary = Gtk.Label(label=entry.description)
        summary.set_xalign(0.0)
        summary.set_wrap(True)
        box.append(summary)

        row.set_child(box)
        return row

    def _clear_container(self, container: Gtk.Widget) -> None:
        remover = getattr(container, "remove_all", None)
        if callable(remover):
            try:
                remover()
                return
            except Exception:  # pragma: no cover - GTK fallback
                pass
        children = getattr(container, "get_children", None)
        if callable(children):
            for child in list(children()):
                remove = getattr(container, "remove", None)
                if callable(remove):
                    try:
                        remove(child)
                    except Exception:  # pragma: no cover - GTK fallback
                        continue

    def _on_row_selected(self, listbox: Gtk.ListBox, row: Optional[Gtk.ListBoxRow]) -> None:
        setattr(listbox, "_selected_row", row)
        if row is None:
            self._select_task(None)
            return
        task_id = getattr(row, "_task_id", None)
        if isinstance(task_id, str):
            self._select_task(task_id)

    def _select_task(self, task_id: Optional[str]) -> None:
        if not task_id:
            self._active_task = None
            self._update_board_selection(None)
            self._update_detail(None)
            return
        entry = self._entry_lookup.get(task_id)
        if entry is None:
            self._update_board_selection(None)
            return
        self._active_task = task_id
        row = self._row_lookup.get(task_id)
        listbox = self._task_list
        if row is not None and listbox is not None:
            getter = getattr(listbox, "get_selected_row", None)
            if callable(getter):
                try:
                    current = getter()
                except Exception:  # pragma: no cover - GTK fallback
                    current = None
            else:
                current = getattr(listbox, "_selected_row", None)
            if current is not row:
                selector = getattr(listbox, "select_row", None)
                if callable(selector):
                    try:
                        selector(row)
                    except Exception:  # pragma: no cover - GTK fallback
                        setattr(listbox, "_selected_row", row)
                else:
                    setattr(listbox, "_selected_row", row)
        self._update_board_selection(task_id)
        self._update_detail(entry)

    # ------------------------------------------------------------------
    # Detail panel
    # ------------------------------------------------------------------
    def _update_detail(self, entry: Optional[_TaskEntry]) -> None:
        if entry is None:
            self._set_label(self._title_label, "No task selected")
            self._set_label(self._status_label, "")
            self._set_label(self._description_label, "Select a task to view details.")
            self._set_label(self._persona_label, "-")
            self._set_label(self._owner_label, "-")
            self._set_label(self._conversation_label, "-")
            self._set_label(self._priority_label, "0")
            self._set_label(self._due_label, "-")
            self._set_label(self._updated_label, "-")
            self._update_action_buttons(None)
            sync_badge_section(self._required_skills_box, [])
            sync_badge_section(self._required_tools_box, [])
            sync_badge_section(self._acceptance_box, [])
            sync_badge_section(self._dependencies_box, [])
            return

        self._set_label(self._title_label, entry.title)
        self._set_label(self._status_label, f"Status: {self._format_status(entry.status)}")
        self._set_label(self._description_label, entry.description)
        self._set_label(self._persona_label, entry.persona or "Unassigned")
        self._set_label(self._owner_label, entry.owner_id or "-")
        self._set_label(self._conversation_label, entry.conversation_id or "-")
        self._set_label(self._priority_label, str(entry.priority))
        self._set_label(self._due_label, entry.due_at or "-")
        self._set_label(self._updated_label, entry.updated_at or "-")
        self._update_action_buttons(entry)

        skill_badges = [(token, ("tag-badge",)) for token in entry.required_skills]
        tool_badges = [(token, ("tag-badge",)) for token in entry.required_tools]
        criteria_badges = [(token, ("tag-badge", "status-ok")) for token in entry.acceptance_criteria]
        dependency_badges: List[Tuple[str, Tuple[str, ...]]] = []
        for dependency in entry.dependencies:
            if isinstance(dependency, Mapping):
                title = dependency.get("title") or dependency.get("id") or "Dependency"
                status = str(dependency.get("status") or "unknown").lower()
                css = self._status_css(status)
                dependency_badges.append((f"{title} ({self._format_status(status)})", css))
            else:
                dependency_badges.append((str(dependency), ("tag-badge",)))

        sync_badge_section(
            self._required_skills_box,
            skill_badges,
            fallback="No skills recorded",
        )
        sync_badge_section(
            self._required_tools_box,
            tool_badges,
            fallback="No tools recorded",
        )
        sync_badge_section(
            self._acceptance_box,
            criteria_badges,
            fallback="No acceptance criteria",
        )
        sync_badge_section(
            self._dependencies_box,
            dependency_badges,
            fallback="No dependencies",
        )

    def _set_label(self, widget: Optional[Gtk.Label], text: str) -> None:
        if widget is None:
            return
        try:
            widget.set_text(text)
        except Exception:  # pragma: no cover - GTK fallback
            setattr(widget, "label", text)

    def _update_action_buttons(self, entry: Optional[_TaskEntry]) -> None:
        box = self._action_box
        if box is None:
            return
        primary = self._primary_action_button
        secondary = self._secondary_action_button
        for button in (primary, secondary):
            if button is not None:
                button.set_visible(False)
                button._target_status = None  # type: ignore[attr-defined]
        if entry is None:
            return

        actions = list(self._ADVANCE_ACTIONS.get(entry.status, ()))
        if not actions:
            return

        if primary is not None and actions:
            label, target = actions[0]
            primary.set_label(label)
            primary.set_visible(True)
            primary._target_status = target  # type: ignore[attr-defined]
        if secondary is not None and len(actions) > 1:
            label, target = actions[1]
            secondary.set_label(label)
            secondary.set_visible(True)
            secondary._target_status = target  # type: ignore[attr-defined]

    def _format_status(self, status: str) -> str:
        if not status:
            return "Unknown"
        text = status.replace("_", " ").strip()
        if not text:
            return "Unknown"
        return text.capitalize()

    def _status_css(self, status: str) -> Tuple[str, ...]:
        normalized = (status or "").lower()
        if normalized in {"done"}:
            return ("tag-badge", "status-ok")
        if normalized in {"cancelled"}:
            return ("tag-badge", "status-error")
        if normalized in {"ready", "in_progress", "review"}:
            return ("tag-badge", "status-warning")
        return ("tag-badge", "status-unknown")

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _on_action_button_clicked(self, button: Gtk.Button) -> None:
        target = getattr(button, "_target_status", None)
        if not target or not isinstance(target, str):
            return
        if not self._active_task or self._active_task not in self._entry_lookup:
            return
        entry = self._entry_lookup[self._active_task]
        self._transition_task(entry, target)

    def _transition_task(self, entry: _TaskEntry, target_status: str) -> None:
        server = getattr(self.ATLAS, "server", None)
        transition = getattr(server, "transition_task", None)
        if not callable(transition):
            self._handle_backend_error("Task transitions are unavailable.")
            return

        context = {"tenant_id": getattr(self.ATLAS, "tenant_id", "default")}
        try:
            payload = transition(
                entry.task_id,
                target_status,
                context=context,
                expected_updated_at=entry.updated_at,
            )
        except Exception as exc:
            logger.error("Failed to transition task %s: %s", entry.task_id, exc, exc_info=True)
            self._handle_backend_error("Unable to update the task status.")
            return

        normalized = self._normalize_entry(payload)
        if normalized is not None:
            self._entry_lookup[normalized.task_id] = normalized
        self._pending_focus_task = entry.task_id
        self._refresh_state()
        notifier = getattr(self.parent_window, "show_success_toast", None)
        if callable(notifier):
            notifier(f"Task moved to {self._format_status(target_status)}")

    # ------------------------------------------------------------------
    # Message bus integration
    # ------------------------------------------------------------------
    def _subscribe_to_bus(self) -> None:
        if self._bus_subscriptions:
            return
        events = ("task.created", "task.updated", "task.status_changed")
        for event_name in events:
            try:
                handle = self.ATLAS.subscribe_event(event_name, self._handle_bus_event)
            except Exception as exc:  # pragma: no cover - subscription fallback
                logger.debug("Unable to subscribe to %s events: %s", event_name, exc, exc_info=True)
                continue
            self._bus_subscriptions.append(handle)

    async def _handle_bus_event(self, payload: Any, *_args: Any) -> None:
        task_id = None
        if isinstance(payload, Mapping):
            task_id = payload.get("task_id")
        if task_id:
            logger.debug("Received task event for %s; scheduling refresh", task_id)
        self._schedule_refresh()

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------
    def _handle_backend_error(self, message: str) -> None:
        handler = getattr(self.parent_window, "show_error_dialog", None)
        if callable(handler):
            handler(message)
        else:
            logger.error("Task management error: %s", message)
