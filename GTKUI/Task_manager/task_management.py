"""GTK task management workspace used by the sidebar."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import GLib, Gtk

from modules.Tools.tool_event_system import subscribe_bus_event

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
        self._persona_filter_combo: Optional[Gtk.ComboBoxText] = None
        self._status_filter_combo: Optional[Gtk.ComboBoxText] = None
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
        self._persona_filter: Optional[str] = None
        self._status_filter: Optional[str] = None
        self._persona_option_lookup: List[Optional[str]] = []
        self._status_option_lookup: List[Optional[str]] = []
        self._active_task: Optional[str] = None
        self._pending_focus_task: Optional[str] = None
        self._refresh_pending = False
        self._bus_subscriptions: List[Any] = []

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

        scroller = Gtk.ScrolledWindow()
        scroller.set_hexpand(False)
        scroller.set_vexpand(True)
        scroller.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        sidebar.append(scroller)

        listbox = Gtk.ListBox()
        listbox.set_selection_mode(Gtk.SelectionMode.SINGLE)
        listbox.connect("row-selected", self._on_row_selected)
        scroller.set_child(listbox)
        self._task_list = listbox

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

        detail_panel.connect("destroy", lambda *_args: self._on_close_request())
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

        badge_container = self._create_badge_container()
        badge_container.set_visible(False)
        container.append(badge_container)
        return badge_container

    def _create_badge_container(self) -> Gtk.Widget:
        flow_class = getattr(Gtk, "FlowBox", None)
        if flow_class is None:
            return Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        container = flow_class()
        setter = getattr(container, "set_selection_mode", None)
        if callable(setter):
            try:
                setter(Gtk.SelectionMode.NONE)
            except Exception:  # pragma: no cover - GTK fallback
                pass
        setter = getattr(container, "set_max_children_per_line", None)
        if callable(setter):
            try:
                setter(6)
            except Exception:  # pragma: no cover - GTK fallback
                pass
        setter = getattr(container, "set_row_spacing", None)
        if callable(setter):
            try:
                setter(6)
            except Exception:  # pragma: no cover - GTK fallback
                pass
        setter = getattr(container, "set_column_spacing", None)
        if callable(setter):
            try:
                setter(6)
            except Exception:  # pragma: no cover - GTK fallback
                pass
        return container

    # ------------------------------------------------------------------
    # Data loading and synchronization
    # ------------------------------------------------------------------
    def _refresh_state(self) -> None:
        entries = self._load_task_entries()
        self._entries = entries
        self._entry_lookup = {entry.task_id: entry for entry in entries}
        self._populate_persona_filter_options()
        self._rebuild_task_list()

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
        server = getattr(self.ATLAS, "server", None)
        list_tasks = getattr(server, "list_tasks", None)
        if not callable(list_tasks):
            logger.warning("ATLAS server does not expose list_tasks; returning empty workspace")
            return []

        params: Dict[str, Any] = {}
        if self._status_filter:
            params["status"] = self._status_filter

        context = {"tenant_id": getattr(self.ATLAS, "tenant_id", "default")}
        try:
            response = list_tasks(params, context=context)
        except Exception as exc:
            logger.error("Failed to load task list: %s", exc, exc_info=True)
            self._handle_backend_error("Unable to load tasks from ATLAS.")
            return []

        items: Iterable[Any]
        if isinstance(response, Mapping):
            items = response.get("items", [])
        elif isinstance(response, Iterable):
            items = response
        else:
            items = []

        entries: List[_TaskEntry] = []
        for raw_entry in items:
            normalized = self._normalize_entry(raw_entry)
            if normalized is not None:
                entries.append(normalized)
        return entries

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

        def _normalize_sequence(value: Any) -> Tuple[str, ...]:
            if isinstance(value, Mapping):
                iterable = value.values()
            else:
                iterable = value
            tokens: List[str] = []
            if isinstance(iterable, Iterable) and not isinstance(iterable, (str, bytes)):
                for item in iterable:
                    text = str(item).strip()
                    if text:
                        tokens.append(text)
            return tuple(tokens)

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
            required_skills=_normalize_sequence(metadata_dict.get("required_skills")),
            required_tools=_normalize_sequence(metadata_dict.get("required_tools")),
            acceptance_criteria=_normalize_sequence(metadata_dict.get("acceptance_criteria")),
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

    def _on_persona_filter_changed(self, combo: Gtk.ComboBoxText) -> None:
        index = getattr(combo, "_active", -1)
        if 0 <= index < len(self._persona_option_lookup):
            selected = self._persona_option_lookup[index]
            self._persona_filter = selected
        else:
            self._persona_filter = None
        self._rebuild_task_list()

    def _on_status_filter_changed(self, combo: Gtk.ComboBoxText) -> None:
        index = getattr(combo, "_active", -1)
        if 0 <= index < len(self._status_option_lookup):
            selected = self._status_option_lookup[index]
            self._status_filter = selected
        else:
            self._status_filter = None
        self._refresh_state()

    # ------------------------------------------------------------------
    # List management
    # ------------------------------------------------------------------
    def _rebuild_task_list(self) -> None:
        listbox = self._task_list
        if listbox is None:
            return
        self._clear_container(listbox)
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

        status_badge = self._create_badge(self._format_status(entry.status), self._status_css(entry.status))
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
            self._update_detail(None)
            return
        entry = self._entry_lookup.get(task_id)
        if entry is None:
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
            self._sync_badge_section(self._required_skills_box, [])
            self._sync_badge_section(self._required_tools_box, [])
            self._sync_badge_section(self._acceptance_box, [])
            self._sync_badge_section(self._dependencies_box, [])
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

        self._sync_badge_section(self._required_skills_box, skill_badges, fallback="No skills recorded")
        self._sync_badge_section(self._required_tools_box, tool_badges, fallback="No tools recorded")
        self._sync_badge_section(self._acceptance_box, criteria_badges, fallback="No acceptance criteria")
        self._sync_badge_section(self._dependencies_box, dependency_badges, fallback="No dependencies")

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

    def _sync_badge_section(
        self,
        container: Optional[Gtk.Widget],
        badges: Sequence[Tuple[str, Sequence[str]]],
        *,
        fallback: Optional[str] = None,
    ) -> None:
        if container is None:
            return
        self._clear_container(container)
        has_badges = False
        for text, css_classes in badges:
            badge = self._create_badge(text, css_classes)
            self._append_badge(container, badge)
            has_badges = True
        if not has_badges and fallback:
            badge = self._create_badge(fallback, ("tag-badge", "status-unknown"))
            self._append_badge(container, badge)
            has_badges = True
        if hasattr(container, "set_visible"):
            container.set_visible(has_badges)

    def _append_badge(self, container: Gtk.Widget, badge: Gtk.Widget) -> None:
        inserter = getattr(container, "insert", None)
        if callable(inserter):
            try:
                inserter(badge, -1)
                return
            except Exception:  # pragma: no cover - GTK fallback
                pass
        appender = getattr(container, "append", None)
        if callable(appender):
            appender(badge)

    def _create_badge(self, text: str, css_classes: Sequence[str]) -> Gtk.Widget:
        label = Gtk.Label(label=text)
        label.set_xalign(0.0)
        label.set_wrap(False)
        try:
            label.add_css_class("tag-badge")
        except Exception:  # pragma: no cover - GTK fallback
            pass
        for css in css_classes:
            if css == "tag-badge":
                continue
            try:
                label.add_css_class(css)
            except Exception:  # pragma: no cover - GTK fallback
                continue
        return label

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
                handle = subscribe_bus_event(event_name, self._handle_bus_event)
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
