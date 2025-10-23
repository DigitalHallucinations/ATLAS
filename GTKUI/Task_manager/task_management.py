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
        self._catalog_entries: List[Mapping[str, Any]] = []
        self._catalog_container: Optional[Gtk.Widget] = None
        self._catalog_hint_label: Optional[Gtk.Label] = None

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
    # Data loading and synchronization
    # ------------------------------------------------------------------
    def _refresh_state(self) -> None:
        entries = self._load_task_entries()
        self._entries = entries
        self._entry_lookup = {entry.task_id: entry for entry in entries}
        self._populate_persona_filter_options()
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
