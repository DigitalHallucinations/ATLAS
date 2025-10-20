"""GTK tool management workspace used by the sidebar."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable as TypingIterable, List, Optional, Sequence, Tuple

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import Gtk

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _ToolEntry:
    """Normalized tool metadata returned by the ATLAS server."""

    name: str
    title: str
    summary: str
    capabilities: List[str]
    auth: Dict[str, Any]
    account_status: Optional[str]
    raw_metadata: Dict[str, Any]


class ToolManagement:
    """Controller responsible for rendering the tool management workspace."""

    _SORT_OPTIONS = [
        ("title_asc", "Title (A–Z)"),
        ("title_desc", "Title (Z–A)"),
        ("name_asc", "Name (A–Z)"),
        ("name_desc", "Name (Z–A)"),
    ]
    _DEFAULT_SORT_KEY = "title_asc"
    _CAPABILITY_ALL_ID = "__all__"

    def __init__(self, atlas: Any, parent_window: Any) -> None:
        self.ATLAS = atlas
        self.parent_window = parent_window

        self.on_open_in_persona: Optional[Callable[[str], None]] = None

        self._widget: Optional[Gtk.Widget] = None
        self._tool_list: Optional[Gtk.ListBox] = None
        self._persona_label: Optional[Gtk.Label] = None
        self._title_label: Optional[Gtk.Label] = None
        self._summary_label: Optional[Gtk.Label] = None
        self._capabilities_label: Optional[Gtk.Label] = None
        self._auth_label: Optional[Gtk.Label] = None
        self._status_badge_box: Optional[Gtk.Box] = None
        self._capability_badge_box: Optional[Gtk.Widget] = None
        self._warning_badge_box: Optional[Gtk.Box] = None
        self._telemetry_label: Optional[Gtk.Label] = None
        self._cta_box: Optional[Gtk.Box] = None
        self._switch: Optional[Gtk.Switch] = None
        self._save_button: Optional[Gtk.Button] = None
        self._bulk_enable_button: Optional[Gtk.Button] = None
        self._bulk_apply_button: Optional[Gtk.Button] = None
        self._reset_button: Optional[Gtk.Button] = None
        self._scope_selector: Optional[Gtk.ComboBoxText] = None
        self._search_entry: Optional[Gtk.SearchEntry] = None
        self._capability_selector: Optional[Gtk.ComboBoxText] = None
        self._sort_selector: Optional[Gtk.ComboBoxText] = None

        self._entries: List[_ToolEntry] = []
        self._entry_lookup: Dict[str, _ToolEntry] = {}
        self._row_lookup: Dict[str, Gtk.Widget] = {}
        self._bulk_checkbox_lookup: Dict[str, Gtk.CheckButton] = {}
        self._visible_entries: List[_ToolEntry] = []

        self._persona_name: Optional[str] = None
        self._enabled_tools: set[str] = set()
        self._baseline_enabled: set[str] = set()
        self._bulk_selection: set[str] = set()
        self._active_tool: Optional[str] = None
        self._suppress_toggle = False
        self._suppress_bulk_signals = False
        self._tool_scope = "persona"
        self._suppress_scope_signal = False
        self._suppress_filter_signals = False

        self._filter_text = ""
        self._capability_filter: Optional[str] = None
        self._sort_key = self._DEFAULT_SORT_KEY
        self._capability_options: List[str] = [self._CAPABILITY_ALL_ID]
        self._sort_option_ids: List[str] = [key for key, _ in self._SORT_OPTIONS]
        self._preferences_loaded_key: Optional[str] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_embeddable_widget(self) -> Gtk.Widget:
        """Return the workspace widget, creating it on first use."""

        if self._widget is None:
            self._widget = self._build_workspace()
        self._refresh_state()
        return self._widget

    def focus_tool(self, tool_name: str) -> bool:
        """Ensure ``tool_name`` is visible and selected in the catalog."""

        if not tool_name:
            return False

        if tool_name not in self._entry_lookup:
            if self._tool_scope != "all":
                self._tool_scope = "all"
                self._sync_scope_widget(bool(self._persona_name))
                self._refresh_state()
        if tool_name not in self._entry_lookup:
            return False

        if tool_name not in {entry.name for entry in self._visible_entries}:
            changed = False
            if self._filter_text:
                self._filter_text = ""
                changed = True
            if self._capability_filter is not None:
                self._capability_filter = None
                changed = True
            if changed:
                self._persist_view_preferences()
                self._sync_filter_widgets()
            self._rebuild_tool_list()

        if tool_name not in {entry.name for entry in self._visible_entries}:
            return False

        self._select_tool(tool_name)
        return True

    def enable_tools_for_persona(self, persona: str, tools: Iterable[str]) -> bool:
        """Enable the provided tools for ``persona`` and persist the change."""

        if not persona:
            self._handle_backend_error("No persona provided for enabling tools.")
            return False

        manager = getattr(self.ATLAS, "persona_manager", None)
        setter = getattr(manager, "set_allowed_tools", None)
        if not callable(setter):
            self._handle_backend_error("Persona manager is unavailable. Unable to save tools.")
            return False

        existing = set(self._load_enabled_tools(persona))
        candidates = {str(tool).strip() for tool in tools if tool}
        desired = sorted(existing | candidates)

        try:
            result = setter(persona, desired)
        except Exception as exc:  # pragma: no cover - backend failure logging
            logger.error("Failed to persist tools for persona '%s': %s", persona, exc, exc_info=True)
            self._handle_backend_error(str(exc) or "Unable to save tool configuration.")
            return False

        if isinstance(result, Mapping) and not result.get("success", True):
            errors = result.get("errors") or []
            message = "; ".join(str(err) for err in errors if err) or str(result.get("error") or "Unable to save tool configuration.")
            self._handle_backend_error(message)
            return False

        self._refresh_state()
        return True

    # ------------------------------------------------------------------
    # Workspace construction
    # ------------------------------------------------------------------
    def _build_workspace(self) -> Gtk.Widget:
        root = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        for setter_name in ("set_margin_top", "set_margin_bottom", "set_margin_start", "set_margin_end"):
            setter = getattr(root, setter_name, None)
            if callable(setter):
                setter(12)

        left_panel = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        left_panel.set_hexpand(False)
        left_panel.set_vexpand(True)

        heading = Gtk.Label(label="Available Tools")
        heading.set_xalign(0.0)
        left_panel.append(heading)

        SearchEntryClass = getattr(Gtk, "SearchEntry", None)
        if SearchEntryClass is None:
            SearchEntryClass = getattr(Gtk, "Entry", Gtk.Widget)
        search_entry = SearchEntryClass()
        try:
            search_entry.set_placeholder_text("Search tools…")
        except Exception:  # pragma: no cover - GTK version differences
            pass
        search_entry.connect("changed", self._on_search_changed)
        search_entry.connect("search-changed", self._on_search_changed)
        left_panel.append(search_entry)
        self._search_entry = search_entry

        filter_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        filter_row.set_hexpand(True)

        capability_selector = Gtk.ComboBoxText()
        capability_selector.set_hexpand(True)
        capability_selector.connect("changed", self._on_capability_filter_changed)
        capability_selector.set_tooltip_text("Filter tools by advertised capability.")
        filter_row.append(capability_selector)
        self._capability_selector = capability_selector

        sort_selector = Gtk.ComboBoxText()
        sort_selector.set_hexpand(False)
        sort_selector.connect("changed", self._on_sort_changed)
        sort_selector.set_tooltip_text("Sort the tool catalog.")
        filter_row.append(sort_selector)
        self._sort_selector = sort_selector

        left_panel.append(filter_row)

        tool_list = Gtk.ListBox()
        tool_list.connect("row-selected", self._on_row_selected)
        self._tool_list = tool_list

        scroller = Gtk.ScrolledWindow()
        scroller.set_hexpand(False)
        scroller.set_vexpand(True)
        scroller.set_child(tool_list)
        left_panel.append(scroller)

        right_panel = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        right_panel.set_hexpand(True)
        right_panel.set_vexpand(True)

        persona_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        persona_row.set_hexpand(True)

        persona_label = Gtk.Label()
        persona_label.set_xalign(0.0)
        persona_label.set_wrap(True)
        persona_label.set_hexpand(True)
        persona_row.append(persona_label)
        self._persona_label = persona_label

        scope_selector = Gtk.ComboBoxText()
        scope_selector.append_text("Persona tools")
        scope_selector.append_text("All tools")
        scope_selector.set_active(0)
        scope_selector.connect("changed", self._on_scope_changed)
        persona_row.append(scope_selector)
        self._scope_selector = scope_selector

        right_panel.append(persona_row)

        title_label = Gtk.Label()
        title_label.set_xalign(0.0)
        try:
            title_label.add_css_class("title-3")
        except Exception:  # pragma: no cover - GTK theme variations
            pass
        right_panel.append(title_label)
        self._title_label = title_label

        summary_label = Gtk.Label()
        summary_label.set_wrap(True)
        summary_label.set_xalign(0.0)
        right_panel.append(summary_label)
        self._summary_label = summary_label

        capabilities_label = Gtk.Label()
        capabilities_label.set_wrap(True)
        capabilities_label.set_xalign(0.0)
        right_panel.append(capabilities_label)
        self._capabilities_label = capabilities_label

        status_badge_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        status_badge_box.set_hexpand(False)
        right_panel.append(status_badge_box)
        self._status_badge_box = status_badge_box

        FlowBoxClass = getattr(Gtk, "FlowBox", Gtk.Box)
        capability_badge_box = FlowBoxClass()
        setter = getattr(capability_badge_box, "set_max_children_per_line", None)
        if callable(setter):
            setter(6)
        selector = getattr(capability_badge_box, "set_selection_mode", None)
        if callable(selector):
            selector(Gtk.SelectionMode.NONE)
        capability_badge_box.set_valign(Gtk.Align.START)
        capability_badge_box.set_hexpand(True)
        capability_badge_box.set_visible(False)
        right_panel.append(capability_badge_box)
        self._capability_badge_box = capability_badge_box

        auth_label = Gtk.Label()
        auth_label.set_wrap(True)
        auth_label.set_xalign(0.0)
        right_panel.append(auth_label)
        self._auth_label = auth_label

        warning_badge_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        warning_badge_box.set_visible(False)
        right_panel.append(warning_badge_box)
        self._warning_badge_box = warning_badge_box

        telemetry_label = Gtk.Label()
        telemetry_label.set_wrap(True)
        telemetry_label.set_xalign(0.0)
        try:
            telemetry_label.add_css_class("dim-label")
        except Exception:  # pragma: no cover - GTK theme variations
            pass
        right_panel.append(telemetry_label)
        self._telemetry_label = telemetry_label

        cta_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        cta_box.set_halign(Gtk.Align.START)
        cta_box.set_visible(False)
        right_panel.append(cta_box)
        self._cta_box = cta_box

        toggle_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        toggle_row.set_hexpand(True)

        toggle_label = Gtk.Label(label="Enable tool for persona")
        toggle_label.set_xalign(0.0)
        toggle_row.append(toggle_label)

        toggle_switch = Gtk.Switch()
        toggle_switch.set_halign(Gtk.Align.END)
        toggle_switch.connect("state-set", self._on_switch_state_set)
        toggle_row.append(toggle_switch)
        self._switch = toggle_switch

        right_panel.append(toggle_row)

        actions = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        actions.set_halign(Gtk.Align.END)

        bulk_enable_button = Gtk.Button(label="Enable selected for persona")
        bulk_enable_button.set_tooltip_text("Enable every checked tool for the active persona.")
        bulk_enable_button.connect("clicked", self._on_bulk_enable_clicked)
        actions.append(bulk_enable_button)
        self._bulk_enable_button = bulk_enable_button

        bulk_apply_button = Gtk.Button(label="Apply recommended set")
        bulk_apply_button.set_tooltip_text(
            "Enable a recommended set of tools based on persona templates and dependencies."
        )
        bulk_apply_button.connect("clicked", self._on_bulk_apply_clicked)
        actions.append(bulk_apply_button)
        self._bulk_apply_button = bulk_apply_button

        save_button = Gtk.Button(label="Save Changes")
        try:
            save_button.add_css_class("suggested-action")
        except Exception:  # pragma: no cover - GTK theme variations
            pass
        save_button.set_tooltip_text("Persist the selected tools for the active persona.")
        save_button.connect("clicked", self._on_save_clicked)
        actions.append(save_button)
        self._save_button = save_button

        reset_button = Gtk.Button(label="Reset")
        reset_button.set_tooltip_text("Reload tool data from ATLAS and discard local changes.")
        reset_button.connect("clicked", self._on_reset_clicked)
        actions.append(reset_button)
        self._reset_button = reset_button

        right_panel.append(actions)

        root.append(left_panel)
        root.append(right_panel)
        return root

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def _refresh_state(self) -> None:
        persona = self._resolve_persona_name()
        self._persona_name = persona
        self._sync_scope_widget(bool(persona))

        preferences_key = self._get_preferences_key(persona)
        if preferences_key != self._preferences_loaded_key:
            self._load_view_preferences(persona)

        persona_filter = persona if self._tool_scope == "persona" and persona else None

        if persona:
            self._set_label(self._persona_label, f"Persona: {persona}")
        else:
            self._set_label(self._persona_label, "Persona: (unavailable)")

        entries = self._load_tool_entries(persona_filter)

        if persona_filter is not None:
            enabled = self._load_enabled_tools(persona_filter)
            self._enabled_tools = set(enabled)
            self._baseline_enabled = set(enabled)
        else:
            self._enabled_tools = set()
            self._baseline_enabled = set()

        self._entries = entries
        self._entry_lookup = {entry.name: entry for entry in entries}
        self._bulk_selection.intersection_update(self._entry_lookup.keys())
        self._populate_capability_selector()
        self._populate_sort_selector()
        self._sync_filter_widgets()
        self._rebuild_tool_list()
        self._update_action_state()

    def _resolve_persona_name(self) -> Optional[str]:
        getter = getattr(self.ATLAS, "get_active_persona_name", None)
        if callable(getter):
            try:
                persona = getter()
            except Exception as exc:  # pragma: no cover - defensive logging only
                logger.error("Failed to determine active persona: %s", exc, exc_info=True)
                return None
            if persona:
                return str(persona)
        return None

    def _load_tool_entries(self, persona: Optional[str]) -> List[_ToolEntry]:
        server = getattr(self.ATLAS, "server", None)
        getter = getattr(server, "get_tools", None)
        if not callable(getter):
            logger.warning("ATLAS server does not expose get_tools; returning empty workspace")
            return []

        kwargs: Dict[str, Any] = {}
        if persona:
            kwargs["persona"] = persona

        try:
            response = getter(**kwargs)
        except Exception as exc:
            logger.error("Failed to load tool metadata: %s", exc, exc_info=True)
            self._handle_backend_error("Unable to load tool metadata from ATLAS.")
            return []

        tools = response.get("tools") if isinstance(response, Mapping) else None
        if not isinstance(tools, Iterable):
            return []

        entries: List[_ToolEntry] = []
        for raw_entry in tools:
            entry = self._normalize_entry(raw_entry)
            if entry is not None:
                entries.append(entry)
        return entries

    def _load_enabled_tools(self, persona: Optional[str]) -> List[str]:
        if not persona:
            return []

        manager = getattr(self.ATLAS, "persona_manager", None)
        getter = getattr(manager, "get_persona", None)
        if not callable(getter):
            return []

        try:
            persona_payload = getter(persona)
        except Exception as exc:
            logger.error("Failed to load persona '%s': %s", persona, exc, exc_info=True)
            self._handle_backend_error("Unable to read persona configuration from ATLAS.")
            return []

        allowed = persona_payload.get("allowed_tools") if isinstance(persona_payload, Mapping) else None
        if not isinstance(allowed, Iterable):
            return []

        normalized: List[str] = []
        for item in allowed:
            text = str(item).strip()
            if text:
                normalized.append(text)
        return normalized

    def _normalize_entry(self, entry: Any) -> Optional[_ToolEntry]:
        if not isinstance(entry, Mapping):
            return None

        raw_name = entry.get("name")
        name = str(raw_name).strip() if raw_name else ""
        if not name:
            return None

        title_source = entry.get("title") or entry.get("display_name") or entry.get("label")
        title = str(title_source).strip() if title_source else name

        summary_source = entry.get("summary") or entry.get("description")
        summary = str(summary_source).strip() if summary_source else "No description available."

        capabilities_raw = entry.get("capabilities") or entry.get("capability_tags")
        capabilities: List[str] = []
        if isinstance(capabilities_raw, Iterable):
            for item in capabilities_raw:
                text = str(item).strip()
                if text:
                    capabilities.append(text)

        auth = entry.get("auth") if isinstance(entry.get("auth"), Mapping) else {}
        account_status = entry.get("account_status")

        normalized_metadata = dict(entry)
        normalized_metadata.setdefault("name", name)

        return _ToolEntry(
            name=name,
            title=title,
            summary=summary,
            capabilities=capabilities,
            auth=dict(auth) if isinstance(auth, Mapping) else {},
            account_status=str(account_status) if account_status else None,
            raw_metadata=normalized_metadata,
        )

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------
    def _rebuild_tool_list(self) -> None:
        if self._tool_list is None:
            return

        children_getter = getattr(self._tool_list, "get_children", None)
        if callable(children_getter):
            rows = list(children_getter())
        else:
            rows = list(getattr(self._tool_list, "children", []) or [])

        for child in rows:
            try:
                self._tool_list.remove(child)
            except Exception:
                continue

        self._row_lookup.clear()
        self._bulk_checkbox_lookup.clear()
        visible_entries = self._derive_visible_entries()
        self._visible_entries = visible_entries

        for entry in visible_entries:
            row = self._create_row(entry)
            self._tool_list.append(row)
            self._row_lookup[entry.name] = row

        self._sync_bulk_checkbox_state()

        if visible_entries:
            desired = self._active_tool
            if not desired or desired not in self._row_lookup:
                desired = visible_entries[0].name
            self._select_tool(desired)
        else:
            try:
                self._tool_list.select_row(None)  # type: ignore[arg-type]
            except Exception:  # pragma: no cover - GTK compatibility fallback
                pass
            self._show_empty_state()

    def _create_row(self, entry: _ToolEntry) -> Gtk.Widget:
        container = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)

        header_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        header_row.set_hexpand(True)

        allow_bulk = self._current_scope_allows_editing()
        checkbox: Optional[Gtk.CheckButton] = None
        if allow_bulk:
            checkbox = Gtk.CheckButton()
            align_cls = getattr(Gtk, "Align", None)
            align_center = getattr(align_cls, "CENTER", None) if align_cls is not None else None
            if align_center is not None:
                try:
                    checkbox.set_halign(align_center)
                except Exception:  # pragma: no cover - GTK compatibility fallbacks
                    pass
                try:
                    checkbox.set_valign(align_center)
                except Exception:  # pragma: no cover - GTK compatibility fallbacks
                    pass
            checkbox.set_tooltip_text("Select this tool for bulk actions.")
            self._suppress_bulk_signals = True
            try:
                setter = getattr(checkbox, "set_active", None)
                if callable(setter):
                    setter(entry.name in self._bulk_selection)
                else:  # pragma: no cover - GTK compatibility fallback
                    checkbox.props.active = entry.name in self._bulk_selection
            finally:
                self._suppress_bulk_signals = False
            checkbox.connect("toggled", self._on_bulk_checkbox_toggled, entry.name)
            header_row.append(checkbox)
            self._bulk_checkbox_lookup[entry.name] = checkbox

        title = Gtk.Label(label=entry.title or entry.name)
        title.set_xalign(0.0)
        title.set_hexpand(True)
        header_row.append(title)

        for badge_text, css_classes in self._iter_status_badges(entry):
            badge = self._create_badge(badge_text, css_classes)
            header_row.append(badge)

        container.append(header_row)

        summary = Gtk.Label(label=entry.summary)
        summary.set_xalign(0.0)
        summary.set_wrap(True)
        try:
            summary.add_css_class("dim-label")
        except Exception:  # pragma: no cover - GTK theme variations
            pass
        container.append(summary)

        FlowBoxClass = getattr(Gtk, "FlowBox", Gtk.Box)
        capability_badge_box = FlowBoxClass()
        selector = getattr(capability_badge_box, "set_selection_mode", None)
        if callable(selector):
            selector(Gtk.SelectionMode.NONE)
        setter = getattr(capability_badge_box, "set_max_children_per_line", None)
        if callable(setter):
            setter(6)
        setter = getattr(capability_badge_box, "set_row_spacing", None)
        if callable(setter):
            setter(2)
        setter = getattr(capability_badge_box, "set_column_spacing", None)
        if callable(setter):
            setter(4)
        for badge_text, css_classes in self._iter_capability_badges(entry):
            badge = self._create_badge(badge_text, css_classes)
            capability_badge_box.insert(badge, -1)
        if capability_badge_box.get_child_at_index(0) is not None:
            container.append(capability_badge_box)

        warnings = list(self._iter_warning_badges(entry))
        if warnings:
            warning_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
            for badge_text, css_classes in warnings:
                warning_box.append(self._create_badge(badge_text, css_classes))
            container.append(warning_box)

        telemetry_text = self._format_last_success(entry)
        if telemetry_text:
            telemetry = Gtk.Label(label=telemetry_text)
            telemetry.set_xalign(0.0)
            try:
                telemetry.add_css_class("caption")
            except Exception:  # pragma: no cover - GTK theme variations
                pass
            container.append(telemetry)

        RowClass = getattr(Gtk, "ListBoxRow", None)
        if RowClass is not None:
            try:
                row = RowClass()
                set_child = getattr(row, "set_child", None)
                if callable(set_child):
                    set_child(container)
                else:  # pragma: no cover - GTK compatibility fallback
                    row.child = container
                return row
            except Exception:  # pragma: no cover - stub compatibility
                pass

        return container

    def _select_tool(self, tool_name: str) -> None:
        entry = self._entry_lookup.get(tool_name)
        row = self._row_lookup.get(tool_name)
        if entry is None or row is None:
            self._active_tool = None
            self._show_empty_state()
            return

        self._active_tool = tool_name

        self._set_label(self._title_label, entry.title or entry.name)
        self._set_label(self._summary_label, entry.summary or "")

        if entry.capabilities:
            capability_text = "Capabilities: " + ", ".join(entry.capabilities)
        else:
            capability_text = "Capabilities: (unspecified)"
        self._set_label(self._capabilities_label, capability_text)

        self._sync_badge_container(self._status_badge_box, self._iter_status_badges(entry))
        self._sync_capability_box(entry)
        self._sync_warning_box(entry)
        self._set_label(self._auth_label, self._format_auth(entry))
        self._set_label(self._telemetry_label, self._format_detail_telemetry(entry))
        self._sync_cta_box(entry)

        if self._tool_list is not None:
            try:
                self._tool_list.select_row(row)
            except Exception:  # pragma: no cover - GTK stub variations
                pass

        allow_edit = self._current_scope_allows_editing()
        self._set_switch_sensitive(allow_edit)
        if allow_edit:
            self._set_switch_active(tool_name in self._enabled_tools)
        else:
            self._set_switch_active(False)
        self._update_action_state()

    def _show_empty_state(self) -> None:
        self._active_tool = None
        self._set_label(self._title_label, "No tool selected")
        self._set_label(self._summary_label, "Select a tool to view configuration details.")
        self._set_label(self._capabilities_label, "Capabilities: (n/a)")
        self._set_label(self._auth_label, "Authentication status: unavailable")
        self._sync_badge_container(self._status_badge_box, [])
        self._clear_container(self._capability_badge_box)
        self._clear_container(self._warning_badge_box)
        if self._capability_badge_box is not None:
            self._capability_badge_box.set_visible(False)
        if self._warning_badge_box is not None:
            self._warning_badge_box.set_visible(False)
        self._set_label(self._telemetry_label, "Telemetry unavailable.")
        if self._cta_box is not None:
            self._cta_box.set_visible(False)
            self._clear_container(self._cta_box)
        self._set_switch_sensitive(self._current_scope_allows_editing())
        self._set_switch_active(False)
        self._update_action_state()

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------
    def _on_search_changed(self, entry: Gtk.SearchEntry) -> None:
        if self._suppress_filter_signals:
            return

        getter = getattr(entry, "get_text", None)
        try:
            text = getter() if callable(getter) else None
        except Exception:  # pragma: no cover - GTK compatibility fallback
            text = None
        if not isinstance(text, str):
            text = getattr(entry, "text", None)
        if not isinstance(text, str):
            props = getattr(entry, "props", None)
            text = getattr(props, "text", "") if props is not None else ""
        new_text = text or ""

        if new_text == self._filter_text:
            return

        self._filter_text = new_text
        self._persist_view_preferences()
        self._rebuild_tool_list()

    def _on_capability_filter_changed(self, combo: Gtk.ComboBoxText) -> None:
        if self._suppress_filter_signals:
            return

        selected = self._get_combo_active_id(combo, self._capability_options)
        normalized = None if not selected or selected == self._CAPABILITY_ALL_ID else selected

        if normalized == self._capability_filter:
            return

        self._capability_filter = normalized
        self._persist_view_preferences()
        self._rebuild_tool_list()

    def _on_sort_changed(self, combo: Gtk.ComboBoxText) -> None:
        if self._suppress_filter_signals:
            return

        selected = self._get_combo_active_id(combo, self._sort_option_ids)
        if not selected:
            selected = self._DEFAULT_SORT_KEY

        if selected not in self._sort_option_ids:
            selected = self._DEFAULT_SORT_KEY

        if selected == self._sort_key:
            return

        self._sort_key = selected
        self._persist_view_preferences()
        self._rebuild_tool_list()

    def _on_row_selected(self, _listbox: Gtk.ListBox, row: Gtk.Widget) -> None:
        for name, candidate in self._row_lookup.items():
            if candidate is row:
                self._select_tool(name)
                break

    def _on_bulk_checkbox_toggled(self, checkbox: Gtk.CheckButton, tool_name: str) -> None:
        if self._suppress_bulk_signals:
            return

        active = False
        getter = getattr(checkbox, "get_active", None)
        if callable(getter):
            try:
                active = bool(getter())
            except Exception:  # pragma: no cover - GTK fallback
                active = False
        if not callable(getter):
            props = getattr(checkbox, "props", None)
            active = bool(getattr(props, "active", False)) if props is not None else bool(getattr(checkbox, "active", False))

        if active:
            self._bulk_selection.add(tool_name)
        else:
            self._bulk_selection.discard(tool_name)

        self._update_action_state()

    def _on_bulk_enable_clicked(self, _button: Gtk.Button) -> None:
        if not self._current_scope_allows_editing():
            return

        if not self._bulk_selection:
            return

        valid_names = set(self._entry_lookup.keys())
        self._enabled_tools.update(name for name in self._bulk_selection if name in valid_names)
        self._sync_active_tool_switch()
        self._update_action_state()

    def _on_bulk_apply_clicked(self, _button: Gtk.Button) -> None:
        if not self._current_scope_allows_editing():
            return

        recommended = self._compute_recommended_bulk_tools()
        if not recommended:
            return

        self._enabled_tools.update(recommended)
        self._sync_active_tool_switch()
        self._update_action_state()

    def _on_switch_state_set(self, _switch: Gtk.Switch, state: bool) -> bool:
        if self._suppress_toggle or self._active_tool is None:
            return False

        if not self._current_scope_allows_editing():
            return False

        if state:
            self._enabled_tools.add(self._active_tool)
        else:
            self._enabled_tools.discard(self._active_tool)

        self._update_action_state()
        return False

    def _on_save_clicked(self, _button: Gtk.Button) -> None:
        if not self._persona_name:
            self._handle_backend_error("No active persona is available for saving.")
            return

        manager = getattr(self.ATLAS, "persona_manager", None)
        setter = getattr(manager, "set_allowed_tools", None)
        if not callable(setter):
            self._handle_backend_error("Persona manager is unavailable. Unable to save tools.")
            return

        payload = sorted(self._enabled_tools)
        try:
            result = setter(self._persona_name, payload)
        except Exception as exc:
            logger.error("Failed to persist tools for persona '%s': %s", self._persona_name, exc, exc_info=True)
            self._handle_backend_error(str(exc) or "Unable to save tool configuration.")
            return

        if isinstance(result, Mapping) and not result.get("success", True):
            errors = result.get("errors") or []
            message = "; ".join(str(err) for err in errors if err) or str(result.get("error") or "Failed to save tool configuration.")
            self._handle_backend_error(message)
            return

        self._refresh_state()

    def _on_reset_clicked(self, _button: Gtk.Button) -> None:
        self._bulk_selection.clear()
        self._refresh_state()
        self._sync_bulk_checkbox_state()

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _format_auth(self, entry: _ToolEntry) -> str:
        auth = entry.auth
        base = entry.account_status or "Authentication: "

        if not entry.account_status:
            required = auth.get("required") if isinstance(auth, Mapping) else None
            if required:
                base += "Required"
            elif required is False:
                base += "Optional"
            else:
                base += "Unknown"

            provider = auth.get("provider") or auth.get("account") if isinstance(auth, Mapping) else None
            if provider:
                base += f" – {provider}"

            detail = None
            if isinstance(auth, Mapping):
                detail = auth.get("status") or auth.get("message")
            if detail:
                base += f" ({detail})"

        warning = None
        if isinstance(auth, Mapping):
            warning = auth.get("error") or auth.get("warning")
        if warning:
            base += f" – {warning}"

        return base

    def _iter_status_badges(self, entry: _ToolEntry) -> TypingIterable[Tuple[str, Sequence[str]]]:
        status_values: List[str] = []
        metadata = entry.raw_metadata or {}
        for key in ("account_status", "status", "state"):
            value = metadata.get(key)
            if isinstance(value, str):
                status_values.append(value)
        auth_status = entry.auth.get("status") if isinstance(entry.auth, Mapping) else None
        if isinstance(auth_status, str):
            status_values.append(auth_status)

        seen: set[str] = set()
        for value in status_values:
            normalized = value.strip()
            if not normalized:
                continue
            lowered = normalized.casefold()
            if lowered in seen:
                continue
            seen.add(lowered)
            css_class = self._status_to_css_class(lowered)
            yield normalized, ("tag-badge", css_class)

    def _iter_capability_badges(self, entry: _ToolEntry) -> TypingIterable[Tuple[str, Sequence[str]]]:
        for capability in entry.capabilities:
            text = capability.strip()
            if text:
                yield text, ("tag-badge", "capability-badge")

        metadata_caps = entry.raw_metadata.get("capability_tags") if isinstance(entry.raw_metadata, Mapping) else None
        if isinstance(metadata_caps, Iterable) and not isinstance(metadata_caps, (str, bytes)):
            for capability in metadata_caps:
                text = str(capability).strip()
                if text and text not in entry.capabilities:
                    yield text, ("tag-badge", "capability-badge")

    def _iter_warning_badges(self, entry: _ToolEntry) -> TypingIterable[Tuple[str, Sequence[str]]]:
        auth = entry.auth if isinstance(entry.auth, Mapping) else {}
        for key in ("warning", "error"):
            value = auth.get(key)
            if isinstance(value, str) and value.strip():
                yield value.strip(), ("tag-badge", "status-warning")

        metadata = entry.raw_metadata or {}
        warnings = metadata.get("warnings")
        if isinstance(warnings, Iterable) and not isinstance(warnings, (str, bytes)):
            for warning in warnings:
                text = str(warning).strip()
                if text:
                    yield text, ("tag-badge", "status-warning")

    def _sync_badge_container(
        self,
        container: Optional[Gtk.Box],
        badges: TypingIterable[Tuple[str, Sequence[str]]],
    ) -> None:
        if container is None:
            return

        self._clear_container(container)
        has_badges = False
        for text, css_classes in badges:
            has_badges = True
            container.append(self._create_badge(text, css_classes))
        container.set_visible(has_badges)

    def _sync_capability_box(self, entry: _ToolEntry) -> None:
        box = self._capability_badge_box
        if box is None:
            return

        self._clear_container(box)
        has_badges = False
        for text, css_classes in self._iter_capability_badges(entry):
            has_badges = True
            badge = self._create_badge(text, css_classes)
            if hasattr(box, "insert"):
                try:
                    box.insert(badge, -1)  # type: ignore[arg-type]
                    continue
                except Exception:  # pragma: no cover - GTK fallback
                    pass
            append = getattr(box, "append", None)
            if callable(append):
                append(badge)
        box.set_visible(has_badges)

    def _sync_warning_box(self, entry: _ToolEntry) -> None:
        box = self._warning_badge_box
        if box is None:
            return

        self._sync_badge_container(box, self._iter_warning_badges(entry))

    def _sync_cta_box(self, entry: _ToolEntry) -> None:
        box = self._cta_box
        if box is None:
            return

        self._clear_container(box)
        actions = list(self._iter_cta_actions(entry))
        if not actions:
            box.set_visible(False)
            return

        for label, callback in actions:
            button = Gtk.Button(label=label)
            try:
                button.add_css_class("suggested-action")
            except Exception:  # pragma: no cover - GTK theme variations
                pass
            button.connect("clicked", callback)
            box.append(button)

        box.set_visible(True)

    def _sync_bulk_checkbox_state(self) -> None:
        allow_bulk = self._current_scope_allows_editing()
        valid_names = set(self._entry_lookup.keys())
        self._bulk_selection.intersection_update(valid_names)

        for name, checkbox in list(self._bulk_checkbox_lookup.items()):
            if not isinstance(checkbox, Gtk.CheckButton):
                continue

            set_visible = getattr(checkbox, "set_visible", None)
            if callable(set_visible):
                set_visible(bool(allow_bulk))

            set_sensitive = getattr(checkbox, "set_sensitive", None)
            if callable(set_sensitive):
                set_sensitive(bool(allow_bulk))

            desired = allow_bulk and name in self._bulk_selection
            self._suppress_bulk_signals = True
            try:
                setter = getattr(checkbox, "set_active", None)
                if callable(setter):
                    setter(desired)
                else:  # pragma: no cover - GTK compatibility fallback
                    checkbox.props.active = desired
            finally:
                self._suppress_bulk_signals = False

    def _sync_active_tool_switch(self) -> None:
        allow_edit = self._current_scope_allows_editing()
        if not allow_edit or not self._active_tool:
            self._set_switch_active(False)
            return

        self._set_switch_active(self._active_tool in self._enabled_tools)

    def _compute_recommended_bulk_tools(self) -> set[str]:
        recommended = set(self._derive_persona_template_tools())

        if self._bulk_selection:
            selection = {name for name in self._bulk_selection if name in self._entry_lookup}
            recommended.update(selection)
            recommended.update(self._derive_dependency_tool_set(selection))

        if recommended:
            recommended.update(self._derive_dependency_tool_set(recommended))

        return {name for name in recommended if name in self._entry_lookup}

    def _derive_persona_template_tools(self) -> set[str]:
        persona_name = self._persona_name
        if not persona_name:
            return set()

        manager = getattr(self.ATLAS, "persona_manager", None)
        if manager is None:
            return set()

        persona_data: Optional[Mapping[str, Any]] = None
        editor_getter = getattr(manager, "get_editor_state", None)
        if callable(editor_getter):
            try:
                snapshot = editor_getter(persona_name)
            except Exception:  # pragma: no cover - defensive logging only
                logger.debug("Failed to load editor state for persona '%s'", persona_name, exc_info=True)
            else:
                if isinstance(snapshot, Mapping):
                    persona_data = snapshot

        if persona_data is None:
            persona_getter = getattr(manager, "get_persona", None)
            if callable(persona_getter):
                try:
                    persona_raw = persona_getter(persona_name)
                except Exception:  # pragma: no cover - defensive logging only
                    logger.debug("Failed to load persona '%s' for recommendations", persona_name, exc_info=True)
                    persona_raw = None
                if isinstance(persona_raw, Mapping):
                    persona_data = persona_raw

        if persona_data is None:
            return set()

        candidates: set[str] = set()
        candidates.update(self._normalize_tool_collection(persona_data.get("allowed_tools")))

        tools_state = persona_data.get("tools") if isinstance(persona_data, Mapping) else None
        if isinstance(tools_state, Mapping):
            for key in ("allowed", "recommended", "template", "defaults"):
                candidates.update(self._normalize_tool_collection(tools_state.get(key)))
            available = tools_state.get("available")
            if isinstance(available, Iterable) and not isinstance(available, (str, bytes)):
                for item in available:
                    if isinstance(item, Mapping) and item.get("enabled"):
                        candidates.update(self._normalize_tool_collection(item.get("name")))

        valid_names = set(self._entry_lookup.keys())
        return {name for name in candidates if name in valid_names}

    def _derive_dependency_tool_set(self, seeds: Iterable[str]) -> set[str]:
        dependencies: set[str] = set()
        queue = [name for name in seeds if name in self._entry_lookup]
        seen: set[str] = set()
        valid_names = set(self._entry_lookup.keys())

        metadata_keys = (
            "depends_on",
            "dependencies",
            "requires",
            "required_tools",
            "after",
            "prerequisites",
            "recommended_tools",
            "bundled_tools",
        )

        while queue:
            current = queue.pop()
            if current in seen:
                continue
            seen.add(current)

            entry = self._entry_lookup.get(current)
            metadata = entry.raw_metadata if entry is not None else None
            if not isinstance(metadata, Mapping):
                continue

            for key in metadata_keys:
                related = metadata.get(key)
                if related is None:
                    continue
                for dependency in self._normalize_tool_collection(related):
                    if dependency in valid_names and dependency not in dependencies:
                        dependencies.add(dependency)
                        queue.append(dependency)

        return dependencies

    def _normalize_tool_collection(self, value: Any) -> set[str]:
        names: set[str] = set()
        if value is None:
            return names

        if isinstance(value, str):
            normalized = value.strip()
            if normalized:
                names.add(normalized)
            return names

        if isinstance(value, Mapping):
            for key in ("name", "tool", "id"):
                candidate = value.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    names.add(candidate.strip())
            nested = value.get("items") or value.get("tools")
            if nested is not None:
                names.update(self._normalize_tool_collection(nested))
            return names

        if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
            for item in value:
                names.update(self._normalize_tool_collection(item))
            return names

        return names

    def _iter_cta_actions(
        self, entry: _ToolEntry
    ) -> TypingIterable[Tuple[str, Callable[[Gtk.Button], None]]]:
        auth = entry.auth if isinstance(entry.auth, Mapping) else {}
        provider_name = self._resolve_provider_name(auth, entry.raw_metadata)
        needs_attention = bool(list(self._iter_warning_badges(entry))) or self._needs_credentials(entry)

        persona_callback = self.on_open_in_persona
        if callable(persona_callback) and self._persona_name and entry.name:
            yield (
                "Configure in persona",
                lambda _btn, tool=entry.name: persona_callback(tool),
            )

        if needs_attention and provider_name:
            yield (
                "Fix credentials",
                lambda _btn, provider=provider_name: self._open_provider_settings(provider),
            )
        elif needs_attention:
            yield (
                "Manage accounts",
                lambda _btn: self._open_accounts_page(),
            )

    def _needs_credentials(self, entry: _ToolEntry) -> bool:
        metadata = entry.raw_metadata or {}
        account_status = metadata.get("account_status") or entry.account_status
        if isinstance(account_status, str) and account_status.strip():
            status = account_status.strip().casefold()
            if status in {"missing", "missing_credentials", "unauthorized", "error"}:
                return True

        auth = entry.auth if isinstance(entry.auth, Mapping) else {}
        required = auth.get("required")
        has_error = isinstance(auth.get("error"), str) and bool(auth.get("error").strip())
        return bool(required) and has_error

    def _resolve_provider_name(
        self, auth: Mapping[str, Any], metadata: Mapping[str, Any] | None
    ) -> Optional[str]:
        candidates = []
        for key in ("provider", "account", "name"):
            value = auth.get(key)
            if isinstance(value, str):
                candidates.append(value)
        if metadata and isinstance(metadata, Mapping):
            provider_value = metadata.get("provider") or metadata.get("account")
            if isinstance(provider_value, str):
                candidates.append(provider_value)

        for candidate in candidates:
            text = candidate.strip()
            if text:
                return text
        return None

    def _open_provider_settings(self, provider: str) -> None:
        window = getattr(self.parent_window, "provider_management", None)
        if window is not None and hasattr(window, "show_provider_settings"):
            try:
                window.show_provider_settings(provider)
                return
            except Exception:  # pragma: no cover - interface guard
                logger.debug("Failed to open provider settings for %s", provider, exc_info=True)

        fallback = getattr(self.parent_window, "show_provider_menu", None)
        if callable(fallback):
            fallback()

    def _open_accounts_page(self) -> None:
        opener = getattr(self.parent_window, "show_accounts_page", None)
        if callable(opener):
            opener()

    def _create_badge(self, text: str, css_classes: Sequence[str]) -> Gtk.Widget:
        box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        label = Gtk.Label(label=text)
        label.set_xalign(0.0)
        label.set_wrap(False)
        for css_class in css_classes:
            try:
                label.add_css_class(css_class)
                box.add_css_class(css_class)
            except Exception:  # pragma: no cover - GTK theme variations
                pass
        box.append(label)
        return box

    def _clear_container(self, container: Optional[Gtk.Widget]) -> None:
        if container is None:
            return

        if hasattr(container, "remove_all"):
            try:
                container.remove_all()  # type: ignore[attr-defined]
                return
            except Exception:  # pragma: no cover - GTK fallback
                pass

        children = self._get_container_children(container)
        remover = getattr(container, "remove", None)
        if callable(remover):
            for child in children:
                try:
                    remover(child)
                except Exception:  # pragma: no cover - GTK fallback
                    continue

    def _get_container_children(self, container: Gtk.Widget) -> List[Gtk.Widget]:
        getter = getattr(container, "get_children", None)
        if callable(getter):
            try:
                return list(getter())
            except Exception:  # pragma: no cover - GTK fallback
                pass

        children: List[Gtk.Widget] = []
        get_first_child = getattr(container, "get_first_child", None)
        if callable(get_first_child):
            current = get_first_child()
            while current is not None:
                children.append(current)
                get_next = getattr(current, "get_next_sibling", None)
                current = get_next() if callable(get_next) else None
        else:
            raw_children = getattr(container, "children", None)
            if isinstance(raw_children, Iterable):
                children.extend(raw_children)
        return children

    def _status_to_css_class(self, status: str) -> str:
        if status in {"ok", "enabled", "healthy", "ready", "active"}:
            return "status-ok"
        if status in {"degraded", "warning", "partial", "stale"}:
            return "status-warning"
        if status in {"missing", "missing_credentials", "error", "failed", "unauthorized", "disabled"}:
            return "status-error"
        return "status-unknown"

    def _format_last_success(self, entry: _ToolEntry) -> Optional[str]:
        timestamp = self._extract_last_success(entry.raw_metadata)
        if timestamp is None:
            return None
        relative = self._format_relative_time(timestamp)
        return f"Last success {relative}"

    def _format_detail_telemetry(self, entry: _ToolEntry) -> str:
        pieces: List[str] = []
        last_success = self._format_last_success(entry)
        if last_success:
            pieces.append(last_success)

        health_summary = self._format_health_summary(entry)
        if health_summary:
            pieces.append(health_summary)

        if not pieces:
            return "Telemetry unavailable."
        return " · ".join(pieces)

    def _format_health_summary(self, entry: _ToolEntry) -> Optional[str]:
        metadata = entry.raw_metadata or {}
        health = metadata.get("health")
        if isinstance(health, Mapping):
            state = health.get("state")
            metrics = health.get("metrics")
        else:
            state = None
            metrics = None

        text_parts: List[str] = []
        if isinstance(state, str) and state.strip():
            text_parts.append(f"State: {state.strip()}")

        if isinstance(metrics, Mapping):
            metric_parts = []
            for key, value in metrics.items():
                metric_parts.append(f"{key}: {value}")
            if metric_parts:
                text_parts.append("Metrics: " + ", ".join(metric_parts))

        if text_parts:
            return " ".join(text_parts)
        return None

    def _extract_last_success(self, metadata: Mapping[str, Any]) -> Optional[datetime]:
        candidates = []
        for key in ("last_success_at", "last_success", "lastSuccess", "last_success_ts"):
            if metadata and key in metadata:
                candidates.append(metadata.get(key))
        if metadata:
            health = metadata.get("health")
            if isinstance(health, Mapping) and "last_success_at" in health:
                candidates.append(health.get("last_success_at"))

        for candidate in candidates:
            parsed = self._parse_timestamp(candidate)
            if parsed is not None:
                return parsed
        return None

    def _parse_timestamp(self, value: Any) -> Optional[datetime]:
        if isinstance(value, (int, float)):
            timestamp = float(value)
            if timestamp > 10**12:  # assume milliseconds
                timestamp /= 1000.0
            try:
                return datetime.fromtimestamp(timestamp, tz=timezone.utc)
            except (OverflowError, OSError, ValueError):
                return None

        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            try:
                return datetime.fromisoformat(text)
            except ValueError:
                try:
                    numeric = float(text)
                except ValueError:
                    return None
                return self._parse_timestamp(numeric)
        return None

    def _format_relative_time(self, timestamp: datetime) -> str:
        now = datetime.now(timezone.utc)
        delta = now - timestamp
        total_seconds = max(int(delta.total_seconds()), 0)

        if total_seconds < 60:
            return "moments ago"
        minutes = total_seconds // 60
        if minutes < 60:
            return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
        hours = minutes // 60
        if hours < 24:
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
        days = hours // 24
        if days < 7:
            return f"{days} day{'s' if days != 1 else ''} ago"
        weeks = days // 7
        if weeks < 4:
            return f"{weeks} week{'s' if weeks != 1 else ''} ago"
        months = days // 30
        if months < 12:
            return f"{months} month{'s' if months != 1 else ''} ago"
        years = days // 365
        return f"{years} year{'s' if years != 1 else ''} ago"

    def _set_label(self, label: Optional[Gtk.Label], text: str) -> None:
        if label is None:
            return
        setter = getattr(label, "set_label", None) or getattr(label, "set_text", None)
        if callable(setter):
            setter(text)
        else:  # pragma: no cover - GTK compatibility fallback
            label.label = text

    def _set_switch_sensitive(self, enabled: bool) -> None:
        if self._switch is None:
            return
        setter = getattr(self._switch, "set_sensitive", None)
        if callable(setter):
            setter(bool(enabled))

    def _set_switch_active(self, active: bool) -> None:
        if self._switch is None:
            return
        setter = getattr(self._switch, "set_active", None)
        self._suppress_toggle = True
        try:
            if callable(setter):
                setter(bool(active))
            else:  # pragma: no cover - GTK compatibility fallback
                self._switch.props.active = bool(active)
        finally:
            self._suppress_toggle = False

    def _update_action_state(self) -> None:
        has_tools = bool(self._entries)
        if not has_tools:
            self._set_button_sensitive(self._save_button, False)
            self._set_button_sensitive(self._bulk_enable_button, False)
            self._set_button_sensitive(self._bulk_apply_button, False)
            self._set_button_sensitive(self._reset_button, False)
            self._sync_bulk_checkbox_state()
            return

        dirty = self._enabled_tools != self._baseline_enabled
        self._set_button_sensitive(self._save_button, dirty and bool(self._persona_name))
        self._set_button_sensitive(self._reset_button, True)

        allow_bulk = self._current_scope_allows_editing()
        selection_available = bool(self._bulk_selection)
        recommended = self._compute_recommended_bulk_tools() if allow_bulk else set()
        self._set_button_sensitive(self._bulk_enable_button, allow_bulk and selection_available)
        self._set_button_sensitive(self._bulk_apply_button, allow_bulk and bool(recommended))

        self._sync_bulk_checkbox_state()

    def _set_button_sensitive(self, button: Optional[Gtk.Button], enabled: bool) -> None:
        if button is None:
            return
        setter = getattr(button, "set_sensitive", None)
        if callable(setter):
            setter(bool(enabled))

    def _populate_capability_selector(self) -> None:
        capabilities = sorted({cap for entry in self._entries for cap in entry.capabilities})
        options = [self._CAPABILITY_ALL_ID] + capabilities
        self._capability_options = options

        if self._capability_filter and self._capability_filter not in capabilities:
            self._capability_filter = None
            if self._preferences_loaded_key is not None:
                self._persist_view_preferences()

        selector = self._capability_selector
        if selector is None:
            return

        self._clear_combo_box(selector)
        append = getattr(selector, "append", None)
        append_text = getattr(selector, "append_text", None)
        for option in options:
            label = "All capabilities" if option == self._CAPABILITY_ALL_ID else option
            try:
                if callable(append):
                    append(option, label)
                elif callable(append_text):
                    append_text(label)
            except Exception:  # pragma: no cover - GTK fallback
                continue

    def _populate_sort_selector(self) -> None:
        self._sort_option_ids = [key for key, _ in self._SORT_OPTIONS]
        selector = self._sort_selector
        if selector is None:
            self._normalize_sort_key(persist=False)
            return

        self._clear_combo_box(selector)
        append = getattr(selector, "append", None)
        append_text = getattr(selector, "append_text", None)
        for key, label in self._SORT_OPTIONS:
            try:
                if callable(append):
                    append(key, label)
                elif callable(append_text):
                    append_text(label)
            except Exception:  # pragma: no cover - GTK fallback
                continue

        self._normalize_sort_key(persist=False)

    def _sync_filter_widgets(self) -> None:
        if not any((self._search_entry, self._capability_selector, self._sort_selector)):
            return

        self._suppress_filter_signals = True
        try:
            if self._search_entry is not None:
                getter = getattr(self._search_entry, "get_text", None)
                try:
                    current = getter() if callable(getter) else None
                except Exception:  # pragma: no cover - GTK fallback
                    current = None
                if not isinstance(current, str):
                    current = getattr(self._search_entry, "text", None)
                if not isinstance(current, str):
                    props = getattr(self._search_entry, "props", None)
                    current = getattr(props, "text", "") if props is not None else ""
                desired = self._filter_text or ""
                if current != desired:
                    setter = getattr(self._search_entry, "set_text", None)
                    try:
                        if callable(setter):
                            setter(desired)
                        else:
                            setattr(self._search_entry, "text", desired)
                    except Exception:  # pragma: no cover - GTK fallback
                        pass

            if self._capability_selector is not None:
                desired_capability = self._capability_filter or self._CAPABILITY_ALL_ID
                self._set_combo_selection(self._capability_selector, self._capability_options, desired_capability)

            if self._sort_selector is not None:
                desired_sort = self._normalize_sort_key(persist=False)
                self._set_combo_selection(self._sort_selector, self._sort_option_ids, desired_sort)
        finally:
            self._suppress_filter_signals = False

    def _clear_combo_box(self, combo: Gtk.ComboBoxText) -> None:
        remover = getattr(combo, "remove_all", None)
        if callable(remover):
            try:
                remover()
                return
            except Exception:  # pragma: no cover - GTK fallback
                pass

        remove = getattr(combo, "remove", None)
        if callable(remove):
            while True:
                try:
                    remove(0)
                except Exception:
                    break

    def _set_combo_selection(self, combo: Gtk.ComboBoxText, options: List[str], desired_id: str) -> None:
        if not options:
            return
        if desired_id not in options:
            desired_id = options[0]

        setter_id = getattr(combo, "set_active_id", None)
        if callable(setter_id):
            try:
                setter_id(desired_id)
                return
            except Exception:  # pragma: no cover - GTK fallback
                pass

        set_active = getattr(combo, "set_active", None)
        if callable(set_active):
            try:
                index = options.index(desired_id)
            except ValueError:
                index = 0
            try:
                set_active(index)
            except Exception:  # pragma: no cover - GTK fallback
                pass

    def _get_combo_active_id(self, combo: Gtk.ComboBoxText, options: List[str]) -> Optional[str]:
        getter = getattr(combo, "get_active_id", None)
        if callable(getter):
            try:
                active_id = getter()
            except Exception:  # pragma: no cover - GTK fallback
                active_id = None
            if isinstance(active_id, str):
                return active_id
            if active_id is not None:
                return str(active_id)

        get_active = getattr(combo, "get_active", None)
        if callable(get_active):
            try:
                index = get_active()
            except Exception:  # pragma: no cover - GTK fallback
                index = None
            if isinstance(index, int) and 0 <= index < len(options):
                return options[index]
        return None

    def _derive_visible_entries(self) -> List[_ToolEntry]:
        query = (self._filter_text or "").strip().casefold()
        capability = self._capability_filter

        visible: List[_ToolEntry] = []
        for entry in self._entries:
            if query:
                haystack_parts = [
                    entry.name,
                    entry.title,
                    entry.summary,
                    " ".join(entry.capabilities),
                ]
                haystack = " ".join(part for part in haystack_parts if part).casefold()
                if query not in haystack:
                    continue

            if capability and capability not in entry.capabilities:
                continue

            visible.append(entry)

        return self._sort_entries(visible)

    def _sort_entries(self, entries: List[_ToolEntry]) -> List[_ToolEntry]:
        if not entries:
            return []

        sort_key = self._normalize_sort_key(persist=False)

        def title_key(item: _ToolEntry) -> tuple[str, str]:
            primary = (item.title or item.name or "").casefold()
            secondary = (item.name or "").casefold()
            return primary, secondary

        def name_key(item: _ToolEntry) -> tuple[str, str]:
            primary = (item.name or "").casefold()
            secondary = (item.title or item.name or "").casefold()
            return primary, secondary

        if sort_key == "title_desc":
            return sorted(entries, key=title_key, reverse=True)
        if sort_key == "name_asc":
            return sorted(entries, key=name_key)
        if sort_key == "name_desc":
            return sorted(entries, key=name_key, reverse=True)
        return sorted(entries, key=title_key)

    def _normalize_sort_key(self, *, persist: bool = False) -> str:
        valid_keys = self._sort_option_ids or [self._DEFAULT_SORT_KEY]
        desired = self._sort_key if self._sort_key in valid_keys else valid_keys[0]
        if desired != self._sort_key:
            self._sort_key = desired
            if persist and self._preferences_loaded_key is not None:
                self._persist_view_preferences()
        return desired

    def _load_view_preferences(self, persona: Optional[str]) -> None:
        storage = self._get_settings_bucket()
        key = self._get_preferences_key(persona)
        record = storage.get(key)
        if not isinstance(record, Mapping):
            record = {}

        text_value = record.get("filter_text")
        self._filter_text = str(text_value) if text_value is not None else ""

        capability_value = record.get("capability")
        if capability_value:
            self._capability_filter = str(capability_value)
        else:
            self._capability_filter = None

        sort_value = record.get("sort_key")
        self._sort_key = str(sort_value) if isinstance(sort_value, str) else self._DEFAULT_SORT_KEY

        self._preferences_loaded_key = key
        self._normalize_sort_key(persist=True)

    def _persist_view_preferences(self) -> None:
        key = self._get_preferences_key(self._persona_name)
        storage = self._get_settings_bucket()

        normalized_sort = self._normalize_sort_key(persist=False)
        capability_value = (
            str(self._capability_filter)
            if self._capability_filter is not None
            else None
        )

        payload = {
            "filter_text": str(self._filter_text or ""),
            "capability": capability_value,
            "sort_key": normalized_sort,
        }

        storage[key] = payload
        self._preferences_loaded_key = key

    def _get_preferences_key(self, persona: Optional[str]) -> str:
        if persona:
            return f"persona::{persona}"
        return "persona::__global__"

    def _get_settings_bucket(self) -> Dict[str, Any]:
        settings = getattr(self.ATLAS, "settings", None)
        if not isinstance(settings, dict):
            settings = {}
            setattr(self.ATLAS, "settings", settings)

        bucket = settings.get("tool_management")
        if not isinstance(bucket, dict):
            bucket = {}
            settings["tool_management"] = bucket

        return bucket

    def _on_scope_changed(self, combo: Gtk.ComboBoxText) -> None:
        if self._suppress_scope_signal:
            return

        getter = getattr(combo, "get_active_text", None)
        label = getter() if callable(getter) else None
        new_scope = "persona" if label == "Persona tools" else "all"

        if new_scope == "persona" and not self._persona_name:
            self._tool_scope = "all"
            self._sync_scope_widget(False)
            return

        if new_scope == self._tool_scope:
            return

        self._tool_scope = new_scope
        self._refresh_state()

    def _sync_scope_widget(self, persona_available: bool) -> None:
        widget = self._scope_selector
        if widget is None:
            return

        desired_scope = self._tool_scope
        if desired_scope == "persona" and not persona_available:
            desired_scope = "all"
            self._tool_scope = "all"

        index = 0 if desired_scope == "persona" else 1
        setter = getattr(widget, "set_active", None)
        self._suppress_scope_signal = True
        try:
            if callable(setter):
                setter(index)
        finally:
            self._suppress_scope_signal = False

        set_sensitive = getattr(widget, "set_sensitive", None)
        if callable(set_sensitive):
            set_sensitive(bool(persona_available))

    def _current_scope_allows_editing(self) -> bool:
        return self._tool_scope == "persona" and bool(self._persona_name)

    def _handle_backend_error(self, message: str) -> None:
        logger.error("Tool management error: %s", message)
        dialog = getattr(self.parent_window, "show_error_dialog", None)
        if callable(dialog):
            try:
                dialog(message)
            except Exception:  # pragma: no cover - user interface edge cases
                logger.debug("Failed to show tool management error dialog", exc_info=True)

