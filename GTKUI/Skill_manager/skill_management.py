"""GTK skill management workspace used by the sidebar."""

from __future__ import annotations

import csv
import json
import logging
import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import GLib, Gtk

from ATLAS.utils import normalize_sequence

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _SkillEntry:
    """Normalized skill metadata returned by the ATLAS server."""

    name: str
    summary: str
    version: Optional[str]
    persona: Optional[str]
    category: Optional[str]
    safety_notes: Optional[str]
    required_tools: List[str]
    required_capabilities: List[str]
    capability_tags: List[str]
    source: Optional[str]
    raw_metadata: Dict[str, Any]
    review_status: Optional[str]
    tester_notes: Optional[str]


@dataclass(slots=True)
class _RequirementStatus:
    """Represents persona compatibility metadata for a skill."""

    persona_name: Optional[str]
    missing_tools: List[str]
    missing_capabilities: List[str]
    satisfied_tools: List[str]
    satisfied_capabilities: List[str]
    evaluation_error: Optional[str] = None
    persona_mismatch: bool = False
    show_dialog: bool = False


class SkillManagement:
    """Controller responsible for rendering the skill management workspace."""

    _CATEGORY_ALL_ID = "__all__categories__"
    _PERSONA_ALL_ID = "__all__personas__"
    _PERSONA_SHARED_ID = "__shared__personas__"
    _DEFAULT_SORT_ORDER = "name_asc"

    def __init__(self, atlas: Any, parent_window: Any) -> None:
        self.ATLAS = atlas
        self.parent_window = parent_window

        self.on_open_in_persona: Optional[Callable[[str], None]] = None

        self._widget: Optional[Gtk.Widget] = None
        self._skill_list: Optional[Gtk.ListBox] = None
        self._title_label: Optional[Gtk.Label] = None
        self._summary_label: Optional[Gtk.Label] = None
        self._persona_label: Optional[Gtk.Label] = None
        self._version_label: Optional[Gtk.Label] = None
        self._category_label: Optional[Gtk.Label] = None
        self._required_tools_label: Optional[Gtk.Label] = None
        self._required_capabilities_label: Optional[Gtk.Label] = None
        self._capability_tags_label: Optional[Gtk.Label] = None
        self._safety_notes_label: Optional[Gtk.Label] = None
        self._source_label: Optional[Gtk.Label] = None
        self._tool_badge_box: Optional[Gtk.Widget] = None
        self._capability_badge_box: Optional[Gtk.Widget] = None
        self._requirement_status_label: Optional[Gtk.Label] = None
        self._review_status_label: Optional[Gtk.Label] = None
        self._action_button_box: Optional[Gtk.Widget] = None
        self._enable_tools_button: Optional[Gtk.Button] = None
        self._open_tool_manager_button: Optional[Gtk.Button] = None
        self._open_persona_button: Optional[Gtk.Button] = None
        self._preview_button: Optional[Gtk.Button] = None
        self._test_button: Optional[Gtk.Button] = None
        self._mark_reviewed_button: Optional[Gtk.Button] = None
        self._test_status_label: Optional[Gtk.Label] = None
        self._test_spinner: Optional[Gtk.Spinner] = None
        self._test_output_view: Optional[Gtk.Widget] = None
        self._test_output_buffer: Optional[Gtk.TextBuffer] = None
        self._tester_notes_view: Optional[Gtk.Widget] = None
        self._tester_notes_buffer: Optional[Gtk.TextBuffer] = None
        self._scope_selector: Optional[Gtk.ComboBoxText] = None
        self._search_entry: Optional[Gtk.SearchEntry] = None
        self._category_combo: Optional[Gtk.ComboBoxText] = None
        self._persona_combo: Optional[Gtk.ComboBoxText] = None
        self._sort_combo: Optional[Gtk.ComboBoxText] = None
        self._recent_changes_box: Optional[Gtk.Widget] = None
        self._recent_changes_label: Optional[Gtk.Label] = None
        self._export_csv_button: Optional[Gtk.Button] = None
        self._export_json_button: Optional[Gtk.Button] = None
        self._export_bundle_button: Optional[Gtk.Button] = None
        self._import_bundle_button: Optional[Gtk.Button] = None
        self._history_box: Optional[Gtk.Widget] = None
        self._history_list_box: Optional[Gtk.Widget] = None
        self._history_records: List[Dict[str, Any]] = []
        self._skill_history_supported = False

        self._entries: List[_SkillEntry] = []
        self._entry_lookup: Dict[str, _SkillEntry] = {}
        self._row_lookup: Dict[str, Gtk.Widget] = {}
        self._row_review_labels: Dict[str, Gtk.Label] = {}
        self._display_entries: List[_SkillEntry] = []
        self._visible_entry_names: Set[str] = set()
        self._active_skill: Optional[str] = None
        self._active_requirement_status: Optional[_RequirementStatus] = None
        self._skill_scope = "persona"
        self._persona_name: Optional[str] = None
        self._suppress_scope_signal = False
        self._suppress_filter_signal = False

        self._filter_text = ""
        self._category_filter: Optional[str] = None
        self._persona_filter: Optional[str] = None
        self._sort_order = self._DEFAULT_SORT_ORDER

        self._category_option_lookup: Dict[str, Optional[str]] = {}
        self._category_reverse_lookup: Dict[Optional[str], str] = {}
        self._persona_option_lookup: Dict[str, Optional[str]] = {}
        self._persona_reverse_lookup: Dict[Optional[str], str] = {}
        self._preferences_loaded_key: Optional[str] = None
        self._requirement_cache: Dict[str, _RequirementStatus] = {}
        self._persona_error_reported = False
        self._preview_in_progress = False
        self._test_in_progress = False
        self._baseline_notes: str = ""
        self._baseline_status: str = ""
        self._recent_changes_snapshot: Optional[Tuple[str, ...]] = None
        self._last_export_directory: Optional[Path] = None
        self._suppress_notes_signal = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_embeddable_widget(self) -> Gtk.Widget:
        """Return the workspace widget, creating it on first use."""

        if self._widget is None:
            self._widget = self._build_workspace()
        self._refresh_state()
        return self._widget

    def focus_skill(self, skill_name: str) -> bool:
        """Ensure ``skill_name`` is visible and selected in the catalog."""

        if not skill_name:
            return False

        if skill_name not in self._entry_lookup:
            if self._skill_scope != "all":
                self._skill_scope = "all"
                self._sync_scope_widget(bool(self._persona_name))
                self._refresh_state()
        if skill_name not in self._entry_lookup:
            return False

        if skill_name not in self._visible_entry_names:
            filters_changed = False
            if self._filter_text:
                self._filter_text = ""
                filters_changed = True
            if self._category_filter is not None:
                self._category_filter = None
                filters_changed = True
            if self._persona_filter is not None:
                self._persona_filter = None
                filters_changed = True
            if filters_changed:
                self._persist_view_preferences()
                self._populate_filter_options()
            self._rebuild_skill_list()

        if skill_name not in self._visible_entry_names:
            return False

        self._select_skill(skill_name)
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

        heading = Gtk.Label(label="Available Skills")
        heading.set_xalign(0.0)
        left_panel.append(heading)

        scope_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        scope_row.set_hexpand(True)

        scope_label = Gtk.Label(label="Skill scope")
        scope_label.set_xalign(0.0)
        scope_label.set_hexpand(True)
        scope_row.append(scope_label)

        scope_selector = Gtk.ComboBoxText()
        scope_selector.append_text("Persona skills")
        scope_selector.append_text("All skills")
        scope_selector.set_active(0)
        scope_selector.connect("changed", self._on_scope_changed)
        scope_row.append(scope_selector)
        self._scope_selector = scope_selector

        left_panel.append(scope_row)

        filter_column = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        filter_column.set_hexpand(True)

        SearchEntryClass = getattr(Gtk, "SearchEntry", None)
        if SearchEntryClass is None:
            SearchEntryClass = getattr(Gtk, "Entry", Gtk.Widget)
        search_entry = SearchEntryClass()
        search_entry.set_hexpand(True)
        try:
            search_entry.set_placeholder_text("Search skills")
        except Exception:  # pragma: no cover - GTK stub variations
            pass
        search_entry.connect("search-changed", self._on_search_changed)
        filter_column.append(search_entry)
        self._search_entry = search_entry

        filter_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        filter_row.set_hexpand(True)

        category_column = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        category_column.set_hexpand(True)
        category_label = Gtk.Label(label="Category")
        category_label.set_xalign(0.0)
        category_column.append(category_label)

        category_combo = Gtk.ComboBoxText()
        category_combo.set_hexpand(True)
        category_combo.connect("changed", self._on_category_filter_changed)
        category_column.append(category_combo)
        self._category_combo = category_combo

        filter_row.append(category_column)

        persona_column = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        persona_column.set_hexpand(True)
        persona_label = Gtk.Label(label="Persona")
        persona_label.set_xalign(0.0)
        persona_column.append(persona_label)

        persona_combo = Gtk.ComboBoxText()
        persona_combo.set_hexpand(True)
        persona_combo.connect("changed", self._on_persona_filter_changed)
        persona_column.append(persona_combo)
        self._persona_combo = persona_combo

        filter_row.append(persona_column)

        filter_column.append(filter_row)

        sort_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        sort_row.set_hexpand(True)

        sort_label = Gtk.Label(label="Sort")
        sort_label.set_xalign(0.0)
        sort_row.append(sort_label)

        sort_combo = Gtk.ComboBoxText()
        sort_combo.set_hexpand(True)
        for key, label in self._get_sort_options().items():
            sort_combo.append(key, label)
        sort_combo.set_active_id(self._normalize_sort_order())
        sort_combo.connect("changed", self._on_sort_order_changed)
        sort_row.append(sort_combo)
        self._sort_combo = sort_combo

        filter_column.append(sort_row)

        left_panel.append(filter_column)

        skill_list = Gtk.ListBox()
        skill_list.connect("row-selected", self._on_row_selected)
        self._skill_list = skill_list

        scroller = Gtk.ScrolledWindow()
        scroller.set_hexpand(False)
        scroller.set_vexpand(True)
        scroller.set_child(skill_list)
        left_panel.append(scroller)

        right_panel = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        right_panel.set_hexpand(True)
        right_panel.set_vexpand(True)

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

        persona_label = Gtk.Label()
        persona_label.set_wrap(True)
        persona_label.set_xalign(0.0)
        right_panel.append(persona_label)
        self._persona_label = persona_label

        version_label = Gtk.Label()
        version_label.set_wrap(True)
        version_label.set_xalign(0.0)
        right_panel.append(version_label)
        self._version_label = version_label

        category_label = Gtk.Label()
        category_label.set_wrap(True)
        category_label.set_xalign(0.0)
        right_panel.append(category_label)
        self._category_label = category_label

        required_tools_label = Gtk.Label()
        required_tools_label.set_wrap(True)
        required_tools_label.set_xalign(0.0)
        right_panel.append(required_tools_label)
        self._required_tools_label = required_tools_label

        FlowBoxClass = getattr(Gtk, "FlowBox", Gtk.Box)
        tool_badge_box = FlowBoxClass()
        configure = getattr(tool_badge_box, "set_selection_mode", None)
        if callable(configure):
            try:
                configure(Gtk.SelectionMode.NONE)
            except Exception:  # pragma: no cover - GTK compatibility fallbacks
                pass
        setter = getattr(tool_badge_box, "set_max_children_per_line", None)
        if callable(setter):
            try:
                setter(4)
            except Exception:  # pragma: no cover - GTK compatibility fallbacks
                pass
        spacing = getattr(tool_badge_box, "set_row_spacing", None)
        if callable(spacing):
            try:
                spacing(4)
            except Exception:  # pragma: no cover - GTK compatibility fallbacks
                pass
        spacing = getattr(tool_badge_box, "set_column_spacing", None)
        if callable(spacing):
            try:
                spacing(6)
            except Exception:  # pragma: no cover - GTK compatibility fallbacks
                pass
        tool_badge_box.set_visible(False)
        right_panel.append(tool_badge_box)
        self._tool_badge_box = tool_badge_box

        required_capabilities_label = Gtk.Label()
        required_capabilities_label.set_wrap(True)
        required_capabilities_label.set_xalign(0.0)
        right_panel.append(required_capabilities_label)
        self._required_capabilities_label = required_capabilities_label

        capability_badge_box = FlowBoxClass()
        configure = getattr(capability_badge_box, "set_selection_mode", None)
        if callable(configure):
            try:
                configure(Gtk.SelectionMode.NONE)
            except Exception:  # pragma: no cover - GTK compatibility fallbacks
                pass
        setter = getattr(capability_badge_box, "set_max_children_per_line", None)
        if callable(setter):
            try:
                setter(4)
            except Exception:  # pragma: no cover - GTK compatibility fallbacks
                pass
        spacing = getattr(capability_badge_box, "set_row_spacing", None)
        if callable(spacing):
            try:
                spacing(4)
            except Exception:  # pragma: no cover - GTK compatibility fallbacks
                pass
        spacing = getattr(capability_badge_box, "set_column_spacing", None)
        if callable(spacing):
            try:
                spacing(6)
            except Exception:  # pragma: no cover - GTK compatibility fallbacks
                pass
        capability_badge_box.set_visible(False)
        right_panel.append(capability_badge_box)
        self._capability_badge_box = capability_badge_box

        capability_tags_label = Gtk.Label()
        capability_tags_label.set_wrap(True)
        capability_tags_label.set_xalign(0.0)
        right_panel.append(capability_tags_label)
        self._capability_tags_label = capability_tags_label

        requirement_status_label = Gtk.Label()
        requirement_status_label.set_wrap(True)
        requirement_status_label.set_xalign(0.0)
        right_panel.append(requirement_status_label)
        self._requirement_status_label = requirement_status_label

        review_status_label = Gtk.Label()
        review_status_label.set_wrap(True)
        review_status_label.set_xalign(0.0)
        right_panel.append(review_status_label)
        self._review_status_label = review_status_label

        recent_changes_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=3)
        recent_changes_box.set_visible(False)
        recent_heading = Gtk.Label(label="Recent changes")
        recent_heading.set_xalign(0.0)
        try:
            recent_heading.add_css_class("title-5")
        except Exception:  # pragma: no cover - GTK theme variations
            pass
        recent_changes_box.append(recent_heading)

        recent_label = Gtk.Label()
        recent_label.set_wrap(True)
        recent_label.set_xalign(0.0)
        recent_changes_box.append(recent_label)

        right_panel.append(recent_changes_box)
        self._recent_changes_box = recent_changes_box
        self._recent_changes_label = recent_label

        export_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        export_row.set_halign(Gtk.Align.END)

        export_csv_button = Gtk.Button(label="Export to CSV")
        export_csv_button.set_tooltip_text("Save the filtered skill list to a CSV file.")
        export_csv_button.connect("clicked", self._on_export_skills_csv_clicked)
        export_row.append(export_csv_button)
        self._export_csv_button = export_csv_button

        export_json_button = Gtk.Button(label="Export to JSON")
        export_json_button.set_tooltip_text("Save the filtered skill list to a JSON file.")
        export_json_button.connect("clicked", self._on_export_skills_json_clicked)
        export_row.append(export_json_button)
        self._export_json_button = export_json_button

        export_bundle_button = Gtk.Button(label="Export bundle")
        export_bundle_button.set_tooltip_text("Export the selected skill to a signed bundle.")
        export_bundle_button.connect("clicked", self._on_export_skill_bundle_clicked)
        export_row.append(export_bundle_button)
        self._export_bundle_button = export_bundle_button

        import_bundle_button = Gtk.Button(label="Import bundle")
        import_bundle_button.set_tooltip_text("Import a skill bundle from disk.")
        import_bundle_button.connect("clicked", self._on_import_skill_bundle_clicked)
        export_row.append(import_bundle_button)
        self._import_bundle_button = import_bundle_button

        right_panel.append(export_row)

        action_button_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        action_button_box.set_visible(False)
        right_panel.append(action_button_box)
        self._action_button_box = action_button_box

        enable_tools_button = Gtk.Button(label="Enable missing tools")
        enable_tools_button.set_tooltip_text("Automatically enable required tools for the active persona.")
        enable_tools_button.connect("clicked", self._on_enable_missing_tools_clicked)
        enable_tools_button.set_visible(False)
        action_button_box.append(enable_tools_button)
        self._enable_tools_button = enable_tools_button

        open_tool_manager_button = Gtk.Button(label="Open tool manager")
        open_tool_manager_button.set_tooltip_text("Review persona tool access in the management workspace.")
        open_tool_manager_button.connect("clicked", self._on_open_tool_manager_clicked)
        open_tool_manager_button.set_visible(False)
        action_button_box.append(open_tool_manager_button)
        self._open_tool_manager_button = open_tool_manager_button

        open_persona_button = Gtk.Button(label="Configure in persona")
        open_persona_button.set_tooltip_text("Open the active persona editor to configure this skill.")
        open_persona_button.connect("clicked", self._on_open_in_persona_clicked)
        open_persona_button.set_visible(False)
        action_button_box.append(open_persona_button)
        self._open_persona_button = open_persona_button

        history_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=3)
        history_box.set_visible(False)
        history_heading = Gtk.Label(label="Recent backend activity")
        history_heading.set_xalign(0.0)
        try:
            history_heading.add_css_class("title-5")
        except Exception:  # pragma: no cover - GTK theme variations
            pass
        history_box.append(history_heading)

        history_list = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        history_list.set_hexpand(True)
        history_box.append(history_list)

        right_panel.append(history_box)
        self._history_box = history_box
        self._history_list_box = history_list

        control_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        right_panel.append(control_row)

        preview_button = Gtk.Button(label="Preview prompt")
        preview_button.set_sensitive(False)
        preview_button.set_tooltip_text("Preview the instruction prompt for this skill.")
        preview_button.connect("clicked", self._on_preview_prompt_clicked)
        control_row.append(preview_button)
        self._preview_button = preview_button

        test_button = Gtk.Button(label="Run test invocation")
        test_button.set_sensitive(False)
        test_button.set_tooltip_text("Execute a lightweight validation call for this skill.")
        test_button.connect("clicked", self._on_run_test_invocation_clicked)
        control_row.append(test_button)
        self._test_button = test_button

        mark_reviewed_button = Gtk.Button(label="Mark as reviewed")
        mark_reviewed_button.set_sensitive(False)
        mark_reviewed_button.set_tooltip_text("Persist review status and tester notes for this skill.")
        mark_reviewed_button.connect("clicked", self._on_mark_reviewed_clicked)
        control_row.append(mark_reviewed_button)
        self._mark_reviewed_button = mark_reviewed_button

        test_panel = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        test_panel.set_hexpand(True)
        right_panel.append(test_panel)

        test_header = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        test_panel.append(test_header)

        test_label = Gtk.Label(label="Test invocation results")
        test_label.set_xalign(0.0)
        test_label.set_hexpand(True)
        test_header.append(test_label)

        test_spinner = Gtk.Spinner()
        test_spinner.set_visible(False)
        test_header.append(test_spinner)
        self._test_spinner = test_spinner

        test_status_label = Gtk.Label()
        test_status_label.set_wrap(True)
        test_status_label.set_xalign(0.0)
        test_panel.append(test_status_label)
        self._test_status_label = test_status_label

        test_output_scroller = Gtk.ScrolledWindow()
        test_output_scroller.set_hexpand(True)
        test_output_scroller.set_vexpand(False)
        test_output_scroller.set_min_content_height(120)
        test_panel.append(test_output_scroller)

        TextViewClass = getattr(Gtk, "TextView", None)
        if TextViewClass is not None:
            test_output_view = TextViewClass()
            setter = getattr(test_output_view, "set_editable", None)
            if callable(setter):
                setter(False)
            setter = getattr(test_output_view, "set_cursor_visible", None)
            if callable(setter):
                setter(False)
            wrap_mode = getattr(Gtk, "WrapMode", None)
            if wrap_mode is not None:
                wrap_setter = getattr(test_output_view, "set_wrap_mode", None)
                if callable(wrap_setter):
                    try:
                        wrap_setter(getattr(wrap_mode, "WORD_CHAR", getattr(wrap_mode, "WORD", None)))
                    except Exception:  # pragma: no cover - GTK compatibility variations
                        pass
            test_output_scroller.set_child(test_output_view)
            self._test_output_view = test_output_view
            buffer_getter = getattr(test_output_view, "get_buffer", None)
            if callable(buffer_getter):
                try:
                    self._test_output_buffer = buffer_getter()
                except Exception:  # pragma: no cover - GTK compatibility variations
                    self._test_output_buffer = None
        else:  # pragma: no cover - GTK fallback path
            fallback_label = Gtk.Label()
            fallback_label.set_xalign(0.0)
            fallback_label.set_wrap(True)
            fallback_label.set_hexpand(True)
            test_output_scroller.set_child(fallback_label)
            self._test_output_view = fallback_label
            self._test_output_buffer = None

        notes_label = Gtk.Label(label="Tester notes")
        notes_label.set_xalign(0.0)
        right_panel.append(notes_label)

        notes_scroller = Gtk.ScrolledWindow()
        notes_scroller.set_hexpand(True)
        notes_scroller.set_vexpand(False)
        notes_scroller.set_min_content_height(100)
        right_panel.append(notes_scroller)

        if TextViewClass is not None:
            notes_view = TextViewClass()
            setter = getattr(notes_view, "set_wrap_mode", None)
            wrap_mode = getattr(Gtk, "WrapMode", None)
            if callable(setter) and wrap_mode is not None:
                try:
                    setter(getattr(wrap_mode, "WORD_CHAR", getattr(wrap_mode, "WORD", None)))
                except Exception:  # pragma: no cover - GTK compatibility variations
                    pass
            setter = getattr(notes_view, "set_hexpand", None)
            if callable(setter):
                setter(True)
            setter = getattr(notes_view, "set_vexpand", None)
            if callable(setter):
                setter(True)
            notes_scroller.set_child(notes_view)
            self._tester_notes_view = notes_view
            buffer_getter = getattr(notes_view, "get_buffer", None)
            if callable(buffer_getter):
                try:
                    self._tester_notes_buffer = buffer_getter()
                except Exception:  # pragma: no cover - GTK compatibility variations
                    self._tester_notes_buffer = None
            if self._tester_notes_buffer is not None:
                try:
                    self._tester_notes_buffer.connect("changed", self._on_notes_buffer_changed)
                except Exception:  # pragma: no cover - GTK compatibility variations
                    pass
        else:  # pragma: no cover - GTK fallback path
            notes_label_fallback = Gtk.Label()
            notes_label_fallback.set_xalign(0.0)
            notes_label_fallback.set_wrap(True)
            notes_label_fallback.set_hexpand(True)
            notes_scroller.set_child(notes_label_fallback)
            self._tester_notes_view = notes_label_fallback
            self._tester_notes_buffer = None

        safety_notes_label = Gtk.Label()
        safety_notes_label.set_wrap(True)
        safety_notes_label.set_xalign(0.0)
        right_panel.append(safety_notes_label)
        self._safety_notes_label = safety_notes_label

        source_label = Gtk.Label()
        source_label.set_wrap(True)
        source_label.set_xalign(0.0)
        right_panel.append(source_label)
        self._source_label = source_label

        root.append(left_panel)
        root.append(right_panel)
        return root

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def _refresh_state(self) -> None:
        persona = self._resolve_persona_name()
        self._persona_name = persona
        self._ensure_preferences_loaded(persona)
        self._sync_scope_widget(bool(persona))

        persona_filter = persona if self._skill_scope == "persona" and persona else None
        entries = self._load_skill_entries(persona_filter)

        self._entries = entries
        self._entry_lookup = {entry.name: entry for entry in entries}
        self._requirement_cache.clear()
        self._active_requirement_status = None
        self._persona_error_reported = False
        self._populate_filter_options()
        self._rebuild_skill_list()
        self._history_records = self._load_skill_history(persona_filter)
        self._sync_history_feed()
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

    def _load_skill_entries(self, persona: Optional[str]) -> List[_SkillEntry]:
        server = getattr(self.ATLAS, "server", None)
        getter = getattr(server, "get_skills", None)
        if not callable(getter):
            logger.warning("ATLAS server does not expose get_skills; returning empty workspace")
            return []

        kwargs: Dict[str, Any] = {}
        if persona:
            kwargs["persona"] = persona

        try:
            response = getter(**kwargs)
        except Exception as exc:
            logger.error("Failed to load skill metadata: %s", exc, exc_info=True)
            self._handle_backend_error("Unable to load skill metadata from ATLAS.")
            return []

        skills = response.get("skills") if isinstance(response, Mapping) else None
        if not isinstance(skills, Iterable):
            return []

        entries: List[_SkillEntry] = []
        for raw_entry in skills:
            entry = self._normalize_entry(raw_entry)
            if entry is not None:
                entries.append(entry)
        return entries

    def _normalize_entry(self, entry: Any) -> Optional[_SkillEntry]:
        if not isinstance(entry, Mapping):
            return None

        raw_name = entry.get("name")
        name = str(raw_name).strip() if raw_name else ""
        if not name:
            return None

        summary_source = entry.get("summary") or entry.get("description")
        summary = str(summary_source).strip() if summary_source else "No description available."

        persona = entry.get("persona")
        category = entry.get("category")
        safety_notes = entry.get("safety_notes")
        version = entry.get("version")
        token_normalizer = lambda value: list(
            normalize_sequence(
                value,
                transform=lambda item: str(item).strip(),
                filter_falsy=True,
                accept_scalar=False,
            )
        )
        required_tools = token_normalizer(entry.get("required_tools"))
        required_capabilities = token_normalizer(entry.get("required_capabilities"))
        capability_tags = token_normalizer(entry.get("capability_tags"))
        source = entry.get("source")

        metadata_payload = entry.get("metadata") if isinstance(entry.get("metadata"), Mapping) else None
        review_status_value = entry.get("review_status")
        tester_notes_value = entry.get("tester_notes")
        if isinstance(metadata_payload, Mapping):
            review_status_value = metadata_payload.get("review_status", review_status_value)
            tester_notes_value = metadata_payload.get("tester_notes", tester_notes_value)

        def _normalize_optional_text(value: Any) -> Optional[str]:
            if value is None:
                return None
            text = str(value).strip()
            return text or None

        review_status = _normalize_optional_text(review_status_value)
        tester_notes = _normalize_optional_text(tester_notes_value)

        normalized_metadata = dict(entry)
        normalized_metadata.setdefault("name", name)
        metadata_copy: Optional[Dict[str, Any]]
        if isinstance(metadata_payload, Mapping):
            metadata_copy = dict(metadata_payload)
        else:
            metadata_copy = None
        if review_status is not None or tester_notes is not None:
            if metadata_copy is None:
                metadata_copy = {}
            if review_status is not None:
                metadata_copy["review_status"] = review_status
            if tester_notes is not None:
                metadata_copy["tester_notes"] = tester_notes
        if metadata_copy is not None:
            normalized_metadata["metadata"] = metadata_copy

        return _SkillEntry(
            name=name,
            summary=summary,
            version=str(version).strip() if version else None,
            persona=str(persona).strip() if persona else None,
            category=str(category).strip() if category else None,
            safety_notes=str(safety_notes).strip() if safety_notes else None,
            required_tools=required_tools,
            required_capabilities=required_capabilities,
            capability_tags=capability_tags,
            source=str(source).strip() if source else None,
            raw_metadata=normalized_metadata,
            review_status=review_status,
            tester_notes=tester_notes,
        )

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------
    def _rebuild_skill_list(self) -> None:
        if self._skill_list is None:
            return

        children_getter = getattr(self._skill_list, "get_children", None)
        if callable(children_getter):
            rows = list(children_getter())
        else:
            rows = list(getattr(self._skill_list, "children", []) or [])

        for child in rows:
            try:
                self._skill_list.remove(child)
            except Exception:
                continue

        self._row_lookup.clear()
        self._row_review_labels.clear()

        filtered_entries = self._apply_filters(self._entries)
        self._display_entries = list(filtered_entries)
        self._visible_entry_names = {entry.name for entry in self._display_entries}

        for entry in self._display_entries:
            row = self._create_row(entry)
            self._skill_list.append(row)
            self._row_lookup[entry.name] = row

        if self._display_entries:
            desired = self._active_skill if self._active_skill in self._visible_entry_names else None
            if desired is None:
                desired = self._display_entries[0].name
            self._select_skill(desired)
        else:
            self._show_empty_state()

        self._update_action_state()

    def _populate_filter_options(self) -> None:
        preferences_changed = False

        categories = sorted({entry.category.strip() for entry in self._entries if entry.category})
        personas = sorted({entry.persona.strip() for entry in self._entries if entry.persona})

        if self._category_filter and self._category_filter not in categories:
            self._category_filter = None
            preferences_changed = True

        if self._persona_filter and self._persona_filter not in personas:
            # Preserve the special shared sentinel "" if no longer available
            if self._persona_filter != "":
                self._persona_filter = None
                preferences_changed = True

        self._suppress_filter_signal = True
        try:
            if self._category_combo is not None:
                self._category_option_lookup = {self._CATEGORY_ALL_ID: None}
                self._category_reverse_lookup = {None: self._CATEGORY_ALL_ID}
                clear = getattr(self._category_combo, "remove_all", None)
                if callable(clear):
                    clear()
                self._category_combo.append(self._CATEGORY_ALL_ID, "All categories")
                for index, category in enumerate(categories):
                    option_id = f"value::cat::{index}"
                    self._category_option_lookup[option_id] = category
                    self._category_reverse_lookup[category] = option_id
                    self._category_combo.append(option_id, category)

                desired_id = self._category_reverse_lookup.get(self._category_filter, self._CATEGORY_ALL_ID)
                setter = getattr(self._category_combo, "set_active_id", None)
                if callable(setter):
                    setter(desired_id)
                else:  # pragma: no cover - GTK compatibility fallback
                    fallback = getattr(self._category_combo, "set_active", None)
                    if callable(fallback):
                        options = list(self._category_option_lookup.keys())
                        index = options.index(desired_id) if desired_id in options else 0
                        fallback(index)

            if self._persona_combo is not None:
                self._persona_option_lookup = {
                    self._PERSONA_ALL_ID: None,
                    self._PERSONA_SHARED_ID: "",
                }
                self._persona_reverse_lookup = {
                    None: self._PERSONA_ALL_ID,
                    "": self._PERSONA_SHARED_ID,
                }
                clear = getattr(self._persona_combo, "remove_all", None)
                if callable(clear):
                    clear()
                self._persona_combo.append(self._PERSONA_ALL_ID, "All personas")
                self._persona_combo.append(self._PERSONA_SHARED_ID, "Shared skills")
                for index, persona in enumerate(personas):
                    option_id = f"value::persona::{index}"
                    self._persona_option_lookup[option_id] = persona
                    self._persona_reverse_lookup[persona] = option_id
                    self._persona_combo.append(option_id, persona)

                desired_id = self._persona_reverse_lookup.get(self._persona_filter, self._PERSONA_ALL_ID)
                setter = getattr(self._persona_combo, "set_active_id", None)
                if callable(setter):
                    setter(desired_id)
                else:  # pragma: no cover - GTK compatibility fallback
                    fallback = getattr(self._persona_combo, "set_active", None)
                    if callable(fallback):
                        options = list(self._persona_option_lookup.keys())
                        index = options.index(desired_id) if desired_id in options else 0
                        fallback(index)

            if self._search_entry is not None:
                setter = getattr(self._search_entry, "set_text", None)
                if callable(setter):
                    setter(self._filter_text)

            if self._sort_combo is not None:
                active_id = self._normalize_sort_order()
                setter = getattr(self._sort_combo, "set_active_id", None)
                if callable(setter):
                    setter(active_id)
                else:  # pragma: no cover - GTK compatibility fallback
                    fallback = getattr(self._sort_combo, "set_active", None)
                    if callable(fallback):
                        options = list(self._get_sort_options().keys())
                        index = options.index(active_id) if active_id in options else 0
                        fallback(index)
        finally:
            self._suppress_filter_signal = False

        if preferences_changed:
            self._persist_view_preferences()

    def _apply_filters(self, entries: List[_SkillEntry]) -> List[_SkillEntry]:
        search = self._filter_text.casefold()
        category = self._category_filter
        persona_filter = self._persona_filter

        filtered: List[_SkillEntry] = []
        for entry in entries:
            if search:
                haystack = f"{entry.name}\n{entry.summary}".casefold()
                if search not in haystack:
                    continue

            if category is not None:
                if (entry.category or "") != category:
                    continue

            if persona_filter is not None:
                if persona_filter == "":
                    if entry.persona:
                        continue
                else:
                    if (entry.persona or "") != persona_filter:
                        continue

            filtered.append(entry)

        return self._sort_entries(filtered)

    def _get_sort_options(self) -> Dict[str, str]:
        return {
            "name_asc": "Name (A→Z)",
            "name_desc": "Name (Z→A)",
            "category_asc": "Category (A→Z)",
            "category_desc": "Category (Z→A)",
        }

    def _normalize_sort_order(self) -> str:
        if self._sort_order not in self._get_sort_options():
            self._sort_order = self._DEFAULT_SORT_ORDER
        return self._sort_order

    def _sort_entries(self, entries: List[_SkillEntry]) -> List[_SkillEntry]:
        sort_key = self._normalize_sort_order()

        def name_key(item: _SkillEntry) -> str:
            return item.name.casefold()

        def category_key(item: _SkillEntry) -> tuple[str, str]:
            return (item.category or "").casefold(), item.name.casefold()

        if sort_key == "name_desc":
            return sorted(entries, key=name_key, reverse=True)
        if sort_key == "category_asc":
            return sorted(entries, key=category_key)
        if sort_key == "category_desc":
            return sorted(entries, key=category_key, reverse=True)
        return sorted(entries, key=name_key)

    def _create_row(self, entry: _SkillEntry) -> Gtk.Widget:
        container = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)

        header_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        header_row.set_hexpand(True)

        title = Gtk.Label(label=entry.name)
        title.set_xalign(0.0)
        title.set_hexpand(True)
        header_row.append(title)

        status = self._get_requirement_status(entry)
        indicator = self._build_requirement_indicator(status, entry)
        if indicator is not None:
            header_row.append(indicator)

        container.append(header_row)

        summary = Gtk.Label(label=entry.summary)
        summary.set_xalign(0.0)
        summary.set_wrap(True)
        try:
            summary.add_css_class("dim-label")
        except Exception:  # pragma: no cover - GTK theme variations
            pass
        container.append(summary)

        review_label = Gtk.Label(label=self._format_review_status_text(entry))
        review_label.set_xalign(0.0)
        review_label.set_wrap(True)
        try:
            review_label.add_css_class("dim-label")
        except Exception:  # pragma: no cover - GTK theme variations
            pass
        container.append(review_label)
        self._row_review_labels[entry.name] = review_label

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

    def _format_review_status_text(self, entry: _SkillEntry, *, include_prefix: bool = False) -> str:
        """Return a human readable review status string for ``entry``."""

        status_text = self._normalize_review_status(entry.review_status)
        if include_prefix:
            return f"Review status: {status_text}"
        return status_text

    def _normalize_review_status(self, status: Optional[str]) -> str:
        """Normalize ``status`` into a display friendly string."""

        if not status:
            return "Needs review"
        text = str(status).strip()
        if not text:
            return "Needs review"
        normalized = text.replace("_", " ").strip()
        lowered = normalized.casefold()
        if lowered in {"reviewed", "complete", "approved"}:
            return "Reviewed"
        if lowered in {"needs review", "needs-review", "pending", "unreviewed"}:
            return "Needs review"
        return normalized[:1].upper() + normalized[1:] if normalized else "Needs review"

    def _update_row_review_status(self, skill_name: str) -> None:
        """Refresh the review label for the skill list row matching ``skill_name``."""

        label = self._row_review_labels.get(skill_name)
        entry = self._entry_lookup.get(skill_name)
        if label is None or entry is None:
            return
        self._set_label(label, self._format_review_status_text(entry))

    def _select_skill(self, skill_name: str) -> None:
        entry = self._entry_lookup.get(skill_name)
        if entry is None:
            return

        row = self._row_lookup.get(skill_name)
        if row is None:
            if self._display_entries:
                fallback = self._display_entries[0].name
                if fallback != skill_name:
                    self._select_skill(fallback)
                else:
                    self._show_empty_state()
            else:
                self._show_empty_state()
            return

        self._active_skill = skill_name
        self._preview_in_progress = False
        self._set_test_running(False)

        status = self._get_requirement_status(entry, refresh=True)
        self._active_requirement_status = status

        persona_scope = entry.persona or "Shared"
        version = entry.version or "Unspecified"
        category = entry.category or "Uncategorized"

        self._set_label(self._title_label, entry.name)
        self._set_label(self._summary_label, entry.summary)
        self._set_label(self._persona_label, f"Persona scope: {persona_scope}")
        self._set_label(self._version_label, f"Version: {version}")
        self._set_label(self._category_label, f"Category: {category}")
        self._set_label(
            self._required_tools_label,
            "Required tools: " + self._format_list(entry.required_tools),
        )
        self._sync_requirement_badges(self._tool_badge_box, entry.required_tools, status, unknown_label="No tools required")
        self._set_label(
            self._required_capabilities_label,
            "Required capabilities: " + self._format_list(entry.required_capabilities),
        )
        self._sync_requirement_badges(
            self._capability_badge_box,
            entry.required_capabilities,
            status,
            kind="capability",
            unknown_label="No capabilities required",
        )
        self._set_label(
            self._capability_tags_label,
            "Capability tags: " + self._format_list(entry.capability_tags),
        )
        safety = entry.safety_notes or "No safety notes provided."
        self._set_label(self._safety_notes_label, f"Safety notes: {safety}")
        source = entry.source or "Source metadata unavailable"
        self._set_label(self._source_label, f"Source: {source}")

        self._set_label(
            self._review_status_label,
            self._format_review_status_text(entry, include_prefix=True),
        )
        self._baseline_status = (entry.review_status or "").strip()
        self._baseline_notes = entry.tester_notes or ""
        self._recent_changes_snapshot = None
        self._set_notes_text(self._baseline_notes)
        self._set_notes_editable(self._supports_review_persistence())
        self._initialize_test_panel(entry)
        self._update_row_review_status(entry.name)

        message = self._describe_requirement_status(status, entry)
        if status.evaluation_error and status.show_dialog and not self._persona_error_reported:
            self._handle_backend_error(status.evaluation_error)
            self._persona_error_reported = True
        elif not status.evaluation_error:
            self._persona_error_reported = False
        self._set_label(self._requirement_status_label, message)
        self._sync_action_buttons(status)

        if self._skill_list is not None:
            try:
                self._skill_list.select_row(row)
            except Exception:  # pragma: no cover - GTK stub variations
                pass

        self._update_action_state()

    def _show_empty_state(self) -> None:
        self._active_skill = None
        self._active_requirement_status = None
        self._baseline_notes = ""
        self._baseline_status = ""
        self._set_label(self._title_label, "No skill selected")
        self._set_label(
            self._summary_label,
            "Select a skill to review configuration and credential requirements.",
        )
        self._set_label(self._persona_label, "Persona scope: (unavailable)")
        self._set_label(self._version_label, "Version: (n/a)")
        self._set_label(self._category_label, "Category: (n/a)")
        self._set_label(self._required_tools_label, "Required tools: (n/a)")
        self._set_label(self._required_capabilities_label, "Required capabilities: (n/a)")
        self._set_label(self._capability_tags_label, "Capability tags: (n/a)")
        self._set_label(self._safety_notes_label, "Safety notes: (n/a)")
        self._set_label(self._source_label, "Source: (n/a)")
        self._set_label(self._requirement_status_label, "Requirement status: (n/a)")
        self._clear_container(self._tool_badge_box)
        if self._tool_badge_box is not None:
            self._tool_badge_box.set_visible(False)
        self._clear_container(self._capability_badge_box)
        if self._capability_badge_box is not None:
            self._capability_badge_box.set_visible(False)
        if self._action_button_box is not None:
            self._action_button_box.set_visible(False)
        if self._enable_tools_button is not None:
            self._enable_tools_button.set_visible(False)
        if self._open_tool_manager_button is not None:
            self._open_tool_manager_button.set_visible(False)
        self._set_label(self._review_status_label, "Review status: (n/a)")
        self._set_notes_text("")
        self._set_notes_editable(False)
        self._preview_in_progress = False
        self._set_test_running(False)
        self._initialize_test_panel(None)
        self._update_action_state()

    def _get_requirement_status(self, entry: _SkillEntry, *, refresh: bool = False) -> _RequirementStatus:
        if not refresh:
            cached = self._requirement_cache.get(entry.name)
            if cached is not None:
                return cached

        status = self._evaluate_requirements(entry)
        self._requirement_cache[entry.name] = status
        return status

    def _evaluate_requirements(self, entry: _SkillEntry) -> _RequirementStatus:
        persona_name = self._persona_name
        context = self._get_persona_context()

        if context is None:
            message = "No active persona selected; requirement status is unknown."
            show_dialog = False
            if persona_name:
                message = "Unable to determine requirement status for the active persona."
                show_dialog = True
            return _RequirementStatus(
                persona_name=persona_name,
                missing_tools=[],
                missing_capabilities=[],
                satisfied_tools=[],
                satisfied_capabilities=[],
                evaluation_error=message,
                persona_mismatch=False,
                show_dialog=show_dialog,
            )

        resolved_persona = context.get("persona_name") or persona_name
        allowed_tools = set(context.get("allowed_tools", []))
        allowed_capabilities = set(context.get("capability_tags", []))

        missing_tools: List[str] = []
        satisfied_tools: List[str] = []
        for tool in entry.required_tools:
            if tool in allowed_tools:
                satisfied_tools.append(tool)
            else:
                missing_tools.append(tool)

        missing_capabilities: List[str] = []
        satisfied_capabilities: List[str] = []
        for capability in entry.required_capabilities:
            if capability in allowed_capabilities:
                satisfied_capabilities.append(capability)
            else:
                missing_capabilities.append(capability)

        persona_mismatch = False
        if entry.persona and resolved_persona:
            persona_mismatch = entry.persona.strip().casefold() != resolved_persona.strip().casefold()
        elif entry.persona and not resolved_persona:
            persona_mismatch = True

        return _RequirementStatus(
            persona_name=resolved_persona,
            missing_tools=missing_tools,
            missing_capabilities=missing_capabilities,
            satisfied_tools=satisfied_tools,
            satisfied_capabilities=satisfied_capabilities,
            evaluation_error=None,
            persona_mismatch=persona_mismatch,
            show_dialog=False,
        )

    def _describe_requirement_status(self, status: _RequirementStatus, entry: _SkillEntry) -> str:
        if status.evaluation_error:
            return status.evaluation_error

        messages: List[str] = []
        if status.persona_mismatch and entry.persona:
            if status.persona_name:
                messages.append(
                    f"Skill is scoped to persona '{entry.persona}', but active persona is '{status.persona_name}'."
                )
            else:
                messages.append(f"Skill is scoped to persona '{entry.persona}'.")

        if status.missing_tools:
            messages.append("Missing tools: " + ", ".join(status.missing_tools))
        if status.missing_capabilities:
            messages.append("Missing capabilities: " + ", ".join(status.missing_capabilities))

        if not messages:
            return "All requirements satisfied for the active persona."
        return " \u2013 ".join(messages)

    def _build_requirement_indicator(
        self, status: _RequirementStatus, entry: _SkillEntry
    ) -> Optional[Gtk.Widget]:
        needs_attention = bool(
            status.evaluation_error or status.missing_tools or status.missing_capabilities or status.persona_mismatch
        )
        if not needs_attention:
            return None

        indicator = Gtk.Label(label="⚠")
        try:
            indicator.set_hexpand(False)
        except Exception:  # pragma: no cover - GTK compatibility fallbacks
            pass
        try:
            indicator.set_xalign(1.0)
        except Exception:  # pragma: no cover - GTK compatibility fallbacks
            pass
        align_cls = getattr(Gtk, "Align", None)
        if align_cls is not None:
            align_value = getattr(align_cls, "END", getattr(align_cls, "FILL", None))
            if align_value is not None:
                try:
                    indicator.set_halign(align_value)
                except Exception:  # pragma: no cover - GTK compatibility fallbacks
                    pass
        indicator.set_tooltip_text(self._describe_requirement_status(status, entry))
        return indicator

    def _create_badge_widget(self, text: str, css_classes: List[str]) -> Gtk.Widget:
        badge = Gtk.Label(label=text)
        try:
            badge.set_xalign(0.5)
        except Exception:  # pragma: no cover - GTK compatibility fallbacks
            pass
        for css in css_classes:
            try:
                badge.add_css_class(css)
            except Exception:  # pragma: no cover - GTK compatibility fallbacks
                continue
        return badge

    def _sync_requirement_badges(
        self,
        container: Optional[Gtk.Widget],
        required_items: Iterable[str],
        status: _RequirementStatus,
        *,
        kind: str = "tool",
        unknown_label: str = "None",
    ) -> None:
        if container is None:
            return

        self._clear_container(container)

        items = [item for item in required_items if item]
        if status.evaluation_error:
            widget = self._create_badge_widget("Status unavailable", ["tag-badge", "status-unknown"])
            self._append_badge(container, widget)
            container.set_visible(True)
            return

        if not items:
            widget = self._create_badge_widget(unknown_label, ["tag-badge", "status-ok"])
            self._append_badge(container, widget)
            container.set_visible(True)
            return

        missing_lookup = set(status.missing_tools if kind == "tool" else status.missing_capabilities)
        for item in items:
            css = ["tag-badge"]
            if item in missing_lookup:
                css.append("status-error")
            else:
                css.append("status-ok")
            widget = self._create_badge_widget(item, css)
            self._append_badge(container, widget)

        container.set_visible(True)

    def _append_badge(self, container: Gtk.Widget, widget: Gtk.Widget) -> None:
        inserter = getattr(container, "insert", None)
        if callable(inserter):
            try:
                inserter(widget, -1)
                return
            except Exception:  # pragma: no cover - GTK compatibility fallbacks
                pass

        appender = getattr(container, "append", None)
        if callable(appender):
            try:
                appender(widget)
            except Exception:  # pragma: no cover - GTK compatibility fallbacks
                pass

    def _sync_action_buttons(self, status: _RequirementStatus) -> None:
        enable_button = self._enable_tools_button
        open_button = self._open_tool_manager_button
        button_box = self._action_button_box

        enable_visible = bool(status.missing_tools) and not status.evaluation_error and bool(self._persona_name)
        if enable_button is not None:
            enable_button.set_visible(enable_visible)
            setter = getattr(enable_button, "set_sensitive", None)
            if callable(setter):
                setter(enable_visible)

        open_visible = bool(
            status.missing_tools or status.missing_capabilities or status.evaluation_error or status.persona_mismatch
        )
        if open_button is not None:
            open_button.set_visible(open_visible)
            setter = getattr(open_button, "set_sensitive", None)
            if callable(setter):
                setter(True)

        persona_callback_available = callable(self.on_open_in_persona) and bool(self._persona_name)
        persona_visible = persona_callback_available and bool(self._active_skill)
        persona_button = self._open_persona_button
        if persona_button is not None:
            persona_button.set_visible(persona_visible)
            setter = getattr(persona_button, "set_sensitive", None)
            if callable(setter):
                setter(persona_visible)

        if button_box is not None:
            button_box.set_visible(enable_visible or open_visible or persona_visible)

    def _get_persona_context(self) -> Optional[Dict[str, Any]]:
        persona_name = self._persona_name
        if not persona_name:
            return None

        manager = getattr(self.ATLAS, "persona_manager", None)
        if manager is None:
            return None

        getter = getattr(manager, "get_current_persona_context", None)
        if callable(getter):
            try:
                context = getter()
            except Exception as exc:  # pragma: no cover - backend failure logging
                logger.error("Failed to load active persona context: %s", exc, exc_info=True)
                return None
            if isinstance(context, Mapping):
                return {
                    "persona_name": context.get("persona_name") or persona_name,
                    "allowed_tools": self._normalize_strings(context.get("allowed_tools")),
                    "capability_tags": self._normalize_strings(context.get("capability_tags")),
                }

        persona_getter = getattr(manager, "get_persona", None)
        if callable(persona_getter):
            try:
                persona_payload = persona_getter(persona_name)
            except Exception as exc:  # pragma: no cover - backend failure logging
                logger.error("Failed to load persona '%s': %s", persona_name, exc, exc_info=True)
                return None
            if isinstance(persona_payload, Mapping):
                return {
                    "persona_name": persona_payload.get("name") or persona_name,
                    "allowed_tools": self._normalize_strings(persona_payload.get("allowed_tools")),
                    "capability_tags": self._normalize_strings(persona_payload.get("capability_tags")),
                }

        return None

    def _normalize_strings(self, values: Any) -> List[str]:
        normalized: List[str] = []
        seen: Set[str] = set()

        if isinstance(values, str):
            candidates = [values]
        elif isinstance(values, Iterable) and not isinstance(values, (str, bytes)):
            candidates = list(values)
        else:
            candidates = []

        for item in candidates:
            if isinstance(item, Mapping):
                candidate_value = item.get("name")
            else:
                candidate_value = item

            if candidate_value is None:
                continue

            text = str(candidate_value).strip()
            if text and text not in seen:
                normalized.append(text)
                seen.add(text)

        return normalized

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------
    def _on_row_selected(self, _listbox: Gtk.ListBox, row: Gtk.Widget) -> None:
        for name, candidate in self._row_lookup.items():
            if candidate is row:
                self._select_skill(name)
                break

    def _on_search_changed(self, entry: Gtk.SearchEntry) -> None:
        if self._suppress_filter_signal:
            return

        getter = getattr(entry, "get_text", None)
        text = getter() if callable(getter) else ""
        normalized = text.strip()

        if normalized == self._filter_text:
            return

        self._filter_text = normalized
        self._persist_view_preferences()
        self._rebuild_skill_list()

    def _on_category_filter_changed(self, combo: Gtk.ComboBoxText) -> None:
        if self._suppress_filter_signal:
            return

        getter = getattr(combo, "get_active_id", None)
        option_id = getter() if callable(getter) else None
        new_value = self._category_option_lookup.get(option_id or "", None)

        if new_value == self._category_filter:
            return

        self._category_filter = new_value
        self._persist_view_preferences()
        self._rebuild_skill_list()

    def _on_persona_filter_changed(self, combo: Gtk.ComboBoxText) -> None:
        if self._suppress_filter_signal:
            return

        getter = getattr(combo, "get_active_id", None)
        option_id = getter() if callable(getter) else None
        new_value = self._persona_option_lookup.get(option_id or "", None)

        if new_value == self._persona_filter:
            return

        self._persona_filter = new_value
        self._persist_view_preferences()
        self._rebuild_skill_list()

    def _on_sort_order_changed(self, combo: Gtk.ComboBoxText) -> None:
        if self._suppress_filter_signal:
            return

        getter = getattr(combo, "get_active_id", None)
        option_id = getter() if callable(getter) else None
        if not option_id:
            return

        if option_id not in self._get_sort_options():
            option_id = self._DEFAULT_SORT_ORDER

        if option_id == self._sort_order:
            return

        self._sort_order = option_id
        self._persist_view_preferences()
        self._rebuild_skill_list()

    def _on_scope_changed(self, combo: Gtk.ComboBoxText) -> None:
        if self._suppress_scope_signal:
            return

        getter = getattr(combo, "get_active_text", None)
        label = getter() if callable(getter) else None
        new_scope = "persona" if label == "Persona skills" else "all"

        if new_scope == "persona" and not self._persona_name:
            self._skill_scope = "all"
            self._sync_scope_widget(False)
            return

        if new_scope == self._skill_scope:
            return

        self._skill_scope = new_scope
        self._refresh_state()

    def _on_notes_buffer_changed(self, _buffer: Gtk.TextBuffer) -> None:
        if self._suppress_notes_signal:
            return
        self._update_action_state()

    def _on_enable_missing_tools_clicked(self, _button: Gtk.Button) -> None:
        status = self._active_requirement_status
        persona_name = self._persona_name
        if status is None or not status.missing_tools or not persona_name:
            return

        tools_to_enable = list(status.missing_tools)
        tool_manager = getattr(self.parent_window, "tool_management", None)
        enabler = getattr(tool_manager, "enable_tools_for_persona", None) if tool_manager else None
        if not callable(enabler):
            try:
                from GTKUI.Tool_manager.tool_management import ToolManagement  # pylint: disable=import-outside-toplevel
            except Exception:  # pragma: no cover - import fallback
                enabler = None
            else:
                enabler = getattr(ToolManagement(self.ATLAS, self.parent_window), "enable_tools_for_persona", None)

        if callable(enabler):
            try:
                success = bool(enabler(persona_name, tools_to_enable))
            except Exception as exc:  # pragma: no cover - defensive logging only
                logger.error("Failed to enable tools via ToolManagement: %s", exc, exc_info=True)
                self._handle_backend_error("Unable to enable tools automatically.")
                return
            if success:
                self._refresh_state()
                if self._active_skill:
                    self._select_skill(self._active_skill)
            return

        persona_manager = getattr(self.ATLAS, "persona_manager", None)
        setter = getattr(persona_manager, "set_allowed_tools", None)
        if not callable(setter):
            self._handle_backend_error("Tool management is unavailable. Unable to enable tools automatically.")
            return

        existing_tools: List[str] = []
        getter = getattr(persona_manager, "get_persona", None)
        if callable(getter):
            try:
                payload = getter(persona_name)
            except Exception as exc:  # pragma: no cover - backend failure logging
                logger.error("Failed to load persona '%s' for enabling tools: %s", persona_name, exc, exc_info=True)
                self._handle_backend_error("Unable to load persona configuration from ATLAS.")
                return
            if isinstance(payload, Mapping):
                existing_tools = self._normalize_strings(payload.get("allowed_tools"))

        desired = sorted({*existing_tools, *tools_to_enable})

        try:
            result = setter(persona_name, desired)
        except Exception as exc:  # pragma: no cover - backend failure logging
            logger.error("Failed to persist enabled tools for persona '%s': %s", persona_name, exc, exc_info=True)
            self._handle_backend_error(str(exc) or "Unable to enable required tools.")
            return

        if isinstance(result, Mapping) and not result.get("success", True):
            errors = result.get("errors") or []
            message = "; ".join(str(err) for err in errors if err) or str(result.get("error") or "Unable to enable required tools.")
            self._handle_backend_error(message)
            return

        self._refresh_state()
        if self._active_skill:
            self._select_skill(self._active_skill)

    def _on_preview_prompt_clicked(self, _button: Gtk.Button) -> None:
        if self._preview_in_progress:
            return

        entry = self._get_active_entry()
        if entry is None:
            return

        method = self._resolve_backend_method("get_skill_details")
        if method is None:
            self._handle_backend_error("Skill prompt preview is not supported by this backend.")
            return

        self._preview_in_progress = True
        self._sync_preview_button_state()

        worker = threading.Thread(target=self._load_prompt_preview, args=(entry, method), daemon=True)
        worker.start()

    def _on_run_test_invocation_clicked(self, _button: Gtk.Button) -> None:
        if self._test_in_progress:
            return

        entry = self._get_active_entry()
        if entry is None:
            return

        method = self._resolve_backend_method(
            "validate_skill",
            "test_skill",
            "run_skill_test",
            "run_skill_validation",
            "invoke_skill_test",
        )
        if method is None:
            self._set_test_results(None, "Test invocations are not supported by this backend.", "")
            self._sync_test_button_state()
            return

        if self._test_status_label is not None:
            self._set_label(self._test_status_label, "Running test invocation…")
        self._set_test_output_text("")
        self._set_test_running(True)

        worker = threading.Thread(target=self._execute_test_runner, args=(entry, method), daemon=True)
        worker.start()

    def _on_mark_reviewed_clicked(self, _button: Gtk.Button) -> None:
        entry = self._get_active_entry()
        if entry is None:
            return

        if not self._supports_review_persistence():
            self._handle_backend_error("Skill metadata updates are not supported by this backend.")
            return

        notes_text = self._get_notes_text()
        status_value = "reviewed"

        if not self._persist_review_metadata(entry, status_value, notes_text):
            return

        self._set_label(
            self._review_status_label,
            self._format_review_status_text(entry, include_prefix=True),
        )
        self._update_row_review_status(entry.name)
        self._baseline_status = "reviewed"
        self._baseline_notes = entry.tester_notes or ""
        self._recent_changes_snapshot = None
        self._update_action_state()

    def _on_open_tool_manager_clicked(self, _button: Gtk.Button) -> None:
        opener = getattr(self.parent_window, "show_tools_menu", None)
        if callable(opener):
            try:
                opener()
            except Exception as exc:  # pragma: no cover - defensive logging only
                logger.error("Failed to open tool management workspace: %s", exc, exc_info=True)
                self._handle_backend_error("Unable to open tool management workspace.")
        else:
            self._handle_backend_error("Tool manager workspace is unavailable.")

    def _on_open_in_persona_clicked(self, _button: Gtk.Button) -> None:
        callback = self.on_open_in_persona
        skill_name = self._active_skill
        if not callable(callback) or not skill_name:
            return

        try:
            callback(skill_name)
        except Exception as exc:  # pragma: no cover - defensive logging only
            logger.error("Failed to open persona configuration for skill '%s': %s", skill_name, exc, exc_info=True)
            self._handle_backend_error("Unable to open persona editor for this skill.")

    def _sync_scope_widget(self, persona_available: bool) -> None:
        widget = self._scope_selector
        if widget is None:
            return

        desired_scope = self._skill_scope
        if desired_scope == "persona" and not persona_available:
            desired_scope = "all"
            self._skill_scope = "all"

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

    def _ensure_preferences_loaded(self, persona: Optional[str]) -> None:
        key = self._get_preferences_key(persona)
        if key == self._preferences_loaded_key:
            return
        self._load_view_preferences(persona)

    def _load_view_preferences(self, persona: Optional[str]) -> None:
        storage = self._get_settings_bucket()
        key = self._get_preferences_key(persona)
        record = storage.get(key)
        if not isinstance(record, Mapping):
            record = {}

        needs_persist = False

        text_value = record.get("filter_text")
        self._filter_text = str(text_value).strip() if isinstance(text_value, str) else ""

        category_value = record.get("category")
        self._category_filter = str(category_value) if isinstance(category_value, str) and category_value else None
        if self._category_filter is not None and not isinstance(category_value, str):
            needs_persist = True

        persona_value = record.get("persona")
        if persona_value == "":
            self._persona_filter = ""
        elif isinstance(persona_value, str) and persona_value:
            self._persona_filter = persona_value
        else:
            self._persona_filter = None
        if persona_value not in (None, "") and not isinstance(persona_value, str):
            needs_persist = True

        sort_value = record.get("sort_order")
        if isinstance(sort_value, str):
            self._sort_order = sort_value
        else:
            self._sort_order = self._DEFAULT_SORT_ORDER
            needs_persist = True
        if self._normalize_sort_order() != sort_value:
            needs_persist = True

        self._preferences_loaded_key = key

        if needs_persist:
            self._persist_view_preferences()

    def _persist_view_preferences(self) -> None:
        storage = self._get_settings_bucket()
        key = self._get_preferences_key(self._persona_name)

        payload = {
            "filter_text": self._filter_text,
            "category": self._category_filter,
            "persona": self._persona_filter,
            "sort_order": self._normalize_sort_order(),
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

        bucket = settings.get("skill_management")
        if not isinstance(bucket, dict):
            bucket = {}
            settings["skill_management"] = bucket

        return bucket

    # ------------------------------------------------------------------
    # Review and test helpers
    # ------------------------------------------------------------------
    def _get_active_entry(self) -> Optional[_SkillEntry]:
        if not self._active_skill:
            return None
        return self._entry_lookup.get(self._active_skill)

    def _load_prompt_preview(self, entry: _SkillEntry, method: Any) -> None:
        persona_scope = entry.persona or self._persona_name
        try:
            response = self._call_backend_with_variants(method, entry.name, persona_scope, {})
        except TypeError:
            GLib.idle_add(
                self._handle_prompt_response,
                entry.name,
                None,
                "Skill prompt preview is not supported by this backend.",
            )
        except Exception as exc:  # pragma: no cover - defensive logging only
            logger.error("Failed to load skill details for '%s': %s", entry.name, exc, exc_info=True)
            GLib.idle_add(
                self._handle_prompt_response,
                entry.name,
                None,
                "Unable to load skill details from ATLAS.",
            )
        else:
            GLib.idle_add(self._handle_prompt_response, entry.name, response, None)

    def _handle_prompt_response(
        self,
        skill_name: str,
        payload: Any,
        error: Optional[str],
    ) -> bool:
        self._preview_in_progress = False
        self._sync_preview_button_state()

        if error:
            self._handle_backend_error(error)
            return False

        if self._active_skill != skill_name:
            return False

        entry = self._entry_lookup.get(skill_name)
        prompt_text = self._extract_prompt_text(payload, entry)
        self._show_prompt_dialog(skill_name, prompt_text)
        return False

    def _extract_prompt_text(self, payload: Any, entry: Optional[_SkillEntry]) -> str:
        prompt: Optional[str] = None
        if isinstance(payload, Mapping):
            prompt = payload.get("instruction_prompt")
            if prompt is None:
                nested = payload.get("skill")
                if isinstance(nested, Mapping):
                    prompt = nested.get("instruction_prompt")
            if prompt is None:
                prompt = payload.get("prompt")
        if prompt is None and entry is not None:
            prompt = entry.raw_metadata.get("instruction_prompt")

        prompt_text = str(prompt).strip() if prompt else ""
        if not prompt_text:
            return "No instruction prompt is available for this skill."
        return prompt_text

    def _show_prompt_dialog(self, skill_name: str, prompt: str) -> None:
        dialog = Gtk.MessageDialog(
            transient_for=self.parent_window,
            modal=True,
            message_type=Gtk.MessageType.INFO,
            buttons=Gtk.ButtonsType.CLOSE,
            text=f"{skill_name} prompt",
        )
        try:
            dialog.set_default_size(640, 480)
        except Exception:  # pragma: no cover - GTK compatibility
            pass

        scroller = Gtk.ScrolledWindow()
        scroller.set_min_content_height(320)
        scroller.set_min_content_width(520)

        TextViewClass = getattr(Gtk, "TextView", None)
        if TextViewClass is not None:
            text_view = TextViewClass()
            setter = getattr(text_view, "set_editable", None)
            if callable(setter):
                setter(False)
            setter = getattr(text_view, "set_cursor_visible", None)
            if callable(setter):
                setter(False)
            wrap_mode = getattr(Gtk, "WrapMode", None)
            if wrap_mode is not None:
                wrap_setter = getattr(text_view, "set_wrap_mode", None)
                if callable(wrap_setter):
                    try:
                        wrap_setter(getattr(wrap_mode, "WORD_CHAR", getattr(wrap_mode, "WORD", None)))
                    except Exception:  # pragma: no cover - GTK compatibility variations
                        pass
            buffer_getter = getattr(text_view, "get_buffer", None)
            if callable(buffer_getter):
                try:
                    buffer = buffer_getter()
                except Exception:  # pragma: no cover - GTK compatibility variations
                    buffer = None
                else:
                    self._set_buffer_text(buffer, prompt)
            scroller.set_child(text_view)
        else:  # pragma: no cover - GTK fallback path
            label = Gtk.Label(label=prompt)
            label.set_wrap(True)
            label.set_xalign(0.0)
            scroller.set_child(label)

        setter = getattr(dialog, "set_extra_child", None)
        if callable(setter):
            try:
                setter(scroller)
            except Exception:  # pragma: no cover - GTK compatibility variations
                pass
        else:  # pragma: no cover - GTK fallback path
            secondary = getattr(dialog, "set_secondary_text", None)
            if callable(secondary):
                secondary(prompt)
            else:
                try:
                    dialog.props.secondary_text = prompt
                except Exception:
                    pass

        dialog.connect("response", lambda d, *_: d.destroy())
        dialog.present()

    def _execute_test_runner(self, entry: _SkillEntry, method: Any) -> None:
        persona_scope = entry.persona or self._persona_name
        try:
            response = self._call_backend_with_variants(method, entry.name, persona_scope, {})
        except TypeError:
            GLib.idle_add(self._on_test_unsupported, entry.name)
        except Exception as exc:  # pragma: no cover - defensive logging only
            logger.error("Test invocation failed for '%s': %s", entry.name, exc, exc_info=True)
            GLib.idle_add(self._on_test_failed, entry.name, str(exc) or "Skill validation failed.")
        else:
            GLib.idle_add(self._on_test_completed, entry.name, response)

    def _on_test_completed(self, skill_name: str, response: Any) -> bool:
        self._set_test_running(False)
        if self._active_skill != skill_name:
            return False
        success, message, details = self._interpret_test_response(response)
        self._set_test_results(success, message, details)
        return False

    def _on_test_failed(self, skill_name: str, error_message: str) -> bool:
        self._set_test_running(False)
        if self._active_skill != skill_name:
            return False
        message = f"Test invocation failed: {error_message}" if error_message else "Test invocation failed."
        self._set_test_results(False, message, error_message)
        return False

    def _on_test_unsupported(self, skill_name: str) -> bool:
        self._set_test_running(False)
        if self._active_skill == skill_name:
            self._set_test_results(None, "Test invocations are not supported by this backend.", "")
            self._sync_test_button_state()
        return False

    def _persist_review_metadata(self, entry: _SkillEntry, status: str, notes: str) -> bool:
        method = self._resolve_backend_method("set_skill_metadata")
        if method is None:
            self._handle_backend_error("Skill metadata updates are not supported by this backend.")
            return False

        metadata_payload = {
            "review_status": status,
            "tester_notes": notes,
        }
        persona_scope = entry.persona or self._persona_name

        try:
            response = self._call_backend_with_variants(
                method,
                entry.name,
                persona_scope,
                {"metadata": metadata_payload},
            )
        except TypeError:
            self._handle_backend_error("Skill metadata updates are not supported by this backend.")
            return False
        except Exception as exc:  # pragma: no cover - defensive logging only
            logger.error("Failed to persist review metadata for '%s': %s", entry.name, exc, exc_info=True)
            self._handle_backend_error("Unable to save review status for this skill.")
            return False

        success = True
        message: Optional[str] = None
        if isinstance(response, Mapping):
            if "success" in response:
                success = bool(response.get("success"))
            if not success:
                message = response.get("message") or response.get("error")
        elif isinstance(response, bool):
            success = response

        if not success:
            self._handle_backend_error(str(message) if message else "Backend rejected the review update.")
            return False

        normalized_status = status.strip()
        entry.review_status = normalized_status
        cleaned_notes = notes.strip()
        entry.tester_notes = cleaned_notes or None

        metadata_record = entry.raw_metadata.get("metadata")
        if not isinstance(metadata_record, dict):
            metadata_record = {}
            entry.raw_metadata["metadata"] = metadata_record
        metadata_record["review_status"] = normalized_status
        metadata_record["tester_notes"] = cleaned_notes
        return True

    def _initialize_test_panel(self, entry: Optional[_SkillEntry]) -> None:
        if entry is None:
            message = "Select a skill to run a test invocation."
        else:
            if not self._supports_test_invocation():
                message = "Test invocations are not supported by this backend."
            else:
                message = "No test invocation has been run for this skill."
        self._set_test_results(None, message, "")

    def _set_test_results(self, status: Optional[bool], message: str, details: Optional[str]) -> None:
        label = self._test_status_label
        if label is not None:
            prefix = ""
            if status is True:
                prefix = "Success: "
            elif status is False:
                prefix = "Failure: "
            self._set_label(label, f"{prefix}{message}" if prefix else message)

        if details is not None:
            self._set_test_output_text(details)

        self._set_test_running(False)

    def _set_test_running(self, running: bool) -> None:
        self._test_in_progress = bool(running)
        spinner = self._test_spinner
        if spinner is not None:
            setter = getattr(spinner, "set_visible", None)
            if callable(setter):
                setter(bool(running))
            toggle = getattr(spinner, "start", None) if running else getattr(spinner, "stop", None)
            if callable(toggle):
                try:
                    toggle()
                except Exception:  # pragma: no cover - GTK compatibility variations
                    pass
        self._sync_test_button_state()

    def _set_test_output_text(self, text: str) -> None:
        if self._test_output_buffer is not None:
            self._set_buffer_text(self._test_output_buffer, text)
        elif isinstance(self._test_output_view, Gtk.Label):
            self._set_label(self._test_output_view, text)

    def _set_notes_text(self, text: str) -> None:
        self._suppress_notes_signal = True
        try:
            if self._tester_notes_buffer is not None:
                self._set_buffer_text(self._tester_notes_buffer, text)
            elif isinstance(self._tester_notes_view, Gtk.Label):
                self._set_label(self._tester_notes_view, text)
        finally:
            self._suppress_notes_signal = False
        self._update_action_state()

    def _get_notes_text(self) -> str:
        if self._tester_notes_buffer is not None:
            return self._get_buffer_text(self._tester_notes_buffer).strip()
        view = self._tester_notes_view
        if isinstance(view, Gtk.Label):
            getter = getattr(view, "get_label", None) or getattr(view, "get_text", None)
            if callable(getter):
                try:
                    return str(getter()).strip()
                except Exception:  # pragma: no cover - GTK compatibility variations
                    return ""
        return ""

    def _set_notes_editable(self, enabled: bool) -> None:
        view = self._tester_notes_view
        if view is None:
            return

        setter = getattr(view, "set_editable", None)
        if callable(setter):
            try:
                setter(bool(enabled))
            except Exception:  # pragma: no cover - GTK compatibility variations
                pass
        cursor = getattr(view, "set_cursor_visible", None)
        if callable(cursor):
            try:
                cursor(bool(enabled))
            except Exception:  # pragma: no cover - GTK compatibility variations
                pass
        sensitive = getattr(view, "set_sensitive", None)
        if callable(sensitive):
            sensitive(bool(enabled))

    def _update_action_state(self) -> None:
        has_entries = bool(self._entries)
        has_visible = bool(self._display_entries)
        self._set_button_sensitive(self._export_csv_button, has_entries and has_visible)
        self._set_button_sensitive(self._export_json_button, has_entries and has_visible)
        self._set_button_sensitive(self._export_bundle_button, bool(self._active_skill))
        self._set_button_sensitive(self._import_bundle_button, True)
        self._sync_recent_changes_panel()
        self._sync_preview_button_state()
        self._sync_test_button_state()
        self._sync_review_button_state()

    def _sync_recent_changes_panel(self) -> None:
        box = self._recent_changes_box
        label = self._recent_changes_label
        if box is None or label is None:
            return

        entry = self._entry_lookup.get(self._active_skill) if self._active_skill else None
        if entry is None:
            setter = getattr(box, "set_visible", None)
            if callable(setter):
                setter(False)
            if self._recent_changes_snapshot not in (None, tuple()):
                logger.debug("Pending skill review changes cleared.")
            self._recent_changes_snapshot = None
            return

        changes = self._compute_pending_skill_changes(entry)
        snapshot = tuple(changes)

        if not changes:
            setter = getattr(box, "set_visible", None)
            if callable(setter):
                setter(False)
            if self._recent_changes_snapshot not in (None, tuple()):
                logger.debug("Pending skill review changes for '%s' cleared.", entry.name)
            self._recent_changes_snapshot = tuple()
            return

        label.set_label("\n".join(changes))
        label.set_wrap(True)
        setter = getattr(box, "set_visible", None)
        if callable(setter):
            setter(True)

        if snapshot != self._recent_changes_snapshot:
            logger.debug(
                "Pending skill review changes for '%s': %s",
                entry.name,
                "; ".join(changes),
            )
        self._recent_changes_snapshot = snapshot

    def _compute_pending_skill_changes(self, entry: _SkillEntry) -> List[str]:
        changes: List[str] = []
        current_notes = self._get_notes_text()
        if current_notes.strip() != (self._baseline_notes or "").strip():
            changes.append("Tester notes updated")

        baseline_status = (self._baseline_status or "").strip()
        if baseline_status.casefold() != "reviewed" and self._supports_review_persistence():
            previous = baseline_status or "unreviewed"
            changes.append(f"Review status will change from '{previous}' to 'reviewed'")
        return changes

    def _load_skill_history(self, persona: Optional[str]) -> List[Dict[str, Any]]:
        server = getattr(self.ATLAS, "server", None)
        method = getattr(server, "list_skill_changes", None)
        self._skill_history_supported = callable(method)
        if not self._skill_history_supported:
            return []

        kwargs: Dict[str, Any] = {}
        if persona:
            kwargs["persona"] = persona

        try:
            response = method(**kwargs) if callable(method) else None
        except Exception as exc:  # pragma: no cover - backend failure logging
            logger.error("Failed to load skill history: %s", exc, exc_info=True)
            self._handle_backend_error("Unable to load skill change history from ATLAS.")
            return []

        records = response
        if isinstance(response, Mapping):
            records = response.get("changes")

        normalized: List[Dict[str, Any]] = []
        if isinstance(records, Iterable):
            for raw in records:
                entry = self._normalize_skill_history_record(raw)
                if entry is not None:
                    normalized.append(entry)
        return normalized[:10]

    def _normalize_skill_history_record(self, record: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(record, Mapping):
            return None

        summary = record.get("summary") or record.get("description")
        action = record.get("action") or record.get("change")
        skill_name = record.get("skill") or record.get("name")
        persona_name = record.get("persona") or record.get("persona_name")
        author = record.get("author") or record.get("user") or record.get("actor")

        if not summary:
            parts: List[str] = []
            if action:
                parts.append(str(action).strip().capitalize())
            if skill_name:
                parts.append(f"skill '{skill_name}'")
            if persona_name:
                parts.append(f"for persona '{persona_name}'")
            summary = " ".join(parts) or "Skill metadata updated."

        timestamp = record.get("timestamp") or record.get("updated_at") or record.get("created_at")
        parsed_timestamp = self._parse_timestamp(timestamp)

        return {
            "summary": str(summary).strip(),
            "author": str(author).strip() if author else "Unknown",
            "timestamp": parsed_timestamp,
            "persona": str(persona_name).strip() if persona_name else None,
        }

    def _sync_history_feed(self) -> None:
        box = self._history_box
        container = self._history_list_box
        if box is None or container is None:
            return

        self._clear_container(container)

        if not self._skill_history_supported:
            label = Gtk.Label(label="History feed is unavailable for this backend.")
            label.set_xalign(0.0)
            label.set_wrap(True)
            container.append(label)
            setter = getattr(box, "set_visible", None)
            if callable(setter):
                setter(True)
            return

        if not self._history_records:
            label = Gtk.Label(label="No recent changes recorded.")
            label.set_xalign(0.0)
            try:
                label.add_css_class("dim-label")
            except Exception:  # pragma: no cover - GTK theme variations
                pass
            container.append(label)
            setter = getattr(box, "set_visible", None)
            if callable(setter):
                setter(True)
            return

        for record in self._history_records:
            timestamp = record.get("timestamp")
            timestamp_text = self._format_history_timestamp(timestamp)
            author = record.get("author") or "Unknown"
            summary = record.get("summary") or "Skill metadata updated."
            persona_label = record.get("persona")
            if persona_label and persona_label != (self._persona_name or ""):
                summary = f"{summary} (persona: {persona_label})"

            entry_label = Gtk.Label()
            entry_label.set_xalign(0.0)
            entry_label.set_wrap(True)
            try:
                entry_label.add_css_class("dim-label")
            except Exception:  # pragma: no cover - GTK theme variations
                pass
            entry_label.set_label(f"{timestamp_text} – {author}: {summary}")
            container.append(entry_label)

        setter = getattr(box, "set_visible", None)
        if callable(setter):
            setter(True)

    def _format_history_timestamp(self, timestamp: Optional[datetime]) -> str:
        if timestamp is None:
            return "Unknown time"
        relative = self._format_relative_time(timestamp)
        iso_text = timestamp.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        return f"{relative} ({iso_text})"

    def _choose_export_path(self, *, title: str, suggested_name: str, pattern: str) -> Optional[str]:
        chooser_cls = getattr(Gtk, "FileChooserNative", None)
        action_enum = getattr(Gtk.FileChooserAction, "SAVE", None) if hasattr(Gtk, "FileChooserAction") else None
        if chooser_cls is None or action_enum is None:
            return None

        chooser = chooser_cls(title=title, transient_for=self.parent_window, modal=True, action=action_enum)
        if hasattr(chooser, "set_current_name"):
            try:
                chooser.set_current_name(suggested_name)
            except Exception:
                pass

        if self._last_export_directory and hasattr(chooser, "set_current_folder"):
            try:
                chooser.set_current_folder(str(self._last_export_directory))
            except Exception:
                pass

        file_filter_cls = getattr(Gtk, "FileFilter", None)
        if callable(file_filter_cls):
            try:
                file_filter = file_filter_cls()
                if hasattr(file_filter, "set_name"):
                    file_filter.set_name(pattern.upper())
                if hasattr(file_filter, "add_pattern"):
                    file_filter.add_pattern(pattern)
                adder = getattr(chooser, "add_filter", None)
                if callable(adder):
                    adder(file_filter)
            except Exception:  # pragma: no cover - GTK compatibility variations
                pass

        response = None
        if hasattr(chooser, "run"):
            try:
                response = chooser.run()
            except Exception:
                response = None
        elif hasattr(chooser, "show"):
            try:
                chooser.show()
                response = getattr(Gtk.ResponseType, "ACCEPT", 1)
            except Exception:
                response = None

        accepted = {
            getattr(Gtk.ResponseType, "ACCEPT", None),
            getattr(Gtk.ResponseType, "OK", None),
            getattr(Gtk.ResponseType, "YES", None),
        }

        filename: Optional[str] = None
        if response in accepted:
            file_obj = getattr(chooser, "get_file", None)
            handle = file_obj() if callable(file_obj) else None
            if handle is not None and hasattr(handle, "get_path"):
                filename = handle.get_path()
            else:
                getter = getattr(chooser, "get_filename", None)
                if callable(getter):
                    filename = getter()

        if hasattr(chooser, "destroy"):
            try:
                chooser.destroy()
            except Exception:
                pass

        if filename:
            path_obj = Path(filename).expanduser().resolve()
            self._last_export_directory = path_obj.parent
            return str(path_obj)
        return None

    def _choose_open_path(self, *, title: str, pattern: str) -> Optional[str]:
        chooser_cls = getattr(Gtk, "FileChooserNative", None)
        action_enum = getattr(Gtk.FileChooserAction, "OPEN", None) if hasattr(Gtk, "FileChooserAction") else None
        if chooser_cls is None or action_enum is None:
            return None

        chooser = chooser_cls(title=title, transient_for=self.parent_window, modal=True, action=action_enum)
        if self._last_export_directory and hasattr(chooser, "set_current_folder"):
            try:
                chooser.set_current_folder(str(self._last_export_directory))
            except Exception:
                pass

        file_filter_cls = getattr(Gtk, "FileFilter", None)
        if callable(file_filter_cls):
            try:
                file_filter = file_filter_cls()
                if hasattr(file_filter, "set_name"):
                    file_filter.set_name(pattern.upper())
                if hasattr(file_filter, "add_pattern"):
                    file_filter.add_pattern(pattern)
                adder = getattr(chooser, "add_filter", None)
                if callable(adder):
                    adder(file_filter)
            except Exception:  # pragma: no cover - GTK compatibility fallback
                pass

        response = None
        if hasattr(chooser, "run"):
            try:
                response = chooser.run()
            except Exception:
                response = None
        elif hasattr(chooser, "show"):
            try:
                chooser.show()
                response = getattr(Gtk.ResponseType, "ACCEPT", 1)
            except Exception:
                response = None

        accepted = {
            getattr(Gtk.ResponseType, "ACCEPT", None),
            getattr(Gtk.ResponseType, "OK", None),
            getattr(Gtk.ResponseType, "YES", None),
        }

        filename: Optional[str] = None
        if response in accepted:
            file_obj = getattr(chooser, "get_file", None)
            if callable(file_obj):
                file_handle = file_obj()
            else:
                file_handle = None
            if file_handle is not None and hasattr(file_handle, "get_path"):
                filename = file_handle.get_path()
            else:
                getter = getattr(chooser, "get_filename", None)
                if callable(getter):
                    filename = getter()

        if hasattr(chooser, "destroy"):
            try:
                chooser.destroy()
            except Exception:
                pass

        if filename:
            path_obj = Path(filename).expanduser().resolve()
            self._last_export_directory = path_obj.parent
            return str(path_obj)
        return None

    def _prompt_signing_key(self, title: str) -> Optional[str]:
        dialog_cls = getattr(Gtk, "Dialog", None)
        entry_cls = getattr(Gtk, "Entry", None)
        if dialog_cls is None or entry_cls is None:
            return None

        dialog = dialog_cls(title=title, transient_for=self.parent_window, modal=True)
        style = getattr(self.parent_window, "style_dialog", None)
        if callable(style):
            try:
                style(dialog)
            except Exception:
                pass

        content_area = getattr(dialog, "get_content_area", lambda: None)()
        if content_area is None and hasattr(dialog, "set_child"):
            box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
            dialog.set_child(box)
            content_area = box

        entry = entry_cls()
        if hasattr(entry, "set_visibility"):
            entry.set_visibility(False)

        label = Gtk.Label(label="Signing key:")
        if content_area is not None and hasattr(content_area, "append"):
            content_area.append(label)
            content_area.append(entry)

        if hasattr(dialog, "add_button"):
            dialog.add_button("Cancel", getattr(Gtk.ResponseType, "CANCEL", 0))
            dialog.add_button("OK", getattr(Gtk.ResponseType, "OK", 1))

        response = None
        if hasattr(dialog, "run"):
            try:
                response = dialog.run()
            except Exception:
                response = None
        else:
            response = getattr(Gtk.ResponseType, "OK", 1)

        key_value = entry.get_text().strip() if hasattr(entry, "get_text") else ""

        if hasattr(dialog, "destroy"):
            try:
                dialog.destroy()
            except Exception:
                pass

        if response == getattr(Gtk.ResponseType, "OK", 1) and key_value:
            return key_value
        return None

    def _show_info_message(self, message: str) -> None:
        if not message:
            return

        dialog = getattr(self.parent_window, "show_info_dialog", None)
        if callable(dialog):
            try:
                dialog(message)
                return
            except Exception:  # pragma: no cover - UI fallback
                logger.debug("Failed to show info dialog", exc_info=True)

        toast = getattr(self.parent_window, "show_success_toast", None)
        if callable(toast):
            try:
                toast(message)
                return
            except Exception:  # pragma: no cover - UI fallback
                logger.debug("Failed to show success toast", exc_info=True)

        logger.info("Skill management notification: %s", message)

    def _serialize_skill_entry(self, entry: _SkillEntry) -> Dict[str, Any]:
        return {
            "name": entry.name,
            "summary": entry.summary,
            "version": entry.version,
            "persona": entry.persona,
            "category": entry.category,
            "required_tools": list(entry.required_tools),
            "required_capabilities": list(entry.required_capabilities),
            "capability_tags": list(entry.capability_tags),
            "review_status": entry.review_status,
            "tester_notes": entry.tester_notes,
        }

    def _export_skills_to_csv(self, path: str, entries: List[_SkillEntry]) -> None:
        try:
            with open(path, "w", encoding="utf-8", newline="") as handle:
                fieldnames = [
                    "name",
                    "summary",
                    "version",
                    "persona",
                    "category",
                    "required_tools",
                    "required_capabilities",
                    "capability_tags",
                    "review_status",
                    "tester_notes",
                ]
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for entry in entries:
                    metadata = self._serialize_skill_entry(entry)
                    writer.writerow(
                        {
                            "name": metadata["name"],
                            "summary": metadata["summary"],
                            "version": metadata["version"] or "",
                            "persona": metadata["persona"] or "",
                            "category": metadata["category"] or "",
                            "required_tools": ", ".join(entry.required_tools),
                            "required_capabilities": ", ".join(entry.required_capabilities),
                            "capability_tags": ", ".join(entry.capability_tags),
                            "review_status": metadata["review_status"] or "",
                            "tester_notes": metadata["tester_notes"] or "",
                        }
                    )
        except Exception as exc:  # pragma: no cover - filesystem issues
            logger.error("Failed to export skill list to CSV: %s", exc, exc_info=True)
            self._handle_backend_error("Unable to export skill list to CSV.")

    def _export_skills_to_json(self, path: str, entries: List[_SkillEntry]) -> None:
        payload = [self._serialize_skill_entry(entry) for entry in entries]
        try:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, ensure_ascii=False)
        except Exception as exc:  # pragma: no cover - filesystem issues
            logger.error("Failed to export skill list to JSON: %s", exc, exc_info=True)
            self._handle_backend_error("Unable to export skill list to JSON.")

    def _on_export_skills_csv_clicked(self, _button: Gtk.Button) -> None:
        entries = list(self._display_entries or self._entries)
        if not entries:
            self._handle_backend_error("No skill data is available to export.")
            return

        path = self._choose_export_path(title="Export skills to CSV", suggested_name="skills.csv", pattern="*.csv")
        if not path:
            return
        self._export_skills_to_csv(path, entries)

    def _on_export_skills_json_clicked(self, _button: Gtk.Button) -> None:
        entries = list(self._display_entries or self._entries)
        if not entries:
            self._handle_backend_error("No skill data is available to export.")
            return

        path = self._choose_export_path(title="Export skills to JSON", suggested_name="skills.json", pattern="*.json")
        if not path:
            return
        self._export_skills_to_json(path, entries)

    def _on_export_skill_bundle_clicked(self, _button: Gtk.Button) -> None:
        skill_name = self._active_skill
        if not skill_name:
            self._handle_backend_error("Select a skill to export.")
            return

        suggested = f"{skill_name}.skillbundle"
        path = self._choose_export_path(
            title=f"Export {skill_name}",
            suggested_name=suggested,
            pattern="*.skillbundle",
        )
        if not path:
            return

        signing_key = self._prompt_signing_key("Enter signing key for skill export")
        if not signing_key:
            self._handle_backend_error("Export cancelled: signing key required.")
            return

        persona_owner: Optional[str] = None
        entry = self._entry_lookup.get(skill_name)
        if entry and entry.persona:
            persona_owner = entry.persona
        if not persona_owner and self._skill_scope == "persona" and self._persona_name:
            persona_owner = self._persona_name

        try:
            response = self.ATLAS.export_skill_bundle(
                skill_name,
                signing_key=signing_key,
                persona=persona_owner,
            )
        except Exception:
            logger.exception("Failed to export skill bundle for %s", skill_name)
            self._handle_backend_error("Failed to export skill bundle.")
            return

        if not response.get("success"):
            self._handle_backend_error(response.get("error") or "Skill export failed.")
            return

        bundle_bytes = response.get("bundle_bytes")
        if not isinstance(bundle_bytes, (bytes, bytearray)):
            self._handle_backend_error("Skill export did not return bundle data.")
            return

        try:
            Path(path).write_bytes(bundle_bytes)
        except OSError:
            self._handle_backend_error(f"Failed to write bundle to {path}.")
            return

        self._show_info_message(f"Exported skill '{skill_name}' to {path}.")

    def _on_import_skill_bundle_clicked(self, _button: Gtk.Button) -> None:
        path = self._choose_open_path(
            title="Import skill bundle",
            pattern="*.skillbundle",
        )
        if not path:
            return

        signing_key = self._prompt_signing_key("Enter signing key for skill import")
        if not signing_key:
            self._handle_backend_error("Import cancelled: signing key required.")
            return

        try:
            bundle_bytes = Path(path).read_bytes()
        except OSError:
            self._handle_backend_error(f"Failed to read bundle from {path}.")
            return

        try:
            response = self.ATLAS.import_skill_bundle(
                bundle_bytes=bundle_bytes,
                signing_key=signing_key,
                rationale="Imported via GTK UI",
            )
        except Exception:
            logger.exception("Failed to import skill bundle from %s", path)
            self._handle_backend_error("Failed to import skill bundle.")
            return

        if not response.get("success"):
            self._handle_backend_error(response.get("error") or "Skill import failed.")
            return

        skill = response.get("skill") or {}
        skill_name = skill.get("name") or "skill"
        self._show_info_message(f"Imported skill '{skill_name}' from {path}.")
        self._refresh_state()

    def _sync_preview_button_state(self) -> None:
        button = self._preview_button
        if button is None:
            return
        enabled = bool(self._active_skill and not self._preview_in_progress and self._resolve_backend_method("get_skill_details"))
        button.set_sensitive(enabled)

    def _sync_test_button_state(self) -> None:
        button = self._test_button
        if button is None:
            return
        supported = bool(self._active_skill and self._supports_test_invocation())
        button.set_sensitive(supported and not self._test_in_progress)

    def _sync_review_button_state(self) -> None:
        button = self._mark_reviewed_button
        if button is None:
            return
        supported = bool(self._active_skill and self._supports_review_persistence())
        button.set_sensitive(supported)

    def _supports_test_invocation(self) -> bool:
        return self._resolve_backend_method(
            "validate_skill",
            "test_skill",
            "run_skill_test",
            "run_skill_validation",
            "invoke_skill_test",
        ) is not None

    def _supports_review_persistence(self) -> bool:
        return self._resolve_backend_method("set_skill_metadata") is not None

    def _resolve_backend_method(self, *names: str) -> Optional[Any]:
        server = getattr(self.ATLAS, "server", None)
        if server is None:
            return None
        for name in names:
            method = getattr(server, name, None)
            if callable(method):
                return method
        return None

    def _call_backend_with_variants(
        self,
        method: Any,
        skill_name: str,
        persona: Optional[str],
        extra: Dict[str, Any],
    ) -> Any:
        base_payload = dict(extra)
        variants: List[Dict[str, Any]] = []
        if persona:
            for key in ("name", "skill", "skill_name"):
                variant = dict(base_payload)
                variant["persona"] = persona
                variant[key] = skill_name
                variants.append(variant)
        for key in ("name", "skill", "skill_name"):
            variant = dict(base_payload)
            variant[key] = skill_name
            variants.append(variant)

        last_exc: Optional[TypeError] = None
        for kwargs in variants:
            try:
                return method(**kwargs)
            except TypeError as exc:
                last_exc = exc
                continue

        if last_exc is not None:
            raise last_exc
        return method(**base_payload)

    def _stringify_details(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, bytes):
            try:
                return value.decode("utf-8")
            except Exception:  # pragma: no cover - decoding fallback
                return value.decode("utf-8", errors="ignore")
        if isinstance(value, (Mapping, list, tuple)):
            try:
                return json.dumps(value, indent=2, sort_keys=True)
            except Exception:  # pragma: no cover - json fallback
                return str(value)
        return str(value)

    def _interpret_test_response(self, response: Any) -> tuple[bool, str, str]:
        success: Optional[bool] = None
        message = ""
        details = ""

        if isinstance(response, Mapping):
            if "success" in response:
                success = bool(response.get("success"))
            error_value = response.get("error")
            if error_value and success is not True:
                message = str(error_value)
                success = False
            status_value = response.get("message") or response.get("status")
            if status_value:
                message = str(status_value)
            for key in ("output", "result", "details", "data", "response"):
                if key in response:
                    details = self._stringify_details(response.get(key))
                    break
            if not details:
                details = self._stringify_details(response)
        else:
            details = self._stringify_details(response)

        if not message:
            message = "Test invocation completed."
        if success is None:
            success = True
        return success, message, details

    def _set_buffer_text(self, buffer: Optional[Gtk.TextBuffer], text: str) -> None:
        if buffer is None:
            return
        setter = getattr(buffer, "set_text", None)
        if callable(setter):
            try:
                setter(text)
            except TypeError:
                setter(text, -1)

    def _get_buffer_text(self, buffer: Optional[Gtk.TextBuffer]) -> str:
        if buffer is None:
            return ""
        getter = getattr(buffer, "get_bounds", None)
        if callable(getter):
            try:
                start, end = getter()
            except Exception:  # pragma: no cover - GTK compatibility variations
                start = end = None
            else:
                text_getter = getattr(buffer, "get_text", None)
                if callable(text_getter) and start is not None and end is not None:
                    try:
                        return text_getter(start, end, True)
                    except TypeError:
                        try:
                            return text_getter(start, end)
                        except Exception:
                            pass
        value = getattr(buffer, "props", None)
        if value is not None:
            text = getattr(value, "text", None)
            if isinstance(text, str):
                return text
        fallback = getattr(buffer, "text", None)
        if isinstance(fallback, str):
            return fallback
        return ""

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _clear_container(self, container: Optional[Gtk.Widget]) -> None:
        if container is None:
            return

        remover = getattr(container, "remove_all", None)
        if callable(remover):
            try:
                remover()
                return
            except Exception:  # pragma: no cover - GTK compatibility fallbacks
                pass

        children = self._get_container_children(container)
        remove_child = getattr(container, "remove", None)
        if callable(remove_child):
            for child in children:
                try:
                    remove_child(child)
                except Exception:  # pragma: no cover - GTK compatibility fallbacks
                    continue

    def _get_container_children(self, container: Gtk.Widget) -> List[Gtk.Widget]:
        getter = getattr(container, "get_children", None)
        if callable(getter):
            try:
                return list(getter())
            except Exception:  # pragma: no cover - GTK compatibility fallbacks
                pass

        return list(getattr(container, "children", []) or [])

    def _set_button_sensitive(self, button: Optional[Gtk.Button], enabled: bool) -> None:
        if button is None:
            return
        setter = getattr(button, "set_sensitive", None)
        if callable(setter):
            setter(bool(enabled))

    def _set_label(self, label: Optional[Gtk.Label], text: str) -> None:
        if label is None:
            return
        setter = getattr(label, "set_label", None) or getattr(label, "set_text", None)
        if callable(setter):
            setter(text)
        else:  # pragma: no cover - GTK compatibility fallback
            label.label = text

    def _format_list(self, items: Iterable[str]) -> str:
        values = [item for item in items if item]
        return ", ".join(values) if values else "None"

    def _parse_timestamp(self, value: Any) -> Optional[datetime]:
        if isinstance(value, (int, float)):
            timestamp = float(value)
            if timestamp > 10**12:
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

    def _handle_backend_error(self, message: str) -> None:
        logger.error("Skill management error: %s", message)
        dialog = getattr(self.parent_window, "show_error_dialog", None)
        if callable(dialog):
            try:
                dialog(message)
            except Exception:  # pragma: no cover - user interface edge cases
                logger.debug("Failed to show skill management error dialog", exc_info=True)
