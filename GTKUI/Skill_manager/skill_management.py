"""GTK skill management workspace used by the sidebar."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import Gtk

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


class SkillManagement:
    """Controller responsible for rendering the skill management workspace."""

    _CATEGORY_ALL_ID = "__all__categories__"
    _PERSONA_ALL_ID = "__all__personas__"
    _PERSONA_SHARED_ID = "__shared__personas__"
    _DEFAULT_SORT_ORDER = "name_asc"

    def __init__(self, atlas: Any, parent_window: Any) -> None:
        self.ATLAS = atlas
        self.parent_window = parent_window

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
        self._scope_selector: Optional[Gtk.ComboBoxText] = None
        self._search_entry: Optional[Gtk.SearchEntry] = None
        self._category_combo: Optional[Gtk.ComboBoxText] = None
        self._persona_combo: Optional[Gtk.ComboBoxText] = None
        self._sort_combo: Optional[Gtk.ComboBoxText] = None

        self._entries: List[_SkillEntry] = []
        self._entry_lookup: Dict[str, _SkillEntry] = {}
        self._row_lookup: Dict[str, Gtk.Widget] = {}
        self._display_entries: List[_SkillEntry] = []
        self._visible_entry_names: Set[str] = set()
        self._active_skill: Optional[str] = None
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_embeddable_widget(self) -> Gtk.Widget:
        """Return the workspace widget, creating it on first use."""

        if self._widget is None:
            self._widget = self._build_workspace()
        self._refresh_state()
        return self._widget

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

        search_entry = Gtk.SearchEntry()
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

        required_capabilities_label = Gtk.Label()
        required_capabilities_label.set_wrap(True)
        required_capabilities_label.set_xalign(0.0)
        right_panel.append(required_capabilities_label)
        self._required_capabilities_label = required_capabilities_label

        capability_tags_label = Gtk.Label()
        capability_tags_label.set_wrap(True)
        capability_tags_label.set_xalign(0.0)
        right_panel.append(capability_tags_label)
        self._capability_tags_label = capability_tags_label

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
        self._populate_filter_options()
        self._rebuild_skill_list()

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

        def _normalize_sequence(value: Any) -> List[str]:
            result: List[str] = []
            if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
                for item in value:
                    text = str(item).strip()
                    if text:
                        result.append(text)
            return result

        persona = entry.get("persona")
        category = entry.get("category")
        safety_notes = entry.get("safety_notes")
        version = entry.get("version")
        required_tools = _normalize_sequence(entry.get("required_tools"))
        required_capabilities = _normalize_sequence(entry.get("required_capabilities"))
        capability_tags = _normalize_sequence(entry.get("capability_tags"))
        source = entry.get("source")

        normalized_metadata = dict(entry)
        normalized_metadata.setdefault("name", name)

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
        title = Gtk.Label(label=entry.name)
        title.set_xalign(0.0)
        container.append(title)

        summary = Gtk.Label(label=entry.summary)
        summary.set_xalign(0.0)
        summary.set_wrap(True)
        try:
            summary.add_css_class("dim-label")
        except Exception:  # pragma: no cover - GTK theme variations
            pass
        container.append(summary)

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
        self._set_label(
            self._required_capabilities_label,
            "Required capabilities: " + self._format_list(entry.required_capabilities),
        )
        self._set_label(
            self._capability_tags_label,
            "Capability tags: " + self._format_list(entry.capability_tags),
        )
        safety = entry.safety_notes or "No safety notes provided."
        self._set_label(self._safety_notes_label, f"Safety notes: {safety}")
        source = entry.source or "Source metadata unavailable"
        self._set_label(self._source_label, f"Source: {source}")

        if self._skill_list is not None:
            try:
                self._skill_list.select_row(row)
            except Exception:  # pragma: no cover - GTK stub variations
                pass

    def _show_empty_state(self) -> None:
        self._active_skill = None
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
    # Utility helpers
    # ------------------------------------------------------------------
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

    def _handle_backend_error(self, message: str) -> None:
        logger.error("Skill management error: %s", message)
        dialog = getattr(self.parent_window, "show_error_dialog", None)
        if callable(dialog):
            try:
                dialog(message)
            except Exception:  # pragma: no cover - user interface edge cases
                logger.debug("Failed to show skill management error dialog", exc_info=True)
