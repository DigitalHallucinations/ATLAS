"""GTK skill management workspace used by the sidebar."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

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

        self._entries: List[_SkillEntry] = []
        self._entry_lookup: Dict[str, _SkillEntry] = {}
        self._row_lookup: Dict[str, Gtk.Widget] = {}
        self._active_skill: Optional[str] = None

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
        entries = self._load_skill_entries(persona)

        self._entries = entries
        self._entry_lookup = {entry.name: entry for entry in entries}
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

        for entry in self._entries:
            row = self._create_row(entry)
            self._skill_list.append(row)
            self._row_lookup[entry.name] = row

        if self._entries:
            self._select_skill(self._entries[0].name)
        else:
            self._show_empty_state()

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

        row = self._row_lookup.get(skill_name)
        if row is not None and self._skill_list is not None:
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
