"""GTK tool management workspace used by the sidebar."""

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

    def __init__(self, atlas: Any, parent_window: Any) -> None:
        self.ATLAS = atlas
        self.parent_window = parent_window

        self._widget: Optional[Gtk.Widget] = None
        self._tool_list: Optional[Gtk.ListBox] = None
        self._persona_label: Optional[Gtk.Label] = None
        self._title_label: Optional[Gtk.Label] = None
        self._summary_label: Optional[Gtk.Label] = None
        self._capabilities_label: Optional[Gtk.Label] = None
        self._auth_label: Optional[Gtk.Label] = None
        self._switch: Optional[Gtk.Switch] = None
        self._save_button: Optional[Gtk.Button] = None
        self._reset_button: Optional[Gtk.Button] = None
        self._scope_selector: Optional[Gtk.ComboBoxText] = None

        self._entries: List[_ToolEntry] = []
        self._entry_lookup: Dict[str, _ToolEntry] = {}
        self._row_lookup: Dict[str, Gtk.Widget] = {}

        self._persona_name: Optional[str] = None
        self._enabled_tools: set[str] = set()
        self._baseline_enabled: set[str] = set()
        self._active_tool: Optional[str] = None
        self._suppress_toggle = False
        self._tool_scope = "persona"
        self._suppress_scope_signal = False

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

        heading = Gtk.Label(label="Available Tools")
        heading.set_xalign(0.0)
        left_panel.append(heading)

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

        auth_label = Gtk.Label()
        auth_label.set_wrap(True)
        auth_label.set_xalign(0.0)
        right_panel.append(auth_label)
        self._auth_label = auth_label

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

        for entry in self._entries:
            row = self._create_row(entry)
            self._tool_list.append(row)
            self._row_lookup[entry.name] = row

        if self._entries:
            self._select_tool(self._entries[0].name)
        else:
            self._show_empty_state()

    def _create_row(self, entry: _ToolEntry) -> Gtk.Widget:
        container = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        title = Gtk.Label(label=entry.title or entry.name)
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

    def _select_tool(self, tool_name: str) -> None:
        entry = self._entry_lookup.get(tool_name)
        if entry is None:
            return

        self._active_tool = tool_name

        self._set_label(self._title_label, entry.title or entry.name)
        self._set_label(self._summary_label, entry.summary or "")

        if entry.capabilities:
            capability_text = "Capabilities: " + ", ".join(entry.capabilities)
        else:
            capability_text = "Capabilities: (unspecified)"
        self._set_label(self._capabilities_label, capability_text)

        self._set_label(self._auth_label, self._format_auth(entry))

        row = self._row_lookup.get(tool_name)
        if row is not None and self._tool_list is not None:
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
        self._set_switch_sensitive(self._current_scope_allows_editing())
        self._set_switch_active(False)
        self._update_action_state()

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------
    def _on_row_selected(self, _listbox: Gtk.ListBox, row: Gtk.Widget) -> None:
        for name, candidate in self._row_lookup.items():
            if candidate is row:
                self._select_tool(name)
                break

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
        self._refresh_state()

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
            self._set_button_sensitive(self._reset_button, False)
            return

        dirty = self._enabled_tools != self._baseline_enabled
        self._set_button_sensitive(self._save_button, dirty and bool(self._persona_name))
        self._set_button_sensitive(self._reset_button, True)

    def _set_button_sensitive(self, button: Optional[Gtk.Button], enabled: bool) -> None:
        if button is None:
            return
        setter = getattr(button, "set_sensitive", None)
        if callable(setter):
            setter(bool(enabled))

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

