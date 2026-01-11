# GTKUI/Persona_manager/persona_management.py
# pyright: reportMissingImports=false, reportMissingModuleSource=false
from __future__ import annotations

import logging

import gi
gi.require_version('Gtk', '4.0')
from gi.repository import Gtk, Gdk, GLib  # type: ignore

import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Tuple, cast, runtime_checkable, TYPE_CHECKING

if TYPE_CHECKING:
    from gi.repository import Gtk as GtkType  # type: ignore[import]


@runtime_checkable
class PersonaFacade(Protocol):
    """Minimal persona-facing surface exposed by the ATLAS core.

    Keeping this lightweight protocol helps Pylance reason about the
    dynamic ATLAS object without pulling backend dependencies into the UI agent.
    """

    @property
    def persona_manager(self) -> Any: ...
    def register_message_dispatcher(self, handler: Callable[[str, str], None]) -> None: ...
    def show_persona_message(self, role: str, message: str) -> None: ...
    def get_persona_names(self) -> List[str]: ...
    def load_persona(self, persona: str) -> None: ...
    def get_current_persona_prompt(self) -> str: ...
    def get_persona_editor_state(self, persona_name: str) -> Optional[Dict[str, Any]]: ...
    def update_persona_from_editor(
        self,
        persona_name: str,
        general: Optional[Dict[str, Any]] = None,
        persona_type: Optional[Dict[str, Any]] = None,
        provider: Optional[Dict[str, Any]] = None,
        speech: Optional[Dict[str, Any]] = None,
        tools: Optional[List[str]] = None,
        skills: Optional[List[str]] = None,
    ) -> Dict[str, Any]: ...
    def export_persona_bundle(self, persona_name: str, *, signing_key: str) -> Dict[str, Any]: ...
    def import_persona_bundle(self, *, bundle_bytes: bytes, signing_key: str, rationale: str) -> Dict[str, Any]: ...
    def attest_persona_review(self, persona_name: str) -> Dict[str, Any]: ...
    def get_persona_review_status(self, persona_name: str) -> Dict[str, Any]: ...
    def get_persona_audit_history(
        self, persona_name: str, limit: int, offset: int
    ) -> Tuple[List[Dict[str, Any]], int]: ...
    def get_persona_metrics(
        self,
        persona_name: str,
        tool_name: Optional[str] = None,
        skill_name: Optional[str] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 10,
        metric_type: Optional[str] = None,
        category: Optional[str] = None,
        recent: bool = False,
    ) -> Dict[str, Any]: ...
    def get_persona_comparison_summary(
        self,
        persona_names: List[str] = [],
        metric_type: Optional[str] = None,
        limit: int = 5,
        category: Optional[str] = None,
        recent: Optional[int] = None,
    ) -> Dict[str, Any]: ...

from .General_Tab.general_tab import GeneralTab
from .Persona_Type_Tab.persona_type_tab import PersonaTypeTab
from .analytics_charts import AnomalyHeatmap, LatencyTimeline
from GTKUI.Utils.styled_window import AtlasWindow
from GTKUI.Utils.utils import apply_css


logger = logging.getLogger(__name__)


class PersonaManagement:
    """
    Manages the persona selection and settings functionality.
    Displays a window listing available personas and allows switching or
    opening the settings for each persona.
    """

    def __init__(self, ATLAS: PersonaFacade, parent_window: Gtk.Window):
        """
        Initializes the PersonaManagement.

        Args:
            ATLAS (ATLAS): The main ATLAS instance.
            parent_window (Gtk.Window): The parent window for the persona menu.
        """
        self.ATLAS: PersonaFacade = ATLAS
        self.parent_window: Gtk.Window = parent_window
        self.persona_window = None
        self._persona_page = None
        self.settings_window = None
        self._settings_stack: Optional[Gtk.Stack] = None
        self._editor_persona_name: Optional[str] = None
        self._persona_message_handler = self._handle_persona_message
        self.ATLAS.register_message_dispatcher(self._persona_message_handler)
        self._current_editor_state = None
        self.tool_rows: Dict[str, Dict[str, Any]] = {}
        self._tool_order: List[str] = []
        self.tool_list_box: Optional[Gtk.ListBox] = None
        self.skill_rows: Dict[str, Dict[str, Any]] = {}
        self._skill_order: List[str] = []
        self.skill_list_box: Optional[Gtk.ListBox] = None
        self._skill_dependency_conflicts: Dict[str, List[str]] = {}
        self.history_list_box: Optional[Gtk.ListBox] = None
        self._history_persona_name: Optional[str] = None
        self._history_page_size: int = 20
        self._history_offset: int = 0
        self._history_total: int = 0
        self._history_load_more_button: Optional[Gtk.Button] = None
        self._history_placeholder: Optional[Gtk.Widget] = None
        self._last_bundle_directory: Optional[Path] = None
        self._analytics_persona_name: Optional[str] = None
        self._analytics_start_entry: Optional[Gtk.Entry] = None
        self._analytics_end_entry: Optional[Gtk.Entry] = None
        self._analytics_summary_labels: Dict[str, Gtk.Label] = {}
        self._analytics_comparison_labels: Dict[str, Gtk.Label] = {}
        self._analytics_tool_list: Optional[Gtk.ListBox] = None
        self._analytics_tool_placeholder: Optional[Gtk.Widget] = None
        self._analytics_anomaly_list: Optional[Gtk.ListBox] = None
        self._analytics_anomaly_placeholder: Optional[Gtk.Widget] = None
        self._analytics_recent_list: Optional[Gtk.ListBox] = None
        self._analytics_recent_placeholder: Optional[Gtk.Widget] = None
        self._analytics_anomaly_heatmap: Optional[AnomalyHeatmap] = None
        self._analytics_latency_chart: Optional[LatencyTimeline] = None
        self._analytics_anomaly_rows: Dict[Gtk.ListBoxRow, Dict[str, Any]] = {}
        self._analytics_anomaly_entries: List[Dict[str, Any]] = []
        self._analytics_recent_entries: List[Dict[str, Any]] = []
        self._theme_monitor_connected: bool = False
        self._review_banner_box: Optional[Gtk.Box] = None
        self._review_banner_label: Optional[Gtk.Label] = None
        self._review_mark_complete_button: Optional[Gtk.Button] = None
        self._review_status: Optional[Dict[str, Any]] = None
        self._review_persona_name: Optional[str] = None
        self._persona_metadata: Dict[str, Mapping[str, Any]] = {}

        self.provider_entry: Optional[Gtk.Entry] = None
        self.model_entry: Optional[Gtk.Entry] = None
        self.speech_provider_entry: Optional[Gtk.Entry] = None
        self.voice_entry: Optional[Gtk.Entry] = None


    # --------------------------- Helpers ---------------------------

    def _abs_icon(self, rel_path: str) -> str:
        """Resolve absolute path to an icon from project root."""
        base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        return os.path.join(base, rel_path)

    def _safe_set_text(self, widget: Any, text: str) -> None:
        """Safely set text on a widget if possible."""
        if widget is None:
            return
        if hasattr(widget, "set_text"):
            # Use getattr to avoid static analysis errors on specific widget types
            getattr(widget, "set_text")(text)
        elif hasattr(widget, "set_label"):
            getattr(widget, "set_label")(text)

    def _safe_get_text(self, widget: Any) -> str:
        """Safely get text from a widget if possible."""
        if widget is None:
            return ""
        if hasattr(widget, "get_text"):
            return str(getattr(widget, "get_text")())
        if hasattr(widget, "get_label"):
            return str(getattr(widget, "get_label")())
        return ""

    def _safe_get_active(self, widget: Any) -> bool:
        """Safely check if a widget is active."""
        if widget is None:
            return False
        if hasattr(widget, "get_active"):
            return bool(getattr(widget, "get_active")())
        return False


    def _style_dialog(self, dialog: Gtk.Dialog) -> None:
        """Ensure dialogs inherit the app styling and dark theme classes."""
        apply_css()
        context = dialog.get_style_context()
        context.add_class("chat-page")
        context.add_class("sidebar")

    def _ensure_theme_monitoring(self) -> None:
        if self._theme_monitor_connected:
            return
        settings = Gtk.Settings.get_default()
        if settings is None:
            return
        try:
            settings.connect("notify::gtk-theme-name", self._on_theme_setting_changed)
            settings.connect(
                "notify::gtk-application-prefer-dark-theme",
                self._on_theme_setting_changed,
            )
            self._theme_monitor_connected = True
        except Exception:
            # Theme notifications are best-effort; ignore if unsupported.
            self._theme_monitor_connected = False

    def _on_theme_setting_changed(self, *_args: Any) -> None:
        self._apply_chart_theme()

    def _apply_chart_theme(self) -> None:
        is_dark = self._is_dark_theme()
        if self._analytics_anomaly_heatmap is not None:
            self._analytics_anomaly_heatmap.set_dark_mode(is_dark)
        if self._analytics_latency_chart is not None:
            self._analytics_latency_chart.set_dark_mode(is_dark)

    def _is_dark_theme(self) -> bool:
        settings = Gtk.Settings.get_default()
        if settings is None:
            return True
        prefer_dark = False
        try:
            prefer_dark = bool(settings.props.gtk_application_prefer_dark_theme)
        except Exception:
            prefer_dark = False
        if prefer_dark:
            return True
        try:
            theme_name = settings.props.gtk_theme_name
        except Exception:
            theme_name = ""
        if isinstance(theme_name, str) and "dark" in theme_name.lower():
            return True
        return False

    def _handle_persona_message(self, role: str, message: str) -> None:
        """Surface persona manager messages via a modal dialog."""
        if not message:
            return

        role_normalized = (role or "").lower()
        if role_normalized == "error":
            message_type = Gtk.MessageType.ERROR
            title = "Error"
        elif role_normalized == "warning":
            message_type = Gtk.MessageType.WARNING
            title = "Warning"
        else:
            message_type = Gtk.MessageType.INFO
            title = "System"

        def _present_dialog():
            dialog = Gtk.MessageDialog(
                transient_for=self.parent_window,
                modal=True,
                message_type=message_type,
                buttons=Gtk.ButtonsType.OK,
                text=title,
            )
            self._style_dialog(dialog)
            dialog.format_secondary_text(message)
            dialog.connect("response", lambda d, *_: d.destroy())
            dialog.present()
            return False

        GLib.idle_add(_present_dialog)

    def _set_widget_visible(self, widget: Optional[Gtk.Widget], visible: bool) -> None:
        if widget is None:
            return
        setter = getattr(widget, "set_visible", None)
        if callable(setter):
            setter(bool(visible))
        else:
            setattr(widget, "_visible", bool(visible))

    def _set_widget_error_state(self, widget: Any, enabled: bool) -> None:
        if widget is None:
            return
        add_class = getattr(widget, "add_css_class", None)
        remove_class = getattr(widget, "remove_css_class", None)
        if not callable(add_class) or not callable(remove_class):
            return
        try:
            if enabled:
                add_class("error")
            else:
                remove_class("error")
        except Exception:
            return

    def _clear_skill_dependency_highlights(self) -> None:
        self._skill_dependency_conflicts.clear()
        for info in self.skill_rows.values():
            check = info.get('check')
            row = info.get('row')
            hint = info.get('hint')
            default_hint = info.get('default_hint')
            default_tooltip = info.get('default_tooltip')

            self._set_widget_error_state(check, False)
            self._set_widget_error_state(row, False)

            set_tooltip = getattr(check, "set_tooltip_text", None)
            if callable(set_tooltip):
                set_tooltip(default_tooltip if default_tooltip else None)

            set_text = getattr(hint, "set_text", None)
            if callable(set_text) and default_hint is not None:
                set_text(default_hint)

    def _apply_skill_dependency_highlights(self, conflicts: Dict[str, List[str]]) -> None:
        if not conflicts:
            self._clear_skill_dependency_highlights()
            return

        self._skill_dependency_conflicts = {key: list(value) for key, value in conflicts.items()}

        for skill_name, missing_tools in conflicts.items():
            info = self.skill_rows.get(skill_name)
            if not info:
                continue

            check = info.get('check')
            row = info.get('row')
            hint = info.get('hint')
            default_hint = info.get('default_hint') or ""
            default_tooltip = info.get('default_tooltip') or ""

            message = f"Enable required tools: {', '.join(missing_tools)}"

            self._set_widget_error_state(check, True)
            self._set_widget_error_state(row, True)

            set_tooltip = getattr(check, "set_tooltip_text", None)
            if callable(set_tooltip):
                tooltip_text = default_tooltip
                if tooltip_text:
                    tooltip_text = f"{tooltip_text}\n\n{message}"
                else:
                    tooltip_text = message
                set_tooltip(tooltip_text)

            set_text = getattr(hint, "set_text", None)
            if callable(set_text):
                if default_hint:
                    set_text(f"{default_hint}\n⚠ Missing tools: {', '.join(missing_tools)}")
                else:
                    set_text(f"⚠ Missing tools: {', '.join(missing_tools)}")

    def _handle_skill_dependency_errors(self, error_messages: List[str]) -> tuple[bool, str]:
        pattern = re.compile(r"Skill '([^']+)' requires missing tools: ([^\n]+)")
        conflicts: Dict[str, List[str]] = {}

        for message in error_messages:
            if not isinstance(message, str):
                continue
            for match in pattern.finditer(message):
                skill_name = match.group(1).strip()
                tools_blob = match.group(2)
                missing_tools = [tool.strip() for tool in tools_blob.split(',') if tool.strip()]
                if not skill_name or not missing_tools:
                    continue
                conflicts[skill_name] = missing_tools

        if not conflicts:
            return (False, "")

        self._apply_skill_dependency_highlights(conflicts)
        guidance_parts = [f"{skill}: {', '.join(tools)}" for skill, tools in conflicts.items()]
        guidance = "Enable the required tools for the highlighted skills (" + "; ".join(guidance_parts) + ")."
        return (True, guidance)

    def _refresh_review_status(self, persona_name: str) -> None:
        if not persona_name:
            self._set_widget_visible(self._review_banner_box, False)
            return

        try:
            status = self.ATLAS.get_persona_review_status(persona_name)
        except Exception as exc:
            self._set_widget_visible(self._review_banner_box, False)
            self.ATLAS.show_persona_message(
                "warning",
                f"Failed to load review status for {persona_name}: {exc}",
            )
            return

        self._review_status = dict(status) if isinstance(status, dict) else None
        self._update_review_banner(persona_name, self._review_status)

    def _update_review_banner(self, persona_name: str, status: Optional[Dict[str, Any]]) -> None:
        box = self._review_banner_box
        label = self._review_banner_label
        button = self._review_mark_complete_button

        if box is None or label is None:
            return

        if not isinstance(status, dict):
            self._set_widget_visible(box, False)
            return

        overdue = bool(status.get("overdue"))
        pending = bool(status.get("pending_task"))
        last_review = status.get("last_review")
        reviewer = status.get("reviewer") or "unknown"
        next_due = status.get("next_due") or status.get("expires_at")

        message_parts: List[str] = []
        if overdue:
            due_text = next_due or "now"
            message_parts.append(f"Review overdue since {due_text}.")
            if pending:
                message_parts.append("A review task has been queued.")
            else:
                message_parts.append("Please attest the review to clear this alert.")
        else:
            if last_review:
                message_parts.append(
                    f"Last reviewed on {last_review} by {reviewer}."
                )
            else:
                message_parts.append("No review attestation recorded yet.")
            if next_due:
                message_parts.append(f"Next review due on {next_due}.")

        text = " ".join(message_parts)
        if hasattr(label, "set_text"):
            try:
                label.set_text(text)
            except Exception:
                pass
        if hasattr(label, "set_label"):
            try:
                label.set_label(text)
            except Exception:
                pass
        label.label = text  # type: ignore[attr-defined]
        self._set_widget_visible(box, True)

        if button is not None:
            button.set_sensitive(not pending)
            tooltip = "Record that this persona review has been completed."
            if pending:
                tooltip = "A review task is pending; complete it before attesting."
            if hasattr(button, "set_tooltip_text"):
                button.set_tooltip_text(tooltip)

    def _on_mark_review_complete(self, *_args: Any) -> None:
        persona_name = self._review_persona_name or ""
        if not persona_name:
            return

        try:
            result = self.ATLAS.attest_persona_review(persona_name)
        except Exception as exc:
            self.ATLAS.show_persona_message(
                "error",
                f"Failed to mark {persona_name} as reviewed: {exc}",
            )
            return

        status = result.get("status") if isinstance(result, dict) else None
        if isinstance(status, dict):
            self._review_status = status
            self._update_review_banner(persona_name, status)

        attestation = result.get("attestation") if isinstance(result, dict) else None
        reviewer = "unknown"
        if isinstance(attestation, dict):
            reviewer = attestation.get("reviewer") or reviewer
        elif isinstance(status, dict):
            reviewer = status.get("reviewer") or reviewer

        self.ATLAS.show_persona_message(
            "system",
            f"Recorded review attestation for {persona_name} by {reviewer}.",
        )

    def _make_icon_widget(self, rel_path: str, fallback_icon_name: str = "emblem-system-symbolic", size: int = 16) -> Gtk.Widget:
        """
        Try to create a Gtk.Picture from a file path; fall back to a themed icon name.
        Returns a Gtk.Picture (file) or Gtk.Image (themed) widget.
        """
        try:
            path = self._abs_icon(rel_path)
            texture = Gdk.Texture.new_from_filename(path)
            pic = Gtk.Picture.new_for_paintable(texture)
            pic.set_size_request(size, size)
            pic.set_content_fit(Gtk.ContentFit.CONTAIN)
            return pic
        except Exception:
            img = Gtk.Image.new_from_icon_name(fallback_icon_name)
            img.set_pixel_size(size)
            return img

    def _choose_file_path(
        self,
        *,
        title: str,
        action: str,
        suggested_name: Optional[str] = None,
    ) -> Optional[str]:
        chooser_cls = getattr(Gtk, "FileChooserNative", None)
        action_enum = getattr(Gtk.FileChooserAction, action.upper(), None) if hasattr(Gtk, "FileChooserAction") else None
        if chooser_cls is None or action_enum is None:
            return None

        chooser = chooser_cls(title=title, transient_for=self.parent_window, modal=True, action=action_enum)
        if suggested_name and hasattr(chooser, "set_current_name"):
            try:
                chooser.set_current_name(suggested_name)
            except Exception:
                pass

        if self._last_bundle_directory and hasattr(chooser, "set_current_folder"):
            try:
                chooser.set_current_folder(str(self._last_bundle_directory))
            except Exception:
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

        accepted_responses = {
            getattr(Gtk.ResponseType, "ACCEPT", None),
            getattr(Gtk.ResponseType, "OK", None),
            getattr(Gtk.ResponseType, "YES", None),
        }

        filename: Optional[str] = None
        if response in accepted_responses:
            file_obj = getattr(chooser, "get_file", lambda: None)()
            if file_obj is not None and hasattr(file_obj, "get_path"):
                filename = file_obj.get_path()
            elif hasattr(chooser, "get_filename"):
                filename = chooser.get_filename()

        if hasattr(chooser, "destroy"):
            try:
                chooser.destroy()
            except Exception:
                pass

        if filename:
            path_obj = Path(filename).expanduser().resolve()
            self._last_bundle_directory = path_obj.parent
            return str(path_obj)

        return None

    def _prompt_signing_key(self, title: str) -> Optional[str]:
        dialog_cls = getattr(Gtk, "Dialog", None)
        entry_cls = getattr(Gtk, "Entry", None)
        if dialog_cls is None or entry_cls is None:
            return None

        dialog = dialog_cls(title=title, transient_for=self.parent_window, modal=True)
        self._style_dialog(dialog)

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

    def _refresh_persona_catalog(self) -> None:
        manager = getattr(self.ATLAS, "persona_manager", None)
        refresher = getattr(manager, "refresh_catalog", None)

        if callable(refresher):
            try:
                refresher()
            except Exception:
                pass

        self._persona_page = None

    def _on_export_persona_clicked(self, _button):
        if not self._current_editor_state:
            self.ATLAS.show_persona_message("error", "No persona loaded for export.")
            return

        persona_name = self._current_editor_state.get("original_name") or ""
        persona_name = str(persona_name).strip()
        if not persona_name:
            self.ATLAS.show_persona_message("error", "Persona name is required for export.")
            return

        path = self._choose_file_path(
            title=f"Export {persona_name}",
            action="SAVE",
            suggested_name=f"{persona_name}.atlasbundle",
        )
        if not path:
            return

        signing_key = self._prompt_signing_key("Enter signing key for export")
        if not signing_key:
            self.ATLAS.show_persona_message("warning", "Export cancelled: signing key required.")
            return

        try:
            response = self.ATLAS.export_persona_bundle(persona_name, signing_key=signing_key)
        except Exception:
            self.ATLAS.show_persona_message("error", "Failed to export persona bundle.")
            return

        if not response.get("success"):
            self.ATLAS.show_persona_message("error", response.get("error") or "Persona export failed.")
            return

        bundle_bytes = response.get("bundle_bytes")
        if not isinstance(bundle_bytes, (bytes, bytearray)):
            self.ATLAS.show_persona_message("error", "Persona export did not return bundle data.")
            return

        try:
            Path(path).write_bytes(bundle_bytes)
        except OSError:
            self.ATLAS.show_persona_message("error", f"Failed to write bundle to {path}.")
            return

        warnings = response.get("warnings") or []
        message = f"Exported persona '{persona_name}' to {path}."
        if warnings:
            message += " " + " ".join(warnings)
        self.ATLAS.show_persona_message("system", message)

    def _on_import_persona_clicked(self, _button):
        path = self._choose_file_path(
            title="Import Persona Bundle",
            action="OPEN",
        )
        if not path:
            return

        signing_key = self._prompt_signing_key("Enter signing key for import")
        if not signing_key:
            self.ATLAS.show_persona_message("warning", "Import cancelled: signing key required.")
            return

        try:
            bundle_bytes = Path(path).read_bytes()
        except OSError:
            self.ATLAS.show_persona_message("error", f"Failed to read bundle from {path}.")
            return

        try:
            response = self.ATLAS.import_persona_bundle(
                bundle_bytes=bundle_bytes,
                signing_key=signing_key,
                rationale="Imported via GTK UI",
            )
        except Exception:
            self.ATLAS.show_persona_message("error", "Failed to import persona bundle.")
            return

        if not response.get("success"):
            self.ATLAS.show_persona_message("error", response.get("error") or "Persona import failed.")
            return

        persona = response.get("persona") or {}
        persona_name = persona.get("name") or "persona"
        warnings = response.get("warnings") or []
        message = f"Imported persona '{persona_name}' from {path}."
        if warnings:
            message += " " + " ".join(warnings)
        self.ATLAS.show_persona_message("system", message)

        self._refresh_persona_catalog()
        if self.persona_window:
            try:
                self.persona_window.close()
            except Exception:
                pass
        self.show_persona_menu()

    # --------------------------- Menu ---------------------------

    def _build_persona_list_widget(self) -> Gtk.Widget:
        """Create the persona selection list suitable for embedding."""
        scroll = Gtk.ScrolledWindow()
        scroll.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        scroll.set_hexpand(True)
        scroll.set_vexpand(True)

        outer = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        outer.set_margin_top(10)
        outer.set_margin_bottom(10)
        outer.set_margin_start(10)
        outer.set_margin_end(10)
        outer.set_valign(Gtk.Align.START)
        scroll.set_child(outer)

        list_box = Gtk.ListBox()
        list_box.set_selection_mode(Gtk.SelectionMode.NONE)
        list_box.set_tooltip_text("Click a row to select the persona; use the gear to edit its settings.")
        list_box.set_valign(Gtk.Align.START)
        outer.append(list_box)

        persona_names = self.ATLAS.get_persona_names() or []
        self._load_persona_metadata(persona_names)

        for persona_name in persona_names:
            row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)

            select_btn = Gtk.Button()
            select_btn.add_css_class("flat")
            select_btn.add_css_class("sidebar")
            select_btn.set_can_focus(True)
            select_btn.set_hexpand(True)
            select_btn.set_halign(Gtk.Align.FILL)
            select_btn.set_tooltip_text(f"Select persona: {persona_name}")

            label_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
            label_box.set_hexpand(True)

            name_lbl = Gtk.Label(label=persona_name)
            name_lbl.set_xalign(0.0)
            name_lbl.set_yalign(0.5)
            name_lbl.get_style_context().add_class("provider-label")
            name_lbl.set_halign(Gtk.Align.START)
            name_lbl.set_hexpand(True)
            label_box.append(name_lbl)

            if self._is_personal_persona(persona_name):
                badge = Gtk.Label(label="Personal")
                badge.set_xalign(0.0)
                badge.add_css_class("tag")
                badge.add_css_class("success")
                badge.set_tooltip_text("Curated for personal creative and productivity flows.")
                label_box.append(badge)

            select_btn.set_child(label_box)

            select_btn.connect("clicked", lambda _b, pname=persona_name: self.select_persona(pname))

            settings_btn = Gtk.Button()
            settings_btn.add_css_class("flat")
            settings_btn.set_can_focus(True)
            settings_btn.set_tooltip_text(f"Open settings for {persona_name}")
            role = getattr(Gtk.AccessibleRole, "BUTTON", None)
            if role is not None:
                settings_btn.set_accessible_role(role)

            gear = self._make_icon_widget("Icons/settings.png", fallback_icon_name="emblem-system-symbolic", size=16)
            settings_btn.set_child(gear)
            settings_btn.connect("clicked", lambda _b, pname=persona_name: self.open_persona_settings(pname))

            row.append(select_btn)
            row.append(settings_btn)

            lrow = Gtk.ListBoxRow()
            lrow.set_child(row)
            list_box.append(lrow)

        hint = Gtk.Label(label="Tip: double-click a row to select.")
        hint.set_tooltip_text("You can also use arrow keys to move and Enter to activate.")
        hint.set_margin_top(6)
        outer.append(hint)

        import_btn = Gtk.Button(label="Import persona…")
        import_btn.set_tooltip_text("Import a persona bundle from disk.")
        import_btn.connect("clicked", self._on_import_persona_clicked)
        outer.append(import_btn)

        def on_row_activated(_lb, lbrow):
            box = lbrow.get_child()
            if isinstance(box, Gtk.Box):
                child = box.get_first_child()
                if isinstance(child, Gtk.Button):
                    child.emit("clicked")

        list_box.connect("row-activated", on_row_activated)

        return scroll

    def get_embeddable_widget(self) -> Gtk.Widget:
        if self._persona_page is None:
            self._persona_page = self._build_persona_list_widget()
        return self._persona_page

    def show_persona_menu(self):
        """
        Displays the "Select Persona" window. This window lists all available
        personas with a label and a settings icon next to each name. Styling is
        provided via :class:`AtlasWindow` so the picker matches the chat UI.
        """
        self.persona_window = AtlasWindow(
            title="Select Persona",
            default_size=(220, 600),
            modal=True,
            transient_for=self.parent_window,
        )
        self.persona_window.set_tooltip_text("Choose a persona or open its settings.")

        # Container inside a scrolled window so long persona lists remain usable
        self.persona_window.set_child(self._build_persona_list_widget())
        self.persona_window.present()

    def select_persona(self, persona: str) -> None:
        """
        Loads the specified persona.

        Args:
            persona (str): The name of the persona to select.
        """
        self.ATLAS.load_persona(persona)
        prompt = self.ATLAS.get_current_persona_prompt()
        print(f"Persona '{persona}' selected with system prompt:\n{prompt}")

    # --------------------------- Settings ---------------------------

    def open_persona_settings(self, persona_name):
        """
        Opens the persona settings window for the specified persona.

        Args:
            persona_name (str): The name of the persona.
        """
        if self.persona_window:
            self.persona_window.close()

        requested_persona = persona_name
        state = self.ATLAS.get_persona_editor_state(requested_persona)
        if state is None:
            self.ATLAS.show_persona_message(
                "error",
                f"Unable to load persona '{requested_persona}' for editing.",
            )
            return

        general_state = state.get('general', {})
        display_name = general_state.get('name') or state.get('original_name') or "Persona"

        if self.settings_window:
            self.settings_window.close()

        self.settings_window = AtlasWindow(
            title=f"Settings for {display_name}",
            default_size=(560, 820),
            modal=True,
            transient_for=self.parent_window,
        )
        self.settings_window.connect("destroy", self._on_settings_window_destroyed)
        self._editor_persona_name = state.get('original_name') or requested_persona
        self.show_persona_settings(state, self.settings_window)

    def show_persona_settings(self, persona_state, settings_window):
        """Populate and display the persona settings window."""
        self._current_editor_state = persona_state
        self.tool_rows = {}
        self._tool_order = []
        self.tool_list_box = None
        self.skill_rows = {}
        self._skill_order = []
        self.skill_list_box = None
        self._settings_stack = None
        settings_window.set_tooltip_text("Configure the persona's details, provider/model, and speech options.")

        # Create a vertical box container for the settings.
        main_vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        main_vbox.set_margin_top(10)
        main_vbox.set_margin_bottom(10)
        main_vbox.set_margin_start(10)
        main_vbox.set_margin_end(10)
        settings_window.set_child(main_vbox)

        self._review_persona_name = (
            persona_state.get('original_name')
            or persona_state.get('general', {}).get('name')
            or ""
        )

        review_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        review_box.set_margin_bottom(4)
        review_box.set_tooltip_text("Persona review policy reminders and actions.")

        review_label = Gtk.Label()
        review_label.set_wrap(True)
        review_label.set_xalign(0.0)
        review_box.append(review_label)

        review_button = Gtk.Button(label="Mark Review Complete")
        review_button.set_tooltip_text("Record that this persona has been reviewed.")
        review_button.connect("clicked", self._on_mark_review_complete)
        review_box.append(review_button)

        main_vbox.append(review_box)

        self._review_banner_box = review_box
        self._review_banner_label = review_label
        self._review_mark_complete_button = review_button
        self._review_status = None
        self._set_widget_visible(review_box, False)

        if self._review_persona_name:
            self._refresh_review_status(self._review_persona_name)

        # Create a Gtk.Stack and Gtk.StackSwitcher for tabbed settings.
        stack = Gtk.Stack()
        stack.set_transition_type(Gtk.StackTransitionType.SLIDE_LEFT_RIGHT)
        stack.set_transition_duration(200)
        # GTK4 quirk: Stack uses hhomogeneous / vhomogeneous (not 'homogeneous')
        if hasattr(stack, "set_hhomogeneous"):
            stack.set_hhomogeneous(True)
        if hasattr(stack, "set_vhomogeneous"):
            stack.set_vhomogeneous(True)

        stack_switcher = Gtk.StackSwitcher()
        stack_switcher.set_stack(stack)
        stack_switcher.set_tooltip_text("Switch between settings tabs.")
        main_vbox.append(stack_switcher)
        main_vbox.append(stack)
        self._settings_stack = stack

        # General Tab (with scrollable box)
        self.general_tab = GeneralTab(persona_state, self.ATLAS)
        general_box = self.general_tab.get_widget()
        scrolled_general_tab = Gtk.ScrolledWindow()
        scrolled_general_tab.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        scrolled_general_tab.set_child(general_box)
        stack.add_titled(scrolled_general_tab, "general", "General")
        scrolled_general_tab.set_tooltip_text("Edit persona name, meaning, and prompt content.")

        # Persona Type Tab (with scrollable box)
        self.persona_type_tab = PersonaTypeTab(persona_state, self.general_tab)
        type_box = self.persona_type_tab.get_widget()
        scrolled_persona_type = Gtk.ScrolledWindow()
        scrolled_persona_type.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        scrolled_persona_type.set_child(type_box)
        scrolled_persona_type.set_tooltip_text("Enable persona roles and configure their specific options.")
        stack.add_titled(scrolled_persona_type, "persona_type", "Persona Type")

        # Provider and Model Tab
        provider_model_box = self.create_provider_model_tab(persona_state)
        provider_model_box.set_tooltip_text("Select which provider/model this persona should use.")
        stack.add_titled(provider_model_box, "provider_model", "Provider & Model")

        # Speech Provider and Voice Tab
        speech_voice_box = self.create_speech_voice_tab(persona_state)
        speech_voice_box.set_tooltip_text("Select speech provider and voice defaults for this persona.")
        stack.add_titled(speech_voice_box, "speech_voice", "Speech & Voice")

        tools_box = self.create_tools_tab(persona_state)
        stack.add_titled(tools_box, "tools", "Tools")

        skills_box = self.create_skills_tab(persona_state)
        stack.add_titled(skills_box, "skills", "Skills")

        analytics_box = self.create_analytics_tab(persona_state)
        analytics_box.set_tooltip_text("Inspect tool usage analytics for this persona.")
        stack.add_titled(analytics_box, "analytics", "Analytics")

        history_box = self.create_history_tab(persona_state)
        history_box.set_tooltip_text("Review persona change history for audit purposes.")
        stack.add_titled(history_box, "history", "History")

        # Save Button at the bottom
        actions_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        export_button = Gtk.Button(label="Export…")
        export_button.set_tooltip_text("Export this persona to a signed bundle.")
        export_button.connect("clicked", self._on_export_persona_clicked)
        actions_box.append(export_button)

        save_button = Gtk.Button(label="Save")
        save_button.set_tooltip_text("Save all changes to this persona.")
        save_button.connect("clicked", lambda _widget: self.save_persona_settings(settings_window))
        actions_box.append(save_button)

        main_vbox.append(actions_box)

        settings_window.present()

    def _on_settings_window_destroyed(self, *_args) -> None:
        self.settings_window = None
        self._settings_stack = None
        self._editor_persona_name = None
        self._current_editor_state = None
        self.tool_rows = {}
        self.tool_list_box = None
        self.skill_rows = {}
        self.skill_list_box = None

    def ensure_persona_settings(self, persona_name: str) -> bool:
        if not persona_name:
            return False

        if self.settings_window is None or self._editor_persona_name != persona_name:
            self.open_persona_settings(persona_name)

        window = self.settings_window
        present = getattr(window, "present", None)
        if callable(present):
            try:
                present()
            except Exception:
                pass

        return self.settings_window is not None

    def focus_tools_tab(self, tool_name: Optional[str] = None) -> bool:
        if not self._ensure_stack_visible("tools"):
            return False

        if not tool_name:
            return True

        info = self.tool_rows.get(tool_name)
        if not info:
            return False

        self._focus_editor_row(self.tool_list_box, info)
        return True

    def focus_skills_tab(self, skill_name: Optional[str] = None) -> bool:
        if not self._ensure_stack_visible("skills"):
            return False

        if not skill_name:
            return True

        info = self.skill_rows.get(skill_name)
        if not info:
            return False

        self._focus_editor_row(self.skill_list_box, info)
        return True

    def _ensure_stack_visible(self, child_name: str) -> bool:
        stack = self._settings_stack
        if not isinstance(stack, Gtk.Stack):
            return False

        setter = getattr(stack, "set_visible_child_name", None)
        if not callable(setter):
            return False

        try:
            setter(child_name)
        except Exception:
            return False

        window = self.settings_window
        present = getattr(window, "present", None)
        if callable(present):
            try:
                present()
            except Exception:
                pass
        return True

    def _safe_append(self, container: Any, child: Any) -> None:
        """Safely append child to container, handling GTK4/Partial Stub issues."""
        if container is None or child is None:
            return
        
        appender = getattr(container, "append", None)
        if callable(appender):
            try:
                appender(child)
                return
            except Exception:
                pass

    def _safe_add_css_class(self, widget: Any, class_name: str) -> None:
        """Safely add CSS class to widget."""
        if widget is None or not class_name:
            return
        
        adder = getattr(widget, "add_css_class", None)
        if callable(adder):
            try:
                adder(class_name)
            except Exception:
                pass
        else:
            # Fallback for GTK3 style contexts if stubs/runtime allow
            style_ctx = getattr(widget, "get_style_context", None)
            if callable(style_ctx):
                try:
                    ctx = style_ctx()
                    add_cls = getattr(ctx, "add_class", None)
                    if callable(add_cls):
                        add_cls(class_name)
                except Exception:
                    pass

    def _focus_editor_row(self, list_box: Optional[Gtk.ListBox], info: Dict[str, Any]) -> None:
        if not isinstance(list_box, Gtk.ListBox):
            return

        row = info.get('row')
        if isinstance(row, Gtk.ListBoxRow):
            # Safe selection
            select_func = getattr(list_box, 'select_row', None)
            if callable(select_func):
                try:
                    select_func(row)
                except Exception:
                    pass

        focus_widget = info.get('check')
        # Try checking widget first
        if isinstance(focus_widget, Gtk.Widget):
            try:
                grab_focus = getattr(focus_widget, 'grab_focus', None)
                if callable(grab_focus):
                    grab_focus()
                return
            except Exception:
                pass

        # Fallback to row focus
        if isinstance(row, Gtk.Widget):
            try:
                grab_focus = getattr(row, 'grab_focus', None)
                if callable(grab_focus):
                    grab_focus()
            except Exception:
                pass

    def _open_tool_management(self, tool_name: str) -> None:
        opener = getattr(self.parent_window, "show_tools_menu", None)
        if callable(opener):
            try:
                opener(tool_name=tool_name)
            except Exception:
                pass

    def _open_skill_management(self, skill_name: str) -> None:
        opener = getattr(self.parent_window, "show_skills_menu", None)
        if callable(opener):
            try:
                opener(skill_name=skill_name)
            except Exception:
                pass

    def create_provider_model_tab(self, persona_state):
        """
        Creates the Provider and Model settings tab.

        Args:
            persona_state (dict): The persona data.

        Returns:
            Gtk.Box: The container with provider and model settings.
        """
        provider_model_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)

        # Provider
        provider_label = Gtk.Label(label="Provider")
        provider_label.set_xalign(0.0)
        provider_label.set_tooltip_text("LLM provider key, e.g., 'OpenAI', 'Anthropic', 'HuggingFace', etc.")
        self.provider_entry = Gtk.Entry()
        # Cast for static analysis
        p_entry = cast(Gtk.Entry, self.provider_entry)
        if hasattr(p_entry, "set_placeholder_text"):
            getattr(p_entry, "set_placeholder_text")("e.g., OpenAI")
            
        provider_defaults = persona_state.get('provider', {})
        self._safe_set_text(self.provider_entry, provider_defaults.get("provider", "openai"))
        
        if hasattr(p_entry, "set_tooltip_text"):
            getattr(p_entry, "set_tooltip_text")("Set which backend/provider this persona uses by default.")
            
        provider_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        self._safe_append(provider_box, provider_label)
        self._safe_append(provider_box, self.provider_entry)
        self._safe_append(provider_model_box, provider_box)

        # Model
        model_label = Gtk.Label(label="Model")
        model_label.set_xalign(0.0)
        model_label.set_tooltip_text("Model identifier, e.g., 'gpt-4o', 'claude-3-opus', 'meta-llama-3-70b'.")
        self.model_entry = Gtk.Entry()
        m_entry = cast(Gtk.Entry, self.model_entry)
        
        if hasattr(m_entry, "set_placeholder_text"):
            getattr(m_entry, "set_placeholder_text")("e.g., gpt-4o")
            
        self._safe_set_text(self.model_entry, provider_defaults.get("model", "gpt-4"))
        
        if hasattr(m_entry, "set_tooltip_text"):
            getattr(m_entry, "set_tooltip_text")("Exact model name to request from the provider.")
            
        model_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        self._safe_append(model_box, model_label)
        self._safe_append(model_box, self.model_entry)
        self._safe_append(provider_model_box, model_box)

        return provider_model_box

    def create_speech_voice_tab(self, persona_state):
        """
        Creates the Speech Provider and Voice settings tab.

        Args:
            persona_state (dict): The persona data.

        Returns:
            Gtk.Box: The container with speech and voice settings.
        """
        speech_voice_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)

        # Speech Provider
        speech_provider_label = Gtk.Label(label="Speech Provider")
        speech_provider_label.set_xalign(0.0)
        speech_provider_label.set_tooltip_text("TTS/STT provider key, e.g., '11labs', 'openai_tts', 'google_tts'.")
        self.speech_provider_entry = Gtk.Entry()
        sp_entry = cast(Gtk.Entry, self.speech_provider_entry)
        if hasattr(sp_entry, "set_placeholder_text"):
            getattr(sp_entry, "set_placeholder_text")("e.g., 11labs")
            
        speech_defaults = persona_state.get('speech', {})
        self._safe_set_text(self.speech_provider_entry, speech_defaults.get("Speech_provider", "11labs"))
        
        if hasattr(sp_entry, "set_tooltip_text"):
            getattr(sp_entry, "set_tooltip_text")("Default speech provider for this persona.")
            
        speech_provider_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        self._safe_append(speech_provider_box, speech_provider_label)
        self._safe_append(speech_provider_box, self.speech_provider_entry)
        self._safe_append(speech_voice_box, speech_provider_box)

        # Voice
        voice_label = Gtk.Label(label="Voice")
        voice_label.set_xalign(0.0)
        voice_label.set_tooltip_text("Voice identifier (depends on the provider).")
        self.voice_entry = Gtk.Entry()
        v_entry = cast(Gtk.Entry, self.voice_entry)
        
        if hasattr(v_entry, "set_placeholder_text"):
            getattr(v_entry, "set_placeholder_text")("e.g., Jack")
            
        self._safe_set_text(self.voice_entry, speech_defaults.get("voice", "jack"))
        
        if hasattr(v_entry, "set_tooltip_text"):
            getattr(v_entry, "set_tooltip_text")("Default voice to synthesize for this persona.")
            
        voice_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        self._safe_append(voice_box, voice_label)
        self._safe_append(voice_box, self.voice_entry)
        self._safe_append(speech_voice_box, voice_box)

        return speech_voice_box

    def _format_timestamp(self, iso_timestamp: str) -> str:
        if not iso_timestamp:
            return "Unknown time"

        text = str(iso_timestamp)
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"

        try:
            moment = datetime.fromisoformat(text)
        except ValueError:
            return iso_timestamp

        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        else:
            moment = moment.astimezone(timezone.utc)

        return moment.strftime("%Y-%m-%d %H:%M UTC")

    def _format_tool_list(self, tools: List[str]) -> str:
        if not tools:
            return "none"
        return ", ".join(tools)

    def _format_tool_change(self, old_tools: List[str], new_tools: List[str]) -> str:
        return f"Tools: {self._format_tool_list(old_tools)} → {self._format_tool_list(new_tools)}"

    def _format_rationale(self, rationale: str) -> str:
        text = (rationale or "").strip()
        if not text:
            return "Rationale: Not provided."
        return f"Rationale: {text}"

    def _format_history_entry(self, entry: Dict[str, Any]) -> str:
        timestamp = self._format_timestamp(entry.get("timestamp", ""))
        username = str(entry.get("username") or "unknown")
        return f"{timestamp} — {username}"

    def _build_history_row(self, entry: Dict[str, Any]) -> Gtk.Widget:
        container = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)

        summary_label = Gtk.Label(label=self._format_history_entry(entry))
        summary_label.set_xalign(0.0)
        self._safe_append(container, summary_label)

        old_tools = entry.get("old_tools") or []
        new_tools = entry.get("new_tools") or []
        tools_label = Gtk.Label(
            label=self._format_tool_change(
                [str(tool) for tool in old_tools],
                [str(tool) for tool in new_tools],
            )
        )
        tools_label.set_xalign(0.0)
        self._safe_append(container, tools_label)

        rationale_label = Gtk.Label(
            label=self._format_rationale(str(entry.get("rationale") or ""))
        )
        rationale_label.set_xalign(0.0)
        self._safe_append(container, rationale_label)

        row = Gtk.ListBoxRow()
        setter = getattr(row, "set_child", None)
        if callable(setter):
            setter(container)
        else:  # pragma: no cover - GTK stubs without set_child
            row.children = [container]
        return row

    def _load_history_page(self, *, reset: bool = False) -> None:
        persona_name = self._history_persona_name
        list_box = self.history_list_box
        if not persona_name or list_box is None:
            return

        if reset:
            existing_children = list(getattr(list_box, "children", []))
            for child in existing_children:
                try:
                    list_box.remove(child)
                except Exception:  # pragma: no cover - stub fallback
                    continue
            self._history_offset = 0
            self._history_total = 0
            self._history_placeholder = None

        try:
            entries, total = self.ATLAS.get_persona_audit_history(
                persona_name,
                offset=self._history_offset,
                limit=self._history_page_size,
            )
        except Exception:
            self.ATLAS.show_persona_message(
                "error",
                "Unable to load persona change history.",
            )
            return

        self._history_total = total

        if reset and not entries:
            placeholder = Gtk.Label(label="No persona changes recorded yet.")
            placeholder.set_xalign(0.0)
            self._safe_append(list_box, placeholder)
            self._history_placeholder = placeholder
        else:
            if self._history_placeholder is not None:
                try:
                    list_box.remove(self._history_placeholder)
                except Exception:  # pragma: no cover - stub fallback
                    pass
                self._history_placeholder = None

            for entry in entries:
                row = self._build_history_row(entry)
                self._safe_append(list_box, row)

            self._history_offset += len(entries)

        has_more = self._history_offset < self._history_total
        if self._history_load_more_button is not None:
            self._history_load_more_button.set_visible(has_more)
            self._history_load_more_button.set_sensitive(has_more)

    def create_history_tab(self, persona_state) -> Gtk.Box:
        persona_name = ""
        if isinstance(persona_state, dict):
            general = persona_state.get('general')
            if isinstance(general, dict):
                persona_name = str(general.get('name') or "")
            if not persona_name:
                persona_name = str(persona_state.get('original_name') or "")

        history_container = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        history_container.set_margin_top(10)
        history_container.set_margin_bottom(10)
        history_container.set_margin_start(10)
        history_container.set_margin_end(10)

        scroll = Gtk.ScrolledWindow()
        scroll.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        scroll.set_hexpand(True)
        scroll.set_vexpand(True)
        self._safe_append(history_container, scroll)

        self.history_list_box = Gtk.ListBox()
        hist_box = cast(Gtk.ListBox, self.history_list_box)
        if hasattr(hist_box, "set_selection_mode"):
             getattr(hist_box, "set_selection_mode")(Gtk.SelectionMode.NONE)
        scroll.set_child(self.history_list_box)

        controls = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        controls.set_valign(Gtk.Align.START)
        load_more_btn = Gtk.Button(label="Load More")
        load_more_btn.set_tooltip_text("Load older persona change events.")
        load_more_btn.connect("clicked", lambda _btn: self._load_history_page())
        self._safe_append(controls, load_more_btn)
        self._safe_append(history_container, controls)

        self._history_persona_name = persona_name
        self._history_offset = 0
        self._history_total = 0
        self._history_page_size = 20
        self._history_load_more_button = load_more_btn
        self._history_placeholder = None

        self._load_history_page(reset=True)

        return history_container

    def _format_tool_hint(self, metadata: Dict[str, Any]) -> str:
        safety = str(metadata.get('safety_level') or 'unspecified').strip()
        if safety:
            safety = safety.capitalize()
        cost_per_call = metadata.get('cost_per_call')
        cost_unit = str(metadata.get('cost_unit') or '').strip()
        if cost_per_call is None:
            cost_text = "n/a"
        else:
            try:
                cost_value = float(cost_per_call)
                if cost_value.is_integer():
                    cost_text = f"{int(cost_value)}"
                else:
                    cost_text = f"{cost_value:.3f}".rstrip('0').rstrip('.')
            except (TypeError, ValueError):
                cost_text = str(cost_per_call)
        if cost_unit:
            cost_display = f"{cost_text} {cost_unit}/call"
        else:
            cost_display = f"{cost_text}/call"
        return f"Safety: {safety or 'Unspecified'} • Cost: {cost_display}"

    def _format_skill_hint(self, metadata: Dict[str, Any]) -> str:
        persona_owner = str(metadata.get('persona') or '').strip()
        if persona_owner:
            persona_display = f"Persona: {persona_owner}"
        else:
            persona_display = "Persona: Shared"

        required_tools = metadata.get('required_tools') or []
        if isinstance(required_tools, list):
            tools_display = ", ".join(str(tool) for tool in required_tools if str(tool))
        else:
            tools_display = str(required_tools)
        tools_hint = f"Requires tools: {tools_display}" if tools_display else "Requires tools: none"

        required_capabilities = metadata.get('required_capabilities') or []
        if isinstance(required_capabilities, list):
            caps_display = ", ".join(str(cap) for cap in required_capabilities if str(cap))
        else:
            caps_display = str(required_capabilities)
        caps_hint = f"Capabilities: {caps_display}" if caps_display else "Capabilities: none"

        category = str(metadata.get('category') or '').strip()
        if category:
            category_hint = f"Category: {category}"
        else:
            category_hint = ""

        summary = str(metadata.get('summary') or '').strip()

        capability_tags = metadata.get('capability_tags') or []
        if isinstance(capability_tags, list):
            tags_display = ", ".join(str(tag) for tag in capability_tags if str(tag))
        else:
            tags_display = str(capability_tags)
        tags_hint = f"Tags: {tags_display}" if tags_display else ""

        notes = str(metadata.get('safety_notes') or '').strip()

        hints = []
        if summary:
            hints.append(summary)
        hints.append(persona_display)
        if category_hint:
            hints.append(category_hint)
        hints.append(tools_hint)
        hints.append(caps_hint)
        if tags_hint:
            hints.append(tags_hint)
        if notes:
            hints.append(notes)

        return " • ".join(hints)

    def _ensure_tool_row_icons(self, icon_name: str) -> Gtk.Widget:
        try:
            image = Gtk.Image.new_from_icon_name(icon_name)
            image.set_pixel_size(14)
            return image
        except Exception:
            return Gtk.Image.new()

    def _safe_add_controller(self, widget: Optional[Gtk.Widget], controller: Any) -> None:
        if not isinstance(widget, Gtk.Widget) or controller is None:
            return
        adder = getattr(widget, "add_controller", None)
        if callable(adder):
            try:
                adder(controller)
            except Exception:
                return

    def _create_drag_content_provider(self, item_name: str):
        content_provider = getattr(Gdk, "ContentProvider", None)
        if content_provider is None:
            return None

        factory = getattr(content_provider, "new_for_value", None)
        if callable(factory):
            try:
                return factory(item_name)
            except Exception:
                return None
        return None

    def _init_reorder_row_interactions(
        self,
        *,
        name: str,
        row: Optional[Gtk.Widget],
        focus_widget: Optional[Gtk.Widget],
        order: List[str],
        move_callback: Callable[[str, int], None],
        info: Dict[str, Any],
    ) -> None:
        drag_source_cls = getattr(Gtk, "DragSource", None)
        drop_target_cls = getattr(Gtk, "DropTarget", None)
        drag_action_enum = getattr(Gdk, "DragAction", None)
        drag_action_move = getattr(drag_action_enum, "MOVE", None) if drag_action_enum else None

        if (
            isinstance(row, Gtk.Widget)
            and drag_source_cls is not None
            and drop_target_cls is not None
            and drag_action_move is not None
        ):
            try:
                drag_source = drag_source_cls()
            except Exception:
                drag_source = None
            if drag_source is not None:
                setter = getattr(drag_source, "set_actions", None)
                if callable(setter):
                    try:
                        setter(drag_action_move)
                    except Exception:
                        pass
                try:
                    drag_source.connect(
                        "prepare",
                        lambda _source, _x, _y, item=name: self._create_drag_content_provider(item),
                    )
                except Exception:
                    pass
                self._safe_add_controller(row, drag_source)

            drop_target = None
            try:
                drop_target = drop_target_cls.new(str, drag_action_move)
            except Exception:
                try:
                    drop_target = drop_target_cls()
                except Exception:
                    drop_target = None
                else:
                    action_setter = getattr(drop_target, "set_actions", None)
                    if callable(action_setter):
                        try:
                            action_setter(drag_action_move)
                        except Exception:
                            pass
                    gtype_setter = getattr(drop_target, "set_gtypes", None)
                    if callable(gtype_setter):
                        try:
                            gtype_setter([str])
                        except Exception:
                            pass

            if drop_target is not None:
                try:
                    drop_target.connect(
                        "drop",
                        lambda _target, value, _x, y, item=name, item_row=row: self._handle_reorder_drop(
                            value,
                            item,
                            y,
                            order,
                            move_callback,
                            item_row,
                        ),
                    )
                except Exception:
                    pass
                self._safe_add_controller(row, drop_target)

        key_controller_cls = getattr(Gtk, "EventControllerKey", None)
        if isinstance(focus_widget, Gtk.Widget) and key_controller_cls is not None:
            try:
                key_controller = key_controller_cls()
            except Exception:
                key_controller = None
            if key_controller is not None:
                try:
                    key_controller.connect(
                        "key-pressed",
                        lambda _controller, keyval, _keycode, state, item=name: self._handle_reorder_key_event(
                            item_name=item,
                            order=order,
                            move_callback=move_callback,
                            keyval=keyval,
                            state=state,
                        ),
                    )
                except Exception:
                    pass
                self._safe_add_controller(focus_widget, key_controller)
                info['keyboard_controller'] = key_controller

    def _handle_reorder_drop(
        self,
        value: Any,
        target_name: str,
        y_pos: float,
        order: List[str],
        move_callback: Callable[[str, int], None],
        row: Optional[Gtk.Widget],
    ) -> bool:
        source_name: Any = value
        if hasattr(value, "get_string"):
            try:
                source_name = value.get_string()
            except Exception:
                source_name = value
        elif hasattr(value, "get_value"):
            try:
                source_name = value.get_value()
            except Exception:
                source_name = value

        if not isinstance(source_name, str):
            return False

        if source_name not in order or target_name not in order:
            return False

        try:
            target_index = order.index(target_name)
        except ValueError:
            return False

        height = 1.0
        if isinstance(row, Gtk.Widget):
            height_getter = getattr(row, "get_allocated_height", None)
            if callable(height_getter):
                try:
                    val = height_getter()
                    measured_height = float(val) if isinstance(val, (int, float)) else 0.0
                except Exception:
                    measured_height = 0.0
                if measured_height > 0.0:
                    height = measured_height

        try:
            relative_y = float(y_pos)
        except (TypeError, ValueError):
            relative_y = 0.0

        if relative_y > height / 2.0:
            target_index += 1

        move_callback(source_name, target_index)
        return True

    def _handle_reorder_key_event(
        self,
        *,
        item_name: str,
        order: List[str],
        move_callback: Callable[[str, int], None],
        keyval: int,
        state: int,
    ) -> bool:
        try:
            current_index = order.index(item_name)
        except ValueError:
            return False

        modifier_enum = getattr(Gdk, "ModifierType", None)
        ctrl_mask = getattr(modifier_enum, "CONTROL_MASK", None) if modifier_enum else None

        try:
            state_value = int(state)
        except (TypeError, ValueError):
            state_value = 0

        ctrl_active = ctrl_mask in (None, 0) or bool(state_value & ctrl_mask)
        if not ctrl_active:
            return False

        key_up = getattr(Gdk, "KEY_Up", 65362)
        key_down = getattr(Gdk, "KEY_Down", 65364)
        key_kp_up = getattr(Gdk, "KEY_KP_Up", key_up)
        key_kp_down = getattr(Gdk, "KEY_KP_Down", key_down)

        if keyval in (key_up, key_kp_up):
            move_callback(item_name, current_index - 1)
            return True
        if keyval in (key_down, key_kp_down):
            move_callback(item_name, current_index + 1)
            return True
        return False

    def create_tools_tab(self, persona_state):
        tools_state = persona_state.get('tools') or {}
        available = tools_state.get('available') or []

        sorted_entries = sorted(
            [entry for entry in available if isinstance(entry, dict)],
            key=lambda entry: entry.get('order', 0),
        )

        tools_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        tools_box.set_margin_top(10)
        tools_box.set_margin_bottom(10)
        tools_box.set_margin_start(10)
        tools_box.set_margin_end(10)
        tools_box.set_tooltip_text("Enable, disable, and reorder tools available to this persona.")

        header = Gtk.Label(label="Select which tools this persona can use.")
        header.set_xalign(0.0)
        header.set_wrap(True)
        header.set_tooltip_text(
            "Toggle tools to enable or disable them. Drag rows or use Ctrl+↑/↓ to change invocation order."
        )
        self._safe_append(tools_box, header)

        list_box = Gtk.ListBox()
        list_box.set_selection_mode(Gtk.SelectionMode.NONE)
        self._safe_add_css_class(list_box, "boxed-list")
        self.tool_list_box = list_box

        for entry in sorted_entries:
            name = entry.get('name')
            if not name:
                continue
            if name not in self._tool_order:
                self._tool_order.append(name)
            metadata = entry.get('metadata') if isinstance(entry, dict) else {}
            if not isinstance(metadata, dict):
                metadata = {}
            display_name = metadata.get('display_name') or metadata.get('title') or metadata.get('name') or name
            enabled = bool(entry.get('enabled'))
            policy_disabled = bool(entry.get('disabled'))
            reason_text = entry.get('disabled_reason')
            if isinstance(reason_text, str):
                reason_text = reason_text.strip()
            else:
                reason_text = str(reason_text).strip() if reason_text else ""

            row_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
            controls = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)

            check = Gtk.CheckButton.new_with_label(str(display_name))
            description = metadata.get('description')
            tooltip_parts: List[str] = []
            if isinstance(description, str) and description.strip():
                tooltip_parts.append(description.strip())

            if policy_disabled:
                check.set_active(False)
                check.set_sensitive(False)
                if not reason_text:
                    reason_text = f"{display_name} is disabled by policy."
                tooltip_parts.append(reason_text)
            else:
                check.set_active(enabled)
                if reason_text:
                    tooltip_parts.append(reason_text)

            tooltip_parts.append(
                "Drag to reorder. Use Ctrl+↑ or Ctrl+↓ for keyboard reordering."
            )
            if tooltip_parts:
                check.set_tooltip_text("\n\n".join(tooltip_parts))
            self._safe_append(controls, check)

            workspace_btn = Gtk.Button()
            self._safe_add_css_class(workspace_btn, "flat")
            workspace_btn.set_can_focus(True)
            workspace_btn.set_tooltip_text("Open this tool in the management workspace.")
            accessible_label = f"Open {display_name} in tool manager"
            set_accessible = getattr(workspace_btn, "set_accessible_name", None)
            if callable(set_accessible):
                set_accessible(accessible_label)
            else:
                update_property = getattr(workspace_btn, "update_property", None)
                if callable(update_property):
                    try:
                        update_property(
                            Gtk.AccessibleProperty.LABEL,
                            GLib.Variant.new_string(accessible_label),
                        )
                    except TypeError as exc:  # pragma: no cover - defensive fallback
                        logger.debug(
                            "Gtk.Button.update_property failed; skipping accessible label: %s",
                            exc,
                        )
                else:
                    logger.debug(
                        "Gtk.Button does not support accessible labels; expected name '%s'",
                        accessible_label,
                    )
            workspace_btn.set_child(self._ensure_tool_row_icons("document-open-symbolic"))
            workspace_btn.connect("clicked", lambda _b, tname=name: self._open_tool_management(tname))
            self._safe_append(controls, workspace_btn)

            badge_widget: Optional[Gtk.Widget] = None
            if reason_text:
                badge = Gtk.Label(label=reason_text)
                badge.set_xalign(0.0)
                self._safe_add_css_class(badge, "tag")
                self._safe_add_css_class(badge, "warning")
                badge.set_tooltip_text(reason_text)
                self._safe_append(controls, badge)
                badge_widget = badge

            self._safe_append(row_box, controls)

            hint = Gtk.Label(label=self._format_tool_hint(metadata))
            hint.set_xalign(0.0)
            self._safe_add_css_class(hint, "dim-label")
            self._safe_append(row_box, hint)

            list_row = Gtk.ListBoxRow()
            list_row.set_child(row_box)
            self._safe_append(list_box, list_row)

            self.tool_rows[name] = {
                'row': list_row,
                'check': check,
                'metadata': metadata,
                'badge': badge_widget,
            }

            self._init_reorder_row_interactions(
                name=name,
                row=list_row,
                focus_widget=check,
                order=self._tool_order,
                move_callback=self._move_tool_row,
                info=self.tool_rows[name],
            )

        self._refresh_tool_reorder_controls()

        scroller = Gtk.ScrolledWindow()
        scroller.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        scroller.set_child(list_box)
        self._safe_append(tools_box, scroller)

        return tools_box

    def create_skills_tab(self, persona_state):
        skills_state = persona_state.get('skills') or {}
        available = skills_state.get('available') or []

        sorted_entries = sorted(
            [entry for entry in available if isinstance(entry, dict)],
            key=lambda entry: entry.get('order', 0),
        )

        skills_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        skills_box.set_margin_top(10)
        skills_box.set_margin_bottom(10)
        skills_box.set_margin_start(10)
        skills_box.set_margin_end(10)
        skills_box.set_tooltip_text("Enable or disable skills available to this persona.")

        header = Gtk.Label(label="Select which skills this persona can access.")
        header.set_xalign(0.0)
        header.set_wrap(True)
        header.set_tooltip_text(
            "Toggle skills to include them in the persona's prompt capabilities. Drag rows or use Ctrl+↑/↓ to change order."
        )
        self._safe_append(skills_box, header)

        list_box = Gtk.ListBox()
        list_box.set_selection_mode(Gtk.SelectionMode.NONE)
        self._safe_add_css_class(list_box, "boxed-list")
        self.skill_list_box = list_box

        for entry in sorted_entries:
            name = entry.get('name')
            if not name:
                continue
            if name not in self._skill_order:
                self._skill_order.append(name)
            metadata = entry.get('metadata') if isinstance(entry, dict) else {}
            if not isinstance(metadata, dict):
                metadata = {}
            display_name = metadata.get('name') or name
            enabled = bool(entry.get('enabled'))
            policy_disabled = bool(entry.get('disabled'))
            reason_text = entry.get('disabled_reason')
            if isinstance(reason_text, str):
                reason_text = reason_text.strip()
            else:
                reason_text = str(reason_text).strip() if reason_text else ""

            row_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
            controls = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)

            check = Gtk.CheckButton.new_with_label(str(display_name))
            description = metadata.get('instruction_prompt')
            tooltip_parts: List[str] = []
            if isinstance(description, str) and description.strip():
                tooltip_parts.append(description.strip())

            if policy_disabled:
                check.set_active(False)
                check.set_sensitive(False)
                if not reason_text:
                    reason_text = f"{display_name} is disabled for this persona."
                tooltip_parts.append(reason_text)
            else:
                check.set_active(enabled)
                if reason_text:
                    tooltip_parts.append(reason_text)

            tooltip_text = ""
            tooltip_parts.append(
                "Drag to reorder. Use Ctrl+↑ or Ctrl+↓ for keyboard reordering."
            )
            if tooltip_parts:
                tooltip_text = "\n\n".join(tooltip_parts)
                check.set_tooltip_text(tooltip_text)
            self._safe_append(controls, check)

            workspace_btn = Gtk.Button()
            self._safe_add_css_class(workspace_btn, "flat")
            workspace_btn.set_can_focus(True)
            workspace_btn.set_tooltip_text("Open this skill in the management workspace.")
            accessible_label = f"Open {display_name} in skill manager"
            set_accessible = getattr(workspace_btn, "set_accessible_name", None)
            if callable(set_accessible):
                set_accessible(accessible_label)
            else:
                update_property = getattr(workspace_btn, "update_property", None)
                if callable(update_property):
                    try:
                        update_property(
                            Gtk.AccessibleProperty.LABEL,
                            GLib.Variant.new_string(accessible_label),
                        )
                    except TypeError as exc:  # pragma: no cover - defensive fallback
                        logger.debug(
                            "Gtk.Button.update_property failed; skipping accessible label: %s",
                            exc,
                        )
                else:
                    logger.debug(
                        "Gtk.Button does not support accessible labels; expected name '%s'",
                        accessible_label,
                    )
            workspace_btn.set_child(self._ensure_tool_row_icons("document-open-symbolic"))
            workspace_btn.connect("clicked", lambda _b, sname=name: self._open_skill_management(sname))
            self._safe_append(controls, workspace_btn)

            badge_widget: Optional[Gtk.Widget] = None
            if reason_text:
                badge = Gtk.Label(label=reason_text)
                badge.set_xalign(0.0)
                self._safe_add_css_class(badge, "tag")
                self._safe_add_css_class(badge, "warning")
                badge.set_tooltip_text(reason_text)
                self._safe_append(controls, badge)
                badge_widget = badge

            self._safe_append(row_box, controls)

            hint = Gtk.Label(label=self._format_skill_hint(metadata))
            hint.set_xalign(0.0)
            self._safe_add_css_class(hint, "dim-label")
            self._safe_append(row_box, hint)

            list_row = Gtk.ListBoxRow()
            list_row.set_child(row_box)
            self._safe_append(list_box, list_row)

            self.skill_rows[name] = {
                'row': list_row,
                'check': check,
                'metadata': metadata,
                'badge': badge_widget,
                'hint': hint,
                'default_hint': hint.get_text(),
                'default_tooltip': tooltip_text,
            }

            self._init_reorder_row_interactions(
                name=name,
                row=list_row,
                focus_widget=check,
                order=self._skill_order,
                move_callback=self._move_skill_row,
                info=self.skill_rows[name],
            )

        self._refresh_skill_reorder_controls()

        scroller = Gtk.ScrolledWindow()
        scroller.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        scroller.set_child(list_box)
        self._safe_append(skills_box, scroller)

        return skills_box

    def _refresh_tool_reorder_controls(self):
        total = len(self._tool_order)
        for index, name in enumerate(self._tool_order):
            info = self.tool_rows.get(name)
            if not info:
                continue
            info['can_move_up'] = index > 0
            info['can_move_down'] = index < total - 1

    def _move_tool_row(self, tool_name: str, target_index: int):
        if tool_name not in self._tool_order or not self.tool_list_box:
            return

        current_index = self._tool_order.index(tool_name)
        try:
            target_index = int(target_index)
        except (TypeError, ValueError):
            return

        target_index = max(0, min(target_index, len(self._tool_order)))
        if target_index == current_index:
            return

        self._tool_order.pop(current_index)
        if target_index > current_index:
            target_index -= 1
        target_index = max(0, min(target_index, len(self._tool_order)))
        self._tool_order.insert(target_index, tool_name)

        info = self.tool_rows.get(tool_name)
        if not info:
            return

        row = info.get('row')
        if not isinstance(row, Gtk.ListBoxRow):
            return

        self.tool_list_box.remove(row)
        self.tool_list_box.insert(row, target_index)
        if hasattr(self.tool_list_box, "invalidate_sort"):
            self.tool_list_box.invalidate_sort()
        self._refresh_tool_reorder_controls()

    def _collect_tool_payload(self) -> List[str]:
        allowed: List[str] = []
        for name in self._tool_order:
            info = self.tool_rows.get(name)
            if not info:
                continue
            check = info.get('check')
            if self._safe_get_active(check):
                allowed.append(name)
        return allowed

    def _refresh_skill_reorder_controls(self):
        total = len(self._skill_order)
        for index, name in enumerate(self._skill_order):
            info = self.skill_rows.get(name)
            if not info:
                continue
            info['can_move_up'] = index > 0
            info['can_move_down'] = index < total - 1

    def _move_skill_row(self, skill_name: str, target_index: int):
        if skill_name not in self._skill_order or not self.skill_list_box:
            return

        current_index = self._skill_order.index(skill_name)
        try:
            target_index = int(target_index)
        except (TypeError, ValueError):
            return

        target_index = max(0, min(target_index, len(self._skill_order)))
        if target_index == current_index:
            return

        self._skill_order.pop(current_index)
        if target_index > current_index:
            target_index -= 1
        target_index = max(0, min(target_index, len(self._skill_order)))
        self._skill_order.insert(target_index, skill_name)

        info = self.skill_rows.get(skill_name)
        if not info:
            return

        row = info.get('row')
        if not isinstance(row, Gtk.ListBoxRow):
            return

        self.skill_list_box.remove(row)
        self.skill_list_box.insert(row, target_index)
        if hasattr(self.skill_list_box, "invalidate_sort"):
            self.skill_list_box.invalidate_sort()
        self._refresh_skill_reorder_controls()

    def _collect_skill_payload(self) -> List[str]:
        allowed: List[str] = []
        for name in self._skill_order:
            info = self.skill_rows.get(name)
            if not info:
                continue
            check = info.get('check')
            if self._safe_get_active(check):
                allowed.append(name)
        return allowed

    def _clear_list_box(self, list_box: Optional[Gtk.ListBox]) -> None:
        if not isinstance(list_box, Gtk.ListBox):
            return
        
        # GTK4 style
        getter = getattr(list_box, "get_first_child", None)
        if callable(getter):
            child = getter()
            while child is not None:
                next_sib_func = getattr(child, "get_next_sibling", None)
                next_sib = next_sib_func() if callable(next_sib_func) else None
                try:
                    remover = getattr(list_box, "remove", None)
                    if callable(remover):
                        remover(child)
                except Exception:
                    pass
                child = next_sib
            return

        # GTK3 style fallback
        getter = getattr(list_box, "get_children", None)
        if callable(getter):
            children = getter()
            if children is not None:
                try:
                    # Specific cast to handle the dynamic return type
                    child_list = cast(List[Gtk.Widget], children)
                    remover = getattr(list_box, "remove", None)
                    if callable(remover):
                        for child in child_list:
                            try:
                                remover(child)
                            except Exception:
                                pass
                except TypeError:
                    pass

    def _parse_analytics_date_entry(
        self, entry: Optional[Gtk.Entry]
    ) -> tuple[Optional[datetime], bool]:
        if not isinstance(entry, Gtk.Entry):
            return (None, True)
        text = self._safe_get_text(entry).strip()
        if not text:
            return (None, True)
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            try:
                parsed = datetime.strptime(text, "%Y-%m-%d")
            except ValueError:
                return (None, False)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        else:
            parsed = parsed.astimezone(timezone.utc)
        return (parsed, True)

    def _parse_analytics_timestamp(self, value: Any) -> Optional[datetime]:
        if isinstance(value, datetime):
            parsed = value
        elif isinstance(value, (int, float)):
            try:
                parsed = datetime.fromtimestamp(float(value), timezone.utc)
            except (TypeError, ValueError):
                return None
        elif isinstance(value, str):
            candidate = value.strip()
            if not candidate:
                return None
            if candidate.endswith("Z"):
                candidate = candidate[:-1] + "+00:00"
            try:
                parsed = datetime.fromisoformat(candidate)
            except ValueError:
                for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
                    try:
                        parsed = datetime.strptime(candidate, fmt)
                        break
                    except ValueError:
                        continue
                else:
                    return None
        else:
            return None

        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        else:
            parsed = parsed.astimezone(timezone.utc)
        return parsed

    def _select_anomaly_by_timestamp(
        self, timestamp: datetime, tolerance_seconds: float = 3600.0
    ) -> bool:
        if not isinstance(self._analytics_anomaly_list, Gtk.ListBox):
            return False
        target = timestamp.astimezone(timezone.utc)
        best_row: Optional[Gtk.ListBoxRow] = None
        best_delta: Optional[float] = None
        for row, entry in self._analytics_anomaly_rows.items():
            entry_ts = self._parse_analytics_timestamp(entry.get("timestamp"))
            if entry_ts is None:
                continue
            delta = abs((entry_ts - target).total_seconds())
            if best_delta is None or delta < best_delta:
                best_delta = delta
                best_row = row
        if best_row is not None and best_delta is not None and best_delta <= tolerance_seconds:
            try:
                select_func = getattr(self._analytics_anomaly_list, "select_row", None)
                if callable(select_func):
                    select_func(best_row)
            except Exception:
                return False
            return True
        return False

    def _on_anomaly_row_selected(
        self, _list_box: Gtk.ListBox, row: Optional[Gtk.ListBoxRow]
    ) -> None:
        if row is None:
            if self._analytics_anomaly_heatmap is not None:
                self._analytics_anomaly_heatmap.highlight_date(None)
            if self._analytics_latency_chart is not None:
                self._analytics_latency_chart.highlight_timestamp(None)
            return
        entry = self._analytics_anomaly_rows.get(row)
        if not entry:
            return
        timestamp = self._parse_analytics_timestamp(entry.get("timestamp"))
        if timestamp is None:
            if self._analytics_anomaly_heatmap is not None:
                self._analytics_anomaly_heatmap.highlight_date(None)
            if self._analytics_latency_chart is not None:
                self._analytics_latency_chart.highlight_timestamp(None)
            return
        if self._analytics_anomaly_heatmap is not None:
            self._analytics_anomaly_heatmap.highlight_date(timestamp)
        if self._analytics_latency_chart is not None:
            self._analytics_latency_chart.highlight_timestamp(timestamp)

    def _on_heatmap_day_selected(self, timestamp: datetime) -> None:
        if not isinstance(self._analytics_anomaly_list, Gtk.ListBox):
            return
        target_date = timestamp.date()
        for row, entry in self._analytics_anomaly_rows.items():
            entry_ts = self._parse_analytics_timestamp(entry.get("timestamp"))
            if entry_ts is not None and entry_ts.date() == target_date:
                try:
                    select_func = getattr(self._analytics_anomaly_list, "select_row", None)
                    if callable(select_func):
                        select_func(row)
                except Exception:
                    pass
                return
        if self._analytics_anomaly_heatmap is not None:
            self._analytics_anomaly_heatmap.highlight_date(timestamp)
        if self._analytics_latency_chart is not None:
            self._analytics_latency_chart.highlight_timestamp(None)

    def _on_timeline_point_selected(self, timestamp: datetime) -> None:
        if self._select_anomaly_by_timestamp(timestamp):
            return
        if self._analytics_latency_chart is not None:
            self._analytics_latency_chart.highlight_timestamp(timestamp)
        if self._analytics_anomaly_heatmap is not None:
            self._analytics_anomaly_heatmap.highlight_date(timestamp)

    def _format_analytics_timestamp(self, value: Optional[str]) -> str:
        if not value:
            return "Unknown"
        try:
            display = value
            if value.endswith("Z"):
                display = value[:-1] + "+00:00"
            parsed = datetime.fromisoformat(display)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            else:
                parsed = parsed.astimezone(timezone.utc)
            return parsed.strftime("%Y-%m-%d %H:%M:%S UTC")
        except ValueError:
            return value

    def _update_comparison_highlights(
        self,
        summary: Optional[Dict[str, Any]],
        persona_name: Optional[str],
    ) -> None:
        """Update comparison highlight labels based on ranking payload."""

        persona_token = str(persona_name or "").strip().lower()

        def _set_label(key: str, text: str) -> None:
            widget = self._analytics_comparison_labels.get(key)
            self._safe_set_text(widget, text)

        _set_label("top", "Top Performer: —")
        _set_label("worst", "Highest Failure Rate: —")
        _set_label("fastest", "Fastest Avg Latency: —")

        if not isinstance(summary, dict):
            return

        rankings = summary.get("rankings")
        if not isinstance(rankings, dict):
            return

        def _first_entry(entries: Any) -> Optional[Dict[str, Any]]:
            if not isinstance(entries, list):
                return None
            for item in entries:
                if isinstance(item, dict) and item.get("persona"):
                    return item
            return None

        def _highlight_suffix(candidate: Optional[str]) -> str:
            if not persona_token or candidate is None:
                return ""
            candidate_token = str(candidate).strip().lower()
            if candidate_token and candidate_token == persona_token:
                return " • Current persona"
            return ""

        top_entry = _first_entry(rankings.get("top_performers"))
        if top_entry:
            name = str(top_entry.get("persona") or "Unknown")
            success_rate = float(top_entry.get("success_rate") or 0.0) * 100
            calls = int(top_entry.get("calls") or 0)
            suffix = _highlight_suffix(name)
            _set_label(
                "top",
                f"Top Performer: {name} ({success_rate:.1f}% success across {calls} calls){suffix}",
            )

        worst_entry = _first_entry(rankings.get("worst_failure_rates"))
        if worst_entry:
            name = str(worst_entry.get("persona") or "Unknown")
            failure_rate = float(worst_entry.get("failure_rate") or 0.0) * 100
            calls = int(worst_entry.get("calls") or 0)
            suffix = _highlight_suffix(name)
            _set_label(
                "worst",
                f"Highest Failure Rate: {name} ({failure_rate:.1f}% across {calls} calls){suffix}",
            )

        fastest_entry = _first_entry(rankings.get("fastest_latency"))
        if fastest_entry:
            name = str(fastest_entry.get("persona") or "Unknown")
            latency = float(fastest_entry.get("average_latency_ms") or 0.0)
            suffix = _highlight_suffix(name)
            _set_label(
                "fastest",
                f"Fastest Avg Latency: {name} ({latency:.1f} ms){suffix}",
            )

    def _refresh_persona_analytics(self, *, show_errors: bool = True) -> None:
        persona_name = self._analytics_persona_name
        if not persona_name:
            return

        start_value, start_valid = self._parse_analytics_date_entry(
            self._analytics_start_entry
        )
        end_value, end_valid = self._parse_analytics_date_entry(
            self._analytics_end_entry
        )

        if not start_valid or not end_valid:
            if show_errors:
                self.ATLAS.show_persona_message(
                    "warning",
                    "Enter dates as YYYY-MM-DD or ISO 8601 timestamps.",
                )
            return

        try:
            metrics = self.ATLAS.get_persona_metrics(
                persona_name,
                start=start_value,
                end=end_value,
                limit=25,
                metric_type="tool",
            )
        except Exception:
            if show_errors:
                self.ATLAS.show_persona_message(
                    "error",
                    "Unable to load persona analytics.",
                )
            return

        try:
            skill_metrics = self.ATLAS.get_persona_metrics(
                persona_name,
                start=start_value,
                end=end_value,
                limit=25,
                metric_type="skill",
            )
        except Exception:
            skill_metrics = {}
            if show_errors:
                self.ATLAS.show_persona_message(
                    "warning",
                    "Unable to load persona skill analytics.",
                )

        totals = metrics.get('totals') if isinstance(metrics, dict) else {}
        if not isinstance(totals, dict):
            totals = {}

        calls_label = self._analytics_summary_labels.get('calls')
        self._safe_set_text(calls_label, str(int(totals.get('calls') or 0)))

        success_label = self._analytics_summary_labels.get('success')
        self._safe_set_text(success_label, str(int(totals.get('success') or 0)))

        failure_label = self._analytics_summary_labels.get('failure')
        self._safe_set_text(failure_label, str(int(totals.get('failure') or 0)))

        success_rate = metrics.get('success_rate') if isinstance(metrics, dict) else 0.0
        if not isinstance(success_rate, (int, float)):
            success_rate = 0.0
        rate_label = self._analytics_summary_labels.get('success_rate')
        self._safe_set_text(rate_label, f"{success_rate * 100:.1f}%")

        avg_latency = metrics.get('average_latency_ms') if isinstance(metrics, dict) else 0.0
        if not isinstance(avg_latency, (int, float)):
            avg_latency = 0.0
        latency_label = self._analytics_summary_labels.get('average_latency_ms')
        self._safe_set_text(latency_label, f"{avg_latency:.1f} ms")

        comparison_summary: Dict[str, Any] = {}
        if hasattr(self.ATLAS, 'get_persona_comparison_summary'):
            try:
                comparison_summary = self.ATLAS.get_persona_comparison_summary(
                    category='tool',
                    recent=5,
                )
            except Exception:
                comparison_summary = {}
                if show_errors:
                    self.ATLAS.show_persona_message(
                        'warning',
                        'Unable to load persona comparison analytics.',
                    )
        self._update_comparison_highlights(comparison_summary, persona_name)

        anomaly_entries: List[Dict[str, Any]] = []
        if isinstance(metrics, dict):
            anomaly_source = metrics.get('recent_anomalies')
            if not isinstance(anomaly_source, list):
                anomaly_source = metrics.get('anomalies')
            if isinstance(anomaly_source, list):
                anomaly_entries.extend(
                    [entry for entry in anomaly_source if isinstance(entry, dict)]
                )
        if not anomaly_entries and isinstance(skill_metrics, dict):
            skill_anomaly_source = skill_metrics.get('recent_anomalies')
            if not isinstance(skill_anomaly_source, list):
                skill_anomaly_source = skill_metrics.get('anomalies')
            if isinstance(skill_anomaly_source, list):
                anomaly_entries.extend(
                    [entry for entry in skill_anomaly_source if isinstance(entry, dict)]
                )

        seen_keys = set()
        deduped_anomalies: List[Dict[str, Any]] = []
        for entry in anomaly_entries:
            metric_name = str(entry.get('metric') or entry.get('category') or '')
            timestamp_key = str(entry.get('timestamp') or '')
            key = (metric_name, timestamp_key)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            deduped_anomalies.append(entry)
        anomaly_entries = deduped_anomalies

        if self._analytics_anomaly_heatmap is not None:
            self._analytics_anomaly_heatmap.update_anomalies(
                anomaly_entries, start_value, end_value
            )
        self._analytics_anomaly_entries = list(anomaly_entries)

        self._clear_list_box(self._analytics_anomaly_list)
        if self._analytics_anomaly_placeholder is not None:
            placeholder_parent = getattr(
                self._analytics_anomaly_placeholder, 'get_parent', lambda: None
            )()
            if placeholder_parent is not None:
                try:
                    placeholder_parent.remove(self._analytics_anomaly_placeholder)
                except Exception:
                    pass
            self._analytics_anomaly_placeholder = None

        self._analytics_anomaly_rows.clear()
        if isinstance(self._analytics_anomaly_list, Gtk.ListBox):
            try:
                unselect_func = getattr(self._analytics_anomaly_list, "unselect_all", None)
                if callable(unselect_func):
                    unselect_func()
            except Exception:
                pass

        if not anomaly_entries:
            placeholder_label = Gtk.Label(label="No anomalies detected in this range.")
            placeholder_label.set_xalign(0.0)
            if isinstance(self._analytics_anomaly_list, Gtk.ListBox):
                placeholder_row = Gtk.ListBoxRow()
                placeholder_row.set_selectable(False)
                placeholder_row.set_activatable(False)
                placeholder_row.set_child(placeholder_label)
                self._safe_append(self._analytics_anomaly_list, placeholder_row)
                self._analytics_anomaly_placeholder = placeholder_row
        else:
            for entry in anomaly_entries:
                metric_name = str(entry.get('metric') or entry.get('category') or 'Metric')
                timestamp = self._format_analytics_timestamp(entry.get('timestamp'))
                observed = entry.get('observed')
                baseline = entry.get('baseline') if isinstance(entry.get('baseline'), dict) else {}
                baseline_mean = baseline.get('mean') if isinstance(baseline, dict) else None
                z_score = baseline.get('z_score') if isinstance(baseline, dict) else None

                if isinstance(observed, (int, float)):
                    if 'failure_rate' in metric_name:
                        observed_text = f"{observed * 100:.1f}% failure"
                        if isinstance(baseline_mean, (int, float)):
                            baseline_text = f"{baseline_mean * 100:.1f}% avg"
                        else:
                            baseline_text = "baseline n/a"
                    else:
                        observed_text = f"{observed:.1f}"
                        if isinstance(baseline_mean, (int, float)):
                            baseline_text = f"{baseline_mean:.1f} baseline"
                        else:
                            baseline_text = "baseline n/a"
                else:
                    observed_text = str(observed)
                    baseline_text = (
                        f"{baseline_mean}" if baseline_mean is not None else "baseline n/a"
                    )

                summary_parts = [f"Observed {observed_text} vs {baseline_text}"]
                if isinstance(z_score, (int, float)):
                    summary_parts.append(f"z-score {z_score:.2f}")
                summary = " • ".join(summary_parts)

                actions = entry.get('suggested_actions')
                action_text = ""
                if isinstance(actions, list):
                    filtered_actions = [str(item) for item in actions if item]
                    if filtered_actions:
                        action_text = "; ".join(filtered_actions)

                row = Gtk.ListBoxRow()
                row_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
                header_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
                metric_label = Gtk.Label(label=metric_name)
                metric_label.set_xalign(0.0)
                time_label = Gtk.Label(label=timestamp)
                time_label.set_xalign(1.0)
                self._safe_add_css_class(time_label, 'dim-label')
                self._safe_append(header_box, metric_label)
                self._safe_append(header_box, time_label)

                summary_label = Gtk.Label(label=summary)
                summary_label.set_xalign(0.0)
                self._safe_add_css_class(summary_label, 'dim-label')
                summary_label.set_wrap(True)

                self._safe_append(row_box, header_box)
                self._safe_append(row_box, summary_label)

                if action_text:
                    action_label = Gtk.Label(label=action_text)
                    action_label.set_xalign(0.0)
                    self._safe_add_css_class(action_label, 'dim-label')
                    action_label.set_wrap(True)
                    self._safe_append(row_box, action_label)

                row.set_child(row_box)
                if isinstance(self._analytics_anomaly_list, Gtk.ListBox):
                    self._safe_append(self._analytics_anomaly_list, row)
                    self._analytics_anomaly_rows[row] = entry

        tool_entries = metrics.get('totals_by_tool') if isinstance(metrics, dict) else []
        if not isinstance(tool_entries, list):
            tool_entries = []

        self._clear_list_box(self._analytics_tool_list)
        if self._analytics_tool_placeholder is not None:
            placeholder_parent = getattr(self._analytics_tool_placeholder, 'get_parent', lambda: None)()
            if placeholder_parent is not None:
                try:
                    placeholder_parent.remove(self._analytics_tool_placeholder)
                except Exception:
                    pass
            self._analytics_tool_placeholder = None

        if not tool_entries:
            placeholder = Gtk.Label(label="No tool activity for this range.")
            placeholder.set_xalign(0.0)
            if isinstance(self._analytics_tool_list, Gtk.ListBox):
                self._safe_append(self._analytics_tool_list, placeholder)
            self._analytics_tool_placeholder = placeholder
        else:
            for entry in tool_entries:
                if not isinstance(entry, dict):
                    continue
                tool_name = str(entry.get('tool') or '')
                if not tool_name:
                    continue
                calls = int(entry.get('calls') or 0)
                success_count = int(entry.get('success') or 0)
                failure_count = int(entry.get('failure') or 0)
                rate = entry.get('success_rate') or 0.0
                if not isinstance(rate, (int, float)):
                    rate = 0.0
                row = Gtk.ListBoxRow()
                box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
                name_label = Gtk.Label(label=tool_name)
                name_label.set_xalign(0.0)
                metrics_label = Gtk.Label(
                    label=(
                        f"{calls} calls • {success_count} success • "
                        f"{failure_count} failure • {rate * 100:.1f}% success"
                    )
                )
                metrics_label.set_xalign(1.0)
                self._safe_add_css_class(metrics_label, 'dim-label')
                self._safe_append(box, name_label)
                self._safe_append(box, metrics_label)
                row.set_child(box)
                if isinstance(self._analytics_tool_list, Gtk.ListBox):
                    self._safe_append(self._analytics_tool_list, row)

        if not isinstance(skill_metrics, dict):
            skill_metrics = {}

        skill_entries = skill_metrics.get('totals_by_skill') if isinstance(skill_metrics, dict) else []
        if not isinstance(skill_entries, list):
            skill_entries = []

        self._clear_list_box(self._analytics_skill_list)
        if self._analytics_skill_placeholder is not None:
            placeholder_parent = getattr(
                self._analytics_skill_placeholder, 'get_parent', lambda: None
            )()
            if placeholder_parent is not None:
                try:
                    placeholder_parent.remove(self._analytics_skill_placeholder)
                except Exception:
                    pass
            self._analytics_skill_placeholder = None

        if not skill_entries:
            placeholder = Gtk.Label(label="No skill activity for this range.")
            placeholder.set_xalign(0.0)
            if isinstance(self._analytics_skill_list, Gtk.ListBox):
                self._safe_append(self._analytics_skill_list, placeholder)
            self._analytics_skill_placeholder = placeholder
        else:
            for entry in skill_entries:
                if not isinstance(entry, dict):
                    continue
                skill_name = str(entry.get('skill') or '')
                if not skill_name:
                    continue
                calls = int(entry.get('calls') or 0)
                success_count = int(entry.get('success') or 0)
                failure_count = int(entry.get('failure') or 0)
                rate = entry.get('success_rate') or 0.0
                if not isinstance(rate, (int, float)):
                    rate = 0.0
                row = Gtk.ListBoxRow()
                box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
                name_label = Gtk.Label(label=skill_name)
                name_label.set_xalign(0.0)
                metrics_label = Gtk.Label(
                    label=(
                        f"{calls} calls • {success_count} success • "
                        f"{failure_count} failure • {rate * 100:.1f}% success"
                    )
                )
                metrics_label.set_xalign(1.0)
                self._safe_add_css_class(metrics_label, 'dim-label')
                self._safe_append(box, name_label)
                self._safe_append(box, metrics_label)
                row.set_child(box)
                if isinstance(self._analytics_skill_list, Gtk.ListBox):
                    self._safe_append(self._analytics_skill_list, row)

        recent_entries = metrics.get('recent') if isinstance(metrics, dict) else []
        if not isinstance(recent_entries, list):
            recent_entries = []

        if self._analytics_latency_chart is not None:
            self._analytics_latency_chart.update_samples(
                recent_entries, start_value, end_value
            )
        self._analytics_recent_entries = list(recent_entries)

        self._clear_list_box(self._analytics_recent_list)
        if self._analytics_recent_placeholder is not None:
            placeholder_parent = getattr(
                self._analytics_recent_placeholder, 'get_parent', lambda: None
            )()
            if placeholder_parent is not None:
                try:
                    placeholder_parent.remove(self._analytics_recent_placeholder)
                except Exception:
                    pass
            self._analytics_recent_placeholder = None

        if not recent_entries:
            placeholder = Gtk.Label(label="No recent executions.")
            placeholder.set_xalign(0.0)
            if isinstance(self._analytics_recent_list, Gtk.ListBox):
                self._safe_append(self._analytics_recent_list, placeholder)
            self._analytics_recent_placeholder = placeholder
        else:
            for entry in recent_entries:
                if not isinstance(entry, dict):
                    continue
                tool_name = str(entry.get('tool') or '')
                status = "Success" if entry.get('success') else "Failure"
                latency = entry.get('latency_ms') or 0.0
                if not isinstance(latency, (int, float)):
                    latency = 0.0
                timestamp = self._format_analytics_timestamp(entry.get('timestamp'))
                row = Gtk.ListBoxRow()
                box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
                header = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
                time_label = Gtk.Label(label=timestamp)
                time_label.set_xalign(0.0)
                status_label = Gtk.Label(label=f"{status} • {latency:.1f} ms")
                status_label.set_xalign(1.0)
                self._safe_add_css_class(status_label, 'dim-label')
                self._safe_append(header, time_label)
                self._safe_append(header, status_label)
                detail = Gtk.Label(label=tool_name or "(unknown tool)")
                detail.set_xalign(0.0)
                self._safe_append(box, header)
                self._safe_append(box, detail)
                row.set_child(box)
                if isinstance(self._analytics_recent_list, Gtk.ListBox):
                    self._safe_append(self._analytics_recent_list, row)

    def create_analytics_tab(self, persona_state) -> Gtk.Box:
        persona_name = ""
        if isinstance(persona_state, dict):
            general = persona_state.get('general')
            if isinstance(general, dict):
                persona_name = str(general.get('name') or "")
            if not persona_name:
                persona_name = str(persona_state.get('original_name') or "")

        container = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        container.set_margin_top(10)
        container.set_margin_bottom(10)
        container.set_margin_start(10)
        container.set_margin_end(10)

        controls = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        controls.set_valign(Gtk.Align.START)
        start_entry = Gtk.Entry()
        start_entry.set_placeholder_text("Start (YYYY-MM-DD)")
        start_entry.set_tooltip_text("Filter analytics to timestamps on or after this value.")
        self._safe_append(controls, start_entry)

        end_entry = Gtk.Entry()
        end_entry.set_placeholder_text("End (YYYY-MM-DD)")
        end_entry.set_tooltip_text("Filter analytics to timestamps on or before this value.")
        self._safe_append(controls, end_entry)

        refresh_button = Gtk.Button(label="Refresh")
        refresh_button.set_tooltip_text("Reload analytics for the selected date range.")
        refresh_button.connect("clicked", lambda _btn: self._refresh_persona_analytics())
        self._safe_append(controls, refresh_button)

        self._safe_append(container, controls)

        summary_grid = Gtk.Grid()
        summary_grid.set_column_spacing(12)
        summary_grid.set_row_spacing(6)
        summary_labels: Dict[str, Gtk.Label] = {}

        summary_items = [
            ("Total Calls", 'calls'),
            ("Success", 'success'),
            ("Failure", 'failure'),
            ("Success Rate", 'success_rate'),
            ("Average Latency", 'average_latency_ms'),
        ]

        for row_index, (label_text, key) in enumerate(summary_items):
            label_widget = Gtk.Label(label=label_text)
            label_widget.set_xalign(0.0)
            value_widget = Gtk.Label(label="0")
            value_widget.set_xalign(1.0)
            self._safe_add_css_class(value_widget, 'dim-label')
            summary_grid.attach(label_widget, 0, row_index, 1, 1)
            summary_grid.attach(value_widget, 1, row_index, 1, 1)
            summary_labels[key] = value_widget

        self._safe_append(container, summary_grid)

        chart_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        chart_row.set_hexpand(True)
        chart_row.set_vexpand(False)
        chart_row.set_homogeneous(True)

        heatmap = AnomalyHeatmap()
        heatmap.set_activate_handler(self._on_heatmap_day_selected)
        self._safe_append(chart_row, heatmap)

        latency_chart = LatencyTimeline()
        latency_chart.set_activate_handler(self._on_timeline_point_selected)
        self._safe_append(chart_row, latency_chart)

        self._safe_append(container, chart_row)

        comparison_label = Gtk.Label(label="Comparison Highlights")
        comparison_label.set_xalign(0.0)
        self._safe_append(container, comparison_label)

        comparison_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        comparison_box.set_hexpand(True)
        self._safe_add_css_class(comparison_box, 'boxed-list')

        top_label = Gtk.Label(label="Top Performer: —")
        top_label.set_xalign(0.0)
        self._safe_add_css_class(top_label, 'dim-label')
        self._safe_append(comparison_box, top_label)

        worst_label = Gtk.Label(label="Highest Failure Rate: —")
        worst_label.set_xalign(0.0)
        self._safe_add_css_class(worst_label, 'dim-label')
        self._safe_append(comparison_box, worst_label)

        fastest_label = Gtk.Label(label="Fastest Avg Latency: —")
        fastest_label.set_xalign(0.0)
        self._safe_add_css_class(fastest_label, 'dim-label')
        self._safe_append(comparison_box, fastest_label)

        self._safe_append(container, comparison_box)

        anomaly_label = Gtk.Label(label="Recent Anomalies")
        anomaly_label.set_xalign(0.0)
        self._safe_append(container, anomaly_label)

        anomaly_scroll = Gtk.ScrolledWindow()
        anomaly_scroll.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        anomaly_scroll.set_min_content_height(140)
        anomaly_scroll.set_hexpand(True)
        anomaly_scroll.set_vexpand(True)
        anomaly_list = Gtk.ListBox()
        anomaly_list.set_selection_mode(Gtk.SelectionMode.SINGLE)
        self._safe_add_css_class(anomaly_list, 'boxed-list')
        anomaly_list.connect('row-selected', self._on_anomaly_row_selected)
        anomaly_scroll.set_child(anomaly_list)
        self._safe_append(container, anomaly_scroll)

        tool_label = Gtk.Label(label="Tool Breakdown")
        tool_label.set_xalign(0.0)
        self._safe_append(container, tool_label)

        tool_scroll = Gtk.ScrolledWindow()
        tool_scroll.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        tool_scroll.set_min_content_height(140)
        tool_scroll.set_hexpand(True)
        tool_scroll.set_vexpand(True)
        tool_list = Gtk.ListBox()
        tool_list.set_selection_mode(Gtk.SelectionMode.NONE)
        self._safe_add_css_class(tool_list, 'boxed-list')
        tool_scroll.set_child(tool_list)
        self._safe_append(container, tool_scroll)

        skill_label = Gtk.Label(label="Skill Breakdown")
        skill_label.set_xalign(0.0)
        self._safe_append(container, skill_label)

        skill_scroll = Gtk.ScrolledWindow()
        skill_scroll.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        skill_scroll.set_min_content_height(140)
        skill_scroll.set_hexpand(True)
        skill_scroll.set_vexpand(True)
        skill_list = Gtk.ListBox()
        skill_list.set_selection_mode(Gtk.SelectionMode.NONE)
        self._safe_add_css_class(skill_list, 'boxed-list')
        skill_scroll.set_child(skill_list)
        self._safe_append(container, skill_scroll)

        recent_label = Gtk.Label(label="Recent Activity")
        recent_label.set_xalign(0.0)
        self._safe_append(container, recent_label)

        recent_scroll = Gtk.ScrolledWindow()
        recent_scroll.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        recent_scroll.set_min_content_height(140)
        recent_scroll.set_hexpand(True)
        recent_scroll.set_vexpand(True)
        recent_list = Gtk.ListBox()
        recent_list.set_selection_mode(Gtk.SelectionMode.NONE)
        self._safe_add_css_class(recent_list, 'boxed-list')
        recent_scroll.set_child(recent_list)
        self._safe_append(container, recent_scroll)

        self._analytics_persona_name = persona_name
        self._analytics_start_entry = start_entry
        self._analytics_end_entry = end_entry
        self._analytics_summary_labels = summary_labels
        self._analytics_comparison_labels = {
            'top': top_label,
            'worst': worst_label,
            'fastest': fastest_label,
        }
        self._analytics_tool_list = tool_list
        self._analytics_skill_list = skill_list
        self._analytics_recent_list = recent_list
        self._analytics_anomaly_list = anomaly_list
        self._analytics_tool_placeholder = None
        self._analytics_skill_placeholder = None
        self._analytics_recent_placeholder = None
        self._analytics_anomaly_placeholder = None
        self._analytics_anomaly_heatmap = heatmap
        self._analytics_latency_chart = latency_chart

        self._ensure_theme_monitoring()
        self._apply_chart_theme()

    def _load_persona_metadata(self, persona_names: List[str]) -> None:
        manager = getattr(self.ATLAS, "persona_manager", None)
        loader = getattr(manager, "get_persona", None)

        metadata: Dict[str, Mapping[str, Any]] = {}
        if callable(loader):
            for name in persona_names:
                try:
                    persona = loader(name)
                except Exception:
                    persona = None

                if isinstance(persona, Mapping):
                    metadata[name] = persona

        self._persona_metadata = metadata

    def _is_personal_persona(self, persona_name: str) -> bool:
        metadata = self._persona_metadata.get(persona_name) or {}

        tags = metadata.get("tags") or metadata.get("persona_tags")
        if isinstance(tags, Mapping):
            tags = tags.values()
        if isinstance(tags, (list, tuple, set)):
            if any(str(tag).strip().lower() == "personal" for tag in tags):
                return True

        audience = metadata.get("audience")
        if isinstance(audience, str) and audience.strip().lower() == "personal":
            return True

        flags = metadata.get("flags")
        if isinstance(flags, Mapping):
            audience_flag = flags.get("audience")
            if isinstance(audience_flag, str) and audience_flag.strip().lower() == "personal":
                return True

        return False

        self._refresh_persona_analytics(show_errors=False)

        return container

    def save_persona_settings(self, settings_window):
        """
        Gathers settings from the General, Persona Type, Provider/Model,
        and Speech/Voice tabs, updates the persona, and then saves the changes.

        Args:
            settings_window (Gtk.Window): The settings window to close after saving.
        """
        self._clear_skill_dependency_highlights()

        if not self._current_editor_state:
            self.ATLAS.show_persona_message(
                "system",
                "Persona data is unavailable; cannot save.",
            )
            return
        general_payload = {
            'name': self.general_tab.get_name(),
            'meaning': self.general_tab.get_meaning(),
            'content': {
                'start_locked': self.general_tab.get_start_locked(),
                'editable_content': self.general_tab.get_editable_content(),
                'end_locked': self.general_tab.get_end_locked(),
            },
        }

        persona_type_payload = self.persona_type_tab.get_values() or {}

        provider_payload = {
            'provider': self._safe_get_text(self.provider_entry),
            'model': self._safe_get_text(self.model_entry),
        }

        speech_payload = {
            'Speech_provider': self._safe_get_text(self.speech_provider_entry),
            'voice': self._safe_get_text(self.voice_entry),
        }

        tools_payload = self._collect_tool_payload()
        skills_payload = self._collect_skill_payload()

        result = self.ATLAS.update_persona_from_editor(
            self._current_editor_state.get('original_name'),
            general_payload,
            persona_type_payload,
            provider_payload,
            speech_payload,
            tools_payload,
            skills_payload,
        )

        if result.get('success'):
            persona_result = result.get('persona')
            target_name = general_payload['name']
            if persona_result:
                target_name = persona_result.get('name') or target_name
            refreshed_state = self.ATLAS.get_persona_editor_state(target_name)
            if refreshed_state is not None:
                self._current_editor_state.clear()
                self._current_editor_state.update(refreshed_state)
            print(f"Settings for {general_payload['name']} saved!")
            settings_window.destroy()
        else:
            error_list = [str(entry) for entry in result.get('errors', []) if str(entry)]
            handled, guidance = self._handle_skill_dependency_errors(error_list)
            error_text = "; ".join(error_list) or "Unable to save persona settings."
            self.ATLAS.show_persona_message("system", f"Failed to save persona: {error_text}")
            if handled and guidance:
                self.ATLAS.show_persona_message("warning", guidance)
