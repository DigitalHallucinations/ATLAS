# GTKUI/sidebar.py

"""Main application window with navigation sidebar and central workspace."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Tuple

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import GLib, Gtk

from GTKUI.Chat.chat_page import ChatPage
from GTKUI.Persona_manager.persona_management import PersonaManagement
from GTKUI.Provider_manager.provider_management import ProviderManagement
from GTKUI.Settings.Speech.speech_settings import SpeechSettings
from GTKUI.Tool_manager import ToolManagement
from GTKUI.Skill_manager import SkillManagement
from GTKUI.UserAccounts.account_dialog import AccountDialog
from GTKUI.Utils.styled_window import AtlasWindow
from GTKUI.Utils.utils import apply_css

logger = logging.getLogger(__name__)


class MainWindow(AtlasWindow):
    """Top-level Atlas window with a navigation column and notebook workspace."""

    def __init__(self, atlas) -> None:
        super().__init__(title="ATLAS")
        safe_width, safe_height = self._calculate_safe_size(1200, 800)
        self.set_default_size(safe_width, safe_height)
        self.ATLAS = atlas
        self._pages: Dict[str, Gtk.Widget] = {}
        self._page_controllers: Dict[str, Any] = {}

        apply_css()

        self.persona_management = PersonaManagement(self.ATLAS, self)
        self.provider_management = ProviderManagement(self.ATLAS, self)
        self.tool_management = ToolManagement(self.ATLAS, self)
        self.skill_management = SkillManagement(self.ATLAS, self)

        layout = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=0)
        layout.set_hexpand(True)
        layout.set_vexpand(True)
        self.set_child(layout)
        self.connect("close-request", self.close_application)

        self.sidebar = _NavigationSidebar(self)
        layout.append(self.sidebar)

        separator = Gtk.Separator(orientation=Gtk.Orientation.VERTICAL)
        layout.append(separator)

        self.notebook = Gtk.Notebook()
        self.notebook.set_hexpand(True)
        self.notebook.set_vexpand(True)
        self.notebook.set_scrollable(True)
        layout.append(self.notebook)

    # ------------------------------------------------------------------
    # Navigation actions
    # ------------------------------------------------------------------
    def show_chat_page(self) -> None:
        if not self._ensure_initialized():
            return

        def factory() -> Gtk.Widget:
            return ChatPage(self.ATLAS)

        page = self._open_or_focus_page("chat", "Chat", factory)
        if page is not None:
            self.sidebar.set_active_item("chat")

    def show_provider_menu(self) -> None:
        if not self._ensure_initialized():
            return

        def factory() -> Gtk.Widget:
            return self.provider_management.get_embeddable_widget()

        page = self._open_or_focus_page("providers", "Providers", factory)
        if page is not None:
            self.sidebar.set_active_item("providers")

    def show_persona_menu(self) -> None:
        if not self._ensure_initialized():
            return

        def factory() -> Gtk.Widget:
            return self.persona_management.get_embeddable_widget()

        page = self._open_or_focus_page("personas", "Personas", factory)
        if page is not None:
            self.sidebar.set_active_item("personas")

    def show_tools_menu(self) -> None:
        if not self._ensure_initialized():
            return

        def factory() -> Gtk.Widget:
            return self.tool_management.get_embeddable_widget()

        page = self._open_or_focus_page("tools", "Tools", factory)
        if page is not None:
            self.sidebar.set_active_item("tools")

    def show_skills_menu(self) -> None:
        if not self._ensure_initialized():
            return

        def factory() -> Gtk.Widget:
            return self.skill_management.get_embeddable_widget()

        page = self._open_or_focus_page("skills", "Skills", factory)
        if page is not None:
            self.sidebar.set_active_item("skills")

    def show_speech_settings(self) -> None:
        if not self._ensure_initialized():
            return

        def factory() -> Gtk.Widget:
            return SpeechSettings(self.ATLAS)

        page = self._open_or_focus_page("speech", "Speech", factory)
        if page is not None:
            self.sidebar.set_active_item("speech")

    def show_settings_page(self) -> None:
        def factory() -> Gtk.Widget:
            box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
            box.set_margin_top(24)
            box.set_margin_bottom(24)
            box.set_margin_start(24)
            box.set_margin_end(24)
            label = Gtk.Label(label="Settings workspaces are coming soon.")
            label.set_wrap(True)
            label.set_justify(Gtk.Justification.CENTER)
            box.append(label)
            return box

        page = self._open_or_focus_page("settings", "Settings", factory)
        if page is not None:
            self.sidebar.set_active_item("settings")

    def handle_history_button(self) -> None:
        if not self._ensure_initialized():
            return
        self.ATLAS.log_history()

    def show_accounts_page(self) -> None:
        if not self._ensure_initialized():
            return

        def factory() -> Tuple[Gtk.Widget, AccountDialog]:
            dialog = AccountDialog(
                self.ATLAS,
                parent=self,
                on_close=lambda: self._close_page("accounts"),
            )
            content = dialog.get_child()
            if content is None:
                fallback = Gtk.Label(label="Account management is unavailable.")
                fallback.set_hexpand(True)
                fallback.set_vexpand(True)
                return fallback, dialog

            dialog.set_child(None)
            content.set_hexpand(True)
            content.set_vexpand(True)

            scroller = Gtk.ScrolledWindow()
            scroller.set_hexpand(True)
            scroller.set_vexpand(True)
            scroller.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
            scroller.set_child(content)

            return scroller, dialog

        result = self._open_or_focus_page("accounts", "Accounts", factory)
        if result is not None:
            self.sidebar.set_active_item("accounts")

    def close_application(self, *_args) -> bool:
        logger.info("Closing application")
        app = self.get_application()
        if app:
            app.quit()
        return False

    # ------------------------------------------------------------------
    # Page management helpers
    # ------------------------------------------------------------------
    def _open_or_focus_page(
        self,
        page_id: str,
        title: str,
        factory: Callable[[], Gtk.Widget | Tuple[Gtk.Widget, Any]],
    ) -> Gtk.Widget | None:
        widget = self._pages.get(page_id)
        if widget is not None:
            self._focus_page(widget)
            return widget

        try:
            result = factory()
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Failed to build %s page: %s", page_id, exc, exc_info=True)
            self.show_error_dialog(str(exc) or f"Unable to open {title} page")
            return None

        controller: Any | None = None
        if isinstance(result, tuple):
            widget, controller = result
        else:
            widget = result
            controller = None

        if widget is None:
            return None

        widget.set_hexpand(True)
        widget.set_vexpand(True)
        tab_label = self._create_tab_header(title, page_id)
        page_index = self.notebook.append_page(widget, tab_label)
        self.notebook.set_tab_reorderable(widget, True)
        self.notebook.set_current_page(page_index)
        self._pages[page_id] = widget
        if controller is not None:
            self._page_controllers[page_id] = controller
        return widget

    def _focus_page(self, widget: Gtk.Widget) -> None:
        page_index = self.notebook.page_num(widget)
        if page_index != -1:
            self.notebook.set_current_page(page_index)

    def _close_page(self, page_id: str) -> None:
        widget = self._pages.pop(page_id, None)
        if widget is None:
            return
        try:
            self.notebook.remove(widget)
        except Exception:  # pragma: no cover - fallback for older GTK stubs
            page_index = self.notebook.page_num(widget)
            if page_index != -1:
                self.notebook.remove_page(page_index)

        controller = self._page_controllers.pop(page_id, None)
        if controller is not None:
            close_request = getattr(controller, "_on_close_request", None)
            if callable(close_request):
                try:
                    close_request()
                except Exception:  # pragma: no cover - defensive cleanup
                    logger.debug("Error during controller close for %s", page_id, exc_info=True)
        if page_id == "providers":
            self.provider_management._provider_page = None
        elif page_id == "personas":
            self.persona_management._persona_page = None

    def _create_tab_header(self, title: str, page_id: str) -> Gtk.Widget:
        box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        box.set_margin_top(4)
        box.set_margin_bottom(4)
        box.set_margin_start(8)
        box.set_margin_end(4)

        label = Gtk.Label(label=title)
        label.set_xalign(0.0)
        label.set_yalign(0.5)
        box.append(label)

        close_btn = Gtk.Button()
        close_btn.add_css_class("flat")
        close_btn.set_tooltip_text(f"Close {title} tab")
        icon = Gtk.Image.new_from_icon_name("window-close")
        icon.set_pixel_size(12)
        close_btn.set_child(icon)
        close_btn.connect("clicked", lambda _btn: self._close_page(page_id))
        box.append(close_btn)

        return box

    def _ensure_initialized(self) -> bool:
        if self.ATLAS.is_initialized():
            return True
        self.show_error_dialog("ATLAS is not fully initialized. Please try again later.")
        return False

    # ------------------------------------------------------------------
    # Dialog helpers
    # ------------------------------------------------------------------
    def show_error_dialog(self, message: str) -> None:
        dialog = Gtk.MessageDialog(
            transient_for=self,
            modal=True,
            message_type=Gtk.MessageType.ERROR,
            buttons=Gtk.ButtonsType.OK,
            text="Initialization Error",
        )
        self._apply_shared_styles(dialog)
        # ``Gtk.MessageDialog.format_secondary_text`` was removed in GTK4.
        # Prefer ``set_secondary_text`` when available and fall back to
        # directly assigning the property for compatibility with GTK 4.x.
        set_secondary = getattr(dialog, "set_secondary_text", None)
        if callable(set_secondary):
            set_secondary(message)
        else:
            try:
                dialog.props.secondary_text = message
            except Exception:  # pragma: no cover - property unavailable
                pass
        dialog.connect("response", lambda d, r: d.destroy())
        dialog.present()

    def _apply_shared_styles(self, widget: Gtk.Widget) -> None:
        try:
            apply_css()
        except Exception:  # pragma: no cover
            pass

        get_context = getattr(widget, "get_style_context", None)
        if not callable(get_context):
            return

        context = get_context()
        if context is None:
            return

        for css_class in ("chat-page", "sidebar"):
            try:
                context.add_class(css_class)
            except Exception:
                continue


class _NavigationSidebar(Gtk.Box):
    """Vertical navigation used along the left edge of the main window."""

    def __init__(self, main_window: MainWindow) -> None:
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        self.main_window = main_window
        self.ATLAS = main_window.ATLAS
        self.persona_management = main_window.persona_management
        self.provider_management = main_window.provider_management
        self.tool_management = main_window.tool_management
        self.skill_management = main_window.skill_management

        self.set_margin_top(4)
        self.set_margin_bottom(4)
        self.set_margin_start(4)
        self.set_margin_end(4)
        self.set_valign(Gtk.Align.FILL)
        self.set_size_request(96, -1)

        content_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        content_box.set_hexpand(True)
        content_box.set_vexpand(True)

        scroller = Gtk.ScrolledWindow()
        scroller.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        scroller.set_vexpand(True)
        scroller.set_child(content_box)
        self.append(scroller)

        self._nav_items: Dict[str, Gtk.ListBoxRow] = {}
        self._row_actions: Dict[Gtk.ListBoxRow, Callable[[], None]] = {}
        self._active_nav_id: str | None = None

        primary_listbox = Gtk.ListBox()
        primary_listbox.set_selection_mode(Gtk.SelectionMode.NONE)
        primary_listbox.set_accessible_role(Gtk.AccessibleRole.LIST)
        primary_listbox.add_css_class("sidebar-nav")
        self._primary_listbox = primary_listbox
        primary_listbox.connect("row-activated", self._on_row_activated)
        content_box.append(primary_listbox)

        self._create_nav_item(
            "providers",
            "Providers",
            self.main_window.show_provider_menu,
            tooltip="Providers",
        )
        self._create_nav_item(
            None,
            "History",
            self.main_window.handle_history_button,
            tooltip="History",
        )
        self._create_nav_item(
            "chat",
            "Chat",
            self.main_window.show_chat_page,
            tooltip="Chat",
        )
        self._create_nav_item(
            "tools",
            "Tools",
            self.main_window.show_tools_menu,
            tooltip="Tools",
        )
        self._create_nav_item(
            "skills",
            "Skills",
            self.main_window.show_skills_menu,
            tooltip="Skills",
        )
        self._create_nav_item(
            "accounts",
            "Accounts",
            self.main_window.show_accounts_page,
            tooltip="Accounts",
        )
        self._create_nav_item(
            "speech",
            "Speech Settings",
            self.main_window.show_speech_settings,
            tooltip="Speech Settings",
        )
        self._create_nav_item(
            "personas",
            "Personas",
            self.main_window.show_persona_menu,
            tooltip="Personas",
        )

        spacer = Gtk.Box()
        spacer.set_vexpand(True)
        content_box.append(spacer)

        separator = Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL)
        separator.set_margin_top(10)
        separator.set_margin_bottom(10)
        content_box.append(separator)

        footer_listbox = Gtk.ListBox()
        footer_listbox.set_selection_mode(Gtk.SelectionMode.NONE)
        footer_listbox.set_accessible_role(Gtk.AccessibleRole.LIST)
        footer_listbox.add_css_class("sidebar-nav")
        footer_listbox.connect("row-activated", self._on_row_activated)
        self._footer_listbox = footer_listbox
        content_box.append(footer_listbox)

        self._create_nav_item(
            "settings",
            "Settings",
            self.main_window.show_settings_page,
            tooltip="Settings",
            container=footer_listbox,
        )

    # ------------------------------------------------------------------
    def _create_nav_item(
        self,
        nav_id: str | None,
        label: str,
        callback: Callable[[], None],
        tooltip: str | None = None,
        container: Gtk.ListBox | None = None,
    ) -> None:
        row = Gtk.ListBoxRow()
        row.set_accessible_role(Gtk.AccessibleRole.LIST_ITEM)
        if hasattr(row, "set_accessible_name"):
            row.set_accessible_name(label)
        elif hasattr(row, "update_property"):
            try:
                row.update_property(
                    Gtk.AccessibleProperty.LABEL, GLib.Variant.new_string(label)
                )
            except TypeError as exc:  # pragma: no cover - defensive fallback
                logger.debug(
                    "Gtk.ListBoxRow.update_property failed; skipping accessible label: %s",
                    exc,
                )
        else:
            logger.debug(
                "Gtk.ListBoxRow does not support accessible names; skipping label assignment"
            )
        row.set_focusable(True)
        row.add_css_class("sidebar-nav-item")
        row.set_tooltip_text(tooltip)

        box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        box.set_margin_top(6)
        box.set_margin_bottom(6)
        box.set_margin_start(10)
        box.set_margin_end(10)

        text = Gtk.Label(label=label)
        text.set_xalign(0.0)
        text.set_halign(Gtk.Align.START)
        text.set_hexpand(True)
        box.append(text)

        row.set_child(box)
        row.set_activatable(True)

        gesture = Gtk.GestureClick()
        gesture.connect("released", lambda _gesture, _n_press, _x, _y: callback())
        row.add_controller(gesture)

        self._row_actions[row] = callback
        target = container if container is not None else self._primary_listbox
        target.append(row)

        if nav_id:
            self._nav_items[nav_id] = row

    def _on_row_activated(self, _listbox: Gtk.ListBox, row: Gtk.ListBoxRow) -> None:
        callback = self._row_actions.get(row)
        if callable(callback):
            callback()

    def set_active_item(self, nav_id: str | None) -> None:
        if nav_id == self._active_nav_id:
            return

        if self._active_nav_id and self._active_nav_id in self._nav_items:
            previous = self._nav_items[self._active_nav_id]
            previous.remove_css_class("active")

        self._active_nav_id = nav_id if nav_id in self._nav_items else None

        if self._active_nav_id:
            current = self._nav_items[self._active_nav_id]
            current.add_css_class("active")


# Backwards compatibility for imports expecting ``Sidebar``
Sidebar = MainWindow
