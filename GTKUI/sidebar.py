# GTKUI/sidebar.py

"""Main application window with navigation sidebar and central workspace."""

from __future__ import annotations

import os
import logging
from typing import Any, Callable, Dict, Tuple

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import Gtk, Gdk

from GTKUI.Chat.chat_page import ChatPage
from GTKUI.Persona_manager.persona_management import PersonaManagement
from GTKUI.Provider_manager.provider_management import ProviderManagement
from GTKUI.Settings.Speech.speech_settings import SpeechSettings
from GTKUI.UserAccounts.account_dialog import AccountDialog
from GTKUI.Utils.styled_window import AtlasWindow
from GTKUI.Utils.utils import apply_css

logger = logging.getLogger(__name__)


class MainWindow(AtlasWindow):
    """Top-level Atlas window with a navigation column and notebook workspace."""

    def __init__(self, atlas) -> None:
        super().__init__(title="ATLAS", default_size=(1200, 800))
        self.ATLAS = atlas
        self._pages: Dict[str, Gtk.Widget] = {}
        self._page_controllers: Dict[str, Any] = {}

        apply_css()

        self.persona_management = PersonaManagement(self.ATLAS, self)
        self.provider_management = ProviderManagement(self.ATLAS, self)

        layout = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=0)
        layout.set_hexpand(True)
        layout.set_vexpand(True)
        self.set_child(layout)

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

        self._open_or_focus_page("chat", "Chat", factory)

    def show_provider_menu(self) -> None:
        if not self._ensure_initialized():
            return

        def factory() -> Gtk.Widget:
            return self.provider_management.get_embeddable_widget()

        self._open_or_focus_page("providers", "Providers", factory)

    def show_persona_menu(self) -> None:
        if not self._ensure_initialized():
            return

        def factory() -> Gtk.Widget:
            return self.persona_management.get_embeddable_widget()

        self._open_or_focus_page("personas", "Personas", factory)

    def show_speech_settings(self) -> None:
        if not self._ensure_initialized():
            return

        def factory() -> Gtk.Widget:
            return SpeechSettings(self.ATLAS)

        self._open_or_focus_page("speech", "Speech", factory)

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

        self._open_or_focus_page("settings", "Settings", factory)

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

        self._open_or_focus_page("accounts", "Accounts", factory)

    def close_application(self) -> None:
        logger.info("Closing application")
        app = self.get_application()
        if app:
            app.quit()

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

    ICON_SIZE = 42

    def __init__(self, main_window: MainWindow) -> None:
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        self.main_window = main_window
        self.ATLAS = main_window.ATLAS
        self.persona_management = main_window.persona_management
        self.provider_management = main_window.provider_management

        self.set_margin_top(10)
        self.set_margin_bottom(10)
        self.set_margin_start(10)
        self.set_margin_end(10)
        self.set_valign(Gtk.Align.FILL)
        self.set_size_request(96, -1)

        self._create_icon("Icons/providers.png", self.main_window.show_provider_menu, "Providers")
        self._create_icon("Icons/history.png", self.main_window.handle_history_button, "History")
        self._create_icon("Icons/chat.png", self.main_window.show_chat_page, "Chat")
        self._create_icon("Icons/user.png", self.main_window.show_accounts_page, "Accounts")
        self._create_icon("Icons/speech.png", self.main_window.show_speech_settings, "Speech Settings")
        self._create_icon("Icons/agent.png", self.main_window.show_persona_menu, "Personas")

        separator = Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL)
        separator.set_margin_top(10)
        separator.set_margin_bottom(10)
        self.append(separator)

        self._create_icon("Icons/settings.png", self.main_window.show_settings_page, "Settings")
        self._create_icon("Icons/power_button.png", self.main_window.close_application, "Quit")

    # ------------------------------------------------------------------
    def _create_icon(self, icon_path: str, callback: Callable[[], None], tooltip: str | None = None) -> None:
        button = Gtk.Button()
        button.add_css_class("icon")
        button.add_css_class("flat")
        button.set_can_focus(True)
        button.set_tooltip_text(tooltip)

        picture = self._load_icon_texture(icon_path)
        button.set_child(picture)
        button.connect("clicked", lambda _btn: callback())
        self.append(button)

    def _load_icon_texture(self, relative_path: str) -> Gtk.Widget:
        def _resolve_path(path: str) -> str:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            return os.path.abspath(os.path.join(current_dir, "..", path))

        try:
            full_path = _resolve_path(relative_path)
            texture = Gdk.Texture.new_from_filename(full_path)
            picture = Gtk.Picture.new_for_paintable(texture)
            picture.set_content_fit(Gtk.ContentFit.CONTAIN)
            picture.set_size_request(self.ICON_SIZE, self.ICON_SIZE)
            return picture
        except Exception as exc:
            logger.error("Error loading icon '%s': %s", relative_path, exc)
            fallback = Gtk.Image.new_from_icon_name("image-missing")
            fallback.set_size_request(self.ICON_SIZE, self.ICON_SIZE)
            return fallback


# Backwards compatibility for imports expecting ``Sidebar``
Sidebar = MainWindow
