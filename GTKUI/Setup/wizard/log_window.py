"""Log window helpers for the setup wizard."""

from __future__ import annotations

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import Gtk

from GTKUI.Utils.styled_window import AtlasWindow


class SetupWizardLogWindow(AtlasWindow):
    """Lightweight window that streams setup logs to an inspector view."""

    def __init__(
        self,
        *,
        application: Gtk.Application | None = None,
        transient_for: Gtk.Window | None = None,
    ) -> None:
        super().__init__(
            title="ATLAS Setup Logs",
            default_size=(720, 480),
            transient_for=transient_for,
        )

        if application is not None:
            try:
                self.set_application(application)
            except Exception:  # pragma: no cover - GTK stubs in tests
                pass

        header = Gtk.HeaderBar()
        try:
            header.set_show_title_buttons(True)
        except Exception:  # pragma: no cover - compatibility shim
            pass

        title_label = Gtk.Label(label="Setup Activity Log")
        if hasattr(title_label, "add_css_class"):
            title_label.add_css_class("heading")
        try:
            header.set_title_widget(title_label)
        except Exception:  # pragma: no cover - GTK3 fallback
            try:
                header.set_title("Setup Activity Log")
            except Exception:
                pass

        try:
            self.set_titlebar(header)
        except Exception:  # pragma: no cover - GTK3 fallback
            pass

        root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        root.set_hexpand(True)
        root.set_vexpand(True)
        self.set_child(root)

        scroller = Gtk.ScrolledWindow()
        scroller.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        scroller.set_hexpand(True)
        scroller.set_vexpand(True)
        root.append(scroller)

        self.text_view = Gtk.TextView()
        self.text_view.set_editable(False)
        self.text_view.set_cursor_visible(False)
        self.text_view.set_monospace(True)
        wrap_mode = getattr(Gtk.WrapMode, "WORD_CHAR", getattr(Gtk.WrapMode, "WORD", None))
        if wrap_mode is not None:
            self.text_view.set_wrap_mode(wrap_mode)
        scroller.set_child(self.text_view)

        self.text_buffer = self.text_view.get_buffer()
