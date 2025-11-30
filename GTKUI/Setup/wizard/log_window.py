"""Log window helpers for the setup wizard."""

from __future__ import annotations

import logging
from typing import Callable, Iterable, Sequence

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import Gtk

from GTKUI.Utils.styled_window import AtlasWindow


class SetupWizardLogWindow(AtlasWindow):
    """Lightweight window that streams setup logs to an inspector view."""

    ALL_COMPONENTS_OPTION = "All components"
    FILTER_LEVELS: tuple[tuple[str, int], ...] = (
        ("Debug", logging.DEBUG),
        ("Info", logging.INFO),
        ("Warning", logging.WARNING),
        ("Error", logging.ERROR),
        ("Critical", logging.CRITICAL),
    )

    def __init__(
        self,
        *,
        application: Gtk.Application | None = None,
        transient_for: Gtk.Window | None = None,
        on_filter_changed: Callable[[int], None] | None = None,
        on_component_changed: Callable[[str | None], None] | None = None,
        initial_filter_level: int | None = None,
        components: Sequence[str] | None = None,
        initial_component: str | None = None,
    ) -> None:
        super().__init__(
            title="ATLAS Setup Logs",
            default_size=(720, 480),
            transient_for=transient_for,
        )

        self._on_filter_changed = on_filter_changed
        self._on_component_changed = on_component_changed
        self._filter_level = initial_filter_level or logging.INFO
        self._components = [self.ALL_COMPONENTS_OPTION]
        if components:
            self._components.extend(list(components))
        self._selected_component = initial_component or self.ALL_COMPONENTS_OPTION
        self._suppress_filter_events = False
        self._suppress_component_events = False

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

        component_widget = self._build_component_widget(self._components)
        filter_widget = self._build_filter_widget(self.FILTER_LEVELS)
        self._set_component_selection(self._selected_component)
        self._set_filter_selection(self._filter_level)

        controls = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        controls.append(component_widget)
        controls.append(filter_widget)

        for pack in (getattr(header, "pack_end", None), getattr(header, "pack_start", None)):
            if callable(pack):
                try:
                    pack(controls)
                    break
                except Exception:  # pragma: no cover - GTK fallback
                    continue

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

    def set_filter_level(self, level: int) -> None:
        """Update the displayed filter without emitting callbacks."""

        self._filter_level = level
        self._set_filter_selection(level)

    def set_filter_component(self, component: str | None) -> None:
        """Update the displayed component without emitting callbacks."""

        self._selected_component = component or self.ALL_COMPONENTS_OPTION
        self._set_component_selection(self._selected_component)

    # -- internal helpers -------------------------------------------------

    def _build_filter_widget(self, levels: Iterable[tuple[str, int]]) -> Gtk.Widget:
        string_levels = list(levels)
        dropdown_cls = getattr(Gtk, "DropDown", None)
        combo_cls = getattr(Gtk, "ComboBoxText", None)

        if dropdown_cls is not None:
            dropdown = dropdown_cls.new_from_strings([label for label, _ in string_levels])
            dropdown.set_margin_end(8)
            dropdown.connect("notify::selected", self._on_dropdown_changed)
            if hasattr(dropdown, "set_tooltip_text"):
                dropdown.set_tooltip_text("Filter visible log entries by level")
            self._filter_widget = dropdown
            return dropdown

        combo: Gtk.Widget | None = None
        if combo_cls is not None:
            try:
                combo = combo_cls()
            except Exception:  # pragma: no cover - GTK stubs
                combo = None

        if combo is None:
            combo = Gtk.Label(label="Filter")
            self._filter_widget = combo
            return combo

        for label, _ in string_levels:
            combo.append_text(label)
        combo.set_margin_end(8)
        try:
            combo.connect("changed", self._on_combo_changed)
        except Exception:  # pragma: no cover - GTK stubs
            pass
        if hasattr(combo, "set_tooltip_text"):
            combo.set_tooltip_text("Filter visible log entries by level")
        self._filter_widget = combo
        return combo

    def _build_component_widget(self, components: Iterable[str]) -> Gtk.Widget:
        component_names = list(components)
        dropdown_cls = getattr(Gtk, "DropDown", None)
        combo_cls = getattr(Gtk, "ComboBoxText", None)

        if dropdown_cls is not None:
            dropdown = dropdown_cls.new_from_strings(component_names)
            dropdown.set_margin_end(8)
            dropdown.connect("notify::selected", self._on_component_dropdown_changed)
            if hasattr(dropdown, "set_tooltip_text"):
                dropdown.set_tooltip_text("Filter visible log entries by component")
            self._component_widget = dropdown
            return dropdown

        combo: Gtk.Widget | None = None
        if combo_cls is not None:
            try:
                combo = combo_cls()
            except Exception:  # pragma: no cover - GTK stubs
                combo = None

        if combo is None:
            combo = Gtk.Label(label="Components")
            self._component_widget = combo
            return combo

        for name in component_names:
            combo.append_text(name)
        combo.set_margin_end(8)
        try:
            combo.connect("changed", self._on_component_combo_changed)
        except Exception:  # pragma: no cover - GTK stubs
            pass
        if hasattr(combo, "set_tooltip_text"):
            combo.set_tooltip_text("Filter visible log entries by component")
        self._component_widget = combo
        return combo

    def _on_dropdown_changed(self, widget: Gtk.Widget, *_: object) -> None:
        if self._suppress_filter_events:
            return
        selected = getattr(widget, "get_selected", None)
        if callable(selected):
            try:
                index = int(selected())
            except Exception:
                return
            self._emit_filter_change(index)

    def _on_combo_changed(self, widget: Gtk.Widget) -> None:
        if self._suppress_filter_events:
            return
        get_active = getattr(widget, "get_active", None)
        if callable(get_active):
            try:
                index = int(get_active())
            except Exception:
                return
            self._emit_filter_change(index)

    def _emit_filter_change(self, index: int) -> None:
        try:
            _, level = self.FILTER_LEVELS[index]
        except Exception:
            return
        self._filter_level = level
        if callable(self._on_filter_changed):
            try:
                self._on_filter_changed(level)
            except Exception:  # pragma: no cover - defensive
                pass

    def _on_component_dropdown_changed(self, widget: Gtk.Widget, *_: object) -> None:
        if self._suppress_component_events:
            return
        selected = getattr(widget, "get_selected", None)
        if callable(selected):
            try:
                index = int(selected())
            except Exception:
                return
            self._emit_component_change(index)

    def _on_component_combo_changed(self, widget: Gtk.Widget) -> None:
        if self._suppress_component_events:
            return
        get_active = getattr(widget, "get_active", None)
        if callable(get_active):
            try:
                index = int(get_active())
            except Exception:
                return
            self._emit_component_change(index)

    def _emit_component_change(self, index: int) -> None:
        try:
            component = self._components[index]
        except Exception:
            return
        self._selected_component = component
        selected_component = (
            None if component == self.ALL_COMPONENTS_OPTION else component
        )
        if callable(self._on_component_changed):
            try:
                self._on_component_changed(selected_component)
            except Exception:  # pragma: no cover - defensive
                pass

    def _set_filter_selection(self, level: int) -> None:
        index = next(
            (i for i, (_, stored_level) in enumerate(self.FILTER_LEVELS) if stored_level == level),
            None,
        )
        if index is None:
            return

        widget = getattr(self, "_filter_widget", None)
        if widget is None:
            return

        self._suppress_filter_events = True
        try:
            if hasattr(widget, "set_selected"):
                setter = getattr(widget, "set_selected", None)
                if callable(setter):
                    try:
                        setter(index)
                        return
                    except Exception:
                        pass
            if hasattr(widget, "set_active"):
                setter = getattr(widget, "set_active", None)
                if callable(setter):
                    try:
                        setter(index)
                    except Exception:
                        pass
        finally:
            self._suppress_filter_events = False

    def _set_component_selection(self, component: str) -> None:
        try:
            index = self._components.index(component)
        except ValueError:
            index = 0

        widget = getattr(self, "_component_widget", None)
        if widget is None:
            return

        self._suppress_component_events = True
        try:
            if hasattr(widget, "set_selected"):
                setter = getattr(widget, "set_selected", None)
                if callable(setter):
                    try:
                        setter(index)
                        return
                    except Exception:
                        pass
            if hasattr(widget, "set_active"):
                setter = getattr(widget, "set_active", None)
                if callable(setter):
                    try:
                        setter(index)
                    except Exception:
                        pass
        finally:
            self._suppress_component_events = False
