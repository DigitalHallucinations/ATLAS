"""Shared GTK window helper for ATLAS UI components."""

from __future__ import annotations

from typing import Sequence
from types import MethodType

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import Gtk, Gdk

from .utils import apply_css


class AtlasWindow(Gtk.Window):
    """GTK window base class that applies the ATLAS shared styling."""

    def __init__(
        self,
        *,
        title: str | None = None,
        default_size: tuple[int, int] | None = None,
        modal: bool | None = None,
        transient_for: Gtk.Window | None = None,
        css_classes: Sequence[str] | None = None,
        apply_styles: bool = True,
    ) -> None:
        super().__init__(title=title)

        if default_size is not None:
            try:
                safe_width, safe_height = self._calculate_safe_size(*default_size)
                self.set_default_size(safe_width, safe_height)
            except Exception:  # pragma: no cover - defensive for stub environments
                pass

        if modal is not None:
            try:
                self.set_modal(modal)
            except Exception:  # pragma: no cover - defensive for stub environments
                pass

        if transient_for is not None:
            try:
                self.set_transient_for(transient_for)
            except Exception:  # pragma: no cover
                pass

        if apply_styles:
            try:
                apply_css()
            except Exception:  # pragma: no cover - styling is best-effort in tests
                pass

        self._apply_style_classes(css_classes)
        self._ensure_set_child()
        self._ensure_signal_support()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _apply_style_classes(self, css_classes: Sequence[str] | None) -> None:
        context = getattr(self, "get_style_context", lambda: None)()
        if context is None:
            return

        for class_name in ("chat-page", "sidebar"):
            try:
                context.add_class(class_name)
            except Exception:  # pragma: no cover - GTK stubs in unit tests
                continue

        if css_classes is None:
            return

        for class_name in css_classes:
            try:
                context.add_class(class_name)
            except Exception:  # pragma: no cover
                continue

    def _ensure_set_child(self) -> None:
        if hasattr(self, "set_child"):
            return

        def _set_child_fallback(_self, child):
            setattr(_self, "_fallback_child", child)

        setattr(self, "set_child", MethodType(_set_child_fallback, self))

    def _ensure_signal_support(self) -> None:
        if not hasattr(self, "_signal_handlers"):
            setattr(self, "_signal_handlers", {})

        if hasattr(self, "connect"):
            return

        def _connect_fallback(_self, signal_name, callback, *args):
            handlers = _self._signal_handlers.setdefault(signal_name, [])
            handlers.append((callback, args))
            return len(handlers) - 1

        setattr(self, "connect", MethodType(_connect_fallback, self))

    # ------------------------------------------------------------------
    # Window sizing helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _clamp_dimension(desired: int, available: int, margin: int) -> int:
        """Clamp a dimension so it never exceeds the available space."""

        usable_space = max(1, available - max(margin, 0))
        return min(desired, usable_space)

    def _calculate_safe_size(self, desired_width: int, desired_height: int) -> tuple[int, int]:
        """Clamp the window size so it never exceeds the primary monitor."""

        margin = 64
        monitor_size = self._get_primary_monitor_size()
        if monitor_size is None:
            return desired_width, desired_height

        monitor_width, monitor_height = monitor_size
        safe_width = self._clamp_dimension(desired_width, monitor_width, margin)
        safe_height = self._clamp_dimension(desired_height, monitor_height, margin)
        return safe_width, safe_height

    def _get_primary_monitor_size(self) -> tuple[int, int] | None:
        """Return the primary monitor geometry if available."""

        try:
            display = Gdk.Display.get_default()
        except Exception:  # pragma: no cover - GTK may be unavailable in tests
            return None

        if display is None:
            return None

        monitor = None
        try:
            monitor = display.get_primary_monitor()
        except Exception:  # pragma: no cover - display may not support the call
            monitor = None

        if monitor is None:
            try:
                surface = getattr(self, "get_surface", lambda: None)()
                if surface is not None:
                    monitor = display.get_monitor_at_surface(surface)
            except Exception:  # pragma: no cover - fallback best effort
                monitor = None

        if monitor is None:
            return None

        try:
            geometry = monitor.get_geometry()
        except Exception:  # pragma: no cover - stub monitors may not support geometry
            return None

        width = getattr(geometry, "width", None)
        height = getattr(geometry, "height", None)
        if width is None or height is None:
            return None

        return int(width), int(height)
