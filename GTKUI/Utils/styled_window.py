"""Shared GTK window helper for ATLAS UI components."""

from __future__ import annotations

from typing import Sequence
from types import MethodType

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import Gtk

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
                self.set_default_size(*default_size)
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
