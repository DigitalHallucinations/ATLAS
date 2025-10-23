"""Reusable GTK widgets shared between task and job management workspaces."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import Gtk


def create_badge_container() -> Gtk.Widget:
    """Return a GTK container suitable for displaying badge style labels."""

    flow_class = getattr(Gtk, "FlowBox", None)
    if flow_class is None:
        return Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)

    container = flow_class()

    setter = getattr(container, "set_selection_mode", None)
    if callable(setter):
        try:
            setter(Gtk.SelectionMode.NONE)
        except Exception:  # pragma: no cover - GTK fallback
            pass

    setter = getattr(container, "set_max_children_per_line", None)
    if callable(setter):
        try:
            setter(6)
        except Exception:  # pragma: no cover - GTK fallback
            pass

    setter = getattr(container, "set_row_spacing", None)
    if callable(setter):
        try:
            setter(6)
        except Exception:  # pragma: no cover - GTK fallback
            pass

    setter = getattr(container, "set_column_spacing", None)
    if callable(setter):
        try:
            setter(6)
        except Exception:  # pragma: no cover - GTK fallback
            pass

    return container


def clear_container(container: Gtk.Widget) -> None:
    """Remove all children from ``container`` using GTK fallbacks where needed."""

    remover = getattr(container, "remove_all", None)
    if callable(remover):
        try:
            remover()
            return
        except Exception:  # pragma: no cover - GTK fallback
            pass

    children = getattr(container, "get_children", None)
    if callable(children):
        for child in list(children()):
            remove = getattr(container, "remove", None)
            if callable(remove):
                try:
                    remove(child)
                except Exception:  # pragma: no cover - GTK fallback
                    continue


def append_badge(container: Gtk.Widget, badge: Gtk.Widget) -> None:
    """Append ``badge`` to ``container`` with graceful GTK fallbacks."""

    inserter = getattr(container, "insert", None)
    if callable(inserter):
        try:
            inserter(badge, -1)
            return
        except Exception:  # pragma: no cover - GTK fallback
            pass

    appender = getattr(container, "append", None)
    if callable(appender):
        appender(badge)


def create_badge(text: str, css_classes: Sequence[str]) -> Gtk.Widget:
    """Create a badge style label widget."""

    label = Gtk.Label(label=text)
    label.set_xalign(0.0)
    label.set_wrap(False)
    try:
        label.add_css_class("tag-badge")
    except Exception:  # pragma: no cover - GTK fallback
        pass

    for css in css_classes:
        if css == "tag-badge":
            continue
        try:
            label.add_css_class(css)
        except Exception:  # pragma: no cover - GTK fallback
            continue

    return label


def sync_badge_section(
    container: Optional[Gtk.Widget],
    badges: Sequence[Tuple[str, Sequence[str]]],
    *,
    fallback: Optional[str] = None,
) -> None:
    """Populate ``container`` with ``badges`` and optional fallback text."""

    if container is None:
        return

    clear_container(container)

    has_badges = False
    for text, css_classes in badges:
        badge = create_badge(text, css_classes)
        append_badge(container, badge)
        has_badges = True

    if not has_badges and fallback:
        badge = create_badge(fallback, ("tag-badge", "status-unknown"))
        append_badge(container, badge)
        has_badges = True

    if hasattr(container, "set_visible"):
        container.set_visible(has_badges)


__all__ = [
    "append_badge",
    "clear_container",
    "create_badge",
    "create_badge_container",
    "sync_badge_section",
]

