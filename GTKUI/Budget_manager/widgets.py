"""Shared widget helpers for budget management UI."""

from __future__ import annotations

from decimal import Decimal
from typing import Optional, Sequence

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import Gtk  # type: ignore[import-untyped]


def format_currency(amount: Decimal, currency: str = "USD") -> str:
    """Format a Decimal amount as currency string."""
    if currency == "USD":
        return f"${amount:,.2f}"
    return f"{amount:,.2f} {currency}"


def format_percentage(value: float) -> str:
    """Format a float as percentage string."""
    return f"{value:.1f}%"


def create_stat_card(
    title: str,
    value: str,
    subtitle: Optional[str] = None,
    css_class: Optional[str] = None,
) -> Gtk.Box:
    """Create a statistic display card widget."""
    card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
    card.add_css_class("budget-stat-card")
    if css_class:
        card.add_css_class(css_class)
    card.set_margin_top(8)
    card.set_margin_bottom(8)
    card.set_margin_start(12)
    card.set_margin_end(12)

    title_label = Gtk.Label(label=title)
    title_label.set_xalign(0.0)
    title_label.add_css_class("stat-card-title")
    card.append(title_label)

    value_label = Gtk.Label(label=value)
    value_label.set_xalign(0.0)
    value_label.add_css_class("stat-card-value")
    card.append(value_label)

    if subtitle:
        subtitle_label = Gtk.Label(label=subtitle)
        subtitle_label.set_xalign(0.0)
        subtitle_label.add_css_class("stat-card-subtitle")
        card.append(subtitle_label)

    return card


def create_progress_bar(
    value: float,
    max_value: float = 100.0,
    *,
    show_text: bool = True,
    warning_threshold: float = 0.8,
    critical_threshold: float = 0.95,
) -> Gtk.Box:
    """Create a budget progress bar with threshold coloring."""
    container = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)

    progress = Gtk.ProgressBar()
    fraction = min(value / max_value, 1.0) if max_value > 0 else 0.0
    progress.set_fraction(fraction)

    if show_text:
        progress.set_show_text(True)
        progress.set_text(format_percentage(fraction * 100))

    # Apply threshold-based styling
    if fraction >= critical_threshold:
        progress.add_css_class("budget-critical")
    elif fraction >= warning_threshold:
        progress.add_css_class("budget-warning")
    else:
        progress.add_css_class("budget-normal")

    container.append(progress)
    return container


def create_badge(text: str, css_class: Optional[str] = None) -> Gtk.Label:
    """Create a badge label."""
    badge = Gtk.Label(label=text)
    badge.add_css_class("budget-badge")
    if css_class:
        badge.add_css_class(css_class)
    return badge


def create_section_header(title: str) -> Gtk.Box:
    """Create a section header with divider."""
    box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
    box.set_margin_top(16)
    box.set_margin_bottom(8)

    label = Gtk.Label(label=title)
    label.set_xalign(0.0)
    label.add_css_class("section-header")
    box.append(label)

    separator = Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL)
    separator.set_hexpand(True)
    separator.set_valign(Gtk.Align.CENTER)
    box.append(separator)

    return box


def clear_container(container: Gtk.Box) -> None:
    """Remove all children from a GTK container."""
    child = container.get_first_child()
    while child is not None:
        next_child = child.get_next_sibling()
        container.remove(child)
        child = next_child


def create_action_button(
    label: str,
    callback,
    *,
    icon_name: Optional[str] = None,
    css_class: Optional[str] = None,
    tooltip: Optional[str] = None,
) -> Gtk.Button:
    """Create an action button with optional icon."""
    if icon_name:
        btn = Gtk.Button()
        box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        icon = Gtk.Image.new_from_icon_name(icon_name)
        box.append(icon)
        box.append(Gtk.Label(label=label))
        btn.set_child(box)
    else:
        btn = Gtk.Button(label=label)

    btn.connect("clicked", callback)

    if css_class:
        btn.add_css_class(css_class)
    if tooltip:
        btn.set_tooltip_text(tooltip)

    return btn
