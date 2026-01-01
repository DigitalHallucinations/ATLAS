"""Documentation browser page builder for Setup wizard.

Uses the shared docs_factory for rendering.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import Gtk

from GTKUI.Docs.docs_factory import (
    create_web_view,
    resolve_default_doc,
)

if TYPE_CHECKING:
    from GTKUI.Setup.setup_wizard import SetupWizardWindow


def build_docs_browser_page(wizard: "SetupWizardWindow") -> Gtk.Widget:
    """Build the documentation browser page for the setup wizard.
    
    Args:
        wizard: The parent setup wizard window.
        
    Returns:
        GTK Widget containing the documentation browser.
    """
    doc_path, doc_hint = resolve_default_doc()

    container = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
    container.set_hexpand(True)
    container.set_vexpand(True)
    container.set_halign(Gtk.Align.FILL)

    heading = Gtk.Label(label="Browse setup documentation")
    heading.set_wrap(True)
    heading.set_xalign(0.0)
    if hasattr(heading, "add_css_class"):
        heading.add_css_class("heading")
    container.append(heading)

    subtitle = Gtk.Label(
        label=(
            "Read the install guide without leaving setup. The embedded view tries to load local docs "
            "automatically and falls back to a placeholder summary if nothing is bundled yet."
        )
    )
    subtitle.set_wrap(True)
    subtitle.set_xalign(0.0)
    container.append(subtitle)

    hint_label = Gtk.Label(label=doc_hint)
    hint_label.set_wrap(True)
    hint_label.set_xalign(0.0)
    if hasattr(hint_label, "add_css_class"):
        hint_label.add_css_class("dim-label")
    container.append(hint_label)

    viewer = create_web_view(doc_path)
    if isinstance(viewer, Gtk.Widget):
        viewer.set_hexpand(True)
        viewer.set_vexpand(True)
        viewer.set_halign(Gtk.Align.FILL)
    container.append(viewer)

    wizard._register_instructions(
        container,
        "Browse the embedded documentation to confirm prerequisites or find answers while keeping your place in the setup flow.",
    )

    return container
