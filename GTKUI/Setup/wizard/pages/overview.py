"""Overview page builder for the setup wizard."""

from __future__ import annotations

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import Gtk

if False:  # pragma: no cover - type checking forward ref
    from GTKUI.Setup.setup_wizard import SetupWizardWindow


def build_overview_page(wizard: "SetupWizardWindow") -> Gtk.Widget:
    box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
    box.set_hexpand(True)
    box.set_vexpand(True)

    heading = Gtk.Label(label="Welcome! Let's get ATLAS ready")
    heading.set_wrap(True)
    heading.set_xalign(0.0)
    if hasattr(heading, "add_css_class"):
        heading.add_css_class("heading")
    box.append(heading)

    summary = Gtk.Label(
        label=(
            "This short walkthrough gathers the essentials so your deployment starts "
            "with sensible defaults. We'll pause along the way to explain what each "
            "choice does."
        )
    )
    summary.set_wrap(True)
    summary.set_xalign(0.0)
    box.append(summary)

    reassurance = Gtk.Label(
        label=(
            "Your answers save automatically—come back to any step from the sidebar "
            "whenever you need to tweak something or resume later."
        )
    )
    reassurance.set_wrap(True)
    reassurance.set_xalign(0.0)
    box.append(reassurance)

    why_callout = wizard._create_overview_callout(
        "Why this matters",
        [
            "Give ATLAS an owner who can finish setup and invite others.",
            "Connect the services that keep conversations safe and responsive.",
            "Set expectations now so future teammates know what was chosen.",
        ],
    )
    box.append(why_callout)

    needs_callout = wizard._create_overview_callout(
        "What you'll need",
        [
            "Contact details for the first administrator.",
            "Connection info for your conversation store (PostgreSQL, SQLite, or MongoDB/Atlas) and supporting services.",
            "API keys or credentials for any model providers you plan to use.",
        ],
    )
    box.append(needs_callout)

    hosting_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
    hosting_box.set_hexpand(False)
    hosting_label = Gtk.Label(
        label=(
            "Here are quick hosting tips for the database, vector store, and model"
            " inference choices you make throughout setup. Update the forms to refresh"
            " the suggestions based on your selections."
        )
    )
    hosting_label.set_wrap(True)
    hosting_label.set_xalign(0.0)
    hosting_box.append(hosting_label)

    hosting_hint = Gtk.Label()
    hosting_hint.set_wrap(True)
    hosting_hint.set_xalign(0.0)
    wizard._hosting_hint_label = hosting_hint
    wizard._refresh_hosting_summary()
    hosting_box.append(hosting_hint)
    box.append(hosting_box)

    cli_label = Gtk.Label(
        label=(
            "Prefer a terminal instead? Run scripts/setup_atlas.py to pick up the "
            "same guided flow from the command line."
        )
    )
    cli_label.set_wrap(True)
    cli_label.set_xalign(0.0)
    box.append(cli_label)

    wizard._register_instructions(
        box,
        (
            "Glance through the overview and check the two callouts so you know what "
            "we'll ask for before you continue to the administrator details."
        ),
    )
    wizard._register_instructions(
        why_callout,
        "Use this to align the setup goals with anyone joining you for the rollout.",
    )
    wizard._register_instructions(
        needs_callout,
        "Gather these items now so the next few forms go quickly.",
    )
    summary_text = hosting_hint.get_text() or ""
    wizard._register_instructions(hosting_hint, summary_text)
    wizard._register_instructions(
        cli_label,
        "You can swap to the terminal helper at any point—the wizard keeps your progress in sync.",
    )

    return box
