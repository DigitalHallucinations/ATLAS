"""Setup type selection page builder."""

from __future__ import annotations

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import Gtk

if False:  # pragma: no cover - type checking forward ref
    from GTKUI.Setup.setup_wizard import SetupWizardWindow


def build_setup_type_page(wizard: "SetupWizardWindow") -> Gtk.Widget:
    box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
    box.set_hexpand(True)
    box.set_vexpand(True)

    heading = Gtk.Label(label="Choose how presets should shape the rest of setup")
    heading.set_wrap(True)
    heading.set_xalign(0.0)
    if hasattr(heading, "add_css_class"):
        heading.add_css_class("heading")
    box.append(heading)

    copy = Gtk.Label(
        label=(
            "Presets apply once to pre-fill the remaining forms. After you tweak a field manually,"
            " picking the same preset again leaves your edits alone."
        )
    )
    copy.set_wrap(True)
    copy.set_xalign(0.0)
    box.append(copy)

    action_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
    action_row.set_hexpand(True)
    docs_button = Gtk.Button(label="View docs")
    docs_button.set_halign(Gtk.Align.START)
    if hasattr(docs_button, "set_tooltip_text"):
        docs_button.set_tooltip_text("Open the embedded setup documentation")
    if hasattr(docs_button, "set_receives_default"):
        docs_button.set_receives_default(True)
    docs_button.connect("clicked", lambda _button: wizard.navigate_to_docs_browser())
    action_row.append(docs_button)
    box.append(action_row)

    button_column = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
    button_column.set_hexpand(False)

    radio_descriptors = [
        (
            "student",
            "Student",
            "Free tier for learners with guidance-focused defaults and usage limits.",
        ),
        (
            "personal",
            "Personal",
            "Everyday users with tiered scaling by user count.",
        ),
        (
            "enthusiast",
            "Enthusiast",
            "Power users with all features unlocked, including experimental capabilities.",
        ),
        (
            "enterprise",
            "Enterprise",
            "Team rollout with Redis, schedulers, and stricter retention defaults.",
        ),
        (
            "regulatory",
            "Regulatory",
            "Enterprise with extended retention, residency controls, and compliance auditing.",
        ),
    ]

    first_button: Gtk.CheckButton | None = None
    for key, title, subtitle in radio_descriptors:
        row = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        toggle = Gtk.CheckButton.new_with_label(title)
        if first_button is None:
            first_button = toggle
        else:
            toggle.set_group(first_button)
        toggle.set_hexpand(False)
        toggle.connect("toggled", wizard._on_setup_type_toggled, key)
        wizard._setup_type_buttons[key] = toggle
        row.append(toggle)

        if subtitle:
            detail = Gtk.Label(label=subtitle)
            detail.set_wrap(True)
            detail.set_xalign(0.0)
            if hasattr(detail, "add_css_class"):
                detail.add_css_class("dim-label")
            row.append(detail)

        button_column.append(row)

    box.append(button_column)

    # Developer mode toggle (available on any tier)
    developer_mode_toggle = Gtk.CheckButton(
        label="Enable Developer mode (local Redis and PostgreSQL for production-like testing)"
    )
    if hasattr(developer_mode_toggle, "set_tooltip_text"):
        developer_mode_toggle.set_tooltip_text(
            "Overlay developer-friendly defaults on any tier: local Redis streams, "
            "PostgreSQL job store, and verbose logging for debugging and development."
        )
    developer_mode_toggle.set_halign(Gtk.Align.START)
    developer_mode_toggle.connect("toggled", wizard._on_developer_mode_toggled)
    wizard._developer_mode_toggle = developer_mode_toggle
    box.append(developer_mode_toggle)

    table = Gtk.Grid(column_spacing=12, row_spacing=6)
    table.set_hexpand(True)
    table.set_vexpand(False)

    headers = ["Area", "Student", "Personal", "Enthusiast", "Enterprise"]
    for col, title in enumerate(headers):
        label = Gtk.Label(label=title)
        label.set_wrap(True)
        label.set_xalign(0.0)
        if hasattr(label, "add_css_class"):
            label.add_css_class("heading")
        table.attach(label, col, 0, 1, 1)
        if col == 1:
            wizard._setup_type_headers["student"] = label
        elif col == 2:
            wizard._setup_type_headers["personal"] = label
        elif col == 3:
            wizard._setup_type_headers["enthusiast"] = label
        elif col == 4:
            wizard._setup_type_headers["enterprise"] = label

    rows = [
        (
            "Message bus",
            "In-memory queue.",
            "In-memory queue.",
            "Local Redis streams.",
            "Redis with shared streams.",
        ),
        (
            "Job scheduling",
            "Disabled.",
            "Disabled.",
            "Enabled with PostgreSQL.",
            "Enabled with dedicated store.",
        ),
        (
            "Key-value store",
            "Reuse conversation DB.",
            "Reuse conversation DB.",
            "Separate PostgreSQL cache.",
            "Separate database cache.",
        ),
        (
            "Retention",
            "7 days, 100 messages.",
            "No limits pre-set.",
            "90 days, 1000 messages.",
            "30 days, 500 messages.",
        ),
        (
            "HTTP server",
            "Auto-start.",
            "Auto-start.",
            "Auto-start.",
            "Manual start.",
        ),
        (
            "Pricing",
            "Free.",
            "Tiered by users.",
            "Paid, all features.",
            "Paid, team features.",
        ),
    ]

    for row_index, (area, student, personal, enthusiast, enterprise) in enumerate(rows, start=1):
        for column, text in enumerate((area, student, personal, enthusiast, enterprise)):
            label = Gtk.Label(label=text)
            label.set_wrap(True)
            label.set_xalign(0.0)
            table.attach(label, column, row_index, 1, 1)

    box.append(table)

    personal_cap_hint = Gtk.Label(
        label=(
            "Personal mode supports up to 5 local profiles. Upgrade to Enterprise to add "
            "more seats and unlock tenancy controls."
        )
    )
    personal_cap_hint.set_wrap(True)
    personal_cap_hint.set_xalign(0.0)
    personal_cap_hint.set_visible(False)
    if hasattr(personal_cap_hint, "add_css_class"):
        personal_cap_hint.add_css_class("dim-label")
    box.append(personal_cap_hint)
    wizard._personal_cap_hint = personal_cap_hint

    local_only_toggle = Gtk.CheckButton(
        label="Keep data on this device (SQLite and in-memory queues)"
    )
    if hasattr(local_only_toggle, "set_tooltip_text"):
        local_only_toggle.set_tooltip_text(
            "Disable remote backends and external connectors for a single-device setup."
        )
    local_only_toggle.set_halign(Gtk.Align.START)
    local_only_toggle.connect("toggled", wizard._on_local_mode_toggled)
    wizard._local_only_toggle = local_only_toggle
    box.append(local_only_toggle)

    instructions = (
        "Pick the preset that matches your rollout. You can always override fields later or"
        " switch presets if plans change."
    )

    wizard._register_instructions(box, instructions)
    wizard._register_instructions(
        table,
        "Scan this comparison so you know which downstream defaults will update when you choose a preset.",
    )

    wizard._sync_setup_type_selection()

    return box
