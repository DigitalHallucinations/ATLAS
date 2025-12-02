"""Hardware preflight page builder."""

from __future__ import annotations

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import Gtk

if False:  # pragma: no cover - type checking forward ref
    from GTKUI.Setup.setup_wizard import SetupWizardWindow


def _build_score_row(title: str, value: str, score: int) -> Gtk.Widget:
    row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
    row.set_hexpand(True)

    label = Gtk.Label(label=title)
    label.set_xalign(0.0)
    label.set_hexpand(True)
    row.append(label)

    value_label = Gtk.Label(label=value)
    value_label.set_xalign(1.0)
    value_label.set_hexpand(True)
    row.append(value_label)

    score_label = Gtk.Label(label=f"Score: {score}/5")
    score_label.set_xalign(1.0)
    if hasattr(score_label, "add_css_class"):
        score_label.add_css_class("dim-label")
    row.append(score_label)

    return row


def build_preflight_page(wizard: "SetupWizardWindow") -> Gtk.Widget:
    profile = wizard._ensure_preflight_profile()
    recommended = wizard.controller.state.setup_recommended_mode or "eco"

    box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
    box.set_hexpand(True)
    box.set_vexpand(True)

    heading = Gtk.Label(label="Preflight check for performance guidance")
    heading.set_wrap(True)
    heading.set_xalign(0.0)
    if hasattr(heading, "add_css_class"):
        heading.add_css_class("heading")
    box.append(heading)

    summary = Gtk.Label(
        label=(
            "We scanned this machine's resources to suggest a PerformanceMode before "
            "you configure databases and providers."
        )
    )
    summary.set_wrap(True)
    summary.set_xalign(0.0)
    box.append(summary)

    recommendation = Gtk.Label(
        label=(
            f"System tier: {profile.tier.title()} — Recommended PerformanceMode: {recommended.title()}"
        )
    )
    recommendation.set_wrap(True)
    recommendation.set_xalign(0.0)
    if hasattr(recommendation, "add_css_class"):
        recommendation.add_css_class("success-label")
    box.append(recommendation)

    scores = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
    scores.set_hexpand(True)
    scores.set_vexpand(False)

    metrics = [
        ("CPU", f"{profile.cpu_cores} cores", profile.cpu_score),
        ("Memory", f"{profile.memory_gb} GiB", profile.memory_score),
        ("Disk", f"{profile.disk_free_gb} GiB free", profile.disk_score),
        ("GPU", f"{profile.gpu_count} detected", profile.gpu_score),
        ("Network", f"{profile.network_speed_mbps} Mbps", profile.network_score),
    ]

    for title, value, score in metrics:
        scores.append(_build_score_row(title, value, score))

    total_label = Gtk.Label(label=f"Total score: {profile.total_score}/25")
    total_label.set_xalign(1.0)
    scores.append(total_label)

    box.append(scores)

    wizard._register_instructions(
        box,
        (
            "Review the quick hardware scan. We use it to tailor defaults like database hosting "
            "tips and provider choices to your environment."
        ),
    )

    return box
