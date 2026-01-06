"""Calendar settings panel for user preferences.

Provides UI for configuring calendar defaults, sync behavior, and display options.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import GLib, Gtk # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


class CalendarSettingsPanel(Gtk.Box):
    """Settings panel for calendar preferences."""

    def __init__(
        self,
        atlas: Any = None,
        on_settings_changed: Optional[Callable[[], None]] = None,
    ) -> None:
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=16)
        self.ATLAS = atlas
        self._on_settings_changed = on_settings_changed

        self.set_margin_top(16)
        self.set_margin_bottom(16)
        self.set_margin_start(16)
        self.set_margin_end(16)

        self._build_ui()
        self._load_settings()

    def _build_ui(self) -> None:
        """Build the settings UI."""
        # Default View Section
        view_section = self._create_section("Default View")

        view_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        view_row.set_margin_start(12)

        view_label = Gtk.Label(label="Default calendar view:")
        view_label.set_xalign(0.0)
        view_label.set_hexpand(True)
        view_row.append(view_label)

        self._view_combo = Gtk.ComboBoxText()
        self._view_combo.append("month", "Month")
        self._view_combo.append("week", "Week")
        self._view_combo.append("day", "Day")
        self._view_combo.append("agenda", "Agenda")
        self._view_combo.set_active_id("month")
        self._view_combo.connect("changed", self._on_setting_changed)
        view_row.append(self._view_combo)

        view_section.append(view_row)
        self.append(view_section)

        # Week Start Section
        week_section = self._create_section("Week Settings")

        week_start_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        week_start_row.set_margin_start(12)

        week_start_label = Gtk.Label(label="Week starts on:")
        week_start_label.set_xalign(0.0)
        week_start_label.set_hexpand(True)
        week_start_row.append(week_start_label)

        self._week_start_combo = Gtk.ComboBoxText()
        self._week_start_combo.append("0", "Monday")
        self._week_start_combo.append("6", "Sunday")
        self._week_start_combo.set_active_id("6")  # Default Sunday
        self._week_start_combo.connect("changed", self._on_setting_changed)
        week_start_row.append(self._week_start_combo)

        week_section.append(week_start_row)
        self.append(week_section)

        # Reminders Section
        reminder_section = self._create_section("Reminders")

        default_reminder_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        default_reminder_row.set_margin_start(12)

        reminder_label = Gtk.Label(label="Default reminder before event:")
        reminder_label.set_xalign(0.0)
        reminder_label.set_hexpand(True)
        default_reminder_row.append(reminder_label)

        self._reminder_combo = Gtk.ComboBoxText()
        self._reminder_combo.append("0", "None")
        self._reminder_combo.append("5", "5 minutes")
        self._reminder_combo.append("10", "10 minutes")
        self._reminder_combo.append("15", "15 minutes")
        self._reminder_combo.append("30", "30 minutes")
        self._reminder_combo.append("60", "1 hour")
        self._reminder_combo.append("1440", "1 day")
        self._reminder_combo.set_active_id("15")  # Default 15 min
        self._reminder_combo.connect("changed", self._on_setting_changed)
        default_reminder_row.append(self._reminder_combo)

        reminder_section.append(default_reminder_row)
        self.append(reminder_section)

        # Sync Section
        sync_section = self._create_section("Sync Settings")

        auto_sync_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        auto_sync_row.set_margin_start(12)

        auto_sync_label = Gtk.Label(label="Auto-sync external calendars")
        auto_sync_label.set_xalign(0.0)
        auto_sync_label.set_hexpand(True)
        auto_sync_row.append(auto_sync_label)

        self._auto_sync_switch = Gtk.Switch()
        self._auto_sync_switch.set_active(True)
        self._auto_sync_switch.connect("notify::active", self._on_setting_changed)
        auto_sync_row.append(self._auto_sync_switch)

        sync_section.append(auto_sync_row)

        sync_interval_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        sync_interval_row.set_margin_start(12)

        sync_interval_label = Gtk.Label(label="Sync interval:")
        sync_interval_label.set_xalign(0.0)
        sync_interval_label.set_hexpand(True)
        sync_interval_row.append(sync_interval_label)

        self._sync_interval_combo = Gtk.ComboBoxText()
        self._sync_interval_combo.append("5", "5 minutes")
        self._sync_interval_combo.append("15", "15 minutes")
        self._sync_interval_combo.append("30", "30 minutes")
        self._sync_interval_combo.append("60", "1 hour")
        self._sync_interval_combo.append("240", "4 hours")
        self._sync_interval_combo.set_active_id("15")
        self._sync_interval_combo.connect("changed", self._on_setting_changed)
        sync_interval_row.append(self._sync_interval_combo)

        sync_section.append(sync_interval_row)

        conflict_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        conflict_row.set_margin_start(12)

        conflict_label = Gtk.Label(label="Conflict resolution:")
        conflict_label.set_xalign(0.0)
        conflict_label.set_hexpand(True)
        conflict_row.append(conflict_label)

        self._conflict_combo = Gtk.ComboBoxText()
        self._conflict_combo.append("ask", "Ask me")
        self._conflict_combo.append("local_wins", "Local changes win")
        self._conflict_combo.append("remote_wins", "Remote changes win")
        self._conflict_combo.append("newest_wins", "Newest changes win")
        self._conflict_combo.set_active_id("ask")
        self._conflict_combo.connect("changed", self._on_setting_changed)
        conflict_row.append(self._conflict_combo)

        sync_section.append(conflict_row)
        self.append(sync_section)

        # Work Hours Section
        work_section = self._create_section("Work Hours")

        work_hours_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        work_hours_row.set_margin_start(12)

        work_start_label = Gtk.Label(label="Work day starts:")
        work_start_label.set_xalign(0.0)
        work_hours_row.append(work_start_label)

        self._work_start_spin = Gtk.SpinButton()
        adjustment_start = Gtk.Adjustment(value=9, lower=0, upper=23, step_increment=1)
        self._work_start_spin.set_adjustment(adjustment_start)
        self._work_start_spin.connect("value-changed", self._on_setting_changed)
        work_hours_row.append(self._work_start_spin)

        work_end_label = Gtk.Label(label="ends:")
        work_end_label.set_margin_start(12)
        work_hours_row.append(work_end_label)

        self._work_end_spin = Gtk.SpinButton()
        adjustment_end = Gtk.Adjustment(value=17, lower=0, upper=23, step_increment=1)
        self._work_end_spin.set_adjustment(adjustment_end)
        self._work_end_spin.connect("value-changed", self._on_setting_changed)
        work_hours_row.append(self._work_end_spin)

        # Spacer
        spacer = Gtk.Box()
        spacer.set_hexpand(True)
        work_hours_row.append(spacer)

        work_section.append(work_hours_row)

        work_days_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        work_days_row.set_margin_start(12)

        work_days_label = Gtk.Label(label="Work days:")
        work_days_label.set_xalign(0.0)
        work_days_row.append(work_days_label)

        self._work_day_checks: dict[str, Gtk.CheckButton] = {}
        for day_name, day_abbr in [
            ("Mon", "1"),
            ("Tue", "2"),
            ("Wed", "3"),
            ("Thu", "4"),
            ("Fri", "5"),
            ("Sat", "6"),
            ("Sun", "0"),
        ]:
            check = Gtk.CheckButton(label=day_name)
            # Default: Mon-Fri
            check.set_active(day_abbr in ["1", "2", "3", "4", "5"])
            check.connect("toggled", self._on_setting_changed)
            self._work_day_checks[day_abbr] = check
            work_days_row.append(check)

        work_section.append(work_days_row)
        self.append(work_section)

        # Actions
        action_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        action_box.set_margin_top(16)
        action_box.set_halign(Gtk.Align.END)

        reset_button = Gtk.Button(label="Reset to Defaults")
        reset_button.connect("clicked", self._on_reset_clicked)
        action_box.append(reset_button)

        save_button = Gtk.Button(label="Save Settings")
        save_button.add_css_class("suggested-action")
        save_button.connect("clicked", self._on_save_clicked)
        action_box.append(save_button)

        self.append(action_box)

        # Status
        self._status_label = Gtk.Label(label="")
        self._status_label.set_xalign(0.0)
        self._status_label.add_css_class("dim-label")
        self.append(self._status_label)

    def _create_section(self, title: str) -> Gtk.Box:
        """Create a settings section with a title."""
        section = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)

        header = Gtk.Label(label=title)
        header.set_xalign(0.0)
        header.add_css_class("heading")
        section.append(header)

        return section

    def _load_settings(self) -> None:
        """Load settings from config or storage."""
        if self.ATLAS is None:
            return

        try:
            config = getattr(self.ATLAS, "config", None)
            if config is None:
                return

            calendar_config = config.get("calendar", {})

            # Default view
            default_view = calendar_config.get("default_view", "month")
            self._view_combo.set_active_id(default_view)

            # Reminder
            default_reminder = str(calendar_config.get("default_reminder_minutes", 15))
            self._reminder_combo.set_active_id(default_reminder)

            # Sync settings
            sync_config = calendar_config.get("sync", {})
            self._auto_sync_switch.set_active(sync_config.get("auto_sync", True))

            sync_interval = str(sync_config.get("sync_interval_minutes", 15))
            self._sync_interval_combo.set_active_id(sync_interval)

            conflict = sync_config.get("conflict_resolution", "ask")
            self._conflict_combo.set_active_id(conflict)

            # Work hours
            work_config = calendar_config.get("work_hours", {})
            work_start = work_config.get("start", "09:00")
            work_end = work_config.get("end", "17:00")

            # Parse hour from "HH:MM"
            try:
                start_hour = int(work_start.split(":")[0])
                end_hour = int(work_end.split(":")[0])
                self._work_start_spin.set_value(start_hour)
                self._work_end_spin.set_value(end_hour)
            except (ValueError, IndexError):
                pass

            # Work days
            work_days = work_config.get("days", [1, 2, 3, 4, 5])
            for day_num, check in self._work_day_checks.items():
                check.set_active(int(day_num) in work_days)

        except Exception as e:
            logger.warning(f"Failed to load calendar settings: {e}")

    def _on_setting_changed(self, *_args) -> None:
        """Handle any setting change."""
        self._status_label.set_text("Settings modified (unsaved)")

    def _on_reset_clicked(self, _button: Gtk.Button) -> None:
        """Reset settings to defaults."""
        self._view_combo.set_active_id("month")
        self._week_start_combo.set_active_id("6")
        self._reminder_combo.set_active_id("15")
        self._auto_sync_switch.set_active(True)
        self._sync_interval_combo.set_active_id("15")
        self._conflict_combo.set_active_id("ask")
        self._work_start_spin.set_value(9)
        self._work_end_spin.set_value(17)

        for day_num, check in self._work_day_checks.items():
            check.set_active(day_num in ["1", "2", "3", "4", "5"])

        self._status_label.set_text("Settings reset to defaults (unsaved)")

    def _on_save_clicked(self, _button: Gtk.Button) -> None:
        """Save current settings."""
        if self.ATLAS is None:
            self._status_label.set_text("Cannot save: ATLAS not available")
            return

        try:
            config = getattr(self.ATLAS, "config", None)
            if config is None:
                self._status_label.set_text("Cannot save: Config not available")
                return

            # Build calendar config
            work_days = [
                int(day_num)
                for day_num, check in self._work_day_checks.items()
                if check.get_active()
            ]

            calendar_config = {
                "default_view": self._view_combo.get_active_id() or "month",
                "default_reminder_minutes": int(
                    self._reminder_combo.get_active_id() or "15"
                ),
                "work_hours": {
                    "start": f"{int(self._work_start_spin.get_value()):02d}:00",
                    "end": f"{int(self._work_end_spin.get_value()):02d}:00",
                    "days": work_days,
                },
                "sync": {
                    "auto_sync": self._auto_sync_switch.get_active(),
                    "sync_interval_minutes": int(
                        self._sync_interval_combo.get_active_id() or "15"
                    ),
                    "conflict_resolution": self._conflict_combo.get_active_id() or "ask",
                },
            }

            # Update config
            if isinstance(config, dict):
                config["calendar"] = calendar_config
            else:
                # Try to update if it has an update method
                update_fn = getattr(config, "update", None)
                if callable(update_fn):
                    update_fn({"calendar": calendar_config})

            # Trigger save if available
            save_config = getattr(self.ATLAS, "save_config", None)
            if callable(save_config):
                save_config()

            self._status_label.set_text("Settings saved")

            if self._on_settings_changed:
                self._on_settings_changed()

        except Exception as e:
            logger.error(f"Failed to save calendar settings: {e}")
            self._status_label.set_text(f"Save failed: {e}")

    def get_settings(self) -> dict:
        """Get current settings as a dictionary."""
        work_days = [
            int(day_num)
            for day_num, check in self._work_day_checks.items()
            if check.get_active()
        ]

        return {
            "default_view": self._view_combo.get_active_id() or "month",
            "week_start": int(self._week_start_combo.get_active_id() or "6"),
            "default_reminder_minutes": int(
                self._reminder_combo.get_active_id() or "15"
            ),
            "auto_sync": self._auto_sync_switch.get_active(),
            "sync_interval_minutes": int(
                self._sync_interval_combo.get_active_id() or "15"
            ),
            "conflict_resolution": self._conflict_combo.get_active_id() or "ask",
            "work_start_hour": int(self._work_start_spin.get_value()),
            "work_end_hour": int(self._work_end_spin.get_value()),
            "work_days": work_days,
        }
