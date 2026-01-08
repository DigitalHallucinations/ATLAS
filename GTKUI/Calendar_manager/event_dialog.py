"""Event add/edit dialog for ATLAS Calendar.

Provides a comprehensive dialog for creating and editing calendar events
with support for recurrence rules, reminders, and category assignment.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, date, time, timedelta
from typing import Any, Dict, List, Optional
from uuid import UUID

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import Gtk, GLib

logger = logging.getLogger(__name__)

# Recurrence frequency options
FREQUENCY_OPTIONS = [
    ("none", "Does not repeat"),
    ("daily", "Daily"),
    ("weekly", "Weekly"),
    ("monthly", "Monthly"),
    ("yearly", "Yearly"),
    ("weekdays", "Every weekday (Mon-Fri)"),
    ("custom", "Custom..."),
]

# Reminder preset options
REMINDER_PRESETS = [
    (0, "At time of event"),
    (5, "5 minutes before"),
    (10, "10 minutes before"),
    (15, "15 minutes before"),
    (30, "30 minutes before"),
    (60, "1 hour before"),
    (120, "2 hours before"),
    (1440, "1 day before"),
    (2880, "2 days before"),
    (10080, "1 week before"),
]

# Event status options
STATUS_OPTIONS = [
    ("confirmed", "Confirmed"),
    ("tentative", "Tentative"),
    ("cancelled", "Cancelled"),
]

# Visibility options
VISIBILITY_OPTIONS = [
    ("public", "Public"),
    ("private", "Private"),
    ("confidential", "Confidential"),
]

# Busy status options
BUSY_STATUS_OPTIONS = [
    ("busy", "Busy"),
    ("free", "Free"),
    ("tentative", "Tentative"),
    ("out_of_office", "Out of Office"),
]


class EventDialog(Gtk.Dialog):
    """Dialog for creating or editing a calendar event."""

    def __init__(
        self,
        parent: Optional[Gtk.Window] = None,
        atlas: Any = None,
        mode: str = "add",
        event_data: Optional[Dict[str, Any]] = None,
        default_category_id: Optional[str] = None,
        default_start: Optional[datetime] = None,
    ) -> None:
        """Initialize the event dialog.

        Args:
            parent: Parent window for modality
            atlas: ATLAS instance for services
            mode: "add" or "edit"
            event_data: Existing event data for edit mode
            default_category_id: Default category to pre-select
            default_start: Default start time for new events
        """
        title = "New Event" if mode == "add" else "Edit Event"
        super().__init__(
            title=title,
            transient_for=parent,
            modal=True,
        )

        self.ATLAS = atlas
        self._mode = mode
        self._event_data = event_data or {}
        self._default_category_id = default_category_id
        self._default_start = default_start or self._round_to_next_half_hour()

        # Widget references
        self._title_entry: Optional[Gtk.Entry] = None
        self._description_view: Optional[Gtk.TextView] = None
        self._location_entry: Optional[Gtk.Entry] = None
        self._all_day_switch: Optional[Gtk.Switch] = None
        self._start_date_button: Optional[Gtk.Button] = None
        self._start_time_entry: Optional[Gtk.Entry] = None
        self._end_date_button: Optional[Gtk.Button] = None
        self._end_time_entry: Optional[Gtk.Entry] = None
        self._category_dropdown: Optional[Gtk.DropDown] = None
        self._frequency_dropdown: Optional[Gtk.DropDown] = None
        self._reminder_box: Optional[Gtk.Box] = None
        self._status_dropdown: Optional[Gtk.DropDown] = None
        self._visibility_dropdown: Optional[Gtk.DropDown] = None
        self._busy_dropdown: Optional[Gtk.DropDown] = None
        self._url_entry: Optional[Gtk.Entry] = None
        self._linked_entities_box: Optional[Gtk.Box] = None
        self._linked_jobs_list: Optional[Gtk.Box] = None
        self._linked_tasks_list: Optional[Gtk.Box] = None

        # State
        self._start_date: date = self._default_start.date()
        self._start_time: time = self._default_start.time()
        self._end_date: date = self._start_date
        self._end_time: time = (self._default_start + timedelta(hours=1)).time()
        self._reminders: List[int] = [15]  # Default: 15 min before
        self._categories: List[Dict[str, Any]] = []

        self.set_default_size(550, -1)
        self.add_button("Cancel", Gtk.ResponseType.CANCEL)

        action_label = "Create" if mode == "add" else "Save"
        action_btn = self.add_button(action_label, Gtk.ResponseType.OK)
        action_btn.add_css_class("suggested-action")

        self._build_form()
        self._load_categories()

        if mode == "edit" and event_data:
            self._populate_from_event(event_data)

    def _round_to_next_half_hour(self) -> datetime:
        """Round current time to next half hour."""
        now = datetime.now()
        minutes = 30 if now.minute < 30 else 60
        return now.replace(minute=0, second=0, microsecond=0) + timedelta(minutes=minutes)

    def _build_form(self) -> None:
        """Build the dialog form."""
        content = self.get_content_area()
        content.set_margin_top(16)
        content.set_margin_bottom(16)
        content.set_margin_start(16)
        content.set_margin_end(16)
        content.set_spacing(12)

        # Scrolled container for long forms
        scrolled = Gtk.ScrolledWindow()
        scrolled.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        scrolled.set_max_content_height(500)
        scrolled.set_propagate_natural_height(True)

        form_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=16)

        # Title (prominent)
        self._title_entry = Gtk.Entry()
        self._title_entry.set_placeholder_text("Event title")
        self._title_entry.add_css_class("title-3")
        form_box.append(self._title_entry)

        # Date/Time section
        datetime_frame = self._build_datetime_section()
        form_box.append(datetime_frame)

        # Location
        location_row = self._build_field_row("Location", icon_name="mark-location-symbolic")
        self._location_entry = Gtk.Entry()
        self._location_entry.set_placeholder_text("Add location")
        self._location_entry.set_hexpand(True)
        location_row.append(self._location_entry)
        form_box.append(location_row)

        # Description
        desc_frame = Gtk.Frame()
        desc_frame.set_label("Description")
        desc_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        desc_box.set_margin_top(8)
        desc_box.set_margin_bottom(8)
        desc_box.set_margin_start(8)
        desc_box.set_margin_end(8)

        self._description_view = Gtk.TextView()
        self._description_view.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
        self._description_view.set_size_request(-1, 80)
        desc_box.append(self._description_view)
        desc_frame.set_child(desc_box)
        form_box.append(desc_frame)

        # Category
        category_row = self._build_field_row("Category", icon_name="view-list-symbolic")
        self._category_dropdown = Gtk.DropDown()
        self._category_dropdown.set_hexpand(True)
        category_row.append(self._category_dropdown)
        form_box.append(category_row)

        # Recurrence
        recurrence_row = self._build_field_row("Repeat", icon_name="view-refresh-symbolic")
        self._frequency_dropdown = self._build_frequency_dropdown()
        self._frequency_dropdown.set_hexpand(True)
        recurrence_row.append(self._frequency_dropdown)
        form_box.append(recurrence_row)

        # Reminders section
        reminder_frame = self._build_reminders_section()
        form_box.append(reminder_frame)

        # Advanced options (collapsible)
        advanced_expander = self._build_advanced_section()
        form_box.append(advanced_expander)

        # Linked entities section (only in edit mode)
        if self._mode == "edit":
            linked_expander = self._build_linked_entities_section()
            form_box.append(linked_expander)

        scrolled.set_child(form_box)
        content.append(scrolled)

    def _build_field_row(
        self, label: str, icon_name: Optional[str] = None
    ) -> Gtk.Box:
        """Build a form field row with label and optional icon."""
        row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)

        if icon_name:
            icon = Gtk.Image.new_from_icon_name(icon_name)
            icon.set_opacity(0.6)
            row.append(icon)

        lbl = Gtk.Label(label=label)
        lbl.set_xalign(0)
        lbl.set_size_request(80, -1)
        row.append(lbl)

        return row

    def _build_datetime_section(self) -> Gtk.Frame:
        """Build the date/time picker section."""
        frame = Gtk.Frame()
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        box.set_margin_top(8)
        box.set_margin_bottom(8)
        box.set_margin_start(8)
        box.set_margin_end(8)

        # All-day toggle
        all_day_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        all_day_label = Gtk.Label(label="All-day event")
        all_day_label.set_hexpand(True)
        all_day_label.set_xalign(0)
        all_day_row.append(all_day_label)

        self._all_day_switch = Gtk.Switch()
        self._all_day_switch.connect("notify::active", self._on_all_day_toggled)
        all_day_row.append(self._all_day_switch)
        box.append(all_day_row)

        # Separator
        box.append(Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL))

        # Start row
        start_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        start_label = Gtk.Label(label="Start")
        start_label.set_size_request(50, -1)
        start_label.set_xalign(0)
        start_row.append(start_label)

        self._start_date_button = Gtk.Button(label=self._format_date(self._start_date))
        self._start_date_button.connect("clicked", self._on_start_date_clicked)
        start_row.append(self._start_date_button)

        self._start_time_entry = Gtk.Entry()
        self._start_time_entry.set_text(self._format_time(self._start_time))
        self._start_time_entry.set_max_width_chars(8)
        self._start_time_entry.set_placeholder_text("HH:MM")
        self._start_time_entry.connect("changed", self._on_start_time_changed)
        start_row.append(self._start_time_entry)

        box.append(start_row)

        # End row
        end_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        end_label = Gtk.Label(label="End")
        end_label.set_size_request(50, -1)
        end_label.set_xalign(0)
        end_row.append(end_label)

        self._end_date_button = Gtk.Button(label=self._format_date(self._end_date))
        self._end_date_button.connect("clicked", self._on_end_date_clicked)
        end_row.append(self._end_date_button)

        self._end_time_entry = Gtk.Entry()
        self._end_time_entry.set_text(self._format_time(self._end_time))
        self._end_time_entry.set_max_width_chars(8)
        self._end_time_entry.set_placeholder_text("HH:MM")
        end_row.append(self._end_time_entry)

        box.append(end_row)
        frame.set_child(box)
        return frame

    def _build_frequency_dropdown(self) -> Gtk.DropDown:
        """Build the recurrence frequency dropdown."""
        model = Gtk.StringList()
        for _, label in FREQUENCY_OPTIONS:
            model.append(label)

        dropdown = Gtk.DropDown(model=model)
        dropdown.set_selected(0)  # "Does not repeat"
        return dropdown

    def _build_reminders_section(self) -> Gtk.Frame:
        """Build the reminders section with add/remove capability."""
        frame = Gtk.Frame()
        frame.set_label("Reminders")

        self._reminder_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        self._reminder_box.set_margin_top(8)
        self._reminder_box.set_margin_bottom(8)
        self._reminder_box.set_margin_start(8)
        self._reminder_box.set_margin_end(8)

        # Add default reminder
        self._add_reminder_row(15)

        # Add button
        add_btn = Gtk.Button()
        add_btn.set_icon_name("list-add-symbolic")
        add_btn.set_tooltip_text("Add reminder")
        add_btn.set_halign(Gtk.Align.START)
        add_btn.connect("clicked", self._on_add_reminder_clicked)
        self._reminder_box.append(add_btn)

        frame.set_child(self._reminder_box)
        return frame

    def _add_reminder_row(self, minutes: int = 15) -> None:
        """Add a reminder row to the reminders section."""
        row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        row.add_css_class("reminder-row")

        # Dropdown for reminder time
        model = Gtk.StringList()
        selected_idx = 0
        for idx, (mins, label) in enumerate(REMINDER_PRESETS):
            model.append(label)
            if mins == minutes:
                selected_idx = idx

        dropdown = Gtk.DropDown(model=model)
        dropdown.set_selected(selected_idx)
        dropdown.set_hexpand(True)
        row.append(dropdown)

        # Remove button
        remove_btn = Gtk.Button()
        remove_btn.set_icon_name("list-remove-symbolic")
        remove_btn.set_tooltip_text("Remove reminder")
        remove_btn.connect("clicked", lambda b: self._remove_reminder_row(row))
        row.append(remove_btn)

        # Insert before the add button
        children = []
        child = self._reminder_box.get_first_child()
        while child:
            children.append(child)
            child = child.get_next_sibling()

        if children:
            # Insert before the last child (add button)
            self._reminder_box.insert_child_after(row, children[-2] if len(children) > 1 else None)
        else:
            self._reminder_box.append(row)

    def _remove_reminder_row(self, row: Gtk.Box) -> None:
        """Remove a reminder row."""
        self._reminder_box.remove(row)

    def _build_advanced_section(self) -> Gtk.Expander:
        """Build the advanced options section."""
        expander = Gtk.Expander(label="Advanced options")

        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        box.set_margin_top(8)
        box.set_margin_start(16)

        # Status
        status_row = self._build_field_row("Status")
        status_model = Gtk.StringList()
        for _, label in STATUS_OPTIONS:
            status_model.append(label)
        self._status_dropdown = Gtk.DropDown(model=status_model)
        self._status_dropdown.set_selected(0)
        self._status_dropdown.set_hexpand(True)
        status_row.append(self._status_dropdown)
        box.append(status_row)

        # Visibility
        vis_row = self._build_field_row("Visibility")
        vis_model = Gtk.StringList()
        for _, label in VISIBILITY_OPTIONS:
            vis_model.append(label)
        self._visibility_dropdown = Gtk.DropDown(model=vis_model)
        self._visibility_dropdown.set_selected(0)
        self._visibility_dropdown.set_hexpand(True)
        vis_row.append(self._visibility_dropdown)
        box.append(vis_row)

        # Busy status
        busy_row = self._build_field_row("Show as")
        busy_model = Gtk.StringList()
        for _, label in BUSY_STATUS_OPTIONS:
            busy_model.append(label)
        self._busy_dropdown = Gtk.DropDown(model=busy_model)
        self._busy_dropdown.set_selected(0)
        self._busy_dropdown.set_hexpand(True)
        busy_row.append(self._busy_dropdown)
        box.append(busy_row)

        # URL / Video call link
        url_row = self._build_field_row("URL", icon_name="web-browser-symbolic")
        self._url_entry = Gtk.Entry()
        self._url_entry.set_placeholder_text("Video call or event URL")
        self._url_entry.set_hexpand(True)
        url_row.append(self._url_entry)
        box.append(url_row)

        expander.set_child(box)
        return expander

    def _build_linked_entities_section(self) -> Gtk.Expander:
        """Build the linked jobs/tasks section for edit mode."""
        expander = Gtk.Expander(label="Linked Jobs & Tasks")

        self._linked_entities_box = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL, spacing=12
        )
        self._linked_entities_box.set_margin_top(8)
        self._linked_entities_box.set_margin_start(16)

        # Linked Jobs section
        jobs_label = Gtk.Label(label="Linked Jobs")
        jobs_label.set_xalign(0)
        jobs_label.add_css_class("heading")
        self._linked_entities_box.append(jobs_label)

        self._linked_jobs_list = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL, spacing=4
        )
        self._linked_jobs_list.set_margin_start(8)
        self._linked_entities_box.append(self._linked_jobs_list)

        # Placeholder for jobs
        jobs_placeholder = Gtk.Label(label="Loading...")
        jobs_placeholder.set_xalign(0)
        jobs_placeholder.add_css_class("dim-label")
        self._linked_jobs_list.append(jobs_placeholder)

        # Linked Tasks section
        tasks_label = Gtk.Label(label="Linked Tasks")
        tasks_label.set_xalign(0)
        tasks_label.add_css_class("heading")
        self._linked_entities_box.append(tasks_label)

        self._linked_tasks_list = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL, spacing=4
        )
        self._linked_tasks_list.set_margin_start(8)
        self._linked_entities_box.append(self._linked_tasks_list)

        # Placeholder for tasks
        tasks_placeholder = Gtk.Label(label="Loading...")
        tasks_placeholder.set_xalign(0)
        tasks_placeholder.add_css_class("dim-label")
        self._linked_tasks_list.append(tasks_placeholder)

        expander.set_child(self._linked_entities_box)

        # Load linked entities asynchronously
        self._load_linked_entities()

        return expander

    def _load_linked_entities(self) -> None:
        """Load linked jobs and tasks for the current event."""
        import asyncio

        async def fetch_and_display():
            event_id = self._event_data.get("id") or self._event_data.get("event_id")
            if not event_id or not self.ATLAS:
                self._update_linked_list(self._linked_jobs_list, [], "jobs")
                self._update_linked_list(self._linked_tasks_list, [], "tasks")
                return

            try:
                service = self.ATLAS.calendar_service
                actor = self.ATLAS.get_current_actor()

                # Fetch linked jobs
                jobs_result = await service.get_linked_jobs(actor, str(event_id))
                jobs = jobs_result.value if jobs_result.is_success else []
                GLib.idle_add(
                    self._update_linked_list,
                    self._linked_jobs_list,
                    jobs,
                    "jobs"
                )

                # Fetch linked tasks
                tasks_result = await service.get_linked_tasks(actor, str(event_id))
                tasks = tasks_result.value if tasks_result.is_success else []
                GLib.idle_add(
                    self._update_linked_list,
                    self._linked_tasks_list,
                    tasks,
                    "tasks"
                )
            except Exception as exc:
                logger.warning(f"Failed to load linked entities: {exc}")
                GLib.idle_add(
                    self._update_linked_list,
                    self._linked_jobs_list,
                    [],
                    "jobs"
                )
                GLib.idle_add(
                    self._update_linked_list,
                    self._linked_tasks_list,
                    [],
                    "tasks"
                )

        asyncio.create_task(fetch_and_display())

    def _update_linked_list(
        self,
        container: Gtk.Box,
        items: List[Dict[str, Any]],
        entity_type: str,
    ) -> bool:
        """Update a linked entities list container.

        Returns True for GLib.idle_add compatibility.
        """
        # Clear existing children
        while True:
            child = container.get_first_child()
            if child is None:
                break
            container.remove(child)

        if not items:
            placeholder = Gtk.Label(
                label=f"No linked {entity_type}"
            )
            placeholder.set_xalign(0)
            placeholder.add_css_class("dim-label")
            container.append(placeholder)
            return True

        for item in items:
            row = self._create_linked_entity_row(item, entity_type)
            container.append(row)

        return True

    def _create_linked_entity_row(
        self,
        item: Dict[str, Any],
        entity_type: str,
    ) -> Gtk.Box:
        """Create a row widget for a linked job or task."""
        row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        row.set_margin_top(2)
        row.set_margin_bottom(2)

        # Icon based on entity type
        icon_name = "emblem-system-symbolic" if entity_type == "jobs" else "view-list-symbolic"
        icon = Gtk.Image.new_from_icon_name(icon_name)
        icon.set_opacity(0.7)
        row.append(icon)

        # Name/title
        if entity_type == "jobs":
            name = item.get("job_name") or item.get("name") or item.get("job_id", "Unknown")
            entity_id = item.get("job_id")
        else:
            name = item.get("task_title") or item.get("title") or item.get("task_id", "Unknown")
            entity_id = item.get("task_id")

        name_label = Gtk.Label(label=str(name))
        name_label.set_xalign(0)
        name_label.set_hexpand(True)
        name_label.set_ellipsize(3)  # PANGO_ELLIPSIZE_END
        row.append(name_label)

        # Status badge if available
        status = item.get("status") or item.get("job_status") or item.get("task_status")
        if status:
            status_badge = Gtk.Label(label=str(status).replace("_", " ").title())
            status_badge.add_css_class("badge")
            self._apply_status_style(status_badge, str(status))
            row.append(status_badge)

        # Link type indicator
        link_type = item.get("link_type")
        if link_type:
            link_label = Gtk.Label(label=f"({link_type})")
            link_label.add_css_class("dim-label")
            link_label.set_opacity(0.6)
            row.append(link_label)

        # Unlink button
        unlink_btn = Gtk.Button()
        unlink_btn.set_icon_name("edit-delete-symbolic")
        unlink_btn.set_tooltip_text(f"Unlink this {entity_type[:-1]}")
        unlink_btn.add_css_class("flat")
        unlink_btn.add_css_class("circular")
        unlink_btn.connect(
            "clicked",
            self._on_unlink_clicked,
            entity_type,
            entity_id,
        )
        row.append(unlink_btn)

        return row

    def _apply_status_style(self, widget: Gtk.Widget, status: str) -> None:
        """Apply CSS class based on status value."""
        status_lower = status.lower()
        if status_lower in ("done", "succeeded", "completed"):
            widget.add_css_class("success")
        elif status_lower in ("running", "in_progress"):
            widget.add_css_class("accent")
        elif status_lower in ("failed", "cancelled"):
            widget.add_css_class("error")
        elif status_lower in ("scheduled", "ready", "pending"):
            widget.add_css_class("warning")
        else:
            widget.add_css_class("dim-label")

    def _on_unlink_clicked(
        self,
        button: Gtk.Button,
        entity_type: str,
        entity_id: Optional[str],
    ) -> None:
        """Handle unlink button click."""
        import asyncio

        if not entity_id or not self.ATLAS:
            return

        event_id = self._event_data.get("id") or self._event_data.get("event_id")
        if not event_id:
            return

        async def do_unlink():
            try:
                service = self.ATLAS.calendar_service
                actor = self.ATLAS.get_current_actor()

                if entity_type == "jobs":
                    result = await service.unlink_from_job(
                        actor, str(event_id), entity_id
                    )
                else:
                    result = await service.unlink_from_task(
                        actor, str(event_id), entity_id
                    )

                if result.is_success:
                    # Refresh the linked entities display
                    self._load_linked_entities()
                else:
                    logger.warning(f"Failed to unlink: {result.error}")
            except Exception as exc:
                logger.warning(f"Failed to unlink entity: {exc}")

        asyncio.create_task(do_unlink())

    def _load_categories(self) -> None:
        """Load categories for the dropdown."""
        async def fetch_categories():
            try:
                if not self.ATLAS:
                    return []
                
                service = self.ATLAS.calendar_service
                actor = self.ATLAS.get_current_actor()
                result = await service.list_categories(actor)
                
                if result.is_success:
                    return [
                        {"id": str(c.id), "name": c.name, "color": c.color, "icon": c.icon}
                        for c in result.data
                    ]
                else:
                    logger.warning(f"Failed to load categories: {result.error}")
                    return []
            except Exception as exc:
                logger.warning(f"Failed to load categories: {exc}")
                return []
        
        async def update_ui():
            self._categories = await fetch_categories()
            
            # Populate dropdown
            model = Gtk.StringList()
            default_idx = 0

            for idx, cat in enumerate(self._categories):
                icon = cat.get("icon", "")
                label = f"{icon} {cat['name']}" if icon else cat["name"]
                model.append(label)

                if self._default_category_id and cat["id"] == self._default_category_id:
                    default_idx = idx

            if self._category_dropdown:
                self._category_dropdown.set_model(model)
                self._category_dropdown.set_selected(default_idx)
        
        asyncio.create_task(update_ui())

    def _populate_from_event(self, event: Dict[str, Any]) -> None:
        """Populate form fields from existing event data."""
        if self._title_entry:
            self._title_entry.set_text(event.get("title", ""))

        if self._location_entry:
            self._location_entry.set_text(event.get("location", ""))

        if self._description_view:
            buffer = self._description_view.get_buffer()
            buffer.set_text(event.get("description", ""))

        # Date/time
        start = event.get("start_time")
        if start:
            if isinstance(start, str):
                start = datetime.fromisoformat(start)
            self._start_date = start.date()
            self._start_time = start.time()
            if self._start_date_button:
                self._start_date_button.set_label(self._format_date(self._start_date))
            if self._start_time_entry:
                self._start_time_entry.set_text(self._format_time(self._start_time))

        end = event.get("end_time")
        if end:
            if isinstance(end, str):
                end = datetime.fromisoformat(end)
            self._end_date = end.date()
            self._end_time = end.time()
            if self._end_date_button:
                self._end_date_button.set_label(self._format_date(self._end_date))
            if self._end_time_entry:
                self._end_time_entry.set_text(self._format_time(self._end_time))

        # All-day
        if self._all_day_switch and event.get("is_all_day"):
            self._all_day_switch.set_active(True)

        # Category
        cat_id = event.get("category_id")
        if cat_id and self._category_dropdown:
            for idx, cat in enumerate(self._categories):
                if cat["id"] == str(cat_id):
                    self._category_dropdown.set_selected(idx)
                    break

        # URL
        if self._url_entry:
            self._url_entry.set_text(event.get("url", ""))

    def _format_date(self, d: date) -> str:
        """Format date for display."""
        return d.strftime("%a, %b %d, %Y")

    def _format_time(self, t: time) -> str:
        """Format time for display (24-hour)."""
        return t.strftime("%H:%M")

    def _parse_time(self, text: str) -> Optional[time]:
        """Parse time string to time object."""
        text = text.strip()
        formats = ["%H:%M", "%I:%M %p", "%I:%M%p", "%H%M"]
        for fmt in formats:
            try:
                return datetime.strptime(text, fmt).time()
            except ValueError:
                continue
        return None

    # ========================================================================
    # Event handlers
    # ========================================================================

    def _on_all_day_toggled(self, switch: Gtk.Switch, _pspec) -> None:
        """Handle all-day toggle."""
        is_all_day = switch.get_active()
        if self._start_time_entry:
            self._start_time_entry.set_sensitive(not is_all_day)
        if self._end_time_entry:
            self._end_time_entry.set_sensitive(not is_all_day)

    def _on_start_date_clicked(self, button: Gtk.Button) -> None:
        """Show date picker for start date."""
        self._show_date_picker(button, is_start=True)

    def _on_end_date_clicked(self, button: Gtk.Button) -> None:
        """Show date picker for end date."""
        self._show_date_picker(button, is_start=False)

    def _show_date_picker(self, button: Gtk.Button, is_start: bool) -> None:
        """Show a calendar popover for date selection."""
        popover = Gtk.Popover()
        popover.set_parent(button)

        calendar = Gtk.Calendar()
        current_date = self._start_date if is_start else self._end_date
        calendar.select_day(
            GLib.DateTime.new_local(
                current_date.year,
                current_date.month,
                current_date.day,
                0, 0, 0
            )
        )

        def on_day_selected(cal):
            gdt = cal.get_date()
            selected = date(gdt.get_year(), gdt.get_month(), gdt.get_day_of_month())
            if is_start:
                self._start_date = selected
                button.set_label(self._format_date(selected))
                # Auto-adjust end date if before start
                if self._end_date < selected:
                    self._end_date = selected
                    if self._end_date_button:
                        self._end_date_button.set_label(self._format_date(selected))
            else:
                self._end_date = selected
                button.set_label(self._format_date(selected))
            popover.popdown()

        calendar.connect("day-selected", on_day_selected)
        popover.set_child(calendar)
        popover.popup()

    def _on_start_time_changed(self, entry: Gtk.Entry) -> None:
        """Handle start time change - auto-adjust end time."""
        parsed = self._parse_time(entry.get_text())
        if parsed:
            old_start = self._start_time
            self._start_time = parsed

            # Maintain duration
            if self._end_time_entry:
                old_duration = datetime.combine(date.min, self._end_time) - datetime.combine(date.min, old_start)
                new_end = datetime.combine(date.min, parsed) + old_duration
                self._end_time = new_end.time()
                self._end_time_entry.set_text(self._format_time(self._end_time))

    def _on_add_reminder_clicked(self, button: Gtk.Button) -> None:
        """Add a new reminder row."""
        self._add_reminder_row(15)

    # ========================================================================
    # Public API
    # ========================================================================

    def get_event_data(self) -> Optional[Dict[str, Any]]:
        """Get the event data from the form.

        Returns:
            Dictionary with event data, or None if validation fails
        """
        # Validate required fields
        title = self._title_entry.get_text().strip() if self._title_entry else ""
        if not title:
            self._show_error("Title is required")
            return None

        # Build datetime
        is_all_day = self._all_day_switch.get_active() if self._all_day_switch else False

        if is_all_day:
            start_dt = datetime.combine(self._start_date, time.min)
            end_dt = datetime.combine(self._end_date, time(23, 59, 59))
        else:
            # Parse time entries
            start_time = self._parse_time(
                self._start_time_entry.get_text() if self._start_time_entry else ""
            ) or self._start_time
            end_time = self._parse_time(
                self._end_time_entry.get_text() if self._end_time_entry else ""
            ) or self._end_time

            start_dt = datetime.combine(self._start_date, start_time)
            end_dt = datetime.combine(self._end_date, end_time)

        # Validate end > start
        if end_dt <= start_dt:
            self._show_error("End time must be after start time")
            return None

        # Get description
        description = ""
        if self._description_view:
            buffer = self._description_view.get_buffer()
            description = buffer.get_text(
                buffer.get_start_iter(),
                buffer.get_end_iter(),
                False
            )

        # Get category
        category_id = None
        if self._category_dropdown and self._categories:
            idx = self._category_dropdown.get_selected()
            if idx < len(self._categories):
                category_id = self._categories[idx]["id"]

        # Get recurrence
        recurrence_rule = None
        if self._frequency_dropdown:
            freq_idx = self._frequency_dropdown.get_selected()
            if freq_idx > 0:  # Not "Does not repeat"
                recurrence_rule = self._build_rrule(freq_idx)

        # Get reminders
        reminders = self._collect_reminders()

        # Get advanced options
        status = STATUS_OPTIONS[self._status_dropdown.get_selected()][0] if self._status_dropdown else "confirmed"
        visibility = VISIBILITY_OPTIONS[self._visibility_dropdown.get_selected()][0] if self._visibility_dropdown else "public"
        busy_status = BUSY_STATUS_OPTIONS[self._busy_dropdown.get_selected()][0] if self._busy_dropdown else "busy"
        url = self._url_entry.get_text().strip() if self._url_entry else ""

        return {
            "title": title,
            "description": description,
            "location": self._location_entry.get_text().strip() if self._location_entry else "",
            "start_time": start_dt,
            "end_time": end_dt,
            "is_all_day": is_all_day,
            "category_id": category_id,
            "recurrence_rule": recurrence_rule,
            "reminders": reminders,
            "status": status,
            "visibility": visibility,
            "busy_status": busy_status,
            "url": url or None,
        }

    def _build_rrule(self, freq_idx: int) -> Optional[str]:
        """Build RRULE string from frequency selection."""
        freq_map = {
            1: "FREQ=DAILY",
            2: "FREQ=WEEKLY",
            3: "FREQ=MONTHLY",
            4: "FREQ=YEARLY",
            5: "FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR",
        }
        return freq_map.get(freq_idx)

    def _collect_reminders(self) -> List[Dict[str, Any]]:
        """Collect reminder settings from the UI."""
        reminders = []
        if not self._reminder_box:
            return reminders

        child = self._reminder_box.get_first_child()
        while child:
            if hasattr(child, "get_css_classes") and "reminder-row" in child.get_css_classes():
                # Find the dropdown in this row
                inner = child.get_first_child()
                while inner:
                    if isinstance(inner, Gtk.DropDown):
                        idx = inner.get_selected()
                        if idx < len(REMINDER_PRESETS):
                            minutes = REMINDER_PRESETS[idx][0]
                            reminders.append({
                                "minutes_before": minutes,
                                "method": "notification",
                            })
                        break
                    inner = inner.get_next_sibling()
            child = child.get_next_sibling()

        return reminders

    def _show_error(self, message: str) -> None:
        """Show an error message."""
        dialog = Gtk.MessageDialog(
            transient_for=self,
            modal=True,
            message_type=Gtk.MessageType.ERROR,
            buttons=Gtk.ButtonsType.OK,
            text=message,
        )
        dialog.connect("response", lambda d, r: d.destroy())
        dialog.present()
