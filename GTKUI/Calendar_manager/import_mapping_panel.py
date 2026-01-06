"""Import mapping panel for calendar sync configuration.

Allows users to configure which external calendars map to which
local categories, and manage sync sources.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple
from uuid import UUID

import gi
gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Gtk, Adw, GLib, Pango

logger = logging.getLogger(__name__)


class SyncSourceRow(Gtk.Box):
    """Row displaying a sync source with its mapped category."""
    
    def __init__(
        self,
        source_id: str,
        source_name: str,
        provider_type: str,
        category_name: Optional[str] = None,
        last_sync: Optional[datetime] = None,
        on_edit: Optional[Callable[[str], None]] = None,
        on_delete: Optional[Callable[[str], None]] = None,
        on_sync: Optional[Callable[[str], None]] = None,
    ):
        super().__init__(
            orientation=Gtk.Orientation.HORIZONTAL,
            spacing=12,
            margin_top=8,
            margin_bottom=8,
            margin_start=12,
            margin_end=12,
        )
        
        self.source_id = source_id
        self._on_edit = on_edit
        self._on_delete = on_delete
        self._on_sync = on_sync
        
        # Provider icon
        icon_name = self._get_provider_icon(provider_type)
        icon = Gtk.Image.new_from_icon_name(icon_name)
        icon.set_pixel_size(24)
        icon.add_css_class("dim-label")
        self.append(icon)
        
        # Info section
        info_box = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=2,
            hexpand=True,
        )
        self.append(info_box)
        
        # Source name
        name_label = Gtk.Label(
            label=source_name,
            xalign=0,
            ellipsize=Pango.EllipsizeMode.END,
        )
        name_label.add_css_class("heading")
        info_box.append(name_label)
        
        # Details row
        details_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        info_box.append(details_box)
        
        # Provider type
        type_label = Gtk.Label(
            label=provider_type.upper(),
            xalign=0,
        )
        type_label.add_css_class("dim-label")
        type_label.add_css_class("caption")
        details_box.append(type_label)
        
        # Category mapping
        if category_name:
            sep = Gtk.Label(label="→")
            sep.add_css_class("dim-label")
            details_box.append(sep)
            
            cat_label = Gtk.Label(label=category_name, xalign=0)
            cat_label.add_css_class("caption")
            details_box.append(cat_label)
        
        # Last sync time
        if last_sync:
            sync_label = Gtk.Label(
                label=f"Synced {self._format_time_ago(last_sync)}",
                xalign=0,
            )
            sync_label.add_css_class("dim-label")
            sync_label.add_css_class("caption")
            sync_label.set_margin_start(12)
            details_box.append(sync_label)
        
        # Action buttons
        button_box = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL,
            spacing=4,
        )
        self.append(button_box)
        
        # Sync now button
        sync_btn = Gtk.Button(icon_name="emblem-synchronizing-symbolic")
        sync_btn.set_tooltip_text("Sync Now")
        sync_btn.add_css_class("flat")
        sync_btn.connect("clicked", self._on_sync_clicked)
        button_box.append(sync_btn)
        
        # Edit button
        edit_btn = Gtk.Button(icon_name="document-edit-symbolic")
        edit_btn.set_tooltip_text("Edit")
        edit_btn.add_css_class("flat")
        edit_btn.connect("clicked", self._on_edit_clicked)
        button_box.append(edit_btn)
        
        # Delete button
        delete_btn = Gtk.Button(icon_name="user-trash-symbolic")
        delete_btn.set_tooltip_text("Remove")
        delete_btn.add_css_class("flat")
        delete_btn.add_css_class("destructive-action")
        delete_btn.connect("clicked", self._on_delete_clicked)
        button_box.append(delete_btn)
    
    def _get_provider_icon(self, provider_type: str) -> str:
        """Get icon name for provider type."""
        icons = {
            "ics": "text-x-generic-symbolic",
            "caldav": "network-server-symbolic",
            "google": "web-browser-symbolic",
            "outlook": "mail-symbolic",
        }
        return icons.get(provider_type.lower(), "folder-symbolic")
    
    def _format_time_ago(self, dt: datetime) -> str:
        """Format datetime as human-readable 'time ago' string."""
        now = datetime.now(timezone.utc)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        
        diff = now - dt
        seconds = diff.total_seconds()
        
        if seconds < 60:
            return "just now"
        elif seconds < 3600:
            minutes = int(seconds / 60)
            return f"{minutes}m ago"
        elif seconds < 86400:
            hours = int(seconds / 3600)
            return f"{hours}h ago"
        else:
            days = int(seconds / 86400)
            return f"{days}d ago"
    
    def _on_sync_clicked(self, button: Gtk.Button) -> None:
        if self._on_sync:
            self._on_sync(self.source_id)
    
    def _on_edit_clicked(self, button: Gtk.Button) -> None:
        if self._on_edit:
            self._on_edit(self.source_id)
    
    def _on_delete_clicked(self, button: Gtk.Button) -> None:
        if self._on_delete:
            self._on_delete(self.source_id)


class AddSourceDialog(Adw.Window):
    """Dialog for adding a new sync source."""
    
    def __init__(
        self,
        parent: Gtk.Window,
        categories: List[Dict[str, Any]],
        on_save: Optional[Callable[[Dict[str, Any]], None]] = None,
        source_data: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            modal=True,
            transient_for=parent,
            default_width=480,
            default_height=560,
            title="Add Calendar Source" if not source_data else "Edit Calendar Source",
        )
        
        self.categories = categories
        self._on_save = on_save
        self._source_data = source_data
        
        self._setup_ui()
        
        if source_data:
            self._populate_data(source_data)
    
    def _setup_ui(self) -> None:
        """Set up the dialog UI."""
        # Main toolbar view
        toolbar_view = Adw.ToolbarView()
        self.set_content(toolbar_view)
        
        # Header bar
        header = Adw.HeaderBar()
        
        cancel_btn = Gtk.Button(label="Cancel")
        cancel_btn.connect("clicked", lambda b: self.close())
        header.pack_start(cancel_btn)
        
        self.save_btn = Gtk.Button(label="Save")
        self.save_btn.add_css_class("suggested-action")
        self.save_btn.connect("clicked", self._on_save_clicked)
        header.pack_end(self.save_btn)
        
        toolbar_view.add_top_bar(header)
        
        # Scrolled content
        scrolled = Gtk.ScrolledWindow(
            vexpand=True,
            hscrollbar_policy=Gtk.PolicyType.NEVER,
        )
        toolbar_view.set_content(scrolled)
        
        # Main content box
        content = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=24,
            margin_top=24,
            margin_bottom=24,
            margin_start=24,
            margin_end=24,
        )
        scrolled.set_child(content)
        
        # Provider type selection
        type_group = Adw.PreferencesGroup(title="Source Type")
        content.append(type_group)
        
        # Provider radio buttons
        self.provider_buttons: Dict[str, Gtk.CheckButton] = {}
        
        providers = [
            ("ics", "ICS File / URL", "Import from .ics file or webcal URL"),
            ("caldav", "CalDAV Server", "Sync with CalDAV server (Nextcloud, etc)"),
        ]
        
        first_button = None
        for provider_id, name, desc in providers:
            row = Adw.ActionRow(
                title=name,
                subtitle=desc,
            )
            
            radio = Gtk.CheckButton()
            if first_button:
                radio.set_group(first_button)
            else:
                first_button = radio
            
            radio.connect("toggled", self._on_provider_changed, provider_id)
            row.add_prefix(radio)
            row.set_activatable_widget(radio)
            
            self.provider_buttons[provider_id] = radio
            type_group.add(row)
        
        # Connection settings group
        self.connection_group = Adw.PreferencesGroup(title="Connection")
        content.append(self.connection_group)
        
        # Name
        self.name_row = Adw.EntryRow(title="Name")
        self.connection_group.add(self.name_row)
        
        # URL/Path
        self.url_row = Adw.EntryRow(title="URL or File Path")
        self.url_row.set_input_purpose(Gtk.InputPurpose.URL)
        self.connection_group.add(self.url_row)
        
        # Browse button for local files
        browse_btn = Gtk.Button(icon_name="folder-open-symbolic")
        browse_btn.set_valign(Gtk.Align.CENTER)
        browse_btn.set_tooltip_text("Browse for file")
        browse_btn.connect("clicked", self._on_browse_clicked)
        self.url_row.add_suffix(browse_btn)
        
        # Username (CalDAV only)
        self.username_row = Adw.EntryRow(title="Username")
        self.connection_group.add(self.username_row)
        
        # Password (CalDAV only)
        self.password_row = Adw.PasswordEntryRow(title="Password")
        self.connection_group.add(self.password_row)
        
        # Mapping settings group
        mapping_group = Adw.PreferencesGroup(title="Category Mapping")
        content.append(mapping_group)
        
        # Category dropdown
        self.category_row = Adw.ComboRow(title="Import to Category")
        
        # Build category model
        category_model = Gtk.StringList()
        for cat in self.categories:
            category_model.append(cat.get("name", "Unknown"))
        
        self.category_row.set_model(category_model)
        mapping_group.add(self.category_row)
        
        # Sync settings group
        sync_group = Adw.PreferencesGroup(title="Sync Settings")
        content.append(sync_group)
        
        # Auto sync toggle
        self.auto_sync_row = Adw.SwitchRow(
            title="Auto Sync",
            subtitle="Automatically sync on a schedule",
        )
        sync_group.add(self.auto_sync_row)
        
        # Sync interval
        self.interval_row = Adw.ComboRow(title="Sync Interval")
        interval_model = Gtk.StringList()
        intervals = ["Every 15 minutes", "Every 30 minutes", "Every hour", "Every 6 hours", "Daily"]
        for interval in intervals:
            interval_model.append(interval)
        self.interval_row.set_model(interval_model)
        self.interval_row.set_selected(2)  # Default to hourly
        sync_group.add(self.interval_row)
        
        # Initialize provider selection
        if first_button:
            first_button.set_active(True)
    
    def _on_provider_changed(self, button: Gtk.CheckButton, provider_id: str) -> None:
        """Handle provider type change."""
        if not button.get_active():
            return
        
        is_caldav = provider_id == "caldav"
        self.username_row.set_visible(is_caldav)
        self.password_row.set_visible(is_caldav)
    
    def _on_browse_clicked(self, button: Gtk.Button) -> None:
        """Open file chooser for ICS file."""
        dialog = Gtk.FileDialog()
        dialog.set_title("Select ICS File")
        
        # Filter for ICS files
        filter_store = Gio.ListStore.new(Gtk.FileFilter)
        
        ics_filter = Gtk.FileFilter()
        ics_filter.set_name("iCalendar Files")
        ics_filter.add_pattern("*.ics")
        ics_filter.add_mime_type("text/calendar")
        filter_store.append(ics_filter)
        
        all_filter = Gtk.FileFilter()
        all_filter.set_name("All Files")
        all_filter.add_pattern("*")
        filter_store.append(all_filter)
        
        dialog.set_filters(filter_store)
        
        dialog.open(self.get_transient_for(), None, self._on_file_selected)
    
    def _on_file_selected(self, dialog: Gtk.FileDialog, result: Any) -> None:
        """Handle file selection."""
        try:
            file = dialog.open_finish(result)
            if file:
                path = file.get_path()
                self.url_row.set_text(path)
                
                # Auto-fill name from filename
                if not self.name_row.get_text():
                    import os
                    name = os.path.splitext(os.path.basename(path))[0]
                    self.name_row.set_text(name)
        except GLib.Error:
            pass  # User cancelled
    
    def _populate_data(self, data: Dict[str, Any]) -> None:
        """Populate form with existing source data."""
        provider = data.get("provider", "ics")
        if provider in self.provider_buttons:
            self.provider_buttons[provider].set_active(True)
        
        self.name_row.set_text(data.get("name", ""))
        self.url_row.set_text(data.get("url", "") or data.get("path", ""))
        self.username_row.set_text(data.get("username", ""))
        self.password_row.set_text(data.get("password", ""))
        
        self.auto_sync_row.set_active(data.get("auto_sync", False))
        
        # Select category
        category_id = data.get("category_id")
        if category_id:
            for i, cat in enumerate(self.categories):
                if cat.get("id") == category_id:
                    self.category_row.set_selected(i)
                    break
    
    def _get_selected_provider(self) -> str:
        """Get the selected provider type."""
        for provider_id, button in self.provider_buttons.items():
            if button.get_active():
                return provider_id
        return "ics"
    
    def _get_sync_interval_minutes(self) -> int:
        """Get the sync interval in minutes."""
        intervals = [15, 30, 60, 360, 1440]
        selected = self.interval_row.get_selected()
        if 0 <= selected < len(intervals):
            return intervals[selected]
        return 60
    
    def _on_save_clicked(self, button: Gtk.Button) -> None:
        """Handle save button click."""
        provider = self._get_selected_provider()
        name = self.name_row.get_text().strip()
        url = self.url_row.get_text().strip()
        
        if not name:
            # TODO: Show validation error
            return
        
        if not url:
            # TODO: Show validation error
            return
        
        # Get selected category
        category_id = None
        selected_idx = self.category_row.get_selected()
        if 0 <= selected_idx < len(self.categories):
            category_id = self.categories[selected_idx].get("id")
        
        source_config = {
            "provider": provider,
            "name": name,
            "url": url,
            "category_id": category_id,
            "auto_sync": self.auto_sync_row.get_active(),
            "sync_interval_minutes": self._get_sync_interval_minutes(),
        }
        
        if provider == "caldav":
            source_config["username"] = self.username_row.get_text()
            source_config["password"] = self.password_row.get_text()
        
        if self._source_data:
            source_config["id"] = self._source_data.get("id")
        
        if self._on_save:
            self._on_save(source_config)
        
        self.close()


# Import Gio for file dialogs
from gi.repository import Gio


class ImportMappingPanel(Gtk.Box):
    """Panel for managing calendar sync sources and mappings."""
    
    def __init__(
        self,
        repository: Any = None,
        on_sync_triggered: Optional[Callable[[str], None]] = None,
    ):
        super().__init__(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=0,
        )
        
        self.repository = repository
        self._on_sync_triggered = on_sync_triggered
        self._sources: List[Dict[str, Any]] = []
        self._categories: List[Dict[str, Any]] = []
        
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        """Set up the panel UI."""
        # Header
        header_box = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL,
            spacing=12,
            margin_top=16,
            margin_bottom=12,
            margin_start=16,
            margin_end=16,
        )
        self.append(header_box)
        
        title_box = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=4,
            hexpand=True,
        )
        header_box.append(title_box)
        
        title = Gtk.Label(
            label="Calendar Sources",
            xalign=0,
        )
        title.add_css_class("title-2")
        title_box.append(title)
        
        subtitle = Gtk.Label(
            label="Import calendars from external sources",
            xalign=0,
        )
        subtitle.add_css_class("dim-label")
        title_box.append(subtitle)
        
        # Add source button
        add_btn = Gtk.Button(
            icon_name="list-add-symbolic",
            tooltip_text="Add Source",
        )
        add_btn.add_css_class("circular")
        add_btn.add_css_class("suggested-action")
        add_btn.connect("clicked", self._on_add_source)
        header_box.append(add_btn)
        
        # Sync all button
        sync_all_btn = Gtk.Button(
            icon_name="emblem-synchronizing-symbolic",
            tooltip_text="Sync All",
        )
        sync_all_btn.add_css_class("circular")
        sync_all_btn.connect("clicked", self._on_sync_all)
        header_box.append(sync_all_btn)
        
        # Separator
        sep = Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL)
        self.append(sep)
        
        # Scrolled list
        scrolled = Gtk.ScrolledWindow(
            vexpand=True,
            hscrollbar_policy=Gtk.PolicyType.NEVER,
        )
        self.append(scrolled)
        
        # Source list container
        self.list_box = Gtk.ListBox(
            selection_mode=Gtk.SelectionMode.NONE,
        )
        self.list_box.add_css_class("boxed-list")
        self.list_box.set_margin_top(12)
        self.list_box.set_margin_bottom(12)
        self.list_box.set_margin_start(16)
        self.list_box.set_margin_end(16)
        scrolled.set_child(self.list_box)
        
        # Empty state
        self.empty_state = Adw.StatusPage(
            icon_name="mail-send-symbolic",
            title="No Calendar Sources",
            description="Add external calendars to sync events into ATLAS",
        )
        self.empty_state.set_visible(False)
        scrolled.set_child(self.empty_state)
        
        # Status bar
        self.status_bar = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL,
            spacing=8,
            margin_top=8,
            margin_bottom=8,
            margin_start=16,
            margin_end=16,
        )
        self.append(self.status_bar)
        
        self.status_label = Gtk.Label(
            label="",
            xalign=0,
            hexpand=True,
        )
        self.status_label.add_css_class("dim-label")
        self.status_label.add_css_class("caption")
        self.status_bar.append(self.status_label)
    
    def set_sources(self, sources: List[Dict[str, Any]]) -> None:
        """Set the list of sync sources."""
        self._sources = sources
        self._refresh_list()
    
    def set_categories(self, categories: List[Dict[str, Any]]) -> None:
        """Set available categories for mapping."""
        self._categories = categories
    
    def _refresh_list(self) -> None:
        """Refresh the source list display."""
        # Clear existing rows
        while True:
            row = self.list_box.get_first_child()
            if row is None:
                break
            self.list_box.remove(row)
        
        if not self._sources:
            self.list_box.set_visible(False)
            self.empty_state.set_visible(True)
            self.status_label.set_text("")
            return
        
        self.list_box.set_visible(True)
        self.empty_state.set_visible(False)
        
        # Build category name map
        cat_names = {cat["id"]: cat["name"] for cat in self._categories}
        
        for source in self._sources:
            category_name = None
            if source.get("category_id"):
                category_name = cat_names.get(source["category_id"])
            
            row = SyncSourceRow(
                source_id=source.get("id", ""),
                source_name=source.get("name", "Unknown"),
                provider_type=source.get("provider", "unknown"),
                category_name=category_name,
                last_sync=source.get("last_sync"),
                on_edit=self._on_edit_source,
                on_delete=self._on_delete_source,
                on_sync=self._on_sync_source,
            )
            self.list_box.append(row)
        
        # Update status
        self.status_label.set_text(f"{len(self._sources)} source(s) configured")
    
    def _on_add_source(self, button: Gtk.Button) -> None:
        """Handle add source button click."""
        window = self.get_root()
        if not isinstance(window, Gtk.Window):
            return
        
        dialog = AddSourceDialog(
            parent=window,
            categories=self._categories,
            on_save=self._save_source,
        )
        dialog.present()
    
    def _on_edit_source(self, source_id: str) -> None:
        """Handle edit source request."""
        source = next((s for s in self._sources if s.get("id") == source_id), None)
        if not source:
            return
        
        window = self.get_root()
        if not isinstance(window, Gtk.Window):
            return
        
        dialog = AddSourceDialog(
            parent=window,
            categories=self._categories,
            on_save=self._save_source,
            source_data=source,
        )
        dialog.present()
    
    def _on_delete_source(self, source_id: str) -> None:
        """Handle delete source request."""
        # Remove from list
        self._sources = [s for s in self._sources if s.get("id") != source_id]
        self._refresh_list()
        
        # TODO: Persist deletion via repository
    
    def _on_sync_source(self, source_id: str) -> None:
        """Handle sync source request."""
        if self._on_sync_triggered:
            self._on_sync_triggered(source_id)
    
    def _on_sync_all(self, button: Gtk.Button) -> None:
        """Handle sync all button click."""
        for source in self._sources:
            if self._on_sync_triggered:
                self._on_sync_triggered(source.get("id", ""))
    
    def _save_source(self, config: Dict[str, Any]) -> None:
        """Save a sync source configuration."""
        import uuid
        
        source_id = config.get("id")
        
        if source_id:
            # Update existing
            for i, source in enumerate(self._sources):
                if source.get("id") == source_id:
                    self._sources[i] = {**source, **config}
                    break
        else:
            # Create new
            config["id"] = str(uuid.uuid4())
            self._sources.append(config)
        
        self._refresh_list()
        
        # TODO: Persist via repository
    
    def update_sync_status(
        self,
        source_id: str,
        status: str,
        last_sync: Optional[datetime] = None,
    ) -> None:
        """Update the sync status for a source."""
        for source in self._sources:
            if source.get("id") == source_id:
                source["sync_status"] = status
                if last_sync:
                    source["last_sync"] = last_sync
                break
        
        self._refresh_list()
