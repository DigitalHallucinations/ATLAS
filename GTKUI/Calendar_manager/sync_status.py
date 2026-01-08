"""Sync status panel for calendar management."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import GLib, Gtk

logger = logging.getLogger(__name__)


def _run_async(coro: Any, callback: Optional[Callable[[Any], None]] = None) -> None:
    """Run an async coroutine from GTK context."""
    def run():
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(coro)
            if callback:
                GLib.idle_add(callback, result)
        except Exception as e:
            logger.error("Async operation failed: %s", e)
            if callback:
                GLib.idle_add(callback, e)
        finally:
            loop.close()
    
    import threading
    thread = threading.Thread(target=run, daemon=True)
    thread.start()


class SyncStatusPanel(Gtk.Box):
    """Panel showing synchronization status for all calendars."""

    def __init__(self, atlas: Any, parent_window: Gtk.Window) -> None:
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        self.ATLAS = atlas
        self._parent = parent_window
        self._status_data: Dict[str, dict] = {}
        self._row_widgets: Dict[str, Dict[str, Gtk.Widget]] = {}
        self._sync_service: Optional[Any] = None
        self._event_subscriptions: List[Any] = []

        self.set_margin_top(12)
        self.set_margin_bottom(12)
        self.set_margin_start(12)
        self.set_margin_end(12)

        self._build_ui()
        self._setup_sync_service()
        self._subscribe_to_events()
        self._load_status()

    def _build_ui(self) -> None:
        """Build the sync status UI."""
        # Header
        header_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        header_box.set_margin_bottom(8)

        title = Gtk.Label(label="Synchronization Status")
        title.add_css_class("title-2")
        title.set_xalign(0.0)
        title.set_hexpand(True)
        header_box.append(title)

        # Sync all button
        sync_all_btn = Gtk.Button()
        sync_all_btn.set_icon_name("emblem-synchronizing-symbolic")
        sync_all_btn.set_tooltip_text("Sync All Calendars")
        sync_all_btn.connect("clicked", self._on_sync_all)
        header_box.append(sync_all_btn)

        # Refresh button
        refresh_btn = Gtk.Button()
        refresh_btn.set_icon_name("view-refresh-symbolic")
        refresh_btn.set_tooltip_text("Refresh Status")
        refresh_btn.connect("clicked", lambda b: self._load_status())
        header_box.append(refresh_btn)

        self.append(header_box)

        # Separator
        self.append(Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL))

        # Status summary
        self._summary_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=24)
        self._summary_box.set_margin_top(8)
        self._summary_box.set_margin_bottom(8)
        self.append(self._summary_box)

        # Separator
        self.append(Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL))

        # Scrolled list of calendars with sync info
        scroll = Gtk.ScrolledWindow()
        scroll.set_vexpand(True)
        scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)

        self._list_box = Gtk.ListBox()
        self._list_box.set_selection_mode(Gtk.SelectionMode.NONE)
        self._list_box.add_css_class("boxed-list")
        scroll.set_child(self._list_box)
        self.append(scroll)

        # Error log section
        error_header = Gtk.Label(label="Recent Errors")
        error_header.add_css_class("title-3")
        error_header.set_xalign(0.0)
        error_header.set_margin_top(12)
        self.append(error_header)

        error_scroll = Gtk.ScrolledWindow()
        error_scroll.set_min_content_height(100)
        error_scroll.set_max_content_height(150)
        error_scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)

        self._error_list = Gtk.ListBox()
        self._error_list.set_selection_mode(Gtk.SelectionMode.NONE)
        self._error_list.add_css_class("boxed-list")
        error_scroll.set_child(self._error_list)
        self.append(error_scroll)

    def _setup_sync_service(self) -> None:
        """Set up CalendarSyncService if available."""
        try:
            # Try to get sync service from ATLAS
            if hasattr(self.ATLAS, "calendar_sync_service"):
                self._sync_service = self.ATLAS.calendar_sync_service
                logger.debug("Using ATLAS calendar sync service")
            elif hasattr(self.ATLAS, "services") and hasattr(self.ATLAS.services, "calendar_sync"):
                self._sync_service = self.ATLAS.services.calendar_sync
                logger.debug("Using ATLAS services calendar sync")
            else:
                # Try to create one if we have a repository
                self._try_create_sync_service()
        except Exception as e:
            logger.warning("Could not set up sync service: %s", e)
            self._sync_service = None

    def _try_create_sync_service(self) -> None:
        """Attempt to create a sync service from available components."""
        try:
            from core.services.calendar import create_sync_service
            
            # Get repository from ATLAS or calendar store
            repository = None
            if hasattr(self.ATLAS, "calendar_repository"):
                repository = self.ATLAS.calendar_repository
            elif hasattr(self.ATLAS, "modules"):
                calendar_store = self.ATLAS.modules.get("calendar_store")
                if calendar_store and hasattr(calendar_store, "repository"):
                    repository = calendar_store.repository
                    
            if repository:
                self._sync_service = create_sync_service(repository=repository)
                logger.info("Created CalendarSyncService")
        except ImportError:
            logger.debug("CalendarSyncService not available")
        except Exception as e:
            logger.debug("Could not create sync service: %s", e)

    def _subscribe_to_events(self) -> None:
        """Subscribe to sync domain events for reactive updates."""
        if not hasattr(self.ATLAS, "message_bus"):
            return
            
        try:
            # Subscribe to sync events
            self._event_subscriptions.append(
                self.ATLAS.message_bus.subscribe(
                    "CalendarSyncCompleted",
                    self._on_sync_event
                )
            )
            self._event_subscriptions.append(
                self.ATLAS.message_bus.subscribe(
                    "CalendarSyncFailed",
                    self._on_sync_event
                )
            )
            self._event_subscriptions.append(
                self.ATLAS.message_bus.subscribe(
                    "CalendarSyncProgress",
                    self._on_sync_progress_event
                )
            )
            logger.debug("Subscribed to sync events")
        except Exception as e:
            logger.warning("Could not subscribe to sync events: %s", e)

    def _on_sync_event(self, event: Any) -> None:
        """Handle sync completed or failed events."""
        GLib.idle_add(self._load_status)

    def _on_sync_progress_event(self, event: Any) -> None:
        """Handle sync progress events."""
        # Update progress indicator if available
        try:
            sync_id = getattr(event, "sync_id", None)
            if sync_id and sync_id in self._row_widgets:
                widgets = self._row_widgets[sync_id]
                percent = getattr(event, "percent", 0)
                # Could update a progress bar here
                logger.debug("Sync %s progress: %.0f%%", sync_id, percent)
        except Exception as e:
            logger.debug("Error handling progress event: %s", e)

    def _load_status(self) -> None:
        """Load sync status from service or fallback to registry."""
        # Try using CalendarSyncService first
        if self._sync_service:
            self._load_status_from_service()
        else:
            self._load_status_from_registry()

    def _load_status_from_service(self) -> None:
        """Load sync status using CalendarSyncService."""
        try:
            from core.services.common import Actor
            
            # Get actor from ATLAS context
            actor = self._get_current_actor()
            
            async def fetch_providers():
                result = await self._sync_service.list_providers(actor)
                return result
            
            def on_providers_loaded(result):
                if isinstance(result, Exception):
                    logger.error("Failed to load providers: %s", result)
                    self._show_empty_state(f"Error: {result}")
                    return
                    
                if not result.success:
                    logger.error("Failed to load providers: %s", result.error)
                    self._show_empty_state(f"Error: {result.error}")
                    return
                
                providers = result.value or []
                self._status_data = {}
                
                for provider_info in providers:
                    name = provider_info.provider_type
                    self._status_data[name] = {
                        "display_name": provider_info.display_name,
                        "backend_type": provider_info.provider_type,
                        "write_enabled": not getattr(provider_info, "read_only", True),
                        "last_sync": None,  # Will be updated from sync history
                        "sync_error": None,
                        "event_count": 0,
                        "supports_push": provider_info.supports_push,
                        "supports_incremental": provider_info.supports_incremental,
                    }
                
                self._update_display()
                # Also fetch sync history for last sync times
                self._fetch_sync_history()
            
            _run_async(fetch_providers(), on_providers_loaded)
            
        except Exception as e:
            logger.error("Failed to load status from service: %s", e)
            self._load_status_from_registry()

    def _fetch_sync_history(self) -> None:
        """Fetch recent sync history to get last sync times."""
        if not self._sync_service:
            return
            
        try:
            actor = self._get_current_actor()
            
            async def fetch_history():
                result = await self._sync_service.get_sync_history(actor, limit=20)
                return result
            
            def on_history_loaded(result):
                if isinstance(result, Exception) or not result.success:
                    return
                    
                history = result.value or []
                # Update status_data with last sync times
                for sync_result in history:
                    provider = sync_result.provider_type
                    if provider in self._status_data:
                        existing_sync = self._status_data[provider].get("last_sync")
                        if existing_sync is None or (
                            sync_result.completed_at and 
                            sync_result.completed_at > existing_sync
                        ):
                            self._status_data[provider]["last_sync"] = sync_result.completed_at
                            if sync_result.status.value == "failed":
                                self._status_data[provider]["sync_error"] = (
                                    sync_result.errors[0] if sync_result.errors else "Unknown error"
                                )
                            else:
                                self._status_data[provider]["sync_error"] = None
                
                self._update_display()
            
            _run_async(fetch_history(), on_history_loaded)
            
        except Exception as e:
            logger.debug("Could not fetch sync history: %s", e)

    def _get_current_actor(self) -> Any:
        """Get the current actor for service calls."""
        from core.services.common import Actor
        
        # Try to get user from ATLAS context
        actor_id = "system"
        tenant_id = "default"
        
        if hasattr(self.ATLAS, "current_user"):
            user = self.ATLAS.current_user
            if hasattr(user, "id"):
                actor_id = str(user.id)
            if hasattr(user, "tenant_id"):
                tenant_id = str(user.tenant_id)
        
        return Actor(
            type="user",
            id=actor_id,
            tenant_id=tenant_id,
            permissions=["calendar:sync", "calendar:read"],
        )

    def _load_status_from_registry(self) -> None:
        """Fallback: Load sync status from CalendarProviderRegistry."""
        try:
            from modules.Tools.Base_Tools.calendar import CalendarProviderRegistry

            registry = CalendarProviderRegistry.get_instance()
            providers = registry.list_providers()

            self._status_data = {}
            for name, info in providers.items():
                backend = info.get("backend")
                self._status_data[name] = {
                    "display_name": info.get("display_name", name),
                    "backend_type": info.get("type", "unknown"),
                    "write_enabled": info.get("write_enabled", False),
                    "last_sync": getattr(backend, "_last_sync", None) if backend else None,
                    "sync_error": getattr(backend, "_last_error", None) if backend else None,
                    "event_count": getattr(backend, "_event_count", 0) if backend else 0,
                }

            self._update_display()

        except ImportError:
            logger.warning("Calendar registry not available")
            self._show_empty_state("Calendar system not initialized")
        except Exception as e:
            logger.error("Failed to load sync status: %s", e)
            self._show_empty_state(f"Error loading status: {e}")

    def _update_display(self) -> None:
        """Update the display with current status data."""
        # Clear existing rows
        while True:
            row = self._list_box.get_row_at_index(0)
            if row is None:
                break
            self._list_box.remove(row)

        self._row_widgets.clear()

        # Update summary
        self._update_summary()

        # Add rows for each calendar
        for name, status in self._status_data.items():
            row = self._create_status_row(name, status)
            self._list_box.append(row)

        # Update error list
        self._update_errors()

    def _update_summary(self) -> None:
        """Update the summary statistics."""
        # Clear summary box
        while True:
            child = self._summary_box.get_first_child()
            if child is None:
                break
            self._summary_box.remove(child)

        total = len(self._status_data)
        synced = sum(1 for s in self._status_data.values() if s.get("last_sync"))
        errors = sum(1 for s in self._status_data.values() if s.get("sync_error"))
        writable = sum(1 for s in self._status_data.values() if s.get("write_enabled"))

        # Create stat boxes
        stats = [
            ("view-list-symbolic", f"{total}", "Total Calendars"),
            ("emblem-ok-symbolic", f"{synced}", "Synced"),
            ("dialog-error-symbolic", f"{errors}", "Errors"),
            ("document-edit-symbolic", f"{writable}", "Writable"),
        ]

        for icon_name, value, label in stats:
            stat_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)

            icon = Gtk.Image.new_from_icon_name(icon_name)
            stat_box.append(icon)

            value_label = Gtk.Label(label=value)
            value_label.add_css_class("title-1")
            stat_box.append(value_label)

            desc_label = Gtk.Label(label=label)
            desc_label.add_css_class("dim-label")
            stat_box.append(desc_label)

            self._summary_box.append(stat_box)

    def _create_status_row(self, name: str, status: dict) -> Gtk.ListBoxRow:
        """Create a row for a calendar's sync status."""
        row = Gtk.ListBoxRow()

        box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        box.set_margin_top(8)
        box.set_margin_bottom(8)
        box.set_margin_start(12)
        box.set_margin_end(12)

        # Status indicator
        has_error = status.get("sync_error") is not None
        indicator = Gtk.Image()
        if has_error:
            indicator.set_from_icon_name("dialog-error-symbolic")
            indicator.add_css_class("error")
        elif status.get("last_sync"):
            indicator.set_from_icon_name("emblem-ok-symbolic")
            indicator.add_css_class("success")
        else:
            indicator.set_from_icon_name("content-loading-symbolic")
            indicator.add_css_class("dim-label")
        box.append(indicator)

        # Info section
        info_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        info_box.set_hexpand(True)

        name_label = Gtk.Label(label=status.get("display_name", name))
        name_label.set_xalign(0.0)
        name_label.add_css_class("heading")
        info_box.append(name_label)

        # Last sync time
        last_sync = status.get("last_sync")
        if last_sync:
            if isinstance(last_sync, datetime):
                sync_text = last_sync.strftime("Last sync: %Y-%m-%d %H:%M")
            else:
                sync_text = f"Last sync: {last_sync}"
        else:
            sync_text = "Not synced"

        sync_label = Gtk.Label(label=sync_text)
        sync_label.set_xalign(0.0)
        sync_label.add_css_class("dim-label")
        sync_label.add_css_class("caption")
        info_box.append(sync_label)

        # Event count
        event_count = status.get("event_count", 0)
        count_label = Gtk.Label(label=f"{event_count} events")
        count_label.set_xalign(0.0)
        count_label.add_css_class("dim-label")
        count_label.add_css_class("caption")
        info_box.append(count_label)

        box.append(info_box)

        # Type badge
        backend_type = status.get("backend_type", "unknown")
        type_label = Gtk.Label(label=backend_type.upper())
        type_label.add_css_class("caption")
        type_label.add_css_class("dim-label")
        box.append(type_label)

        # Sync button
        sync_btn = Gtk.Button()
        sync_btn.set_icon_name("emblem-synchronizing-symbolic")
        sync_btn.set_tooltip_text("Sync Now")
        sync_btn.connect("clicked", self._on_sync_calendar, name)
        box.append(sync_btn)

        row.set_child(box)

        # Store widgets for updates
        self._row_widgets[name] = {
            "indicator": indicator,
            "sync_label": sync_label,
            "count_label": count_label,
            "sync_btn": sync_btn,
        }

        return row

    def _update_errors(self) -> None:
        """Update the error log."""
        # Clear existing
        while True:
            row = self._error_list.get_row_at_index(0)
            if row is None:
                break
            self._error_list.remove(row)

        # Collect errors
        errors: List[tuple] = []
        for name, status in self._status_data.items():
            error = status.get("sync_error")
            if error:
                errors.append((name, str(error)))

        if not errors:
            # Show "no errors" message
            row = Gtk.ListBoxRow()
            label = Gtk.Label(label="No sync errors")
            label.add_css_class("dim-label")
            label.set_margin_top(8)
            label.set_margin_bottom(8)
            row.set_child(label)
            self._error_list.append(row)
            return

        for name, error_msg in errors:
            row = Gtk.ListBoxRow()
            box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
            box.set_margin_top(4)
            box.set_margin_bottom(4)
            box.set_margin_start(8)
            box.set_margin_end(8)

            icon = Gtk.Image.new_from_icon_name("dialog-error-symbolic")
            icon.add_css_class("error")
            box.append(icon)

            name_label = Gtk.Label(label=f"{name}:")
            name_label.add_css_class("heading")
            box.append(name_label)

            error_label = Gtk.Label(label=error_msg)
            error_label.set_xalign(0.0)
            error_label.set_hexpand(True)
            error_label.set_wrap(True)
            error_label.add_css_class("dim-label")
            box.append(error_label)

            row.set_child(box)
            self._error_list.append(row)

    def _show_empty_state(self, message: str) -> None:
        """Show empty state message."""
        # Clear list
        while True:
            row = self._list_box.get_row_at_index(0)
            if row is None:
                break
            self._list_box.remove(row)

        # Add message row
        row = Gtk.ListBoxRow()
        label = Gtk.Label(label=message)
        label.add_css_class("dim-label")
        label.set_margin_top(24)
        label.set_margin_bottom(24)
        row.set_child(label)
        self._list_box.append(row)

    def _on_sync_calendar(self, button: Gtk.Button, name: str) -> None:
        """Handle sync button click for a single calendar."""
        button.set_sensitive(False)
        button.set_icon_name("content-loading-symbolic")

        # Try using sync service first
        if self._sync_service:
            self._sync_with_service(name, button)
        else:
            self._sync_with_registry(name, button)

    def _sync_with_service(self, name: str, button: Gtk.Button) -> None:
        """Sync using CalendarSyncService."""
        try:
            from core.services.calendar import SyncConfiguration, SyncDirection
            
            actor = self._get_current_actor()
            
            # Create sync configuration
            config = SyncConfiguration(
                provider_type=name,
                account_name=name,
                calendar_id="default",
                config={},
                direction=SyncDirection.IMPORT,
            )
            
            async def do_sync():
                result = await self._sync_service.start_sync(actor, config)
                return result
            
            def on_sync_result(result):
                if isinstance(result, Exception):
                    GLib.idle_add(self._sync_complete, name, str(result))
                elif result.success:
                    GLib.idle_add(self._sync_complete, name, None)
                else:
                    GLib.idle_add(self._sync_complete, name, result.error)
            
            _run_async(do_sync(), on_sync_result)
            
        except Exception as e:
            logger.error("Service sync failed for %s: %s", name, e)
            # Fall back to registry sync
            self._sync_with_registry(name, button)

    def _sync_with_registry(self, name: str, button: Gtk.Button) -> None:
        """Fallback: Sync using CalendarProviderRegistry."""
        def do_sync():
            try:
                from modules.Tools.Base_Tools.calendar import CalendarProviderRegistry

                registry = CalendarProviderRegistry.get_instance()
                backend = registry.get_provider(name)

                if backend and hasattr(backend, "sync"):
                    backend.sync()
                    logger.info("Synced calendar: %s", name)

                GLib.idle_add(self._sync_complete, name, None)
            except Exception as e:
                logger.error("Sync failed for %s: %s", name, e)
                GLib.idle_add(self._sync_complete, name, str(e))

        import threading
        thread = threading.Thread(target=do_sync, daemon=True)
        thread.start()

    def _sync_complete(self, name: str, error: Optional[str]) -> None:
        """Handle sync completion on main thread."""
        widgets = self._row_widgets.get(name, {})
        sync_btn = widgets.get("sync_btn")

        if sync_btn:
            sync_btn.set_sensitive(True)
            sync_btn.set_icon_name("emblem-synchronizing-symbolic")

        # Update status
        if name in self._status_data:
            if error:
                self._status_data[name]["sync_error"] = error
            else:
                self._status_data[name]["last_sync"] = datetime.now(timezone.utc)
                self._status_data[name]["sync_error"] = None

        self._update_display()

        # Notify via message bus (domain events are already published by service)
        if hasattr(self.ATLAS, "message_bus"):
            self.ATLAS.message_bus.publish(
                "CALENDAR_SYNC",
                {"calendar": name, "success": error is None, "error": error},
            )

    def _on_sync_all(self, button: Gtk.Button) -> None:
        """Handle sync all button click."""
        button.set_sensitive(False)

        # Try using sync service first
        if self._sync_service:
            self._sync_all_with_service(button)
        else:
            self._sync_all_with_registry(button)

    def _sync_all_with_service(self, button: Gtk.Button) -> None:
        """Sync all calendars using CalendarSyncService."""
        try:
            from core.services.calendar import SyncConfiguration, SyncDirection
            
            actor = self._get_current_actor()
            
            async def do_sync_all():
                results = []
                for name in self._status_data.keys():
                    config = SyncConfiguration(
                        provider_type=name,
                        account_name=name,
                        calendar_id="default",
                        config={},
                        direction=SyncDirection.IMPORT,
                    )
                    try:
                        result = await self._sync_service.start_sync(actor, config)
                        results.append((name, result))
                    except Exception as e:
                        logger.error("Failed to sync %s: %s", name, e)
                return results
            
            def on_all_complete(results):
                GLib.idle_add(self._sync_all_complete, button)
            
            _run_async(do_sync_all(), on_all_complete)
            
        except Exception as e:
            logger.error("Service sync all failed: %s", e)
            self._sync_all_with_registry(button)

    def _sync_all_with_registry(self, button: Gtk.Button) -> None:
        """Fallback: Sync all using CalendarProviderRegistry."""
        def do_sync_all():
            try:
                from modules.Tools.Base_Tools.calendar import CalendarProviderRegistry

                registry = CalendarProviderRegistry.get_instance()
                providers = registry.list_providers()

                for name, info in providers.items():
                    backend = info.get("backend")
                    if backend and hasattr(backend, "sync"):
                        try:
                            backend.sync()
                            logger.info("Synced calendar: %s", name)
                        except Exception as e:
                            logger.error("Failed to sync %s: %s", name, e)

                GLib.idle_add(self._sync_all_complete, button)
            except Exception as e:
                logger.error("Sync all failed: %s", e)
                GLib.idle_add(self._sync_all_complete, button)

        import threading
        thread = threading.Thread(target=do_sync_all, daemon=True)
        thread.start()

    def _sync_all_complete(self, button: Gtk.Button) -> None:
        """Handle sync all completion."""
        button.set_sensitive(True)
        self._load_status()

        if hasattr(self.ATLAS, "message_bus"):
            self.ATLAS.message_bus.publish(
                "CALENDAR_SYNC", {"calendar": "all", "success": True}
            )

    def refresh(self) -> None:
        """Public method to refresh status."""
        self._load_status()
