"""Category panel for managing calendar categories.

Provides a GTK 4 panel displaying all calendar categories with
visibility toggles, color indicators, and management actions.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import GLib, Gtk, Pango # type: ignore[import-untyped]

from .category_dialog import CategoryDialog

logger = logging.getLogger(__name__)


class CategoryPanel(Gtk.Box):
    """Panel displaying calendar categories with visibility controls.

    This panel shows all calendar categories from the Master Calendar store,
    allowing users to:
    - Toggle category visibility
    - Set the default category
    - Add, edit, and delete custom categories
    - View built-in categories (with restricted editing)
    """

    def __init__(self, atlas: Any, parent_controller: Any) -> None:
        """Initialize the category panel.

        Args:
            atlas: ATLAS instance for accessing services.
            parent_controller: Parent CalendarManagement controller.
        """
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        self.ATLAS = atlas
        self._controller = parent_controller
        self._category_rows: Dict[str, Gtk.ListBoxRow] = {}
        self._repo = None

        self.set_hexpand(True)
        self.set_vexpand(True)

        self._build_ui()
        GLib.idle_add(self._init_repository)

    def _build_ui(self) -> None:
        """Build the panel UI."""
        # Header with quick actions
        header = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        header.set_margin_start(16)
        header.set_margin_end(16)
        header.set_margin_top(12)
        header.set_margin_bottom(8)

        # Show/Hide All buttons
        show_all_btn = Gtk.Button(label="Show All")
        show_all_btn.add_css_class("flat")
        show_all_btn.connect("clicked", self._on_show_all_clicked)
        header.append(show_all_btn)

        hide_all_btn = Gtk.Button(label="Hide All")
        hide_all_btn.add_css_class("flat")
        hide_all_btn.connect("clicked", self._on_hide_all_clicked)
        header.append(hide_all_btn)

        # Spacer
        spacer = Gtk.Box()
        spacer.set_hexpand(True)
        header.append(spacer)

        # Add category button
        add_btn = Gtk.Button()
        add_btn.set_icon_name("list-add-symbolic")
        add_btn.set_tooltip_text("Add Category")
        add_btn.add_css_class("suggested-action")
        add_btn.connect("clicked", self._on_add_clicked)
        header.append(add_btn)

        self.append(header)

        # Separator
        self.append(Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL))

        # Scrolled window for category list
        scrolled = Gtk.ScrolledWindow()
        scrolled.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        scrolled.set_hexpand(True)
        scrolled.set_vexpand(True)

        # List box for categories
        self._list_box = Gtk.ListBox()
        self._list_box.set_selection_mode(Gtk.SelectionMode.NONE)
        self._list_box.add_css_class("boxed-list")
        self._list_box.set_margin_start(16)
        self._list_box.set_margin_end(16)
        self._list_box.set_margin_top(8)
        self._list_box.set_margin_bottom(16)

        scrolled.set_child(self._list_box)
        self.append(scrolled)

        # Empty state placeholder
        self._empty_label = Gtk.Label(
            label="No categories found.\n\nClick + to add a category."
        )
        self._empty_label.set_justify(Gtk.Justification.CENTER)
        self._empty_label.add_css_class("dim-label")
        self._empty_label.set_margin_top(48)
        self._empty_label.set_visible(False)
        self.append(self._empty_label)

        # Loading indicator
        self._loading_spinner = Gtk.Spinner()
        self._loading_spinner.set_size_request(32, 32)
        self._loading_spinner.set_margin_top(48)
        self._loading_spinner.set_halign(Gtk.Align.CENTER)
        self._loading_spinner.start()
        self.append(self._loading_spinner)

    def _init_repository(self) -> bool:
        """Initialize the calendar store repository."""
        try:
            from modules.calendar_store import CalendarStoreRepository
            from sqlalchemy.orm import sessionmaker

            # Get the database session factory from ATLAS or create one
            session_factory = self._get_session_factory()
            if session_factory:
                self._repo = CalendarStoreRepository(session_factory)
                self.refresh()
            else:
                logger.warning("No database session factory available")
                self._show_error_state("Database not configured")
        except ImportError as exc:
            logger.warning("Calendar store not available: %s", exc)
            self._show_error_state("Calendar store not available")
        except Exception as exc:
            logger.error("Failed to initialize category repository: %s", exc)
            self._show_error_state(f"Error: {exc}")

        return False

    def _get_session_factory(self) -> Optional[Any]:
        """Get the SQLAlchemy session factory from ATLAS."""
        try:
            # Try to get from ATLAS services
            if hasattr(self.ATLAS, "services"):
                services = self.ATLAS.services
                if hasattr(services, "calendar_session_factory"):
                    return services.calendar_session_factory
                if hasattr(services, "session_factory"):
                    return services.session_factory

            # Try to create from config
            from core.config import ConfigManager

            config = ConfigManager()
            db_url = config.get_config("database.url") or config.get_config("database.connection_string")

            if db_url:
                from sqlalchemy import create_engine
                from sqlalchemy.orm import sessionmaker
                from modules.calendar_store import create_schema

                engine = create_engine(db_url)
                create_schema(engine, seed_categories=True)
                return sessionmaker(bind=engine)

        except Exception as exc:
            logger.debug("Could not get session factory: %s", exc)

        return None

    def _show_error_state(self, message: str) -> None:
        """Show an error state in the panel."""
        self._loading_spinner.stop()
        self._loading_spinner.set_visible(False)
        self._list_box.set_visible(False)
        self._empty_label.set_label(message)
        self._empty_label.set_visible(True)

    def refresh(self) -> None:
        """Refresh the category list from the database."""
        GLib.idle_add(self._do_refresh)

    def _do_refresh(self) -> bool:
        """Perform the refresh on the main thread."""
        self._loading_spinner.stop()
        self._loading_spinner.set_visible(False)

        # Clear existing rows
        while True:
            row = self._list_box.get_first_child()
            if row is None:
                break
            self._list_box.remove(row)
        self._category_rows.clear()

        if self._repo is None:
            self._show_error_state("Repository not initialized")
            return False

        try:
            categories = self._repo.list_categories(include_hidden=True)

            if not categories:
                self._list_box.set_visible(False)
                self._empty_label.set_label(
                    "No categories found.\n\nClick + to add a category."
                )
                self._empty_label.set_visible(True)
            else:
                self._empty_label.set_visible(False)
                self._list_box.set_visible(True)

                for cat in categories:
                    row = self._create_category_row(cat)
                    self._list_box.append(row)
                    self._category_rows[cat["id"]] = row

        except Exception as exc:
            logger.error("Failed to load categories: %s", exc)
            self._show_error_state(f"Failed to load: {exc}")

        return False

    def _create_category_row(self, category: Dict[str, Any]) -> Gtk.ListBoxRow:
        """Create a list row widget for a category."""
        row = Gtk.ListBoxRow()
        row.set_activatable(False)

        box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        box.set_margin_top(8)
        box.set_margin_bottom(8)
        box.set_margin_start(12)
        box.set_margin_end(12)

        # Visibility toggle (checkbox)
        visible_check = Gtk.CheckButton()
        visible_check.set_active(category.get("is_visible", True))
        visible_check.set_tooltip_text("Show/hide in calendar")
        visible_check.connect(
            "toggled", self._on_visibility_toggled, category["id"]
        )
        box.append(visible_check)

        # Color indicator
        color = category.get("color", "#4285F4")
        color_box = Gtk.Box()
        color_box.set_size_request(16, 16)
        color_box.add_css_class("category-color-dot")
        css_provider = Gtk.CssProvider()
        css_provider.load_from_data(
            f".category-color-dot {{ background-color: {color}; border-radius: 50%; }}".encode()
        )
        color_box.get_style_context().add_provider(
            css_provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )
        box.append(color_box)

        # Icon
        icon = category.get("icon", "📅")
        icon_label = Gtk.Label(label=icon)
        icon_label.set_size_request(24, -1)
        box.append(icon_label)

        # Name and badges
        text_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        text_box.set_hexpand(True)

        name_label = Gtk.Label(label=category.get("name", "Unknown"))
        name_label.set_xalign(0.0)
        name_label.set_ellipsize(Pango.EllipsizeMode.END)
        if not category.get("is_visible", True):
            name_label.add_css_class("dim-label")
        text_box.append(name_label)

        # Default badge
        if category.get("is_default"):
            default_badge = Gtk.Label(label="Default")
            default_badge.add_css_class("caption")
            default_badge.add_css_class("accent")
            text_box.append(default_badge)

        # Built-in badge
        if category.get("is_builtin"):
            builtin_badge = Gtk.Label(label="Built-in")
            builtin_badge.add_css_class("caption")
            builtin_badge.add_css_class("dim-label")
            text_box.append(builtin_badge)

        box.append(text_box)

        # Action buttons
        action_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)

        # Set as default button (if not already default)
        if not category.get("is_default"):
            default_btn = Gtk.Button()
            default_btn.set_icon_name("starred-symbolic")
            default_btn.set_tooltip_text("Set as Default")
            default_btn.add_css_class("flat")
            default_btn.connect(
                "clicked", self._on_set_default_clicked, category["id"]
            )
            action_box.append(default_btn)

        # Edit button
        edit_btn = Gtk.Button()
        edit_btn.set_icon_name("document-edit-symbolic")
        edit_btn.set_tooltip_text("Edit")
        edit_btn.add_css_class("flat")
        edit_btn.connect("clicked", self._on_edit_clicked, category)
        action_box.append(edit_btn)

        # Delete button (only for custom categories)
        if not category.get("is_builtin"):
            delete_btn = Gtk.Button()
            delete_btn.set_icon_name("user-trash-symbolic")
            delete_btn.set_tooltip_text("Delete")
            delete_btn.add_css_class("flat")
            delete_btn.add_css_class("destructive-action")
            delete_btn.connect("clicked", self._on_delete_clicked, category)
            action_box.append(delete_btn)

        box.append(action_box)
        row.set_child(box)
        return row

    # -------------------------------------------------------------------------
    # Event handlers
    # -------------------------------------------------------------------------

    def _on_show_all_clicked(self, button: Gtk.Button) -> None:
        """Handle Show All button click."""
        if self._repo is None:
            return
        try:
            categories = self._repo.list_categories(include_hidden=True)
            for cat in categories:
                if not cat.get("is_visible"):
                    self._repo.toggle_category_visibility(cat["id"], visible=True)
            self.refresh()
        except Exception as exc:
            logger.error("Failed to show all categories: %s", exc)

    def _on_hide_all_clicked(self, button: Gtk.Button) -> None:
        """Handle Hide All button click."""
        if self._repo is None:
            return
        try:
            categories = self._repo.list_categories(include_hidden=True)
            for cat in categories:
                if cat.get("is_visible"):
                    self._repo.toggle_category_visibility(cat["id"], visible=False)
            self.refresh()
        except Exception as exc:
            logger.error("Failed to hide all categories: %s", exc)

    def _on_add_clicked(self, button: Gtk.Button) -> None:
        """Handle Add Category button click."""
        dialog = CategoryDialog(
            parent=self._get_parent_window(),
            atlas=self.ATLAS,
            mode="add",
        )
        dialog.connect("response", self._on_add_dialog_response)
        dialog.present()

    def _on_add_dialog_response(
        self, dialog: CategoryDialog, response_id: int
    ) -> None:
        """Handle add category dialog response."""
        if response_id == Gtk.ResponseType.OK:
            data = dialog.get_category_data()
            if data and self._repo:
                try:
                    self._repo.create_category(
                        name=data["name"],
                        color=data["color"],
                        icon=data.get("icon"),
                        description=data.get("description"),
                        is_visible=data.get("is_visible", True),
                        is_default=data.get("is_default", False),
                    )
                    self.refresh()
                except Exception as exc:
                    logger.error("Failed to create category: %s", exc)
                    self._show_error_toast(f"Failed to create: {exc}")
        dialog.destroy()

    def _on_visibility_toggled(
        self, check: Gtk.CheckButton, category_id: str
    ) -> None:
        """Handle visibility checkbox toggle."""
        if self._repo is None:
            return
        try:
            self._repo.toggle_category_visibility(
                category_id, visible=check.get_active()
            )
            # Update the row styling
            self.refresh()
        except Exception as exc:
            logger.error("Failed to toggle visibility: %s", exc)
            # Revert the checkbox
            check.set_active(not check.get_active())

    def _on_set_default_clicked(self, button: Gtk.Button, category_id: str) -> None:
        """Handle Set as Default button click."""
        if self._repo is None:
            return
        try:
            self._repo.set_default_category(category_id)
            self.refresh()
        except Exception as exc:
            logger.error("Failed to set default category: %s", exc)

    def _on_edit_clicked(
        self, button: Gtk.Button, category: Dict[str, Any]
    ) -> None:
        """Handle Edit button click."""
        dialog = CategoryDialog(
            parent=self._get_parent_window(),
            atlas=self.ATLAS,
            mode="edit",
            category=category,
        )
        dialog.connect("response", self._on_edit_dialog_response, category["id"])
        dialog.present()

    def _on_edit_dialog_response(
        self, dialog: CategoryDialog, response_id: int, category_id: str
    ) -> None:
        """Handle edit category dialog response."""
        if response_id == Gtk.ResponseType.OK:
            data = dialog.get_category_data()
            if data and self._repo:
                try:
                    changes = {}
                    for key in ["name", "color", "icon", "description", "is_visible"]:
                        if key in data:
                            changes[key] = data[key]

                    if data.get("is_default"):
                        self._repo.set_default_category(category_id)

                    if changes:
                        self._repo.update_category(category_id, changes=changes)

                    self.refresh()
                except Exception as exc:
                    logger.error("Failed to update category: %s", exc)
                    self._show_error_toast(f"Failed to update: {exc}")
        dialog.destroy()

    def _on_delete_clicked(
        self, button: Gtk.Button, category: Dict[str, Any]
    ) -> None:
        """Handle Delete button click."""
        # Show confirmation dialog
        dialog = Gtk.MessageDialog(
            transient_for=self._get_parent_window(),
            modal=True,
            message_type=Gtk.MessageType.QUESTION,
            buttons=Gtk.ButtonsType.YES_NO,
            text=f"Delete category '{category.get('name')}'?",
        )
        dialog.format_secondary_text(
            "Events in this category will be moved to the default category."
        )
        dialog.connect("response", self._on_delete_confirm, category["id"])
        dialog.present()

    def _on_delete_confirm(
        self, dialog: Gtk.MessageDialog, response_id: int, category_id: str
    ) -> None:
        """Handle delete confirmation dialog response."""
        if response_id == Gtk.ResponseType.YES and self._repo:
            try:
                self._repo.delete_category(category_id)
                self.refresh()
            except Exception as exc:
                logger.error("Failed to delete category: %s", exc)
                self._show_error_toast(f"Failed to delete: {exc}")
        dialog.destroy()

    def _get_parent_window(self) -> Optional[Gtk.Window]:
        """Get the parent window for dialogs."""
        if hasattr(self._controller, "parent_window"):
            return self._controller.parent_window
        return None

    def _show_error_toast(self, message: str) -> None:
        """Show an error toast notification."""
        logger.warning("Category panel error: %s", message)
        # Could integrate with ATLAS toast system if available


__all__ = ["CategoryPanel"]
