"""Main documentation browser page with sidebar navigation.

Uses the shared docs_factory for rendering.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import Gtk, Pango

from GTKUI.Utils.utils import apply_css
from GTKUI.Docs.docs_factory import (
    DOC_EXTENSIONS,
    create_web_view,
    create_placeholder_view,
    discover_repo_root,
)

logger = logging.getLogger(__name__)


class DocsPage(Gtk.Box):
    """Docs browser with sidebar navigation for browsing documentation."""

    def __init__(self, atlas: Any) -> None:
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        self.ATLAS = atlas
        self._current_doc_path: Path | None = None

        # Find the docs root
        self._repo_root = discover_repo_root()
        self._docs_root = self._repo_root / "docs"

        for setter_name in (
            "set_margin_top",
            "set_margin_bottom",
            "set_margin_start",
            "set_margin_end",
        ):
            setter = getattr(self, setter_name, None)
            if callable(setter):
                try:
                    setter(12)
                except Exception:  # pragma: no cover - GTK fallback
                    continue

        apply_css()
        self.set_hexpand(True)
        self.set_vexpand(True)

        # Header
        heading = Gtk.Label(label="Documentation Browser")
        heading.set_wrap(True)
        heading.set_xalign(0.0)
        if hasattr(heading, "add_css_class"):
            heading.add_css_class("heading")
        self.append(heading)

        subtitle = Gtk.Label(
            label="Browse project documentation. Select a file from the tree or use the file picker to open external docs."
        )
        subtitle.set_wrap(True)
        subtitle.set_xalign(0.0)
        self.append(subtitle)

        # Toolbar with file picker
        toolbar = self._build_toolbar()
        self.append(toolbar)

        # Main content: sidebar + viewer
        paned = Gtk.Paned(orientation=Gtk.Orientation.HORIZONTAL)
        paned.set_hexpand(True)
        paned.set_vexpand(True)
        paned.set_shrink_start_child(False)
        paned.set_shrink_end_child(False)
        paned.set_position(280)

        # Sidebar with tree view
        sidebar = self._build_sidebar()
        paned.set_start_child(sidebar)

        # Document viewer container
        self._viewer_container = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        self._viewer_container.set_hexpand(True)
        self._viewer_container.set_vexpand(True)
        self._viewer_container.set_halign(Gtk.Align.FILL)
        self._viewer_container.set_valign(Gtk.Align.FILL)

        viewer_frame = Gtk.Frame()
        viewer_frame.set_child(self._viewer_container)
        viewer_frame.set_hexpand(True)
        viewer_frame.set_vexpand(True)
        paned.set_end_child(viewer_frame)

        self.append(paned)

        # Status bar
        self._status_label = Gtk.Label()
        self._status_label.set_wrap(True)
        self._status_label.set_xalign(0.0)
        self._status_label.add_css_class("dim-label")
        self.append(self._status_label)

        # Load initial placeholder
        self._load_placeholder()

    def _build_toolbar(self) -> Gtk.Box:
        """Build the toolbar with path entry and buttons."""
        toolbar = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        toolbar.set_hexpand(True)

        self._path_entry = Gtk.Entry()
        self._path_entry.set_hexpand(True)
        self._path_entry.set_can_focus(True)
        self._path_entry.set_placeholder_text("Enter a documentation path…")
        self._path_entry.connect("activate", self._on_load_requested)

        browse_button = Gtk.Button(label="Browse…")
        browse_button.set_can_focus(True)
        browse_button.connect("clicked", self._on_choose_doc)

        load_button = Gtk.Button(label="Load")
        load_button.set_can_focus(True)
        load_button.set_receives_default(True)
        load_button.connect("clicked", self._on_load_requested)

        refresh_button = Gtk.Button(label="Refresh Tree")
        refresh_button.set_can_focus(True)
        refresh_button.connect("clicked", self._on_refresh_tree)

        toolbar.append(self._path_entry)
        toolbar.append(browse_button)
        toolbar.append(load_button)
        toolbar.append(refresh_button)

        return toolbar

    def _build_sidebar(self) -> Gtk.Widget:
        """Build the sidebar with documentation tree."""
        sidebar_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        sidebar_box.set_size_request(250, -1)

        # Tree label
        tree_label = Gtk.Label(label="Documentation Files")
        tree_label.set_xalign(0.0)
        tree_label.set_margin_start(6)
        tree_label.set_margin_top(6)
        if hasattr(tree_label, "add_css_class"):
            tree_label.add_css_class("heading")
        sidebar_box.append(tree_label)

        # Scrolled window for tree
        scroll = Gtk.ScrolledWindow()
        scroll.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        scroll.set_hexpand(True)
        scroll.set_vexpand(True)

        # Create tree store: icon, display_name, full_path, is_folder
        self._tree_store = Gtk.TreeStore.new([str, str, str, bool])
        self._populate_tree()

        # Tree view
        self._tree_view = Gtk.TreeView(model=self._tree_store)
        self._tree_view.set_headers_visible(False)
        self._tree_view.set_enable_tree_lines(True)
        self._tree_view.set_activate_on_single_click(False)

        # Icon + Name column
        column = Gtk.TreeViewColumn()

        icon_renderer = Gtk.CellRendererText()
        column.pack_start(icon_renderer, False)
        column.add_attribute(icon_renderer, "text", 0)

        name_renderer = Gtk.CellRendererText()
        name_renderer.set_property("ellipsize", Pango.EllipsizeMode.MIDDLE)
        column.pack_start(name_renderer, True)
        column.add_attribute(name_renderer, "text", 1)

        self._tree_view.append_column(column)

        # Connect signals
        self._tree_view.connect("row-activated", self._on_tree_row_activated)
        selection = self._tree_view.get_selection()
        selection.set_mode(Gtk.SelectionMode.SINGLE)

        scroll.set_child(self._tree_view)
        sidebar_box.append(scroll)

        # Expand the docs folder by default
        self._tree_view.expand_all()

        return sidebar_box

    def _populate_tree(self, parent_iter: Any = None, directory: Path | None = None) -> None:
        """Recursively populate the tree with documentation files."""
        if directory is None:
            # Clear existing items
            self._tree_store.clear()

            # Add docs folder as root
            if self._docs_root.exists():
                docs_iter = self._tree_store.append(None, [
                    "📁", "docs", str(self._docs_root), True
                ])
                self._populate_tree(docs_iter, self._docs_root)

            # Add README.md if exists
            readme = self._repo_root / "README.md"
            if readme.exists():
                self._tree_store.append(None, [
                    "📄", "README.md", str(readme), False
                ])
            return

        # Get sorted entries (folders first, then files)
        try:
            entries = sorted(directory.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except PermissionError:
            return

        for entry in entries:
            # Skip hidden files and common non-doc directories
            if entry.name.startswith("."):
                continue
            if entry.name in {"__pycache__", "node_modules", ".git"}:
                continue
            # Skip _audit and other underscore directories but allow files
            if entry.is_dir() and entry.name.startswith("_"):
                continue

            if entry.is_dir():
                # Check if folder contains any documentation files
                if self._has_doc_files(entry):
                    folder_iter = self._tree_store.append(parent_iter, [
                        "📁", entry.name, str(entry), True
                    ])
                    self._populate_tree(folder_iter, entry)
            elif entry.suffix.lower() in DOC_EXTENSIONS:
                icon = self._get_file_icon(entry)
                self._tree_store.append(parent_iter, [
                    icon, entry.name, str(entry), False
                ])

    def _has_doc_files(self, directory: Path) -> bool:
        """Check if a directory contains any documentation files."""
        try:
            for entry in directory.iterdir():
                if entry.name.startswith("."):
                    continue
                if entry.is_dir():
                    if entry.name.startswith("_"):
                        continue
                    if self._has_doc_files(entry):
                        return True
                elif entry.suffix.lower() in DOC_EXTENSIONS:
                    return True
        except PermissionError:
            pass
        return False

    def _get_file_icon(self, path: Path) -> str:
        """Get an appropriate icon for a file type."""
        suffix = path.suffix.lower()
        if suffix in {".md", ".markdown"}:
            return "📝"
        elif suffix in {".html", ".htm"}:
            return "🌐"
        elif suffix == ".txt":
            return "📄"
        elif suffix == ".rst":
            return "📃"
        return "📄"

    def _on_tree_row_activated(self, tree_view: Gtk.TreeView, path: Gtk.TreePath, column: Gtk.TreeViewColumn) -> None:
        """Handle double-click on tree row."""
        model = tree_view.get_model()
        iter_ = model.get_iter(path)
        if iter_ is None:
            return

        is_folder = model.get_value(iter_, 3)
        if is_folder:
            # Toggle expansion
            if tree_view.row_expanded(path):
                tree_view.collapse_row(path)
            else:
                tree_view.expand_row(path, False)
        else:
            # Load the document
            file_path = model.get_value(iter_, 2)
            self._load_document(Path(file_path))

    def _on_refresh_tree(self, _button: Gtk.Button) -> None:
        """Refresh the documentation tree."""
        self._populate_tree()
        self._tree_view.expand_all()
        self._update_status("Documentation tree refreshed.")

    def _on_choose_doc(self, _button: Gtk.Button) -> None:
        """Open file chooser dialog."""
        chooser_cls = getattr(Gtk, "FileChooserNative", None)
        action_enum = getattr(Gtk.FileChooserAction, "OPEN", None) if hasattr(Gtk, "FileChooserAction") else None
        if chooser_cls is None or action_enum is None:
            self._update_status("File chooser unavailable. Enter a path manually.")
            return

        dialog = chooser_cls(
            title="Select documentation file",
            transient_for=self.get_root(),
            action=action_enum,
        )
        dialog.set_modal(True)
        response = dialog.run()
        try:
            if response == Gtk.ResponseType.ACCEPT:
                file_obj = dialog.get_file()
                if file_obj is not None:
                    chosen = file_obj.get_path()
                    if chosen:
                        self._path_entry.set_text(chosen)
                        self._load_document(Path(chosen))
        finally:
            dialog.destroy()

    def _on_load_requested(self, _widget: Gtk.Widget) -> None:
        """Load document from path entry."""
        text = self._path_entry.get_text().strip()
        if not text:
            self._load_placeholder()
            return
        self._load_document(Path(text))

    def _load_placeholder(self) -> None:
        """Show placeholder content."""
        viewer = create_web_view(None)
        self._replace_viewer(viewer)
        self._update_status("Select a document from the sidebar to view it.")

    def _load_document(self, path: Path) -> None:
        """Load and display a document."""
        self._current_doc_path = path
        self._path_entry.set_text(str(path))

        if not path.exists():
            viewer = create_placeholder_view(path, f"File not found: {path}")
            self._replace_viewer(viewer)
            self._update_status(f"File not found: {path}")
            return

        # Create web view with navigation callback for relative links
        viewer = create_web_view(path, on_navigate=self._load_document)
        self._replace_viewer(viewer)

        # Show relative path if within docs
        try:
            rel_path = path.relative_to(self._repo_root)
            self._update_status(f"Viewing: {rel_path}")
        except ValueError:
            self._update_status(f"Viewing: {path}")

    def _replace_viewer(self, widget: Gtk.Widget) -> None:
        """Replace the current viewer with a new widget."""
        child = self._viewer_container.get_first_child()
        while child is not None:
            next_child = child.get_next_sibling()
            self._viewer_container.remove(child)
            child = next_child
        self._viewer_container.append(widget)
        widget.grab_focus()

    def _update_status(self, message: str) -> None:
        """Update the status bar message."""
        self._status_label.set_text(message)
