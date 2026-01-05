"""Document Upload Dialog for RAG knowledge base ingestion.

Provides a comprehensive GTK4 dialog for uploading documents to knowledge bases.
Features:
- File picker with multi-file support
- Drag-and-drop zone
- Knowledge base selector
- Progress tracking
- Metadata entry (title, tags, source)
- File type validation
- Size limit enforcement
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Gdk", "4.0")
from gi.repository import Gdk, GLib, Gtk

from GTKUI.Utils.styled_window import AtlasWindow
from GTKUI.Utils.utils import apply_css
from modules.logging.logger import setup_logger

if TYPE_CHECKING:
    from core.services.rag import RAGService
    from modules.storage.knowledge import KnowledgeStore
    from modules.storage.knowledge.base import KnowledgeBase

logger = setup_logger(__name__)


# Supported file extensions for upload
SUPPORTED_EXTENSIONS = {
    # Text and documents
    ".txt": "Plain Text",
    ".md": "Markdown",
    ".markdown": "Markdown",
    ".html": "HTML",
    ".htm": "HTML",
    ".pdf": "PDF",
    ".json": "JSON",
    ".csv": "CSV",
    # Code files
    ".py": "Python",
    ".js": "JavaScript",
    ".ts": "TypeScript",
    ".jsx": "JSX",
    ".tsx": "TSX",
    ".java": "Java",
    ".c": "C",
    ".cpp": "C++",
    ".h": "C Header",
    ".hpp": "C++ Header",
    ".cs": "C#",
    ".go": "Go",
    ".rs": "Rust",
    ".rb": "Ruby",
    ".php": "PHP",
    ".swift": "Swift",
    ".kt": "Kotlin",
    ".scala": "Scala",
    ".sh": "Shell Script",
    ".bash": "Bash Script",
    ".sql": "SQL",
    ".yaml": "YAML",
    ".yml": "YAML",
    ".toml": "TOML",
    ".xml": "XML",
    ".css": "CSS",
    ".scss": "SCSS",
    ".less": "LESS",
}

# Default max file size in MB
DEFAULT_MAX_FILE_SIZE_MB = 50


@dataclass
class UploadFile:
    """Represents a file queued for upload."""

    path: Path
    name: str
    size: int
    extension: str
    status: str = "pending"  # pending, uploading, success, error, skipped
    error: Optional[str] = None
    custom_title: Optional[str] = None


@dataclass
class UploadProgress:
    """Progress tracking for upload operations."""

    total_files: int = 0
    completed_files: int = 0
    current_file: str = ""
    is_uploading: bool = False
    errors: List[str] = field(default_factory=list)

    @property
    def progress_fraction(self) -> float:
        if self.total_files == 0:
            return 0.0
        return self.completed_files / self.total_files


class DocumentUploadDialog(AtlasWindow):
    """Dialog for uploading documents to RAG knowledge bases.

    Usage:
        dialog = DocumentUploadDialog(
            rag_service=rag_service,
            knowledge_store=knowledge_store,
            parent=main_window,
            on_complete=lambda results: print(f"Uploaded {len(results)} files"),
        )
        dialog.present()
    """

    def __init__(
        self,
        *,
        rag_service: Optional["RAGService"] = None,
        knowledge_store: Optional["KnowledgeStore"] = None,
        parent: Optional[Gtk.Window] = None,
        on_complete: Optional[Callable[[List[Dict[str, Any]]], None]] = None,
        max_file_size_mb: float = DEFAULT_MAX_FILE_SIZE_MB,
    ) -> None:
        """Initialize the document upload dialog.

        Args:
            rag_service: RAG service for file ingestion.
            knowledge_store: Knowledge store for KB listing.
            parent: Parent window for modal positioning.
            on_complete: Callback with upload results on completion.
            max_file_size_mb: Maximum file size in MB.
        """
        super().__init__(
            title="Upload Documents",
            default_size=(600, 700),
            modal=True,
            transient_for=parent,
            css_classes=["document-upload-dialog"],
        )

        self._rag_service = rag_service
        self._knowledge_store = knowledge_store
        self._on_complete = on_complete
        self._max_file_size_mb = max_file_size_mb
        self._max_file_size_bytes = int(max_file_size_mb * 1024 * 1024)

        self._files: List[UploadFile] = []
        self._progress = UploadProgress()
        self._knowledge_bases: List["KnowledgeBase"] = []
        self._selected_kb_id: Optional[str] = None

        self._build_ui()
        self._setup_drag_drop()
        self._load_knowledge_bases()

    def _build_ui(self) -> None:
        """Build the dialog UI."""
        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        main_box.set_margin_start(16)
        main_box.set_margin_end(16)
        main_box.set_margin_top(16)
        main_box.set_margin_bottom(16)

        # Knowledge base selector
        kb_section = self._build_kb_selector()
        main_box.append(kb_section)

        # Source tabs (Files / URL)
        self._source_notebook = Gtk.Notebook()
        self._source_notebook.set_vexpand(True)

        # Files tab
        files_tab = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        files_tab.set_margin_top(8)

        # Drop zone / file list area
        self._drop_zone = self._build_drop_zone()
        files_tab.append(self._drop_zone)

        # File list (hidden initially)
        self._file_list_container = self._build_file_list()
        self._file_list_container.set_visible(False)
        files_tab.append(self._file_list_container)

        files_label = Gtk.Label(label="Files")
        self._source_notebook.append_page(files_tab, files_label)

        # URL tab
        url_tab = self._build_url_tab()
        url_label = Gtk.Label(label="URL")
        self._source_notebook.append_page(url_tab, url_label)

        main_box.append(self._source_notebook)

        # Metadata section (collapsible)
        metadata_section = self._build_metadata_section()
        main_box.append(metadata_section)

        # Progress section
        self._progress_section = self._build_progress_section()
        self._progress_section.set_visible(False)
        main_box.append(self._progress_section)

        # Action buttons
        actions = self._build_actions()
        main_box.append(actions)

        self.set_child(main_box)

    def _build_kb_selector(self) -> Gtk.Box:
        """Build the knowledge base selector section."""
        section = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)

        label = Gtk.Label(label="Target Knowledge Base")
        label.add_css_class("heading")
        label.set_xalign(0)
        section.append(label)

        # Dropdown for KB selection
        self._kb_dropdown = Gtk.DropDown()
        self._kb_dropdown.set_tooltip_text("Select knowledge base for document ingestion")
        self._kb_dropdown.connect("notify::selected", self._on_kb_selected)

        # Initially show loading state
        loading_model = Gtk.StringList.new(["Loading knowledge bases..."])
        self._kb_dropdown.set_model(loading_model)
        self._kb_dropdown.set_sensitive(False)

        section.append(self._kb_dropdown)

        # New KB button
        new_kb_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        new_kb_box.set_margin_top(4)

        new_kb_btn = Gtk.Button(label="+ Create New Knowledge Base")
        new_kb_btn.add_css_class("flat")
        new_kb_btn.connect("clicked", self._on_create_kb_clicked)
        new_kb_box.append(new_kb_btn)

        section.append(new_kb_box)

        return section

    def _build_drop_zone(self) -> Gtk.Box:
        """Build the drag-and-drop zone."""
        drop_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        drop_box.add_css_class("drop-zone")
        drop_box.set_vexpand(True)
        drop_box.set_valign(Gtk.Align.CENTER)
        drop_box.set_margin_top(20)
        drop_box.set_margin_bottom(20)

        # Icon
        icon = Gtk.Image.new_from_icon_name("folder-documents-symbolic")
        icon.set_pixel_size(64)
        icon.add_css_class("dim-label")
        drop_box.append(icon)

        # Instructions
        title = Gtk.Label(label="Drop files here")
        title.add_css_class("title-2")
        drop_box.append(title)

        subtitle = Gtk.Label(
            label=f"or click to browse • Max {self._max_file_size_mb}MB per file"
        )
        subtitle.add_css_class("dim-label")
        drop_box.append(subtitle)

        # Supported formats
        formats = Gtk.Label(
            label="Supported: PDF, TXT, Markdown, HTML, JSON, CSV, Python, and more"
        )
        formats.add_css_class("caption")
        formats.add_css_class("dim-label")
        drop_box.append(formats)

        # Button row for files and folder
        btn_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        btn_row.set_halign(Gtk.Align.CENTER)
        btn_row.set_margin_top(12)

        # Browse files button
        browse_btn = Gtk.Button(label="Browse Files")
        browse_btn.add_css_class("suggested-action")
        browse_btn.connect("clicked", self._on_browse_clicked)
        btn_row.append(browse_btn)

        # Browse folder button
        folder_btn = Gtk.Button(label="Browse Folder")
        folder_btn.set_tooltip_text("Select a folder to upload all supported files recursively")
        folder_btn.connect("clicked", self._on_browse_folder_clicked)
        btn_row.append(folder_btn)

        drop_box.append(btn_row)

        # Make the whole zone clickable
        click_controller = Gtk.GestureClick()
        click_controller.connect("released", self._on_drop_zone_clicked)
        drop_box.add_controller(click_controller)

        return drop_box

    def _build_file_list(self) -> Gtk.Box:
        """Build the file list section."""
        section = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        section.set_vexpand(True)

        # Header with file count and clear button
        header = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)

        self._file_count_label = Gtk.Label(label="0 files selected")
        self._file_count_label.set_xalign(0)
        self._file_count_label.set_hexpand(True)
        header.append(self._file_count_label)

        add_more_btn = Gtk.Button(label="Add More")
        add_more_btn.add_css_class("flat")
        add_more_btn.connect("clicked", self._on_browse_clicked)
        header.append(add_more_btn)

        clear_btn = Gtk.Button(label="Clear All")
        clear_btn.add_css_class("flat")
        clear_btn.add_css_class("destructive-action")
        clear_btn.connect("clicked", self._on_clear_files)
        header.append(clear_btn)

        section.append(header)

        # Scrollable file list
        scrolled = Gtk.ScrolledWindow()
        scrolled.set_vexpand(True)
        scrolled.set_min_content_height(150)
        scrolled.set_max_content_height(250)
        scrolled.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)

        self._file_list_box = Gtk.ListBox()
        self._file_list_box.add_css_class("boxed-list")
        self._file_list_box.set_selection_mode(Gtk.SelectionMode.NONE)
        scrolled.set_child(self._file_list_box)

        section.append(scrolled)

        return section

    def _build_metadata_section(self) -> Gtk.Box:
        """Build the metadata entry section."""
        section = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)

        # Collapsible header
        expander = Gtk.Expander(label="Metadata (Optional)")
        expander.add_css_class("heading")

        content = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        content.set_margin_top(8)

        # Tags entry
        tags_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        tags_label = Gtk.Label(label="Tags")
        tags_label.set_xalign(0)
        tags_label.add_css_class("caption")
        tags_box.append(tags_label)

        self._tags_entry = Gtk.Entry()
        self._tags_entry.set_placeholder_text("Enter tags separated by commas")
        self._tags_entry.set_tooltip_text("e.g., documentation, api, reference")
        tags_box.append(self._tags_entry)

        content.append(tags_box)

        # Source entry
        source_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        source_label = Gtk.Label(label="Source")
        source_label.set_xalign(0)
        source_label.add_css_class("caption")
        source_box.append(source_label)

        self._source_entry = Gtk.Entry()
        self._source_entry.set_placeholder_text("Enter source URL or reference")
        self._source_entry.set_tooltip_text("Optional: URL or reference for provenance")
        source_box.append(self._source_entry)

        content.append(source_box)

        # Duplicate handling
        dup_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        dup_box.set_margin_top(8)

        self._check_duplicates = Gtk.CheckButton()
        self._check_duplicates.set_active(True)
        dup_box.append(self._check_duplicates)

        dup_label = Gtk.Label(label="Check for duplicate documents")
        dup_label.set_tooltip_text("Skip files with identical content already in the KB")
        dup_box.append(dup_label)

        content.append(dup_box)

        expander.set_child(content)
        section.append(expander)

        return section

    def _build_url_tab(self) -> Gtk.Box:
        """Build the URL ingestion tab."""
        tab = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        tab.set_margin_start(8)
        tab.set_margin_end(8)
        tab.set_margin_top(16)
        tab.set_margin_bottom(8)

        # Instructions
        info_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        info_box.set_halign(Gtk.Align.CENTER)

        icon = Gtk.Image.new_from_icon_name("web-browser-symbolic")
        icon.set_pixel_size(48)
        icon.add_css_class("dim-label")
        info_box.append(icon)

        title = Gtk.Label(label="Import from URL")
        title.add_css_class("title-3")
        info_box.append(title)

        subtitle = Gtk.Label(label="Enter a web page URL to fetch and ingest its content")
        subtitle.add_css_class("dim-label")
        info_box.append(subtitle)

        tab.append(info_box)

        # URL entry
        url_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        url_box.set_margin_top(16)

        url_label = Gtk.Label(label="Web Page URL")
        url_label.set_xalign(0)
        url_box.append(url_label)

        url_entry_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)

        self._url_entry = Gtk.Entry()
        self._url_entry.set_placeholder_text("https://example.com/docs/page.html")
        self._url_entry.set_hexpand(True)
        self._url_entry.connect("activate", self._on_url_add_clicked)
        url_entry_row.append(self._url_entry)

        add_url_btn = Gtk.Button(label="Add")
        add_url_btn.add_css_class("suggested-action")
        add_url_btn.connect("clicked", self._on_url_add_clicked)
        url_entry_row.append(add_url_btn)

        url_box.append(url_entry_row)
        tab.append(url_box)

        # URL list
        url_list_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        url_list_box.set_margin_top(8)

        url_list_label = Gtk.Label(label="URLs to import")
        url_list_label.set_xalign(0)
        url_list_label.add_css_class("caption")
        url_list_box.append(url_list_label)

        url_scrolled = Gtk.ScrolledWindow()
        url_scrolled.set_vexpand(True)
        url_scrolled.set_min_content_height(100)
        url_scrolled.set_max_content_height(150)
        url_scrolled.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)

        self._url_list = Gtk.ListBox()
        self._url_list.add_css_class("boxed-list")
        self._url_list.set_selection_mode(Gtk.SelectionMode.NONE)
        url_scrolled.set_child(self._url_list)

        url_list_box.append(url_scrolled)
        tab.append(url_list_box)

        # Initialize URL storage
        self._urls: List[str] = []

        return tab

    def _on_url_add_clicked(self, widget: Gtk.Widget) -> None:
        """Handle URL add button click."""
        url = self._url_entry.get_text().strip()
        if not url:
            return

        # Basic URL validation
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        # Check for duplicates
        if url in self._urls:
            return

        self._urls.append(url)
        self._url_entry.set_text("")
        self._refresh_url_list()
        self._update_upload_button_state()

    def _refresh_url_list(self) -> None:
        """Refresh the URL list display."""
        # Clear existing rows
        while True:
            child = self._url_list.get_first_child()
            if child is None:
                break
            self._url_list.remove(child)

        for url in self._urls:
            row = self._create_url_row(url)
            self._url_list.append(row)

    def _create_url_row(self, url: str) -> Gtk.ListBoxRow:
        """Create a row for the URL list."""
        row = Gtk.ListBoxRow()
        row.set_activatable(False)

        box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        box.set_margin_start(8)
        box.set_margin_end(8)
        box.set_margin_top(6)
        box.set_margin_bottom(6)

        # URL icon
        icon = Gtk.Image.new_from_icon_name("web-browser-symbolic")
        box.append(icon)

        # URL text
        label = Gtk.Label(label=url)
        label.set_xalign(0)
        label.set_ellipsize(3)  # PANGO_ELLIPSIZE_END
        label.set_hexpand(True)
        box.append(label)

        # Remove button
        remove_btn = Gtk.Button.new_from_icon_name("window-close-symbolic")
        remove_btn.add_css_class("flat")
        remove_btn.add_css_class("circular")
        remove_btn.set_tooltip_text("Remove")
        remove_btn.connect("clicked", lambda b, u=url: self._remove_url(u))
        box.append(remove_btn)

        row.set_child(box)
        return row

    def _remove_url(self, url: str) -> None:
        """Remove a URL from the list."""
        if url in self._urls:
            self._urls.remove(url)
            self._refresh_url_list()
            self._update_upload_button_state()

    def _build_progress_section(self) -> Gtk.Box:
        """Build the upload progress section."""
        section = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        section.set_margin_top(8)

        # Current file label
        self._current_file_label = Gtk.Label(label="Preparing upload...")
        self._current_file_label.set_xalign(0)
        self._current_file_label.add_css_class("caption")
        section.append(self._current_file_label)

        # Progress bar
        self._progress_bar = Gtk.ProgressBar()
        self._progress_bar.set_show_text(True)
        section.append(self._progress_bar)

        return section

    def _build_actions(self) -> Gtk.Box:
        """Build the action buttons."""
        actions = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        actions.set_halign(Gtk.Align.END)
        actions.set_margin_top(8)

        # Cancel button
        cancel_btn = Gtk.Button(label="Cancel")
        cancel_btn.connect("clicked", self._on_cancel)
        actions.append(cancel_btn)

        # Upload button
        self._upload_btn = Gtk.Button(label="Upload")
        self._upload_btn.add_css_class("suggested-action")
        self._upload_btn.set_sensitive(False)
        self._upload_btn.connect("clicked", self._on_upload)
        actions.append(self._upload_btn)

        return actions

    def _setup_drag_drop(self) -> None:
        """Set up drag and drop handling."""
        drop_target = Gtk.DropTarget.new(Gdk.FileList, Gdk.DragAction.COPY)
        drop_target.connect("drop", self._on_drop)
        drop_target.connect("enter", self._on_drag_enter)
        drop_target.connect("leave", self._on_drag_leave)
        self._drop_zone.add_controller(drop_target)

    def _on_drag_enter(self, target: Gtk.DropTarget, x: float, y: float) -> Gdk.DragAction:
        """Handle drag enter on drop zone."""
        self._drop_zone.add_css_class("drop-zone-hover")
        return Gdk.DragAction.COPY

    def _on_drag_leave(self, target: Gtk.DropTarget) -> None:
        """Handle drag leave on drop zone."""
        self._drop_zone.remove_css_class("drop-zone-hover")

    def _on_drop(
        self, target: Gtk.DropTarget, value: Gdk.FileList, x: float, y: float
    ) -> bool:
        """Handle file drop."""
        self._drop_zone.remove_css_class("drop-zone-hover")

        files = value.get_files()
        for gfile in files:
            path = gfile.get_path()
            if path:
                self._add_file(Path(path))

        return True

    def _on_browse_clicked(self, button: Gtk.Button) -> None:
        """Open file chooser dialog."""
        self._open_file_chooser()

    def _on_browse_folder_clicked(self, button: Gtk.Button) -> None:
        """Open folder chooser dialog for batch upload."""
        self._open_folder_chooser()

    def _on_drop_zone_clicked(
        self, gesture: Gtk.GestureClick, n_press: int, x: float, y: float
    ) -> None:
        """Handle click on drop zone."""
        self._open_file_chooser()

    def _open_file_chooser(self) -> None:
        """Open the file chooser dialog."""
        dialog = Gtk.FileDialog()
        dialog.set_title("Select Documents to Upload")

        # Create file filter for supported types
        filter_all_supported = Gtk.FileFilter()
        filter_all_supported.set_name("All Supported Files")
        for ext in SUPPORTED_EXTENSIONS:
            filter_all_supported.add_pattern(f"*{ext}")
            filter_all_supported.add_pattern(f"*{ext.upper()}")

        filter_text = Gtk.FileFilter()
        filter_text.set_name("Text Documents")
        for ext in [".txt", ".md", ".markdown"]:
            filter_text.add_pattern(f"*{ext}")

        filter_pdf = Gtk.FileFilter()
        filter_pdf.set_name("PDF Documents")
        filter_pdf.add_pattern("*.pdf")

        filter_code = Gtk.FileFilter()
        filter_code.set_name("Code Files")
        for ext in [
            ".py", ".js", ".ts", ".jsx", ".tsx", ".java",
            ".c", ".cpp", ".h", ".go", ".rs", ".rb"
        ]:
            filter_code.add_pattern(f"*{ext}")

        filters = Gtk.FilterListModel()
        filter_list = [filter_all_supported, filter_text, filter_pdf, filter_code]

        # Open multiple files
        dialog.open_multiple(self, None, self._on_files_selected)

    def _open_folder_chooser(self) -> None:
        """Open folder chooser for batch upload."""
        dialog = Gtk.FileDialog()
        dialog.set_title("Select Folder for Batch Upload")
        dialog.select_folder(self, None, self._on_folder_selected)

    def _on_folder_selected(
        self, dialog: Gtk.FileDialog, result: Any
    ) -> None:
        """Handle folder selection from dialog."""
        try:
            gfile = dialog.select_folder_finish(result)
            path = gfile.get_path()
            if path:
                self._add_folder(Path(path))
        except GLib.Error as e:
            if e.code != Gtk.DialogError.DISMISSED:
                logger.warning("Folder selection error: %s", e.message)

    def _on_files_selected(
        self, dialog: Gtk.FileDialog, result: Any
    ) -> None:
        """Handle file selection from dialog."""
        try:
            files = dialog.open_multiple_finish(result)
            for i in range(files.get_n_items()):
                gfile = files.get_item(i)
                path = gfile.get_path()
                if path:
                    self._add_file(Path(path))
        except GLib.Error as e:
            if e.code != Gtk.DialogError.DISMISSED:
                logger.warning("File selection error: %s", e.message)

    def _add_file(self, path: Path) -> None:
        """Add a file to the upload queue."""
        # Check if file exists
        if not path.exists():
            self._show_error(f"File not found: {path.name}")
            return

        # Handle directories recursively
        if path.is_dir():
            self._add_folder(path)
            return

        # Check extension
        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            self._show_error(f"Unsupported file type: {ext}")
            return

        # Check file size
        size = path.stat().st_size
        if size > self._max_file_size_bytes:
            self._show_error(
                f"File too large: {path.name} ({size / (1024*1024):.1f}MB > {self._max_file_size_mb}MB)"
            )
            return

        # Check for duplicates
        if any(f.path == path for f in self._files):
            return

        upload_file = UploadFile(
            path=path,
            name=path.name,
            size=size,
            extension=ext,
        )
        self._files.append(upload_file)
        self._refresh_file_list()

    def _add_folder(self, folder_path: Path, max_depth: int = 10) -> None:
        """Add all supported files from a folder recursively.
        
        Args:
            folder_path: Path to the folder.
            max_depth: Maximum recursion depth to prevent runaway scanning.
        """
        if max_depth <= 0:
            return

        if not folder_path.is_dir():
            return

        added_count = 0
        skipped_count = 0

        try:
            for item in folder_path.iterdir():
                if item.name.startswith("."):
                    # Skip hidden files/folders
                    continue

                if item.is_dir():
                    # Recurse into subdirectories
                    self._add_folder(item, max_depth - 1)
                elif item.is_file():
                    ext = item.suffix.lower()
                    if ext in SUPPORTED_EXTENSIONS:
                        # Check size
                        try:
                            size = item.stat().st_size
                            if size <= self._max_file_size_bytes:
                                # Check for duplicates
                                if not any(f.path == item for f in self._files):
                                    upload_file = UploadFile(
                                        path=item,
                                        name=item.name,
                                        size=size,
                                        extension=ext,
                                    )
                                    self._files.append(upload_file)
                                    added_count += 1
                            else:
                                skipped_count += 1
                        except OSError:
                            skipped_count += 1
                    else:
                        skipped_count += 1

        except PermissionError:
            logger.warning("Permission denied accessing folder: %s", folder_path)

        # Refresh after folder scan
        self._refresh_file_list()
        
        if added_count > 0:
            logger.info("Added %d files from folder: %s", added_count, folder_path.name)

    def _refresh_file_list(self) -> None:
        """Refresh the file list display."""
        # Show/hide drop zone vs file list
        has_files = len(self._files) > 0
        self._drop_zone.set_visible(not has_files)
        self._file_list_container.set_visible(has_files)

        # Update file count
        count = len(self._files)
        self._file_count_label.set_label(
            f"{count} file{'s' if count != 1 else ''} selected"
        )

        # Clear and rebuild list
        while True:
            child = self._file_list_box.get_first_child()
            if child is None:
                break
            self._file_list_box.remove(child)

        for upload_file in self._files:
            row = self._create_file_row(upload_file)
            self._file_list_box.append(row)

        # Update upload button state
        self._update_upload_button_state()

    def _create_file_row(self, upload_file: UploadFile) -> Gtk.ListBoxRow:
        """Create a row for the file list."""
        row = Gtk.ListBoxRow()
        row.set_activatable(False)

        box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        box.set_margin_start(12)
        box.set_margin_end(12)
        box.set_margin_top(8)
        box.set_margin_bottom(8)

        # File icon based on type
        icon_name = self._get_icon_for_extension(upload_file.extension)
        icon = Gtk.Image.new_from_icon_name(icon_name)
        icon.set_pixel_size(24)
        box.append(icon)

        # File info
        info_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        info_box.set_hexpand(True)

        name_label = Gtk.Label(label=upload_file.name)
        name_label.set_xalign(0)
        name_label.set_ellipsize(3)  # PANGO_ELLIPSIZE_END
        info_box.append(name_label)

        # Size and type
        type_name = SUPPORTED_EXTENSIONS.get(upload_file.extension, "Unknown")
        size_str = self._format_size(upload_file.size)
        meta_label = Gtk.Label(label=f"{type_name} • {size_str}")
        meta_label.set_xalign(0)
        meta_label.add_css_class("caption")
        meta_label.add_css_class("dim-label")
        info_box.append(meta_label)

        box.append(info_box)

        # Status indicator
        if upload_file.status == "success":
            status_icon = Gtk.Image.new_from_icon_name("emblem-ok-symbolic")
            status_icon.add_css_class("success")
            box.append(status_icon)
        elif upload_file.status == "error":
            status_icon = Gtk.Image.new_from_icon_name("dialog-error-symbolic")
            status_icon.add_css_class("error")
            status_icon.set_tooltip_text(upload_file.error or "Upload failed")
            box.append(status_icon)
        elif upload_file.status == "uploading":
            spinner = Gtk.Spinner()
            spinner.set_spinning(True)
            box.append(spinner)
        else:
            # Remove button for pending files
            remove_btn = Gtk.Button.new_from_icon_name("window-close-symbolic")
            remove_btn.add_css_class("flat")
            remove_btn.add_css_class("circular")
            remove_btn.set_tooltip_text("Remove")
            remove_btn.connect("clicked", lambda b, f=upload_file: self._remove_file(f))
            box.append(remove_btn)

        row.set_child(box)
        return row

    def _get_icon_for_extension(self, ext: str) -> str:
        """Get icon name for file extension."""
        ext = ext.lower()
        if ext == ".pdf":
            return "x-office-document-symbolic"
        elif ext in [".py", ".js", ".ts", ".java", ".c", ".cpp", ".go", ".rs"]:
            return "text-x-generic-symbolic"
        elif ext in [".md", ".markdown"]:
            return "text-x-generic-symbolic"
        elif ext in [".html", ".htm"]:
            return "text-html-symbolic"
        elif ext in [".json", ".yaml", ".yml", ".toml", ".xml"]:
            return "text-x-generic-symbolic"
        elif ext == ".csv":
            return "x-office-spreadsheet-symbolic"
        else:
            return "text-x-generic-symbolic"

    def _format_size(self, size: int) -> str:
        """Format file size for display."""
        if size < 1024:
            return f"{size} B"
        elif size < 1024 * 1024:
            return f"{size / 1024:.1f} KB"
        else:
            return f"{size / (1024 * 1024):.1f} MB"

    def _remove_file(self, upload_file: UploadFile) -> None:
        """Remove a file from the queue."""
        self._files.remove(upload_file)
        self._refresh_file_list()

    def _on_clear_files(self, button: Gtk.Button) -> None:
        """Clear all files from the queue."""
        self._files.clear()
        self._refresh_file_list()

    def _load_knowledge_bases(self) -> None:
        """Load available knowledge bases asynchronously."""
        if not self._knowledge_store:
            self._show_no_kb_warning()
            return

        async def fetch_kbs():
            try:
                kbs = await self._knowledge_store.list_knowledge_bases()
                GLib.idle_add(self._on_kbs_loaded, kbs)
            except Exception as e:
                logger.error("Failed to load knowledge bases: %s", e)
                GLib.idle_add(self._on_kbs_load_error, str(e))

        asyncio.create_task(fetch_kbs())

    def _on_kbs_loaded(self, kbs: List["KnowledgeBase"]) -> None:
        """Handle loaded knowledge bases."""
        self._knowledge_bases = kbs

        if not kbs:
            self._show_no_kb_warning()
            return

        # Create string list for dropdown
        names = [kb.name for kb in kbs]
        model = Gtk.StringList.new(names)
        self._kb_dropdown.set_model(model)
        self._kb_dropdown.set_sensitive(True)

        # Select first KB by default
        if kbs:
            self._kb_dropdown.set_selected(0)
            self._selected_kb_id = kbs[0].id

        self._update_upload_button_state()

    def _on_kbs_load_error(self, error: str) -> None:
        """Handle KB loading error."""
        model = Gtk.StringList.new([f"Error: {error[:50]}..."])
        self._kb_dropdown.set_model(model)

    def _show_no_kb_warning(self) -> None:
        """Show warning when no knowledge bases are available."""
        model = Gtk.StringList.new(["No knowledge bases available"])
        self._kb_dropdown.set_model(model)
        self._kb_dropdown.set_sensitive(False)

    def _on_kb_selected(self, dropdown: Gtk.DropDown, param: Any) -> None:
        """Handle knowledge base selection."""
        selected = dropdown.get_selected()
        if selected < len(self._knowledge_bases):
            self._selected_kb_id = self._knowledge_bases[selected].id
            self._update_upload_button_state()

    def _on_create_kb_clicked(self, button: Gtk.Button) -> None:
        """Handle create new KB button click."""
        # This would open a KB creation dialog
        # For now, show a placeholder message
        logger.info("Create KB clicked - KB Manager integration pending")
        self._show_info("Knowledge Base Manager coming soon!")

    def _update_upload_button_state(self) -> None:
        """Update the upload button enabled state."""
        has_content = len(self._files) > 0 or len(getattr(self, "_urls", [])) > 0
        can_upload = (
            has_content
            and self._selected_kb_id is not None
            and not self._progress.is_uploading
        )
        self._upload_btn.set_sensitive(can_upload)

    def _on_upload(self, button: Gtk.Button) -> None:
        """Start the upload process."""
        if not self._rag_service:
            self._show_error("RAG service not available")
            return

        if not self._selected_kb_id:
            self._show_error("Please select a knowledge base")
            return

        urls = getattr(self, "_urls", [])
        if not self._files and not urls:
            self._show_error("No files or URLs to upload")
            return

        # Gather metadata
        tags_text = self._tags_entry.get_text().strip()
        tags = [t.strip() for t in tags_text.split(",") if t.strip()] if tags_text else []
        source = self._source_entry.get_text().strip() or None

        metadata = {}
        if tags:
            metadata["tags"] = tags
        if source:
            metadata["source"] = source

        # Start upload
        self._start_upload(metadata)

    def _start_upload(self, metadata: Dict[str, Any]) -> None:
        """Execute the upload process."""
        urls = getattr(self, "_urls", [])
        total_items = len(self._files) + len(urls)
        
        self._progress = UploadProgress(
            total_files=total_items,
            is_uploading=True,
        )
        self._progress_section.set_visible(True)
        self._upload_btn.set_sensitive(False)
        
        # Get duplicate check setting
        check_duplicates = getattr(self, "_check_duplicates", None)
        should_check_duplicates = check_duplicates.get_active() if check_duplicates else True

        async def do_upload():
            results: List[Dict[str, Any]] = []
            item_index = 0
            duplicates_skipped = 0

            # Process files first
            for i, upload_file in enumerate(self._files):
                upload_file.status = "uploading"
                GLib.idle_add(self._update_progress_ui, upload_file.name, item_index)
                GLib.idle_add(self._refresh_file_list)

                try:
                    # Read file content for duplicate check
                    if should_check_duplicates and self._knowledge_store:
                        content = upload_file.path.read_text(errors="ignore")
                        existing_doc = await self._knowledge_store.find_duplicate(
                            self._selected_kb_id, content
                        )
                        
                        if existing_doc:
                            upload_file.status = "skipped"
                            upload_file.error = f"Duplicate of '{existing_doc.title}'"
                            duplicates_skipped += 1
                            results.append({
                                "file": upload_file.name,
                                "success": False,
                                "error": f"Duplicate of '{existing_doc.title}'",
                                "skipped": True,
                            })
                            item_index += 1
                            self._progress.completed_files = item_index
                            GLib.idle_add(self._refresh_file_list)
                            continue

                    success = await self._rag_service.ingest_file(
                        kb_id=self._selected_kb_id,
                        file_path=upload_file.path,
                        title=upload_file.custom_title or upload_file.name,
                        metadata=metadata,
                    )

                    if success:
                        upload_file.status = "success"
                        results.append({
                            "file": upload_file.name,
                            "success": True,
                        })
                    else:
                        upload_file.status = "error"
                        upload_file.error = "Ingestion failed"
                        results.append({
                            "file": upload_file.name,
                            "success": False,
                            "error": "Ingestion failed",
                        })

                except Exception as e:
                    upload_file.status = "error"
                    upload_file.error = str(e)
                    results.append({
                        "file": upload_file.name,
                        "success": False,
                        "error": str(e),
                    })
                    logger.error("Upload error for %s: %s", upload_file.name, e)

                item_index += 1
                self._progress.completed_files = item_index
                GLib.idle_add(self._refresh_file_list)

            # Process URLs
            for url in urls:
                GLib.idle_add(self._update_progress_ui, url[:50] + "...", item_index)

                try:
                    success = await self._ingest_url(url, metadata)
                    if success:
                        results.append({
                            "file": url,
                            "success": True,
                            "type": "url",
                        })
                    else:
                        results.append({
                            "file": url,
                            "success": False,
                            "error": "URL ingestion failed",
                            "type": "url",
                        })
                except Exception as e:
                    results.append({
                        "file": url,
                        "success": False,
                        "error": str(e),
                        "type": "url",
                    })
                    logger.error("URL ingestion error for %s: %s", url, e)

                item_index += 1
                self._progress.completed_files = item_index

            self._progress.is_uploading = False
            GLib.idle_add(self._on_upload_complete, results)

        asyncio.create_task(do_upload())

    async def _ingest_url(self, url: str, metadata: Dict[str, Any]) -> bool:
        """Ingest content from a URL.
        
        Args:
            url: The URL to fetch and ingest.
            metadata: Metadata to attach to the document.
            
        Returns:
            True if ingestion succeeded.
        """
        import aiohttp
        from urllib.parse import urlparse

        try:
            # Fetch URL content
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status != 200:
                        logger.warning("URL fetch failed with status %d: %s", response.status, url)
                        return False

                    content_type = response.headers.get("content-type", "")
                    content = await response.text()

            # Extract title from URL or content
            parsed = urlparse(url)
            title = parsed.path.split("/")[-1] or parsed.netloc

            # Add URL to metadata
            url_metadata = {**metadata, "source_url": url}

            # Ingest as text
            if self._rag_service:
                return await self._rag_service.ingest_text(
                    kb_id=self._selected_kb_id,
                    title=title,
                    content=content,
                    source_uri=url,
                    metadata=url_metadata,
                )
            return False

        except Exception as e:
            logger.error("Failed to ingest URL %s: %s", url, e)
            return False

    def _update_progress_ui(self, current_file: str, index: int) -> None:
        """Update progress UI on main thread."""
        self._current_file_label.set_label(f"Uploading: {current_file}")
        fraction = (index + 1) / self._progress.total_files
        self._progress_bar.set_fraction(fraction)
        self._progress_bar.set_text(f"{index + 1} of {self._progress.total_files}")

    def _on_upload_complete(self, results: List[Dict[str, Any]]) -> None:
        """Handle upload completion."""
        successful = sum(1 for r in results if r.get("success"))
        failed = len(results) - successful

        self._current_file_label.set_label(
            f"Complete: {successful} succeeded, {failed} failed"
        )
        self._progress_bar.set_fraction(1.0)
        self._progress_bar.set_text("Done")

        if self._on_complete:
            self._on_complete(results)

        # Change upload button to close
        self._upload_btn.set_label("Close")
        self._upload_btn.set_sensitive(True)
        self._upload_btn.disconnect_by_func(self._on_upload)
        self._upload_btn.connect("clicked", lambda b: self.close())

    def _on_cancel(self, button: Gtk.Button) -> None:
        """Handle cancel button click."""
        self.close()

    def _show_error(self, message: str) -> None:
        """Show an error message."""
        logger.warning("Upload dialog error: %s", message)
        # Could show a toast or inline error

    def _show_info(self, message: str) -> None:
        """Show an info message."""
        logger.info("Upload dialog info: %s", message)
        # Could show a toast


__all__ = [
    "DocumentUploadDialog",
    "UploadFile",
    "UploadProgress",
    "SUPPORTED_EXTENSIONS",
]
