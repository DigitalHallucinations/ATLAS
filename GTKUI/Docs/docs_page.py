from __future__ import annotations

import html as html_escape
import logging
from pathlib import Path
from typing import Any

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import Gtk, GLib, Pango

from GTKUI.Utils.utils import apply_css


def _load_webkit() -> tuple[bool, Any | None]:
    """Attempt to load a GTK4-compatible WebKit namespace."""

    gtk_major = getattr(Gtk, "get_major_version", lambda: 0)()
    target_versions = ["6.0", "5.0"] if gtk_major >= 4 else ["4.1", "4.0"]

    for version in target_versions:
        try:
            gi.require_version("WebKit", version)
            from gi.repository import WebKit as WebKitNS  # type: ignore
        except (ImportError, ValueError, gi.RepositoryError):
            continue
        return True, WebKitNS

    return False, None


WEBKIT_AVAILABLE, WebKit = _load_webkit()

logger = logging.getLogger(__name__)

# File extensions to display in the docs browser
DOC_EXTENSIONS = {".md", ".markdown", ".html", ".htm", ".txt", ".rst"}

DEFAULT_PLACEHOLDER_HTML = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>ATLAS Docs</title>
    <style>
      :root { color-scheme: light dark; }
      body { font-family: sans-serif; margin: 1.5rem; line-height: 1.5; }
      h1 { margin-top: 0; }
      .card { border: 1px solid #dadada; border-radius: 8px; padding: 1rem; background: #fafafa; }
      ul { padding-left: 1.25rem; }
      code { background: #f0f0f0; padding: 0.1rem 0.25rem; border-radius: 4px; }
      @media (prefers-color-scheme: dark) {
        .card { background: #2a2a2a; border-color: #444; }
        code { background: #333; }
      }
    </style>
  </head>
  <body>
    <div class="card">
      <h1>ATLAS Documentation</h1>
      <p>Browse the project documentation using the sidebar on the left.</p>
      <ul>
        <li>Click on any document in the tree to view it.</li>
        <li>Folders can be expanded to reveal their contents.</li>
        <li>Use the file picker to open additional Markdown or HTML files.</li>
      </ul>
      <p>Select a document from the sidebar to get started.</p>
    </div>
  </body>
</html>"""


class DocsPage(Gtk.Box):
    """Docs browser with sidebar navigation for browsing documentation."""

    def __init__(self, atlas: Any) -> None:
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        self.ATLAS = atlas
        self._current_doc_path: Path | None = None

        # Find the docs root
        self._repo_root = Path(__file__).resolve().parents[2]
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
        viewer = self._create_web_view(None)
        self._replace_viewer(viewer)
        self._update_status("Select a document from the sidebar to view it.")

    def _load_document(self, path: Path) -> None:
        """Load and display a document."""
        self._current_doc_path = path
        self._path_entry.set_text(str(path))

        if not path.exists():
            viewer = self._create_placeholder_view(path, f"File not found: {path}")
            self._replace_viewer(viewer)
            self._update_status(f"File not found: {path}")
            return

        viewer = self._create_web_view(path)
        self._replace_viewer(viewer)

        # Show relative path if within docs
        try:
            rel_path = path.relative_to(self._repo_root)
            self._update_status(f"Viewing: {rel_path}")
        except ValueError:
            self._update_status(f"Viewing: {path}")

    def _create_web_view(self, doc_path: Path | None) -> Gtk.Widget:
        """Create a WebKit view for rendering documents."""
        if WebKit is None:
            logger.debug("WebKit is unavailable; falling back to text view.")
            if doc_path is None:
                return self._create_placeholder_view(None, "WebKit unavailable. Using text preview.")
            return self._create_text_view(doc_path)

        web_view = WebKit.WebView()
        web_view.set_hexpand(True)
        web_view.set_vexpand(True)
        web_view.set_can_focus(True)

        # Handle navigation for relative links
        web_view.connect("decide-policy", self._on_decide_policy)

        if doc_path is None:
            web_view.load_html(DEFAULT_PLACEHOLDER_HTML, "about:blank")
            return web_view

        if doc_path.suffix.lower() in {".md", ".markdown"}:
            rendered = self._render_markdown(doc_path)
            web_view.load_html(rendered, doc_path.parent.as_uri() + "/")
            return web_view

        try:
            web_view.load_uri(doc_path.as_uri())
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Failed to load documentation %s: %s", doc_path, exc, exc_info=True)
            return self._create_text_view(doc_path)
        return web_view

    def _on_decide_policy(self, web_view: Any, decision: Any, decision_type: Any) -> bool:
        """Handle navigation decisions to intercept relative links."""
        try:
            if decision_type.value_name == "WEBKIT_POLICY_DECISION_TYPE_NAVIGATION_ACTION":
                nav_action = decision.get_navigation_action()
                request = nav_action.get_request()
                uri = request.get_uri()

                # Check if it's a local file link
                if uri.startswith("file://"):
                    file_path = Path(uri.replace("file://", ""))
                    if file_path.suffix.lower() in DOC_EXTENSIONS:
                        # Load in our viewer instead
                        GLib.idle_add(lambda: self._load_document(file_path))
                        decision.ignore()
                        return True
        except Exception:
            pass  # Let default handling proceed

        decision.use()
        return True

    def _render_markdown(self, doc_path: Path) -> str:
        """Render a markdown file to HTML."""
        try:
            import markdown  # type: ignore
        except ImportError:
            logger.debug("Markdown package unavailable; returning raw content for %s", doc_path)
            return self._wrap_raw_content(doc_path)

        try:
            raw = doc_path.read_text(encoding="utf-8")
        except OSError as exc:  # pragma: no cover - defensive logging
            logger.error("Failed to read %s: %s", doc_path, exc, exc_info=True)
            return DEFAULT_PLACEHOLDER_HTML

        md_converter = markdown.Markdown(
            extensions=[
                "fenced_code",
                "tables",
                "toc",
                "attr_list",
                "md_in_html",
                "sane_lists",
                "smarty",
            ],
            output_format="html5",
            extension_configs={
                "toc": {
                    "permalink": True,
                    "anchorlink": True,
                    "title": "Table of Contents",
                    "toc_depth": "2-6",
                }
            },
        )
        html_content = md_converter.convert(raw)
        toc = md_converter.toc or ""

        return self._wrap_html_content(doc_path.name, toc, html_content)

    def _wrap_raw_content(self, doc_path: Path) -> str:
        """Wrap raw file content in basic HTML styling."""
        try:
            raw = doc_path.read_text(encoding="utf-8")
        except OSError:
            raw = "Unable to read file content."

        # Basic markdown-to-HTML conversion for raw display
        escaped = html_escape.escape(raw)

        # Simple transforms for better readability
        lines = escaped.split("\n")
        formatted_lines = []
        in_code_block = False

        for line in lines:
            if line.startswith("```"):
                in_code_block = not in_code_block
                if in_code_block:
                    formatted_lines.append("<pre><code>")
                else:
                    formatted_lines.append("</code></pre>")
            elif in_code_block:
                formatted_lines.append(line)
            elif line.startswith("# "):
                formatted_lines.append(f"<h1>{line[2:]}</h1>")
            elif line.startswith("## "):
                formatted_lines.append(f"<h2>{line[3:]}</h2>")
            elif line.startswith("### "):
                formatted_lines.append(f"<h3>{line[4:]}</h3>")
            elif line.startswith("#### "):
                formatted_lines.append(f"<h4>{line[5:]}</h4>")
            elif line.startswith("- ") or line.startswith("* "):
                formatted_lines.append(f"<li>{line[2:]}</li>")
            elif line.strip() == "":
                formatted_lines.append("<br>")
            else:
                formatted_lines.append(f"<p>{line}</p>")

        content = "\n".join(formatted_lines)
        return self._wrap_html_content(doc_path.name, "", content)

    def _wrap_html_content(self, title: str, toc: str, content: str) -> str:
        """Wrap content in a styled HTML document."""
        return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>{title}</title>
    <style>
      :root {{
        color-scheme: light dark;
      }}
      body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        margin: 1.5rem;
        line-height: 1.6;
        max-width: 900px;
      }}
      h1, h2, h3, h4, h5, h6 {{
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
        font-weight: 600;
      }}
      h1 {{ font-size: 2rem; border-bottom: 1px solid rgba(128,128,128,0.3); padding-bottom: 0.3rem; }}
      h2 {{ font-size: 1.5rem; border-bottom: 1px solid rgba(128,128,128,0.2); padding-bottom: 0.2rem; }}
      h3 {{ font-size: 1.25rem; }}
      p {{ margin: 0.75rem 0; }}
      a {{ color: #0366d6; text-decoration: none; }}
      a:hover {{ text-decoration: underline; }}
      pre, code {{
        font-family: "JetBrains Mono", "Fira Code", "SF Mono", Consolas, monospace;
        font-size: 0.9em;
      }}
      pre {{
        padding: 1rem;
        background: rgba(128,128,128,0.1);
        border-radius: 8px;
        overflow-x: auto;
        border: 1px solid rgba(128,128,128,0.2);
      }}
      code {{
        background: rgba(128,128,128,0.1);
        padding: 0.15rem 0.4rem;
        border-radius: 4px;
      }}
      pre code {{
        background: none;
        padding: 0;
      }}
      blockquote {{
        border-left: 4px solid rgba(128,128,128,0.3);
        margin: 1rem 0;
        padding: 0.5rem 1rem;
        color: rgba(128,128,128,0.8);
        background: rgba(128,128,128,0.05);
        border-radius: 0 8px 8px 0;
      }}
      ul, ol {{
        padding-left: 1.5rem;
        margin: 0.75rem 0;
      }}
      li {{
        margin: 0.25rem 0;
      }}
      table {{
        border-collapse: collapse;
        width: 100%;
        margin: 1rem 0;
      }}
      th, td {{
        border: 1px solid rgba(128,128,128,0.3);
        padding: 0.5rem 0.75rem;
        text-align: left;
      }}
      th {{
        background: rgba(128,128,128,0.1);
        font-weight: 600;
      }}
      tr:nth-child(even) {{
        background: rgba(128,128,128,0.05);
      }}
      .toc {{
        border: 1px solid rgba(128,128,128,0.2);
        border-radius: 8px;
        padding: 1rem;
        background: rgba(128,128,128,0.05);
        margin-bottom: 1.5rem;
      }}
      .toc h1 {{
        margin: 0 0 0.5rem 0;
        font-size: 1.1rem;
        border: none;
        padding: 0;
      }}
      .toc ul {{
        padding-left: 1.25rem;
        margin: 0;
        list-style: disc;
      }}
      .toc a {{
        text-decoration: none;
      }}
      .toc a:hover {{
        text-decoration: underline;
      }}
      img {{
        max-width: 100%;
        height: auto;
        border-radius: 8px;
      }}
      hr {{
        border: none;
        border-top: 1px solid rgba(128,128,128,0.3);
        margin: 2rem 0;
      }}
      /* Dark mode adjustments */
      @media (prefers-color-scheme: dark) {{
        a {{ color: #58a6ff; }}
      }}
    </style>
    <script type="module">
      import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs";
      mermaid.initialize({{ startOnLoad: true }});
    </script>
  </head>
  <body>
    <article>
      {toc}
      {content}
    </article>
  </body>
</html>"""

    def _create_text_view(self, doc_path: Path) -> Gtk.Widget:
        """Create a styled text view for documents when WebKit is unavailable."""
        scroller = Gtk.ScrolledWindow()
        scroller.set_hexpand(True)
        scroller.set_vexpand(True)
        scroller.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)

        try:
            content = doc_path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.debug("Failed to read %s: %s", doc_path, exc, exc_info=True)
            content = f"Unable to read file: {doc_path}\n\nError: {exc}"

        buffer = Gtk.TextBuffer()
        buffer.set_text(content)

        text_view = Gtk.TextView.new_with_buffer(buffer)
        text_view.set_editable(False)
        text_view.set_cursor_visible(False)
        text_view.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
        text_view.set_hexpand(True)
        text_view.set_vexpand(True)
        text_view.set_can_focus(True)
        text_view.set_margin_top(12)
        text_view.set_margin_bottom(12)
        text_view.set_margin_start(12)
        text_view.set_margin_end(12)

        # Try to set a monospace font for better code display
        if hasattr(text_view, "set_monospace"):
            text_view.set_monospace(True)

        scroller.set_child(text_view)
        return scroller

    def _create_placeholder_view(self, doc_path: Path | None, reason: str) -> Gtk.Widget:
        """Create a placeholder view for errors or missing content."""
        scroller = Gtk.ScrolledWindow()
        scroller.set_hexpand(True)
        scroller.set_vexpand(True)
        scroller.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)

        buffer = Gtk.TextBuffer()
        preview_lines = [
            "ATLAS Documentation",
            "",
            reason,
            "",
        ]
        if doc_path is not None:
            preview_lines.append(f"Path: {doc_path}")

        buffer.set_text("\n".join(preview_lines))

        text_view = Gtk.TextView.new_with_buffer(buffer)
        text_view.set_editable(False)
        text_view.set_cursor_visible(False)
        text_view.set_wrap_mode(Gtk.WrapMode.WORD)
        text_view.set_hexpand(True)
        text_view.set_vexpand(True)
        text_view.set_can_focus(True)
        text_view.set_margin_top(12)
        text_view.set_margin_bottom(12)
        text_view.set_margin_start(12)
        text_view.set_margin_end(12)
        scroller.set_child(text_view)
        return scroller

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
