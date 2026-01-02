"""Shared documentation rendering utilities.

This module provides a factory pattern for creating documentation viewers
with consistent WebKit loading, markdown rendering, and styling across
the application.
"""

from __future__ import annotations

import html as html_escape
import logging
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import Gtk, GLib  # type: ignore[attr-defined]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# WebKit Loading
# ---------------------------------------------------------------------------

_REPOSITORY_ERRORS: tuple[type[BaseException], ...] = (ImportError, ValueError)
_repository_error = getattr(gi, "RepositoryError", None)
if isinstance(_repository_error, type) and issubclass(_repository_error, BaseException):
    _REPOSITORY_ERRORS = (*_REPOSITORY_ERRORS, _repository_error)


def _load_webkit() -> tuple[bool, Any | None]:
    """Attempt to load a GTK4-compatible WebKit namespace.
    
    Tries WebKit (GTK4) first, then falls back to WebKit2 (GTK3 compat).
    """
    gtk_major = getattr(Gtk, "get_major_version", lambda: 0)()
    target_versions = ["6.0", "5.0"] if gtk_major >= 4 else ["4.1", "4.0"]

    for ns_name in ["WebKit", "WebKit2"]:
        for version in target_versions:
            try:
                gi.require_version(ns_name, version)
                if ns_name == "WebKit":
                    from gi.repository import WebKit as WebKitNS  # type: ignore
                else:
                    from gi.repository import WebKit2 as WebKitNS  # type: ignore
                logger.debug("Loaded %s %s", ns_name, version)
                return True, WebKitNS
            except _REPOSITORY_ERRORS:
                continue

    logger.warning("WebKit not available; falling back to text view")
    return False, None


WEBKIT_AVAILABLE, WebKit = _load_webkit()

# ---------------------------------------------------------------------------
# Document Extensions
# ---------------------------------------------------------------------------

DOC_EXTENSIONS = {".md", ".markdown", ".html", ".htm", ".txt", ".rst"}

# ---------------------------------------------------------------------------
# HTML Templates & Styling
# ---------------------------------------------------------------------------

DEFAULT_PLACEHOLDER_HTML = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>ATLAS Documentation</title>
    <style>
      :root { color-scheme: light dark; }
      body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 1.5rem; line-height: 1.5; }
      h1 { margin-top: 0; }
      .card { border: 1px solid rgba(128,128,128,0.3); border-radius: 8px; padding: 1rem; background: rgba(128,128,128,0.05); }
      ul { padding-left: 1.25rem; }
      code { background: rgba(128,128,128,0.15); padding: 0.1rem 0.25rem; border-radius: 4px; }
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


def get_html_styles() -> str:
    """Return the shared CSS styles for rendered documentation."""
    return """
      :root {
        color-scheme: light dark;
      }
      body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        margin: 1.5rem;
        line-height: 1.6;
        max-width: 900px;
      }
      h1, h2, h3, h4, h5, h6 {
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
        font-weight: 600;
      }
      h1 { font-size: 2rem; border-bottom: 1px solid rgba(128,128,128,0.3); padding-bottom: 0.3rem; }
      h2 { font-size: 1.5rem; border-bottom: 1px solid rgba(128,128,128,0.2); padding-bottom: 0.2rem; }
      h3 { font-size: 1.25rem; }
      p { margin: 0.75rem 0; }
      a { color: #0366d6; text-decoration: none; }
      a:hover { text-decoration: underline; }
      pre, code {
        font-family: "JetBrains Mono", "Fira Code", "SF Mono", Consolas, monospace;
        font-size: 0.9em;
      }
      pre {
        padding: 1rem;
        background: rgba(128,128,128,0.1);
        border-radius: 8px;
        overflow-x: auto;
        border: 1px solid rgba(128,128,128,0.2);
      }
      code {
        background: rgba(128,128,128,0.1);
        padding: 0.15rem 0.4rem;
        border-radius: 4px;
      }
      pre code {
        background: none;
        padding: 0;
      }
      blockquote {
        border-left: 4px solid rgba(128,128,128,0.3);
        margin: 1rem 0;
        padding: 0.5rem 1rem;
        color: rgba(128,128,128,0.8);
        background: rgba(128,128,128,0.05);
        border-radius: 0 8px 8px 0;
      }
      ul, ol {
        padding-left: 1.5rem;
        margin: 0.75rem 0;
      }
      li {
        margin: 0.25rem 0;
      }
      table {
        border-collapse: collapse;
        width: 100%;
        margin: 1rem 0;
      }
      th, td {
        border: 1px solid rgba(128,128,128,0.3);
        padding: 0.5rem 0.75rem;
        text-align: left;
      }
      th {
        background: rgba(128,128,128,0.1);
        font-weight: 600;
      }
      tr:nth-child(even) {
        background: rgba(128,128,128,0.05);
      }
      .toc {
        border: 1px solid rgba(128,128,128,0.2);
        border-radius: 8px;
        padding: 1rem;
        background: rgba(128,128,128,0.05);
        margin-bottom: 1.5rem;
      }
      .toc h1 {
        margin: 0 0 0.5rem 0;
        font-size: 1.1rem;
        border: none;
        padding: 0;
      }
      .toc ul {
        padding-left: 1.25rem;
        margin: 0;
        list-style: disc;
      }
      .toc a {
        text-decoration: none;
      }
      .toc a:hover {
        text-decoration: underline;
      }
      img {
        max-width: 100%;
        height: auto;
        border-radius: 8px;
      }
      hr {
        border: none;
        border-top: 1px solid rgba(128,128,128,0.3);
        margin: 2rem 0;
      }
      /* Dark mode adjustments */
      @media (prefers-color-scheme: dark) {
        a { color: #58a6ff; }
      }
      /* Mermaid diagram styling */
      pre.mermaid {
        background: transparent;
        border: none;
        text-align: center;
      }
    """


def wrap_html_content(title: str, toc: str, content: str) -> str:
    """Wrap content in a styled HTML document."""
    styles = get_html_styles()
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>{title}</title>
    <style>{styles}</style>
    <script type="module">
      import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs";
      // Transform fenced code blocks with language-mermaid class to mermaid format
      document.querySelectorAll('pre > code.language-mermaid').forEach(codeEl => {{
        const preEl = codeEl.parentElement;
        const content = codeEl.textContent;
        preEl.className = 'mermaid';
        preEl.textContent = content;
      }});
      mermaid.initialize({{ startOnLoad: true, theme: 'default' }});
    </script>
  </head>
  <body>
    <article>
      {toc}
      {content}
    </article>
  </body>
</html>"""


# ---------------------------------------------------------------------------
# Markdown Rendering
# ---------------------------------------------------------------------------

def render_markdown(doc_path: Path) -> str:
    """Render a markdown file to styled HTML.
    
    Falls back to basic formatting if the markdown package is unavailable.
    """
    try:
        import markdown  # type: ignore
    except ImportError:
        logger.debug("Markdown package unavailable; using basic rendering for %s", doc_path)
        return render_raw_markdown(doc_path)

    try:
        raw = doc_path.read_text(encoding="utf-8")
    except OSError as exc:
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
        output_format="html",
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
    toc = getattr(md_converter, "toc", "") or ""

    return wrap_html_content(doc_path.name, toc, html_content)


def render_raw_markdown(doc_path: Path) -> str:
    """Render markdown with basic formatting when markdown package unavailable."""
    try:
        raw = doc_path.read_text(encoding="utf-8")
    except OSError:
        raw = "Unable to read file content."

    escaped = html_escape.escape(raw)
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
        elif line.startswith("##### "):
            formatted_lines.append(f"<h5>{line[6:]}</h5>")
        elif line.startswith("###### "):
            formatted_lines.append(f"<h6>{line[7:]}</h6>")
        elif line.startswith("- ") or line.startswith("* "):
            formatted_lines.append(f"<li>{line[2:]}</li>")
        elif line.startswith("> "):
            formatted_lines.append(f"<blockquote>{line[2:]}</blockquote>")
        elif line.strip() == "":
            formatted_lines.append("<br>")
        else:
            formatted_lines.append(f"<p>{line}</p>")

    content = "\n".join(formatted_lines)
    return wrap_html_content(doc_path.name, "", content)


# ---------------------------------------------------------------------------
# Document Viewer Factory
# ---------------------------------------------------------------------------

@runtime_checkable
class DocViewerCallback(Protocol):
    """Protocol for document navigation callbacks."""
    def __call__(self, path: Path) -> None: ...


class DocsViewerFactory:
    """Factory for creating documentation viewer widgets.
    
    Usage:
        factory = DocsViewerFactory()
        
        # Create a WebKit view for a document
        viewer = factory.create_viewer(Path("docs/README.md"))
        
        # Create with navigation callback for link handling
        viewer = factory.create_viewer(
            Path("docs/README.md"),
            on_navigate=lambda p: load_document(p)
        )
        
        # Create placeholder
        viewer = factory.create_placeholder("Select a document to view")
    """
    
    def __init__(self) -> None:
        self._webkit = WebKit
        self._available = WEBKIT_AVAILABLE
    
    @property
    def webkit_available(self) -> bool:
        """Check if WebKit is available for rendering."""
        return self._available
    
    def create_viewer(
        self,
        doc_path: Path,
        on_navigate: DocViewerCallback | None = None,
    ) -> Gtk.Widget:
        """Create a viewer widget for the given document.
        
        Args:
            doc_path: Path to the document to display
            on_navigate: Optional callback for handling navigation to other docs
            
        Returns:
            A GTK widget displaying the document
        """
        if not doc_path.exists():
            return self.create_placeholder(f"File not found: {doc_path}")
        
        if self._webkit is None:
            return self._create_text_view(doc_path)
        
        return self._create_web_view(doc_path, on_navigate)
    
    def create_placeholder(self, message: str | None = None) -> Gtk.Widget:
        """Create a placeholder viewer widget.
        
        Args:
            message: Optional message to display
            
        Returns:
            A GTK widget showing placeholder content
        """
        if self._webkit is None:
            return self._create_text_placeholder(message)
        
        web_view = self._webkit.WebView()
        web_view.set_hexpand(True)
        web_view.set_vexpand(True)
        web_view.set_can_focus(True)
        
        if message:
            html = DEFAULT_PLACEHOLDER_HTML.replace(
                "<p>Select a document from the sidebar to get started.</p>",
                f"<p>{html_escape.escape(message)}</p>"
            )
            web_view.load_html(html, "about:blank")
        else:
            web_view.load_html(DEFAULT_PLACEHOLDER_HTML, "about:blank")
        
        return web_view
    
    def _create_web_view(
        self,
        doc_path: Path,
        on_navigate: DocViewerCallback | None = None,
    ) -> Gtk.Widget:
        """Create a WebKit-based viewer."""
        assert self._webkit is not None, "WebKit is required for web view"
        web_view = self._webkit.WebView()
        web_view.set_hexpand(True)
        web_view.set_vexpand(True)
        web_view.set_can_focus(True)
        
        if on_navigate is not None:
            web_view.connect(
                "decide-policy",
                self._make_policy_handler(on_navigate),
            )
        
        if doc_path.suffix.lower() in {".md", ".markdown"}:
            rendered = render_markdown(doc_path)
            web_view.load_html(rendered, doc_path.parent.as_uri() + "/")
        else:
            try:
                web_view.load_uri(doc_path.as_uri())
            except Exception as exc:
                logger.error("Failed to load %s: %s", doc_path, exc, exc_info=True)
                return self._create_text_view(doc_path)
        
        return web_view
    
    def _make_policy_handler(
        self,
        on_navigate: DocViewerCallback,
    ):
        """Create a policy decision handler for link navigation."""
        def handler(web_view: Any, decision: Any, decision_type: Any) -> bool:
            try:
                if decision_type.value_name == "WEBKIT_POLICY_DECISION_TYPE_NAVIGATION_ACTION":
                    nav_action = decision.get_navigation_action()
                    request = nav_action.get_request()
                    uri = request.get_uri()
                    
                    if uri.startswith("file://"):
                        file_path = Path(uri.replace("file://", ""))
                        if file_path.suffix.lower() in DOC_EXTENSIONS:
                            GLib.idle_add(lambda: on_navigate(file_path))
                            decision.ignore()
                            return True
            except Exception:
                pass
            
            decision.use()
            return True
        
        return handler
    
    def _create_text_view(self, doc_path: Path) -> Gtk.Widget:
        """Create a text-based viewer as fallback."""
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

        if hasattr(text_view, "set_monospace"):
            text_view.set_monospace(True)

        scroller.set_child(text_view)
        return scroller
    
    def _create_text_placeholder(self, message: str | None) -> Gtk.Widget:
        """Create a text-based placeholder."""
        scroller = Gtk.ScrolledWindow()
        scroller.set_hexpand(True)
        scroller.set_vexpand(True)
        scroller.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)

        buffer = Gtk.TextBuffer()
        lines = [
            "ATLAS Documentation",
            "",
            message or "Select a document to view.",
        ]
        buffer.set_text("\n".join(lines))

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


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------

_default_factory: DocsViewerFactory | None = None


def get_factory() -> DocsViewerFactory:
    """Get the default docs viewer factory instance."""
    global _default_factory
    if _default_factory is None:
        _default_factory = DocsViewerFactory()
    return _default_factory


def create_doc_viewer(
    doc_path: Path,
    on_navigate: DocViewerCallback | None = None,
) -> Gtk.Widget:
    """Create a document viewer widget.
    
    Convenience function that uses the default factory.
    """
    return get_factory().create_viewer(doc_path, on_navigate)


def create_placeholder(message: str | None = None) -> Gtk.Widget:
    """Create a placeholder viewer widget.
    
    Convenience function that uses the default factory.
    """
    return get_factory().create_placeholder(message)
