"""Shared documentation rendering factory for ATLAS.

This module provides unified markdown rendering and WebKit view creation
used by both the main Docs browser and the Setup wizard's docs page.
"""

from __future__ import annotations

import html as html_escape
import logging
from pathlib import Path
from typing import Any

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import Gtk, GLib  # type: ignore[attr-defined]

logger = logging.getLogger(__name__)

# Build a robust tuple of repository errors for exception handling
_REPOSITORY_ERRORS: tuple[type[BaseException], ...] = (ImportError, ValueError)
_repository_error = getattr(gi, "RepositoryError", None)
if isinstance(_repository_error, type) and issubclass(_repository_error, BaseException):
    _REPOSITORY_ERRORS = (*_REPOSITORY_ERRORS, _repository_error)

# File extensions recognized as documentation
DOC_EXTENSIONS = frozenset({".md", ".markdown", ".html", ".htm", ".txt", ".rst"})


def _load_webkit() -> tuple[bool, Any | None]:
    """Attempt to load a GTK4-compatible WebKit namespace."""
    gtk_major = getattr(Gtk, "get_major_version", lambda: 0)()
    target_versions = ["6.0", "5.0"] if gtk_major >= 4 else ["4.1", "4.0"]

    for version in target_versions:
        try:
            gi.require_version("WebKit", version)
            from gi.repository import WebKit as WebKitNS  # type: ignore
            return True, WebKitNS
        except _REPOSITORY_ERRORS:
            continue

    return False, None


WEBKIT_AVAILABLE, WebKit = _load_webkit()


DEFAULT_PLACEHOLDER_HTML = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>ATLAS Docs</title>
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


def get_html_template(title: str, toc: str, content: str) -> str:
    """Wrap content in a fully styled HTML document.
    
    Args:
        title: The document title.
        toc: Table of contents HTML (can be empty).
        content: The main HTML content.
        
    Returns:
        Complete HTML document string.
    """
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
      @media (prefers-color-scheme: dark) {{
        a {{ color: #58a6ff; }}
      }}
      /* Mermaid diagram styling */
      pre.mermaid {{
        background: transparent;
        border: none;
        text-align: center;
      }}
    </style>
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


def render_markdown(doc_path: Path) -> str:
    """Render a markdown file to HTML.
    
    Uses the markdown library if available, otherwise falls back to
    basic regex-based conversion.
    
    Args:
        doc_path: Path to the markdown file.
        
    Returns:
        Complete HTML document string.
    """
    try:
        import markdown  # type: ignore
    except ImportError:
        logger.debug("Markdown package unavailable; using fallback rendering for %s", doc_path)
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
                "permalink": "",
                "anchorlink": False,
                "title": "Table of Contents",
                "toc_depth": "2-6",
            }
        },
    )
    html_content = md_converter.convert(raw)
    toc = getattr(md_converter, "toc", "") or ""

    return get_html_template(doc_path.name, toc, html_content)


def render_raw_markdown(doc_path: Path) -> str:
    """Render markdown with basic fallback conversion when markdown library is unavailable.
    
    Args:
        doc_path: Path to the markdown file.
        
    Returns:
        Complete HTML document string with basic formatting.
    """
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
    return get_html_template(doc_path.name, "", content)


def create_web_view(
    doc_path: Path | None = None,
    on_navigate: Any | None = None,
) -> Gtk.Widget:
    """Create a WebKit view for rendering documents.
    
    Args:
        doc_path: Path to document to load, or None for placeholder.
        on_navigate: Optional callback for link navigation (receives Path).
        
    Returns:
        GTK Widget containing the rendered document.
    """
    if WebKit is None:
        logger.debug("WebKit is unavailable; falling back to text view.")
        if doc_path is None:
            return create_placeholder_view(None, "WebKit unavailable. Using text preview.")
        return create_text_view(doc_path)

    web_view = WebKit.WebView()
    web_view.set_hexpand(True)
    web_view.set_vexpand(True)
    web_view.set_can_focus(True)

    # Handle navigation for relative links if callback provided
    if on_navigate is not None:
        def on_decide_policy(wv: Any, decision: Any, decision_type: Any) -> bool:
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

        web_view.connect("decide-policy", on_decide_policy)

    if doc_path is None:
        web_view.load_html(DEFAULT_PLACEHOLDER_HTML, "about:blank")
        return web_view

    if doc_path.suffix.lower() in {".md", ".markdown"}:
        rendered = render_markdown(doc_path)
        web_view.load_html(rendered, doc_path.parent.as_uri() + "/")
        return web_view

    try:
        web_view.load_uri(doc_path.as_uri())
    except Exception as exc:
        logger.error("Failed to load documentation %s: %s", doc_path, exc, exc_info=True)
        return create_text_view(doc_path)

    return web_view


def create_text_view(doc_path: Path) -> Gtk.Widget:
    """Create a styled text view for documents when WebKit is unavailable.
    
    Args:
        doc_path: Path to the document file.
        
    Returns:
        GTK ScrolledWindow containing a TextView.
    """
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


def create_placeholder_view(doc_path: Path | None, reason: str) -> Gtk.Widget:
    """Create a placeholder view for errors or missing content.
    
    Args:
        doc_path: Path that was attempted, or None.
        reason: Explanation message to display.
        
    Returns:
        GTK ScrolledWindow containing a TextView with the message.
    """
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


def discover_repo_root() -> Path:
    """Find the repository root by looking for README.md.
    
    Returns:
        Path to the repository root directory.
    """
    current = Path(__file__).resolve()
    for candidate in (current.parent, *current.parents):
        if (candidate / "README.md").exists():
            return candidate
    return current.parent


def resolve_default_doc() -> tuple[Path | None, str]:
    """Find the default documentation file to display.
    
    Searches for docs/index.html, docs/README.md, then README.md.
    
    Returns:
        Tuple of (path or None, hint message).
    """
    repo_root = discover_repo_root()
    default_locations = (
        Path("docs/index.html"),
        Path("docs/README.md"),
        Path("README.md"),
    )
    for relative in default_locations:
        candidate = repo_root / relative
        if candidate.exists():
            return candidate, f"Loading {candidate.name} from {candidate.parent}"
    return None, "No bundled documentation found. Loading placeholder content."
