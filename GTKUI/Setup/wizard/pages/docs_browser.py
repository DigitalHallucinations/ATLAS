"""Documentation browser page builder."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import Gtk

_REPOSITORY_ERRORS: tuple[type[BaseException], ...] = (ImportError, ValueError)
_repository_error = getattr(gi, "RepositoryError", None)
if isinstance(_repository_error, type) and issubclass(_repository_error, BaseException):
    _REPOSITORY_ERRORS = (*_REPOSITORY_ERRORS, _repository_error)


def _load_webkit2() -> tuple[bool, Any | None]:
    """Attempt to load a GTK4-compatible WebKit2 namespace."""

    gtk_major = getattr(Gtk, "get_major_version", lambda: 0)()
    target_versions = ["6.0", "5.0"] if gtk_major >= 4 else ["4.1", "4.0"]

    for version in target_versions:
        try:
            gi.require_version("WebKit2", version)
            from gi.repository import WebKit2 as WebKit2NS  # type: ignore
        except _REPOSITORY_ERRORS:
            continue
        return True, WebKit2NS

    return False, None


WEBKIT_AVAILABLE, WebKit2 = _load_webkit2()

DEFAULT_DOC_LOCATIONS = (
    Path("docs/index.html"),
    Path("docs/README.md"),
    Path("README.md"),
)

DEFAULT_PLACEHOLDER_HTML = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>ATLAS Docs Preview</title>
    <style>
      body { font-family: sans-serif; margin: 1.5rem; line-height: 1.5; }
      h1 { margin-top: 0; }
      .card { border: 1px solid #dadada; border-radius: 8px; padding: 1rem; background: #fafafa; }
      ul { padding-left: 1.25rem; }
      code { background: #f0f0f0; padding: 0.1rem 0.25rem; border-radius: 4px; }
    </style>
  </head>
  <body>
    <div class="card">
      <h1>ATLAS Setup Docs</h1>
      <p>Use this preview to confirm installation requirements while you walk through setup.</p>
      <ul>
        <li>Offline bundles will appear here automatically once packaged.</li>
        <li>For now, you can open <code>README.md</code> or the docs index from the repository root.</li>
        <li>Use the breadcrumbs in the left rail to return to the setup forms.</li>
      </ul>
      <p>If nothing loads automatically, start with the local README to review prerequisites.</p>
    </div>
  </body>
</html>"""


def _discover_repo_root() -> Path:
    current = Path(__file__).resolve()
    for candidate in (current.parent, *current.parents):
        if (candidate / "README.md").exists():
            return candidate
    return current.parent


def _resolve_default_doc() -> tuple[Path | None, str]:
    repo_root = _discover_repo_root()
    for relative in DEFAULT_DOC_LOCATIONS:
        candidate = repo_root / relative
        if candidate.exists():
            return candidate, f"Loading {candidate.name} from {candidate.parent}"
    return None, "No bundled documentation found. Loading placeholder content."


def _create_web_view(doc_path: Path | None) -> Gtk.Widget:
    if WebKit2 is None:
        return _create_placeholder_view(doc_path)

    web_view = WebKit2.WebView()
    web_view.set_hexpand(True)
    web_view.set_vexpand(True)
    web_view.set_can_focus(True)

    if doc_path is None:
        web_view.load_html(DEFAULT_PLACEHOLDER_HTML, "about:blank")
        return web_view

    if doc_path.suffix.lower() in {".md", ".markdown"}:
        rendered = _render_markdown(doc_path)
        web_view.load_html(rendered, doc_path.parent.as_uri())
        return web_view

    web_view.load_uri(doc_path.as_uri())

    return web_view


def _create_placeholder_view(doc_path: Path | None) -> Gtk.Widget:
    scroller = Gtk.ScrolledWindow()
    scroller.set_hexpand(True)
    scroller.set_vexpand(True)
    scroller.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)

    buffer = Gtk.TextBuffer()
    preview_lines = [
        "ATLAS Setup Docs",
        "",
        "Preview documentation is shown here. Use this panel to review installation guidance",
        "without leaving the setup flow.",
        "",
    ]
    if doc_path is not None:
        try:
            preview_lines.append(f"Default doc path: {doc_path}")
            preview_lines.append("")
            preview_lines.extend(doc_path.read_text(encoding="utf-8").splitlines()[:80])
        except OSError:  # pragma: no cover - defensive
            preview_lines.append(f"Unable to read {doc_path}. Showing placeholder content instead.")
    else:
        preview_lines.append("No bundled documentation found. A placeholder summary is shown instead.")
    buffer.set_text("\n".join(preview_lines))

    text_view = Gtk.TextView.new_with_buffer(buffer)
    text_view.set_editable(False)
    text_view.set_cursor_visible(False)
    text_view.set_wrap_mode(Gtk.WrapMode.WORD)
    text_view.set_hexpand(True)
    text_view.set_vexpand(True)
    text_view.set_margin_top(6)
    text_view.set_margin_bottom(6)
    text_view.set_margin_start(6)
    text_view.set_margin_end(6)
    scroller.set_child(text_view)
    return scroller


def _render_markdown(doc_path: Path) -> str:
    try:
        import markdown  # type: ignore
    except ImportError:
        return DEFAULT_PLACEHOLDER_HTML

    raw = doc_path.read_text(encoding="utf-8")
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
    html = md_converter.convert(raw)
    toc = md_converter.toc or ""
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>{doc_path.name}</title>
    <style>
      :root {{
        color-scheme: light dark;
        --surface: #1f1f1f;
        --card: #262626;
        --muted: #b8b8b8;
      }}
      body {{
        font-family: sans-serif;
        margin: 1.5rem;
        line-height: 1.6;
      }}
      h1, h2, h3, h4, h5, h6 {{
        margin-top: 1.25rem;
        margin-bottom: 0.5rem;
      }}
      pre, code {{
        font-family: "JetBrains Mono", "Fira Code", monospace;
      }}
      pre {{
        padding: 0.75rem;
        background: rgba(0,0,0,0.08);
        border-radius: 8px;
        overflow: auto;
      }}
      code {{
        background: rgba(0,0,0,0.05);
        padding: 0.1rem 0.35rem;
        border-radius: 6px;
      }}
      blockquote {{
        border-left: 3px solid rgba(0,0,0,0.2);
        margin: 0.75rem 0;
        padding-left: 0.75rem;
        color: var(--muted);
      }}
      ul, ol {{
        padding-left: 1.5rem;
      }}
      table {{
        border-collapse: collapse;
        width: 100%;
        margin: 1rem 0;
      }}
      th, td {{
        border: 1px solid rgba(0,0,0,0.15);
        padding: 0.5rem 0.75rem;
        text-align: left;
      }}
      th {{
        background: rgba(0,0,0,0.08);
      }}
      .toc {{
        border: 1px solid rgba(0,0,0,0.1);
        border-radius: 8px;
        padding: 0.75rem;
        background: rgba(0,0,0,0.03);
        margin-bottom: 1rem;
      }}
      .toc h1 {{
        margin: 0 0 0.5rem 0;
        font-size: 1.05rem;
      }}
      .toc ul {{
        padding-left: 1.25rem;
        margin: 0;
        list-style: disc;
      }}
      .toc a {{
        text-decoration: none;
      }}
      .toc a:hover, .toc a:focus {{
        text-decoration: underline;
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
      {html}
    </article>
  </body>
</html>"""


def build_docs_browser_page(wizard: "SetupWizardWindow") -> Gtk.Widget:
    doc_path, doc_hint = _resolve_default_doc()

    container = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
    container.set_hexpand(True)
    container.set_vexpand(True)
    container.set_halign(Gtk.Align.FILL)

    heading = Gtk.Label(label="Browse setup documentation")
    heading.set_wrap(True)
    heading.set_xalign(0.0)
    if hasattr(heading, "add_css_class"):
        heading.add_css_class("heading")
    container.append(heading)

    subtitle = Gtk.Label(
        label=(
            "Read the install guide without leaving setup. The embedded view tries to load local docs "
            "automatically and falls back to a placeholder summary if nothing is bundled yet."
        )
    )
    subtitle.set_wrap(True)
    subtitle.set_xalign(0.0)
    container.append(subtitle)

    hint_label = Gtk.Label(label=doc_hint)
    hint_label.set_wrap(True)
    hint_label.set_xalign(0.0)
    if hasattr(hint_label, "add_css_class"):
        hint_label.add_css_class("dim-label")
    container.append(hint_label)

    viewer = _create_web_view(doc_path)
    if isinstance(viewer, Gtk.Widget):
        viewer.set_hexpand(True)
        viewer.set_vexpand(True)
        viewer.set_halign(Gtk.Align.FILL)
    container.append(viewer)

    wizard._register_instructions(
        container,
        "Browse the embedded documentation to confirm prerequisites or find answers while keeping your place in the setup flow.",
    )

    return container


if False:  # pragma: no cover - type checking forward ref
    from GTKUI.Setup.setup_wizard import SetupWizardWindow
