from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import Gtk

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

DEFAULT_DOC_LOCATIONS = (
    Path("docs/index.html"),
    Path("docs/README.md"),
    Path("README.md"),
)

DEFAULT_PLACEHOLDER_HTML = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>ATLAS Docs</title>
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
      <h1>ATLAS Documentation</h1>
      <p>Browse local documentation or load an alternate file without leaving the app.</p>
      <ul>
        <li>The viewer will try to load bundled docs automatically.</li>
        <li>Use the file picker to open additional Markdown or HTML files.</li>
        <li>If nothing is available locally, this placeholder remains visible.</li>
      </ul>
      <p>If the viewer fails to load, you can still copy the doc path and open it externally.</p>
    </div>
  </body>
</html>"""


class DocsPage(Gtk.Box):
    """Docs browser with graceful fallbacks for missing local bundles."""

    def __init__(self, atlas: Any) -> None:
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        self.ATLAS = atlas
        self._viewer_container = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        self._status_label = Gtk.Label()
        self._path_entry = Gtk.Entry()
        self._default_doc_path, hint = self._resolve_default_doc()

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

        heading = Gtk.Label(label="Documentation")
        heading.set_wrap(True)
        heading.set_xalign(0.0)
        if hasattr(heading, "add_css_class"):
            heading.add_css_class("heading")
        self.append(heading)

        subtitle = Gtk.Label(
            label=(
                "Review bundled docs in a focused tab. Use the file picker to open other Markdown or "
                "HTML files locally, or press Enter after editing the path field."
            )
        )
        subtitle.set_wrap(True)
        subtitle.set_xalign(0.0)
        self.append(subtitle)

        hint_label = Gtk.Label(label=hint)
        hint_label.set_wrap(True)
        hint_label.set_xalign(0.0)
        if hasattr(hint_label, "add_css_class"):
            hint_label.add_css_class("dim-label")
        self.append(hint_label)

        self._path_entry.set_hexpand(True)
        self._path_entry.set_can_focus(True)
        self._path_entry.set_placeholder_text("Enter a local documentation path…")
        if self._default_doc_path is not None:
            self._path_entry.set_text(str(self._default_doc_path))
        self._path_entry.connect("activate", self._on_load_requested)

        browse_button = Gtk.Button(label="Browse…")
        browse_button.set_can_focus(True)
        browse_button.connect("clicked", self._on_choose_doc)

        load_button = Gtk.Button(label="Load")
        load_button.set_can_focus(True)
        load_button.set_receives_default(True)
        load_button.connect("clicked", self._on_load_requested)

        controls = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        controls.set_hexpand(True)
        controls.append(self._path_entry)
        controls.append(browse_button)
        controls.append(load_button)
        self.append(controls)

        self._status_label.set_wrap(True)
        self._status_label.set_xalign(0.0)
        self._status_label.add_css_class("dim-label")

        self._viewer_container.set_hexpand(True)
        self._viewer_container.set_vexpand(True)
        self._viewer_container.set_halign(Gtk.Align.FILL)
        self._viewer_container.set_valign(Gtk.Align.FILL)
        self.append(self._viewer_container)
        self.append(self._status_label)

        self._load_view(self._default_doc_path, hint)

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------
    def _resolve_default_doc(self) -> tuple[Path | None, str]:
        repo_root = Path(__file__).resolve().parents[2]
        for relative in DEFAULT_DOC_LOCATIONS:
            candidate = repo_root / relative
            if candidate.exists():
                return candidate, f"Loading {candidate.name} from {candidate.parent}"
        return None, "No bundled documentation found. A placeholder will be shown instead."

    def _on_choose_doc(self, _button: Gtk.Button) -> None:
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
                        self._load_view(Path(chosen), f"Loaded {chosen}")
        finally:
            dialog.destroy()

    def _on_load_requested(self, _widget: Gtk.Widget) -> None:
        text = self._path_entry.get_text().strip()
        if not text:
            self._load_view(None, "No path provided. Showing placeholder content.")
            return
        self._load_view(Path(text))

    def _load_view(self, path: Path | None, hint: str | None = None) -> None:
        viewer: Gtk.Widget
        status_message: str

        if path is None:
            viewer = (
                self._create_web_view(None)
                if WebKit is not None
                else self._create_placeholder_view(None, hint)
            )
            status_message = hint or "Showing placeholder content."
        elif path.exists():
            viewer = self._create_web_view(path)
            status_message = hint or f"Loaded {path}"
        else:
            viewer = self._create_placeholder_view(path, f"{path} was not found.")
            status_message = f"{path} was not found. Showing placeholder content."

        self._replace_viewer(viewer)
        self._update_status(status_message)

    def _create_web_view(self, doc_path: Path | None) -> Gtk.Widget:
        if WebKit is None:
            logger.debug("WebKit is unavailable; falling back to placeholder view.")
            return self._create_placeholder_view(doc_path, "WebKit is unavailable. Showing a preview instead.")

        web_view = WebKit.WebView()
        web_view.set_hexpand(True)
        web_view.set_vexpand(True)
        web_view.set_can_focus(True)

        if doc_path is None:
            web_view.load_html(DEFAULT_PLACEHOLDER_HTML, "about:blank")
            return web_view

        try:
            web_view.load_uri(doc_path.as_uri())
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Failed to load documentation %s: %s", doc_path, exc, exc_info=True)
            return self._create_placeholder_view(doc_path, f"Unable to load {doc_path}. Showing a preview instead.")
        return web_view

    def _create_placeholder_view(self, doc_path: Path | None, reason: str | None) -> Gtk.Widget:
        scroller = Gtk.ScrolledWindow()
        scroller.set_hexpand(True)
        scroller.set_vexpand(True)
        scroller.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)

        buffer = Gtk.TextBuffer()
        preview_lines = [
            "ATLAS Documentation",
            "",
            reason or "Showing placeholder content.",
            "",
        ]
        if doc_path is not None and doc_path.exists():
            try:
                preview_lines.append(f"Previewing {doc_path}")
                preview_lines.append("")
                preview_lines.extend(doc_path.read_text(encoding="utf-8").splitlines()[:120])
            except OSError as exc:  # pragma: no cover - defensive
                logger.debug("Failed to read %s: %s", doc_path, exc, exc_info=True)
                preview_lines.append(f"Unable to read {doc_path}. Showing placeholder instead.")
        elif doc_path is not None:
            preview_lines.append(f"{doc_path} was not found locally.")

        buffer.set_text("\n".join(preview_lines))

        text_view = Gtk.TextView.new_with_buffer(buffer)
        text_view.set_editable(False)
        text_view.set_cursor_visible(False)
        text_view.set_wrap_mode(Gtk.WrapMode.WORD)
        text_view.set_hexpand(True)
        text_view.set_vexpand(True)
        text_view.set_can_focus(True)
        text_view.set_margin_top(6)
        text_view.set_margin_bottom(6)
        text_view.set_margin_start(6)
        text_view.set_margin_end(6)
        scroller.set_child(text_view)
        return scroller

    def _replace_viewer(self, widget: Gtk.Widget) -> None:
        child = self._viewer_container.get_first_child()
        while child is not None:
            next_child = child.get_next_sibling()
            self._viewer_container.remove(child)
            child = next_child
        self._viewer_container.append(widget)
        widget.grab_focus()

    def _update_status(self, message: str) -> None:
        self._status_label.set_text(message)
