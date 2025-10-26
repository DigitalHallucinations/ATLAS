"""GTK front-end that launches the standalone setup utility."""

from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Callable

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("GLib", "2.0")
from gi.repository import GLib, Gtk

from ATLAS.setup import SetupWizardController as CoreSetupWizardController

Callback = Callable[[], None]
ErrorCallback = Callable[[BaseException], None]


class SetupWizardWindow(Gtk.Window):
    """Minimal helper window that runs the CLI setup utility."""

    def __init__(
        self,
        *,
        application: Gtk.Application,
        atlas: Any | None,
        on_success: Callback,
        on_error: ErrorCallback,
        error: BaseException | None = None,
        run_setup: Callable[[], int] | None = None,
    ) -> None:
        super().__init__()
        if hasattr(self, "set_title"):
            self.set_title("ATLAS Setup Utility")
        self.set_application(application)
        self._on_success = on_success
        self._on_error = on_error
        self._run_setup = run_setup or self._execute_setup_script
        self._worker: threading.Thread | None = None

        self.controller = CoreSetupWizardController(atlas=atlas)

        root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        root.set_margin_top(18)
        root.set_margin_bottom(18)
        root.set_margin_start(18)
        root.set_margin_end(18)
        self.set_child(root)

        header = Gtk.Label(
            label=(
                "ATLAS must be configured using the standalone setup utility. "
                "Press the button below to launch it in a terminal session."
            )
        )
        header.set_wrap(True)
        header.set_xalign(0.0)
        root.append(header)

        self._status_label = Gtk.Label()
        self._status_label.set_wrap(True)
        self._status_label.set_xalign(0.0)
        if error is not None:
            self.display_error(error)
        root.append(self._status_label)

        self._run_button = Gtk.Button(label="Run setup utility")
        self._run_button.connect("clicked", self._on_run_clicked)
        root.append(self._run_button)

    def _on_run_clicked(self, *_: object) -> None:
        if self._worker and self._worker.is_alive():
            return

        self._status_label.set_text("Running setup utility…")
        if hasattr(self._run_button, "set_sensitive"):
            self._run_button.set_sensitive(False)

        def _run() -> None:
            try:
                returncode = self._run_setup()
            except Exception as exc:  # pragma: no cover - error surfaced asynchronously
                GLib.idle_add(self._report_failure, exc)
            else:
                GLib.idle_add(self._report_result, returncode)

        self._worker = threading.Thread(target=_run, daemon=True)
        self._worker.start()

    def _report_failure(self, error: BaseException) -> None:
        self.display_error(error)
        if hasattr(self._run_button, "set_sensitive"):
            self._run_button.set_sensitive(True)
        self._worker = None
        self._on_error(error)

    def _report_result(self, returncode: int) -> None:
        if returncode == 0:
            self._status_label.set_text(
                "Setup complete. Restart ATLAS to sign in with the new administrator account."
            )
            if hasattr(self._run_button, "set_sensitive"):
                self._run_button.set_sensitive(True)
            self._worker = None
            self._on_success()
        else:
            message = (
                "The setup utility exited with status %d. Review the terminal output for details." % returncode
            )
            self.display_error(RuntimeError(message))

    def display_error(self, error: BaseException) -> None:
        text = str(error) or error.__class__.__name__
        self._status_label.set_text(text)
        if hasattr(self._status_label, "set_css_classes"):
            self._status_label.set_css_classes(["error-text"])  # type: ignore[attr-defined]

    def _execute_setup_script(self) -> int:
        script = Path(__file__).resolve().parents[2] / "scripts" / "setup_atlas.py"
        return subprocess.run([sys.executable, str(script)], check=False).returncode


class SetupWizardController(CoreSetupWizardController):  # type: ignore[misc]
    """Backwards-compatible import shim for legacy callers."""

    pass
