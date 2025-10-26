"""Helpers for managing the initial GTK application startup."""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Optional

import gi

gi.require_version('Gtk', '4.0')
from gi.repository import Gtk

AtlasFactory = Callable[[], Any]
LoopRunner = Callable[[Any], Any]


class FirstRunCoordinator:
    """Coordinate the initial GTK startup sequence for ATLAS."""

    def __init__(
        self,
        *,
        application: Gtk.Application,
        atlas_factory: AtlasFactory,
        main_window_cls: type,
        setup_window_cls: type | None = None,
        loop_runner: LoopRunner = asyncio.run,
    ) -> None:
        self._application = application
        self._atlas_factory = atlas_factory
        self._main_window_cls = main_window_cls
        self._setup_window_cls = setup_window_cls or SetupWizardWindow
        self._loop_runner = loop_runner

        self.atlas: Any | None = None
        self.main_window: Any | None = None
        self.setup_window: Any | None = None

    def activate(self) -> None:
        """Attempt to start the application, falling back to the setup wizard."""

        try:
            self._ensure_atlas()
        except RuntimeError as exc:
            self._present_setup(error=exc)
            return

        self._initialize_or_present_setup()

    # -- private helpers -------------------------------------------------

    def _ensure_atlas(self) -> Any:
        if self.atlas is None:
            self.atlas = self._atlas_factory()
        return self.atlas

    def _initialize_or_present_setup(self) -> None:
        try:
            atlas = self._ensure_atlas()
        except RuntimeError as exc:
            self._present_setup(error=exc)
            return

        try:
            self._execute_initialization(atlas)
        except RuntimeError as exc:
            self._present_setup(error=exc)
            return

        self._present_main_window()

    def _execute_initialization(self, atlas: Any) -> None:
        coroutine = atlas.initialize()
        try:
            self._loop_runner(coroutine)
        except Exception:
            if hasattr(coroutine, 'close'):
                coroutine.close()
            raise

    def _present_main_window(self) -> None:
        self._close_setup()
        atlas = self._ensure_atlas()
        self.main_window = self._main_window_cls(atlas)
        if hasattr(self.main_window, "set_application"):
            self.main_window.set_application(self._application)
        if hasattr(self.main_window, "present"):
            self.main_window.present()

    def _present_setup(self, *, error: Optional[BaseException] = None) -> None:
        if self.setup_window is None:
            self.setup_window = self._setup_window_cls(
                application=self._application,
                atlas=self.atlas,
                on_success=self._handle_setup_success,
                on_error=self._handle_setup_error,
                error=error,
            )
            if hasattr(self.setup_window, "set_application"):
                self.setup_window.set_application(self._application)
        elif error is not None and hasattr(self.setup_window, "display_error"):
            self.setup_window.display_error(error)

        if hasattr(self.setup_window, "present"):
            self.setup_window.present()

    def _close_setup(self) -> None:
        if self.setup_window is not None and hasattr(self.setup_window, "close"):
            self.setup_window.close()
        self.setup_window = None

    def _handle_setup_success(self) -> None:
        try:
            if self.atlas is None:
                self.atlas = self._atlas_factory()
        except RuntimeError as exc:
            self._present_setup(error=exc)
            return

        try:
            self._execute_initialization(self.atlas)
        except RuntimeError as exc:
            self._present_setup(error=exc)
            return

        self._present_main_window()

    def _handle_setup_error(self, error: BaseException) -> None:
        self._present_setup(error=error)


class SetupWizardWindow(Gtk.Window):
    """A minimal GTK window guiding users through the first-run setup."""

    def __init__(
        self,
        *,
        application: Gtk.Application,
        atlas: Any | None,
        on_success: Callable[[], None],
        on_error: Callable[[BaseException], None],
        error: BaseException | None = None,
    ) -> None:
        super().__init__(title="ATLAS Setup")
        self._on_success = on_success
        self._on_error = on_error
        self._atlas = atlas
        self.set_application(application)

        self.set_default_size(480, 320)

        root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        root.set_margin_top(24)
        root.set_margin_bottom(24)
        root.set_margin_start(24)
        root.set_margin_end(24)
        self.set_child(root)

        intro = Gtk.Label(label="Complete the setup steps to finish configuring ATLAS.")
        intro.set_wrap(True)
        intro.set_justify(Gtk.Justification.FILL)
        root.append(intro)

        self._error_label = Gtk.Label()
        self._error_label.set_wrap(True)
        self._error_label.set_visible(False)
        self._error_label.set_css_classes(["error-label"])
        root.append(self._error_label)

        button_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        root.append(button_box)

        continue_button = Gtk.Button(label="Finish Setup")
        continue_button.connect("clicked", self._on_continue_clicked)
        button_box.append(continue_button)

        retry_button = Gtk.Button(label="Report Problem")
        retry_button.connect("clicked", self._on_report_problem)
        button_box.append(retry_button)

        if error is not None:
            self.display_error(error)

    # -- GTK callbacks ---------------------------------------------------

    def _on_continue_clicked(self, _button: Gtk.Button) -> None:
        self._on_success()

    def _on_report_problem(self, _button: Gtk.Button) -> None:
        self._on_error(RuntimeError("Setup workflow reported a problem."))

    # -- public helpers --------------------------------------------------

    def display_error(self, error: BaseException) -> None:
        message = str(error) or error.__class__.__name__
        self._error_label.set_text(message)
        self._error_label.set_visible(True)

    # Convenience hooks for tests and other automation layers ------------

    def emit_success(self) -> None:
        """Programmatically trigger the success callback."""

        self._on_success()

    def emit_error(self, error: BaseException) -> None:
        """Programmatically trigger the error callback with ``error``."""

        self._on_error(error)
