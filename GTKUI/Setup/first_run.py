"""Helpers for managing the initial GTK application startup."""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Optional, Protocol

import gi

gi.require_version('Gtk', '4.0')
from gi.repository import Gtk

from .setup_wizard import SetupWizardWindow

LoopRunner = Callable[[Any], Any]


class AtlasProvider(Protocol):
    """Protocol describing an object that can provide an ATLAS instance."""

    def get_atlas(self) -> Any:
        ...


class FirstRunCoordinator:
    """Coordinate the initial GTK startup sequence for ATLAS."""

    def __init__(
        self,
        *,
        application: Gtk.Application,
        atlas_provider: AtlasProvider,
        main_window_cls: type,
        setup_window_cls: type | None = None,
        loop_runner: LoopRunner = asyncio.run,
    ) -> None:
        self._application = application
        self._atlas_provider = atlas_provider
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
            self.atlas = self._atlas_provider.get_atlas()
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
                self.atlas = self._atlas_provider.get_atlas()
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
