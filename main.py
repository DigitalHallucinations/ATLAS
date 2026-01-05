# main.py

import logging

import gi

gi.require_version('Gtk', '4.0')
from gi.repository import Gtk

from core.ATLAS import ATLAS
from core.setup_marker import is_setup_complete
from GTKUI.Setup.first_run import FirstRunCoordinator
from GTKUI.sidebar import MainWindow
from core.providers.atlas_provider import AtlasProvider

logger = logging.getLogger(__name__)


def main() -> None:
    app = Gtk.Application(application_id='com.example.atlas')

    def on_activate(app: Gtk.Application) -> None:
        atlas_provider = AtlasProvider(atlas_cls=ATLAS, setup_check=is_setup_complete)
        coordinator = FirstRunCoordinator(
            application=app,
            atlas_provider=atlas_provider,
            main_window_cls=MainWindow,
        )

        coordinator.activate()

        # Retain references while the GTK application is running.
        app._first_run_coordinator = coordinator  # type: ignore[attr-defined]
        app._atlas_instance = coordinator.atlas  # type: ignore[attr-defined]
        app._main_window = coordinator.main_window  # type: ignore[attr-defined]
        app._setup_window = coordinator.setup_window  # type: ignore[attr-defined]
        app._atlas_provider = atlas_provider  # type: ignore[attr-defined]

    app.connect('activate', on_activate)
    app.run()


if __name__ == "__main__":
    main()
