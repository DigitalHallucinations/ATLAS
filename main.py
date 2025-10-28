# main.py

import logging

import gi

gi.require_version('Gtk', '4.0')
from gi.repository import Gtk

from ATLAS.ATLAS import ATLAS
from ATLAS.setup_marker import is_setup_complete
from GTKUI.Setup.first_run import FirstRunCoordinator
from GTKUI.sidebar import MainWindow

logger = logging.getLogger(__name__)


def main() -> None:
    app = Gtk.Application(application_id='com.example.atlas')

    def on_activate(app: Gtk.Application) -> None:
        atlas_instance: ATLAS | None = None

        def atlas_factory() -> ATLAS:
            nonlocal atlas_instance
            if not is_setup_complete():
                logger.error(
                    "ATLAS setup is incomplete. Launching the setup wizard."
                )
                raise RuntimeError("ATLAS setup is incomplete.")

            if atlas_instance is None:
                atlas_instance = ATLAS()
            return atlas_instance

        coordinator = FirstRunCoordinator(
            application=app,
            atlas_factory=atlas_factory,
            main_window_cls=MainWindow,
        )

        coordinator.activate()

        # Retain references while the GTK application is running.
        app._first_run_coordinator = coordinator  # type: ignore[attr-defined]
        app._atlas_instance = coordinator.atlas  # type: ignore[attr-defined]
        app._main_window = coordinator.main_window  # type: ignore[attr-defined]
        app._setup_window = coordinator.setup_window  # type: ignore[attr-defined]

    app.connect('activate', on_activate)
    app.run()


if __name__ == "__main__":
    main()
